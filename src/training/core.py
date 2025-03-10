from typing import Any, Callable

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float

import wandb
from src.distributions import AnnealedDistribution, Target
from src.mcmc.sampling import sample_with_mcmc
from src.utils.distributions import sample_monotonic_uniform_ordered
from src.utils.eval import (
    aggregate_eval_metrics,
    evaluate_model,
    log_metrics,
    save_model_if_best,
)
from src.utils.optimization import get_optimizer, inverse_power_schedule, power_schedule

from .config import TrainingExperimentConfig
from .loss import Particle, loss_fn
from .normalizing_constant import estimate_log_Z_t, estimate_log_Z_t_online


def generate_samples_with_optional_mcmc(
    key: jax.random.PRNGKey,
    v_theta: Callable,
    ts: jnp.ndarray,
    path_distribution: AnnealedDistribution,
    config: TrainingExperimentConfig,
    use_mcmc: bool = False,
    force_finite: bool = False,
    lambda_factor: Float[Array, ""] = 1.0,
):
    """
    Generate samples with or without MCMC correction.
    
    Args:
        key: Random key
        v_theta: Velocity field model
        ts: Time steps
        path_distribution: Annealed distribution
        config: Training experiment configuration
        use_mcmc: Whether to use MCMC correction
        force_finite: Whether to replace non-finite values with finite ones
        lambda_factor: Factor controlling the contribution of the velocity field
        
    Returns:
        Samples dictionary
    """
    # We generate samples in full precision
    ts_compute = config.mp_policy.cast_to_output(ts)
    
    # Determine MCMC method
    mcmc_method = config.mcmc.method if use_mcmc else "none"
    
    # Generate initial samples
    initial_samples = path_distribution.sample_initial(key, (config.sampling.num_particles,)).astype(config.mp_policy.output_dtype)

    # Use unified MCMC interface
    samples = sample_with_mcmc(
        key=key,
        initial_samples=initial_samples,
        v_theta=v_theta,
        time_dependent_log_density=path_distribution.time_dependent_log_prob,
        mcmc_method=mcmc_method,
        ts=ts_compute,
        shift_fn=config.density.shift_fn,
        use_shortcut=config.training.use_shortcut,
        num_steps=config.mcmc.num_steps,
        integration_steps=config.mcmc.num_integration_steps,
        eta=config.mcmc.step_size,
        ess_threshold=0.6,  # Default value
        estimate_covariance=False,  # Default value
        solver=config.integration.method,
        lambda_factor=lambda_factor,
    )
    
    if force_finite:
        samples["positions"] = jnp.nan_to_num(
            samples["positions"], nan=0.0, posinf=1.0, neginf=-1.0
        )
    chex.assert_type(samples["positions"], config.mp_policy.output_dtype)
    chex.assert_shape(samples["positions"], (ts.shape[0], config.sampling.num_particles, None))

    return samples

jitted_loss_fn = eqx.filter_jit(loss_fn)

def train_velocity_field(
    key: jax.random.PRNGKey,
    initial_density: Target,
    target_density: Target,
    v_theta: Callable[[chex.Array, float], chex.Array],
    config: TrainingExperimentConfig,
) -> Any:
    """Train a velocity field using either standard or decoupled loss function."""
    best_metrics = []
    model_version = 0

    path_distribution = AnnealedDistribution(
        initial_density=initial_density,
        target_density=target_density,
        method=config.density.annealing_path,
    )

    current_end_time = config.sampling.num_timesteps

    # Lambda factor scheduling
    lambda_max = config.mcmc.lambda_max  # Default to 1.0 if not specified
    lambda_epochs = config.mcmc.lambda_epochs  # Default to 1 if not specified
    lambda_total_steps = lambda_epochs * config.training.steps_per_epoch

    def compute_lambda_factor(step):
        # Exponentially increase lambda from 0 to lambda_max
        # Calculate progress as a combination of epoch and step
        total_progress = step / lambda_total_steps
        
        if total_progress >= lambda_epochs:
            return jnp.array(lambda_max, dtype=jnp.float32)
        # Exponential growth from almost 0 to lambda_max
        return jnp.array(lambda_max * (1.0 - jnp.exp(-5.0 * total_progress / lambda_epochs)), dtype=jnp.float32)

    # Set up base time steps
    if config.integration.schedule == "linear":
        base_ts = jnp.linspace(0, 1.0, current_end_time)
    elif config.integration.schedule == "inverse_power":
        base_ts = inverse_power_schedule(
            current_end_time,
            end_time=1.0,
            gamma=0.5,
        )
    elif config.integration.schedule == "power":
        base_ts = power_schedule(
            current_end_time,
            end_time=1.0,
            gamma=0.25,
        )
    else:
        raise ValueError(f"Unknown schedule {config.integration.schedule}")

    # Use a Python list to store log_Z_t between loop iterations
    # This is a mutable reference that can be updated in the loop
    log_Z_t_ref = [None]
    
    # Optimizer setup
    if config.training.gradient_clip_norm is not None:
        gradient_clipping = optax.clip_by_global_norm(
            config.training.gradient_clip_norm
        )
    elif config.training.gradient_clip is not None:
        gradient_clipping = optax.clip(config.training.gradient_clip)
    else:
        gradient_clipping = optax.identity()

    lr_schedule = config.training.learning_rate
    if config.training.use_schedule:
        total_steps = config.training.num_epochs * config.training.steps_per_epoch

        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=config.training.schedule_init_value,
            peak_value=config.training.learning_rate,
            warmup_steps=config.training.schedule_warmup_steps,
            decay_steps=int(total_steps * config.training.schedule_decay_fraction),
            end_value=config.training.schedule_end_value,
        )

    base_optimizer = get_optimizer(
        config.training.optimizer,
        lr_schedule,
        weight_decay=config.training.weight_decay,
        b1=config.training.beta1,
        b2=config.training.beta2,
        eps=config.training.epsilon,
        momentum=config.training.momentum,
        nesterov=config.training.nesterov,
        noise_scale=config.training.noise_scale,
    )

    optimizer = optax.chain(optax.zero_nans(), gradient_clipping, base_optimizer)
    # optimizer: optax.GradientTransformation = optax.apply_if_finite(optimizer, 5)
    if config.training.every_k_schedule > 1:
        optimizer = optax.MultiSteps(
            optimizer, every_k_schedule=config.training.every_k_schedule
        )

    opt_state = optimizer.init(eqx.filter(v_theta, eqx.is_inexact_array))
    
    # Function to get current learning rate from schedule
    def get_current_lr(step_count):
        if config.training.use_schedule:
            return lr_schedule(step_count)
        else:
            return config.training.learning_rate

    @eqx.filter_jit
    def step(key, v_theta, opt_state, particles):
        key, dropout_key = jax.random.split(key)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(
            v_theta,
            particles,
            path_distribution.time_derivative,
            path_distribution.score_fn,
            config.density.shift_fn,
            config.training.estimator,
            key=key,
            n_probes=config.training.n_probes,
            combined_loss=config.training.use_combined_loss,
            shortcut_weight=config.training.shortcut_weight,
            random_alpha=config.training.random_alpha,
            dropout_key=dropout_key if config.model.dropout is not None else None,
        )
            
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(v_theta, eqx.is_array))
        v_theta = eqx.apply_updates(v_theta, updates)
        return v_theta, opt_state, loss

    shortcut_size_d = 1.0 / jnp.array(
        [2**e for e in range(int(jnp.floor(jnp.log2(128))) + 1)]
    )

    prev_sum = None
    prev_count = None
    mcmc_samples = None

    for epoch in range(config.training.num_epochs):
        # Calculate current lambda_factor based on the epoch
        current_lambda = compute_lambda_factor(epoch * config.training.steps_per_epoch)
        if not config.offline:
            wandb.log({"lambda_factor": current_lambda})
        else:
            print(f"Epoch {epoch}, Lambda Factor: {current_lambda}")
            
        # Handle time steps for this epoch
        if config.integration.continuous_time:
            key, subkey = jax.random.split(key)
            current_ts = sample_monotonic_uniform_ordered(subkey, base_ts, True)
        else:
            current_ts = base_ts

        # Sample generation
        if config.training.use_decoupled_loss:            
            # Only estimate log_Z_t according to the configured frequency
            should_estimate_log_z = (epoch % config.training.log_z_estimation_frequency == 0) or (epoch == 0) or (log_Z_t_ref[0] is None)
            
            if should_estimate_log_z:
                key, subkey = jax.random.split(key)
                mcmc_samples = generate_samples_with_optional_mcmc(
                    subkey, v_theta, current_ts, path_distribution, config, 
                    use_mcmc=True, force_finite=True, lambda_factor=current_lambda
                )

                if not config.integration.continuous_time:
                    key, subkey = jax.random.split(key)
                    log_Z_t, prev_sum, prev_count = estimate_log_Z_t_online(
                        mcmc_samples["positions"],
                        mcmc_samples["weights"],
                        current_ts,
                        path_distribution.time_derivative,
                        v_theta=v_theta,
                        score_fn=path_distribution.score_fn,
                        use_control_variate=config.mcmc.use_control_variate,
                        use_shortcut=config.training.use_shortcut,
                        prev_log_sum=prev_sum,
                        prev_count=prev_count,
                    )
                else:
                    log_Z_t = estimate_log_Z_t(
                        mcmc_samples["positions"],
                        mcmc_samples["weights"],
                        current_ts,
                        path_distribution.time_derivative,
                        v_theta=v_theta,
                        score_fn=path_distribution.score_fn,
                        use_control_variate=config.mcmc.use_control_variate,
                        use_shortcut=config.training.use_shortcut,
                    )
                log_Z_t = jax.lax.stop_gradient(log_Z_t)
                
                # Update last_log_Z_t for future epochs
                log_Z_t_ref[0] = log_Z_t
                
                if not config.offline:
                    log_Z_t_to_log = jnp.nan_to_num(log_Z_t, nan=0.0, posinf=1.0, neginf=-1.0)
                    wandb.log({"log_Z_t": log_Z_t_to_log})
                    if "ess" in mcmc_samples:
                        wandb.log({"ess": mcmc_samples["ess"]})
                else:
                    print("Log Z: ", log_Z_t)
                    if "ess" in mcmc_samples:
                        print("MCMC Samples ESS: ", mcmc_samples["ess"])
            else:
                # Reuse the log_Z_t from the previous estimation
                log_Z_t = log_Z_t_ref[0]
                if not config.offline:
                    wandb.log({"log_Z_t (reused)": jnp.nan_to_num(log_Z_t, nan=0.0, posinf=1.0, neginf=-1.0)})
                
            key, subkey = jax.random.split(key)
            v_theta_samples = generate_samples_with_optional_mcmc(
                subkey, v_theta, current_ts, path_distribution, config,
                use_mcmc=False, force_finite=True, lambda_factor=current_lambda
            )
            samples = jnp.concatenate(
                [mcmc_samples["positions"], v_theta_samples["positions"]], axis=1
            )
        else:
            key, subkey = jax.random.split(key)
            samples = generate_samples_with_optional_mcmc(
                key, v_theta, current_ts, path_distribution, config,
                use_mcmc=True, force_finite=True, lambda_factor=current_lambda
            )
            if isinstance(samples, dict):
                samples = samples["positions"]
            log_Z_t = None

        epoch_loss = 0.0
        key, subkey = jax.random.split(key)
        num_particles = (
            config.sampling.num_particles * 2
            if config.training.use_decoupled_loss
            else config.sampling.num_particles
        )
        particles = Particle(
            x=samples.reshape(num_particles * current_ts.shape[0], -1),
            t=jnp.repeat(current_ts, num_particles),
            log_Z_t=jnp.repeat(log_Z_t, num_particles),
            d=jax.random.choice(
                subkey,
                shortcut_size_d,
                (num_particles * current_ts.shape[0],),
                replace=True,
            )
            if config.training.use_shortcut
            else None,
        )

        for s in range(config.training.steps_per_epoch):
            # Update lambda factor for each step within the epoch
            key, subkey = jax.random.split(key)
            indices = jax.random.choice(
                subkey,
                particles.x.shape[0],
                (config.training.time_batch_size * config.sampling.batch_size,),
            )
            training_particles = Particle(
                x=particles.x[indices],
                t=particles.t[indices],
                log_Z_t=particles.log_Z_t[indices],
                d=particles.d[indices] if particles.d is not None else None,
            )

            key, subkey = jax.random.split(key)
            v_theta, opt_state, loss = step(
                subkey, v_theta, opt_state, training_particles
            )
            epoch_loss += loss
            if s % 20 == 0:
                if not config.offline:
                    wandb.log({"loss": loss, "learning_rate": get_current_lr(epoch * config.training.steps_per_epoch + s)})
                else:
                    print(f"Epoch {epoch}, Step {s}, Loss: {loss}, Learning Rate: {get_current_lr(epoch * config.training.steps_per_epoch + s)}")

        avg_loss = epoch_loss / config.training.steps_per_epoch
        # We now calculate the validation loss

        key, dropout_key = jax.random.split(key)
        val_loss = jitted_loss_fn(
            v_theta,
            training_particles,
            path_distribution.time_derivative,
            path_distribution.score_fn,
            config.density.shift_fn,
            "none",
            key=key,
            combined_loss=False,
            shortcut_weight=config.training.shortcut_weight,
            random_alpha=config.training.random_alpha,
            dropout_key=dropout_key if config.model.dropout is not None else None,
        )

        if not config.offline:
            wandb.log({"epoch": epoch, "val_loss": val_loss, "average_loss": avg_loss, "epoch_learning_rate": get_current_lr(epoch * config.training.steps_per_epoch)})
        else:
            print(f"Epoch {epoch}, Average Loss: {avg_loss}, Learning Rate: {get_current_lr(epoch * config.training.steps_per_epoch)}")
            print(f"Epoch {epoch}, Validation Loss: {val_loss}")


        if epoch % config.training.eval_frequency == 0:
            # Run multiple evaluations
            all_eval_results = []
            for _ in range(3):
                key, subkey = jax.random.split(key)
                eval_metrics = evaluate_model(
                    subkey,
                    v_theta,
                    config,
                    path_distribution,
                    target_density,
                    current_end_time,
                )
                all_eval_results.append(eval_metrics)

            # Process and log metrics
            aggregated_metrics = aggregate_eval_metrics(all_eval_results)
            log_metrics(aggregated_metrics, config)

            # Handle model saving
            if not config.offline:
                best_metrics, model_version = save_model_if_best(
                    v_theta,
                    aggregated_metrics,
                    best_metrics,
                    model_version,
                    target_density,
                )

    # Save final model state
    if not config.offline and len(best_metrics) > 0:
        # Log summary of best models
        wandb.run.summary["best_metrics"] = [w2 for w2, _ in best_metrics]
        wandb.run.summary["best_model_versions"] = [ver for _, ver in best_metrics]
        wandb.finish()

    return v_theta
