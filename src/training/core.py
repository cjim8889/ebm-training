from typing import Any, Callable

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
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
from src.utils.optimization import get_optimizer, inverse_power_schedule, power_schedule, focus_schedule
from src.utils.schedule import constant_then_cyclic_cosine_schedule

from .config import TrainingExperimentConfig
from .loss import Particle, loss_fn, calculate_validation_loss_and_plot
from .normalizing_constant import (
    estimate_log_Z_t,
    estimate_log_Z_t_with_TI,
)

@eqx.filter_jit
def random_rotation_3d(key):
    key_angle, key_axis = jax.random.split(key)
    angle = jax.random.uniform(key_angle, shape=(), minval=0.0, maxval=2 * jnp.pi)
    # Sample a random axis uniformly from the sphere.
    axis = jax.random.normal(key_axis, shape=(3,))
    axis = axis / jnp.linalg.norm(axis)
    # Construct the skew-symmetric matrix for the axis.
    K = jnp.array([[0, -axis[2], axis[1]],
                   [axis[2], 0, -axis[0]],
                   [-axis[1], axis[0], 0]])
    I = jnp.eye(3)
    # Rodrigues rotation formula
    R = I + jnp.sin(angle) * K + (1 - jnp.cos(angle)) * (K @ K)
    return R

@eqx.filter_jit
def augment_chain(chain, key, translation_scale, num_particles):
    # Reshape the chain into (num_particles, 3)
    chain = chain.reshape((num_particles, 3))
    # Split the key for rotation and translation
    key_rot, key_trans = jax.random.split(key)
    # Generate a random 3D rotation matrix
    R = random_rotation_3d(key_rot)
    # Apply the rotation (note the transpose, since each row is a particle's coordinates)
    chain_rot = jnp.dot(chain, R.T)
    # Sample a random translation vector (3D)
    translation = jax.random.uniform(
        key_trans,
        shape=(1,),
        minval=-translation_scale,
        maxval=translation_scale
    )
    # Apply the translation
    chain_aug = chain_rot + translation
    # Flatten back to a 1D array (num_particles * 3)
    return chain_aug.reshape(-1)

batch_augment_chain = jax.vmap(augment_chain, in_axes=(0, 0, None, None))

def generate_samples_with_optional_mcmc(
    key: jax.random.PRNGKey,
    v_theta: Callable,
    ts: jnp.ndarray,
    path_distribution: AnnealedDistribution,
    config: TrainingExperimentConfig,
    mcmc_method: str = None,
    force_finite: bool = False,
    num_samples: int = None,
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
        mcmc_method: MCMC method to use (e.g., "smc", "hmc")
        force_finite: Whether to replace non-finite values with finite ones
        lambda_factor: Factor controlling the contribution of the velocity field
        
    Returns:
        Samples dictionary
    """
    # We generate samples in full precision
    ts_compute = config.mp_policy.cast_to_output(ts)
    
    # Determine MCMC method
    mcmc_method = config.mcmc.method if mcmc_method is None else mcmc_method

    _num_samples = config.sampling.num_particles if num_samples is None else num_samples
    
    # Generate initial samples
    initial_samples = path_distribution.sample_initial(key, (_num_samples,)).astype(config.mp_policy.output_dtype)

    # Use unified MCMC interface
    samples = sample_with_mcmc(
        key=key,
        initial_samples=initial_samples,
        v_theta=v_theta,
        time_dependent_log_density=path_distribution.time_dependent_log_prob,
        incremental_log_delta=path_distribution.incremental_log_delta,
        mcmc_method=mcmc_method,
        ts=ts_compute,
        shift_fn=config.density.shift_fn,
        use_shortcut=config.training.use_shortcut,
        num_steps=config.mcmc.num_steps,
        integration_steps=config.mcmc.num_integration_steps,
        eta=config.mcmc.step_size,
        ess_threshold=config.mcmc.ess_threshold,  # Default value
        estimate_covariance=False,  # Default value
        solver=config.integration.method,
        lambda_factor=lambda_factor,
    )
    
    if force_finite:
        samples["positions"] = jnp.nan_to_num(
            samples["positions"], nan=0.0, posinf=1.0, neginf=-1.0
        )
    chex.assert_type(samples["positions"], config.mp_policy.output_dtype)
    chex.assert_shape(samples["positions"], (None, _num_samples, None))

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
    base_ts = None

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
            gamma=0.15,
        )
    elif config.integration.schedule == "focus":
        base_ts = focus_schedule(
            current_end_time,
            end_time=1.0,
            gamma=0.7,
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
        lr_schedule = constant_then_cyclic_cosine_schedule(
            constant_value=config.training.learning_rate,
            initial_steps=config.training.schedule_warmup_steps,
            cycle_steps=config.training.steps_per_epoch * config.training.schedule_decay_epoch,
            peak_value=config.training.learning_rate,
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

    mcmc_samples = None
    current_ts = None

    key, subkey = jax.random.split(key)
    validation_ts = jnp.linspace(0, 1.0, current_end_time)
    validation_set = generate_samples_with_optional_mcmc(
        key=subkey,
        v_theta=v_theta,
        ts=validation_ts,
        path_distribution=path_distribution,
        config=config,
        mcmc_method="smc",
        force_finite=True,
        num_samples=config.training.time_batch_size * config.sampling.batch_size,
    )

    validation_log_Z_t = estimate_log_Z_t(
        xs=validation_set["positions"],
        weights=validation_set["weights"],
        ts=validation_ts,
        time_derivative_log_density=path_distribution.time_derivative,
    ).flatten()

    validation_particles = Particle(
        x=validation_set["positions"].reshape(-1, 39),
        t=validation_ts.repeat(config.training.time_batch_size * config.sampling.batch_size),
        log_Z_t=validation_log_Z_t.repeat(config.training.time_batch_size * config.sampling.batch_size),
    )

    N = config.sampling.num_timesteps * config.sampling.num_particles
    weights = jnp.ones(N)
    boundary = int(0.2 * N)
    weights = weights.at[:boundary].set(1.5)
    weights = weights.at[-boundary:].set(1.5)
    if config.training.use_decoupled_loss:
        weights = jnp.tile(weights, 2)

    p = weights / jnp.sum(weights)

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
            current_ts = base_ts if current_ts is None else current_ts

        # Only estimate log_Z_t according to the configured frequency
        should_estimate_log_z = (epoch % config.training.log_z_estimation_frequency == 0) or (epoch == 0) or (log_Z_t_ref[0] is None)
        
        if should_estimate_log_z:
            key, subkey = jax.random.split(key)
            mcmc_samples = generate_samples_with_optional_mcmc(
                subkey, v_theta, current_ts, path_distribution, config, 
                mcmc_method=config.mcmc.method, force_finite=True, lambda_factor=current_lambda
            )

            if config.mcmc.method == "asmc":
                current_ts = mcmc_samples["ts"]
                base_ts = mcmc_samples["ts"]

                print("Using Adaptive SMC for log Z estimation as well as tempering schedule selection")
                print("Current time steps: ", current_ts)


            if config.training.use_TI:
                # Use Thermodynamic Integration for log Z estimation
                print("Using Thermodynamic Integration for log Z estimation (Does not support control variate)")
                log_Z_t = estimate_log_Z_t_with_TI(
                    mcmc_samples["positions"],
                    mcmc_samples["weights"],
                    current_ts,
                    path_distribution.time_derivative,
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

            log_Z_t = log_Z_t.flatten()
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

        epoch_loss = 0.0
        key, subkey = jax.random.split(key)
        num_particles = (
            config.sampling.num_particles * 2
            if config.training.use_decoupled_loss
            else config.sampling.num_particles
        )
        # num_particles = config.sampling.num_particles
        # Sample generation
        if config.training.use_decoupled_loss:            
            key, subkey = jax.random.split(key)
            v_theta_samples = generate_samples_with_optional_mcmc(
                subkey, v_theta, current_ts, path_distribution, config,
                mcmc_method="none", force_finite=True, lambda_factor=current_lambda
            )
            samples = jnp.concatenate(
                [mcmc_samples["positions"], v_theta_samples["positions"]], axis=1
            )
        else:
            samples = mcmc_samples["positions"]

        time_steps, num_particles, d = samples.shape

        for s in range(config.training.steps_per_epoch):
            # Update lambda factor for each step within the epoch
            key, subkey = jax.random.split(key)

            chain_indices = jax.random.choice(
                subkey, num_particles, shape=(config.training.time_batch_size,), replace=False
            )
            selected_chains = samples[:, chain_indices, :].reshape(
                time_steps * config.training.time_batch_size, -1
            )

            # Apply augmentation
            if config.training.augment:
                # Get the batch size from selected_chains.
                batch_size = selected_chains.shape[0]
                # Generate a unique key for each datapoint in the batch.
                keys_aug = jax.random.split(key, batch_size)
                # Vectorize the augmentation over the batch dimension.
                selected_chains = batch_augment_chain(selected_chains, keys_aug, config.training.translation_scale, config.density.n_particles)

            # Apply translation augmentation
            if config.training.perturb:
                key, subkey = jax.random.split(key)
                selected_chains = selected_chains + jax.random.normal(
                    subkey, selected_chains.shape, dtype=config.mp_policy.output_dtype
                ) * config.training.perturbation_scale


            selected_t = jnp.repeat(current_ts, config.training.time_batch_size)
            selected_log_Z_t = jnp.repeat(log_Z_t, config.training.time_batch_size)
  
            training_particles = Particle(
                x=selected_chains,
                t=selected_t,
                log_Z_t=selected_log_Z_t,
                d=None,
                # d=particles.d[indices] if particles.d is not None else None,
                # loss_weight=particles.loss_weight[indices] if particles.loss_weight is not None else None,
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
            for _ in range(1):
                key, subkey = jax.random.split(key)
                eval_metrics = evaluate_model(
                    subkey,
                    v_theta,
                    config,
                    path_distribution,
                    target_density,
                    current_end_time,
                    current_ts=current_ts if config.mcmc.method == "asmc" else None,
                )
                all_eval_results.append(eval_metrics)

            # Process and log metrics
            aggregated_metrics = aggregate_eval_metrics(all_eval_results)
            log_metrics(aggregated_metrics, config)

            # Calculate validation loss
            validation_loss, validation_plt = calculate_validation_loss_and_plot(
                v_theta,
                validation_particles,
                path_distribution,
                validation_ts,
                time_batch_size=config.training.time_batch_size,
                batch_size=config.sampling.batch_size,
            )

            if not config.offline:
                wandb.log({"validation_loss": validation_loss})
                wandb.log({"validation_loss_plot": wandb.Image(validation_plt)})
                
                best_metrics, model_version = save_model_if_best(
                    v_theta,
                    aggregated_metrics,
                    best_metrics,
                    model_version,
                    target_density,
                )
            else:
                print(f"Validation Loss: {validation_loss}")
                plt.show()

            plt.close(validation_plt)

    # Save final model state
    if not config.offline and len(best_metrics) > 0:
        # Log summary of best models
        wandb.run.summary["best_metrics"] = [w2 for w2, _ in best_metrics]
        wandb.run.summary["best_model_versions"] = [ver for _, ver in best_metrics]
        wandb.finish()

    return v_theta
