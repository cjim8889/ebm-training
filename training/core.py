from typing import Any, Callable

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

import wandb
from distributions import AnnealedDistribution, Target
from utils.distributions import sample_monotonic_uniform_ordered
from utils.hmc import generate_samples_with_hmc_correction
from utils.smc import (
    generate_samples_with_euler_smc,
    generate_samples_with_smc,
    systematic_resampling,
)
from utils.integration import euler_integrate, generate_samples
from utils.optimization import get_optimizer, inverse_power_schedule, power_schedule
from .config import TrainingExperimentConfig
from .loss import loss_fn, estimate_log_Z_t


def train_velocity_field(
    key: jax.random.PRNGKey,
    initial_density: Target,
    target_density: Target,
    v_theta: Callable[[chex.Array, float], chex.Array],
    config: TrainingExperimentConfig,
) -> Any:
    """Train a velocity field using either standard or decoupled loss function.

    Args:
        key: Random key for JAX operations
        initial_density: Initial density distribution
        target_density: Target density distribution
        v_theta: Velocity field function
        config: Training configuration

    Returns:
        Trained velocity field function
    """
    path_distribution = AnnealedDistribution(
        initial_density=initial_density,
        target_density=target_density,
        method=config.density.annealing_path,
    )

    # Initialize current end time
    if config.progressive.enable:
        current_end_time = config.progressive.initial_timesteps
    else:
        current_end_time = config.sampling.num_timesteps

    # Set up optimizer
    if config.training.gradient_clip_norm is not None:
        gradient_clipping = optax.clip_by_global_norm(
            config.training.gradient_clip_norm
        )
    else:
        gradient_clipping = optax.identity()

    base_optimizer = get_optimizer(
        config.training.optimizer,
        config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        b1=config.training.beta1,
        b2=config.training.beta2,
        eps=config.training.epsilon,
        momentum=config.training.momentum,
        nesterov=config.training.nesterov,
        noise_scale=config.training.noise_scale,
    )
    optimizer = optax.chain(optax.zero_nans(), gradient_clipping, base_optimizer)
    optimizer: optax.GradientTransformation = optax.apply_if_finite(optimizer, 5)

    opt_state = optimizer.init(eqx.filter(v_theta, eqx.is_inexact_array))
    integrator = euler_integrate

    def _generate(key: jax.random.PRNGKey, ts: jnp.ndarray, force_finite: bool = False):
        samples = generate_samples(
            key,
            v_theta,
            config.sampling.num_particles,
            ts,
            path_distribution.sample_initial,
            integrator,
            config.density.shift_fn,
            use_shortcut=False,
        )

        if force_finite:
            samples["positions"] = jnp.nan_to_num(
                samples["positions"], nan=0.0, posinf=1.0, neginf=-1.0
            )

        return samples

    def _generate_mcmc(
        key: jax.random.PRNGKey, ts: jnp.ndarray, force_finite: bool = False
    ):
        if config.mcmc.method == "hmc":
            samples = generate_samples_with_hmc_correction(
                key=key,
                v_theta=v_theta,
                sample_fn=path_distribution.sample_initial,
                time_dependent_log_density=path_distribution.time_dependent_log_prob,
                num_samples=config.sampling.num_particles,
                ts=ts,
                integration_fn=integrator,
                num_steps=config.mcmc.num_steps,
                integration_steps=config.mcmc.num_integration_steps,
                eta=config.mcmc.step_size,
                rejection_sampling=config.mcmc.with_rejection,
                shift_fn=config.density.shift_fn,
                use_shortcut=False,
            )
        elif config.mcmc.method == "esmc":
            samples = generate_samples_with_euler_smc(
                key=key,
                v_theta=v_theta,
                time_dependent_log_density=path_distribution.time_dependent_log_prob,
                num_samples=config.sampling.num_particles,
                ts=ts,
                sample_fn=path_distribution.sample_initial,
                num_steps=config.mcmc.num_steps,
                integration_steps=config.mcmc.num_integration_steps,
                eta=config.mcmc.step_size,
                rejection_sampling=config.mcmc.with_rejection,
                shift_fn=config.density.shift_fn,
                use_shortcut=False,
            )
        elif config.mcmc.method == "smc":
            samples = generate_samples_with_smc(
                key=key,
                time_dependent_log_density=path_distribution.time_dependent_log_prob,
                num_samples=config.sampling.num_particles,
                ts=ts,
                sample_fn=path_distribution.sample_initial,
                num_steps=config.mcmc.num_steps,
                integration_steps=config.mcmc.num_integration_steps,
                eta=config.mcmc.step_size,
                rejection_sampling=config.mcmc.with_rejection,
                shift_fn=config.density.shift_fn,
                estimate_covariance=False,
                blackjax_hmc=True,
            )
        elif config.mcmc.method == "vsmc":
            samples = generate_samples_with_smc(
                key=key,
                time_dependent_log_density=path_distribution.time_dependent_log_prob,
                num_samples=config.sampling.num_particles,
                ts=ts,
                sample_fn=path_distribution.sample_initial,
                num_steps=config.mcmc.num_steps,
                integration_steps=config.mcmc.num_integration_steps,
                eta=config.mcmc.step_size,
                rejection_sampling=config.mcmc.with_rejection,
                shift_fn=config.density.shift_fn,
                v_theta=v_theta,
                estimate_covariance=False,
                blackjax_hmc=True,
            )
        else:
            samples = _generate(key, ts, force_finite)

        if force_finite:
            if isinstance(samples, dict):
                samples["positions"] = jnp.nan_to_num(
                    samples["positions"], nan=0.0, posinf=1.0, neginf=-1.0
                )
            else:
                samples = jnp.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)

        return samples

    @eqx.filter_jit
    def step(v_theta, opt_state, xs, ts, log_Z_t=None):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(
            v_theta,
            xs,
            ts,
            time_derivative_log_density=path_distribution.time_derivative,
            score_fn=path_distribution.score_fn,
            dt_log_density_clip=config.integration.dt_clip,
            log_Z_t=log_Z_t,
        )

        updates, opt_state = optimizer.update(grads, opt_state, v_theta)
        v_theta = eqx.apply_updates(v_theta, updates)

        return v_theta, opt_state, loss

    for epoch in range(config.training.num_epochs):
        # Update end time if needed
        if epoch % config.progressive.update_frequency == 0:
            if (
                config.progressive.enable
                and current_end_time < config.sampling.num_timesteps
                and epoch != 0
            ):
                current_end_time += config.progressive.timestep_increment
                current_end_time = min(current_end_time, config.sampling.num_timesteps)

            # Update time steps based on new end time
            if config.integration.schedule == "linear":
                current_ts = jnp.linspace(
                    0,
                    current_end_time * 1.0 / config.sampling.num_timesteps,
                    current_end_time,
                )
            elif config.integration.schedule == "inverse_power":
                current_ts = inverse_power_schedule(
                    current_end_time,
                    end_time=current_end_time * 1.0 / config.sampling.num_timesteps,
                    gamma=0.5,
                )
            elif config.integration.schedule == "power":
                current_ts = power_schedule(
                    current_end_time,
                    end_time=current_end_time * 1.0 / config.sampling.num_timesteps,
                    gamma=0.25,
                )

            if config.integration.continuous_time:
                key, subkey = jax.random.split(key)
                current_ts = sample_monotonic_uniform_ordered(subkey, current_ts, True)

        # Generate samples based on training mode
        if config.training.use_decoupled_loss:
            key, subkey = jax.random.split(key)
            mcmc_samples = _generate_mcmc(subkey, current_ts, force_finite=True)
            key, subkey = jax.random.split(key)
            log_Z_t = estimate_log_Z_t(
                mcmc_samples["positions"],
                mcmc_samples["weights"],
                current_ts,
                path_distribution.time_derivative,
            )

            if not config.offline:
                log_Z_t = jnp.nan_to_num(log_Z_t, nan=0.0, posinf=1.0, neginf=-1.0)
                wandb.log({"log_Z_t": log_Z_t})
                if "ess" in mcmc_samples:
                    wandb.log({"ess": mcmc_samples["ess"]})
            else:
                print("Log Z: ", log_Z_t)
                print("Current TS: ", current_ts)
                if "ess" in mcmc_samples:
                    print("MCMC Samples ESS: ", mcmc_samples["ess"])

            key, subkey = jax.random.split(key)
            v_theta_samples = _generate(subkey, current_ts, force_finite=True)

            resampling_keys = jax.random.split(subkey, current_ts.shape[0])
            good_samples_indices = systematic_resampling(
                resampling_keys,
                mcmc_samples["weights"],
                config.sampling.num_particles // 2,
            )
            good_samples = jax.vmap(lambda x, i: x[i])(
                mcmc_samples["positions"], good_samples_indices
            )

            samples = jnp.concatenate(
                [good_samples, v_theta_samples["positions"]], axis=1
            )
        else:
            key, subkey = jax.random.split(key)
            samples = _generate_mcmc(key, current_ts, force_finite=True)
            if isinstance(samples, dict):
                samples = samples["positions"]
            log_Z_t = None

        epoch_loss = 0.0

        for s in range(config.training.steps_per_epoch):
            # Subsample particles
            key, subkey = jax.random.split(key)
            samps = jax.random.choice(
                subkey, samples, (config.sampling.batch_size,), replace=False, axis=1
            )

            # Subsample time points
            key, subkey = jax.random.split(key)
            time_indices = jax.random.choice(
                subkey,
                jnp.arange(current_ts.shape[0]),
                (config.training.time_batch_size,),
                replace=False,
            )
            time_indices = jnp.sort(time_indices)  # Keep time points ordered
            batch_ts = current_ts[time_indices]
            batch_samples = samps[time_indices]

            if log_Z_t is not None:
                batch_log_Z_t = log_Z_t[time_indices]
            else:
                batch_log_Z_t = None

            v_theta, opt_state, loss = step(
                v_theta, opt_state, batch_samples, batch_ts, batch_log_Z_t
            )

            epoch_loss += loss
            if s % 20 == 0:
                if not config.offline:
                    wandb.log({"loss": loss})
                else:
                    print(f"Epoch {epoch}, Step {s}, Loss: {loss}")

        avg_loss = epoch_loss / config.training.steps_per_epoch
        if not config.offline:
            wandb.log(
                {"epoch": epoch, "average_loss": avg_loss, "end_time": current_end_time}
            )
        else:
            print(
                f"Epoch {epoch}, Average Loss: {avg_loss} Current End Time: {current_end_time}"
            )

        if epoch % config.training.eval_frequency == 0:
            eval_ts = jnp.linspace(
                0,
                current_end_time * 1.0 / config.sampling.num_timesteps,
                current_end_time,
            )
            key, subkey = jax.random.split(key)
            val_samples = generate_samples(
                subkey,
                v_theta,
                config.sampling.num_particles,
                eval_ts,
                path_distribution.sample_initial,
                integrator,
                config.density.shift_fn,
                use_shortcut=False,
            )

            # Evaluate samples using target density
            key, subkey = jax.random.split(key)
            eval_samples = jax.random.choice(
                subkey,
                val_samples["positions"][-1],
                (config.sampling.num_particles,),
                replace=False,
            )

            if target_density.TIME_DEPENDENT:
                eval_metrics = target_density.evaluate(eval_samples, float(eval_ts[-1]))
            else:
                eval_metrics = target_density.evaluate(eval_samples)

            # Log metrics to wandb
            if not config.offline:
                wandb.log(
                    {
                        f"validation_samples_{config.sampling.num_timesteps}_step": wandb.Image(
                            eval_metrics["figure"]
                        ),
                    }
                )
            else:
                plt.show()

            plt.close(eval_metrics["figure"])

    # Save trained model to wandb
    if not config.offline:
        eqx.tree_serialise_leaves("v_theta.eqx", v_theta)
        artifact = wandb.Artifact(
            name=f"velocity_field_model_{wandb.run.id}", type="model"
        )
        artifact.add_file(local_path="v_theta.eqx", name="model")
        artifact.save()

        wandb.log_artifact(artifact)
        wandb.finish()

    return v_theta
