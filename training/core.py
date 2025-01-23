from typing import Any, Callable, Dict, List, Tuple

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
    generate_samples_with_smc,
)
from utils.integration import (
    euler_integrate,
    generate_samples,
    generate_samples_with_different_ts,
)
from utils.optimization import get_optimizer, inverse_power_schedule, power_schedule
from .config import TrainingExperimentConfig
from .loss import loss_fn, Particle
from .normalizing_constant import estimate_log_Z_t


def evaluate_model(
    key: jax.random.PRNGKey,
    v_theta: Callable,
    config: TrainingExperimentConfig,
    path_distribution: AnnealedDistribution,
    integrator: Callable,
    target_density: Target,
    current_end_time: int,
) -> Dict[str, Any]:
    """Run a single evaluation pass and return metrics."""
    total_eval_metrics = {}

    if config.training.use_shortcut:
        eval_ts = [
            jnp.linspace(0, 1.0, eval_step)
            for eval_step in config.training.shortcut_size
        ]
        key, subkey = jax.random.split(key)
        val_samples = generate_samples_with_different_ts(
            subkey,
            v_theta,
            config.sampling.num_particles,
            eval_ts,
            path_distribution.sample_initial,
            integrator,
            config.density.shift_fn,
            use_shortcut=config.training.use_shortcut,
        )

        for i, es in enumerate(config.training.shortcut_size):
            eval_samples = val_samples[i][-1]
            key, subkey = jax.random.split(key)
            eval_metrics = target_density.evaluate(
                subkey,
                eval_samples,
                time=float(eval_ts[i][-1]),
                use_shortcut=config.training.use_shortcut,
                ts=eval_ts[i],
                v_theta=v_theta,
                base_log_prob_fn=path_distribution.base_log_prob,
                base_sample_fn=path_distribution.sample_initial,
            )
            total_eval_metrics[f"validation_{es}_step"] = eval_metrics
    else:
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
            use_shortcut=config.training.use_shortcut,
        )
        eval_samples = val_samples["positions"][-1]
        key, subkey = jax.random.split(key)
        eval_metrics = target_density.evaluate(
            subkey,
            eval_samples,
            time=float(eval_ts[-1]),
            use_shortcut=config.training.use_shortcut,
            ts=eval_ts,
            v_theta=v_theta,
            base_log_prob_fn=path_distribution.base_log_prob,
            base_sample_fn=path_distribution.sample_initial,
        )
        total_eval_metrics[f"validation_{config.sampling.num_timesteps}_step"] = (
            eval_metrics
        )

    return total_eval_metrics


def aggregate_eval_metrics(
    all_eval_results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate metrics across multiple evaluation runs with proper figure cleanup."""
    aggregated_metrics = {}

    for step_key in all_eval_results[0].keys():
        step_metrics = [result[step_key] for result in all_eval_results]
        agg_metrics = {}
        figures = []

        # Collect metrics and figures from all runs
        for run_idx, metrics in enumerate(step_metrics):
            # Collect figures for later cleanup
            if "figure" in metrics:
                figures.append(metrics["figure"])

            # Process numerical metrics
            for metric_name in metrics.keys():
                if metric_name == "figure":
                    continue

                # Initialize storage if first run
                if run_idx == 0:
                    agg_metrics[f"{metric_name}_mean"] = []
                    agg_metrics[f"{metric_name}_var"] = []

                # Collect values
                agg_metrics[f"{metric_name}_mean"].append(metrics[metric_name])

        # Close all but last figure
        for fig in figures[:-1]:
            plt.close(fig)

        # Add last figure to metrics if exists
        if figures:
            agg_metrics["figure"] = figures[-1]

        # Calculate final mean/var for numerical metrics
        for metric_name in list(agg_metrics.keys()):
            if "_mean" in metric_name:
                base_name = metric_name.replace("_mean", "")
                values = jnp.array(agg_metrics.pop(metric_name))
                agg_metrics[f"{base_name}_mean"] = jnp.mean(values)
                agg_metrics[f"{base_name}_var"] = jnp.var(values)

        aggregated_metrics[step_key] = agg_metrics

    return aggregated_metrics


def log_metrics(
    aggregated_metrics: Dict[str, Dict[str, Any]], config: TrainingExperimentConfig
):
    """Handle metric logging to appropriate destinations."""
    if not config.offline:
        for step_key, metrics in aggregated_metrics.items():
            figure = metrics.pop("figure", None)
            prefixed_metrics = {f"{step_key}/{k}": v for k, v in metrics.items()}
            if figure is not None:
                prefixed_metrics[f"{step_key}/figure"] = wandb.Image(figure)
            wandb.log(prefixed_metrics)
            plt.close(figure)
    else:
        for step_key, metrics in aggregated_metrics.items():
            figure = metrics.pop("figure", None)
            print(f"Evaluation results for {step_key}:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value}")
            if figure is not None:
                plt.show()
                plt.close(figure)


def save_model_if_best(
    v_theta: Any,
    aggregated_metrics: Dict[str, Dict[str, Any]],
    best_w2_distances: List[Tuple[float, int]],
    model_version: int,
) -> Tuple[List[Tuple[float, int]], int]:
    """Save model if it improves upon previous best metrics."""
    if not wandb.run:
        return best_w2_distances, model_version

    largest_step_key = max(aggregated_metrics.keys())
    largest_step_metrics = aggregated_metrics[largest_step_key]
    current_w2 = largest_step_metrics.get("w2_distance_mean", None)

    if current_w2 is None:
        return best_w2_distances, model_version

    should_save = False
    if len(best_w2_distances) < 3:
        should_save = True
    elif current_w2 < max(w2 for w2, _ in best_w2_distances):
        should_save = True

    if should_save:
        model_version += 1
        model_name = f"velocity_field_model_{wandb.run.id}"
        model_path = f"{model_name}_v{model_version}_w2_{current_w2:.4f}.eqx"

        eqx.tree_serialise_leaves(model_path, v_theta)
        artifact = wandb.Artifact(
            name=model_name,
            type="model",
            metadata={
                "w2_distance": current_w2,
                "version": model_version,
            },
        )
        artifact.add_file(local_path=model_path, name="model.eqx")

        best_w2_distances.append((current_w2, model_version))
        best_w2_distances.sort()
        if len(best_w2_distances) > 3:
            best_w2_distances.pop()

        rank = len([w2 for w2, _ in best_w2_distances if w2 <= current_w2])
        aliases = [f"top{rank}"] if rank <= 3 else []
        if rank == 1:
            aliases.append("best")
        wandb.log_artifact(artifact, aliases=aliases)

    return best_w2_distances, model_version


def train_velocity_field(
    key: jax.random.PRNGKey,
    initial_density: Target,
    target_density: Target,
    v_theta: Callable[[chex.Array, float], chex.Array],
    config: TrainingExperimentConfig,
) -> Any:
    """Train a velocity field using either standard or decoupled loss function."""
    best_w2_distances = []
    model_version = 0

    path_distribution = AnnealedDistribution(
        initial_density=initial_density,
        target_density=target_density,
        method=config.density.annealing_path,
    )

    current_end_time = config.sampling.num_timesteps

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

    # Optimizer setup
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
            use_shortcut=config.training.use_shortcut,
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
                use_shortcut=config.training.use_shortcut,
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
                use_shortcut=config.training.use_shortcut,
            )
        else:
            samples = _generate(key, ts, force_finite)

        if force_finite:
            samples["positions"] = jnp.nan_to_num(
                samples["positions"], nan=0.0, posinf=1.0, neginf=-1.0
            )
        return samples

    @eqx.filter_jit
    def step(v_theta, opt_state, particles):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(
            v_theta,
            particles,
            path_distribution.time_derivative,
            path_distribution.score_fn,
            config.density.shift_fn,
        )
        updates, opt_state = optimizer.update(grads, opt_state, v_theta)
        v_theta = eqx.apply_updates(v_theta, updates)
        return v_theta, opt_state, loss

    shortcut_size_d = 1.0 / jnp.array(config.training.shortcut_size)

    for epoch in range(config.training.num_epochs):
        # Handle time steps for this epoch
        if config.integration.continuous_time:
            key, subkey = jax.random.split(key)
            current_ts = sample_monotonic_uniform_ordered(subkey, base_ts, True)
        else:
            current_ts = base_ts

        # Sample generation
        if config.training.use_decoupled_loss:
            key, subkey = jax.random.split(key)
            mcmc_samples = _generate_mcmc(subkey, current_ts, force_finite=True)
            key, subkey = jax.random.split(key)
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
            samples = jnp.concatenate(
                [mcmc_samples["positions"], v_theta_samples["positions"]], axis=1
            )
        else:
            key, subkey = jax.random.split(key)
            samples = _generate_mcmc(key, current_ts, force_finite=True)
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
            v_theta, opt_state, loss = step(v_theta, opt_state, training_particles)
            epoch_loss += loss
            if s % 20 == 0:
                if not config.offline:
                    wandb.log({"loss": loss})
                else:
                    print(f"Epoch {epoch}, Step {s}, Loss: {loss}")

        avg_loss = epoch_loss / config.training.steps_per_epoch
        if not config.offline:
            wandb.log({"epoch": epoch, "average_loss": avg_loss})
        else:
            print(f"Epoch {epoch}, Average Loss: {avg_loss}")

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
                    euler_integrate,
                    target_density,
                    current_end_time,
                )
                all_eval_results.append(eval_metrics)

            # Process and log metrics
            aggregated_metrics = aggregate_eval_metrics(all_eval_results)
            log_metrics(aggregated_metrics, config)

            # Handle model saving
            if not config.offline:
                best_w2_distances, model_version = save_model_if_best(
                    v_theta, aggregated_metrics, best_w2_distances, model_version
                )

    # Save final model state
    if not config.offline and len(best_w2_distances) > 0:
        # Log summary of best models
        wandb.run.summary["best_w2_distances"] = [w2 for w2, _ in best_w2_distances]
        wandb.run.summary["best_model_versions"] = [ver for _, ver in best_w2_distances]
        wandb.finish()

    return v_theta
