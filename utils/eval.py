from typing import Any, Callable, Dict, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import wandb
from distributions import AnnealedDistribution, Target
from training.config import TrainingExperimentConfig


def evaluate_model(
    key: jax.random.PRNGKey,
    v_theta: Callable,
    config: TrainingExperimentConfig,
    path_distribution: AnnealedDistribution,
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

        for i, es in enumerate(config.training.shortcut_size):
            key, eval_key = jax.random.split(key)
            eval_metrics = target_density.evaluate(
                eval_key,
                use_shortcut=config.training.use_shortcut,
                ts=eval_ts[i],
                v_theta=v_theta,
                base_density=path_distribution.initial_density,
            )
            total_eval_metrics[f"validation_{es}_step"] = eval_metrics
    else:
        eval_ts = jnp.linspace(
            0,
            current_end_time * 1.0 / config.sampling.num_timesteps,
            current_end_time,
        )

        key, eval_key = jax.random.split(key)
        eval_metrics = target_density.evaluate(
            eval_key,
            use_shortcut=config.training.use_shortcut,
            ts=eval_ts,
            v_theta=v_theta,
            base_density=path_distribution.initial_density,
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
    best_metrics: List[Tuple[float, int]],
    model_version: int,
    target_density: Target,
) -> Tuple[List[Tuple[float, int]], int]:
    """Save model if it improves upon previous best metrics."""
    if not wandb.run:
        return best_metrics, model_version

    largest_step_key = max(aggregated_metrics.keys())
    largest_step_metrics = aggregated_metrics[largest_step_key]
    current_metric = largest_step_metrics.get(target_density.TARGET_METRIC, None)

    if current_metric is None:
        return best_metrics, model_version

    should_save = False
    if len(best_metrics) < 3:
        should_save = True
    elif current_metric < max(w2 for w2, _ in best_metrics):
        should_save = True

    if should_save:
        model_version += 1
        model_name = f"velocity_field_model_{wandb.run.id}"
        metric_name = target_density.TARGET_METRIC.replace("_mean", "")
        model_path = (
            f"{model_name}_v{model_version}_{metric_name}_{current_metric:.4f}.eqx"
        )

        eqx.tree_serialise_leaves(model_path, v_theta)
        artifact = wandb.Artifact(
            name=model_name,
            type="model",
            metadata={
                metric_name: current_metric,
                "version": model_version,
            },
        )
        artifact.add_file(local_path=model_path, name="model.eqx")

        best_metrics.append((current_metric, model_version))
        best_metrics.sort()
        if len(best_metrics) > 3:
            best_metrics.pop()

        rank = len([m for m, _ in best_metrics if m <= current_metric])
        aliases = [f"top{rank}"] if rank <= 3 else []
        if rank == 1:
            aliases.append("best")
        wandb.log_artifact(artifact, aliases=aliases)

    return best_metrics, model_version
