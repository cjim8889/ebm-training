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
    best_metrics: List[Tuple[Tuple[float, ...], int]],
    model_version: int,
    target_density: Target,
) -> Tuple[List[Tuple[Tuple[float, ...], int]], int]:
    """Save model if it improves upon previous best metrics with hierarchical metrics support."""
    if not wandb.run:
        return best_metrics, model_version

    # Get metrics in order of descending importance
    largest_step_key = max(aggregated_metrics.keys())
    largest_step_metrics = aggregated_metrics[largest_step_key]

    # Extract metric values based on target specification
    metric_values = []
    for metric_spec in target_density.TARGET_METRIC:
        metric_name, lower_better = metric_spec
        value = largest_step_metrics.get(f"{metric_name}_mean", None)
        if value is None:
            return best_metrics, model_version  # If any metric is missing, abort
        metric_values.append((value, lower_better))

    # Convert to comparable tuple (flip values if higher is better)
    comparable_values = tuple(
        val if lower_better else -val for val, lower_better in metric_values
    )
    raw_values = tuple(val for val, _ in metric_values)

    # Check if this is better than any existing in top 10
    should_save = False
    if len(best_metrics) < 10:
        should_save = True
    else:
        # Get worst in current top 10 (since list is maintained in sorted order)
        worst_in_best = best_metrics[-1][0]
        if comparable_values < worst_in_best:
            should_save = True

    if should_save:
        model_version += 1
        model_name = f"velocity_field_model_{wandb.run.id}"

        # Create metric string for filename
        metric_str = "_".join(
            f"{name}_{val:.4f}"
            for (name, _), val in zip(target_density.TARGET_METRIC, raw_values)
        )
        model_path = f"{model_name}_v{model_version}_{metric_str}.eqx"

        eqx.tree_serialise_leaves(model_path, v_theta)

        # Create metadata with all metrics
        metadata = {
            "version": model_version,
            **{
                name: val
                for (name, _), val in zip(target_density.TARGET_METRIC, raw_values)
            },
        }

        artifact = wandb.Artifact(
            name=model_name,
            type="model",
            metadata=metadata,
        )
        artifact.add_file(local_path=model_path, name="model.eqx")

        # Insert and maintain sorted order
        best_metrics.append((comparable_values, model_version))
        best_metrics.sort()

        # Keep only top 10
        if len(best_metrics) > 10:
            best_metrics = best_metrics[:10]

        # Determine ranking
        better_models = sum(1 for m in best_metrics if m[0] < comparable_values)
        rank = better_models + 1  # Actual 1-based position in sorted list

        aliases = []
        if rank == 1:
            aliases.append("best")
        if rank < 5:
            aliases.append(f"top{rank}")

        wandb.log_artifact(artifact, aliases=aliases)

    return best_metrics, model_version
