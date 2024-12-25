from typing import Any, Callable, List, Optional

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

import wandb
from distributions import AnnealedDistribution, Target
from utils.distributions import (
    sample_monotonic_uniform_ordered,
)
from utils.hmc import (
    generate_samples_with_hmc_correction,
)
from utils.integration import (
    euler_integrate,
    generate_samples,
    generate_samples_with_different_ts,
)
from utils.optimization import get_optimizer, inverse_power_schedule

from .loss import shortcut_loss_fn


# Main training loop for training a velocity field with shortcut
def train_velocity_field_with_shortcut(
    key: jax.random.PRNGKey,
    initial_density: Target,
    target_density: Target,
    v_theta: Callable[[chex.Array, float], chex.Array],
    shift_fn: Callable[[chex.Array], chex.Array] = lambda x: x,
    N: int = 512,
    B: int = 256,
    C: int = 64,
    T: int = 32,
    enable_end_time_progression: bool = False,
    target_end_time: float = 1.0,  # Final target end time
    initial_end_time: float = 0.2,  # Starting end time
    end_time_steps: int = 5,  # Number of steps to reach target end time
    update_end_time_every: int = 200,  # Update frequency in epochs
    num_epochs: int = 200,
    num_steps: int = 100,
    learning_rate: float = 1e-03,
    gradient_norm: float = 1.0,
    mcmc_type: str = "hmc",
    num_mcmc_steps: int = 5,
    num_mcmc_integration_steps: int = 3,
    eta: float = 0.01,
    schedule: str = "inverse_power",
    continuous_schedule: bool = False,
    integrator: str = "euler",
    optimizer: str = "adamw",
    with_rejection_sampling: bool = False,
    eval_steps: List[int] = [4, 8, 16, 32],
    offline: bool = False,
    d_distribution: str = "uniform",
    target: str = "gmm",
    eval_every: int = 20,
    dt_log_density_clip: Optional[float] = None,
    debug: bool = False,
    **kwargs: Any,
) -> Any:
    path_distribution = AnnealedDistribution(
        initial_distribution=initial_density,
        target_distribution=target_density,
        dim=initial_density.dim,
    )

    # Initialize current end time
    if enable_end_time_progression:
        end_time_steps = jnp.linspace(initial_end_time, target_end_time, end_time_steps)
        current_end_time = float(end_time_steps[0])
        current_end_time_idx = -1
    else:
        current_end_time = target_end_time

    # Set up optimizer
    gradient_clipping = optax.clip_by_global_norm(gradient_norm)
    base_optimizer = get_optimizer(optimizer, learning_rate)
    optimizer = optax.chain(optax.zero_nans(), gradient_clipping, base_optimizer)
    optimizer: optax.GradientTransformation = optax.apply_if_finite(optimizer, 5)

    opt_state = optimizer.init(eqx.filter(v_theta, eqx.is_inexact_array))
    integrator = euler_integrate

    # Set up shortcut distribution
    if d_distribution == "log":
        d_dis = 1.0 / jnp.array(
            [2**e for e in range(int(jnp.floor(jnp.log2(128))) + 1)]
        )

    def _generate(key: jax.random.PRNGKey, force_finite: bool = False):
        if mcmc_type == "hmc":
            samples = generate_samples_with_hmc_correction(
                key=key,
                v_theta=v_theta,
                sample_fn=path_distribution.sample_initial,
                time_dependent_log_density=path_distribution.time_dependent_log_prob,
                num_samples=N,
                ts=current_ts,
                integration_fn=integrator,
                num_steps=num_mcmc_steps,
                integration_steps=num_mcmc_integration_steps,
                eta=eta,
                rejection_sampling=with_rejection_sampling,
                shift_fn=shift_fn,
                use_shortcut=True,
            )
        else:
            samples = generate_samples(
                key,
                v_theta,
                N,
                current_ts,
                path_distribution.sample_initial,
                integrator,
                shift_fn,
                use_shortcut=True,
            )

        if force_finite:
            samples = jnp.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)

        return samples

    @eqx.filter_jit
    def step(v_theta, opt_state, xs, cxs, ts, ds):
        loss, grads = eqx.filter_value_and_grad(shortcut_loss_fn)(
            v_theta,
            xs,
            cxs,
            ts,
            ds,
            time_derivative_log_density=path_distribution.time_derivative,
            score_fn=path_distribution.score_fn,
            shift_fn=shift_fn,
            dt_log_density_clip=dt_log_density_clip,
        )

        updates, opt_state = optimizer.update(grads, opt_state, v_theta)
        v_theta = eqx.apply_updates(v_theta, updates)

        return v_theta, opt_state, loss

    for epoch in range(num_epochs):
        # Update end time if needed
        if epoch % update_end_time_every == 0:
            if (
                enable_end_time_progression
                and current_end_time_idx < len(end_time_steps) - 1
            ):
                current_end_time_idx += 1
                current_end_time = float(end_time_steps[current_end_time_idx])

            # Update time steps based on new end time
            if schedule == "linear":
                current_ts = jnp.linspace(0, current_end_time, T)
            elif schedule == "inverse_power":
                current_ts = inverse_power_schedule(
                    T, end_time=current_end_time, gamma=0.5
                )

            if continuous_schedule:
                key, subkey = jax.random.split(key)
                current_ts = sample_monotonic_uniform_ordered(subkey, current_ts, True)

        key, subkey = jax.random.split(key)
        samples = _generate(key, force_finite=True)
        epoch_loss = 0.0

        key, subkey = jax.random.split(key)
        if d_distribution == "uniform":
            sampled_ds = jax.random.uniform(subkey, (T, C), minval=0.0, maxval=1.0)
        elif d_distribution == "log":
            sampled_ds = jax.random.choice(subkey, d_dis, (T, C), replace=True)

        for s in range(num_steps):
            key, subkey = jax.random.split(key)
            samps, samps_cxs = jnp.split(
                jax.random.choice(subkey, samples, (B + C,), replace=False, axis=1),
                [B],
                axis=1,
            )

            v_theta, opt_state, loss = step(
                v_theta, opt_state, samps, samps_cxs, current_ts, sampled_ds
            )

            epoch_loss += loss
            if s % 20 == 0:
                if not offline:
                    wandb.log({"loss": loss})
                else:
                    print(f"Epoch {epoch}, Step {s}, Loss: {loss}")

        avg_loss = epoch_loss / num_steps
        if not offline:
            wandb.log(
                {"epoch": epoch, "average_loss": avg_loss, "end_time": current_end_time}
            )
        else:
            print(
                f"Epoch {epoch}, Average Loss: {avg_loss} Current End Time: {current_end_time}"
            )

        if epoch % eval_every == 0:
            tss = [
                jnp.linspace(0, current_end_time, eval_step) for eval_step in eval_steps
            ]
            key, subkey = jax.random.split(key)
            val_samples = generate_samples_with_different_ts(
                subkey,
                v_theta,
                N,
                tss,
                path_distribution.sample_initial,
                integrator,
                shift_fn,
                use_shortcut=True,
            )

            for i, es in enumerate(eval_steps):
                if target == "gmm":
                    fig = target_density.visualise(val_samples[i][-1])
                    if not offline:
                        wandb.log(
                            {
                                f"validation_samples_{es}_step": wandb.Image(fig),
                            }
                        )
                    else:
                        plt.show()

                    plt.close(fig)

                elif target == "mw32":
                    key, subkey = jax.random.split(key)
                    fig = target_density.visualise(
                        jax.random.choice(
                            subkey, val_samples[i][-1], (100,), replace=False
                        )
                    )

                    if not offline:
                        wandb.log(
                            {
                                f"validation_samples_{es}_step": wandb.Image(fig),
                            }
                        )
                    else:
                        plt.show()

                    plt.close(fig)
                elif (
                    target == "dw4"
                    or target == "dw4o"
                    or target == "lj13"
                    or target == "sclj13"
                    or target == "tlj13"
                    or target == "lj13b"
                ):
                    key, subkey = jax.random.split(key)
                    fig = target_density.visualise(val_samples[i][-1][:1024])

                    if not offline:
                        wandb.log(
                            {
                                f"validation_samples_{es}_step": wandb.Image(fig),
                            }
                        )
                    else:
                        plt.show()

                    plt.close(fig)

    # Save trained model to wandb
    if not offline:
        eqx.tree_serialise_leaves("v_theta.eqx", v_theta)
        artifact = wandb.Artifact(
            name=f"velocity_field_model_{wandb.run.id}", type="model"
        )
        artifact.add_file(local_path="v_theta.eqx", name="model")
        artifact.save()

        wandb.log_artifact(artifact)
        wandb.finish()

    return v_theta
