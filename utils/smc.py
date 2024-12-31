from typing import Callable, Tuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from .hmc import sample_hamiltonian_monte_carlo
from blackjax.smc.ess import ess
from blackjax.smc.resampling import systematic


@jax.jit
def log_weights_to_weights(log_weights: jnp.ndarray) -> jnp.ndarray:
    """
    Convert log weights to weights.

    Args:
        log_weights: Log weights. Shape: (num_samples,).

    Returns:
        Weights: Weights. Shape: (num_samples,).
    """
    log_sum_w = jax.scipy.special.logsumexp(log_weights)
    log_normalized_weights = log_weights - log_sum_w
    weights = jnp.exp(log_normalized_weights)

    return weights


@eqx.filter_jit
def generate_samples_with_smc(
    key: jax.random.PRNGKey,
    time_dependent_log_density: Callable[[chex.Array, float], float],
    num_samples: int,
    ts: jnp.ndarray,
    sample_fn: Callable[[jax.random.PRNGKey, Tuple[int, ...]], jnp.ndarray],
    num_steps: int = 10,
    integration_steps: int = 3,
    eta: float = 0.1,
    rejection_sampling: bool = False,
    shift_fn: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
    ess_threshold: float = 0.5,
    resampling_fn: Callable[
        [jax.random.PRNGKey, jnp.ndarray, int], jnp.ndarray
    ] = systematic,
) -> jnp.ndarray:
    batched_shift_fn = jax.vmap(shift_fn)
    batched_hmc = jax.vmap(
        lambda key, x, t: sample_hamiltonian_monte_carlo(
            key,
            time_dependent_log_density,
            x,
            t,
            num_steps,
            integration_steps,
            eta,
            rejection_sampling,
            shift_fn,
        ),
        in_axes=(0, 0, None),
    )

    key, subkey = jax.random.split(key)
    initial_samples = sample_fn(subkey, (num_samples,))
    log_weights = jnp.full((num_samples,), -jnp.log(num_samples))
    sample_keys = jax.random.split(key, num_samples * ts.shape[0]).reshape(
        ts.shape[0], num_samples, -1
    )

    particles = {
        "positions": initial_samples,
        "log_weights": log_weights,
    }

    def _delta(positions, t_delta):
        # Prevent division by zero by adding a small epsilon
        log_density_ratio = time_dependent_log_density(
            positions, 1.0
        ) - time_dependent_log_density(positions, 0.0)
        return jnp.exp(log_density_ratio * t_delta)

    batched_delta = jax.vmap(_delta, in_axes=(0, None))

    def _resample(key, positions, log_weights):
        """
        Resample particles based on their log weights.

        Args:
            key: JAX PRNG key.
            positions: Current particle positions. Shape: (num_samples, ...).
            log_weights: Current log weights. Shape: (num_samples,).

        Returns:
            new_positions: Resampled particle positions. Shape: (num_samples, ...).
            new_log_weights: Reset log weights. Shape: (num_samples,).
        """
        # Normalize log_weights to prevent numerical underflow/overflow
        weights = log_weights_to_weights(log_weights)
        # Perform resampling to obtain indices
        indices = resampling_fn(key, weights, num_samples)  # Shape: (num_samples,)

        # Resample positions
        new_positions = jnp.take(positions, indices, axis=0)

        # Reset log_weights to uniform
        new_log_weights = jnp.log(jnp.full((num_samples,), 1.0 / num_samples))

        return new_positions, new_log_weights

    def step(carry, inputs):
        keys, t = inputs
        particles_prev, t_prev = carry

        prev_positions = particles_prev["positions"]
        prev_log_weights = particles_prev["log_weights"]

        d = t - t_prev

        # Apply shift function
        shifted_positions = batched_shift_fn(
            prev_positions
        )  # Shape: (num_samples, ...)

        # Apply HMC to propagate particles
        propagated_positions = batched_hmc(
            keys, shifted_positions, t
        )  # Shape: (num_samples, ...)

        # Compute incremental weights
        w_delta = batched_delta(propagated_positions, d)  # Shape: (num_samples,)

        # Update log weights in log space
        next_log_weights = prev_log_weights + jnp.log(
            w_delta + 1e-8
        )  # Shape: (num_samples,)

        ess_val = ess(log_weights=next_log_weights)  # Scalar
        ess_percentage = ess_val / num_samples  # Scalar

        # Define the condition for resampling
        def do_resample():
            # Resample particles
            new_positions, new_log_weights = _resample(
                keys[0], propagated_positions, next_log_weights
            )
            return {"positions": new_positions, "log_weights": new_log_weights}

        def do_nothing():
            # Keep the particles as is with normalized log weights
            log_weights_normalized = next_log_weights - jax.scipy.special.logsumexp(
                next_log_weights
            )
            return {
                "positions": propagated_positions,
                "log_weights": log_weights_normalized,
            }

        # Conditionally resample based on ESS percentage
        particles_new = jax.lax.cond(
            ess_percentage < ess_threshold,
            do_resample,
            do_nothing,
        )

        # Update time
        new_carry = (particles_new, t)

        # Output current particles
        return new_carry, particles_new

    # Perform the SMC over all time steps
    _, output = jax.lax.scan(
        step,
        (particles, 0.0),  # Initial carry: particles and initial time
        (sample_keys, ts),  # Inputs: resampled keys and time steps
    )

    weights = log_weights_to_weights(output["log_weights"])

    return {
        "positions": output["positions"],
        "weights": weights,
    }


@eqx.filter_jit
def generate_samples_with_euler_smc(
    key: jax.random.PRNGKey,
    v_theta: Callable[[jnp.ndarray, float], jnp.ndarray],
    time_dependent_log_density: Callable[[chex.Array, float], float],
    num_samples: int,
    ts: jnp.ndarray,
    sample_fn: Callable[[jax.random.PRNGKey, Tuple[int, ...]], jnp.ndarray],
    num_steps: int = 10,
    integration_steps: int = 3,
    eta: float = 0.1,
    rejection_sampling: bool = False,
    shift_fn: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
    use_shortcut: bool = False,
) -> jnp.ndarray:
    batched_shift_fn = jax.vmap(shift_fn)
    batched_hmc = jax.vmap(
        lambda key, x, t: sample_hamiltonian_monte_carlo(
            key,
            time_dependent_log_density,
            x,
            t,
            num_steps,
            integration_steps,
            eta,
            rejection_sampling,
            shift_fn,
        ),
        in_axes=(0, 0, None),
    )

    key, subkey = jax.random.split(key)
    initial_samples = sample_fn(subkey, (num_samples,))
    sample_keys = jax.random.split(key, num_samples * ts.shape[0]).reshape(
        ts.shape[0], num_samples, 2
    )

    def step(carry, xs):
        keys, t = xs

        x_prev, t_prev = carry
        d = t - t_prev

        if use_shortcut:
            samples = x_prev + d * jax.vmap(lambda x: v_theta(x, t, d))(x_prev)
        else:
            samples = x_prev + d * jax.vmap(lambda x: v_theta(x, t))(x_prev)

        samples = batched_shift_fn(samples)
        samples = batched_hmc(keys, samples, t)

        return (samples, t), samples

    _, output = jax.lax.scan(step, (initial_samples, 0.0), (sample_keys, ts))

    uniform_weights = jnp.full(
        (
            ts.shape[0],
            num_samples,
        ),
        1.0 / num_samples,
    )
    return {
        "positions": output,
        "weights": uniform_weights,
    }
