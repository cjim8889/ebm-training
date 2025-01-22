from typing import Callable, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .distributions import divergence_velocity, divergence_velocity_with_shortcut


@eqx.filter_jit
def euler_integrate(
    v_theta: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    initial_samples: jnp.ndarray,
    ts: jnp.ndarray,
    shift_fn: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
    use_shortcut: bool = False,
) -> jnp.ndarray:
    batched_shift_fn = jax.vmap(shift_fn)

    def step(carry, t):
        x_prev, t_prev = carry
        d = t - t_prev

        if use_shortcut:
            samples = x_prev + d * jax.vmap(lambda x: v_theta(x, t, d))(x_prev)
        else:
            samples = x_prev + d * jax.vmap(lambda x: v_theta(x, t))(x_prev)

        samples = batched_shift_fn(samples)

        return (samples, t), samples

    _, output = jax.lax.scan(step, (initial_samples, 0.0), ts)
    return output


@eqx.filter_jit
def generate_samples(
    key: jax.random.PRNGKey,
    v_theta: Callable[[jnp.ndarray, float], jnp.ndarray],
    num_samples: int,
    ts: jnp.ndarray,
    sample_fn: Callable[[jax.random.PRNGKey, Tuple[int, ...]], jnp.ndarray],
    integration_fn: Callable[
        [Callable[[jnp.ndarray, float], jnp.ndarray], jnp.ndarray, jnp.ndarray],
        jnp.ndarray,
    ] = euler_integrate,
    shift_fn: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
    use_shortcut: bool = False,
) -> jnp.ndarray:
    key, subkey = jax.random.split(key)
    initial_samples = sample_fn(subkey, (num_samples,))
    samples = integration_fn(v_theta, initial_samples, ts, shift_fn, use_shortcut)
    weights = jnp.ones((ts.shape[0], num_samples)) / num_samples

    return {
        "positions": samples,
        "weights": weights,
    }


@eqx.filter_jit
def generate_samples_with_initial_values(
    v_theta: Callable[[jnp.ndarray, float], jnp.ndarray],
    initial_samples: jnp.ndarray,
    ts: jnp.ndarray,
    integration_fn: Callable[
        [Callable[[jnp.ndarray, float], jnp.ndarray], jnp.ndarray, jnp.ndarray],
        jnp.ndarray,
    ] = euler_integrate,
    shift_fn: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
    use_shortcut: bool = False,
):
    samples = integration_fn(v_theta, initial_samples, ts, shift_fn, use_shortcut)
    return samples


@eqx.filter_jit
def generate_samples_with_different_ts(
    key: jax.random.PRNGKey,
    v_theta: Callable[[jnp.ndarray, float], jnp.ndarray],
    num_samples: int,
    tss: List[jnp.ndarray],
    sample_fn: Callable[[jax.random.PRNGKey, Tuple[int, ...]], jnp.ndarray],
    integration_fn: Callable[
        [Callable[[jnp.ndarray, float], jnp.ndarray], jnp.ndarray, jnp.ndarray],
        jnp.ndarray,
    ] = euler_integrate,
    shift_fn: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
    use_shortcut: bool = False,
) -> jnp.ndarray:
    key, subkey = jax.random.split(key)
    initial_samples = sample_fn(subkey, (num_samples,))

    samples = [
        integration_fn(v_theta, initial_samples, ts, shift_fn, use_shortcut)
        for ts in tss
    ]
    return samples


@eqx.filter_jit
def generate_samples_with_log_prob(
    v_theta: Callable,
    initial_samples: jnp.ndarray,
    initial_log_probs: jnp.ndarray,
    ts: jnp.ndarray,
    use_shortcut: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if use_shortcut:
        batched_v_theta = jax.vmap(v_theta, in_axes=(0, None, None))
    else:
        batched_v_theta = jax.vmap(v_theta, in_axes=(0, None))

    def step(carry, t):
        x_prev, log_prob_prev, t_prev = carry

        dt = t - t_prev
        if use_shortcut:
            v_t = batched_v_theta(x_prev, t, dt)
        else:
            v_t = batched_v_theta(x_prev, t)

        x_next = x_prev + dt * v_t

        # Compute divergence
        if use_shortcut:
            div_v_t = jax.vmap(
                lambda x: divergence_velocity_with_shortcut(v_theta, x, t, jnp.abs(dt))
            )(x_prev)
        else:
            div_v_t = jax.vmap(lambda x: divergence_velocity(v_theta, x, t))(x_prev)
        log_prob_next = log_prob_prev - dt * div_v_t

        return (x_next, log_prob_next, t), None

    # Initialize carry with initial samples and their log-probabilities
    carry = (initial_samples, initial_log_probs, 0.0)
    (samples, log_probs, _), _ = jax.lax.scan(step, carry, ts)

    return samples, log_probs
