from typing import Callable, Tuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from .hmc import sample_hamiltonian_monte_carlo


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

    return output
