from typing import Callable, List, Tuple, Dict

import equinox as eqx
import jax
import jax.numpy as jnp
import diffrax

from .distributions import divergence_velocity, divergence_velocity_with_shortcut
from .ode import solve_neural_ode_diffrax


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

    _, output = jax.lax.scan(step, (initial_samples, ts[0]), ts)
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
) -> Dict[str, jnp.ndarray]:
    key, subkey = jax.random.split(key)
    initial_samples = sample_fn(subkey, (num_samples,))
    samples = integration_fn(v_theta, initial_samples, ts, shift_fn, use_shortcut)
    weights = jnp.ones((ts.shape[0], num_samples)) / num_samples

    return {
        "positions": samples,
        "weights": weights,
    }


@eqx.filter_jit
def generate_samples_with_diffrax(
    key: jax.random.PRNGKey,
    v_theta: Callable[[jnp.ndarray, float], jnp.ndarray],
    num_samples: int,
    ts: jnp.ndarray,
    sample_fn: Callable[[jax.random.PRNGKey, Tuple[int, ...]], jnp.ndarray],
    use_shortcut: bool = False,
    solver: str = "Euler",
) -> Dict[str, jnp.ndarray]:
    initial_samples = sample_fn(key, (num_samples,))
    final_samples, _ = solve_neural_ode_diffrax(
        v_theta=v_theta,
        y0=initial_samples,
        ts=ts,
        use_shortcut=use_shortcut,
        exact_logp=True,
        forward=True,
        save_trajectory=True,
        solver=diffrax.Tsit5() if solver == "Tsit5" else diffrax.Euler(),
    )
    return {
        "positions": final_samples,
        "weights": jnp.ones((ts.shape[0], num_samples)) / num_samples,
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


@eqx.filter_jit
def generate_samples_with_log_prob_diffrax(
    v_theta: Callable,
    initial_samples: jnp.ndarray,
    initial_log_probs: jnp.ndarray,
    ts: jnp.ndarray,
    use_shortcut: bool = False,
    solver: str = "Euler",
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    final_samples, final_log_probs = solve_neural_ode_diffrax(
        v_theta=v_theta,
        y0=initial_samples,
        ts=ts,
        log_p0=initial_log_probs,
        use_shortcut=use_shortcut,
        exact_logp=True,
        forward=True,
        solver=diffrax.Tsit5() if solver == "Tsit5" else diffrax.Euler(),
    )
    return final_samples, final_log_probs