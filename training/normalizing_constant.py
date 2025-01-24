from typing import Callable, Optional

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from utils.distributions import (
    divergence_velocity,
    divergence_velocity_with_shortcut,
    hutchinson_divergence_velocity2,
)


def control_variate_epsilon(
    v_theta: Callable[[chex.Array, float], chex.Array],
    x: chex.Array,
    t: float,
    score_fn: Callable[[chex.Array, float], chex.Array],
    d: Optional[chex.Array] = None,
    use_hutchinson: bool = False,
    key: Optional[jax.random.PRNGKey] = None,
    n_probes: int = 5,
) -> float:
    """Use control variate to reduce variance of the normalizing constant estimate.

    Args:
        v_theta: The velocity field function taking (x, t) and returning velocity vector
        x: The point at which to compute the error
        t: Current time
        score_fn: Score function taking (x, t) and returning gradient of log density
        d: Shortcut distance
        use_hutchinson: Whether to use Hutchinson's trick
        key: PRNG key for Hutchinson's trick

    Returns:
        float: Local error in satisfying the Liouville equation
    """

    if d is not None:
        if use_hutchinson:
            div_v = hutchinson_divergence_velocity2(
                key, v_theta, x, t, d=d, n_probes=n_probes
            )
        else:
            div_v = divergence_velocity_with_shortcut(v_theta, x, t, d=d)
        v = v_theta(x, t, d)
    else:
        if use_hutchinson:
            div_v = hutchinson_divergence_velocity2(
                key, v_theta, x, t, n_probes=n_probes
            )
        else:
            div_v = divergence_velocity(v_theta, x, t)
        v = v_theta(x, t)

    return jnp.nan_to_num(
        div_v + jnp.dot(v, score_fn(x, t)),
        posinf=1.0,
        neginf=-1.0,
    )


batched_control_variate_epsilon = jax.vmap(
    control_variate_epsilon, in_axes=(None, 0, None, None, None)
)
time_batched_control_variate_epsilon = eqx.filter_jit(
    jax.vmap(batched_control_variate_epsilon, in_axes=(None, 0, 0, None, None))
)


@eqx.filter_jit
def estimate_log_Z_t(
    xs: chex.Array,
    weights: chex.Array,
    ts: chex.Array,
    time_derivative_log_density: Callable[[chex.Array, float], float],
    v_theta: Callable[[chex.Array, float], chex.Array] = None,
    score_fn: Callable[[chex.Array, float], chex.Array] = None,
    use_control_variate: bool = False,
    use_shortcut: bool = False,
) -> chex.Array:
    """Estimate the log partition function using weighted samples.

    Args:
        xs: Samples from the distribution
        weights: Importance weights for the samples
        ts: Time points
        time_derivative_log_density: Function computing time derivative of log density
        v_theta: Velocity field function
        use_control_variate: Whether to use control variate
        use_shortcut: Whether to use shortcut distance

    Returns:
        Estimate of log partition function
    """
    dt_log_unormalised_density = jax.vmap(
        lambda xs, t: jax.vmap(lambda x: time_derivative_log_density(x, t))(xs),
        in_axes=(0, 0),
    )(xs, ts)

    if use_control_variate:
        if use_shortcut:
            d = jnp.diff(ts, axis=-1)[0]
        else:
            d = None

        epsilons = time_batched_control_variate_epsilon(v_theta, xs, ts, score_fn, d)
        dt_log_unormalised_density = dt_log_unormalised_density + epsilons

    return jnp.sum(dt_log_unormalised_density * weights, axis=-1, keepdims=True)
