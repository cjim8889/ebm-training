from typing import Callable, Optional

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from utils.distributions import divergence_velocity, divergence_velocity_with_shortcut


def shortcut_epsilon(
    v_theta: Callable[[chex.Array, float, float], chex.Array],
    x: chex.Array,
    dt_log_density: float,
    t: float,
    d: float,
    score_fn: Callable[[chex.Array, float], chex.Array],
) -> float:
    """Computes the local error in satisfying the Liouville equation.

    Args:
        v_theta: The velocity field function taking (x, t) and returning velocity vector
        x: The point at which to compute the error
        dt_log_density: Time derivative of log density at (x, t)
        t: Current time
        d: Time step
        score_fn: Score function taking (x, t) and returning gradient of log density

    Returns:
        float: Local error in satisfying the Liouville equation
    """
    score = score_fn(x, t)
    div_v = divergence_velocity_with_shortcut(v_theta, x, t, d)
    v = v_theta(x, t, d)
    lhs = div_v + jnp.dot(v, score)

    return jnp.nan_to_num(lhs + dt_log_density, nan=0.0, posinf=1.0, neginf=-1.0)


batched_shortcut_epsilon = jax.vmap(
    shortcut_epsilon, in_axes=(None, 0, 0, None, None, None)
)
time_batched_shortcut_epsilon = eqx.filter_jit(
    jax.vmap(batched_shortcut_epsilon, in_axes=(None, 0, 0, 0, 0, None))
)


def epsilon(
    v_theta: Callable[[chex.Array, float], chex.Array],
    x: chex.Array,
    dt_log_density: float,
    t: float,
    score_fn: Callable[[chex.Array, float], chex.Array],
) -> float:
    """Computes the local error in satisfying the Liouville equation.

    Args:
        v_theta: The velocity field function taking (x, t) and returning velocity vector
        x: The point at which to compute the error
        dt_log_density: Time derivative of log density at (x, t)
        t: Current time
        score_fn: Score function taking (x, t) and returning gradient of log density

    Returns:
        float: Local error in satisfying the Liouville equation
    """
    score = score_fn(x, t)
    div_v = divergence_velocity(v_theta, x, t)
    v = v_theta(x, t)
    lhs = div_v + jnp.dot(v, score)

    return jnp.nan_to_num(lhs + dt_log_density, nan=0.0, posinf=1.0, neginf=-1.0)


batched_epsilon = jax.vmap(epsilon, in_axes=(None, 0, 0, None, None))
time_batched_epsilon = eqx.filter_jit(
    jax.vmap(batched_epsilon, in_axes=(None, 0, 0, 0, None))
)


def shortcut(
    v_theta: Callable[[chex.Array, float, float], chex.Array],
    x: chex.Array,
    t: chex.Array,
    d: chex.Array,
    shift_fn: Callable[[chex.Array], chex.Array],
):
    real_d = (jnp.clip(t + 2 * d, 0.0, 1.0) - t) / 2.0

    s_t = v_theta(x, t, real_d)
    x_t = shift_fn(x + s_t * real_d)

    s_td = v_theta(x_t, t + real_d, real_d)
    s_target = jax.lax.stop_gradient(s_t + s_td) / 2.0

    error = (v_theta(x, t, 2 * real_d) - s_target) ** 2

    return error


batched_shortcut = jax.vmap(shortcut, in_axes=(None, 0, None, 0, None))


@eqx.filter_jit
def time_batched_shortcut_loss(
    v_theta: Callable[[chex.Array, float, float], chex.Array],
    xs: chex.Array,
    ts: chex.Array,
    ds: chex.Array,
    shift_fn: Callable[[chex.Array], chex.Array],
) -> chex.Array:
    return jnp.mean(
        jax.vmap(batched_shortcut, in_axes=(None, 0, 0, 0, None))(
            v_theta, xs, ts, ds, shift_fn
        )
    )


def shortcut_loss_fn(
    v_theta: Callable[[chex.Array, float, float], chex.Array],
    xs: chex.Array,
    cxs: chex.Array,
    ts: chex.Array,
    ds: chex.Array,
    time_derivative_log_density: Callable[[chex.Array, float], float],
    score_fn: Callable[[chex.Array, float], chex.Array],
    shift_fn: Callable[[chex.Array], chex.Array],
    dt_log_density_clip: Optional[float] = None,
) -> float:
    dt_log_unormalised_density = jax.vmap(
        lambda xs, t: jax.vmap(lambda x: time_derivative_log_density(x, t))(xs),
        in_axes=(0, 0),
    )(xs, ts)

    dt_log_density = jnp.nan_to_num(
        dt_log_unormalised_density
        - jnp.mean(dt_log_unormalised_density, axis=-1, keepdims=True),
        nan=0.0,
    )

    if dt_log_density_clip is not None:
        dt_log_density = jnp.clip(
            dt_log_density, -dt_log_density_clip, dt_log_density_clip
        )

    dss = jnp.diff(ts, append=1.0)
    epsilons = time_batched_shortcut_epsilon(
        v_theta, xs, dt_log_density, ts, dss, score_fn
    )

    short_cut_loss = time_batched_shortcut_loss(v_theta, cxs, ts, ds, shift_fn)

    return jnp.mean(epsilons**2) + short_cut_loss


def loss_fn(
    v_theta: Callable[[chex.Array, float], chex.Array],
    xs: chex.Array,
    ts: chex.Array,
    time_derivative_log_density: Callable[[chex.Array, float], float],
    score_fn: Callable[[chex.Array, float], chex.Array],
    dt_log_density_clip: Optional[float] = None,
) -> float:
    dt_log_unormalised_density = jax.vmap(
        lambda xs, t: jax.vmap(lambda x: time_derivative_log_density(x, t))(xs),
        in_axes=(0, 0),
    )(xs, ts)

    dt_log_density = jnp.nan_to_num(
        dt_log_unormalised_density
        - jnp.mean(dt_log_unormalised_density, axis=-1, keepdims=True),
        nan=0.0,
    )

    if dt_log_density_clip is not None:
        dt_log_density = jnp.clip(
            dt_log_density, -dt_log_density_clip, dt_log_density_clip
        )

    epsilons = time_batched_epsilon(v_theta, xs, dt_log_density, ts, score_fn)
    return jnp.mean(epsilons**2)
