from typing import Callable, Optional, Tuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from src.utils.distributions import (
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
        n_probes: Number of probes for Hutchinson's trick

    Returns:
        float: Local error in satisfying the Liouville equation
    """
    # Calculate divergence using appropriate method
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

    # Get score and calculate dot product with better numerical stability
    score = score_fn(x, t)
    v_dot_score = jnp.sum(v * score)  # element-wise multiply then sum is more stable
    
    # Calculate final result and handle NaN/inf values
    result = div_v + v_dot_score
    
    return jnp.nan_to_num(
        result,
        nan=0.0,
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
        score_fn: Score function
        use_control_variate: Whether to use control variate
        use_shortcut: Whether to use shortcut distance
        mp_policy: JAX Mixed Precision policy

    Returns:
        Estimate of log partition function
    """
    # Compute time derivative with appropriate precision
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

    # Perform weighted sum with better numerical stability
    result = jnp.sum(dt_log_unormalised_density * weights, axis=-1, keepdims=True)
    
    return jnp.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)


@eqx.filter_jit
def estimate_log_Z_t_online(
    xs: chex.Array,
    weights: chex.Array,
    ts: chex.Array,
    time_derivative_log_density: callable,
    v_theta: Optional[callable] = None,
    score_fn: Optional[callable] = None,
    use_control_variate: bool = False,
    use_shortcut: bool = False,
    prev_log_sum: Optional[chex.Array] = None,
    prev_count: Optional[int] = None,
) -> Tuple[chex.Array, chex.Array, int]:
    """
    Update an online estimate of log Z by combining the current batch estimate
    with previous batches.
    
    Returns:
        combined_log_Z: The updated log partition function estimate.
        new_sum: Updated accumulator for sum of Z estimates.
        new_count: Updated batch count.
    """
    # Compute the current batch's log Z estimate.
    batch_log_Z = estimate_log_Z_t(
        xs,
        weights,
        ts,
        time_derivative_log_density,
        v_theta=v_theta,
        score_fn=score_fn,
        use_control_variate=use_control_variate,
        use_shortcut=use_shortcut,
    )
    
    # If no previous accumulator exists, initialize.
    if prev_log_sum is None or prev_count is None:
        new_log_sum = batch_log_Z
        new_count = 1
    else:
        # Combine the previous log sum with the new batch's log Z.
        new_log_sum = jnp.logaddexp(prev_log_sum, batch_log_Z)
        new_count = prev_count + 1
    
    # The combined estimator is the log of the average:
    # log(mean Z) = log_sum - log(count)
    combined_log_Z = new_log_sum - jnp.log(new_count)
    
    return combined_log_Z, new_log_sum, new_count