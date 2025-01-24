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


class Particle(eqx.Module):
    x: chex.Array
    t: chex.Array
    log_Z_t: chex.Array
    d: Optional[chex.Array] = None


@eqx.filter_jit
def epsilon(
    v_theta: Callable[[chex.Array, float, float], chex.Array]
    | Callable[[chex.Array, float], chex.Array],
    particle: Particle,
    score_fn: Callable[[chex.Array, float], chex.Array],
    time_derivative_log_density: Callable[[chex.Array, float], float],
) -> chex.Array:
    """Computes the local error using a Particle instance."""
    x, t, log_Z_t, d = particle.x, particle.t, particle.log_Z_t, particle.d
    dt_log_unormalised = time_derivative_log_density(x, t)
    dt_log_density = dt_log_unormalised - log_Z_t

    score = score_fn(x, t)

    if d is not None:
        div_v = divergence_velocity_with_shortcut(v_theta, x, t, d)
        v = v_theta(x, t, d)
    else:
        div_v = divergence_velocity(v_theta, x, t)
        v = v_theta(x, t)

    lhs = div_v + jnp.dot(v, score)
    return jnp.nan_to_num(lhs + dt_log_density, nan=0.0, posinf=1.0, neginf=-1.0)


batched_epsilon = jax.vmap(epsilon, in_axes=(None, 0, None, None))


@eqx.filter_jit
def epsilon_with_hutchinson(
    key: jax.random.PRNGKey,
    v_theta: Callable[[chex.Array, float, float], chex.Array],
    particle: Particle,
    score_fn: Callable[[chex.Array, float], chex.Array],
    time_derivative_log_density: Callable[[chex.Array, float], float],
    n_probes: int,
):
    x, t, log_Z_t, d = particle.x, particle.t, particle.log_Z_t, particle.d
    dt_log_unormalised = time_derivative_log_density(x, t)
    dt_log_density = dt_log_unormalised - log_Z_t

    score = score_fn(x, t)
    div_v = hutchinson_divergence_velocity2(key, v_theta, x, t, d=d, n_probes=n_probes)
    if d is not None:
        v = v_theta(x, t, d)
    else:
        v = v_theta(x, t)

    return jnp.nan_to_num(
        div_v + jnp.dot(v, score) + dt_log_density,
        posinf=1.0,
        neginf=-1.0,
    )


batched_epsilon_with_hutchinson = jax.vmap(
    epsilon_with_hutchinson, in_axes=(0, None, 0, None, None, None)
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


batched_shortcut = jax.vmap(shortcut, in_axes=(None, 0, 0, 0, None))


def loss_fn(
    v_theta: Callable[[chex.Array, float], chex.Array],
    particles: Particle,
    time_derivative_log_density: Callable[[chex.Array, float], float],
    score_fn: Callable[[chex.Array, float], chex.Array],
    shift_fn: Callable[[chex.Array], chex.Array] = lambda x: x,
    use_hutchinson: bool = False,
    key: Optional[jax.random.PRNGKey] = None,
    n_probes: int = 5,
) -> float:
    """Computes the loss for training the velocity field.

    Args:
        v_theta: The velocity field function taking (x, t) and returning velocity vector
        particles: Batch of particles, shape (batch_size, num_samples, dim)
        time_derivative_log_density: Function computing time derivative of log density
        score_fn: Score function taking (x, t) and returning gradient of log density
        shift_fn: Function taking (x) and returning shifted x

    Returns:
        float: Mean squared error in satisfying the Liouville equation
    """

    if use_hutchinson:
        keys = jax.random.split(key, particles.x.shape[0])
        epsilons = batched_epsilon_with_hutchinson(
            keys,
            v_theta,
            particles,
            score_fn,
            time_derivative_log_density,
            n_probes,
        )
    else:
        epsilons = batched_epsilon(
            v_theta, particles, score_fn, time_derivative_log_density
        )

    # Compute L1 and L2 loss for epsilons
    l1_loss = jnp.mean(jnp.abs(epsilons))  # L1 (MAE)
    l2_loss = jnp.mean(epsilons**2)  # L2 (MSE)
    combined_loss = 0.5 * l1_loss + 0.5 * l2_loss  # Adjust weights as needed

    if particles.d is not None:
        short_cut_loss = batched_shortcut(
            v_theta, particles.x, particles.t, particles.d, shift_fn
        )
        return combined_loss + 0.5 * jnp.mean(short_cut_loss)
    else:
        return combined_loss
