from typing import Callable

import chex
import equinox as eqx
import optax
import jax
import jax.numpy as jnp


def batched_remove_mean(x, n_particles, n_spatial_dim):
    x = x.reshape(-1, n_particles, n_spatial_dim)
    x = x - jnp.mean(x, axis=1, keepdims=True)
    return x.reshape(-1, n_particles * n_spatial_dim)


def remove_mean(x, n_particles, n_spatial_dim):
    x = x.reshape(n_particles, n_spatial_dim)
    x = x - jnp.mean(x, axis=0, keepdims=True)
    return x.reshape(n_particles * n_spatial_dim)


def remove_mean_decorator(step_fn, n_particles, n_spatial_dim):
    """Wraps HMC step to remove mean at each iteration"""

    def wrapped_step(key, state):
        state, info = step_fn(key, state)
        # Remove mean from position
        state = state._replace(
            position=remove_mean(
                state.position, n_particles=n_particles, n_spatial_dim=n_spatial_dim
            )
        )
        return state, info

    return wrapped_step


def compute_distances(x, n_particles, n_dimensions, min_dr=1e-8):
    x = x.reshape(n_particles, n_dimensions)

    # Get indices of upper triangular pairs
    i, j = jnp.triu_indices(n_particles, k=1)

    # Calculate displacements between pairs
    dx = x[i] - x[j]

    # Compute distances
    distances = optax.safe_norm(dx, axis=-1, min_norm=min_dr, ord=2)
    # Repeat distances for each pair
    distances = jnp.repeat(distances, 2)
    
    return distances


@eqx.filter_jit
def divergence_velocity_with_shortcut(
    v_theta: Callable[[chex.Array, float], chex.Array],
    x: chex.Array,
    t: float,
    d: float,
) -> float:
    def v_x(x):
        return v_theta(x, t, d)

    jacobian = jax.jacfwd(v_x)(x)
    div_v = jnp.trace(jacobian)
    return div_v


@eqx.filter_jit
def divergence_velocity(
    v_theta: Callable[[chex.Array, float], chex.Array],
    x: chex.Array,
    t: float,
) -> float:
    def v_x(x):
        return v_theta(x, t)

    jacobian = jax.jacfwd(v_x)(x)
    div_v = jnp.trace(jacobian)
    return div_v


@eqx.filter_jit
def sample_monotonic_uniform_ordered(
    key: jax.random.PRNGKey, bounds: chex.Array, include_endpoints: bool = True
) -> chex.Array:
    def step(carry, info):
        t_prev = carry
        t_current = info

        return t_current, jnp.array([t_prev, t_current])

    _, ordered_pairs = jax.lax.scan(step, bounds[0], bounds[1:])

    if include_endpoints:
        ordered_pairs = jnp.concatenate(
            [ordered_pairs, jnp.array([[1.0, 1.0]])], axis=0
        )

    samples = jax.random.uniform(
        key, bounds.shape, minval=ordered_pairs[:, 0], maxval=ordered_pairs[:, 1]
    )
    return samples

@eqx.filter_jit
def get_inverse_temperature(t, T_initial, T_final, method='geometric'):
    """
    Compute the inverse temperature beta(t) given a parameter t in [0,1],
    initial and final temperatures, and the interpolation method.

    Args:
        t (float or jnp.ndarray): Parameter in [0, 1] indicating the position along the tempering path.
        T_initial (float): Initial temperature at t=0.
        T_final (float): Final temperature at t=1.
        method (str): Interpolation method, either 'linear' or 'geometric'.

    Returns:
        float or jnp.ndarray: Computed inverse temperature beta(t).
    """
    # Ensure t is within [0,1]
    # t = jnp.clip(t, 0.0, 1.0)

    # Compute inverse temperatures
    beta_initial = 1.0 / T_initial
    beta_final = 1.0 / T_final

    if method == 'linear':
        # Linear interpolation in beta-space
        beta_t = beta_initial + t * (beta_final - beta_initial)
    elif method == 'geometric':
        # Geometric interpolation in beta-space
        # Equivalent to logarithmic spacing
        log_beta_initial = jnp.log(beta_initial)
        log_beta_final = jnp.log(beta_final)
        log_beta_t = log_beta_initial + t * (log_beta_final - log_beta_initial)
        beta_t = jnp.exp(log_beta_t)
    else:
        raise ValueError("Unsupported interpolation method. Choose 'linear' or 'geometric'.")

    return beta_t