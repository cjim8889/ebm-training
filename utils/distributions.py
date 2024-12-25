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
