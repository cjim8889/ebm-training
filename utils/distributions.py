from typing import Callable

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import ott
import ot as pot


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


def compute_distances(x, n_particles, n_dimensions, min_dr=1e-8, repeat=True):
    x = x.reshape(n_particles, n_dimensions)

    # Get indices of upper triangular pairs
    i, j = jnp.triu_indices(n_particles, k=1)

    # Calculate displacements between pairs
    dx = x[i] - x[j]

    # Compute distances
    distances = optax.safe_norm(dx, axis=-1, min_norm=min_dr, ord=2)
    if repeat:
        # Repeat distances for each pair
        distances = jnp.repeat(distances, 2)

    return distances


@jax.jit
def compute_w2_distance(x, y):
    return ott.tools.unreg.wassdis_p(x=x, y=y, p=2.0)


def compute_w2_distance_pot(x, y):
    a = jnp.ones(x.shape[0]) / x.shape[0]
    b = jnp.ones(y.shape[0]) / y.shape[0]
    M = pot.dist(x, y)

    return jnp.sqrt(pot.emd2(a, b, M))


def compute_w2_distance_1d_pot(x, y):
    return pot.wasserstein_1d(x, y, p=2.0)


def compute_total_variation_distance(
    samples_p, samples_q, num_bins=200, lower_bound=-5.0, upper_bound=5.0
):
    """
    Compute the Total Variation (TV) distance between two distributions (p and q) in N-dimensional space.

    Args:
    samples_p: Array of shape (n_samples, d) representing samples from distribution P
    samples_q: Array of shape (n_samples, d) representing samples from distribution Q
    num_bins: Number of bins to use for histogram estimation per dimension
    lower_bound: Lower bound of the sample space for each dimension
    upper_bound: Upper bound of the sample space for each dimension

    Returns:
    TV distance: Scalar value representing the Total Variation distance
    """
    # Create bin edges for each dimension
    bin_edges = [
        jnp.linspace(lower_bound, upper_bound, num_bins + 1)
        for _ in range(samples_p.shape[1])
    ]

    # Compute histograms for both distributions (normalized)
    hist_p, _ = jnp.histogramdd(samples_p, bins=bin_edges, density=True)
    hist_q, _ = jnp.histogramdd(samples_q, bins=bin_edges, density=True)

    # Normalize histograms explicitly to ensure their sum is 1
    hist_p /= jnp.sum(hist_p)
    hist_q /= jnp.sum(hist_q)

    # Compute Total Variation distance as the half sum of absolute differences
    tv_distance = 0.5 * jnp.sum(jnp.abs(hist_p - hist_q))

    return tv_distance


compute_total_variation_distance = jax.jit(
    compute_total_variation_distance, static_argnums=(2, 3, 4)
)


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
def get_inverse_temperature(t, T_initial, T_final, method="geometric"):
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

    if method == "linear":
        # Linear interpolation in beta-space
        beta_t = beta_initial + t * (beta_final - beta_initial)
    elif method == "geometric":
        # Geometric interpolation in beta-space
        # Equivalent to logarithmic spacing
        log_beta_initial = jnp.log(beta_initial)
        log_beta_final = jnp.log(beta_final)
        log_beta_t = log_beta_initial + t * (log_beta_final - log_beta_initial)
        beta_t = jnp.exp(log_beta_t)
    else:
        raise ValueError(
            "Unsupported interpolation method. Choose 'linear' or 'geometric'."
        )

    return beta_t
