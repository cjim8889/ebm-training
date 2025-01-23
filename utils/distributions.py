from typing import Callable, Tuple

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
    M = pot.dist(x, y, metric="euclidean")

    G = pot.emd(a, b, M)
    w2_dist = jnp.sum(G * M) / G.sum()

    return w2_dist


def compute_wasserstein_distance_pot(x, y):
    a = jnp.ones(x.shape[0]) / x.shape[0]
    b = jnp.ones(y.shape[0]) / y.shape[0]
    M = pot.dist(x, y, metric="euclidean")
    M_sq = M**2

    w2_dist = jnp.sqrt(pot.emd2(a, b, M_sq))
    w1_dist = pot.emd2(a, b, M)

    return w1_dist, w2_dist


def compute_w2_distance_1d_pot(x, y):
    return pot.wasserstein_1d(x, y, p=2.0)


def compute_w1_distance_1d_pot(x, y):
    return pot.wasserstein_1d(x, y, p=1.0)


def compute_total_variation_distance(
    samples_p, samples_q, num_bins=200, lower_bound=-5.0, upper_bound=5.0
):
    """
    Compute the Total Variation (TV) distance between two distributions (p and q) in N-dimensional space.

    Args:
    samples_p: Array of shape (n_samples, d) representing samples from distribution P
    samples_q: Array of shape (n_samples, d) representing samples from distribution Q
    num_bins: Number of bins to use for histogram estimation per dimension
    lower_bound: Lower bound of the sample space for each dimension. If None, computed from data.
    upper_bound: Upper bound of the sample space for each dimension. If None, computed from data.

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
) -> chex.Array:
    def v_x(x):
        return v_theta(x, t, d)

    return jnp.trace(jax.jacfwd(v_x)(x))


@eqx.filter_jit
def divergence_velocity(
    v_theta: Callable[[chex.Array, float], chex.Array],
    x: chex.Array,
    t: float,
) -> float:
    def v_x(x):
        return v_theta(x, t)

    return jnp.trace(jax.jacfwd(v_x)(x))


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


@eqx.filter_jit
def reverse_time_flow(
    v_theta: Callable,
    final_samples: jnp.ndarray,
    final_time: float,
    ts: jnp.ndarray,
    use_shortcut: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Reverse ts to integrate backward
    ts_rev = ts[::-1]

    if use_shortcut:
        batched_v_theta = jax.vmap(v_theta, in_axes=(0, None, None))
    else:
        batched_v_theta = jax.vmap(v_theta, in_axes=(0, None))

    def step(carry, t):
        x_next, log_prob_next, t_next = carry

        dt = t - t_next  # dt is negative for backward integration
        if use_shortcut:
            v_t = batched_v_theta(x_next, t, jnp.abs(dt))
        else:
            v_t = batched_v_theta(x_next, t)
        x_prev = x_next + dt * v_t  # Since dt < 0, this moves backward

        # Compute divergence
        if use_shortcut:
            div_v_t = jax.vmap(
                lambda x: divergence_velocity_with_shortcut(v_theta, x, t, jnp.abs(dt))
            )(x_next)
        else:
            div_v_t = jax.vmap(lambda x: divergence_velocity(v_theta, x, t))(x_next)
        log_prob_prev = log_prob_next - dt * div_v_t  # Accumulate log_prob

        return (x_prev, log_prob_prev, t), None

    # Initialize carry with final samples and zero log-probabilities
    num_samples = final_samples.shape[0]
    initial_log_probs = jnp.zeros(num_samples)
    carry = (final_samples, initial_log_probs, final_time)

    (xs, log_probs, _), _ = jax.lax.scan(step, carry, ts_rev)

    return xs, log_probs


@eqx.filter_jit
def estimate_kl_divergence(
    v_theta: Callable[[chex.Array, float, float], chex.Array],
    num_samples: int,
    key: jax.random.PRNGKey,
    ts: chex.Array,
    log_prob_p_fn: Callable[[chex.Array], chex.Array],
    sample_p_fn: Callable[[jax.random.PRNGKey, int], chex.Array],
    base_log_prob_fn: Callable[[chex.Array], chex.Array],
    final_time: float = 1.0,
    use_shortcut: bool = False,
) -> chex.Array:
    # Generate samples from p(x)
    samples_p = sample_p_fn(key, (num_samples,))
    log_probs_p = log_prob_p_fn(samples_p)  # Compute log p(x) for these samples

    # Perform reverse-time integration to compute samples and log probabilities under q(x)
    samples_rev, log_probs_q = reverse_time_flow(
        v_theta, samples_p, final_time, ts, use_shortcut
    )

    # Compute log q(x(T)) = log q(x(0)) + accumulated log_probs
    base_log_probs = base_log_prob_fn(samples_rev)  # Compute log q(x(0))
    log_q_x = base_log_probs - log_probs_q

    log_w = log_probs_p - log_q_x
    # Compute KL divergence: KL(p || q) = E_p[log p(x) - log q(x)]
    kl_divergence = jnp.mean(log_w)

    return kl_divergence


@jax.jit
def compute_log_effective_sample_size(
    log_p: chex.Array, log_q: chex.Array
) -> chex.Array:
    """
    Compute the log Effective Sample Size (log ESS Fraction) given samples from `q` and log-probability functions.

    **Parameters:**


    **Returns:**
        Array: Scalar representing log(ESS / N).
    """
    # Ensure shapes match
    if log_p.shape != log_q.shape:
        raise ValueError(
            f"Shape mismatch: log_p has shape {log_p.shape}, log_q has shape {log_q.shape}."
        )

    log_w = log_p - log_q  # Log importance weights (unnormalized)
    log_sum_w = jax.scipy.special.logsumexp(log_w)  # log(Σ exp(log_w))
    log_sum_w_sq = jax.scipy.special.logsumexp(2.0 * log_w)  # log(Σ exp(2*log_w))
    log_ess_frac = 2.0 * log_sum_w - log_sum_w_sq

    return log_ess_frac
