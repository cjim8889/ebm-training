from typing import Callable, Tuple, Optional

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import ott
import ot as pot

from .ode import solve_neural_ode_diffrax


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
    # Compute squared Euclidean distance
    M = jnp.sum((x[:, None] - y[None, :]) ** 2, axis=-1)
    G = pot.emd(a, b, M)
    w2_dist = jnp.sqrt(jnp.sum(G * M))  # W2 is sqrt of OT cost for squared distance
    return w2_dist


def compute_wasserstein_distance_pot(x, y, num_itermax=1e7):
    a = jnp.ones(x.shape[0]) / x.shape[0]
    b = jnp.ones(y.shape[0]) / y.shape[0]
    M = pot.dist(x, y, metric="euclidean")
    M_sq = M**2

    w2_dist = jnp.sqrt(pot.emd2(a, b, M_sq, numItermax=num_itermax))
    w1_dist = pot.emd2(a, b, M, numItermax=num_itermax)

    return w1_dist, w2_dist


def compute_w2_sinkhorn_distance(x, y, reg=1.0, num_itermax=1e5):
    a = jnp.ones(x.shape[0]) / x.shape[0]
    b = jnp.ones(y.shape[0]) / y.shape[0]
    M = pot.dist(x, y)
    w2_dist = jnp.sqrt(
        pot.sinkhorn2(
            a, b, M, reg=reg, numItermax=int(num_itermax), method="sinkhorn_log"
        )
    )
    return w2_dist


def compute_w2_sinkhorn_distance_ot(x, y, reg=None, num_itermax=1e5):
    div, _ = ott.tools.sinkhorn_divergence.sinkdiv(
        x=x, y=y, epsilon=reg, cost_fn=ott.geometry.costs.SqEuclidean()
    )
    return jnp.sqrt(div)


compute_w2_sinkhorn_distance_ot = jax.jit(
    compute_w2_sinkhorn_distance_ot, static_argnums=(2, 3)
)


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
def hutchinson_divergence_velocity(
    key: jax.random.PRNGKey,
    v_theta: Callable,
    x: chex.Array,
    t: float,
    d: Optional[float] = None,
    n_probes: int = 1,
) -> chex.Array:
    """
    Hutchinson estimator for divergence with:
    - Rademacher probe vectors for reduced variance
    - Support for multiple probes via vmap
    - Optional shortcut parameter 'd'
    """
    # Generate multiple keys for probe vectors
    keys = jax.random.split(key, n_probes)

    def compute_probe(subkey):
        eps = jax.random.rademacher(subkey, shape=x.shape, dtype=jnp.float32)
        eps = jax.lax.stop_gradient(eps)
        # Handle both standard and shortcut cases
        if d is not None:
            v_fn = lambda x: v_theta(x, t, d)
        else:
            v_fn = lambda x: v_theta(x, t)

        # Efficient vector-Jacobian product
        _, f_vjp = jax.vjp(v_fn, x)
        jvp_eps = f_vjp(eps)[0]
        return jnp.sum(eps * jvp_eps)

    # Vectorize over probes and average results
    estimates = jax.vmap(compute_probe)(keys)
    return jnp.mean(estimates)


@eqx.filter_jit
def hutchinson_divergence_velocity2(
    v_theta: Callable,
    x: chex.Array,
    t: float,
    eps: chex.Array,
    d: Optional[float] = None,
) -> chex.Array:
    # Compute VJP once and reuse for all probes
    v_fn = lambda x: v_theta(x, t, d) if d is not None else v_theta(x, t)
    _, f_vjp = jax.vjp(v_fn, x)

    # Batched computation using vmap
    jvp_eps = jax.vmap(f_vjp)(eps)[0]  # [n_probes, ...]

    # Efficient trace estimation using Einstein summation
    estimates = jnp.einsum("i...,i...->i", eps, jvp_eps)

    return jnp.mean(estimates)


@eqx.filter_jit
def hutchinson_divergence_velocity_single_probe(
    v_theta: Callable,
    x: chex.Array,
    t: chex.Array,
    eps: chex.Array,  # Probe vector now passed as argument
    d: Optional[float] = None,
) -> chex.Array:
    """Compute divergence using single pre-defined probe vector."""
    # Validate probe vector shape
    chex.assert_shape(eps, x.shape)

    # Compute VJP once for the given probe
    v_fn = lambda x: v_theta(x, t, d) if d is not None else v_theta(x, t)
    _, f_vjp = jax.vjp(v_fn, x)

    # Direct computation without vmap (single probe)
    jvp_eps = f_vjp(eps)[0]

    # Simplified trace estimation for single probe
    trace_estimate = jnp.sum(eps * jvp_eps)

    return trace_estimate


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
        log_prob_prev = log_prob_next + dt * div_v_t  # Accumulate log_prob

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
    samples_rev, log_probs_q = solve_neural_ode_diffrax(
        v_theta=v_theta,
        y0=samples_p,
        ts=ts[::-1],
        log_p0=None,
        use_shortcut=use_shortcut,
        exact_logp=True,
        forward=False,
    )

    # Compute log q(x(T)) = log q(x(0)) + accumulated log_probs
    base_log_probs = base_log_prob_fn(samples_rev)  # Compute log q(x(0))
    log_q_x = base_log_probs + log_probs_q

    log_w = log_probs_p - log_q_x
    # Compute KL divergence: KL(p || q) = E_p[log p(x) - log q(x)]
    kl_divergence = jnp.mean(log_w)

    return kl_divergence


@jax.jit
def compute_log_effective_sample_size(
    log_p: chex.Array, log_q: chex.Array
) -> chex.Array:
    log_w = log_p - log_q
    log_sum_w = jax.scipy.special.logsumexp(log_w)
    log_ess = 2 * log_sum_w - jax.scipy.special.logsumexp(2.0 * log_w)
    return log_ess
