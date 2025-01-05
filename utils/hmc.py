from typing import Callable, Tuple, Optional
import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from .integration import (
    euler_integrate,
    generate_samples,
    generate_samples_with_initial_values,
)


@eqx.filter_jit
def sample_hamiltonian_monte_carlo(
    key: jax.random.PRNGKey,
    time_dependent_log_density: Callable[[chex.Array, float], float],  # (dim,) -> float
    x: chex.Array,  # (dim,)
    t: float,
    num_steps: int = 10,
    integration_steps: int = 3,
    eta: float = 0.1,
    rejection_sampling: bool = False,
    shift_fn: Callable[[chex.Array], chex.Array] = lambda x: x,  # (dim,) -> (dim,)
    covariance: Optional[chex.Array] = None,  # (dim,) or (dim, dim)
) -> chex.Array:  # (dim,)
    dim = x.shape[-1]

    # Handle covariance cases
    if covariance is None:
        # Identity covariance case
        sqrt_cov = inv_sqrt_cov = jnp.ones((dim,))
        is_diagonal = True
    else:
        is_diagonal = covariance.ndim == 1
        if is_diagonal:
            # Diagonal case - already in vector form
            sqrt_cov = jnp.sqrt(covariance)
            inv_sqrt_cov = 1.0 / sqrt_cov
        else:
            # Full matrix case
            reg_cov = covariance + 1e-6 * jnp.eye(dim)
            sqrt_cov = jnp.linalg.cholesky(reg_cov)
            inv_sqrt_cov = jax.scipy.linalg.solve_triangular(
                sqrt_cov, jnp.eye(dim), lower=True
            )

    # Pre-compute operation functions based on covariance type
    if is_diagonal:

        def momentum_op(v):
            return inv_sqrt_cov * v

        def sample_momentum(z):
            return sqrt_cov * z
    else:

        def momentum_op(v):
            return inv_sqrt_cov @ v

        def sample_momentum(z):
            return sqrt_cov @ z

    grad_log_prob = jax.grad(lambda x: time_dependent_log_density(x, t))

    def kinetic_energy(v):
        scaled_v = momentum_op(v)
        return 0.5 * jnp.sum(scaled_v**2)

    def hamiltonian(x, v):
        return -time_dependent_log_density(x, t) + kinetic_energy(v)

    def integration_step(carry, _):
        x, v = carry
        # Apply momentum operation once and reuse result
        scaled_v = momentum_op(momentum_op(v))
        x = x + eta * scaled_v
        x = shift_fn(x)
        v = v + eta * grad_log_prob(x)
        return (x, v), _

    def hmc_step(x_current, inputs):
        x = x_current
        v, accept_key = inputs

        current_h = hamiltonian(x, v)
        v = v + 0.5 * eta * grad_log_prob(x)

        # Leapfrog integration
        (x, v), _ = jax.lax.scan(
            integration_step, (x, v), None, length=integration_steps, unroll=4
        )

        v = v + 0.5 * eta * grad_log_prob(x)
        x = shift_fn(x)

        if rejection_sampling:
            proposed_h = hamiltonian(x, v)
            accept_ratio = jnp.minimum(1.0, jnp.exp(current_h - proposed_h))
            accept = jax.random.uniform(accept_key) < accept_ratio
            return jax.lax.cond(
                accept, lambda _: x, lambda _: x_current, operand=None
            ), None
        else:
            return x, None

    # Generate all momentum samples at once
    momentum_key, accept_keys = jax.random.split(key)
    accept_keys = jax.random.split(accept_keys, num_steps)
    z = jax.random.normal(momentum_key, (num_steps, dim))
    vs = jax.vmap(sample_momentum)(z)

    # Run the chain with pre-sampled momentum
    final_x, _ = jax.lax.scan(hmc_step, x, (vs, accept_keys))
    return final_x


@eqx.filter_jit
def time_batched_sample_hamiltonian_monte_carlo(
    key: jax.random.PRNGKey,
    time_dependent_log_density: Callable[[chex.Array, float], float],  # (dim,) -> float
    xs: chex.Array,  # (time, batch, dim)
    ts: chex.Array,  # (time,)
    num_steps: int = 10,
    integration_steps: int = 3,
    eta: float = 0.1,
    rejection_sampling: bool = False,
    shift_fn: Callable[[chex.Array], chex.Array] = lambda x: x,  # (dim,) -> (dim,)
    covariance: Optional[chex.Array] = None,  # (time, dim) or (time, dim, dim)
) -> chex.Array:  # (time, batch, dim)
    batch_size, num_chains = xs.shape[1:3]
    keys = jax.random.split(key, xs.shape[0] * num_chains).reshape(
        xs.shape[0], num_chains, -1
    )

    time_chain_sampler = jax.vmap(
        jax.vmap(
            sample_hamiltonian_monte_carlo,
            in_axes=(0, None, 0, None, None, None, None, None, None, None),
        ),
        in_axes=(
            0,
            None,
            0,
            0,
            None,
            None,
            None,
            None,
            None,
            0 if covariance is not None else None,
        ),
    )

    final_xs = time_chain_sampler(
        keys,
        time_dependent_log_density,
        xs,
        ts,
        num_steps,
        integration_steps,
        eta,
        rejection_sampling,
        shift_fn,
        covariance,
    )

    return final_xs


@eqx.filter_jit
def generate_samples_with_hmc_correction_and_initial_values(
    key: jax.random.PRNGKey,
    v_theta: Callable[[jnp.ndarray, float], jnp.ndarray],
    initial_samples: jnp.ndarray,
    ts: jnp.ndarray,
    time_dependent_log_density: Callable[[jnp.ndarray, float], float],
    integration_fn: Callable[
        [Callable[[jnp.ndarray, float], jnp.ndarray], jnp.ndarray, jnp.ndarray],
        jnp.ndarray,
    ] = euler_integrate,
    num_steps: int = 3,
    integration_steps: int = 3,
    eta: float = 0.01,
    rejection_sampling: bool = False,
    shift_fn: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
    use_shortcut: bool = False,
    covariance: Optional[chex.Array] = None,
) -> jnp.ndarray:
    initial_samps = generate_samples_with_initial_values(
        v_theta=v_theta,
        initial_samples=initial_samples,
        ts=ts,
        integration_fn=integration_fn,
        shift_fn=shift_fn,
        use_shortcut=use_shortcut,
    )

    final_samples = time_batched_sample_hamiltonian_monte_carlo(
        key,
        time_dependent_log_density,
        initial_samps,
        ts,
        num_steps,
        integration_steps,
        eta,
        rejection_sampling,
        shift_fn,
        covariance,
    )

    return final_samples


@eqx.filter_jit
def generate_samples_with_hmc_correction(
    key: jax.random.PRNGKey,
    v_theta: Callable[[jnp.ndarray, float], jnp.ndarray],
    sample_fn: Callable[[jax.random.PRNGKey, Tuple[int, ...]], jnp.ndarray],
    time_dependent_log_density: Callable[[jnp.ndarray, float], float],
    num_samples: int,
    ts: jnp.ndarray,
    integration_fn: Callable[
        [Callable[[jnp.ndarray, float], jnp.ndarray], jnp.ndarray, jnp.ndarray],
        jnp.ndarray,
    ] = euler_integrate,
    num_steps: int = 3,
    integration_steps: int = 3,
    eta: float = 0.01,
    rejection_sampling: bool = False,
    shift_fn: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
    use_shortcut: bool = False,
    covariance: Optional[chex.Array] = None,
) -> jnp.ndarray:
    key, subkey = jax.random.split(key)
    initial_samples = generate_samples(
        subkey,
        v_theta,
        num_samples,
        ts,
        sample_fn,
        integration_fn,
        shift_fn,
        use_shortcut=use_shortcut,
    )

    key, subkey = jax.random.split(key)
    final_samples = time_batched_sample_hamiltonian_monte_carlo(
        subkey,
        time_dependent_log_density,
        initial_samples,
        ts,
        num_steps,
        integration_steps,
        eta,
        rejection_sampling,
        shift_fn,
        covariance,
    )

    weights = (
        jnp.ones(
            (
                ts.shape[0],
                num_samples,
            )
        )
        / num_samples
    )
    return {
        "positions": final_samples,
        "weights": weights,
    }
