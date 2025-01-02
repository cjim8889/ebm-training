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
    time_dependent_log_density: Callable[[chex.Array, float], float],
    x: chex.Array,
    t: float,
    num_steps: int = 10,
    integration_steps: int = 3,
    eta: float = 0.1,
    rejection_sampling: bool = False,
    shift_fn: Callable[[chex.Array], chex.Array] = lambda x: x,
    covariance: Optional[chex.Array] = None,
) -> chex.Array:
    dim = x.shape[-1]

    # Handle covariance cases
    if covariance is None:
        covariance = jnp.eye(dim)
        inv_covariance = covariance
    else:
        # Check if diagonal covariance (1D array) or full matrix
        if covariance.ndim == 1:
            inv_covariance = 1.0 / covariance
            covariance = jnp.diag(covariance)
        else:
            inv_covariance = jnp.linalg.inv(covariance)

    grad_log_prob = jax.grad(lambda x: time_dependent_log_density(x, t))

    def kinetic_energy(v):
        return 0.5 * v.T @ inv_covariance @ v

    def hamiltonian(x, v):
        return -time_dependent_log_density(x, t) + kinetic_energy(v)

    def integration_step(carry, _):
        x, v = carry
        x = x + eta * inv_covariance @ v
        # Apply the modular wrapping to enforce PBC
        x = shift_fn(x)

        v = v + eta * grad_log_prob(x)
        return (x, v), _

    def hmc_step(x_current, key):
        x = x_current
        key, subkey = jax.random.split(key)

        # Sample momentum
        v = jax.random.normal(subkey, (dim,))
        current_h = hamiltonian(x, v)

        # Initial half step for momentum
        v = v + 0.5 * eta * grad_log_prob(x)

        # Leapfrog integration
        (x, v), _ = jax.lax.scan(
            integration_step, (x, v), None, length=integration_steps
        )

        # Final half step for momentum
        v = v + 0.5 * eta * grad_log_prob(x)

        # Finalize the proposal
        x = shift_fn(x)

        if rejection_sampling:
            # Compute acceptance probability
            proposed_h = hamiltonian(x, v)
            accept_ratio = jnp.minimum(1.0, jnp.exp(current_h - proposed_h))

            # Accept or reject
            key, subkey = jax.random.split(key)
            uniform_sample = jax.random.uniform(subkey)
            accept = uniform_sample < accept_ratio

            new_x = jax.lax.cond(accept, lambda _: x, lambda _: x_current, operand=None)

            return new_x, None
        else:
            return x, None

    # Run the chain
    keys = jax.random.split(key, num_steps)

    # return hmc_step(init_state, keys[0])
    final_x, _ = jax.lax.scan(hmc_step, x, keys)

    return final_x


@eqx.filter_jit
def time_batched_sample_hamiltonian_monte_carlo(
    key: jax.random.PRNGKey,
    time_dependent_log_density: Callable[[chex.Array, float], float],
    xs: chex.Array,
    ts: chex.Array,
    num_steps: int = 10,
    integration_steps: int = 3,
    eta: float = 0.1,
    rejection_sampling: bool = False,
    shift_fn: Callable[[chex.Array], chex.Array] = lambda x: x,
    covariance: Optional[chex.Array] = None,
) -> chex.Array:
    keys = jax.random.split(key, xs.shape[0] * xs.shape[1])
    reshaped_keys = keys.reshape((xs.shape[0], xs.shape[1], -1))

    if covariance is None:
        final_xs = jax.vmap(
            lambda xs, t, keys: jax.vmap(
                lambda x, subkey: sample_hamiltonian_monte_carlo(
                    subkey,
                    time_dependent_log_density,
                    x,
                    t,
                    num_steps,
                    integration_steps,
                    eta,
                    rejection_sampling,
                    shift_fn,
                    None,
                )
            )(xs, keys)
        )(xs, ts, reshaped_keys)
    else:
        final_xs = jax.vmap(
            lambda xs, t, keys, cov: jax.vmap(
                lambda x, subkey: sample_hamiltonian_monte_carlo(
                    subkey,
                    time_dependent_log_density,
                    x,
                    t,
                    num_steps,
                    integration_steps,
                    eta,
                    rejection_sampling,
                    shift_fn,
                    cov,
                )
            )(xs, keys)
        )(xs, ts, reshaped_keys, covariance)

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
