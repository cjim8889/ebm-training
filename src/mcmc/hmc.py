from typing import Callable, Optional

import blackjax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


@eqx.filter_jit
def sample_hamiltonian_monte_carlo_blackjax(
    key: PRNGKeyArray,
    time_dependent_log_density: Callable[[Float[Array, "dim"], float], float],
    x: Float[Array, "dim"],
    t: float,
    step_size: Float[Array, ""], # Now explicitly passed, can be 0D array or float
    inverse_mass_matrix: Optional[Float[Array, "dim dim"]],
    num_integration_steps: int, # Now explicitly passed
    num_hmc_steps: int = 1, # Renamed to avoid confusion, typically 1 inside SMC step
    **kwargs, # Keep for potential future use / compatibility if needed elsewhere
) -> Float[Array, "dim"]:
    """
    Hamiltonian Monte Carlo using BlackJAX.
    
    Args:
        key: Random key
        time_dependent_log_density: Log density function
        x: Initial position
        t: Time parameter
        num_steps: Number of HMC steps
        integration_steps: Number of integration steps per HMC step
        step_size: Step size
        inverse_mass_matrix: Optional covariance matrix/diagonal
        **kwargs: Additional arguments (for compatibility)
        
    Returns:
        Final position after HMC
    """
    # Use provided inv mass matrix, default to identity if None
    dim = x.shape[-1]
    _inverse_mass_matrix = jnp.eye(dim) if inverse_mass_matrix is None else inverse_mass_matrix

    # Initialize Blackjax HMC kernel with passed parameters
    hmc = blackjax.hmc(
        logdensity_fn=lambda state: time_dependent_log_density(state, t),
        step_size=step_size,
        inverse_mass_matrix=_inverse_mass_matrix,
        num_integration_steps=num_integration_steps,
    )
    hmc_kernel = jax.jit(hmc.step)
    initial_state = hmc.init(x)

    # Define the loop body for sequential HMC steps
    @jax.jit
    def one_step(state, rng_key):
        state, _ = hmc_kernel(rng_key, state)
        return state, state # Carry the state, output the state

    # Perform HMC steps using lax.scan
    keys = jax.random.split(key, num_hmc_steps) # Use num_hmc_steps
    final_state, _ = jax.lax.scan(one_step, initial_state, keys)

    return final_state.position # Return only the final position


@eqx.filter_jit
def sample_nuts_blackjax(
    key: PRNGKeyArray,
    time_dependent_log_density: Callable[[Float[Array, "dim"], float], float],
    x: Float[Array, "dim"],
    t: float,
    num_steps: int = 10,
    step_size: float = 0.1,
    covariance: Optional[Float[Array, "dim dim"]] = None,
    max_num_doublings: int = 10,
    divergence_threshold: int = 1000,
    **kwargs,
) -> Float[Array, "dim"]:
    """
    No-U-Turn Sampler (NUTS) using BlackJAX.

    Args:
        key: Random key
        time_dependent_log_density: Log density function dependent on state and time
        x: Initial position
        t: Time parameter
        num_steps: Number of NUTS steps to take
        step_size: Step size for the integrator
        covariance: Optional covariance matrix/diagonal for inverse mass matrix
        shift_fn: Function to shift samples (e.g., for periodic boundaries)
        max_num_doublings: Maximum number of doublings in the NUTS trajectory expansion
        divergence_threshold: Threshold for divergence detection
        **kwargs: Additional arguments (for compatibility)

    Returns:
        Final position after NUTS sampling
    """
    dim = x.shape[-1]
    inverse_mass_matrix = jnp.eye(dim) if covariance is None else covariance

    logdensity_fn = lambda state: time_dependent_log_density(state, t)

    nuts_kernel_factory = blackjax.nuts(
        logdensity_fn=logdensity_fn,
        step_size=step_size,
        inverse_mass_matrix=inverse_mass_matrix,
        max_num_doublings=max_num_doublings,
        divergence_threshold=divergence_threshold,
    )
    nuts_step = jax.jit(nuts_kernel_factory.step)
    initial_state = nuts_kernel_factory.init(x) # Assuming init only needs position

    @jax.jit
    def one_step(state, rng_key):
        state, info = nuts_step(rng_key, state)
        # Assuming NUTSState is a Pytree (like NamedTuple/dataclass) with a 'position' attribute
        return state, state

    keys = jax.random.split(key, num_steps)
    final_state, _ = jax.lax.scan(one_step, initial_state, keys)

    return final_state.position


@eqx.filter_jit
def propagate_with_hmc(
    key: PRNGKeyArray,
    initial_samples: Float[Array, "num_timesteps num_samples dim"],
    time_dependent_log_density: Callable[[Float[Array, "dim"], float], float],
    ts: Float[Array, "num_timesteps"],
    num_steps: int = 10,
    integration_steps: int = 3,
    eta: float = 0.1,
    shift_fn: Callable[[Float[Array, "dim"]], Float[Array, "dim"]] = lambda x: x,
    covariance: Optional[Float[Array, "dim dim"]] = None,
) -> Float[Array, "num_timesteps num_samples dim"]:
    """
    Propagate samples using HMC.
    
    Args:
        key: Random key
        initial_samples: Initial particle positions
        time_dependent_log_density: Log density function
        ts: Time steps
        num_steps: Number of HMC steps
        integration_steps: Number of integration steps per HMC step
        eta: Step size
        shift_fn: Function to shift samples
        covariance: Covariance matrix (optional)
        
    Returns:
        Dictionary with propagated samples and weights
    """
    # Time-batched BlackJAX HMC
    batch_size = initial_samples.shape[1]
    keys = jax.random.split(key, ts.shape[0] * batch_size).reshape(
        ts.shape[0], batch_size, -1
    )
        
    time_chain_sampler = jax.vmap(
        jax.vmap(
            sample_hamiltonian_monte_carlo_blackjax,
            in_axes=(0, None, 0, None, None, None, None, None, None),
        ),
        in_axes=(
            0,
            None,
            0,
            0,
            None,
            None,
            None,
            0 if covariance is not None else None,
            None,
        ),
    )
    
    final_samples = time_chain_sampler(
        keys,
        time_dependent_log_density,
        initial_samples,
        ts,
        num_steps,
        integration_steps,
        eta,
        covariance,
        shift_fn,
    )
    
    return final_samples
