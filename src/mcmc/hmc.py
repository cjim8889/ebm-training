from typing import Callable, Dict, Optional

import blackjax
import equinox as eqx
import jax
import jax.numpy as jnp
from blackjax.mcmc.hmc import HMCState
from jaxtyping import Array, Float, PRNGKeyArray


@eqx.filter_jit
def sample_hamiltonian_monte_carlo_blackjax(
    key: PRNGKeyArray,
    time_dependent_log_density: Callable[[Float[Array, "dim"], float], float],
    x: Float[Array, "dim"],
    t: float,
    num_steps: int = 10,
    integration_steps: int = 3,
    eta: float = 0.1,
    covariance: Optional[Float[Array, "dim dim"]] = None,
    shift_fn: Callable[[Float[Array, "dim"]], Float[Array, "dim"]] = lambda x: x,
    **kwargs,
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
        eta: Step size
        covariance: Optional covariance matrix/diagonal
        shift_fn: Function to shift samples (e.g., for periodic boundaries)
        **kwargs: Additional arguments (for compatibility)
        
    Returns:
        Final position after HMC
    """
    dim = x.shape[-1]
    inverse_mass_matrix = jnp.eye(dim) if covariance is None else covariance

    hmc = blackjax.hmc(
        lambda x: time_dependent_log_density(x, t),
        eta,
        inverse_mass_matrix,
        integration_steps,
    )
    hmc_kernel = jax.jit(hmc.step)
    initial_state = hmc.init(x)

    @jax.jit
    def one_step(state, rng_key):
        state, _ = hmc_kernel(rng_key, state)
        state = HMCState(
            position=shift_fn(state.position),
            logdensity=state.logdensity,
            logdensity_grad=state.logdensity_grad,
        )
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
