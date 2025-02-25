from typing import Callable, Dict, Optional, Union

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from src.mcmc.smc import generate_samples_with_smc
from src.ode import solve_neural_ode_diffrax

from .hmc import propagate_with_hmc


@eqx.filter_jit
def sample_with_mcmc(
    key: PRNGKeyArray,
    initial_samples: Float[Array, "num_samples dim"],
    v_theta: Optional[Callable[[Float[Array, "dim"], float], Float[Array, "dim"]]] = None,
    time_dependent_log_density: Optional[Callable[[Float[Array, "dim"], float], float]] = None,
    mcmc_method: str = "none",
    ts: Optional[Float[Array, "num_timesteps"]] = None,
    initial_log_probs: Optional[Float[Array, "num_samples"]] = None,
    shift_fn: Callable[[Float[Array, "dim"]], Float[Array, "dim"]] = lambda x: x,
    use_shortcut: bool = False,
    num_steps: int = 10,
    integration_steps: int = 3,
    eta: float = 0.1,
    ess_threshold: float = 0.6,
    estimate_covariance: bool = False,
    covariance: Optional[Float[Array, "dim dim"]] = None,
    solver: str = "Euler",
    **kwargs
) -> Dict[str, Union[Float[Array, "num_timesteps num_samples dim"], 
                    Float[Array, "num_timesteps num_samples"],
                    Float[Array, "num_timesteps"]]]:
    """
    Unified interface for generating samples with MCMC methods.
    
    This function provides a standardized interface for all MCMC sampling
    methods, including direct sampling (no MCMC), Hamiltonian Monte Carlo (HMC),
    and Sequential Monte Carlo (SMC).
    
    Args:
        key: Random key
        initial_samples: Initial samples with shape (num_samples, dim)
        v_theta: Velocity field model (optional)
        time_dependent_log_density: Log density function (t, x) -> log p(x, t)
        mcmc_method: MCMC method to use ("none", "hmc", "smc", "vsmc")
        ts: Time steps with shape (num_timesteps,)
        initial_log_probs: Optional log probabilities for initial samples with shape (num_samples,)
        shift_fn: Function to shift samples (for periodic boundaries, etc.)
        use_shortcut: Whether to use shortcut mechanism for velocity field
        num_steps: Number of MCMC steps
        integration_steps: Number of integration steps per MCMC step
        eta: Step size for MCMC
        ess_threshold: Threshold for resampling in SMC
        estimate_covariance: Whether to estimate covariance in SMC
        covariance: Fixed covariance matrix (optional) with shape (dim, dim)
        solver: Integration solver to use ("Euler" or "Tsit5")
        **kwargs: Additional arguments
        
    Returns:
        Dictionary containing:
        - positions: Generated samples with shape (num_timesteps, num_samples, dim)
        - weights: Sample weights with shape (num_timesteps, num_samples)
        - (optional) ess: Effective sample size for SMC with shape (num_timesteps,)
        - (optional) log_probs: Log probabilities with shape (num_timesteps, num_samples)
    """
    # Assert initial_samples dimensions
    chex.assert_rank(initial_samples, 2)

    if initial_log_probs is None:
        initial_log_probs = jnp.zeros(initial_samples.shape[0])

    chex.assert_rank(initial_log_probs, 1)
    
    # Apply MCMC based on the specified method
    if mcmc_method == "none":
        # No MCMC, just uniform weights and run the ode
        output_samples, _ = solve_neural_ode_diffrax(
            v_theta=v_theta,
            y0=initial_samples,
            ts=ts,
            use_shortcut=use_shortcut,
            exact_logp=True,
            forward=True,
            save_trajectory=True,
            solver=solver,
        )

        weights = jnp.ones((ts.shape[0], initial_samples.shape[0])) / initial_samples.shape[0]
        return {
            "positions": output_samples,
            "weights": weights,
        }
    
    elif mcmc_method == "hmc":
        # Use HMC and first run the ode
        intermediate_samples, _ = solve_neural_ode_diffrax(
            v_theta=v_theta,
            y0=initial_samples,
            ts=ts,
            use_shortcut=use_shortcut,
            exact_logp=True,
            forward=True,
            save_trajectory=True,
            solver=solver,
        )

        key, subkey = jax.random.split(key)
        final_samples = propagate_with_hmc(
            key=subkey,
            initial_samples=intermediate_samples,
            time_dependent_log_density=time_dependent_log_density,
            ts=ts,
            num_steps=num_steps,
            integration_steps=integration_steps,
            eta=eta,
            shift_fn=shift_fn,
            covariance=covariance,
        )

        return {
            "positions": final_samples,
            "weights": jnp.ones((ts.shape[0], initial_samples.shape[0])) / initial_samples.shape[0],
        }
    
    elif mcmc_method in ["smc", "vsmc"]:
        # Use SMC or VSMC
        key, subkey = jax.random.split(key)
        
        # For VSMC, we use the velocity field
        v_theta_smc = v_theta if mcmc_method == "vsmc" else None
        
        return generate_samples_with_smc(
            key=subkey,
            initial_samples=initial_samples,
            time_dependent_log_density=time_dependent_log_density,
            ts=ts,
            num_steps=num_steps,
            integration_steps=integration_steps,
            eta=eta,
            shift_fn=shift_fn,
            ess_threshold=ess_threshold,
            estimate_covariance=estimate_covariance,
            v_theta=v_theta_smc,
            use_shortcut=use_shortcut,
        )
    
    else:
        raise ValueError(f"Unknown MCMC method: {mcmc_method}") 