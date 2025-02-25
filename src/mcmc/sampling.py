from typing import Callable, Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .hmc import (
    sample_hamiltonian_monte_carlo,
    sample_hamiltonian_monte_carlo_blackjax,
    time_batched_sample_hamiltonian_monte_carlo,
)
from .integration import (
    euler_integrate,
    generate_samples,
    generate_samples_with_diffrax,
)
from .smc import generate_samples_with_smc


@eqx.filter_jit
def sample_with_mcmc(
    key: jax.random.PRNGKey,
    v_theta: Optional[Callable] = None,
    sample_fn: Optional[Callable] = None,
    time_dependent_log_density: Optional[Callable] = None,
    mcmc_method: str = "none",
    num_samples: int = 1000,
    ts: Optional[jnp.ndarray] = None,
    initial_samples: Optional[jnp.ndarray] = None,
    initial_log_probs: Optional[jnp.ndarray] = None,
    shift_fn: Callable = lambda x: x,
    use_shortcut: bool = False,
    num_steps: int = 10,
    integration_steps: int = 3,
    eta: float = 0.1,
    rejection_sampling: bool = False,
    ess_threshold: float = 0.6,
    estimate_covariance: bool = False,
    covariance: Optional[jnp.ndarray] = None,
    use_blackjax: bool = True,
    solver: str = "Euler",
    **kwargs
) -> Dict[str, jnp.ndarray]:
    """
    Unified interface for generating samples with MCMC methods.
    
    This function provides a standardized interface for all MCMC sampling
    methods, including direct sampling (no MCMC), Hamiltonian Monte Carlo (HMC),
    and Sequential Monte Carlo (SMC).
    
    Args:
        key: Random key
        v_theta: Velocity field model (optional)
        sample_fn: Function to sample from the base distribution
        time_dependent_log_density: Log density function (t, x) -> log p(x, t)
        mcmc_method: MCMC method to use ("none", "hmc", "smc", "vsmc")
        num_samples: Number of samples to generate
        ts: Time steps
        initial_samples: Optional pre-generated samples
        initial_log_probs: Optional log probabilities for initial samples
        shift_fn: Function to shift samples (for periodic boundaries, etc.)
        use_shortcut: Whether to use shortcut mechanism for velocity field
        num_steps: Number of MCMC steps
        integration_steps: Number of integration steps per MCMC step
        eta: Step size for MCMC
        rejection_sampling: Whether to use rejection sampling
        ess_threshold: Threshold for resampling in SMC
        estimate_covariance: Whether to estimate covariance in SMC
        covariance: Fixed covariance matrix (optional)
        use_blackjax: Whether to use blackjax for HMC
        solver: Integration solver to use ("Euler" or "Tsit5")
        **kwargs: Additional arguments
        
    Returns:
        Dictionary containing:
        - positions: Generated samples
        - weights: Sample weights
        - (optional) ess: Effective sample size for SMC
        - (optional) log_probs: Log probabilities
    """
    # Generate initial samples if not provided
    if initial_samples is None:
        key, subkey = jax.random.split(key)
        
        # Generate initial samples
        use_flow = mcmc_method != "none" and v_theta is not None
        samples = generate_initial_samples(
            subkey,
            sample_fn,
            num_samples,
            ts,
            v_theta,
            use_flow=use_flow,
            use_shortcut=use_shortcut,
            shift_fn=shift_fn,
            solver=solver,
        )
        
        # If no MCMC or we've already used the flow model, return directly
        if mcmc_method == "none" or use_flow:
            return samples
            
        initial_samples = samples["positions"]
    
    # Apply MCMC based on the specified method
    if mcmc_method == "none":
        # No MCMC, just uniform weights
        weights = jnp.ones((ts.shape[0], initial_samples.shape[1])) / initial_samples.shape[1]
        return {
            "positions": initial_samples,
            "weights": weights,
        }
    
    elif mcmc_method == "hmc":
        # Use HMC
        key, subkey = jax.random.split(key)
        return propagate_with_hmc(
            subkey,
            initial_samples,
            time_dependent_log_density,
            ts,
            num_steps,
            integration_steps,
            eta,
            rejection_sampling,
            shift_fn,
            covariance,
            use_blackjax,
        )
    
    elif mcmc_method in ["smc", "vsmc"]:
        # Use SMC or VSMC
        key, subkey = jax.random.split(key)
        
        # For VSMC, we use the velocity field
        v_theta_smc = v_theta if mcmc_method == "vsmc" else None
        
        return generate_samples_with_smc(
            key=subkey,
            time_dependent_log_density=time_dependent_log_density,
            num_samples=num_samples,
            ts=ts,
            sample_fn=sample_fn,
            num_steps=num_steps,
            integration_steps=integration_steps,
            eta=eta,
            rejection_sampling=rejection_sampling,
            shift_fn=shift_fn,
            ess_threshold=ess_threshold,
            estimate_covariance=estimate_covariance,
            blackjax_hmc=use_blackjax,
            v_theta=v_theta_smc,
            use_shortcut=use_shortcut,
            initial_samples=initial_samples,
            initial_log_weights=initial_log_probs,
        )
    
    else:
        raise ValueError(f"Unknown MCMC method: {mcmc_method}") 