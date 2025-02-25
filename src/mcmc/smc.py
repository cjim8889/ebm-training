from typing import Callable, Dict, Optional, Tuple, Union

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from blackjax.smc.ess import ess
from blackjax.smc.resampling import systematic
from jaxtyping import Array, Float, Int, PRNGKeyArray

from .hmc import sample_hamiltonian_monte_carlo_blackjax


@jax.jit
def log_weights_to_weights(log_weights: Float[Array, "num_samples"]) -> Float[Array, "num_samples"]:
    """
    Convert log weights to weights.

    Args:
        log_weights: Log weights. Shape: (num_samples,).

    Returns:
        Weights: Weights. Shape: (num_samples,).
    """
    log_sum_w = jax.scipy.special.logsumexp(log_weights)
    log_normalized_weights = log_weights - log_sum_w
    weights = jnp.exp(log_normalized_weights)

    return weights


@eqx.filter_jit
def _estimate_covariance(
    positions: Float[Array, "num_samples dim"],
    weights: Optional[Float[Array, "num_samples"]] = None,
    diagonal: bool = True,
    regularization: float = 1e-6,
) -> Float[Array, "dim dim"]:
    N, d = positions.shape
    # Handle weights
    if weights is None:
        weights = jnp.ones(N) / N
    else:
        # Normalize weights to sum to 1
        weights = weights / jnp.sum(weights)

    if not diagonal:
        return jnp.cov(
            positions, rowvar=False, aweights=weights
        ) + regularization * jnp.eye(d)
    else:
        mean = jnp.average(positions, weights=weights, axis=0)
        # Compute weighted variance for each dimension
        # Weighted squared deviations
        squared_devs = (positions - mean) ** 2
        var = jnp.average(squared_devs, weights=weights, axis=0)
        # Construct diagonal covariance matrix
        cov_diag = jnp.diag(var)
        # Add regularization
        cov_diag += regularization * jnp.eye(d)
        return cov_diag


@eqx.filter_jit
def generate_samples_with_smc(
    key: PRNGKeyArray,
    initial_samples: Float[Array, "num_samples dim"],
    time_dependent_log_density: Callable[[Float[Array, "dim"], float], float],
    ts: Float[Array, "num_timesteps"],
    num_steps: int = 10,
    integration_steps: int = 3,
    eta: float = 0.1,
    shift_fn: Callable[[Float[Array, "dim"]], Float[Array, "dim"]] = lambda x: x,
    ess_threshold: float = 0.6,
    resampling_fn: Callable[
        [PRNGKeyArray, Float[Array, "num_samples"], int], Int[Array, "num_samples"]
    ] = systematic,
    covariances: Optional[Float[Array, "num_timesteps dim dim"]] = None,
    estimate_covariance: bool = False,
    v_theta: Optional[Callable[[Float[Array, "dim"], float], Float[Array, "dim"]]] = None,
    use_shortcut: bool = False,
    initial_log_weights: Optional[Float[Array, "num_samples"]] = None,
) -> Dict[str, Union[Float[Array, "num_timesteps num_samples dim"], 
                     Float[Array, "num_timesteps num_samples"], 
                     Float[Array, "num_timesteps"]]]:
    batched_shift_fn = jax.vmap(shift_fn)
    batched_hmc = jax.vmap(
        lambda key, x, t, covariance: sample_hamiltonian_monte_carlo_blackjax(
            key,
            time_dependent_log_density,
            x,
            t,
            num_steps,
            integration_steps,
            eta,
            covariance,
            shift_fn,
        ),
        in_axes=(0, 0, None, None),
    )

    num_samples = initial_samples.shape[0]
    # Initialize particles with provided samples or generate new ones
    chex.assert_rank(initial_samples, 2)
    if initial_log_weights is None:
        initial_log_weights = jnp.full((num_samples,), -jnp.log(num_samples))

    chex.assert_rank(initial_log_weights, 1)
    
    # Split keys for all samples and all timesteps, but exclude the first timestep (since we use initial_samples)
    num_scan_steps = ts.shape[0] - 1  # We need one less step than the number of timestamps
    sample_keys = jax.random.split(key, num_samples * num_scan_steps).reshape(
        num_scan_steps, num_samples, -1
    )

    # Initial particles at time ts[0]
    particles = {
        "positions": initial_samples,
        "log_weights": initial_log_weights,
        "ess": jnp.array(1.0),
    }

    def _delta(positions, t, t_prev):
        return time_dependent_log_density(
            positions, t
        ) - time_dependent_log_density(positions, t_prev)

    batched_delta = jax.vmap(_delta, in_axes=(0, None, None))

    if v_theta is not None:
        if use_shortcut:
            batched_v_theta = jax.vmap(v_theta, in_axes=(0, None, None))
        else:
            batched_v_theta = jax.vmap(v_theta, in_axes=(0, None))

    def _resample(key: PRNGKeyArray, 
                  positions: Float[Array, "num_samples dim"], 
                  log_weights: Float[Array, "num_samples"]
                 ) -> Tuple[Float[Array, "num_samples dim"], Float[Array, "num_samples"]]:
        """
        Resample particles based on their log weights.

        Args:
            key: JAX PRNG key.
            positions: Current particle positions. Shape: (num_samples, ...).
            log_weights: Current log weights. Shape: (num_samples,).

        Returns:
            new_positions: Resampled particle positions. Shape: (num_samples, ...).
            new_log_weights: Reset log weights. Shape: (num_samples,).
        """
        # Normalize log_weights to prevent numerical underflow/overflow
        weights = log_weights_to_weights(log_weights)
        # Perform resampling to obtain indices
        indices = resampling_fn(key, weights, num_samples)  # Shape: (num_samples,)

        # Resample positions
        new_positions = jnp.take(positions, indices, axis=0)

        # Reset log_weights to uniform
        new_log_weights = jnp.full((num_samples,), -jnp.log(num_samples))

        return new_positions, new_log_weights

    def step(carry, inputs):
        particles_prev, t_idx = carry
        keys, cov_t = inputs
        
        # Get current and next time from ts
        t_prev = ts[t_idx]
        t = ts[t_idx + 1]
        
        # Time step for ODE integration
        d = t - t_prev
        
        # Use provided covariance or estimate it if needed
        if covariances is None and estimate_covariance:
            cov = _estimate_covariance(
                particles_prev["positions"], 
                log_weights_to_weights(particles_prev["log_weights"]), 
                diagonal=True
            )
        elif covariances is None:
            # Use default identity covariance when neither provided nor estimated
            cov = None
        else:
            cov = cov_t

        # Compute ESS and Resample if necessary
        ess_val = ess(log_weights=particles_prev["log_weights"])  # Scalar
        ess_percentage = ess_val / num_samples  # Scalar

        # Define the condition for resampling
        def do_resample():
            resample_key, _ = jax.random.split(keys[0])
            # Resample particles
            new_positions, new_log_weights = _resample(
                resample_key, particles_prev["positions"], particles_prev["log_weights"]
            )

            return {"positions": new_positions, "log_weights": new_log_weights}

        def do_nothing():
            # Keep the particles as is with normalized log weights
            log_weights_normalized = particles_prev["log_weights"] - jax.scipy.special.logsumexp(
                particles_prev["log_weights"]
            )
            return {
                "positions": particles_prev["positions"],
                "log_weights": log_weights_normalized,
            }

        # Conditionally resample based on ESS percentage
        particles_new = jax.lax.cond(
            ess_percentage < ess_threshold,
            do_resample,
            do_nothing,
        )
        particles_new["ess"] = ess_percentage

        # Apply shift function
        shifted_positions = batched_shift_fn(
            particles_new["positions"]
        )  # Shape: (num_samples, ...)

        # If v_theta is provided, use it to propagate particles first
        if v_theta is not None:
            if use_shortcut:
                # Match Euler's behavior by using t_prev and absolute dt
                propagated_positions = shifted_positions + d * batched_v_theta(
                    shifted_positions, t_prev, jnp.abs(d)
                )
            else:
                propagated_positions = shifted_positions + d * batched_v_theta(
                    shifted_positions, t_prev
                )
        else:
            propagated_positions = shifted_positions

        # Apply HMC to propagate particles
        propagated_positions = batched_hmc(
            keys, propagated_positions, t, cov
        )  # Shape: (num_samples, ...)

        # Compute incremental weights
        w_delta = batched_delta(propagated_positions, t, t_prev)
        # Update log weights in log space
        next_log_weights = particles_new["log_weights"] + w_delta
        next_log_weights = next_log_weights - jax.scipy.special.logsumexp(
            next_log_weights
        )

        # Update carry with new particles and next time index
        # Include the ess key to match the input carry structure
        new_carry = (
            {
                "positions": propagated_positions,
                "log_weights": next_log_weights,
                "ess": particles_new["ess"],  # Make sure to include ess in the new carry
            },
            t_idx + 1,
        )

        # Output particles at time t
        return new_carry, {
            "positions": propagated_positions,
            "log_weights": next_log_weights,
            "ess": particles_new["ess"],
        }

    # Prepare covariances for scan if provided
    if covariances is not None:
        scan_covariances = covariances[1:]  # Skip first element since we match with ts[1:] in the scan
    else:
        scan_covariances = None
    
    # Run scan over time indices from 0 to num_timesteps-2
    # This will generate particles at times ts[1] to ts[num_timesteps-1]
    _, scan_particles = jax.lax.scan(
        step,
        (particles, 0),  # Initial carry: particles at ts[0] and time index 0
        (sample_keys, scan_covariances),  # Inputs for each time step
    )
    
    # Now we need to include the initial particles at ts[0]
    all_positions = jnp.concatenate([
        jnp.expand_dims(particles["positions"], axis=0),
        scan_particles["positions"]
    ], axis=0)
    
    all_log_weights = jnp.concatenate([
        jnp.expand_dims(particles["log_weights"], axis=0),
        scan_particles["log_weights"]
    ], axis=0)
    
    all_ess = jnp.concatenate([
        jnp.expand_dims(particles["ess"], axis=0),
        scan_particles["ess"]
    ], axis=0)
    
    # Convert all log weights to weights
    all_weights = jax.vmap(log_weights_to_weights)(all_log_weights)

    return {
        "positions": all_positions,
        "weights": all_weights,
        "ess": all_ess,
    }

def systematic_resampling(
    keys: PRNGKeyArray, 
    weights: Float[Array, "num_timesteps num_samples"], 
    size: int
) -> Int[Array, "num_timesteps size"]:
    """
    Perform batched systematic resampling.

    Args:
        keys: PRNG keys for each timestep. Shape: (num_timesteps, 2)
        weights: Normalized weights for resampling. Shape: (num_timesteps, num_samples)
        size: Number of samples to resample per timestep.

    Returns:
        Indices of resampled particles. Shape: (num_timesteps, size)
    """

    def single_resample(key, w, size):
        """Resample indices for a single timestep."""
        positions = (jnp.arange(size) + jax.random.uniform(key)) / size
        cumulative_sum = jnp.cumsum(w)
        indices = jnp.searchsorted(cumulative_sum, positions, side="right")
        return indices

    return jax.vmap(single_resample, in_axes=(0, 0, None))(keys, weights, size)
