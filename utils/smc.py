from typing import Callable, Tuple, Optional

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from .hmc import sample_hamiltonian_monte_carlo, sample_hamiltonian_monte_carlo_blackjax
from blackjax.smc.ess import ess
from blackjax.smc.resampling import systematic


@jax.jit
def log_weights_to_weights(log_weights: jnp.ndarray) -> jnp.ndarray:
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
def estimate_covariance(
    positions: chex.Array,
    weights: Optional[chex.Array] = None,
    diagonal: bool = True,
    regularization: float = 1e-6,
) -> chex.Array:
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
    key: jax.random.PRNGKey,
    time_dependent_log_density: Callable[[chex.Array, float], float],
    num_samples: int,
    ts: jnp.ndarray,
    sample_fn: Callable[[jax.random.PRNGKey, Tuple[int, ...]], jnp.ndarray],
    num_steps: int = 10,
    integration_steps: int = 3,
    eta: float = 0.1,
    rejection_sampling: bool = False,
    shift_fn: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
    ess_threshold: float = 0.5,
    resampling_fn: Callable[
        [jax.random.PRNGKey, jnp.ndarray, int], jnp.ndarray
    ] = systematic,
    incremental_log_delta: Optional[Callable[[chex.Array, float], float]] = None,
    covariances: Optional[chex.Array] = None,
    estimate_covariance: bool = False,
    blackjax_hmc: bool = True,
    v_theta: Optional[Callable[[chex.Array, float], chex.Array]] = None,
    use_shortcut: bool = False,
) -> jnp.ndarray:
    batched_shift_fn = jax.vmap(shift_fn)
    if blackjax_hmc:
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
    else:
        batched_hmc = jax.vmap(
            lambda key, x, t, covariance: sample_hamiltonian_monte_carlo(
                key,
                time_dependent_log_density,
                x,
                t,
                num_steps,
                integration_steps,
                eta,
                rejection_sampling,
                shift_fn,
                covariance,
            ),
            in_axes=(0, 0, None, None),
        )

    key, subkey = jax.random.split(key)
    initial_samples = sample_fn(subkey, (num_samples,))
    log_weights = jnp.full((num_samples,), -jnp.log(num_samples))
    sample_keys = jax.random.split(key, num_samples * ts.shape[0]).reshape(
        ts.shape[0], num_samples, -1
    )

    particles = {
        "positions": initial_samples,
        "log_weights": log_weights,
    }

    def _delta(positions, t, t_prev):
        if incremental_log_delta is not None:
            return incremental_log_delta(positions, t - t_prev)
        else:
            return time_dependent_log_density(
                positions, t
            ) - time_dependent_log_density(positions, t_prev)

    batched_delta = jax.vmap(_delta, in_axes=(0, None, None))

    if v_theta is not None:
        if use_shortcut:
            batched_v_theta = jax.vmap(v_theta, in_axes=(0, None, None))
        else:
            batched_v_theta = jax.vmap(v_theta, in_axes=(0, None))

    def _resample(key, positions, log_weights):
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
        keys, t, cov = inputs
        particles_prev, t_prev = carry

        prev_positions = particles_prev["positions"]
        prev_log_weights = particles_prev["log_weights"]
        d = t - t_prev
        if covariances is None and estimate_covariance:
            cov = estimate_covariance(
                prev_positions, log_weights_to_weights(prev_log_weights), diagonal=True
            )

        # Compute ESS and Resample if necessary
        ess_val = ess(log_weights=prev_log_weights)  # Scalar
        ess_percentage = ess_val / num_samples  # Scalar

        # Define the condition for resampling
        def do_resample():
            resample_key, _ = jax.random.split(keys[0])
            # Resample particles
            new_positions, new_log_weights = _resample(
                resample_key, prev_positions, prev_log_weights
            )

            return {"positions": new_positions, "log_weights": new_log_weights}

        def do_nothing():
            # Keep the particles as is with normalized log weights
            log_weights_normalized = prev_log_weights - jax.scipy.special.logsumexp(
                prev_log_weights
            )
            return {
                "positions": prev_positions,
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
                propagated_positions = shifted_positions + d * batched_v_theta(
                    shifted_positions, t_prev, d
                )
            else:
                propagated_positions = shifted_positions + d * batched_v_theta(
                    shifted_positions, t_prev
                )

        # Apply HMC to propagate particles
        propagated_positions = batched_hmc(
            keys, shifted_positions, t, cov
        )  # Shape: (num_samples, ...)

        # Compute incremental weights
        w_delta = batched_delta(propagated_positions, t, t_prev)
        # Update log weights in log space
        next_log_weights = particles_new["log_weights"] + w_delta
        next_log_weights = next_log_weights - jax.scipy.special.logsumexp(
            next_log_weights
        )

        # Update time
        new_carry = (
            {
                "positions": propagated_positions,
                "log_weights": next_log_weights,
            },
            t,
        )

        # Output current particles
        return new_carry, particles_new

    # Perform the SMC over all time steps
    _, output = jax.lax.scan(
        step,
        (particles, 0.0),  # Initial carry: particles and initial time
        (sample_keys, ts, covariances),  # Inputs: resampled keys and time steps
    )

    weights = jax.vmap(log_weights_to_weights)(output["log_weights"])

    return {
        "positions": output["positions"],
        "weights": weights,
        "ess": output["ess"],
    }


@eqx.filter_jit
def generate_samples_with_euler_smc(
    key: jax.random.PRNGKey,
    v_theta: Callable[[jnp.ndarray, float], jnp.ndarray],
    time_dependent_log_density: Callable[[chex.Array, float], float],
    num_samples: int,
    ts: jnp.ndarray,
    sample_fn: Callable[[jax.random.PRNGKey, Tuple[int, ...]], jnp.ndarray],
    num_steps: int = 10,
    integration_steps: int = 3,
    eta: float = 0.1,
    rejection_sampling: bool = False,
    shift_fn: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
    use_shortcut: bool = False,
) -> jnp.ndarray:
    batched_shift_fn = jax.vmap(shift_fn)
    batched_hmc = jax.vmap(
        lambda key, x, t: sample_hamiltonian_monte_carlo(
            key,
            time_dependent_log_density,
            x,
            t,
            num_steps,
            integration_steps,
            eta,
            rejection_sampling,
            shift_fn,
        ),
        in_axes=(0, 0, None),
    )

    key, subkey = jax.random.split(key)
    initial_samples = sample_fn(subkey, (num_samples,))
    sample_keys = jax.random.split(key, num_samples * ts.shape[0]).reshape(
        ts.shape[0], num_samples, 2
    )

    def step(carry, xs):
        keys, t = xs

        x_prev, t_prev = carry
        d = t - t_prev

        if use_shortcut:
            samples = x_prev + d * jax.vmap(lambda x: v_theta(x, t, d))(x_prev)
        else:
            samples = x_prev + d * jax.vmap(lambda x: v_theta(x, t))(x_prev)

        samples = batched_shift_fn(samples)
        samples = batched_hmc(keys, samples, t)

        return (samples, t), samples

    _, output = jax.lax.scan(step, (initial_samples, 0.0), (sample_keys, ts))

    uniform_weights = jnp.full(
        (
            ts.shape[0],
            num_samples,
        ),
        1.0 / num_samples,
    )
    return {
        "positions": output,
        "weights": uniform_weights,
    }


def systematic_resampling(
    keys: chex.PRNGKey, weights: jnp.ndarray, size: int
) -> jnp.ndarray:
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


class SampleBuffer:
    """Optimized buffer to store and manage samples at each time step. All samples have uniform weights after resampling."""

    def __init__(self, buffer_size: int = 10240, min_update_size: int = 1024):
        """
        Args:
            buffer_size: Maximum number of samples to store per time step
            min_update_size: Minimum number of samples required before using buffer for estimation
        """
        self.buffer_size = buffer_size
        self.min_update_size = min_update_size
        self.samples = None  # Shape: (num_timesteps, buffer_size, dim)
        self.sample_counts = None  # Shape: (num_timesteps,)

    # @eqx.filter_jit
    def add_samples(
        self,
        key: chex.PRNGKey,
        new_samples: chex.Array,  # Shape: (num_timesteps, num_new_samples, dim)
        new_weights: chex.Array,  # Shape: (num_timesteps, num_new_samples)
    ):
        """Add new samples and weights for all time steps.
        If total samples exceed buffer_size, resample to buffer_size.

        Args:
            key: Random key for resampling
            new_samples: New samples to add. Shape: (num_timesteps, num_new_samples, dim)
            new_weights: Normalized weights for new samples. Shape: (num_timesteps, num_new_samples)
        """
        num_timesteps, num_new_samples, dim = new_samples.shape

        if self.samples is None:
            # Initialize samples and sample_counts
            sampled_size = jnp.minimum(num_new_samples, self.buffer_size)
            keys = jax.random.split(key, num_timesteps)
            indices = systematic_resampling(keys, new_weights, sampled_size)
            # Gather the new samples based on the resampled indices
            new_selected_samples = jnp.take_along_axis(
                new_samples, indices[..., None], axis=1
            )
            self.samples = new_selected_samples
            self.sample_counts = sampled_size
            return

        # Determine the number of samples to add per timestep
        available_space = self.buffer_size - self.sample_counts
        sampled_size = jnp.minimum(
            jnp.maximum(available_space, self.min_update_size), num_new_samples
        )
        sampled_size = jnp.minimum(sampled_size, num_new_samples)

        # Split keys for each timestep
        keys = jax.random.split(key, num_timesteps)

        # Resample indices using systematic resampling per timestep
        indices = systematic_resampling(keys, new_weights, sampled_size)

        # Gather the new samples based on the resampled indices
        new_selected_samples = jnp.take_along_axis(
            new_samples, indices[..., None], axis=1
        )

        # Concatenate existing samples with new samples
        self.samples = jnp.concatenate([self.samples, new_selected_samples], axis=1)

        # Update sample counts
        self.sample_counts = self.sample_counts + sampled_size

        # Resample to buffer_size if exceeded
        exceed_mask = self.sample_counts > self.buffer_size
        if jnp.any(exceed_mask):
            # Generate new keys for resampling
            subkeys = jax.random.split(key, num_timesteps)
            # Uniform weights for resampling
            uniform_weights = (
                jnp.ones((num_timesteps, self.buffer_size)) / self.buffer_size
            )
            # Resample indices uniformly
            resample_indices = systematic_resampling(
                subkeys, uniform_weights, self.buffer_size
            )
            # Resample the samples to maintain buffer_size
            self.samples = jnp.take_along_axis(
                self.samples, resample_indices[..., None], axis=1
            )
            # Update sample counts
            self.sample_counts = jnp.minimum(self.sample_counts, self.buffer_size)

    @eqx.filter_jit
    def get_samples(self, key: chex.PRNGKey, num_samples: Optional[int] = None):
        """Get samples from the buffer.

        Args:
            key: Random key for sampling
            num_samples: Optional number of samples to return. If None, returns all samples.
                        If specified, samples are resampled using systematic resampling.

        Returns:
            samples: Array of samples. Shape: (num_timesteps, num_samples, dim)
            weights: Array of uniform weights. Shape: (num_timesteps, num_samples)
        """
        if self.samples is None:
            return None, None

        num_timesteps, current_size, dim = self.samples.shape

        if num_samples is None or num_samples == current_size:
            uniform_weights = jnp.ones((num_timesteps, current_size)) / current_size
            return self.samples, uniform_weights

        # Resample to get the requested number of samples
        keys = jax.random.split(key, num_timesteps)
        uniform_weights = jnp.ones((num_timesteps, current_size)) / current_size

        # Apply systematic resampling for each timestep
        indices = systematic_resampling(keys, uniform_weights, num_samples)

        # Gather resampled samples
        resampled_samples = jnp.take_along_axis(
            self.samples, indices[..., None], axis=1
        )

        # Uniform weights for resampled samples
        new_uniform_weights = jnp.ones((num_timesteps, num_samples)) / num_samples
        return resampled_samples, new_uniform_weights

    @eqx.filter_jit
    def estimate_covariance(
        self, key: chex.PRNGKey, num_samples: Optional[int] = None
    ) -> Optional[chex.Array]:
        """Estimate covariance at each time step using uniform weights.

        Returns:
            Array of shape (num_timesteps, dim, dim) or None if insufficient samples
        """
        samples, _ = self.get_samples(key, num_samples=num_samples)
        if samples is None:
            return None

        # Compute mean per timestep
        mean = jnp.mean(
            samples, axis=1, keepdims=True
        )  # Shape: (num_timesteps, 1, dim)
        # Center the samples
        centered = samples - mean  # Shape: (num_timesteps, num_samples, dim)
        # Compute covariance using einsum for batch operations
        cov = jnp.einsum("tnd,tne->tde", centered, centered) / (samples.shape[1] - 1)
        return cov  # Shape: (num_timesteps, dim, dim)

    @eqx.filter_jit
    def estimate_log_Z_t(
        self,
        key: chex.PRNGKey,
        time_derivative_log_density: Callable[[chex.Array, float], float],
        ts: chex.Array,
        num_samples: Optional[int] = None,
    ) -> Optional[chex.Array]:
        """Estimate log_Z_t at each time step using uniform weights.

        Args:
            key: Random key for sampling
            time_derivative_log_density: Function to compute time derivative of log density
            ts: Array of time steps. Shape: (num_timesteps,)
            num_samples: Optional number of samples to use for estimation

        Returns:
            Array of shape (num_timesteps, 1) or None if insufficient samples
        """
        samples, _ = self.get_samples(key, num_samples=num_samples)
        if samples is None or ts is None:
            return None

        # Vectorize the time_derivative_log_density function
        batched_density = jax.vmap(
            lambda t, xs: jnp.mean(
                jax.vmap(lambda x: time_derivative_log_density(x, t))(xs)
            ),
            in_axes=(0, 0),
        )(ts, samples)  # Shape: (num_timesteps,)

        return batched_density[..., None]  # Shape: (num_timesteps, 1)
