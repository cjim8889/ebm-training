import jax.numpy as jnp
import distrax
import chex
import jax
from .base import Target
from typing import Optional
import matplotlib.pyplot as plt


class TranslationInvariantGaussian(Target):
    TIME_DEPENDENT = False

    def __init__(
        self,
        N: int = 3,
        D: int = 2,
        sigma: float = 1.0,
        box_size: Optional[jnp.ndarray] = None,  # [D] or scalar for isotropic
        plot_bound_factor: float = 3.0,
        num_shift: int = 3,
        wrap: bool = False,
        **kwargs,
    ):
        """
        Initializes the Translation-Invariant Gaussian distribution.

        Args:
            N (int): Number of points.
            D (int): Dimensionality of each point.
            sigma (float or array-like): Standard deviation. Scalar for isotropic, or array of shape [D].
            box_size (Optional[jnp.ndarray]): Size of the periodic box. If None, wrapping is disabled.
            plot_bound_factor (float): Factor to determine plot boundaries based on sigma.
            num_shift (int): Number of shifts per dimension for wrapping.
            wrap (bool): Whether to apply periodic boundary conditions.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            dim=N * D,
            log_Z=0.0,  # Not used since it's a wrapped Gaussian
            can_sample=True,
            n_plots=1,
            n_model_samples_eval=1000,
            n_target_samples_eval=1000,
            **kwargs,
        )

        self.N = N
        self.D = D
        self.wrap = wrap
        self.num_shift = num_shift

        # Handle sigma
        self.sigma = jnp.asarray(sigma)
        if self.sigma.ndim == 0:
            # Scalar sigma: isotropic covariance
            self.scale_diag = jnp.full((D,), self.sigma)
        else:
            # Per-dimension sigma
            if self.sigma.shape[0] != D:
                raise ValueError(
                    f"Sigma shape {self.sigma.shape} does not match dimension {D}."
                )
            self.scale_diag = self.sigma

        # Handle box_size
        if self.wrap:
            if box_size is None:
                raise ValueError("box_size must be provided if wrap is True.")
            self.box_size = jnp.asarray(box_size)
            if self.box_size.shape == ():
                # Scalar box size: same for all dimensions
                self.box_size = jnp.full((D,), self.box_size)
            elif self.box_size.shape[0] != D:
                raise ValueError(
                    f"box_size shape {self.box_size.shape} does not match dimension {D}."
                )
        else:
            self.box_size = None

        # Degrees of freedom: (N-1)*D
        self.degrees_of_freedom = (self.N - 1) * self.D

        # Normalizing constant
        self.log_normalizing_constant = -0.5 * self.degrees_of_freedom * jnp.log(
            2 * jnp.pi
        ) - jnp.sum(jnp.log(self.scale_diag))

        # Initialize the Multivariate Normal Distribution
        # We model the N*D variables, but enforce mean zero in log_prob and sampling
        self.distribution = distrax.MultivariateNormalDiag(
            loc=jnp.zeros(self.N * self.D),
            scale_diag=jnp.tile(self.scale_diag, (self.N,)),
        )

        # Determine plot bounds based on sigma
        if self.sigma.ndim == 0:
            bound = self.sigma * plot_bound_factor
        else:
            bound = jnp.max(self.sigma) * plot_bound_factor
        self._plot_bound = float(bound)

        if self.wrap:
            # Prepare shifts for periodic boundary conditions
            # For simplicity, assume shifts are the same across all dimensions
            k = jnp.arange(-self.num_shift, self.num_shift + 1)
            shifts = jnp.stack(jnp.meshgrid(*([k] * self.D), indexing="ij"), axis=-1)
            self.kL = (
                shifts.reshape(-1, self.D) * self.box_size
            )  # Shape: [(2*num_shift+1)^D, D]
        else:
            self.kL = None

    def remove_mean(self, x: chex.Array) -> chex.Array:
        """
        Removes the mean across the N points.

        Args:
            x (chex.Array): Input array of shape [N, D].

        Returns:
            chex.Array: Mean-zero projected array of shape [N, D].
        """
        mean = jnp.mean(x, axis=0, keepdims=True)  # Shape: [1, D]
        return x - mean  # Broadcasting

    def log_prob(self, x: chex.Array) -> float:
        """
        Computes the log-probability of the input under the translation-invariant Gaussian.

        Args:
            x (chex.Array): Input array of shape [N, D].

        Returns:
            float: Log-probability of the input.
        """
        x = x.reshape(self.N, self.D)  # Shape: [N, D]
        # Remove mean to enforce mean-zero constraint
        x_centered = self.remove_mean(x)  # Shape: [N, D]

        # Compute sum of squares
        r2 = jnp.sum(x_centered**2)  # Scalar

        # Compute log-probability
        log_px = -0.5 * r2 + self.log_normalizing_constant  # Scalar

        if self.wrap:
            # If wrapping is enabled, sum over all possible shifts
            # This accounts for periodic boundary conditions
            # Compute log_prob for each shifted configuration and use log-sum-exp

            # Add shifts to the sample
            # x_shifted has shape [(num_shifts)^D, N, D]
            x_shifted = x_centered[None, :, :] + self.kL[:, None, :]  # Broadcasting

            # Remove mean again after shifting
            x_shifted = self.remove_mean(
                x_shifted.reshape(-1, self.D)
            )  # Shape: [num_shifts^D, D]
            # Reshape back to [num_shifts^D, N, D]
            x_shifted = x_shifted.reshape(-1, self.N, self.D)

            # Compute sum of squares for each shifted sample
            r2_shifted = jnp.sum(x_shifted**2, axis=(1, 2))  # Shape: [num_shifts^D]

            # Compute log_prob for shifted samples
            log_px_shifted = (
                -0.5 * r2_shifted + self.log_normalizing_constant
            )  # Shape: [num_shifts^D]

            # Use log-sum-exp to aggregate probabilities
            max_log_px = jnp.max(log_px_shifted)  # Scalar
            log_px = max_log_px + jnp.log(
                jnp.sum(jnp.exp(log_px_shifted - max_log_px))
            )  # Scalar

        return log_px

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape = ()) -> chex.Array:
        """
        Generates samples from the translation-invariant Gaussian.

        Args:
            seed (chex.PRNGKey): Random seed.
            sample_shape (chex.Shape): Shape of the samples to generate.

        Returns:
            chex.Array: Sampled array of shape [sample_shape, N, D].
        """
        # Total number of variables: N * D
        total_sample_shape = (
            sample_shape if isinstance(sample_shape, tuple) else (sample_shape,)
        )
        samples = self.distribution.sample(
            seed=seed, sample_shape=total_sample_shape
        )  # Shape: [*sample_shape, N*D]

        # Reshape to [*sample_shape, N, D]
        samples = samples.reshape(*total_sample_shape, self.N, self.D)

        samples_centered = samples - jnp.mean(samples, axis=1, keepdims=True)

        if self.wrap:
            # Apply periodic boundary conditions using modulo operation
            samples_centered = jnp.mod(samples_centered, self.box_size)

        return samples_centered.reshape(total_sample_shape[0], -1)

    def score(self, value: chex.Array) -> chex.Array:
        """
        Computes the gradient of the log-probability with respect to the input.

        Args:
            value (chex.Array): Input array of shape [N, D].

        Returns:
            chex.Array: Gradient array of shape [N, D].
        """
        # Compute gradient
        return jax.grad(self.log_prob)(value)

    def visualise(self, samples) -> plt.Figure:
        raise NotImplementedError
