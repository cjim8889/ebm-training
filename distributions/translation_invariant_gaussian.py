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
        plot_bound_factor: float = 3.0,
        **kwargs,
    ):
        """
        Initializes the Translation-Invariant Gaussian distribution.

        Args:
            N (int): Number of points.
            D (int): Dimensionality of each point.
            sigma (float or array-like): Standard deviation. Scalar for isotropic, or array of shape [D].
            plot_bound_factor (float): Factor to determine plot boundaries based on sigma.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            dim=N * D,
            log_Z=0.0,
            can_sample=True,
            n_plots=1,
            n_model_samples_eval=1000,
            n_target_samples_eval=1000,
            **kwargs,
        )

        self.N = N
        self.D = D

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

        # Degrees of freedom: (N-1)*D
        self.degrees_of_freedom = (self.N - 1) * self.D

        # Normalizing constant
        # Note: We need to account for the projection to mean-zero subspace
        self.log_normalizing_constant = -0.5 * self.degrees_of_freedom * jnp.log(
            2 * jnp.pi
        ) - self.degrees_of_freedom * jnp.log(self.sigma)

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
        log_px = -0.5 * r2 / (self.sigma**2) + self.log_normalizing_constant  # Scalar

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

        # Project to mean-zero subspace
        samples_centered = samples - jnp.mean(samples, axis=1, keepdims=True)

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
