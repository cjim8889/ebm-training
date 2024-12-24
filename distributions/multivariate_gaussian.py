import jax.numpy as jnp
import distrax
import chex
from .base import Target
import matplotlib.pyplot as plt


class MultivariateGaussian(Target):
    TIME_DEPENDENT = False

    def __init__(
        self,
        dim: int = 2,
        mean: float = 0.0,
        sigma: float = 1.0,
        plot_bound_factor: float = 3.0,
    ):
        super().__init__(
            dim=dim,
            log_Z=0.0,
            can_sample=True,
            n_plots=1,
            n_model_samples_eval=1000,
            n_target_samples_eval=1000,
        )
        self.sigma = jnp.asarray(sigma)
        if self.sigma.ndim == 0:
            scale_diag = jnp.full((dim,), self.sigma)
        else:
            if self.sigma.shape[0] != dim:
                raise ValueError(
                    f"Sigma shape {self.sigma.shape} does not match dimension {dim}."
                )
            scale_diag = self.sigma

        self.mean = jnp.asarray(mean)

        self.distribution = distrax.MultivariateNormalDiag(
            loc=self.mean, scale_diag=scale_diag
        )

    def log_prob(self, x: chex.Array) -> chex.Array:
        return self.distribution.log_prob(x)

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape = ()) -> chex.Array:
        return self.distribution.sample(seed=seed, sample_shape=sample_shape)

    def visualise(self, samples: chex.Array) -> plt.Figure:
        fig, ax = plt.subplots(1, figsize=(6, 6))
        if self.dim == 2:
            # Plot contour lines for the distribution
            # Create a grid
            grid_size = 100
            x_lin = jnp.linspace(
                self.mean[0] - 3 * self.sigma[0],
                self.mean[0] + 3 * self.sigma[0],
                grid_size,
            )
            y_lin = jnp.linspace(
                self.mean[1] - 3 * self.sigma[1],
                self.mean[1] + 3 * self.sigma[1],
                grid_size,
            )
            X, Y = jnp.meshgrid(x_lin, y_lin)
            grid = jnp.stack([X, Y], axis=-1).reshape(-1, 2)  # Shape: (grid_size**2, 2)

            # Compute log_prob for each grid point
            log_probs = self.log_prob(grid).reshape(grid_size, grid_size)

            # Plot contours
            ax.contour(X, Y, log_probs, levels=20, cmap="viridis")
            ax.set_xlim(
                self.mean[0] - 3 * self.sigma[0], self.mean[0] + 3 * self.sigma[0]
            )
            ax.set_ylim(
                self.mean[1] - 3 * self.sigma[1], self.mean[1] + 3 * self.sigma[1]
            )

            # Overlay scatter plot of samples
            ax.scatter(
                samples[:, 0],
                samples[:, 1],
                alpha=0.5,
                s=10,
                color="red",
                label="Samples",
            )

            ax.set_title("Multivariate Gaussian (2D)")
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            ax.legend()
            ax.grid(True)

        return fig
