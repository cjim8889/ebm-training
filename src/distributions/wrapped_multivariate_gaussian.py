import jax.numpy as jnp
import distrax
import chex
import jax
from .base import Target
import matplotlib.pyplot as plt


class WrappedMultivariateGaussian(Target):
    TIME_DEPENDENT = False

    def __init__(
        self,
        dim: int = 2,
        sigma: float = 1.0,
        box_size: float = 15.0,
        plot_bound_factor: float = 3.0,
        num_shift: int = 3,
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            log_Z=0.0,  # Not used since it's a wrapped Gaussian
            can_sample=True,
            n_plots=1,
            n_model_samples_eval=1000,
            n_target_samples_eval=1000,
            **kwargs,
        )

        self.sigma = jnp.asarray(sigma)
        if self.sigma.ndim == 0:
            # Scalar sigma: isotropic covariance
            self.scale_diag = jnp.full((dim,), self.sigma)
        else:
            # Per-dimension sigma
            if self.sigma.shape[0] != dim:
                raise ValueError(
                    f"Sigma shape {self.sigma.shape} does not match dimension {dim}."
                )
            self.scale_diag = self.sigma

        self.box_size = box_size
        self.num_shift = num_shift  # Number of shifts per dimension

        # Initialize the Multivariate Normal Distribution
        self.distribution = distrax.MultivariateNormalDiag(
            loc=jnp.zeros(dim), scale_diag=self.scale_diag
        )

        # Determine plot bounds based on sigma
        if self.sigma.ndim == 0:
            bound = self.sigma * plot_bound_factor
        else:
            bound = jnp.max(self.sigma) * plot_bound_factor
        self._plot_bound = float(bound)

        k = jnp.arange(-self.num_shift, self.num_shift + 1)
        self.kL = k * self.box_size

    def batch_log_prob(self, x: chex.Array) -> chex.Array:
        return jax.vmap(self.log_prob)(x)

    def log_prob(self, x: chex.Array) -> chex.Array:
        mu = self.distribution.loc[0]
        sigma = self.distribution.scale_diag[0]

        x_expanded = x[:, None]  # Shape: (D, 1)
        x_kL = x_expanded + self.kL  # Shape: (D, 2K + 1)

        # Compute exponentials: -((y_i + kL - mu)^2) / (2*sigma^2)
        exponent = -0.5 * ((x_kL - mu) / sigma) ** 2  # Shape: (D, 2K + 1)

        # Compute log-sum-exp for each dimension
        max_exponent = jnp.max(exponent, axis=1, keepdims=True)  # Shape: (D, 1)
        sum_exp = jnp.sum(jnp.exp(exponent - max_exponent), axis=1)  # Shape: (D,)
        log_sum_exp = jnp.squeeze(max_exponent, axis=1) + jnp.log(
            sum_exp
        )  # Shape: (D,)

        # Compute the log_prob for each dimension
        log_prob_dim = (
            -jnp.log(sigma * jnp.sqrt(2 * jnp.pi)) + log_sum_exp
        )  # Shape: (D,)

        # Total log_prob is the sum over all dimensions
        total_log_prob = jnp.sum(log_prob_dim)  # Scalar

        return total_log_prob

    def score(self, value: chex.Array) -> chex.Array:
        # Compute the gradient
        score = jax.grad(self.log_prob)(value)

        return score

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape = ()) -> chex.Array:
        # Sample from the base Gaussian
        samples = self.distribution.sample(seed=seed, sample_shape=sample_shape)

        # Apply modulo operation to enforce PBC
        wrapped_samples = jnp.mod(samples, self.box_size)

        return wrapped_samples

    def visualise(
        self,
        samples: chex.Array,
    ) -> plt.Figure:
        fig, ax = plt.subplots(1, figsize=(6, 6))
        if self.dim == 2:
            # Plot contour lines for the distribution within the box
            # Create a grid within [0, box_size)
            grid_size = 100
            x_lin = jnp.linspace(0, self.box_size[0], grid_size)
            y_lin = jnp.linspace(0, self.box_size[1], grid_size)
            X, Y = jnp.meshgrid(x_lin, y_lin)
            grid = jnp.stack([X, Y], axis=-1).reshape(-1, 2)  # Shape: (grid_size**2, 2)

            # Compute log_prob for each grid point
            log_probs = self.batch_log_prob(grid).reshape(grid_size, grid_size)

            # Plot contours
            ax.contour(X, Y, log_probs, levels=20, cmap="viridis")
            ax.set_xlim(0, self.box_size[0])
            ax.set_ylim(0, self.box_size[1])

            # Overlay scatter plot of samples
            ax.scatter(
                samples[:, 0],
                samples[:, 1],
                alpha=0.5,
                s=10,
                color="red",
                label="Samples",
            )

            # Draw box boundaries
            rect = plt.Rectangle(
                (0, 0),
                self.box_size[0],
                self.box_size[1],
                linewidth=2,
                edgecolor="black",
                facecolor="none",
            )
            ax.add_patch(rect)

            ax.set_title("Wrapped Multivariate Gaussian (2D)")
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            ax.legend()
            ax.grid(True)

            # Optionally, plot marginal distributions or scatter plots for selected pairs

        return fig
