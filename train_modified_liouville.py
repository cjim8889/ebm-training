import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import pandas as pd
import distrax
import wandb
import optax
import argparse
import chex

import abc
from typing import Optional, Callable, Union, Tuple, Dict, Any, Sequence, List
from jax import Array
from itertools import product

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import combinations

# from IPython.display import HTML
#

plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.figsize"] = [6.0, 4.0]

# Set the default precision for matmul operations to the highest available


def plot_contours_2D(
    log_prob_func, ax: Optional[plt.Axes] = None, bound: float = 3, levels: int = 20
):
    """Plot the contours of a 2D log prob function."""
    if ax is None:
        fig, ax = plt.subplots(1)
    n_points = 200
    x_points_dim1 = np.linspace(-bound, bound, n_points)
    x_points_dim2 = np.linspace(-bound, bound, n_points)
    x_points = np.array(list(product(x_points_dim1, x_points_dim2)))
    log_probs = log_prob_func(x_points)
    log_probs = jnp.clip(log_probs, a_min=-1000, a_max=None)
    x1 = x_points[:, 0].reshape(n_points, n_points)
    x2 = x_points[:, 1].reshape(n_points, n_points)
    z = log_probs.reshape(n_points, n_points)
    ax.contour(x1, x2, z, levels=levels)


def plot_marginal_pair(
    samples: chex.Array,
    ax: Optional[plt.Axes] = None,
    marginal_dims: Tuple[int, int] = (0, 1),
    bounds: Tuple[float, float] = (-5, 5),
    alpha: float = 0.5,
):
    """Plot samples from marginal of distribution for a given pair of dimensions."""
    if not ax:
        fig, ax = plt.subplots(1)
    samples = jnp.clip(samples, bounds[0], bounds[1])
    ax.plot(
        samples[:, marginal_dims[0]], samples[:, marginal_dims[1]], "o", alpha=alpha
    )


def plot_history(history):
    """Agnostic history plotter for quickly plotting a dictionary of logging info."""
    figure, axs = plt.subplots(len(history), 1, figsize=(7, 3 * len(history.keys())))
    if len(history.keys()) == 1:
        axs = [axs]  # make iterable
    elif len(history.keys()) == 0:
        return
    for i, key in enumerate(history):
        data = pd.Series(history[key])
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        if sum(data.isna()) > 0:
            data = data.dropna()
            print(f"NaN encountered in {key} history")
        axs[i].plot(data)
        axs[i].set_title(key)
    plt.tight_layout()


LogProbFn = Callable[[chex.Array], chex.Array]


def calculate_log_forward_ess(
    log_w: chex.Array, mask: Optional[chex.Array] = None, log_Z: Optional[float] = None
) -> chex.Array:
    """Calculate forward ess, either using exact log_Z if it is known, or via estimating it from the samples.
    NB: log_q = p(x)/q(x) where x ~ p(x).
    """
    if mask is None:
        mask = jnp.ones_like(log_w)

    chex.assert_equal_shape((log_w, mask))
    log_w = jnp.where(mask, log_w, jnp.zeros_like(log_w))  # make sure log_w finite

    if log_Z is None:
        log_z_inv = jax.nn.logsumexp(-log_w, b=mask) - jnp.log(jnp.sum(mask))
    else:
        log_z_inv = -log_Z

    # log ( Z * E_p[p(x)/q(x)] )
    log_z_times_expectation_p_of_p_div_q = jax.nn.logsumexp(log_w, b=mask) - jnp.log(
        jnp.sum(mask)
    )
    # ESS (as fraction of 1) = 1/E_p[p(x)/q(x)]
    # ESS = Z / ( Z * E_p[p(x)/q(x)] )
    # Log ESS = - log Z^{-1} -  log ( Z * E_p[p(x)/q(x)] )
    log_forward_ess = -log_z_inv - log_z_times_expectation_p_of_p_div_q
    return log_forward_ess


class Target(abc.ABC):
    """Abstraction of target distribution that allows our training and evaluation scripts to be generic."""

    def __init__(
        self,
        dim: int,
        log_Z: Optional[float],
        can_sample: bool,
        n_plots: int,
        n_model_samples_eval: int,
        n_target_samples_eval: Optional[int],
    ):
        self.n_model_samples_eval = n_model_samples_eval
        self.n_target_samples_eval = n_target_samples_eval
        self._dim = dim
        self._log_Z = log_Z
        self._n_plots = n_plots
        self._can_sample = can_sample

    @property
    def dim(self) -> int:
        """Dimensionality of the problem."""
        return self._dim

    @property
    def n_plots(self) -> int:
        """Number of matplotlib axes that samples are visualized on."""
        return self._n_plots

    @property
    def can_sample(self) -> bool:
        """Whether the target may be sampled form."""
        return self._can_sample

    @property
    def log_Z(self) -> Union[int, None]:
        """Log normalizing constant if available."""
        return self._log_Z

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        raise NotImplementedError

    @abc.abstractmethod
    def log_prob(self, value: chex.Array) -> chex.Array:
        """(Possibly unnormalized) target probability density."""

    def evaluate(
        self,
        model_log_prob_fn: LogProbFn,
        model_sample_and_log_prob_fn: Callable[
            [chex.PRNGKey, chex.Shape], Tuple[chex.Array, chex.Array]
        ],
        key: chex.PRNGKey,
    ) -> dict:
        """Evaluate a model. Note that reverse ESS will be estimated separately, so should not be estimated here."""
        key1, key2 = jax.random.split(key)

        info = {}

        if self.can_sample:
            assert self.n_target_samples_eval is not None
            samples_p = self.sample(key1, (self.n_target_samples_eval,))
            log_prob_q = model_log_prob_fn(samples_p)
            log_prob_p = self.log_prob(samples_p)
            log_w = log_prob_p - log_prob_q
            log_forward_ess = calculate_log_forward_ess(log_w, log_Z=self.log_Z)
            info.update(
                log_lik=jnp.mean(log_prob_q),
                log_forward_ess=log_forward_ess,
                forward_ess=jnp.exp(log_forward_ess),
            )
        return info

    @abc.abstractmethod
    def visualise(
        self,
        samples: chex.Array,
        axes: List[plt.Axes],
    ) -> None:
        """Visualise samples from the model."""


class GMM(Target):
    def __init__(
        self,
        dim: int = 2,
        n_mixes: int = 40,
        loc_scaling: float = 40,
        scale_scaling: float = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__(
            dim=dim,
            log_Z=0.0,
            can_sample=True,
            n_plots=1,
            n_model_samples_eval=1000,
            n_target_samples_eval=1000,
        )

        self.seed = seed
        self.n_mixes = n_mixes

        key = jax.random.PRNGKey(seed)
        logits = jnp.ones(n_mixes)
        mean = (
            jax.random.uniform(shape=(n_mixes, dim), key=key, minval=-1.0, maxval=1.0)
            * loc_scaling
        )
        scale = jnp.ones(shape=(n_mixes, dim)) * scale_scaling

        mixture_dist = distrax.Categorical(logits=logits)
        components_dist = distrax.Independent(
            distrax.Normal(loc=mean, scale=scale), reinterpreted_batch_ndims=1
        )
        self.distribution = distrax.MixtureSameFamily(
            mixture_distribution=mixture_dist,
            components_distribution=components_dist,
        )

        self._plot_bound = loc_scaling * 1.5

    def log_prob(self, x: chex.Array) -> chex.Array:
        log_prob = self.distribution.log_prob(x)
        return log_prob

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape = ()) -> chex.Array:
        return self.distribution.sample(seed=seed, sample_shape=sample_shape)

    def visualise(
        self,
        samples: chex.Array,
    ) -> plt.Figure:
        """Visualise samples from the model."""
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        plot_marginal_pair(samples, ax, bounds=(-self._plot_bound, self._plot_bound))
        if self.dim == 2:
            plot_contours_2D(self.log_prob, ax, bound=self._plot_bound, levels=50)

        return fig


class MultiDimGMM:
    def __init__(
        self,
        n_dim: int = 2,
        n_mixes: int = 40,
        loc_scaling: float = 40,
        scale_scaling: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.n_dim = n_dim
        self.n_mixes = n_mixes
        self.loc_scaling = loc_scaling
        self.scale_scaling = scale_scaling
        self.seed = seed

        key = jax.random.PRNGKey(seed)
        logits = jnp.ones(n_mixes)
        mean = (
            jax.random.uniform(shape=(n_mixes, n_dim), key=key, minval=-1.0, maxval=1.0)
            * loc_scaling
        )
        scale = jnp.ones(shape=(n_mixes, n_dim)) * scale_scaling

        mixture_dist = distrax.Categorical(logits=logits)
        components_dist = distrax.Independent(
            distrax.Normal(loc=mean, scale=scale), reinterpreted_batch_ndims=1
        )
        self.distribution = distrax.MixtureSameFamily(
            mixture_distribution=mixture_dist,
            components_distribution=components_dist,
        )

        self._plot_bound = loc_scaling * 1.5
        self.rng = np.random.default_rng(888)

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.distribution.log_prob(x)

    def sample(self, seed: chex.PRNGKey, sample_shape: tuple = ()) -> jnp.ndarray:
        return self.distribution.sample(seed=seed, sample_shape=sample_shape)

    def visualise(self, samples: jnp.ndarray) -> plt.Figure:
        """Visualise pairwise marginal distributions."""
        dims = list(range(self.n_dim))
        dim_pairs = list(combinations(dims, 2))

        # Randomly select at most 4 pairs of dimensions
        num_pairs = min(len(dim_pairs), 4)
        if len(dim_pairs) > 4:
            dim_pairs = self.rng.choice(dim_pairs, size=4, replace=False)
        else:
            dim_pairs = dim_pairs[:num_pairs]

        # Set up the subplot grid
        if num_pairs == 1:
            nrows, ncols = 1, 1
        elif num_pairs == 2:
            nrows, ncols = 1, 2
        elif num_pairs == 3:
            nrows, ncols = 2, 2  # Will have one empty subplot
        else:
            nrows, ncols = 2, 2

        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False
        )

        axes = axes.flatten()

        for idx, (dim1, dim2) in enumerate(dim_pairs):
            ax = axes[idx]
            marginal_samples = samples[:, [dim1, dim2]]
            self._plot_marginal_pair(marginal_samples, ax)
            self._plot_marginal_contours(ax, (dim1, dim2))
            ax.set_title(f"Dimensions {dim1} vs {dim2}")
            ax.set_xlabel(f"Dimension {dim1}")
            ax.set_ylabel(f"Dimension {dim2}")

        # Hide any unused axes
        for idx in range(num_pairs, nrows * ncols):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        return fig

    def _plot_marginal_contours(
        self,
        ax: Optional[plt.Axes],
        dims: Tuple[int, int],
        bound: float = None,
        levels: int = 20,
    ):
        """Plot the contours of the marginal 2D log prob function for specified dimensions."""
        if bound is None:
            bound = self._plot_bound
        n_points = 100
        x_points_dim1 = np.linspace(-bound, bound, n_points)
        x_points_dim2 = np.linspace(-bound, bound, n_points)
        x1_grid, x2_grid = np.meshgrid(x_points_dim1, x_points_dim2)
        grid_shape = x1_grid.shape
        x_grid = jnp.stack([x1_grid.ravel(), x2_grid.ravel()], axis=-1)

        # Compute marginal log probabilities over the selected dimensions
        log_probs = self._marginal_log_prob(x_grid, dims)
        log_probs = jnp.clip(log_probs, a_min=-1000, a_max=None)
        z = log_probs.reshape(grid_shape)
        ax.contour(x1_grid, x2_grid, z, levels=levels, cmap="viridis")

    def _marginal_log_prob(self, x: jnp.ndarray, dims: Tuple[int, int]) -> jnp.ndarray:
        """Compute the marginal log probability over the specified dimensions."""
        # Extract the means and scales for the specified dimensions
        mean = self.distribution.components_distribution.distribution.loc[:, dims]
        scale = self.distribution.components_distribution.distribution.scale[:, dims]
        logits = self.distribution.mixture_distribution.logits

        # Create a new mixture distribution over the selected dimensions
        mixture_dist = distrax.Categorical(logits=logits)
        components_dist = distrax.Independent(
            distrax.Normal(loc=mean, scale=scale), reinterpreted_batch_ndims=1
        )
        marginal_distribution = distrax.MixtureSameFamily(
            mixture_distribution=mixture_dist,
            components_distribution=components_dist,
        )
        return marginal_distribution.log_prob(x)

    def _plot_marginal_pair(
        self,
        samples: jnp.ndarray,
        ax: Optional[plt.Axes],
        bounds: Tuple[float, float] = None,
        alpha: float = 0.5,
    ):
        """Plot samples from the marginal distribution for a given pair of dimensions."""
        if bounds is None:
            bounds = (-self._plot_bound, self._plot_bound)
        samples_clipped = jnp.clip(samples, bounds[0], bounds[1])
        ax.scatter(
            samples_clipped[:, 0],
            samples_clipped[:, 1],
            alpha=alpha,
            s=10,
            label="Samples",
        )
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        ax.legend()


class MultivariateGaussian(Target):
    """
    Multivariate Gaussian Distribution with Sigma * I covariance.

    Parameters:
    - dim: int
        Dimensionality of the Gaussian.
    - sigma: float or jnp.ndarray, default=1.0
        Standard deviation of the distribution. If a float is provided, the same sigma is used for all dimensions.
        If an array is provided, it should have shape (dim,) for per-dimension sigma.
    - plot_bound_factor: float, default=3.0
        Factor to determine the plotting bounds based on sigma.
    """

    def __init__(
        self, dim: int = 2, sigma: float = 1.0, plot_bound_factor: float = 3.0, **kwargs
    ):
        """
        Initializes the MultivariateGaussian distribution.

        Args:
            dim (int): Dimensionality of the Gaussian.
            sigma (float or jnp.ndarray): Standard deviation(s). Scalar for isotropic, array for anisotropic.
            plot_bound_factor (float): Factor to determine the plotting bounds.
            **kwargs: Additional arguments to pass to the base Target class.
        """
        super().__init__(
            dim=dim,
            log_Z=0.0,  # Not used since it's a single Gaussian
            can_sample=True,
            n_plots=1,
            n_model_samples_eval=1000,
            n_target_samples_eval=1000,
            **kwargs,
        )

        self.sigma = jnp.asarray(sigma)
        if self.sigma.ndim == 0:
            # Scalar sigma: isotropic covariance
            scale_diag = jnp.full((dim,), self.sigma)
        else:
            # Per-dimension sigma
            if self.sigma.shape[0] != dim:
                raise ValueError(
                    f"Sigma shape {self.sigma.shape} does not match dimension {dim}."
                )
            scale_diag = self.sigma

        # Initialize the Multivariate Normal Distribution
        self.distribution = distrax.MultivariateNormalDiag(
            loc=jnp.zeros(dim), scale_diag=scale_diag
        )

        # Determine plot bounds based on sigma
        if self.sigma.ndim == 0:
            bound = self.sigma * plot_bound_factor
        else:
            bound = jnp.max(self.sigma) * plot_bound_factor
        self._plot_bound = float(bound)

    def log_prob(self, x: chex.Array) -> chex.Array:
        """
        Computes the log probability density of the input samples.

        Args:
            x (chex.Array): Input samples with shape (..., dim).

        Returns:
            chex.Array: Log probability densities.
        """
        return self.distribution.log_prob(x)

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape = ()) -> chex.Array:
        """
        Generates samples from the distribution.

        Args:
            seed (chex.PRNGKey): JAX random key for reproducibility.
            sample_shape (chex.Shape): Shape of the samples to generate.

        Returns:
            chex.Array: Generated samples with shape `sample_shape + (dim,)`.
        """
        return self.distribution.sample(seed=seed, sample_shape=sample_shape)

    def visualise(
        self,
        samples: chex.Array,
        axes: List[plt.Axes],
    ) -> None:
        """
        Visualizes the distribution and samples.

        Args:
            samples (chex.Array): Samples to visualize.
            axes (List[plt.Axes]): List of matplotlib axes for plotting.
        """
        assert (
            len(axes) == self.n_plots
        ), f"Expected {self.n_plots} axes, got {len(axes)}."

        ax = axes[0]
        if self.dim == 2:
            # Plot contour lines for the distribution
            plot_contours_2D(self.log_prob, ax, bound=self._plot_bound, levels=20)
            # Overlay scatter plot of samples
            plot_marginal_pair(
                samples, ax, bounds=(-self._plot_bound, self._plot_bound)
            )
        else:
            # For higher dimensions, visualize projections or other summaries
            ax.hist(samples, bins=50, density=True)
            ax.set_title(f"Histogram of samples in 1D projection (dim={self.dim})")
            ax.set_xlim(-self._plot_bound, self._plot_bound)
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")


class TimeVelocityField(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, key, input_dim, hidden_dim, depth=3):
        # Define an MLP with time as an input
        self.mlp = eqx.nn.MLP(
            in_size=input_dim + 1,  # x and t
            out_size=input_dim,
            width_size=hidden_dim,
            activation=jax.nn.sigmoid,
            depth=depth,
            key=key,
        )

    def __call__(self, x, t):
        # Concatenate x and t
        t_expanded = jnp.array([t])
        x_t = jnp.concatenate([x, t_expanded], axis=-1)

        return self.mlp(x_t)


@eqx.filter_jit
def divergence_velocity(
    v_theta: Callable[[Array, float], Array], x: Array, t: float
) -> float:
    """
    Compute the divergence of the velocity field v_theta at point x and time t.

    Args:
        v_theta (Callable[[Array, float], Array]): The velocity field function that takes x and t and returns the velocity vector.
        x (Array): The point at which to compute the divergence.
        t (float): The time at which to compute the divergence.

    Returns:
        float: The divergence of v_theta at (x, t).
    """

    def v_x(x):
        return v_theta(x, t)

    jacobian = jax.jacrev(v_x)(x)
    div_v = jnp.trace(jacobian)
    return div_v


def inverse_tangent_schedule(T=64, alpha=5):
    x_atan = jnp.linspace(0, 1, T)
    t_atan = 1 - (jnp.arctan(alpha * x_atan) / jnp.arctan(alpha))
    return jnp.flip(t_atan)


def inverse_power_schedule(T=64, gamma=0.5):
    x_pow = np.linspace(0, 1, T)
    t_pow = 1 - x_pow**gamma
    return jnp.flip(t_pow)


def sample_langevin_dynamics(
    key: chex.PRNGKey,
    xs: chex.Array,
    t: float,
    score_function: Callable[[Array, float], Array],
    num_steps: int = 3,
    eta: float = 0.01,
) -> chex.Array:
    """
    Sample from the Langevin dynamics using the score function and noise.

    Args:
        key (chex.PRNGKey): JAX random key for randomness.
        xs (chex.Array): Initial samples of shape (N, D).
        t (float): Current time.
        score_function (Callable[[Array, float], Array]): Function to compute the score (gradient of log-density).
        num_steps (int, optional): Number of Langevin steps to perform. Defaults to 3.
        eta (float, optional): Step size for Langevin dynamics. Defaults to 0.01.

    Returns:
        chex.Array: The samples after applying Langevin dynamics.
    """

    def step(carry, _):
        i, x, subkey = carry
        noise = jax.random.normal(subkey, shape=x.shape)
        g = jax.vmap(lambda x: score_function(x, t))(x)
        x = x + (eta**2 / 2.0) * g + eta * noise
        return (i + 1, x, jax.random.fold_in(subkey, i)), None

    (_, final_xs, _), _ = jax.lax.scan(step, (0, xs, key), None, length=num_steps)
    return final_xs


@eqx.filter_jit
def time_batched_sample_langevin_dynamics(
    key: chex.PRNGKey,
    xs: chex.Array,
    ts: chex.Array,
    score_function: Callable[[Array, float], Array],
    num_steps: int = 3,
    eta: float = 0.01,
) -> chex.Array:
    """
    Apply Langevin dynamics over a batch of time steps and samples.

    Args:
        key (chex.PRNGKey): JAX random key.
        xs (chex.Array): Samples array of shape (N, D).
        ts (chex.Array): Times array of shape (N,).
        score_function (Callable[[Array, float], Array]): Function to compute the score.
        num_steps (int, optional): Number of Langevin steps. Defaults to 3.
        eta (float, optional): Step size. Defaults to 0.01.

    Returns:
        chex.Array: The samples after applying Langevin dynamics.
    """

    keys = jax.random.split(key, ts.shape[0] + 1)
    key, subkeys = keys[0], keys[1:]
    final_xs = jax.vmap(
        lambda x, t, subkey: sample_langevin_dynamics(
            subkey, x, t, score_function, num_steps, eta
        )
    )(xs, ts, subkeys)
    return final_xs


def sample_hamiltonian_monte_carlo(
    key: jax.random.PRNGKey,
    time_dependent_log_density: Callable[[Array, float], float],
    x: Array,
    t: float,
    num_steps: int = 10,
    integration_steps: int = 3,
    eta: float = 0.1,
    rejection_sampling: bool = False,
) -> Array:
    """
    Perform a single chain of Hamiltonian Monte Carlo sampling.

    Args:
        key (jax.random.PRNGKey): Random key for randomness.
        time_dependent_log_density (Callable[[Array, float], float]): Function to compute log-density at given x and t.
        x (Array): Initial sample of shape (D,).
        t (float): Current time.
        num_steps (int, optional): Number of HMC steps. Defaults to 10.
        integration_steps (int, optional): Number of leapfrog integration steps. Defaults to 3.
        eta (float, optional): Step size for integration. Defaults to 0.1.

    Returns:
        Array: The sample after HMC steps.
    """
    dim = x.shape[-1]
    covariance = jnp.eye(dim)
    inv_covariance = covariance
    grad_log_prob = jax.grad(lambda x: time_dependent_log_density(x, t))

    def kinetic_energy(v):
        return 0.5 * v.T @ inv_covariance @ v

    def hamiltonian(x, v):
        return -time_dependent_log_density(x, t) + kinetic_energy(v)

    def integration_step(carry, _):
        x, v = carry
        x = x + eta * inv_covariance @ v
        v = v + eta * grad_log_prob(x)
        return (x, v), _

    def hmc_step(x_current, key):
        x = x_current
        key, subkey = jax.random.split(key)

        # Sample momentum
        v = jax.random.normal(subkey, (dim,))
        current_h = hamiltonian(x, v)

        # Initial half step for momentum
        v = v + 0.5 * eta * grad_log_prob(x)

        # Leapfrog integration
        (x, v), _ = jax.lax.scan(
            integration_step, (x, v), None, length=integration_steps
        )

        # Final half step for momentum
        v = v + 0.5 * eta * grad_log_prob(x)

        if rejection_sampling:
            # Compute acceptance probability
            proposed_h = hamiltonian(x, v)
            accept_ratio = jnp.minimum(1.0, jnp.exp(current_h - proposed_h))

            # Accept or reject
            key, subkey = jax.random.split(key)
            uniform_sample = jax.random.uniform(subkey)
            accept = uniform_sample < accept_ratio

            new_x = jax.lax.cond(accept, lambda _: x, lambda _: x_current, operand=None)

            return new_x, None
        else:
            return x, None

    # Run the chain
    keys = jax.random.split(key, num_steps)

    # return hmc_step(init_state, keys[0])
    final_x, _ = jax.lax.scan(hmc_step, x, keys)

    return final_x


@eqx.filter_jit
def time_batched_sample_hamiltonian_monte_carlo(
    key: jax.random.PRNGKey,
    time_dependent_log_density: Callable[[Array, float], float],
    xs: chex.Array,
    ts: chex.Array,
    num_steps: int = 10,
    integration_steps: int = 3,
    eta: float = 0.1,
    rejection_sampling: bool = False,
) -> chex.Array:
    """
    Apply HMC sampling over batches of samples and times.

    Args:
        key (jax.random.PRNGKey): Random key.
        time_dependent_log_density (Callable[[Array, float], float]): Log-density function.
        xs (chex.Array): Samples array of shape (N, D).
        ts (chex.Array): Times array of shape (N,).
        num_steps (int, optional): Number of HMC steps. Defaults to 10.
        integration_steps (int, optional): Number of leapfrog steps. Defaults to 3.
        eta (float, optional): Step size. Defaults to 0.1.

    Returns:
        chex.Array: Samples after HMC steps.
    """
    keys = jax.random.split(key, xs.shape[0] * xs.shape[1])

    final_xs = jax.vmap(
        lambda xs, t, keys: jax.vmap(
            lambda x, subkey: sample_hamiltonian_monte_carlo(
                subkey,
                time_dependent_log_density,
                x,
                t,
                num_steps,
                integration_steps,
                eta,
                rejection_sampling,
            )
        )(xs, keys)
    )(xs, ts, keys.reshape((xs.shape[0], xs.shape[1], -1)))

    return final_xs


@eqx.filter_jit
def generate_samples_without_weight(
    v_theta: Callable,
    initial_samples: jnp.ndarray,
    ts: jnp.ndarray,
) -> jnp.ndarray:
    def step(carry, t):
        x_prev, t_prev = carry
        samples = x_prev + (t - t_prev) * jax.vmap(lambda x: v_theta(x, t))(x_prev)
        return (samples, t), samples

    _, output = jax.lax.scan(step, (initial_samples, 0.0), ts)

    return output


def generate_samples(
    v_theta: Callable,
    num_samples: int,
    key: jnp.ndarray,
    ts: jnp.ndarray,
    sample_fn: Callable,
) -> jnp.ndarray:
    key, subkey = jax.random.split(key)
    initial_samples = sample_fn(subkey, num_samples)
    samples = generate_samples_without_weight(v_theta, initial_samples, ts)
    return samples


@eqx.filter_jit
def generate_samples_with_langevin(
    v_theta: Callable,
    num_samples: int,
    key: jnp.ndarray,
    ts: jnp.ndarray,
    sample_fn: Callable,
    score_function: Callable,
    num_steps: int = 3,
    eta: float = 0.01,
) -> jnp.ndarray:
    key, subkey = jax.random.split(key)
    initial_samples = sample_fn(subkey, num_samples)
    samples = generate_samples_without_weight(v_theta, initial_samples, ts)
    final_samples = time_batched_sample_langevin_dynamics(
        subkey, samples, ts, score_function, num_steps, eta
    )

    return final_samples


@eqx.filter_jit
def generate_samples_with_hmc(
    v_theta: Callable,
    sample_fn: Callable,
    time_dependent_log_density: Callable,
    num_samples: int,
    key: jnp.ndarray,
    ts: jnp.ndarray,
    num_steps: int = 3,
    integration_steps: int = 3,
    eta: float = 0.01,
    rejection_sampling: bool = False,
) -> jnp.ndarray:
    key, subkey = jax.random.split(key)
    initial_samples = sample_fn(subkey, num_samples)
    samples = generate_samples_without_weight(v_theta, initial_samples, ts)
    final_samples = time_batched_sample_hamiltonian_monte_carlo(
        subkey,
        time_dependent_log_density,
        samples,
        ts,
        num_steps,
        integration_steps,
        eta,
        rejection_sampling,
    )

    return final_samples


@eqx.filter_jit
def generate_samples_with_log_prob(
    v_theta: Callable,
    initial_samples: jnp.ndarray,
    initial_log_probs: jnp.ndarray,
    ts: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    def step(carry, t):
        x_prev, log_prob_prev, t_prev = carry

        dt = t - t_prev
        v_t = jax.vmap(lambda x: v_theta(x, t))(x_prev)

        x_next = x_prev + dt * v_t

        # Compute divergence
        div_v_t = jax.vmap(lambda x: divergence_velocity(v_theta, x, t))(x_prev)
        log_prob_next = log_prob_prev - dt * div_v_t

        return (x_next, log_prob_next, t), None

    # Initialize carry with initial samples and their log-probabilities
    carry = (initial_samples, initial_log_probs, 0.0)
    (samples, log_probs, _), _ = jax.lax.scan(step, carry, ts)

    return samples, log_probs


@eqx.filter_jit
def reverse_time_flow(
    v_theta: Callable,
    final_samples: jnp.ndarray,
    final_time: float,
    ts: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Reverse ts to integrate backward
    ts_rev = ts[::-1]

    def step(carry, t):
        x_next, log_prob_next, t_next = carry

        dt = t - t_next  # dt is negative for backward integration
        v_t = jax.vmap(lambda x: v_theta(x, t))(x_next)
        x_prev = x_next + dt * v_t  # Since dt < 0, this moves backward

        # Compute divergence
        div_v_t = jax.vmap(lambda x: divergence_velocity(v_theta, x, t))(x_next)
        log_prob_prev = log_prob_next + dt * div_v_t  # Accumulate log_prob

        return (x_prev, log_prob_prev, t), None

    # Initialize carry with final samples and zero log-probabilities
    num_samples = final_samples.shape[0]
    initial_log_probs = jnp.zeros(num_samples)
    carry = (final_samples, initial_log_probs, final_time)

    (xs, log_probs, _), _ = jax.lax.scan(step, carry, ts_rev)

    return xs, log_probs


@eqx.filter_jit
def generate_samples_with_log_prob_rk4(
    v_theta: Callable,
    initial_samples: jnp.ndarray,
    initial_log_probs: jnp.ndarray,
    ts: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    def step(carry, t):
        x_prev, log_prob_prev, t_prev = carry
        dt = t - t_prev

        # RK4 for x
        k1 = jax.vmap(lambda x: v_theta(x, t_prev))(x_prev)
        k2 = jax.vmap(lambda x: v_theta(x, t_prev + 0.5 * dt))(x_prev + 0.5 * dt * k1)
        k3 = jax.vmap(lambda x: v_theta(x, t_prev + 0.5 * dt))(x_prev + 0.5 * dt * k2)
        k4 = jax.vmap(lambda x: v_theta(x, t_prev + dt))(x_prev + dt * k3)
        x_next = x_prev + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # RK4 for divergence
        d1 = jax.vmap(lambda x: divergence_velocity(v_theta, x, t_prev))(x_prev)
        d2 = jax.vmap(lambda x: divergence_velocity(v_theta, x, t_prev + 0.5 * dt))(
            x_prev + 0.5 * dt * k1
        )
        d3 = jax.vmap(lambda x: divergence_velocity(v_theta, x, t_prev + 0.5 * dt))(
            x_prev + 0.5 * dt * k2
        )
        d4 = jax.vmap(lambda x: divergence_velocity(v_theta, x, t_prev + dt))(
            x_prev + dt * k3
        )
        divergence_avg = (d1 + 2 * d2 + 2 * d3 + d4) / 6.0
        log_prob_next = log_prob_prev - dt * divergence_avg

        return (x_next, log_prob_next, t), None

    # Initialize carry with initial samples and their log-probabilities
    carry = (initial_samples, initial_log_probs, ts[0] if len(ts) > 0 else 0.0)
    (samples, log_probs, _), _ = jax.lax.scan(step, carry, ts)

    return samples, log_probs


@eqx.filter_jit
def reverse_time_flow_rk4(
    v_theta: Callable,
    final_samples: jnp.ndarray,
    final_log_probs: jnp.ndarray,
    final_time: float,
    ts: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Reverse ts to integrate backward
    ts_rev = ts[::-1]

    def step(carry, t):
        x_next, log_prob_next, t_next = carry
        dt = t - t_next  # dt is negative for backward integration

        # RK4 for x (backward)
        k1 = jax.vmap(lambda x: v_theta(x, t_next))(x_next)
        k2 = jax.vmap(lambda x: v_theta(x, t_next + 0.5 * dt))(x_next + 0.5 * dt * k1)
        k3 = jax.vmap(lambda x: v_theta(x, t_next + 0.5 * dt))(x_next + 0.5 * dt * k2)
        k4 = jax.vmap(lambda x: v_theta(x, t_next + dt))(x_next + dt * k3)
        x_prev = x_next + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # RK4 for divergence (backward)
        d1 = jax.vmap(lambda x: divergence_velocity(v_theta, x, t_next))(x_next)
        d2 = jax.vmap(lambda x: divergence_velocity(v_theta, x, t_next + 0.5 * dt))(
            x_next + 0.5 * dt * k1
        )
        d3 = jax.vmap(lambda x: divergence_velocity(v_theta, x, t_next + 0.5 * dt))(
            x_next + 0.5 * dt * k2
        )
        d4 = jax.vmap(lambda x: divergence_velocity(v_theta, x, t_next + dt))(
            x_next + dt * k3
        )
        divergence_avg = (d1 + 2 * d2 + 2 * d3 + d4) / 6.0
        log_prob_prev = (
            log_prob_next - dt * divergence_avg
        )  # Note the sign change for backward

        return (x_prev, log_prob_prev, t), None

    # Initialize carry with final samples and their log-probabilities
    carry = (final_samples, final_log_probs, final_time)
    (xs, log_probs, _), _ = jax.lax.scan(step, carry, ts_rev)

    return xs, log_probs


@jax.jit
def compute_log_effective_sample_size(log_p: Array, log_q: Array) -> Array:
    """
    Compute the log Effective Sample Size (log ESS Fraction) between two sets of log probabilities.

    The Effective Sample Size (ESS) is a measure of how representative a set of weighted samples is
    of the target distribution. It quantifies the number of independent, equally weighted samples
    that the weighted samples are equivalent to. To enhance numerical stability, this function
    operates entirely in log-space.

    **Definition:**
        ESS = (sum_i w_i)^2 / sum_i w_i^2

    **Log-Space Transformation:**
        log(ESS / N) = 2 * logsumexp(log_w) - logsumexp(2 * log_w) - log(N)

    Here,
        w_i = p(x_i) / q(x_i),
        N = number of samples,
        log_w = log_p - log_q,
        ESS Fraction = ESS / N

    **Parameters:**
        log_p (Array): Log probabilities from distribution p. Shape: (N,)
        log_q (Array): Log probabilities from distribution q. Shape: (N,)

    **Returns:**
        Array: Scalar representing log(ESS / N), where ESS is the Effective Sample Size
               and N is the total number of samples. This value ranges from log(1/N) to log(1).

    **Example:**
        >>> import jax.numpy as jnp
        >>> log_p = jnp.log(jnp.array([0.5, 0.5]))
        >>> log_q = jnp.log(jnp.array([0.5, 0.5]))
        >>> compute_log_effective_sample_size(log_p, log_q)
        DeviceArray(-0.6931472, dtype=float32)

    **Notes:**
        - The ESS Fraction is normalized to range between 1/N and 1, where N is the number of samples.
        - A higher ESS Fraction indicates better representation of the target distribution.
        - This function assumes that `log_p` and `log_q` are aligned such that each `log_p[i]` corresponds
          to `log_q[i]` for the same sample `x_i`.
    """
    # Ensure that log_p and log_q have the same shape
    if log_p.shape != log_q.shape:
        raise ValueError(
            f"Shape mismatch: log_p has shape {log_p.shape}, "
            f"but log_q has shape {log_q.shape}."
        )

    # Number of samples (N)
    n_samples = log_p.shape[0]

    # Compute log weights: log_w_i = log_p(x_i) - log_q(x_i)
    log_w = log_p - log_q  # Shape: (N,)

    # Compute log(sum_i w_i) using logsumexp for numerical stability
    # Since w_i = p(x_i) / q(x_i), sum_i w_i = sum_i exp(log_w_i)
    log_sum_w = jax.scipy.special.logsumexp(log_w)  # Scalar

    # Compute log(sum_i w_i^2) using logsumexp on 2 * log_w
    # sum_i w_i^2 = sum_i exp(2 * log_w_i)
    log_sum_w_sq = jax.scipy.special.logsumexp(2.0 * log_w)  # Scalar

    # Compute log(ESS Fraction) = 2 * log(sum_i w_i) - log(sum_i w_i^2) - log(N)
    log_ess_frac = (2.0 * log_sum_w) - log_sum_w_sq - jnp.log(n_samples)  # Scalar

    return log_ess_frac


@jax.jit
def compute_expectation_log_q(log_probs_q: Array) -> Array:
    """
    Compute the expectation of log probabilities from distribution q.

    Args:
        log_probs_q (Array): Log probabilities from distribution q. Shape: (N,)

    Returns:
        Array: The mean of log_probs_q as a scalar.
    """
    expectation: Array = jnp.mean(log_probs_q)
    return expectation


@eqx.filter_jit
def estimate_diagnostics(
    v_theta: Callable[[Array, float], Array],
    num_samples: int,
    key: jax.random.PRNGKey,
    ts: Array,
    log_prob_p_fn: Callable[[Array], Array],
    sample_p_fn: Callable[[jax.random.PRNGKey, int], Array],
    sample_initial_fn: Callable[[jax.random.PRNGKey, int], Array],
    base_log_prob: Callable[[Array], Array],
    final_time: float = 1.0,
) -> Tuple[Array, Array, Array]:
    """
    Estimate diagnostic metrics including KL divergence, ESS, and expectation of log q.

    This function performs the following steps:
    1. Generates samples from distribution p(x).
    2. Computes log probabilities under p(x).
    3. Performs reverse-time integration to compute samples and log probabilities under q(x).
    4. Computes the KL divergence between p and q.
    5. Computes the Effective Sample Size (ESS).
    6. Computes the expectation of log q(x).

    Args:
        v_theta (Callable[[Array, float], Array]): Vector field function for reverse-time flow.
        num_samples (int): Number of samples to generate.
        key (PRNGKey): PRNG key for random number generation.
        ts (Array): Time steps for the reverse-time integration.
        log_prob_p_fn (Callable[[Array], Array]): Function to compute log probabilities under p(x).
        sample_fn (Callable[[PRNGKey, int], Array]): Function to generate samples from p(x).
        base_log_prob (Callable[[Array], Array]): Function to compute log probabilities of the base distribution q(x(0)).
        final_time (float, optional): The final time for reverse-time integration. Defaults to 1.0.

    Returns:
        Tuple[Array, Array, Array]: A tuple containing:
            - KL divergence between p and q.
            - Effective Sample Size (ESS).
            - Expectation of log q(x).
    """
    # Generate samples from p(x)
    key, subkey = jax.random.split(key)
    samples_p: Array = sample_p_fn(subkey, num_samples)
    log_probs_p: Array = log_prob_p_fn(samples_p)  # Compute log p(x) for these samples

    # Perform reverse-time integration to compute samples and log probabilities under q(x)
    # samples_rev, log_probs_q = reverse_time_flow_rk4(
    #     v_theta, samples_p, jnp.zeros((num_samples,), dtype=jnp.float32), final_time, ts
    # )
    samples_rev, log_probs_q = reverse_time_flow(v_theta, samples_p, final_time, ts)

    # Compute log q(x(T)) = log q(x(0)) + accumulated log_probs
    base_log_probs: Array = base_log_prob(samples_rev)  # Compute log q(x(0))
    log_q_x: Array = base_log_probs + log_probs_q

    # Compute KL divergence: KL(p || q) = E_p[log p(x) - log q(x)]
    kl_divergence: Array = jnp.mean(log_probs_p - log_q_x)

    # Compute Effective Sample Size (ESS)
    key, subkey = jax.random.split(key)
    initial_samples = sample_initial_fn(subkey, num_samples)
    initial_log_probs = base_log_prob(initial_samples)
    samples_q, log_q_samples_q = generate_samples_with_log_prob(
        v_theta, initial_samples, initial_log_probs, ts
    )
    ess: Array = jnp.exp(
        compute_log_effective_sample_size(log_prob_p_fn(samples_q), log_q_samples_q)
    )

    # Compute expectation of log q(x)
    expectation_log_q: Array = compute_expectation_log_q(log_q_x)

    return kl_divergence, ess, expectation_log_q


def epsilon(v_theta, x, dt_log_density, t, score_fn):
    score = score_fn(x, t)
    div_v = divergence_velocity(v_theta, x, t)
    v = v_theta(x, t)
    lhs = div_v + jnp.dot(v, score)
    return jnp.nan_to_num(lhs + dt_log_density, posinf=1.0, neginf=-1.0)


batched_epsilon = jax.vmap(epsilon, in_axes=(None, 0, 0, None, None))
time_batched_epsilon = eqx.filter_jit(
    jax.vmap(batched_epsilon, in_axes=(None, 0, 0, 0, None))
)


def loss_fn(v_theta, xs, ts, time_derivative_log_density, score_fn):
    dt_log_unormalised_density = jax.vmap(
        lambda xs, t: jax.vmap(lambda x: time_derivative_log_density(x, t))(xs),
        in_axes=(0, 0),
    )(xs, ts)
    dt_log_density = dt_log_unormalised_density - jnp.mean(
        dt_log_unormalised_density, axis=-1, keepdims=True
    )
    epsilons = time_batched_epsilon(v_theta, xs, dt_log_density, ts, score_fn)
    return jnp.mean(epsilons**2)


def train_velocity_field(
    initial_density: Target,
    target_density: Target,
    v_theta: Callable[[Array, float], Array],
    key: jax.random.PRNGKey,
    N: int = 512,
    B: int = 64,
    num_epochs: int = 200,
    num_steps: int = 100,
    optimiser: str = "adam",
    learning_rate: float = 1e-03,
    momentum: float = 0.9,
    nestrov: bool = True,
    T: int = 32,
    gradient_norm: float = 1.0,
    mcmc_type: str = "langevin",
    num_mcmc_steps: int = 5,
    num_mcmc_integration_steps: int = 3,
    eta: float = 0.01,
    schedule: str = "linear",
    schedule_alpha: float = 5.0,  # Added
    schedule_gamma: float = 0.5,
    run_name: str = "velocity_field_training",
    **kwargs: Any,
) -> Any:
    """
    Train the velocity field v_theta to match initial and target densities.

    Args:
        initial_density (Target): The initial density distribution.
        target_density (Target): The target density distribution.
        v_theta (Callable[[Array, float], Array]): Velocity field function to train.
        key (jax.random.PRNGKey): Random key for randomness.
        N (int, optional): Number of samples per batch. Defaults to 512.
        num_epochs (int, optional): Number of training epochs. Defaults to 200.
        num_steps (int, optional): Number of steps per epoch. Defaults to 100.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-03.
        T (int, optional): Number of time steps. Defaults to 32.
        gradient_norm (float, optional): Gradient clipping norm. Defaults to 1.0.
        mcmc_type (str, optional): Type of MCMC sampler ('langevin' or 'hmc'). Defaults to "langevin".
        num_mcmc_steps (int, optional): Number of MCMC steps. Defaults to 5.
        num_mcmc_integration_steps (int, optional): Number of integration steps for MCMC. Defaults to 3.
        eta (float, optional): Step size for MCMC samplers. Defaults to 0.01.
        schedule (str, optional): Time schedule type ('linear', 'inverse_tangent', 'inverse_power'). Defaults to "linear".
        **kwargs (Any): Additional arguments.

    Returns:
        Any: Trained velocity field v_theta.
    """
    print("Arguments and Their Types & Values:")
    print(
        f"1. initial_density (Type: {type(initial_density).__name__}) = {initial_density}"
    )
    print(
        f"2. target_density (Type: {type(target_density).__name__}) = {target_density}"
    )
    print(f"3. v_theta (Type: {type(v_theta).__name__}) = {v_theta}")
    print(f"4. key (Type: {type(key).__name__}) = {key}")
    print(f"5. N (Type: {type(N).__name__}) = {N}")
    print(f"6. num_epochs (Type: {type(num_epochs).__name__}) = {num_epochs}")
    print(f"7. num_steps (Type: {type(num_steps).__name__}) = {num_steps}")
    print(f"8. optimiser (Type: {type(optimiser).__name__}) = '{optimiser}'")
    print(f"9. learning_rate (Type: {type(learning_rate).__name__}) = {learning_rate}")
    print(f"10. momentum (Type: {type(momentum).__name__}) = {momentum}")
    print(f"11. nestrov (Type: {type(nestrov).__name__}) = {nestrov}")
    print(f"12. T (Type: {type(T).__name__}) = {T}")
    print(f"13. gradient_norm (Type: {type(gradient_norm).__name__}) = {gradient_norm}")
    print(f"14. mcmc_type (Type: {type(mcmc_type).__name__}) = '{mcmc_type}'")
    print(
        f"15. num_mcmc_steps (Type: {type(num_mcmc_steps).__name__}) = {num_mcmc_steps}"
    )
    print(
        f"16. num_mcmc_integration_steps (Type: {type(num_mcmc_integration_steps).__name__}) = {num_mcmc_integration_steps}"
    )
    print(f"17. eta (Type: {type(eta).__name__}) = {eta}")
    print(f"18. schedule (Type: {type(schedule).__name__}) = '{schedule}'")
    print(
        f"19. schedule_alpha (Type: {type(schedule_alpha).__name__}) = {schedule_alpha}"
    )
    print(
        f"20. schedule_gamma (Type: {type(schedule_gamma).__name__}) = {schedule_gamma}"
    )
    print(f"21. run_name (Type: {type(run_name).__name__}) = '{run_name}'")

    if kwargs:
        print("\nKeyword Arguments (kwargs) and Their Types & Values:")
        for key_arg, value in kwargs.items():
            print(f"{key_arg} (Type: {type(value).__name__}) = {value}")

    # Handle logging hyperparameters
    wandb.init(
        project="liouville",
        config={
            "input_dim": initial_density.dim,
            "T": T,
            "N": N,
            "num_epochs": num_epochs,
            "num_steps": num_steps,
            "optimiser": optimiser,
            "learning_rate": learning_rate,
            "momentum": momentum,
            "nestrov": nestrov,
            "gradient_norm": gradient_norm,
            "hidden_dim": v_theta.mlp.width_size,
            "depth": v_theta.mlp.depth,
            "mcmc_type": mcmc_type,
            "num_mcmc_steps": num_mcmc_steps,
            "num_mcmc_integration_steps": num_mcmc_integration_steps,
            "eta": eta,
            "schedule": schedule,
            **kwargs,
        },
        name=run_name,
        reinit=True,
    )

    # Set up various functions
    def time_dependent_log_density(x, t):
        return (1 - t) * initial_density.log_prob(x) + t * target_density.log_prob(x)

    def score_function(x, t):
        return jax.grad(lambda x: time_dependent_log_density(x, t))(x)

    def time_derivative_log_density(x, t):
        return jax.grad(lambda t: time_dependent_log_density(x, t))(t)

    def sample_initial(key, num_samples):
        return initial_density.sample(key, (num_samples,))

    def sample_target(key, num_samples):
        return target_density.sample(key, (num_samples,))

    # Set up optimizer
    gradient_clipping = optax.clip_by_global_norm(gradient_norm)

    if optimiser == "adam":
        opt = optax.adam(learning_rate)
    elif optimiser == "sgd":
        opt = optax.sgd(learning_rate, momentum=momentum, nesterov=nestrov)
    elif optimiser == "adamw":
        opt = optax.adamw(learning_rate)
    elif optimiser == "adamax":
        opt = optax.adamax(learning_rate)
    else:
        opt = optax.adam(learning_rate)

    optimizer = optax.chain(gradient_clipping, opt)
    opt_state = optimizer.init(eqx.filter(v_theta, eqx.is_inexact_array))

    # Generate time steps
    key, subkey = jax.random.split(key)
    if schedule == "linear":
        ts = jnp.linspace(0, 1, T)
    elif schedule == "inverse_tangent":
        ts = inverse_tangent_schedule(T, alpha=schedule_alpha)
    elif schedule == "inverse_power":
        ts = inverse_power_schedule(T, gamma=schedule_gamma)
    else:
        ts = jnp.linspace(0, 1, T)

    @eqx.filter_jit
    def step(v_theta, opt_state, xs):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(
            v_theta,
            xs,
            ts,
            time_derivative_log_density=time_derivative_log_density,
            score_fn=score_function,
        )
        updates, opt_state = optimizer.update(grads, opt_state, v_theta)
        v_theta = eqx.apply_updates(v_theta, updates)
        return v_theta, opt_state, loss

    for epoch in range(num_epochs):
        key, subkey = jax.random.split(key)
        if mcmc_type == "langevin":
            samples = generate_samples_with_langevin(
                v_theta,
                N,
                subkey,
                ts,
                sample_initial,
                score_function,
                num_mcmc_steps,
                eta,
            )
        elif mcmc_type == "hmc":
            samples = generate_samples_with_hmc(
                v_theta,
                sample_initial,
                time_dependent_log_density,
                N,
                subkey,
                ts,
                num_mcmc_steps,
                num_mcmc_integration_steps,
                eta,
                False,
            )
        else:
            samples = generate_samples(v_theta, N, subkey, ts, sample_initial)

        epoch_loss = 0.0
        for s in range(num_steps):
            key, subkey = jax.random.split(key)
            samps = jax.random.choice(subkey, samples, (B,), replace=False, axis=1)
            v_theta, opt_state, loss = step(v_theta, opt_state, samps)
            epoch_loss += loss

            if s % 20 == 0:
                wandb.log({"loss": loss})

        avg_loss = epoch_loss / num_steps
        print(f"Epoch {epoch}, Average Loss: {avg_loss}")
        wandb.log({"epoch": epoch, "average_loss": avg_loss})

        if epoch % 20 == 0:
            linear_ts = jnp.linspace(0, 1, T)
            key, subkey = jax.random.split(key)
            val_samples = generate_samples(
                v_theta, N, subkey, linear_ts, sample_initial
            )
            fig = target_density.visualise(val_samples[-1])

            wandb.log({"validation_samples": wandb.Image(fig)})

            key, subkey = jax.random.split(key)
            kl_div, ess, E_log_q = estimate_diagnostics(
                v_theta,
                10000,
                subkey,
                linear_ts,
                target_density.log_prob,
                sample_target,
                sample_initial,
                initial_density.log_prob,
            )

            wandb.log(
                {
                    "forward_kl_divergence": kl_div,
                    "ESS": ess,
                    "Expectation_log_q": E_log_q,
                }
            )

            print(f"KL Divergence: {kl_div}, ESS: {ess}, E[log q]: {E_log_q}")

            plt.close(fig)

    wandb.finish()
    return v_theta


# Define Main Function
def main(args):
    # Initialize WandB
    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key)

    if args.input_dim == 2:
        gmm = GMM(
            dim=2,
            n_mixes=args.gmm_n_mixes,
            seed=subkey[0],
        )
    elif args.input_dim > 2:
        gmm = MultiDimGMM(
            n_dim=args.input_dim,
            n_mixes=args.gmm_n_mixes,
            seed=subkey[0],
        )
    else:
        raise ValueError("Input dimension must be at least 2.")

    initial = MultivariateGaussian(
        dim=args.input_dim,
        sigma=args.initial_sigma,
    )

    # Initialize the velocity field
    key, model_key = jax.random.split(key)
    v_theta = TimeVelocityField(
        model_key,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    )

    # Train the velocity field
    key, subkey = jax.random.split(key)
    train_velocity_field(
        initial_density=initial,
        target_density=gmm,
        v_theta=v_theta,
        key=subkey,
        N=args.N,
        B=args.B,
        num_epochs=args.num_epochs,
        num_steps=args.num_steps,
        optimiser=args.optimiser,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        T=args.T,
        gradient_norm=args.gradient_norm,
        mcmc_type=args.mcmc_type,  # Added
        num_mcmc_steps=args.num_mcmc_steps,  # Added
        num_mcmc_integration_steps=args.num_mcmc_integration_steps,  # Added
        eta=args.eta,
        schedule=args.schedule,  # Added
        schedule_alpha=args.schedule_alpha,  # Added
        schedule_gamma=args.schedule_gamma,
        nestrov=True,
        run_name=args.run_name,
    )


# Define Argument Parser
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train a Velocity Field with Configurable Hyperparameters"
    )

    # General Hyperparameters
    parser.add_argument(
        "--seed", type=int, default=80801, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="liouville",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="velocity_field_training",
        help="Name of the WandB run.",
    )

    # Training Hyperparameters
    parser.add_argument(
        "--input_dim", type=int, default=2, help="Dimensionality of the problem."
    )
    parser.add_argument("--T", type=int, default=64, help="Number of time steps.")
    parser.add_argument(
        "--N",
        type=int,
        default=1024,
        help="Number of samples for training at each time step.",
    )
    parser.add_argument(
        "--B",
        type=int,
        default=64,
        help="Number of samples to use for each training step",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=2000, help="Number of training epochs."
    )
    parser.add_argument(
        "--num_steps", type=int, default=100, help="Number of training steps per epoch."
    )
    parser.add_argument(
        "--optimiser",
        type=str,
        choices=["adam", "sgd", "adamw", "adamax"],
        default="adam",
        help="Optimizer to use for training.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="Momentum for the optimizer."
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="Hidden dimension size of the MLP."
    )
    parser.add_argument(
        "--depth", type=int, default=3, help="Depth (number of layers) of the MLP."
    )
    parser.add_argument(
        "--gradient_norm", type=float, default=1.0, help="Gradient clipping norm."
    )

    # GMM Hyperparameters
    parser.add_argument(
        "--gmm_n_mixes", type=int, default=40, help="Number of mixtures in the GMM."
    )

    # Initial Distribution Hyperparameters
    parser.add_argument(
        "--initial_sigma",
        type=float,
        default=20.0,
        help="Sigma (standard deviation) for the initial Gaussian.",
    )

    # MCMC Hyperparameters
    parser.add_argument(
        "--mcmc_type",
        type=str,
        choices=["langevin", "hmc"],
        default="hmc",
        help="Type of MCMC sampler to use ('langevin' or 'hmc').",
    )
    parser.add_argument(
        "--num_mcmc_steps",
        type=int,
        default=5,
        help="Number of MCMC steps to perform.",
    )
    parser.add_argument(
        "--num_mcmc_integration_steps",
        type=int,
        default=5,
        help="Number of integration steps for MCMC samplers (applicable for HMC).",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="Step size parameter for MCMC samplers.",
    )

    # Time Schedule Hyperparameters
    parser.add_argument(
        "--schedule",
        type=str,
        choices=["linear", "inverse_tangent", "inverse_power"],
        default="linear",
        help="Time schedule type ('linear', 'inverse_tangent', 'inverse_power'). Defaults to 'linear'.",
    )
    parser.add_argument(
        "--schedule_alpha",
        type=float,
        default=5.0,
        help="Alpha parameter for the inverse_tangent schedule. Applicable only if schedule is 'inverse_tangent'.",
    )
    parser.add_argument(
        "--schedule_gamma",
        type=float,
        default=0.5,
        help="Gamma parameter for the inverse_power schedule. Applicable only if schedule is 'inverse_power'.",
    )

    return parser.parse_args()


# Entry Point
if __name__ == "__main__":
    args = parse_arguments()
    main(args)
