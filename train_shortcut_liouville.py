import jax
import jax.numpy as jnp
import equinox as eqx
import distrax
import optax
import chex
import wandb
import numpy as np
import argparse
from jax_md import energy, space
from matplotlib import pyplot as plt
from itertools import product
from typing import Optional, Callable, Union, Tuple, Any, List


# Base Target class and utility functions
class Target:
    """Base class for distributions"""

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
        return self._dim

    @property
    def n_plots(self) -> int:
        return self._n_plots

    @property
    def can_sample(self) -> bool:
        return self._can_sample

    @property
    def log_Z(self) -> Union[int, None]:
        return self._log_Z

    def log_prob(self, value: chex.Array) -> chex.Array:
        raise NotImplementedError


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


def rejection_sampling(
    n_samples: int,
    proposal: distrax.Distribution,
    target_log_prob_fn: Callable,
    k: float,
    key: chex.PRNGKey,
) -> chex.Array:
    """Rejection sampling. See Pattern Recognition and ML by Bishop Chapter 11.1"""
    # Note: This currently is not written to work inside of jax.jit or jax.vmap.
    key1, key2, key3 = jax.random.split(key, 3)
    n_samples_propose = n_samples * 10
    z_0, log_q_z0 = proposal._sample_n_and_log_prob(key, n=n_samples_propose)
    u_0 = (
        jax.random.uniform(key=key2, shape=(n_samples_propose,)) * k * jnp.exp(log_q_z0)
    )
    accept = jnp.exp(target_log_prob_fn(z_0)) > u_0
    samples = z_0[accept]
    if samples.shape[0] >= n_samples:
        return samples[:n_samples]
    else:
        required_samples = n_samples - samples.shape[0]
        new_samples = rejection_sampling(
            required_samples, proposal, target_log_prob_fn, k, key3
        )
        samples = jnp.concatenate([samples, new_samples], axis=0)
        return samples


class Energy:
    """
    https://zenodo.org/record/3242635#.YNna8uhKjIW
    """

    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def _energy(self, x):
        raise NotImplementedError()

    def energy(self, x, temperature=None):
        assert x.shape[-1] == self._dim, "`x` does not match `dim`"
        if temperature is None:
            temperature = 1.0
        return self._energy(x) / temperature

    def force(self, x, temperature=None):
        e_func = lambda x: jnp.sum(self.energy(x, temperature=temperature))
        return -jax.grad(e_func)(x)


class DoubleWellEnergy(Energy):
    def __init__(self):
        dim = 2
        a = -0.5
        b = -6.0
        c = 1.0
        super().__init__(dim)
        self._a = a
        self._b = b
        self._c = c

    def _energy(self, x):
        d = x[:, [0]]
        v = x[:, 1:]
        e1 = self._a * d + self._b * d**2 + self._c * d**4
        e2 = jnp.sum(0.5 * v**2, axis=-1, keepdims=True)
        return e1 + e2

    def log_prob(self, x):
        if len(x.shape) == 1:
            x = jnp.expand_dims(x, axis=0)
        return jnp.squeeze(-self.energy(x) - self.log_Z)

    @property
    def log_Z(self):
        log_Z_dim0 = jnp.log(11784.50927)
        log_Z_dim1 = 0.5 * jnp.log(2 * jnp.pi)
        return log_Z_dim0 + log_Z_dim1

    def sample_first_dimension(self, key: chex.Array, n: int) -> chex.Array:
        # see fab.sampling.rejection_sampling_test.py
        if self._a == -0.5 and self._b == -6 and self._c == 1.0:
            # Define target.
            def target_log_prob(x):
                return -(x**4) + 6 * x**2 + 1 / 2 * x

            TARGET_Z = 11784.50927

            # Define proposal params
            component_mix = jnp.array([0.2, 0.8])
            means = jnp.array([-1.7, 1.7])
            scales = jnp.array([0.5, 0.5])

            # Define proposal
            mix = distrax.Categorical(component_mix)
            com = distrax.Normal(means, scales)

            proposal = distrax.MixtureSameFamily(
                mixture_distribution=mix, components_distribution=com
            )

            k = TARGET_Z * 3

            samples = rejection_sampling(
                n_samples=n,
                proposal=proposal,
                target_log_prob_fn=target_log_prob,
                k=k,
                key=key,
            )
            return samples
        else:
            raise NotImplementedError

    def sample(self, key: chex.PRNGKey, shape: chex.Shape):
        if self._a == -0.5 and self._b == -6 and self._c == 1.0:
            assert len(shape) == 1
            key1, key2 = jax.random.split(key=key)
            dim1_samples = self.sample_first_dimension(key=key1, n=shape[0])
            dim2_samples = distrax.Normal(jnp.array(0.0), jnp.array(1.0)).sample(
                seed=key2, sample_shape=shape
            )
            return jnp.stack([dim1_samples, dim2_samples], axis=-1)
        else:
            raise NotImplementedError


class ManyWellEnergy(Target):
    def __init__(self, dim: int = 32):
        assert dim % 2 == 0
        self.n_wells = dim // 2
        self.double_well_energy = DoubleWellEnergy()

        log_Z = self.double_well_energy.log_Z * self.n_wells
        super().__init__(
            dim=dim,
            log_Z=log_Z,
            can_sample=False,
            n_plots=1,
            n_model_samples_eval=2000,
            n_target_samples_eval=10000,
        )

        self.centre = 1.7
        self.max_dim_for_all_modes = (
            40  # otherwise we get memory issues on huuuuge test set
        )
        if self.dim < self.max_dim_for_all_modes:
            dim_1_vals_grid = jnp.meshgrid(
                *[jnp.array([-self.centre, self.centre]) for _ in range(self.n_wells)]
            )
            dim_1_vals = jnp.stack([dim.flatten() for dim in dim_1_vals_grid], axis=-1)
            n_modes = 2**self.n_wells
            assert n_modes == dim_1_vals.shape[0]
            test_set = jnp.zeros((n_modes, dim))
            test_set = test_set.at[:, jnp.arange(dim) % 2 == 0].set(dim_1_vals)
            self.modes_test_set = test_set
        else:
            raise NotImplementedError("still need to implement this")

        self.shallow_well_bounds = [-1.75, -1.65]
        self.deep_well_bounds = [1.7, 1.8]
        self._plot_bound = 3.0

        self.double_well_samples = self.double_well_energy.sample(
            key=jax.random.PRNGKey(0), shape=(int(1e6),)
        )

        if self.n_target_samples_eval < self.modes_test_set.shape[0]:
            print("Evaluation occuring on subset of the modes test set.")

    def log_prob(self, x):
        return jnp.sum(
            jnp.stack(
                [
                    self.double_well_energy.log_prob(x[..., i * 2 : i * 2 + 2])
                    for i in range(self.n_wells)
                ],
                axis=-1,
            ),
            axis=-1,
        )

    def log_prob_2D(self, x):
        """Marginal 2D pdf - useful for plotting."""
        return self.double_well_energy.log_prob(x)

    def get_target_log_prob_marginal_pair(self, i: int, j: int, total_dim: int):
        def log_prob_marginal_pair(x_2d):
            x = jnp.zeros((x_2d.shape[0], total_dim))
            x = x.at[:, i].set(x_2d[:, 0])
            x = x.at[:, j].set(x_2d[:, 1])
            return self.log_prob(x)

        return log_prob_marginal_pair

    def visualise(self, samples: chex.Array) -> plt.Figure:
        alpha = 0.3
        plotting_bounds = (-3, 3)
        dim = samples.shape[-1]
        fig, axs = plt.subplots(2, 2, sharex="row", sharey="row", figsize=(10, 8))

        for i in range(2):
            for j in range(2):
                target_log_prob = self.get_target_log_prob_marginal_pair(i, j + 2, dim)
                plot_contours_2D(
                    target_log_prob, bound=self._plot_bound, ax=axs[i, j], levels=20
                )
                plot_marginal_pair(
                    samples,
                    marginal_dims=(i, j + 2),
                    ax=axs[i, j],
                    bounds=plotting_bounds,
                    alpha=alpha,
                )

                if j == 0:
                    axs[i, j].set_ylabel(f"$x_{{{i + 1}}}$")
                if i == 1:
                    axs[i, j].set_xlabel(f"$x_{{{j + 3}}}$")

        plt.tight_layout()
        return fig

    def sample(self, key: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        key1, key2 = jax.random.split(key)
        dw_sample_indices = jax.random.randint(
            minval=0,
            maxval=self.double_well_samples.shape[0],
            key=key1,
            shape=(sample_shape[0] * self.n_wells,),
        )
        dw_samples = self.double_well_samples[dw_sample_indices]
        samples_p = jnp.reshape(dw_samples, (-1, self.dim))

        return samples_p

    def get_eval_samples(
        self, key: chex.PRNGKey, n: int
    ) -> Tuple[chex.Array, chex.Array]:
        key1, key2 = jax.random.split(key)
        dw_sample_indices = jax.random.randint(
            minval=0,
            maxval=self.double_well_samples.shape[0],
            key=key1,
            shape=(n * self.n_wells,),
        )
        dw_samples = self.double_well_samples[dw_sample_indices]
        samples_p = jnp.reshape(dw_samples, (-1, self.dim))

        if n < self.modes_test_set.shape[0]:
            mode_sample_indices = jax.random.choice(
                a=jnp.arange(self.modes_test_set.shape[0]),
                key=key2,
                shape=(n,),
                replace=False,
            )
            samples_modes = self.modes_test_set[mode_sample_indices]
        else:
            samples_modes = self.modes_test_set
        return samples_p, samples_modes


class LennardJonesEnergy(Target):
    def __init__(
        self,
        dim: int,
        n_particles: int,
        data_path: str,
        data_path_train: Optional[str] = None,
        data_path_val: Optional[str] = None,
        key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    ):
        super().__init__(
            dim=dim,
            log_Z=None,
            can_sample=False,
            n_plots=10,
            n_model_samples_eval=1000,
            n_target_samples_eval=None,
        )
        self.n_particles = n_particles
        self.n_spatial_dim = dim // n_particles
        self.key = key

        self.data_path = data_path
        self.data_path_train = data_path_train
        self.data_path_val = data_path_val

        self._test_set = self.setup_test_set()
        self._val_set = self.setup_val_set()
        self._train_set = self.setup_train_set()

        self._mask = jnp.triu(jnp.ones((n_particles, n_particles), dtype=bool), k=1)

        displacement_fn, _ = space.periodic(5.0)
        self.energy_fn = energy.lennard_jones_pair(
            displacement_fn, sigma=1.0, epsilon=1.0
        )

    def log_prob(self, x: chex.Array) -> chex.Array:
        return -self.energy_fn(x.reshape(self.n_particles, self.n_spatial_dim))

    def sample(
        self, key: jax.random.PRNGKey, sample_shape: chex.Shape = ()
    ) -> chex.Array:
        raise NotImplementedError(
            "Sampling is not implemented for MultiDoubleWellEnergy"
        )

    def setup_test_set(self):
        data = np.load(self.data_path, allow_pickle=True)
        data = remove_mean(jnp.array(data), self.n_particles, self.n_spatial_dim)
        return data

    def setup_train_set(self):
        data = np.load(self.data_path, allow_pickle=True)
        data = remove_mean(jnp.array(data), self.n_particles, self.n_spatial_dim)
        return data

    def setup_val_set(self):
        data = np.load(self.data_path, allow_pickle=True)
        data = remove_mean(jnp.array(data), self.n_particles, self.n_spatial_dim)
        return data

    def interatomic_dist(self, x):
        x = x.reshape(-1, self.n_particles, self.n_spatial_dim)
        diffs = x[:, :, None, :] - x[:, None, :, :]
        mask = jnp.triu(jnp.ones((self.n_particles, self.n_particles)), k=1) == 1
        distances = jnp.linalg.norm(diffs, axis=-1)
        distances = distances[:, mask]
        return distances

    def batched_log_prob(self, xs):
        return jax.vmap(self.log_prob)(xs)

    def visualise(self, samples: chex.Array) -> plt.Figure:
        self.key, subkey = jax.random.split(self.key)
        test_data_smaller = jax.random.choice(
            subkey, self._test_set, shape=(1000,), replace=False
        )

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        dist_samples = self.interatomic_dist(samples)
        dist_test = self.interatomic_dist(test_data_smaller)

        axs[0].hist(
            dist_samples.flatten(),
            bins=100,
            alpha=0.5,
            density=True,
            histtype="step",
            linewidth=4,
        )
        axs[0].hist(
            dist_test.flatten(),
            bins=100,
            alpha=0.5,
            density=True,
            histtype="step",
            linewidth=4,
        )
        axs[0].set_xlabel("Interatomic distance")
        axs[0].legend(["Generated data", "Test data"])

        energy_samples = -self.batched_log_prob(samples)
        energy_test = -self.batched_log_prob(test_data_smaller)

        min_energy = -26
        max_energy = 0

        axs[1].hist(
            energy_test,
            bins=100,
            density=True,
            alpha=0.4,
            range=(min_energy, max_energy),
            color="g",
            histtype="step",
            linewidth=4,
            label="Test data",
        )
        axs[1].hist(
            energy_samples,
            bins=100,
            density=True,
            alpha=0.4,
            range=(min_energy, max_energy),
            color="r",
            histtype="step",
            linewidth=4,
            label="Generated data",
        )
        axs[1].set_xlabel("Energy")
        axs[1].legend()

        fig.canvas.draw()
        return fig


class WrappedMultivariateGaussian(Target):
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


class MultivariateGaussian(Target):
    def __init__(
        self, dim: int = 2, sigma: float = 1.0, plot_bound_factor: float = 3.0
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

        self.distribution = distrax.MultivariateNormalDiag(
            loc=jnp.zeros(dim), scale_diag=scale_diag
        )

    def log_prob(self, x: chex.Array) -> chex.Array:
        return self.distribution.log_prob(x)

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape = ()) -> chex.Array:
        return self.distribution.sample(seed=seed, sample_shape=sample_shape)


class GMM(Target):
    def __init__(
        self,
        key: chex.PRNGKey,
        dim: int = 2,
        n_mixes: int = 40,
        loc_scaling: float = 40,
        scale_scaling: float = 1.0,
    ) -> None:
        super().__init__(
            dim=dim,
            log_Z=0.0,
            can_sample=True,
            n_plots=1,
            n_model_samples_eval=1000,
            n_target_samples_eval=1000,
        )

        self.n_mixes = n_mixes

        logits = jnp.ones(n_mixes)
        mean = jnp.array(
            [
                [-0.2995, 21.4577],
                [-32.9218, -29.4376],
                [-15.4062, 10.7263],
                [-0.7925, 31.7156],
                [-3.5498, 10.5845],
                [-12.0885, -7.8626],
                [-38.2139, -26.4913],
                [-16.4889, 1.4817],
                [15.8134, 24.0009],
                [-27.1176, -17.4185],
                [14.5287, 33.2155],
                [-8.2320, 29.9325],
                [-6.4473, 4.2326],
                [36.2190, -37.1068],
                [-25.1815, -10.1266],
                [-15.5920, 34.5600],
                [-25.9272, -18.4133],
                [-27.9456, -37.4624],
                [-23.3496, 34.3839],
                [17.8487, 19.3869],
                [2.1037, -20.5073],
                [6.7674, -37.3478],
                [-28.9026, -20.6212],
                [25.2375, 23.4529],
                [-17.7398, -1.4433],
                [25.5824, 39.7653],
                [15.8753, 5.4037],
                [26.8195, -23.5521],
                [7.4538, -31.0122],
                [-27.7234, -20.6633],
                [18.0989, 16.0864],
                [-23.6941, 12.0843],
                [21.9589, -5.0487],
                [1.5273, 9.2682],
                [24.8151, 38.4078],
                [-30.8249, -14.6588],
                [15.7204, 33.1420],
                [34.8083, 35.2943],
                [7.9606, -34.7833],
                [3.6797, -25.0242],
            ]
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
        fig, ax = plt.subplots(1, figsize=(6, 6))
        plot_marginal_pair(samples, ax, bounds=(-self._plot_bound, self._plot_bound))
        if self.dim == 2:
            plot_contours_2D(self.log_prob, ax, bound=self._plot_bound, levels=50)

        return fig


def compute_distances(x, n_particles, n_dimensions, mask, L, epsilon=1e-8):
    x = x.reshape(n_particles, n_dimensions)

    # Compute pairwise distances efficiently using broadcasting
    # Shape of diffs: [n_particles, n_particles, n_dimensions]
    diffs = x[:, jnp.newaxis, :] - x[jnp.newaxis, :, :]

    # Apply Minimum Image Convention
    diffs = diffs - L * jnp.round(diffs / L)

    # Compute squared distances and add epsilon for numerical stability
    sq_dists = jnp.sum(diffs**2, axis=-1) + epsilon

    # Compute the square root to get distances
    distances = jnp.sqrt(sq_dists)

    # Apply the mask to filter out invalid pairs (e.g., self-pairs)
    distances = distances[mask]

    return distances


# MultiDoubleWellEnergy class conforming to Target interface
class MultiDoubleWellEnergy(Target):
    def __init__(
        self,
        dim: int,
        n_particles: int,
        data_path_val: Optional[str] = None,
        data_path_test: Optional[str] = None,
        key: jax.random.PRNGKey = jax.random.PRNGKey(0),
        a: float = 0.9,
        b: float = -4.0,
        c: float = 0.0,
        offset: float = 4.0,
        L: float = 10.0,
    ):
        super().__init__(
            dim=dim,
            log_Z=None,
            can_sample=False,
            n_plots=10,
            n_model_samples_eval=1000,
            n_target_samples_eval=None,
        )
        self.n_particles = n_particles
        self.n_spatial_dim = dim // n_particles
        self.key = key

        self.data_path_val = data_path_val
        self.data_path_test = data_path_test

        self.val_set_size = 1000

        self.a = a
        self.b = b
        self.c = c
        self.offset = offset
        self.L = L

        self._val_set = self.setup_val_set()
        self._test_set = self.setup_test_set()

        self._mask = jnp.triu(jnp.ones((n_particles, n_particles), dtype=bool), k=1)

    def multi_double_well_energy(self, x):
        dists = compute_distances(
            x, self.n_particles, self.n_spatial_dim, self._mask, self.L
        )
        dists = dists - self.offset

        energies = self.a * dists**4 + self.b * dists**2 + self.c

        total_energy = jnp.sum(energies)
        return total_energy

    def log_prob(self, x: chex.Array) -> chex.Array:
        return -self.multi_double_well_energy(x)

    def score(self, x: chex.Array) -> chex.Array:
        return jax.grad(self.log_prob)(x)

    def sample(
        self, key: jax.random.PRNGKey, sample_shape: chex.Shape = ()
    ) -> chex.Array:
        raise NotImplementedError(
            "Sampling is not implemented for MultiDoubleWellEnergy"
        )

    def setup_test_set(self):
        data = np.load(self.data_path_test, allow_pickle=True)
        data = data % self.L
        return data

    def setup_val_set(self):
        data = np.load(self.data_path_val, allow_pickle=True)
        data = data % self.L
        return data

    def interatomic_dist(self, x):
        x = x.reshape(-1, self.n_particles, self.n_spatial_dim)
        distances = jax.vmap(
            lambda x: compute_distances(
                x, self.n_particles, self.n_spatial_dim, self._mask, self.L
            )
        )(x)

        return distances

    def batched_log_prob(self, xs):
        return jax.vmap(self.log_prob)(xs)

    def visualise(self, samples: chex.Array) -> plt.Figure:
        self.key, subkey = jax.random.split(self.key)
        test_data_smaller = jax.random.choice(
            subkey, self._test_set, shape=(1000,), replace=False
        )

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        dist_samples = self.interatomic_dist(samples)
        dist_test = self.interatomic_dist(test_data_smaller)

        axs[0].hist(
            dist_samples.flatten(),
            bins=100,
            alpha=0.5,
            density=True,
            histtype="step",
            linewidth=4,
        )
        axs[0].hist(
            dist_test.flatten(),
            bins=100,
            alpha=0.5,
            density=True,
            histtype="step",
            linewidth=4,
        )
        axs[0].set_xlabel("Interatomic distance")
        axs[0].legend(["Generated data", "Test data"])

        energy_samples = -self.batched_log_prob(samples)
        energy_test = -self.batched_log_prob(test_data_smaller)

        min_energy = -26
        max_energy = 0

        axs[1].hist(
            energy_test,
            bins=100,
            density=True,
            alpha=0.4,
            range=(min_energy, max_energy),
            color="g",
            histtype="step",
            linewidth=4,
            label="Test data",
        )
        axs[1].hist(
            energy_samples,
            bins=100,
            density=True,
            alpha=0.4,
            range=(min_energy, max_energy),
            color="r",
            histtype="step",
            linewidth=4,
            label="Generated data",
        )
        axs[1].set_xlabel("Energy")
        axs[1].legend()

        fig.canvas.draw()
        return fig


class AnnealedDistribution(Target):
    def __init__(
        self,
        initial_distribution: Target,
        target_distribution: Target,
        dim: int = 2,
        max_score: float = 1.0,
    ):
        super().__init__(
            dim=dim,
            log_Z=0.0,
            can_sample=False,
            n_plots=1,
            n_model_samples_eval=1000,
            n_target_samples_eval=1000,
        )
        self.initial_distribution = initial_distribution
        self.target_distribution = target_distribution
        self.max_score = max_score

    def log_prob(self, xs: chex.Array) -> chex.Array:
        return self.time_dependent_log_prob(xs, 1.0)

    def time_dependent_log_prob(self, xs: chex.Array, t: chex.Array) -> chex.Array:
        initial_prob = (1 - t) * self.initial_distribution.log_prob(xs)
        target_prob = t * self.target_distribution.log_prob(xs)
        return initial_prob + target_prob

    def time_derivative(self, xs: chex.Array, t: float) -> chex.Array:
        return jax.grad(lambda t: self.time_dependent_log_prob(xs, t))(t)

    def score_fn(self, xs: chex.Array, t: float) -> chex.Array:
        return jax.grad(lambda x: self.time_dependent_log_prob(x, t))(xs)

    def sample_initial(self, key: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        return self.initial_distribution.sample(key, sample_shape)


class ShortcutTimeVelocityField(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, key, input_dim, hidden_dim, depth=3):
        # Define an MLP with time as an input
        self.mlp = eqx.nn.MLP(
            in_size=input_dim + 2,  # x, t, and d
            out_size=input_dim,
            width_size=hidden_dim,
            activation=jax.nn.sigmoid,
            depth=depth,
            key=key,
        )

    def __call__(self, xs, t, d):
        # Concatenate x and t
        t_expanded = jnp.array([t, d])
        x_td = jnp.concatenate([xs, t_expanded], axis=-1)

        return self.mlp(x_td)


class ShortcutTimeVelocityFieldWithPairwiseFeature(eqx.Module):
    mlp: eqx.nn.MLP
    n_particles: int
    n_spatial_dim: int

    def __init__(self, key, n_particles, n_spatial_dim, hidden_dim, depth=3):
        self.n_particles = n_particles
        self.n_spatial_dim = n_spatial_dim
        input_dim = n_particles * n_spatial_dim
        num_pairwise = n_particles * (n_particles - 1) // 2
        self.mlp = eqx.nn.MLP(
            in_size=input_dim + 2 + num_pairwise,  # x, t, d, pairwise distances
            out_size=input_dim,
            width_size=hidden_dim,
            activation=jax.nn.sigmoid,
            depth=depth,
            key=key,
        )

    def __call__(self, xs, t, d):
        # Reshape xs to (n_particles, n_spatial_dim)
        xs_reshaped = xs.reshape(self.n_particles, self.n_spatial_dim)
        # Compute pairwise distances
        diffs = (
            xs_reshaped[:, None, :] - xs_reshaped[None, :, :]
        )  # Shape (n_particles, n_particles, n_spatial_dim)
        dists = jnp.linalg.norm(diffs, axis=-1)  # Shape (n_particles, n_particles)
        # Get unique pairwise distances
        idx_upper = jnp.triu_indices(self.n_particles, k=1)
        pairwise_dists = dists[idx_upper]  # Shape (num_pairwise,)
        # Flatten xs back to original shape
        xs_flat = xs_reshaped.flatten()
        # Concatenate xs_flat, t, d, pairwise_dists
        t_d = jnp.array([t, d])
        x_td = jnp.concatenate([xs_flat, t_d, pairwise_dists], axis=0)
        return self.mlp(x_td)


# Utility functions
@eqx.filter_jit
def euler_integrate(
    v_theta: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    initial_samples: jnp.ndarray,
    ts: jnp.ndarray,
    shift_fn: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
) -> jnp.ndarray:
    def step(carry, t):
        x_prev, t_prev = carry
        d = t - t_prev
        samples = x_prev + d * jax.vmap(lambda x: v_theta(x, t, d))(x_prev)
        samples = shift_fn(samples)

        return (samples, t), samples

    _, output = jax.lax.scan(step, (initial_samples, 0.0), ts)
    return output


def sample_hamiltonian_monte_carlo(
    key: jax.random.PRNGKey,
    time_dependent_log_density: Callable[[chex.Array, float], float],
    x: chex.Array,
    t: float,
    num_steps: int = 10,
    integration_steps: int = 3,
    eta: float = 0.1,
    rejection_sampling: bool = False,
    shift_fn: Callable[[chex.Array], chex.Array] = lambda x: x,
) -> chex.Array:
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
        # Apply the modular wrapping to enforce PBC
        x = shift_fn(x)

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

        # Finalize the proposal
        x = shift_fn(x)

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
    time_dependent_log_density: Callable[[chex.Array, float], float],
    xs: chex.Array,
    ts: chex.Array,
    num_steps: int = 10,
    integration_steps: int = 3,
    eta: float = 0.1,
    rejection_sampling: bool = False,
    shift_fn: Callable[[chex.Array], chex.Array] = lambda x: x,
) -> chex.Array:
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
                shift_fn,
            )
        )(xs, keys)
    )(xs, ts, keys.reshape((xs.shape[0], xs.shape[1], -1)))

    return final_xs


@eqx.filter_jit
def generate_samples(
    key: jax.random.PRNGKey,
    v_theta: Callable[[jnp.ndarray, float], jnp.ndarray],
    num_samples: int,
    ts: jnp.ndarray,
    sample_fn: Callable[[jax.random.PRNGKey, Tuple[int, ...]], jnp.ndarray],
    integration_fn: Callable[
        [Callable[[jnp.ndarray, float], jnp.ndarray], jnp.ndarray, jnp.ndarray],
        jnp.ndarray,
    ] = euler_integrate,
    shift_fn: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
) -> jnp.ndarray:
    key, subkey = jax.random.split(key)
    initial_samples = sample_fn(subkey, (num_samples,))
    samples = integration_fn(v_theta, initial_samples, ts, shift_fn)
    return samples


@eqx.filter_jit
def generate_samples_with_different_ts(
    key: jax.random.PRNGKey,
    v_theta: Callable[[jnp.ndarray, float], jnp.ndarray],
    num_samples: int,
    tss: List[jnp.ndarray],
    sample_fn: Callable[[jax.random.PRNGKey, Tuple[int, ...]], jnp.ndarray],
    integration_fn: Callable[
        [Callable[[jnp.ndarray, float], jnp.ndarray], jnp.ndarray, jnp.ndarray],
        jnp.ndarray,
    ] = euler_integrate,
    shift_fn: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
) -> jnp.ndarray:
    key, subkey = jax.random.split(key)
    initial_samples = sample_fn(subkey, (num_samples,))

    samples = [integration_fn(v_theta, initial_samples, ts, shift_fn) for ts in tss]
    return samples


@eqx.filter_jit
def generate_samples_with_hmc_correction(
    key: jax.random.PRNGKey,
    v_theta: Callable[[jnp.ndarray, float], jnp.ndarray],
    sample_fn: Callable[[jax.random.PRNGKey, Tuple[int, ...]], jnp.ndarray],
    time_dependent_log_density: Callable[[jnp.ndarray, float], float],
    num_samples: int,
    ts: jnp.ndarray,
    integration_fn: Callable[
        [Callable[[jnp.ndarray, float], jnp.ndarray], jnp.ndarray, jnp.ndarray],
        jnp.ndarray,
    ] = euler_integrate,
    num_steps: int = 3,
    integration_steps: int = 3,
    eta: float = 0.01,
    rejection_sampling: bool = False,
    shift_fn: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
) -> jnp.ndarray:
    key, subkey = jax.random.split(key)
    initial_samples = generate_samples(
        subkey, v_theta, num_samples, ts, sample_fn, integration_fn, shift_fn
    )

    key, subkey = jax.random.split(key)
    final_samples = time_batched_sample_hamiltonian_monte_carlo(
        subkey,
        time_dependent_log_density,
        initial_samples,
        ts,
        num_steps,
        integration_steps,
        eta,
        rejection_sampling,
        shift_fn,
    )

    return final_samples


@eqx.filter_jit
def divergence_velocity(
    v_theta: Callable[[chex.Array, float], chex.Array],
    x: chex.Array,
    t: float,
    d: float,
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
        return v_theta(x, t, d)

    jacobian = jax.jacfwd(v_x)(x)
    div_v = jnp.trace(jacobian)
    return div_v


def epsilon(
    v_theta: Callable[[chex.Array, float], chex.Array],
    x: chex.Array,
    dt_log_density: float,
    t: float,
    d: float,
    score_fn: Callable[[chex.Array, float], chex.Array],
) -> float:
    """Computes the local error in satisfying the Liouville equation.

    Args:
        v_theta: The velocity field function taking (x, t) and returning velocity vector
        x: The point at which to compute the error
        dt_log_density: Time derivative of log density at (x, t)
        t: Current time
        score_fn: Score function taking (x, t) and returning gradient of log density

    Returns:
        float: Local error in satisfying the Liouville equation
    """
    score = score_fn(x, t)
    div_v = divergence_velocity(v_theta, x, t, d)
    v = v_theta(x, t, d)
    lhs = div_v + jnp.dot(v, score)
    return jnp.nan_to_num(lhs + dt_log_density, posinf=1.0, neginf=-1.0)


batched_epsilon = jax.vmap(epsilon, in_axes=(None, 0, 0, None, None, None))
time_batched_epsilon = eqx.filter_jit(
    jax.vmap(batched_epsilon, in_axes=(None, 0, 0, 0, 0, None))
)


def shortcut(
    v_theta: Callable[[chex.Array, float, float], chex.Array],
    x: chex.Array,
    t: chex.Array,
    d: chex.Array,
    shift_fn: Callable[[chex.Array], chex.Array],
):
    real_d = (jnp.clip(t + 2 * d, 0.0, 1.0) - t) / 2.0

    s_t = v_theta(x, t, real_d)
    x_t = shift_fn(x + s_t * real_d)

    s_td = v_theta(x_t, t + real_d, real_d)
    s_target = jax.lax.stop_gradient(s_t + s_td) / 2.0

    error = (v_theta(x, t, 2 * real_d) - s_target) ** 2

    return error


batched_shortcut = jax.vmap(shortcut, in_axes=(None, 0, None, 0, None))


@eqx.filter_jit
def time_batched_shortcut_loss(
    v_theta: Callable[[chex.Array, float, float], chex.Array],
    xs: chex.Array,
    ts: chex.Array,
    ds: chex.Array,
    shift_fn: Callable[[chex.Array], chex.Array],
) -> chex.Array:
    return jnp.mean(
        jax.vmap(batched_shortcut, in_axes=(None, 0, 0, 0, None))(
            v_theta, xs, ts, ds, shift_fn
        )
    )


def loss_fn(
    v_theta: Callable[[chex.Array, float], chex.Array],
    xs: chex.Array,
    cxs: chex.Array,
    ts: chex.Array,
    ds: chex.Array,
    time_derivative_log_density: Callable[[chex.Array, float], float],
    score_fn: Callable[[chex.Array, float], chex.Array],
    shift_fn: Callable[[chex.Array], chex.Array],
) -> float:
    dt_log_unormalised_density = jax.vmap(
        lambda xs, t: jax.vmap(lambda x: time_derivative_log_density(x, t))(xs),
        in_axes=(0, 0),
    )(xs, ts)
    dt_log_density = dt_log_unormalised_density - jnp.mean(
        dt_log_unormalised_density, axis=-1, keepdims=True
    )

    dss = jnp.diff(ts, append=1.0)
    epsilons = time_batched_epsilon(v_theta, xs, dt_log_density, ts, dss, score_fn)

    short_cut_loss = time_batched_shortcut_loss(v_theta, cxs, ts, ds, shift_fn)

    return jnp.mean(epsilons**2) + short_cut_loss


@eqx.filter_jit
def sample_monotonic_uniform_ordered(
    key: jax.random.PRNGKey, bounds: chex.Array, include_endpoints: bool = True
) -> chex.Array:
    def step(carry, info):
        t_prev = carry
        t_current = info

        return t_current, jnp.array([t_prev, t_current])

    _, ordered_pairs = jax.lax.scan(step, bounds[0], bounds[1:])

    if include_endpoints:
        ordered_pairs = jnp.concatenate(
            [ordered_pairs, jnp.array([[1.0, 1.0]])], axis=0
        )

    samples = jax.random.uniform(
        key, bounds.shape, minval=ordered_pairs[:, 0], maxval=ordered_pairs[:, 1]
    )
    return samples


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
        v_t = jax.vmap(lambda x: v_theta(x, t, jnp.abs(dt)))(x_next)
        x_prev = x_next + dt * v_t  # Since dt < 0, this moves backward

        # Compute divergence
        div_v_t = jax.vmap(lambda x: divergence_velocity(v_theta, x, t, jnp.abs(dt)))(
            x_next
        )
        log_prob_prev = log_prob_next + dt * div_v_t  # Accumulate log_prob

        return (x_prev, log_prob_prev, t), None

    # Initialize carry with final samples and zero log-probabilities
    num_samples = final_samples.shape[0]
    initial_log_probs = jnp.zeros(num_samples)
    carry = (final_samples, initial_log_probs, final_time)

    (xs, log_probs, _), _ = jax.lax.scan(step, carry, ts_rev)

    return xs, log_probs


@eqx.filter_jit
def estimate_kl_div(
    v_theta: Callable[[chex.Array, float, float], chex.Array],
    num_samples: int,
    key: jax.random.PRNGKey,
    ts: chex.Array,
    log_prob_p_fn: Callable[[chex.Array], chex.Array],
    sample_p_fn: Callable[[jax.random.PRNGKey, int], chex.Array],
    base_log_prob: Callable[[chex.Array], chex.Array],
    final_time: float = 1.0,
) -> chex.Array:
    # Generate samples from p(x)
    key, subkey = jax.random.split(key)
    samples_p = sample_p_fn(subkey, (num_samples,))
    log_probs_p = log_prob_p_fn(samples_p)  # Compute log p(x) for these samples

    # Perform reverse-time integration to compute samples and log probabilities under q(x)
    samples_rev, log_probs_q = reverse_time_flow(v_theta, samples_p, final_time, ts)

    # Compute log q(x(T)) = log q(x(0)) + accumulated log_probs
    base_log_probs = base_log_prob(samples_rev)  # Compute log q(x(0))
    log_q_x = base_log_probs + log_probs_q

    # Compute KL divergence: KL(p || q) = E_p[log p(x) - log q(x)]
    kl_divergence = jnp.mean(log_probs_p - log_q_x)

    return kl_divergence


def imq_kernel(x, y, gamma=1.0):
    """IMQ kernel function."""
    return 1.0 / jnp.sqrt(1.0 + gamma * jnp.sum((x - y) ** 2))


def kernelized_stein_discrepancy(X, score_function, kernel_func, gamma=1.0):
    """
    Optimized computation of kernelized Stein discrepancy between a sample X and a target distribution p.
    X: Samples (n, d), where n is the number of samples, d is the dimension.
    score_function: The gradient of the log of the target distribution p(x, t).
    kernel_func: Kernel function (e.g., IMQ kernel).
    """
    n = X.shape[0]

    # Compute pairwise kernel matrix for X, X
    pairwise_kernel = jax.vmap(
        lambda xi: jax.vmap(lambda xj: kernel_func(xi, xj, gamma))(X)
    )(X)

    # Compute score function gradients for each sample
    scores = jax.vmap(score_function, in_axes=(0, None))(X, 1.0)

    # Compute pairwise terms for the Stein discrepancy
    # scores[i] * scores[j] computes the dot product of the scores for each pair
    score_dots = jax.vmap(
        lambda score_xi: jax.vmap(lambda score_xj: jnp.dot(score_xi, score_xj))(scores)
    )(scores)

    # Combine the pairwise kernel matrix and score dot products
    stein_terms = jnp.sum(
        pairwise_kernel * (score_dots + jnp.sum(scores**2, axis=1)[:, None])
    )

    # Return the discrepancy
    return stein_terms / (n**2)


def inverse_power_schedule(T=64, gamma=0.5):
    x_pow = jnp.linspace(0, 1, T)
    t_pow = 1 - x_pow**gamma
    return jnp.flip(t_pow)


def get_optimizer(name: str, learning_rate: float) -> optax.GradientTransformation:
    """Creates optimizer based on name and learning rate.

    Args:
        name: Name of optimizer ('adam', 'adamw', 'sgd', 'rmsprop')
        learning_rate: Learning rate for the optimizer

    Returns:
        optax.GradientTransformation: The configured optimizer
    """
    if name == "adam":
        return optax.adam(learning_rate)
    elif name == "adamw":
        return optax.adamw(learning_rate)
    elif name == "sgd":
        return optax.sgd(learning_rate)
    elif name == "rmsprop":
        return optax.rmsprop(learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


# Main training loop
def train_velocity_field(
    key: jax.random.PRNGKey,
    initial_density: Target,
    target_density: Target,
    v_theta: Callable[[chex.Array, float], chex.Array],
    shift_fn: Callable[[chex.Array], chex.Array] = lambda x: x,
    N: int = 512,
    B: int = 256,
    C: int = 64,
    L: float = 10.0,
    num_epochs: int = 200,
    num_steps: int = 100,
    learning_rate: float = 1e-03,
    T: int = 32,
    gradient_norm: float = 1.0,
    mcmc_type: str = "hmc",
    num_mcmc_steps: int = 5,
    num_mcmc_integration_steps: int = 3,
    eta: float = 0.01,
    schedule: str = "inverse_power",
    continuous_schedule: bool = False,
    integrator: str = "euler",
    optimizer: str = "adamw",
    with_rejection_sampling: bool = False,
    eval_steps: List[int] = [4, 8, 16, 32],
    offline: bool = False,
    d_distribution: str = "uniform",
    target: str = "gmm",
    eval_every: int = 20,
    **kwargs: Any,
) -> Any:
    path_distribution = AnnealedDistribution(
        initial_distribution=initial_density,
        target_distribution=target_density,
        dim=initial_density.dim,
    )

    # Set up optimizer
    gradient_clipping = optax.clip_by_global_norm(gradient_norm)
    base_optimizer = get_optimizer(optimizer, learning_rate)
    optimizer = optax.chain(gradient_clipping, base_optimizer)
    opt_state = optimizer.init(eqx.filter(v_theta, eqx.is_inexact_array))

    integrator = euler_integrate
    # Generate time steps
    key, subkey = jax.random.split(key)
    if schedule == "linear":
        ts = jnp.linspace(0, 1, T)
    elif schedule == "inverse_power":
        ts = inverse_power_schedule(T, gamma=0.5)
    else:
        ts = jnp.linspace(0, 1, T)

    sampled_ts = ts
    if d_distribution == "log":
        d_dis = 1.0 / jnp.array(
            [2**e for e in range(int(jnp.floor(jnp.log2(128))) + 1)]
        )

    @eqx.filter_jit
    def step(v_theta, opt_state, xs, cxs, ts, ds):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(
            v_theta,
            xs,
            cxs,
            ts,
            ds,
            time_derivative_log_density=path_distribution.time_derivative,
            score_fn=path_distribution.score_fn,
            shift_fn=shift_fn,
        )
        updates, opt_state = optimizer.update(grads, opt_state, v_theta)
        v_theta = eqx.apply_updates(v_theta, updates)
        return v_theta, opt_state, loss

    for epoch in range(num_epochs):
        key, subkey = jax.random.split(key)
        if mcmc_type == "hmc":
            samples = generate_samples_with_hmc_correction(
                key=subkey,
                v_theta=v_theta,
                sample_fn=path_distribution.sample_initial,
                time_dependent_log_density=path_distribution.time_dependent_log_prob,
                num_samples=N,
                ts=sampled_ts,
                integration_fn=integrator,
                num_steps=num_mcmc_steps,
                integration_steps=num_mcmc_integration_steps,
                eta=eta,
                rejection_sampling=with_rejection_sampling,
                shift_fn=shift_fn,
            )
        else:
            samples = generate_samples(
                subkey,
                v_theta,
                N,
                sampled_ts,
                path_distribution.sample_initial,
                integrator,
                shift_fn,
            )

        epoch_loss = 0.0

        key, subkey = jax.random.split(key)
        if d_distribution == "uniform":
            sampled_ds = jax.random.uniform(subkey, (T, C), minval=0.0, maxval=1.0)
        else:
            sampled_ds = jax.random.choice(subkey, d_dis, (T, C), replace=True)

        for s in range(num_steps):
            key, subkey = jax.random.split(key)
            samps, samps_cxs = jnp.split(
                jax.random.choice(subkey, samples, (B + C,), replace=False, axis=1),
                [B],
                axis=1,
            )

            v_theta, opt_state, loss = step(
                v_theta, opt_state, samps, samps_cxs, sampled_ts, sampled_ds
            )
            epoch_loss += loss

            if s % 20 == 0:
                if not offline:
                    wandb.log({"loss": loss})
                else:
                    print(f"Epoch {epoch}, Step {s}, Loss: {loss}")

        avg_loss = epoch_loss / num_steps
        if not offline:
            wandb.log({"epoch": epoch, "average_loss": avg_loss})
        else:
            print(f"Epoch {epoch}, Average Loss: {avg_loss}")

        if epoch % eval_every == 0:
            tss = [jnp.linspace(0, 1, eval_step) for eval_step in eval_steps]
            key, subkey = jax.random.split(key)
            val_samples = generate_samples_with_different_ts(
                subkey,
                v_theta,
                N,
                tss,
                path_distribution.sample_initial,
                integrator,
                shift_fn,
            )

            for i, es in enumerate(eval_steps):
                if target == "gmm":
                    fig = target_density.visualise(val_samples[i][-1])

                    key, subkey = jax.random.split(key)
                    kl_div = estimate_kl_div(
                        v_theta,
                        N,
                        key,
                        tss[i],
                        target_density.log_prob,
                        target_density.sample,
                        initial_density.log_prob,
                    )

                    if not offline:
                        wandb.log(
                            {
                                f"validation_samples_{es}_step": wandb.Image(fig),
                                f"kl_div_{es}_step": kl_div,
                            }
                        )
                    else:
                        plt.show()

                    plt.close(fig)

                elif target == "mw32":
                    key, subkey = jax.random.split(key)
                    fig = target_density.visualise(
                        jax.random.choice(
                            subkey, val_samples[i][-1], (100,), replace=False
                        )
                    )

                    key, subkey = jax.random.split(key)
                    ksd_div = kernelized_stein_discrepancy(
                        val_samples[i][-1][:1280],
                        path_distribution.score_fn,
                        imq_kernel,
                    )

                    if not offline:
                        wandb.log(
                            {
                                f"validation_samples_{es}_step": wandb.Image(fig),
                                f"ksd_div_{es}_step": ksd_div,
                            }
                        )
                    else:
                        plt.show()

                    plt.close(fig)
                elif target == "dw4" or target == "lj13":
                    key, subkey = jax.random.split(key)
                    fig = target_density.visualise(val_samples[i][-1][:1024])
                    # key, subkey = jax.random.split(key)
                    # ksd_div = kernelized_stein_discrepancy(
                    #     val_samples[i][-1][:1280],
                    #     path_distribution.score_fn,
                    #     imq_kernel,
                    # )

                    if not offline:
                        wandb.log(
                            {
                                f"validation_samples_{es}_step": wandb.Image(fig),
                                # f"ksd_div_{es}_step": ksd_div,
                            }
                        )
                    else:
                        plt.show()

        # Resample ts according to gamma range
        if continuous_schedule:
            key, subkey = jax.random.split(key)
            sampled_ts = sample_monotonic_uniform_ordered(subkey, ts, True)

    # Save trained model to wandb
    if not offline:
        eqx.tree_serialise_leaves("v_theta.eqx", v_theta)
        artifact = wandb.Artifact(
            name=f"velocity_field_model_{wandb.run.id}", type="model"
        )
        artifact.add_file(local_path="v_theta.eqx", name="model")
        artifact.save()

        wandb.log_artifact(artifact)
        wandb.finish()
    return v_theta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-epochs", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--shortcut-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=5120)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument(
        "--box-size",
        "-L",
        type=float,
        default=10.0,
        help="Size of the box",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--num-timesteps", type=int, default=128)
    parser.add_argument("--mcmc-steps", type=int, default=5)
    parser.add_argument("--mcmc-integration-steps", type=int, default=3)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--initial-sigma", type=float, default=20.0)
    parser.add_argument("--eval-steps", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--network", type=str, default="mlp", choices=["mlp", "pdn"])
    parser.add_argument(
        "--target", type=str, default="gmm", choices=["gmm", "mw32", "dw4", "lj13"]
    )
    parser.add_argument(
        "--d-distribution", type=str, choices=["uniform", "log"], default="uniform"
    )
    parser.add_argument(
        "--schedule",
        type=str,
        choices=["linear", "inverse_power"],
        default="linear",
    )
    parser.add_argument(
        "--integrator", type=str, choices=["euler", "rk4"], default="euler"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "adamw", "sgd", "rmsprop"],
        default="adamw",
    )
    parser.add_argument("--continuous-schedule", action="store_true")
    parser.add_argument("--with-rejection-sampling", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--eval-every", type=int, default=20)
    args = parser.parse_args()

    if args.debug:
        jax.config.update("jax_debug_nans", True)
        jax.config.update("jax_debug_infs", True)

    # Set random seed
    key = jax.random.PRNGKey(args.seed)
    shift_fn = lambda x: x

    # Set up distributions
    if args.target == "gmm":
        input_dim = 2
        key, subkey = jax.random.split(key)
        # Initialize distributions
        initial_density = MultivariateGaussian(dim=input_dim, sigma=args.initial_sigma)
        target_density = GMM(subkey, dim=input_dim)
    elif args.target == "mw32":
        input_dim = 32
        key, subkey = jax.random.split(key)
        initial_density = MultivariateGaussian(dim=input_dim, sigma=args.initial_sigma)
        target_density = ManyWellEnergy(dim=input_dim)
    elif args.target == "dw4":
        input_dim = 8
        key, subkey = jax.random.split(key)
        # initial_density = MultivariateGaussian(dim=input_dim, sigma=args.initial_sigma)
        initial_density = WrappedMultivariateGaussian(
            dim=input_dim, sigma=args.initial_sigma, box_size=args.box_size, num_shift=3
        )
        target_density = MultiDoubleWellEnergy(
            dim=input_dim,
            n_particles=4,
            data_path_test="test_split_DW4.npy",
            data_path_val="val_split_DW4.npy",
            key=subkey,
            L=args.box_size,
        )
        shift_fn = lambda x: jnp.mod(x, args.box_size)

    elif args.target == "lj13":
        input_dim = 39
        key, subkey = jax.random.split(key)
        initial_density = MultivariateGaussian(dim=input_dim, sigma=args.initial_sigma)
        target_density = LennardJonesEnergy(
            dim=input_dim,
            n_particles=13,
            data_path="test_split_LJ13-1000.npy",
            data_path_train="train_split_LJ13-1000.npy",
            data_path_val="val_split_LJ13-1000.npy",
            key=subkey,
        )

    # Initialize velocity field
    key, model_key = jax.random.split(key)
    if args.network == "mlp":
        v_theta = ShortcutTimeVelocityField(
            model_key, input_dim=input_dim, hidden_dim=args.hidden_dim, depth=args.depth
        )
    elif args.network == "pdn":
        v_theta = ShortcutTimeVelocityFieldWithPairwiseFeature(
            model_key,
            n_particles=4 if args.target == "dw4" else 13,
            n_spatial_dim=2 if args.target == "dw4" else 3,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
        )

    if not args.offline:
        # Handle logging hyperparameters
        wandb.init(
            project="shortcut_continuous_liouville",
            config={
                "input_dim": initial_density.dim,
                "T": args.num_timesteps,
                "N": args.num_samples,
                "C": args.shortcut_size,
                "L": args.box_size,
                "num_epochs": args.num_epochs,
                "num_steps": args.num_steps,
                "num_shifts": 3,
                "learning_rate": args.learning_rate,
                "gradient_norm": 1.0,
                "hidden_dim": v_theta.mlp.width_size,
                "depth": v_theta.mlp.depth,
                "mcmc_type": "hmc",
                "num_mcmc_steps": args.mcmc_steps,
                "num_mcmc_integration_steps": args.mcmc_integration_steps,
                "eta": args.eta,
                "schedule": args.schedule,
                "optimizer": args.optimizer,
                "integrator": args.integrator,
                "with_rejection_sampling": args.with_rejection_sampling,
                "continuous_schedule": args.continuous_schedule,
                "d_distribution": args.d_distribution,
                "target": args.target,
                "network": args.network,
            },
            reinit=True,
            tags=[args.target, args.network],
        )

    # Train model
    v_theta = train_velocity_field(
        key=key,
        initial_density=initial_density,
        target_density=target_density,
        v_theta=v_theta,
        shift_fn=shift_fn,
        N=args.num_samples,
        B=args.batch_size,
        L=args.box_size,
        num_epochs=args.num_epochs,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        T=args.num_timesteps,
        num_mcmc_steps=args.mcmc_steps,
        num_mcmc_integration_steps=args.mcmc_integration_steps,
        mcmc_type="hmc",
        eta=args.eta,
        schedule=args.schedule,
        integrator=args.integrator,
        optimizer=args.optimizer,
        with_rejection_sampling=args.with_rejection_sampling,
        continuous_schedule=args.continuous_schedule,
        eval_steps=args.eval_steps,
        offline=args.offline,
        d_distribution=args.d_distribution,
        target=args.target,
        eval_every=args.eval_every,
        network=args.network,
    )


if __name__ == "__main__":
    main()