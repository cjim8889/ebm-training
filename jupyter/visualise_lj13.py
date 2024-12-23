import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import pandas as pd
import distrax
import optax
import blackjax
import chex

import abc
from typing import Optional, Callable, Union, Tuple, Dict, Any, Sequence, List
from itertools import product


from jax.scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from IPython.display import HTML

plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.figsize"] = [6.0, 4.0]

jax.config.update("jax_platform_name", "cpu")

key = jax.random.PRNGKey(12391)


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


def plot_contours_2D_with_time(
    log_prob_func: Callable[[np.ndarray, float], np.ndarray],
    t: float,
    ax: Optional[plt.Axes] = None,
    bound: float = 3,
    levels: int = 20,
):
    """Plot the contours of a 2D log prob function at time t."""
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.clear()  # Clear previous contours
    n_points = 200
    x_points_dim1 = np.linspace(-bound, bound, n_points)
    x_points_dim2 = np.linspace(-bound, bound, n_points)
    x_points = np.array(list(product(x_points_dim1, x_points_dim2)))
    log_probs = log_prob_func(x_points, t)
    log_probs = jnp.clip(log_probs, a_min=-1000, a_max=None)
    x1 = x_points[:, 0].reshape(n_points, n_points)
    x2 = x_points[:, 1].reshape(n_points, n_points)
    z = log_probs.reshape(n_points, n_points)
    contour = ax.contourf(x1, x2, z, levels=levels, cmap="viridis")
    ax.set_title(f"Time t = {t:.2f}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    return contour


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

    def score(self, value: chex.Array) -> chex.Array:
        """Gradient of log_prob w.r.t. value."""
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
    ) -> plt.Figure:
        """Visualise samples from the model."""


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


# Utility function
def remove_mean(x, n_particles, n_spatial_dim):
    x = x.reshape(n_particles, n_spatial_dim)
    x = x - jnp.mean(x, axis=0, keepdims=True)
    return x.reshape(n_particles * n_spatial_dim)


def remove_mean_decorator(step_fn, n_particles, n_spatial_dim):
    """Wraps HMC step to remove mean at each iteration"""

    def wrapped_step(key, state):
        state, info = step_fn(key, state)
        # Remove mean from position
        state = state._replace(
            position=remove_mean(
                state.position, n_particles=n_particles, n_spatial_dim=n_spatial_dim
            )
        )
        return state, info

    return wrapped_step


def estimate_mass_matrix_from_samples(samples):
    """
    Estimate mass matrix from samples using sample covariance.

    Args:
        samples: Array of shape (N, dim) containing N samples

    Returns:
        mass_matrix: Array of shape (dim, dim) containing estimated mass matrix
    """
    # Calculate sample covariance matrix
    cov = jnp.cov(samples, rowvar=False)

    # Add small diagonal offset for numerical stability
    cov = cov + 1e-6 * jnp.eye(cov.shape[0])

    # Return inverse covariance as mass matrix
    return jnp.linalg.inv(cov)


class TimeDependentLennardJonesEnergyButler(Target):
    TIME_DEPENDENT = True

    def __init__(
        self,
        dim: int,
        n_particles: int,
        alpha: float = 0.5,
        sigma: float = 1.0,
        epsilon_val: float = 1.0,
        min_dr: float = 1e-4,
        n: float = 1,
        m: float = 1,
        c: float = 0.5,
        log_prob_clip: float = None,
        score_norm: float = 1.0,
        data_path_val: str = "../data/val_split_LJ13-1000.npy",
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

        self.alpha = alpha
        self.sigma = sigma
        self.epsilon_val = epsilon_val
        self.min_dr = min_dr
        self.n = n
        self.m = m
        self.c = c

        self.log_prob_clip = log_prob_clip
        self.score_norm = score_norm

        self.data_path_val = data_path_val

        self._val_set = self.setup_val_set()

    def setup_val_set(self):
        data = np.load(self.data_path_val, allow_pickle=True)
        return data

    def find_min_energy_position(self, initial_position, tol=1e-6):
        result = minimize(
            lambda x: self.compute_time_dependent_lj_energy(x, 1.0),
            initial_position,
            method="BFGS",
            tol=tol,
        )
        return result.x

    def initialize_position(self, key: jax.random.PRNGKey):
        # Start with a random normal position
        initial_position = jax.random.normal(key, (self.dim,))
        # Optionally, scale positions to avoid overlaps
        initial_position = initial_position * self.sigma * 1.1
        # Perform energy minimization
        optimized_position = self.find_min_energy_position(initial_position)

        # Center the initial position
        optimized_position = remove_mean(
            optimized_position, self.n_particles, self.n_spatial_dim
        )

        return optimized_position

    def _generate_validation_set(
        self,
        key: jax.random.PRNGKey,
        initial_position,
        inverse_mass_matrix=None,
        step_size=0.01,
        num_samples=1000,
        num_warmup=1000,
        thinning=10,
        num_integration_steps=10,
        divergence_threshold=10000,
    ):
        """Generate validation set using NUTS sampler"""
        if inverse_mass_matrix is None:
            inverse_mass_matrix = jnp.ones(self.dim)
        # Setup NUTS sampler
        nuts = blackjax.hmc(
            self.log_prob,
            inverse_mass_matrix=inverse_mass_matrix,
            divergence_threshold=divergence_threshold,
            step_size=step_size,
            num_integration_steps=num_integration_steps,
        )
        # nuts = nuts._replace(
        #     step=remove_mean_decorator(nuts.step, self.n_particles, self.n_spatial_dim)
        # )

        initial_state = nuts.init(initial_position)

        @jax.jit
        def one_step(carry, key):
            state, info = carry
            state, info = nuts.step(key, state)
            return (state, info), state.position

        # Generate samples
        keys = jax.random.split(key, num_samples + num_warmup)

        initial_state, initial_info = nuts.step(keys[0], initial_state)
        (final_state, final_info), samples = jax.lax.scan(
            one_step, (initial_state, initial_info), keys
        )

        # Apply thinning and discard warmup
        samples = samples[num_warmup:]
        samples = samples[::thinning]

        # Apply shift function and center of mass correction
        # samples = self.shift_fn(samples)
        samples = jax.vmap(
            lambda samples: remove_mean(samples, self.n_particles, self.n_spatial_dim)
        )(samples)

        return samples, final_info

    def multichain_sampling(
        self,
        key: jax.random.PRNGKey,
        inverse_mass_matrix=None,
        num_chains=10,
        step_size=0.01,
        num_samples=1000,
        num_warmup=1000,
        thinning=10,
        num_integration_steps=10,
        divergence_threshold=10000,
    ):
        """Generate multiple chains using NUTS sampler"""

        keys = jax.random.split(key, num_chains)
        initial_positions = jax.vmap(self.initialize_position)(keys)

        if inverse_mass_matrix is None:
            inverse_mass_matrix = jnp.ones(self.dim)

        keys = jax.random.split(keys[0], num_chains)
        samples = jax.vmap(
            lambda key, initial_position: self._generate_validation_set(
                key,
                initial_position,
                inverse_mass_matrix,
                step_size,
                num_samples,
                num_warmup,
                thinning,
                num_integration_steps,
                divergence_threshold,
            )[0]
        )(keys, initial_positions)

        return samples.reshape(-1, self.dim)

    def soft_core_lennard_jones_potential(
        self,
        pairwise_dr: jnp.ndarray,
        t: float,
    ) -> jnp.ndarray:
        """
        Compute the time-dependent soft-core Lennard-Jones potential.

        Args:
            pairwise_dr (jnp.ndarray): Pairwise distances of shape [n_pairs].
            t (float): Time parameter, influencing the strength of the potential (lambda).

        Returns:
            jnp.ndarray: Time-dependent soft-core Lennard-Jones potential energy of shape [].
        """

        inv_r6 = (self.sigma / (pairwise_dr + self.alpha * (1 - t) ** self.m)) ** 6
        inv_r12 = inv_r6**2

        lj_energy = self.epsilon_val * t**self.n * (inv_r12 - 2 * inv_r6)
        total_lj_energy = jnp.sum(lj_energy, axis=-1)

        return total_lj_energy

    def soft_core_lennard_jones_potential_1(
        self,
        pairwise_dr: jnp.ndarray,
        t: float,
    ) -> jnp.ndarray:
        """
        Compute the time-dependent soft-core Lennard-Jones potential.

        Args:
            pairwise_dr (jnp.ndarray): Pairwise distances of shape [n_pairs].
            t (float): Time parameter, influencing the strength of the potential (lambda).

        Returns:
            jnp.ndarray: Time-dependent soft-core Lennard-Jones potential energy of shape [].
        """

        # U(λ, r) = 0.5 * epsilon * λ^n * [ (α_LJ * (1 - λ)^m + (r/sigma)^6)^-2 - (α_LJ * (1 - λ)^m + (r/sigma)^6)^-1 ]
        lambda_ = t

        inv_r6 = (pairwise_dr / self.sigma) ** 6
        soft_core_term = self.alpha * (1 - lambda_) ** self.m + inv_r6
        lj_energy = (
            0.5
            * self.epsilon_val
            * lambda_**self.n
            * (soft_core_term**-1 - soft_core_term**-2)
        )

        # Sum over all pairs to get total energy per sample
        total_lj_energy = jnp.sum(lj_energy, axis=-1)

        return total_lj_energy

    def compute_distances(self, x, epsilon=1e-8):
        x = x.reshape(self.n_particles, self.n_spatial_dim)

        # Get indices of upper triangular pairs
        i, j = jnp.triu_indices(self.n_particles, k=1)

        # Calculate displacements between pairs
        dx = x[i] - x[j]

        # Compute distances
        distances = optax.safe_norm(dx, axis=-1, min_norm=self.min_dr)

        return distances

    def harmonic_potential(self, x):
        """
        Compute the harmonic potential energy.

        E^osc(x) = 1/2 * Σ ||xi - x_COM||^2
        """
        x = x.reshape(self.n_particles, self.n_spatial_dim)
        x_com = jnp.mean(x, axis=0)
        distances_to_com = optax.safe_norm(
            x - x_com,
            axis=-1,
            min_norm=0.0,
        )

        return 0.5 * jnp.sum(distances_to_com**2)

    def compute_time_dependent_lj_energy(
        self,
        x: jnp.ndarray,
        t: float,
    ) -> jnp.ndarray:
        """
        Compute the total time-dependent soft-core Lennard-Jones energy for a batch of samples.

        Args:
            x (jnp.ndarray): Input array of shape [n_particles * n_spatial_dim].
            t (float): Time parameter.

        Returns:
            jnp.ndarray: Total time-dependent Lennard-Jones energy.
        """
        pairwise_dr = self.compute_distances(
            x.reshape(self.n_particles, self.n_spatial_dim)
        )
        lj_energy = self.soft_core_lennard_jones_potential(pairwise_dr, t)
        if self.log_prob_clip is not None:
            lj_energy = jnp.clip(lj_energy, -self.log_prob_clip, self.log_prob_clip)

        harmonic_energy = self.harmonic_potential(x)

        return lj_energy + self.c * harmonic_energy

    def log_prob(self, x: chex.Array) -> chex.Array:
        return -self.compute_time_dependent_lj_energy(x, 1.0)

    def time_dependent_log_prob(self, x: chex.Array, t: float) -> chex.Array:
        p_t = -self.compute_time_dependent_lj_energy(x, t)
        return p_t

    def score(self, x: chex.Array, t: float) -> chex.Array:
        sc = jax.grad(self.time_dependent_log_prob, argnums=0)(x, t)
        norm = optax.safe_norm(sc, axis=-1, min_norm=1e-6)
        scale = jnp.clip(self.score_norm / (norm + 1e-6), a_min=0.0, a_max=1.0)

        return sc * scale

    def sample(
        self, key: jax.random.PRNGKey, sample_shape: chex.Shape = ()
    ) -> chex.Array:
        raise NotImplementedError(
            "Sampling is not implemented for TimeDependentLennardJonesEnergy"
        )

    def interatomic_dist(self, x):
        x = x.reshape(-1, self.n_particles, self.n_spatial_dim)
        distances = jax.vmap(lambda x: self.compute_distances(x))(x)

        return distances

    def batched_log_prob(self, xs, t):
        return jax.vmap(self.time_dependent_log_prob, in_axes=(0, None))(xs, t)

    def visualise(self, samples: chex.Array) -> plt.Figure:
        # Fill samples nan values with zeros
        samples = jnp.nan_to_num(samples, nan=0.0, posinf=100.0, neginf=-100.0)

        # Since we don't have a test set, we will just visualize the samples
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        dist_samples = self.interatomic_dist(samples)
        dist_val = self.interatomic_dist(self._val_set)

        axs[0].hist(
            dist_samples.flatten(),
            bins=100,
            alpha=0.5,
            density=True,
            histtype="step",
            linewidth=4,
        )

        axs[0].hist(
            dist_val.flatten(),
            bins=100,
            alpha=0.5,
            density=True,
            histtype="step",
            linewidth=4,
        )
        axs[0].set_xlabel("Interatomic distance")
        axs[0].legend(["Generated data", "Ground truth samples"])

        energy_samples = -self.batched_log_prob(samples, 1.0)
        energy_samples_val = -self.batched_log_prob(self._val_set, 1.0)

        # Clip energy values for visualization
        energy_samples = jnp.nan_to_num(
            energy_samples, nan=0.0, posinf=1000.0, neginf=-1000.0
        )
        energy_samples_val = jnp.nan_to_num(
            energy_samples_val, nan=0.0, posinf=1000.0, neginf=-1000.0
        )

        # Determine histogram range from cleaned data
        min_energy = jnp.min(jnp.concatenate([energy_samples, energy_samples_val]))
        max_energy = jnp.max(jnp.concatenate([energy_samples, energy_samples_val]))

        # Add padding to range
        energy_range = (
            min_energy - 0.1 * abs(min_energy),
            max_energy + 0.1 * abs(max_energy),
        )

        axs[1].hist(
            energy_samples,
            bins=100,
            density=True,
            alpha=0.4,
            range=energy_range,
            color="r",
            histtype="step",
            linewidth=4,
            label="Generated data",
        )

        axs[1].hist(
            energy_samples_val,
            bins=100,
            density=True,
            alpha=0.4,
            range=energy_range,
            color="g",
            histtype="step",
            linewidth=4,
            label="Ground truth samples",
        )
        axs[1].set_xlabel("Energy")
        axs[1].legend()

        fig.canvas.draw()
        return fig


if __name__ == "__main__":
    butler = TimeDependentLennardJonesEnergyButler(
        dim=39,
        n_particles=13,
        alpha=0,
        sigma=1.0,
        epsilon_val=1.0,
        min_dr=1e-5,
        n=1,
        m=1,
        log_prob_clip=100.0,
        data_path_val="data/test_split_LJ13-1000.npy",
    )

    key, subkey = jax.random.split(key)
    samples = butler.multichain_sampling(
        subkey,
        inverse_mass_matrix=None,
        num_chains=20,
        step_size=0.1,
        num_samples=210000,
        num_warmup=200000,
        thinning=1,
        num_integration_steps=5,
    )

    butler.visualise(samples[::10])
