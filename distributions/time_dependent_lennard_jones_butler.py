import chex
import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize
import matplotlib.pyplot as plt
import optax

from utils.distributions import compute_distances

from .base import Target


class TimeDependentLennardJonesEnergyButler(Target):
    TIME_DEPENDENT = True

    def __init__(
        self,
        dim: int,
        n_particles: int,
        alpha: float = 0.5,
        sigma: float = 1.0,
        sigma_cutoff: float = 2.5,
        epsilon_val: float = 1.0,
        min_dr: float = 1e-4,
        n: float = 1,
        m: float = 1,
        c: float = 0.5,
        log_prob_clip: float = None,
        score_norm: float = None,
        include_harmonic: bool = False,
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
        self.cutoff = sigma_cutoff * sigma

        self.epsilon_val = epsilon_val
        self.min_dr = min_dr
        self.n = n
        self.m = m
        self.c = c

        self.log_prob_clip = log_prob_clip
        self.score_norm = score_norm
        self.include_harmonic = include_harmonic

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
        initial_position = jax.random.uniform(key, (self.dim,), minval=-3.0, maxval=3.0)
        # Optionally, scale positions to avoid overlaps
        initial_position = initial_position * self.sigma * 1.2
        # Perform energy minimization
        optimized_position = self.find_min_energy_position(initial_position)

        return optimized_position

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

        inv_r6 = (pairwise_dr / self.sigma) ** 6
        soft_core_term = self.alpha * (1 - t) ** self.m + inv_r6
        lj_energy = (
            self.epsilon_val * t**self.n * (soft_core_term**-2 - 2 * soft_core_term**-1)
        )

        # Apply cutoff: set energy to zero for distances > cutoff
        lj_energy = jnp.where(pairwise_dr <= self.cutoff, lj_energy, 0.0)
        # Sum over all pairs to get total energy per sample
        total_lj_energy = jnp.sum(lj_energy, axis=-1)

        return total_lj_energy

    def harmonic_potential(self, x):
        """
        Compute the harmonic potential energy.

        E^osc(x) = 1/2 * Σ ||xi - x_COM||^2
        """
        x = x.reshape(self.n_particles, self.n_spatial_dim)
        x_com = jnp.mean(x, axis=0)
        distances_to_com = optax.safe_norm(
            x - x_com,
            ord=2,
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
        pairwise_dr = compute_distances(
            x,
            n_particles=self.n_particles,
            n_dimensions=self.n_spatial_dim,
            min_dr=self.min_dr,
        )
        lj_energy = self.soft_core_lennard_jones_potential(pairwise_dr, t)

        if self.include_harmonic:
            harmonic_energy = self.harmonic_potential(x)

            return lj_energy + self.c * harmonic_energy
        else:
            return lj_energy

    def log_prob(self, x: chex.Array) -> chex.Array:
        return self.time_dependent_log_prob(x, 1.0)

    def time_dependent_log_prob(self, x: chex.Array, t: float) -> chex.Array:
        p_t = -self.compute_time_dependent_lj_energy(x, t)

        if self.log_prob_clip is not None:
            p_t = jnp.clip(p_t, a_min=-self.log_prob_clip, a_max=self.log_prob_clip)

        return p_t

    def score(self, x: chex.Array, t: float) -> chex.Array:
        sc = jax.grad(self.time_dependent_log_prob, argnums=0)(x, t)

        if self.score_norm is not None:
            norm = optax.safe_norm(sc, axis=-1, min_norm=1e-6)
            scale = jnp.clip(self.score_norm / (norm + 1e-6), a_min=0.0, a_max=1.0)

            return sc * scale
        else:
            return sc

    def sample(
        self, key: jax.random.PRNGKey, sample_shape: chex.Shape = ()
    ) -> chex.Array:
        raise NotImplementedError(
            "Sampling is not implemented for TimeDependentLennardJonesEnergy"
        )

    def interatomic_dist(self, x):
        x = x.reshape(-1, self.n_particles, self.n_spatial_dim)
        distances = jax.vmap(
            lambda x: compute_distances(x, self.n_particles, self.n_spatial_dim)
        )(x)

        return distances

    def batched_log_prob(self, xs, t):
        return jax.vmap(self.time_dependent_log_prob, in_axes=(0, None))(xs, t)

    def visualise_with_time(self, samples: chex.Array, t: float) -> plt.Figure:
        # Fill samples nan values with zeros
        samples = jnp.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)

        # Since we don't have a test set, we will just visualize the samples
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        dist_samples = self.interatomic_dist(samples)

        axs[0].hist(
            dist_samples.flatten(),
            bins=100,
            alpha=0.5,
            density=True,
            histtype="step",
            linewidth=4,
        )
        axs[0].set_xlabel("Interatomic distance")
        axs[0].legend(["Generated data at t={}".format(t)])

        energy_samples = -self.batched_log_prob(samples, t)
        # Clip energy values for visualization
        energy_samples = jnp.nan_to_num(
            energy_samples, nan=0.0, posinf=100.0, neginf=-100.0
        )

        # Determine histogram range from cleaned data
        min_energy = jnp.min(energy_samples)
        max_energy = jnp.max(energy_samples)

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
            label="Generated data at t={}".format(t),
        )
        axs[1].set_xlabel("Energy")
        axs[1].legend()

        fig.canvas.draw()
        return fig

    def visualise(self, samples: chex.Array) -> plt.Figure:
        return self.visualise_with_time(samples, 1.0)


class TimeDependentLennardJonesEnergyButlerWithTemperatureTempered(
    TimeDependentLennardJonesEnergyButler
):
    def __init__(
        self,
        dim,
        n_particles,
        alpha=0.5,
        sigma=1,
        sigma_cutoff=2.5,
        epsilon_val=1,
        min_dr=0.0001,
        n=1,
        m=1,
        c=0.5,
        initial_temperature=250.0,
        annealing_order=1.0,
        log_prob_clip=None,
        score_norm=None,
        include_harmonic=False,
    ):
        super().__init__(
            dim,
            n_particles,
            alpha,
            sigma,
            sigma_cutoff,
            epsilon_val,
            min_dr,
            n,
            m,
            c,
            log_prob_clip,
            score_norm,
            include_harmonic,
        )

        self.initial_temperature = initial_temperature
        self.annealing_order = annealing_order

    def time_dependent_log_prob(self, x: chex.Array, t: float) -> chex.Array:
        p_t = -self.compute_time_dependent_lj_energy(x, t)
        p_t = p_t / (1 + self.initial_temperature * (1 - t) ** self.annealing_order)

        if self.log_prob_clip is not None:
            p_t = jnp.clip(p_t, a_min=-self.log_prob_clip, a_max=self.log_prob_clip)

        return p_t