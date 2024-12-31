from typing import Callable

import blackjax
import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jax.scipy.optimize import minimize

from utils.distributions import remove_mean, remove_mean_decorator, compute_distances

from .base import Target


class SoftCoreLennardJonesEnergy(Target):
    TIME_DEPENDENT = False

    def __init__(
        self,
        dim: int,
        n_particles: int,
        sigma: float = 1.0,
        epsilon_val: float = 1.0,
        alpha: float = 0.1,
        shift_fn: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
        min_dr: float = 1e-3,
        c: float = 1.0,
        include_harmonic: bool = False,
        **kwargs,
    ):
        """
        Compute the Soft-core Lennard-Jones energy for a batch of samples.

        Args:
            x (jnp.ndarray): Input array of shape [n_particles * n_spatial_dim].
            sigma (float): Finite distance at which the inter-particle potential is zero.
            epsilon_val (float): Depth of the potential well.
            alpha (float): Smoothing parameter for the soft-core potential.
        Returns:
            jnp.ndarray: Total Lennard-Jones energy for each sample, shape [batch_size].
        """

        super().__init__(
            dim=dim,
            log_Z=None,
            can_sample=False,
            n_plots=1,
            n_model_samples_eval=1000,
            n_target_samples_eval=None,
            **kwargs,
        )
        self.n_particles = n_particles
        self.n_spatial_dim = dim // n_particles
        self.sigma = sigma
        self.epsilon_val = epsilon_val
        self.alpha = alpha
        self.c = c
        self.include_harmonic = include_harmonic

        self.min_dr = min_dr
        self.shift_fn = shift_fn

    def filter_samples(self, samples, max_energy=None, max_interatomic_dist=None):
        """Filter samples based on energy and interatomic distance thresholds.

        Args:
            samples (jnp.ndarray): Array of shape (n_samples, n_particles * n_spatial_dim)
                containing the samples to filter
            max_energy (float, optional): Maximum allowed energy. Samples with higher
                energy will be filtered out. Defaults to None (no energy filtering).
            max_interatomic_dist (float, optional): Maximum allowed interatomic distance.
                Samples with any pairwise distance greater than this will be filtered out.
                Defaults to None (no distance filtering).

        Returns:
            jnp.ndarray: Filtered samples array
            jnp.ndarray: Boolean mask indicating which samples passed the filters
        """
        n_samples = len(samples)
        mask = jnp.ones(n_samples, dtype=bool)

        # Filter by energy if threshold provided
        if max_energy is not None:
            energies = jax.vmap(self.compute_soft_core_lj_energy)(samples)
            energy_mask = energies <= max_energy
            mask = mask & energy_mask

        # Filter by interatomic distances if threshold provided
        if max_interatomic_dist is not None:
            # Compute pairwise distances for all samples
            distances = jax.vmap(self.compute_distances)(samples)
            # Create mask for samples where all distances are below threshold
            distance_mask = jnp.all(distances <= max_interatomic_dist, axis=1)
            mask = mask & distance_mask

        return samples[mask], mask

    def find_min_energy_position(self, initial_position, tol=1e-6):
        result = minimize(
            self.compute_soft_core_lj_energy, initial_position, method="BFGS", tol=tol
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
        optimized_position = self.shift_fn(optimized_position)
        optimized_position = remove_mean(
            optimized_position, self.n_particles, self.n_spatial_dim
        )

        return optimized_position

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

    @eqx.filter_jit
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
        nuts = nuts._replace(
            step=remove_mean_decorator(nuts.step, self.n_particles, self.n_spatial_dim)
        )

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
        samples = self.shift_fn(samples)
        samples = jax.vmap(
            lambda samples: remove_mean(samples, self.n_particles, self.n_spatial_dim)
        )(samples)

        return samples, final_info

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

    def soft_core_lennard_jones_potential(
        self, pairwise_dr: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute the Soft-core Lennard-Jones potential.
        Args:
            pairwise_dr (jnp.ndarray): Pairwise distances of shape [n_pairs].
        Returns:
            jnp.ndarray: Soft-core Lennard-Jones potential energy of shape [].
        """
        # Compute (sigma / (r + alpha))^6 and (sigma / (r + alpha))^12
        inv_r = self.sigma / (pairwise_dr + self.alpha)
        inv_r6 = inv_r**6
        inv_r12 = inv_r6**2
        # Compute LJ potential: 4 * epsilon * (inv_r12 - inv_r6)
        lj_energy = self.epsilon_val * (inv_r12 - 2 * inv_r6)

        # Apply cutoff: set energy to zero for distances > cutoff
        lj_energy = jnp.where(pairwise_dr <= 2.5 * self.sigma, lj_energy, 0.0)
        # Sum over all pairs to get total energy per sample
        total_lj_energy = jnp.sum(lj_energy, axis=-1)

        return total_lj_energy

    def compute_soft_core_lj_energy(self, x: jnp.ndarray) -> jnp.ndarray:
        pairwise_dr = compute_distances(
            x, self.n_particles, self.n_spatial_dim, min_dr=self.min_dr
        )
        lj_energy = self.soft_core_lennard_jones_potential(pairwise_dr)

        if self.include_harmonic:
            harmonic_energy = self.harmonic_potential(x)

            return lj_energy + self.c * harmonic_energy
        else:
            return lj_energy

    def log_prob(self, x: chex.Array) -> chex.Array:
        return -self.compute_soft_core_lj_energy(x)

    def batched_log_prob(self, xs):
        return jax.vmap(self.log_prob)(xs)

    def sample(self, key: jax.random.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        """Not implemented as sampling directly is difficult."""
        raise NotImplementedError("Direct sampling from LJ potential not implemented")

    def interatomic_dist(self, x):
        x = x.reshape(-1, self.n_particles, self.n_spatial_dim)
        distances = jax.vmap(
            lambda x: compute_distances(x, self.n_particles, self.n_spatial_dim)
        )(x)

        return distances

    def visualise(self, samples: chex.Array) -> plt.Figure:
        """Visualize samples against validation set"""

        dist_samples = self.interatomic_dist(samples)
        energy_samples = -self.batched_log_prob(samples)

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].hist(
            dist_samples.flatten(),
            bins=100,
            alpha=0.5,
            density=True,
            histtype="step",
            linewidth=4,
        )
        axs[0].set_xlabel("Interatomic distance")

        axs[1].hist(
            energy_samples,
            bins=100,
            density=True,
            alpha=0.4,
            range=(energy_samples.min(), energy_samples.max()),
            color="r",
            histtype="step",
            linewidth=4,
            label="Generated data",
        )
        axs[1].set_xlabel("Energy")

        fig.canvas.draw()
        return fig
