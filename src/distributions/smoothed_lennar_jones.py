from typing import Callable

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from src.ode import generate_samples_with_log_prob
from src.utils.distributions import (
    compute_distances,
    compute_log_effective_sample_size,
)

from .base import Target


class QuadraticSmoothedLJ(Target):
    TIME_DEPENDENT = False
    TARGET_METRIC = (
        ("ess", False),
    )

    def __init__(
        self,
        dim: int,
        n_particles: int,
        sigma: float = 1.0,
        epsilon_val: float = 1.0,
        r_min: float = 0.1,
        V_max: float = 100.0,
        min_dr: float = 1e-4,
        c: float = 0.5,
        include_harmonic: bool = True,
        log_prob_clip: float = None,
        log_prob_clip_min: float = None,
        log_prob_clip_max: float = None,
        **kwargs,
    ):
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
        self.r_min = r_min
        self.V_max = V_max
        self.min_dr = min_dr
        self.c = c
        self.include_harmonic = include_harmonic
        self.log_prob_clip = log_prob_clip
        self.log_prob_clip_min = log_prob_clip_min
        self.log_prob_clip_max = log_prob_clip_max

        # Compute polynomial coefficients
        self.a, self.b, self.c_coeff = self.compute_quadratic_coefficients()

    def compute_quadratic_coefficients(self):
        r_min = self.r_min
        sigma = self.sigma
        epsilon = self.epsilon_val
        V_max = self.V_max

        # Compute V_LJ(r_min)
        u = sigma / r_min
        V_LJ = epsilon * (u**12 - 2 * u**6)

        # Compute V_LJ'(r_min)
        V_LJ_prime = -12 * epsilon * sigma / r_min**2 * (u**11 - u**5)

        # Compute coefficients
        c = V_max
        a = (V_LJ_prime * r_min + V_max - V_LJ) / (r_min**2)
        b = V_LJ_prime - 2 * a * r_min

        return a, b, c

    def smoothed_lennard_jones_potential(self, pairwise_dr: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the smoothed Lennard-Jones potential using quadratic polynomial for r < r_min.
        """
        sigma = self.sigma
        epsilon = self.epsilon_val
        r_min = self.r_min
        a, b, c = self.a, self.b, self.c_coeff

        # Standard LJ for r >= r_min
        u = sigma / pairwise_dr
        lj_energy = epsilon * (u**12 - 2 * u**6)

        # Polynomial for r < r_min
        poly_energy = a * pairwise_dr**2 + b * pairwise_dr + c

        # Combine using where
        energy = jnp.where(pairwise_dr < r_min, poly_energy, lj_energy)

        # Apply cutoff
        energy = jnp.where(pairwise_dr <= 2.5 * sigma, energy, 0.0)

        return energy
    
    def harmonic_potential(self, x):
        """
        Compute the harmonic potential energy.

        E^osc(x) = 1/2 * Σ ||xi - x_COM||^2
        """
        x = x.reshape(self.n_particles, self.n_spatial_dim)
        x_com = jnp.mean(x, axis=0, keepdims=True)
        distances_to_com = optax.safe_norm(
            x - x_com,
            ord=2,
            axis=-1,
            min_norm=0.0,
        )

        return 0.5 * jnp.sum(distances_to_com**2)

    def compute_smoothed_lj_energy(self, x: jnp.ndarray) -> jnp.ndarray:
        pairwise_dr = compute_distances(x, self.n_particles, self.n_spatial_dim, min_dr=self.min_dr)
        lj_energy = self.smoothed_lennard_jones_potential(pairwise_dr)
        total_lj_energy = jnp.sum(lj_energy, axis=-1)

        if self.include_harmonic:
            harmonic_energy = self.harmonic_potential(x)
            return total_lj_energy + self.c * harmonic_energy
        else:
            return total_lj_energy

    def log_prob(self, x: chex.Array) -> chex.Array:
        p_t = -self.compute_smoothed_lj_energy(x)

        if self.log_prob_clip is not None:
            clip_min = -self.log_prob_clip
            clip_max = self.log_prob_clip
        else:
            clip_min = self.log_prob_clip_min
            clip_max = self.log_prob_clip_max

        if clip_min is not None or clip_max is not None:
            p_t = jnp.clip(p_t, a_min=clip_min, a_max=clip_max)

        return p_t
    
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
            linewidth=2,
        )
        axs[0].set_xlabel("Interatomic distance")

        axs[1].hist(
            energy_samples,
            bins=100,
            density=True,
            alpha=0.4,
            range=(energy_samples.min(), energy_samples.max()),
            histtype="step",
            linewidth=2,
        )
        axs[1].set_xlabel("Energy")

        fig.canvas.draw()
        return fig

    def evaluate(
        self,
        key: chex.PRNGKey,
        *,
        v_theta: Callable,
        ts: chex.Array,
        base_density: Target,
        use_shortcut: bool = False,
        **kwargs,
    ):
        metrics = {}

        key, sample_key = jax.random.split(key)
        initial_samples = base_density.sample(
            sample_key, (self.n_model_samples_eval,)
        )  # Sample from base distribution q_0
        initial_log_probs = jax.vmap(base_density.log_prob)(initial_samples)

        samples_q, samples_log_q = generate_samples_with_log_prob(
            v_theta=v_theta,
            initial_samples=initial_samples,
            initial_log_probs=initial_log_probs,
            ts=ts,
            use_shortcut=use_shortcut,
        )

        metrics["figure"] = self.visualise(samples_q)

        log_prob_samples = self.batched_log_prob(samples_q)

        ess = compute_log_effective_sample_size(
            log_p=log_prob_samples,
            log_q=samples_log_q,
        )
        metrics["ess"] = jnp.exp(ess)

        return metrics
