import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from .base import Target
import chex


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

        # U(λ, r) = 0.5 * epsilon * λ^n * [ (α_LJ * (1 - λ)^m + (r/sigma)^6)^-2 - (α_LJ * (1 - λ)^m + (r/sigma)^6)^-1 ]
        lambda_ = t

        inv_r6 = (pairwise_dr / self.sigma) ** 6
        soft_core_term = self.alpha * (1 - lambda_) ** self.m + inv_r6
        lj_energy = (
            self.epsilon_val
            * lambda_**self.n
            * (soft_core_term**-2 - 2 * soft_core_term**-1)
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

        axs[0].hist(
            dist_samples.flatten(),
            bins=100,
            alpha=0.5,
            density=True,
            histtype="step",
            linewidth=4,
        )
        axs[0].set_xlabel("Interatomic distance")
        axs[0].legend(["Generated data"])

        energy_samples = -self.batched_log_prob(samples, 1.0)
        # Clip energy values for visualization
        energy_samples = jnp.nan_to_num(
            energy_samples, nan=0.0, posinf=1000.0, neginf=-1000.0
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
            label="Generated data",
        )
        axs[1].set_xlabel("Energy")
        axs[1].legend()

        fig.canvas.draw()
        return fig
