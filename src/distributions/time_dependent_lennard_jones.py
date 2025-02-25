import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from .base import Target
import chex


class TimeDependentLennardJonesEnergy(Target):
    TIME_DEPENDENT = True

    def __init__(
        self,
        dim: int,
        n_particles: int,
        alpha: float = 0.5,
        sigma: float = 1.0,
        epsilon_val: float = 1.0,
        min_dr: float = 1e-4,
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

    def time_dependent_lennard_jones_potential(
        self,
        pairwise_dr: jnp.ndarray,
        t: float,
    ) -> jnp.ndarray:
        """
        Compute the time-dependent Lennard-Jones potential.

        Args:
            pairwise_dr (jnp.ndarray): Pairwise distances of shape [n_pairs].
            t (float): Time parameter, influencing the strength of the potential.

        Returns:
            jnp.ndarray: Time-dependent Lennard-Jones potential energy of shape [].
        """
        # Compute (sigma / r)^6 and (sigma / r)^12
        inv_r6 = (self.sigma / (pairwise_dr + self.alpha * (1 - t))) ** 6
        inv_r12 = inv_r6**2

        # Compute LJ potential: 4 * epsilon * (inv_r12 - inv_r6)
        lj_energy = 4 * self.epsilon_val * (inv_r12 - inv_r6)

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
        distances = jnp.maximum(
            jnp.sqrt(jnp.sum(dx**2, axis=-1) + epsilon), self.min_dr
        )

        return distances

    def compute_time_dependent_lj_energy(
        self,
        x: jnp.ndarray,
        t: float,
    ) -> jnp.ndarray:
        """
        Compute the total time-dependent Lennard-Jones energy for a batch of samples.

        Args:
            x (jnp.ndarray): Input array of shape [n_particles * n_spatial_dim].
            t (float): Time parameter.

        Returns:
            jnp.ndarray: Total time-dependent Lennard-Jones energy.
        """
        pairwise_dr = self.compute_distances(
            x.reshape(self.n_particles, self.n_spatial_dim)
        )
        lj_energy = self.time_dependent_lennard_jones_potential(pairwise_dr, t)
        return lj_energy

    def log_prob(self, x: chex.Array) -> chex.Array:
        return -self.compute_time_dependent_lj_energy(x, 1.0)

    def time_dependent_log_prob(self, x: chex.Array, t: float) -> chex.Array:
        return -self.compute_time_dependent_lj_energy(x, t)

    def score(self, x: chex.Array, t: float) -> chex.Array:
        return jax.grad(self.log_prob, argnums=0)(x, t)

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

        min_energy = jnp.min(energy_samples)
        max_energy = jnp.max(energy_samples)

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

        return fig
