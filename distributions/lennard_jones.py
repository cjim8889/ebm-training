import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from utils.distributions import batched_remove_mean

from .base import Target


class LennardJonesEnergy(Target):
    TIME_DEPENDENT = False

    def __init__(
        self,
        dim: int,
        n_particles: int,
        data_path_train: str,
        data_path_test: str,
        data_path_val: str,
        log_prob_clip: float = 100.0,
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

        self.data_path_test = data_path_test
        self.data_path_val = data_path_val
        self.data_path_train = data_path_train

        self.log_prob_clip = log_prob_clip

        self._train_set = self.setup_train_set()
        self._test_set = self.setup_test_set()
        self._val_set = self.setup_val_set()

    def safe_lennard_jones_potential(
        self,
        pairwise_dr: jnp.ndarray,
        sigma: float = 1.0,
        epsilon_val: float = 1.0,
    ) -> jnp.ndarray:
        """
        Compute the Lennard-Jones potential in a safe manner to prevent numerical instability.

        Args:
            pairwise_dr (jnp.ndarray): Pairwise distances of shape [n_pairs].
            sigma (float): Finite distance at which the inter-particle potential is zero.
            epsilon_val (float): Depth of the potential well.

        Returns:
            jnp.ndarray: Lennard-Jones potential energy of shape [].
        """
        # Compute (sigma / r)^6 and (sigma / r)^12
        inv_r = sigma / pairwise_dr
        inv_r6 = inv_r**6
        inv_r12 = inv_r6**2

        # Compute LJ potential: 4 * epsilon * (inv_r12 - inv_r6)
        lj_energy = epsilon_val * (inv_r12 - 2 * inv_r6)

        # Sum over all pairs to get total energy per sample
        total_lj_energy = jnp.sum(lj_energy, axis=-1)

        return total_lj_energy

    def compute_distances(self, x, epsilon=1e-8, min_dr: float = 1e-3):
        x = x.reshape(self.n_particles, self.n_spatial_dim)

        # Get indices of upper triangular pairs
        i, j = jnp.triu_indices(self.n_particles, k=1)

        # Calculate displacements between pairs
        dx = x[i] - x[j]

        # Compute distances
        distances = jnp.maximum(jnp.sqrt(jnp.sum(dx**2, axis=-1) + epsilon), min_dr)

        return distances

    def compute_safe_lj_energy(
        self,
        x: jnp.ndarray,
        sigma: float = 1.0,
        epsilon_val: float = 1.0,
        min_dr: float = 1e-2,
    ) -> jnp.ndarray:
        """
        Compute the total Lennard-Jones energy for a batch of samples in a safe manner.

        Args:
            x (jnp.ndarray): Input array of shape [n_particles * n_spatial_dim].
            sigma (float): Finite distance at which the inter-particle potential is zero.
            epsilon_val (float): Depth of the potential well.
            min_dr (float): Minimum allowed distance to prevent division by zero.

        Returns:
            jnp.ndarray: Total Lennard-Jones energy for each sample, shape [batch_size].
        """
        pairwise_dr = self.compute_distances(
            x.reshape(self.n_particles, self.n_spatial_dim), min_dr
        )
        lj_energy = self.safe_lennard_jones_potential(pairwise_dr, sigma, epsilon_val)
        return lj_energy

    def log_prob(self, x: chex.Array) -> chex.Array:
        return jnp.clip(
            jnp.nan_to_num(
                -self.compute_safe_lj_energy(x),
                nan=0.0,
                posinf=self.log_prob_clip,
                neginf=-self.log_prob_clip,
            ),
            min=-self.log_prob_clip,
            max=self.log_prob_clip,
        )

    def score(self, x: chex.Array) -> chex.Array:
        return jax.grad(self.log_prob)(x)

    def sample(
        self, key: jax.random.PRNGKey, sample_shape: chex.Shape = ()
    ) -> chex.Array:
        raise NotImplementedError(
            "Sampling is not implemented for MultiDoubleWellEnergy"
        )

    def setup_train_set(self):
        data = np.load(self.data_path_train, allow_pickle=True)
        data = batched_remove_mean(
            jnp.array(data), self.n_particles, self.n_spatial_dim
        )
        return data

    def setup_test_set(self):
        data = np.load(self.data_path_test, allow_pickle=True)
        data = batched_remove_mean(
            jnp.array(data), self.n_particles, self.n_spatial_dim
        )
        return data

    def setup_val_set(self):
        data = np.load(self.data_path_val, allow_pickle=True)
        data = batched_remove_mean(
            jnp.array(data), self.n_particles, self.n_spatial_dim
        )
        return data

    def interatomic_dist(self, x):
        x = x.reshape(-1, self.n_particles, self.n_spatial_dim)
        distances = jax.vmap(lambda x: self.compute_distances(x))(x)

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
