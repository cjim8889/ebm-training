import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import numpy as np

from utils.distributions import batched_remove_mean, compute_distances

from .base import Target


class LennardJonesEnergy(Target):
    TIME_DEPENDENT = False

    def __init__(
        self,
        dim: int,
        n_particles: int,
        data_path_train: str = None,
        data_path_test: str = None,
        data_path_val: str = None,
        c: float = 1.0,
        log_prob_clip: float = None,
        log_prob_clip_min: float = None,
        log_prob_clip_max: float = None,
        include_harmonic: bool = False,
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

        self.c = c
        self.include_harmonic = include_harmonic

        # self._train_set = self.setup_train_set()
        self._test_set = self.setup_test_set()
        # self._val_set = self.setup_val_set()

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
        pairwise_dr = compute_distances(x, self.n_particles, self.n_spatial_dim, min_dr)
        lj_energy = self.safe_lennard_jones_potential(pairwise_dr, sigma, epsilon_val)

        if self.include_harmonic:
            lj_energy += self.c * self.harmonic_potential(x)

        return lj_energy

    def log_prob(self, x: chex.Array) -> chex.Array:
        p_t = -self.compute_safe_lj_energy(x)

        # Handle legacy log_prob_clip parameter for backward compatibility
        if self.log_prob_clip is not None:
            clip_min = -self.log_prob_clip
            clip_max = self.log_prob_clip
        else:
            clip_min = self.log_prob_clip_min
            clip_max = self.log_prob_clip_max

        if clip_min is not None or clip_max is not None:
            p_t = jnp.clip(p_t, a_min=clip_min, a_max=clip_max)

        return p_t

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
        return jax.vmap(
            lambda x: compute_distances(x, self.n_particles, self.n_spatial_dim)
        )(x)

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

        min_energy = -100
        max_energy = 100

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
