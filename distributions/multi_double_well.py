import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from .base import Target
from typing import Optional
import chex


class MultiDoubleWellEnergy(Target):
    TIME_DEPENDENT = False

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

        self._val_set = self.setup_val_set()
        self._test_set = self.setup_test_set()

        self._mask = jnp.triu(jnp.ones((n_particles, n_particles), dtype=bool), k=1)

    def compute_distances(self, x, epsilon=1e-8):
        x = x.reshape(self.n_particles, self.n_spatial_dim)

        # Get indices of upper triangular pairs
        i, j = jnp.triu_indices(self.n_particles, k=1)

        # Calculate displacements between pairs
        dx = x[i] - x[j]

        # Compute distances
        distances = jnp.sqrt(jnp.sum(dx**2, axis=-1) + epsilon)

        return distances

    def batched_remove_mean(self, x):
        return x - jnp.mean(x, axis=1, keepdims=True)

    def multi_double_well_energy(self, x):
        dists = self.compute_distances(x)
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
        data = self.batched_remove_mean(data)
        return data

    def setup_val_set(self):
        data = np.load(self.data_path_val, allow_pickle=True)
        data = self.batched_remove_mean(data)
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
