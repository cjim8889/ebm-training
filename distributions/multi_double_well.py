import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from .base import Target
from typing import Optional
import chex

from utils.distributions import (
    compute_wasserstein_distance_pot,
    compute_w1_distance_1d_pot,
    compute_w2_distance_1d_pot,
    compute_total_variation_distance,
)


class MultiDoubleWellEnergy(Target):
    TIME_DEPENDENT = False

    def __init__(
        self,
        dim: int,
        n_particles: int,
        data_path_test: Optional[str] = None,
        key: jax.random.PRNGKey = jax.random.PRNGKey(0),
        a: float = 0.9,
        b: float = -4.0,
        c: float = 0.0,
        offset: float = 4.0,
        log_prob_clip: Optional[float] = None,
        log_prob_clip_min: Optional[float] = None,
        log_prob_clip_max: Optional[float] = None,
        n_samples_eval: int = 2048,
    ):
        super().__init__(
            dim=dim,
            log_Z=None,
            can_sample=False,
            n_plots=10,
            n_model_samples_eval=n_samples_eval,
            n_target_samples_eval=n_samples_eval,
        )
        self.n_particles = n_particles
        self.n_spatial_dim = dim // n_particles
        self.key = key

        self.data_path_test = data_path_test

        self.a = a
        self.b = b
        self.c = c
        self.offset = offset

        self.log_prob_clip = log_prob_clip
        self.log_prob_clip_min = log_prob_clip_min
        self.log_prob_clip_max = log_prob_clip_max

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
        p_t = -self.multi_double_well_energy(x)

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

    def setup_test_set(self):
        data = np.load(self.data_path_test, allow_pickle=True)
        data = self.batched_remove_mean(data)
        return data

    def interatomic_dist(self, x):
        x = x.reshape(-1, self.n_particles, self.n_spatial_dim)
        distances = jax.vmap(lambda x: self.compute_distances(x))(x)

        return distances

    def batched_log_prob(self, xs):
        return jax.vmap(self.log_prob)(xs)

    def visualise(self, samples: chex.Array) -> plt.Figure:
        samples = jnp.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)

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

    def evaluate(self, key, samples, time=None, **kwargs):
        metrics = super().evaluate(key, samples, time=time, **kwargs)
        if samples.shape[0] > self.n_model_samples_eval:
            samples = samples[: self.n_model_samples_eval]

        true_samples = self._test_set[: self.n_target_samples_eval]

        x_w1_distance, x_w2_distance = compute_wasserstein_distance_pot(
            samples, true_samples
        )

        metrics["w1_distance"] = x_w1_distance
        metrics["w2_distance"] = x_w2_distance

        log_prob_samples = self.batched_log_prob(samples)
        log_prob_true_samples = self.batched_log_prob(true_samples)

        e_w2_distance = compute_w2_distance_1d_pot(
            log_prob_samples,
            log_prob_true_samples,
        )

        e_w1_distance = compute_w1_distance_1d_pot(
            log_prob_samples,
            log_prob_true_samples,
        )

        metrics["e_w2_distance"] = e_w2_distance
        metrics["e_w1_distance"] = e_w1_distance

        true_interatomic_dist = self.interatomic_dist(true_samples).reshape(-1, 1)
        samples_interatomic_dist = self.interatomic_dist(samples).reshape(-1, 1)

        dist_total_variation = compute_total_variation_distance(
            samples_interatomic_dist,
            true_interatomic_dist,
            num_bins=200,
            lower_bound=-10,
            upper_bound=10,
        )

        metrics["dist_total_variation"] = dist_total_variation

        return metrics
