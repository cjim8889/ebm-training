from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from utils.distributions import (
    compute_w1_distance_1d_pot,
    compute_w2_distance_1d_pot,
    compute_wasserstein_distance_pot,
    estimate_kl_divergence,
)
from utils.plotting import plot_contours_2D, plot_marginal_pair

from .base import Target
from .double_well import DoubleWellEnergy


class ManyWellEnergy(Target):
    TIME_DEPENDENT = False

    def __init__(self, dim: int = 32, n_samples_eval: int = 1024):
        assert dim % 2 == 0
        self.n_wells = dim // 2
        self.double_well_energy = DoubleWellEnergy()

        log_Z = self.double_well_energy.log_Z * self.n_wells
        super().__init__(
            dim=dim,
            log_Z=log_Z,
            can_sample=False,
            n_plots=1,
            n_model_samples_eval=n_samples_eval,
            n_target_samples_eval=n_samples_eval,
        )

        self.centre = 1.7
        self.max_dim_for_all_modes = (
            40  # otherwise we get memory issues on huuuuge test set
        )
        if self.dim < self.max_dim_for_all_modes:
            dim_1_vals_grid = jnp.meshgrid(
                *[jnp.array([-self.centre, self.centre]) for _ in range(self.n_wells)]
            )
            dim_1_vals = jnp.stack([dim.flatten() for dim in dim_1_vals_grid], axis=-1)
            n_modes = 2**self.n_wells
            assert n_modes == dim_1_vals.shape[0]
            test_set = jnp.zeros((n_modes, dim))
            test_set = test_set.at[:, jnp.arange(dim) % 2 == 0].set(dim_1_vals)
            self.modes_test_set = test_set
        else:
            raise NotImplementedError("still need to implement this")

        self.shallow_well_bounds = [-1.75, -1.65]
        self.deep_well_bounds = [1.7, 1.8]
        self._plot_bound = 3.0

        self.double_well_samples = self.double_well_energy.sample(
            key=jax.random.PRNGKey(0), shape=(int(1e6),)
        )

        if self.n_target_samples_eval < self.modes_test_set.shape[0]:
            print("Evaluation occuring on subset of the modes test set.")

    def log_prob(self, x):
        return jnp.sum(
            jnp.stack(
                [
                    self.double_well_energy.log_prob(x[..., i * 2 : i * 2 + 2])
                    for i in range(self.n_wells)
                ],
                axis=-1,
            ),
            axis=-1,
        )

    def log_prob_2D(self, x):
        """Marginal 2D pdf - useful for plotting."""
        return self.double_well_energy.log_prob(x)

    def get_target_log_prob_marginal_pair(self, i: int, j: int, total_dim: int):
        def log_prob_marginal_pair(x_2d):
            x = jnp.zeros((x_2d.shape[0], total_dim))
            x = x.at[:, i].set(x_2d[:, 0])
            x = x.at[:, j].set(x_2d[:, 1])
            return self.log_prob(x)

        return log_prob_marginal_pair

    def visualise(self, samples: chex.Array) -> plt.Figure:
        alpha = 0.3
        plotting_bounds = (-3, 3)
        dim = samples.shape[-1]
        fig, axs = plt.subplots(2, 2, sharex="row", sharey="row", figsize=(10, 8))

        for i in range(2):
            for j in range(2):
                target_log_prob = self.get_target_log_prob_marginal_pair(i, j + 2, dim)
                plot_contours_2D(
                    target_log_prob, bound=self._plot_bound, ax=axs[i, j], levels=20
                )
                plot_marginal_pair(
                    samples,
                    marginal_dims=(i, j + 2),
                    ax=axs[i, j],
                    bounds=plotting_bounds,
                    alpha=alpha,
                )

                if j == 0:
                    axs[i, j].set_ylabel(f"$x_{{{i + 1}}}$")
                if i == 1:
                    axs[i, j].set_xlabel(f"$x_{{{j + 3}}}$")

        plt.tight_layout()
        return fig

    def evaluate(
        self,
        key: jax.Array,
        samples: jax.Array,
        time: jax.Array | None = None,
        **kwargs,
    ) -> dict:
        metrics = super().evaluate(key, samples, time=time, **kwargs)
        if samples.shape[0] > self.n_model_samples_eval:
            samples = samples[: self.n_model_samples_eval]

        true_samples = self.sample(
            key, (min(self.n_model_samples_eval, samples.shape[0]),)
        )

        x_w1_distance, x_w2_distance = compute_wasserstein_distance_pot(
            samples, true_samples
        )

        log_prob_samples = self.log_prob(samples)
        log_prob_true_samples = self.log_prob(true_samples)

        e_w2_distance = compute_w2_distance_1d_pot(
            log_prob_samples,
            log_prob_true_samples,
        )

        e_w1_distance = compute_w1_distance_1d_pot(
            log_prob_samples,
            log_prob_true_samples,
        )

        metrics["w2_distance"] = x_w2_distance
        metrics["w1_distance"] = x_w1_distance
        metrics["e_w2_distance"] = e_w2_distance
        metrics["e_w1_distance"] = e_w1_distance

        # KL divergence and log ESS
        key, kl_key = jax.random.split(key)
        kl_divergence = estimate_kl_divergence(
            v_theta=kwargs["v_theta"],
            num_samples=samples.shape[0],
            key=kl_key,
            ts=kwargs["ts"],
            log_prob_p_fn=self.log_prob,
            sample_p_fn=self.sample,
            base_log_prob_fn=kwargs["base_log_prob_fn"],
            final_time=1.0,
            use_shortcut=kwargs["use_shortcut"],
        )

        metrics["kl_divergence"] = kl_divergence

        return metrics

    def sample(self, key: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        key1, key2 = jax.random.split(key)
        dw_sample_indices = jax.random.randint(
            minval=0,
            maxval=self.double_well_samples.shape[0],
            key=key1,
            shape=(sample_shape[0] * self.n_wells,),
        )
        dw_samples = self.double_well_samples[dw_sample_indices]
        samples_p = jnp.reshape(dw_samples, (-1, self.dim))

        return samples_p

    def get_eval_samples(
        self, key: chex.PRNGKey, n: int
    ) -> Tuple[chex.Array, chex.Array]:
        key1, key2 = jax.random.split(key)
        dw_sample_indices = jax.random.randint(
            minval=0,
            maxval=self.double_well_samples.shape[0],
            key=key1,
            shape=(n * self.n_wells,),
        )
        dw_samples = self.double_well_samples[dw_sample_indices]
        samples_p = jnp.reshape(dw_samples, (-1, self.dim))

        if n < self.modes_test_set.shape[0]:
            mode_sample_indices = jax.random.choice(
                a=jnp.arange(self.modes_test_set.shape[0]),
                key=key2,
                shape=(n,),
                replace=False,
            )
            samples_modes = self.modes_test_set[mode_sample_indices]
        else:
            samples_modes = self.modes_test_set
        return samples_p, samples_modes
