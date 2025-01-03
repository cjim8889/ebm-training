import jax.numpy as jnp
import distrax
import chex
import jax
from .base import Target
import matplotlib.pyplot as plt
from utils.plotting import plot_contours_2D, plot_marginal_pair


class GMM(Target):
    TIME_DEPENDENT = False

    def __init__(
        self,
        key: chex.PRNGKey,
        dim: int = 2,
        n_mixes: int = 40,
        loc_scaling: float = 40,
        scale_scaling: float = 1.0,
        fixed_mean: bool = True,
    ) -> None:
        super().__init__(
            dim=dim,
            log_Z=0.0,
            can_sample=True,
            n_plots=1,
            n_model_samples_eval=1000,
            n_target_samples_eval=1000,
        )

        self.n_mixes = n_mixes

        logits = jnp.ones(n_mixes)
        if fixed_mean:
            mean = jnp.array(
                [
                    [-0.2995, 21.4577],
                    [-32.9218, -29.4376],
                    [-15.4062, 10.7263],
                    [-0.7925, 31.7156],
                    [-3.5498, 10.5845],
                    [-12.0885, -7.8626],
                    [-38.2139, -26.4913],
                    [-16.4889, 1.4817],
                    [15.8134, 24.0009],
                    [-27.1176, -17.4185],
                    [14.5287, 33.2155],
                    [-8.2320, 29.9325],
                    [-6.4473, 4.2326],
                    [36.2190, -37.1068],
                    [-25.1815, -10.1266],
                    [-15.5920, 34.5600],
                    [-25.9272, -18.4133],
                    [-27.9456, -37.4624],
                    [-23.3496, 34.3839],
                    [17.8487, 19.3869],
                    [2.1037, -20.5073],
                    [6.7674, -37.3478],
                    [-28.9026, -20.6212],
                    [25.2375, 23.4529],
                    [-17.7398, -1.4433],
                    [25.5824, 39.7653],
                    [15.8753, 5.4037],
                    [26.8195, -23.5521],
                    [7.4538, -31.0122],
                    [-27.7234, -20.6633],
                    [18.0989, 16.0864],
                    [-23.6941, 12.0843],
                    [21.9589, -5.0487],
                    [1.5273, 9.2682],
                    [24.8151, 38.4078],
                    [-30.8249, -14.6588],
                    [15.7204, 33.1420],
                    [34.8083, 35.2943],
                    [7.9606, -34.7833],
                    [3.6797, -25.0242],
                ]
            )
        else:
            mean = jax.random.normal(key, shape=(n_mixes, dim)) * loc_scaling

        scale = jnp.ones(shape=(n_mixes, dim)) * scale_scaling

        mixture_dist = distrax.Categorical(logits=logits)
        components_dist = distrax.Independent(
            distrax.Normal(loc=mean, scale=scale), reinterpreted_batch_ndims=1
        )
        self.distribution = distrax.MixtureSameFamily(
            mixture_distribution=mixture_dist,
            components_distribution=components_dist,
        )

        self._plot_bound = loc_scaling * 1.5

    def log_prob(self, x: chex.Array) -> chex.Array:
        log_prob = self.distribution.log_prob(x)
        return log_prob

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape = ()) -> chex.Array:
        return self.distribution.sample(seed=seed, sample_shape=sample_shape)

    def visualise(
        self,
        samples: chex.Array,
    ) -> plt.Figure:
        """Visualise samples from the model."""
        fig, ax = plt.subplots(1, figsize=(6, 6))
        plot_marginal_pair(samples, ax, bounds=(-self._plot_bound, self._plot_bound))
        if self.dim == 2:
            plot_contours_2D(self.log_prob, ax, bound=self._plot_bound, levels=50)

        return fig
