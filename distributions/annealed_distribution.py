from .base import Target
from utils.distributions import get_inverse_temperature
import jax
import chex


class AnnealedDistribution(Target):
    TIME_DEPENDENT = True

    def __init__(
        self,
        initial_density: Target,
        target_density: Target,
        method: str = "linear",
    ):
        super().__init__(
            dim=initial_density.dim,
            log_Z=0.0,
            can_sample=False,
            n_plots=1,
            n_model_samples_eval=1000,
            n_target_samples_eval=1000,
        )
        self.initial_density = initial_density
        self.target_density = target_density
        self.method = method

    def log_prob(self, xs: chex.Array) -> chex.Array:
        return self.time_dependent_log_prob(xs, 1.0)

    def time_dependent_log_prob(self, xs: chex.Array, t: chex.Array) -> chex.Array:
        if self.method == "linear":
            beta = t
        else:
            beta = get_inverse_temperature(t, 250., 1.0)

        initial_prob = (1 - beta) * self.initial_density.log_prob(xs)

        if self.target_density.TIME_DEPENDENT:
            target_prob = beta * self.target_density.time_dependent_log_prob(xs, t)
        else:
            target_prob = beta * self.target_density.log_prob(xs)

        return initial_prob + target_prob

    def incremental_log_delta(self, xs: chex.Array, dt: float) -> chex.Array:
        return -self.time_dependent_log_prob(xs, 1.0) * dt

    def time_derivative(self, xs: chex.Array, t: float) -> chex.Array:
        return jax.grad(lambda t: self.time_dependent_log_prob(xs, t))(t)

    def score_fn(self, xs: chex.Array, t: float) -> chex.Array:
        return jax.grad(lambda x: self.time_dependent_log_prob(x, t))(xs)

    def sample_initial(self, key: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        return self.initial_density.sample(key, sample_shape)
