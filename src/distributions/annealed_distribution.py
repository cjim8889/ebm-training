import chex
import jax

from src.utils.distributions import get_inverse_temperature

from .base import Target


class AnnealedDistribution(Target):
    TIME_DEPENDENT = True

    def __init__(
        self,
        initial_density: Target,
        target_density: Target,
        method: str = "linear",
        prior_regularization: bool = False,
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
        self.prior_regularization = prior_regularization
        self.method = method

    def log_prob(self, xs: chex.Array) -> chex.Array:
        return self.time_dependent_log_prob(xs, 1.0)

    def base_log_prob(self, xs: chex.Array) -> chex.Array:
        return self.initial_density.log_prob(xs)

    def time_dependent_log_prob(self, xs: chex.Array, t: chex.Array) -> chex.Array:
        if self.method == "linear":
            beta = t
        elif self.method == "inverse_power":
            beta = (1 - (1 - t) ** 0.5)
        else:
            beta = get_inverse_temperature(t, 250.0, 1.0)
        
        if self.prior_regularization:
            initial_prob = self.initial_density.log_prob(xs)
        else:
            initial_prob = (1 - beta) * self.initial_density.log_prob(xs)
        

        if self.target_density.TIME_DEPENDENT:
            target_prob = beta * self.target_density.time_dependent_log_prob(xs, t)
        else:
            target_prob = beta * self.target_density.log_prob(xs)

        return initial_prob + target_prob

    def incremental_log_delta(self, xs: chex.Array, dt: float) -> chex.Array:
        return dt * (
            self.target_density.log_prob(xs) - self.initial_density.log_prob(xs)
        )

    def time_derivative(self, xs: chex.Array, t: float) -> chex.Array:
        return jax.grad(lambda t: self.time_dependent_log_prob(xs, t))(t)

    def score_fn(self, xs: chex.Array, t: float) -> chex.Array:
        return jax.grad(lambda x: self.time_dependent_log_prob(x, t))(xs)

    def sample_initial(self, key: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        return self.initial_density.sample(key, sample_shape)
