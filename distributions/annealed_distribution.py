from .base import Target
import jax
import chex


class AnnealedDistribution(Target):
    TIME_DEPENDENT = True

    def __init__(
        self,
        initial_distribution: Target,
        target_distribution: Target,
        dim: int = 2,
        max_score: float = 1.0,
    ):
        super().__init__(
            dim=dim,
            log_Z=0.0,
            can_sample=False,
            n_plots=1,
            n_model_samples_eval=1000,
            n_target_samples_eval=1000,
        )
        self.initial_distribution = initial_distribution
        self.target_distribution = target_distribution
        self.max_score = max_score

    def log_prob(self, xs: chex.Array) -> chex.Array:
        return self.time_dependent_log_prob(xs, 1.0)

    def time_dependent_log_prob(self, xs: chex.Array, t: chex.Array) -> chex.Array:
        initial_prob = (1 - t) * self.initial_distribution.log_prob(xs)

        if self.target_distribution.TIME_DEPENDENT:
            target_prob = t * self.target_distribution.time_dependent_log_prob(xs, t)
        else:
            target_prob = t * self.target_distribution.log_prob(xs)

        return initial_prob + target_prob

    def time_derivative(self, xs: chex.Array, t: float) -> chex.Array:
        return jax.grad(lambda t: self.time_dependent_log_prob(xs, t))(t)

    def score_fn(self, xs: chex.Array, t: float) -> chex.Array:
        return jax.grad(lambda x: self.time_dependent_log_prob(x, t))(xs)

    def sample_initial(self, key: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        return self.initial_distribution.sample(key, sample_shape)
