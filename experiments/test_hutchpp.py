import jax
import jax.numpy as jnp

from src.distributions import AnnealedDistribution, MultivariateGaussian, QuadraticSmoothedLJ

from src.models.mlp import VelocityFieldTwo
from src.training.core import Particle
from src.training.loss import epsilon_with_hutchinson_Q
from src.utils.distributions import divergence_velocity_with_shortcut
from src.utils.hutchpp import get_Q_velocity, hutchinson_divergence_Q

initial_density = MultivariateGaussian(dim=39, mean=0, sigma=2)
target_density = QuadraticSmoothedLJ(dim=39, n_particles=13, r_min=0.8)

annealed_distribution = AnnealedDistribution(initial_density, target_density)

# Setup
key = jax.random.PRNGKey(0)
mlp = VelocityFieldTwo(
    key=key,
    dim=39,
    hidden_dim=4,
    depth=3,
    shortcut=True,
)

batch_size = 1024
key, subkey = jax.random.split(key)
pos_batch = jax.random.normal(subkey, (batch_size, 39))
pos = pos_batch[0]

t = jnp.array(0.5)
sigma = jnp.array(0.125)

Q, primals = get_Q_velocity(mlp, pos, t, 10, d=sigma)

epsilon = epsilon_with_hutchinson_Q(
    mlp,
    Particle(pos, t, sigma, sigma),
    annealed_distribution.score_fn,
    annealed_distribution.time_derivative,
    jax.random.normal(key, (10, 39)),
    Q,
)

print(epsilon)

# estimate = hutchinson_divergence_Q(mlp, pos, t, jax.random.normal(key, (10, 256)), Q, d=sigma)
# print(estimate)

# ground_truth = divergence_velocity_with_shortcut(mlp, pos, t, d=sigma)
# print(ground_truth)