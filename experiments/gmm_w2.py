import jax
import jax.numpy as jnp
import ot as pot
import time

from distributions import GMM
from distributions.multivariate_gaussian import MultivariateGaussian
from utils.distributions import compute_wasserstein_distance_pot

key = jax.random.PRNGKey(1234)


initial_density = MultivariateGaussian(dim=2, mean=0, sigma=20)

key, subkey = jax.random.split(key)
target_density = GMM(subkey, dim=2)

key, subkey = jax.random.split(key)
samples_1 = target_density.sample(subkey, (1000,))

key, subkey = jax.random.split(key)
samples_2 = target_density.sample(subkey, (1000,))

w1_dist, w2_dist = compute_wasserstein_distance_pot(samples_1, samples_2)

key, subkey = jax.random.split(key)
random_samples = jax.random.normal(subkey, (1000, 2)) * 25

w1_dist_random, w2_dist_random = compute_wasserstein_distance_pot(
    samples_1, random_samples
)
print(
    f"Distances - W1: {w1_dist}, W2: {w2_dist}, W1 (random): {w1_dist_random}, W2 (random): {w2_dist_random}"
)
