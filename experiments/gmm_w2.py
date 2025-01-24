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
samples_1 = target_density.sample(subkey, (5000,))

key, subkey = jax.random.split(key)
samples_2 = target_density.sample(subkey, (5000,))

w1_dist, w2_dist = compute_wasserstein_distance_pot(samples_1, samples_2)


M = pot.dist(samples_1, samples_2)
a, b = (
    jnp.ones(samples_1.shape[0]) / samples_1.shape[0],
    jnp.ones(samples_2.shape[0]) / samples_2.shape[0],
)

# Time EMD calculation
start_time = time.time()
w2_dist_2 = jnp.sqrt(pot.emd2(a, b, M, numThreads=1))
emd_time = time.time() - start_time
print(f"EMD calculation time: {emd_time:.4f} seconds")

# Time Sinkhorn calculation
start_time = time.time()
w2_dist_sinkhorn = jnp.sqrt(pot.sinkhorn2(a, b, M, reg=1.0))
sinkhorn_time = time.time() - start_time
print(f"Sinkhorn calculation time: {sinkhorn_time:.4f} seconds")

print(
    f"Distances - W1: {w1_dist}, W2: {w2_dist}, W2 (EMD): {w2_dist_2}, W2 (Sinkhorn): {w2_dist_sinkhorn}"
)
