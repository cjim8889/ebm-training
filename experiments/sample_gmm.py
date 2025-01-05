import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from distributions import AnnealedDistribution
from distributions.multivariate_gaussian import MultivariateGaussian
from distributions import GMM

from utils.smc import generate_samples_with_smc, SampleBuffer

key = jax.random.PRNGKey(1234)


initial_density = MultivariateGaussian(dim=2, mean=0, sigma=20)

key, subkey = jax.random.split(key)
target_density = GMM(subkey, dim=2)
path_density = AnnealedDistribution(
    initial_density=initial_density, target_density=target_density, method="geometric"
)

key, subkey = jax.random.split(key)
ts = jnp.linspace(0, 1, 128)


# buffer = SampleBuffer(
#     buffer_size=25600, min_update_size=1024
# )

# warmup_counts = 5

# for i in range(warmup_counts):
#     key, subkey = jax.random.split(key)
#     samples = generate_samples_with_smc(
#         key=key,
#         time_dependent_log_density=path_density.time_dependent_log_prob,
#         num_samples=5120,
#         ts=ts,
#         sample_fn=path_density.sample_initial,
#         num_steps=10,
#         integration_steps=20,
#         eta=0.1,
#         rejection_sampling=True,
#         ess_threshold=0.5,
#         # incremental_log_delta=path_density.incremental_log_delta
#     )

#     key, subkey = jax.random.split(key)
#     buffer.add_samples(key, samples["positions"], samples["weights"])
#     print(f"Warmup {i} done")

print("Warmup done")
keys = jax.random.split(key, 3)
key = keys[0]
subkey = keys[1]
covariance_key = keys[2]

samples = generate_samples_with_smc(
    key=subkey,
    time_dependent_log_density=path_density.time_dependent_log_prob,
    num_samples=10240,
    ts=ts,
    sample_fn=path_density.sample_initial,
    num_steps=5,
    integration_steps=5,
    eta=0.83,
    rejection_sampling=True,
    ess_threshold=0.5,
)
fig = target_density.visualise(samples["positions"][-1])
# plt.show()
print(samples["ess"])
plt.savefig("gmm.png")
