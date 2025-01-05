import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from distributions import AnnealedDistribution
from distributions.multivariate_gaussian import MultivariateGaussian
from distributions.time_dependent_lennard_jones_butler import (
    TimeDependentLennardJonesEnergyButler,
    TimeDependentLennardJonesEnergyButlerWithTemperatureTempered,
)
from distributions import SoftCoreLennardJonesEnergy

from utils.optimization import power_schedule, inverse_power_schedule
from utils.smc import generate_samples_with_smc, SampleBuffer

key = jax.random.PRNGKey(1234)

initial_density = MultivariateGaussian(dim=39, mean=0, sigma=3)
target_density = TimeDependentLennardJonesEnergyButler(
    dim=39,
    n_particles=13,
    alpha=0.5,
    sigma=1.0,
    epsilon_val=1.0,
    min_dr=1e-4,
    n=1,
    m=1,
    c=0.5,
    include_harmonic=True,
    # log_prob_clip=100.0,
)

# target_density = SoftCoreLennardJonesEnergy(
#     dim=39,
#     n_particles=13,
#     sigma=1.0,
#     epsilon_val=1.0,
#     min_dr=1e-4,
#     alpha=0.2,
#     c=0.5,
#     include_harmonic=True,
#     log_prob_clip=100.,
# )

path_density = AnnealedDistribution(
    initial_density=initial_density, target_density=target_density, method="geometric"
)

reference_density = SoftCoreLennardJonesEnergy(
    dim=39,
    n_particles=13,
    sigma=1.0,
    epsilon_val=1.0,
    min_dr=1e-3,
    alpha=0.0,
    c=0.5,
    include_harmonic=True,
)
key, subkey = jax.random.split(key)
ts = jnp.linspace(0, 1, 128)

# buffer = SampleBuffer(
#     buffer_size=25600, min_update_size=1024
# )

# warmup_counts = 2

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
#         eta=0.015,
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
    num_samples=25600,
    ts=ts,
    sample_fn=path_density.sample_initial,
    num_steps=20,
    integration_steps=50,
    eta=0.015,
    rejection_sampling=True,
    ess_threshold=0.5,
    # covariances=buffer.estimate_covariance(covariance_key, num_samples=10240),
    # incremental_log_delta=path_density.incremental_log_delta
)
fig = reference_density.visualise(samples["positions"][-1])
# plt.show()
plt.savefig("lj13c.png")
