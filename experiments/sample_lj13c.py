import blackjax
import blackjax.adaptation
import blackjax.adaptation.mass_matrix
import blackjax.smc.resampling as resampling
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from distributions import AnnealedDistribution
from distributions.multivariate_gaussian import MultivariateGaussian
from distributions.time_dependent_lennard_jones_butler import (
    TimeDependentLennardJonesEnergyButler, TimeDependentLennardJonesEnergyButlerWithTemperatureTempered
)
from distributions import SoftCoreLennardJonesEnergy

from utils.optimization import power_schedule, inverse_power_schedule
from utils.smc import generate_samples_with_smc

# plt.rcParams["figure.dpi"] = 300
# plt.rcParams["figure.figsize"] = [6.0, 4.0]

jax.config.update("jax_platform_name", "cpu")

key = jax.random.PRNGKey(1234)


initial_density = MultivariateGaussian(dim=39, mean=0, sigma=3)
# target_density = TimeDependentLennardJonesEnergyButler(
#     dim=39,
#     n_particles=13,
#     alpha=0.5,
#     sigma=1.0,
#     epsilon_val=1.0,
#     min_dr=1e-4,
#     n=1,
#     m=1,
#     c=0.5,
#     include_harmonic=True,
#     initial_temperature=100.,
#     # log_prob_clip=100.0,
# )
target_density = TimeDependentLennardJonesEnergyButler(
    dim=39,
    n_particles=13,
    alpha=0.2,
    sigma=1.0,
    epsilon_val=1.0,
    min_dr=1e-4,
    n=1,
    m=1,
    c=0.5,
    include_harmonic=True,
    # cubic_spline=True,
    # log_prob_clip=100.0,
)

initial_density = MultivariateGaussian(dim=39, mean=0.0, sigma=1.0)
path_density = AnnealedDistribution(
    initial_density=initial_density, target_density=target_density
)

key, subkey = jax.random.split(key)
gamma = 0.5
# ts = jnp.linspace(0, 1, 128)
ts = (1 + gamma) ** (jnp.linspace(0, 1, 128) + 1) - 1
ts = ts / ts[-1]
# print(ts)

# print(inverse_power_schedule(T=128, end_time=1.0, gamma=0.5))
# ts = inverse_power_schedule(T=128, end_time=1.0, gamma=0.5)
samples = generate_samples_with_smc(
    key=key,
    time_dependent_log_density=path_density.time_dependent_log_prob,
    num_samples=2048,
    ts=ts,
    sample_fn=path_density.sample_initial,
    num_steps=10,
    integration_steps=10,
    eta=0.01,
    rejection_sampling=True,
    ess_threshold=0.6,
    # incremental_log_delta=path_density.incremental_log_delta,
)

cov = samples["covariances"]
key, subkey = jax.random.split(key)
samples = generate_samples_with_smc(
    key=key,
    time_dependent_log_density=path_density.time_dependent_log_prob,
    num_samples=2048,
    ts=ts,
    sample_fn=path_density.sample_initial,
    num_steps=10,
    integration_steps=10,
    eta=0.01,
    rejection_sampling=True,
    ess_threshold=0.6,
    covariances=cov,
)
fig = target_density.visualise(samples["positions"][-1])
# plt.show()
plt.savefig("lj13c.png")
