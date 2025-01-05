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

from utils.smc import generate_samples_with_smc

key = jax.random.PRNGKey(1234)

initial_density = MultivariateGaussian(dim=39, mean=0, sigma=1)
# target_density = TimeDependentLennardJonesEnergyButler(
#     dim=39,
#     n_particles=13,
#     alpha=0.2,
#     sigma=1.0,
#     epsilon_val=1.0,
#     min_dr=1e-4,
#     n=1,
#     m=1,
#     c=0.5,
#     include_harmonic=True,
# )
target_density = SoftCoreLennardJonesEnergy(
    dim=39,
    n_particles=13,
    alpha=0.2,
    sigma=1.0,
    epsilon_val=1.0,
    min_dr=1e-4,
    c=0.5,
    include_harmonic=True,
)

path_density = AnnealedDistribution(
    initial_density=initial_density, target_density=target_density, method="linear"
)
ts = jnp.linspace(0, 1, 128)
print("Warmup done")
keys = jax.random.split(key, 3)
key = keys[0]
subkey = keys[1]
covariance_key = keys[2]

samples = generate_samples_with_smc(
    key=subkey,
    time_dependent_log_density=path_density.time_dependent_log_prob,
    num_samples=2560,
    ts=ts,
    sample_fn=path_density.sample_initial,
    num_steps=20,
    integration_steps=10,
    eta=0.018,
    rejection_sampling=True,
    ess_threshold=0.5,
    estimate_covariance=False,
    blackjax_hmc=True,
)
print("Sampling done")
print("ESS", samples["ess"])
fig = target_density.visualise(samples["positions"][-1])
plt.savefig("lj13c-3.png")
