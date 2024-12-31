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
    TimeDependentLennardJonesEnergyButler,
)

from utils.smc import generate_samples_with_smc

# plt.rcParams["figure.dpi"] = 300
# plt.rcParams["figure.figsize"] = [6.0, 4.0]

jax.config.update("jax_platform_name", "cpu")

key = jax.random.PRNGKey(1234)


initial_density = MultivariateGaussian(dim=39, mean=0, sigma=3)
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
    cubic_spline=True,
    # log_prob_clip=100.0,
)

initial_density = MultivariateGaussian(dim=39, mean=0.0, sigma=1.0)
path_density = AnnealedDistribution(
    initial_density=initial_density, target_density=target_density
)

key, subkey = jax.random.split(key)
ts = jnp.linspace(0, 1.0, num=128)

samples = generate_samples_with_smc(
    key=key,
    time_dependent_log_density=path_density.time_dependent_log_prob,
    num_samples=1000,
    ts=ts,
    sample_fn=path_density.sample_initial,
    num_steps=20,
    integration_steps=10,
    eta=0.02,
    rejection_sampling=True,
    ess_threshold=0.5,
)

fig = target_density.visualise(samples)
plt.show()
