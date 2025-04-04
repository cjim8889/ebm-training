import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.distributions import (
    AnnealedDistribution,
    QuadraticSmoothedLJ,
    MultivariateGaussian,
)
from src.mcmc.sampling import generate_samples_with_smc

jax.config.update("jax_platform_name", "cpu")

key = jax.random.PRNGKey(1234)

target_density = QuadraticSmoothedLJ(
    dim=55*3,
    n_particles=55,
    r_min=0.8,
    include_harmonic=True,
)


initial_density = MultivariateGaussian(dim=55*3, sigma=2.0)
path_density = AnnealedDistribution(
    initial_density=initial_density,
    target_density=target_density,
)

ts = jnp.linspace(0, 1, 128)
print("Warmup done")
keys = jax.random.split(key, 3)
key = keys[0]
subkey = keys[1]
covariance_key = keys[2]


initial_samples = path_density.sample_initial(subkey, 2560)
samples = generate_samples_with_smc(
    key=subkey,
    initial_samples=initial_samples,
    time_dependent_log_density=path_density.time_dependent_log_prob,
    ts=ts,
    num_steps=10,
    integration_steps=10,
    eta=0.02,
    ess_threshold=0.5,
    estimate_covariance=False,
)
print("Sampling done")
print("ESS", samples["ess"])
fig = target_density.visualise(samples["positions"][-1])
plt.show()
plt.savefig("lj55q.png")
