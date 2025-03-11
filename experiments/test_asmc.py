import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.distributions import (
    AnnealedDistribution,
    QuadraticSmoothedLJ,
    TranslationInvariantGaussian,
)
from src.mcmc.adaptive_smc import generate_samples_with_adaptive_smc

jax.config.update("jax_platform_name", "cpu")

key = jax.random.PRNGKey(333)

target_density = QuadraticSmoothedLJ(
    dim=39,
    n_particles=13,
    include_harmonic=True,
    r_min=0.8,
)


initial_density = TranslationInvariantGaussian(N=13, D=3, sigma=2.0)
path_density = AnnealedDistribution(
    initial_density=initial_density,
    target_density=target_density,
    method="linear",
)

print("Warmup done")
keys = jax.random.split(key, 3)
key = keys[0]
subkey = keys[1]
covariance_key = keys[2]


initial_samples = path_density.sample_initial(subkey, 5120)
samples = generate_samples_with_adaptive_smc(
    key=subkey,
    initial_samples=initial_samples,
    time_dependent_log_density=path_density.time_dependent_log_prob,
    incremental_log_delta=path_density.incremental_log_delta,
    t0=0.0,
    max_steps=128,
    mcmc_steps=20,
    integration_steps=10,
    eta=0.02,
    ess_threshold=0.6,
)
print("Sampling done")
print(f"ESS: {samples["diagnostics"]["ess"]}")
print(f"Beta: {samples["diagnostics"]["beta"]}")

print(samples["weights"].shape)

fig = target_density.visualise(samples["positions"][-1])

plt.show()