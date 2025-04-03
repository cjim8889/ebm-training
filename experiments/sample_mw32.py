import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Use src prefix consistently based on sample_lj13c.py structure
from src.distributions import (
    AnnealedDistribution,
    ManyWellEnergy,
    MultivariateGaussian, # Added initial distribution
)
from src.mcmc.sampling import generate_samples_with_smc

jax.config.update("jax_platform_name", "cpu") # Added from lj13c

key = jax.random.PRNGKey(1234)

# Target density remains ManyWellEnergy
target_density = ManyWellEnergy(dim=32)

# Define an initial density (standard Gaussian matching the target dimension)
initial_density = MultivariateGaussian(dim=32, mean=0, sigma=2.)

# Define the annealed path using geometric annealing
path_density = AnnealedDistribution(
    initial_density=initial_density,
    target_density=target_density,
)

# Time schedule (same as lj13c)
ts = jnp.linspace(0, 1, 128)

print("Warmup done") # Added from lj13c
keys = jax.random.split(key, 3)
key = keys[0]
subkey = keys[1]
# covariance_key = keys[2] # Not used if estimate_covariance=False

# Sample initial points from the initial density via the path
initial_samples = path_density.sample_initial(subkey, 2560) # Same sample size as original mw32

# Generate samples using SMC (parameters from lj13c)
print("Starting SMC sampling...")
samples = generate_samples_with_smc(
    key=subkey, # Reusing subkey for sampling
    initial_samples=initial_samples,
    time_dependent_log_density=path_density.time_dependent_log_prob,
    ts=ts,
    num_steps=10,
    integration_steps=5,
    eta=0.1,
    ess_threshold=0.5,
    estimate_covariance=False,
)
print("Sampling done") # Added from lj13c
print("ESS", samples["ess"]) # Added from lj13c

# Visualize the final samples obtained from SMC
fig = target_density.visualise(samples["positions"][-1])
plt.savefig("mw32_smc_samples.png") # Save the figure
plt.show() # Optional: uncomment to display the plot interactively
