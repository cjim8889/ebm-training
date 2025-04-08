import os

import jax
import jax.numpy as jnp
import numpy as np  # Added for loading parameters
import matplotlib.pyplot as plt

from src.distributions import (
    AnnealedDistribution,
    MultivariateGaussian,
    QuadraticSmoothedLJ,
)
# Updated import path/function name if it changed (it didn't visibly, but good practice)
from src.mcmc.smc import generate_samples_with_smc
from src.mcmc.adaptive_smc import generate_samples_with_adaptive_smc

jax.config.update("jax_platform_name", "cpu")

key = jax.random.PRNGKey(8888)

target_density = QuadraticSmoothedLJ(
    dim=39,
    n_particles=13,
    include_harmonic=True,
    r_min=0.01,
    min_dr=1e-6,
    # r_min=0.8,
)

initial_density = MultivariateGaussian(dim=39, sigma=1.0)
path_density = AnnealedDistribution(
    initial_density=initial_density,
    target_density=target_density,
)

ts = jnp.linspace(0, 1, 128)
print("Setup done")

keys = jax.random.split(key, 2) # Only need 2 keys now
key = keys[0]
subkey = keys[1]


initial_samples = path_density.sample_initial(key, (10240,))
print("Starting SMC sampling...")
samples = generate_samples_with_adaptive_smc(
    key=subkey,
    initial_samples=initial_samples,
    time_dependent_log_density=path_density.time_dependent_log_prob,
    t0=0.0,
    max_steps=128,
    mcmc_steps=10,      # Renamed from num_steps
    integration_steps=15,   # Kept as default/fallback if adaptive params fail
    eta=0.01,               # Kept as default/fallback if adaptive params fail
    ess_threshold=0.6,
    incremental_log_delta=path_density.incremental_log_delta,
    # estimate_covariance=False, # Removed argument
)
print("Sampling done")

# --- Visualization and Saving ---
print("Visualizing final samples...")
final_positions = samples["positions"][-1]
fig = target_density.visualise(final_positions)

plt.show()
# Save the samples to a local file
save_path = "data/lj13c_adaptive_smc_samples.npz" # Changed filename slightly
os.makedirs(os.path.dirname(save_path), exist_ok=True)
jnp.savez(
    save_path,
    positions=final_positions,
    weights=samples["weights"][-1], # Save final weights too
    ess=samples["ess"],            # Save ESS history
    times=ts,
)
print(f"Samples saved to {save_path}")

# Save the figure
fig_save_path = "lj13c_adaptive_smc_samples.png" # Changed filename slightly
plt.savefig(fig_save_path)
print(f"Figure saved to {fig_save_path}")
plt.close(fig) # Close the plot to free memory
