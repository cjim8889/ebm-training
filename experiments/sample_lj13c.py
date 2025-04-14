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

jax.config.update("jax_platform_name", "cpu")

key = jax.random.PRNGKey(8888)

target_density = QuadraticSmoothedLJ(
    dim=39,
    n_particles=13,
    include_harmonic=True,
    r_min=0.01,
    min_dr=1e-6,
    c=1,
    # r_min=0.8,
)

initial_density = MultivariateGaussian(dim=39, sigma=2.)
path_density = AnnealedDistribution(
    initial_density=initial_density,
    target_density=target_density,
)

ts = jnp.linspace(0, 1, 128)
print("Setup done")

# --- Load Adaptive HMC Parameters ---
param_path = "data/lj13_adaptive_smc_params.npz"
try:
    # Load parameters and convert to a standard dict if needed (np.load returns NpzFile)
    # JAX/Equinox usually handles dict-like structures, but explicit dict is safer.
    loaded_params = np.load(param_path)
    adaptive_params = {key: loaded_params[key] for key in loaded_params.files}
    print(f"Loaded adaptive HMC parameters from {param_path}")
    # Optional: Convert numpy arrays back to JAX arrays if necessary for downstream JAX functions
    adaptive_params = jax.tree.map(jnp.asarray, adaptive_params)
except FileNotFoundError:
    print(f"Warning: Adaptive parameters file not found at {param_path}. Running without adaptive parameters.")
    adaptive_params = None
except Exception as e:
    print(f"Error loading adaptive parameters: {e}. Running without adaptive parameters.")
    adaptive_params = None
# ------------------------------------
adaptive_params = None

keys = jax.random.split(key, 2) # Only need 2 keys now
key = keys[0]
subkey = keys[1]


initial_samples = path_density.sample_initial(key, (10240,))
print("Starting SMC sampling...")
samples = generate_samples_with_smc(
    key=subkey,
    initial_samples=initial_samples,
    time_dependent_log_density=path_density.time_dependent_log_prob,
    ts=ts,
    num_mcmc_steps=15,      # Renamed from num_steps
    integration_steps=10,   # Kept as default/fallback if adaptive params fail
    eta=0.02,               # Kept as default/fallback if adaptive params fail
    ess_threshold=0.6,
    hmc_parameters=adaptive_params, # Pass loaded parameters
    # incremental_delta=path_density.incremental_log_delta,
    # estimate_covariance=False, # Removed argument
)
print("Sampling done")
# Access the final ESS value
final_ess = samples["ess"][-1]
print(f"Final ESS percentage: {final_ess:.4f}")

# --- Visualization and Saving ---
print("Visualizing final samples...")
final_positions = samples["positions"][-1]
fig = target_density.visualise(final_positions)

plt.show()
# Save the samples to a local file
save_path = "data/lj13c_smc_regularized_samples_2.npz" # Changed filename slightly
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
fig_save_path = "lj13c_smc_regularized_samples_2.png" # Changed filename slightly
plt.savefig(fig_save_path)
print(f"Figure saved to {fig_save_path}")
plt.close(fig) # Close the plot to free memory
