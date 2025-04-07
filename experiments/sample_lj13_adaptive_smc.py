import os

import blackjax
import jax
import jax.numpy as jnp

from src.distributions.multivariate_gaussian import MultivariateGaussian
from src.distributions.smoothed_lennar_jones import QuadraticSmoothedLJ
from src.distributions.annealed_distribution import AnnealedDistribution

# plt.rcParams["figure.dpi"] = 300
# plt.rcParams["figure.figsize"] = [6.0, 4.0]

jax.config.update("jax_platform_name", "cpu")

key = jax.random.PRNGKey(1234)


initial_density = MultivariateGaussian(dim=39, mean=0, sigma=2.)
target_density = QuadraticSmoothedLJ(
    dim=39,
    n_particles=13,
    include_harmonic=True,
)   

path_density = AnnealedDistribution(
    initial_density=initial_density,
    target_density=target_density,
)

all_step_sizes = []
all_inv_mass_matrices = []
all_num_integration_steps = []

for t in jnp.linspace(0, 1, 128):
    print(f"Running adaptation for t={t:.4f}") # Add progress indicator
    log_density = lambda x: path_density.time_dependent_log_prob(x, t).squeeze()

    key, subkey = jax.random.split(key)
    initial_position = initial_density.sample(subkey, (1,))
    warmup = blackjax.window_adaptation(
        blackjax.hmc,
        log_density,
        num_integration_steps=10,
        initial_step_size=1.0,
        target_acceptance_rate=0.6,
        progress_bar=False, # Disable inner progress bar for cleaner output
    )

    key, warmup_key, sample_key = jax.random.split(key, 3)

    (state, parameters), _ = warmup.run(
        warmup_key,
        initial_position,
        num_steps=10000, # Note: 10000 steps per t might be slow for 128 t values.
    )
    print("HMC Warmup done for t=", t)
    # Store parameters instead of printing individually
    all_step_sizes.append(parameters["step_size"])
    all_inv_mass_matrices.append(parameters["inverse_mass_matrix"])
    all_num_integration_steps.append(parameters["num_integration_steps"])
    print(f"Step size: {parameters['step_size']}")
    print(f"Inverse mass matrix: {parameters['inverse_mass_matrix']}")

# Combine parameters into a single pytree
combined_parameters = {
    "step_size": jnp.stack(all_step_sizes),
    "inverse_mass_matrix": jnp.stack(all_inv_mass_matrices),
    "num_integration_steps": jnp.array(all_num_integration_steps),
}

# Ensure the data directory exists
save_path = "data/lj13_adaptive_smc_params.npz"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Save the combined parameters pytree
jnp.savez(save_path, **combined_parameters)
print(f"\nCombined parameters saved to {save_path}")
