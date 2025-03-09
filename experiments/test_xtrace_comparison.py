import jax
import jax.numpy as jnp
import jmp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.models.transformer_v3 import ParticleTransformerV3
from src.utils.distributions import (
    divergence_velocity_with_shortcut,
    hutchinson_divergence_velocity2,
)
from src.utils.hutchpp import divergence_velocity_hutchpp
from src.utils.xtrace import divergence_velocity_xtrace

# Set up policy and model
policy = jmp.Policy(
    param_dtype=jnp.float32,
    compute_dtype=jnp.float32,
    output_dtype=jnp.float32,
)

key = jax.random.PRNGKey(12345)
mlp = ParticleTransformerV3(
    n_particles=13,
    n_spatial_dim=3,
    hidden_size=128,
    num_layers=6,
    num_heads=4,
    dropout_rate=0.1,
    attn_dropout_rate=0.1,
    key=key,
    shortcut=True,
    mp_policy=policy,
)

# Parameters
batch_size = 1024  # Total number of positions available
B = 500  # Number of positions to use for comparison
n_probes = 20
t = jnp.array(0.5)
sigma = jnp.array(0.125)

# Generate random positions (each of dimension 39)
key, subkey = jax.random.split(key)
pos_batch = jax.random.normal(subkey, (batch_size, 39))
positions = pos_batch[:B]  # Use first B positions

# Vectorize the ground truth, X-Trace, and Hutchinson++ estimators:
v_ground_truth = jax.vmap(lambda pos: divergence_velocity_with_shortcut(mlp, pos, t, d=sigma))
v_xtrace = jax.vmap(lambda pos: divergence_velocity_xtrace(mlp, pos, t, n_probes=n_probes, d=sigma)[0])
v_hutchpp = jax.vmap(lambda pos: divergence_velocity_hutchpp(mlp, pos, t, d=sigma, n_probes=n_probes)[0])

# For the Hutchinson estimator, we need to generate a new eps for each sample.
def hutchinson_fn(pos, key):
    eps = jax.random.rademacher(key, (n_probes, 39), dtype=jnp.float32)
    return hutchinson_divergence_velocity2(mlp, pos, t, eps, d=sigma)[0]

# Generate B keys for the Hutchinson estimator
key, subkey = jax.random.split(key)
hutchinson_keys = jax.random.split(subkey, B)
v_hutchinson = jax.vmap(hutchinson_fn, in_axes=(0, 0))

# Run the vectorized estimators over the B positions:
ground_truths = v_ground_truth(positions)
xtrace_estimates = v_xtrace(positions)
hutchpp_estimates = v_hutchpp(positions)
hutchinson_estimates = v_hutchinson(positions, hutchinson_keys)

# Convert from JAX arrays to numpy arrays for statistics/plotting
ground_truths_np = np.array(ground_truths)
xtrace_estimates_np = np.array(xtrace_estimates)
hutchpp_estimates_np = np.array(hutchpp_estimates)
hutchinson_estimates_np = np.array(hutchinson_estimates)

# Calculate absolute errors
xtrace_errors_np = np.abs(xtrace_estimates_np - ground_truths_np)
hutchpp_errors_np = np.abs(hutchpp_estimates_np - ground_truths_np)
hutchinson_errors_np = np.abs(hutchinson_estimates_np - ground_truths_np)

# Calculate statistics for each estimator
estimators = ["X-Trace", "Hutchinson++", "Hutchinson"]
mean_errors = [
    np.mean(xtrace_errors_np),
    np.mean(hutchpp_errors_np),
    np.mean(hutchinson_errors_np)
]
error_variances = [
    np.var(xtrace_errors_np),
    np.var(hutchpp_errors_np),
    np.var(hutchinson_errors_np)
]
estimate_variances = [
    np.var(xtrace_estimates_np),
    np.var(hutchpp_estimates_np),
    np.var(hutchinson_estimates_np)
]

# Print statistics
print("\nResults:")
print(f"{'Estimator':<15} {'Mean Error':<15} {'Error Variance':<15} {'Estimate Variance':<15}")
print("-" * 60)
for i, estimator in enumerate(estimators):
    print(f"{estimator:<15} {mean_errors[i]:<15.6f} {error_variances[i]:<15.6f} {estimate_variances[i]:<15.6f}")

# Plotting
plt.figure(figsize=(15, 10))

# Plot 1: Mean Absolute Error
plt.subplot(2, 2, 1)
plt.bar(estimators, mean_errors)
plt.title('Mean Absolute Error')
plt.ylabel('Error')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Plot 2: Error Variance
plt.subplot(2, 2, 2)
plt.bar(estimators, error_variances)
plt.title('Error Variance')
plt.ylabel('Variance')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Plot 3: Estimate Variance
plt.subplot(2, 2, 3)
plt.bar(estimators, estimate_variances)
plt.title('Estimate Variance')
plt.ylabel('Variance')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Plot 4: Error Distribution via Boxplot
plt.subplot(2, 2, 4)
plt.boxplot([xtrace_errors_np, hutchpp_errors_np, hutchinson_errors_np], labels=estimators)
plt.title('Error Distribution')
plt.ylabel('Absolute Error')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('estimator_comparison.png')
plt.show()

print("\nPlot saved as 'estimator_comparison.png'")
