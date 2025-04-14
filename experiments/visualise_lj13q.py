import jax.numpy as jnp
import jax.random as random

import matplotlib.pyplot as plt

from src.distributions import QuadraticSmoothedLJ

key = random.PRNGKey(42)  # Seed for reproducibility

target_density = QuadraticSmoothedLJ(
    dim=39,
    n_particles=13,
    include_harmonic=True,
    min_dr=1e-6,
    # r_min=0.8,
)

# Load the three types of samples
gt_samples = jnp.load("data/gt_lj13_samples.npz")["positions"]
efm_samples_full = jnp.load("data/efm_LJ13_all.npy")
n_efm_total = efm_samples_full.shape[0]
n_efm_subsample = 400_000
key, subkey = random.split(key)
efm_samples = random.choice(subkey, efm_samples_full, (n_efm_subsample,), replace=False)
del efm_samples_full # Free up memory

idem_samples = jnp.load("data/idem_lj13_samples.npz")["positions"]
nfs_samples = jnp.load("data/lj13q_samples_128_steps_trajectory.npz")["positions"][-1]
stan_samples = jnp.load("data/lj13_stan_samples.npz")["positions"]
smc_regularized_samples = jnp.load("data/lj13c_smc_regularized_samples.npz")["positions"]

print(f"EFM samples shape: {efm_samples.shape}")
print(f"GT samples shape: {gt_samples.shape}")
print(f"IDEM samples shape: {idem_samples.shape}")
print(f"NFS samples shape: {nfs_samples.shape}")
print(f"Stan samples shape: {stan_samples.shape}")

# Calculate interatomic distances for all sample types
dist_gt = target_density.interatomic_dist(gt_samples)
dist_idem = target_density.interatomic_dist(idem_samples)
dist_nfs = target_density.interatomic_dist(nfs_samples)
dist_stan = target_density.interatomic_dist(stan_samples)
dist_efm = target_density.interatomic_dist(efm_samples)  # Added for EFM
dist_smc_regularized = target_density.interatomic_dist(smc_regularized_samples)

# Calculate energies for all sample types
energy_gt = -target_density.batched_log_prob(gt_samples)
energy_idem = -target_density.batched_log_prob(idem_samples)
energy_nfs = -target_density.batched_log_prob(nfs_samples)
energy_stan = -target_density.batched_log_prob(stan_samples)
energy_efm = -target_density.batched_log_prob(efm_samples)  # Added for EFM
energy_smc_regularized = -target_density.batched_log_prob(smc_regularized_samples)

# Create subplots for visualization
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot interatomic distances
axs[0].hist(
    dist_gt.flatten(),
    bins=100,
    alpha=0.5,
    density=True,
    histtype="step",
    linewidth=2,
    label="IDEM Provided Ground Truth",
    color="blue"
)
axs[0].hist(
    dist_idem.flatten(),
    bins=100,
    alpha=0.5,
    density=True,
    histtype="step",
    linewidth=2,
    label="IDEM",
    color="green"
)
axs[0].hist(
    dist_nfs.flatten(),
    bins=100,
    alpha=0.5,
    density=True,
    histtype="step",
    linewidth=2,
    label="Previous NFS",
    color="red"
)
axs[0].hist(
    dist_stan.flatten(),
    bins=100,
    alpha=0.5,
    density=True,
    histtype="step",
    linewidth=2,
    label="Stan",
    color="orange"
)
axs[0].hist(  # Added for EFM
    dist_efm.flatten(),
    bins=100,
    alpha=0.5,
    density=True,
    histtype="step",
    linewidth=2,
    label="EFM (Subsampled)",
    color="purple"
)
axs[0].hist(
    dist_smc_regularized.flatten(),
    bins=100,
    alpha=0.5,
    density=True,
    histtype="step",
    linewidth=2,
    label="SMC Regularized",
    color="cyan"
)
axs[0].set_xlabel("Interatomic distance")
axs[0].set_ylabel("Density")
axs[0].set_title("Distribution of Interatomic Distances")
axs[0].legend()

# Plot energies
axs[1].hist(
    energy_gt,
    bins=100,
    density=True,
    alpha=0.5,
    range=(-65, 0),
    histtype="step",
    linewidth=2,
    label="IDEM Provided Ground Truth",
    color="blue"
)
axs[1].hist(
    energy_idem,
    bins=100,
    density=True,
    alpha=0.5,
    range=(-65, 0),
    histtype="step",
    linewidth=2,
    label="IDEM",
    color="green"
)
axs[1].hist(
    energy_nfs,
    bins=100,
    density=True,
    alpha=0.5,
    range=(-65, 0),
    histtype="step",
    linewidth=2,
    label="Previous NFS",
    color="red"
)
axs[1].hist(
    energy_stan,
    bins=100,
    density=True,
    alpha=0.5,
    range=(-65, 0),
    histtype="step",
    linewidth=2,
    label="Stan",
    color="orange"
)
axs[1].hist(  # Added for EFM
    energy_efm,
    bins=100,
    density=True,
    alpha=0.5,
    range=(-65, 0),  # Assuming same range is appropriate
    histtype="step",
    linewidth=2,
    label="EFM (Subsampled)",
    color="purple"
)
axs[1].hist(
    energy_smc_regularized,
    bins=100,
    density=True,
    alpha=0.5,
    range=(-65, 0), # Assuming same range is appropriate
    histtype="step",
    linewidth=2,
    label="SMC Regularized",
    color="cyan"
)
axs[1].set_xlabel("Energy")
axs[1].set_ylabel("Density")
axs[1].set_title("Distribution of Energy Values")
axs[1].legend()

# Add a main title
fig.suptitle("Comparison of Sample Distributions", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the title

# Save the figure
plt.savefig("lj13q_comparison.png", dpi=300)

# Show the plot
plt.show()