import jax.numpy as jnp

import matplotlib.pyplot as plt

from src.distributions import QuadraticSmoothedLJ


target_density = QuadraticSmoothedLJ(
    dim=39,
    n_particles=13,
    r_min=0.8,
)

# Load the three types of samples
gt_samples = jnp.load("data/gt_lj13_samples.npz")["positions"]
idem_samples = jnp.load("data/idem_lj13_samples.npz")["positions"]
nfs_samples = jnp.load("data/lj13q_samples_128_steps_trajectory.npz")["positions"][-1]

# Calculate interatomic distances for all sample types
dist_gt = target_density.interatomic_dist(gt_samples)
dist_idem = target_density.interatomic_dist(idem_samples)
dist_nfs = target_density.interatomic_dist(nfs_samples)

# Calculate energies for all sample types
energy_gt = -target_density.batched_log_prob(gt_samples)
energy_idem = -target_density.batched_log_prob(idem_samples)
energy_nfs = -target_density.batched_log_prob(nfs_samples)

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
    label="Ground Truth",
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
    label="NFS",
    color="red"
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
    label="Ground Truth",
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
    label="NFS",
    color="red"
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