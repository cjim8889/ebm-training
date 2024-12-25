import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from distributions import AnnealedDistribution
from distributions.multivariate_gaussian import MultivariateGaussian
from distributions.time_dependent_lennard_jones_butler import (
    TimeDependentLennardJonesEnergyButler,
)

# plt.rcParams["figure.dpi"] = 300
# plt.rcParams["figure.figsize"] = [6.0, 4.0]

jax.config.update("jax_platform_name", "cpu")

key = jax.random.PRNGKey(12391)

initial_density = MultivariateGaussian(dim=2, mean=0.0, sigma=5.0)
target_density = TimeDependentLennardJonesEnergyButler(
    dim=2,
    n_particles=2,
    alpha=0.2,
    sigma=1.0,
    epsilon_val=1.0,
    min_dr=1e-6,
    n=1,
    m=1,
    c=0.5,
    include_harmonic=True,
)

path_density = AnnealedDistribution(
    initial_density=initial_density, target_density=target_density
)

# Create position grid
positions = jnp.linspace(-10, 10, 500)
X, Y = jnp.meshgrid(positions, positions)
positions_flat = jnp.stack([X.ravel(), Y.ravel()], axis=1)  # Shape: (2500, 2)

# Define time points
times = jnp.linspace(0, 1, 4)


# Pre-compile log probability functions with JIT
@jax.jit
def compute_log_prob_path(pos_batch, t):
    return jax.vmap(lambda x: path_density.time_dependent_log_prob(x, t))(pos_batch)


@jax.jit
def compute_log_prob_target(pos_batch, t):
    return jax.vmap(lambda x: target_density.time_dependent_log_prob(x, t))(pos_batch)


# Compute distances (Euclidean) for all positions
@jax.jit
def compute_distances(pos_batch):
    # pos_batch has shape (N, 2), representing [x1, x2]
    return jnp.abs(pos_batch[:, 0] - pos_batch[:, 1])


# Prepare the figure with subplots
# Now, 4 rows (Path, Target, Path Log Prob vs Distance, Target Log Prob vs Distance) and len(times) columns
fig, axes = plt.subplots(4, len(times), figsize=(4 * len(times), 20))
plt.subplots_adjust(wspace=0.4, hspace=0.6)

# Iterate over each time point and compute log probabilities in a batched manner
for idx, t in enumerate(times):
    # Compute log probabilities for the current time
    Z_path_flat = compute_log_prob_path(positions_flat, t)  # Shape: (2500,)
    Z_target_flat = compute_log_prob_target(positions_flat, t)  # Shape: (2500,)

    # Reshape the flat arrays back to grid shape for contour plots
    Z_path = Z_path_flat.reshape(X.shape)  # Shape: (50, 50)
    Z_target = Z_target_flat.reshape(X.shape)  # Shape: (50, 50)

    # Plot Path Density in the first row
    c1 = axes[0, idx].contourf(X, Y, Z_path, levels=20, cmap="viridis")
    fig.colorbar(c1, ax=axes[0, idx])
    if idx == 0:
        axes[0, idx].set_ylabel("Path Distribution", fontsize=14)
    axes[0, idx].set_title(f"t = {t:.2f}")
    axes[0, idx].set_xlabel("x₁")
    axes[0, idx].set_ylabel("x₂")

    # Plot Target Density in the second row
    c2 = axes[1, idx].contourf(X, Y, Z_target, levels=20, cmap="viridis")
    fig.colorbar(c2, ax=axes[1, idx])
    if idx == 0:
        axes[1, idx].set_ylabel("Target Distribution", fontsize=14)
    axes[1, idx].set_title(f"t = {t:.2f}")
    axes[1, idx].set_xlabel("x₁")
    axes[1, idx].set_ylabel("x₂")

    # Compute distances
    distances = compute_distances(positions_flat)  # Shape: (2500,)

    # Compute Negative Log Probabilities
    neg_log_prob_path = -Z_path_flat  # Shape: (2500,)
    neg_log_prob_target = -Z_target_flat  # Shape: (2500,)

    # Bin the distances and compute average negative log probabilities per bin
    num_bins = 50
    bin_edges = jnp.linspace(distances.min(), distances.max(), num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Digitize the distances to find bin indices
    bin_indices = jnp.digitize(distances, bin_edges) - 1  # Shape: (2500,)
    bin_indices = jnp.clip(bin_indices, 0, num_bins - 1)

    # Compute sum and count per bin for Path
    sum_neg_log_prob_path = jnp.bincount(
        bin_indices, weights=neg_log_prob_path, minlength=num_bins
    )
    count_path = jnp.bincount(bin_indices, minlength=num_bins)
    mean_neg_log_prob_path = sum_neg_log_prob_path / count_path
    mean_neg_log_prob_path = jnp.where(count_path > 0, mean_neg_log_prob_path, 0)

    # Compute sum and count per bin for Target
    sum_neg_log_prob_target = jnp.bincount(
        bin_indices, weights=neg_log_prob_target, minlength=num_bins
    )
    count_target = jnp.bincount(bin_indices, minlength=num_bins)
    mean_neg_log_prob_target = sum_neg_log_prob_target / count_target
    mean_neg_log_prob_target = jnp.where(count_target > 0, mean_neg_log_prob_target, 0)

    # Plot Path Negative Log Probability vs Distance in the third row
    axes[2, idx].plot(bin_centers, mean_neg_log_prob_path, color="blue", label="Path")
    axes[2, idx].set_xlabel("Distance |x₁ - x₂|")
    axes[2, idx].set_ylabel("-Log Prob")
    if idx == 0:
        axes[2, idx].set_ylabel("Path -Log Prob", fontsize=14)
    axes[2, idx].set_title(f"t = {t:.2f}")
    axes[2, idx].legend()

    # Plot Target Negative Log Probability vs Distance in the fourth row
    axes[3, idx].plot(
        bin_centers, mean_neg_log_prob_target, color="red", label="Target"
    )
    axes[3, idx].set_xlabel("Distance |x₁ - x₂|")
    axes[3, idx].set_ylabel("-Log Prob")
    if idx == 0:
        axes[3, idx].set_ylabel("Target -Log Prob", fontsize=14)
    axes[3, idx].set_title(f"t = {t:.2f}")
    axes[3, idx].legend()

# # Add a main title for the entire figure
# fig.suptitle(
#     "Distributions and Negative Log Probabilities at Various Time Points",
#     fontsize=18,
#     y=0.95,
# )

# Improve layout and display the plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
