import argparse
import numpy as np
import cmdstanpy
from pathlib import Path
import matplotlib.pyplot as plt # Added for plotting

# Constants
N_PARTICLES = 13
N_SPATIAL_DIM = 3
D = N_PARTICLES * N_SPATIAL_DIM

# --- Helper functions for Visualization ---

def lennard_jones_potential(r_sq, sigma_sq=1.0, epsilon=1.0):
    """Computes the LJ potential given squared distance."""
    # Add small epsilon to avoid division by zero if r_sq is exactly 0
    r_sq_safe = r_sq + 1e-12
    r6_inv = (sigma_sq / r_sq_safe)**3
    r12_inv = r6_inv**2
    return epsilon * (r12_inv - 2.0 * r6_inv)

def harmonic_potential(x_flat, n_particles, n_spatial_dim, c=0.5):
    """Computes the harmonic potential."""
    x = x_flat.reshape(n_particles, n_spatial_dim)
    x_com = np.mean(x, axis=0, keepdims=True)
    displacements = x - x_com
    sq_distances_to_com = np.sum(displacements**2, axis=-1)
    return 0.5 * c * np.sum(sq_distances_to_com)

def compute_total_energy(x_flat, n_particles, n_spatial_dim, c=0.5, sigma_sq=1.0, epsilon=1.0):
    """Computes the total LJ + Harmonic energy for a single configuration."""
    x = x_flat.reshape(n_particles, n_spatial_dim)
    total_lj_energy = 0.0
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            disp = x[i] - x[j]
            r_sq = np.sum(disp**2)
            # Multiply by 2.0 to match the modified Stan model (repeat=True logic)
            total_lj_energy += 2.0 * lennard_jones_potential(r_sq, sigma_sq, epsilon)

    harm_energy = harmonic_potential(x_flat, n_particles, n_spatial_dim, c)
    return total_lj_energy + harm_energy

def batch_compute_total_energy(samples, n_particles, n_spatial_dim, c=0.5):
    """Computes total energy for a batch of samples."""
    energies = np.array([
        compute_total_energy(sample, n_particles, n_spatial_dim, c)
        for sample in samples
    ])
    return energies

def batch_compute_distances(samples, n_particles, n_spatial_dim):
    """Computes all unique pairwise distances for a batch of samples using NumPy."""
    all_distances_list = []
    num_samples = samples.shape[0]
    num_pairs = n_particles * (n_particles - 1) // 2
    all_distances = np.zeros((num_samples, num_pairs)) # Preallocate array

    for idx, sample in enumerate(samples):
        x = sample.reshape(n_particles, n_spatial_dim)
        pair_idx = 0
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                diff = x[i] - x[j]
                dist = np.sqrt(np.sum(diff**2))
                all_distances[idx, pair_idx] = dist
                pair_idx += 1

    # Flatten the array for the histogram as before
    return all_distances.flatten()


def visualize_samples(positions, n_particles, n_spatial_dim, output_filename="stan_samples_visualization.png"):
    """Creates histograms of interatomic distances and energies."""
    print("Calculating distances and energies for visualization...")
    # Calculate distances
    distances = batch_compute_distances(positions, n_particles, n_spatial_dim)

    # Calculate energies (using the exact potential from the Stan model)
    energies = batch_compute_total_energy(positions, n_particles, n_spatial_dim)

    print("Plotting histograms...")
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Distance histogram
    axs[0].hist(distances, bins=100, density=True, alpha=0.7, label="Stan Samples")
    axs[0].set_xlabel("Interatomic distance")
    axs[0].set_ylabel("Density")
    axs[0].set_title("Distribution of Interatomic Distances")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # Energy histogram
    # Determine reasonable range, might need adjustment
    energy_range = (np.min(energies) - 5, np.max(energies) + 5)
    axs[1].hist(energies, bins=100, density=True, alpha=0.7, label="Stan Samples", range=energy_range)
    axs[1].set_xlabel("Total Energy (LJ + Harmonic)")
    axs[1].set_ylabel("Density")
    axs[1].set_title("Distribution of Energy Values")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    print(f"Saving visualization to {output_filename}...")
    plt.savefig(output_filename)
    print("Visualization saved.")
    plt.close(fig) # Close the plot to free memory


# --- Main Function ---

def main(args):
    """Compiles and runs the Stan sampler for LJ13 + Harmonic potential."""
    stan_file = Path(__file__).parent / "lj13_harmonic.stan"
    if not stan_file.exists():
        print(f"Error: Stan file not found at {stan_file}")
        return

    print(f"Using Stan file: {stan_file}")

    # Ensure output directory exists
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {output_path}")

    # Load and compile the Stan model
    # cmdstanpy will automatically compile if needed
    print("Loading Stan model...")
    try:
        model = cmdstanpy.CmdStanModel(stan_file=str(stan_file))
        print("Model loaded successfully.")
        print(f"CmdStan path: {cmdstanpy.cmdstan_path()}")
        print(f"Model name: {model.name}")
    except Exception as e:
        print(f"Error loading/compiling Stan model: {e}")
        print("Please ensure CmdStan is installed and configured correctly.")
        print("See: https://mc-stan.org/cmdstanpy/installation.html")
        return

    # Prepare data for Stan
    stan_data = {
        "N": N_PARTICLES,
        "n_spatial_dim": N_SPATIAL_DIM,
        "D": D,
    }
    print(f"Data for Stan: {stan_data}")

    # Run the NUTS sampler
    print(
        f"Running NUTS sampler with {args.num_chains} chains, "
        f"{args.num_samples} samples, {args.num_warmup} warmup steps..."
    )
    try:
        fit = model.sample(
            data=stan_data,
            seed=args.seed,
            chains=args.num_chains,
            iter_sampling=args.num_samples,
            iter_warmup=args.num_warmup,
            thin=args.thinning,
            show_progress=True,
            # Sampler control parameters
            max_treedepth=args.max_depth,
            # adapt_delta=0.9, # Optional: Adjust adapt_delta if divergences persist
        )
        print("Sampling completed.")
    except Exception as e:
        print(f"Error during Stan sampling: {e}")
        return

    # Print summary and diagnostics
    print("\nSampler Summary:")
    print(fit.summary())
    print("\nSampler Diagnostics:")
    print(fit.diagnose())

    # Extract samples for the 'x' parameter
    # Shape will be (num_chains * num_samples, D)
    position_samples = fit.stan_variable("x")
    print(f"\nExtracted samples shape: {position_samples.shape}")

    # Save samples to NPZ file
    print(f"Saving samples to {output_path}...")
    np.savez(output_path, positions=position_samples)
    print("Samples saved successfully.")

    # --- Visualization ---
    if args.plot:
        plot_filename = output_path.stem + "_visualization.png" # e.g., lj13_harmonic_stan_samples_visualization.png
        visualize_samples(
            positions=position_samples,
            n_particles=N_PARTICLES,
            n_spatial_dim=N_SPATIAL_DIM,
            output_filename=plot_filename
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample LJ13+Harmonic potential using Stan (cmdstanpy)."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/lj13_stan_samples.npz",
        help="Path to save the output samples (NPZ format).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of post-warmup samples per chain.",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=1000,
        help="Number of warmup (adaptation) samples per chain.",
    )
    parser.add_argument(
        "--num-chains",
        type=int,
        default=4,
        help="Number of independent Markov chains.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for the sampler."
    )
    parser.add_argument(
        "--thinning",
        type=int,
        default=1,
        help="Period for saving samples. Saves every Nth iteration.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=10, # Stan's default max_depth
        help="Maximum NUTS tree depth.",
    )
    parser.add_argument(
        "--plot",
        action="store_true", # Flag, doesn't take a value
        help="If set, generate and save histograms of distances and energies.",
    )

    args = parser.parse_args()
    main(args)