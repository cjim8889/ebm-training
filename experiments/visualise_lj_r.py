import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from distributions import AnnealedDistribution
from distributions.multivariate_gaussian import MultivariateGaussian
from distributions.time_dependent_lennard_jones_butler import (
    TimeDependentLennardJonesEnergyButler,
)

jax.config.update("jax_platform_name", "cpu")

key = jax.random.PRNGKey(12391)

target_density = TimeDependentLennardJonesEnergyButler(
    dim=1,
    n_particles=1,
    alpha=0.2,
    sigma=1.0,
    epsilon_val=1.0,
    min_dr=1e-4,
    n=1,
    m=1,
    c=0.5,
    include_harmonic=True,
    cubic_spline=True,
)


r = jnp.linspace(0.25, 2, 1000).reshape(
    1000, 1
)  # Start from 0.1 to avoid division by zero
potential = jax.vmap(
    lambda x: target_density.soft_core_lennard_jones_potential(x, 1.0)
)(r)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(r, potential, "b-", linewidth=2)
plt.grid(True, linestyle="--", alpha=0.7)
plt.xlabel("Pairwise Distance (r)", fontsize=12)
plt.ylabel("Potential Energy", fontsize=12)
plt.title("Soft-Core Lennard-Jones Potential", fontsize=14)

# Add axis lines
plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
plt.axvline(x=0, color="k", linestyle="-", alpha=0.3)

# Adjust layout and display
plt.tight_layout()
plt.show()
