from typing import Literal, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

# === Moved Schedule Functions (from src/utils/optimization.py) ===

def inverse_power_schedule(T: int = 64, end_time: float = 1.0, gamma: float = 0.25) -> Float[Array, " T"]:
    """Generates time steps using an inverse power law schedule."""
    x_pow = jnp.linspace(0, end_time, T)
    t_pow = 1 - x_pow**gamma
    return jnp.flip(t_pow)

def power_schedule(T: int = 64, end_time: float = 1.0, gamma: float = 0.25) -> Float[Array, " T"]:
    """Generates time steps using a power law schedule."""
    x_pow = jnp.linspace(0, end_time, T)
    t_pow = x_pow**gamma
    return t_pow

def focus_schedule(T: int = 64, end_time: float = 1.0, gamma: float = 0.6) -> Float[Array, " T*2"]:
    """Generates time steps focusing density near gamma. Note: Output length is 2*T."""
    # Note: Original implementation returns 2*T points. Consider if this is intended.
    # If T points are desired, adjust the linspace counts or slice the output.
    # For now, keeping original behavior.
    num_points_part1 = T // 2 # Example adjustment for T points total
    num_points_part2 = T - num_points_part1
    x_1 = jnp.linspace(0, gamma, num_points_part1)
    x_2 = jnp.linspace(gamma, end_time, num_points_part2)[1:] # Avoid duplicating gamma if T is even
    # Original implementation:
    # x_1 = jnp.linspace(0, gamma, T)
    # x_2 = jnp.linspace(gamma, end_time, T)
    return jnp.concatenate((x_1, x_2))


# === Moved Sampling Function (from src/utils/distributions.py) ===

@eqx.filter_jit
def sample_monotonic_uniform_ordered(
    key: jax.random.PRNGKey, bounds: Float[Array, " time"], include_endpoints: bool = True
) -> Float[Array, " time"]:
    def step(carry, info):
        t_prev = carry
        t_current = info

        return t_current, jnp.array([t_prev, t_current])

    _, ordered_pairs = jax.lax.scan(step, bounds[0], bounds[1:])

    if include_endpoints:
        ordered_pairs = jnp.concatenate(
            [ordered_pairs, jnp.array([[1.0, 1.0]])], axis=0
        )

    samples = jax.random.uniform(
        key, bounds.shape, minval=ordered_pairs[:, 0], maxval=ordered_pairs[:, 1]
    )
    return samples

# === New Helper Functions ===

# Define the Literal type for schedule names, including 'focus'
ScheduleName = Literal["linear", "inverse_power", "power", "focus"]

def setup_time_schedule(
    schedule: ScheduleName,
    num_timesteps: int,
    end_time: float = 1.0,
    gamma: Optional[float] = None # Optional gamma for relevant schedules
) -> Float[Array, " time"]:
    """
    Sets up the base time steps for integration based on the specified schedule.

    Args:
        schedule: The name of the schedule to use ('linear', 'inverse_power', 'power', 'focus').
        num_timesteps: The desired number of time steps (T).
        end_time: The final time point (usually 1.0).
        gamma: The gamma parameter required for 'inverse_power', 'power', and 'focus' schedules.

    Returns:
        An array of time steps. Note: 'focus' might return 2*T points based on original util impl.
    """
    if schedule == "linear":
        base_ts = jnp.linspace(0, end_time, num_timesteps)
    elif schedule == "inverse_power":
        if gamma is None:
            raise ValueError("Gamma must be provided for 'inverse_power' schedule.")
        base_ts = inverse_power_schedule(num_timesteps, end_time=end_time, gamma=gamma)
    elif schedule == "power":
        if gamma is None:
            raise ValueError("Gamma must be provided for 'power' schedule.")
        base_ts = power_schedule(num_timesteps, end_time=end_time, gamma=gamma)
    elif schedule == "focus":
        if gamma is None:
            raise ValueError("Gamma must be provided for 'focus' schedule.")
        # Note: Check if the output size (potentially 2*T) is handled correctly downstream.
        base_ts = focus_schedule(num_timesteps, end_time=end_time, gamma=gamma)
        # If exactly T points are needed for focus, adjust here or in focus_schedule itself.
        # Example: if base_ts.shape[0] != num_timesteps: base_ts = jnp.linspace(0, end_time, num_timesteps) # Fallback?
    else:
        # Should be caught by Literal type hint, but good practice to have runtime check.
        raise ValueError(f"Unknown schedule: {schedule}")

    return base_ts


def sample_continuous_time(
    key: jax.random.PRNGKey, base_ts: Float[Array, " time"]
) -> Float[Array, " time"]:
    """
    Samples new time steps monotonically based on existing base time steps.

    Args:
        key: JAX random key.
        base_ts: The base time steps (sorted array, e.g., from setup_time_schedule).

    Returns:
        A new array of sampled time steps, ordered monotonically, matching the size of base_ts.
    """
    # Assuming include_endpoints=True is the desired behavior based on core.py usage
    return sample_monotonic_uniform_ordered(key, base_ts, include_endpoints=True)