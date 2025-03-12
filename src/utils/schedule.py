import jax.numpy as jnp
import optax

def constant_then_cyclic_cosine_schedule(
    constant_value: float,
    initial_steps: int,
    cycle_steps: int,
    peak_value: float,
    end_value: float = 1e-06,
    exponent: float = 1.0,
) -> optax.Schedule:
    """
    Returns an optax schedule that outputs a constant learning rate for the first
    `initial_steps` steps and then follows a cyclic cosine schedule for subsequent steps.

    Args:
        constant_value: Learning rate during the initial constant phase.
        initial_steps: Number of steps for which the learning rate remains constant.
        cycle_steps: Number of steps in each cosine cycle.
        peak_value: Maximum learning rate at the start of each cycle.
        end_value: Minimum learning rate at the end of each cycle (default: 0.0).
        exponent: Exponent to modify the cosine curve shape. The cosine factor becomes
                  (0.5 * (1 + cos(pi * t / cycle_steps))) ** exponent.
    
    Returns:
        A function mapping a training step (int) to a learning rate (float).
    """
    def schedule_fn(step: int) -> jnp.ndarray:
        # Convert step to a JAX array to ensure compatibility with JIT compilation.
        step = jnp.asarray(step, dtype=jnp.int32)
        # For steps >= initial_steps, compute the cyclic cosine schedule.
        # The cycle phase is computed using modulus operation.
        cycle_phase = (step - initial_steps) % cycle_steps
        cosine_factor = 0.5 * (1 + jnp.cos(jnp.pi * cycle_phase / cycle_steps))
        cosine_factor = cosine_factor ** exponent
        lr_cyclic = end_value + (peak_value - end_value) * cosine_factor

        # Use constant_value for the initial phase.
        lr = jnp.where(step < initial_steps, constant_value, lr_cyclic)
        return lr

    return schedule_fn
