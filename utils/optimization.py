import jax.numpy as jnp
import optax


def inverse_power_schedule(T=64, end_time=1.0, gamma=0.5):
    x_pow = jnp.linspace(0, end_time, T)
    t_pow = 1 - x_pow**gamma
    return jnp.flip(t_pow)


def power_schedule(T=64, end_time=1.0, gamma=0.25):
    x_pow = jnp.linspace(0, end_time, T)
    t_pow = x_pow**gamma
    return t_pow


def soft_clip(x, min_val, max_val, alpha=10.0):
    return min_val + (max_val - min_val) / (1 + jnp.exp(-alpha * (x - min_val)))


def get_optimizer(name: str, learning_rate: float) -> optax.GradientTransformation:
    """Creates optimizer based on name and learning rate.

    Args:
        name: Name of optimizer ('adam', 'adamw', 'sgd', 'rmsprop')
        learning_rate: Learning rate for the optimizer

    Returns:
        optax.GradientTransformation: The configured optimizer
    """
    if name == "adam":
        return optax.adam(learning_rate)
    elif name == "adamw":
        return optax.adamw(learning_rate)
    elif name == "sgd":
        return optax.sgd(learning_rate)
    elif name == "rmsprop":
        return optax.rmsprop(learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {name}")
