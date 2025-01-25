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


def get_optimizer(
    name: str,
    learning_rate: float | optax.Schedule,
    weight_decay: float = 0.0,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    momentum: float = 0.9,
    nesterov: bool = False,
    **kwargs,
) -> optax.GradientTransformation:
    """Creates optimizer based on name and parameters.

    Args:
        name: Name of optimizer ('adam', 'adamw', 'sgd', 'rmsprop', 'adafactor', 'adagrad', 'adadelta', 'lamb', 'lion', 'adamax', 'fromage', 'noisy_sgd')
        learning_rate: Learning rate for the optimizer
        weight_decay: Weight decay coefficient (L2 regularization)
        b1: First moment decay rate (for Adam-like optimizers)
        b2: Second moment decay rate (for Adam-like optimizers)
        eps: Small constant for numerical stability
        momentum: Momentum coefficient for SGD
        nesterov: Whether to use Nesterov momentum
        **kwargs: Additional optimizer-specific parameters

    Returns:
        optax.GradientTransformation: The configured optimizer
    """
    if weight_decay > 0 and name not in ["adamw", "lamb"]:
        # Add weight decay as a separate transformation for optimizers that don't include it
        base = None
        if name == "adam":
            base = optax.adam(learning_rate=learning_rate, b1=b1, b2=b2, eps=eps)
        elif name == "sgd":
            base = optax.sgd(
                learning_rate=learning_rate, momentum=momentum, nesterov=nesterov
            )
        elif name == "rmsprop":
            base = optax.rmsprop(
                learning_rate=learning_rate,
                decay=b1,
                eps=eps,
                momentum=momentum,
                nesterov=nesterov,
            )
        elif name == "adafactor":
            base = optax.adafactor(learning_rate=learning_rate)
        elif name == "adagrad":
            base = optax.adagrad(learning_rate=learning_rate, eps=eps)
        elif name == "adadelta":
            base = optax.adadelta(learning_rate=learning_rate, eps=eps)
        elif name == "lion":
            base = optax.lion(learning_rate=learning_rate, b1=b1, b2=b2)
        elif name == "adamax":
            base = optax.adamax(learning_rate=learning_rate, b1=b1, b2=b2, eps=eps)
        elif name == "fromage":
            base = optax.fromage(learning_rate=learning_rate)
        elif name == "noisy_sgd":
            base = optax.noisy_sgd(
                learning_rate=learning_rate, eta=kwargs.get("noise_scale", 0.01)
            )

        if base is not None:
            return optax.chain(optax.add_decayed_weights(weight_decay), base)

    # Optimizers with built-in weight decay or no weight decay needed
    if name == "adamw":
        return optax.adamw(
            learning_rate=learning_rate,
            b1=b1,
            b2=b2,
            eps=eps,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    elif name == "lamb":
        return optax.lamb(
            learning_rate=learning_rate,
            b1=b1,
            b2=b2,
            eps=eps,
            weight_decay=weight_decay,
        )
    elif name == "adam":
        return optax.adam(learning_rate=learning_rate, b1=b1, b2=b2, eps=eps)
    elif name == "sgd":
        return optax.sgd(
            learning_rate=learning_rate, momentum=momentum, nesterov=nesterov
        )
    elif name == "rmsprop":
        return optax.rmsprop(
            learning_rate=learning_rate,
            decay=b1,
            eps=eps,
            momentum=momentum,
            nesterov=nesterov,
        )
    elif name == "adafactor":
        return optax.adafactor(learning_rate=learning_rate)
    elif name == "adagrad":
        return optax.adagrad(learning_rate=learning_rate, eps=eps)
    elif name == "adadelta":
        return optax.adadelta(learning_rate=learning_rate, eps=eps)
    elif name == "lion":
        return optax.lion(learning_rate=learning_rate, b1=b1, b2=b2)
    elif name == "adamax":
        return optax.adamax(learning_rate=learning_rate, b1=b1, b2=b2, eps=eps)
    elif name == "fromage":
        return optax.fromage(learning_rate=learning_rate)
    elif name == "noisy_sgd":
        return optax.noisy_sgd(
            learning_rate=learning_rate, eta=kwargs.get("noise_scale", 0.01)
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")
