from typing import Callable
import equinox as eqx
import jax
import jax.numpy as jnp


def xavier_init(
    weight: jnp.ndarray, key: jax.random.PRNGKey, scale: float = 1.0
) -> jnp.ndarray:
    """Xavier (Glorot) initialization."""
    out, in_ = weight.shape
    bound = jnp.sqrt(6 / (in_ + out))
    return scale * jax.random.uniform(
        key, shape=(out, in_), minval=-bound, maxval=bound
    )


def init_linear_weights(
    model: eqx.Module, init_fn: Callable, key: jax.random.PRNGKey, scale: float = 1.0
) -> eqx.Module:
    """Initialize weights of all Linear layers in a model using the given initialization function."""
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_weights = lambda m: [
        x.weight
        for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
        if is_linear(x)
    ]
    weights = get_weights(model)
    new_weights = [
        init_fn(weight, subkey, scale)
        for weight, subkey in zip(weights, jax.random.split(key, len(weights)))
    ]
    return eqx.tree_at(get_weights, model, new_weights)
