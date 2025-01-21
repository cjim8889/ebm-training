from typing import Callable, Optional, Tuple
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


def get_fully_connected_senders_receivers(
    n_node: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute senders and receivers for fully connected graph."""
    # Create all pairs of indices more efficiently using a single arange
    idx = jnp.arange(n_node)
    # Use broadcasting to avoid meshgrid
    senders = jnp.repeat(idx, n_node - 1)
    receivers = jnp.concatenate(
        [jnp.concatenate([idx[:i], idx[i + 1 :]]) for i in range(n_node)]
    )
    return senders, receivers


def segment_sum(
    data: jnp.ndarray, segment_ids: jnp.ndarray, num_segments: int
) -> jnp.ndarray:
    """Sum segments of an array based on segment indices."""
    return jax.ops.segment_sum(data, segment_ids, num_segments)


def segment_mean(
    data: jnp.ndarray, segment_ids: jnp.ndarray, num_segments: int
) -> jnp.ndarray:
    seg_sum = jax.ops.segment_sum(data, segment_ids, num_segments)
    seg_count = jax.ops.segment_sum(jnp.ones_like(data), segment_ids, num_segments)
    seg_count = jnp.maximum(seg_count, 1)  # Avoid 0 division
    return seg_sum / seg_count


class EGNNLayer(eqx.Module):
    pos_mlp: eqx.nn.MLP
    pos_aggregate_fn: Callable
    normalize: bool
    eps: float
    dt: float
    senders: jnp.ndarray
    receivers: jnp.ndarray
    n_node: int

    def __init__(
        self,
        n_node: int,
        hidden_size: int,
        key: jax.random.PRNGKey,
        normalize: bool = False,
        tanh: bool = False,
        dt: float = 0.001,
        eps: float = 1e-8,
    ):
        self.dt = dt
        self.n_node = n_node
        self.normalize = normalize
        self.eps = eps
        self.pos_aggregate_fn = segment_mean

        # Precompute senders and receivers for the fixed number of nodes
        self.senders, self.receivers = get_fully_connected_senders_receivers(n_node)

        # Position update network - takes radial distance and time as input
        self.pos_mlp = eqx.nn.MLP(
            in_size=2,  # radial distance and time
            out_size=1,
            width_size=hidden_size,
            depth=1,
            activation=jax.nn.silu,
            final_activation=jax.nn.tanh if tanh else lambda x: x,
            key=key,
        )
        self.pos_mlp = init_linear_weights(self.pos_mlp, xavier_init, key, scale=dt)

    def _pos_update(
        self,
        radial: jnp.ndarray,
        coord_diff: jnp.ndarray,
        t: float,
    ) -> jnp.ndarray:
        edge_features = jnp.concatenate([radial, jnp.full_like(radial, t)], axis=-1)
        edge_scalars = jax.vmap(self.pos_mlp)(edge_features)
        trans = jnp.clip(coord_diff * edge_scalars, -100.0, 100.0)
        return self.pos_aggregate_fn(trans, self.senders, self.n_node)

    def _coord2radial(self, coord: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        coord_diff = coord[self.senders] - coord[self.receivers]
        radial = jnp.sum(coord_diff**2, axis=1, keepdims=True)
        if self.normalize:
            norm = jnp.sqrt(radial + self.eps)
            coord_diff = coord_diff / norm
        return radial, coord_diff

    def __call__(
        self,
        pos: jnp.ndarray,
        t: float,
    ) -> jnp.ndarray:
        # Compute radial features and coordinate differences
        radial, coord_diff = self._coord2radial(pos)

        # Update positions using both radial features and time
        pos_update = self._pos_update(radial, coord_diff, t)
        new_pos = pos + pos_update

        return new_pos


class EGNN(eqx.Module):
    hidden_size: int
    num_layers: int
    layers: list
    normalize: bool
    tanh: bool
    n_node: int

    def __init__(
        self,
        n_node: int,
        hidden_size: int,
        key: jax.random.PRNGKey,
        num_layers: int = 4,
        normalize: bool = False,
        tanh: bool = False,
    ):
        keys = jax.random.split(key, num_layers)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.normalize = normalize
        self.tanh = tanh
        self.n_node = n_node

        # EGNN layers
        self.layers = [
            EGNNLayer(
                n_node=n_node,
                hidden_size=hidden_size,
                key=keys[i],
                normalize=normalize,
                tanh=tanh,
            )
            for i in range(num_layers)
        ]

    def __call__(self, pos: jnp.ndarray, t: float) -> jnp.ndarray:
        pos = pos.reshape(self.n_node, -1)

        # Message passing
        for layer in self.layers:
            pos = layer(pos, t)

        pos = pos.reshape(-1)
        return pos
