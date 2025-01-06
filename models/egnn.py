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
    # Create all pairs of indices
    idx = jnp.arange(n_node)
    senders, receivers = jnp.meshgrid(idx, idx)
    # Remove self-connections
    mask = senders != receivers
    senders = senders[mask]
    receivers = receivers[mask]
    return senders, receivers


def segment_sum(
    data: jnp.ndarray, segment_ids: jnp.ndarray, num_segments: int
) -> jnp.ndarray:
    """Sum segments of an array based on segment indices."""
    return jax.ops.segment_sum(data, segment_ids, num_segments)


class EGNNLayer(eqx.Module):
    edge_mlp: eqx.nn.MLP
    node_mlp: eqx.nn.MLP
    pos_mlp: eqx.nn.MLP
    attention_mlp: Optional[eqx.nn.MLP]
    pos_aggregate_fn: Callable
    msg_aggregate_fn: Callable
    residual: bool
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
        output_size: int,
        key: jax.random.PRNGKey,
        blocks: int = 1,
        act_fn: Callable = jax.nn.silu,
        residual: bool = True,
        attention: bool = False,
        normalize: bool = False,
        tanh: bool = False,
        dt: float = 0.001,
        eps: float = 1e-8,
    ):
        keys = jax.random.split(key, 4)
        self.dt = dt
        self.n_node = n_node
        self.residual = residual
        self.normalize = normalize
        self.eps = eps
        self.pos_aggregate_fn = segment_sum
        self.msg_aggregate_fn = segment_sum

        # Precompute senders and receivers for the fixed number of nodes
        self.senders, self.receivers = get_fully_connected_senders_receivers(n_node)

        # Message network
        edge_in_size = hidden_size * 2 + 1  # incoming + outgoing + radial
        self.edge_mlp = eqx.nn.MLP(
            in_size=edge_in_size,
            out_size=hidden_size,
            width_size=hidden_size,
            depth=blocks,
            activation=act_fn,
            final_activation=jax.nn.silu,
            key=keys[0],
        )
        self.edge_mlp = init_linear_weights(self.edge_mlp, xavier_init, keys[0])

        # Update network
        node_in_size = hidden_size * 2  # node features + aggregated messages
        self.node_mlp = eqx.nn.MLP(
            in_size=node_in_size,
            out_size=output_size,
            width_size=hidden_size,
            depth=blocks,
            activation=act_fn,
            key=keys[1],
        )
        self.node_mlp = init_linear_weights(self.node_mlp, xavier_init, keys[1])

        # Position update network
        self.pos_mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=1,
            width_size=hidden_size,
            depth=1,
            activation=act_fn,
            final_activation=jax.nn.tanh if tanh else lambda x: x,
            key=keys[2],
        )
        self.pos_mlp = init_linear_weights(self.pos_mlp, xavier_init, keys[2], scale=dt)

        # Attention network
        self.attention_mlp = None
        if attention:
            self.attention_mlp = eqx.nn.MLP(
                in_size=hidden_size,
                out_size=hidden_size,
                width_size=hidden_size,
                depth=1,
                activation=jax.nn.sigmoid,
                final_activation=jax.nn.sigmoid,
                key=keys[3],
            )
            self.attention_mlp = init_linear_weights(
                self.attention_mlp, xavier_init, keys[3]
            )

    def _pos_update(
        self,
        edges: jnp.ndarray,
        coord_diff: jnp.ndarray,
    ) -> jnp.ndarray:
        trans = coord_diff * jax.vmap(self.pos_mlp)(edges)
        trans = jnp.clip(trans, -100, 100)
        return self.pos_aggregate_fn(trans, self.senders, self.n_node)

    def _message(
        self,
        incoming: jnp.ndarray,
        outgoing: jnp.ndarray,
        radial: jnp.ndarray,
    ) -> jnp.ndarray:
        msg = jnp.concatenate([incoming, outgoing, radial], axis=-1)
        msg = jax.vmap(self.edge_mlp)(msg)
        if self.attention_mlp is not None:
            att = jax.vmap(self.attention_mlp)(msg)
            msg = msg * att
        return msg

    def _update(
        self,
        nodes: jnp.ndarray,
        msg: jnp.ndarray,
        node_attribute: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        x = jnp.concatenate([nodes, msg], axis=-1)
        if node_attribute is not None:
            x = jnp.concatenate([x, node_attribute], axis=-1)
        x = jax.vmap(self.node_mlp)(x)
        if self.residual:
            x = nodes + x
        return x

    def _coord2radial(self, coord: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        coord_diff = coord[self.senders] - coord[self.receivers]
        radial = jnp.sum(coord_diff**2, 1)[:, jnp.newaxis]
        if self.normalize:
            norm = jnp.sqrt(radial)
            coord_diff = coord_diff / (norm + self.eps)
        return radial, coord_diff

    def __call__(
        self,
        nodes: jnp.ndarray,
        pos: jnp.ndarray,
        node_attribute: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Compute radial features and coordinate differences
        radial, coord_diff = self._coord2radial(pos)

        # Compute messages
        incoming_nodes = nodes[self.receivers]
        outgoing_nodes = nodes[self.senders]
        messages = self._message(incoming_nodes, outgoing_nodes, radial)

        # Aggregate messages
        aggregated_messages = self.msg_aggregate_fn(
            messages, self.receivers, self.n_node
        )

        # Update nodes
        new_nodes = self._update(nodes, aggregated_messages, node_attribute)

        # Update positions
        pos_update = self._pos_update(messages, coord_diff)
        new_pos = pos + pos_update

        return new_nodes, new_pos


class EGNN(eqx.Module):
    hidden_size: int
    num_layers: int
    layers: list
    embedding: eqx.nn.Linear
    act_fn: Callable
    residual: bool
    attention: bool
    normalize: bool
    tanh: bool
    n_node: int

    def __init__(
        self,
        n_node: int,
        hidden_size: int,
        key: jax.random.PRNGKey,
        act_fn: Callable = jax.nn.silu,
        num_layers: int = 4,
        residual: bool = True,
        attention: bool = False,
        normalize: bool = False,
        tanh: bool = False,
    ):
        keys = jax.random.split(key, num_layers + 2)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.act_fn = act_fn
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.tanh = tanh
        self.n_node = n_node

        # Input embedding and output readout
        self.embedding = eqx.nn.Linear(1, hidden_size, key=keys[0])
        self.embedding = init_linear_weights(self.embedding, xavier_init, keys[0])

        # EGNN layers
        self.layers = [
            EGNNLayer(
                n_node=n_node,
                hidden_size=hidden_size,
                output_size=hidden_size,
                key=keys[i + 1],
                act_fn=act_fn,
                residual=residual,
                attention=attention,
                normalize=normalize,
                tanh=tanh,
            )
            for i in range(num_layers)
        ]

    def __call__(self, pos: jnp.ndarray, t: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        pos = pos.reshape(self.n_node, -1)
        # Input node embedding
        h = self.embedding(jnp.array([t]))
        h = h.repeat(self.n_node, axis=0).reshape(self.n_node, self.hidden_size)

        # Message passing
        for layer in self.layers:
            h, pos = layer(h, pos)

        pos = pos.reshape(-1)
        return pos
