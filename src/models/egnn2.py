from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from utils.models import init_linear_weights, xavier_init

from .egnn import get_fully_connected_senders_receivers, GEONORM
from .mlp import MLPWithNorm


class EGNNLayer(eqx.Module):
    phi_e: MLPWithNorm  # Edge message function
    phi_h: MLPWithNorm  # Node feature update function
    phi_x: MLPWithNorm  # Position update scalar function
    normalize: bool
    eps: float
    senders: jnp.ndarray
    receivers: jnp.ndarray
    seg_count_senders: jnp.ndarray
    n_node: int
    shortcut: bool
    num_nearest_neighbors: Optional[int]

    def __init__(
        self,
        n_node: int,
        hidden_size: int,
        key: jax.random.PRNGKey,
        mlp_depth: int = 2,
        normalize: bool = False,
        tanh: bool = False,
        eps: float = 1.0,
        dt: float = 0.01,
        shortcut: bool = False,
        num_nearest_neighbors: int = None,
        mixed_precision: bool = False,
        norm: str = "rms",
    ):
        self.n_node = n_node
        self.normalize = normalize
        self.eps = eps
        self.shortcut = shortcut
        self.senders, self.receivers = get_fully_connected_senders_receivers(n_node)
        self.num_nearest_neighbors = num_nearest_neighbors

        self.seg_count_senders = (
            num_nearest_neighbors * jnp.ones(n_node, dtype=jnp.float32)
            if num_nearest_neighbors
            else (n_node - 1) * jnp.ones(n_node, dtype=jnp.float32)
        )
        self.seg_count_senders = jnp.maximum(self.seg_count_senders, 1.0)

        keys = jax.random.split(key, 3)

        # phi_e: processes [h_senders, h_receivers, radial, t, (d)]
        in_size_e = 2 * hidden_size + 2 + (1 if shortcut else 0)
        self.phi_e = MLPWithNorm(
            in_size=in_size_e,
            out_size=hidden_size,  # Message dimension
            width_size=hidden_size,
            depth=mlp_depth,
            activation=jax.nn.silu,
            key=keys[0],
            mixed_precision=mixed_precision,
            norm=norm,
        )

        # phi_h: processes [h_i, aggregated messages]
        self.phi_h = MLPWithNorm(
            in_size=2 * hidden_size,
            out_size=hidden_size,
            width_size=hidden_size,
            depth=mlp_depth,
            activation=jax.nn.silu,
            key=keys[1],
            mixed_precision=mixed_precision,
            norm=norm,
        )

        # phi_x: processes m_ij to scalar for position update
        self.phi_x = MLPWithNorm(
            in_size=hidden_size,
            out_size=1,
            width_size=hidden_size,
            depth=mlp_depth,
            activation=jax.nn.tanh if tanh else lambda x: x,
            key=keys[2],
            mixed_precision=mixed_precision,
            norm=norm,
        )

        # Initialize weights
        self.phi_e = init_linear_weights(self.phi_e, xavier_init, keys[0])
        self.phi_h = init_linear_weights(self.phi_h, xavier_init, keys[1])
        self.phi_x = init_linear_weights(self.phi_x, xavier_init, keys[2], scale=dt)

    def __call__(self, pos, h, t, d=None):
        n_node = pos.shape[0]
        k = self.num_nearest_neighbors

        # Compute full coord_diff and radial once
        coord_diff_full = pos[:, None] - pos[None, :]  # [n, n, d]
        radial_full = jnp.sum(coord_diff_full**2, axis=-1)  # [n, n]

        # Determine senders and receivers
        if k is not None:
            # Select top-k nearest neighbors
            masked_radial = radial_full.at[jnp.diag_indices(n_node)].set(jnp.inf)
            _, receivers = jax.lax.top_k(-masked_radial, k=k)
            senders = jnp.repeat(jnp.arange(n_node), k)
            receivers = receivers.flatten()
        else:
            senders, receivers = self.senders, self.receivers

        # Extract coord_diff and radial for selected edges
        coord_diff = coord_diff_full[senders, receivers]  # [num_edges, d]
        radial = radial_full[senders, receivers][:, None]  # [num_edges, 1]

        # Normalize coord_diff if required, reusing radial
        if self.normalize:
            norm = jnp.sqrt(radial + self.eps)
            coord_diff = coord_diff / norm

        # Compute edge features
        h_senders = h[senders]
        h_receivers = h[receivers]
        t_feature = jnp.full((len(senders), 1), t)
        edge_input = (
            jnp.concatenate(
                [
                    h_senders,
                    h_receivers,
                    radial,
                    t_feature,
                    jnp.full((len(senders), 1), d),
                ],
                axis=-1,
            )
            if self.shortcut
            else jnp.concatenate([h_senders, h_receivers, radial, t_feature], axis=-1)
        )
        m_ij = jax.vmap(self.phi_e)(edge_input)

        # Update node features
        agg_i = jax.ops.segment_sum(m_ij, senders, self.n_node)
        h_input = jnp.concatenate([h, agg_i], axis=-1)
        h_update = jax.vmap(self.phi_h)(h_input)
        h_new = h + h_update

        # Update positions
        edge_scalars = jax.vmap(self.phi_x)(m_ij).squeeze(-1)
        trans = coord_diff * edge_scalars[:, None]
        seg_sum = jax.ops.segment_sum(trans, senders, self.n_node)
        pos_update = seg_sum / self.seg_count_senders[:, None]
        pos_new = pos + pos_update

        return pos_new, h_new


class EGNNWithLearnableNodeFeatures(eqx.Module):
    layers: list
    n_node: int
    shortcut: bool
    h: jnp.ndarray  # Learnable node features

    def __init__(
        self,
        n_node: int,
        hidden_size: int,
        key: jax.random.PRNGKey,
        num_layers: int = 4,
        mlp_depth: int = 2,
        normalize: bool = False,
        tanh: bool = False,
        shortcut: bool = False,
        num_nearest_neighbors: int = None,
        mixed_precision: bool = False,
        geonorm: bool = False,
        norm: str = "rms",
    ):
        self.n_node = n_node
        self.shortcut = shortcut
        keys = jax.random.split(key, num_layers + 1)  # Extra key for h

        self.h = (
            jax.random.normal(keys[-1], (n_node, hidden_size)) * 0.1
        )  # Initialize node features

        self.layers = []
        for i, key in enumerate(keys[:-1]):
            self.layers.append(
                EGNNLayer(
                    n_node=n_node,
                    hidden_size=hidden_size,
                    key=key,
                    normalize=normalize,
                    mlp_depth=mlp_depth,
                    tanh=tanh,
                    shortcut=shortcut,
                    num_nearest_neighbors=num_nearest_neighbors,
                    mixed_precision=mixed_precision,
                    norm=norm,
                )
            )
            if geonorm and i < num_layers - 1:
                key, _ = jax.random.split(key)
                self.layers.append(GEONORM(n_node, key=key))

    def __call__(self, pos, t, d=None):
        pos = pos.reshape(self.n_node, -1)
        h = self.h  # Initial node features
        mu0 = jnp.mean(pos, axis=0, keepdims=True)

        for layer in self.layers:
            if isinstance(layer, EGNNLayer):
                if self.shortcut:
                    pos, h = layer(pos, h, t, d)
                else:
                    pos, h = layer(pos, h, t)
            else:  # GEONORM
                pos = layer(pos, mu0)
        return pos.reshape(-1)
