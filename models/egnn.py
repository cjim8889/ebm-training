from typing import Optional, Tuple
import equinox as eqx
import jax
import jax.numpy as jnp

from utils.models import xavier_init, init_linear_weights
from .mlp import MLPWithLayerNorm


def get_fully_connected_senders_receivers(
    n_node: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Vectorized computation of fully connected graph edges."""
    idx = jnp.arange(n_node)
    senders, receivers = jnp.meshgrid(idx, idx, indexing="ij")
    mask = senders != receivers
    return senders[mask], receivers[mask]


class EGNNLayer(eqx.Module):
    pos_mlp: eqx.nn.MLP
    normalize: bool
    eps: float
    dt: float
    senders: jnp.ndarray
    receivers: jnp.ndarray
    seg_count_senders: jnp.ndarray
    n_node: int
    shortcut: bool  # New: Conditionally include step size 'd'

    def __init__(
        self,
        n_node: int,
        hidden_size: int,
        key: jax.random.PRNGKey,
        normalize: bool = False,
        tanh: bool = False,
        dt: float = 0.001,
        eps: float = 1e-8,
        shortcut: bool = False,  # New parameter
        mixed_precision: bool = False,
    ):
        self.dt = dt
        self.n_node = n_node
        self.normalize = normalize
        self.eps = eps
        self.shortcut = shortcut  # Store shortcut flag

        # Precompute graph structure
        self.senders, self.receivers = get_fully_connected_senders_receivers(n_node)

        # Precompute normalization constants
        self.seg_count_senders = (n_node - 1) * jnp.ones(n_node, dtype=jnp.float32)
        self.seg_count_senders = jnp.maximum(self.seg_count_senders, 1.0)

        # Dynamic MLP input size based on shortcut
        mlp_in_size = 3 if shortcut else 2  # radial + (t, d)
        self.pos_mlp = MLPWithLayerNorm(
            in_size=mlp_in_size,
            out_size=1,
            width_size=hidden_size,
            depth=2,
            activation=jax.nn.silu,
            final_activation=jax.nn.tanh if tanh else lambda x: x,
            key=key,
            mixed_precision=mixed_precision,
        )

        self.pos_mlp = init_linear_weights(self.pos_mlp, xavier_init, key, scale=dt)

    def _pos_update(
        self,
        radial: jnp.ndarray,
        coord_diff: jnp.ndarray,
        t: float,
        d: Optional[float] = None,
    ) -> jnp.ndarray:
        """Enhanced position update with optional step size."""
        # Conditionally build edge features
        if self.shortcut:
            edge_features = jnp.concatenate(
                [
                    radial,
                    jnp.full_like(radial, t),  # Time feature
                    jnp.full_like(radial, d),  # Step size feature
                ],
                axis=-1,
            )
        else:
            edge_features = jnp.concatenate([radial, jnp.full_like(radial, t)], axis=-1)

        # Process all edges in single batch
        edge_scalars = jax.vmap(self.pos_mlp)(edge_features).squeeze(-1)

        trans = coord_diff * edge_scalars[:, None]
        seg_sum = jax.ops.segment_sum(trans, self.senders, self.n_node)
        return seg_sum / self.seg_count_senders[:, None]

    def _coord2radial(self, coord: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Efficient coordinate to radial conversion."""
        coord_diff = coord[self.senders] - coord[self.receivers]
        radial = jnp.sum(coord_diff**2, axis=1, keepdims=True)
        if self.normalize:
            norm = jnp.sqrt(radial + self.eps)
            coord_diff = coord_diff / norm
        return radial, coord_diff

    def __call__(self, *args):
        """Flexible call signature based on shortcut configuration."""
        if self.shortcut:
            pos, t, d = args
        else:
            pos, t = args

        radial, coord_diff = self._coord2radial(pos)
        pos_update = self._pos_update(
            radial, coord_diff, t, d if self.shortcut else None
        )
        return pos + pos_update


class EGNN(eqx.Module):
    layers: list
    n_node: int
    shortcut: bool  # Expose shortcut configuration

    def __init__(
        self,
        n_node: int,
        hidden_size: int,
        key: jax.random.PRNGKey,
        num_layers: int = 4,
        normalize: bool = False,
        tanh: bool = False,
        shortcut: bool = False,  # New configuration
        mixed_precision: bool = False,
    ):
        self.n_node = n_node
        self.shortcut = shortcut
        keys = jax.random.split(key, num_layers)

        self.layers = [
            EGNNLayer(
                n_node=n_node,
                hidden_size=hidden_size,
                key=k,
                normalize=normalize,
                tanh=tanh,
                shortcut=shortcut,  # Propagate configuration
                mixed_precision=mixed_precision,
            )
            for k in keys
        ]

    def __call__(self, *args):
        """Dynamic forward pass supporting optional step size."""
        if self.shortcut:
            pos, t, d = args
        else:
            pos, t = args

        pos = pos.reshape(self.n_node, -1)
        for layer in self.layers:
            if self.shortcut:
                pos = layer(pos, t, d)
            else:
                pos = layer(pos, t)
        return pos.reshape(-1)
