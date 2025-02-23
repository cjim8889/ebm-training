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


class GEONORM(eqx.Module):
    """SE(3)-Equivariant Geometric Normalization Layer"""

    g: jnp.ndarray  # Learnable scale parameters [N,1]
    b: jnp.ndarray  # Learnable direction coefficients [N,1]
    eps: float

    def __init__(
        self,
        num_particles: int,
        eps: float = 1e-5,
        key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    ):
        g_key, b_key = jax.random.split(key)
        self.g = jax.random.normal(g_key, (num_particles, 1)) * 0.1
        self.b = jax.random.normal(b_key, (num_particles, 1)) * 0.1
        self.eps = eps

    def __call__(self, x: jnp.ndarray, mu0: jnp.ndarray) -> jnp.ndarray:
        """Input shape: [N,D] (N particles in D-dimensional space)"""
        # Initialize mu0 on first call
        mu = jnp.mean(x, axis=0, keepdims=True)  # Current centroid [1,D]
        x_centered = x - mu  # Translation-invariant coordinates [N,D]

        # SE(3)-invariant radial distances
        sigma = jnp.linalg.norm(x_centered, axis=1, keepdims=True) + self.eps  # [N,1]

        # Directional alignment term
        mu_diff = mu - mu0  # [1,D]
        mu_diff_norm = (
            jnp.linalg.norm(mu_diff, axis=-1, keepdims=True) + self.eps
        )  # [1,1]
        direction = mu_diff / mu_diff_norm  # [1,D]

        # Geometric normalization
        x_norm = (self.g * x_centered) / sigma  # Scaled translation-equivariant term
        directional_term = self.b * direction  # Rotation-equivariant alignment

        return x_norm + directional_term + mu0  # [N,D]


class EGNNLayer(eqx.Module):
    pos_mlp: eqx.nn.MLP
    normalize: bool
    eps: float
    senders: jnp.ndarray
    receivers: jnp.ndarray
    seg_count_senders: jnp.ndarray
    n_node: int
    shortcut: bool  # New: Conditionally include step size 'd'
    num_nearest_neighbors: Optional[int]  # New: Number of nearest neighbors

    def __init__(
        self,
        n_node: int,
        hidden_size: int,
        key: jax.random.PRNGKey,
        normalize: bool = False,
        tanh: bool = False,
        eps: float = 1.0,
        dt: float = 0.01,
        shortcut: bool = False,  # New parameter
        num_nearest_neighbors: int = 5,
        mixed_precision: bool = False,
    ):
        self.n_node = n_node
        self.normalize = normalize
        self.eps = eps
        self.shortcut = shortcut  # Store shortcut flag

        # Precompute graph structure
        self.senders, self.receivers = get_fully_connected_senders_receivers(n_node)
        self.num_nearest_neighbors = num_nearest_neighbors

        # Precompute normalization constants
        self.seg_count_senders = num_nearest_neighbors * jnp.ones(
            n_node, dtype=jnp.float32
        )
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
            rms_norm=True,
        )

        self.pos_mlp = init_linear_weights(self.pos_mlp, xavier_init, key, scale=dt)

    def _pos_update(
        self,
        senders: jnp.ndarray,
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
        seg_sum = jax.ops.segment_sum(trans, senders, self.n_node)
        return seg_sum / self.seg_count_senders[:, None]

    def __call__(self, *args):
        """Flexible call signature based on shortcut configuration."""
        if self.shortcut:
            pos, t, d = args
        else:
            pos, t = args

        n_node = pos.shape[0]
        k = self.num_nearest_neighbors

        # Compute pairwise distances
        coord_diff = pos[:, None] - pos[None, :]  # [n, n, d]
        radial = jnp.sum(coord_diff**2, axis=-1)  # [n, n]

        # Find k-nearest neighbors (excluding self)
        if k is not None:
            masked_radial = radial.at[jnp.diag_indices(n_node)].set(jnp.inf)
            _, receivers = jax.lax.top_k(-masked_radial, k=k)  # [n, k]
            senders = jnp.repeat(jnp.arange(n_node), k)
            receivers = receivers.flatten()
        else:  # Fully connected
            senders, receivers = self.senders, self.receivers

        # Compute edge features for selected pairs
        coord_diff = pos[senders] - pos[receivers]
        if self.normalize:
            norm = jnp.sqrt(jnp.sum(coord_diff**2, axis=1, keepdims=True) + self.eps)
            coord_diff = coord_diff / norm

        radial = jnp.sum(coord_diff**2, axis=1, keepdims=True)

        pos_update = self._pos_update(
            senders, radial, coord_diff, t, d if self.shortcut else None
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
        num_nearest_neighbors: int = None,
        mixed_precision: bool = False,
        geonorm: bool = False,
    ):
        self.n_node = n_node
        self.shortcut = shortcut
        keys = jax.random.split(key, num_layers)

        self.layers = []

        for i, key in enumerate(keys):
            self.layers.append(
                EGNNLayer(
                    n_node=n_node,
                    hidden_size=hidden_size,
                    key=key,
                    normalize=normalize,
                    tanh=tanh,
                    shortcut=shortcut,
                    num_nearest_neighbors=num_nearest_neighbors,
                    mixed_precision=mixed_precision,
                )
            )
            if geonorm and i < num_layers - 1:
                key, _ = jax.random.split(key)
                self.layers.append(GEONORM(n_node, key=key))

    def __call__(self, *args):
        """Dynamic forward pass supporting optional step size."""
        if self.shortcut:
            pos, t, d = args
        else:
            pos, t = args

        pos = pos.reshape(self.n_node, -1)
        mu0 = jnp.mean(pos, axis=0, keepdims=True)

        for layer in self.layers:
            if isinstance(layer, EGNNLayer):
                if self.shortcut:
                    pos = layer(pos, t, d)
                else:
                    pos = layer(pos, t)
            else:
                pos = layer(pos, mu0)
        return pos.reshape(-1)
