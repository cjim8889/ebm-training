import equinox as eqx
import jax
import jax.numpy as jnp

from utils.distributions import compute_distances
from .mlp import MLPWithLayerNorm


class EfficientPairwiseInteraction(eqx.Module):
    """Replace O(N^2) pairwise features with attention-based aggregation."""

    key_proj: eqx.nn.Linear
    value_proj: eqx.nn.Linear
    time_dist_proj: eqx.nn.Linear
    heads: int = eqx.static_field()

    def __init__(self, dim, heads=4, key=None, shortcut=False):
        keys = jax.random.split(key, 3)
        self.key_proj = eqx.nn.Linear(
            1, dim * heads, key=keys[0]
        )  # Distance projection
        self.value_proj = eqx.nn.Linear(1, dim, key=keys[1])  # Distance values
        self.time_dist_proj = eqx.nn.Linear(
            2 if shortcut else 1, dim * heads, key=keys[2]
        )  # Time/distance
        self.heads = heads

    def __call__(self, dists, t, d=None):
        dists = dists.reshape(-1, 1)
        # dists: (N*(N-1)/2, 1)
        keys_all_heads = jax.vmap(self.key_proj)(dists)  # (N*(N-1)/2, dim * heads)
        values = jax.vmap(self.value_proj)(dists)  # (N*(N-1)/2, dim)
        B = jnp.concatenate([t, d] if d is not None else [t]).reshape(
            -1
        )  # (1,) or (2,)
        queries_all_heads = self.time_dist_proj(B)  # (dim * heads,)

        keys = jnp.split(
            keys_all_heads, self.heads, axis=-1
        )  # list of (heads, N*(N-1)/2, dim)
        queries = jnp.split(
            queries_all_heads, self.heads, axis=-1
        )  # list of (heads, dim)

        keys_stacked = jnp.stack(keys)  # (heads, N*(N-1)/2, dim)
        queries_stacked = jnp.stack(queries)  # (heads, dim)

        def process_head(key_head, query_head, values):
            attn = jax.nn.softmax(
                query_head @ key_head.T
            )  # (N*(N-1)/2,)  q: (dim,), k.T: (dim, N*(N-1)/2)
            output_h = (
                attn @ values
            )  # (dim,) attn: (N*(N-1)/2,), values: (N*(N-1)/2, dim)
            return output_h

        vmap_process_head = jax.vmap(process_head, in_axes=(0, 0, None), out_axes=0)
        head_outputs_vmap = vmap_process_head(
            keys_stacked, queries_stacked, values
        )  # (heads, dim)
        return head_outputs_vmap.sum(axis=0)


# class EfficientMLP(eqx.Module):
#     layers: list
#     activation: callable = eqx.static_field()
#     mixed_precision: bool = eqx.static_field()

#     def __init__(
#         self,
#         in_size,
#         out_size,
#         hidden_size,
#         depth,
#         key,
#         activation=jax.nn.relu,
#         rms_norm=True,
#         mixed_precision=False,
#     ):
#         keys = jax.random.split(key, depth + 1)
#         layers = []
#         sizes = [in_size] + [hidden_size] * depth + [out_size]

#         for i in range(len(sizes) - 1):
#             layers.append(eqx.nn.Linear(sizes[i], sizes[i + 1], key=keys[i]))
#             if i < len(sizes) - 2 and rms_norm:  # Skip norm on last layer
#                 layers.append(eqx.nn.RMSNorm(hidden_size))
#             if i < len(sizes) - 2:
#                 layers.append(eqx.nn.Lambda(activation))

#         self.layers = layers
#         self.activation = activation

#     def __call__(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x


class OptimizedVelocityField(eqx.Module):
    pairwise: EfficientPairwiseInteraction
    mlp: MLPWithLayerNorm
    shortcut: bool = eqx.static_field()
    n_particles: int = eqx.static_field()
    n_spatial_dims: int = eqx.static_field()

    def __init__(
        self,
        key,
        dim,
        hidden_size,
        n_particles,
        n_spatial_dims,
        depth=4,
        heads=4,
        shortcut=False,
        mixed_precision=False,
    ):
        self.shortcut = shortcut
        p_key, m_key = jax.random.split(key)

        self.pairwise = EfficientPairwiseInteraction(
            hidden_size // heads, heads, p_key, shortcut
        )

        mlp_in = dim + hidden_size // heads + (2 if shortcut else 1)
        self.mlp = MLPWithLayerNorm(
            mlp_in,
            dim,
            hidden_size,
            depth,
            m_key,
            activation=jax.nn.gelu,
            mixed_precision=mixed_precision,
            rms_norm=True,
        )

        self.n_particles = n_particles
        self.n_spatial_dims = n_spatial_dims

    def __call__(self, x, t, d=None):
        dists = compute_distances(
            x,
            n_particles=self.n_particles,
            n_dimensions=self.n_spatial_dims,
            repeat=False,
        )  # Optimized distance

        # Efficient attention-based aggregation
        pair_feat = self.pairwise(dists, t, d if self.shortcut else None)

        # Concatenate features
        features = [x, pair_feat, t]
        if self.shortcut:
            features.append(d)

        out = self.mlp(jnp.concatenate(features))
        return out
