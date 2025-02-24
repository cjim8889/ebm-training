from typing import List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class EmbedderBlock(eqx.Module):
    spatial_embedder: eqx.nn.MLP
    time_embedder: eqx.nn.MLP
    d_embedder: Optional[eqx.nn.MLP]
    layernorm: eqx.nn.LayerNorm
    n_particles: int
    embedding_size: int
    shortcut: bool = eqx.field(static=True)

    def __init__(
        self,
        n_particles: int,
        n_spatial_dim: int,
        embedding_size: int,
        key: jax.random.PRNGKey,
        shortcut: bool = False,
    ):
        self.shortcut = shortcut
        self.n_particles = n_particles
        self.embedding_size = embedding_size

        spatial_key, time_key, d_key = jax.random.split(key, 3)
        # Spatial embedding: Linear layer for particle positions
        self.spatial_embedder = eqx.nn.MLP(n_spatial_dim, embedding_size, width_size=32, depth=2, key=spatial_key)

        # Time embedding: Small MLP for time `t`
        self.time_embedder = eqx.nn.MLP(1, embedding_size, width_size=32, depth=2, key=time_key)

        # Conditional embedding for `d` if shortcut is True
        self.d_embedder = eqx.nn.MLP(1, embedding_size, width_size=32, depth=2, key=d_key) if shortcut else None

        # Layer normalization
        self.layernorm = eqx.nn.LayerNorm(embedding_size)

    def __call__(self, xs, t, d=None):
        # Embed spatial positions
        xs_embed = jax.vmap(self.spatial_embedder)(xs)  # [n_particles, embedding_size]

        # Embed time `t`
        if jnp.ndim(t) == 0:
            t = jnp.expand_dims(t, 0)
        t_embed = self.time_embedder(t)  # [embedding_size, ]
        t_embed = t_embed.reshape(1, -1)  # Reshape to [1, embedding_size]

        # Combine embeddings: xs + t
        combined = xs_embed + t_embed

        # If shortcut, embed `d` and add to combined embeddings
        if self.shortcut:
            if jnp.ndim(d) == 0:
                d = jnp.expand_dims(d, 0)
            d_embed = self.d_embedder(d)  # [embedding_size, ]
            d_embed = d_embed.reshape(1, -1)  # Reshape to [1, embedding_size]
            combined += d_embed

        # Apply layer normalization
        return jax.vmap(self.layernorm)(combined)


class SimplifiedAttentionBlock(eqx.Module):
    """Optimized attention block without positional embeddings."""

    attention: eqx.nn.MultiheadAttention
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout
    num_heads: int = eqx.field(static=True)

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        key: jax.random.PRNGKey,
    ):
        self.num_heads = num_heads
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=hidden_size,
            dropout_p=attention_dropout_rate,
            key=key,
        )
        self.layernorm = eqx.nn.LayerNorm(hidden_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(self, inputs, enable_dropout=False, key=None):
        attn_key, dropout_key = (
            jax.random.split(key) if key is not None else (None, None)
        )

        # Self-attention with residual connection
        attn_out = self.attention(
            query=inputs,
            key_=inputs,
            value=inputs,
            inference=not enable_dropout,
            key=attn_key,
        )
        attn_out = self.dropout(attn_out, key=dropout_key, inference=not enable_dropout)
        attn_out = inputs + attn_out
        return jax.vmap(self.layernorm)(attn_out)

class EfficientFFN(eqx.Module):
    conv: eqx.nn.Conv1d
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    linear3: eqx.nn.Linear
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(self, hidden_size, dropout_rate, key):
        # Depthwise separable convolution for local interactions
        self.conv = eqx.nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1,
            groups=hidden_size,  # Depthwise convolution
            key=key,
        )
        # Three-layer FFN
        self.linear1 = eqx.nn.Linear(hidden_size, hidden_size * 4, key=key)
        self.linear2 = eqx.nn.Linear(hidden_size * 4, hidden_size * 4, key=key)
        self.linear3 = eqx.nn.Linear(hidden_size * 4, hidden_size, key=key)
        self.layernorm = eqx.nn.LayerNorm(hidden_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(self, x, enable_dropout=False, key=None):
        # Apply depthwise convolution across particles
        x_conv = self.conv(x.transpose(1, 0)).transpose(1, 0)  # [n_particles, hidden_size]

        # Add convolution output to input
        x = x + x_conv

        # Apply FFN with three layers
        residual = x
        x = jax.vmap(self.linear1)(x)
        x = jax.nn.gelu(x)
        x = self.dropout(x, key=key, inference=not enable_dropout)
        x = jax.vmap(self.linear2)(x)
        x = jax.nn.gelu(x)
        x = self.dropout(x, key=key, inference=not enable_dropout)
        x = jax.vmap(self.linear3)(x)
        return jax.vmap(self.layernorm)(residual + x)


class TransformerLayer(eqx.Module):
    """Combined transformer layer with optimized components."""

    attn: SimplifiedAttentionBlock
    ffn: EfficientFFN

    def __init__(self, hidden_size, num_heads, dropout_rate, attn_dropout_rate, key):
        key1, key2 = jax.random.split(key)
        self.attn = SimplifiedAttentionBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attn_dropout_rate,
            key=key1,
        )
        self.ffn = EfficientFFN(
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            key=key2,
        )

    def __call__(self, x, enable_dropout=False, key=None):
        attn_key, ffn_key = jax.random.split(key) if key is not None else (None, None)
        x = self.attn(x, enable_dropout, attn_key)
        return self.ffn(x, enable_dropout, ffn_key)


class ParticleTransformerV2(eqx.Module):
    """Efficient transformer with optional d-conditioning."""

    n_spatial_dim: int = eqx.field(static=True)
    embedder: EmbedderBlock
    layers: List[TransformerLayer]
    predictor: eqx.nn.Linear
    shortcut: bool = eqx.field(static=True)

    def __init__(
        self,
        n_particles: int,
        n_spatial_dim: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dropout_rate: float,
        attn_dropout_rate: float,
        key: jax.random.PRNGKey,
        shortcut: bool = False,
    ):
        self.shortcut = shortcut
        e_key, l_key, p_key = jax.random.split(key, 3)

        self.embedder = EmbedderBlock(
            n_particles=n_particles,
            n_spatial_dim=n_spatial_dim,
            embedding_size=hidden_size,
            key=e_key,
            shortcut=shortcut,
        )

        self.layers = [
            TransformerLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                attn_dropout_rate=attn_dropout_rate,
                key=k,
            )
            for k in jax.random.split(l_key, num_layers)
        ]

        self.predictor = eqx.nn.MLP(hidden_size, n_spatial_dim, width_size=hidden_size//2, depth=2, key=p_key)

        self.n_spatial_dim = n_spatial_dim

    def __call__(
        self,
        xs: Float[Array, "..."],
        t: Float,
        d: Optional[Float] = None,
        *,
        enable_dropout: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Float[Array, "..."]:
        if self.shortcut and d is None:
            raise ValueError("d must be provided when shortcut is enabled")

        xs = xs.reshape(-1, self.n_spatial_dim)
        x = self.embedder(xs, t, d=d if self.shortcut else None)

        for layer in self.layers:
            x = layer(x, enable_dropout, key)
            if key is not None:
                key, _ = jax.random.split(key)

        return jax.vmap(self.predictor)(x).flatten()
