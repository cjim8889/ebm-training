from typing import Dict, List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class EmbedderBlock(eqx.Module):
    particle_embedder: eqx.nn.Linear
    layernorm: eqx.nn.LayerNorm
    n_particles: int
    n_spatial_dim: int
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
        in_dim = n_spatial_dim + 2 if shortcut else n_spatial_dim + 1

        self.particle_embedder = eqx.nn.Linear(
            in_features=in_dim,
            out_features=embedding_size,
            use_bias=True,
            key=key,
        )
        # Correct LayerNorm shape to feature dimension only
        self.layernorm = eqx.nn.LayerNorm(shape=(embedding_size,))

        self.n_particles = n_particles
        self.n_spatial_dim = n_spatial_dim

    def __call__(self, xs, t, d=None):
        if self.shortcut:
            d = jnp.broadcast_to(d, (xs.shape[0], 1))
            t = jnp.broadcast_to(t, (xs.shape[0], 1))
            xs = jnp.concatenate([xs, t, d], axis=-1)
        else:
            t = jnp.broadcast_to(t, (xs.shape[0], 1))
            xs = jnp.concatenate([xs, t], axis=-1)

        embedded = jax.vmap(self.particle_embedder)(xs)
        # Apply LayerNorm with vmap per-particle
        return jax.vmap(self.layernorm)(embedded)


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
    """Optimized feed-forward network with parameter reuse."""

    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(self, hidden_size, dropout_rate, key):
        key1, key2 = jax.random.split(key)
        self.linear1 = eqx.nn.Linear(hidden_size, hidden_size * 4, key=key1)
        self.linear2 = eqx.nn.Linear(hidden_size * 4, hidden_size, key=key2)
        self.layernorm = eqx.nn.LayerNorm(hidden_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(self, x, enable_dropout=False, key=None):
        residual = x
        # Apply vmap to linear layers to process each particle
        x = jax.vmap(self.linear1)(x)
        x = jax.nn.gelu(x)
        x = self.dropout(x, key=key, inference=not enable_dropout)
        x = jax.vmap(self.linear2)(x)
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
        attn_key, ffn_key = jax.random.split(key) if key else (None, None)
        x = self.attn(x, enable_dropout, attn_key)
        return self.ffn(x, enable_dropout, ffn_key)


class ParticleTransformer(eqx.Module):
    """Efficient transformer with optional d-conditioning."""

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

        self.predictor = eqx.nn.Linear(hidden_size, n_spatial_dim, key=p_key)

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

        xs = xs.reshape(-1, self.embedder.n_spatial_dim)
        x = self.embedder(xs, t, d=d if self.shortcut else None)

        for layer in self.layers:
            x = layer(x, enable_dropout, key)
            if key is not None:
                key, _ = jax.random.split(key)

        return jax.vmap(self.predictor)(x).flatten()
