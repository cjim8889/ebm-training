from typing import List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
from jaxtyping import Array, Float


class EmbedderBlock(eqx.Module):
    particle_embedder: eqx.nn.Linear
    layernorm: eqx.nn.LayerNorm
    n_particles: int
    n_spatial_dim: int
    shortcut: bool = eqx.field(static=True)
    mp_policy: jmp.Policy = eqx.field(static=True)

    def __init__(
        self,
        n_particles: int,
        n_spatial_dim: int,
        embedding_size: int,
        key: jax.random.PRNGKey,
        mp_policy: jmp.Policy,
        shortcut: bool = False,
    ):
        self.shortcut = shortcut
        self.mp_policy = mp_policy
        in_dim = n_spatial_dim + 2 if shortcut else n_spatial_dim + 1

        self.particle_embedder = eqx.nn.MLP(
            in_size=in_dim,
            out_size=embedding_size,
            width_size=64,
            depth=3,
            activation=jax.nn.silu,
            use_bias=True,
            key=key,
            dtype=mp_policy.param_dtype,
        )
        # Correct LayerNorm shape to feature dimension only
        self.layernorm = eqx.nn.LayerNorm(shape=(embedding_size,), dtype=jnp.float32)

        self.n_particles = n_particles
        self.n_spatial_dim = n_spatial_dim

    def __call__(self, xs, t, d=None):
        if self.shortcut:
            d = jnp.broadcast_to(d, (xs.shape[0], 1))
            t = jnp.broadcast_to(t, (xs.shape[0], 1))
            input = jnp.concatenate([xs, t, d], axis=-1)
        else:
            t = jnp.broadcast_to(t, (xs.shape[0], 1))
            input = jnp.concatenate([xs, t], axis=-1)

        input = self.mp_policy.cast_to_compute(input)
        embedder = self.mp_policy.cast_to_compute(self.particle_embedder)

        embedded = jax.vmap(embedder)(input)
        # Apply LayerNorm with vmap per-particle using fp32
        return self.mp_policy.cast_to_output(jax.vmap(self.layernorm)(embedded.astype(jnp.float32)))


class SimplifiedAttentionBlock(eqx.Module):
    """Optimized attention block without positional embeddings."""

    attention: eqx.nn.MultiheadAttention
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout
    mp_policy: jmp.Policy = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        key: jax.random.PRNGKey,
        mp_policy: jmp.Policy,
    ):
        self.mp_policy = mp_policy
        self.num_heads = num_heads
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=hidden_size,
            dropout_p=attention_dropout_rate,
            key=key,
            dtype=mp_policy.param_dtype,
        )
        self.layernorm = eqx.nn.LayerNorm(hidden_size, dtype=jnp.float32)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(self, inputs, enable_dropout=False, key=None):
        attn_key, dropout_key = (
            jax.random.split(key) if key is not None else (None, None)
        )
        inputs = self.mp_policy.cast_to_output(inputs)
        attention = self.mp_policy.cast_to_output(self.attention)

        # Self-attention with residual connection
        attn_out = attention(
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
    mp_policy: jmp.Policy = eqx.field(static=True)
    def __init__(self, hidden_size, dropout_rate, key, mp_policy: jmp.Policy):
        self.mp_policy = mp_policy
        key1, key2 = jax.random.split(key)
        self.linear1 = eqx.nn.Linear(hidden_size, hidden_size * 4, key=key1, dtype=mp_policy.param_dtype)
        self.linear2 = eqx.nn.Linear(hidden_size * 4, hidden_size, key=key2, dtype=mp_policy.param_dtype)
        self.layernorm = eqx.nn.LayerNorm(hidden_size, dtype=jnp.float32)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(self, x, enable_dropout=False, key=None):
        x = self.mp_policy.cast_to_compute(x)
        linear1 = self.mp_policy.cast_to_compute(self.linear1)
        linear2 = self.mp_policy.cast_to_compute(self.linear2)
        layernorm = self.mp_policy.cast_to_compute(self.layernorm)

        residual = x
        # Apply vmap to linear layers to process each particle
        x = jax.vmap(linear1)(x)
        # Cast to FP32 before GELU activation for improved numerical stability
        x = jax.nn.gelu(x.astype(jnp.float32))
        x = self.dropout(x, key=key, inference=not enable_dropout)
        x = jax.vmap(linear2)(x) + residual
        return self.mp_policy.cast_to_output(jax.vmap(layernorm)(x.astype(jnp.float32)))


class TransformerLayer(eqx.Module):
    """Combined transformer layer with optimized components."""

    attn: SimplifiedAttentionBlock
    ffn: EfficientFFN
    mp_policy: jmp.Policy = eqx.field(static=True)

    def __init__(self, hidden_size, num_heads, dropout_rate, attn_dropout_rate, key, mp_policy: jmp.Policy):
        self.mp_policy = mp_policy
        key1, key2 = jax.random.split(key)
        self.attn = SimplifiedAttentionBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attn_dropout_rate,
            key=key1,
            mp_policy=mp_policy,
        )
        self.ffn = EfficientFFN(
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            key=key2,
            mp_policy=mp_policy,
        )

    def __call__(self, x, enable_dropout=False, key=None):
        attn_key, ffn_key = jax.random.split(key) if key is not None else (None, None)

        x = self.mp_policy.cast_to_compute(x)
        x = self.attn(x, enable_dropout, attn_key)
        return self.mp_policy.cast_to_output(self.ffn(x, enable_dropout, ffn_key))


class ParticleTransformer(eqx.Module):
    """Efficient transformer with optional d-conditioning."""

    embedder: EmbedderBlock
    layers: List[TransformerLayer]
    predictor: eqx.nn.Linear
    shortcut: bool = eqx.field(static=True)
    mp_policy: jmp.Policy = eqx.field(static=True)

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
        mp_policy: jmp.Policy,
        shortcut: bool = False,
    ):
        self.shortcut = shortcut
        self.mp_policy = mp_policy
        e_key, l_key, p_key = jax.random.split(key, 3)

        self.embedder = EmbedderBlock(
            n_particles=n_particles,
            n_spatial_dim=n_spatial_dim,
            embedding_size=hidden_size,
            key=e_key,
            shortcut=shortcut,
            mp_policy=mp_policy,
        )

        self.layers = [
            TransformerLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                attn_dropout_rate=attn_dropout_rate,
                key=k,
                mp_policy=mp_policy,
            )
            for k in jax.random.split(l_key, num_layers)
        ]

        self.predictor = eqx.nn.Linear(hidden_size, n_spatial_dim, key=p_key, dtype=mp_policy.param_dtype)

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
        
        xs = self.mp_policy.cast_to_compute(xs)
        t = self.mp_policy.cast_to_compute(t)
        if d is not None:
            d = self.mp_policy.cast_to_compute(d)

        predictor = self.mp_policy.cast_to_compute(self.predictor)

        xs = xs.reshape(-1, self.embedder.n_spatial_dim)
        x = self.embedder(xs, t, d=d if self.shortcut else None)

        for layer in self.layers:
            x = layer(x, enable_dropout, key)
            if key is not None:
                key, _ = jax.random.split(key)

        return self.mp_policy.cast_to_output(jax.vmap(predictor)(x).flatten())
