from typing import Callable, List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class MixedPrecisionMLP(eqx.Module):
    layers: list
    activation: Callable = eqx.field(static=True)
    mixed_precision: bool = eqx.field(static=True)

    def __init__(
        self,
        in_size: int,
        out_size: int,
        width_size: int,
        depth: int,
        key: jax.random.PRNGKey,
        activation: Callable = jax.nn.silu,
        mixed_precision: bool = False,
    ):
        keys = jax.random.split(key, depth + 1)
        layers = []
        current_size = in_size
        for i in range(depth):
            layers.append(eqx.nn.Linear(current_size, width_size, key=keys[i]))
            layers.append(eqx.nn.Lambda(activation))
            current_size = width_size
        layers.append(eqx.nn.Linear(current_size, out_size, key=keys[-1]))
        self.layers = layers
        self.activation = activation
        self.mixed_precision = mixed_precision

    def __call__(self, x):
        if self.mixed_precision:
            x = x.astype(jnp.bfloat16)
            for layer in self.layers[:-1]:
                if isinstance(layer, eqx.nn.Linear):
                    weight = layer.weight.astype(jnp.bfloat16)
                    bias = layer.bias.astype(jnp.bfloat16) if layer.bias is not None else None
                    x = jnp.dot(x, weight.T) + bias
                else:  # activation
                    x = layer(x)
            final_layer = self.layers[-1]
            weight = final_layer.weight.astype(jnp.bfloat16)
            bias = final_layer.bias.astype(jnp.bfloat16) if final_layer.bias is not None else None
            x = jnp.dot(x, weight.T) + bias
            return x.astype(jnp.float32)
        else:
            for layer in self.layers[:-1]:
                x = layer(x)
            return self.layers[-1](x)

class EmbedderBlock(eqx.Module):
    spatial_embedder: MixedPrecisionMLP
    time_embedder: MixedPrecisionMLP
    d_embedder: Optional[MixedPrecisionMLP]
    layernorm: eqx.nn.LayerNorm
    n_particles: int
    embedding_size: int
    shortcut: bool = eqx.field(static=True)
    mixed_precision: bool = eqx.field(static=True)

    def __init__(
        self,
        n_particles: int,
        n_spatial_dim: int,
        embedding_size: int,
        key: jax.random.PRNGKey,
        shortcut: bool = False,
        mixed_precision: bool = False,
    ):
        self.shortcut = shortcut
        self.n_particles = n_particles
        self.embedding_size = embedding_size
        self.mixed_precision = mixed_precision

        spatial_key, time_key, d_key = jax.random.split(key, 3)
        self.spatial_embedder = MixedPrecisionMLP(
            n_spatial_dim, embedding_size, width_size=32, depth=2, key=spatial_key, mixed_precision=mixed_precision
        )
        self.time_embedder = MixedPrecisionMLP(
            1, embedding_size, width_size=32, depth=2, key=time_key, mixed_precision=mixed_precision
        )
        self.d_embedder = (
            MixedPrecisionMLP(1, embedding_size, width_size=32, depth=2, key=d_key, mixed_precision=mixed_precision)
            if shortcut
            else None
        )

        # Layer normalization
        self.layernorm = eqx.nn.LayerNorm(embedding_size)

    def __call__(self, xs, t, d=None):
        if self.mixed_precision:
            xs = xs.astype(jnp.bfloat16)
            t = t.astype(jnp.bfloat16)
            if d is not None:
                d = d.astype(jnp.bfloat16)

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

        # Layer normalization in float32 for stability
        combined_fp32 = combined.astype(jnp.float32)
        return jax.vmap(self.layernorm)(combined_fp32)


class SimplifiedAttentionBlock(eqx.Module):
    """Optimized attention block without positional embeddings."""

    attention: eqx.nn.MultiheadAttention
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout
    num_heads: int = eqx.field(static=True)
    mixed_precision: bool = eqx.field(static=True)

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        key: jax.random.PRNGKey,
        mixed_precision: bool = False,
    ):
        self.num_heads = num_heads
        self.mixed_precision = mixed_precision
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=hidden_size,
            dropout_p=attention_dropout_rate,
            key=key,
            dtype=jnp.bfloat16 if mixed_precision else jnp.float32,
        )
        self.layernorm = eqx.nn.LayerNorm(hidden_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(self, inputs, enable_dropout=False, key=None):
        attn_key, dropout_key = (
            jax.random.split(key) if key is not None else (None, None)
        )

        if self.mixed_precision:
            inputs = inputs.astype(jnp.bfloat16)

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

        # Layer normalization in float32 for stability
        attn_out = attn_out.astype(jnp.float32)
        return jax.vmap(self.layernorm)(attn_out)

class EfficientFFN(eqx.Module):
    """
    An efficient feed-forward network (FFN) module with mixed precision support.
    
    Args:
        hidden_size (int): The size of the hidden dimension.
        dropout_rate (float): The dropout rate to apply after activations.
        key (jax.random.PRNGKey): Random key for initializing the layers.
        mixed_precision (bool, optional): Whether to use mixed precision (bfloat16 for efficiency,
            float32 for stability). Defaults to False.
    """
    conv: eqx.nn.Conv1d
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    linear3: eqx.nn.Linear
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout
    mixed_precision: bool = eqx.field(static=True)

    def __init__(
        self,
        hidden_size: int,
        dropout_rate: float,
        key: jax.random.PRNGKey,
        mixed_precision: bool = False,
    ):
        self.mixed_precision = mixed_precision
        key, subkey = jax.random.split(key)
        self.conv = eqx.nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1,
            groups=hidden_size,  # Depthwise convolution
            key=subkey,
            dtype=jnp.bfloat16 if mixed_precision else jnp.float32,
        )
        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
        self.linear1 = eqx.nn.Linear(hidden_size, hidden_size * 4, key=subkey1, dtype=jnp.bfloat16 if mixed_precision else jnp.float32)
        self.linear2 = eqx.nn.Linear(hidden_size * 4, hidden_size * 4, key=subkey2, dtype=jnp.bfloat16 if mixed_precision else jnp.float32)
        self.linear3 = eqx.nn.Linear(hidden_size * 4, hidden_size, key=subkey3, dtype=jnp.bfloat16 if mixed_precision else jnp.float32)
        self.layernorm = eqx.nn.LayerNorm(hidden_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        x: Float[Array, "n_particles hidden_size"],
        enable_dropout: bool = False,
        key: Optional[jax.random.PRNGKey] = None
    ) -> Float[Array, "n_particles hidden_size"]:
        """
        Forward pass of the EfficientFFN module.

        Args:
            x (Float[Array, "n_particles hidden_size"]): Input tensor.
            enable_dropout (bool, optional): Whether to enable dropout during the forward pass.
                Defaults to False.
            key (Optional[jax.random.PRNGKey], optional): Random key for dropout. Required if
                enable_dropout is True. Defaults to None.

        Returns:
            Float[Array, "n_particles hidden_size"]: Output tensor after applying the FFN.
        """
        # Cast input to bfloat16 if mixed precision is enabled
        if self.mixed_precision:
            x = x.astype(jnp.bfloat16)

        # Apply depthwise convolution across particles
        x_conv = self.conv(x.transpose(1, 0)).transpose(1, 0)  # [n_particles, hidden_size]
        x = x + x_conv

        # Apply FFN with three layers
        residual = x
        x = jax.vmap(self.linear1)(x)
        x = jax.nn.gelu(x)
        if enable_dropout and key is not None:
            x = self.dropout(x, key=key, inference=not enable_dropout)
        
        x = jax.vmap(self.linear2)(x)
        x = jax.nn.gelu(x)
        if enable_dropout and key is not None:
            x = self.dropout(x, key=key, inference=not enable_dropout)
        
        x = jax.vmap(self.linear3)(x)
        # Add residual connection
        x = residual + x

        # Apply layer normalization in float32 for stability
        if self.mixed_precision:
            x = x.astype(jnp.float32)
        return jax.vmap(self.layernorm)(x)

class TransformerLayer(eqx.Module):
    """Combined transformer layer with optimized components."""

    attn: SimplifiedAttentionBlock
    ffn: EfficientFFN

    def __init__(self, hidden_size, num_heads, dropout_rate, attn_dropout_rate, key, mixed_precision=False):
        key1, key2 = jax.random.split(key)
        self.attn = SimplifiedAttentionBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attn_dropout_rate,
            key=key1,
            mixed_precision=mixed_precision,
        )
        self.ffn = EfficientFFN(
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            key=key2,
            mixed_precision=mixed_precision,
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
    mixed_precision: bool = eqx.field(static=True)

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
        mixed_precision: bool = False,
    ):
        self.shortcut = shortcut
        self.mixed_precision = mixed_precision
        e_key, l_key, p_key = jax.random.split(key, 3)

        self.embedder = EmbedderBlock(
            n_particles=n_particles,
            n_spatial_dim=n_spatial_dim,
            embedding_size=hidden_size,
            key=e_key,
            shortcut=shortcut,
            mixed_precision=mixed_precision,
        )

        self.layers = [
            TransformerLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                attn_dropout_rate=attn_dropout_rate,
                key=k,
                mixed_precision=mixed_precision,
            )
            for k in jax.random.split(l_key, num_layers)
        ]

        self.predictor = eqx.nn.MLP(hidden_size, n_spatial_dim, width_size=hidden_size//2, depth=2, key=p_key, dtype=jnp.bfloat16 if mixed_precision else jnp.float32)

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

        if self.mixed_precision:
            x = x.astype(jnp.bfloat16)

        output = jax.vmap(self.predictor)(x).flatten()
        return output.astype(jnp.float32)
