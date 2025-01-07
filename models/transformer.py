from typing import Dict, List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class PositionalEncoding(eqx.Module):
    d_model: int
    max_len: int = 5000
    pos_encoding: jnp.ndarray

    def __init__(self, d_model, max_len=5000):
        self.d_model = d_model
        position = jnp.arange(0, max_len).reshape(-1, 1)  # (max_len, 1)
        div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(jnp.log(10000.0) / d_model))
        pe = jnp.zeros((max_len, d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        self.pos_encoding = pe  # (max_len, d_model)

    def __call__(self, x):
        # x shape: (seq_len, d_model)
        seq_len = x.shape[0]
        return x + self.pos_encoding[:seq_len, :]


class EmbedderBlock(eqx.Module):
    particle_embedder: eqx.nn.Linear
    layernorm: eqx.nn.LayerNorm

    def __init__(
        self,
        n_particles: int,
        n_spatial_dim: int,
        embedding_size: int,
        key: jax.random.PRNGKey,
    ):
        self.particle_embedder = eqx.nn.Linear(
            in_features=n_spatial_dim + 1,
            out_features=embedding_size,
            use_bias=True,
            key=key,
        )
        self.layernorm = eqx.nn.LayerNorm(shape=(n_particles, embedding_size))

    def __call__(
        self,
        xs: Float[Array, "n_particles n_spatial_dim"],
        t: Float,
    ) -> Float[Array, "n_particles embedding_size"]:
        t = jnp.broadcast_to(t, (xs.shape[0], 1))
        xs = jnp.concatenate([xs, t], axis=-1)
        embedded = jax.vmap(self.particle_embedder)(xs)
        embedded = self.layernorm(embedded)
        return embedded


class AttentionBlock(eqx.Module):
    """A single transformer attention block."""

    attention: eqx.nn.MultiheadAttention
    layernorm: eqx.nn.LayerNorm
    rope_embeddings: eqx.nn.RotaryPositionalEmbedding
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
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            dropout_p=attention_dropout_rate,
            key=key,
        )
        self.rope_embeddings = eqx.nn.RotaryPositionalEmbedding(
            embedding_size=hidden_size // num_heads
        )
        self.layernorm = eqx.nn.LayerNorm(shape=hidden_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        inputs: Float[Array, "seq_len hidden_size"],
        enable_dropout: bool = False,
        key: "jax.random.PRNGKey" = None,
    ) -> Float[Array, "seq_len hidden_size"]:
        def process_heads(
            query_heads: Float[Array, "seq_length num_heads qk_size"],
            key_heads: Float[Array, "seq_length num_heads qk_size"],
            value_heads: Float[Array, "seq_length num_heads vo_size"],
        ) -> tuple[
            Float[Array, "seq_length num_heads qk_size"],
            Float[Array, "seq_length num_heads qk_size"],
            Float[Array, "seq_length num_heads vo_size"],
        ]:
            query_heads = jax.vmap(self.rope_embeddings, in_axes=1, out_axes=1)(
                query_heads
            )
            key_heads = jax.vmap(self.rope_embeddings, in_axes=1, out_axes=1)(key_heads)

            return query_heads, key_heads, value_heads

        attention_key, dropout_key = (
            (None, None) if key is None else jax.random.split(key)
        )

        attention_output = self.attention(
            query=inputs,
            key_=inputs,
            value=inputs,
            inference=not enable_dropout,
            key=attention_key,
            process_heads=process_heads,
        )

        result = attention_output
        result = self.dropout(result, inference=not enable_dropout, key=dropout_key)
        result = result + inputs
        result = jax.vmap(self.layernorm)(result)
        return result


class FeedForwardBlock(eqx.Module):
    """A single transformer feed forward block."""

    mlp: eqx.nn.Linear
    output: eqx.nn.Linear
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout_rate: float,
        key: jax.random.PRNGKey,
    ):
        mlp_key, output_key = jax.random.split(key)
        self.mlp = eqx.nn.Linear(
            in_features=hidden_size, out_features=intermediate_size, key=mlp_key
        )
        self.output = eqx.nn.Linear(
            in_features=intermediate_size, out_features=hidden_size, key=output_key
        )

        self.layernorm = eqx.nn.LayerNorm(shape=hidden_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        inputs: Float[Array, " hidden_size"],
        enable_dropout: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Float[Array, " hidden_size"]:
        # Feed-forward.
        hidden = self.mlp(inputs)
        hidden = jax.nn.gelu(hidden)

        # Project back to input size.
        output = self.output(hidden)
        output = self.dropout(output, inference=not enable_dropout, key=key)

        # Residual and layer norm.
        output += inputs
        output = self.layernorm(output)

        return output


class TransformerLayer(eqx.Module):
    """A single transformer layer."""

    attention_block: AttentionBlock
    ff_block: FeedForwardBlock

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        key: jax.random.PRNGKey,
    ):
        attention_key, ff_key = jax.random.split(key)

        self.attention_block = AttentionBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            key=attention_key,
        )
        self.ff_block = FeedForwardBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout_rate=dropout_rate,
            key=ff_key,
        )

    def __call__(
        self,
        inputs: Float[Array, "seq_len hidden_size"],
        *,
        enable_dropout: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Float[Array, "seq_len hidden_size"]:
        attn_key, ff_key = (None, None) if key is None else jax.random.split(key)
        attention_output = self.attention_block(
            inputs, enable_dropout=enable_dropout, key=attn_key
        )
        seq_len = inputs.shape[0]
        ff_keys = None if ff_key is None else jax.random.split(ff_key, num=seq_len)
        output = jax.vmap(self.ff_block, in_axes=(0, None, 0))(
            attention_output, enable_dropout, ff_keys
        )
        return output


class TimeVelocityFieldTransformer(eqx.Module):
    """Full BERT encoder."""

    embedder_block: EmbedderBlock
    layers: List[TransformerLayer]
    pooler: eqx.nn.Linear
    n_particles: int
    n_spatial_dim: int

    def __init__(
        self,
        n_particles: int,
        n_spatial_dim: int,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        key: jax.random.PRNGKey,
    ):
        embedder_key, layer_key, pooler_key = jax.random.split(key, num=3)
        self.embedder_block = EmbedderBlock(
            n_particles=n_particles,
            n_spatial_dim=n_spatial_dim,
            embedding_size=hidden_size,
            key=embedder_key,
        )
        layer_keys = jax.random.split(layer_key, num=num_layers)
        self.layers = []
        for layer_key in layer_keys:
            self.layers.append(
                TransformerLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                    key=layer_key,
                )
            )

        self.pooler = eqx.nn.Linear(
            in_features=hidden_size, out_features=n_spatial_dim, key=pooler_key
        )

        self.n_particles = n_particles
        self.n_spatial_dim = n_spatial_dim

    def __call__(
        self,
        xs: Float[Array, "..."],
        t: Float[Array, "..."],
        *,
        enable_dropout: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Dict[str, Array]:
        xs = xs.reshape(self.n_particles, self.n_spatial_dim)
        key, l_key = jax.random.split(key) if key is not None else (None, None)

        x = self.embedder_block(xs, t)
        for layer in self.layers:
            cl_key, l_key = (None, None) if l_key is None else jax.random.split(l_key)
            x = layer(x, enable_dropout=enable_dropout, key=cl_key)

        pooled = jax.vmap(self.pooler)(x)

        return pooled.reshape((self.n_particles * self.n_spatial_dim,))
