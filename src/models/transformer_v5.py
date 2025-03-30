from typing import List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
from jaxtyping import Array, Float

from src.utils.models import init_linear_weights, xavier_init, zero_init


# Helper modulation function (same as DiT’s modulate)
def modulate(x: Float[Array, " ... "], shift: Float[Array, " ... "], scale: Float[Array, " ... "]) -> Float[Array, " ... "]:
    return x * (1 + scale) + shift


class TimeEmbedding(eqx.Module):
    net: eqx.nn.MLP
    frequency_embedding_size: int = eqx.field(static=True)

    def __init__(self, 
            hidden_size: int, 
            frequency_embedding_size: int, 
            key: jax.random.PRNGKey, 
            embedder_width: int = 128, 
            embedder_depth: int = 3,
            shortcut: bool = False,
        ):


        self.net = eqx.nn.MLP(
            in_size=frequency_embedding_size * 2 if shortcut else frequency_embedding_size,
            out_size=hidden_size,
            width_size=embedder_width,
            depth=embedder_depth,
            activation=jax.nn.silu,
            use_bias=True,
            key=key,
        )

        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: Float[Array, "..."], 
        dim: int, 
        max_period: int = 10000
    ) -> Float[Array, "... dim"]:
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: A 1-D array of N indices (may be fractional).
            dim: The dimension of the output.
            max_period: Controls the minimum frequency of the embeddings.

        Returns:
            A (N, dim) array of positional embeddings.
        """
        half = dim // 2
        # Compute frequencies.
        freqs = jnp.exp(
            -jnp.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half
        )
        # Ensure t is a scalar float32.
        t = jnp.asarray(t, dtype=jnp.float32)
        args = t * freqs
        # Concatenate cosine and sine embeddings.
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        # Pad with an extra zero if dim is odd.
        if dim % 2:
            embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[..., :1])], axis=-1)
        return embedding
    
    def __call__(self, t: Float[Array, ""], d: Optional[Float[Array, ""]] = None,) -> Float[Array, "hidden_size"]:
        # Compute the sinusoidal embeddings.
        t_freq = TimeEmbedding.timestep_embedding(t, self.frequency_embedding_size)

        # If d is provided, concatenate it with the embeddings.
        if d is not None:
            d_freq = TimeEmbedding.timestep_embedding(d, self.frequency_embedding_size)
            t_freq = jnp.concatenate([t_freq, d_freq], axis=-1)
        
        # Pass the embeddings through the MLP.
        t_emb = self.net(t_freq)
        return t_emb
    
class EfficientFFN(eqx.Module):
    """Optimized feed-forward network with parameter reuse."""
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    mp_policy: jmp.Policy = eqx.field(static=True)

    def __init__(self, input_size: int, hidden_size: int, key: jax.random.PRNGKey, mp_policy: jmp.Policy):
        self.mp_policy = mp_policy
        key1, key2 = jax.random.split(key, 2)
        self.linear1 = eqx.nn.Linear(input_size, hidden_size, key=key1, dtype=mp_policy.param_dtype)
        self.linear2 = eqx.nn.Linear(hidden_size, input_size, key=key2, dtype=mp_policy.param_dtype)

    def __call__(
        self, 
        x: Float[Array, "num_particles hidden_size"],
    ) -> Float[Array, "num_particles hidden_size"]:
        x = self.mp_policy.cast_to_compute(x)
        linear1 = self.mp_policy.cast_to_compute(self.linear1)
        linear2 = self.mp_policy.cast_to_compute(self.linear2)
        
        residual = x
        x = jax.vmap(linear1)(x)
        x = jax.nn.gelu(self.mp_policy.cast_to_param(x))
        x = jax.vmap(linear2)(x) + residual
        x = self.mp_policy.cast_to_compute(x)

        return x

###############################################################################
#      Adaptive LayerNorm Modulation (produces 6 parameters per block)       #
###############################################################################
class AdaptiveLayerNormModulation(eqx.Module):
    linear: eqx.nn.Linear
    count: int = eqx.field(static=True)

    def __init__(self, hidden_size: int, count: int, key: jax.random.PRNGKey, mp_policy: jmp.Policy):
        # Mimic: SiLU -> Linear(hidden_size, count*hidden_size)
        self.count = count
        self.linear = eqx.nn.Linear(hidden_size, count * hidden_size, key=key, dtype=mp_policy.param_dtype)

        # Initialize weights to zero
        self.linear = init_linear_weights(self.linear, xavier_init, key=key)

    def __call__(self, c: Float[Array, "hidden_size"]) -> List[Float[Array, "hidden_size"]]:
        # c: conditioning vector (shape: (hidden_size,) or (batch, hidden_size))
        params = self.linear(jax.nn.silu(c))  # shape: (count*hidden_size,)
        return jnp.split(params, self.count, axis=-1)  # returns [shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn]



class DiTBlock(eqx.Module):
    layernorm1: eqx.nn.LayerNorm
    layernorm2: eqx.nn.LayerNorm
    attention: eqx.nn.MultiheadAttention
    rotary_embeddings: eqx.nn.RotaryPositionalEmbedding
    modulation: AdaptiveLayerNormModulation
    ffn: EfficientFFN
    mp_policy: jmp.Policy = eqx.field(static=True)

    def __init__(self, hidden_size: int, num_heads: int, key: jax.random.PRNGKey, mp_policy: jmp.Policy, mlp_ratio: float = 4.0):
        self.mp_policy = mp_policy
        self.layernorm1 = eqx.nn.LayerNorm(hidden_size, dtype=self.mp_policy.param_dtype)
        self.layernorm2 = eqx.nn.LayerNorm(hidden_size, dtype=self.mp_policy.param_dtype)

        key1, key2, key3 = jax.random.split(key, 3)
        
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=hidden_size,
            key=key1,
            dtype=self.mp_policy.param_dtype,
        )
        
        self.ffn = EfficientFFN(
            input_size=hidden_size,
            hidden_size=int(hidden_size * mlp_ratio),
            key=key2,
            mp_policy=mp_policy,
        )

        self.rotary_embeddings = eqx.nn.RotaryPositionalEmbedding(
            embedding_size=hidden_size // num_heads,
            theta=10000.0,
            dtype=self.mp_policy.param_dtype,
        )

        self.modulation = AdaptiveLayerNormModulation(hidden_size, count=6, key=key3, mp_policy=mp_policy)


    def __call__(self, 
            x: Float[Array, "num_particles hidden_size"],
            c: Float[Array, "hidden_size"],
        ) -> Float[Array, "num_particles hidden_size"]:
        def process_heads(
            query_heads: Float[Array, "num_particles num_heads qk_size"],
            key_heads: Float[Array, "num_particles num_heads qk_size"],
            value_heads: Float[Array, "num_particles num_heads vo_size"]
        ) -> tuple[Float[Array, "num_particles num_heads qk_size"], 
                  Float[Array, "num_particles num_heads qk_size"], 
                  Float[Array, "num_particles num_heads vo_size"]]:
            query_heads = jax.vmap(self.rotary_embeddings, in_axes=1, out_axes=1)(query_heads)
            key_heads = jax.vmap(self.rotary_embeddings, in_axes=1, out_axes=1)(key_heads)
            return query_heads, key_heads, value_heads
        
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = self.modulation(c)
        # Expand to per-particle shape:
        shift_attn = jnp.broadcast_to(shift_attn, x.shape)
        scale_attn = jnp.broadcast_to(scale_attn, x.shape)
        gate_attn  = jnp.broadcast_to(gate_attn, x.shape)
        shift_ffn  = jnp.broadcast_to(shift_ffn, x.shape)
        scale_ffn  = jnp.broadcast_to(scale_ffn, x.shape)
        gate_ffn   = jnp.broadcast_to(gate_ffn, x.shape)


        # Attention branch:
        x_norm_attn = jax.vmap(self.layernorm1)(x)
        x_mod_attn = modulate(x_norm_attn, shift_attn, scale_attn)
        attn_out = self.attention(
            query=x_mod_attn,
            key_=x_mod_attn,
            value=x_mod_attn,
            inference=True,
            process_heads=process_heads,
        )
        x = x + gate_attn * attn_out

        # FFN branch:
        x_norm_ffn = jax.vmap(self.layernorm2)(x)
        x_mod_ffn = modulate(x_norm_ffn, shift_ffn, scale_ffn)
        ffn_out = self.ffn(x_mod_ffn)
        x = x + gate_ffn * ffn_out

        return x
    
###############################################################################
#                   New DiT-Style Transformer Layer                           #
###############################################################################
class FinalLayer(eqx.Module):
    """
    The final layer of DiT.
    """
    norm_final: eqx.nn.LayerNorm
    linear: eqx.nn.Linear
    adaLN_modulation: AdaptiveLayerNormModulation

    def __init__(self, hidden_size: int, output_size: int, key: jax.random.PRNGKey, mp_policy: jmp.Policy):
        self.norm_final = eqx.nn.LayerNorm(hidden_size, use_bias=False, use_weight=False, eps=1e-6)
        key1, key2 = jax.random.split(key)
        self.linear = eqx.nn.Linear(hidden_size, output_size, key=key2, dtype=mp_policy.param_dtype)
        self.adaLN_modulation = AdaptiveLayerNormModulation(hidden_size, count=2, key=key1, mp_policy=mp_policy)
        # Initialize weights to zero
        self.linear = init_linear_weights(self.linear, zero_init, key=key2)

    def __call__(self, x: Float[Array, "num_particles hidden_size"], c: Float[Array, "hidden_size"]) -> Float[Array, "num_particles output_size"]:
        shift, scale = self.adaLN_modulation(c)
        # Expand to per-particle shape:
        shift = jnp.broadcast_to(shift, x.shape)
        scale = jnp.broadcast_to(scale, x.shape)

        x = modulate(jax.vmap(self.norm_final)(x), shift, scale)
        x = jax.vmap(self.linear)(x)
        return x


###############################################################################
#                (Unchanged) Embedder, Attention, and FFN Blocks              #
###############################################################################
class EmbedderBlock(eqx.Module):
    particle_embedder: eqx.nn.MLP
    layernorm: eqx.nn.LayerNorm
    shortcut: bool = eqx.field(static=True)
    mp_policy: jmp.Policy = eqx.field(static=True)

    def __init__(
        self,
        n_spatial_dim: int,
        embedding_size: int,
        key: jax.random.PRNGKey,
        mp_policy: jmp.Policy,
        embedder_width: int = 128,
        embedder_depth: int = 3,
        embedder_activation: callable = jax.nn.silu,
        shortcut: bool = False,
    ):
        self.shortcut = shortcut
        self.mp_policy = mp_policy
        in_dim = n_spatial_dim
        self.particle_embedder = eqx.nn.MLP(
            in_size=in_dim,
            out_size=embedding_size,
            width_size=embedder_width,
            depth=embedder_depth,
            activation=embedder_activation,
            use_bias=True,
            key=key,
            dtype=mp_policy.param_dtype,
        )
        self.layernorm = eqx.nn.LayerNorm(shape=(embedding_size,), dtype=jnp.float32)

    def __call__(
        self, 
        xs: Float[Array, "num_particles spatial_dim"],
    ) -> Float[Array, "num_particles embedding_dim"]:
        input = xs

        input = self.mp_policy.cast_to_compute(input)
        embedder = self.mp_policy.cast_to_compute(self.particle_embedder)
        embedded = jax.vmap(embedder)(input)
        return self.mp_policy.cast_to_output(embedded)


###############################################################################
#                ParticleTransformerV4 using DiT-style layers                  #
###############################################################################
class ParticleTransformerV5(eqx.Module):
    """
    Efficient transformer with DiT-style adaptive layer norm conditioning.
    """
    embedder: EmbedderBlock
    time_embedder: TimeEmbedding
    layers: List[DiTBlock]
    predictor: FinalLayer
    shortcut: bool = eqx.field(static=True)
    mp_policy: jmp.Policy = eqx.field(static=True)
    n_spatial_dim: int = eqx.field(static=True)
    
    def __init__(
        self,
        n_particles: int,
        n_spatial_dim: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        key: jax.random.PRNGKey,
        mp_policy: jmp.Policy,
        frequency_embedding_size: int = 256,
        shortcut: bool = False,
    ):
        self.shortcut = shortcut
        self.mp_policy = mp_policy
        self.n_spatial_dim = n_spatial_dim

        e_key, l_key, p_key, init_key, t_key, d_key = jax.random.split(key, 6)

        self.embedder = EmbedderBlock(
            n_spatial_dim=n_spatial_dim,
            embedding_size=hidden_size,
            key=e_key,
            shortcut=shortcut,
            mp_policy=mp_policy,
        )

        self.time_embedder = TimeEmbedding(
            hidden_size=hidden_size,
            frequency_embedding_size=frequency_embedding_size,
            key=t_key,
            shortcut=shortcut,
        )

        self.layers = [
            DiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                key=k,
                mp_policy=mp_policy,
            )
            for k in jax.random.split(l_key, num_layers)
        ]

        self.predictor = FinalLayer(hidden_size, n_spatial_dim, key=p_key, mp_policy=mp_policy)

        key_1 = jax.random.split(init_key, 1)[0]
        self.embedder = init_linear_weights(self.embedder, xavier_init, key_1, scale=0.1)

    def __call__(
        self,
        xs: Float[Array, "num_particles spatial_dim"],
        t: Float[Array, ""],
        d: Optional[Float[Array, ""]] = None,
    ) -> Float[Array, "num_particles spatial_dim"]:
        if self.shortcut and d is None:
            raise ValueError("d must be provided when shortcut is enabled")
        
        xs = self.mp_policy.cast_to_compute(xs)
        t = self.mp_policy.cast_to_compute(t)
        if d is not None:
            d = self.mp_policy.cast_to_compute(d)

        predictor = self.mp_policy.cast_to_compute(self.predictor)

        xs = xs.reshape(-1, self.n_spatial_dim)
        x = self.embedder(xs)
        c = self.time_embedder(t, d)

        for layer in self.layers:
            x = layer(x, c)

        return self.mp_policy.cast_to_output(predictor(x, c).flatten())
