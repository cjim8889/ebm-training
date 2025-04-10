from typing import List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
from jaxtyping import Array, Float

from src.utils.models import init_linear_weights, xavier_init, zero_init # Assuming these are defined elsewhere


# Helper modulation function (same as DiT’s modulate)
def modulate(x: Float[Array, " ... "], shift: Float[Array, " ... "], scale: Float[Array, " ... "]) -> Float[Array, " ... "]:
    """Applies adaptive layer norm modulation."""
    return x * (1 + scale) + shift


class TimeEmbeddingV8(eqx.Module):
    """
    Time Embedding module using sinusoidal embeddings for t and
    adaptive layer norm modulation for the step size d.
    """
    net: eqx.nn.MLP
    d_modulator: Optional[eqx.nn.Linear] # Layer to generate shift/scale from d
    frequency_embedding_size: int = eqx.field(static=True)
    embedding_size: int = eqx.field(static=True)
    use_shortcut: bool = eqx.field(static=True) # Static flag to indicate if shortcut is active

    def __init__(self,
            embedding_size: int,
            frequency_embedding_size: int,
            key: jax.random.PRNGKey,
            embedder_width: int = 128,
            embedder_depth: int = 3,
            use_shortcut: bool = True, # Control whether to use d modulation
        ):
        self.embedding_size = embedding_size
        self.frequency_embedding_size = frequency_embedding_size
        self.use_shortcut = use_shortcut

        net_key, d_mod_key = jax.random.split(key, 2)

        # MLP processes only the t sinusoidal embedding
        self.net = eqx.nn.MLP(
            in_size=frequency_embedding_size,
            out_size=embedding_size,
            width_size=embedder_width,
            depth=embedder_depth,
            activation=jax.nn.silu,
            use_bias=True,
            key=net_key,
        )

        # Linear layer to produce shift and scale from d for modulation
        if self.use_shortcut:
            self.d_modulator = eqx.nn.MLP(
                in_size=1, # d is a scalar
                out_size=2 * embedding_size, # Output: shift_d and scale_d
                width_size=embedder_width,
                depth=embedder_depth,
                activation=jax.nn.silu,
                key=d_mod_key,
                use_bias=True  # Bias is crucial for zero initialization
            )
            # Initialize weights and biases to zero for no initial effect
            self.d_modulator = init_linear_weights(self.d_modulator, zero_init, key=d_mod_key)
        else:
            self.d_modulator = None

    @staticmethod
    def timestep_embedding(
        t: Float[Array, "..."],
        dim: int,
        max_period: int = 10000
    ) -> Float[Array, "... dim"]:
        """
        Create sinusoidal timestep embeddings. (Unchanged)
        """
        half = dim // 2
        freqs = jnp.exp(
            -jnp.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half
        )
        t = jnp.asarray(t, dtype=jnp.float32)
        args = t * freqs
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[..., :1])], axis=-1)
        return embedding

    def __call__(self, t: Float[Array, ""], d: Optional[Float[Array, ""]] = None) -> Float[Array, "embedding_size"]:
        """
        Computes the time embedding for t, optionally modulated by d.

        Args:
            t: The current time step.
            d: The step size (optional). If None or if modulate_d is False,
               d has no effect.

        Returns:
            The final time embedding vector.
        """
        # Compute the sinusoidal embeddings for t.
        t_freq = TimeEmbeddingV8.timestep_embedding(t, self.frequency_embedding_size)

        # Pass the t embeddings through the MLP.
        t_emb = self.net(t_freq) # Shape: (embedding_size,)

        # If d is provided and modulation is enabled, compute and apply modulation.
        if d is not None and self.use_shortcut:
            if self.d_modulator is None:
                 # Should not happen if use_shortcut is True, but safeguard
                 raise ValueError("d_modulator is not initialized, but d was provided and use_shortcut is True.")

            # Ensure d is scalar and reshape for the linear layer
            d_scalar = jnp.asarray(d, dtype=jnp.float32).reshape(1)

            # Generate shift and scale parameters from d
            # Output shape: (2 * embedding_size,)
            d_params = self.d_modulator(d_scalar)

            # Split into shift and scale
            # Each has shape: (embedding_size,)
            shift_d, scale_d = jnp.split(d_params, 2, axis=-1)

            # Modulate the t embedding
            final_emb = modulate(t_emb, shift_d, scale_d)
            # print(f"Modulating with d={d:.4f}, initial t_emb norm: {jnp.linalg.norm(t_emb):.4f}, final_emb norm: {jnp.linalg.norm(final_emb):.4f}")
            # print(f"shift_d norm: {jnp.linalg.norm(shift_d):.4f}, scale_d norm: {jnp.linalg.norm(scale_d):.4f}")

        else:
            # If d is not provided or modulation is disabled, use t_emb directly
            final_emb = t_emb

        return final_emb

# --- Rest of the components (largely unchanged, but using TimeEmbeddingV8) ---

class EfficientFFN(eqx.Module):
    """Optimized feed-forward network with parameter reuse."""
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    mp_policy: jmp.Policy = eqx.field(static=True)

    def __init__(self, input_size: int, key: jax.random.PRNGKey, mp_policy: jmp.Policy):
        self.mp_policy = mp_policy
        key1, key2 = jax.random.split(key, 2)
        # Note: In DiT, the FFN hidden dim is 4x input_size
        ffn_hidden_dim = input_size * 4
        self.linear1 = eqx.nn.Linear(input_size, ffn_hidden_dim, key=key1, dtype=mp_policy.param_dtype)
        self.linear2 = eqx.nn.Linear(ffn_hidden_dim, input_size, key=key2, dtype=mp_policy.param_dtype)

    def __call__(
        self,
        x: Float[Array, "num_particles hidden_size"],
    ) -> Float[Array, "num_particles hidden_size"]:
        x = self.mp_policy.cast_to_compute(x)
        # Cast weights only if needed - Equinox handles this internally usually
        # linear1 = self.mp_policy.cast_to_compute(self.linear1)
        # linear2 = self.mp_policy.cast_to_compute(self.linear2)

        residual = x
        # Apply layers per-particle using vmap
        x = jax.vmap(self.linear1)(x)
        x = jax.nn.silu(x) # Apply activation element-wise
        x = jax.vmap(self.linear2)(x)

        x = self.mp_policy.cast_to_param(x) # Cast back before residual add if needed
        x = x + residual
        x = self.mp_policy.cast_to_compute(x) # Cast back to compute

        return x

class AdaptiveLayerNormModulation(eqx.Module):
    """Generates modulation parameters (shift, scale, etc.) from conditioning."""
    linear: eqx.nn.Linear
    count: int = eqx.field(static=True) # Number of parameters to generate per hidden_size dim

    def __init__(self, embedding_size: int, count: int, key: jax.random.PRNGKey, mp_policy: jmp.Policy):
        self.count = count
        # SiLU activation is applied *before* the linear layer in DiT's AdaLNModulation
        # The linear layer maps the SILU'd embedding to the required number of parameters
        self.linear = eqx.nn.Linear(embedding_size, count * embedding_size, key=key, dtype=mp_policy.param_dtype, use_bias=True)

        # Initialize weights to zero - standard practice for AdaLN modulators in DiT
        # This ensures the initial modulation is identity (scale=0, shift=0)
        self.linear = init_linear_weights(self.linear, zero_init, key=key)

    def __call__(self, c: Float[Array, "hidden_size"]) -> List[Float[Array, "hidden_size"]]:
        """
        Args:
            c: Conditioning vector (e.g., time embedding). Shape: (hidden_size,)

        Returns:
            A list of parameter vectors (e.g., [shift, scale, gate,...]),
            each of shape (hidden_size,).
        """
        # Apply SiLU activation to the conditioning vector
        activated_c = jax.nn.silu(c)

        # Generate all parameters with the linear layer
        params = self.linear(activated_c)  # Shape: (count * hidden_size,)

        # Split into individual parameter vectors
        # Returns a list of arrays, each of shape (hidden_size,)
        return jnp.split(params, self.count, axis=-1)


class DiTBlock(eqx.Module):
    """A single block of the Transformer based on DiT."""
    layernorm1: eqx.nn.LayerNorm
    layernorm2: eqx.nn.LayerNorm
    attention: eqx.nn.MultiheadAttention
    rotary_embeddings: eqx.nn.RotaryPositionalEmbedding
    modulation: AdaptiveLayerNormModulation # Generates shift/scale/gate for this block
    ffn: EfficientFFN
    mp_policy: jmp.Policy = eqx.field(static=True)
    embedding_size: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)

    def __init__(self, embedding_size: int, num_heads: int, key: jax.random.PRNGKey, mp_policy: jmp.Policy):
        self.mp_policy = mp_policy
        self.embedding_size = embedding_size
        self.num_heads = num_heads

        # Layer norms applied *before* modulation, as per DiT
        # Note: DiT uses LayerNorm without bias/weight, relying on modulation
        self.layernorm1 = eqx.nn.LayerNorm(embedding_size, use_bias=False, use_weight=False, eps=1e-6, dtype=self.mp_policy.param_dtype)
        self.layernorm2 = eqx.nn.LayerNorm(embedding_size, use_bias=False, use_weight=False, eps=1e-6, dtype=self.mp_policy.param_dtype)

        key1, key2, key3 = jax.random.split(key, 3)

        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=embedding_size, # Query size matches embedding size
            key=key1,
            use_query_bias=True, # Standard MHA biases
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            dtype=self.mp_policy.param_dtype,
        )

        self.ffn = EfficientFFN(
            input_size=embedding_size,
            key=key2,
            mp_policy=mp_policy,
        )

        # RoPE applied per head
        assert embedding_size % num_heads == 0, "Embedding size must be divisible by num_heads"
        head_dim = embedding_size // num_heads
        self.rotary_embeddings = eqx.nn.RotaryPositionalEmbedding(
            embedding_size=head_dim,
            theta=10000.0,
            dtype=self.mp_policy.param_dtype,
        )

        # AdaLN modulation layer: generates 6 params (shift/scale/gate for attn/ffn)
        self.modulation = AdaptiveLayerNormModulation(embedding_size, count=6, key=key3, mp_policy=mp_policy)


    def __call__(self,
            x: Float[Array, "num_particles embedding_size"],
            c: Float[Array, "embedding_size"], # Conditioning (time+d embedding)
        ) -> Float[Array, "num_particles embedding_size"]:

        num_particles = x.shape[0]

        # 1. Generate Modulation Parameters from conditioning c
        #    c comes from TimeEmbeddingV8(t, d)
        #    Shapes: (embedding_size,) for each parameter
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = self.modulation(c)

        # Expand modulation params for broadcasting: (1, embedding_size) -> (num_particles, embedding_size)
        # JAX should broadcast automatically if shapes are (embedding_size,), but being explicit can help clarity.
        # shift_attn = jnp.broadcast_to(shift_attn, x.shape) # Not strictly needed if shapes are correct
        # scale_attn = jnp.broadcast_to(scale_attn, x.shape)
        # gate_attn  = jnp.broadcast_to(gate_attn, x.shape)
        # shift_ffn  = jnp.broadcast_to(shift_ffn, x.shape)
        # scale_ffn  = jnp.broadcast_to(scale_ffn, x.shape)
        # gate_ffn   = jnp.broadcast_to(gate_ffn, x.shape)


        # 2. Attention Branch
        residual_attn = x
        # Apply LayerNorm first, then modulate
        x_norm_attn = jax.vmap(self.layernorm1)(x) # Shape: (num_particles, embedding_size)
        x_mod_attn = modulate(x_norm_attn, shift_attn, scale_attn) # Modulate per particle

        # Define RoPE processing for attention heads
        def process_heads_rope(
            query_heads: Float[Array, "num_particles num_heads qk_size"],
            key_heads: Float[Array, "num_particles num_heads qk_size"],
            value_heads: Float[Array, "num_particles num_heads vo_size"]
        ) -> Tuple[Float[Array, "num_particles num_heads qk_size"],
                  Float[Array, "num_particles num_heads qk_size"],
                  Float[Array, "num_particles num_heads vo_size"]]:
            # Apply RoPE to Query and Key heads
            # vmap over the head dimension (axis=1)
            query_heads = jax.vmap(self.rotary_embeddings, in_axes=1, out_axes=1)(query_heads)
            key_heads = jax.vmap(self.rotary_embeddings, in_axes=1, out_axes=1)(key_heads)
            return query_heads, key_heads, value_heads

        # Apply Multihead Attention with RoPE
        attn_out = self.attention(
            query=x_mod_attn, # Use modulated input for Q, K, V
            key_=x_mod_attn,
            value=x_mod_attn,
            inference=True, # Assuming inference mode, adjust if training needed
            process_heads=process_heads_rope,
        ) # Shape: (num_particles, embedding_size)

        # Apply gating and add residual
        x = residual_attn + gate_attn * attn_out

        # 3. FFN Branch
        residual_ffn = x
        # Apply LayerNorm first, then modulate
        x_norm_ffn = jax.vmap(self.layernorm2)(x) # Shape: (num_particles, embedding_size)
        x_mod_ffn = modulate(x_norm_ffn, shift_ffn, scale_ffn) # Modulate per particle

        # Apply FeedForward network
        ffn_out = self.ffn(x_mod_ffn) # Shape: (num_particles, embedding_size)

        # Apply gating and add residual
        x = residual_ffn + gate_ffn * ffn_out

        return x

class FinalLayer(eqx.Module):
    """ The final layer of the Transformer, adapted from DiT. """
    norm_final: eqx.nn.LayerNorm
    linear: eqx.nn.Linear
    adaLN_modulation: AdaptiveLayerNormModulation # Generates shift/scale for final LN

    def __init__(self, embedding_size: int, output_size: int, key: jax.random.PRNGKey, mp_policy: jmp.Policy):
        # Final LayerNorm without learnable params, modulation provides them
        self.norm_final = eqx.nn.LayerNorm(embedding_size, use_bias=False, use_weight=False, eps=1e-6)
        key1, key2 = jax.random.split(key)

        # Final linear projection to output dimensions
        self.linear = eqx.nn.Linear(embedding_size, output_size, key=key2, dtype=mp_policy.param_dtype, use_bias=True)

        # Modulation layer specific to the final layer (generates 2 params: shift, scale)
        self.adaLN_modulation = AdaptiveLayerNormModulation(embedding_size, count=2, key=key1, mp_policy=mp_policy)

        # Initialize the final projection layer's weights/biases to zero (as done in DiT)
        self.linear = init_linear_weights(self.linear, zero_init, key=key2)

    def __call__(self, x: Float[Array, "num_particles embedding_size"], c: Float[Array, "embedding_size"]) -> Float[Array, "num_particles output_size"]:
        # Generate final shift and scale from conditioning vector c
        shift, scale = self.adaLN_modulation(c) # Shapes: (embedding_size,)

        # Apply final LayerNorm, then modulate
        x_norm = jax.vmap(self.norm_final)(x)
        x_mod = modulate(x_norm, shift, scale) # Modulate per particle

        # Apply final linear projection
        x_out = jax.vmap(self.linear)(x_mod)
        return x_out


class EmbedderBlock(eqx.Module):
    """ Embeds particle spatial coordinates. (Unchanged Structurally) """
    particle_embedder: eqx.nn.MLP
    layernorm: eqx.nn.LayerNorm
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
    ):
        self.mp_policy = mp_policy
        self.particle_embedder = eqx.nn.MLP(
            in_size=n_spatial_dim,
            out_size=embedding_size,
            width_size=embedder_width,
            depth=embedder_depth,
            activation=embedder_activation,
            use_bias=True,
            key=key,
            dtype=mp_policy.param_dtype,
        )
        # Standard LayerNorm after embedding
        self.layernorm = eqx.nn.LayerNorm(shape=(embedding_size,), dtype=jnp.float32) # Usually float32 for stability

    def __call__(
        self,
        xs: Float[Array, "num_particles spatial_dim"],
    ) -> Float[Array, "num_particles embedding_dim"]:
        xs = self.mp_policy.cast_to_compute(xs)
        # embedder = self.mp_policy.cast_to_compute(self.particle_embedder) # Equinox handles this

        # Apply embedder MLP per particle
        embedded = jax.vmap(self.particle_embedder)(xs)
        embedded = self.mp_policy.cast_to_param(embedded) # Cast before LayerNorm if needed

        # Apply LayerNorm per particle
        embedded_norm = jax.vmap(self.layernorm)(embedded)
        embedded_norm = self.mp_policy.cast_to_compute(embedded_norm) # Cast back to compute

        return embedded_norm


###############################################################################
#               ParticleTransformerV8 using new TimeEmbeddingV8               #
###############################################################################
class ParticleTransformerV8(eqx.Module):
    """
    Efficient transformer with DiT-style adaptive layer norm conditioning
    for time (t) and step size (d).
    """
    embedder: EmbedderBlock
    time_embedder: TimeEmbeddingV8 # Use the new time embedder
    layers: List[DiTBlock]
    predictor: FinalLayer
    mp_policy: jmp.Policy = eqx.field(static=True)
    n_spatial_dim: int = eqx.field(static=True)
    embedding_size: int = eqx.field(static=True)

    # def __init__(
    #     self,
    #     n_particles: int, # Often not needed directly by the model structure itself
    #     n_spatial_dim: int,
    #     num_layers: int,
    #     num_heads: int,
    #     embedding_size: int, # Renamed hidden_size to embedding_size for consistency
    #     key: jax.random.PRNGKey,
    #     mp_policy: jmp.Policy,
    #     frequency_embedding_size: int = 256,
    #     embedder_width: int = 128,
    #     embedder_depth: int = 2,
    #     time_embedder_width: int = 128, # Allow separate control
    #     time_embedder_depth: int = 3,
    #     use_shortcut: bool = True, # Flag to enable/disable d modulation
    # ):
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
        embedder_width: int = 128,
        embedder_depth: int = 2,
        embedding_size: int = 128,
        use_shortcut: bool = False,
    ):
        self.mp_policy = mp_policy
        self.n_spatial_dim = n_spatial_dim
        self.embedding_size = embedding_size

        e_key, t_key, l_key, p_key, init_key = jax.random.split(key, 5)

        self.embedder = EmbedderBlock(
            n_spatial_dim=n_spatial_dim,
            embedding_size=embedding_size,
            key=e_key,
            mp_policy=mp_policy,
            embedder_width=embedder_width,
            embedder_depth=embedder_depth,
        )

        # Use the new TimeEmbeddingV8
        self.time_embedder = TimeEmbeddingV8(
            embedding_size=embedding_size,
            frequency_embedding_size=frequency_embedding_size,
            embedder_depth=embedder_depth,
            embedder_width=embedder_width,
            key=t_key,
            use_shortcut=use_shortcut, # Pass the flag
        )

        self.layers = [
            DiTBlock(
                embedding_size=embedding_size,
                num_heads=num_heads,
                key=k,
                mp_policy=mp_policy,
            )
            for k in jax.random.split(l_key, num_layers)
        ]

        self.predictor = FinalLayer(
            embedding_size=embedding_size,
            output_size=n_spatial_dim,
            key=p_key,
            mp_policy=mp_policy
        )

        # Initialize embedder weights (optional, but often helpful)
        # Consider initializing other parts too if needed
        self.embedder = init_linear_weights(self.embedder, xavier_init, init_key, scale=1.)


    def __call__(
        self,
        xs: Float[Array, "num_particles spatial_dim"],
        t: Float[Array, ""],
        d: Optional[Float[Array, ""]] = None,
    ) -> Float[Array, "num_particles spatial_dim"]:
        """
        Forward pass of the Particle Transformer V8.

        Args:
            xs: Input particle positions/features.
            t: Current time step.
            d: Step size (optional, used for modulation if enabled).

        Returns:
            Predicted output (e.g., velocities or displacements) for each particle.
        """
        # Cast inputs if using mixed precision
        xs = self.mp_policy.cast_to_compute(xs)
        t = self.mp_policy.cast_to_compute(t)
        if d is not None:
            d = self.mp_policy.cast_to_compute(d)

        # 1. Embed particle features
        # Ensure input shape is correct (redundant if already correct)
        xs = xs.reshape(-1, self.n_spatial_dim)
        x = self.embedder(xs) # Shape: (num_particles, embedding_size)

        # 2. Compute conditioning vector from time t and step size d
        # c shape: (embedding_size,)
        c = self.time_embedder(t, d)
        c = self.mp_policy.cast_to_compute(c) # Ensure conditioning is in compute dtype

        # 3. Pass through Transformer blocks
        for layer in self.layers:
            # Cast layer to compute dtype if necessary (Equinox usually handles this)
            # layer = self.mp_policy.cast_to_compute(layer)
            x = layer(x, c) # x shape remains (num_particles, embedding_size)

        # 4. Final prediction layer
        # predictor = self.mp_policy.cast_to_compute(self.predictor)
        # predictor takes final x and the conditioning c
        output = self.predictor(x, c) # Shape: (num_particles, n_spatial_dim)

        # Flatten if the original input was flat (common in some uses)
        # Consider the expected output shape. If it should match input xs exactly:
        # output = output.reshape(xs.shape) # Reshape to match original input shape if needed

        # Cast output to final desired dtype
        return self.mp_policy.cast_to_output(output.flatten())

