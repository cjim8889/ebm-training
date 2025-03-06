import functools as ft
import math
import warnings
from functools import partial
from typing import Callable, List, Optional, Union, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
from equinox.nn import Dropout, Linear
from jax import random as jrandom
from jaxtyping import Array, Bool, Float, PRNGKeyArray


def dot_product_attention_weights(
    query: Float[Array, "q_seq qk_size"],
    key: Float[Array, "kv_seq qk_size"],
    distance_bias: Float[Array, "q_seq kv_seq"],
    mask: Optional[Bool[Array, "q_seq kv_seq"]] = None,
) -> Float[Array, "q_seq kv_seq"]:
    query = query / math.sqrt(query.shape[-1])
    logits = jnp.einsum("sd,Sd->sS", query, key)
    logits = logits + distance_bias

    if mask is not None:
        if mask.shape != logits.shape:
            raise ValueError(
                f"mask must have shape (query_seq_length, "
                f"kv_seq_length)=({query.shape[0]}, "
                f"{key.shape[0]}). Got {mask.shape}."
            )
        logits = jnp.where(mask, logits, jnp.finfo(logits.dtype).min)
        logits = cast(Array, logits)

    with jax.numpy_dtype_promotion("standard"):
        dtype = jnp.result_type(logits.dtype, jnp.float32)
    weights = jax.nn.softmax(logits.astype(dtype)).astype(logits.dtype)
    return weights


def dot_product_attention(
    query: Float[Array, "q_seq qk_size"],
    key_: Float[Array, "kv_seq qk_size"],
    value: Float[Array, "kv_seq v_size"],
    pairwise_distances: Float[Array, "q_seq q_seq"],
    w: Float[Array, ""],
    gamma: Float[Array, ""],
    mask: Optional[Bool[Array, "q_seq kv_seq"]] = None,
    dropout: Optional[Dropout] = None,
    *,
    key: Optional[PRNGKeyArray] = None,
    inference: Optional[bool] = None,
) -> Float[Array, "q_seq v_size"]:
    distance_bias = w + jnp.exp(-gamma * pairwise_distances ** 2)
    weights = dot_product_attention_weights(query, key_, distance_bias, mask)
    if dropout is not None:
        weights = dropout(weights, key=key, inference=inference)
    attn = jnp.einsum("sS,Sd->sd", weights, value)
    return attn


class MultiheadAttention(eqx.Module, strict=True):
    r"""
    Computes

    $$\text{MultiheadAttention}(Q, K, V)
      = \sum_i \text{Attention}\left(QW^Q_i, KW^K_i, VW^V_i\right)W^O_i$$

    where:

    - The inputs are
      $Q \in \mathbb{R}^{d_\text{seq} \times d_\text{query}}$,
      $K \in \mathbb{R}^{d_\text{seq} \times d_\text{key}}$,
      $V \in \mathbb{R}^{d_\text{seq} \times d_\text{value}}$.
      These are referred to as query, key, and value respectively. Meanwhile
      $d_\text{seq}$ is the sequence length, and $d_\text{query}$, $d_\text{key}$,
      $d_\text{value}$ are numbers of channels.

    - The trainable weights are
    $W^Q_i \in \mathbb{R}^{d_\text{query} \times d_\text{qk}}$,
    $W^K_i \in \mathbb{R}^{d_\text{key} \times d_\text{qk}}$,
    $W^V_i \in \mathbb{R}^{d_\text{value} \times d_\text{vo}}$,
    $W^O_i \in \mathbb{R}^{d_\text{vo} \times d_\text{output}}$,
    with $i \in \{1, \ldots, h\}$, where $h$ is the number of heads, and $d_\text{qk}$,
    $d_\text{vo}$, $d_\text{output}$ are hyperparameters.

    - $\text{Attention}$ is defined as
      $\text{Attention}(\widetilde{Q}, \widetilde{K}, \widetilde{V})
       = \text{softmax}(\frac{\widetilde{Q}\widetilde{K}^\intercal}
                             {\sqrt{d_\text{qk}}})\widetilde{V}$.

    ??? cite

        [Attention is All You Need](https://arxiv.org/abs/1706.03762)

        ```bibtex
        @inproceedings{vaswani2017attention,
            author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and
                    Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and
                    Kaiser, {\L}ukasz and Polosukhin, Illia},
            booktitle={Advances in Neural Information Processing Systems},
            publisher={Curran Associates, Inc.},
            title={Attention is All You Need},
            volume={30},
            year={2017}
        }
        ```

    !!! faq "FAQ"

        Different software libraries often implement multihead attention in slightly
        different ways. Some of them will or won't add on biases by default. Most of
        them will fix the values of $d_\text{qk}, d_\text{vo}, d_\text{output}$ in
        terms of $d_\text{query}$ or $d_\text{key}$ or $d_\text{value}$. Equinox
        chooses to expose all of these as options.

        Relative to the original
        [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper: our
        $d_\text{qk}$ is their "$d_k$". Our $d_\text{vo}$ is their "$d_\text{v}$". They
        fix $d_\text{query} = d_\text{key} = d_\text{value} = d_\text{output}$ and
        refer to it as "$d_\text{model}$".
    """

    query_proj: Linear
    key_proj: Linear
    value_proj: Linear
    output_proj: Linear
    dropout: Dropout
    w: Float[Array, ""]
    gamma: Float[Array, ""]

    num_heads: int = eqx.field(static=True)
    query_size: int = eqx.field(static=True)
    key_size: int = eqx.field(static=True)
    value_size: int = eqx.field(static=True)
    output_size: int = eqx.field(static=True)
    qk_size: int = eqx.field(static=True)
    vo_size: int = eqx.field(static=True)
    use_query_bias: bool = eqx.field(static=True)
    use_key_bias: bool = eqx.field(static=True)
    use_value_bias: bool = eqx.field(static=True)
    use_output_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        num_heads: int,
        query_size: int,
        key_size: Optional[int] = None,
        value_size: Optional[int] = None,
        output_size: Optional[int] = None,
        qk_size: Optional[int] = None,
        vo_size: Optional[int] = None,
        use_query_bias: bool = False,
        use_key_bias: bool = False,
        use_value_bias: bool = False,
        use_output_bias: bool = False,
        dropout_p: float = 0.0,
        inference: bool = False,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        r"""**Arguments:**

        - `num_heads`: Number of parallel attention heads $h$.
        - `query_size`: Number of input channels for query $Q$.
        - `key_size`: Number of input channels for key $K$. Defaults to `query_size`.
        - `value_size`: Number of input channels for value $V$. Defaults to
            `query_size`.
        - `output_size`: Number of output channels. Defaults to `query_size`.
        - `qk_size`: Number of channels to compare query and key over, per head.
            Defaults to `query_size // num_heads`.
        - `vo_size`: Number of channels to compare attention-weighted value and output
            over, per head. Defaults to `query_size // num_heads`.
        - `use_query_bias`: Whether to use a bias term in the query projections.
        - `use_key_bias`: Whether to use a bias term in the key projections.
        - `use_value_bias`: Whether to use a bias term in the value projections.
        - `use_output_bias`: Whether to use a bias term in the output projection.
        - `dropout_p`: Dropout probability on attention weights.
        - `inference`: Whether to actually apply dropout at all. If `True` then dropout
            is not applied. If `False` then dropout is applied. This may be toggled
            with [`equinox.nn.inference_mode`][] or overridden during
            [`equinox.nn.MultiheadAttention.__call__`][].
        - `dtype`: The dtype to use for all trainable parameters in this layer.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
            on whether JAX is in 64-bit mode.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        dtype = jnp.float32 if dtype is None else dtype
        qkey, kkey, vkey, okey = jrandom.split(key, 4)

        if key_size is None:
            key_size = query_size
        if value_size is None:
            value_size = query_size
        if qk_size is None:
            qk_size = query_size // num_heads
        if vo_size is None:
            vo_size = query_size // num_heads
        if output_size is None:
            output_size = query_size

        self.query_proj = Linear(
            query_size,
            num_heads * qk_size,
            use_bias=use_query_bias,
            dtype=dtype,
            key=qkey,
        )
        self.key_proj = Linear(
            key_size, num_heads * qk_size, use_bias=use_key_bias, dtype=dtype, key=kkey
        )
        self.value_proj = Linear(
            value_size,
            num_heads * vo_size,
            use_bias=use_value_bias,
            dtype=dtype,
            key=vkey,
        )
        self.output_proj = Linear(
            num_heads * vo_size,
            output_size,
            use_bias=use_output_bias,
            dtype=dtype,
            key=okey,
        )
        self.dropout = Dropout(dropout_p, inference=inference)

        self.w = jnp.zeros(())
        self.gamma = jnp.ones(())

        self.num_heads = num_heads
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        self.output_size = output_size
        self.qk_size = qk_size
        self.vo_size = vo_size
        self.use_query_bias = use_query_bias
        self.use_key_bias = use_key_bias
        self.use_value_bias = use_value_bias
        self.use_output_bias = use_output_bias

    @jax.named_scope("eqx.nn.MultiheadAttention")
    def __call__(
        self,
        query: Float[Array, "q_seq q_size"],
        key_: Float[Array, "kv_seq k_size"],
        value: Float[Array, "kv_seq v_size"],
        pairwise_distances: Float[Array, "q_seq q_seq"],
        mask: Union[
            None, Bool[Array, "q_seq kv_seq"], Bool[Array, "num_heads q_seq kv_seq"]
        ] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
        inference: Optional[bool] = None,
        deterministic: Optional[bool] = None,
        process_heads: Optional[
            Callable[
                [
                    Float[Array, "seq_length num_heads qk_size"],
                    Float[Array, "seq_length num_heads qk_size"],
                    Float[Array, "seq_length num_heads vo_size"],
                ],
                tuple[
                    Float[Array, "seq_length num_heads qk_size"],
                    Float[Array, "seq_length num_heads qk_size"],
                    Float[Array, "seq_length num_heads vo_size"],
                ],
            ]
        ] = None,
    ) -> Float[Array, "q_seq o_size"]:
        """**Arguments:**

        - `query`: Query embedding. Should be a JAX array of shape
            `(query_seq_length, query_size)`.
        - `key_`: Key embedding. Should be a JAX array of shape
            `(kv_seq_length, key_size)`.
        - `value`: Value embedding. Should be a JAX array of shape
            `(kv_seq_length, value_size)`.
        - `pairwise_distances`: Pairwise distances between query and key. Should be a JAX array of shape
            `(query_seq_length, kv_seq_length)`.
        - `mask`: Optional mask preventing attention to certain positions. Should either
            be a JAX array of shape `(query_seq_length, kv_seq_length)`, or (for custom
            per-head masking) `(num_heads, query_seq_length, kv_seq_length)`. A value of
            `False` at a position indicates that position should be ignored.
        - `key`: A `jax.random.PRNGKey` used for dropout. Unused if `dropout = 0`.
            (Keyword only argument.)
        - `inference`: As [`equinox.nn.Dropout.__call__`][]. (Keyword only
            argument.)
        - `deterministic`: (Deprecated in favour of `inference`.)
        - `process_heads`: A function that takes in the query, key, and value heads and
            returns new query, key, and value heads. For example, this can be
            used to implement relative positional embeddings -
            see e.g. `RotaryPositionalEmbedding`for an example. (Keyword only argument.)

        **Returns:**

        A JAX array of shape `(query_seq_length, output_size)`.
        """

        if deterministic is not None:
            inference = deterministic
            warnings.warn(
                "MultiheadAttention()(deterministic=...) is deprecated "
                "in favour of MultiheadAttention()(inference=...)"
            )

        query_seq_length, _ = query.shape
        kv_seq_length, _ = key_.shape
        kv_seq_length2, _ = value.shape
        if kv_seq_length != kv_seq_length2:
            # query length can be different
            raise ValueError("key and value must both be sequences of equal length.")

        query_heads = self._project(self.query_proj, query)
        key_heads = self._project(self.key_proj, key_)
        value_heads = self._project(self.value_proj, value)

        if process_heads is not None:
            q_shape, k_shape, v_shape = (
                query_heads.shape,
                key_heads.shape,
                value_heads.shape,
            )
            query_heads, key_heads, value_heads = process_heads(
                query_heads, key_heads, value_heads
            )

            if (
                query_heads.shape != q_shape
                or key_heads.shape != k_shape
                or value_heads.shape != v_shape
            ):
                raise ValueError(
                    "process_heads must not change the shape of the heads."
                )

        attn_fn = partial(
            dot_product_attention, dropout=self.dropout, inference=inference
        )
        keys = None if key is None else jax.random.split(key, query_heads.shape[1])
        if mask is not None and mask.ndim == 3:
            # Batch `mask` and `keys` down their 0-th dimension.
            attn = jax.vmap(attn_fn, in_axes=1, out_axes=1)(
                query_heads, key_heads, value_heads, pairwise_distances, self.w, self.gamma, mask=mask, key=keys,
            )
        else:
            # Batch `keys` down its 0-th dimension.
            attn = jax.vmap(ft.partial(attn_fn, pairwise_distances=pairwise_distances, w=self.w, gamma=self.gamma, mask=mask), in_axes=1, out_axes=1)(
                query_heads, key_heads, value_heads, key=keys
            )
        attn = attn.reshape(query_seq_length, -1)

        return jax.vmap(self.output_proj)(attn)

    def _project(self, proj, x):
        seq_length, _ = x.shape
        projection = jax.vmap(proj)(x)
        return projection.reshape(seq_length, self.num_heads, -1)



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

    def __call__(
        self, 
        xs: Float[Array, "num_particles spatial_dim"],
        t: Float[Array, ""],
        d: Optional[Float[Array, ""]] = None
    ) -> Float[Array, "num_particles embedding_dim"]:
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
    rope_embeddings: eqx.nn.RotaryPositionalEmbedding
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
        theta: float = 10000.0,
    ):
        self.mp_policy = mp_policy
        self.num_heads = num_heads
        self.attention = MultiheadAttention(
            num_heads=num_heads,
            query_size=hidden_size,
            dropout_p=attention_dropout_rate,
            key=key,
            dtype=mp_policy.param_dtype,
        )
        self.layernorm = eqx.nn.LayerNorm(hidden_size, dtype=jnp.float32)
        self.dropout = eqx.nn.Dropout(dropout_rate)
        self.rope_embeddings = eqx.nn.RotaryPositionalEmbedding(
            embedding_size=hidden_size // num_heads,
            theta=theta,
            dtype=mp_policy.param_dtype,
        )

    def __call__(
        self, 
        inputs: Float[Array, "num_particles hidden_size"],
        pairwise_distances: Float[Array, "num_particles num_particles"],
        enable_dropout: bool = False, 
        key: Optional[jax.random.PRNGKey] = None
    ) -> Float[Array, "num_particles hidden_size"]:
        def process_heads(
            query_heads: Float[Array, "num_particles num_heads qk_size"],
            key_heads: Float[Array, "num_particles num_heads qk_size"],
            value_heads: Float[Array, "num_particles num_heads vo_size"]
        ) -> tuple[
            Float[Array, "num_particles num_heads qk_size"],
            Float[Array, "num_particles num_heads qk_size"],
            Float[Array, "num_particles num_heads vo_size"]
        ]:
            query_heads = jax.vmap(self.rope_embeddings,
                                   in_axes=1,
                                   out_axes=1)(query_heads)
            key_heads = jax.vmap(self.rope_embeddings,
                                 in_axes=1,
                                 out_axes=1)(key_heads)

            return query_heads, key_heads, value_heads
        

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
            pairwise_distances=pairwise_distances,
            inference=not enable_dropout,
            key=attn_key,
            process_heads=process_heads,
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

    def __call__(
        self, 
        x: Float[Array, "num_particles hidden_size"],
        enable_dropout: bool = False, 
        key: Optional[jax.random.PRNGKey] = None
    ) -> Float[Array, "num_particles hidden_size"]:
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

    def __init__(self, hidden_size, num_heads, dropout_rate, attn_dropout_rate, key, mp_policy: jmp.Policy, theta: float = 10000.0):
        self.mp_policy = mp_policy
        key1, key2 = jax.random.split(key)
        self.attn = SimplifiedAttentionBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attn_dropout_rate,
            key=key1,
            mp_policy=mp_policy,
            theta=theta,
        )
        self.ffn = EfficientFFN(
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            key=key2,
            mp_policy=mp_policy,
        )

    def __call__(
        self, 
        x: Float[Array, "num_particles hidden_size"],
        pairwise_distances: Float[Array, "num_particles num_particles"],
        enable_dropout: bool = False, 
        key: Optional[jax.random.PRNGKey] = None
    ) -> Float[Array, "num_particles hidden_size"]:
        attn_key, ffn_key = jax.random.split(key) if key is not None else (None, None)

        pairwise_distances = self.mp_policy.cast_to_compute(pairwise_distances)
        x = self.mp_policy.cast_to_compute(x)
        x = self.attn(x, pairwise_distances, enable_dropout, attn_key)
        return self.mp_policy.cast_to_output(self.ffn(x, enable_dropout, ffn_key))


class ParticleTransformerV3(eqx.Module):
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
        theta: float = 10000.0,
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
                theta=theta,
            )
            for k in jax.random.split(l_key, num_layers)
        ]

        self.predictor = eqx.nn.Linear(hidden_size, n_spatial_dim, key=p_key, dtype=mp_policy.param_dtype)

    def __call__(
        self,
        xs: Float[Array, "num_particles * spatial_dim"],
        t: Float[Array, ""],
        d: Optional[Float[Array, ""]] = None,
        *,
        enable_dropout: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Float[Array, "num_particles * spatial_dim"]:
        if self.shortcut and d is None:
            raise ValueError("d must be provided when shortcut is enabled")
        
        xs = self.mp_policy.cast_to_compute(xs)
        t = self.mp_policy.cast_to_compute(t)
        if d is not None:
            d = self.mp_policy.cast_to_compute(d)

        predictor = self.mp_policy.cast_to_compute(self.predictor)

        xs = xs.reshape(-1, self.embedder.n_spatial_dim)
        x = self.embedder(xs, t, d=d if self.shortcut else None)
        pairwise_distances = jnp.linalg.norm(xs[:, None, :] - xs[None, :, :], axis=-1)
        for layer in self.layers:
            x = layer(x, pairwise_distances, enable_dropout, key)
            if key is not None:
                key, _ = jax.random.split(key)

        return self.mp_policy.cast_to_output(jax.vmap(predictor)(x).flatten())
