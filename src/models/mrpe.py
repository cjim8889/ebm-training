import jax
import jax.numpy as jnp
from typing import Optional, Any
from equinox import Module, field
from jax.random import PRNGKey

class MolecularRotaryPositionalEmbedding(Module, strict=True):
    """
    A rotary positional encoding module adapted for molecular data.
    
    Given 3D molecular coordinates (e.g. for 13 particles), this module:
    
      1. Projects each coordinate x ∈ ℝ³ into a higher-dimensional space:
            z = proj * x,   with proj ∈ ℝ^(embedding_size × 3)
      2. For each channel (pair of coordinates) k = 0,...,embedding_size//2 - 1, it
         computes:
            theta^(k) = (u_k ⋅ x) / (theta^(2k/embedding_size))
         where u_k ∈ ℝ³ is a learned (or fixed) frequency vector.
      3. It then applies a rotary transform per channel:
            z'_pair = [z_even, z_odd] rotated by theta^(k)
         via
            z'_pair = z_pair * cos(theta^(k)) + rotate_half(z_pair)* sin(theta^(k))
    
    Finally, the transformed embedding z_rot is returned.
    """
    embedding_size: int = field(static=True)
    theta: float = field(static=True, default=10_000.0)
    dtype: Any = field(static=True, default_factory=jnp.float32)
    proj: jnp.ndarray  # Shape: (embedding_size, 3)
    freq_vectors: jnp.ndarray  # Shape: (embedding_size//2, 3)

    def __init__(
        self,
        embedding_size: int,
        key: PRNGKey,
        freq_key: PRNGKey,
        theta: float = 10_000.0,
        dtype: Optional[Any] = None,
        proj: Optional[jnp.ndarray] = None,
        freq_vectors: Optional[jnp.ndarray] = None,
    ):
        """
        **Arguments:**
        
          - `embedding_size`: The size of the embedding space (must be even).
          - `key`: A JAX PRNGKey used to initialize the projection matrix.
          - `freq_key`: A JAX PRNGKey used to initialize the frequency vectors.
          - `theta`: A scaling constant. Defaults to 10,000.
          - `dtype`: Floating point dtype for the module.
          - `proj`: Optionally provide a (pre‑initialized) projection matrix of shape
                    (embedding_size, 3).
          - `freq_vectors`: Optionally provide learned frequency vectors of shape
                           (embedding_size//2, 3).
        """
        dtype = jnp.float32 if dtype is None else dtype
        if embedding_size % 2 != 0:
            raise ValueError("`embedding_size` must be even.")
        self.embedding_size = embedding_size
        self.theta = theta
        self.dtype = dtype

        # Initialize projection matrix (from ℝ³ to ℝ^(embedding_size))
        if proj is None:
            self.proj = jax.random.normal(key, (embedding_size, 3), dtype=dtype)
        else:
            if proj.shape != (embedding_size, 3):
                raise ValueError("`proj` must have shape (embedding_size, 3).")
            self.proj = proj

        # Initialize frequency vectors (one per channel)
        if freq_vectors is None:
            self.freq_vectors = jax.random.normal(freq_key, (embedding_size // 2, 3), dtype=dtype)
        else:
            if freq_vectors.shape != (embedding_size // 2, 3):
                raise ValueError("`freq_vectors` must have shape (embedding_size//2, 3).")
            self.freq_vectors = freq_vectors

    @staticmethod
    def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
        """
        Splits the last dimension into two halves and rotates them.
        For an input x of shape (..., embedding_size), returns:
          concat(-x[..., embedding_size//2:], x[..., :embedding_size//2])
        """
        d = x.shape[-1]
        d_2 = d // 2
        return jnp.concatenate([-x[..., d_2:], x[..., :d_2]], axis=-1)

    @jax.named_scope("MolecularRotaryPositionalEmbedding")
    def __call__(self, x: jnp.ndarray, *, key: Optional[PRNGKey] = None) -> jnp.ndarray:
        """
        **Arguments:**
        
          - `x`: A JAX array of shape `(num_particles, 3)` representing the 3D coordinates
                 of the particles (e.g. 13 particles).
          - `key`: Ignored; provided for compatibility with the Equinox API.
        
        **Returns:**
        
          A JAX array of shape `(num_particles, embedding_size)` with the rotary positional
          encoding applied.
        """
        if x.ndim != 2 or x.shape[-1] != 3:
            raise ValueError("Input `x` must have shape `(num_particles, 3)`.")
        
        # 1. Project the 3D coordinates into the embedding space.
        #    z has shape: (num_particles, embedding_size)
        z = jnp.dot(x, self.proj.T)

        # 2. For each channel k, compute scaling factor and angles.
        half_d = self.embedding_size // 2
        channel_indices = jnp.arange(half_d)  # Shape: (half_d,)
        # Compute scaling factor s[k] = theta^(2k/embedding_size)
        scaling = self.theta ** ((2 * channel_indices) / self.embedding_size)  # (half_d,)

        # Compute the dot product of each 3D coordinate with each frequency vector.
        # (x has shape (num_particles, 3), freq_vectors has shape (half_d, 3))
        # Resulting in an array of shape (num_particles, half_d)
        angles = jnp.dot(x, self.freq_vectors.T) / scaling

        # 3. Compute cosine and sine values.
        cos_angles = jnp.cos(angles)  # (num_particles, half_d)
        sin_angles = jnp.sin(angles)  # (num_particles, half_d)

        # Duplicate these values to match the embedding size (for even/odd pairs).
        # After repeating, each array has shape: (num_particles, embedding_size)
        cos_angles = jnp.repeat(cos_angles, 2, axis=-1)
        sin_angles = jnp.repeat(sin_angles, 2, axis=-1)

        # 4. Apply the rotary transformation.
        # Using the same pairing scheme as in standard RoPE:
        #      z_rot = z * cos_angles + rotate_half(z) * sin_angles
        z_rot = (z * cos_angles) + (MolecularRotaryPositionalEmbedding.rotate_half(z) * sin_angles)
        return z_rot
