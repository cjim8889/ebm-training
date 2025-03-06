import jax
import jax.numpy as jnp
import jmp

from src.models.mrpe import MolecularRotaryPositionalEmbedding

# Initialize PRNG key
key = jax.random.PRNGKey(42)

# Example parameters
n_particles = 13
n_spatial_dim = 3
input_dim = n_particles * n_spatial_dim
hidden_dim = 32
num_heads = 4
ff_dim = 64
depth = 2
shortcut = False
mp_policy = jmp.Policy(jnp.float32, jnp.float32, jnp.float32)


embedding = MolecularRotaryPositionalEmbedding(
    embedding_size=32,
    key=key,
    freq_key=key,
    theta=1000.0,
    dtype=jnp.float32,
)
# Example input
inputs = jax.random.normal(key, (13, 3))
output = embedding(inputs)

print(output.shape)  # (13 * 3)
print(output)
print(inputs)
