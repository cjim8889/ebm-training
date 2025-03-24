import jax
import jax.numpy as jnp
import jmp

from src.models.transformer_v4 import ParticleTransformerV4

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


net = ParticleTransformerV4(
    n_particles=n_particles,
    n_spatial_dim=n_spatial_dim,
    hidden_size=hidden_dim,
    num_layers=depth,
    num_heads=num_heads,
    key=key,
    mp_policy=mp_policy,
)

# Example input
inputs = jax.random.normal(key, (n_particles, n_spatial_dim))
output = net(inputs, jnp.array(0))

print(output.shape)  # (13 * 3)
