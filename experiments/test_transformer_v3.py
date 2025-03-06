import jax
import jax.numpy as jnp
import jmp

from src.models.transformer_v3 import ParticleTransformerV3

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


net = ParticleTransformerV3(
    n_particles=n_particles,
    n_spatial_dim=n_spatial_dim,
    hidden_size=hidden_dim,
    num_layers=depth,
    num_heads=num_heads,
    dropout_rate=0.1,
    attn_dropout_rate=0.1,
    key=key,
    shortcut=True,
    mp_policy=mp_policy,
    theta=100.0,
)

# Example input
inputs = jnp.ones((13, 3))
output = net(inputs, 0, 0.15, enable_dropout=True, key=key)

print(output.shape)  # (13 * 3)
