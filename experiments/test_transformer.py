import jax
import jax.numpy as jnp
from models.transformer import ParticleTransformer


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


net = ParticleTransformer(
    n_particles=n_particles,
    n_spatial_dim=n_spatial_dim,
    hidden_size=hidden_dim,
    num_layers=depth,
    num_heads=num_heads,
    dropout_rate=0.1,
    attn_dropout_rate=0.1,
    key=key,
    shortcut=True,
)

# Example input
inputs = jnp.ones((13, 3))
output = net(inputs, 0, 0.15)

print(output.shape)  # (13, 3)
