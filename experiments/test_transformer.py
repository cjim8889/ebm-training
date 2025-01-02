import jax
import jax.numpy as jnp
from models.transformer import AttentionBlock, TimeVelocityFieldTransformer


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


net = TimeVelocityFieldTransformer(
    n_particles=n_particles,
    n_spatial_dim=n_spatial_dim,
    hidden_size=hidden_dim,
    intermediate_size=ff_dim,
    num_layers=depth,
    num_heads=num_heads,
    dropout_rate=0.1,
    attention_dropout_rate=0.1,
    key=key,
)

# Example input
inputs = jnp.ones((13, 3))
output = net(inputs)

print(output.shape)  # (13, 3)
