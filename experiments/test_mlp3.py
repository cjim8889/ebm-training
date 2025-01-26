import jax
import jax.numpy as jnp

from models.mlp import VelocityFieldThree


key = jax.random.PRNGKey(0)

mlp = VelocityFieldThree(
    key=key,
    n_particles=4,
    n_spatial_dim=2,
    hidden_dim=32,
    depth=2,
    shortcut=True,
)

pos = jax.random.normal(key, (8,))

pos = mlp(pos, jnp.array(0.5), jnp.array(0.125))

print(pos.shape)
