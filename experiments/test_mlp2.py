import jax
import jax.numpy as jnp

from models.mlp import VelocityFieldTwo


key = jax.random.PRNGKey(0)

mlp = VelocityFieldTwo(
    key=key,
    dim=32,
    hidden_dim=32,
    depth=2,
    shortcut=True,
)

pos = jax.random.normal(key, (32,))

pos = mlp(pos, jnp.array(0.5), jnp.array(0.125))

print(pos.shape)
