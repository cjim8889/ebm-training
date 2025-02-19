import jax
import jax.numpy as jnp

from models.imlp import InvariantFeatureNet


key = jax.random.PRNGKey(0)

mlp = InvariantFeatureNet(
    key=key,
    n_particles=13,
    n_spatial_dim=3,
    hidden_dim=32,
    depth=3,
    shortcut=True,
    mixed_precision=True,
)

pos = jax.random.normal(key, (13 * 3))

pos = mlp(pos, jnp.array(0.5), jnp.array(0.125))

print(pos.shape)
