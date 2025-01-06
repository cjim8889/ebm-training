import jax
import jax.numpy as jnp

from models.egnn import EGNN


key = jax.random.PRNGKey(0)

egnn = EGNN(
    n_node=13,
    hidden_size=32,
    key=key,
    num_layers=2,
)

nodes = jnp.zeros((1,))
pos = jax.random.normal(key, (13, 3))

pos = egnn(pos, nodes)

print(pos.shape)
