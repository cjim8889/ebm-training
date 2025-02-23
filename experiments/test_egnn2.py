import jax
import jax.numpy as jnp

from models.egnn2 import EGNNWithLearnableNodeFeatures


key = jax.random.PRNGKey(0)

egnn = EGNNWithLearnableNodeFeatures(
    n_node=13,
    hidden_size=32,
    key=key,
    num_layers=2,
    normalize=True,
    num_nearest_neighbors=5,
    geonorm=True,
)

nodes = jnp.zeros((1,))
pos = jax.random.normal(key, (13, 3))

pos = egnn(pos, nodes)

print(pos.shape)
