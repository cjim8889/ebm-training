import jax
import jax.numpy as jnp
import equinox as eqx
from models.omlp import OptimizedVelocityField

key, subkey = jax.random.split(jax.random.PRNGKey(0))
model = OptimizedVelocityField(
    key=subkey,
    dim=8,
    hidden_size=64,
    n_particles=4,
    n_spatial_dims=2,
    depth=4,
    heads=4,
    shortcut=True,
)
net = jax.jit(
    lambda x, t, d: model(x, t, d),
    static_argnums=None,
)


x = jax.random.uniform(key, (8,))


# Time and distance are concatenated
output = net(x, jnp.array([0.0]), jnp.array([1.0]))

print(output.shape)  # (8,)
