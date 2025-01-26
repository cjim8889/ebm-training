import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


from distributions import ManyWellEnergy


key = jax.random.PRNGKey(1234)
target_density = ManyWellEnergy(dim=32)

samples = target_density.sample(key, (2560,))

fig = target_density.visualise(samples)
plt.show()
