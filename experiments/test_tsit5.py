import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from distributions import AnnealedDistribution, ManyWellEnergy, MultivariateGaussian
from models.mlp import VelocityFieldTwo
from utils.ode import solve_neural_ode_diffrax

# Create a key for model initialization
key = jax.random.PRNGKey(0)

v_theta = VelocityFieldTwo(
    key=key,
    dim=32,
    hidden_dim=128,
    depth=4,
    shortcut=True,
)


samples, log_probs = solve_neural_ode_diffrax(
    v_theta,
    y0=jnp.zeros((64, 32)),
    t0=0,
    t1=1,
    dt=0.1,
    save_trajectory=True,
    use_shortcut=True,
    forward=True,
    max_steps=128,
)

print(samples.shape)
print(log_probs.shape)
