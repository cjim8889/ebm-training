import os

import equinox as eqx
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

import wandb
from distributions import GMM, MultivariateGaussian
from models.mlp import VelocityFieldTwo
from utils.integration import (
    generate_samples,
    generate_samples_with_diffrax,
    solve_neural_ode_diffrax,
    euler_integrate,
)

run = wandb.init()
artifact = run.use_artifact(
    "iclac/liouville_workshop/velocity_field_model_iwik1qe2:v17", type="model"
)

artifact_dir = artifact.download()

# Create a key for model initialization
key = jax.random.PRNGKey(0)

v_theta = VelocityFieldTwo(
    key=key,
    dim=2,
    hidden_dim=128,
    depth=4,
    shortcut=True,
)
# Load the saved parameters into the model
v_theta = eqx.tree_deserialise_leaves(f"{artifact_dir}/model.eqx", v_theta)

initial_density = MultivariateGaussian(dim=2, sigma=25.0)
target_density = GMM(key, dim=2)

key, sample_key = jax.random.split(key)

ts = jnp.linspace(0, 1, 128)
initial_samples = initial_density.sample(sample_key, (512,))

diffrax_euler_samples, _ = solve_neural_ode_diffrax(
    v_theta=v_theta,
    y0=initial_samples,
    ts=ts,
    use_shortcut=True,
    exact_logp=True,
)

euler_samples = euler_integrate(
    v_theta=v_theta, initial_samples=initial_samples, ts=ts, use_shortcut=True
)
euler_samples = euler_samples[-1]

diff_norms = jnp.linalg.norm(diffrax_euler_samples - euler_samples, axis=1)
mean_diff = jnp.mean(diff_norms)

print("Mean L2 distance between diffrax and Euler samples:", mean_diff)

fig = target_density.visualise(diffrax_euler_samples)
plt.show()

fig = target_density.visualise(euler_samples)
plt.show()
