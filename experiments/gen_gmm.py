import os

import equinox as eqx
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

import wandb
from distributions import GMM, MultivariateGaussian
from models.mlp import VelocityFieldTwo
from utils.integration import generate_samples

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

for step in [1, 8, 16, 32, 64]:
    ts = jnp.linspace(0, 1, step)
    samples = generate_samples(
        sample_key, v_theta, 5000, ts, initial_density.sample, use_shortcut=True
    )

    # Save the samples to a local file
    save_path = f"data/gmm_samples_{step}_steps.npz"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    jnp.savez(
        save_path,
        positions=samples["positions"][-1],
        times=ts,
    )
    print(f"Samples saved to {save_path}")

    fig = target_density.visualise(samples["positions"][-1])
    plt.show()
