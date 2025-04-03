import os

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import wandb
from src.distributions import ManyWellEnergy, MultivariateGaussian
from src.models import VelocityFieldTwo
from src.ode import generate_samples


run = wandb.init()
artifact = run.use_artifact(
    "iclac/liouville_workshop_corrected/velocity_field_model_51eigt87:v19", type="model"
)

artifact_dir = artifact.download()

# Create a key for model initialization
key = jax.random.PRNGKey(0)

v_theta = VelocityFieldTwo(
    key=key,
    dim=32,
    hidden_dim=128,
    depth=4,
    shortcut=False,
)
# Load the saved parameters into the model
v_theta = eqx.tree_deserialise_leaves(f"{artifact_dir}/model.eqx", v_theta)

initial_density = MultivariateGaussian(dim=32, sigma=2.0)
target_density = ManyWellEnergy(dim=32)

key, sample_key = jax.random.split(key)

for step in [1, 8, 16, 32, 64, 128]:
    ts = jnp.linspace(0, 1, step)
    samples = generate_samples(
        key=sample_key,
        v_theta=v_theta,
        num_samples=5000,
        ts=ts,
        sample_fn=initial_density.sample,
        use_shortcut=False,
        save_trajectory=False,
    )

    # Save the samples to a local file
    save_path = f"data/mw32_samples_no_shortcut_{step}_steps.npz"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    jnp.savez(
        save_path,
        positions=samples["positions"],
        times=ts,
    )
    print(f"Samples saved to {save_path}")

    fig = target_density.visualise(samples["positions"])
    plt.show()
