import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os

import wandb
from distributions import ManyWellEnergy, MultivariateGaussian
from models.mlp import VelocityFieldTwo
from training.config import TrainingConfig, TrainingExperimentConfig
from utils.integration import generate_samples_with_Tsit5

config = TrainingExperimentConfig(
    offline=False,
    training=TrainingConfig(use_shortcut=True, shortcut_size=[32, 64, 128]),
)
run = wandb.init()
artifact = run.use_artifact(
    "iclac/liouville_workshop/velocity_field_model_tgsgkczf:v72", type="model"
)

artifact_dir = artifact.download()

# Create a key for model initialization
key = jax.random.PRNGKey(0)

v_theta = VelocityFieldTwo(
    key=key,
    dim=32,
    hidden_dim=128,
    depth=4,
    shortcut=True,
)
# Load the saved parameters into the model
v_theta = eqx.tree_deserialise_leaves(f"{artifact_dir}/model.eqx", v_theta)

initial_density = MultivariateGaussian(dim=32, sigma=2.0)
target_density = ManyWellEnergy(dim=32)

key, sample_key = jax.random.split(key)

ts = jnp.linspace(0, 1, 128)
samples = generate_samples_with_Tsit5(
    sample_key, v_theta, 5000, ts, initial_density.sample, use_shortcut=True
)

# Save the samples to a local file
save_path = "data/mw32_samples.npz"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
jnp.savez(
    save_path,
    positions=samples["positions"][-1],
    times=ts,
)
print(f"Samples saved to {save_path}")

target_density.visualise(samples["positions"][-1])
