import os

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import wandb
from distributions import (
    AnnealedDistribution,
    MultiDoubleWellEnergy,
    MultivariateGaussian,
)
from models.mlp import TimeVelocityFieldWithPairwiseFeature
from training.config import TrainingConfig, TrainingExperimentConfig
from utils.integration import generate_samples

jax.config.update("jax_debug_nans", True)

config = TrainingExperimentConfig(
    offline=False,
    training=TrainingConfig(use_shortcut=True, shortcut_size=[32, 64, 128]),
)
run = wandb.init()
artifact = run.use_artifact(
    "iclac/liouville_workshop/velocity_field_model_tbyio36t:v17", type="model"
)


artifact_dir = artifact.download()

# Create a key for model initialization
key = jax.random.PRNGKey(238)

v_theta = TimeVelocityFieldWithPairwiseFeature(
    key=key,
    n_particles=4,
    n_spatial_dim=2,
    hidden_dim=512,
    depth=4,
    shortcut=True,
)
# Load the saved parameters into the model
v_theta = eqx.tree_deserialise_leaves(f"{artifact_dir}/model.eqx", v_theta)

initial_density = MultivariateGaussian(dim=8, sigma=2.0)
target_density = MultiDoubleWellEnergy(
    dim=8,
    n_particles=4,
    data_path_test="data/test_split_DW4.npy",
    n_samples_eval=2048,
)

path_distribution = AnnealedDistribution(
    initial_density,
    target_density,
)
key, sample_key = jax.random.split(key)

for step in [1, 8, 16, 32, 64, 128]:
    ts = jnp.linspace(0, 1, step)
    samples = generate_samples(
        sample_key, v_theta, 5000, ts, initial_density.sample, use_shortcut=True
    )

    # Save the samples to a local file
    save_path = f"data/dw4_samples_{step}_steps.npz"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    jnp.savez(
        save_path,
        positions=samples["positions"][-1],
        times=ts,
    )
    print(f"Samples saved to {save_path}")

    fig = target_density.visualise(samples["positions"][-1])
    plt.show()
