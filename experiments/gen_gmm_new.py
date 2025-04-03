import os

import equinox as eqx
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

import wandb
from src.distributions import GMM, MultivariateGaussian
from src.models import VelocityFieldTwo
from src.ode import generate_samples
import jmp


run = wandb.init()
artifact = run.use_artifact(
    "iclac/liouville_workshop_corrected/velocity_field_model_vbkyganu:v19", type="model"
)

artifact_dir = artifact.download()

# Create a key for model initialization
key = jax.random.PRNGKey(0)

policy = jmp.Policy(
    param_dtype=jnp.float32,
    compute_dtype=jnp.float32,
    output_dtype=jnp.float32,
)
# v_theta = ParticleTransformerV5(
#     n_particles=2,
#     n_spatial_dim=1,
#     hidden_size=128,
#     num_layers=4,
#     num_heads=4,
#     mp_policy=policy,
#     key=key,
#     shortcut=False,
# )

v_theta = VelocityFieldTwo(
    key=key,
    dim=2,
    hidden_dim=128,
    depth=4,
    shortcut=False,
)

# Load the saved parameters into the model
v_theta = eqx.tree_deserialise_leaves(f"{artifact_dir}/model.eqx", v_theta)

initial_density = MultivariateGaussian(dim=2, sigma=25.0)
target_density = GMM(key, dim=2)

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
    save_path = f"data/gmm_samples_no_shortcut_{step}_steps.npz"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    jnp.savez(
        save_path,
        positions=samples["positions"],
        times=ts,
    )
    print(f"Samples saved to {save_path}")

    fig = target_density.visualise(samples["positions"])
    plt.show()
