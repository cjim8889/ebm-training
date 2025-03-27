import os

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import matplotlib.pyplot as plt

import wandb
from src.distributions import MultivariateGaussian, QuadraticSmoothedLJ
from src.models.transformer_v4 import ParticleTransformerV4
from src.ode import generate_samples

run = wandb.init()
artifact = run.use_artifact(
    "iclac/liouville_workshop_corrected/velocity_field_model_zghxbhqo:v3", type="model"
)


artifact_dir = artifact.download()

# Create a key for model initialization
key = jax.random.PRNGKey(0)

mp_policy = jmp.Policy(
    param_dtype=jnp.float32,
    compute_dtype=jnp.float32,
    output_dtype=jnp.float32,
)

v_theta = ParticleTransformerV4(
    n_particles=13,
    n_spatial_dim=3,
    hidden_size=128,
    num_layers=6,
    num_heads=4,
    key=key,
    mp_policy=mp_policy
)
# Load the saved parameters into the model
v_theta = eqx.tree_deserialise_leaves(f"{artifact_dir}/model.eqx", v_theta)

initial_density = MultivariateGaussian(dim=39, sigma=2.0)
target_density = QuadraticSmoothedLJ(
    dim=39,
    n_particles=13,
    r_min=0.8,
)

key, sample_key = jax.random.split(key)

for step in [128]:
    ts = jnp.linspace(0, 1, step)
    samples = generate_samples(
        key=sample_key,
        v_theta=v_theta,
        num_samples=5000,
        ts=ts,
        sample_fn=initial_density.sample,
        use_shortcut=False,
    )

    # Save the samples to a local file
    save_path = f"data/lj13q_samples_{step}_steps_trajectory.npz"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(samples["positions"][-1].shape)
    jnp.savez(
        save_path,
        positions=samples["positions"],
        times=ts,
    )
    print(f"Samples saved to {save_path}")

    fig = target_density.visualise(samples["positions"][-1])
    plt.show()
