import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import wandb
from src.distributions import (
    AnnealedDistribution,
    MultiDoubleWellEnergy,
    MultivariateGaussian,
)
from src.mcmc.sampling import sample_with_mcmc
from src.models.mlp import TimeVelocityFieldWithPairwiseFeature
from src.training.config import TrainingConfig, TrainingExperimentConfig
from src.training.loss import Particle, calculate_validation_loss_and_plot
from src.training.normalizing_constant import estimate_log_Z_t

jax.config.update("jax_debug_nans", True)

config = TrainingExperimentConfig(
    offline=False,
    training=TrainingConfig(use_shortcut=True, shortcut_size=[32, 64, 128]),
)
run = wandb.init(project="ebm-training-corrected")
artifact = run.use_artifact(
    "iclac/liouville_workshop_corrected/velocity_field_model_vozggoyb:v41", type="model"
)


artifact_dir = artifact.download()

# Create a key for model initialization
key = jax.random.PRNGKey(520019)

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
ts = jnp.linspace(0, 1, 128)

# Generate samples for validation log Z_t
key, subkey = jax.random.split(key)
initial_samples = path_distribution.sample_initial(subkey, 2048)
validation_set = sample_with_mcmc(
    key=subkey,
    initial_samples=initial_samples,
    v_theta=v_theta,
    time_dependent_log_density=path_distribution.time_dependent_log_prob,
    incremental_log_delta=path_distribution.incremental_log_delta,
    ts=ts,
    mcmc_method="smc",
    num_steps=15,
    integration_steps=10,
    eta=0.02,
    ess_threshold=0.5,
)

validation_log_Z_t = estimate_log_Z_t(
    xs=validation_set["positions"],
    weights=validation_set["weights"],
    ts=ts,
    time_derivative_log_density=path_distribution.time_derivative,
).flatten()

validation_particles = Particle(
    x=validation_set["positions"].reshape(-1, 8),
    t=ts.repeat(2048),
    log_Z_t=validation_log_Z_t.repeat(2048),
    d=jnp.ones(2048 * 128) / 128.,
)


validation_loss, plot = calculate_validation_loss_and_plot(
    v_theta=v_theta,
    particles=validation_particles,
    path_distribution=path_distribution,
    ts=ts,
    time_batch_size=8,
    batch_size=2048,
)

plt.savefig("validation_loss.png")
plt.show()