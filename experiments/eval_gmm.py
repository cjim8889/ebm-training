import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import wandb
from distributions import GMM, AnnealedDistribution, MultivariateGaussian
from models.mlp import VelocityFieldTwo
from training.config import TrainingConfig, TrainingExperimentConfig
from utils.eval import aggregate_eval_metrics, evaluate_model, log_metrics

config = TrainingExperimentConfig(
    offline=False,
    training=TrainingConfig(use_shortcut=True, shortcut_size=[32, 64, 128]),
)
run = wandb.init()
artifact = run.use_artifact(
    "iclac/liouville_workshop/velocity_field_model_iwik1qe2:v17", type="model"
)

# "iclac/liouville_workshop/velocity_field_model_iwik1qe2:v17", type="model"

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

path_distribution = AnnealedDistribution(
    initial_density,
    target_density,
)
key, sample_key = jax.random.split(key)

all_eval_results = []
for _ in range(5):
    key, subkey = jax.random.split(key)
    eval_metrics = evaluate_model(
        subkey,
        v_theta,
        config,
        path_distribution,
        target_density,
        1.0,
    )
    all_eval_results.append(eval_metrics)

# Process and log metrics
aggregated_metrics = aggregate_eval_metrics(all_eval_results)
log_metrics(aggregated_metrics, config)
print(aggregated_metrics)

plt.show()
