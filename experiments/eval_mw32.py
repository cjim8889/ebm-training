import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import wandb
from distributions import ManyWellEnergy, MultivariateGaussian
from models.mlp import (
    VelocityFieldTwo,
)
from utils.distributions import (
    estimate_kl_divergence,
    reverse_time_flow,
    compute_log_effective_sample_size,
)
from utils.integration import (
    generate_samples_with_log_prob_rk4,
    generate_samples_with_log_prob,
)
from utils.ode import reverse_time_flow_diffrax

run = wandb.init()
artifact = run.use_artifact(
    "iclac/liouville_workshop/velocity_field_model_34iyfj4y:v70", type="model"
)
artifact_dir = artifact.download()

# Create a key for model initialization
key = jax.random.PRNGKey(0)

# Initialize the model with the same architecture
# Note: You'll need to use the same parameters as the original model
# This is just an example - adjust parameters based on your original model
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
# div = estimate_kl_divergence(
#     v_theta,
#     num_samples=1024,
#     key=sample_key,
#     ts=ts,
#     log_prob_p_fn=target_density.log_prob,
#     sample_p_fn=target_density.sample,
#     base_log_prob_fn=initial_density.log_prob,
#     use_shortcut=True,
# )

initial_samples = initial_density.sample(key, (1024,))
samples, log_probs = generate_samples_with_log_prob(
    v_theta,
    initial_samples,
    initial_log_probs=initial_density.log_prob(initial_samples),
    ts=ts,
    use_shortcut=True,
)

# print("Forward KL: ", div)


ground_truth_samples = target_density.sample(key, (1024,))
samples_rev_diffrax, log_probs_rev_diffrax = reverse_time_flow_diffrax(
    v_theta,
    ground_truth_samples,
    1.0,
    dt=1.0 / 128,
    use_shortcut=True,
    exact_logp=True,
)

# Compute log q(x(T)) = log q(x(0)) + accumulated log_probs
base_log_probs = initial_density.log_prob(samples_rev_diffrax)  # Compute log q(x(0))
log_q_x = base_log_probs + log_probs_rev_diffrax

log_w = target_density.log_prob(ground_truth_samples) - log_q_x
# Compute KL divergence: KL(p || q) = E_p[log p(x) - log q(x)]
# kl_divergence = jnp.mean((log_w * jnp.exp(log_w)) - (jnp.exp(log_w) - 1))
kl_divergence = jnp.mean(log_w)

ESS = compute_log_effective_sample_size(
    target_density.log_prob(ground_truth_samples),
    log_q_x,
)

print("Diffrax KL: ", kl_divergence)
print("Diffrax ESS: ", jnp.exp(ESS))

samples_rev, log_probs_rev = reverse_time_flow(
    v_theta,
    ground_truth_samples,
    1.0,
    ts=ts,
    use_shortcut=True,
)

base_log_probs = initial_density.log_prob(samples_rev)
log_q_x = base_log_probs + log_probs_rev

log_w = target_density.log_prob(ground_truth_samples) - log_q_x
kl_divergence = jnp.mean(log_w)

print("Euler KL: ", kl_divergence)


dist_initial = jnp.mean((initial_samples - samples_rev_diffrax) ** 2)
print(dist_initial)


# log_probs_rev_total = log_probs_rev + initial_density.log_prob(samples_rev)

# diff_probs = jnp.mean((log_probs_rev_total - log_probs) ** 2)
# diff_samples = jnp.mean((samples_rev - initial_samples) ** 2)
# print("Diff probs: ", diff_probs)
# print("Diff samples: ", diff_samples)

# fig = target_density.visualise(samples)

# plt.show()
