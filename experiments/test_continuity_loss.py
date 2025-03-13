import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import matplotlib.pyplot as plt

import wandb
from src.distributions import (
    AnnealedDistribution,
    MultivariateGaussian,
    QuadraticSmoothedLJ,
)
from src.mcmc.sampling import generate_samples_with_smc
from src.models.transformer_v2 import ParticleTransformerV2
from src.training.loss import batched_epsilon, Particle
from src.training.normalizing_constant import estimate_log_Z_t

jax.config.update("jax_platform_name", "cpu")

key = jax.random.PRNGKey(1234)

run = wandb.init()
artifact = run.use_artifact(
    "iclac/liouville_workshop_corrected/velocity_field_model_pi9i11pv:v2", type="model"
)

artifact_dir = artifact.download()

# Create a key for model initialization
key = jax.random.PRNGKey(0)

mp_policy = jmp.Policy(
    param_dtype=jnp.float32,
    compute_dtype=jnp.float32,
    output_dtype=jnp.float32,
)

v_theta = ParticleTransformerV2(
    n_particles=13,
    n_spatial_dim=3,
    hidden_size=128,
    num_layers=6,
    num_heads=4,
    dropout_rate=None,
    attn_dropout_rate=None,
    key=key,
    mp_policy=mp_policy
)
# Load the saved parameters into the model
v_theta = eqx.tree_deserialise_leaves(f"{artifact_dir}/model.eqx", v_theta)

target_density = QuadraticSmoothedLJ(
    dim=39,
    n_particles=13,
    include_harmonic=True,
    r_min=0.8,
)


initial_density = MultivariateGaussian(dim=39, sigma=2.0)
path_density = AnnealedDistribution(
    initial_density=initial_density,
    target_density=target_density,
    method="linear",
)

ts = jnp.linspace(0, 1, 128)

keys = jax.random.split(key, 3)
key = keys[0]
subkey = keys[1]
covariance_key = keys[2]


initial_samples = path_density.sample_initial(subkey, 16)
samples = generate_samples_with_smc(
    key=subkey,
    initial_samples=initial_samples,
    time_dependent_log_density=path_density.time_dependent_log_prob,
    ts=ts,
    num_steps=15,
    integration_steps=10,
    eta=0.02,
    ess_threshold=0.5,
    estimate_covariance=False,
)

log_Z_t = estimate_log_Z_t(
    xs=samples["positions"],
    weights=samples["weights"],
    ts=ts,
    time_derivative_log_density=path_density.time_derivative,
).flatten()

particles = Particle(
    x=samples["positions"].reshape(-1, 39),
    t=ts.repeat(initial_samples.shape[0]),
    log_Z_t=log_Z_t.repeat(initial_samples.shape[0]),
)


losses = batched_epsilon(
    v_theta,
    particles,
    path_density.score_fn,
    path_density.time_derivative,
)

losses = losses.reshape(
    ts.shape[0], -1
)

# Calculate the mean and variance for each time step (along the batch dimension)
loss_mean = jnp.mean(losses, axis=1)  # shape: (time,)
loss_var = jnp.var(losses, axis=1)    # shape: (time,)
loss_std = jnp.sqrt(loss_var)         # standard deviation

# Optionally, convert JAX arrays to numpy arrays for plotting
loss_mean = jnp.array(loss_mean)
loss_std = jnp.array(loss_std)
ts_np = jnp.array(ts)

# Plot the mean with a shaded region representing ±1 standard deviation
plt.figure(figsize=(10, 6))
plt.plot(ts_np, loss_mean, label="Mean Loss", color="blue")
plt.fill_between(ts_np, loss_mean - loss_std, loss_mean + loss_std, color="blue", alpha=0.3, label="Std Dev")
plt.xlabel("Time")
plt.ylabel("Loss")
plt.title("Loss over Time with Batch Statistics")
plt.legend()
plt.show()




