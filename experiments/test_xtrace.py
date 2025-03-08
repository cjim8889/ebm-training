import jax
import jax.numpy as jnp
import jmp

from src.models.transformer_v3 import ParticleTransformerV3
from src.utils.distributions import (
    divergence_velocity_with_shortcut,
    hutchinson_divergence_velocity2,
)
from src.utils.hutchpp import divergence_velocity_hutchpp
from src.utils.xtrace import divergence_velocity_xtrace

policy = jmp.Policy(
    param_dtype=jnp.float32,
    compute_dtype=jnp.float32,
    output_dtype=jnp.float32,
)
# Setup
key = jax.random.PRNGKey(12345)
mlp = ParticleTransformerV3(
    n_particles=13,
    n_spatial_dim=3,
    hidden_size=128,
    num_layers=3,
    num_heads=4,
    dropout_rate=0.1,
    attn_dropout_rate=0.1,
    key=key,
    shortcut=True,
    mp_policy=policy,
)

batch_size = 1024
key, subkey = jax.random.split(key)
pos_batch = jax.random.normal(subkey, (batch_size, 39))
pos = pos_batch[0]

t = jnp.array(0.5)
sigma = jnp.array(0.125)

estimate, primals  = divergence_velocity_xtrace(mlp, pos, t, n_probes=8, d=sigma)
print(f"Estimate: {estimate}")
print(f"Primals: {primals}")

estimate_hutchpp, _, primals_hutchpp = divergence_velocity_hutchpp(mlp, pos, t, d=sigma, n_probes=8)
print(f"Estimate hutchpp: {estimate_hutchpp}")
print(f"Primals hutchpp: {primals_hutchpp}")

eps = jax.random.rademacher(key, (8, 39), dtype=jnp.float32)
estimate_hutchinson, primals_hutchinson = hutchinson_divergence_velocity2(mlp, pos, t, eps, d=sigma,)
print(f"Estimate hutchinson: {estimate_hutchinson}")
print(f"Primals hutchinson: {primals_hutchinson}")

ground_truth = divergence_velocity_with_shortcut(mlp, pos, t, d=sigma)
print(f"Ground truth: {ground_truth}")