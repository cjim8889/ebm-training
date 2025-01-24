import jax
import jax.numpy as jnp
from jax.profiler import start_trace, stop_trace
import time

from models.mlp import VelocityFieldTwo
from utils.distributions import (
    hutchinson_divergence_velocity,
    divergence_velocity_with_shortcut,
    hutchinson_divergence_velocity2,
)


def benchmark_fn(fn, *args):
    # Run forward
    out = fn(*args)
    jax.block_until_ready(out)

    # Run backward
    grad_fn = jax.grad(lambda *a: jnp.sum(fn(*a)))
    grad = grad_fn(*args)
    jax.block_until_ready(grad)

    return out


# Setup
key = jax.random.PRNGKey(0)
mlp = VelocityFieldTwo(
    key=key,
    dim=256,
    hidden_dim=256,
    depth=6,
    shortcut=True,
)

batch_size = 1024
key, subkey = jax.random.split(key)
pos_batch = jax.random.normal(subkey, (batch_size, 256))
pos = pos_batch[0]

t = jnp.array(0.5)
sigma = jnp.array(0.125)

# Start recording
start_trace("/tmp/tensorboard")

try:
    # Individual calculations
    _ = benchmark_fn(
        lambda p: hutchinson_divergence_velocity(key, mlp, p, t, sigma, n_probes=5), pos
    )

    _ = benchmark_fn(
        lambda p: hutchinson_divergence_velocity2(key, mlp, p, t, sigma, n_probes=5),
        pos,
    )

    _ = benchmark_fn(lambda p: divergence_velocity_with_shortcut(mlp, p, t, sigma), pos)

    # Batched calculations
    batch_hutchinson = jax.vmap(
        lambda p: hutchinson_divergence_velocity(key, mlp, p, t, sigma, n_probes=5)
    )
    batch_hutchinson2 = jax.vmap(
        lambda p: hutchinson_divergence_velocity2(key, mlp, p, t, sigma, n_probes=5)
    )
    batch_shortcut = jax.vmap(
        lambda p: divergence_velocity_with_shortcut(mlp, p, t, sigma)
    )

    _ = benchmark_fn(batch_hutchinson, pos_batch)
    _ = benchmark_fn(batch_hutchinson2, pos_batch)
    _ = benchmark_fn(batch_shortcut, pos_batch)

finally:
    # Ensure trace is stopped even if there's an error
    stop_trace()
    jax.profiler.save_device_memory_profile("memory.prof")


print("""
Trace saved to /tmp/tensorboard.
Visualize with:
    tensorboard --logdir=/tmp/tensorboard
""")
