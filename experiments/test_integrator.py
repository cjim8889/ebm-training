
import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import matplotlib.pyplot as plt

from src.distributions import GMM, MultivariateGaussian
from src.models.transformer_v2 import ParticleTransformerV2
from src.ode import solve_neural_ode_diffrax, solve_neural_ode_euler

# Create a key for model initialization
key = jax.random.PRNGKey(0)


policy = jmp.Policy(
    param_dtype=jnp.float32,
    compute_dtype=jnp.float32,
    output_dtype=jnp.float32,
)
v_theta = ParticleTransformerV2(
    n_particles=2,
    n_spatial_dim=3,
    hidden_size=128,
    num_layers=2,
    num_heads=4,
    dropout_rate=None,
    attn_dropout_rate=None,
    key=key,
    mp_policy=policy,
)

initial_density = MultivariateGaussian(dim=6, sigma=2.0)

key, sample_key = jax.random.split(key)

ts = jnp.linspace(0, 1, 128)
initial_samples = initial_density.sample(sample_key, (128,))

diffrax_result, _ =solve_neural_ode_diffrax(
    v_theta=v_theta,
    y0=initial_samples,
    ts=ts,
    use_shortcut=False,
)

euler_result, _ = solve_neural_ode_euler(
    v_theta=v_theta,
    y0=initial_samples,
    ts=ts,
    use_shortcut=False,
)

allclose = jnp.allclose(diffrax_result, euler_result, atol=1e-5)
print("All close:", allclose)