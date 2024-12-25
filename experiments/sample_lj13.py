import blackjax.adaptation
import blackjax.adaptation.mass_matrix
import jax
import jax.numpy as jnp
import blackjax
import matplotlib.pyplot as plt

from distributions import AnnealedDistribution
from distributions.multivariate_gaussian import MultivariateGaussian
from distributions.time_dependent_lennard_jones_butler import (
    TimeDependentLennardJonesEnergyButler,
)

# plt.rcParams["figure.dpi"] = 300
# plt.rcParams["figure.figsize"] = [6.0, 4.0]

jax.config.update("jax_platform_name", "cpu")

key = jax.random.PRNGKey(12391)

target_density = TimeDependentLennardJonesEnergyButler(
    dim=39,
    n_particles=13,
    alpha=0.2,
    sigma=1.0,
    epsilon_val=1.0,
    min_dr=1e-4,
    n=1,
    m=1,
    c=0.5,
    include_harmonic=True,
    log_prob_clip=100.0,
)

initial_density = MultivariateGaussian(dim=39, mean=0.0, sigma=1.0)


def smc_inference_loop(rng_key, smc_kernel, initial_state):
    """Run the temepered SMC algorithm.

    We run the adaptive algorithm until the tempering parameter lambda reaches the value
    lambda=1.

    """

    def cond(carry):
        i, state, _k = carry
        return state.lmbda < 1

    def one_step(carry):
        i, state, k = carry
        k, subk = jax.random.split(k, 2)
        state, _ = smc_kernel(subk, state)
        return i + 1, state, k

    n_iter, final_state, _ = jax.lax.while_loop(
        cond, one_step, (0, initial_state, rng_key)
    )

    return n_iter, final_state


def inference_loop(key: jax.random.PRNGKey, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


inference_loop_multiple_chains = jax.vmap(
    inference_loop,
    in_axes=(0, None, 0, None),
)


key, subkey = jax.random.split(key)
initial_position = target_density.initialize_position(subkey)

print("Initial position:", initial_position)
warmup = blackjax.window_adaptation(
    blackjax.rmhmc,
    target_density.log_prob,
    num_integration_steps=20,
    initial_step_size=0.2,
    target_acceptance_rate=0.9,
    progress_bar=True,
)


key, warmup_key, sample_key = jax.random.split(key, 3)

(state, parameters), _ = warmup.run(
    warmup_key,
    initial_position,
    num_steps=10000,
)
print("Warmup done")
print("Parameters:", parameters)

hmc = blackjax.hmc(target_density.log_prob, **parameters)
kernel = jax.jit(hmc.step)

num_chains = 32
key, sample_key = jax.random.split(key)
sample_keys = jax.random.split(sample_key, num_chains)

initial_states = jax.vmap(hmc.init)(
    jax.vmap(target_density.initialize_position)(sample_keys)
)

vmap_states = inference_loop_multiple_chains(
    sample_keys, kernel, initial_states, 200000
)
_ = vmap_states.position.block_until_ready()

samples = vmap_states.position[
    :,
    190000,
].reshape(-1, 39)

print("Samples shape:", samples.shape)
fig = target_density.visualise(samples)
plt.show()
