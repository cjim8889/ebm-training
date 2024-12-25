import blackjax.adaptation
import blackjax.adaptation.mass_matrix
import jax
import blackjax
import blackjax.smc.resampling as resampling
import matplotlib.pyplot as plt

from distributions.multivariate_gaussian import MultivariateGaussian
from distributions.time_dependent_lennard_jones_butler import (
    TimeDependentLennardJonesEnergyButler,
)

# plt.rcParams["figure.dpi"] = 300
# plt.rcParams["figure.figsize"] = [6.0, 4.0]

jax.config.update("jax_platform_name", "cpu")

key = jax.random.PRNGKey(1234)

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


key, subkey = jax.random.split(key)
initial_position = target_density.initialize_position(subkey)
warmup = blackjax.window_adaptation(
    blackjax.hmc,
    target_density.log_prob,
    num_integration_steps=10,
    initial_step_size=1.0,
    target_acceptance_rate=0.8,
    # progress_bar=True,
)


key, warmup_key, sample_key = jax.random.split(key, 3)

(state, parameters), _ = warmup.run(
    warmup_key,
    initial_position,
    num_steps=2000,
)
print("HMC Warmup done")
print("Step size:", parameters["step_size"])

hmc = blackjax.hmc(target_density.log_prob, **parameters)
kernel = jax.jit(hmc.step)

target_ess = 0.6
num_mcmc_steps = 100

tempered = blackjax.adaptive_tempered_smc(
    initial_density.log_prob,
    target_density.log_prob,
    blackjax.hmc.build_kernel(),
    blackjax.hmc.init,
    blackjax.smc.extend_params(parameters),
    resampling_fn=resampling.systematic,
    target_ess=target_ess,
    num_mcmc_steps=num_mcmc_steps,
)

tempered_kernel = jax.jit(tempered.step)

num_particles = 10240
sample_keys = jax.random.split(key, num_particles)
initial_positions = jax.vmap(target_density.initialize_position)(sample_keys)
initial_state = tempered.init(initial_positions)


key, subkey = jax.random.split(key, 2)
n_iter, final_state = smc_inference_loop(subkey, tempered_kernel, initial_state)

print("SMC done")
print("Number of iterations:", n_iter)
# print("Final state:", final_state)

samples = final_state.particles
print("Samples shape:", samples.shape)
fig = target_density.visualise(samples)
plt.show()
