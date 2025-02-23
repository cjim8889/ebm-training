import blackjax.adaptation
import blackjax.adaptation.mass_matrix
import jax
import jax.numpy as jnp
import blackjax
import blackjax.smc.resampling as resampling
import matplotlib.pyplot as plt

from distributions import TranslationInvariantGaussian, SoftCoreLennardJonesEnergy

# plt.rcParams["figure.dpi"] = 300
# plt.rcParams["figure.figsize"] = [6.0, 4.0]

jax.config.update("jax_platform_name", "cpu")

key = jax.random.PRNGKey(1234)

target_density = SoftCoreLennardJonesEnergy(
    dim=39,
    n_particles=13,
    include_harmonic=True,
)


initial_density = TranslationInvariantGaussian(N=13, D=3, sigma=2.0)


def smc_inference_loop(rng_key, smc_kernel, initial_state, ts):
    """Run the temepered SMC algorithm.

    We run the adaptive algorithm until the tempering parameter lambda reaches the value
    lambda=1.

    """

    def one_step(carry, t):
        i, state, k = carry
        k, subk = jax.random.split(k, 2)
        state, _ = smc_kernel(subk, state, lmbda=t)
        return (i + 1, state, k), state

    _, final_state = jax.lax.scan(one_step, (0, initial_state, rng_key), ts)

    return final_state


key, subkey = jax.random.split(key)
initial_position = initial_density.sample(subkey, (1,)).reshape(-1)
warmup = blackjax.window_adaptation(
    blackjax.hmc,
    target_density.log_prob,
    num_integration_steps=10,
    initial_step_size=1.0,
    target_acceptance_rate=0.6,
    progress_bar=True,
)


key, warmup_key, sample_key = jax.random.split(key, 3)

(state, parameters), _ = warmup.run(
    warmup_key,
    initial_position,
    num_steps=10000,
)

print("HMC Warmup done")
print("Step size:", parameters["step_size"])
parameters["inverse_mass_matrix"] = jnp.eye(39)
print(parameters)

hmc = blackjax.hmc(target_density.log_prob, **parameters)
kernel = jax.jit(hmc.step)

target_ess = 0.6
num_mcmc_steps = 20

tempered = blackjax.tempered_smc(
    initial_density.log_prob,
    target_density.log_prob,
    blackjax.hmc.build_kernel(),
    blackjax.hmc.init,
    blackjax.smc.extend_params(parameters),
    resampling_fn=resampling.systematic,
    num_mcmc_steps=num_mcmc_steps,
)

tempered_kernel = jax.jit(tempered.step)

num_particles = 10240
key, sample_key = jax.random.split(key)
initial_positions = initial_density.sample(sample_key, (num_particles,))
initial_state = tempered.init(initial_positions)

ts = jnp.linspace(0.0, 1.0, 128)

key, subkey = jax.random.split(key, 2)
final_state = smc_inference_loop(subkey, tempered_kernel, initial_state, ts)

print("SMC done")
# print("Final state:", final_state)

samples = final_state.particles[-1]
print("Samples shape:", samples.shape)
fig = target_density.visualise(samples)

plt.savefig("lj13c.png")
# plt.show()
