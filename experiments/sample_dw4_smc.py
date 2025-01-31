import blackjax.adaptation
import blackjax.adaptation.mass_matrix
import jax
import jax.numpy as jnp
import blackjax
import blackjax.smc.resampling as resampling
import matplotlib.pyplot as plt

from distributions.multivariate_gaussian import MultivariateGaussian
from distributions import MultiDoubleWellEnergy

# plt.rcParams["figure.dpi"] = 300
# plt.rcParams["figure.figsize"] = [6.0, 4.0]

jax.config.update("jax_platform_name", "cpu")

key = jax.random.PRNGKey(1234)


key, density_key = jax.random.split(key)
initial_density = MultivariateGaussian(dim=8, mean=0, sigma=3.0)
target_density = MultiDoubleWellEnergy(
    dim=8, n_particles=4, data_path_test="data/test_split_DW4.npy", key=density_key
)


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


# key, subkey = jax.random.split(key)
# initial_position = initial_density.sample(subkey, (1,))
# warmup = blackjax.window_adaptation(
#     blackjax.hmc,
#     target_density.log_prob,
#     num_integration_steps=15,
#     initial_step_size=1.0,
#     target_acceptance_rate=0.6,
#     progress_bar=True,
# )


# key, warmup_key, sample_key = jax.random.split(key, 3)

# (state, parameters), _ = warmup.run(
#     warmup_key,
#     initial_position,
#     num_steps=10000,
# )
# print("HMC Warmup done")
# print("Step size:", parameters["step_size"])
# print("Num integration steps:", parameters["num_integration_steps"])
parameters = {
    "step_size": 0.1,
    "num_integration_steps": 15,
    "inverse_mass_matrix": jnp.eye(8),
}

hmc = blackjax.hmc(target_density.log_prob, **parameters)
kernel = jax.jit(hmc.step)

target_ess = 0.6
num_mcmc_steps = 20

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
key, subkey = jax.random.split(key)
initial_positions = initial_density.sample(subkey, (num_particles,))
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
plt.savefig("dw4.png")
# plt.show()
