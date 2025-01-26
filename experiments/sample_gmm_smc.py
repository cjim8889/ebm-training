import blackjax.adaptation
import blackjax.adaptation.mass_matrix
import jax
import blackjax
import blackjax.smc.resampling as resampling
import matplotlib.pyplot as plt

from distributions.multivariate_gaussian import MultivariateGaussian
from distributions import GMM


key = jax.random.PRNGKey(1234)


initial_density = MultivariateGaussian(dim=2, mean=0, sigma=20)
key, subkey = jax.random.split(key)
target_density = GMM(subkey, dim=2)

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


warmup = blackjax.window_adaptation(
    blackjax.hmc,
    target_density.log_prob,
    num_integration_steps=10,
    initial_step_size=1.0,
    target_acceptance_rate=0.8,
    progress_bar=True,
)

key, subkey = jax.random.split(key)
initial_position = initial_density.sample(subkey, (1,)).reshape(-1)
print("Initial position:", initial_position.shape)
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
num_mcmc_steps = 10

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
key, sample_key = jax.random.split(key)
initial_positions = initial_density.sample(sample_key, (num_particles,)).reshape(num_particles, -1)
initial_state = tempered.init(initial_positions)


key, subkey = jax.random.split(key, 2)
n_iter, final_state = smc_inference_loop(subkey, tempered_kernel, initial_state)

print("SMC done")
print("Number of iterations:", n_iter)
# print("Final state:", final_state)

samples = final_state.particles
print("Samples shape:", samples.shape)
fig = target_density.visualise(samples)

plt.savefig("gmm_smc.png")
# plt.show()
