import blackjax.adaptation
import blackjax.adaptation.mass_matrix
import jax
import jax.numpy as jnp
import blackjax
import blackjax.smc.resampling as resampling
import matplotlib.pyplot as plt

from distributions.multivariate_gaussian import MultivariateGaussian
from distributions import MultiDoubleWellEnergy, AnnealedDistribution

from utils.smc import generate_samples_with_smc

# plt.rcParams["figure.dpi"] = 300
# plt.rcParams["figure.figsize"] = [6.0, 4.0]

jax.config.update("jax_platform_name", "cpu")

key = jax.random.PRNGKey(1234)


key, density_key = jax.random.split(key)
initial_density = MultivariateGaussian(dim=8, mean=0, sigma=2.0)
target_density = MultiDoubleWellEnergy(
    dim=8, n_particles=4, data_path_test="data/test_split_DW4.npy", key=density_key
)

path_density = AnnealedDistribution(
    initial_density=initial_density,
    target_density=target_density,
    method="inverse_power",
)
ts = jnp.linspace(0, 1, 128)

key, subkey = jax.random.split(key)
samples = generate_samples_with_smc(
    key=subkey,
    time_dependent_log_density=path_density.time_dependent_log_prob,
    num_samples=5120,
    ts=ts,
    sample_fn=initial_density.sample,
    num_steps=10,
    integration_steps=10,
    eta=0.1,
    rejection_sampling=True,
    ess_threshold=0.5,
    estimate_covariance=False,
    blackjax_hmc=True,
)

print("SMC done")

fig = target_density.visualise(samples["positions"][-1])

plt.show()
# plt.show()
