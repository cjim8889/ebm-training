import jax
import jax.numpy as jnp
import distrax
import chex

from typing import Callable


def rejection_sampling(
    n_samples: int,
    proposal: distrax.Distribution,
    target_log_prob_fn: Callable,
    k: float,
    key: chex.PRNGKey,
) -> chex.Array:
    """Rejection sampling. See Pattern Recognition and ML by Bishop Chapter 11.1"""
    # Note: This currently is not written to work inside of jax.jit or jax.vmap.
    key1, key2, key3 = jax.random.split(key, 3)
    n_samples_propose = n_samples * 10
    z_0, log_q_z0 = proposal._sample_n_and_log_prob(key, n=n_samples_propose)
    u_0 = (
        jax.random.uniform(key=key2, shape=(n_samples_propose,)) * k * jnp.exp(log_q_z0)
    )
    accept = jnp.exp(target_log_prob_fn(z_0)) > u_0
    samples = z_0[accept]
    if samples.shape[0] >= n_samples:
        return samples[:n_samples]
    else:
        required_samples = n_samples - samples.shape[0]
        new_samples = rejection_sampling(
            required_samples, proposal, target_log_prob_fn, k, key3
        )
        samples = jnp.concatenate([samples, new_samples], axis=0)
        return samples


class Energy:
    """
    https://zenodo.org/record/3242635#.YNna8uhKjIW
    """

    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def _energy(self, x):
        raise NotImplementedError()

    def energy(self, x, temperature=None):
        assert x.shape[-1] == self._dim, "`x` does not match `dim`"
        if temperature is None:
            temperature = 1.0
        return self._energy(x) / temperature

    def force(self, x, temperature=None):
        return -jax.grad(lambda x: jnp.sum(self.energy(x, temperature=temperature)))(x)


class DoubleWellEnergy(Energy):
    TIME_DEPENDENT = False

    def __init__(self):
        dim = 2
        a = -0.5
        b = -6.0
        c = 1.0
        super().__init__(dim)
        self._a = a
        self._b = b
        self._c = c

    def _energy(self, x):
        d = x[:, [0]]
        v = x[:, 1:]
        e1 = self._a * d + self._b * d**2 + self._c * d**4
        e2 = jnp.sum(0.5 * v**2, axis=-1, keepdims=True)
        return e1 + e2

    def log_prob(self, x):
        if len(x.shape) == 1:
            x = jnp.expand_dims(x, axis=0)
        return jnp.squeeze(-self.energy(x) - self.log_Z)

    @property
    def log_Z(self):
        log_Z_dim0 = jnp.log(11784.50927)
        log_Z_dim1 = 0.5 * jnp.log(2 * jnp.pi)
        return log_Z_dim0 + log_Z_dim1

    def sample_first_dimension(self, key: chex.Array, n: int) -> chex.Array:
        # see fab.sampling.rejection_sampling_test.py
        if self._a == -0.5 and self._b == -6 and self._c == 1.0:
            # Define target.
            def target_log_prob(x):
                return -(x**4) + 6 * x**2 + 1 / 2 * x

            TARGET_Z = 11784.50927

            # Define proposal params
            component_mix = jnp.array([0.2, 0.8])
            means = jnp.array([-1.7, 1.7])
            scales = jnp.array([0.5, 0.5])

            # Define proposal
            mix = distrax.Categorical(component_mix)
            com = distrax.Normal(means, scales)

            proposal = distrax.MixtureSameFamily(
                mixture_distribution=mix, components_distribution=com
            )

            k = TARGET_Z * 3

            samples = rejection_sampling(
                n_samples=n,
                proposal=proposal,
                target_log_prob_fn=target_log_prob,
                k=k,
                key=key,
            )
            return samples
        else:
            raise NotImplementedError

    def sample(self, key: chex.PRNGKey, shape: chex.Shape):
        if self._a == -0.5 and self._b == -6 and self._c == 1.0:
            assert len(shape) == 1
            key1, key2 = jax.random.split(key=key)
            dim1_samples = self.sample_first_dimension(key=key1, n=shape[0])
            dim2_samples = distrax.Normal(jnp.array(0.0), jnp.array(1.0)).sample(
                seed=key2, sample_shape=shape
            )
            return jnp.stack([dim1_samples, dim2_samples], axis=-1)
        else:
            raise NotImplementedError
