import jax
import jax.numpy as jnp
import equinox as eqx
import distrax
import blackjax
import wandb
import optax
import chex
import pickle
import argparse
from typing import Optional, Callable, Union, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


# Base Target class and utility functions
class Target:
    """Base class for distributions"""

    def __init__(
        self,
        dim: int,
        log_Z: Optional[float],
        can_sample: bool,
        n_plots: int,
        n_model_samples_eval: int,
        n_target_samples_eval: Optional[int],
    ):
        self.n_model_samples_eval = n_model_samples_eval
        self.n_target_samples_eval = n_target_samples_eval
        self._dim = dim
        self._log_Z = log_Z
        self._n_plots = n_plots
        self._can_sample = can_sample

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def n_plots(self) -> int:
        return self._n_plots

    @property
    def can_sample(self) -> bool:
        return self._can_sample

    @property
    def log_Z(self) -> Union[int, None]:
        return self._log_Z

    def log_prob(self, value: chex.Array) -> chex.Array:
        raise NotImplementedError


class MultivariateGaussian(Target):
    def __init__(
        self, dim: int = 2, sigma: float = 1.0, plot_bound_factor: float = 3.0
    ):
        super().__init__(
            dim=dim,
            log_Z=0.0,
            can_sample=True,
            n_plots=1,
            n_model_samples_eval=1000,
            n_target_samples_eval=1000,
        )
        self.sigma = jnp.asarray(sigma)
        if self.sigma.ndim == 0:
            scale_diag = jnp.full((dim,), self.sigma)
        else:
            if self.sigma.shape[0] != dim:
                raise ValueError(
                    f"Sigma shape {self.sigma.shape} does not match dimension {dim}."
                )
            scale_diag = self.sigma

        self.distribution = distrax.MultivariateNormalDiag(
            loc=jnp.zeros(dim), scale_diag=scale_diag
        )

    def log_prob(self, x: chex.Array) -> chex.Array:
        return self.distribution.log_prob(x)

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape = ()) -> chex.Array:
        return self.distribution.sample(seed=seed, sample_shape=sample_shape)


class BetaTarget(Target):
    def __init__(self, X: chex.Array, y: chex.Array, dim: int = 2) -> None:
        super().__init__(
            dim=dim,
            log_Z=0.0,
            can_sample=True,
            n_plots=1,
            n_model_samples_eval=1000,
            n_target_samples_eval=1000,
        )
        self.X = X
        self.y = y
        self.prior = MultivariateGaussian(dim=dim, sigma=1.0)

    @eqx.filter_jit
    def log_prob(self, beta: chex.Array) -> chex.Array:
        prior_log_prob = self.prior.log_prob(beta)
        z = jnp.einsum("ij,j->i", self.X, beta)
        target_log_prob = jnp.sum(self.y * z - jax.nn.softplus(z))
        return prior_log_prob + target_log_prob


class AnnealedDistribution(Target):
    def __init__(
        self, initial_distribution: Target, target_distribution: Target, dim: int = 2
    ):
        super().__init__(
            dim=dim,
            log_Z=0.0,
            can_sample=False,
            n_plots=1,
            n_model_samples_eval=1000,
            n_target_samples_eval=1000,
        )
        self.initial_distribution = initial_distribution
        self.target_distribution = target_distribution

    def log_prob(self, xs: chex.Array) -> chex.Array:
        return self.time_dependent_log_prob(xs, 1.0)

    def time_dependent_log_prob(self, xs: chex.Array, t: chex.Array) -> chex.Array:
        initial_prob = (1 - t) * self.initial_distribution.log_prob(xs)
        target_prob = t * self.target_distribution.log_prob(xs)
        return initial_prob + target_prob

    def time_derivative(self, xs: chex.Array, t: float) -> chex.Array:
        return jax.grad(lambda t: self.time_dependent_log_prob(xs, t))(t)

    def score_fn(self, xs: chex.Array, t: float) -> chex.Array:
        return jax.grad(lambda x: self.time_dependent_log_prob(x, t))(xs)

    def sample_initial(self, key: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        return self.initial_distribution.sample(key, sample_shape)


class TimeVelocityField(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, key, input_dim, hidden_dim, depth=3):
        self.mlp = eqx.nn.MLP(
            in_size=input_dim + 1,
            out_size=input_dim,
            width_size=hidden_dim,
            activation=jax.nn.sigmoid,
            depth=depth,
            key=key,
        )

    def __call__(self, xs, t):
        t_expanded = jnp.array([t])
        x_t = jnp.concatenate([xs, t_expanded], axis=-1)
        return self.mlp(x_t)


# Utility functions
@eqx.filter_jit
def euler_integrate(
    v_theta: Callable[[jnp.ndarray, float], jnp.ndarray],
    initial_samples: jnp.ndarray,
    ts: jnp.ndarray,
) -> jnp.ndarray:
    def step(carry, t):
        x_prev, t_prev = carry
        samples = x_prev + (t - t_prev) * jax.vmap(lambda x: v_theta(x, t))(x_prev)
        return (samples, t), samples

    _, output = jax.lax.scan(step, (initial_samples, 0.0), ts)

    return output


@eqx.filter_jit
def rk4_integrate(
    v_theta: Callable[[jnp.ndarray, float], jnp.ndarray],
    initial_samples: jnp.ndarray,
    ts: jnp.ndarray,
) -> jnp.ndarray:
    """Integrate ODE using 4th order Runge-Kutta method.

    Args:
        v_theta: Velocity field function
        initial_samples: Initial positions shape (num_samples, dim)
        ts: Time points to evaluate at

    Returns:
        Array of positions at each time point
    """

    def step(carry, t):
        x, t_prev = carry
        dt = t - t_prev

        # Evaluate at initial point
        k1 = jax.vmap(lambda x_i: v_theta(x_i, t_prev))(x)

        # Evaluate at midpoint using k1
        t_mid = t_prev + 0.5 * dt
        x_mid1 = x + 0.5 * dt * k1
        k2 = jax.vmap(lambda x_i: v_theta(x_i, t_mid))(x_mid1)

        # Evaluate at midpoint using k2
        x_mid2 = x + 0.5 * dt * k2
        k3 = jax.vmap(lambda x_i: v_theta(x_i, t_mid))(x_mid2)

        # Evaluate at endpoint using k3
        x_end = x + dt * k3
        k4 = jax.vmap(lambda x_i: v_theta(x_i, t))(x_end)

        # Combine steps with standard RK4 weights
        dx = (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (dt / 6.0)
        new_x = x + dx

        return (new_x, t), new_x

    _, output = jax.lax.scan(step, (initial_samples, 0.0), ts)

    return output


def sample_hamiltonian_monte_carlo(
    key: jax.random.PRNGKey,
    time_dependent_log_density: Callable[[chex.Array, float], float],
    x: chex.Array,
    t: float,
    num_steps: int = 10,
    integration_steps: int = 3,
    eta: float = 0.1,
    rejection_sampling: bool = False,
) -> chex.Array:
    """
    Perform a single chain of Hamiltonian Monte Carlo sampling.

    Args:
        key (jax.random.PRNGKey): Random key for randomness.
        time_dependent_log_density (Callable[[Array, float], float]): Function to compute log-density at given x and t.
        x (Array): Initial sample of shape (D,).
        t (float): Current time.
        num_steps (int, optional): Number of HMC steps. Defaults to 10.
        integration_steps (int, optional): Number of leapfrog integration steps. Defaults to 3.
        eta (float, optional): Step size for integration. Defaults to 0.1.

    Returns:
        Array: The sample after HMC steps.
    """
    dim = x.shape[-1]
    covariance = jnp.eye(dim)
    inv_covariance = covariance
    grad_log_prob = jax.grad(lambda x: time_dependent_log_density(x, t))

    def kinetic_energy(v):
        return 0.5 * v.T @ inv_covariance @ v

    def hamiltonian(x, v):
        return -time_dependent_log_density(x, t) + kinetic_energy(v)

    def integration_step(carry, _):
        x, v = carry
        x = x + eta * inv_covariance @ v
        v = v + eta * grad_log_prob(x)
        return (x, v), _

    def hmc_step(x_current, key):
        x = x_current
        key, subkey = jax.random.split(key)

        # Sample momentum
        v = jax.random.normal(subkey, (dim,))
        current_h = hamiltonian(x, v)

        # Initial half step for momentum
        v = v + 0.5 * eta * grad_log_prob(x)

        # Leapfrog integration
        (x, v), _ = jax.lax.scan(
            integration_step, (x, v), None, length=integration_steps
        )

        # Final half step for momentum
        v = v + 0.5 * eta * grad_log_prob(x)

        if rejection_sampling:
            # Compute acceptance probability
            proposed_h = hamiltonian(x, v)
            accept_ratio = jnp.minimum(1.0, jnp.exp(current_h - proposed_h))

            # Accept or reject
            key, subkey = jax.random.split(key)
            uniform_sample = jax.random.uniform(subkey)
            accept = uniform_sample < accept_ratio

            new_x = jax.lax.cond(accept, lambda _: x, lambda _: x_current, operand=None)

            return new_x, None
        else:
            return x, None

    # Run the chain
    keys = jax.random.split(key, num_steps)

    # return hmc_step(init_state, keys[0])
    final_x, _ = jax.lax.scan(hmc_step, x, keys)

    return final_x


@eqx.filter_jit
def time_batched_sample_hamiltonian_monte_carlo(
    key: jax.random.PRNGKey,
    time_dependent_log_density: Callable[[chex.Array, float], float],
    xs: chex.Array,
    ts: chex.Array,
    num_steps: int = 10,
    integration_steps: int = 3,
    eta: float = 0.1,
    rejection_sampling: bool = False,
) -> chex.Array:
    """
    Apply HMC sampling over batches of samples and times.

    Args:
        key (jax.random.PRNGKey): Random key.
        time_dependent_log_density (Callable[[Array, float], float]): Log-density function.
        xs (chex.Array): Samples array of shape (N, D).
        ts (chex.Array): Times array of shape (N,).
        num_steps (int, optional): Number of HMC steps. Defaults to 10.
        integration_steps (int, optional): Number of leapfrog steps. Defaults to 3.
        eta (float, optional): Step size. Defaults to 0.1.

    Returns:
        chex.Array: Samples after HMC steps.
    """
    keys = jax.random.split(key, xs.shape[0] * xs.shape[1])

    final_xs = jax.vmap(
        lambda xs, t, keys: jax.vmap(
            lambda x, subkey: sample_hamiltonian_monte_carlo(
                subkey,
                time_dependent_log_density,
                x,
                t,
                num_steps,
                integration_steps,
                eta,
                rejection_sampling,
            )
        )(xs, keys)
    )(xs, ts, keys.reshape((xs.shape[0], xs.shape[1], -1)))

    return final_xs


@eqx.filter_jit
def generate_samples(
    key: jax.random.PRNGKey,
    v_theta: Callable[[jnp.ndarray, float], jnp.ndarray],
    num_samples: int,
    ts: jnp.ndarray,
    sample_fn: Callable[[jax.random.PRNGKey, Tuple[int, ...]], jnp.ndarray],
    integration_fn: Callable[
        [Callable[[jnp.ndarray, float], jnp.ndarray], jnp.ndarray, jnp.ndarray],
        jnp.ndarray,
    ] = euler_integrate,
) -> jnp.ndarray:
    key, subkey = jax.random.split(key)
    initial_samples = sample_fn(subkey, (num_samples,))
    samples = integration_fn(v_theta, initial_samples, ts)
    return samples


@eqx.filter_jit
def generate_samples_with_hmc_correction(
    key: jax.random.PRNGKey,
    v_theta: Callable[[jnp.ndarray, float], jnp.ndarray],
    sample_fn: Callable[[jax.random.PRNGKey, Tuple[int, ...]], jnp.ndarray],
    time_dependent_log_density: Callable[[jnp.ndarray, float], float],
    num_samples: int,
    ts: jnp.ndarray,
    integration_fn: Callable[
        [Callable[[jnp.ndarray, float], jnp.ndarray], jnp.ndarray, jnp.ndarray],
        jnp.ndarray,
    ] = euler_integrate,
    num_steps: int = 3,
    integration_steps: int = 3,
    eta: float = 0.01,
    rejection_sampling: bool = False,
) -> jnp.ndarray:
    key, subkey = jax.random.split(key)
    initial_samples = generate_samples(
        subkey, v_theta, num_samples, ts, sample_fn, integration_fn
    )

    key, subkey = jax.random.split(key)
    final_samples = time_batched_sample_hamiltonian_monte_carlo(
        subkey,
        time_dependent_log_density,
        initial_samples,
        ts,
        num_steps,
        integration_steps,
        eta,
        rejection_sampling,
    )

    return final_samples


@eqx.filter_jit
def divergence_velocity(
    v_theta: Callable[[chex.Array, float], chex.Array], x: chex.Array, t: float
) -> float:
    """
    Compute the divergence of the velocity field v_theta at point x and time t.

    Args:
        v_theta (Callable[[Array, float], Array]): The velocity field function that takes x and t and returns the velocity vector.
        x (Array): The point at which to compute the divergence.
        t (float): The time at which to compute the divergence.

    Returns:
        float: The divergence of v_theta at (x, t).
    """

    def v_x(x):
        return v_theta(x, t)

    jacobian = jax.jacfwd(v_x)(x)
    div_v = jnp.trace(jacobian)
    return div_v


def epsilon(
    v_theta: Callable[[chex.Array, float], chex.Array],
    x: chex.Array,
    dt_log_density: float,
    t: float,
    score_fn: Callable[[chex.Array, float], chex.Array],
) -> float:
    """Computes the local error in satisfying the Liouville equation.

    Args:
        v_theta: The velocity field function taking (x, t) and returning velocity vector
        x: The point at which to compute the error
        dt_log_density: Time derivative of log density at (x, t)
        t: Current time
        score_fn: Score function taking (x, t) and returning gradient of log density

    Returns:
        float: Local error in satisfying the Liouville equation
    """
    score = score_fn(x, t)
    div_v = divergence_velocity(v_theta, x, t)
    v = v_theta(x, t)
    lhs = div_v + jnp.dot(v, score)
    return jnp.nan_to_num(lhs + dt_log_density, posinf=1.0, neginf=-1.0)


batched_epsilon = jax.vmap(epsilon, in_axes=(None, 0, 0, None, None))
time_batched_epsilon = eqx.filter_jit(
    jax.vmap(batched_epsilon, in_axes=(None, 0, 0, 0, None))
)


def loss_fn(
    v_theta: Callable[[chex.Array, float], chex.Array],
    xs: chex.Array,
    ts: chex.Array,
    time_derivative_log_density: Callable[[chex.Array, float], float],
    score_fn: Callable[[chex.Array, float], chex.Array],
) -> float:
    """Computes the loss for training the velocity field.

    Args:
        v_theta: The velocity field function taking (x, t) and returning velocity vector
        xs: Batch of points, shape (batch_size, num_samples, dim)
        ts: Batch of times, shape (batch_size,)
        time_derivative_log_density: Function computing time derivative of log density
        score_fn: Score function taking (x, t) and returning gradient of log density

    Returns:
        float: Mean squared error in satisfying the Liouville equation
    """
    dt_log_unormalised_density = jax.vmap(
        lambda xs, t: jax.vmap(lambda x: time_derivative_log_density(x, t))(xs),
        in_axes=(0, 0),
    )(xs, ts)
    dt_log_density = dt_log_unormalised_density - jnp.mean(
        dt_log_unormalised_density, axis=-1, keepdims=True
    )
    epsilons = time_batched_epsilon(v_theta, xs, dt_log_density, ts, score_fn)
    return jnp.mean(epsilons**2)


def evaluate_predictive_prob(beta: chex.Array, X_test: chex.Array) -> chex.Array:
    z = jnp.einsum("ij,j->i", X_test, beta)
    return jax.nn.sigmoid(z)


batched_evaluate_predictive_prob = jax.vmap(evaluate_predictive_prob, in_axes=(0, None))


@eqx.filter_jit
def sample_monotonic_uniform_ordered(
    key: jax.random.PRNGKey, bounds: chex.Array, include_endpoints: bool = True
) -> chex.Array:
    def step(carry, info):
        t_prev = carry
        t_current = info

        return t_current, jnp.array([t_prev, t_current])

    _, ordered_pairs = jax.lax.scan(step, bounds[0], bounds[1:])

    if include_endpoints:
        ordered_pairs = jnp.concatenate(
            [ordered_pairs, jnp.array([[1.0, 1.0]])], axis=0
        )

    samples = jax.random.uniform(
        key, bounds.shape, minval=ordered_pairs[:, 0], maxval=ordered_pairs[:, 1]
    )
    return samples


def run_nuts_chain(target, key, step_size=0.1, num_samples=1000, num_warmup=1000):
    """Run HMC chain for BetaTarget"""
    # Setup step size and mass matrix
    inverse_mass_matrix = jnp.ones(target.dim)

    # Initialize state
    initial_position = jnp.zeros(target.dim)

    # Create HMC kernel
    hmc = blackjax.nuts(
        target.log_prob,
        step_size=step_size,
        inverse_mass_matrix=inverse_mass_matrix,
    )

    state = hmc.init(initial_position)

    @jax.jit
    def one_step(state, key):
        state, _ = hmc.step(key, state)
        return state, state.position

    # Generate random keys
    keys = jax.random.split(key, num_samples + num_warmup)

    # Run the chain
    final_state, samples = jax.lax.scan(one_step, state, keys)

    return samples[num_warmup:]  # Discard warmup samples


def evaluate_predictive_accuracy(
    betas: chex.Array, X_test: chex.Array, y_test: chex.Array
) -> chex.Array:
    probs = jnp.mean(batched_evaluate_predictive_prob(betas, X_test), axis=0)
    y_preds = probs > 0.5
    accuracy = accuracy_score(y_test, y_preds)
    f1 = f1_score(y_test, y_preds, average="weighted")
    recall = recall_score(y_test, y_preds, average="weighted")
    precision = precision_score(y_test, y_preds, average="weighted")
    return accuracy, f1, recall, precision


def evaluate_posterior_agreement(
    betas: chex.Array, betas_hmc: chex.Array, X_test: chex.Array
) -> chex.Array:
    probs = jnp.mean(batched_evaluate_predictive_prob(betas, X_test), axis=0)
    probs_hmc = jnp.mean(batched_evaluate_predictive_prob(betas_hmc, X_test), axis=0)
    y_preds = probs > 0.5
    y_preds_hmc = probs_hmc > 0.5

    agreement = jnp.mean(y_preds == y_preds_hmc)

    return agreement


@eqx.filter_jit
def estimate_mmd(X, Y, sigma=1.0):
    def gaussian_kernel_matrix(X, Y, sigma):
        X_norm = jnp.sum(X**2, axis=1).reshape(-1, 1)
        Y_norm = jnp.sum(Y**2, axis=1).reshape(1, -1)
        K = jnp.exp(-(X_norm + Y_norm - 2 * jnp.dot(X, Y.T)) / (2 * sigma**2))
        return K

    K_xx = gaussian_kernel_matrix(X, X, sigma)
    K_yy = gaussian_kernel_matrix(Y, Y, sigma)
    K_xy = gaussian_kernel_matrix(X, Y, sigma)

    m = X.shape[0]
    n = Y.shape[0]

    mmd = (
        (jnp.sum(K_xx) - jnp.trace(K_xx)) / (m * (m - 1))
        + (jnp.sum(K_yy) - jnp.trace(K_yy)) / (n * (n - 1))
        - 2 * jnp.sum(K_xy) / (m * n)
    )
    return mmd


def inverse_power_schedule(T=64, gamma=0.5):
    x_pow = jnp.linspace(0, 1, T)
    t_pow = 1 - x_pow**gamma
    return jnp.flip(t_pow)


def get_optimizer(name: str, learning_rate: float) -> optax.GradientTransformation:
    """Creates optimizer based on name and learning rate.

    Args:
        name: Name of optimizer ('adam', 'adamw', 'sgd', 'rmsprop')
        learning_rate: Learning rate for the optimizer

    Returns:
        optax.GradientTransformation: The configured optimizer
    """
    if name == "adam":
        return optax.adam(learning_rate)
    elif name == "adamw":
        return optax.adamw(learning_rate)
    elif name == "sgd":
        return optax.sgd(learning_rate)
    elif name == "rmsprop":
        return optax.rmsprop(learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


# Main training loop
def train_velocity_field_for_blr(
    key: jax.random.PRNGKey,
    initial_density: Target,
    target_density: Target,
    v_theta: Callable[[chex.Array, float], chex.Array],
    N: int = 512,
    B: int = 256,
    num_epochs: int = 200,
    num_steps: int = 100,
    learning_rate: float = 1e-03,
    T: int = 32,
    gradient_norm: float = 1.0,
    mcmc_type: str = "hmc",
    num_mcmc_steps: int = 5,
    num_mcmc_integration_steps: int = 3,
    eta: float = 0.01,
    schedule: str = "inverse_power",
    continuous_schedule: bool = False,
    gamma_range: Tuple[float, float] = (0.2, 0.8),
    integrator: str = "euler",
    X_test: chex.Array = None,
    y_test: chex.Array = None,
    optimizer: str = "adamw",
    eval_samples: int = 25600,
    with_rejection_sampling: bool = False,
    eval_with_hmc: bool = True,
    **kwargs: Any,
) -> Any:
    path_distribution = AnnealedDistribution(
        initial_distribution=initial_density,
        target_distribution=target_density,
        dim=initial_density.dim,
    )

    if eval_with_hmc:
        # Eval with HMC
        key, subkey = jax.random.split(key)
        betas_from_hmc = run_nuts_chain(
            target_density,
            subkey,
            num_samples=eval_samples * 5 + 10000,
            num_warmup=10000,
        )
        # Discard warmup samples and thin
        betas_from_hmc = betas_from_hmc[::5]

        accuracy_hmc, f1_hmc, recall_hmc, precision_hmc = evaluate_predictive_accuracy(
            betas_from_hmc, X_test, y_test
        )

        wandb.log(
            {
                "accuracy_hmc": accuracy_hmc,
                "f1_hmc": f1_hmc,
                "recall_hmc": recall_hmc,
                "precision_hmc": precision_hmc,
            }
        )

    # Set up optimizer
    gradient_clipping = optax.clip_by_global_norm(gradient_norm)
    base_optimizer = get_optimizer(optimizer, learning_rate)
    optimizer = optax.chain(gradient_clipping, base_optimizer)
    opt_state = optimizer.init(eqx.filter(v_theta, eqx.is_inexact_array))

    integrator = euler_integrate if integrator == "euler" else rk4_integrate
    # Generate time steps
    key, subkey = jax.random.split(key)
    if schedule == "linear":
        ts = jnp.linspace(0, 1, T)
    elif schedule == "inverse_power":
        ts = inverse_power_schedule(T, gamma=0.5)
    else:
        ts = jnp.linspace(0, 1, T)

    sampled_ts = ts

    @eqx.filter_jit
    def step(v_theta, opt_state, xs, ts):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(
            v_theta,
            xs,
            ts,
            time_derivative_log_density=path_distribution.time_derivative,
            score_fn=path_distribution.score_fn,
        )
        updates, opt_state = optimizer.update(grads, opt_state, v_theta)
        v_theta = eqx.apply_updates(v_theta, updates)
        return v_theta, opt_state, loss

    for epoch in range(num_epochs):
        key, subkey = jax.random.split(key)
        if mcmc_type == "hmc":
            samples = generate_samples_with_hmc_correction(
                key=subkey,
                v_theta=v_theta,
                sample_fn=path_distribution.sample_initial,
                time_dependent_log_density=path_distribution.time_dependent_log_prob,
                num_samples=N,
                ts=sampled_ts,
                integration_fn=integrator,
                num_steps=num_mcmc_steps,
                integration_steps=num_mcmc_integration_steps,
                eta=eta,
                rejection_sampling=with_rejection_sampling,
            )
        else:
            samples = generate_samples(
                subkey,
                v_theta,
                N,
                sampled_ts,
                path_distribution.sample_initial,
                integrator,
            )

        epoch_loss = 0.0
        for s in range(num_steps):
            key, subkey = jax.random.split(key)
            samps = jax.random.choice(subkey, samples, (B,), replace=False, axis=1)
            v_theta, opt_state, loss = step(v_theta, opt_state, samps, sampled_ts)
            epoch_loss += loss

            if s % 20 == 0:
                wandb.log({"loss": loss})

        avg_loss = epoch_loss / num_steps
        print(f"Epoch {epoch}, Average Loss: {avg_loss}")
        wandb.log({"epoch": epoch, "average_loss": avg_loss})

        if epoch % 10 == 0:
            linear_ts = jnp.linspace(0, 1, T)
            betas = generate_samples(
                subkey,
                v_theta,
                eval_samples,
                linear_ts,
                path_distribution.sample_initial,
                integrator,
            )
            betas = betas[-1]
            if X_test is not None and y_test is not None:
                accuracy, f1, recall, precision = evaluate_predictive_accuracy(
                    betas, X_test, y_test
                )
                print(
                    f"Accuracy: {accuracy}, F1: {f1}, Recall: {recall}, Precision: {precision}"
                )
                wandb.log(
                    {
                        "accuracy": accuracy,
                        "f1": f1,
                        "recall": recall,
                        "precision": precision,
                    }
                )

                if eval_with_hmc:
                    agreement = evaluate_posterior_agreement(
                        betas, betas_from_hmc, X_test
                    )
                    print(f"Posterior agreement with HMC: {agreement}")

                    mmd = estimate_mmd(betas, betas_from_hmc)

                    wandb.log({"posterior_agreement": agreement, "mmd": mmd})

        # Resample ts according to gamma range
        if continuous_schedule:
            key, subkey = jax.random.split(key)
            sampled_ts = sample_monotonic_uniform_ordered(subkey, ts, True)

    # Save trained model to wandb
    eqx.tree_serialise_leaves("v_theta.eqx", v_theta)
    artifact = wandb.Artifact(name=f"velocity_field_model_{wandb.run.id}", type="model")
    artifact.add_file(local_path="v_theta.eqx", name="model")
    artifact.save()

    wandb.log_artifact(artifact)
    wandb.finish()
    return v_theta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="ionosphere.pkl")
    parser.add_argument("--num-epochs", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=5120)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--num-timesteps", type=int, default=128)
    parser.add_argument("--mcmc-steps", type=int, default=5)
    parser.add_argument("--mcmc-integration-steps", type=int, default=3)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--initial-sigma", type=float, default=20.0)
    parser.add_argument("--eval-samples", type=int, default=25600)
    parser.add_argument("--split", type=float, default=0.15)
    parser.add_argument(
        "--schedule",
        type=str,
        choices=["linear", "inverse_power"],
        default="inverse_power",
    )
    parser.add_argument(
        "--integrator", type=str, choices=["euler", "rk4"], default="euler"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "adamw", "sgd", "rmsprop"],
        default="adamw",
    )
    parser.add_argument(
        "--normalize-features",
        action="store_true",
        help="Normalize each feature dimension to zero mean and unit variance",
    )
    parser.add_argument("--continuous-schedule", action="store_true")
    parser.add_argument("--with-rejection-sampling", action="store_true")
    parser.add_argument("--gamma-min", type=float, default=0.4)
    parser.add_argument("--gamma-max", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Set random seed
    key = jax.random.PRNGKey(args.seed)

    # Load data
    with open(args.data_path, "rb") as f:
        X, y = pickle.load(f)

    # Preprocess data
    y = jnp.clip(y, 0, 1)
    # Add feature normalization
    if args.normalize_features:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    key, subkey = jax.random.split(key)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.split, random_state=int(subkey[0])
    )

    # Input Dimension
    input_dim = X_train.shape[1]

    # Initialize distributions
    initial_density = MultivariateGaussian(dim=input_dim, sigma=args.initial_sigma)
    target_density = BetaTarget(X=X_train, y=y_train, dim=input_dim)

    # Initialize velocity field
    key, model_key = jax.random.split(key)
    v_theta = TimeVelocityField(
        model_key, input_dim=input_dim, hidden_dim=args.hidden_dim, depth=args.depth
    )

    # Handle logging hyperparameters
    wandb.init(
        project="continuous_liouville_blr",
        config={
            "input_dim": initial_density.dim,
            "data_path": args.data_path,
            "T": args.num_timesteps,
            "N": args.num_samples,
            "num_epochs": args.num_epochs,
            "num_steps": args.num_steps,
            "learning_rate": args.learning_rate,
            "gradient_norm": 1.0,
            "hidden_dim": v_theta.mlp.width_size,
            "depth": v_theta.mlp.depth,
            "mcmc_type": "hmc",
            "num_mcmc_steps": args.mcmc_steps,
            "num_mcmc_integration_steps": args.mcmc_integration_steps,
            "eta": args.eta,
            "schedule": args.schedule,
            "gamma_range": args.gamma_min,
            "optimizer": args.optimizer,
            "normalize_features": args.normalize_features,
            "eval_samples": args.eval_samples,
            "integrator": args.integrator,
            "with_rejection_sampling": args.with_rejection_sampling,
            "continuous_schedule": args.continuous_schedule,
            "eval_with_hmc": True,
        },
        name="velocity_field_training",
        reinit=True,
    )

    # Train model
    v_theta = train_velocity_field_for_blr(
        key=key,
        initial_density=initial_density,
        target_density=target_density,
        v_theta=v_theta,
        N=args.num_samples,
        B=args.batch_size,
        num_epochs=args.num_epochs,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        T=args.num_timesteps,
        num_mcmc_steps=args.mcmc_steps,
        num_mcmc_integration_steps=args.mcmc_integration_steps,
        mcmc_type="hmc",
        eta=args.eta,
        schedule=args.schedule,
        gamma_range=(args.gamma_min, args.gamma_max),
        integrator=args.integrator,
        X_test=X_test,
        y_test=y_test,
        optimizer=args.optimizer,
        eval_samples=args.eval_samples,
        with_rejection_sampling=args.with_rejection_sampling,
        continuous_schedule=args.continuous_schedule,
    )


if __name__ == "__main__":
    main()
