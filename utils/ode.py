from typing import Callable, Tuple
import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from functools import partial


@eqx.filter_jit
def solve_neural_ode_diffrax(
    v_theta: Callable,
    y0: Float[Array, "batch dim"],
    t0: float,
    t1: float,
    dt: float,
    log_p0: Float[Array, "batch"] = None,
    use_shortcut: bool = False,
    exact_logp: bool = True,
    key: jax.random.PRNGKey = None,
    forward: bool = True,
    save_trajectory: bool = False,
    max_steps: int = 128,
) -> Tuple[Float[Array, "batch dim"], Float[Array, "batch"]]:
    """
    Solve the neural ODE using Diffrax.

    Args:
        v_theta: Velocity field accepting (t, y, dt) when use_shortcut=True
        final_samples: Target distribution samples [batch, dim]
        final_time: End time of forward process
        dt: Time step for integration
        use_shortcut: Whether velocity field uses time step dt
        exact_logp: Compute exact divergence vs Hutchinson's estimator
        key: PRNG key for Hutchinson's estimator

    Returns:
        (base_samples, log_probs)
    """

    # Prepare augmented state (samples + log_probs)
    if log_p0 is None:
        initial_log_probs = jnp.zeros((y0.shape[0],))
    else:
        initial_log_probs = log_p0

    augmented_state = (y0, initial_log_probs)

    # Configure solver and step controller
    solver = diffrax.ReversibleHeun()
    dt0 = dt  # Initial negative step for backward integration

    # Prepare arguments based on computation mode
    if exact_logp:
        term = diffrax.ODETerm(
            partial(_exact_logp_wrapper_with_shortcut, forward=forward)
            if use_shortcut
            else partial(_exact_logp_wrapper, forward=forward)
        )
        eps = jnp.zeros_like(y0)  # Dummy
    else:
        term = diffrax.ODETerm(
            partial(_approx_logp_wrapper_with_shortcut, forward=forward)
            if use_shortcut
            else partial(_approx_logp_wrapper, forward=forward)
        )
        if key is None:
            key = jax.random.PRNGKey(0)
        eps = jax.random.rademacher(key, y0.shape, dtype=y0.dtype)

    # Special handling for shortcut: precompute absolute time steps
    if use_shortcut:
        args = (eps, v_theta, jnp.abs(dt))
    else:
        args = (eps, v_theta)

    # Solve the reverse-time ODE
    sols = jax.vmap(
        lambda x: diffrax.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=x,
            args=args,
            saveat=diffrax.SaveAt(steps=True) if save_trajectory else None,
            stepsize_controller=diffrax.ConstantStepSize(),
            max_steps=max_steps,
        )
    )(augmented_state)

    # Extract final state and accumulated log probabilities
    samples, log_probs = sols.ys
    if save_trajectory:
        return jnp.transpose(samples, axes=(1, 0, 2)), jnp.transpose(
            log_probs, axes=(1, 0)
        )
    else:
        return samples.reshape(y0.shape), log_probs.reshape(-1)


@eqx.filter_jit
def reverse_time_flow_diffrax(
    v_theta: Callable,
    final_samples: Float[Array, "batch dim"],
    final_time: float,
    dt: float,
    use_shortcut: bool = False,
    exact_logp: bool = True,
    key: jax.random.PRNGKey = None,
) -> Tuple[Float[Array, "batch dim"], Float[Array, "batch"]]:
    """
    Reverse-time flow with shortcut support using Diffrax.

    Args:
        v_theta: Velocity field accepting (t, y, dt) when use_shortcut=True
        final_samples: Target distribution samples [batch, dim]
        final_time: End time of forward process
        ts: Time points for integration (will be reversed)
        use_shortcut: Whether velocity field uses time step dt
        exact_logp: Compute exact divergence vs Hutchinson's estimator
        key: PRNG key for Hutchinson's estimator

    Returns:
        (base_samples, log_probs)
    """
    return solve_neural_ode_diffrax(
        v_theta,
        y0=final_samples,
        t0=final_time,
        t1=0.0,
        dt=dt,
        use_shortcut=use_shortcut,
        exact_logp=exact_logp,
        key=key,
    )


def _exact_logp_wrapper_with_shortcut(
    t: float,
    state: Tuple[Float[Array, "batch dim"], Float[Array, "batch"]],
    args: Tuple,
    forward: bool = True,
) -> Tuple[Tuple, Float[Array, "batch"]]:
    """Exact divergence computation with shortcut support."""
    y, logp = state
    eps, func, dt_abs = args

    def fn(y):
        return func(y, t, dt_abs)

    trace = jnp.trace(jax.jacfwd(fn)(y))
    f = fn(y)

    return f, -trace if forward else trace


def _approx_logp_wrapper_with_shortcut(
    t: float,
    state: Tuple[Float[Array, "batch dim"], Float[Array, "batch"]],
    args: Tuple,
    forward: bool = True,
) -> Tuple[Tuple, Float[Array, "batch"]]:
    """Approximate divergence with shortcut support."""
    y, logp = state
    eps, func, dt_abs = args

    def fn(y):
        return func(y, t, dt_abs)

    # Hutchinson's trace estimate
    f, vjp_fn = jax.vjp(fn, y)
    (eps_dfdy,) = vjp_fn(eps)
    trace_estimate = jnp.sum(eps_dfdy * eps, axis=-1)
    return f, trace_estimate


# Original exact/approx wrappers from example (modified for completeness)
def _exact_logp_wrapper(t, state, args, forward: bool = True):
    y, logp = state
    eps, func = args

    def fn(y):
        return func(y, t)  # No dt for non-shortcut

    trace = jnp.trace(jax.jacfwd(fn)(y))
    f = fn(y)

    return f, -trace if forward else trace


def _approx_logp_wrapper(t, state, args, forward: bool = True):
    y, logp = state
    eps, func = args

    def fn(y):
        return func(y, t)  # No dt for non-shortcut

    f, vjp_fn = jax.vjp(fn, y)
    (eps_dfdy,) = vjp_fn(eps)
    trace_estimate = jnp.sum(eps_dfdy * eps, axis=-1)
    return f, trace_estimate
