from typing import Callable, Dict, Optional, Tuple, Union

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from blackjax.smc.ess import ess
from blackjax.smc.resampling import systematic
from blackjax.smc.solver import dichotomy  # dichotomy solver
from jaxtyping import Array, Float, Int, PRNGKeyArray

from .hmc import sample_hamiltonian_monte_carlo_blackjax
from .smc import log_weights_to_weights


def generate_samples_with_adaptive_smc(
    key: PRNGKeyArray,
    initial_samples: Float[Array, "num_samples dim"],
    time_dependent_log_density: Callable[[Float[Array, "dim"], float], float],
    incremental_log_delta: Callable[[Float[Array, "dim"], float], float],
    t0: float = 0.0,
    max_steps: int = 128,
    mcmc_steps: int = 10,
    integration_steps: int = 10,
    eta: float = 0.1,
    ess_threshold: float = 0.5,
    resampling_fn: Callable[
        [PRNGKeyArray, Float[Array, "num_samples"], int], Int[Array, "num_samples"]
    ] = systematic,
    v_theta: Optional[Callable[[Float[Array, "dim"], float], Float[Array, "dim"]]] = None,
    use_shortcut: bool = False,
    initial_log_weights: Optional[Float[Array, "num_samples"]] = None,
    lambda_factor: float = 1.0,
) -> Dict[str, Union[
    Float[Array, "num_timesteps num_samples dim"],
    Float[Array, "num_timesteps num_samples"],
    Float[Array, "num_timesteps"]
]]:
    # Prepare the batched functions.
    batched_hmc = jax.vmap(
        lambda key, x, beta, covariance: sample_hamiltonian_monte_carlo_blackjax(
            key,
            time_dependent_log_density,
            x,
            beta,
            mcmc_steps,
            integration_steps,
            eta,
            covariance,
            lambda x: x,  # placeholder for shift_fn
        ),
        in_axes=(0, 0, None, None),
    )
    batched_log_delta = jax.vmap(incremental_log_delta, in_axes=(0, None))
    
    if v_theta is not None:
        if use_shortcut:
            batched_v_theta = jax.vmap(v_theta, in_axes=(0, None, None))
        else:
            batched_v_theta = jax.vmap(v_theta, in_axes=(0, None))
    
    num_samples, dim = initial_samples.shape
    chex.assert_rank(initial_samples, 2)
    if initial_log_weights is None:
        initial_log_weights = jnp.full((num_samples,), -jnp.log(num_samples))
    chex.assert_rank(initial_log_weights, 1)
    
    # Pre-allocate diagnostic arrays (maximum length = max_steps+1).
    diag_positions = jnp.zeros((max_steps + 1, num_samples, dim))
    diag_log_weights = jnp.zeros((max_steps + 1, num_samples))
    diag_ess = jnp.zeros((max_steps + 1,))
    diag_beta = jnp.zeros((max_steps + 1,))
    diag_delta = jnp.zeros((max_steps,))

    # Set initial diagnostics.
    diag_positions = diag_positions.at[0].set(initial_samples)
    diag_log_weights = diag_log_weights.at[0].set(initial_log_weights)
    diag_ess = diag_ess.at[0].set(ess(initial_log_weights) / num_samples)
    diag_beta = diag_beta.at[0].set(t0)

    # Pre-split keys for each step.
    step_keys = jax.random.split(key, max_steps)
    
    # Initial particles.
    particles = {
        "positions": initial_samples,
        "log_weights": initial_log_weights,
        "ess": ess(initial_log_weights) / num_samples,
    }
    current_beta = t0
    init_state = (particles, current_beta, 0, diag_positions, diag_log_weights, diag_ess, diag_beta, diag_delta, step_keys)
    
    def _resample(key: PRNGKeyArray, positions: Float[Array, "num_samples dim"], log_weights: Float[Array, "num_samples"]
                 ) -> Tuple[Float[Array, "num_samples dim"], Float[Array, "num_samples"]]:
        weights = log_weights_to_weights(log_weights)
        indices = resampling_fn(key, weights, num_samples)
        new_positions = jnp.take(positions, indices, axis=0)
        new_log_weights = jnp.full((num_samples,), -jnp.log(num_samples))
        return new_positions, new_log_weights

    def cond_fun(state):
        # Continue if beta is less than 1 and we have not exhausted max_steps.
        _, beta, i, *_ = state
        return jnp.logical_and(beta < 1.0, i < max_steps)
    
    def body_fun(state):
        particles_prev, current_beta, i, diag_positions, diag_log_weights, diag_ess, diag_beta, diag_delta, step_keys = state
        step_key = step_keys[i]
        
        # --- Resampling step based on ESS ---
        current_ess = ess(particles_prev["log_weights"]) / num_samples
        def do_resample():
            rk, _ = jax.random.split(step_key)
            new_pos, new_logw = _resample(rk, particles_prev["positions"], particles_prev["log_weights"])
            return {"positions": new_pos, "log_weights": new_logw}
        def do_nothing():
            logw_norm = particles_prev["log_weights"] - jax.scipy.special.logsumexp(particles_prev["log_weights"])
            return {"positions": particles_prev["positions"], "log_weights": logw_norm}
        particles_new = jax.lax.cond(current_ess < ess_threshold, do_resample, do_nothing)
        particles_new["ess"] = current_ess

        # --- Adaptive selection of beta increment (delta) ---
        def fun(delta):
            # Compute the incremental log weights for candidate delta.
            inc_log_w = batched_log_delta(particles_new["positions"], delta)
            # Use log-sum-exp for numerical stability.
            lse = jax.scipy.special.logsumexp(inc_log_w)
            lse2 = jax.scipy.special.logsumexp(2 * inc_log_w)
            ess_candidate = jnp.exp(2 * lse - lse2)
            return ess_candidate / num_samples - ess_threshold
        
        min_delta = 1e-8
        max_delta = 1.0 - current_beta
        # If the full step (delta = max_delta) already gives a sufficiently high ESS,
        # then we choose that; otherwise, use the dichotomy solver.
        candidate = jnp.where(fun(max_delta) > 0,
                              max_delta,
                              dichotomy(fun, min_delta, max_delta, eps=1e-4, max_iter=100))
        delta = candidate
        new_beta = current_beta + delta
        
        # --- Propagate particles ---
        if v_theta is not None:
            if use_shortcut:
                propagated_positions = particles_new["positions"] + lambda_factor * batched_v_theta(
                    particles_new["positions"], current_beta, delta
                )
            else:
                propagated_positions = particles_new["positions"] + lambda_factor * batched_v_theta(
                    particles_new["positions"], current_beta
                )
        else:
            propagated_positions = particles_new["positions"]
        # Apply MCMC propagation (e.g. HMC) at the new tempering level.
        keys = jax.random.split(step_key, num_samples)
        propagated_positions = batched_hmc(keys, propagated_positions, new_beta, None)
        
        # --- Update log weights ---
        w_delta = batched_log_delta(propagated_positions, delta)
        next_log_weights = particles_new["log_weights"] + w_delta
        next_log_weights = next_log_weights - jax.scipy.special.logsumexp(next_log_weights)
        
        new_particles = {
            "positions": propagated_positions,
            "log_weights": next_log_weights,
            "ess": ess(particles_new["log_weights"] + w_delta) / num_samples,
        }
        
        # --- Update diagnostics ---
        diag_positions = diag_positions.at[i + 1].set(propagated_positions)
        diag_log_weights = diag_log_weights.at[i + 1].set(next_log_weights)
        diag_ess = diag_ess.at[i + 1].set(new_particles["ess"])
        diag_beta = diag_beta.at[i + 1].set(new_beta)
        diag_delta = diag_delta.at[i].set(delta)
        
        new_state = (new_particles, new_beta, i + 1, diag_positions, diag_log_weights, diag_ess, diag_beta, diag_delta, step_keys)
        return new_state

    # Run the while loop until beta reaches 1 or we hit max_steps.
    final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)
    
    # Unpack the final state.
    _, _, steps_taken, diag_positions, diag_log_weights, diag_ess, diag_beta, diag_delta, _ = final_state

    # Truncate diagnostics to the number of steps taken + 1.
    diag_positions = diag_positions[: steps_taken + 1]
    diag_log_weights = diag_log_weights[: steps_taken + 1]
    diag_ess = diag_ess[: steps_taken + 1]
    diag_beta = diag_beta[: steps_taken + 1]
    diag_delta = diag_delta[: steps_taken]

    # Convert all log weights to normalized weights.
    diag_weights = jax.vmap(log_weights_to_weights)(diag_log_weights)
    
    return {
        "positions": diag_positions,
        "weights": diag_weights,
        "diagnostics": {
            "ess": diag_ess,
            "beta": diag_beta,
            "delta": diag_delta,
        },
    }
