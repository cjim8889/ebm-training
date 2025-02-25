from typing import Callable, Dict, Tuple, Union

import diffrax
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .diffrax import solve_neural_ode_diffrax
from .euler import solve_neural_ode_euler

@eqx.filter_jit
def generate_samples(
    key: PRNGKeyArray,
    v_theta: Callable[[Float[Array, "dim"], float], Float[Array, "dim"]],
    num_samples: int,
    ts: Float[Array, "num_timesteps"],
    sample_fn: Callable[[PRNGKeyArray, Tuple[int, ...]], Float[Array, "num_samples dim"]],
    use_shortcut: bool = False,
    solver: str = "Euler",
    **kwargs,
) -> Dict[str, Union[Float[Array, "num_timesteps num_samples dim"], 
                     Float[Array, "num_timesteps num_samples"]]]:
    # TODO: Add shift_fn

    initial_samples = sample_fn(key, (num_samples,))
    final_samples, _ = solve_neural_ode_euler(
        v_theta=v_theta,
        y0=initial_samples,
        ts=ts,
        use_shortcut=use_shortcut,
        exact_logp=True,
        forward=True,
        save_trajectory=True,
        solver=diffrax.Tsit5() if solver == "Tsit5" else diffrax.Euler(),
    )
    return {
        "positions": final_samples,
        "weights": jnp.ones((ts.shape[0], num_samples)) / num_samples,
    }

@eqx.filter_jit
def generate_samples_with_log_prob(
    v_theta: Callable[[Float[Array, "dim"], float], Float[Array, "dim"]],
    initial_samples: Float[Array, "num_samples dim"],
    initial_log_probs: Float[Array, "num_samples"],
    ts: Float[Array, "num_timesteps"],
    use_shortcut: bool = False,
    solver: str = "Euler",
    **kwargs,
) -> Tuple[Float[Array, "num_timesteps num_samples dim"], 
           Float[Array, "num_timesteps num_samples"]]:
    final_samples, final_log_probs = solve_neural_ode_euler(
        v_theta=v_theta,
        y0=initial_samples,
        ts=ts,
        log_p0=initial_log_probs,
        use_shortcut=use_shortcut,
        exact_logp=True,
        forward=True,
        solver=diffrax.Tsit5() if solver == "Tsit5" else diffrax.Euler(),
    )
    return final_samples, final_log_probs