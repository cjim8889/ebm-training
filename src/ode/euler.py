import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import Callable, Tuple, Union


@eqx.filter_jit
def solve_neural_ode_euler(
    v_theta: Callable,
    y0: Float[Array, "batch dim"],
    ts: Float[Array, "steps"],
    use_shortcut: bool = False,
    save_trajectory: bool = False,
    **kwargs
) -> Tuple[Union[Float[Array, "batch dim"], Float[Array, "steps batch dim"]], 
           Union[Float[Array, "batch"], Float[Array, "steps batch"]]]:
    """
    A simple Euler integrator for neural ODEs as a drop-in replacement for solve_neural_ode_diffrax.
    This implementation doesn't compute log probability updates but maintains API compatibility.
    
    Args:
        v_theta: Velocity field function
        y0: Initial state [batch, dim]
        ts: Time points for integration [steps]
        use_shortcut: Whether velocity field uses time step dt
        save_trajectory: Whether to save intermediate states
        
    Returns:
        samples: Final samples [batch, dim] or trajectory [steps, batch, dim]
        log_probs: Unchanged log probabilities [batch] or replicated [steps, batch]
    """
    batch_size, dim = y0.shape
    num_timesteps = ts.shape[0]
    
    # Define Euler step for a single sample
    def euler_step(y, t, next_t):
        dt = next_t - t
        if use_shortcut:
            dy = v_theta(y, t, jnp.abs(dt))
        else:
            dy = v_theta(y, t)
        return y + dt * dy
    
    # Vectorize over batch dimension
    vmap_euler_step = jax.vmap(euler_step, in_axes=(0, None, None), out_axes=0)
    
    # Define scan function
    def scan_fn(y, t_idx):
        t = ts[t_idx]
        next_t = ts[t_idx + 1]
        next_y = vmap_euler_step(y, t, next_t)
        return next_y, next_y
    
    if save_trajectory:
        # Compute trajectory using scan
        _, trajectory_without_y0 = jax.lax.scan(
            scan_fn, y0, jnp.arange(num_timesteps - 1)
        )
        
        # Prepend y0 to get full trajectory
        trajectory = jnp.concatenate([
            jnp.expand_dims(y0, axis=0),
            trajectory_without_y0
        ], axis=0)
        
        return trajectory, None
    else:
        # Just compute final state using scan
        final_y, _ = jax.lax.scan(
            scan_fn, y0, jnp.arange(num_timesteps - 1)
        )
        
        return final_y, None 