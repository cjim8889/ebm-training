from typing import Callable, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


@eqx.filter_jit
def _get_scale(W: Array, D: Array, n: int, k: int) -> Array:
    return (n - k + 1) / (n - jax.numpy.linalg.norm(W, axis=0) ** 2 + jnp.abs(D) ** 2)

@eqx.filter_jit
def divergence_velocity_xtrace(
    v_theta: Callable,
    x: Float[Array, "D"],
    t: Float[Array, ""],
    *,
    n_probes: int = 5,
    d: Optional[Float[Array, ""]] = None,
    dropout_key: Optional[PRNGKeyArray] = None,
) -> Tuple[Float[Array, ""], Float[Array, ""], Float[Array, "D"]]:
    """
    XTrace divergence estimation for a velocity field v_theta, mathematically equivalent
    to the improved XTrace estimator implementation.

    Returns a tuple of (trace_estimate, standard_error, primals), where primals is the
    result of v_theta(x, t, ...) and the error is computed over the m = n_probes//2 probe estimates.
    """
    # Define v_fn based on dropout and additional arguments
    if dropout_key is not None:
        if d is not None:
            v_fn = lambda z: v_theta(z, t, d, enable_dropout=True, key=dropout_key)
        else:
            v_fn = lambda z: v_theta(z, t, enable_dropout=True, key=dropout_key)
    else:
        if d is not None:
            v_fn = lambda z: v_theta(z, t, d)
        else:
            v_fn = lambda z: v_theta(z, t)
    
    # Compute the value and VJP for v_fn at x
    primals, f_vjp = jax.vjp(v_fn, x)
    
    def matvec(vec: Float[Array, "D"]) -> Float[Array, "D"]:
        return f_vjp(vec)[0]
    
    key_rng = dropout_key if dropout_key is not None else jax.random.PRNGKey(0)
    n = x.shape[0]            # analogous to n in the estimator
    m = n_probes // 2         # number of probe pairs
    
    # Draw probe directions (satisfying zero-mean, unit-covariance)
    Omega = jax.random.normal(key_rng, (n, m), dtype=x.dtype)
    Omega = jnp.sqrt(n) * (Omega / jnp.linalg.norm(Omega, axis=0))
    
    # Apply the linearized operator (via the VJP matvec) to Omega
    Y = jax.vmap(matvec, in_axes=1, out_axes=1)(Omega)
    
    # Compute a low-rank orthonormal basis for Y
    Q, R = jax.scipy.linalg.qr(Y, mode="economic")
    Z = jax.vmap(matvec, in_axes=1, out_axes=1)(Q)
    
    # Form the necessary matrices as in XTrace:
    H = Q.T @ Z
    W = Q.T @ Omega
    T = Z.T @ Omega
    
    # Compute S = (inv(R)).T and normalize its columns
    S = jnp.linalg.inv(R).T
    s_norm = jnp.linalg.norm(S, axis=0)
    S = S / s_norm  # now each column of S is normalized
    
    # Compute intermediate quantities per probe (i indexes each column)
    SW_d = jnp.sum(S * W, axis=0)         # s_i^T w_i for each i
    TW_d = jnp.sum(T * W, axis=0)           # t_i^T w_i
    SHS_d = jnp.sum(S * (H @ S), axis=0)    # s_i^T (H s_i)
    HW = H @ W
    WHW_d = jnp.sum(W * HW, axis=0)         # w_i^T (H w_i)
    
    # Compute the additional correction terms:
    term1 = SW_d * jnp.sum((T - H.T @ W) * S, axis=0)
    term2 = (jnp.abs(SW_d) ** 2) * SHS_d
    term3 = jnp.conjugate(SW_d) * jnp.sum(S * (R - HW), axis=0)
    
    # Compute scaling factor using _get_scale (improved variant)
    scale = _get_scale(W, SW_d, n, n_probes)
    
    # Combine to form the per-probe estimates, matching the first implementation:
    estimates = jnp.trace(H) - SHS_d + (WHW_d - TW_d + term1 + term2 + term3) * scale
    
    # Aggregate the estimates (mean and standard error)
    trace_est = jnp.mean(estimates)
    # std_err = jnp.std(estimates) / jnp.sqrt(m)
    
    return trace_est, primals
