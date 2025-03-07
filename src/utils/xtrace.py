from typing import Callable, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


# @eqx.filter_jit
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
    XTrace divergence estimation for a velocity field v_theta.
    """
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
    
    primals, f_vjp = jax.vjp(v_fn, x)
    
    def matvec(vec: Float[Array, "D"]) -> Float[Array, "D"]:
        return f_vjp(vec)[0]
    
    key_rng = dropout_key if dropout_key is not None else jax.random.PRNGKey(0)
    D = x.shape[0]
    p = n_probes // 2
    

    Omega = jax.random.rademacher(key_rng, (D, p), dtype=x.dtype)

    Y = jax.vmap(matvec, in_axes=1, out_axes=1)(Omega)

    Q, R = jax.scipy.linalg.qr(Y, mode="economic")
    Z = jax.vmap(matvec, in_axes=1, out_axes=1)(Q)
    
    H = Q.T @ Z
    W = Q.T @ Omega
    T = Z.T @ Omega
    
    S = jnp.linalg.inv(R.T)
    col_norms = jnp.linalg.norm(S, axis=0)
    S_norm = S / col_norms
    
    trace_H = jnp.trace(H)
    
    @jax.jit
    def per_probe_estimator(
        s: Float[Array, "p"], 
        r_col: Float[Array, "p"],  # Now using columns of R
        w: Float[Array, "p"], 
        t_vec: Float[Array, "p"]
    ) -> Float[Array, ""]:
        dot_ws = jnp.dot(w, s)
        x_vec = w - dot_ws * s
        term1 = trace_H - jnp.dot(s, H @ s)
        term2 = dot_ws * jnp.dot(s, r_col)  # Corrected term using R's column
        term3 = -jnp.dot(t_vec, x_vec)
        term4 = jnp.dot(x_vec, H @ x_vec)
        return term1 + term2 + term3 + term4
    
    # Pass R's columns instead of S's columns
    estimates = jax.vmap(per_probe_estimator, in_axes=1, out_axes=0)(S_norm, R, W, T)
    
    trace_est = jnp.mean(estimates)
    # error2 = jnp.sum((estimates - trace_est) ** 2) / (p * (p - 1))
    # error_est = jnp.sqrt(error2)
    
    return trace_est, primals