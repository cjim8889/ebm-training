from typing import Callable, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


@eqx.filter_jit
def divergence_velocity_xtrace(
    v_theta: Callable,  # velocity function: v_theta(x, t, ...) mapping R^D -> R^D
    x: Float[Array, "D"],     # state, shape [D]
    t: Float[Array, ""],     # scalar time or parameter
    *,
    m: int,             # even integer, total number of matvecs (m/2 probes)
    d: Optional[Float[Array, ""]] = None,  # optional extra argument to v_theta
    dropout_key: Optional[PRNGKeyArray] = None,
) -> Tuple[Float[Array, ""], Float[Array, ""], Float[Array, "D"]]:
    """
    XTrace divergence estimation for a velocity field v_theta.
    
    This function estimates the divergence (i.e. trace of the Jacobian)
    of v_theta at point x and time t using an XTrace-style estimator.
    
    Args:
      v_theta: A callable representing the velocity field. It should have a signature
               like v_theta(x, t, d, enable_dropout, key) if dropout is used, or
               v_theta(x, t, d) / v_theta(x, t) otherwise.
      x: Array of shape [D] representing the state.
      t: Scalar time or parameter.
      m: An even integer specifying the total number of matvecs to use (m/2 probes).
      d: Optional extra argument to pass to v_theta.
      dropout_key: Optional PRNGKey used for randomness (and for dropout if needed).
    
    Returns:
      A tuple (div_est, error_est, primals) where:
        - div_est is the estimated divergence (scalar),
        - error_est is an error estimate,
        - primals is the computed v_theta(x, t, ...) at the given x.
    """
    # Define v_fn to handle optional dropout and extra argument d
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
    
    # Compute primals = v_fn(x) and get the VJP function.
    primals, f_vjp = jax.vjp(v_fn, x)
    # Define a function for the matrix-vector product with A = J_v(x)^T.
    # Although f_vjp returns J_v(x)^T*g, we use it in place of A*.
    def matvec(vec: Float[Array, "D"]) -> Float[Array, "D"]:
        return f_vjp(vec)[0]
    
    # Use dropout_key for randomness if provided, else a default key.
    key_rng = dropout_key if dropout_key is not None else jax.random.PRNGKey(0)
    
    D = x.shape[0]
    p = m // 2  # number of probes
    
    # 1. Draw random matrix Omega with entries in {+1, -1} of shape [D, p]
    Omega: Float[Array, "D p"] = jax.random.rademacher(key_rng, (D, p), dtype=x.dtype)  # maps 0,1 -> -1,1
    
    # 2. Compute Y = A @ Omega, where A = J_v(x)^T.
    #    We use jax.vmap to apply matvec to each column of Omega.
    Y: Float[Array, "D p"] = jax.vmap(matvec, in_axes=1, out_axes=1)(Omega)  # shape: [D, p]
    
    # 3. Compute the economy-size QR decomposition of Y: Y = Q R.
    qr_result = jnp.linalg.qr(Y, mode="reduced")
    Q: Float[Array, "D p"] = qr_result[0]  # Q: [D, p]
    R: Float[Array, "p p"] = qr_result[1]  # R: [p, p]
    
    # 4. Compute Z = A @ Q.
    Z: Float[Array, "D p"] = jax.vmap(matvec, in_axes=1, out_axes=1)(Q)  # shape: [D, p]
    
    # 5. Compute small matrices H, W, T.
    H: Float[Array, "p p"] = Q.T @ Z       # shape: [p, p]
    W: Float[Array, "p p"] = Q.T @ Omega   # shape: [p, p]
    T: Float[Array, "p p"] = Z.T @ Omega   # shape: [p, p]
    
    # 6. Compute S = (R^T)^{-1} and then normalize each column.
    S: Float[Array, "p p"] = jnp.linalg.inv(R.T)  # S: [p, p] (unnormalized columns)
    col_norms: Float[Array, "p"] = jnp.linalg.norm(S, axis=0)  # norms for each column (shape: [p])
    S_norm: Float[Array, "p p"] = S / col_norms  # normalized columns; each column s_i has unit norm
    
    trace_H: Float[Array, ""] = jnp.trace(H)
    
    # 7. Define the estimator per probe.
    def per_probe_estimator(
        s: Float[Array, "p"], 
        r: Float[Array, "p"], 
        w: Float[Array, "p"], 
        t_vec: Float[Array, "p"]
    ) -> Float[Array, ""]:
        dot_ws = jnp.dot(w, s)
        x_vec: Float[Array, "p"] = w - dot_ws * s
        term1 = trace_H - jnp.dot(s, H @ s)
        term2 = dot_ws * jnp.dot(s, r)  # note: jnp.dot(s, r) is the norm of the unnormalized s column.
        term3 = - jnp.dot(t_vec, x_vec)
        term4 = jnp.dot(x_vec, H @ x_vec)
        return term1 + term2 + term3 + term4
    
    # 8. Vectorize the estimator over the p probes (columns of S_norm, S, W, T).
    estimates: Float[Array, "p"] = jax.vmap(per_probe_estimator, in_axes=1, out_axes=0)(S_norm, S, W, T)  # shape: [p]
    
    # 9. Compute the final divergence estimate and error estimate.
    trace_est: Float[Array, ""] = jnp.mean(estimates)  # estimated trace of A, which equals div(v_theta)
    error2: Float[Array, ""] = jnp.sum((estimates - trace_est) ** 2) / (p * (p - 1))
    error_est: Float[Array, ""] = jnp.sqrt(error2)
    
    # Return divergence estimate, error estimate, and primals (v_theta(x, t, ...)).
    return trace_est, error_est, primals
