from typing import Callable, Optional, Tuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as linalg
from jaxtyping import Array, Float, PRNGKeyArray

@eqx.filter_jit
def divergence_velocity_hutchpp(
    key: PRNGKeyArray,
    v_theta: Callable, 
    x: Float[Array, "D"], 
    t: Float[Array, ""], 
    *,
    n_probes: int = 4,
    d: Optional[Float[Array, ""]] = None,
    dropout_key: Optional[PRNGKeyArray] = None,
) -> Tuple[Float, Float, Float]:
    r"""
    Combine the logic of:
      (1) Estimating a rank-r subspace Q that approximates the top-r eigen-directions 
          of the Jacobian \(\mathbf{J}_v(\mathbf{x})\), 
      (2) Performing a Hutch++-style trace (divergence) estimate 
          using that subspace plus random residual directions.

    This yields an estimate of \(\mathrm{div}\,v_\theta(\mathbf{x}, t)\) 
    while reusing a single VJP call for efficiency.

    :param v_theta:
        The velocity/drift function, e.g. `v_theta(x, t, ...)`.
        Must map \(\mathbb{R}^D \to \mathbb{R}^D\).
    :param x:
        Input array of shape [D]. (Adjust if your state has different shape.)
    :param t:
        Scalar time or diffusion parameter, passed to `v_theta`.
    :param r:
        Rank for the top subspace approximation.
    :param n_probes:
        Number of random Hutchinson probes for the residual part.
    :param d:
        Optional extra argument to pass to `v_theta`.
    :param dropout_key:
        Optional PRNG key for randomness (both for Q subspace and for `v_theta` if it uses dropout).

    :returns:
        `(div_est, Q, primals)`, where
         - `div_est` is a scalar float (the trace estimate, i.e. divergence),
         - `Q` is the orthonormal subspace [D, r],
         - `primals` is `v_theta(x, t, ...)` at the given `x`.

    **Hutch++ Outline**

    1. Generate `r` random vectors \(S\in \mathbb{R}^{D\times r}\).
    2. Apply \( \mathbf{J}_v(\mathbf{x})\) to each column of `S`, factor via QR => \(\mathbf{Q}\).
    3. `partial_trace = \mathrm{trace}\bigl(\mathbf{Q}^\mathsf{T} \mathbf{J}_v(\mathbf{x}) \mathbf{Q}\bigr)`.
    4. Generate `n_probes` random vectors (the “epsilons”) 
       for the residual in the orthogonal complement of \(\mathbf{Q}\).
    5. Average their standard Hutchinson dot products in that orthogonal space.
    6. Sum partial_trace + residual_est => final trace estimate.
    """

    #----------------------#
    # 1) Define v_fn(x)   #
    #----------------------#
    # Decide whether to enable dropout (and pass key) or not:
    if dropout_key is not None:
        # pass `enable_dropout=True` and `key=dropout_key`
        if d is not None:
            v_fn = lambda z: v_theta(z, t, d, enable_dropout=True, key=dropout_key)
        else:
            v_fn = lambda z: v_theta(z, t, enable_dropout=True, key=dropout_key)
    else:
        # No dropout, no key
        if d is not None:
            v_fn = lambda z: v_theta(z, t, d)
        else:
            v_fn = lambda z: v_theta(z, t)

    chex.assert_rank(x, 1)  # x should be [D,]

    #-----------------------------------------#
    # 2) Single VJP to reuse for all jvp calls
    #-----------------------------------------#
    primals, f_vjp = jax.vjp(v_fn, x)  
    # primals = v_fn(x) => shape [D]
    # f_vjp -> function that does: given g in R^D, return J(x)^T*g

    def jvp_fn(vec: Array) -> Array:
        # J(x)*vec
        return f_vjp(vec)[0]

    #--------------------------------------#
    # 3) Build Q from r random directions
    #--------------------------------------#
    key, subkey = random.split(key, 2)
    # S shape [D, r]
    S = random.rademacher(subkey, (x.shape[0], n_probes), dtype=x.dtype)
    chex.assert_shape(S, (x.shape[0], n_probes))

    # A_S = J(x)*S => shape [D, r]
    A_S = jax.vmap(jvp_fn, in_axes=1, out_axes=1)(S)
    chex.assert_shape(A_S, (x.shape[0], n_probes))

    # Factor => Q in [D, r]
    Q, _ = linalg.qr(A_S, mode="economic")
    chex.assert_shape(Q, (x.shape[0], n_probes))

    #----------------------------------------------#
    # 4) partial_trace = trace(Q^T * J_f(x) * Q )
    #----------------------------------------------#
    # We'll compute A_Q = J(x)*Q => [D, r], then do sum of diag(Q^T A_Q)
    A_Q = jax.vmap(jvp_fn, in_axes=1, out_axes=1)(Q)
    chex.assert_shape(A_Q, (x.shape[0], n_probes))

    partial_trace = jnp.einsum("dr,dr->", Q, A_Q)
    chex.assert_shape(partial_trace, ())

    #----------------------------------------------#
    # 5) residual trace with n_probes random vectors
    #----------------------------------------------#
    # eps shape [n_probes, D]
    eps = random.rademacher(key, (n_probes, x.shape[0]), dtype=x.dtype)
    chex.assert_shape(eps, (n_probes, x.shape[0]))

    def residual_trace(e):
        #  e_proj = e - Q @ (Q^T e)
        e_proj = e - Q @ (Q.T @ e)
        #  Ae_proj = J(x)*e_proj
        Ae_proj = jvp_fn(e_proj)
        #  Ae_proj_ortho = Ae_proj - Q @ (Q^T Ae_proj)
        Ae_proj_ortho = Ae_proj - Q @ (Q.T @ Ae_proj)
        return jnp.dot(e_proj, Ae_proj_ortho)

    # Vectorize over the batch of eps
    estimates = jax.vmap(residual_trace)(eps)  # shape [n_probes]
    chex.assert_shape(estimates, (n_probes,))

    residual_est = jnp.mean(estimates)
    chex.assert_shape(residual_est, ())

    #----------------------------------------------#
    # 6) Final divergence estimate
    #----------------------------------------------#
    div_est = partial_trace + residual_est

    return div_est, Q, primals


@eqx.filter_jit
def divergence_velocity_hutchpp2(
    key: PRNGKeyArray,
    v_theta: Callable, 
    x: Float[Array, "D"], 
    t: Float[Array, ""], 
    *,
    n_probes: int = 4,
    d: Optional[Float[Array, ""]] = None,
    dropout_key: Optional[PRNGKeyArray] = None,
) -> Tuple[Float, Float, Float]:
    r"""
    Combine the logic of:
      (1) Estimating a rank-r subspace Q that approximates the top-r eigen-directions 
          of the Jacobian \(\mathbf{J}_v(\mathbf{x})\), 
      (2) Performing a Hutch++-style trace (divergence) estimate 
          using that subspace plus random residual directions.

    This yields an estimate of \(\mathrm{div}\,v_\theta(\mathbf{x}, t)\) 
    while reusing a single VJP call for efficiency.

    :param v_theta:
        The velocity/drift function, e.g. `v_theta(x, t, ...)`.
        Must map \(\mathbb{R}^D \to \mathbb{R}^D\).
    :param x:
        Input array of shape [D]. (Adjust if your state has different shape.)
    :param t:
        Scalar time or diffusion parameter, passed to `v_theta`.
    :param r:
        Rank for the top subspace approximation.
    :param n_probes:
        Number of random Hutchinson probes for the residual part.
    :param d:
        Optional extra argument to pass to `v_theta`.
    :param dropout_key:
        Optional PRNG key for randomness (both for Q subspace and for `v_theta` if it uses dropout).

    :returns:
        `(div_est, Q, primals)`, where
         - `div_est` is a scalar float (the trace estimate, i.e. divergence),
         - `Q` is the orthonormal subspace [D, r],
         - `primals` is `v_theta(x, t, ...)` at the given `x`.

    **Hutch++ Outline**

    1. Generate `r` random vectors \(S\in \mathbb{R}^{D\times r}\).
    2. Apply \( \mathbf{J}_v(\mathbf{x})\) to each column of `S`, factor via QR => \(\mathbf{Q}\).
    3. `partial_trace = \mathrm{trace}\bigl(\mathbf{Q}^\mathsf{T} \mathbf{J}_v(\mathbf{x}) \mathbf{Q}\bigr)`.
    4. Generate `n_probes` random vectors (the “epsilons”) 
       for the residual in the orthogonal complement of \(\mathbf{Q}\).
    5. Average their standard Hutchinson dot products in that orthogonal space.
    6. Sum partial_trace + residual_est => final trace estimate.
    """

    #----------------------#
    # 1) Define v_fn(x)   #
    #----------------------#
    # Decide whether to enable dropout (and pass key) or not:
    if dropout_key is not None:
        # pass `enable_dropout=True` and `key=dropout_key`
        if d is not None:
            v_fn = lambda z: v_theta(z, t, d, enable_dropout=True, key=dropout_key)
        else:
            v_fn = lambda z: v_theta(z, t, enable_dropout=True, key=dropout_key)
    else:
        # No dropout, no key
        if d is not None:
            v_fn = lambda z: v_theta(z, t, d)
        else:
            v_fn = lambda z: v_theta(z, t)

    chex.assert_rank(x, 1)  # x should be [D,]

    #-----------------------------------------#
    # 2) Single VJP to reuse for all jvp calls
    #-----------------------------------------#
    primals, f_vjp = jax.vjp(v_fn, x)  
    # primals = v_fn(x) => shape [D]
    # f_vjp -> function that does: given g in R^D, return J(x)^T*g

    def jvp_fn(vec: Array) -> Array:
        # J(x)*vec
        return f_vjp(vec)[0]
    
    mv = jax.vmap(jvp_fn, in_axes=1, out_axes=1)  # J(x)*vec
    # mv is a vectorized version of the JVP function

    #--------------------------------------#
    # 3) Build Q from r random directions
    #--------------------------------------#
    # S shape [D, r]
    m = n_probes // 3
    S = random.rademacher(key, (x.shape[0], 2*m), dtype=x.dtype)

    X1 = S[:, :m] 
    X2 = S[:, m:]

    # A_S = J(x)*S => shape [D, r]
    Y = mv(X1)

    # compute Q, _ = QR(Y) (orthogonal matrix)
    Q, _ = jnp.linalg.qr(Y)

    # compute G = X2 - Q @ (Q.T @ X2)
    G = X2 - Q @ (Q.T @ X2)

    # estimate trace = tr(Q.T @ A @ Q) + tr(G.T @ A @ G) / k
    AQ = mv(Q)
    AG = mv(G)
    trace_est = jnp.sum(AQ * Q) + jnp.sum(AG * G) / (G.shape[1])

    return trace_est, _, primals
