from typing import Callable, Optional, Tuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as linalg
from jaxtyping import Array, Float, PRNGKeyArray


@eqx.filter_jit
def get_Q_velocity(
    v_theta: Callable,
    x: Float[Array, "D"],
    t: float,
    r: Optional[int] = 4,
    d: Optional[float] = None,
    dropout_key: Optional[PRNGKeyArray] = None,
) -> Tuple[Float[Array, "D r"], Float[Array, "D"]]:
    """
    Obtain an orthonormal basis Q ∈ R^{D x r} that approximates the top-r
    eigen-directions of the Jacobian of v_theta(·, t) at x.

    :param v_theta: The velocity/drift function, e.g. v_theta(x, t, ...).
    :param x: Input array (dimension D).
    :param t: Scalar time or diffusion parameter.
    :param r: Rank for the top subspace approximation.
    :param d: Optional extra argument to pass to v_theta.
    :param dropout_key: PRNG key (optional) for dropout or randomization.
    :returns: Q, shape [D, r], whose columns form an approximate top-r subspace.
    """
    # Create a function that applies v_theta(x, t, ...) and captures the VJP.
    if dropout_key is not None:
        v_fn = lambda z: v_theta(z, t, d, enable_dropout=True, key=dropout_key) \
                 if d is not None else v_theta(z, t, enable_dropout=True, key=dropout_key)
    else:
        v_fn = lambda z: v_theta(z, t, d) if d is not None else v_theta(z, t)

    # Pull back once so that we can do repeated vector–Jacobian products.
    _, f_vjp = jax.vjp(v_fn, x)

    # Sample r random vectors in R^D (assume x.shape == (D,)).
    # If x has a different shape, adapt accordingly.
    rng = dropout_key if dropout_key is not None else jax.random.PRNGKey(0)
    S = random.normal(rng, (x.shape[0], r))  # shape [D, r]
    chex.assert_shape(S, (x.shape[0], r))

    # Multiply Jacobian J_v(x) by each column of S => (J_v(x)*S). shape also [D, r]
    def jvp_col(col):
        return f_vjp(col)[0]  # J_f(x)*col
    # vmap over the columns of S (axis=1)
    A_S = jax.vmap(jvp_col, in_axes=1, out_axes=1)(S)  # shape [D, r]
    chex.assert_shape(A_S, (x.shape[0], r))

    # QR factorization to get orthonormal basis for the columns of (J_v(x) * S)
    # Q is [D, r]; R is [r, r]. We only keep Q.
    Q, _ = linalg.qr(A_S, mode="economic")
    chex.assert_shape(Q, (x.shape[0], r))
    return Q


@eqx.filter_jit
def hutchinson_divergence_Q(
    v_theta: Callable,
    x: Float[Array, "D"],
    t: float,
    eps: Float[Array, "n_probes D"],
    Q: Float[Array, "D r"],
    d: Optional[float] = None,
    dropout_key: Optional[PRNGKeyArray] = None,
) -> Float[Array, ""]:
    """
    Hutch++-style divergence (trace of Jacobian) estimator. 
    Same signature as `hutchinson_divergence_velocity2`, but with an extra Q argument.
    
    :param v_theta: The velocity/drift function, e.g. v_theta(x, t, ...).
    :param x: Input array (dimension D).
    :param t: Scalar time or diffusion parameter.
    :param eps: Array of random probes for the residual, shape [n_probes, D].
    :param Q: Orthonormal subspace basis, shape [D, r], from `get_Q_velocity`.
    :param d: Optional extra argument to pass to v_theta.
    :param dropout_key: PRNG key (optional) for dropout or randomization.
    :returns: A scalar (float) estimate of trace(J_v(x)), i.e., div v_theta(x,t).
    """
    # Ensure eps has the same dtype as x
    eps = eps.astype(x.dtype)
    
    # Add shape assertions
    chex.assert_shape(x, (x.shape[0],))
    chex.assert_shape(eps, (eps.shape[0], x.shape[0]))
    chex.assert_shape(Q, (x.shape[0], Q.shape[1]))

    # Define a function f(x) = v_theta(x, t, ...)
    if dropout_key is not None:
        v_fn = lambda z: v_theta(z, t, d, enable_dropout=True, key=dropout_key) \
                 if d is not None else v_theta(z, t, enable_dropout=True, key=dropout_key)
    else:
        v_fn = lambda z: v_theta(z, t, d) if d is not None else v_theta(z, t)

    # Pull back once to reuse the VJP function
    primals, f_vjp = jax.vjp(v_fn, x)

    # -- 1) Deterministic part: trace(Q^T * J_f * Q) --
    def jvp_fn(v):
        return f_vjp(v)[0]
    # A_Q = J_f(x)*Q => shape [D, r]
    A_Q = jax.vmap(jvp_fn, in_axes=1, out_axes=1)(Q)
    chex.assert_shape(A_Q, (x.shape[0], Q.shape[1]))
    
    # partial_trace = sum over diag(Q^T * A_Q)
    partial_trace = jnp.einsum("dr,dr->", Q, A_Q)
    chex.assert_shape(partial_trace, ())

    # -- 2) Stochastic part on the residual: (I - Q Q^T) * A * (I - Q Q^T) --
    # We'll project eps onto the orthogonal complement and average the usual Hutch. dot products.
    def residual_trace(e):
        # Project e onto (I - QQ^T)
        e_proj = e - Q @ (Q.T @ e)
        Ae_proj = jvp_fn(e_proj)
        Ae_proj_ortho = Ae_proj - Q @ (Q.T @ Ae_proj)
        return jnp.dot(e_proj, Ae_proj_ortho)

    # Vectorize over the batch of eps
    estimates = jax.vmap(residual_trace)(eps)  # shape [n_probes]
    chex.assert_shape(estimates, (eps.shape[0],))
    
    residual_est = jnp.mean(estimates)
    chex.assert_shape(residual_est, ())

    return partial_trace + residual_est, primals

@eqx.filter_jit
def divergence_velocity_hutchpp(
    v_theta: Callable, 
    x: Float[Array, "D"], 
    t: Float[Array, ""], 
    *,
    r: int = 4,
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
    rng = dropout_key if dropout_key is not None else jax.random.PRNGKey(0)

    # S shape [D, r]
    S = random.normal(rng, (x.shape[0], r), dtype=x.dtype)
    chex.assert_shape(S, (x.shape[0], r))

    # A_S = J(x)*S => shape [D, r]
    A_S = jax.vmap(jvp_fn, in_axes=1, out_axes=1)(S)
    chex.assert_shape(A_S, (x.shape[0], r))

    # Factor => Q in [D, r]
    Q, _ = linalg.qr(A_S, mode="economic")
    chex.assert_shape(Q, (x.shape[0], r))

    #----------------------------------------------#
    # 4) partial_trace = trace(Q^T * J_f(x) * Q )
    #----------------------------------------------#
    # We'll compute A_Q = J(x)*Q => [D, r], then do sum of diag(Q^T A_Q)
    A_Q = jax.vmap(jvp_fn, in_axes=1, out_axes=1)(Q)
    chex.assert_shape(A_Q, (x.shape[0], r))

    partial_trace = jnp.einsum("dr,dr->", Q, A_Q)
    chex.assert_shape(partial_trace, ())

    #----------------------------------------------#
    # 5) residual trace with n_probes random vectors
    #----------------------------------------------#
    # eps shape [n_probes, D]
    rng_eps = random.split(rng, 2)[-1]  # or any sub-key you like
    eps = random.normal(rng_eps, (n_probes, x.shape[0]), dtype=x.dtype)
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
