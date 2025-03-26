import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


@eqx.filter_jit
def random_rotation_3d(key: jax.random.PRNGKey) -> Float[Array, "3 3"]:
    """Generates a random 3D rotation matrix."""
    key_angle, key_axis = jax.random.split(key)
    angle = jax.random.uniform(key_angle, shape=(), minval=0.0, maxval=2 * jnp.pi)
    # Sample a random axis uniformly from the sphere.
    axis = jax.random.normal(key_axis, shape=(3,))
    axis = axis / jnp.linalg.norm(axis)
    # Construct the skew-symmetric matrix for the axis.
    K = jnp.array([[0, -axis[2], axis[1]],
                   [axis[2], 0, -axis[0]],
                   [-axis[1], axis[0], 0]])
    I = jnp.eye(3)
    # Rodrigues rotation formula
    R = I + jnp.sin(angle) * K + (1 - jnp.cos(angle)) * (K @ K)
    return R


@eqx.filter_jit
def augment_chain(
    chain: Float[Array, "dim"],
    key: jax.random.PRNGKey,
    translation_scale: float,
    num_particles: int # This seems to relate to the structure within 'dim', e.g., num_particles * 3D coords
) -> Float[Array, "dim"]:
    """Applies random rotation and translation to a single chain (assumed to be flattened coordinates)."""
    dim = chain.shape[0]
    # Assuming dim represents num_particles * 3 coordinates
    if dim % 3 != 0:
        # Add a check or clarification on how num_particles relates to dim if not num_particles*3
        print(f"Warning: augment_chain dimension {dim} is not a multiple of 3.")
        # Fallback or error? For now, proceed assuming it's intended structure.
        num_coords = dim // 3 # This might be incorrect if structure isn't (N, 3)
    else:
        num_coords = dim // 3

    # Reshape the chain into (num_coords, 3) - Requires careful validation of input structure
    try:
        chain_reshaped = chain.reshape((num_coords, 3))
    except ValueError as e:
        print(f"Error reshaping chain in augment_chain: dim={dim}, num_coords={num_coords}. {e}")
        # Potentially return chain unchanged or raise error
        return chain # Return unchanged on error

    # Split the key for rotation and translation
    key_rot, key_trans = jax.random.split(key)
    # Generate a random 3D rotation matrix
    R = random_rotation_3d(key_rot)
    # Apply the rotation
    chain_rot = jnp.dot(chain_reshaped, R.T)
    # Sample a random translation vector (3D) - Applied uniformly to all coords in the chain
    translation = jax.random.uniform(
        key_trans,
        shape=(1, 3), # Generate a single 3D vector
        minval=-translation_scale,
        maxval=translation_scale
    )
    # Apply the translation (broadcasts over num_coords)
    chain_aug = chain_rot + translation
    # Flatten back to a 1D array
    return chain_aug.reshape(-1)


# Vmap over batch dimension (first axis) and keys (first axis)
# Pass num_particles correctly based on config.density.n_particles used in the original call site
batch_augment_chain = jax.vmap(augment_chain, in_axes=(0, 0, None, None))