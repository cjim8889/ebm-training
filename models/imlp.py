import equinox as eqx
import jax.numpy as jnp

from utils.distributions import compute_distances

from .mlp import MLPWithNorm


class InvariantFeatureNet(eqx.Module):
    mlp: MLPWithNorm
    n_particles: int
    n_spatial_dim: int
    shortcut: bool

    def __init__(
        self,
        key,
        n_particles,
        n_spatial_dim,
        hidden_dim,
        depth=3,
        shortcut=False,
        mixed_precision=False,
    ):
        self.shortcut = shortcut
        self.n_particles = n_particles
        self.n_spatial_dim = n_spatial_dim
        input_dim = n_particles * n_spatial_dim
        num_pairwise = n_particles * (n_particles - 1) // 2

        if shortcut:
            self.mlp = MLPWithNorm(
                in_size=input_dim + 2 + num_pairwise + self.n_spatial_dim,  # x, t, d, pairwise distances
                out_size=input_dim,
                width_size=hidden_dim,
                depth=depth,
                key=key,
                mixed_precision=mixed_precision,
                norm="rms",
            )
        else:
            self.mlp = MLPWithNorm(
                in_size=input_dim + 1 + num_pairwise + self.n_spatial_dim,  # x, t, pairwise distances
                out_size=input_dim,
                width_size=hidden_dim,
                depth=depth,
                key=key,
                mixed_precision=mixed_precision,
                norm="rms",
            )

    def __call__(self, *input):
        if self.shortcut:
            assert len(input) == 3
            xs, t, d = input
            td_concat = jnp.concatenate(
                [jnp.array([t]), jnp.expand_dims(d, -1)], axis=-1
            )
        else:
            assert len(input) == 2
            xs, t = input
            td_concat = jnp.array([t])

        xs = xs.reshape(-1, self.n_spatial_dim)
        centriod = jnp.mean(xs, axis=0, keepdims=True)
        xs = xs - centriod
        # Compute pairwise distances
        dists = compute_distances(
            xs, self.n_particles, self.n_spatial_dim, repeat=False, min_dr=1e-4
        )
        x_concat = jnp.concatenate([xs.flatten(), td_concat, dists.flatten(), centriod.flatten()], axis=0)

        return self.mlp(x_concat)
