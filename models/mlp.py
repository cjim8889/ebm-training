import equinox as eqx
import jax
import jax.numpy as jnp

from utils.distributions import compute_distances


class TimeVelocityField(eqx.Module):
    mlp: eqx.nn.MLP
    shortcut: bool

    def __init__(self, key, input_dim, hidden_dim, depth=3, shortcut=False):
        # Define an MLP with time as an input
        self.shortcut = shortcut

        if shortcut:
            self.mlp = eqx.nn.MLP(
                in_size=input_dim + 2,  # x, t, and d
                out_size=input_dim,
                width_size=hidden_dim,
                activation=jax.nn.sigmoid,
                depth=depth,
                key=key,
            )
        else:
            self.mlp = eqx.nn.MLP(
                in_size=input_dim + 1,  # x, and t
                out_size=input_dim,
                width_size=hidden_dim,
                activation=jax.nn.sigmoid,
                depth=depth,
                key=key,
            )

    def __call__(self, *input):
        if self.shortcut:
            assert len(input) == 3
            xs, t, d = input
            x_concat = jnp.concatenate([xs, jnp.array([t, d])], axis=-1)
        else:
            assert len(input) == 2
            xs, t = input
            x_concat = jnp.concatenate([xs, jnp.array([t])], axis=-1)

        return self.mlp(x_concat)


class TimeVelocityFieldWithPairwiseFeature(eqx.Module):
    mlp: eqx.nn.MLP
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
    ):
        self.shortcut = shortcut
        self.n_particles = n_particles
        self.n_spatial_dim = n_spatial_dim
        input_dim = n_particles * n_spatial_dim
        num_pairwise = n_particles * (n_particles - 1) // 2

        if shortcut:
            self.mlp = eqx.nn.MLP(
                in_size=input_dim + 2 + num_pairwise,  # x, t, d, pairwise distances
                out_size=input_dim,
                width_size=hidden_dim,
                activation=jax.nn.sigmoid,
                depth=depth,
                key=key,
            )
        else:
            self.mlp = eqx.nn.MLP(
                in_size=input_dim + 1 + num_pairwise,  # x, t, pairwise distances
                out_size=input_dim,
                width_size=hidden_dim,
                activation=jax.nn.sigmoid,
                depth=depth,
                key=key,
            )

    def __call__(self, *input):
        if self.shortcut:
            assert len(input) == 3
            xs, t, d = input
            x_concat = jnp.concatenate([xs, jnp.array([t, d])], axis=-1)
        else:
            assert len(input) == 2
            xs, t = input
            x_concat = jnp.concatenate([xs, jnp.array([t])], axis=-1)

        # Compute pairwise distances
        dists = compute_distances(
            xs, self.n_particles, self.n_spatial_dim, repeat=False, min_dr=1e-4
        )
        x_concat = jnp.concatenate([x_concat, dists.flatten()], axis=0)

        return self.mlp(x_concat)


class EquivariantTimeVelocityField(eqx.Module):
    mlp: eqx.nn.MLP
    n_particles: int
    n_spatial_dim: int
    min_dr: float

    def __init__(
        self, key, n_particles, n_spatial_dim, hidden_dim, depth=3, min_dr=1e-4
    ):
        self.n_particles = n_particles
        self.n_spatial_dim = n_spatial_dim
        self.min_dr = min_dr
        num_pairwise = n_particles * (n_particles - 1) // 2

        self.mlp = eqx.nn.MLP(
            in_size=1 + num_pairwise,
            out_size=n_particles * n_spatial_dim,
            width_size=hidden_dim,
            activation=jax.nn.gelu,
            depth=depth,
            key=key,
        )

    def __call__(self, *input):
        assert len(input) == 2
        xs, t = input

        interatomic_distances = compute_distances(
            xs, self.n_particles, self.n_spatial_dim, repeat=False, min_dr=self.min_dr
        )
        features = jnp.concatenate(
            [interatomic_distances.flatten(), jnp.array([t])], axis=-1
        )
        return self.mlp(features)
