import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from utils.distributions import compute_distances
from utils.models import init_linear_weights, xavier_init


class MLPWithLayerNorm(eqx.Module):
    layers: list
    activation: callable

    def __init__(
        self, in_size, out_size, width_size, depth, key, activation=jax.nn.silu
    ):
        keys = jax.random.split(key, depth + 1)
        layers = []
        current_size = in_size

        # Hidden layers
        for i in range(depth):
            layers.append(eqx.nn.Linear(current_size, width_size, key=keys[i]))
            layers.append(eqx.nn.LayerNorm(width_size))
            current_size = width_size

        # Output layer
        layers.append(eqx.nn.Linear(current_size, out_size, key=keys[-1]))

        self.layers = layers
        self.activation = activation

    def __call__(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            if isinstance(layer, eqx.nn.Linear):
                x = layer(x)
                x = self.activation(x)
            else:  # LayerNorm
                x = layer(x)
        return self.layers[-1](x)  # Final layer without activation


class TimeVelocityField(eqx.Module):
    mlp: MLPWithLayerNorm
    shortcut: bool
    dt: float

    def __init__(
        self,
        key,
        input_dim,
        hidden_dim,
        depth=3,
        shortcut=False,
        dt: float = 0.01,
    ):
        self.shortcut = shortcut
        self.dt = dt
        if shortcut:
            self.mlp = MLPWithLayerNorm(
                in_size=input_dim + 2,  # x, t, and d
                out_size=input_dim,
                width_size=hidden_dim,
                depth=depth,
                key=key,
            )
        else:
            self.mlp = MLPWithLayerNorm(
                in_size=input_dim + 1,  # x, and t
                out_size=input_dim,
                width_size=hidden_dim,
                depth=depth,
                key=key,
            )

        self.mlp = init_linear_weights(self.mlp, xavier_init, key, scale=dt)

    def __call__(self, *input):
        if self.shortcut:
            assert len(input) == 3
            xs, t, d = input
            x_concat = jnp.concatenate(
                [xs, jnp.array([t]), jnp.expand_dims(d, -1)], axis=-1
            )
        else:
            assert len(input) == 2
            xs, t = input
            x_concat = jnp.concatenate([xs, jnp.array([t])], axis=-1)

        return self.mlp(x_concat)


class TimeVelocityFieldWithPairwiseFeature(eqx.Module):
    mlp: MLPWithLayerNorm
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
            self.mlp = MLPWithLayerNorm(
                in_size=input_dim + 2 + num_pairwise,  # x, t, d, pairwise distances
                out_size=input_dim,
                width_size=hidden_dim,
                depth=depth,
                key=key,
            )
        else:
            self.mlp = MLPWithLayerNorm(
                in_size=input_dim + 1 + num_pairwise,  # x, t, pairwise distances
                out_size=input_dim,
                width_size=hidden_dim,
                depth=depth,
                key=key,
            )

    def __call__(self, *input):
        if self.shortcut:
            assert len(input) == 3
            xs, t, d = input
            x_concat = jnp.concatenate(
                [xs, jnp.array([t]), jnp.expand_dims(d, -1)], axis=-1
            )
        else:
            assert len(input) == 2
            xs, t = input
            x_concat = jnp.concatenate([xs, jnp.array([t])], axis=-1)

        # Compute pairwise distances
        dists = compute_distances(
            xs, self.n_particles, self.n_spatial_dim, repeat=False, min_dr=1e-4
        )
        x_concat = jnp.concatenate([x_concat, dists.flatten()], axis=0)

        return xs + self.mlp(x_concat)


class EquivariantTimeVelocityField(eqx.Module):
    mlp: MLPWithLayerNorm
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

        self.mlp = MLPWithLayerNorm(
            in_size=1 + num_pairwise,
            out_size=n_particles * n_spatial_dim,
            width_size=hidden_dim,
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


class AdaptiveFeatureProjection(eqx.Module):
    """Conditioning module for time and distance"""

    time_mlp: eqx.nn.MLP
    dist_mlp: eqx.nn.MLP
    transform: eqx.nn.Linear

    def __init__(self, dim, key):
        t_key, d_key, proj_key = jax.random.split(key, 3)
        self.time_mlp = eqx.nn.MLP(1, dim, dim, 2, key=t_key)
        self.dist_mlp = eqx.nn.MLP(1, dim, dim, 2, key=d_key)
        self.transform = eqx.nn.Linear(dim * 2, dim, key=proj_key)

    def __call__(self, t: chex.Array, d: chex.Array):
        t_feat = jax.nn.silu(self.time_mlp(t))
        d_feat = jax.nn.silu(self.dist_mlp(d))

        concat = jnp.concatenate([t_feat, d_feat], axis=-1)
        return self.transform(concat)


class VelocityFieldTwo(eqx.Module):
    input_proj: eqx.nn.Linear
    blocks: list
    norm: eqx.nn.LayerNorm
    output_proj: eqx.nn.Linear
    conditioning: AdaptiveFeatureProjection
    shortcut: bool
    dt: float

    def __init__(self, key, dim, hidden_dim, depth=6, shortcut=False, dt=0.01):
        keys = jax.random.split(key, 6)
        self.shortcut = shortcut
        self.dt = dt

        # Input processing
        in_dim = dim + 2 if shortcut else dim + 1
        self.input_proj = eqx.nn.Linear(in_dim, hidden_dim, key=keys[0])

        # Residual blocks
        self.blocks = [
            eqx.nn.Sequential(
                [
                    eqx.nn.Linear(hidden_dim, hidden_dim, key=k),
                    eqx.nn.LayerNorm(hidden_dim),
                    eqx.nn.Lambda(jax.nn.gelu),
                ]
            )
            for k in jax.random.split(keys[1], depth)
        ]

        # Conditioning system
        self.conditioning = AdaptiveFeatureProjection(hidden_dim, keys[2])

        # Output projection
        self.norm = eqx.nn.LayerNorm(hidden_dim)
        self.output_proj = eqx.nn.Linear(hidden_dim, dim, key=keys[3])
        self._init_weights(keys[4])

    def _init_weights(self, key):
        """Ensure stable initialization with dt scaling"""
        self.output_proj = init_linear_weights(
            self.output_proj, xavier_init, key, scale=self.dt
        )

    def __call__(self, x: chex.Array, t: float, d: float = None):
        if isinstance(d, float):
            d = jnp.array([d])
        if isinstance(t, float):
            t = jnp.array([t])

        d = d.reshape(1)
        t = t.reshape(1)
        # Prepare inputs
        if self.shortcut:
            inputs = jnp.concatenate([x, t, d])
        else:
            inputs = jnp.concatenate([x, t])

        # Project to hidden space
        h = self.input_proj(inputs)

        # Get conditioning features
        cond = self.conditioning(t, d) if self.shortcut else 0.0

        # Process through blocks
        for block in self.blocks:
            h = block(h + cond)  # Additive conditioning

        # Final projection
        return self.output_proj(self.norm(h))
