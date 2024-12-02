import PIL
import torch
import numpy as np
import matplotlib.pyplot as plt

from bgflow import Energy
from bgflow.utils import distance_vectors, distances_from_vectors

from .base import BaseEnergyFunction
from ..utils.data_utils import remove_mean

def sample_from_array(array, size):
    idx = np.random.choice(array.shape[0], size=size)
    return array[idx]


def lennard_jones_energy_torch(r, eps=1.0, rm=1.0):
    p = 0.9
    lj = eps * ((rm / r) ** 12 - 2 * (rm / r) ** 6)
    return lj


class LennardJonesPotential(Energy):
    def __init__(
        self,
        dim,
        n_particles,
        eps=1.0,
        rm=1.0,
        oscillator=True,
        oscillator_scale=1.0,
        two_event_dims=True,
        energy_factor=1.0,
    ):
        """Energy for a Lennard-Jones cluster.

        Parameters
        ----------
        dim : int
            Number of degrees of freedom ( = space dimension x n_particles)
        n_particles : int
            Number of Lennard-Jones particles
        eps : float
            LJ well depth epsilon
        rm : float
            LJ well radius R_min
        oscillator : bool
            Whether to use a harmonic oscillator as an external force
        oscillator_scale : float
            Force constant of the harmonic oscillator energy
        two_event_dims : bool
            If True, the energy expects inputs with two event dimensions (particle_id, coordinate).
            Else, use only one event dimension.
        """
        if two_event_dims:
            super().__init__([n_particles, dim // n_particles])
        else:
            super().__init__(dim)
        self._n_particles = n_particles
        self._n_dims = dim // n_particles

        self._eps = eps
        self._rm = rm
        self.oscillator = oscillator
        self._oscillator_scale = oscillator_scale

        # this is to match the eacf energy with the eq-fm energy
        # for lj13, to match the eacf set energy_factor=0.5
        self._energy_factor = energy_factor

    def _energy(self, x):
        batch_shape = x.shape[: -len(self.event_shape)]
        x = x.view(*batch_shape, self._n_particles, self._n_dims)

        dists = distances_from_vectors(
            distance_vectors(x.view(-1, self._n_particles, self._n_dims))
        )

        lj_energies = lennard_jones_energy_torch(dists, self._eps, self._rm)
        # lj_energies = torch.clip(lj_energies, -1e4, 1e4)
        lj_energies = lj_energies.view(*batch_shape, -1).sum(dim=-1) * self._energy_factor

        if self.oscillator:
            osc_energies = 0.5 * self._remove_mean(x).pow(2).sum(dim=(-2, -1)).view(*batch_shape)
            lj_energies = lj_energies + osc_energies * self._oscillator_scale

        return lj_energies[:, None]

    def _remove_mean(self, x):
        x = x.view(-1, self._n_particles, self._n_dims)
        return x - torch.mean(x, dim=1, keepdim=True)

    def _energy_numpy(self, x):
        x = torch.Tensor(x)
        return self._energy(x).cpu().numpy()

    def _log_prob(self, x):
        return -self._energy(x)


class LennardJonesEnergy(BaseEnergyFunction):
    def __init__(
        self,
        dimensionality,
        n_particles,
        data_path,
        data_path_train=None,
        data_path_val=None,
        device="cpu",
        data_normalization_factor=1.0,
        energy_factor=1.0,
        is_molecule=True,
    ):
        self.n_particles = n_particles
        self.n_spatial_dim = dimensionality // n_particles

        if self.n_particles != 13 and self.n_particles != 55:
            raise NotImplementedError

        self.data_normalization_factor = data_normalization_factor

        self.data_path = data_path
        self.data_path_train = data_path_train
        self.data_path_val = data_path_val

        # self.data_path = get_original_cwd() + "/" + data_path
        # self.data_path_train = get_original_cwd() + "/" + data_path_train
        # self.data_path_val = get_original_cwd() + "/" + data_path_val

        if self.n_particles == 13:
            self.name = "LJ13_efm"
        elif self.n_particles == 55:
            self.name = "LJ55"

        self.device = device

        self.lennard_jones = LennardJonesPotential(
            dim=dimensionality,
            n_particles=n_particles,
            eps=1.0,
            rm=1.0,
            oscillator=True,
            oscillator_scale=1.0,
            two_event_dims=False,
            energy_factor=energy_factor,
        )

        super().__init__(dimensionality=dimensionality, is_molecule=is_molecule)

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        return self.lennard_jones._log_prob(samples).squeeze(-1)
    
    def log_prob(self, samples: torch.Tensor) -> torch.Tensor:
        return self(samples)
    