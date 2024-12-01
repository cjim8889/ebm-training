import torch
from bgflow import MultiDoubleWellPotential

from .base import BaseEnergyFunction

class MultiDoubleWellEnergy(BaseEnergyFunction):
    def __init__(
        self,
        dimensionality=8,
        n_particles=4,
        is_molecule=True,
    ):
        self.n_particles = n_particles
        self.n_spatial_dim = dimensionality // n_particles

        self.multi_double_well = MultiDoubleWellPotential(
            dim=dimensionality,
            n_particles=n_particles,
            a=0.9,
            b=-4,
            c=0,
            offset=4,
            two_event_dims=False,
        )

        super().__init__(dimensionality=dimensionality, is_molecule=is_molecule)

    def log_prob(self, samples: torch.Tensor) -> torch.Tensor:
        return -self.multi_double_well.energy(samples).squeeze(-1)
    
    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        return self.log_prob(samples)
