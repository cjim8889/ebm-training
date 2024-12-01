import PIL
import torch
import numpy as np
import matplotlib.pyplot as plt
from bgflow import MultiDoubleWellPotential

from .base import BaseEnergyFunction
from ..utils.data_utils import remove_mean

class MultiDoubleWellEnergy(BaseEnergyFunction):
    def __init__(
        self,
        dimensionality,
        n_particles,
        data_path,
        data_path_train=None,
        data_path_val=None,
        data_from_efm=True,  # if True, data from EFM
        device="cpu",
        is_molecule=True,
    ):
        self.n_particles = n_particles
        self.n_spatial_dim = dimensionality // n_particles

        self.data_from_efm = data_from_efm

        if data_from_efm:
            self.name = "DW4_EFM"
        else:
            self.name = "DW4_EACF"

        if self.data_from_efm:
            if data_path_train is None:
                raise ValueError("DW4 is from EFM. No train data path provided")
            if data_path_val is None:
                raise ValueError("DW4 is from EFM. No val data path provided")

        self.data_path = data_path
        self.data_path_train = data_path_train
        self.data_path_val = data_path_val

        self.val_set_size = 1000
        self.test_set_size = 1000
        self.train_set_size = 100000

        self.device = device

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

    def setup_test_set(self):
        if self.data_from_efm:
            data = np.load(self.data_path, allow_pickle=True)

        else:
            all_data = np.load(self.data_path, allow_pickle=True)
            data = all_data[0][-self.test_set_size :]
            del all_data

        data = remove_mean(torch.tensor(data), self.n_particles, self.n_spatial_dim).to(
            self.device
        )

        return data

    def setup_train_set(self):
        if self.data_from_efm:
            data = np.load(self.data_path_train, allow_pickle=True)

        else:
            all_data = np.load(self.data_path, allow_pickle=True)
            data = all_data[0][: self.train_set_size]
            del all_data

        data = remove_mean(torch.tensor(data), self.n_particles, self.n_spatial_dim).to(
            self.device
        )

        return data

    def setup_val_set(self):
        if self.data_from_efm:
            data = np.load(self.data_path_val, allow_pickle=True)

        else:
            all_data = np.load(self.data_path, allow_pickle=True)
            data = all_data[0][-self.test_set_size - self.val_set_size : -self.test_set_size]
            del all_data

        data = remove_mean(torch.tensor(data), self.n_particles, self.n_spatial_dim).to(
            self.device
        )
        return data

    def interatomic_dist(self, x):
        batch_shape = x.shape[: -len(self.multi_double_well.event_shape)]
        x = x.view(*batch_shape, self.n_particles, self.n_spatial_dim)

        # Compute the pairwise interatomic distances
        # removes duplicates and diagonal
        distances = x[:, None, :, :] - x[:, :, None, :]
        distances = distances[
            :,
            torch.triu(torch.ones((self.n_particles, self.n_particles)), diagonal=1) == 1,
        ]
        dist = torch.linalg.norm(distances, dim=-1)
        return dist

    def get_dataset_fig(self, samples):
        test_data_smaller = self.sample_test_set(1000)

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        dist_samples = self.interatomic_dist(samples).detach().cpu()
        dist_test = self.interatomic_dist(test_data_smaller).detach().cpu()

        axs[0].hist(
            dist_samples.view(-1),
            bins=100,
            alpha=0.5,
            density=True,
            histtype="step",
            linewidth=4,
        )
        axs[0].hist(
            dist_test.view(-1),
            bins=100,
            alpha=0.5,
            density=True,
            histtype="step",
            linewidth=4,
        )
        axs[0].set_xlabel("Interatomic distance")
        axs[0].legend(["generated data", "test data"])

        energy_samples = -self(samples).detach().detach().cpu()
        energy_test = -self(test_data_smaller).detach().detach().cpu()

        min_energy = -26
        max_energy = 0

        axs[1].hist(
            energy_test.cpu(),
            bins=100,
            density=True,
            alpha=0.4,
            range=(min_energy, max_energy),
            color="g",
            histtype="step",
            linewidth=4,
            label="test data",
        )
        axs[1].hist(
            energy_samples.cpu(),
            bins=100,
            density=True,
            alpha=0.4,
            range=(min_energy, max_energy),
            color="r",
            histtype="step",
            linewidth=4,
            label="generated data",
        )
        axs[1].set_xlabel("Energy")
        axs[1].legend()

        fig.canvas.draw()
        return PIL.Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
