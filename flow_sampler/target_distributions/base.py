from typing import Optional, Dict

from abc import ABC, abstractmethod
import torch
import numpy as np
from flow_sampler.types_ import LogProbFunc

class TargetDistribution(ABC):

    @abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """returns (unnormalised) log probability of samples x"""
        raise NotImplementedError

    def performance_metrics(self, samples: torch.Tensor, log_w: torch.Tensor,
                            log_q_fn: Optional[LogProbFunc] = None,
                            batch_size: Optional[int] = None) -> Dict:
        """
        Check performance metrics using samples & log weights from the model, as well as it's
        probability density function (if defined).
        Args:
            samples: Samples from the trained model.
            log_w: Log importance weights from the trained model.
            log_q_fn: Log probability density function of the trained model, if defined.
            batch_size: If performance metrics are aggregated over many points that require network
                forward passes, batch_size ensures that the forward passes don't overload GPU
                memory by doing all the points together.

        Returns:
            info: A dictionary of performance measures, specific to the defined
            target_distribution, that evaluate how well the trained model approximates the target.
        """
        raise NotImplementedError


    def sample(self, shape):
        raise NotImplementedError
    

class BaseEnergyFunction(ABC):
    def __init__(
        self,
        dimensionality: int,
        is_molecule: Optional[bool] = False,
        normalization_min: Optional[float] = None,
        normalization_max: Optional[float] = None,
    ):
        self._dimensionality = dimensionality

        self._test_set = self.setup_test_set()
        self._val_set = self.setup_val_set()
        self._train_set = None

        self.normalization_min = normalization_min
        self.normalization_max = normalization_max

        self._is_molecule = is_molecule

    def setup_test_set(self) -> Optional[torch.Tensor]:
        return None

    def setup_train_set(self) -> Optional[torch.Tensor]:
        return None

    def setup_val_set(self) -> Optional[torch.Tensor]:
        return None

    @property
    def _can_normalize(self) -> bool:
        return self.normalization_min is not None and self.normalization_max is not None

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if x is None or not self._can_normalize:
            return x

        mins = self.normalization_min
        maxs = self.normalization_max

        # [ 0, 1 ]
        x = (x - mins) / (maxs - mins + 1e-5)
        # [ -1, 1 ]
        return x * 2 - 1

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        if x is None or not self._can_normalize:
            return x

        mins = self.normalization_min
        maxs = self.normalization_max

        x = (x + 1) / 2
        return x * (maxs - mins) + mins

    def sample_test_set(
        self, num_points: int, normalize: bool = False, full: bool = False
    ) -> Optional[torch.Tensor]:
        if self.test_set is None:
            return None

        if full:
            outs = self.test_set
        else:
            idxs = torch.randperm(len(self.test_set))[:num_points]
            outs = self.test_set[idxs]
        if normalize:
            outs = self.normalize(outs)

        return outs

    def sample_train_set(self, num_points: int, normalize: bool = False) -> Optional[torch.Tensor]:
        if self.train_set is None:
            self._train_set = self.setup_train_set()

        idxs = torch.randperm(len(self.train_set))[:num_points]
        outs = self.train_set[idxs]
        if normalize:
            outs = self.normalize(outs)

        return outs

    def sample_val_set(self, num_points: int, normalize: bool = False) -> Optional[torch.Tensor]:
        if self.val_set is None:
            return None

        idxs = torch.randperm(len(self.val_set))[:num_points]
        outs = self.val_set[idxs]
        if normalize:
            outs = self.normalize(outs)

        return outs

    @property
    def dimensionality(self) -> int:
        return self._dimensionality

    @property
    def is_molecule(self) -> Optional[bool]:
        return self._is_molecule

    @property
    def test_set(self) -> Optional[torch.Tensor]:
        return self._test_set

    @property
    def val_set(self) -> Optional[torch.Tensor]:
        return self._val_set

    @property
    def train_set(self) -> Optional[torch.Tensor]:
        return self._train_set

    @abstractmethod
    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def score(self, samples: torch.Tensor) -> torch.Tensor:
        grad_fxn = torch.func.grad(self.__call__)
        vmapped_grad = torch.vmap(grad_fxn)
        return vmapped_grad(samples)

    def save_samples(
        self,
        samples: torch.Tensor,
        dataset_name: str,
    ) -> None:
        np.save(f"{dataset_name}_samples.npy", samples.cpu().numpy())
