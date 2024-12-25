from abc import ABC, abstractmethod

import numpy as np
import torch


class BaseTimeSchedule(ABC):
    @abstractmethod
    def __call__():
        pass


class LinearTimeSchedule(BaseTimeSchedule):
    def __init__(self, T):
        self.T = T

    def __call__(self):
        return torch.linspace(0, 1, self.T)


class InverseTangentTimeSchedule(BaseTimeSchedule):
    def __init__(self, T, alpha=5):
        self.T = T
        self.alpha = alpha

    def __call__(self):
        x_atan = torch.linspace(0, 1, self.T)
        t_atan = 1 - (torch.atan(self.alpha * x_atan) / np.arctan(self.alpha))
        return torch.flip(t_atan, dims=(0,))


class InversePowerTimeSchedule(BaseTimeSchedule):
    def __init__(self, T, gamma=.5):
        self.T = T
        self.gamma = gamma

    def __call__(self):
        x_pow = torch.linspace(0, 1, self.T)
        t_pow = 1 - x_pow ** self.gamma
        return torch.flip(t_pow, dims=(0,))

