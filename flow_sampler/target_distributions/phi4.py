# adapted from https://github.com/mathisgerdes/continuous-flow-lft/blob/master/jaxlft/phi4.py
import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple

def phi4_action(phi: torch.Tensor, m2: float = 1, lam: float = None) -> torch.Tensor:
    """Compute the Euclidean action for the scalar phi^4 theory.

    The Lagrangian density is kin(phi) + m2 * phi + l * phi^4

    Args:
        phi: Single field configuration of shape L^d.
        m2: Mass squared term (can be negative).
        lam: Coupling constant for phi^4 term.

    Returns:
        Scalar, the action of the field configuration..
    """

    phis2 = phi ** 2
    a = m2 * phis2
    if lam is not None:
        a += lam * (phis2 ** 2)
    
    # Kinetic term
    for d in range(phi.dim()):
        a += (torch.roll(phi, shifts=1, dims=d) - phi) ** 2

    action = a.sum()
    return action

@dataclass
class Phi4Theory:
    """Scalar phi^4 theory."""
    shape: Tuple[int, ...]
    m2: float
    lam: float = None

    @property
    def lattice_size(self):
        return np.prod(self.shape)

    @property
    def dim(self):
        return len(self.shape)
    
    @property
    def vmapped_action(self):
        return torch.vmap(
            phi4_action,
            in_dims=(0, None, None),
            randomness="different",
        )

    def action(self, phis: torch.Tensor, m2: float = None, lam: float = None, half: bool = False) -> torch.Tensor:
        """Compute the phi^4 action."""
        lam = self.lam if lam is None else lam
        m2 = self.m2 if m2 is None else m2
        if phis.dim() == self.dim:
            assert phis.shape == self.shape
            action = phi4_action(phis, m2, lam)
            return action / 2 if half else action
        else:
            assert phis.shape[1:] == self.shape
            action = self.vmapped_action(phis, m2, lam)
            return action / 2 if half else action
        
    def log_prob(self, phis: torch.Tensor, m2: float = None, lam: float = None, half: bool = False) -> torch.Tensor:
        """Compute the phi^4 log probability."""
        lam = self.lam if lam is None else lam
        m2 = self.m2 if m2 is None else m2
        return -self.action(phis, m2, lam, half)

