import torch
import numpy as np
from einops import rearrange, repeat

from typing import Callable

def time_batched_sample_hamiltonian_monte_carlo(
    time_dependent_log_density: Callable,
    xs: torch.Tensor,
    ts: torch.Tensor,
    num_steps: int = 10,
    integration_steps: int = 3,
    eta: float = 0.1,
    rejection_sampling: bool = False,
):
    """
    Apply Hamiltonian Monte Carlo to samples.

    Args:
        time_dependent_log_density (Callable): Log-density function.
        xs (torch.Tensor): Samples tensor of shape (T, N, D).
        ts (torch.Tensor): Time tensor of shape (T,).
        num_steps (int): Number of HMC steps. Defaults to 10.
        integration_steps (int): Number of leapfrog steps. Defaults to 3.
        eta (float): Step size. Defaults to 0.1.

    Returns:
        torch.Tensor: Final samples tensor of shape (T, N, D).
    """
    ts = repeat(ts, 't -> t n', n=xs.size(1)).to(xs.device)
    dim = xs.shape[-1]
    covariance = repeat(torch.eye(dim).to(xs.device), 'd1 d2 -> t n d1 d2', t=xs.size(0), n=xs.size(1))
    inv_covariance = covariance

    def score_function(x, t):
        x = x.detach().requires_grad_(True)
        log_p = time_dependent_log_density(x, t)
        return torch.autograd.grad(log_p.sum(), x)[0]

    def kinetic_energy(v):
        vT = rearrange(v, 't n d -> t n 1 d')
        v = rearrange(v, 't n d -> t n d 1')
        return rearrange(0.5 * vT @ inv_covariance @ v, 't n 1 1 -> t n')

    def hamiltonian(x, v):
        return -time_dependent_log_density(x, ts) + kinetic_energy(v)

    for step_ in range(num_steps):
        x_current = xs

        # sampling momentum
        v = torch.randn_like(xs)
        current_h = hamiltonian(xs, v)

        # Initial half step for momentum
        v = v - 0.5 * eta * score_function(xs, ts)

        # Leapfrog integration
        for _ in range(integration_steps):
            xs = xs + eta * (inv_covariance @ v.unsqueeze(-1)).squeeze(-1)
            v = v + eta * score_function(xs, ts)

        # Final half step for momentum
        v = v + 0.5 * eta * score_function(xs, ts)

        if rejection_sampling:
            # Compute acceptance probability
            proposed_h = hamiltonian(xs, v)
            accept_prob = torch.exp(current_h - proposed_h)

            # Accept or reject
            uniform_samples = torch.rand_like(accept_prob)
            mask = rearrange(uniform_samples < accept_prob, 't n -> t n 1').float()
            xs = mask * xs + (1 - mask) * x_current

    return xs

def is_nan(x):
    return torch.isnan(x).any()
