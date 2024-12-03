import torch
import numpy as np
from einops import rearrange, repeat

from typing import Callable

def inverse_tangent_schedule(T=64, alpha=5):
    x_atan = torch.linspace(0, 1, T)
    t_atan = 1 - (torch.atan(alpha * x_atan) / np.arctan(alpha))
    return torch.flip(t_atan, dims=(0,))

def inverse_power_schedule(T=64, gamma=0.5):
    x_pow = torch.linspace(0, 1, T)
    t_pow = 1 - x_pow**gamma
    return torch.flip(t_pow, dims=(0,))

def time_schedule(T, schedule_alpha, schedule_gamma):
    return {
        'linear': lambda: torch.linspace(0, 1, T),
        "inverse_tangent": lambda: inverse_tangent_schedule(T, schedule_alpha),
        "inverse_power": lambda: inverse_power_schedule(T, schedule_gamma),
    }


def generate_samples_with_euler_method(v_theta, initial_samples, ts):
    """
    Generate samples using the Euler method.
        t = 0 -> t = 1 : noise -> data
    """
    device = initial_samples.device
    samples = initial_samples
    t_prev = ts[:-1]
    t_next = ts[1:]

    samples_list = [initial_samples]
    for t_p, t_n in zip(t_prev, t_next):
        t = torch.ones(samples.size(0), device=device).unsqueeze(1) * t_p
        with torch.no_grad():
            samples = samples + v_theta(samples, t) * (t_n - t_p)
        samples_list.append(samples)
 
    samples = torch.stack(samples_list, dim=0)
    return samples

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

    for _ in range(num_steps):
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

def time_batched_sample_langevin_dynamics(
    xs: torch.Tensor,
    ts: torch.Tensor,
    score_function: Callable,
    num_steps: int = 3,
    eta: float = 0.01,
):
    """
    Apply Langevin dynamics to samples.

    Args:
        xs (torch.Tensor): Samples tensor of shape (T, N, D).
        ts (torch.Tensor): Time tensor of shape (T,).
        score_function (Callable): Score function.
        num_steps (int): Number of Langevin dynamics steps. Defaults to 3.
        eta (float): Step size. Defaults to 0.01.

    Returns:
        torch.Tensor: Final samples tensor of shape (T, N, D).

    """

    ts = repeat(ts, 't -> t n', n=xs.size(1)).to(xs.device)

    for _ in range(num_steps):
        noise = torch.randn_like(xs)
        g = score_function(xs, ts)
        xs = xs + 0.5 * eta**2 * g + eta * noise

    return xs


def generate_samples(v_theta, num_samples, ts, sample_fn):
    initial_samples = sample_fn(num_samples)
    samples = generate_samples_with_euler_method(v_theta, initial_samples, ts)
    return rearrange(samples, 't n d -> n t d')

def generate_samples_with_hmc(
    v_theta: Callable,
    sample_fn: Callable,
    time_dependent_log_density: Callable,
    num_samples: int,
    ts: torch.Tensor,
    num_steps: int = 3,
    integration_steps: int = 3,
    eta: float = 0.01,
    rejection_sampling: bool = False,
):
    initial_samples = sample_fn(num_samples)
    samples = generate_samples_with_euler_method(v_theta, initial_samples, ts)  # (T, N, D)
    final_samples = time_batched_sample_hamiltonian_monte_carlo(
        time_dependent_log_density,
        samples,
        ts,
        num_steps,
        integration_steps,
        eta,
        rejection_sampling,
    )

    return rearrange(final_samples, 't n d -> n t d')

def generate_samples_with_langevin_dynamics(
    v_theta: Callable,
    num_samples: int,
    ts: torch.Tensor,
    sample_fn: Callable,
    score_function: Callable,
    num_steps: int = 3,
    eta: float = 0.01,
):
    initial_samples = sample_fn(num_samples)
    samples = generate_samples_with_euler_method(v_theta, initial_samples, ts)  # (T, N, D)
    final_samples = time_batched_sample_langevin_dynamics(
        samples,
        ts,
        score_function,
        num_steps,
        eta,
    )

    return rearrange(final_samples, 't n d -> n t d')
