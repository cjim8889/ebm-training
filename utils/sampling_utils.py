import torch
import numpy as np

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
        samples = samples + v_theta(samples, t) * (t_n - t_p)
        samples_list.append(samples)
 
    samples = torch.stack(samples_list, dim=0)
    return samples

def generate_samples(v_theta, num_samples, ts, sample_fn):
    initial_samples = sample_fn(num_samples)
    return generate_samples_with_euler_method(v_theta, initial_samples, ts)
