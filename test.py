import torch
import numpy as np
import matplotlib.pyplot as plt

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

linear = time_schedule(128, 5.0, 0.5)['linear']().numpy()
inverse_tangent = time_schedule(128, 5.0, 0.5)['inverse_tangent']().numpy()
inverse_power = time_schedule(128, 5.0, 0.5)['inverse_power']().numpy()

print(linear)
print(inverse_power)

plt.plot(linear, label='linear')
plt.plot(inverse_tangent, label='inverse_tangent')
plt.plot(inverse_power, label='inverse_power')
plt.legend()
plt.savefig('time_schedule.png')
