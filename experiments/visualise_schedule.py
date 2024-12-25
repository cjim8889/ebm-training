import jax.numpy as jnp
import matplotlib.pyplot as plt


# Define the inverse power schedule function
def inverse_power_schedule(T=64, end_time=1.0, gamma=0.5):
    x_pow = jnp.linspace(0, end_time, T)
    t_pow = 1 - x_pow**gamma
    return jnp.flip(t_pow)


def power_schedule(T=64, end_time=1.0, gamma=0.5):
    x_pow = jnp.linspace(0, end_time, T)
    t_pow = x_pow**gamma
    return t_pow


# Generate the schedule data
T = 64  # Number of time steps
gamma = 0.8  # Gamma parameter for the inverse power law
end_time = 1.0  # End time (usually 1.0)

inverse_schedule = inverse_power_schedule(T, end_time, gamma)
schedule = power_schedule(T, end_time, gamma)

# Plotting the schedule
plt.plot(jnp.linspace(0, end_time, T), schedule, label=f"gamma={gamma}")
plt.plot(jnp.linspace(0, end_time, T), inverse_schedule, label=f"Inverse gamma={gamma}")
plt.xlabel("Time")
plt.ylabel("Schedule Value")
plt.title("Inverse Power Schedule")
plt.legend()
plt.grid(True)
plt.show()
