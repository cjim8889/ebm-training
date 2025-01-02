import jax
import jax.numpy as jnp
from utils.smc import SampleBuffer


# Initialize a sample buffer with custom size
buffer = SampleBuffer(buffer_size=1024, min_update_size=128)
key = jax.random.PRNGKey(0)

# Create first batch of samples
num_timesteps = 3
sample_dim = 2
num_samples = 256  # Less than buffer size

key1, key = jax.random.split(key)
samples1 = jax.random.normal(key1, (num_timesteps, num_samples, sample_dim))
weights1 = jnp.ones((num_timesteps, num_samples)) / num_samples  # Uniform weights

print("First batch shape:", samples1.shape)
print("First batch weights shape:", weights1.shape)

# Add first batch to buffer
buffer.add_samples(key1, samples1, weights1)
print("\nAfter adding first batch (should be padded):")
print("Buffer samples shape:", buffer.samples.shape)
print("Valid samples per timestep:", buffer.sample_counts)

# Create second batch with different number of samples
key2, key = jax.random.split(key)
num_samples2 = 512
samples2 = jax.random.normal(key2, (num_timesteps, num_samples2, sample_dim))
weights2 = jnp.ones((num_timesteps, num_samples2)) / num_samples2

print("\nSecond batch shape:", samples2.shape)
print("Second batch weights shape:", weights2.shape)

# Add second batch to buffer (should trigger resampling)
buffer.add_samples(key2, samples2, weights2)
print("\nAfter adding second batch (should be resampled to buffer size):")
print("Buffer samples shape:", buffer.samples.shape)
print("Valid samples per timestep:", buffer.sample_counts)

# Get samples from buffer
key3, key = jax.random.split(key)
samples, ts = buffer.get_samples(key3)
print("\nRetrieved samples shape:", samples.shape)
print("Retrieved timesteps shape:", ts.shape)

# Print some sample statistics
print("\nSample statistics:")
print("Mean of samples:", jnp.mean(samples, axis=(0, 1)))
print("Std of samples:", jnp.std(samples, axis=(0, 1)))
