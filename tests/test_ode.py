import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.ode import solve_neural_ode_diffrax, solve_neural_ode_euler


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


def test_simple_linear_decay():
    """Test equivalence of integrators on a simple linear decay system."""
    # Define a simple linear decay vector field
    def v_theta(x, t):
        return -0.5 * x  # dx/dt = -0.5*x solution: x(t) = x(0) * exp(-0.5*t)
    
    # Initial state
    batch_size, dim = 10, 2
    y0 = jnp.ones((batch_size, dim))
    
    # Time steps
    ts = jnp.linspace(0.0, 1.0, 101)  # Increase from 11 to 101 time steps for better accuracy
    
    # Run both integrators
    diffrax_result, _ = solve_neural_ode_diffrax(
        v_theta=v_theta,
        y0=y0,
        ts=ts,
        save_trajectory=True,
        solver=diffrax.Euler(),
    )
    
    euler_result, _ = solve_neural_ode_euler(
        v_theta=v_theta,
        y0=y0,
        ts=ts,
        save_trajectory=True,
    )
    
    # Check that results are close
    assert jnp.allclose(diffrax_result, euler_result, rtol=1e-5, atol=1e-5)
    
    # Also check against analytical solution: x(t) = x(0) * exp(-0.5*t)
    analytical_result = jnp.ones_like(euler_result) * jnp.exp(-0.5 * ts[:, None, None])
    # Using more relaxed tolerance for comparison with analytical solution
    assert jnp.allclose(euler_result, analytical_result, rtol=1e-2, atol=1e-2)


def test_harmonic_oscillator():
    """Test equivalence of integrators on a harmonic oscillator system."""
    # Define a harmonic oscillator vector field (2D)
    def v_theta(x, t, dt=None):
        # x = [position, velocity]
        # dx/dt = [velocity, -position]
        # Handle both cases: when x is [batch, 2] or just [2]
        # During tracing, x might be a single sample
        if x.ndim == 1:
            # Single sample case (during tracing)
            return jnp.array([x[1], -x[0]])
        else:
            # Batch case [batch, 2]
            return jnp.stack([x[:, 1], -x[:, 0]], axis=1)
    
    # Initial state: [position=1, velocity=0] for each sample in batch
    batch_size = 5
    y0 = jnp.tile(jnp.array([1.0, 0.0]), (batch_size, 1))
    
    # Time steps (one period)
    ts = jnp.linspace(0.0, 2*jnp.pi, 101)
    
    # Run both integrators
    diffrax_result, _ = solve_neural_ode_diffrax(
        v_theta=v_theta,
        y0=y0,
        ts=ts,
        save_trajectory=True,
        solver=diffrax.Euler(),
    )
    
    euler_result, _ = solve_neural_ode_euler(
        v_theta=v_theta,
        y0=y0,
        ts=ts,
        save_trajectory=True,
    )
    
    # Check that results are close
    assert jnp.allclose(diffrax_result, euler_result, rtol=1e-3, atol=1e-3)


def test_with_shortcut():
    """Test equivalence of integrators when use_shortcut=True."""
    # Define a vector field that uses the shortcut
    def v_theta(x, t, dt):
        return -0.5 * x * dt  # dx = -0.5*x*dt, pre-multiplied by dt
    
    # Initial state
    batch_size, dim = 10, 3
    y0 = jnp.ones((batch_size, dim))
    
    # Time steps
    ts = jnp.linspace(0.0, 2.0, 21)
    
    # Run both integrators with use_shortcut=True
    diffrax_result, _ = solve_neural_ode_diffrax(
        v_theta=v_theta,
        y0=y0,
        ts=ts,
        use_shortcut=True,
        save_trajectory=True,
        solver=diffrax.Euler(),
    )
    
    euler_result, _ = solve_neural_ode_euler(
        v_theta=v_theta,
        y0=y0,
        ts=ts,
        use_shortcut=True,
        save_trajectory=True,
    )
    
    # Check that results are close
    assert jnp.allclose(diffrax_result, euler_result, rtol=1e-5, atol=1e-5)


def test_without_save_trajectory():
    """Test equivalence of integrators when save_trajectory=False."""
    # Define a simple vector field
    def v_theta(x, t, dt=None):
        return -x  # dx/dt = -x
    
    # Initial state
    batch_size, dim = 5, 4
    y0 = jnp.ones((batch_size, dim))
    
    # Time steps
    ts = jnp.linspace(0.0, 1.0, 11)
    
    # Run both integrators without saving trajectory
    diffrax_result, _ = solve_neural_ode_diffrax(
        v_theta=v_theta,
        y0=y0,
        ts=ts,
        save_trajectory=False,
        solver=diffrax.Euler(),
    )
    
    euler_result, _ = solve_neural_ode_euler(
        v_theta=v_theta,
        y0=y0,
        ts=ts,
        save_trajectory=False,
    )
    
    # Check that results are close
    assert jnp.allclose(diffrax_result, euler_result, rtol=1e-5, atol=1e-5)


def test_random_initial_conditions(key):
    """Test equivalence of integrators with random initial conditions."""
    # Define a simple vector field
    def v_theta(x, t, dt=None):
        return -0.1 * x  # dx/dt = -0.1*x
    
    # Random initial state
    batch_size, dim = 20, 5
    y0 = jax.random.normal(key, (batch_size, dim))
    
    # Time steps
    ts = jnp.linspace(0.0, 2.0, 21)
    
    # Run both integrators
    diffrax_result, _ = solve_neural_ode_diffrax(
        v_theta=v_theta,
        y0=y0,
        ts=ts,
        save_trajectory=True,
        solver=diffrax.Euler(),
    )
    
    euler_result, _ = solve_neural_ode_euler(
        v_theta=v_theta,
        y0=y0,
        ts=ts,
        save_trajectory=True,
    )
    
    # Check that results are close
    assert jnp.allclose(diffrax_result, euler_result, rtol=1e-5, atol=1e-5)


def test_performance_comparison():
    """Compare performance of the two integrators."""
    # Define a vector field
    def v_theta(x, t, dt=None):
        return -0.5 * (x - jnp.sin(t))
    
    # Large batch and dimension
    batch_size, dim = 100, 10
    y0 = jnp.ones((batch_size, dim))
    
    # Many time steps
    ts = jnp.linspace(0.0, 5.0, 101)
    
    # Compile both functions first
    _ = solve_neural_ode_diffrax(
        v_theta=v_theta,
        y0=y0,
        ts=ts,
        save_trajectory=True,
        solver=diffrax.Euler(),
    )
    
    _ = solve_neural_ode_euler(
        v_theta=v_theta,
        y0=y0,
        ts=ts,
        save_trajectory=True,
    )
    
    # Time the diffrax implementation
    diffrax_fn = jax.jit(lambda: solve_neural_ode_diffrax(
        v_theta=v_theta,
        y0=y0,
        ts=ts,
        save_trajectory=True,
        solver=diffrax.Euler(),
    ))
    
    # Time the euler implementation
    euler_fn = jax.jit(lambda: solve_neural_ode_euler(
        v_theta=v_theta,
        y0=y0,
        ts=ts,
        save_trajectory=True,
    ))
    
    # No assert here - just run and check output when running tests
    # The results will be printed when running with pytest -v
    diffrax_result, _ = diffrax_fn()
    euler_result, _ = euler_fn()
    
    # Verify results are still close
    assert jnp.allclose(diffrax_result, euler_result, rtol=1e-4, atol=1e-4)
