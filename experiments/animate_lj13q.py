import numpy as np
from manim import *

# Configuration
NPZ_FILE_PATH = "data/lj13q_samples_128_steps_trajectory.npz"
SAMPLE_INDEX = 0  # Which sample trajectory to visualize
PARTICLE_RADIUS = 0.1
ANIMATION_RUN_TIME_PER_STEP = 1/24 # 24 steps per second

class LJ13Animation(ThreeDScene):
    def construct(self):
        # 1. Load and Prepare Data
        try:
            data = np.load(NPZ_FILE_PATH)
            # Assuming shape: (time_steps, num_samples, num_particles * 3)
            positions_all_samples = data['positions']
            times = data.get('times', np.linspace(0, 1, positions_all_samples.shape[0])) # Get times if available, else generate
        except FileNotFoundError:
            self.add(Text(f"Error: Could not find data file\n{NPZ_FILE_PATH}", color=RED))
            return
        except KeyError:
             self.add(Text(f"Error: 'positions' key not found in\n{NPZ_FILE_PATH}", color=RED))
             return

        if SAMPLE_INDEX >= positions_all_samples.shape[1]:
             self.add(Text(f"Error: SAMPLE_INDEX {SAMPLE_INDEX} out of bounds\nfor {positions_all_samples.shape[1]} samples", color=RED))
             return

        # Select the trajectory for the specified sample
        trajectory = positions_all_samples[:, SAMPLE_INDEX, :]
        num_time_steps, flattened_dim = trajectory.shape

        if flattened_dim % 3 != 0:
            self.add(Text(f"Error: Flattened dimension {flattened_dim}\nis not divisible by 3", color=RED))
            return

        num_particles = flattened_dim // 3

        # Reshape to (time_steps, num_particles, 3)
        try:
            trajectory = trajectory.reshape((num_time_steps, num_particles, 3))
        except ValueError as e:
             self.add(Text(f"Error reshaping data: {e}", color=RED))
             return

        # 2. Setup Scene
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-3, 3, 1],
            x_length=6,
            y_length=6,
            z_length=6,
        )
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

        # 3. Create Particles (Spheres)
        particles = VGroup()
        for i in range(num_particles):
            initial_pos = trajectory[0, i, :]
            sphere = Sphere(radius=PARTICLE_RADIUS, fill_opacity=0.8).move_to(initial_pos)
            # Assign a color based on index for visual distinction (optional)
            sphere.set_color(interpolate_color(BLUE, GREEN, i / (num_particles -1 if num_particles > 1 else 1)))
            particles.add(sphere)

        # Add initial state
        self.add(axes, particles)
        self.wait(0.5) # Pause briefly on the initial state

        # 4. Animate Trajectory
        for t in range(1, num_time_steps):
            animations = []
            for i, sphere in enumerate(particles):
                new_position = trajectory[t, i, :]
                animations.append(sphere.animate.move_to(new_position))

            self.play(*animations, run_time=ANIMATION_RUN_TIME_PER_STEP)
            # Optional: Add a time indicator
            # time_text.set_value(times[t]) # Requires creating time_text earlier

        self.wait(1) # Hold the final frame

# To render this animation, run the following command in your terminal:
# manim -pql experiments/animate_lj13q.py LJ13Animation