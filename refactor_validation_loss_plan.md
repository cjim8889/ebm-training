        # Refactoring Plan: `calculate_validation_loss_and_plot` using `jax.lax.fori_loop`

**Goal:** Replace the Python loop iterating over mini-batches in `src/training/loss.py::calculate_validation_loss_and_plot` with `jax.lax.fori_loop` to potentially improve performance by keeping the iteration logic within the JAX computation graph.

**Refactoring Steps:**

1.  **Function Signature:** Maintain the existing function signature:
    ```python
    def calculate_validation_loss_and_plot(
        v_theta: Callable,
        particles: Particle,
        path_distribution: AnnealedDistribution,
        ts: jnp.ndarray,
        time_batch_size: int = 8,
        batch_size: int = 128,
    ):
        # ... implementation ...
        return mean_loss, fig
    ```

2.  **Pre-computation (Outside `fori_loop`):**
    *   Calculate `total_samples = particles.x.shape[0]`.
    *   Calculate `mini_batch_size = time_batch_size * batch_size`.
    *   Calculate `mini_batch = int(jnp.ceil(total_samples / mini_batch_size))`.
    *   Determine the expected shape of the full loss array before reshaping: `(total_samples,)`.
    *   Determine the dtype based on `particles.x.dtype`.

3.  **State Initialization (for `fori_loop`):**
    *   Initialize an empty JAX array to store the results from `batched_epsilon` for all samples:
        ```python
        initial_losses = jnp.zeros((total_samples,), dtype=particles.x.dtype)
        ```
    *   The state passed through the loop will be this `initial_losses` array.

4.  **Define Loop Body Function (`loop_body`):**
    *   This function will perform the computation for a single mini-batch.
    *   Signature: `def loop_body(i, current_losses_state):`
        *   `i`: The current loop iteration index (from 0 to `mini_batch - 1`).
        *   `current_losses_state`: The JAX array holding losses computed so far.
    *   Inside `loop_body`:
        *   Calculate `start = i * mini_batch_size`.
        *   Calculate `current_batch_actual_size = jnp.minimum(mini_batch_size, total_samples - start)`. This handles the last potentially smaller batch.
        *   Extract the current mini-batch of particles using `jax.lax.dynamic_slice`. Ensure correct slicing for all `Particle` fields (`x`, `t`, `log_Z_t`, and optional `d`).
        *   Call `batched_epsilon` on the extracted `batch_particles`.
        *   Update the state array (`current_losses_state`) with the computed `losses_batch` using `jax.lax.dynamic_update_slice` at the correct `start` index.
        *   Return the `updated_losses_state`.

5.  **Execute `fori_loop`:**
    *   Call `jax.lax.fori_loop(0, mini_batch, loop_body, initial_losses)` to execute the loop. The result will be the `all_losses` array containing epsilon values for all samples.

6.  **Post-processing (After `fori_loop`):**
    *   Reshape `all_losses` to `(ts.shape[0], -1)`.
    *   Square the losses: `losses = losses ** 2`.
    *   Calculate statistics (`mean_loss`, `loss_mean`, `loss_std`) using `jnp.mean`, `jnp.var`, `jnp.sqrt`.
    *   Generate the plot using `matplotlib.pyplot`.

7.  **JIT Compilation:**
    *   Consider JIT-compiling the core computational part (steps 2-6 up to statistics calculation) by potentially wrapping it in a helper function, leaving the `matplotlib` plotting outside the JIT boundary.

**Conceptual Diagram:**

```mermaid
graph TD
    A[Input Particles, v_theta, path_dist, ts] --> B{Precompute Sizes};
    B --> C{Initialize `initial_losses` Array};
    C --> D{jax.lax.fori_loop};

    subgraph fori_loop Execution
        direction LR
        E[loop_body(i, state)] --> F[dynamic_slice Particles];
        F --> G[Call batched_epsilon];
        G --> H[dynamic_update_slice State];
        H --> I[Return Updated State];
    end

    D -- Iterates 0 to mini_batch-1 --> E;
    D -- Returns Final State --> J[Final `all_losses` Array];

    J --> K{Reshape & Square Losses};
    K --> L{Calculate Statistics};
    L --> M{Plotting (matplotlib)};
    L --> N[Return mean_loss];
    M --> O[Return fig];

    style D fill:#f9f,stroke:#333,stroke-width:2px