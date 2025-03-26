# Refactoring Plan for `src/training/core.py` - Time Step Handling

**Objective:** Isolate the logic related to handling time steps (`ts`) from `src/training/core.py` into a dedicated module (`src/training/time_utils.py`) to improve separation of concerns and code clarity.

**Approved Plan:**

1.  **Create New Module:** Create a new file `src/training/time_utils.py`.
2.  **Move Schedule Functions:**
    *   Move `inverse_power_schedule`, `power_schedule`, and `focus_schedule` from `src/utils/optimization.py` to `src/training/time_utils.py`.
    *   Move `sample_monotonic_uniform_ordered` from `src/utils/distributions.py` to `src/training/time_utils.py`.
3.  **Create Helper Functions in `time_utils.py`:**
    *   **`setup_time_schedule`**: Create a function `setup_time_schedule(config: IntegrationConfig, num_timesteps: int) -> Float[Array, " time"]`. This function will contain the logic currently in `core.py::_setup_time_steps` (lines 230-241), calling the appropriate schedule function (`linear`, `inverse_power`, `power`, `focus`) based on `config.schedule`.
    *   **`sample_continuous_time`**: Create a function `sample_continuous_time(key: jax.random.PRNGKey, base_ts: Float[Array, " time"]) -> Float[Array, " time"]`. This will wrap the call to the moved `sample_monotonic_uniform_ordered` function.
4.  **Refactor `core.py`:**
    *   Remove the internal helper function `_setup_time_steps`.
    *   Update imports: Remove imports for the moved functions from `utils` and add imports for `setup_time_schedule` and `sample_continuous_time` from `src.training.time_utils`.
    *   In `train_velocity_field`, replace the call to `_setup_time_steps(config)` with `time_utils.setup_time_schedule(config.integration, config.sampling.num_timesteps)` to generate `base_ts`.
    *   In `_run_training_loop` (around line 832), replace the direct call to `sample_monotonic_uniform_ordered` with `time_utils.sample_continuous_time(subkey_time, base_ts)`.
    *   In `_maybe_estimate_log_z` (around line 463), replace the direct call to `sample_monotonic_uniform_ordered` with `time_utils.sample_continuous_time(subkey_time, base_ts)`.
5.  **Update `config.py`:**
    *   Add `"focus"` to the `Literal` type hint for `IntegrationConfig.schedule` (line 85) to match its usage in `core.py`.

**Visual Plan (Mermaid):**

```mermaid
graph TD
    subgraph src/training/core.py (Before)
        A_old[train_velocity_field] --> F_old[_setup_time_steps]
        F_old --> G_old[utils.optimization schedules]
        B_old[_run_training_loop] --> J_old[utils.distributions.sample_monotonic_uniform_ordered]
        C_old[_maybe_estimate_log_z] --> J_old
    end

    subgraph src/training/time_utils.py (Proposed)
        K[setup_time_schedule] --> M{Moved schedules}
        L[sample_continuous_time] --> P[Moved sample_monotonic_uniform_ordered]
    end

    subgraph src/training/core.py (After)
        A_new[train_velocity_field] --> K
        B_new[_run_training_loop] --> L
        C_new[_maybe_estimate_log_z] --> L
    end

    style F_old fill:#f9f,stroke:#333,stroke-width:2px
    style G_old fill:#f9f,stroke:#333,stroke-width:2px
    style J_old fill:#f9f,stroke:#333,stroke-width:2px
    style K fill:#9cf,stroke:#333,stroke-width:2px
    style L fill:#9cf,stroke:#333,stroke-width:2px
    style M fill:#9cf,stroke:#333,stroke-width:2px
    style P fill:#9cf,stroke:#333,stroke-width:2px