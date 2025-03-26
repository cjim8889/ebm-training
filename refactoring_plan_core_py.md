# Refactoring Plan for src/training/core.py

This document outlines the plan to refactor `src/training/core.py`, primarily focusing on decomposing the `train_velocity_field` function to improve readability and maintainability while preserving functionality.

## Analysis Summary

- The `train_velocity_field` function is excessively long (~400 lines) and handles multiple responsibilities: initialization, optimizer setup, time step generation, lambda scheduling, the main training loop, Log Z estimation, sample generation, data augmentation, loss calculation, logging, validation, evaluation, and model saving.
- This violates the single-responsibility principle and makes the code hard to follow and modify.
- Other functions (`random_rotation_3d`, `augment_chain`, `generate_samples_with_optional_mcmc`) appear reasonably well-structured.

## Proposed Refactoring Structure

The core idea is to break `train_velocity_field` into smaller, focused helper functions.

```mermaid
graph TD
    A[train_velocity_field (Original)] --> B{Initialization};
    A --> C{Optimizer Setup};
    A --> D{Validation Setup};
    A --> E{Training Loop};
    E --> F{Epoch Setup};
    F --> G{Log Z Estimation?};
    G -- Yes --> H{Estimate Log Z};
    G -- No --> I{Reuse Log Z};
    H --> J{Prepare Samples};
    I --> J;
    J --> K{Step Loop};
    K --> L{Prepare Batch};
    L --> M{Augment/Perturb};
    M --> N{Execute Step};
    N --> O{Log Step Loss};
    K -- End Loop --> P{Log Epoch Loss};
    P --> Q{Calculate Validation Loss};
    Q --> R{Log Epoch Summary};
    R --> S{Evaluation?};
    S -- Yes --> T{Perform Evaluation};
    T --> U{Save Best Model};
    S -- No --> V[End Epoch];
    U --> V;
    E -- End Loop --> W{Final Logging};

    subgraph Refactored Structure
        direction LR
        AA[train_velocity_field] --> BB[_initialize_training];
        AA --> CC[_run_training_loop];
        AA --> DD[_finalize_training];

        BB --> EE[_setup_optimizer];
        BB --> FF[_setup_time_steps];
        BB --> GG[_setup_path_distribution];
        BB --> HH[_generate_initial_validation_set];

        CC --> II[_run_epoch];
        II --> JJ[_maybe_estimate_log_z];
        II --> KK[_prepare_epoch_samples];
        II --> LL[_run_steps_for_epoch];
        II --> MM[_calculate_and_log_epoch_metrics];
        II --> NN[_maybe_evaluate_and_save];

        LL --> OO[_prepare_step_batch];
        LL --> PP[_apply_augmentations];
        LL --> QQ[_execute_training_step];

        NN --> RR[_evaluate_model_performance];
        NN --> SS[_save_model_if_best];
    end
```

## Detailed Steps

1.  **Create Helper Functions for Initialization:**
    *   `_setup_optimizer(config)`: Returns configured `optax` optimizer.
    *   `_setup_time_steps(config)`: Returns `base_ts` array.
    *   `_setup_path_distribution(initial_density, target_density, config)`: Returns `AnnealedDistribution`.
    *   `_generate_initial_validation_set(key, v_theta, ts, path_distribution, config)`: Returns `validation_particles`.
    *   `_initialize_training_state(...)`: Calls helpers and initializes state (`opt_state`, `log_Z_t_ref`, etc.).

2.  **Decompose the Main Loop:**
    *   `_run_training_loop(...)`: Contains the main `for epoch in range(...)` loop, calling `_run_epoch`.
    *   `_run_epoch(...)`: Performs work for a single epoch, calling epoch-specific helpers. Returns updated state.

3.  **Extract Epoch Logic into Helpers:**
    *   `_compute_lambda_factor(...)`: Calculates lambda factor (already exists).
    *   `_maybe_estimate_log_z(...)`: Handles Log Z estimation logic and logging. Returns `log_Z_t`, updated `current_ts`, `mcmc_samples`.
    *   `_prepare_epoch_samples(...)`: Generates samples for the epoch's steps. Returns `samples` array.
    *   `_run_steps_for_epoch(...)`: Contains the inner `for s in range(...)` loop. Returns updated state and `total_epoch_loss`.
    *   `_calculate_and_log_epoch_metrics(...)`: Calculates and logs average/validation loss.
    *   `_maybe_evaluate_and_save(...)`: Handles periodic evaluation, metric logging, and model saving. Returns updated state.

4.  **Extract Step Logic into Helpers:**
    *   `_prepare_step_batch(...)`: Prepares `training_particles` for a step. Returns `key`, `training_particles`.
    *   `_apply_augmentations(...)`: Applies augmentations. Returns `key`, augmented `selected_chains`.
    *   `_execute_training_step(...)`: Executes the JIT-compiled `step` function and logs step loss. Returns updated state and step `loss`.

5.  **Finalization:**
    *   `_finalize_training(...)`: Handles final WandB summary logging.

6.  **Review `generate_samples_with_optional_mcmc`:** Check for potential minor improvements.

7.  **Docstrings and Comments:** Add/update documentation for all new/modified functions.

8.  **Variable Naming:** Review variable names for clarity and consistency.

## Next Steps

- Implement the refactoring according to this plan, likely in "code" mode.
- Test thoroughly to ensure functionality remains unchanged.