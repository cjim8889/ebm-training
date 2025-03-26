# Logfire Instrumentation Plan for src/training/core.py

**1. Goal:**
Instrument key functions within `src/training/core.py` using Logfire to capture performance metrics, specifically execution duration.

**2. Approach:**
We'll primarily use the `@logfire.instrument()` decorator. This is generally cleaner for instrumenting entire functions compared to manually adding `with logfire.span(): ...` blocks. The decorator automatically captures the function's execution time, arguments (by default), and return values.

**3. Implementation Steps:**

*   **Add Import:** Add `import logfire` at the top of `src/training/core.py`.
*   **Ensure Configuration:** Verify that `logfire.configure()` is called appropriately in the main script that runs the training (e.g., `main.py` or an experiment script like `experiments/gen_lj13q.py`). Logfire needs to be configured before any instrumentation is used. We won't add the configuration call within `core.py` itself.
*   **Apply Decorators:** Add the `@logfire.instrument()` decorator to the following functions in `src/training/core.py`. We'll use a basic message template like `@logfire.instrument('Executing {__qualname__}')` which automatically includes the function name.
    *   `generate_samples_with_optional_mcmc`
    *   `_setup_optimizer`
    *   `_setup_path_distribution`
    *   `_generate_initial_validation_set`
    *   `_prepare_step_batch`
    *   `_apply_augmentations`
    *   `_compute_lambda_factor`
    *   `_maybe_estimate_log_z`
    *   `_prepare_epoch_samples`
    *   `_run_steps_for_epoch`
    *   `_calculate_and_log_epoch_metrics`
    *   `_maybe_evaluate_and_save`
    *   `_run_training_loop`
    *   `_finalize_training`
    *   `train_velocity_field`
*   **Note on JIT:** We will *not* apply the decorator directly to the JIT-compiled function `_execute_jitted_step`. Instrumenting JITted functions directly can sometimes cause issues or add undesirable overhead within the compiled code. The time spent in the JITted function will be included in the span of its calling function (`_run_steps_for_epoch`).

**4. High-Level Flow with Instrumentation Points:**

```mermaid
graph TD
    A[Start Training: train_velocity_field] --> B{Initialization};
    B --> C{_setup_optimizer};
    B --> D[time_utils.setup_time_schedule];
    B --> E{_setup_path_distribution};
    B --> F{_generate_initial_validation_set};
    F --> G[Run Training Loop: _run_training_loop];
    G --> H{Epoch Loop};
    H --> I{_compute_lambda_factor};
    H --> J[Update Time Steps];
    H --> K{_maybe_estimate_log_z};
    K -- Generates/Provides --> L[MCMC Samples];
    H --> M{_prepare_epoch_samples};
    M -- Uses --> L;
    M --> N[Epoch Samples Pool];
    H --> O{Run Steps for Epoch: _run_steps_for_epoch};
    O --> P{Step Loop};
    P --> Q{_prepare_step_batch};
    P --> R{_apply_augmentations};
    P --> S[Execute JITted Step];
    O --> T[Calculate Avg Epoch Loss];
    H --> U{_calculate_and_log_epoch_metrics};
    H --> V{_maybe_evaluate_and_save};
    V -- Calls --> W[evaluate_model];
    V -- Calls --> X[calculate_validation_loss_and_plot];
    V -- Calls --> Y[save_model_if_best];
    G --> Z{Finalization: _finalize_training};

    %% Instrumentation Points (using @logfire.instrument)
    style A fill:#f9f,stroke:#333,stroke-width:2px;
    style C fill:#f9f,stroke:#333,stroke-width:2px;
    style E fill:#f9f,stroke:#333,stroke-width:2px;
    style F fill:#f9f,stroke:#333,stroke-width:2px;
    style G fill:#f9f,stroke:#333,stroke-width:2px;
    style I fill:#f9f,stroke:#333,stroke-width:2px;
    style K fill:#f9f,stroke:#333,stroke-width:2px;
    style M fill:#f9f,stroke:#333,stroke-width:2px;
    style O fill:#f9f,stroke:#333,stroke-width:2px;
    style Q fill:#f9f,stroke:#333,stroke-width:2px;
    style R fill:#f9f,stroke:#333,stroke-width:2px;
    style U fill:#f9f,stroke:#333,stroke-width:2px;
    style V fill:#f9f,stroke:#333,stroke-width:2px;
    style Z fill:#f9f,stroke:#333,stroke-width:2px;

    %% Also instrument generate_samples_with_optional_mcmc
    subgraph core.py Functions to Instrument
        A; C; E; F; G; I; K; M; O; Q; R; U; V; Z;
        AA[generate_samples_with_optional_mcmc]
        style AA fill:#f9f,stroke:#333,stroke-width:2px;
    end

    F -- Calls --> AA;
    K -- Calls --> AA;

    %% Indicate non-instrumented JIT step
    style S fill:#eee,stroke:#999,stroke-width:1px,stroke-dasharray: 5 5;

    %% Indicate external calls (not instrumented here)
    style D fill:#ccf,stroke:#333,stroke-width:1px;
    style W fill:#ccf,stroke:#333,stroke-width:1px;
    style X fill:#ccf,stroke:#333,stroke-width:1px;
    style Y fill:#ccf,stroke:#333,stroke-width:1px;