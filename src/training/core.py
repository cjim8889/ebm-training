# Refactored src/training/core.py

from typing import Any, Callable, Dict, List, Optional, Tuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jaxtyping import Array, Float, PyTree

import wandb
import logfire # Added logfire import
from src.distributions import AnnealedDistribution, Target
from src.mcmc.sampling import sample_with_mcmc
from src.utils.eval import (
    aggregate_eval_metrics,
    evaluate_model,
    log_metrics,
    save_model_if_best,
)
from src.utils.optimization import get_optimizer  # Keep get_optimizer
from src.utils.schedule import constant_then_cyclic_cosine_schedule

# Added: Import new time utils
from . import time_utils
from .config import TrainingExperimentConfig
from .loss import Particle, calculate_validation_loss_and_plot, loss_fn
from .normalizing_constant import (
    estimate_log_Z_t,
)
from .augmentation import batch_augment_chain # Added for refactoring

# === Sample Generation (Unchanged, added type hints) ===

@logfire.instrument('Executing {__qualname__}')
def generate_samples_with_optional_mcmc(
    key: jax.random.PRNGKey,
    v_theta: Callable,
    ts: Float[Array, " time"],
    path_distribution: AnnealedDistribution,
    config: TrainingExperimentConfig,
    mcmc_method: Optional[str] = None,
    force_finite: bool = False,
    num_samples: Optional[int] = None,
    lambda_factor: Float[Array, ""] = 1.0,
) -> Dict[str, Any]:
    """
    Generate samples with or without MCMC correction.

    Args:
        key: Random key
        v_theta: Velocity field model
        ts: Time steps
        path_distribution: Annealed distribution
        config: Training experiment configuration
        mcmc_method: MCMC method to use (e.g., "smc", "hmc", "none"). Defaults to config.mcmc.method.
        force_finite: Whether to replace non-finite values with finite ones
        num_samples: Number of sample particles. Defaults to config.sampling.num_particles.
        lambda_factor: Factor controlling the contribution of the velocity field

    Returns:
        Samples dictionary containing at least "positions" and "weights".
    """
    # We generate samples in full precision
    ts_compute = config.mp_policy.cast_to_output(ts)

    # Determine MCMC method
    mcmc_method = config.mcmc.method if mcmc_method is None else mcmc_method

    _num_samples = config.sampling.num_particles if num_samples is None else num_samples

    # Generate initial samples
    initial_samples = path_distribution.sample_initial(key, (_num_samples,)).astype(config.mp_policy.output_dtype)

    # Use unified MCMC interface
    samples = sample_with_mcmc(
        key=key,
        initial_samples=initial_samples,
        v_theta=v_theta,
        time_dependent_log_density=path_distribution.time_dependent_log_prob,
        incremental_log_delta=path_distribution.incremental_log_delta,
        mcmc_method=mcmc_method,
        ts=ts_compute,
        shift_fn=config.density.shift_fn,
        use_shortcut=config.training.use_shortcut,
        num_steps=config.mcmc.num_steps,
        integration_steps=config.mcmc.num_integration_steps,
        eta=config.mcmc.step_size,
        ess_threshold=config.mcmc.ess_threshold,
        estimate_covariance=False, # Currently hardcoded
        solver=config.integration.method,
        lambda_factor=lambda_factor,
    )

    if force_finite:
        samples["positions"] = jnp.nan_to_num(
            samples["positions"], nan=0.0, posinf=1.0, neginf=-1.0
        )
    chex.assert_type(samples["positions"], config.mp_policy.output_dtype)
    chex.assert_shape(samples["positions"], (None, _num_samples, None)) # (time, num_samples, dim)

    return samples

# === Jitted Loss Function (Unchanged) ===
jitted_loss_fn = eqx.filter_jit(loss_fn)

# === Refactored Training Logic ===

# --- Initialization Helpers ---

@logfire.instrument('Executing {__qualname__}')
def _setup_optimizer(config: TrainingExperimentConfig) -> Tuple[optax.GradientTransformation, Callable]:
    """Sets up the optimizer and learning rate schedule based on the configuration."""
    if config.training.gradient_clip_norm is not None:
        gradient_clipping = optax.clip_by_global_norm(config.training.gradient_clip_norm)
    elif config.training.gradient_clip is not None:
        gradient_clipping = optax.clip(config.training.gradient_clip)
    else:
        gradient_clipping = optax.identity()

    lr_schedule_fn: Callable[[int], float]
    if config.training.use_schedule:
        lr_schedule_fn = constant_then_cyclic_cosine_schedule(
            constant_value=config.training.learning_rate,
            initial_steps=config.training.schedule_warmup_steps,
            cycle_steps=config.training.steps_per_epoch * config.training.schedule_decay_epoch,
            peak_value=config.training.learning_rate,
            end_value=config.training.schedule_end_value,
        )
    else:
        # If not using schedule, return a function that always returns the constant LR
        lr_schedule_fn = lambda step: config.training.learning_rate


    base_optimizer = get_optimizer(
        config.training.optimizer,
        lr_schedule_fn, # Pass the function itself
        weight_decay=config.training.weight_decay,
        b1=config.training.beta1,
        b2=config.training.beta2,
        eps=config.training.epsilon,
        momentum=config.training.momentum,
        nesterov=config.training.nesterov,
        noise_scale=config.training.noise_scale,
    )

    optimizer = optax.chain(optax.zero_nans(), gradient_clipping, base_optimizer)

    if config.training.every_k_schedule > 1:
        optimizer = optax.MultiSteps(
            optimizer, every_k_schedule=config.training.every_k_schedule
        )

    # Consider adding apply_if_finite if needed:
    # optimizer = optax.apply_if_finite(optimizer, max_finite_updates=5)

    return optimizer, lr_schedule_fn # Return schedule_fn as well for logging


@logfire.instrument('Executing {__qualname__}')
def _setup_path_distribution(
    initial_density: Target,
    target_density: Target,
    config: TrainingExperimentConfig
) -> AnnealedDistribution:
    """Creates the annealed path distribution."""
    return AnnealedDistribution(
        initial_density=initial_density,
        target_density=target_density,
        method=config.density.annealing_path,
    )

@logfire.instrument('Executing {__qualname__}')
def _generate_initial_validation_set(
    key: jax.random.PRNGKey,
    v_theta: Callable, # Should be PyTree
    validation_ts: Float[Array, " time"],
    path_distribution: AnnealedDistribution,
    config: TrainingExperimentConfig
) -> Tuple[jax.random.PRNGKey, Particle]:
    """Generates the initial set of particles for validation."""
    key, subkey = jax.random.split(key)
    # Use separate config or decide on validation set size
    num_val_samples = config.density.n_samples_eval

    print(f"Generating validation set with {num_val_samples} samples...")

    validation_samples_dict = generate_samples_with_optional_mcmc(
        key=subkey,
        v_theta=v_theta,
        ts=validation_ts,
        path_distribution=path_distribution,
        config=config,
        mcmc_method="smc", # Use SMC for validation set generation
        force_finite=True,
        num_samples=num_val_samples,
    )

    # Estimate Log Z for the validation set times
    validation_log_Z_t = estimate_log_Z_t(
        xs=validation_samples_dict["positions"],
        weights=validation_samples_dict["weights"],
        ts=validation_ts,
        time_derivative_log_density=path_distribution.time_derivative,
        # Don't use control variate for validation Log Z estimation for consistency?
    ).flatten()

    # Reshape positions: (time, num_samples, dim) -> (time * num_samples, dim)
    num_time_steps, _, dim = validation_samples_dict["positions"].shape
    reshaped_positions = validation_samples_dict["positions"].reshape(-1, dim)

    # Repeat ts and log_Z_t to match particles
    repeated_t = jnp.repeat(validation_ts, num_val_samples)
    repeated_log_Z_t = jnp.repeat(validation_log_Z_t, num_val_samples)

    validation_particles = Particle(
        x=reshaped_positions,
        t=repeated_t,
        log_Z_t=repeated_log_Z_t,
    )
    return key, validation_particles


# --- Step Logic Helpers ---

# Keep the core step JITted for performance
@eqx.filter_jit
def _execute_jitted_step(
    key: jax.random.PRNGKey,
    v_theta: PyTree,
    opt_state: PyTree,
    optimizer: optax.GradientTransformation, # Static arg
    particles: Particle, # Dynamic arg
    path_distribution_time_derivative: Callable, # Static arg
    path_distribution_score_fn: Callable, # Static arg
    config_density_shift_fn: Callable, # Static arg
    config_training_estimator: str, # Static arg
    config_training_n_probes: int, # Static arg
    config_training_use_combined_loss: bool, # Static arg
    config_training_shortcut_weight: float, # Static arg
    config_training_random_alpha: bool, # Static arg
    config_model_dropout: Optional[float], # Static arg
) -> Tuple[PyTree, PyTree, Float[Array, ""]]:
    """Performs a single training step (loss, gradients, update). JIT compiled."""
    key, dropout_key = jax.random.split(key)

    # Define the loss function specific to this step's context
    def compute_loss(model):
        return loss_fn(
            model,
            particles,
            path_distribution_time_derivative,
            path_distribution_score_fn,
            config_density_shift_fn,
            config_training_estimator,
            key=key, # Use the outer key passed to this function
            n_probes=config_training_n_probes,
            combined_loss=config_training_use_combined_loss,
            shortcut_weight=config_training_shortcut_weight,
            random_alpha=config_training_random_alpha,
            dropout_key=dropout_key if config_model_dropout is not None else None,
        )

    loss, grads = eqx.filter_value_and_grad(compute_loss)(v_theta)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(v_theta, eqx.is_array))
    v_theta = eqx.apply_updates(v_theta, updates)
    return v_theta, opt_state, loss


@logfire.instrument('Executing {__qualname__}')
def _prepare_step_batch(
    key: jax.random.PRNGKey,
    samples: Float[Array, "time batch dim"],
    current_ts: Float[Array, " time"],
    log_Z_t: Float[Array, " time"],
    config: TrainingExperimentConfig,
) -> Tuple[jax.random.PRNGKey, Particle, Float[Array, "batch_size dim"]]:
    """Selects a batch of chains and prepares the Particle object for a training step."""
    key, subkey = jax.random.split(key)
    time_steps, num_total_particles, dim = samples.shape
    # Number of distinct trajectories to sample per step
    num_chains_per_step = config.training.time_batch_size

    if num_chains_per_step > num_total_particles:
        print(f"Warning: time_batch_size ({num_chains_per_step}) > num available particles ({num_total_particles}). Sampling with replacement.")
        replace = True
    else:
        replace = False

    chain_indices = jax.random.choice(
        subkey, num_total_particles, shape=(num_chains_per_step,), replace=replace
    )
    # Select chains: (time_steps, num_chains_per_step, dim)
    selected_chains_time_major = samples[:, chain_indices, :]
    # Reshape to batch major: (time_steps * num_chains_per_step, dim)
    # This combines time and chain index into the batch dimension for the step
    selected_chains = selected_chains_time_major.reshape(time_steps * num_chains_per_step, dim)

    # Repeat time and log_Z_t for the selected batch
    selected_t = jnp.repeat(current_ts, num_chains_per_step)
    selected_log_Z_t = jnp.repeat(log_Z_t, num_chains_per_step)

    training_particles = Particle(
        x=selected_chains, # Will be potentially augmented later
        t=selected_t,
        log_Z_t=selected_log_Z_t,
        # d and loss_weight are not used in the original code snippet for training_particles
        d=None,
        loss_weight=None,
    )
    # Return selected_chains before augmentation for the augmentation function
    return key, training_particles, selected_chains

@logfire.instrument('Executing {__qualname__}')
def _apply_augmentations(
    key: jax.random.PRNGKey,
    selected_chains: Float[Array, "batch_size dim"],
    config: TrainingExperimentConfig,
) -> Tuple[jax.random.PRNGKey, Float[Array, "batch_size dim"]]:
    """Applies configured augmentations to the selected chains."""
    augmented_chains = selected_chains
    if config.training.augment:
        with logfire.span('Augmenting chains'):
            key, subkey = jax.random.split(key)
            batch_size = augmented_chains.shape[0]
            keys_aug = jax.random.split(subkey, batch_size)
            # Pass n_particles from density config, assuming it matches the structure
            augmented_chains = batch_augment_chain(
                augmented_chains, keys_aug, config.training.translation_scale, config.density.n_particles
            )

    if config.training.perturb:
        with logfire.span('Adding noise to chains'):
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(
                subkey, augmented_chains.shape, dtype=config.mp_policy.output_dtype
            ) * config.training.perturbation_scale
            augmented_chains = augmented_chains + noise

    return key, augmented_chains


# --- Epoch Logic Helpers ---

@logfire.instrument('Executing {__qualname__}')
def _compute_lambda_factor(
    global_step: int,
    lambda_total_steps: int,
    lambda_max: float,
    # lambda_epochs: int # Removed as redundant if lambda_total_steps is correct
) -> Float[Array, ""]:
    """Calculates the lambda factor based on the current training progress."""
    progress_ratio = jnp.minimum(1.0, global_step / lambda_total_steps)
    # Exponential growth from almost 0 to lambda_max
    return jnp.array(lambda_max * (1.0 - jnp.exp(-5.0 * progress_ratio)), dtype=jnp.float32)


@logfire.instrument('Executing {__qualname__}')
def _maybe_estimate_log_z(
    key: jax.random.PRNGKey,
    epoch: int,
    v_theta: PyTree,
    current_ts: Float[Array, " time"],
    base_ts: Float[Array, " time"], # Needed for continuous time sampling
    path_distribution: AnnealedDistribution,
    config: TrainingExperimentConfig,
    log_Z_t_ref: List[Optional[Float[Array, " time"]]],
    current_lambda: Float[Array, ""]
) -> Tuple[jax.random.PRNGKey, Float[Array, " time"], Float[Array, " time"], Optional[Dict[str, Any]]]:
    """Estimates log_Z_t if required for the current epoch, otherwise reuses the previous value."""
    mcmc_samples = None # Initialize
    should_estimate = (epoch % config.training.log_z_estimation_frequency == 0) or \
                      (epoch == 0) or \
                      (log_Z_t_ref[0] is None)

    ts_for_estimation = current_ts # Start with current ts

    if should_estimate:
        print(f"Epoch {epoch}: Estimating log Z(t)...")
        key, subkey_sample, subkey_time = jax.random.split(key, 3)

        # Handle time steps for estimation (especially if continuous)
        if config.integration.continuous_time:
             # Resample time steps for estimation if continuous
             # Changed: Use time_utils.sample_continuous_time
             ts_for_estimation = time_utils.sample_continuous_time(subkey_time, base_ts)
             print(f"Epoch {epoch}: Resampled continuous time for Log Z estimation.")

        mcmc_samples = generate_samples_with_optional_mcmc(
            subkey_sample, v_theta, ts_for_estimation, path_distribution, config,
            mcmc_method=config.mcmc.method, force_finite=True, lambda_factor=current_lambda
        )

        # Note: ASMC specific updates were removed here (lines 473-482).
        # If ASMC were used, it would update current_ts and ts_for_estimation.

        # Estimate Log Z using the determined time steps (ts_for_estimation)
        print(f"Using standard log Z estimation (Control Variate: {config.mcmc.use_control_variate}).")
        log_Z_t = estimate_log_Z_t(
            mcmc_samples["positions"],
            mcmc_samples["weights"],
            ts_for_estimation, # Use potentially updated ts
            path_distribution.time_derivative,
            v_theta=v_theta,
            score_fn=path_distribution.score_fn,
            use_control_variate=config.mcmc.use_control_variate,
            use_shortcut=config.training.use_shortcut,
        )

        log_Z_t = log_Z_t.flatten()
        log_Z_t = jax.lax.stop_gradient(log_Z_t)
        log_Z_t_ref[0] = log_Z_t # Update the shared reference

        # Logging
        log_Z_t_to_log = jnp.nan_to_num(log_Z_t, nan=0.0, posinf=1.0, neginf=-1.0)
        log_data = {"log_Z_t": log_Z_t_to_log, "epoch": epoch}
        if "ess" in mcmc_samples:
             log_data["ess"] = mcmc_samples["ess"]

        if not config.offline:
            wandb.log(log_data)
        else:
            print(f"Epoch {epoch}, Log Z(t) estimated (shape {log_Z_t.shape})")
            if "ess" in mcmc_samples:
                print(f"Epoch {epoch}, MCMC Samples ESS: {mcmc_samples['ess']}")

    else:
        # Reuse the previous estimation
        log_Z_t = log_Z_t_ref[0]
        if log_Z_t is None:
             # This should not happen if estimation occurs at epoch 0
             raise RuntimeError("log_Z_t_ref[0] is None, but should have been estimated in epoch 0.")
        print(f"Epoch {epoch}: Reusing previous log Z(t) estimation.")
        if not config.offline:
            wandb.log({"log_Z_t (reused)": jnp.nan_to_num(log_Z_t, nan=0.0, posinf=1.0, neginf=-1.0), "epoch": epoch})

    # Return the potentially updated current_ts (especially if ASMC was used)
    return key, log_Z_t, current_ts, mcmc_samples


@logfire.instrument('Executing {__qualname__}')
def _prepare_epoch_samples(
    key: jax.random.PRNGKey,
    v_theta: PyTree,
    current_ts: Float[Array, " time"],
    path_distribution: AnnealedDistribution,
    config: TrainingExperimentConfig,
    mcmc_samples_from_logz: Optional[Dict[str, Any]], # Samples from log Z estimation step
    current_lambda: Float[Array, ""]
) -> Tuple[jax.random.PRNGKey, Float[Array, "time batch dim"]]:
    """Prepares the pool of samples for the epoch's training steps."""
    base_mcmc_samples: Dict[str, Any]

    if mcmc_samples_from_logz is None:
        # This happens if log Z wasn't estimated this epoch. Need to generate base samples.
        # Use the standard MCMC method defined in config for generating training samples.
        print("Epoch: Generating base MCMC samples as none were provided (log Z reused).")
        key, subkey = jax.random.split(key)
        base_mcmc_samples = generate_samples_with_optional_mcmc(
            subkey, v_theta, current_ts, path_distribution, config,
            mcmc_method=config.mcmc.method, force_finite=True, lambda_factor=current_lambda
        )
    else:
        # Reuse samples generated during log Z estimation
        print("Epoch: Reusing MCMC samples generated during log Z estimation.")
        base_mcmc_samples = mcmc_samples_from_logz

    base_positions = base_mcmc_samples["positions"] # Shape (time, num_particles, dim)

    if config.training.use_decoupled_loss:
        print("Epoch: Generating additional samples for decoupled loss.")
        key, subkey = jax.random.split(key)
        # Generate samples using only the velocity field (no MCMC correction)
        v_theta_samples_dict = generate_samples_with_optional_mcmc(
            subkey, v_theta, current_ts, path_distribution, config,
            mcmc_method="none", force_finite=True, lambda_factor=current_lambda,
            # Ensure same number of particles as base_mcmc_samples
            num_samples=base_positions.shape[1]
        )
        v_theta_samples = v_theta_samples_dict["positions"]
        # Concatenate along the particle batch dimension
        samples = jnp.concatenate([base_positions, v_theta_samples], axis=1)
        print(f"Epoch: Concatenated samples for decoupled loss. New shape: {samples.shape}")
    else:
        samples = base_positions
        print(f"Epoch: Using standard samples. Shape: {samples.shape}")

    # Basic shape check
    chex.assert_rank(samples, 3) # time, batch, dim

    return key, samples


@logfire.instrument('Executing {__qualname__}')
def _run_steps_for_epoch(
    key: jax.random.PRNGKey,
    v_theta: PyTree,
    opt_state: PyTree,
    optimizer: optax.GradientTransformation,
    lr_schedule_fn: Callable[[int], float], # Pass the schedule function
    epoch: int,
    samples: Float[Array, "time batch dim"],
    current_ts: Float[Array, " time"],
    log_Z_t: Float[Array, " time"],
    path_distribution: AnnealedDistribution,
    config: TrainingExperimentConfig,
) -> Tuple[jax.random.PRNGKey, PyTree, PyTree, Float[Array, ""]]:
    """Runs all training steps within a single epoch."""
    epoch_loss = 0.0
    steps_per_epoch = config.training.steps_per_epoch

    # Pre-compile static arguments for the JITted step function
    static_args = (
        optimizer,
        path_distribution.time_derivative,
        path_distribution.score_fn,
        config.density.shift_fn,
        config.training.estimator,
        config.training.n_probes,
        config.training.use_combined_loss,
        config.training.shortcut_weight,
        config.training.random_alpha,
        config.model.dropout,
    )

    for s in range(steps_per_epoch):
        global_step_count = epoch * steps_per_epoch + s
        key, subkey_batch, subkey_aug, subkey_step = jax.random.split(key, 4)

        # 1. Prepare Batch
        subkey_batch, training_particles_pre_aug, selected_chains = _prepare_step_batch(
            subkey_batch, samples, current_ts, log_Z_t, config
        )

        # 2. Apply Augmentations
        subkey_aug, augmented_chains = _apply_augmentations(
            subkey_aug, selected_chains, config
        )

        # Update particles with augmented chains
        training_particles = training_particles_pre_aug._replace(x=augmented_chains)

        # 3. Execute Training Step (using the JITted function)
        v_theta, opt_state, loss = _execute_jitted_step(
            subkey_step, v_theta, opt_state, training_particles, *static_args
        )
        epoch_loss += loss

        # 4. Log Step Loss (periodically)
        if s % 20 == 0: # Log every 20 steps
            current_lr = lr_schedule_fn(global_step_count)
            step_metrics = {"loss": loss, "learning_rate": current_lr, "epoch": epoch, "step": s, "global_step": global_step_count}
            if not config.offline:
                wandb.log(step_metrics)
            else:
                print(f"Epoch {epoch}, Step {s}, Loss: {loss:.4f}, LR: {current_lr:.6f}")

    avg_epoch_loss = epoch_loss / steps_per_epoch
    return key, v_theta, opt_state, avg_epoch_loss


@logfire.instrument('Executing {__qualname__}')
def _calculate_and_log_epoch_metrics(
    key: jax.random.PRNGKey,
    epoch: int,
    avg_train_loss: Float[Array, ""],
    v_theta: PyTree,
    validation_particles: Particle, # Use the pre-generated validation set
    path_distribution: AnnealedDistribution,
    config: TrainingExperimentConfig,
    lr_schedule_fn: Callable[[int], float], # Pass the schedule function
) -> Tuple[jax.random.PRNGKey, Float[Array, ""]]:
    """Calculates validation loss and logs epoch summary metrics."""
    key, dropout_key = jax.random.split(key)
    global_step_count = epoch * config.training.steps_per_epoch # LR at start of epoch

    # Calculate validation loss using the dedicated validation set
    # Use settings consistent with how validation should be performed (e.g., no dropout, specific estimator)
    val_loss = jitted_loss_fn(
        v_theta,
        validation_particles, # Use the dedicated validation set
        path_distribution.time_derivative,
        path_distribution.score_fn,
        config.density.shift_fn,
        estimator="none", # Typically 'none' for validation
        key=key, # Provide a key
        n_probes=config.training.n_probes, # Use same probes? Or 1? Let's use config for now.
        combined_loss=False, # No combined loss for validation
        shortcut_weight=config.training.shortcut_weight, # Include shortcut? Yes, likely.
        random_alpha=False, # No random alpha for validation
        # Dropout during validation? Usually no. Let's disable it here.
        dropout_key=None, # dropout_key if config.model.dropout is not None and config.training.get('dropout_in_val', False) else None,
    )

    epoch_lr = lr_schedule_fn(global_step_count)
    epoch_metrics = {
        "epoch": epoch,
        "average_train_loss": avg_train_loss,
        "validation_loss_epoch_end": val_loss, # More specific name
        "epoch_learning_rate": epoch_lr,
        "global_step": (epoch + 1) * config.training.steps_per_epoch # Step count at end of epoch
    }

    if not config.offline:
        wandb.log(epoch_metrics)
    
    print(f"--- Epoch {epoch} Summary ---")
    print(f"  Avg Train Loss: {avg_train_loss:.4f}")
    print(f"  Validation Loss: {val_loss:.4f}")
    print(f"  Learning Rate at Epoch Start: {epoch_lr:.6f}")
    print("----------------------")

    return key, val_loss # Return validation loss for potential use in saving


@logfire.instrument('Executing {__qualname__}')
def _maybe_evaluate_and_save(
    key: jax.random.PRNGKey,
    epoch: int,
    v_theta: PyTree,
    config: TrainingExperimentConfig,
    path_distribution: AnnealedDistribution,
    target_density: Target,
    validation_particles: Particle, # For plotting validation loss curve
    validation_ts: Float[Array, " time"], # For plotting validation loss curve
    best_metrics: List[Tuple[float, int]], # List of (metric_value, version)
    model_version: int,
) -> Tuple[jax.random.PRNGKey, List[Tuple[float, int]], int]:
    """Performs model evaluation and saves the best model periodically."""
    if epoch % config.training.eval_frequency == 0:
        print(f"--- Epoch {epoch}: Running Evaluation ---")
        # Run multiple evaluations (original code had loop for 1 iteration)
        all_eval_results = []
        num_eval_runs = config.training.get("num_eval_runs", 1) # Make configurable
        for i in range(num_eval_runs):
            key, subkey = jax.random.split(key)
            print(f"  Evaluation Run {i+1}/{num_eval_runs}...")
            eval_metrics = evaluate_model(
                subkey,
                v_theta,
                config,
                path_distribution,
                target_density,
                config.sampling.num_timesteps, # Pass num_timesteps from config
            )
            all_eval_results.append(eval_metrics)

        # Process and log metrics
        aggregated_metrics = aggregate_eval_metrics(all_eval_results)
        log_metrics(aggregated_metrics, config, epoch=epoch) # Pass epoch for logging

        # Calculate and plot validation loss curve
        print("  Calculating validation loss curve...")
        # Ensure calculate_validation_loss_and_plot uses appropriate settings (e.g., no dropout)
        validation_loss_curve, validation_plt = calculate_validation_loss_and_plot(
            v_theta,
            validation_particles,
            path_distribution,
            validation_ts, # Use the ts corresponding to validation_particles
            time_batch_size=config.training.time_batch_size, # Reuse config params
            batch_size=config.sampling.batch_size,
            # Pass other relevant config if needed by the function
        )
        # validation_loss_curve is the mean loss per time step

        if not config.offline:
            # Log the mean validation loss across time
            mean_validation_loss_curve = jnp.mean(validation_loss_curve)
            wandb.log({
                "validation_loss_curve_mean": mean_validation_loss_curve,
                "validation_loss_plot": wandb.Image(validation_plt),
                "epoch": epoch
            })
            print(f"  Logged validation loss curve (Mean: {mean_validation_loss_curve:.4f}) and plot to WandB.")

            # Save model if best based on aggregated metrics (e.g., W2)
            # Make the primary metric configurable
            primary_metric_key = config.training.get('primary_eval_metric', 'wasserstein2_mean')
            if primary_metric_key in aggregated_metrics:
                 current_metric_value = aggregated_metrics[primary_metric_key]
                 print(f"  Checking if model is best based on {primary_metric_key}: {current_metric_value:.4f}")
                 # save_model_if_best needs the metric value, not the whole dict
                 best_metrics, model_version = save_model_if_best(
                     v_theta,
                     current_metric_value, # Pass the specific metric value
                     best_metrics,
                     model_version,
                     target_density, # Pass target density for saving context
                     # metric_key=primary_metric_key, # Function likely uses the passed value directly
                 )
            else:
                 print(f"  Warning: Primary metric '{primary_metric_key}' not found in evaluation results {list(aggregated_metrics.keys())}. Skipping save_model_if_best.")

        else:
            # Show plot locally if offline
            print(f"  Validation Loss Curve (Mean): {jnp.mean(validation_loss_curve):.4f}")
            # Check if plot is None before showing
            if validation_plt is not None:
                plt.show() # Display the plot generated by calculate_validation_loss_and_plot
            else:
                print("  (Plotting disabled or failed)")


        # Close the plot figure if it exists
        if validation_plt is not None:
            plt.close(validation_plt)
        print(f"--- Epoch {epoch}: Evaluation Complete ---")

    return key, best_metrics, model_version


# --- Main Training Loop ---

@logfire.instrument('Executing {__qualname__}')
def _run_training_loop(
    key: jax.random.PRNGKey,
    v_theta: PyTree,
    opt_state: PyTree,
    optimizer: optax.GradientTransformation,
    lr_schedule_fn: Callable[[int], float],
    path_distribution: AnnealedDistribution,
    target_density: Target, # Needed for evaluation saving
    base_ts: Float[Array, " time"],
    validation_particles: Particle,
    validation_ts: Float[Array, " time"], # TS for validation particles
    log_Z_t_ref: List[Optional[Float[Array, " time"]]],
    best_metrics: List[Tuple[float, int]],
    model_version: int,
    config: TrainingExperimentConfig,
) -> Tuple[PyTree, List[Tuple[float, int]]]:
    """Runs the main training loop over all epochs."""

    current_ts = base_ts # Initialize current_ts
    mcmc_samples_from_logz: Optional[Dict[str, Any]] = None # Samples from last logZ step

    lambda_max = config.mcmc.lambda_max
    lambda_epochs = config.mcmc.lambda_epochs
    # Ensure lambda_total_steps is at least 1 to avoid division by zero
    lambda_total_steps = max(1, lambda_epochs * config.training.steps_per_epoch)


    for epoch in range(config.training.num_epochs):
        print(f"\n=== Starting Epoch {epoch}/{config.training.num_epochs - 1} ===")
        key, subkey_epoch = jax.random.split(key)
        global_step_start_epoch = epoch * config.training.steps_per_epoch

        # 1. Calculate Lambda Factor for the epoch
        current_lambda = _compute_lambda_factor(global_step_start_epoch, lambda_total_steps, lambda_max)
        if not config.offline:
            wandb.log({"lambda_factor": current_lambda, "epoch": epoch, "global_step": global_step_start_epoch})
        else:
            print(f"Epoch {epoch}, Lambda Factor: {current_lambda:.4f}")

        # 2. Update Time Steps if Continuous
        # Note: Previously, ASMC handled time step updates within _maybe_estimate_log_z.
        if config.integration.continuous_time: # Removed redundant 'and config.mcmc.method != "asmc"'
            subkey_epoch, subkey_time = jax.random.split(subkey_epoch)
            # Changed: Use time_utils.sample_continuous_time
            current_ts = time_utils.sample_continuous_time(subkey_time, base_ts)
            print(f"Epoch {epoch}: Sampled new continuous time steps.")

        # 3. Estimate Log Z (if needed) - This might update current_ts if ASMC is used
        subkey_epoch, log_Z_t, current_ts, mcmc_samples_from_logz = _maybe_estimate_log_z(
            subkey_epoch, epoch, v_theta, current_ts, base_ts, path_distribution, config, log_Z_t_ref, current_lambda
        )

        # 4. Prepare Samples for Epoch Steps
        subkey_epoch, epoch_training_samples = _prepare_epoch_samples(
            subkey_epoch, v_theta, current_ts, path_distribution, config, mcmc_samples_from_logz, current_lambda
        )

        # 5. Run Training Steps for Epoch
        subkey_epoch, v_theta, opt_state, avg_epoch_loss = _run_steps_for_epoch(
            subkey_epoch, v_theta, opt_state, optimizer, lr_schedule_fn, epoch,
            epoch_training_samples, current_ts, log_Z_t, path_distribution, config
        )

        # 6. Calculate and Log Epoch Metrics (Validation Loss)
        subkey_epoch, _ = _calculate_and_log_epoch_metrics( # val_loss not needed here
            subkey_epoch, epoch, avg_epoch_loss, v_theta, validation_particles,
            path_distribution, config, lr_schedule_fn
        )

        # 7. Evaluate and Save Model (Periodically)
        subkey_epoch, best_metrics, model_version = _maybe_evaluate_and_save(
            subkey_epoch, epoch, v_theta, config, path_distribution, target_density,
            validation_particles, validation_ts, best_metrics, model_version
        )

        print(f"=== Finished Epoch {epoch} ===")

    return v_theta, best_metrics # Return final model and best metrics list


# --- Finalization Helper ---

@logfire.instrument('Executing {__qualname__}')
def _finalize_training(
    config: TrainingExperimentConfig,
    best_metrics: List[Tuple[float, int]]
):
    """Logs final summary information and finishes WandB run."""
    if not config.offline and wandb.run is not None:
        if len(best_metrics) > 0:
            print("\n--- Training Complete ---")
            print("Best Model Metrics (Metric, Version):", best_metrics)
            # Log summary to WandB
            try:
                wandb.run.summary["best_metrics_values"] = [metric for metric, _ in best_metrics]
                wandb.run.summary["best_metrics_versions"] = [version for _, version in best_metrics]
                # Log the single best metric value if desired
                # Assuming lower is better for the primary metric used in save_model_if_best
                best_metric_overall = min(best_metrics, key=lambda item: item[0])
                wandb.run.summary["best_metric_overall_value"] = best_metric_overall[0]
                wandb.run.summary["best_metric_overall_version"] = best_metric_overall[1]
                print(f"Logged best metrics summary to WandB. Best overall: {best_metric_overall}")
            except Exception as e:
                print(f"Error logging summary to WandB: {e}")
        else:
            print("\n--- Training Complete (No best models saved based on criteria) ---")

        try:
            wandb.finish()
            print("WandB run finished.")
        except Exception as e:
            print(f"Error finishing WandB run: {e}")
    else:
        print("\n--- Training Complete (Offline Mode or WandB not initialized) ---")


# --- Main Entry Point ---

@logfire.instrument('Executing {__qualname__}')
def train_velocity_field(
    key: jax.random.PRNGKey,
    initial_density: Target,
    target_density: Target,
    v_theta: PyTree, # Explicitly PyTree (Equinox module)
    config: TrainingExperimentConfig,
) -> PyTree:
    """
    Trains a velocity field model (v_theta).

    Args:
        key: JAX random key.
        initial_density: The initial distribution P_0.
        target_density: The target distribution P_1.
        v_theta: The velocity field model (an Equinox module).
        config: The training configuration.

    Returns:
        The trained velocity field model.
    """
    print("--- Starting Velocity Field Training ---")
    print(f"Config: {config}") # Log config at start

    # 1. Initialization
    key, subkey_init_opt, subkey_init_ts, subkey_init_path, subkey_init_val, subkey_loop = jax.random.split(key, 6)

    best_metrics: List[Tuple[float, int]] = []
    model_version = 0
    log_Z_t_ref: List[Optional[Float[Array, " time"]]] = [None] # Mutable reference for log Z

    optimizer, lr_schedule_fn = _setup_optimizer(config)
    # Changed: Use time_utils.setup_time_schedule
    base_ts = time_utils.setup_time_schedule(
        schedule=config.integration.schedule,
        num_timesteps=config.sampling.num_timesteps,
        gamma=config.integration.get('gamma', None) # Pass gamma if needed
    )
    path_distribution = _setup_path_distribution(initial_density, target_density, config)
    opt_state = optimizer.init(eqx.filter(v_theta, eqx.is_inexact_array))

    # Generate validation set using a fixed linear schedule for consistency
    validation_ts_init = jnp.linspace(0, 1.0, config.sampling.num_timesteps)
    subkey_init_val, validation_particles = _generate_initial_validation_set(
        subkey_init_val, v_theta, validation_ts_init, path_distribution, config
    )
    print(f"Initialized Optimizer, Time Steps (Base shape: {base_ts.shape}), Path Distribution.")
    print(f"Generated Initial Validation Set (Particles shape: {validation_particles.x.shape}, TS shape: {validation_ts_init.shape}).")


    # 2. Run Training Loop
    v_theta, best_metrics = _run_training_loop(
        key=subkey_loop, # Use the dedicated key for the loop
        v_theta=v_theta,
        opt_state=opt_state,
        optimizer=optimizer,
        lr_schedule_fn=lr_schedule_fn,
        path_distribution=path_distribution,
        target_density=target_density,
        base_ts=base_ts,
        validation_particles=validation_particles,
        validation_ts=validation_ts_init, # Pass the ts used for validation particles
        log_Z_t_ref=log_Z_t_ref,
        best_metrics=best_metrics,
        model_version=model_version, # Pass initial version
        config=config,
    )

    # 3. Finalization
    _finalize_training(config, best_metrics)

    print("--- Finished Velocity Field Training ---")
    return v_theta
