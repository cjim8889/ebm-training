from dataclasses import dataclass, field
from typing import Optional, Literal, Callable, Any, List


@dataclass
class SamplingConfig:
    num_particles: int = 512  # N: Number of particles to simulate
    batch_size: int = 256  # B: Batch size for training
    num_timesteps: int = 32  # T: Number of timesteps for integration


@dataclass
class TrainingConfig:
    num_epochs: int = 200
    steps_per_epoch: int = 100
    learning_rate: float = 1e-3
    gradient_clip_norm: Optional[float] = None
    eval_frequency: int = 20
    optimizer: Literal[
        "adam",
        "adamw",
        "sgd",
        "rmsprop",
        "adafactor",
        "adagrad",
        "adadelta",
        "lamb",
        "lion",
        "adamax",
        "fromage",
        "noisy_sgd",
    ] = "adamw"
    use_decoupled_loss: bool = False  # Whether to use decoupled loss function
    # Optimizer parameters
    weight_decay: float = 0.0
    beta1: float = 0.9  # b1 for Adam-like optimizers
    beta2: float = 0.999  # b2 for Adam-like optimizers
    epsilon: float = 1e-8  # eps for numerical stability
    momentum: float = 0.9  # momentum for SGD
    nesterov: bool = False  # whether to use Nesterov momentum
    noise_scale: float = 0.01  # eta for noisy SGD
    time_batch_size: int = 32  # Number of time points to use in each batch
    shortcut_size: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    use_shortcut: bool = False
    use_hutchinson: bool = False
    n_probes: int = 5
    every_k_schedule: int = 1


@dataclass
class MCMCConfig:
    method: Literal["hmc", "smc", "esmc", "vsmc"] = "hmc"
    num_steps: int = 5
    num_integration_steps: int = 3
    step_size: float = 0.01  # eta: MCMC step size
    with_rejection: bool = False
    use_control_variate: bool = False


@dataclass
class IntegrationConfig:
    method: Literal["euler", "rk4"] = "euler"
    schedule: Literal["linear", "inverse_power", "power"] = "linear"
    continuous_time: bool = False
    dt_clip: Optional[float] = None


@dataclass
class ProgressiveTrainingConfig:
    enable: bool = False
    initial_timesteps: int = 16
    timestep_increment: int = 2
    update_frequency: int = 100


@dataclass
class ModelConfig:
    hidden_dim: int = 256
    num_layers: int = 3
    architecture: Literal["mlp", "pdn", "transformer", "emlp", "egnn"] = "mlp"


@dataclass
class DensityConfig:
    target_type: Literal[
        "gmm",
        "mw32",
        "dw4",
        "lj13",
        "sclj13",
        "dw4o",
        "tlj13",
        "lj13b",
        "lj13bt",
        "lj13c",
    ] = "gmm"
    initial_sigma: float = 20.0
    score_norm: Optional[float] = None
    annealing_path: Literal["linear", "geometric"] = "linear"
    shift_fn: Callable[[Any], Any] = field(default_factory=lambda: lambda x: x)
    input_dim: Optional[int] = None
    # LJ specific parameters
    n_particles: Optional[int] = None
    n_spatial_dim: Optional[int] = None
    alpha: Optional[float] = None
    epsilon_val: Optional[float] = None
    min_dr: Optional[float] = 1e-3
    m: Optional[int] = 1
    n: Optional[int] = 1.0
    c: Optional[float] = 0.5
    log_prob_clip: Optional[float] = None
    log_prob_clip_min: Optional[float] = None
    log_prob_clip_max: Optional[float] = None
    soft_clip: bool = False
    include_harmonic: bool = True
    cubic_spline: bool = False
    # Data paths for some targets
    data_path_test: Optional[str] = None
    data_path_val: Optional[str] = None
    data_path_train: Optional[str] = None
    n_samples_eval: Optional[int] = 1024


@dataclass
class TrainingExperimentConfig:
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    mcmc: MCMCConfig = field(default_factory=MCMCConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)
    progressive: ProgressiveTrainingConfig = field(
        default_factory=ProgressiveTrainingConfig
    )
    model: ModelConfig = field(default_factory=ModelConfig)
    density: DensityConfig = field(default_factory=DensityConfig)
    offline: bool = False
    debug: bool = False
    mixed_precision: bool = False
