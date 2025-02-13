import argparse

import jax
import jax.numpy as jnp

import wandb
from distributions import (
    GMM,
    LennardJonesEnergy,
    ManyWellEnergy,
    MultiDoubleWellEnergy,
    MultivariateGaussian,
    SoftCoreLennardJonesEnergy,
    TimeDependentLennardJonesEnergy,
    TimeDependentLennardJonesEnergyButler,
    TimeDependentLennardJonesEnergyButlerWithTemperatureTempered,
    TranslationInvariantGaussian,
)
from models import (
    TimeVelocityField,
    TimeVelocityFieldWithPairwiseFeature,
    ParticleTransformer,
    EquivariantTimeVelocityField,
    EGNN,
    VelocityFieldTwo,
    VelocityFieldThree,
)
from training.core import train_velocity_field
from training.config import (
    SamplingConfig,
    TrainingConfig,
    MCMCConfig,
    IntegrationConfig,
    ProgressiveTrainingConfig,
    ModelConfig,
    TrainingExperimentConfig,
    DensityConfig,
)


def main():
    parser = argparse.ArgumentParser()

    # Model configuration
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument(
        "--network",
        type=str,
        default="mlp",
        choices=["mlp", "pdn", "transformer", "emlp", "egnn", "mlp2", "mlp3"],
    )

    # Sampling configuration
    parser.add_argument("--num-samples", type=int, default=5120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-timesteps", type=int, default=128)

    # Training configuration
    parser.add_argument("--num-epochs", type=int, default=10000)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gradient-norm", type=float, default=1.0)
    parser.add_argument("--eval-frequency", type=int, default=20)
    parser.add_argument("--n-samples-eval", type=int, default=1024)
    parser.add_argument(
        "--use-cv",
        action="store_true",
        help="Whether to use control variate",
    )
    parser.add_argument(
        "--use-shortcut", action="store_true", help="Whether to use shortcut"
    )
    parser.add_argument(
        "--use-hutchinson",
        action="store_true",
        help="Whether to use Hutchinson's trick",
    )
    parser.add_argument(
        "--n-probes",
        type=int,
        default=2,
        help="Number of probes to use in Hutchinson's trick",
    )
    parser.add_argument(
        "--shortcut-size",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128],
    )
    parser.add_argument(
        "--time-batch-size",
        type=int,
        default=32,
        help="Number of time points to use in each batch",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=[
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
        ],
        default="adamw",
    )
    # Optimizer parameters
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay coefficient (L2 regularization)",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="First moment decay rate (for Adam-like optimizers)",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="Second moment decay rate (for Adam-like optimizers)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Small constant for numerical stability",
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="Momentum coefficient for SGD"
    )
    parser.add_argument(
        "--nesterov", action="store_true", help="Whether to use Nesterov momentum"
    )
    parser.add_argument(
        "--noise-scale", type=float, default=0.01, help="Noise scale for noisy SGD"
    )

    # MCMC configuration
    parser.add_argument(
        "--mcmc-method", type=str, default="hmc", choices=["hmc", "smc", "esmc", "vsmc"]
    )
    parser.add_argument("--mcmc-steps", type=int, default=5)
    parser.add_argument("--mcmc-integration-steps", type=int, default=3)
    parser.add_argument("--mcmc-step-size", type=float, default=0.2)
    parser.add_argument("--with-rejection", action="store_true")

    # Integration configuration
    parser.add_argument(
        "--integration-method", type=str, choices=["euler", "rk4"], default="euler"
    )
    parser.add_argument(
        "--schedule",
        type=str,
        choices=["linear", "inverse_power", "power"],
        default="linear",
    )
    parser.add_argument("--continuous-time", action="store_true")
    parser.add_argument("--dt-clip", type=float, default=None)

    # Progressive training configuration
    parser.add_argument("--enable-progression", action="store_true")
    parser.add_argument("--initial-timesteps", type=int, default=16)
    parser.add_argument("--timestep-increment", type=int, default=2)
    parser.add_argument("--progression-frequency", type=int, default=100)

    # Density configuration
    parser.add_argument(
        "--target",
        type=str,
        default="gmm",
        choices=[
            "gmm",
            "gmm10",
            "mw32",
            "dw4",
            "lj13",
            "sclj13",
            "dw4o",
            "tlj13",
            "lj13b",
            "lj13bt",
            "lj13c",
            "sclj13t",
            "lj13ct",
        ],
    )
    parser.add_argument("--initial-sigma", type=float, default=20.0)
    parser.add_argument("--score-norm", type=float, default=None)
    parser.add_argument(
        "--annealing-path",
        type=str,
        default="linear",
        choices=["linear", "geometric", "inverse_power"],
    )
    parser.add_argument("--shift", action="store_true")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
    )
    parser.add_argument("--epsilon-val", type=float, default=1.0)
    parser.add_argument("--min-dr", type=float, default=1e-3)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--c", type=float, default=0.5)
    parser.add_argument("--log-prob-clip", type=float, default=None)
    parser.add_argument("--log-prob-clip-min", type=float, default=None)
    parser.add_argument("--log-prob-clip-max", type=float, default=None)
    parser.add_argument("--soft-clip", action="store_true")
    parser.add_argument("--include-harmonic", action="store_true")
    parser.add_argument("--cubic-spline", action="store_true")
    parser.add_argument("--data-path-test", type=str, default=None)
    parser.add_argument("--data-path-val", type=str, default=None)
    parser.add_argument("--data-path-train", type=str, default=None)

    # Other configuration
    parser.add_argument(
        "--use-decoupled-loss",
        action="store_true",
        help="Whether to use decoupled loss function",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    # if args.debug:
    # jax.config.update("jax_debug_nans", True)
    # jax.config.update("jax_debug_infs", True)

    # Set random seed
    key = jax.random.PRNGKey(args.seed)

    # Create configuration objects
    sampling_config = SamplingConfig(
        num_particles=args.num_samples,
        batch_size=args.batch_size,
        num_timesteps=args.num_timesteps,
    )

    training_config = TrainingConfig(
        num_epochs=args.num_epochs,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        gradient_clip_norm=args.gradient_norm,
        eval_frequency=args.eval_frequency,
        optimizer=args.optimizer,
        use_decoupled_loss=args.use_decoupled_loss,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
        momentum=args.momentum,
        nesterov=args.nesterov,
        noise_scale=args.noise_scale,
        time_batch_size=args.time_batch_size,
        use_shortcut=args.use_shortcut,
        shortcut_size=args.shortcut_size,
        use_hutchinson=args.use_hutchinson,
        n_probes=args.n_probes,
    )

    mcmc_config = MCMCConfig(
        method=args.mcmc_method,
        num_steps=args.mcmc_steps,
        num_integration_steps=args.mcmc_integration_steps,
        step_size=args.mcmc_step_size,
        with_rejection=args.with_rejection,
        use_control_variate=args.use_cv,
    )

    integration_config = IntegrationConfig(
        method=args.integration_method,
        schedule=args.schedule,
        continuous_time=args.continuous_time,
        dt_clip=args.dt_clip,
    )

    progressive_config = ProgressiveTrainingConfig(
        enable=args.enable_progression,
        initial_timesteps=args.initial_timesteps,
        timestep_increment=args.timestep_increment,
        update_frequency=args.progression_frequency,
    )

    model_config = ModelConfig(
        hidden_dim=args.hidden_dim, num_layers=args.depth, architecture=args.network
    )

    # Set up shift function
    def shift_fn(x):
        return x - jnp.mean(x, axis=0, keepdims=True) if args.shift else x

    # Set up input dimensions and other target-specific parameters
    if args.target == "gmm":
        input_dim = 2
        n_particles = None
        n_spatial_dim = None
    elif args.target == "gmm10":
        input_dim = 10
        n_particles = None
        n_spatial_dim = None
    elif args.target == "mw32":
        input_dim = 32
        n_particles = None
        n_spatial_dim = None
    elif args.target in ["dw4", "dw4o"]:
        input_dim = 8
        n_particles = 4
        n_spatial_dim = 2
    else:  # LJ variants
        input_dim = 39
        n_particles = 13
        n_spatial_dim = 3

    density_config = DensityConfig(
        target_type=args.target,
        initial_sigma=args.initial_sigma,
        score_norm=args.score_norm,
        annealing_path=args.annealing_path,
        shift_fn=shift_fn,
        input_dim=input_dim,
        n_particles=n_particles,
        n_spatial_dim=n_spatial_dim,
        alpha=args.alpha,
        epsilon_val=args.epsilon_val,
        min_dr=args.min_dr,
        m=args.m,
        n=args.n,
        c=args.c,
        log_prob_clip=args.log_prob_clip,
        log_prob_clip_min=args.log_prob_clip_min,
        log_prob_clip_max=args.log_prob_clip_max,
        soft_clip=args.soft_clip,
        include_harmonic=args.include_harmonic,
        cubic_spline=args.cubic_spline,
        data_path_test=args.data_path_test,
        data_path_val=args.data_path_val,
        data_path_train=args.data_path_train,
        n_samples_eval=args.n_samples_eval,
    )

    config = TrainingExperimentConfig(
        sampling=sampling_config,
        training=training_config,
        mcmc=mcmc_config,
        integration=integration_config,
        progressive=progressive_config,
        model=model_config,
        density=density_config,
        offline=args.offline,
        debug=args.debug,
    )

    # Initialize distributions based on density config
    key, subkey = jax.random.split(key)
    if config.density.target_type == "gmm":
        initial_density = MultivariateGaussian(
            mean=jnp.zeros(config.density.input_dim),
            dim=config.density.input_dim,
            sigma=config.density.initial_sigma,
        )
        target_density = GMM(subkey, dim=config.density.input_dim)
    elif config.density.target_type == "gmm10":
        target_density = GMM(subkey, dim=config.density.input_dim, fixed_mean=False)
        initial_density = MultivariateGaussian(
            mean=jnp.zeros(config.density.input_dim),
            dim=config.density.input_dim,
            sigma=config.density.initial_sigma,
        )
    elif config.density.target_type == "mw32":
        initial_density = MultivariateGaussian(
            mean=jnp.zeros(config.density.input_dim),
            dim=config.density.input_dim,
            sigma=config.density.initial_sigma,
        )
        target_density = ManyWellEnergy(dim=config.density.input_dim)
    elif config.density.target_type == "dw4":
        initial_density = TranslationInvariantGaussian(
            N=config.density.n_particles,
            D=config.density.n_spatial_dim,
            sigma=config.density.initial_sigma,
            wrap=False,
        )
        target_density = MultiDoubleWellEnergy(
            dim=config.density.input_dim,
            n_particles=config.density.n_particles,
            data_path_test=config.density.data_path_test,
            data_path_val=config.density.data_path_val,
            key=subkey,
            log_prob_clip=config.density.log_prob_clip,
            log_prob_clip_min=config.density.log_prob_clip_min,
            log_prob_clip_max=config.density.log_prob_clip_max,
        )
    elif config.density.target_type == "dw4o":
        initial_density = MultivariateGaussian(
            mean=jnp.zeros(config.density.input_dim),
            dim=config.density.input_dim,
            sigma=config.density.initial_sigma,
        )
        target_density = MultiDoubleWellEnergy(
            dim=config.density.input_dim,
            n_particles=config.density.n_particles,
            data_path_test=config.density.data_path_test,
            key=subkey,
            log_prob_clip=config.density.log_prob_clip,
            log_prob_clip_min=config.density.log_prob_clip_min,
            log_prob_clip_max=config.density.log_prob_clip_max,
        )
    elif config.density.target_type == "lj13":
        initial_density = MultivariateGaussian(
            dim=config.density.input_dim,
            mean=jnp.zeros(config.density.input_dim),
            sigma=config.density.initial_sigma,
        )
        target_density = LennardJonesEnergy(
            dim=config.density.input_dim,
            n_particles=config.density.n_particles,
            data_path_test=config.density.data_path_test,
            data_path_val=config.density.data_path_val,
            data_path_train=config.density.data_path_train,
            log_prob_clip=config.density.log_prob_clip,
            log_prob_clip_min=config.density.log_prob_clip_min,
            log_prob_clip_max=config.density.log_prob_clip_max,
            key=subkey,
        )
    elif config.density.target_type == "tlj13":
        initial_density = MultivariateGaussian(
            dim=config.density.input_dim,
            mean=jnp.zeros(config.density.input_dim),
            sigma=config.density.initial_sigma,
        )
        target_density = TimeDependentLennardJonesEnergy(
            dim=config.density.input_dim,
            n_particles=config.density.n_particles,
            alpha=config.density.alpha,
            min_dr=config.density.min_dr,
        )
    elif config.density.target_type == "lj13b":
        initial_density = TranslationInvariantGaussian(
            N=config.density.n_particles,
            D=config.density.n_spatial_dim,
            sigma=config.density.initial_sigma,
        )
        target_density = TimeDependentLennardJonesEnergyButler(
            dim=config.density.input_dim,
            n_particles=config.density.n_particles,
            sigma=1.0,
            alpha=config.density.alpha,
            epsilon_val=config.density.epsilon_val,
            min_dr=config.density.min_dr,
            m=config.density.m,
            n=config.density.n,
            c=config.density.c,
            log_prob_clip=config.density.log_prob_clip,
            log_prob_clip_min=config.density.log_prob_clip_min,
            log_prob_clip_max=config.density.log_prob_clip_max,
            soft_clip=config.density.soft_clip,
            score_norm=config.density.score_norm,
            include_harmonic=config.density.include_harmonic,
        )
    elif config.density.target_type == "lj13c":
        initial_density = MultivariateGaussian(
            dim=config.density.input_dim,
            mean=jnp.zeros(config.density.input_dim),
            sigma=config.density.initial_sigma,
        )
        target_density = TimeDependentLennardJonesEnergyButler(
            dim=config.density.input_dim,
            n_particles=config.density.n_particles,
            sigma=1.0,
            alpha=config.density.alpha,
            epsilon_val=config.density.epsilon_val,
            min_dr=config.density.min_dr,
            m=config.density.m,
            n=config.density.n,
            c=config.density.c,
            log_prob_clip=config.density.log_prob_clip,
            log_prob_clip_min=config.density.log_prob_clip_min,
            log_prob_clip_max=config.density.log_prob_clip_max,
            soft_clip=config.density.soft_clip,
            score_norm=config.density.score_norm,
            include_harmonic=config.density.include_harmonic,
            cubic_spline=config.density.cubic_spline,
        )
    elif config.density.target_type == "lj13ct":
        initial_density = TranslationInvariantGaussian(
            N=config.density.n_particles,
            D=config.density.n_spatial_dim,
            sigma=config.density.initial_sigma,
            wrap=False,
        )
        target_density = TimeDependentLennardJonesEnergyButler(
            dim=config.density.input_dim,
            n_particles=config.density.n_particles,
            sigma=1.0,
            alpha=config.density.alpha,
            epsilon_val=config.density.epsilon_val,
            min_dr=config.density.min_dr,
            m=config.density.m,
            n=config.density.n,
            c=config.density.c,
            log_prob_clip=config.density.log_prob_clip,
            log_prob_clip_min=config.density.log_prob_clip_min,
            log_prob_clip_max=config.density.log_prob_clip_max,
            soft_clip=config.density.soft_clip,
            score_norm=config.density.score_norm,
            include_harmonic=config.density.include_harmonic,
            cubic_spline=config.density.cubic_spline,
        )
    elif config.density.target_type == "lj13bt":
        initial_density = MultivariateGaussian(
            dim=config.density.input_dim,
            mean=jnp.zeros(config.density.input_dim),
            sigma=config.density.initial_sigma,
        )
        target_density = TimeDependentLennardJonesEnergyButlerWithTemperatureTempered(
            dim=config.density.input_dim,
            n_particles=config.density.n_particles,
            sigma=1.0,
            alpha=config.density.alpha,
            epsilon_val=config.density.epsilon_val,
            min_dr=config.density.min_dr,
            m=config.density.m,
            n=config.density.n,
            c=config.density.c,
            log_prob_clip=config.density.log_prob_clip,
            log_prob_clip_min=config.density.log_prob_clip_min,
            log_prob_clip_max=config.density.log_prob_clip_max,
            soft_clip=config.density.soft_clip,
            score_norm=config.density.score_norm,
            include_harmonic=config.density.include_harmonic,
        )
    elif config.density.target_type == "sclj13":
        initial_density = MultivariateGaussian(
            dim=config.density.input_dim,
            mean=jnp.zeros(config.density.input_dim),
            sigma=config.density.initial_sigma,
        )
        target_density = SoftCoreLennardJonesEnergy(
            dim=config.density.input_dim,
            n_particles=config.density.n_particles,
            sigma=1.0,
            epsilon_val=config.density.epsilon_val,
            alpha=config.density.alpha,
            shift_fn=config.density.shift_fn,
            min_dr=config.density.min_dr,
            c=config.density.c,
            include_harmonic=config.density.include_harmonic,
            log_prob_clip=config.density.log_prob_clip,
            log_prob_clip_min=config.density.log_prob_clip_min,
            log_prob_clip_max=config.density.log_prob_clip_max,
        )
    elif config.density.target_type == "sclj13t":
        initial_density = TranslationInvariantGaussian(
            N=config.density.n_particles,
            D=config.density.n_spatial_dim,
            sigma=config.density.initial_sigma,
        )
        target_density = SoftCoreLennardJonesEnergy(
            dim=config.density.input_dim,
            n_particles=config.density.n_particles,
            sigma=1.0,
            epsilon_val=config.density.epsilon_val,
            alpha=config.density.alpha,
            shift_fn=config.density.shift_fn,
            min_dr=config.density.min_dr,
            c=config.density.c,
            include_harmonic=config.density.include_harmonic,
            log_prob_clip=config.density.log_prob_clip,
            log_prob_clip_min=config.density.log_prob_clip_min,
            log_prob_clip_max=config.density.log_prob_clip_max,
        )

    # Initialize velocity field
    key, model_key = jax.random.split(key)
    if config.model.architecture == "mlp":
        v_theta = TimeVelocityField(
            model_key,
            input_dim=config.density.input_dim,
            hidden_dim=config.model.hidden_dim,
            depth=config.model.num_layers,
            shortcut=config.training.use_shortcut,
        )
    elif config.model.architecture == "mlp2":
        v_theta = VelocityFieldTwo(
            model_key,
            dim=config.density.input_dim,
            hidden_dim=config.model.hidden_dim,
            depth=config.model.num_layers,
            shortcut=config.training.use_shortcut,
        )
    elif config.model.architecture == "pdn":
        v_theta = TimeVelocityFieldWithPairwiseFeature(
            model_key,
            n_particles=config.density.n_particles,
            n_spatial_dim=config.density.n_spatial_dim,
            hidden_dim=config.model.hidden_dim,
            depth=config.model.num_layers,
            shortcut=config.training.use_shortcut,
        )
    elif config.model.architecture == "transformer":
        v_theta = ParticleTransformer(
            n_particles=config.density.n_particles,
            n_spatial_dim=config.density.n_spatial_dim,
            hidden_size=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            num_heads=4,
            dropout_rate=0.0,
            attn_dropout_rate=0.0,
            key=model_key,
            shortcut=config.training.use_shortcut,
        )
    elif config.model.architecture == "emlp":
        v_theta = EquivariantTimeVelocityField(
            key=model_key,
            n_particles=config.density.n_particles,
            n_spatial_dim=config.density.n_spatial_dim,
            hidden_dim=config.model.hidden_dim,
            depth=config.model.num_layers,
            min_dr=config.density.min_dr,
        )
    elif config.model.architecture == "egnn":
        v_theta = EGNN(
            key=model_key,
            n_node=config.density.n_particles,
            hidden_size=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            normalize=True,
            tanh=True,
            shortcut=config.training.use_shortcut,
        )
    elif config.model.architecture == "mlp3":
        v_theta = VelocityFieldThree(
            key=model_key,
            n_particles=config.density.n_particles,
            n_spatial_dim=config.density.n_spatial_dim,
            hidden_dim=config.model.hidden_dim,
            depth=config.model.num_layers,
            equivariant_dim=32,
            shortcut=config.training.use_shortcut,
        )

    if not config.offline:
        # Handle logging hyperparameters
        wandb.init(
            project="liouville_workshop",
            config=vars(config),
            reinit=True,
            tags=[
                config.density.target_type,
                config.model.architecture,
                "decoupled" if config.training.use_decoupled_loss else "standard",
                "shortcut" if config.training.use_shortcut else "noshortcut",
                "cv" if config.mcmc.use_control_variate else "no_cv",
            ],
        )

    if config.debug:
        config.training.num_epochs = 2
        import jax.profiler as profiler

        with profiler.trace("profile"):
            # Train model
            v_theta = train_velocity_field(
                key=key,
                initial_density=initial_density,
                target_density=target_density,
                v_theta=v_theta,
                config=config,
            )
    else:
        # Train model
        v_theta = train_velocity_field(
            key=key,
            initial_density=initial_density,
            target_density=target_density,
            v_theta=v_theta,
            config=config,
        )


if __name__ == "__main__":
    main()
