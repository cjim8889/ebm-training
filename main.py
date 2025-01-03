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
from models import TimeVelocityField, TimeVelocityFieldWithPairwiseFeature
from training import train_velocity_field, train_velocity_field_with_decoupled_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-epochs", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=5120)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--num-timesteps", type=int, default=128)
    parser.add_argument(
        "--mcmc-type", type=str, default="hmc", choices=["hmc", "smc", "esmc"]
    )
    parser.add_argument("--mcmc-steps", type=int, default=5)
    parser.add_argument("--mcmc-integration-steps", type=int, default=3)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--initial-sigma", type=float, default=20.0)
    parser.add_argument("--network", type=str, default="mlp", choices=["mlp", "pdn"])
    parser.add_argument("--dt-pt-clip", type=float, default=None)
    parser.add_argument("--soft-clip", action="store_true")
    parser.add_argument("--pt-clip", type=float, default=None)
    parser.add_argument(
        "--target",
        type=str,
        default="gmm",
        choices=[
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
        ],
    )
    parser.add_argument(
        "--schedule",
        type=str,
        choices=["linear", "inverse_power", "power"],
        default="linear",
    )
    parser.add_argument(
        "--integrator", type=str, choices=["euler", "rk4"], default="euler"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "adamw", "sgd", "rmsprop"],
        default="adamw",
    )
    parser.add_argument("--continuous-schedule", action="store_true")
    parser.add_argument("--with-rejection-sampling", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--target-end-time", type=float, default=1.0)
    parser.add_argument("--initial-end-time", type=float, default=0.2)
    parser.add_argument("--end-time-steps", type=int, default=10)
    parser.add_argument("--update-end-time-every", type=int, default=100)
    parser.add_argument("--enable-end-time-progression", action="store_true")
    parser.add_argument("--gradient-norm", type=float, default=1.0)
    parser.add_argument("--score-norm", type=float, default=None)
    parser.add_argument(
        "--method", type=str, default="default", choices=["default", "decoupled"]
    )

    args = parser.parse_args()

    if args.debug:
        jax.config.update("jax_debug_nans", True)
        jax.config.update("jax_debug_infs", True)

    # Set random seed
    key = jax.random.PRNGKey(args.seed)

    def shift_fn(x):
        return x

    # Set up distributions
    if args.target == "gmm":
        input_dim = 2
        key, subkey = jax.random.split(key)
        # Initialize distributions
        initial_density = MultivariateGaussian(
            mean=jnp.zeros(input_dim), dim=input_dim, sigma=args.initial_sigma
        )
        target_density = GMM(subkey, dim=input_dim)
    elif args.target == "mw32":
        input_dim = 32
        key, subkey = jax.random.split(key)
        initial_density = MultivariateGaussian(
            mean=jnp.zeros(input_dim), dim=input_dim, sigma=args.initial_sigma
        )
        target_density = ManyWellEnergy(dim=input_dim)
    elif args.target == "dw4":
        input_dim = 8
        key, subkey = jax.random.split(key)
        initial_density = TranslationInvariantGaussian(
            N=4, D=2, sigma=args.initial_sigma, wrap=False
        )
        target_density = MultiDoubleWellEnergy(
            dim=input_dim,
            n_particles=4,
            data_path_test="data/test_split_DW4.npy",
            data_path_val="data/val_split_DW4.npy",
            key=subkey,
        )

        def shift_fn(x):
            return x - jnp.mean(x, axis=0, keepdims=True)
    elif args.target == "dw4o":
        input_dim = 8
        key, subkey = jax.random.split(key)
        initial_density = MultivariateGaussian(
            mean=jnp.zeros(input_dim), dim=input_dim, sigma=args.initial_sigma
        )
        target_density = MultiDoubleWellEnergy(
            dim=input_dim,
            n_particles=4,
            data_path_test="data/test_split_DW4.npy",
            data_path_val="data/val_split_DW4.npy",
            key=subkey,
        )
    elif args.target == "lj13":
        input_dim = 39
        key, subkey = jax.random.split(key)

        initial_density = MultivariateGaussian(
            dim=input_dim, mean=jnp.zeros(input_dim), sigma=args.initial_sigma
        )
        target_density = LennardJonesEnergy(
            dim=input_dim,
            n_particles=13,
            data_path_test="data/test_split_LJ13-1000.npy",
            data_path_val="data/val_split_LJ13-1000.npy",
            data_path_train="data/train_split_LJ13-1000.npy",
            log_prob_clip=args.pt_clip,
            key=subkey,
        )
    elif args.target == "tlj13":
        input_dim = 39
        key, subkey = jax.random.split(key)

        initial_density = MultivariateGaussian(
            dim=input_dim, mean=jnp.zeros(input_dim), sigma=args.initial_sigma
        )
        target_density = TimeDependentLennardJonesEnergy(
            dim=input_dim,
            n_particles=13,
            alpha=2.0,
            min_dr=1e-3,
        )
    elif args.target == "lj13b":
        input_dim = 39
        key, subkey = jax.random.split(key)

        initial_density = MultivariateGaussian(
            dim=input_dim, mean=jnp.zeros(input_dim), sigma=args.initial_sigma
        )

        target_density = TimeDependentLennardJonesEnergyButler(
            dim=input_dim,
            n_particles=13,
            sigma=1.0,
            alpha=0.1,
            epsilon_val=1.0,
            min_dr=1e-3,
            m=1,
            n=1,
            c=0.5,
            log_prob_clip=args.pt_clip,
            soft_clip=args.soft_clip,
            score_norm=args.score_norm,
            include_harmonic=True,
        )

        def shift_fn(x):
            return x - jnp.mean(x, axis=0, keepdims=True)
    elif args.target == "lj13c":
        input_dim = 39
        key, subkey = jax.random.split(key)

        initial_density = MultivariateGaussian(
            dim=input_dim, mean=jnp.zeros(input_dim), sigma=args.initial_sigma
        )

        target_density = TimeDependentLennardJonesEnergyButler(
            dim=input_dim,
            n_particles=13,
            sigma=1.0,
            alpha=0.1,
            epsilon_val=1.0,
            min_dr=1e-3,
            m=1,
            n=1,
            c=0.5,
            log_prob_clip=args.pt_clip,
            soft_clip=args.soft_clip,
            score_norm=args.score_norm,
            include_harmonic=True,
            cubic_spline=True,
        )
    elif args.target == "lj13bt":
        input_dim = 39
        key, subkey = jax.random.split(key)

        initial_density = MultivariateGaussian(
            dim=input_dim, mean=jnp.zeros(input_dim), sigma=args.initial_sigma
        )

        target_density = TimeDependentLennardJonesEnergyButlerWithTemperatureTempered(
            dim=input_dim,
            n_particles=13,
            sigma=1.0,
            alpha=0.1,
            epsilon_val=1.0,
            min_dr=1e-3,
            m=1,
            n=1,
            c=0.5,
            log_prob_clip=args.pt_clip,
            soft_clip=args.soft_clip,
            score_norm=args.score_norm,
            include_harmonic=True,
        )
    elif args.target == "sclj13":
        input_dim = 39
        key, subkey = jax.random.split(key)

        initial_density = MultivariateGaussian(
            dim=input_dim, mean=jnp.zeros(input_dim), sigma=args.initial_sigma
        )
        target_density = SoftCoreLennardJonesEnergy(
            key=subkey,
            dim=input_dim,
            n_particles=13,
            sigma=1.0,
            epsilon_val=1.0,
            alpha=0.1,
            shift_fn=shift_fn,
            min_dr=1e-3,
        )

    # Initialize velocity field
    key, model_key = jax.random.split(key)
    if args.network == "mlp":
        v_theta = TimeVelocityField(
            model_key,
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
        )
    elif args.network == "pdn":
        v_theta = TimeVelocityFieldWithPairwiseFeature(
            model_key,
            n_particles=4 if args.target in ["dw4", "dw4o"] else 13,
            n_spatial_dim=2 if args.target in ["dw4", "dw4o"] else 3,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
        )

    if not args.offline:
        # Handle logging hyperparameters
        wandb.init(
            project="liouville",
            config={
                "input_dim": initial_density.dim,
                "T": args.num_timesteps,
                "N": args.num_samples,
                "num_epochs": args.num_epochs,
                "num_steps": args.num_steps,
                "learning_rate": args.learning_rate,
                "gradient_norm": args.gradient_norm,
                "hidden_dim": v_theta.mlp.width_size,
                "depth": v_theta.mlp.depth,
                "mcmc_type": args.mcmc_type,
                "num_mcmc_steps": args.mcmc_steps,
                "num_mcmc_integration_steps": args.mcmc_integration_steps,
                "eta": args.eta,
                "schedule": args.schedule,
                "optimizer": args.optimizer,
                "integrator": args.integrator,
                "initial_sigma": args.initial_sigma,
                "with_rejection_sampling": args.with_rejection_sampling,
                "continuous_schedule": args.continuous_schedule,
                "target": args.target,
                "network": args.network,
                "dt_log_density_clip": args.dt_pt_clip,
                "log_density_clip": args.pt_clip,
                "soft_clip": args.soft_clip,
                "target_end_time": args.target_end_time,
                "initial_end_time": args.initial_end_time,
                "end_time_steps": args.end_time_steps,
                "update_end_time_every": args.update_end_time_every,
                "enable_end_time_progression": args.enable_end_time_progression,
                "score_norm": args.score_norm,
                "method": args.method,
            },
            reinit=True,
            tags=[
                args.target,
                args.network,
                args.method,
                "no_shortcut",
            ],
        )

    # Train model
    if args.method == "default":
        v_theta = train_velocity_field(
            key=key,
            initial_density=initial_density,
            target_density=target_density,
            v_theta=v_theta,
            shift_fn=shift_fn,
            N=args.num_samples,
            B=args.batch_size,
            T=args.num_timesteps,
            num_epochs=args.num_epochs,
            num_steps=args.num_steps,
            learning_rate=args.learning_rate,
            num_mcmc_steps=args.mcmc_steps,
            num_mcmc_integration_steps=args.mcmc_integration_steps,
            mcmc_type=args.mcmc_type,
            eta=args.eta,
            schedule=args.schedule,
            integrator=args.integrator,
            optimizer=args.optimizer,
            with_rejection_sampling=args.with_rejection_sampling,
            continuous_schedule=args.continuous_schedule,
            offline=args.offline,
            target=args.target,
            eval_every=args.eval_every,
            network=args.network,
            dt_log_density_clip=args.dt_pt_clip,
            target_end_time=args.target_end_time,
            initial_end_time=args.initial_end_time,
            end_time_steps=args.end_time_steps,
            update_end_time_every=args.update_end_time_every,
            enable_end_time_progression=args.enable_end_time_progression,
            gradient_norm=args.gradient_norm,
        )
    elif args.method == "decoupled":
        v_theta = train_velocity_field_with_decoupled_loss(
            key=key,
            initial_density=initial_density,
            target_density=target_density,
            v_theta=v_theta,
            shift_fn=shift_fn,
            N=args.num_samples,
            B=args.batch_size,
            T=args.num_timesteps,
            num_epochs=args.num_epochs,
            num_steps=args.num_steps,
            learning_rate=args.learning_rate,
            num_mcmc_steps=args.mcmc_steps,
            num_mcmc_integration_steps=args.mcmc_integration_steps,
            mcmc_type=args.mcmc_type,
            eta=args.eta,
            schedule=args.schedule,
            integrator=args.integrator,
            optimizer=args.optimizer,
            with_rejection_sampling=args.with_rejection_sampling,
            continuous_schedule=args.continuous_schedule,
            offline=args.offline,
            target=args.target,
            eval_every=args.eval_every,
            network=args.network,
            dt_log_density_clip=args.dt_pt_clip,
            target_end_time=args.target_end_time,
            initial_end_time=args.initial_end_time,
            end_time_steps=args.end_time_steps,
            update_end_time_every=args.update_end_time_every,
            enable_end_time_progression=args.enable_end_time_progression,
            gradient_norm=args.gradient_norm,
        )


if __name__ == "__main__":
    main()
