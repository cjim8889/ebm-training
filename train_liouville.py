import argparse
import torch
import numpy as np
from einops import rearrange

from torch import Tensor
from typing import Optional, Callable, Union, Tuple, Dict, Any, Sequence, List

from utils.torch_utils import seed_everything
from utils.mog_utils import GMM, MultivariateGaussian
from models.mlp_models import TimeVelocityField
from utils.sampling_utils import time_schedule, generate_samples
from utils.loss_utils import loss_fn

def train_velocity_field(
    initial_density,
    target_density,
    v_theta: Callable[[Tensor, float], Tensor],
    optimiser: torch.optim.Optimizer,
    N: int = 512,
    B: int = 64,
    num_epochs: int = 200,
    num_steps: int = 100,
    learning_rate: float = 1e-03,
    momentum: float = 0.9,
    nestrov: bool = True,
    T: int = 32,
    gradient_norm: float = 1.0,
    mcmc_type: str = "langevin",
    num_mcmc_steps: int = 5,
    num_mcmc_integration_steps: int = 3,
    eta: float = 0.01,
    schedule: str = "linear",
    schedule_alpha: float = 5.0,  # Added
    schedule_gamma: float = 0.5,
    run_name: str = "velocity_field_training",
    **kwargs: Any,
) -> Any:
    """
    Train the velocity field v_theta to match initial and target densities.

    Args:
        initial_density (Target): The initial density distribution.
        target_density (Target): The target density distribution.
        v_theta (Callable[[Array, float], Array]): Velocity field function to train.
        key (jax.random.PRNGKey): Random key for randomness.
        N (int, optional): Number of samples per batch. Defaults to 512.
        num_epochs (int, optional): Number of training epochs. Defaults to 200.
        num_steps (int, optional): Number of steps per epoch. Defaults to 100.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-03.
        T (int, optional): Number of time steps. Defaults to 32.
        gradient_norm (float, optional): Gradient clipping norm. Defaults to 1.0.
        mcmc_type (str, optional): Type of MCMC sampler ('langevin' or 'hmc'). Defaults to "langevin".
        num_mcmc_steps (int, optional): Number of MCMC steps. Defaults to 5.
        num_mcmc_integration_steps (int, optional): Number of integration steps for MCMC. Defaults to 3.
        eta (float, optional): Step size for MCMC samplers. Defaults to 0.01.
        schedule (str, optional): Time schedule type ('linear', 'inverse_tangent', 'inverse_power'). Defaults to "linear".
        **kwargs (Any): Additional arguments.

    Returns:
        Any: Trained velocity field v_theta.
    """

    # Set up various functions
    def sample_initial(num_samples):
        return initial_density.sample((num_samples,))
    
    def time_dependent_log_density(x, t):
        return (1 - t) * initial_density.log_prob(x) + t * target_density.log_prob(x)
    
    def time_derivative_log_density(x, t):
        # t = t.requires_grad_(True)
        # log_p = time_dependent_log_density(x, t)
        # return torch.autograd.grad(log_p.sum(), t)[0]
        return - initial_density.log_prob(x) + target_density.log_prob(x)

    def score_function(x, t):
        x = x.requires_grad_(True)
        log_p = time_dependent_log_density(x, t)
        return torch.autograd.grad(log_p.sum(), x)[0]
    
    
    ts = time_schedule(T, schedule_alpha, schedule_gamma)[schedule]()

    for epoch in range(num_epochs):
        if mcmc_type == "langevin":
            samples = generate_samples(v_theta, N, ts, sample_initial)
        elif mcmc_type == "hmc":
            samples = generate_samples(v_theta, N, ts, sample_initial)
        else:
            samples = generate_samples(v_theta, N, ts, sample_initial)

        epoch_loss = 0.0
        for s in range(num_steps):

            samps = samples[
                torch.randperm(samples.size(0))[:B]
            ]   # (B, T, D)

            optimiser.zero_grad()
            loss = loss_fn(
                v_theta, samps, ts, 
                time_derivative_log_density=time_derivative_log_density,
                score_fn=score_function
            )
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
            print(loss.item())
            exit()

def main(args):
    gmm = GMM(
        dim=args.input_dim, 
        n_mixes=args.gmm_n_mixes, 
        loc_scaling=40, 
        log_var_scaling=1,
        seed=args.seed,
        device=args.device,
    )

    initial = MultivariateGaussian(
        dim=args.input_dim,
        sigma=args.initial_sigma,
        device=args.device,
    )

    v_theta = TimeVelocityField(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    ).to(args.device)

    if args.optimiser == "adam":
        optimizer = torch.optim.Adam(
            v_theta.parameters(), lr=args.learning_rate
        )
    elif args.optimiser == "sgd":
        optimizer = torch.optim.SGD(
            v_theta.parameters(), lr=args.learning_rate, momentum=args.momentum
        )
    elif args.optimiser == "adamw":
        optimizer = torch.optim.AdamW(
            v_theta.parameters(), lr=args.learning_rate
        )
    elif args.optimiser == "adamax":
        optimizer = torch.optim.Adamax(
            v_theta.parameters(), lr=args.learning_rate
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimiser}")
    
    train_velocity_field(
        initial_density=initial,
        target_density=gmm,
        v_theta=v_theta,
        optimiser=optimizer,
        N=args.N,
        B=args.B,
        num_epochs=args.num_epochs,
        num_steps=args.num_steps,
        T=args.T,
        gradient_norm=args.gradient_norm,
        mcmc_type=args.mcmc_type,  # Added
        num_mcmc_steps=args.num_mcmc_steps,  # Added
        num_mcmc_integration_steps=args.num_mcmc_integration_steps,  # Added
        eta=args.eta,
        schedule=args.schedule,  # Added
        schedule_alpha=args.schedule_alpha,  # Added
        schedule_gamma=args.schedule_gamma,
        nestrov=True,
        run_name=args.run_name,
    )


# Define Argument Parser
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train a Velocity Field with Configurable Hyperparameters"
    )

    # General Hyperparameters
    parser.add_argument(
        "--seed", type=int, default=80801, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="liouville",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="velocity_field_training",
        help="Name of the WandB run.",
    )

    # Training Hyperparameters
    parser.add_argument(
        "--input_dim", type=int, default=2, help="Dimensionality of the problem."
    )
    parser.add_argument("--T", type=int, default=64, help="Number of time steps.")
    parser.add_argument(
        "--N",
        type=int,
        default=1024,
        help="Number of samples for training at each time step.",
    )
    parser.add_argument(
        "--B",
        type=int,
        default=64,
        help="Number of samples to use for each training step",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=2000, help="Number of training epochs."
    )
    parser.add_argument(
        "--num_steps", type=int, default=100, help="Number of training steps per epoch."
    )
    parser.add_argument(
        "--optimiser",
        type=str,
        choices=["adam", "sgd", "adamw", "adamax"],
        default="adam",
        help="Optimizer to use for training.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="Momentum for the optimizer."
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="Hidden dimension size of the MLP."
    )
    parser.add_argument(
        "--depth", type=int, default=3, help="Depth (number of layers) of the MLP."
    )
    parser.add_argument(
        "--gradient_norm", type=float, default=1.0, help="Gradient clipping norm."
    )

    # GMM Hyperparameters
    parser.add_argument(
        "--gmm_n_mixes", type=int, default=40, help="Number of mixtures in the GMM."
    )

    # Initial Distribution Hyperparameters
    parser.add_argument(
        "--initial_sigma",
        type=float,
        default=20.0,
        help="Sigma (standard deviation) for the initial Gaussian.",
    )

    # MCMC Hyperparameters
    parser.add_argument(
        "--mcmc_type",
        type=str,
        choices=["langevin", "hmc"],
        default="hmc",
        help="Type of MCMC sampler to use ('langevin' or 'hmc').",
    )
    parser.add_argument(
        "--num_mcmc_steps",
        type=int,
        default=5,
        help="Number of MCMC steps to perform.",
    )
    parser.add_argument(
        "--num_mcmc_integration_steps",
        type=int,
        default=5,
        help="Number of integration steps for MCMC samplers (applicable for HMC).",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="Step size parameter for MCMC samplers.",
    )

    # Time Schedule Hyperparameters
    parser.add_argument(
        "--schedule",
        type=str,
        choices=["linear", "inverse_tangent", "inverse_power"],
        default="linear",
        help="Time schedule type ('linear', 'inverse_tangent', 'inverse_power'). Defaults to 'linear'.",
    )
    parser.add_argument(
        "--schedule_alpha",
        type=float,
        default=5.0,
        help="Alpha parameter for the inverse_tangent schedule. Applicable only if schedule is 'inverse_tangent'.",
    )
    parser.add_argument(
        "--schedule_gamma",
        type=float,
        default=0.5,
        help="Gamma parameter for the inverse_power schedule. Applicable only if schedule is 'inverse_power'.",
    )

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    seed_everything(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(args)
