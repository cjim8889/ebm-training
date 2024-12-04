import torch
import numpy as np
import wandb
import hydra
from omegaconf import DictConfig
from einops import rearrange
import matplotlib.pyplot as plt

from torch import Tensor
from typing import Optional, Callable, Union, Tuple, Dict, Any, Sequence, List

from flow_sampler.utils.torch_utils import seed_everything
from flow_sampler.target_distributions.multi_double_well import MultiDoubleWellEnergy
from flow_sampler.utils.mog_utils import MultivariateGaussian
from flow_sampler.models.mlp_models import TimeVelocityField
from flow_sampler.models.egnn import EGNN_dynamics
from flow_sampler.utils.sampling_utils import time_schedule, generate_samples, generate_samples_with_hmc, generate_samples_with_langevin_dynamics
from flow_sampler.utils.loss_utils import loss_fn
from flow_sampler.utils.data_utils import remove_mean


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
    gradient_norm_clip: float = 1.0,
    mcmc_type: str = "langevin",
    num_mcmc_steps: int = 5,
    num_mcmc_integration_steps: int = 3,
    eta: float = 0.01,
    schedule: str = "linear",
    schedule_alpha: float = 5.0,  # Added
    schedule_gamma: float = 0.5,
    run_name: str = "velocity_field_training",
    input_dim: int = 2,
    hidden_dim: int = 128,
    depth: int = 3,
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
        gradient_norm_clip (float, optional): Gradient clipping norm. Defaults to 1.0.
        mcmc_type (str, optional): Type of MCMC sampler ('langevin' or 'hmc'). Defaults to "langevin".
        num_mcmc_steps (int, optional): Number of MCMC steps. Defaults to 5.
        num_mcmc_integration_steps (int, optional): Number of integration steps for MCMC. Defaults to 3.
        eta (float, optional): Step size for MCMC samplers. Defaults to 0.01.
        schedule (str, optional): Time schedule type ('linear', 'inverse_tangent', 'inverse_power'). Defaults to "linear".
        **kwargs (Any): Additional arguments.

    Returns:
        Any: Trained velocity field v_theta.
    """

    # Handle logging hyperparameters
    wandb.init(
        project="liouville",
        config={
            "input_dim": input_dim,
            "T": T,
            "N": N,
            "num_epochs": num_epochs,
            "num_steps": num_steps,
            "optimiser": optimiser,
            "learning_rate": learning_rate,
            "momentum": momentum,
            "nestrov": nestrov,
            "gradient_norm_clip": gradient_norm_clip,
            "hidden_dim": hidden_dim,
            "depth": depth,
            "mcmc_type": mcmc_type,
            "num_mcmc_steps": num_mcmc_steps,
            "num_mcmc_integration_steps": num_mcmc_integration_steps,
            "eta": eta,
            "schedule": schedule,
            **kwargs,
        },
        name=run_name,
        reinit=True,
        mode="online",    # online disabled
    )

    # Set up various functions
    def sample_initial(num_samples):
        return initial_density.sample((num_samples,))
    
    def time_dependent_log_density(x, t):
        l, n, d = x.shape
        x = rearrange(x, "l n d -> (l n) d")
        t = rearrange(t, "l n -> (l n)")
        log_prob = (1 - t) * initial_density.log_prob(x) + t * target_density.log_prob(x)
        return rearrange(log_prob, "(l n) -> l n", l=l)

    def time_derivative_log_density(x, t):
        # t = t.requires_grad_(True)
        # log_p = time_dependent_log_density(x, t)
        # return torch.autograd.grad(log_p.sum(), t)[0]
        b, n, d = x.shape
        x = rearrange(x, "b n d -> (b n) d")
        dlog_prob = - initial_density.log_prob(x) + target_density.log_prob(x)
        return rearrange(dlog_prob, "(b n) -> b n", b=b)

    def score_function(x, t):
        x = x.detach().requires_grad_(True)
        log_p = time_dependent_log_density(x, t)
        return torch.autograd.grad(log_p.sum(), x)[0]

    ts = time_schedule(T, schedule_alpha, schedule_gamma)[schedule]()

    for epoch in range(num_epochs):
        if mcmc_type == "langevin":
            samples = generate_samples_with_langevin_dynamics(
                v_theta,
                N,
                ts,
                sample_initial,
                score_function,
                num_mcmc_steps,
                eta,
            )
        elif mcmc_type == "hmc":
            samples = generate_samples_with_hmc(
                v_theta,
                sample_initial,
                time_dependent_log_density,
                N,
                ts,
                num_mcmc_steps,
                num_mcmc_integration_steps,
                eta,
                False,
            )
        else:
            samples = generate_samples(v_theta, N, ts, sample_initial)

        epoch_loss = 0.0
        for s in range(num_steps):

            with torch.no_grad():
                samps = samples[
                    torch.randperm(samples.size(0))[:B]
                ].detach()   # (B, T, D)

            # add random samples among the time steps
            optimiser.zero_grad(set_to_none=True)
            idxs = torch.randperm(samps.size(1))[:32]
            samps = samps[:, idxs, :]
            ts_ = ts[idxs]

            loss = loss_fn(
                v_theta, samps, ts_, 
                time_derivative_log_density=time_derivative_log_density,
                score_fn=score_function
            )
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()

            if s % 20 == 0:
                wandb.log({"loss": loss.item()})

        avg_loss = epoch_loss / num_steps
        print(f"Epoch {epoch}, Average Loss: {avg_loss}")
        wandb.log({"epoch": epoch, "average_loss": avg_loss})

        if epoch % 20 == 0:
            linear_ts = torch.linspace(0, 1, T)
            val_samples = generate_samples(
                v_theta, 1024, linear_ts, sample_initial
            )[:, -1, :].detach().cpu()
            val_samples = remove_mean(val_samples, target_density.n_particles, target_density.n_spatial_dim)
            fig = target_density.get_dataset_fig(val_samples)
            wandb.log({f"generative_samples": wandb.Image(fig)})

def run(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target = MultiDoubleWellEnergy(
        dimensionality=cfg.target.input_dim,
        n_particles=cfg.target.n_particles,
        data_path=cfg.target.data_path,
        data_path_train=cfg.target.data_path_train,
        data_path_val=cfg.target.data_path_val,
        data_from_efm=cfg.target.data_from_efm,
        device=device,
    )

    initial = MultivariateGaussian(
        dim=cfg.target.input_dim,
        sigma=cfg.flow.initial_sigma,
        device=device,
    )

    # v_theta = TimeVelocityField(
    #     input_dim=cfg.target.input_dim,
    #     hidden_dim=cfg.model.hidden_dim,
    #     depth=cfg.model.depth,
    # ).to(device)
    v_theta = EGNN_dynamics(
        n_particles=cfg.model.n_particles,
        n_dimension=cfg.model.n_dimension,
        hidden_nf=cfg.model.hidden_nf,
        n_layers=cfg.model.n_layers,
        recurrent=cfg.model.recurrent,
        attention=cfg.model.attention,
        condition_time=cfg.model.condition_time,
        tanh=cfg.model.tanh,
        agg=cfg.model.agg,
    ).to(device)

    opt_params = cfg.optimiser
    if opt_params.name == "adam":
        optimizer = torch.optim.Adam(
            v_theta.parameters(), lr=opt_params.learning_rate
        )
    elif opt_params.name == "sgd":
        optimizer = torch.optim.SGD(
            v_theta.parameters(), lr=opt_params.learning_rate, momentum=opt_params.momentum
        )
    elif opt_params.name == "adamw":
        optimizer = torch.optim.AdamW(
            v_theta.parameters(), lr=opt_params.learning_rate
        )
    elif opt_params.name == "adamax":
        optimizer = torch.optim.Adamax(
            v_theta.parameters(), lr=opt_params.learning_rate
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_params.name}")

    train_velocity_field(
        initial_density=initial,
        target_density=target,
        v_theta=v_theta,
        optimiser=optimizer,
        N=cfg.training.N,
        B=cfg.training.B,
        num_epochs=cfg.training.num_epochs,
        num_steps=cfg.training.num_steps,
        T=cfg.flow.T,
        gradient_norm_clip=cfg.training.gradient_norm_clip,
        mcmc_type=cfg.flow.mcmc_type,  # Added
        num_mcmc_steps=cfg.flow.num_mcmc_steps,  # Added
        num_mcmc_integration_steps=cfg.flow.num_mcmc_integration_steps,  # Added
        eta=cfg.flow.eta,
        schedule=cfg.schedule.name,  # Added
        schedule_alpha=cfg.schedule.schedule_alpha,  # Added
        schedule_gamma=cfg.schedule.schedule_gamma,
        nestrov=True,
        run_name=cfg.wandb.run_name,
        input_dim=2,
        hidden_dim=128,
        depth=3,
    )

@hydra.main(config_path="../configs", config_name="dw4.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    run(cfg)


if __name__ == '__main__':
    pass
