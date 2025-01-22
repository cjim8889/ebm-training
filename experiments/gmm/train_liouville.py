import os
import argparse
import torch
import numpy as np
import wandb
import hydra
from omegaconf import DictConfig

from torch import Tensor
from typing import Optional, Callable, Union, Tuple, Dict, Any, Sequence, List

from flow_sampler.utils.torch_utils import seed_everything
from flow_sampler.utils.mog_utils import GMM, MultivariateGaussian, plot_MoG40
from flow_sampler.models.mlp_models import TimeVelocityField
from flow_sampler.utils.sampling_utils import time_schedule, generate_samples, generate_samples_with_hmc, generate_samples_with_langevin_dynamics
from flow_sampler.utils.loss_utils import loss_fn, get_dt_logZt
from flow_sampler.utils.gradient_varest import GradientVarianceEstimator
from flow_sampler.utils.evaluate import eval_data_w2, eval_energy_w2, eval_total_variation

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
    dt_logZt_estor: str = "mcmc",
    ckpt_log_path: str = None,
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
            "dt_logZt_estor": dt_logZt_estor,
            **kwargs,
        },
        name=run_name,
        reinit=True,
        mode="disabled",    # online disabled
    )

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
        x = x.detach().requires_grad_(True)
        log_p = time_dependent_log_density(x, t)
        return torch.autograd.grad(log_p.sum(), x)[0]
    
    ts = time_schedule(T, schedule_alpha, schedule_gamma)[schedule]()

    variance_estimator = GradientVarianceEstimator(v_theta)

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

        std_dt_log_Zt_way1, std_dt_log_Zt_way2 = get_dt_logZt(v_theta, samples, ts, time_derivative_log_density, score_function)
        wandb.log({
            "mean_std/std_dt_log_Zt_mcmc": np.mean(std_dt_log_Zt_way1), 
            "mean_std/std_dt_log_Zt_mcmc_velocity": np.mean(std_dt_log_Zt_way2)
        })
        wandb.log({
            "last10steps_std/std_dt_log_Zt_mcmc": np.mean(std_dt_log_Zt_way1[-10:]), 
            "last10steps_std/std_dt_log_Zt_mcmc_velocity": np.mean(std_dt_log_Zt_way2[-10:])
        })
        wandb.log({
            "first10steps_std/std_dt_log_Zt_mcmc": np.mean(std_dt_log_Zt_way1[:10]), 
            "first10steps_std/std_dt_log_Zt_mcmc_velocity": np.mean(std_dt_log_Zt_way2[:10])
        })

        epoch_loss = 0.0
        for s in range(num_steps):

            with torch.no_grad():
                samps = samples[
                    torch.randperm(samples.size(0))[:B]
                ].detach()   # (B, T, D)

            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(
                v_theta, samps, ts, 
                time_derivative_log_density=time_derivative_log_density,
                score_fn=score_function,
                dt_logZt_estor=dt_logZt_estor,
            )
            loss.backward()

            # Track gradient variance before optimizer step
            variance_estimator.update()

            optimiser.step()
            epoch_loss += loss.item()

            if s % 20 == 0:
                wandb.log({"loss": loss.item()})
        
        avg_loss = epoch_loss / num_steps
        print(f"Epoch {epoch}, Average Loss: {avg_loss}")
        wandb.log({"epoch": epoch, "average_loss": avg_loss})

        grad_summary = variance_estimator.get_summary()
        wandb.log({"var_grad/mean_var": grad_summary['mean_variance']})

        if epoch % 50 == 0:
            # visualise generated samples
            for sample_steps in [4, 8, 16, T]:
                linear_ts = torch.linspace(0, 1, sample_steps)
                val_samples = generate_samples(
                    v_theta, 10000, linear_ts, sample_initial
                )[:, -1, :].detach().cpu()

                fig = plot_MoG40(
                    log_prob_function=target_density.log_prob,
                    samples=val_samples, 
                )
                wandb.log({f"samples_T={sample_steps}": wandb.Image(fig)})

            # evaluate metrics
            test_samples = generate_samples(
                    v_theta, 1000, linear_ts, sample_initial
                )[:, -1, :].detach()
            # data_w2_dist = eval_data_w2(target_density, test_samples)
            energy_w2_dist = eval_energy_w2(target_density, test_samples)

            # save checkpoint
            torch.save(v_theta.state_dict(), f"velocity_field_{epoch}.pt")

def run(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.ckpt_log_path, exist_ok=True)

    gmm = GMM(
        dim=cfg.target.input_dim, 
        n_mixes=cfg.target.gmm_n_mixes, 
        loc_scaling=40, 
        log_var_scaling=1,
        seed=cfg.seed,
        device=device,
    )

    initial = MultivariateGaussian(
        dim=cfg.target.input_dim,
        sigma=cfg.flow.initial_sigma,
        device=device,
    )

    v_theta = TimeVelocityField(
        input_dim=cfg.target.input_dim,
        hidden_dim=cfg.model.hidden_dim,
        depth=cfg.model.depth,
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
        target_density=gmm,
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
        run_name=cfg.wandb.run_name + f'_{cfg.dt_logZt_estor}',
        input_dim=2,
        hidden_dim=128,
        depth=3,
        dt_logZt_estor=cfg.dt_logZt_estor,
        ckpt_log_path=cfg.ckpt_log_path
    )

@hydra.main(config_path="../configs", config_name="gmm.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    run(cfg)



if __name__ == '__main__':
    pass
