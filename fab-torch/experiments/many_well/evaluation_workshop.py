import hydra
from typing import Optional, Tuple
import itertools

import ot as pot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from omegaconf import DictConfig
import torch
from collections import defaultdict

from fab.target_distributions.many_well import ManyWellEnergy
from experiments.load_model_for_eval import load_model


PATH = os.getcwd()


def evaluate_many_well(cfg: DictConfig, path_to_model: str, target, num_samples=int(5e4)):
    test_set_exact = target.sample((num_samples, ))
    test_set_log_prob_over_p = torch.mean(target.log_prob(test_set_exact) - target.log_Z).cpu().item()
    test_set_modes_log_prob_over_p = torch.mean(target.log_prob(target._test_set_modes) - target.log_Z)
    print(f"test set log prob under p: {test_set_log_prob_over_p:.2f}")
    print(f"modes test set log prob under p: {test_set_modes_log_prob_over_p:.2f}")
    model = load_model(cfg, target, path_to_model)
    eval = model.get_eval_info(num_samples, 500)
    return eval

def evaluate_metrics(model, target, gen_samples):
    data_set = target.sample((gen_samples.shape[0], ))
    gen_samples = gen_samples.cpu()

    lop_p = target.log_prob(data_set) if target.normalised else target.log_prob(data_set) - target.log_Z  
    log_q = model.flow._nf_model.log_prob(data_set.to('cuda')).cpu()
    f_kl = torch.mean(lop_p - log_q).item()


    energies = target.log_prob(data_set)
    generated_energies = target.log_prob(gen_samples)
    e_w2_dist = np.sqrt(pot.emd2_1d(energies.numpy(), generated_energies.numpy()))



    energies = energies.numpy().reshape(-1)
    generated_energies= generated_energies.numpy().reshape(-1)
    H_data_set, x_data_set = np.histogram(energies, bins=200)
    H_generated_samples, _ = np.histogram(generated_energies, bins=(x_data_set))
    total_var = (
        0.5
        * np.abs(
            H_data_set / H_data_set.sum() - H_generated_samples / H_generated_samples.sum()
        ).sum()
    )


    # distance_matrix = pot.dist(data_set.numpy(), gen_samples.numpy(), metric='euclidean')
    # distance_matrix = distance_matrix**2
    # src, dist = np.ones(len(data_set)) / len(data_set), np.ones(len(gen_samples)) / len(gen_samples)
    # x_w2_dist = np.sqrt(pot.emd2(src, dist, distance_matrix))
    a, b = pot.unif(gen_samples.shape[0]), pot.unif(data_set.shape[0])
    M = torch.cdist(gen_samples, data_set) ** 2
    ret = pot.emd2(a, b, M.detach().cpu().numpy(), numItermax=1e7)
    x_w2_dist = np.sqrt(ret)


    return {
        'e_w2_dist': e_w2_dist,
        'x_w2_dist': x_w2_dist,
        'e_tv': total_var,
        'f_kl': f_kl
    }

def get_target_log_prob_marginal_pair(log_prob, i: int, j: int, total_dim: int):
    def log_prob_marginal_pair(x_2d):
        x = torch.zeros((x_2d.shape[0], total_dim))
        x[:, i] = x_2d[:, 0]
        x[:, j] = x_2d[:, 1]
        return log_prob(x)
    return log_prob_marginal_pair


def plot_contours(log_prob_func,
                  ax: Optional[plt.Axes] = None,
                  bounds: Tuple[float, float] = (-5.0, 5.0),
                  grid_width_n_points: int = 20,
                  n_contour_levels: Optional[int] = None,
                  log_prob_min: float = -1000.0):
    """Plot contours of a log_prob_func that is defined on 2D"""
    if ax is None:
        fig, ax = plt.subplots(1)
    x_points_dim1 = torch.linspace(bounds[0], bounds[1], grid_width_n_points)
    x_points_dim2 = x_points_dim1
    x_points = torch.tensor(list(itertools.product(x_points_dim1, x_points_dim2)))
    log_p_x = log_prob_func(x_points).detach()
    log_p_x = torch.clamp_min(log_p_x, log_prob_min)
    log_p_x = log_p_x.reshape((grid_width_n_points, grid_width_n_points))
    x_points_dim1 = x_points[:, 0].reshape((grid_width_n_points, grid_width_n_points)).numpy()
    x_points_dim2 = x_points[:, 1].reshape((grid_width_n_points, grid_width_n_points)).numpy()
    if n_contour_levels:
        ax.contour(x_points_dim1, x_points_dim2, log_p_x, levels=n_contour_levels)
    else:
        ax.contour(x_points_dim1, x_points_dim2, log_p_x)

def plot_marginal_pair(samples: torch.Tensor,
                  ax: Optional[plt.Axes] = None,
                  marginal_dims: Tuple[int, int] = (0, 1),
                  bounds: Tuple[float, float] = (-5.0, 5.0),
                  alpha: float = 0.5,
                  markersize: int = 3):
    """Plot samples from marginal of distribution for a given pair of dimensions."""
    if not ax:
        fig, ax = plt.subplots(1)
    samples = torch.clamp(samples, bounds[0], bounds[1])
    samples = samples.cpu().detach()
    ax.plot(samples[:, marginal_dims[0]], samples[:, marginal_dims[1]], "o", alpha=alpha, markersize=markersize)

def visualise_samples_1plot(samples, target, name="",):
    alpha = 0.3
    markersize = 2.0
    dim = samples.shape[-1]
    plotting_bounds = (-3, 3)

    fig, axs = plt.subplots(1, 1, figsize=(4, 3.6))
    target_log_prob = get_target_log_prob_marginal_pair(target.log_prob, 0, 2, dim)
    plot_contours(target_log_prob, bounds=plotting_bounds, ax=axs,
                        n_contour_levels=30, grid_width_n_points=100)
    plot_marginal_pair(samples, marginal_dims=(0, 2),
                            ax=axs, bounds=plotting_bounds, alpha=alpha, markersize=markersize)
    plt.axis("off")
    plt.savefig(f"{PATH}/{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.)
    plt.close()

def visualise_samples_2plots(samples, target, name="",):
    alpha = 0.3
    markersize = 2.0
    dim = samples.shape[-1]
    plotting_bounds = (-3, 3)   

    fig, axs = plt.subplots(2, 1, figsize=(4, 7.2))
    fig.subplots_adjust(hspace=0.01)

    for idx, (i,j) in enumerate([(0,3), (1,2)]):
        target_log_prob = get_target_log_prob_marginal_pair(target.log_prob, i, j, dim)
        plot_contours(target_log_prob, bounds=plotting_bounds, ax=axs[idx],
                        n_contour_levels=30, grid_width_n_points=100)
        plot_marginal_pair(samples, marginal_dims=(i, j),
                            ax=axs[idx], bounds=plotting_bounds, alpha=alpha, markersize=markersize)
        axs[idx].set_axis_off()

    plt.savefig(f"{PATH}/{name}_2plots.png", dpi=300, bbox_inches="tight", pad_inches=0.)
    plt.close()

def visualise_samples(samples, target, name="",):
    # visualise_samples_1plot(samples, target, name=name)
    visualise_samples_2plots(samples, target, name=name)

def plot_energy_hist(fab_samples, target):
    data_set = target.sample((5000,))
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))

    gt_energy = -(target.log_prob(data_set) - target.log_Z)
    fab_energy = -(target.log_prob(fab_samples) - target.log_Z)

    nfs_samples = np.load(f"{PATH}/ckpts/nfs_mw32_samples.npz")['positions']
    nfs_samples = torch.tensor(nfs_samples)
    nfs_energy = -(target.log_prob(nfs_samples) - target.log_Z)

    idem_samples = np.load(f"{PATH}/ckpts/idem_mw32_samples.npz")['positions']
    idem_samples = torch.tensor(idem_samples)
    idem_energy = -(target.log_prob(idem_samples) - target.log_Z)

    lfis_samples = np.load(f"{PATH}/ckpts/lfis_mw32_samples.npz")['positions']
    lfis_samples = torch.tensor(lfis_samples)
    lfis_energy = -(target.log_prob(lfis_samples) - target.log_Z)

    for energy, label in zip([gt_energy, fab_energy, idem_energy, lfis_energy, nfs_energy], ['Ground Truth', 'FAB', 'IDEM', 'LFIS', r"$\text{NFS}^2$ (ours)"]):
    # for energy, label in zip([gt_energy], ['Ground Truth']):
        energy = energy.cpu().detach()
        axs.hist(
            energy.view(-1),
            bins=100,
            alpha=0.5,
            density=True,
            range=(5, 75),
            histtype="step",
            linewidth=2,
            label=label,
        )

    axs.set_xlabel("Energy", fontsize=13)
    axs.legend()
    plt.savefig(f"{PATH}/mw32_hist.png", dpi=300, bbox_inches="tight", pad_inches=0.)
    exit()



@hydra.main(config_path="../config", config_name="many_well.yaml")
def main(cfg: DictConfig):
    n_samples = 1000
    target = ManyWellEnergy(cfg.target.dim, a=-0.5, b=-6, use_gpu=False)
    path_to_model = f"{PATH}/ckpts/mw_model.pt"
    model = load_model(cfg, target, path_to_model)

    # data_set = target.sample((5000,))
    # visualise_samples(data_set, target, name="data_samples")

    # nfs_samples = np.load(f"{PATH}/ckpts/nfs_mw32_samples.npz")['positions']
    # visualise_samples(torch.tensor(nfs_samples), target, name="nfs_samples")

    # samples = model.flow.sample((5000,)).detach()
    # visualise_samples(samples, target, name="fab_samples")

    samples = model.flow.sample((5000,)).detach()
    plot_energy_hist(samples, target)
    exit()

    results = defaultdict(list)
    for _ in range(10):
        samples = model.flow.sample((n_samples,)).detach()
        metrics = evaluate_metrics(model, target, samples)
        for k, v in metrics.items():
            results[k].append(v)

    for k, v in results.items():
        mean = np.mean(v)
        std = np.std(v)
        print(f"{k}: {mean} +- {std}")



if __name__ == '__main__':
    main()
