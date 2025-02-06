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


def visualise_samples(target, path="", name="", plotting_bounds=(-1.4 * 40, 1.4 * 40)):
    alpha = 0.3
    markersize = 2.0
    dim = 32
    plotting_bounds = (-3, 3)


    samples_list = []
    steps = [8, 16, 32, 64, 128]
    
    for step in steps:
        samples = np.load(f"{path}/mw32_samples_{step}_steps.npz")['positions']
        samples_list.append(samples)

    fig, axs = plt.subplots(1, len(steps), figsize=(25, 4))
    # fig.subplots_adjust(hspace=0.01, wspace=0.1)

    target_log_prob = get_target_log_prob_marginal_pair(target.log_prob, 0, 2, dim)
    for i in range(len(steps)):
        samples = torch.tensor(samples_list[i])
        plot_contours(target_log_prob, bounds=plotting_bounds, ax=axs[i],
                        n_contour_levels=30, grid_width_n_points=100)
        plot_marginal_pair(samples, marginal_dims=(0, 2),
                                ax=axs[i], bounds=plotting_bounds, alpha=alpha, markersize=markersize)
        axs[i].set_axis_off()
        axs[i].set_title(f"{steps[i]} steps", fontsize=20)

    plt.savefig(f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.)
    plt.close()

@hydra.main(config_path="../config", config_name="many_well.yaml")
def main(cfg: DictConfig):
    target = ManyWellEnergy(cfg.target.dim, a=-0.5, b=-6, use_gpu=False)

    # path = f"{PATH}/ckpts/samples/mw32/nfs"
    # visualise_samples(target, path=path, name=f"{PATH}/nfs_mw32_diff_stpes")

    # path = f"{PATH}/ckpts/samples/mw32/idem"
    # visualise_samples(target, path=path, name=f"{PATH}/idem_mw32_diff_stpes")

    path = f"{PATH}/ckpts/samples/mw32/lfis"
    visualise_samples(target, path=path, name=f"{PATH}/lfis_mw32_diff_stpes")

if __name__ == '__main__':
    main()
