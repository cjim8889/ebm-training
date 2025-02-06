import hydra
import matplotlib.pyplot as plt
from typing import Optional, Tuple

import ot as pot
import numpy as np
import os
from omegaconf import DictConfig
import torch
from collections import defaultdict

from fab.target_distributions.gmm import GMM
from experiments.load_model_for_eval import load_model
from fab.utils.plotting import plot_contours, plot_marginal_pair


PATH = os.getcwd()

def setup_target(cfg, num_samples):
    # Setup target
    torch.manual_seed(0)  #  Always 0 for GMM problem
    target = GMM(dim=cfg.target.dim, n_mixes=cfg.target.n_mixes,
                 loc_scaling=cfg.target.loc_scaling, log_var_scaling=cfg.target.log_var_scaling,
                 use_gpu=False, n_test_set_samples=num_samples)
    if cfg.training.use_64_bit:
        target = target.double()
    return target


def _plot_marginal_pair(samples: torch.Tensor,
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
    samples_list = []
    steps = [8, 16, 32, 64, 128]
    
    for step in steps:
        samples = np.load(f"{path}/gmm_samples_{step}_steps.npz")['positions']
        samples_list.append(samples)

    fig, axs = plt.subplots(1, len(steps), figsize=(25, 4))
    # fig.subplots_adjust(hspace=0.01, wspace=0.5)
    for i in range(len(steps)):
        samples = torch.tensor(samples_list[i])
        plot_contours(
            target.log_prob,
            bounds=plotting_bounds,
            ax=axs[i],
            n_contour_levels=50,
            grid_width_n_points=200,
        )
        _plot_marginal_pair(samples, ax=axs[i], bounds=plotting_bounds, markersize=1.5)
        axs[i].set_axis_off()
        axs[i].set_title(f"{steps[i]} steps", fontsize=20)

    plt.savefig(f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.)
    plt.close()

@hydra.main(config_path="../config", config_name="gmm.yaml")
def main(cfg: DictConfig, debug=False):
    target = setup_target(cfg, 1000)

    # path = f"{PATH}/ckpts/samples/gmm/nfs/"
    # visualise_samples(target, path=path, name=f"{PATH}/nfs_gmm_diff_stpes")

    path = f"{PATH}/ckpts/samples/gmm/idem/"
    visualise_samples(target, path=path, name=f"{PATH}/idem_gmm_diff_stpes")
    
    # path = f"{PATH}/ckpts/samples/gmm/lfis/"
    # visualise_samples(target, path=path, name=f"{PATH}/lfis_gmm_diff_stpes")

if __name__ == '__main__':
    main()
