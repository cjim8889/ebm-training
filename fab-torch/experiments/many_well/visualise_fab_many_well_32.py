import os
import urllib

import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
from hydra import compose, initialize
import torch

from fab.utils.plotting import plot_contours, plot_marginal_pair
from fab.target_distributions.many_well import ManyWellEnergy
from experiments.setup_run import setup_model
from experiments.many_well.many_well_visualise_all_marginal_pairs import get_target_log_prob_marginal_pair

with initialize(version_base=None, config_path="../config/", job_name="colab_app"):
    cfg = compose(config_name=f"many_well")

target = ManyWellEnergy(cfg.target.dim, a=-0.5, b=-6, use_gpu=False)
model = setup_model(cfg, target)

# Download weights from huggingface, and load them into the model
urllib.request.urlretrieve('https://huggingface.co/VincentStimper/fab/resolve/main/many_well/model.pt', 'model.pt')
model.load("model.pt", map_location="cpu")

# Sample from the model
n_samples: int = 200
samples_flow = model.flow.sample((n_samples,)).detach()

# Visualise samples
alpha = 0.3
plotting_bounds = (-3, 3)
dim = cfg.target.dim
fig, axs = plt.subplots(2, 2, sharex="row", sharey="row")

for i in range(2):
    for j in range(2):
        target_log_prob = get_target_log_prob_marginal_pair(target.log_prob, i, j + 2, dim)
        plot_contours(target_log_prob, bounds=plotting_bounds, ax=axs[i, j],
                      n_contour_levels=20, grid_width_n_points=100)
        plot_marginal_pair(samples_flow, marginal_dims=(i, j+2),
                           ax=axs[i, j], bounds=plotting_bounds, alpha=alpha)


        if j == 0:
            axs[i, j].set_ylabel(f"$x_{i + 1}$")
        if i == 1:
            axs[i, j].set_xlabel(f"$x_{j + 1 + 2}$")

plt.tight_layout()
plt.savefig("plots/fab_many_well_32.png")