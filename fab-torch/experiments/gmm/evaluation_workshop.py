import hydra
import matplotlib.pyplot as plt

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

def eval_data_total_variation(dimensionality, data_set, generated_samples):
    bins = (200, ) * dimensionality
    all_data = torch.cat([data_set, generated_samples], dim=0)
    min_vals, _ = all_data.min(dim=0)
    max_vals, _ = all_data.max(dim=0)

    ranges = tuple((min_vals[i].item(), max_vals[i].item()) for i in range(dimensionality))  # tuple of (min, max) for each dimension
    ranges = tuple(item for subtuple in ranges for item in subtuple)
    hist_p, _ = torch.histogramdd(data_set.cpu(), bins=bins, range=ranges)
    hist_q, _ = torch.histogramdd(generated_samples.cpu(), bins=bins, range=ranges)
    
    p_dist = hist_p / hist_p.sum()
    q_dist = hist_q / hist_q.sum()
    
    total_var = 0.5 * torch.abs(p_dist - q_dist).sum()
    return total_var.item()

def evaluate_metrics(target, gen_samples):
    data_set = target.sample((gen_samples.shape[0], ))
    gen_samples = gen_samples.cpu()

    energies = target.log_prob(data_set)
    generated_energies = target.log_prob(gen_samples)
    e_w2_dist = np.sqrt(pot.emd2_1d(energies.numpy(), generated_energies.numpy()))
    x_tv = eval_data_total_variation(2, data_set, gen_samples)

    return {
        'e_w2_dist': e_w2_dist,
        'x_tv': x_tv
    }

# use base config of GMM but overwrite for specific model.
@hydra.main(config_path="../config", config_name="gmm.yaml")
def main(cfg: DictConfig, debug=False):
    n_samples = 1000
    target = setup_target(cfg, n_samples)
    path_to_model = f"{PATH}/ckpts/gmm_model.pt"
    model = load_model(cfg, target, path_to_model)
    model.set_ais_target(min_is_target=False)

    results = defaultdict(list)
    for _ in range(5):
        samples = model.flow.sample((n_samples,)).detach()
        metrics = evaluate_metrics(target, samples)
        for k, v in metrics.items():
            results[k].append(v)

    for k, v in results.items():
        mean = np.mean(v)
        std = np.std(v)
        print(f"{k}: {mean} +- {std}")

    if debug:
        alpha = 0.3
        plotting_bounds = (-cfg.target.loc_scaling * 1.4, cfg.target.loc_scaling * 1.4)
        fig, ax = plt.subplots()
        # samples_flow = target.sample((n_samples, ))
        samples_flow = model.flow.sample((n_samples,)).detach()
        plot_marginal_pair(samples_flow, ax=ax, bounds=plotting_bounds, alpha=alpha)
        plot_contours(target.log_prob, bounds=plotting_bounds, ax=ax, n_contour_levels=50,
                        grid_width_n_points=200)
        plt.savefig(f"{PATH}/gen_samples.png")

if __name__ == '__main__':
    main()
