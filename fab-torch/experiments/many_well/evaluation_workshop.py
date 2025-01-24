import hydra

import ot as pot
import numpy as np
import pandas as pd
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

def evaluate_metrics(target, gen_samples):
    data_set = target.sample((gen_samples.shape[0], ))
    gen_samples = gen_samples.cpu()

    energies = target.log_prob(data_set)
    generated_energies = target.log_prob(gen_samples)
    e_w2_dist = np.sqrt(pot.emd2_1d(energies.numpy(), generated_energies.numpy()))

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
        'x_w2_dist': x_w2_dist
    }

@hydra.main(config_path="../config", config_name="many_well.yaml")
def main(cfg: DictConfig):
    n_samples = 1000
    target = ManyWellEnergy(cfg.target.dim, a=-0.5, b=-6, use_gpu=False)
    path_to_model = f"{PATH}/ckpts/mw_model.pt"
    model = load_model(cfg, target, path_to_model)

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



if __name__ == '__main__':
    main()
