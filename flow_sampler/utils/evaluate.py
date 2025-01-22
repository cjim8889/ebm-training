import ot as pot
import numpy as np
import torch

def eval_data_w2(target_density, generated_samples):
    data_set = target_density.test_set
    
    distance_matrix = pot.dist(data_set.cpu().numpy(), generated_samples.cpu().numpy(), metric='euclidean')
    distance_matrix = distance_matrix**2
    src, dist = np.ones(len(data_set)) / len(data_set), np.ones(len(generated_samples)) / len(generated_samples)
    w2_dist = np.sqrt(pot.emd2(src, dist, distance_matrix))
    return w2_dist

def eval_energy_w2(target_density, generated_samples):
    data_set = target_density.test_set

    energies = target_density.log_prob(data_set)
    generated_energies = target_density.log_prob(generated_samples)
    w2_dist = np.sqrt(pot.emd2_1d(energies.cpu().numpy(), generated_energies.cpu().numpy()))
    return w2_dist

def eval_data_total_variation(target_density, generated_samples):
    data_set = target_density.test_set

    bins = (200, ) * target_density.dimensionality
    all_data = torch.cat([data_set, generated_samples], dim=0)
    min_vals, _ = all_data.min(dim=0)
    max_vals, _ = all_data.max(dim=0)

    ranges = tuple((min_vals[i].item(), max_vals[i].item()) for i in range(target_density.dimensionality))  # tuple of (min, max) for each dimension
    ranges = tuple(item for subtuple in ranges for item in subtuple)
    hist_p, _ = torch.histogramdd(data_set.cpu(), bins=bins, range=ranges)
    hist_q, _ = torch.histogramdd(generated_samples.cpu(), bins=bins, range=ranges)
    
    p_dist = hist_p / hist_p.sum()
    q_dist = hist_q / hist_q.sum()
    
    total_var = 0.5 * torch.abs(p_dist - q_dist).sum()
    return total_var.item()


def eval_dist_total_variation():
    pass