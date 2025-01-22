import ot as pot
import numpy as np
import torch

def eval_data_w2(target_density, generated_samples):
    data_set = target_density.test_set
    
    distance_matrix = pot.dist(data_set.cpu().numpy(), generated_samples.cpu().numpy(), metric='euclidean')
    src, dist = np.ones(len(data_set)) / len(data_set), np.ones(len(generated_samples)) / len(generated_samples)
    G = pot.emd(src, dist, distance_matrix)
    w2_dist = np.sum(G * distance_matrix) / G.sum()
    return w2_dist

def eval_energy_w2(target_density, generated_samples):
    data_set = target_density.test_set

    energies = target_density.log_prob(data_set)
    generated_energies = target_density.log_prob(generated_samples)
    energy_w2 = pot.emd2_1d(energies.cpu().numpy(), generated_energies.cpu().numpy())
    print(energies.shape, generated_energies.shape)
    print(energy_w2)
    exit()

def eval_total_variation():
    pass