import torch


def get_target_log_prob_marginal_pair(log_prob, i: int, j: int, total_dim: int):
    def log_prob_marginal_pair(x_2d):
        x = torch.zeros((x_2d.shape[0], total_dim))
        x[:, i] = x_2d[:, 0]
        x[:, j] = x_2d[:, 1]
        return log_prob(x)
    return log_prob_marginal_pair
