import torch
from einops import rearrange, repeat

def loss_fn(v_theta, xs, ts, time_derivative_log_density, score_fn):
    ts = ts.to(xs.device)
    ts = repeat(ts, 't -> n t', n=xs.size(0))
    
    dt_log_unormalised_density = time_derivative_log_density(xs, ts)
    dt_log_density = dt_log_unormalised_density - dt_log_unormalised_density.mean(dim=0, keepdim=True)

    score = score_fn(xs, ts)

    xs.requires_grad_(True)
    v = v_theta(xs, ts.unsqueeze(-1))
    div_v = torch.zeros(xs.shape[:2], device=xs.device)
    for i in range(xs.shape[2]):
        div_v += torch.autograd.grad(v[..., i].sum(), xs, create_graph=True)[0][..., i]

    lhs = div_v + (v * score).sum(dim=-1)
    eps = (lhs + dt_log_density).nan_to_num_(posinf=1.0, neginf=-1.0, nan=0.0)
    return (eps**2).mean()