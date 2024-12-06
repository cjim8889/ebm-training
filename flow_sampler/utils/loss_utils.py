import torch
from einops import rearrange, repeat

def loss_fn(v_theta, xs, ts, time_derivative_log_density, score_fn):
    ts = ts.to(xs.device)
    ts = repeat(ts, 't -> n t', n=xs.size(0))
    
    dt_log_unormalised_density = time_derivative_log_density(xs, ts)
    dt_log_density = dt_log_unormalised_density - dt_log_unormalised_density.mean(dim=0, keepdim=True)

    score = score_fn(xs, ts)

    b, t, d = xs.shape
    xs = rearrange(xs, "b t d -> (b t) d")
    ts = rearrange(ts, "b t -> (b t)")
    
    xs_detached = xs.detach().requires_grad_(True)
    v = v_theta(xs_detached, ts.unsqueeze(-1))

    # Calculate divergence
    # div_v = torch.zeros(xs_detached.shape[:1], device=xs_detached.device)
    # for i in range(xs_detached.shape[-1]):
    #     if i < xs_detached.shape[-1] - 1:
    #         div_v += torch.autograd.grad(
    #             v[..., i].sum(), 
    #             xs_detached, 
    #             create_graph=True,  # Need this for second-order gradients
    #             retain_graph=True   # Retain graph for subsequent iterations
    #         )[0][..., i]
    #     else:
    #         # On last iteration, we don't need to retain the graph
    #         div_v += torch.autograd.grad(
    #             v[..., i].sum(), 
    #             xs_detached, 
    #             create_graph=True
    #         )[0][..., i]

    # Alternative way to calculate divergence
    div_v = torch.zeros(xs_detached.shape[:1], device=xs_detached.device)
    for i in range(xs_detached.shape[-1]):
        div_v += torch.autograd.grad(v[..., i].sum(), xs_detached, create_graph=True, retain_graph=True)[0][..., i]
    
    div_v = rearrange(div_v, "(b t) -> b t", b=b)
    v = rearrange(v, "(b t) d -> b t d", b=b)

    lhs = div_v + (v * score).sum(dim=-1)
    eps = (lhs + dt_log_density).nan_to_num_(posinf=1.0, neginf=-1.0, nan=0.0)
    return (eps**2).mean()