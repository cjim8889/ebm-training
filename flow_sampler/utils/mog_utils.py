import torch
import itertools
import numpy as np
from matplotlib import pyplot as plt

class GMM(torch.nn.Module):
    def __init__(self, dim, n_mixes, loc_scaling, log_var_scaling=1., seed=0,
                 n_test_set_samples=1000, device="cpu"):
        super(GMM, self).__init__()

        # fix mean and variance for reproducibility
        torch.manual_seed(0)
        self.n_mixes = n_mixes
        self.dim = dim
        self.n_test_set_samples = n_test_set_samples

        mean = (torch.rand((n_mixes, dim), ) - 0.5)*2 * loc_scaling
        log_var = torch.ones((n_mixes, dim)) * log_var_scaling

        self.register_buffer("cat_probs", torch.ones(n_mixes))
        self.register_buffer("locs", mean)
        self.register_buffer("scale_trils", torch.diag_embed(torch.nn.functional.softplus(log_var)))
        self.device = device
        self.to(self.device)

        # reset seed
        torch.manual_seed(seed)

    def to(self, device):
        if device == "cuda":
            if torch.cuda.is_available():
                self.cuda()
        else:
            self.cpu()

    @property
    def distribution(self):
        mix = torch.distributions.Categorical(self.cat_probs.to(self.device))
        com = torch.distributions.MultivariateNormal(self.locs.to(self.device),
                                                     scale_tril=self.scale_trils.to(self.device),
                                                     validate_args=False)
        return torch.distributions.MixtureSameFamily(mixture_distribution=mix,
                                                     component_distribution=com,
                                                     validate_args=False)

    @property
    def test_set(self) -> torch.Tensor:
        return self.sample((self.n_test_set_samples, ))

    def log_prob(self, x: torch.Tensor):
        log_prob = self.distribution.log_prob(x)
        mask = torch.zeros_like(log_prob)
        mask[log_prob < -1e4] = - torch.tensor(float("inf"))
        log_prob = log_prob + mask
        return log_prob

    def sample(self, shape=(1,)):
        return self.distribution.sample(shape)

class MultivariateGaussian:
    """
    Multivariate Gaussian Distribution with Sigma * I covariance.

    Parameters:
    - dim: int
        Dimensionality of the Gaussian.
    - sigma: float or jnp.ndarray, default=1.0
        Standard deviation of the distribution. If a float is provided, the same sigma is used for all dimensions.
        If an array is provided, it should have shape (dim,) for per-dimension sigma.
    - plot_bound_factor: float, default=3.0
        Factor to determine the plotting bounds based on sigma.
    """

    def __init__(
        self, dim: int = 2, sigma: float = 1.0, device: str = 'cuda', **kwargs
    ):
        """
        Initializes the MultivariateGaussian distribution.

        Args:
            dim (int): Dimensionality of the Gaussian.
            sigma (float or np.array): Standard deviation(s). Scalar for isotropic, array for anisotropic.
            plot_bound_factor (float): Factor to determine the plotting bounds.
            **kwargs: Additional arguments to pass to the base Target class.
        """

        self.sigma = np.array(sigma)
        if self.sigma.ndim == 0:
            # Scalar sigma: isotropic covariance
            scale_diag = torch.ones(dim) * self.sigma
        else:
            # Per-dimension sigma
            if self.sigma.shape[0] != dim:
                raise ValueError(
                    f"Sigma shape {self.sigma.shape} does not match dimension {dim}."
                )
            scale_diag = self.sigma

        self.distribution = torch.distributions.MultivariateNormal(
            loc=torch.zeros(dim, device=device), 
            covariance_matrix=torch.diag(scale_diag ** 2).to(device)
        )


    def log_prob(self, x):
        """
        Computes the log probability density of the input samples.

        Args:
            x (torch.Tensor): Input samples with shape (..., dim).

        Returns:
            torch.Tensor: Log probability densities.
        """
        return self.distribution.log_prob(x)

    def sample(self, sample_shape):
        """
        Generates samples from the distribution.

        Args:
            sample_shape (torch.Size): Shape of the samples to generate.

        Returns:
            torch.Tensor: Generated samples with shape `sample_shape + (dim,)`.
        """
        return self.distribution.sample(sample_shape=sample_shape)


def plot_contours(log_prob_func,
                  samples = None,
                  ax = None,
                  bounds = (-5.0, 5.0),
                  grid_width_n_points = 20,
                  n_contour_levels = None,
                  log_prob_min = -1000.0,
                  device='cpu',
                  plot_marginal_dims=[0, 1],
                  s=2,
                  alpha=0.6,
                  title=None,
                  plt_show=True,
                  xy_tick=True,):
    """Plot contours of a log_prob_func that is defined on 2D"""
    if ax is None:
        fig, ax = plt.subplots(1)
    x_points_dim1 = torch.linspace(bounds[0], bounds[1], grid_width_n_points)
    x_points_dim2 = x_points_dim1
    x_points = torch.tensor(list(itertools.product(x_points_dim1, x_points_dim2)), device=device)
    log_p_x = log_prob_func(x_points).cpu().detach()
    log_p_x = torch.clamp_min(log_p_x, log_prob_min)
    log_p_x = log_p_x.reshape((grid_width_n_points, grid_width_n_points))
    x_points_dim1 = x_points[:, 0].reshape((grid_width_n_points, grid_width_n_points)).cpu().numpy()
    x_points_dim2 = x_points[:, 1].reshape((grid_width_n_points, grid_width_n_points)).cpu().numpy()
    if n_contour_levels:
        ax.contour(x_points_dim1, x_points_dim2, log_p_x, levels=n_contour_levels)
    else:
        ax.contour(x_points_dim1, x_points_dim2, log_p_x)

    if samples is not None:
        samples = np.clip(samples, bounds[0], bounds[1])
        ax.scatter(samples[:, plot_marginal_dims[0]], samples[:, plot_marginal_dims[1]], s=s, alpha=alpha)
        ### x,y ticks
        if xy_tick:
            ax.set_xticks([-40,0,40])
            ax.set_yticks([-40,0,40])
        ### size of ticks
        ax.tick_params(axis='both', which='major', labelsize=15)
    if title:
        ax.set_title(title)
        ### size of title 
        ax.title.set_fontsize(40)
    if plt_show:
        plt.show()

    return fig


def plot_MoG40(log_prob_function,samples, save_path=None, save_name=None,title=None):
    fig = plot_contours(log_prob_function, samples=samples.detach().cpu().numpy(), bounds=(-56,56), n_contour_levels=50, grid_width_n_points=200, device="cuda",title=title,plt_show=False)
    # plt.savefig(f"""tmp.png""")
    plt.close(fig)

    return fig
