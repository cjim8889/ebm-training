from typing import Callable, Optional, Union

import chex
import matplotlib.pyplot as plt


class Target:
    """Base class for distributions"""

    TIME_DEPENDENT = False
    TARGET_METRIC = "w2_distance_mean"

    def __init__(
        self,
        dim: int,
        log_Z: Optional[float],
        can_sample: bool,
        n_plots: int,
        n_model_samples_eval: int,
        n_target_samples_eval: Optional[int],
    ):
        self.n_model_samples_eval = n_model_samples_eval
        self.n_target_samples_eval = n_target_samples_eval
        self._dim = dim
        self._log_Z = log_Z
        self._n_plots = n_plots
        self._can_sample = can_sample

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def n_plots(self) -> int:
        return self._n_plots

    @property
    def can_sample(self) -> bool:
        return self._can_sample

    @property
    def log_Z(self) -> Union[int, None]:
        return self._log_Z

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape = ()) -> chex.Array:
        raise NotImplementedError

    def log_prob(self, value: chex.Array) -> chex.Array:
        raise NotImplementedError

    def time_dependent_log_prob(self, value: chex.Array, time: float) -> chex.Array:
        raise NotImplementedError

    def visualise(self, samples: chex.Array) -> plt.Figure:
        raise NotImplementedError

    def visualise_with_time(self, samples: chex.Array, time: float) -> plt.Figure:
        raise NotImplementedError

    def evaluate(
        self,
        key: chex.PRNGKey,
        v_theta: Callable,
        ts: chex.Array,
        use_shortcut: bool = False,
        **kwargs,
    ) -> dict:
        raise NotImplementedError
