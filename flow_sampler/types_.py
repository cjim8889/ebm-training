from typing import Callable, Tuple, Mapping, Any, Iterator
import torch
import abc

LogProbFunc = Callable[[torch.Tensor], torch.Tensor]