from __future__ import annotations

from typing import Any, Callable
import torch
from torch import Tensor, nn
from . import register_noise_model


@register_noise_model("L1_stochastic_parameter_noise")
class L1StochasticParameterNoise(nn.Module):
    """L1: Stochastic parameter/thermal noise wrapper.

    Adds independent Gaussian noise drawn per forward step.
    """

    def __init__(self, dynamics: nn.Module | Callable, sigma: float = 0.01) -> None:
        super().__init__()
        self.dynamics = dynamics
        self.sigma = float(sigma)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            if "dynamics" in self.__dict__:
                return getattr(self.dynamics, name)
            raise

    def forward(self, state: Tensor, t: Tensor, drive: Tensor) -> Tensor:
        vel = self.dynamics(state, t, drive)
        noise = torch.randn_like(vel) * self.sigma
        return vel + noise
