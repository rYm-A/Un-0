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
        self.register_buffer("sigma", torch.tensor(float(sigma)))

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            if "dynamics" in self.__dict__:
                return getattr(self.dynamics, name)
            raise

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "sigma" and isinstance(value, (int, float)):
            if "_buffers" in self.__dict__ and "sigma" in self._buffers and self._buffers["sigma"] is not None:
                self._buffers["sigma"].fill_(float(value))
                return
        super().__setattr__(name, value)

    def forward(self, state: Tensor, t: Tensor, drive: Tensor) -> Tensor:
        vel = self.dynamics(state, t, drive)
        noise = torch.randn_like(vel) * self.sigma
        return vel + noise
