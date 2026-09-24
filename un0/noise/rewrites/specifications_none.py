from __future__ import annotations
import torch
from torch import Tensor, nn
from typing import Callable
from un0.noise import register_noise_model

@register_noise_model("none")
class NoneNoise(nn.Module):
    """Agent synthesized physical hardware noise wrapper."""

    def __init__(self, dynamics: nn.Module | Callable, sigma: float = 0.01) -> None:
        super().__init__()
        self.dynamics = dynamics
        self.sigma = float(sigma)

    def forward(self, state: Tensor, t: Tensor, drive: Tensor) -> Tensor:
        vel = self.dynamics(state, t, drive)
        noise = torch.randn_like(vel) * self.sigma
        return vel + noise
