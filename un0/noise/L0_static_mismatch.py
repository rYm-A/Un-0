from __future__ import annotations

from typing import Callable
import torch
from torch import Tensor, nn
from . import register_noise_model


@register_noise_model("L0_static_mismatch")
class L0StaticMismatchNoise(nn.Module):
    """L0: Static mismatch noise wrapper.

    Applies a fixed, static perturbation tensor to the velocity computation.
    """

    def __init__(self, dynamics: nn.Module | Callable, sigma: float = 0.01) -> None:
        super().__init__()
        self.dynamics = dynamics
        self.sigma = float(sigma)
        self.register_buffer("static_mismatch", None, persistent=False)

    def forward(self, state: Tensor, t: Tensor, drive: Tensor) -> Tensor:
        vel = self.dynamics(state, t, drive)
        if (
            self.static_mismatch is None
            or self.static_mismatch.shape != vel.shape
            or self.static_mismatch.device != vel.device
        ):
            self.static_mismatch = torch.randn_like(vel) * self.sigma
        return vel + self.static_mismatch
