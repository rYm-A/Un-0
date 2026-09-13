from __future__ import annotations

from typing import Callable
import torch
from torch import Tensor, nn
from . import register_noise_model


@register_noise_model("L3_correlated_drift")
class L3CorrelatedDriftNoise(nn.Module):
    """L3: Correlated drift noise wrapper."""

    def __init__(
        self,
        dynamics: nn.Module | Callable,
        power_law_exponent: float = 0.02,
        drift_rate: float = 0.001,
    ) -> None:
        super().__init__()
        self.dynamics = dynamics
        self.power_law_exponent = float(power_law_exponent)
        self.drift_rate = float(drift_rate)

    def forward(self, state: Tensor, t: Tensor, drive: Tensor) -> Tensor:
        vel = self.dynamics(state, t, drive)
        # Time-dependent spatial drift: scales with t^exponent and time
        t_scalar = float(t.item()) if isinstance(t, Tensor) and t.numel() == 1 else float(t)
        drift_scale = self.drift_rate * (1.0 + (abs(t_scalar) ** self.power_law_exponent))
        # Correlated spatial modulation across last dimension
        dim = vel.shape[-1]
        spatial_pattern = torch.sin(torch.arange(dim, device=vel.device, dtype=vel.dtype) * 0.1)
        drift = drift_scale * spatial_pattern.unsqueeze(0)
        return vel + drift
