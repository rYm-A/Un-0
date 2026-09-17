from __future__ import annotations

from typing import Any, Callable
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

        dim = None
        curr = dynamics
        while curr is not None:
            if hasattr(curr, "state_dim") and isinstance(getattr(curr, "state_dim"), int):
                dim = getattr(curr, "state_dim")
                break
            if hasattr(curr, "n") and isinstance(getattr(curr, "n"), int):
                dim = getattr(curr, "n") + getattr(curr, "n_cond", 0)
                break
            curr = getattr(curr, "dynamics", None)

        if dim is not None:
            self.register_buffer("static_mismatch", torch.zeros(1, dim), persistent=False)
        else:
            self.register_buffer("static_mismatch", None, persistent=False)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            if "dynamics" in self.__dict__:
                return getattr(self.dynamics, name)
            raise

    def forward(self, state: Tensor, t: Tensor, drive: Tensor) -> Tensor:
        vel = self.dynamics(state, t, drive)
        if self.static_mismatch is not None and self.static_mismatch.shape[-1] == vel.shape[-1]:
            return vel + self.static_mismatch.to(device=vel.device, dtype=vel.dtype)
        mismatch = torch.randn(1, vel.shape[-1], device=vel.device, dtype=vel.dtype) * self.sigma
        return vel + mismatch
