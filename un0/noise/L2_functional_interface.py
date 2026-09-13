from __future__ import annotations

from typing import Callable
import torch
from torch import Tensor, nn
from . import register_noise_model


def _quantize(x: Tensor, num_bits: int, val_min: float = -10.0, val_max: float = 10.0) -> Tensor:
    """Quantize tensor x to num_bits over [val_min, val_max]."""
    if num_bits >= 32 or num_bits <= 0:
        return x
    levels = (2**num_bits) - 1
    x_clamped = torch.clamp(x, val_min, val_max)
    normalized = (x_clamped - val_min) / (val_max - val_min)
    quantized = torch.round(normalized * levels) / levels
    return quantized * (val_max - val_min) + val_min


@register_noise_model("L2_functional_interface")
class L2FunctionalInterfaceNoise(nn.Module):
    """L2: Functional interface DAC/ADC quantization & clipping wrapper."""

    def __init__(
        self,
        dynamics: nn.Module | Callable,
        dac_bits: int = 8,
        adc_bits: int = 8,
        clip_range: tuple[float, float] = (-10.0, 10.0),
    ) -> None:
        super().__init__()
        self.dynamics = dynamics
        self.dac_bits = int(dac_bits)
        self.adc_bits = int(adc_bits)
        self.clip_min, self.clip_max = clip_range

    def forward(self, state: Tensor, t: Tensor, drive: Tensor) -> Tensor:
        # DAC quantization on state and drive
        q_state = _quantize(state, self.dac_bits, self.clip_min, self.clip_max)
        q_drive = _quantize(drive, self.dac_bits, self.clip_min, self.clip_max)
        vel = self.dynamics(q_state, t, q_drive)
        # ADC quantization on velocity output
        return _quantize(vel, self.adc_bits, self.clip_min, self.clip_max)
