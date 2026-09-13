from __future__ import annotations

import torch
from torch import nn


def compile_model(
    model: nn.Module,
    backend: str = "inductor",
    mode: str = "default",
    dynamic: bool = True,
    **kwargs,
) -> nn.Module:
    """Wrap a PyTorch model with torch.compile."""
    try:
        return torch.compile(model, backend=backend, mode=mode, dynamic=dynamic, **kwargs)
    except Exception as e:
        print(f"Warning: torch.compile failed with error: {e}. Returning uncompiled model.")
        return model
