from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from torch import nn

logger = logging.getLogger("un0.compiler.compile_model")


def compile_model(
    model: nn.Module,
    backend: str = "inductor",
    mode: str = "default",
    dynamic: bool = True,
    target_device: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    """Wrap a PyTorch model with torch.compile using AOT / Inductor backend.

    Supports dynamic batch sizes >= 1 without upper bound.

    Args:
        model: PyTorch Module to compile.
        backend: Compiler backend ('inductor', 'aot_eager', etc.).
        mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune').
        dynamic: Whether to enable dynamic shapes (dynamic batch size).
        target_device: Optional target device ('mps', 'cuda', 'cpu').
        **kwargs: Extra arguments passed to torch.compile.

    Returns:
        Compiled nn.Module.
    """
    if target_device is not None:
        try:
            model = model.to(target_device)
        except Exception as dev_err:
            logger.warning("Could not move model to target device '%s': %s", target_device, dev_err)

    if hasattr(model, "eval") and callable(getattr(model, "eval")):
        model.eval()

    # Configure Torch Dynamo optimizations for HOP and scalar captures
    try:
        torch._dynamo.config.capture_scalar_outputs = True
    except Exception:
        pass

    try:
        compiled = torch.compile(
            model,
            backend=backend,
            mode=mode,
            dynamic=dynamic,
            **kwargs,
        )
        return compiled
    except Exception as exc:
        logger.warning(
            "torch.compile failed with backend '%s': %s. Falling back to aot_eager / uncompiled model.",
            backend,
            exc,
        )
        try:
            return torch.compile(model, backend="aot_eager", dynamic=dynamic)
        except Exception:
            return model
