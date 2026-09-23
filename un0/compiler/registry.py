"""Compiler Rewrite Registry for Un-0 Kuramoto models.

Provides registration, lookup, and management of functionalization and AOT compilation
rewrites (e.g. PyTorch Higher-Order Operators, while_loop, unrolled functional loops).
Mirrors un0.solver and un0.noise registry patterns.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

import torch
from torch import nn

logger = logging.getLogger("un0.compiler.registry")

COMPILER_REGISTRY: Dict[str, Callable] = {}


def register_compiler_rewrite(name: str | Callable | None = None) -> Callable:
    """Decorator to register a compiler functionalization rewrite in COMPILER_REGISTRY.

    Signature of registered rewrite callable:
        rewrite_fn(
            model: nn.Module,
            solver_name: str,
            num_steps: int,
            dt: float,
            integration_time: float,
            target_device: str = "mps",
            **kwargs,
        ) -> nn.Module
    """
    if callable(name):
        fn = name
        rewrite_name = fn.__name__
        COMPILER_REGISTRY[rewrite_name] = fn
        return fn

    def decorator(fn: Callable) -> Callable:
        rewrite_name = name if name is not None else fn.__name__
        COMPILER_REGISTRY[rewrite_name] = fn
        return fn

    return decorator


def get_compiler_rewrite(name: str) -> Callable:
    """Lookup a registered compiler rewrite function by name.

    Args:
        name: Registered identifier of the compiler rewrite.

    Returns:
        Callable functionalizing the Kuramoto dynamics model.
    """
    if name not in COMPILER_REGISTRY:
        # Dynamically discover rewrites stored in un0/compiler/rewrites/
        from pathlib import Path
        import importlib.util

        rewrites_dir = Path(__file__).parent / "rewrites"
        if rewrites_dir.is_dir():
            for py_file in rewrites_dir.glob("*.py"):
                if py_file.name == "__init__.py":
                    continue
                if name in py_file.stem:
                    try:
                        mod_name = f"un0.compiler.rewrites.{py_file.stem}"
                        spec = importlib.util.spec_from_file_location(mod_name, py_file)
                        if spec and spec.loader:
                            mod = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(mod)
                            if name not in COMPILER_REGISTRY:
                                for candidate_name in (name, "rewrite", "rewrite_fn", py_file.stem):
                                    candidate_fn = getattr(mod, candidate_name, None)
                                    if candidate_fn is not None and callable(candidate_fn):
                                        COMPILER_REGISTRY[name] = candidate_fn
                                        break
                                if name not in COMPILER_REGISTRY:
                                    for attr_name in dir(mod):
                                        val = getattr(mod, attr_name)
                                        if callable(val) and not attr_name.startswith("_") and getattr(val, "__module__", "") == mod_name:
                                            COMPILER_REGISTRY[name] = val
                                            break
                    except Exception as exc:
                        logger.warning("Auto-load rewrite failed for %s: %s", py_file, exc)
    if name not in COMPILER_REGISTRY:
        raise KeyError(
            f"Compiler rewrite {name!r} not found in registry. "
            f"Available rewrites: {list(COMPILER_REGISTRY.keys())}"
        )
    return COMPILER_REGISTRY[name]


# Register built-in compiler rewrites
@register_compiler_rewrite("hop_while_loop")
def _hop_while_loop_rewrite(
    model: nn.Module,
    solver_name: str = "rk4",
    num_steps: int = 25,
    dt: float = 0.04,
    integration_time: float = 1.0,
    target_device: str = "mps",
    **kwargs,
) -> nn.Module:
    """Functionalize dynamics using PyTorch Higher-Order Operator (while_loop)."""
    from .functional_hop import functionalize_kuramoto_model

    return functionalize_kuramoto_model(
        model=model,
        solver_name=solver_name,
        num_steps=num_steps,
        dt=dt,
        integration_time=integration_time,
        use_hop=True,
    )


@register_compiler_rewrite("unrolled_functional")
def _unrolled_functional_rewrite(
    model: nn.Module,
    solver_name: str = "rk4",
    num_steps: int = 25,
    dt: float = 0.04,
    integration_time: float = 1.0,
    target_device: str = "mps",
    **kwargs,
) -> nn.Module:
    """Functionalize dynamics using static unrolled steps without Python graph breaks."""
    from .functional_hop import functionalize_kuramoto_model

    return functionalize_kuramoto_model(
        model=model,
        solver_name=solver_name,
        num_steps=num_steps,
        dt=dt,
        integration_time=integration_time,
        use_hop=False,
    )


__all__ = [
    "COMPILER_REGISTRY",
    "register_compiler_rewrite",
    "get_compiler_rewrite",
]
