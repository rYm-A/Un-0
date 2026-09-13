from __future__ import annotations

from typing import Any, Callable, Dict

NOISE_REGISTRY: Dict[str, Any] = {}


def register_noise_model(name: str | Callable | None = None) -> Callable:
    """Decorator to register a noise model wrapper in NOISE_REGISTRY."""
    if callable(name):
        fn = name
        noise_name = fn.__name__
        NOISE_REGISTRY[noise_name] = fn
        return fn

    def decorator(fn: Callable) -> Callable:
        noise_name = name if name is not None else fn.__name__
        NOISE_REGISTRY[noise_name] = fn
        return fn

    return decorator


def get_noise_model(name: str) -> Any:
    """Lookup a registered noise model by name."""
    if name not in NOISE_REGISTRY:
        raise KeyError(
            f"Noise model {name!r} not found in registry. Available noise models: {list(NOISE_REGISTRY.keys())}"
        )
    return NOISE_REGISTRY[name]


# Import noise modules to trigger registration
from . import L0_static_mismatch, L1_stochastic_parameter_noise, L2_functional_interface, L3_correlated_drift

__all__ = ["NOISE_REGISTRY", "register_noise_model", "get_noise_model"]
