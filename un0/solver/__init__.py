from __future__ import annotations

from typing import Any, Callable, Dict

SOLVER_REGISTRY: Dict[str, Callable] = {}


def register_solver(name: str | Callable | None = None) -> Callable:
    """Decorator to register an ODE solver function in SOLVER_REGISTRY."""
    if callable(name):
        fn = name
        solver_name = fn.__name__
        SOLVER_REGISTRY[solver_name] = fn
        return fn

    def decorator(fn: Callable) -> Callable:
        solver_name = name if name is not None else fn.__name__
        SOLVER_REGISTRY[solver_name] = fn
        return fn

    return decorator


def get_solver(name: str) -> Callable:
    """Lookup a registered solver function by name."""
    if name not in SOLVER_REGISTRY:
        raise KeyError(
            f"Solver {name!r} not found in registry. Available solvers: {list(SOLVER_REGISTRY.keys())}"
        )
    return SOLVER_REGISTRY[name]


# Import solver modules to trigger registration
from . import euler, euler_backward, par_ode, parareal, rk4

__all__ = ["SOLVER_REGISTRY", "register_solver", "get_solver"]
