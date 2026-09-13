from __future__ import annotations

from typing import Callable
import torch
from torch import Tensor
from . import register_solver


@register_solver("euler")
def euler_step(
    rhs: Callable[[Tensor, Tensor, Tensor], Tensor],
    state: Tensor,
    time_grid: Tensor,
    drive: Tensor,
    **kwargs,
) -> Tensor:
    """Explicit first-order Euler ODE solver."""
    trajectory = [state]
    current_state = state
    num_steps = len(time_grid) - 1
    for i in range(num_steps):
        t = time_grid[i]
        dt = time_grid[i + 1] - t
        vel = rhs(current_state, t, drive)
        current_state = current_state + dt * vel
        trajectory.append(current_state)
    return torch.stack(trajectory, dim=0)
