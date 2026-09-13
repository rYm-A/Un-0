from __future__ import annotations

from typing import Callable
import torch
from torch import Tensor
from . import register_solver


@register_solver("euler_backward")
def euler_backward_step(
    rhs: Callable[[Tensor, Tensor, Tensor], Tensor],
    state: Tensor,
    time_grid: Tensor,
    drive: Tensor,
    max_iter: int = 5,
    **kwargs,
) -> Tensor:
    """Implicit (backward) Euler ODE solver via fixed-point iteration."""
    trajectory = [state]
    current_state = state
    num_steps = len(time_grid) - 1
    for i in range(num_steps):
        t_next = time_grid[i + 1]
        dt = t_next - time_grid[i]
        # Predictor: explicit Euler step as initial guess
        next_state = current_state + dt * rhs(current_state, time_grid[i], drive)
        # Fixed-point iteration
        for _ in range(max_iter):
            next_state = current_state + dt * rhs(next_state, t_next, drive)
        current_state = next_state
        trajectory.append(current_state)
    return torch.stack(trajectory, dim=0)
