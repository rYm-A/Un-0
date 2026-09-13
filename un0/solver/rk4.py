from __future__ import annotations

from typing import Callable
import torch
from torch import Tensor
from . import register_solver


@register_solver("rk4")
def rk4_step(
    rhs: Callable[[Tensor, Tensor, Tensor], Tensor],
    state: Tensor,
    time_grid: Tensor,
    drive: Tensor,
    **kwargs,
) -> Tensor:
    """Explicit fourth-order Runge-Kutta ODE solver."""
    trajectory = [state]
    current_state = state
    num_steps = len(time_grid) - 1
    for i in range(num_steps):
        t = time_grid[i]
        t_next = time_grid[i + 1]
        dt = t_next - t
        half_dt = dt * 0.5
        t_mid = t + half_dt

        k1 = rhs(current_state, t, drive)
        k2 = rhs(current_state + half_dt * k1, t_mid, drive)
        k3 = rhs(current_state + half_dt * k2, t_mid, drive)
        k4 = rhs(current_state + dt * k3, t_next, drive)

        current_state = current_state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        trajectory.append(current_state)
    return torch.stack(trajectory, dim=0)
