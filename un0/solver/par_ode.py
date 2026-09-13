from __future__ import annotations

from typing import Callable
import torch
from torch import Tensor
from . import register_solver


@register_solver("par_ode")
def par_ode_step(
    rhs: Callable[[Tensor, Tensor, Tensor], Tensor],
    state: Tensor,
    time_grid: Tensor,
    drive: Tensor,
    num_iterations: int = 4,
    **kwargs,
) -> Tensor:
    """Parallelized ODE integration scheme across time steps."""
    num_steps = len(time_grid) - 1
    trajectory = torch.stack([state] * (num_steps + 1), dim=0)
    dts = time_grid[1:] - time_grid[:-1]  # shape (num_steps,)

    for _ in range(num_iterations):
        curr_states = trajectory[:-1]  # shape (num_steps, batch, dim)
        times = time_grid[:-1]          # shape (num_steps,)

        rates = torch.stack([
            rhs(curr_states[i], times[i], drive) for i in range(num_steps)
        ], dim=0)

        integrated = torch.cumsum(rates * dts.view(-1, 1, 1), dim=0)
        next_trajectory = [state]
        for i in range(num_steps):
            next_trajectory.append(state + integrated[i])
        trajectory = torch.stack(next_trajectory, dim=0)

    return trajectory
