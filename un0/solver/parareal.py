from __future__ import annotations

from typing import Callable
import torch
from torch import Tensor
from . import register_solver


@register_solver("parareal")
def parareal_step(
    rhs: Callable[[Tensor, Tensor, Tensor], Tensor],
    state: Tensor,
    time_grid: Tensor,
    drive: Tensor,
    num_iterations: int = 2,
    **kwargs,
) -> Tensor:
    """Parallel-in-time Parareal predictor-corrector solver algorithm."""
    num_steps = len(time_grid) - 1
    if num_steps <= 1:
        trajectory = [state]
        curr = state
        for i in range(num_steps):
            dt = time_grid[i + 1] - time_grid[i]
            curr = curr + dt * rhs(curr, time_grid[i], drive)
            trajectory.append(curr)
        return torch.stack(trajectory, dim=0)

    # Coarse solver G (1-step coarse Euler between grid points)
    def G(x: Tensor, t_start: Tensor, t_end: Tensor) -> Tensor:
        dt = t_end - t_start
        return x + dt * rhs(x, t_start, drive)

    # Fine solver F (2-substep RK2 between grid points)
    def F(x: Tensor, t_start: Tensor, t_end: Tensor) -> Tensor:
        dt = (t_end - t_start) * 0.5
        t_mid = t_start + dt
        k1 = x + dt * rhs(x, t_start, drive)
        x_mid = x + dt * rhs(k1, t_start, drive)
        k2 = x_mid + dt * rhs(x_mid, t_mid, drive)
        return x_mid + dt * rhs(k2, t_mid, drive)

    # Initial coarse prediction X^0 across all time steps
    X = [state]
    for i in range(num_steps):
        X.append(G(X[i], time_grid[i], time_grid[i + 1]))

    # Parareal iterations
    for _iter in range(num_iterations):
        X_next = [state]
        fine_results = [F(X[i], time_grid[i], time_grid[i + 1]) for i in range(num_steps)]
        for i in range(num_steps):
            g_next = G(X_next[i], time_grid[i], time_grid[i + 1])
            g_old = G(X[i], time_grid[i], time_grid[i + 1])
            corrected = g_next + fine_results[i] - g_old
            X_next.append(corrected)
        X = X_next

    return torch.stack(X, dim=0)
