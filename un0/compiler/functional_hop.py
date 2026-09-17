"""PyTorch Higher-Order Operators (HOP) functionalization for Kuramoto models.

Rewrites the continuous Kuramoto dynamics integration loop into a functional form
using PyTorch HOPs (e.g., torch._higher_order_ops.while_loop) compatible with
Torch Dynamo, AOTAutograd, and Inductor, supporting dynamic batch sizes >= 1.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import torch
from torch import Tensor, nn

logger = logging.getLogger("un0.compiler.functional_hop")

# Check HOP availability
try:
    from torch._higher_order_ops.while_loop import while_loop
    _HOP_WHILE_LOOP_AVAILABLE = True
except ImportError:
    while_loop = None
    _HOP_WHILE_LOOP_AVAILABLE = False


class FunctionalHOPKuramotoDynamics(nn.Module):
    """Functionalized Kuramoto model using PyTorch HOPs for AOT compilation.

    Replaces dynamic Python while loops and external ODE integrators (such as
    torchdiffeq.odeint) with a functionalized loop based on PyTorch's
    while_loop higher-order operator. This enables whole-graph capture
    by TorchDynamo and codegen via TorchInductor with dynamic batch shapes (B >= 1).
    """

    def __init__(
        self,
        dynamics: nn.Module | Callable,
        solver_name: str = "rk4",
        num_steps: int = 25,
        dt: float = 0.04,
        integration_time: float = 1.0,
        use_hop: bool = True,
    ) -> None:
        """Initialize HOP Kuramoto dynamics wrapper.

        Args:
            dynamics: Base Kuramoto dynamics module or noise-wrapped dynamics.
            solver_name: Solver algorithm name ('rk4', 'euler', etc.).
            num_steps: Number of integration steps.
            dt: Time step size. If None or 0, derived from integration_time / num_steps.
            integration_time: Total integration horizon.
            use_hop: Whether to use torch._higher_order_ops.while_loop if available.
        """
        super().__init__()
        self.dynamics = dynamics
        self.solver_name = str(solver_name).lower()
        self.num_steps = max(1, int(num_steps))
        if dt and dt > 0:
            self.dt = float(dt)
        else:
            self.dt = float(integration_time) / self.num_steps
        self.integration_time = float(integration_time)
        self.use_hop = bool(use_hop and _HOP_WHILE_LOOP_AVAILABLE)

        # Expose attributes of underlying dynamics (n, n_cond, state_dim, num_classes, etc.)
        for attr in ("n", "n_cond", "state_dim", "num_classes", "omega", "K", "K_drive"):
            curr = dynamics
            while curr is not None:
                if hasattr(curr, attr):
                    setattr(self, attr, getattr(curr, attr))
                    break
                curr = getattr(curr, "dynamics", None)

    def _step_euler(
        self,
        state: Tensor,
        t: Tensor,
        dt_val: Tensor,
        drive: Tensor,
    ) -> Tensor:
        """Single explicit Euler step."""
        vel = self.dynamics(state, t, drive)
        return state + dt_val * vel

    def _step_rk4(
        self,
        state: Tensor,
        t: Tensor,
        dt_val: Tensor,
        drive: Tensor,
    ) -> Tensor:
        """Single explicit 4th-order Runge-Kutta step."""
        half_dt = dt_val * 0.5
        t_mid = t + half_dt
        t_next = t + dt_val

        k1 = self.dynamics(state, t, drive)
        k2 = self.dynamics(state + half_dt * k1, t_mid, drive)
        k3 = self.dynamics(state + half_dt * k2, t_mid, drive)
        k4 = self.dynamics(state + dt_val * k3, t_next, drive)

        return state + (dt_val / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _integrate_hop(self, initial_state: Tensor, drive: Tensor) -> Tensor:
        """Integrate dynamics using PyTorch while_loop HOP."""
        max_steps = torch.tensor(self.num_steps, dtype=torch.int64, device=initial_state.device)
        dt_t = torch.tensor(self.dt, dtype=initial_state.dtype, device=initial_state.device)
        step_0 = torch.tensor(0, dtype=torch.int64, device=initial_state.device)

        def cond_fn(step, s, max_s, dt_val):
            return step < max_s

        if self.solver_name == "euler":
            def body_fn(step, s, max_s, dt_val):
                t = step.to(dt_val.dtype) * dt_val
                new_s = self._step_euler(s, t, dt_val, drive)
                # Clone unchanged loop constants to avoid input-to-output aliasing in HOP tracer
                return step + 1, new_s, max_s.clone(), dt_val.clone()
        else:
            def body_fn(step, s, max_s, dt_val):
                t = step.to(dt_val.dtype) * dt_val
                new_s = self._step_rk4(s, t, dt_val, drive)
                # Clone unchanged loop constants to avoid input-to-output aliasing in HOP tracer
                return step + 1, new_s, max_s.clone(), dt_val.clone()

        result = while_loop(cond_fn, body_fn, (step_0, initial_state, max_steps, dt_t))
        return result[1]

    def _integrate_functional_unroll(self, initial_state: Tensor, drive: Tensor) -> Tensor:
        """Fallback functional unrolled integration loop for eager / non-HOP backends."""
        state = initial_state
        dt_val = torch.tensor(self.dt, dtype=initial_state.dtype, device=initial_state.device)
        for i in range(self.num_steps):
            t = torch.tensor(i * self.dt, dtype=initial_state.dtype, device=initial_state.device)
            if self.solver_name == "euler":
                state = self._step_euler(state, t, dt_val, drive)
            else:
                state = self._step_rk4(state, t, dt_val, drive)
        return state

    def forward(self, state: Tensor, drive: Optional[Tensor] = None) -> Tensor:
        """Compute integrated phase trajectory ending state.

        Supports dynamic batch size B >= 1 without upper bound.

        Args:
            state: Initial phases tensor of shape `(batch, state_dim)`.
            drive: Optional class drive tensor of shape `(batch, n, n_cond)`.

        Returns:
            Final state tensor of shape `(batch, state_dim)`.
        """
        if drive is None:
            n_osc = getattr(self, "n", 16)
            n_cond = getattr(self, "n_cond", 4)
            drive = torch.zeros(state.shape[0], n_osc, n_cond, device=state.device, dtype=state.dtype)

        if self.use_hop and _HOP_WHILE_LOOP_AVAILABLE:
            try:
                return self._integrate_hop(state, drive)
            except Exception as hop_err:
                logger.debug("HOP execution fell back to functional unroll: %s", hop_err)
                return self._integrate_functional_unroll(state, drive)
        return self._integrate_functional_unroll(state, drive)


def functionalize_kuramoto_model(
    model: nn.Module,
    solver_name: str = "rk4",
    num_steps: int = 25,
    dt: float = 0.04,
    integration_time: float = 1.0,
    use_hop: bool = True,
) -> FunctionalHOPKuramotoDynamics:
    """Create a functionalized HOP Kuramoto dynamics model for compilation."""
    return FunctionalHOPKuramotoDynamics(
        dynamics=model,
        solver_name=solver_name,
        num_steps=num_steps,
        dt=dt,
        integration_time=integration_time,
        use_hop=use_hop,
    )
