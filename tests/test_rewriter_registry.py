from __future__ import annotations

import pytest
import torch
from torch import nn

from un0.compiler import compile_model
from un0.model import ConditionalKuramotoDynamics
from un0.noise import (
    NOISE_REGISTRY,
    get_noise_model,
    register_noise_model,
)
from un0.solver import (
    SOLVER_REGISTRY,
    get_solver,
    register_solver,
)


# --- Tests for Solver Registry ---


def test_solver_registry_contains_builtins() -> None:
    expected = {"euler", "rk4", "euler_backward", "parareal", "par_ode"}
    assert expected.issubset(set(SOLVER_REGISTRY.keys()))


def test_solver_registration_and_lookup() -> None:
    @register_solver("test_dummy_solver")
    def dummy_solver(rhs, state, time_grid, drive):
        return torch.stack([state] * len(time_grid), dim=0)

    solver = get_solver("test_dummy_solver")
    assert solver is dummy_solver

    with pytest.raises(KeyError, match="not found in registry"):
        get_solver("non_existent_solver")


@pytest.mark.parametrize("solver_name", ["euler", "rk4", "euler_backward", "parareal", "par_ode"])
def test_solver_execution(solver_name: str) -> None:
    dynamics = ConditionalKuramotoDynamics(
        n_oscillators=4,
        n_conditional_oscillators=2,
        num_classes=2,
    )
    batch_size = 3
    state = torch.randn(batch_size, dynamics.state_dim)
    drive = torch.randn(batch_size, dynamics.n, dynamics.n_cond)
    time_grid = torch.linspace(0.0, 1.0, 5)

    solver_fn = get_solver(solver_name)
    trajectory = solver_fn(dynamics, state, time_grid, drive)

    assert trajectory.shape == (5, batch_size, dynamics.state_dim)
    assert not torch.isnan(trajectory).any()


# --- Tests for Noise Registry ---


def test_noise_registry_contains_builtins() -> None:
    expected = {
        "L0_static_mismatch",
        "L1_stochastic_parameter_noise",
        "L2_functional_interface",
        "L3_correlated_drift",
    }
    assert expected.issubset(set(NOISE_REGISTRY.keys()))


def test_noise_registration_and_lookup() -> None:
    @register_noise_model("test_dummy_noise")
    class DummyNoise(nn.Module):
        def __init__(self, dynamics):
            super().__init__()
            self.dynamics = dynamics

        def forward(self, state, t, drive):
            return self.dynamics(state, t, drive)

    noise_cls = get_noise_model("test_dummy_noise")
    assert noise_cls is DummyNoise

    with pytest.raises(KeyError, match="not found in registry"):
        get_noise_model("non_existent_noise")


@pytest.mark.parametrize(
    "noise_name",
    [
        "L0_static_mismatch",
        "L1_stochastic_parameter_noise",
        "L2_functional_interface",
        "L3_correlated_drift",
    ],
)
def test_noise_model_execution(noise_name: str) -> None:
    dynamics = ConditionalKuramotoDynamics(
        n_oscillators=4,
        n_conditional_oscillators=2,
        num_classes=2,
    )
    batch_size = 3
    state = torch.randn(batch_size, dynamics.state_dim)
    drive = torch.randn(batch_size, dynamics.n, dynamics.n_cond)
    t = torch.tensor(0.5)

    noise_cls = get_noise_model(noise_name)
    noisy_dynamics = noise_cls(dynamics)

    vel = noisy_dynamics(state, t, drive)
    assert vel.shape == (batch_size, dynamics.state_dim)
    assert not torch.isnan(vel).any()


# --- Tests for Compiler Wrapper ---


def test_compile_model_wrapper() -> None:
    class SimpleModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    model = SimpleModule()
    compiled = compile_model(model, backend="aot_eager", dynamic=True)

    x = torch.randn(2, 4)
    out = compiled(x)
    assert out.shape == (2, 2)
