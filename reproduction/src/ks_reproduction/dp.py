"""PyTorch value-function benchmark and loader for the published Julia cache."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .interpolation import bilinear

# The Julia cache uses (good, employed), (bad, employed),
# (good, unemployed), (bad, unemployed).  The package canonical order is
# (bad, unemployed), (bad, employed), (good, unemployed), (good, employed).
CANONICAL_TO_JULIA = torch.tensor([3, 1, 2, 0], dtype=torch.long)


def capital_grid(
    size: int = 100,
    minimum: float = 0.1,
    maximum: float = 1000.0,
    degree: int = 7,
    *,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    unit = torch.linspace(0.0, 1.0, size, dtype=dtype)
    result = minimum + (maximum - minimum) * unit.pow(degree)
    result[0], result[-1] = minimum, maximum
    return result


def aggregate_capital_grid(
    size: int = 20,
    minimum: float = 0.1,
    maximum: float = 50.0,
    *,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    return torch.linspace(minimum, maximum, size, dtype=dtype)


@dataclass(frozen=True)
class DPReference:
    policy: Tensor  # [individual capital, aggregate capital, canonical shock]
    value: Tensor
    alm_good: tuple[float, float]
    alm_bad: tuple[float, float]
    r2_good: float
    r2_bad: float
    k_grid: Tensor
    K_grid: Tensor

    @classmethod
    def load(cls, directory: str | Path) -> DPReference:
        directory = Path(directory)
        arrays = {
            "policy": np.load(directory / "kss_k_opt.npy"),
            "value": np.load(directory / "kss_value.npy"),
            "B": np.load(directory / "B.npy"),
            "R2": np.load(directory / "R2.npy"),
        }
        if arrays["policy"].shape != (100, 20, 4):
            raise ValueError(f"unexpected DP policy shape: {arrays['policy'].shape}")
        if arrays["value"].shape != arrays["policy"].shape:
            raise ValueError("DP value and policy arrays must have identical shapes")
        # Convert the cache once so every downstream module uses named/canonical regimes.
        order = CANONICAL_TO_JULIA.numpy()
        policy = torch.from_numpy(
            np.ascontiguousarray(arrays["policy"][:, :, order])
        ).to(torch.float64)
        value = torch.from_numpy(
            np.ascontiguousarray(arrays["value"][:, :, order])
        ).to(torch.float64)
        good_a, good_b, bad_a, bad_b = (float(item) for item in arrays["B"])
        r2_good, r2_bad = (float(item) for item in arrays["R2"])
        return cls(
            policy=policy,
            value=value,
            alm_good=(good_a, good_b),
            alm_bad=(bad_a, bad_b),
            r2_good=r2_good,
            r2_bad=r2_bad,
            k_grid=capital_grid(),
            K_grid=aggregate_capital_grid(),
        )

    def next_capital(self, states: Tensor) -> Tensor:
        """Evaluate the cached policy for states ``(k, K, z, epsilon)``."""

        states = states.to(dtype=torch.float64, device="cpu")
        if states.ndim != 2 or states.shape[1] != 4:
            raise ValueError("states must have shape [batch, 4]")
        shock = (2 * states[:, 2].long() + states[:, 3].long()).clamp(0, 3)
        result = torch.empty(states.shape[0], dtype=torch.float64)
        for index in range(4):
            mask = shock == index
            if mask.any():
                result[mask] = bilinear(
                    self.k_grid,
                    self.K_grid,
                    self.policy[:, :, index],
                    states[mask, 0],
                    states[mask, 1],
                )
        return result

    def saving_ratio(self, states: Tensor) -> Tensor:
        capital = states[:, 0].to(torch.float64).clamp_min(1e-12)
        return (self.next_capital(states) / capital).unsqueeze(1)


@dataclass
class VFISolution:
    value: Tensor
    policy: Tensor
    iterations: int
    error: float


class TorchVFISolver:
    """A tested, discrete-action Bellman solver for a fixed aggregate law.

    It intentionally avoids the broken legacy Python solver.  The full thesis
    reproduction uses the archived Julia benchmark by default; this class makes
    the benchmark recomputable and useful for reduced-grid tests.
    """

    def __init__(
        self,
        *,
        k_grid: Tensor,
        K_grid: Tensor,
        transition: Tensor,
        z_values: tuple[float, float] = (0.99, 1.01),
        unemployment: tuple[float, float] = (0.10, 0.04),
        alpha: float = 0.36,
        beta: float = 0.8,
        delta: float = 0.025,
        labor_endowment: float = 1.0 / 0.9,
        unemployment_income: float = 0.0,
        alm_bad: tuple[float, float] = (0.0, 1.0),
        alm_good: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self.k_grid = k_grid.to(torch.float64)
        self.K_grid = K_grid.to(torch.float64)
        self.transition = transition.to(torch.float64)
        if self.transition.shape != (4, 4):
            raise ValueError("joint transition matrix must be 4 by 4")
        self.z_values = z_values
        self.unemployment = unemployment
        self.alpha, self.beta, self.delta = alpha, beta, delta
        self.labor_endowment = labor_endowment
        self.unemployment_income = unemployment_income
        self.alm_bad, self.alm_good = alm_bad, alm_good

    def _aggregate_next(self, K: Tensor, z_index: int) -> Tensor:
        intercept, slope = self.alm_bad if z_index == 0 else self.alm_good
        return torch.exp(intercept + slope * torch.log(K)).clamp(self.K_grid[0], self.K_grid[-1])

    def _wealth(self, k: Tensor, K: Tensor, z_index: int, employed: int) -> Tensor:
        z = self.z_values[z_index]
        labor = self.labor_endowment * (1.0 - self.unemployment[z_index])
        interest = self.alpha * z * (labor / K.clamp_min(1e-12)).pow(1.0 - self.alpha)
        wage = (1.0 - self.alpha) * z * (K / labor).pow(self.alpha)
        labor_income = wage * (
            employed * self.labor_endowment + (1 - employed) * self.unemployment_income
        )
        return (interest + 1.0 - self.delta) * k + labor_income

    def solve(
        self,
        *,
        tolerance: float = 1e-5,
        max_iterations: int = 10_000,
        initial_value: Tensor | None = None,
    ) -> VFISolution:
        nk, nK = self.k_grid.numel(), self.K_grid.numel()
        value = (
            torch.zeros((nk, nK, 4), dtype=torch.float64)
            if initial_value is None
            else initial_value.clone().to(torch.float64)
        )
        policy = torch.empty_like(value)
        error = float("inf")
        for iteration in range(1, max_iterations + 1):
            old = value.clone()
            updated = torch.empty_like(value)
            for aggregate_index, K in enumerate(self.K_grid):
                for shock in range(4):
                    z_index, employed = divmod(shock, 2)
                    K_next = self._aggregate_next(K, z_index)
                    continuation = torch.zeros(nk, dtype=torch.float64)
                    for next_shock in range(4):
                        continuation += self.transition[shock, next_shock] * bilinear(
                            self.k_grid,
                            self.K_grid,
                            old[:, :, next_shock],
                            self.k_grid,
                            K_next.expand_as(self.k_grid),
                        )
                    wealth = self._wealth(self.k_grid, K, z_index, employed)
                    consumption = wealth[:, None] - self.k_grid[None, :]
                    feasible = consumption > 0.0
                    objective = torch.where(
                        feasible,
                        torch.log(consumption.clamp_min(torch.finfo(torch.float64).tiny))
                        + self.beta * continuation[None, :],
                        torch.full_like(consumption, -torch.inf),
                    )
                    updated[:, aggregate_index, shock], choices = objective.max(dim=1)
                    policy[:, aggregate_index, shock] = self.k_grid[choices]
            error = float((updated - old).abs().max())
            value = updated
            if error < tolerance:
                return VFISolution(value, policy, iteration, error)
        return VFISolution(value, policy, max_iterations, error)


def dp_metadata(reference: DPReference) -> dict[str, Any]:
    return {
        "policy_shape": list(reference.policy.shape),
        "value_shape": list(reference.value.shape),
        "alm": {
            "good": list(reference.alm_good),
            "bad": list(reference.alm_bad),
        },
        "r2": {"good": reference.r2_good, "bad": reference.r2_bad},
    }
