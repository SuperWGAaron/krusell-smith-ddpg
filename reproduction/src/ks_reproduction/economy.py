"""Economic equations shared by the DP and PyTorch DDPG implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .config import ALMCoefficients, CompatibilityConfig, EconomyConfig, GridConfig

BAD = 0
GOOD = 1
UNEMPLOYED = 0
EMPLOYED = 1

_DEFAULT_ECONOMY = EconomyConfig()
_DEFAULT_COMPATIBILITY = CompatibilityConfig()
_DEFAULT_GRID = GridConfig()


def _is_tensor(value: Any) -> bool:
    return torch.is_tensor(value)


def _where(condition: Any, left: Any, right: Any) -> Any:
    if _is_tensor(condition):
        left_value = torch.as_tensor(left, dtype=condition.dtype, device=condition.device)
        right_value = torch.as_tensor(right, dtype=condition.dtype, device=condition.device)
        return torch.where(condition, left_value, right_value)
    return np.where(condition, left, right)


def _log(value: Any) -> Any:
    return torch.log(value) if _is_tensor(value) else np.log(value)


def _exp(value: Any) -> Any:
    return torch.exp(value) if _is_tensor(value) else np.exp(value)


def _maximum(value: Any, minimum: float) -> Any:
    if _is_tensor(value):
        return torch.clamp_min(value, minimum)
    return np.maximum(value, minimum)


def _regime_value(regime: Any, bad_value: float, good_value: float) -> Any:
    """Select a value under the canonical BAD=0, GOOD=1 convention."""

    if _is_tensor(regime):
        _validate_binary_tensor(regime, name="aggregate regime")
        bad = torch.as_tensor(bad_value, dtype=torch.get_default_dtype(), device=regime.device)
        good = torch.as_tensor(good_value, dtype=torch.get_default_dtype(), device=regime.device)
        return torch.where(regime == GOOD, good, bad)
    array = np.asarray(regime)
    if not np.all((array == BAD) | (array == GOOD)):
        raise ValueError("aggregate regime must contain only BAD=0 or GOOD=1")
    result = np.where(array == GOOD, good_value, bad_value)
    return result.item() if result.ndim == 0 else result


def _validate_binary_tensor(value: torch.Tensor, *, name: str) -> None:
    if not bool(torch.all((value == 0) | (value == 1)).item()):
        raise ValueError(f"{name} must contain only zero or one")


def productivity(regime: Any, config: EconomyConfig = _DEFAULT_ECONOMY) -> Any:
    return _regime_value(regime, config.productivity_bad, config.productivity_good)


def unemployment_rate(regime: Any, config: EconomyConfig = _DEFAULT_ECONOMY) -> Any:
    return _regime_value(regime, config.unemployment_bad, config.unemployment_good)


def labor_endowment_scale(
    config: EconomyConfig = _DEFAULT_ECONOMY,
    compatibility: CompatibilityConfig = _DEFAULT_COMPATIBILITY,
) -> float:
    """Return employed labor units.

    The manuscript states employed labor is one.  The retained code instead
    normalises labor by bad-state employment, ``1 / (1 - u_bad)``.  The latter
    is used only in legacy-faithful mode.
    """

    if compatibility.active("use_normalized_labor_endowment"):
        return 1.0 / (1.0 - config.unemployment_bad)
    return 1.0


def aggregate_labor(
    regime: Any,
    config: EconomyConfig = _DEFAULT_ECONOMY,
    compatibility: CompatibilityConfig = _DEFAULT_COMPATIBILITY,
) -> Any:
    scale = labor_endowment_scale(config, compatibility)
    return scale * (1.0 - unemployment_rate(regime, config))


def effective_individual_labor(
    employment: Any,
    config: EconomyConfig = _DEFAULT_ECONOMY,
    compatibility: CompatibilityConfig = _DEFAULT_COMPATIBILITY,
) -> Any:
    """Labor-equivalent income entering the household budget constraint."""

    scale = labor_endowment_scale(config, compatibility)
    if _is_tensor(employment):
        _validate_binary_tensor(employment, name="employment")
    else:
        array = np.asarray(employment)
        if not np.all((array == UNEMPLOYED) | (array == EMPLOYED)):
            raise ValueError("employment must contain only UNEMPLOYED=0 or EMPLOYED=1")
    return employment * scale + (1.0 - employment) * config.unemployment_income


def production(
    aggregate_capital: Any,
    labor: Any,
    technology: Any,
    alpha: float = 0.36,
) -> Any:
    """Cobb--Douglas output ``z K^alpha L^(1-alpha)``."""

    return technology * aggregate_capital**alpha * labor ** (1.0 - alpha)


def interest_rate(
    aggregate_capital: Any,
    labor: Any,
    technology: Any,
    alpha: float = 0.36,
    *,
    minimum_capital: float = 1.0e-12,
) -> Any:
    """Competitive rental rate of capital."""

    capital = _maximum(aggregate_capital, minimum_capital)
    return alpha * technology * (labor / capital) ** (1.0 - alpha)


def wage_rate(
    aggregate_capital: Any,
    labor: Any,
    technology: Any,
    alpha: float = 0.36,
) -> Any:
    """Competitive wage rate."""

    return (1.0 - alpha) * technology * (aggregate_capital / labor) ** alpha


def factor_prices(
    aggregate_capital: Any,
    regime: Any,
    config: EconomyConfig = _DEFAULT_ECONOMY,
    compatibility: CompatibilityConfig = _DEFAULT_COMPATIBILITY,
) -> tuple[Any, Any]:
    technology = productivity(regime, config)
    labor = aggregate_labor(regime, config, compatibility)
    return (
        interest_rate(aggregate_capital, labor, technology, config.alpha),
        wage_rate(aggregate_capital, labor, technology, config.alpha),
    )


def available_resources(
    individual_capital: Any,
    aggregate_capital: Any,
    regime: Any,
    employment: Any,
    config: EconomyConfig = _DEFAULT_ECONOMY,
    compatibility: CompatibilityConfig = _DEFAULT_COMPATIBILITY,
) -> Any:
    """Resources available for consumption and next-period capital."""

    interest, wage = factor_prices(aggregate_capital, regime, config, compatibility)
    labor_income = effective_individual_labor(employment, config, compatibility)
    return (interest + 1.0 - config.delta) * individual_capital + wage * labor_income


def next_capital_from_action(individual_capital: Any, action: Any) -> Any:
    """Map the DDPG action ratio ``a=k'/k`` to next-period capital."""

    return individual_capital * action


def consumption(resources: Any, next_capital: Any) -> Any:
    return resources - next_capital


def crra_utility(consumption_value: Any, risk_aversion: float = 1.0) -> Any:
    """CRRA utility, including log utility at unit risk aversion."""

    if risk_aversion < 0.0:
        raise ValueError("risk_aversion must be non-negative")
    if math_isclose(risk_aversion, 1.0):
        return _log(consumption_value)
    return (consumption_value ** (1.0 - risk_aversion) - 1.0) / (1.0 - risk_aversion)


def math_isclose(left: float, right: float, *, tolerance: float = 1.0e-12) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def safe_reward(
    consumption_value: Any,
    config: EconomyConfig = _DEFAULT_ECONOMY,
    *,
    feasible: Any | None = None,
) -> Any:
    """Utility with the thesis's finite punishment for infeasible choices."""

    positive = consumption_value > config.minimum_positive
    feasible_mask = positive if feasible is None else (positive & feasible)
    safe_consumption = _maximum(consumption_value, config.minimum_positive)
    utility = crra_utility(safe_consumption, config.risk_aversion)
    if _is_tensor(utility):
        punishment = torch.as_tensor(
            config.invalid_reward,
            dtype=utility.dtype,
            device=utility.device,
        )
        return torch.where(feasible_mask, utility, punishment)
    return np.where(feasible_mask, utility, config.invalid_reward)


def next_aggregate_capital(
    aggregate_capital: Any,
    regime: Any,
    coefficients: ALMCoefficients,
) -> Any:
    """Apply the named log-linear aggregate law of motion."""

    intercept = _regime_value(regime, coefficients.bad_intercept, coefficients.good_intercept)
    slope = _regime_value(regime, coefficients.bad_slope, coefficients.good_slope)
    return _exp(intercept + slope * _log(aggregate_capital))


@dataclass(frozen=True)
class HouseholdStep:
    next_individual_capital: Any
    next_aggregate_capital: Any
    resources: Any
    consumption: Any
    reward: Any
    feasible: Any


def household_step(
    individual_capital: Any,
    aggregate_capital: Any,
    regime: Any,
    employment: Any,
    action: Any,
    coefficients: ALMCoefficients,
    config: EconomyConfig = _DEFAULT_ECONOMY,
    compatibility: CompatibilityConfig = _DEFAULT_COMPATIBILITY,
) -> HouseholdStep:
    next_individual = next_capital_from_action(individual_capital, action)
    resources = available_resources(
        individual_capital,
        aggregate_capital,
        regime,
        employment,
        config,
        compatibility,
    )
    current_consumption = consumption(resources, next_individual)
    feasible = (next_individual > 0.0) & (next_individual < resources)
    reward = safe_reward(current_consumption, config, feasible=feasible)
    next_aggregate = next_aggregate_capital(aggregate_capital, regime, coefficients)
    return HouseholdStep(
        next_individual_capital=next_individual,
        next_aggregate_capital=next_aggregate,
        resources=resources,
        consumption=current_consumption,
        reward=reward,
        feasible=feasible & (current_consumption > config.minimum_positive),
    )


def discount_factors(beta: float, horizon: int, *, like: Any | None = None) -> Any:
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must lie in (0, 1)")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if _is_tensor(like):
        powers = torch.arange(horizon, dtype=like.dtype, device=like.device)
        return torch.as_tensor(beta, dtype=like.dtype, device=like.device) ** powers
    return np.power(beta, np.arange(horizon, dtype=np.float64))


def discounted_sum(rewards: Any, beta: float, *, axis: int = -1) -> Any:
    horizon = rewards.shape[axis]
    discounts = discount_factors(beta, horizon, like=rewards)
    shape = [1] * rewards.ndim
    shape[axis] = horizon
    discounts = discounts.reshape(shape)
    if _is_tensor(rewards):
        return torch.sum(rewards * discounts, dim=axis)
    return np.sum(rewards * discounts, axis=axis)


def individual_capital_grid(config: GridConfig = _DEFAULT_GRID) -> np.ndarray:
    unit = np.arange(config.individual_size, dtype=np.float64) / (config.individual_size - 1)
    grid = config.individual_min + (
        config.individual_max - config.individual_min
    ) * unit**config.individual_curvature
    grid[0], grid[-1] = config.individual_min, config.individual_max
    return grid


def aggregate_capital_grid(config: GridConfig = _DEFAULT_GRID) -> np.ndarray:
    return np.linspace(config.aggregate_min, config.aggregate_max, config.aggregate_size)


__all__ = [
    "BAD",
    "EMPLOYED",
    "GOOD",
    "HouseholdStep",
    "UNEMPLOYED",
    "aggregate_capital_grid",
    "aggregate_labor",
    "available_resources",
    "consumption",
    "crra_utility",
    "discount_factors",
    "discounted_sum",
    "effective_individual_labor",
    "factor_prices",
    "household_step",
    "individual_capital_grid",
    "interest_rate",
    "labor_endowment_scale",
    "next_aggregate_capital",
    "next_capital_from_action",
    "production",
    "productivity",
    "safe_reward",
    "unemployment_rate",
    "wage_rate",
]
