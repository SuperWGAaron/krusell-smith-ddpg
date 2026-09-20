"""Representative-agent environment and vectorized population simulations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor

from .config import ALMCoefficients, ReproductionConfig
from .economy import (
    available_resources,
    crra_utility,
    discounted_sum,
    household_step,
    next_aggregate_capital,
)
from .transitions import (
    TransitionMatrices,
    build_transition_matrices,
    sample_aggregate_path,
    sample_employment_paths,
    sample_joint_next_state,
)


class SavingRatioPolicy(Protocol):
    def __call__(self, states: Tensor) -> Tensor: ...


def _module_device_dtype(module: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
    parameter = next(module.parameters(), None)
    if parameter is None:
        return torch.device("cpu"), torch.float32
    return parameter.device, parameter.dtype


class ActorPolicy:
    """No-grad, batched policy adapter for an Actor module."""

    def __init__(self, actor: torch.nn.Module) -> None:
        self.actor = actor

    def __call__(self, states: Tensor) -> Tensor:
        device, dtype = _module_device_dtype(self.actor)
        was_training = self.actor.training
        self.actor.eval()
        try:
            with torch.no_grad():
                return self.actor(states.to(device=device, dtype=dtype)).detach().cpu()
        finally:
            self.actor.train(was_training)


class EnsemblePolicy:
    """Equal-weight function-space average used by thesis fictitious play."""

    def __init__(self, policies: Sequence[Callable[[Tensor], Tensor]]) -> None:
        if not policies:
            raise ValueError("an ensemble needs at least one policy")
        self.policies = tuple(policies)

    def __call__(self, states: Tensor) -> Tensor:
        predictions = [
            torch.as_tensor(policy(states), dtype=torch.float64)
            for policy in self.policies
        ]
        return torch.stack(predictions, dim=0).mean(dim=0)


def policy_saving_ratio(policy: Any, states: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(states, dtype=torch.float32)
    if hasattr(policy, "saving_ratio"):
        result = policy.saving_ratio(tensor)
    elif hasattr(policy, "select_action"):
        try:
            result = policy.select_action(tensor, explore=False)
        except TypeError:
            result = policy.select_action(tensor, add_noise=False)
    elif hasattr(policy, "act"):
        try:
            result = policy.act(tensor, explore=False)
        except TypeError:
            result = policy.act(tensor)
    elif callable(policy):
        result = policy(tensor)
    else:
        raise TypeError("policy must be callable or expose saving_ratio/select_action/act")
    values = torch.as_tensor(result).detach().cpu().numpy()
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.shape != (states.shape[0],):
        raise ValueError(f"policy returned shape {values.shape}, expected {(states.shape[0],)}")
    return values.astype(np.float64, copy=False)


@dataclass(frozen=True)
class EnvironmentStep:
    state: np.ndarray
    reward: float
    terminated: bool
    info: dict[str, float | int | bool]


class KrusellSmithEnvironment:
    """Single-household DDPG environment with no global random state."""

    def __init__(
        self,
        config: ReproductionConfig,
        rng: np.random.Generator,
        *,
        matrices: TransitionMatrices | None = None,
        alm: ALMCoefficients | None = None,
    ) -> None:
        self.config = config
        self.rng = rng
        self.matrices = matrices or build_transition_matrices(config.economy, config.transitions)
        self.alm = alm or config.initial_alm
        self.steps = 0

    def set_alm(self, coefficients: ALMCoefficients) -> None:
        self.alm = coefficients

    def reset(self) -> np.ndarray:
        self.steps = 0
        upper = 2.0 * self.config.economy.initial_aggregate_capital
        k, K = self.rng.uniform(self.config.economy.minimum_positive, upper, size=2)
        aggregate, employment = self.rng.integers(0, 2, size=2)
        return np.array([k, K, aggregate, employment], dtype=np.float32)

    def step(self, state: np.ndarray, action: float | np.ndarray) -> EnvironmentStep:
        self.steps += 1
        k, K, aggregate_raw, employment_raw = (float(value) for value in state)
        aggregate, employment = int(aggregate_raw), int(employment_raw)
        action_value = float(np.asarray(action).reshape(-1)[0])
        next_aggregate_state, next_employment = sample_joint_next_state(
            self.matrices,
            aggregate,
            employment,
            self.rng,
            self.config.compatibility,
        )
        outcome = household_step(
            k,
            K,
            aggregate,
            employment,
            action_value,
            self.alm,
            self.config.economy,
            self.config.compatibility,
        )
        feasible = bool(np.asarray(outcome.feasible).item())
        terminal_horizon = self.steps >= self.config.training.horizon
        terminated = (
            not feasible
            or terminal_horizon
            or float(outcome.next_individual_capital) <= self.config.economy.minimum_positive
            or float(outcome.consumption) <= self.config.economy.minimum_positive
        )
        reward = float(np.asarray(outcome.reward).item())
        if terminal_horizon and not self.config.compatibility.active("keep_last_period_reward"):
            reward = 0.0
        next_state = np.array(
            [
                float(outcome.next_individual_capital),
                float(outcome.next_aggregate_capital),
                int(next_aggregate_state),
                int(next_employment),
            ],
            dtype=np.float32,
        )
        return EnvironmentStep(
            state=next_state,
            reward=reward,
            terminated=terminated,
            info={
                "consumption": float(outcome.consumption),
                "resources": float(outcome.resources),
                "feasible": feasible,
                "horizon_reached": terminal_horizon,
            },
        )


@dataclass(frozen=True)
class PopulationPaths:
    individual_capital: np.ndarray
    aggregate_capital: np.ndarray
    aggregate_state: np.ndarray
    employment: np.ndarray
    resources: np.ndarray
    consumption: np.ndarray
    utility: np.ndarray
    discounted_utility: np.ndarray

    def save_dict(self) -> dict[str, np.ndarray]:
        return {
            "individual_capital": self.individual_capital,
            "aggregate_capital": self.aggregate_capital,
            "aggregate_state": self.aggregate_state,
            "employment": self.employment,
            "resources": self.resources,
            "consumption": self.consumption,
            "utility": self.utility,
            "discounted_utility": self.discounted_utility,
        }


class PopulationSimulator:
    def __init__(
        self,
        config: ReproductionConfig,
        *,
        matrices: TransitionMatrices | None = None,
    ) -> None:
        self.config = config
        self.matrices = matrices or build_transition_matrices(config.economy, config.transitions)

    def shocks(
        self,
        population: int,
        horizon: int,
        rng: np.random.Generator,
        *,
        aggregate_path: np.ndarray | None = None,
        initial_aggregate_state: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        initial_state = (
            self.config.transitions.simulation_initial_aggregate_state
            if initial_aggregate_state is None
            else initial_aggregate_state
        )
        aggregate = (
            sample_aggregate_path(
                self.matrices,
                horizon,
                rng,
                initial_state=initial_state,
            )
            if aggregate_path is None
            else np.asarray(aggregate_path, dtype=np.int64)
        )
        if aggregate.shape != (horizon,):
            raise ValueError("aggregate path has the wrong horizon")
        employment = sample_employment_paths(
            self.matrices,
            aggregate,
            population,
            rng,
            self.config.economy,
            self.config.compatibility,
        )
        return aggregate, employment

    def simulate(
        self,
        policy: Any,
        *,
        population: int,
        horizon: int,
        rng: np.random.Generator,
        aggregate_path: np.ndarray | None = None,
        employment_paths: np.ndarray | None = None,
        initial_aggregate_state: int | None = None,
    ) -> PopulationPaths:
        aggregate, generated_employment = self.shocks(
            population,
            horizon,
            rng,
            aggregate_path=aggregate_path,
            initial_aggregate_state=initial_aggregate_state,
        )
        employment = (
            generated_employment
            if employment_paths is None
            else np.asarray(employment_paths, dtype=np.int64)
        )
        if employment.shape != (population, horizon):
            raise ValueError("employment paths must have shape [population, horizon]")
        k = np.zeros((population, horizon), dtype=np.float64)
        K = np.zeros(horizon, dtype=np.float64)
        resources = np.zeros_like(k)
        consumption = np.zeros_like(k)
        utility = np.zeros_like(k)
        k[:, 0] = self.config.economy.initial_individual_capital
        K[0] = self.config.economy.initial_aggregate_capital

        last_decision = horizon if not self.config.compatibility.active(
            "leave_final_evaluation_utility_zero"
        ) else horizon - 1
        for time in range(last_decision):
            states = np.column_stack(
                (
                    k[:, time],
                    np.full(population, K[time]),
                    np.full(population, aggregate[time]),
                    employment[:, time],
                )
            ).astype(np.float32)
            ratio = policy_saving_ratio(policy, states)
            next_k = k[:, time] * ratio
            current_resources = available_resources(
                k[:, time],
                K[time],
                aggregate[time],
                employment[:, time],
                self.config.economy,
                self.config.compatibility,
            )
            current_consumption = np.asarray(current_resources) - next_k
            current_utility = np.full(population, np.nan, dtype=np.float64)
            valid = current_consumption > 0.0
            current_utility[valid] = crra_utility(
                current_consumption[valid], self.config.economy.risk_aversion
            )
            resources[:, time] = current_resources
            consumption[:, time] = current_consumption
            utility[:, time] = current_utility
            if time < horizon - 1:
                k[:, time + 1] = next_k
                K[time + 1] = float(np.mean(next_k))

        discounted = discounted_sum(utility, self.config.economy.beta, axis=1)
        return PopulationPaths(
            k,
            K,
            aggregate,
            employment,
            resources,
            consumption,
            utility,
            discounted,
        )


def simulate_alm_path(
    initial_capital: float,
    aggregate_states: np.ndarray,
    coefficients: ALMCoefficients,
) -> np.ndarray:
    states = np.asarray(aggregate_states, dtype=np.int64)
    result = np.empty(states.size, dtype=np.float64)
    result[0] = initial_capital
    for time in range(states.size - 1):
        result[time + 1] = next_aggregate_capital(
            result[time], states[time], coefficients
        )
    return result
