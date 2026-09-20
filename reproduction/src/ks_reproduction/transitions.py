"""Krusell--Smith aggregate and idiosyncratic Markov transitions.

All public arrays use the explicit ordering ``BAD=0, GOOD=1`` and
``UNEMPLOYED=0, EMPLOYED=1``.  Joint states are therefore
``(bad, unemployed), (bad, employed), (good, unemployed), (good, employed)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .config import CompatibilityConfig, EconomyConfig, TransitionConfig


class AggregateState(IntEnum):
    BAD = 0
    GOOD = 1


class EmploymentState(IntEnum):
    UNEMPLOYED = 0
    EMPLOYED = 1


JOINT_STATE_NAMES = (
    "bad_unemployed",
    "bad_employed",
    "good_unemployed",
    "good_employed",
)

_DEFAULT_ECONOMY = EconomyConfig()
_DEFAULT_TRANSITIONS = TransitionConfig()
_DEFAULT_COMPATIBILITY = CompatibilityConfig()


def joint_index(aggregate_state: int, employment_state: int) -> int:
    """Correct row-major joint-state index ``2*z + epsilon``."""

    z = _binary_scalar(aggregate_state, name="aggregate_state")
    epsilon = _binary_scalar(employment_state, name="employment_state")
    return 2 * z + epsilon


def legacy_joint_index(aggregate_state: int, employment_state: int) -> int:
    """Indexing bug in the retained TensorFlow environment (``z + epsilon``)."""

    z = _binary_scalar(aggregate_state, name="aggregate_state")
    epsilon = _binary_scalar(employment_state, name="employment_state")
    return z + epsilon


def decode_joint_index(index: int) -> tuple[AggregateState, EmploymentState]:
    if int(index) not in range(4):
        raise ValueError("joint-state index must lie in {0, 1, 2, 3}")
    return AggregateState(int(index) // 2), EmploymentState(int(index) % 2)


def _binary_scalar(value: int, *, name: str) -> int:
    result = int(value)
    if result not in (0, 1):
        raise ValueError(f"{name} must be zero or one")
    return result


def _validate_stochastic_matrix(matrix: np.ndarray, *, name: str) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite probabilities")
    if np.any(matrix < -1.0e-12) or np.any(matrix > 1.0 + 1.0e-12):
        raise ValueError(f"{name} contains probabilities outside [0, 1]")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1.0e-12, rtol=0.0):
        raise ValueError(f"rows of {name} must sum to one")


@dataclass(frozen=True)
class TransitionMatrices:
    """All transition matrices in canonical named state order.

    ``employment[z, z_next, epsilon, epsilon_next]`` stores idiosyncratic
    transitions conditional on the aggregate-state transition.
    """

    aggregate: np.ndarray
    employment: np.ndarray
    joint: np.ndarray

    def validate(self) -> None:
        if self.aggregate.shape != (2, 2):
            raise ValueError("aggregate transition matrix must have shape (2, 2)")
        if self.employment.shape != (2, 2, 2, 2):
            raise ValueError("employment transitions must have shape (2, 2, 2, 2)")
        if self.joint.shape != (4, 4):
            raise ValueError("joint transition matrix must have shape (4, 4)")
        _validate_stochastic_matrix(self.aggregate, name="aggregate transition matrix")
        for z in range(2):
            for z_next in range(2):
                _validate_stochastic_matrix(
                    self.employment[z, z_next],
                    name=f"employment transition ({z}->{z_next})",
                )
        _validate_stochastic_matrix(self.joint, name="joint transition matrix")

    def employment_given(
        self,
        aggregate_state: int,
        next_aggregate_state: int,
    ) -> np.ndarray:
        z = _binary_scalar(aggregate_state, name="aggregate_state")
        z_next = _binary_scalar(next_aggregate_state, name="next_aggregate_state")
        return self.employment[z, z_next]

    def joint_row(
        self,
        aggregate_state: int,
        employment_state: int,
        *,
        legacy_index_bug: bool = False,
    ) -> np.ndarray:
        indexer = legacy_joint_index if legacy_index_bug else joint_index
        return self.joint[indexer(aggregate_state, employment_state)]


def build_transition_matrices(
    economy: EconomyConfig = _DEFAULT_ECONOMY,
    transitions: TransitionConfig = _DEFAULT_TRANSITIONS,
) -> TransitionMatrices:
    """Derive the transition matrices from the thesis duration restrictions."""

    economy.validate()
    transitions.validate()

    p_bad_bad = 1.0 - 1.0 / transitions.average_duration_bad
    p_good_good = 1.0 - 1.0 / transitions.average_duration_good
    aggregate = np.array(
        [
            [p_bad_bad, 1.0 - p_bad_bad],
            [1.0 - p_good_good, p_good_good],
        ],
        dtype=np.float64,
    )

    p_uu_bad_bad = 1.0 - 1.0 / transitions.unemployment_duration_bad
    p_uu_good_good = 1.0 - 1.0 / transitions.unemployment_duration_good
    p_uu_good_bad = (
        transitions.unemployment_persistence_gb_over_bb * p_uu_bad_bad
    )
    p_uu_bad_good = (
        transitions.unemployment_persistence_bg_over_gg * p_uu_good_good
    )

    unemployment = (economy.unemployment_bad, economy.unemployment_good)
    p_uu = np.empty((2, 2), dtype=np.float64)
    p_uu[AggregateState.BAD, AggregateState.BAD] = p_uu_bad_bad
    p_uu[AggregateState.BAD, AggregateState.GOOD] = p_uu_bad_good
    p_uu[AggregateState.GOOD, AggregateState.BAD] = p_uu_good_bad
    p_uu[AggregateState.GOOD, AggregateState.GOOD] = p_uu_good_good

    employment = np.empty((2, 2, 2, 2), dtype=np.float64)
    for z in range(2):
        for z_next in range(2):
            probability_uu = p_uu[z, z_next]
            probability_eu = (
                unemployment[z_next] - unemployment[z] * probability_uu
            ) / (1.0 - unemployment[z])
            employment[z, z_next] = np.array(
                [
                    [probability_uu, 1.0 - probability_uu],
                    [probability_eu, 1.0 - probability_eu],
                ],
                dtype=np.float64,
            )

    joint = np.empty((4, 4), dtype=np.float64)
    for z in range(2):
        for epsilon in range(2):
            row = joint_index(z, epsilon)
            for z_next in range(2):
                for epsilon_next in range(2):
                    column = joint_index(z_next, epsilon_next)
                    joint[row, column] = (
                        aggregate[z, z_next]
                        * employment[z, z_next, epsilon, epsilon_next]
                    )

    result = TransitionMatrices(aggregate=aggregate, employment=employment, joint=joint)
    result.validate()
    for array in (result.aggregate, result.employment, result.joint):
        array.setflags(write=False)
    return result


def sample_categorical(probabilities: Sequence[float], rng: np.random.Generator) -> int:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("probabilities must be a non-empty one-dimensional vector")
    if np.any(values < 0.0) or not np.isclose(values.sum(), 1.0):
        raise ValueError("probabilities must be non-negative and sum to one")
    draw = int(np.searchsorted(np.cumsum(values), rng.random(), side="right"))
    return min(draw, values.size - 1)


def sample_aggregate_path(
    matrices: TransitionMatrices,
    horizon: int,
    rng: np.random.Generator,
    *,
    initial_state: int | None = None,
) -> np.ndarray:
    """Sample exactly ``horizon`` aggregate states, including the initial one."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    path = np.empty(horizon, dtype=np.int64)
    path[0] = int(rng.integers(0, 2)) if initial_state is None else _binary_scalar(
        initial_state,
        name="initial_state",
    )
    for time in range(1, horizon):
        path[time] = sample_categorical(matrices.aggregate[path[time - 1]], rng)
    return path


def _force_unemployment_count(
    employment: np.ndarray,
    target_unemployed: int,
    rng: np.random.Generator,
) -> None:
    unemployed = np.flatnonzero(employment == EmploymentState.UNEMPLOYED)
    difference = int(target_unemployed) - unemployed.size
    if difference > 0:
        employed = np.flatnonzero(employment == EmploymentState.EMPLOYED)
        selected = rng.choice(employed, size=difference, replace=False)
        employment[selected] = EmploymentState.UNEMPLOYED
    elif difference < 0:
        selected = rng.choice(unemployed, size=-difference, replace=False)
        employment[selected] = EmploymentState.EMPLOYED


def sample_employment_paths(
    matrices: TransitionMatrices,
    aggregate_path: Sequence[int],
    population: int,
    rng: np.random.Generator,
    economy: EconomyConfig = _DEFAULT_ECONOMY,
    compatibility: CompatibilityConfig = _DEFAULT_COMPATIBILITY,
    *,
    initial_states: Sequence[int] | None = None,
    force_exact_shares: bool | None = None,
) -> np.ndarray:
    """Sample employment conditional on a common aggregate path.

    In legacy-faithful mode populations of at least 100 are adjusted after
    sampling to contain exactly ``floor(u_z * population)`` unemployed agents,
    matching the retained code.
    """

    z_path = np.asarray(aggregate_path, dtype=np.int64)
    if z_path.ndim != 1 or z_path.size == 0:
        raise ValueError("aggregate_path must be a non-empty vector")
    if not np.all((z_path == 0) | (z_path == 1)):
        raise ValueError("aggregate_path must contain only zero or one")
    if population <= 0:
        raise ValueError("population must be positive")

    employment = np.empty((population, z_path.size), dtype=np.int64)
    if initial_states is None:
        initial_unemployment = (
            economy.unemployment_bad
            if z_path[0] == AggregateState.BAD
            else economy.unemployment_good
        )
        employment[:, 0] = (rng.random(population) > initial_unemployment).astype(np.int64)
    else:
        initial = np.asarray(initial_states, dtype=np.int64)
        if initial.shape != (population,) or not np.all((initial == 0) | (initial == 1)):
            raise ValueError("initial_states must be a binary vector of length population")
        employment[:, 0] = initial

    for time in range(1, z_path.size):
        conditional = matrices.employment_given(z_path[time - 1], z_path[time])
        uniforms = rng.random(population)
        probability_unemployed = conditional[employment[:, time - 1], EmploymentState.UNEMPLOYED]
        employment[:, time] = (uniforms > probability_unemployed).astype(np.int64)

    enforce = (
        compatibility.active("force_exact_unemployment_shares")
        if force_exact_shares is None
        else bool(force_exact_shares)
    )
    if enforce and population >= 100:
        for time, z in enumerate(z_path):
            rate = (
                economy.unemployment_bad
                if z == AggregateState.BAD
                else economy.unemployment_good
            )
            _force_unemployment_count(employment[:, time], int(rate * population), rng)
    return employment


def sample_joint_next_state(
    matrices: TransitionMatrices,
    aggregate_state: int,
    employment_state: int,
    rng: np.random.Generator,
    compatibility: CompatibilityConfig = _DEFAULT_COMPATIBILITY,
) -> tuple[AggregateState, EmploymentState]:
    row = matrices.joint_row(
        aggregate_state,
        employment_state,
        legacy_index_bug=compatibility.active("use_joint_state_index_bug"),
    )
    return decode_joint_index(sample_categorical(row, rng))


def sample_joint_path(
    matrices: TransitionMatrices,
    horizon: int,
    rng: np.random.Generator,
    compatibility: CompatibilityConfig = _DEFAULT_COMPATIBILITY,
    *,
    initial_state: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the representative-agent transition used during DDPG training."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    aggregate = np.empty(horizon, dtype=np.int64)
    employment = np.empty(horizon, dtype=np.int64)
    if initial_state is None:
        aggregate[0], employment[0] = rng.integers(0, 2, size=2)
    else:
        aggregate[0] = _binary_scalar(initial_state[0], name="initial aggregate state")
        employment[0] = _binary_scalar(initial_state[1], name="initial employment state")
    for time in range(1, horizon):
        next_aggregate, next_employment = sample_joint_next_state(
            matrices,
            aggregate[time - 1],
            employment[time - 1],
            rng,
            compatibility,
        )
        aggregate[time] = next_aggregate
        employment[time] = next_employment
    return aggregate, employment


def stationary_distribution(matrix: np.ndarray) -> np.ndarray:
    """Return the unique finite-state stationary distribution."""

    values = np.asarray(matrix, dtype=np.float64)
    _validate_stochastic_matrix(values, name="transition matrix")
    system = np.vstack((values.T - np.eye(values.shape[0]), np.ones(values.shape[0])))
    target = np.append(np.zeros(values.shape[0]), 1.0)
    distribution, *_ = np.linalg.lstsq(system, target, rcond=None)
    distribution = np.maximum(distribution, 0.0)
    distribution /= distribution.sum()
    return distribution


__all__ = [
    "AggregateState",
    "EmploymentState",
    "JOINT_STATE_NAMES",
    "TransitionMatrices",
    "build_transition_matrices",
    "decode_joint_index",
    "joint_index",
    "legacy_joint_index",
    "sample_aggregate_path",
    "sample_categorical",
    "sample_employment_paths",
    "sample_joint_next_state",
    "sample_joint_path",
    "stationary_distribution",
]
