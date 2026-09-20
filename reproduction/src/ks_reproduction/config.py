"""Typed configuration for the Krusell--Smith thesis reproduction.

The original sources spread parameters across the manuscript, Python scripts,
and Julia notebooks.  This module gives every run one serialisable source of
truth.  The canonical aggregate-state ordering is always ``[bad, good]``;
conversion helpers are provided for the legacy DP arrays, whose ordering was
``[good, bad]``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on Python 3.10
    import tomli as tomllib


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _pair(values: Sequence[float], *, name: str) -> tuple[float, float]:
    _require(len(values) == 2, f"{name} must contain exactly two values")
    result = (float(values[0]), float(values[1]))
    _require(all(math.isfinite(value) for value in result), f"{name} must be finite")
    return result


@dataclass(frozen=True)
class ALMCoefficients:
    """Log-linear aggregate law of motion with explicit regime names.

    The law is ``log(K_next) = intercept + slope * log(K)``.  Canonical flat
    arrays use ``[bad_intercept, bad_slope, good_intercept, good_slope]``.
    The retained DP cache used the opposite regime order, so callers must use
    :meth:`from_legacy_dp` rather than relying on positional conventions.
    """

    bad_intercept: float = 0.0
    bad_slope: float = 1.0
    good_intercept: float = 0.0
    good_slope: float = 1.0

    def __post_init__(self) -> None:
        values = self.to_bad_good()
        _require(all(math.isfinite(value) for value in values), "ALM coefficients must be finite")

    @classmethod
    def identity(cls) -> ALMCoefficients:
        """Return the thesis's initial constant-capital ALM."""

        return cls()

    @classmethod
    def from_regime_pairs(
        cls,
        *,
        bad: Sequence[float],
        good: Sequence[float],
    ) -> ALMCoefficients:
        bad_intercept, bad_slope = _pair(bad, name="bad")
        good_intercept, good_slope = _pair(good, name="good")
        return cls(bad_intercept, bad_slope, good_intercept, good_slope)

    @classmethod
    def from_bad_good(cls, values: Sequence[float]) -> ALMCoefficients:
        """Construct from canonical DDPG-style ``[bad, good]`` ordering."""

        _require(len(values) == 4, "ALM vector must contain four values")
        return cls(*(float(value) for value in values))

    @classmethod
    def from_legacy_ddpg(cls, values: Sequence[float]) -> ALMCoefficients:
        """Construct from legacy DDPG ``[bad_intercept, bad_slope, good...]``."""

        return cls.from_bad_good(values)

    @classmethod
    def from_legacy_dp(cls, values: Sequence[float]) -> ALMCoefficients:
        """Construct from legacy DP ``[good_intercept, good_slope, bad...]``."""

        _require(len(values) == 4, "legacy DP ALM vector must contain four values")
        good_intercept, good_slope, bad_intercept, bad_slope = (float(value) for value in values)
        return cls(bad_intercept, bad_slope, good_intercept, good_slope)

    def to_bad_good(self) -> tuple[float, float, float, float]:
        return (
            float(self.bad_intercept),
            float(self.bad_slope),
            float(self.good_intercept),
            float(self.good_slope),
        )

    def to_legacy_ddpg(self) -> tuple[float, float, float, float]:
        return self.to_bad_good()

    def to_legacy_dp(self) -> tuple[float, float, float, float]:
        return (
            float(self.good_intercept),
            float(self.good_slope),
            float(self.bad_intercept),
            float(self.bad_slope),
        )

    def as_regime_pairs(self) -> dict[str, tuple[float, float]]:
        return {
            "bad": (float(self.bad_intercept), float(self.bad_slope)),
            "good": (float(self.good_intercept), float(self.good_slope)),
        }

    def for_regime(self, regime: str | int) -> tuple[float, float]:
        if regime in ("bad", "b", 0):
            return float(self.bad_intercept), float(self.bad_slope)
        if regime in ("good", "g", 1):
            return float(self.good_intercept), float(self.good_slope)
        raise ValueError(f"unknown aggregate regime: {regime!r}")

    def predict(self, aggregate_capital: float, regime: str | int) -> float:
        _require(aggregate_capital > 0.0, "aggregate capital must be positive")
        intercept, slope = self.for_regime(regime)
        return math.exp(intercept + slope * math.log(float(aggregate_capital)))


@dataclass(frozen=True)
class EconomyConfig:
    alpha: float = 0.36
    beta: float = 0.80
    delta: float = 0.025
    risk_aversion: float = 1.0
    productivity_bad: float = 0.99
    productivity_good: float = 1.01
    unemployment_bad: float = 0.10
    unemployment_good: float = 0.04
    unemployment_income: float = 0.0
    initial_individual_capital: float = 40.0
    initial_aggregate_capital: float = 40.0
    invalid_reward: float = -20.0
    minimum_positive: float = 1.0e-4

    def validate(self) -> None:
        _require(0.0 < self.alpha < 1.0, "alpha must lie in (0, 1)")
        _require(0.0 < self.beta < 1.0, "beta must lie in (0, 1)")
        _require(0.0 <= self.delta <= 1.0, "delta must lie in [0, 1]")
        _require(self.risk_aversion >= 0.0, "risk_aversion must be non-negative")
        _require(
            0.0 < self.productivity_bad < self.productivity_good,
            "productivity states must satisfy 0 < bad < good",
        )
        _require(
            0.0 <= self.unemployment_good < self.unemployment_bad < 1.0,
            "unemployment rates must satisfy 0 <= good < bad < 1",
        )
        _require(self.unemployment_income >= 0.0, "unemployment income must be non-negative")
        _require(
            self.initial_individual_capital > 0.0,
            "initial individual capital must be positive",
        )
        _require(
            self.initial_aggregate_capital > 0.0,
            "initial aggregate capital must be positive",
        )
        _require(self.minimum_positive > 0.0, "minimum_positive must be positive")


@dataclass(frozen=True)
class TransitionConfig:
    average_duration_bad: float = 8.0
    average_duration_good: float = 8.0
    unemployment_duration_bad: float = 2.5
    unemployment_duration_good: float = 1.5
    unemployment_persistence_gb_over_bb: float = 1.25
    unemployment_persistence_bg_over_gg: float = 0.75
    simulation_initial_aggregate_state: int = 1

    def validate(self) -> None:
        for name in (
            "average_duration_bad",
            "average_duration_good",
            "unemployment_duration_bad",
            "unemployment_duration_good",
        ):
            _require(getattr(self, name) > 1.0, f"{name} must be greater than one")
        _require(
            self.unemployment_persistence_gb_over_bb > 0.0,
            "unemployment_persistence_gb_over_bb must be positive",
        )
        _require(
            self.unemployment_persistence_bg_over_gg > 0.0,
            "unemployment_persistence_bg_over_gg must be positive",
        )
        _require(
            self.simulation_initial_aggregate_state in {0, 1},
            "simulation_initial_aggregate_state must be 0 (bad) or 1 (good)",
        )


@dataclass(frozen=True)
class GridConfig:
    individual_min: float = 0.1
    individual_max: float = 1000.0
    individual_size: int = 100
    individual_curvature: float = 7.0
    aggregate_min: float = 0.1
    aggregate_max: float = 50.0
    aggregate_size: int = 20

    def validate(self) -> None:
        _require(0.0 < self.individual_min < self.individual_max, "invalid individual grid bounds")
        _require(self.individual_size >= 2, "individual_size must be at least two")
        _require(self.individual_curvature > 0.0, "individual_curvature must be positive")
        _require(0.0 < self.aggregate_min < self.aggregate_max, "invalid aggregate grid bounds")
        _require(self.aggregate_size >= 2, "aggregate_size must be at least two")


@dataclass(frozen=True)
class NetworkConfig:
    state_size: int = 4
    action_size: int = 1
    hidden_sizes: tuple[int, int] = (200, 100)
    actor_learning_rate: float = 1.0e-5
    critic_learning_rate: float = 5.0e-5
    target_update_rate: float = 0.01
    batch_size: int = 1024
    replay_capacity: int = 1_000_000
    replay_sample_with_replacement: bool = True
    action_bound: float = 1.0
    actor_output_scale: float = 2.0
    normalization: str = "legacy_frozen"
    initialization: str = "legacy_glorot_normal"
    ou_mean: float = 0.0
    ou_sigma: float = 0.05
    ou_theta: float = 0.20
    ou_dt: float = 0.01

    # Read-only aliases used by the model/replay components and by familiar
    # DDPG terminology.  They are intentionally not duplicated in TOML.
    @property
    def state_dim(self) -> int:
        return self.state_size

    @property
    def state_dims(self) -> tuple[int]:
        return (self.state_size,)

    @property
    def action_dim(self) -> int:
        return self.action_size

    @property
    def action_dims(self) -> int:
        return self.action_size

    @property
    def hidden_dims(self) -> tuple[int, int]:
        return self.hidden_sizes

    @property
    def max_size(self) -> int:
        return self.replay_capacity

    @property
    def mu(self) -> float:
        return self.ou_mean

    @property
    def sigma(self) -> float:
        return self.ou_sigma

    @property
    def theta(self) -> float:
        return self.ou_theta

    @property
    def dt(self) -> float:
        return self.ou_dt

    @property
    def batch_norm_mode(self) -> str:
        return self.normalization

    @property
    def normalization_eps(self) -> None:
        return None

    def validate(self) -> None:
        _require(self.state_size == 4, "the thesis state is (k, K, z, epsilon) and has size four")
        _require(self.action_size == 1, "the thesis action is the scalar ratio k'/k")
        _require(len(self.hidden_sizes) == 2, "hidden_sizes must contain two widths")
        _require(all(size > 0 for size in self.hidden_sizes), "hidden widths must be positive")
        _require(self.actor_learning_rate > 0.0, "actor learning rate must be positive")
        _require(self.critic_learning_rate > 0.0, "critic learning rate must be positive")
        _require(0.0 < self.target_update_rate <= 1.0, "target update rate must lie in (0, 1]")
        _require(self.batch_size > 0, "batch_size must be positive")
        _require(self.replay_capacity >= self.batch_size, "replay capacity must cover a batch")
        _require(self.action_bound > 0.0, "action_bound must be positive")
        _require(self.actor_output_scale > 0.0, "actor_output_scale must be positive")
        _require(
            self.normalization in {"legacy_frozen", "batch_norm", "none"},
            "unsupported normalization mode",
        )
        _require(bool(self.initialization), "initialization must not be empty")
        _require(self.ou_sigma >= 0.0, "OU sigma must be non-negative")
        _require(self.ou_theta >= 0.0, "OU theta must be non-negative")
        _require(self.ou_dt > 0.0, "OU dt must be positive")


@dataclass(frozen=True)
class ExplorationConfig:
    """Exploration process used while collecting DDPG transitions."""

    kind: str = "ou"
    uniform_low: float = 0.0
    uniform_high: float = 1.0

    def validate(self) -> None:
        _require(self.kind in {"ou", "uniform"}, "exploration kind must be 'ou' or 'uniform'")
        _require(
            math.isfinite(self.uniform_low) and math.isfinite(self.uniform_high),
            "uniform exploration bounds must be finite",
        )
        _require(
            self.uniform_high > self.uniform_low,
            "uniform exploration requires uniform_high > uniform_low",
        )


@dataclass(frozen=True)
class TrainingConfig:
    rounds: int = 5
    horizon: int = 30
    first_round_episodes: int = 10_000
    episodes_per_round: int = 5_000
    alm_population: int = 10_000
    alm_simulations: int = 100
    alm_tolerance: float = 1.0e-3
    score_window: int = 1_000
    alm_initial_aggregate_state: int = 0
    pretrain_actor: bool = False
    pretrain_steps: int = 100
    pretrain_batch_size: int = 1_000
    pretrain_target_ratio: float = 0.9

    def validate(self) -> None:
        _require(self.rounds > 0, "rounds must be positive")
        _require(self.horizon > 1, "horizon must exceed one")
        _require(self.first_round_episodes > 0, "first_round_episodes must be positive")
        _require(self.episodes_per_round > 0, "episodes_per_round must be positive")
        _require(self.alm_population > 0, "alm_population must be positive")
        _require(self.alm_simulations > 0, "alm_simulations must be positive")
        _require(self.alm_tolerance > 0.0, "alm_tolerance must be positive")
        _require(self.score_window > 0, "score_window must be positive")
        _require(
            self.alm_initial_aggregate_state in {0, 1},
            "training alm_initial_aggregate_state must be 0 (bad) or 1 (good)",
        )
        _require(self.pretrain_steps > 0, "pretrain_steps must be positive")
        _require(self.pretrain_batch_size > 0, "pretrain_batch_size must be positive")
        _require(
            0.0 < self.pretrain_target_ratio <= 1.0,
            "pretrain_target_ratio must lie in (0, 1]",
        )

    @property
    def total_episodes(self) -> int:
        return self.first_round_episodes + (self.rounds - 1) * self.episodes_per_round


@dataclass(frozen=True)
class EvaluationConfig:
    aggregate_paths: int = 500
    population_per_path: int = 5_000
    horizon: int = 30
    alm_paths: int = 300
    policy_capital_min: float = 1.0e-3
    policy_capital_max: float = 80.0
    policy_capital_step: float = 1.0
    figure_dpi: int = 200
    utility_initial_aggregate_state: int = 1
    alm_initial_aggregate_state: int = 0
    comparison_initial_aggregate_state: int = 1

    def validate(self) -> None:
        _require(self.aggregate_paths > 0, "aggregate_paths must be positive")
        _require(self.population_per_path > 0, "population_per_path must be positive")
        _require(self.horizon > 1, "evaluation horizon must exceed one")
        _require(self.alm_paths > 0, "alm_paths must be positive")
        _require(
            0.0 < self.policy_capital_min < self.policy_capital_max,
            "invalid policy plot range",
        )
        _require(self.policy_capital_step > 0.0, "policy_capital_step must be positive")
        _require(self.figure_dpi > 0, "figure_dpi must be positive")
        for name in (
            "utility_initial_aggregate_state",
            "alm_initial_aggregate_state",
            "comparison_initial_aggregate_state",
        ):
            _require(
                getattr(self, name) in {0, 1},
                f"evaluation {name} must be 0 (bad) or 1 (good)",
            )

    @property
    def utility_sample_size(self) -> int:
        return self.aggregate_paths * self.population_per_path


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 0
    deterministic_torch: bool = True
    device: str = "auto"
    dtype: str = "float32"

    def validate(self) -> None:
        _require(self.seed >= 0, "seed must be non-negative")
        _require(self.device in {"auto", "cpu", "cuda", "mps"}, "unsupported device")
        _require(self.dtype in {"float32", "float64"}, "dtype must be float32 or float64")


@dataclass(frozen=True)
class CompatibilityConfig:
    """Switches needed to emulate the retained TensorFlow implementation.

    When ``legacy_faithful`` is false, consumers should ignore all legacy
    switches and use the model-correct implementation.  Keeping each quirk
    named makes experiment metadata honest and permits one-at-a-time audits.
    """

    legacy_faithful: bool = True
    use_normalized_labor_endowment: bool = True
    use_joint_state_index_bug: bool = True
    use_soft_target_initialization_bug: bool = True
    discount_reported_training_score: bool = True
    keep_ou_state_between_episodes: bool = True
    keep_last_period_reward: bool = True
    force_exact_unemployment_shares: bool = True
    evaluate_methods_on_separate_shocks: bool = True
    leave_final_evaluation_utility_zero: bool = True

    def active(self, option: str) -> bool:
        if option == "legacy_faithful":
            return self.legacy_faithful
        if option not in {field.name for field in fields(self)}:
            raise KeyError(f"unknown compatibility option: {option}")
        return self.legacy_faithful and bool(getattr(self, option))


@dataclass(frozen=True)
class ReproductionConfig:
    schema_version: int = 1
    experiment_name: str = "thesis_baseline"
    verification_profile: str = "thesis_numerical"
    economy: EconomyConfig = EconomyConfig()
    transitions: TransitionConfig = TransitionConfig()
    grids: GridConfig = GridConfig()
    network: NetworkConfig = NetworkConfig()
    exploration: ExplorationConfig = ExplorationConfig()
    training: TrainingConfig = TrainingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    compatibility: CompatibilityConfig = CompatibilityConfig()
    initial_alm: ALMCoefficients = ALMCoefficients()

    @property
    def ddpg(self) -> NetworkConfig:
        """Compatibility alias used by the DDPG component constructors."""

        return self.network

    def validate(self) -> None:
        _require(self.schema_version == 1, "unsupported configuration schema version")
        _require(bool(self.experiment_name.strip()), "experiment_name must not be empty")
        _require(
            self.verification_profile in {"thesis_numerical", "structural"},
            "verification_profile must be 'thesis_numerical' or 'structural'",
        )
        self.economy.validate()
        self.transitions.validate()
        self.grids.validate()
        self.network.validate()
        self.exploration.validate()
        self.training.validate()
        _require(
            self.training.pretrain_target_ratio <= self.network.action_bound,
            "pretrain_target_ratio must not exceed the actor action bound",
        )
        self.evaluation.validate()
        self.runtime.validate()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, strict: bool = True) -> ReproductionConfig:
        allowed = {field.name for field in fields(cls)}
        if strict:
            unknown = set(data) - allowed
            _require(not unknown, f"unknown top-level configuration keys: {sorted(unknown)}")

        def section(
            key: str,
            section_type: type[Any],
            *,
            conversions: Mapping[str, Any] | None = None,
        ) -> Any:
            raw = dict(data.get(key, {}))
            allowed_section = {field.name for field in fields(section_type)}
            if strict:
                unknown_section = set(raw) - allowed_section
                _require(not unknown_section, f"unknown keys in [{key}]: {sorted(unknown_section)}")
            else:
                raw = {
                    field_name: value
                    for field_name, value in raw.items()
                    if field_name in allowed_section
                }
            for field_name, converter in (conversions or {}).items():
                if field_name in raw:
                    raw[field_name] = converter(raw[field_name])
            return section_type(**raw)

        config = cls(
            schema_version=int(data.get("schema_version", 1)),
            experiment_name=str(data.get("experiment_name", "thesis_baseline")),
            verification_profile=str(
                data.get("verification_profile", "thesis_numerical")
            ),
            economy=section("economy", EconomyConfig),
            transitions=section("transitions", TransitionConfig),
            grids=section("grids", GridConfig),
            network=section(
                "network",
                NetworkConfig,
                conversions={"hidden_sizes": lambda value: tuple(int(item) for item in value)},
            ),
            exploration=section("exploration", ExplorationConfig),
            training=section("training", TrainingConfig),
            evaluation=section("evaluation", EvaluationConfig),
            runtime=section("runtime", RuntimeConfig),
            compatibility=section("compatibility", CompatibilityConfig),
            initial_alm=section("initial_alm", ALMCoefficients),
        )
        config.validate()
        return config

    @classmethod
    def from_toml(cls, path: str | Path, *, strict: bool = True) -> ReproductionConfig:
        with Path(path).open("rb") as handle:
            return cls.from_dict(tomllib.load(handle), strict=strict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_toml(self) -> str:
        self.validate()
        data = self.to_dict()
        lines = [
            f"schema_version = {self.schema_version}",
            f"experiment_name = {json.dumps(self.experiment_name)}",
            f"verification_profile = {json.dumps(self.verification_profile)}",
        ]
        for section_name in (
            "economy",
            "transitions",
            "grids",
            "network",
            "exploration",
            "training",
            "evaluation",
            "runtime",
            "compatibility",
            "initial_alm",
        ):
            lines.extend(("", f"[{section_name}]"))
            for key, value in data[section_name].items():
                lines.append(f"{key} = {_toml_value(value)}")
        return "\n".join(lines) + "\n"

    def write_toml(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_toml(), encoding="utf-8")
        return destination


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        _require(math.isfinite(value), "TOML output does not support non-finite floats")
        return repr(value)
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value: {value!r}")


def load_config(path: str | Path, *, strict: bool = True) -> ReproductionConfig:
    return ReproductionConfig.from_toml(path, strict=strict)


def save_config(config: ReproductionConfig, path: str | Path) -> Path:
    return config.write_toml(path)


__all__ = [
    "ALMCoefficients",
    "CompatibilityConfig",
    "EconomyConfig",
    "EvaluationConfig",
    "ExplorationConfig",
    "GridConfig",
    "NetworkConfig",
    "ReproductionConfig",
    "RuntimeConfig",
    "TrainingConfig",
    "TransitionConfig",
    "load_config",
    "save_config",
]
