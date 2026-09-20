"""Manuscript-scale evaluation, raw evidence, tables, and numerical figures."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifacts import RunLayout, atomic_write_json
from .config import ALMCoefficients, ReproductionConfig
from .dp import DPReference
from .economy import next_aggregate_capital
from .plotting import (
    alm_coefficients_figure,
    alm_figure,
    alm_history_figure,
    capital_path_figure,
    comparison_figure,
    consumption_capital_figure,
    policy_figure,
    score_figure,
    utility_density_figure,
)
from .runtime import RNGStream, RNGStreams, resolve_device
from .simulation import (
    ActorPolicy,
    PopulationPaths,
    PopulationSimulator,
    policy_saving_ratio,
    simulate_alm_path,
)
from .statistics import SummaryStatistics, independent_t_test
from .training import _load_actor_snapshot


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _checkpoint(layout: RunLayout, requested_round: int, device: torch.device) -> dict[str, Any]:
    path = layout.checkpoint(requested_round)
    if not path.is_file():
        path = layout.checkpoints / "last.pt"
    if not path.is_file():
        raise FileNotFoundError(f"no checkpoint is available for round {requested_round}")
    return _torch_load(path, device)


def _round_components(
    checkpoint: Mapping[str, Any],
    requested_round: int,
    config: ReproductionConfig,
    device: torch.device,
) -> tuple[ActorPolicy, ALMCoefficients, tuple[float, float], int]:
    snapshots = checkpoint["actor_snapshots"]
    if not snapshots:
        raise ValueError("checkpoint contains no actor snapshot")
    actual_round = min(requested_round, len(snapshots))
    actor = _load_actor_snapshot(snapshots[actual_round - 1], config, device, torch.float32)
    alm_history = checkpoint["alm_history"]
    r2_history = checkpoint["r2_history"]
    history_index = min(actual_round, len(alm_history) - 1)
    coefficients = ALMCoefficients.from_bad_good(alm_history[history_index])
    r2_bad, r2_good = r2_history[min(actual_round, len(r2_history) - 1)]
    return ActorPolicy(actor), coefficients, (float(r2_bad), float(r2_good)), actual_round


def _accepted_utility_samples(
    simulator: PopulationSimulator,
    policy: Any,
    config: ReproductionConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    accepted: list[np.ndarray] = []
    attempts = 0
    target_paths = config.evaluation.aggregate_paths
    while len(accepted) < target_paths:
        attempts += 1
        paths = simulator.simulate(
            policy,
            population=config.evaluation.population_per_path,
            horizon=config.evaluation.horizon,
            rng=rng,
            initial_aggregate_state=(
                config.evaluation.utility_initial_aggregate_state
            ),
        )
        if np.isfinite(paths.discounted_utility).all():
            accepted.append(paths.discounted_utility)
        if attempts > target_paths * 20:
            raise RuntimeError(
                f"could not obtain {target_paths} finite utility batches after {attempts} attempts"
            )
    return np.concatenate(accepted)


def _common_path_pair(
    simulator: PopulationSimulator,
    first_policy: Any,
    second_policy: Any,
    config: ReproductionConfig,
    rng: np.random.Generator,
) -> tuple[PopulationPaths, PopulationPaths]:
    population = 1
    horizon = config.evaluation.horizon
    aggregate, employment = simulator.shocks(
        population,
        horizon,
        rng,
        initial_aggregate_state=config.evaluation.comparison_initial_aggregate_state,
    )
    first = simulator.simulate(
        first_policy,
        population=population,
        horizon=horizon,
        rng=rng,
        aggregate_path=aggregate,
        employment_paths=employment,
    )
    second = simulator.simulate(
        second_policy,
        population=population,
        horizon=horizon,
        rng=rng,
        aggregate_path=aggregate,
        employment_paths=employment,
    )
    return first, second


def _alm_metrics(
    simulator: PopulationSimulator,
    policy: Any,
    coefficients: ALMCoefficients,
    r2: tuple[float, float],
    config: ReproductionConfig,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    capital_paths: list[np.ndarray] = []
    state_paths: list[np.ndarray] = []
    desired_paths = config.evaluation.alm_paths
    maximum_paths = max(desired_paths, 2) * 20
    for _ in range(maximum_paths):
        paths = simulator.simulate(
            policy,
            population=config.evaluation.population_per_path,
            horizon=config.evaluation.horizon,
            rng=rng,
            initial_aggregate_state=config.evaluation.alm_initial_aggregate_state,
        )
        capital_paths.append(paths.aggregate_capital)
        state_paths.append(paths.aggregate_state)
        if len(capital_paths) >= desired_paths:
            observed_regimes = np.concatenate([path[:-1] for path in state_paths])
            if np.any(observed_regimes == 0) and np.any(observed_regimes == 1):
                break
    else:
        raise RuntimeError(
            "ALM evaluation did not observe both aggregate regimes after "
            f"{maximum_paths} paths"
        )
    capital = np.asarray(capital_paths)
    states = np.asarray(state_paths)
    current = capital[:, :-1]
    following = capital[:, 1:]
    regimes = states[:, :-1]
    predicted = np.empty_like(following)
    for regime in (0, 1):
        mask = regimes == regime
        predicted[mask] = next_aggregate_capital(
            current[mask], regime, coefficients
        )

    def regime_metrics(regime: int, fit_r2: float) -> dict[str, float | int]:
        mask = regimes == regime
        intercept, slope = coefficients.for_regime(regime)
        return {
            "intercept": intercept,
            "slope": slope,
            "r_squared": fit_r2,
            "rmse": float(np.sqrt(np.mean(np.square(following[mask] - predicted[mask])))),
            "observations": int(np.count_nonzero(mask)),
        }

    metrics = {
        "bad": regime_metrics(0, r2[0]),
        "good": regime_metrics(1, r2[1]),
    }
    arrays = {
        "capital": capital,
        "aggregate_state": states,
        "predicted_next": predicted,
    }
    return metrics, arrays


def _plot_policy(
    destination: Path,
    policy: Any,
    config: ReproductionConfig,
) -> None:
    capital = np.arange(
        config.evaluation.policy_capital_min,
        config.evaluation.policy_capital_max,
        config.evaluation.policy_capital_step,
        dtype=np.float64,
    )
    K = np.full_like(capital, config.economy.initial_aggregate_capital)
    good = np.ones_like(capital)
    unemployed_states = np.column_stack(
        (capital, K, good, np.zeros_like(capital))
    ).astype(np.float32)
    employed_states = np.column_stack((capital, K, good, np.ones_like(capital))).astype(np.float32)
    unemployed_next = capital * policy_saving_ratio(policy, unemployed_states)
    employed_next = capital * policy_saving_ratio(policy, employed_states)
    policy_figure(destination, capital, unemployed_next, employed_next)


def _plot_alm(destination: Path, coefficients: ALMCoefficients, config: ReproductionConfig) -> None:
    capital = np.arange(
        config.evaluation.policy_capital_min,
        config.evaluation.policy_capital_max,
        config.evaluation.policy_capital_step,
        dtype=np.float64,
    )
    bad = next_aggregate_capital(capital, 0, coefficients)
    good = next_aggregate_capital(capital, 1, coefficients)
    alm_figure(destination, capital, bad, good)


def _write_tables(summary: Mapping[str, Any], layout: RunLayout) -> None:
    utility = summary["discounted_utility"]
    csv_path = layout.metrics / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "count", "mean", "standard_deviation", "maximum", "minimum"])
        for method in ("dynamic_programming", "ddpg_round_5", "ddpg_round_20"):
            row = utility[method]
            writer.writerow(
                [
                    method,
                    row["count"],
                    row["mean"],
                    row["standard_deviation"],
                    row["maximum"],
                    row["minimum"],
                ]
            )
    latex = [
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Method & Mean & Std. dev. & Max & Min \\",
        r"\hline",
    ]
    for label, method in (
        ("Value function iteration", "dynamic_programming"),
        ("DDPG (round 5)", "ddpg_round_5"),
        ("DDPG (round 20)", "ddpg_round_20"),
    ):
        row = utility[method]
        latex.append(
            f"{label} & {row['mean']:.4f} & {row['standard_deviation']:.4f} & "
            f"{row['maximum']:.4f} & {row['minimum']:.4f} \\\\"
        )
    latex.extend((r"\hline", r"\end{tabular}", ""))
    (layout.metrics / "thesis_table.tex").write_text("\n".join(latex), encoding="utf-8")


class ThesisEvaluator:
    def __init__(
        self,
        config: ReproductionConfig,
        layout: RunLayout,
        benchmark_directory: Path,
    ) -> None:
        self.config = config
        self.layout = layout
        self.benchmark_directory = benchmark_directory
        self.device = resolve_device(config.runtime.device)
        self.streams = RNGStreams(config.runtime.seed)
        self.simulator = PopulationSimulator(config)
        self.dp = DPReference.load(benchmark_directory)

    def evaluate(self) -> dict[str, Any]:
        checkpoint5 = _checkpoint(self.layout, 5, self.device)
        checkpoint20 = _checkpoint(self.layout, 20, self.device)
        policy5, alm5, r2_5, actual5 = _round_components(
            checkpoint5, 5, self.config, self.device
        )
        policy20, alm20, r2_20, actual20 = _round_components(
            checkpoint20, 20, self.config, self.device
        )

        utility_dp = _accepted_utility_samples(
            self.simulator,
            self.dp,
            self.config,
            self.streams.numpy(RNGStream.EVALUATION_DP_SHOCKS),
        )
        utility5 = _accepted_utility_samples(
            self.simulator,
            policy5,
            self.config,
            self.streams.numpy(f"{RNGStream.EVALUATION_DDPG_SHOCKS.value}.round5"),
        )
        utility20 = _accepted_utility_samples(
            self.simulator,
            policy20,
            self.config,
            self.streams.numpy(f"{RNGStream.EVALUATION_DDPG_SHOCKS.value}.round20"),
        )
        stats_dp = SummaryStatistics.from_array(utility_dp)
        stats5 = SummaryStatistics.from_array(utility5)
        stats20 = SummaryStatistics.from_array(utility20)
        t_test = independent_t_test(stats5, stats_dp, equal_variance=True)

        dp_coefficients = ALMCoefficients.from_legacy_dp(
            (*self.dp.alm_good, *self.dp.alm_bad)
        )
        dp_alm, dp_arrays = _alm_metrics(
            self.simulator,
            self.dp,
            dp_coefficients,
            (self.dp.r2_bad, self.dp.r2_good),
            self.config,
            self.streams.numpy("evaluation_alm.dp"),
        )
        ddpg5_alm, ddpg5_arrays = _alm_metrics(
            self.simulator,
            policy5,
            alm5,
            r2_5,
            self.config,
            self.streams.numpy("evaluation_alm.ddpg5"),
        )
        ddpg20_alm, ddpg20_arrays = _alm_metrics(
            self.simulator,
            policy20,
            alm20,
            r2_20,
            self.config,
            self.streams.numpy("evaluation_alm.ddpg20"),
        )

        summary = {
            "discounted_utility": {
                "dynamic_programming": stats_dp.to_dict(),
                "ddpg_round_5": stats5.to_dict(),
                "ddpg_round_20": stats20.to_dict(),
                "comparison": {
                    "relative_mean_difference": (stats5.mean - stats_dp.mean) / stats_dp.mean,
                    "independent_samples_t": t_test.statistic,
                    "degrees_of_freedom": t_test.degrees_of_freedom,
                    "p_value": t_test.p_value_two_sided,
                },
            },
            "aggregate_law_of_motion": {
                "dynamic_programming": dp_alm,
                "ddpg_round_5": ddpg5_alm,
                "ddpg_round_20": ddpg20_alm,
            },
            "checkpoint_mapping": {
                "requested_round_5": actual5,
                "requested_round_20": actual20,
            },
        }
        atomic_write_json(self.layout.metrics / "summary.json", summary)
        _write_tables(summary, self.layout)
        np.savez_compressed(
            self.layout.data / "discounted_utility.npz",
            dynamic_programming=utility_dp,
            ddpg_round_5=utility5,
            ddpg_round_20=utility20,
        )
        for name, arrays in (
            ("dp_alm_evaluation", dp_arrays),
            ("ddpg_round_5_alm_evaluation", ddpg5_arrays),
            ("ddpg_round_20_alm_evaluation", ddpg20_arrays),
        ):
            np.savez_compressed(self.layout.data / f"{name}.npz", **arrays)

        self.plot(
            checkpoint5=checkpoint5,
            checkpoint20=checkpoint20,
            policy5=policy5,
            policy20=policy20,
            alm5=alm5,
            alm20=alm20,
            utility_dp=utility_dp,
            utility5=utility5,
            utility20=utility20,
            dp_arrays=dp_arrays,
            ddpg5_arrays=ddpg5_arrays,
            ddpg20_arrays=ddpg20_arrays,
        )
        return summary

    def plot_saved(self) -> None:
        """Regenerate figures from saved evaluation arrays and checkpoints.

        This keeps the standalone ``plot`` stage read-only with respect to the
        expensive simulations: it never reruns manuscript-scale evaluation.
        """

        checkpoint5 = _checkpoint(self.layout, 5, self.device)
        checkpoint20 = _checkpoint(self.layout, 20, self.device)
        policy5, alm5, _, _ = _round_components(checkpoint5, 5, self.config, self.device)
        policy20, alm20, _, _ = _round_components(checkpoint20, 20, self.config, self.device)

        utility_path = self.layout.data / "discounted_utility.npz"
        if not utility_path.is_file():
            raise FileNotFoundError(
                f"saved evaluation samples are missing at {utility_path}; run evaluate first"
            )
        with np.load(utility_path) as values:
            utility_dp = values["dynamic_programming"].copy()
            utility5 = values["ddpg_round_5"].copy()
            utility20 = values["ddpg_round_20"].copy()

        def arrays(name: str) -> dict[str, np.ndarray]:
            path = self.layout.data / f"{name}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"saved ALM evaluation arrays are missing at {path}")
            with np.load(path) as values:
                return {key: values[key].copy() for key in values.files}

        self.plot(
            checkpoint5=checkpoint5,
            checkpoint20=checkpoint20,
            policy5=policy5,
            policy20=policy20,
            alm5=alm5,
            alm20=alm20,
            utility_dp=utility_dp,
            utility5=utility5,
            utility20=utility20,
            dp_arrays=arrays("dp_alm_evaluation"),
            ddpg5_arrays=arrays("ddpg_round_5_alm_evaluation"),
            ddpg20_arrays=arrays("ddpg_round_20_alm_evaluation"),
        )

    def plot(
        self,
        *,
        checkpoint5: Mapping[str, Any],
        checkpoint20: Mapping[str, Any],
        policy5: Any,
        policy20: Any,
        alm5: ALMCoefficients,
        alm20: ALMCoefficients,
        utility_dp: np.ndarray,
        utility5: np.ndarray,
        utility20: np.ndarray,
        dp_arrays: Mapping[str, np.ndarray],
        ddpg5_arrays: Mapping[str, np.ndarray],
        ddpg20_arrays: Mapping[str, np.ndarray],
    ) -> None:
        dp_dir = self.layout.figures / "DP"
        dp_coefficients = ALMCoefficients.from_legacy_dp((*self.dp.alm_good, *self.dp.alm_bad))
        _plot_policy(dp_dir / "DP_policy_good_state_eng.jpg", self.dp, self.config)
        _plot_alm(dp_dir / "DP_Equilibrium_ALM_eng.jpg", dp_coefficients, self.config)
        dp_recursive = simulate_alm_path(
            self.config.economy.initial_aggregate_capital,
            dp_arrays["aggregate_state"][0],
            dp_coefficients,
        )
        capital_path_figure(
            dp_dir / "DP_Equilibrium_K_path_eng.jpg",
            dp_arrays["capital"][0],
            dp_recursive,
        )

        for requested, checkpoint, policy, coefficients, utilities, arrays in (
            (5, checkpoint5, policy5, alm5, utility5, ddpg5_arrays),
            (20, checkpoint20, policy20, alm20, utility20, ddpg20_arrays),
        ):
            destination = self.layout.figures / f"round_{requested}"
            _plot_policy(destination / "policy_good_state_eng.jpg", policy, self.config)
            _plot_alm(destination / "Equilibrium_ALM_eng.jpg", coefficients, self.config)
            recursive = simulate_alm_path(
                self.config.economy.initial_aggregate_capital,
                arrays["aggregate_state"][0],
                coefficients,
            )
            capital_path_figure(
                destination / "Equilibrium_K_path_eng.jpg",
                arrays["capital"][0],
                recursive,
            )
            history = np.asarray(checkpoint["alm_history"], dtype=np.float64)
            upto = min(requested + 1, len(history))
            alm_coefficients_figure(destination / "ALM_coeff_eng.jpg", history[:upto])
            aggregate_path = arrays["aggregate_state"][0]
            histories = {
                index: simulate_alm_path(
                    self.config.economy.initial_aggregate_capital,
                    aggregate_path,
                    ALMCoefficients.from_bad_good(item),
                )
                for index, item in enumerate(history[:upto])
            }
            alm_history_figure(
                destination / "Equilibrium_ALM_after_each_round_selected_eng.jpg",
                histories,
            )
            scores = checkpoint["discounted_returns"]
            score_figure(
                destination / "score_history_eng.jpg",
                scores,
                window=self.config.training.score_window,
            )
            dp_path, ddpg_path = _common_path_pair(
                self.simulator,
                self.dp,
                policy,
                self.config,
                self.streams.numpy(f"plots.common_path.round{requested}"),
            )
            consumption_capital_figure(
                destination / "comparison_consumption_capital_eng.jpg",
                dp_path.consumption[0],
                dp_path.individual_capital[0],
                ddpg_path.consumption[0],
                ddpg_path.individual_capital[0],
            )
            comparison_figure(
                destination / "comparison_income_eng.jpg",
                dp_path.resources[0],
                ddpg_path.resources[0],
                ylabel="income",
            )
            comparison_figure(
                destination / "comparison_utility_eng.jpg",
                dp_path.utility[0],
                ddpg_path.utility[0],
                ylabel="utility",
            )
            utility_density_figure(
                destination / "utility_eng.jpg", utility_dp, utilities
            )
