"""Matplotlib figures with manuscript-compatible names."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (6.4, 4.8),
            "figure.dpi": 100,
            "savefig.dpi": 200,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )
    return plt


def _save(fig, path: Path) -> Path:
    plt = _pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    # Keep the 6.4 x 4.8 inch canvas at 200 dpi (1280 x 960 pixels), matching
    # the manuscript assets. ``bbox_inches='tight'`` would change that contract.
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def policy_figure(
    path: Path,
    capital: np.ndarray,
    unemployed_next: np.ndarray,
    employed_next: np.ndarray,
) -> Path:
    plt = _pyplot()
    fig, axis = plt.subplots()
    axis.plot(capital, unemployed_next, "b--", label="unemployed")
    axis.plot(capital, employed_next, "r-", label="employed")
    axis.plot(capital, capital, "k:", label="45 degree")
    axis.set(xlabel="capital today", ylabel="capital tomorrow")
    axis.legend()
    return _save(fig, path)


def alm_figure(
    path: Path,
    capital: np.ndarray,
    bad_next: np.ndarray,
    good_next: np.ndarray,
) -> Path:
    plt = _pyplot()
    fig, axis = plt.subplots()
    axis.plot(capital, bad_next, "b--", label="bad")
    axis.plot(capital, good_next, "r-", label="good")
    axis.plot(capital, capital, "k:", label="45 degree")
    axis.set(xlabel="aggregate capital today", ylabel="aggregate capital tomorrow")
    axis.legend()
    return _save(fig, path)


def capital_path_figure(path: Path, simulated: np.ndarray, approximated: np.ndarray) -> Path:
    plt = _pyplot()
    fig, axis = plt.subplots()
    axis.plot(approximated, "b--", label="approximation by ALM")
    axis.plot(simulated, "r-", label="simulation")
    axis.set(xlabel="period", ylabel="aggregate capital")
    axis.legend()
    return _save(fig, path)


def score_figure(path: Path, scores: Sequence[float], window: int = 1000) -> Path:
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        values = np.zeros(1)
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    indices = np.arange(values.size)
    starts = np.maximum(0, indices + 1 - window)
    moving = (cumulative[indices + 1] - cumulative[starts]) / (indices + 1 - starts)
    plt = _pyplot()
    fig, axis = plt.subplots()
    axis.plot(moving)
    axis.set(xlabel="game", ylabel="average discounted return")
    return _save(fig, path)


def alm_coefficients_figure(path: Path, coefficients: np.ndarray) -> Path:
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.ndim != 2 or coefficients.shape[1] != 4:
        raise ValueError("ALM coefficient history must have shape [round, 4]")
    plt = _pyplot()
    fig, axis = plt.subplots()
    lines = axis.plot(coefficients)
    axis.legend(lines, ["intercept bad", "slope bad", "intercept good", "slope good"])
    axis.set(xlabel="round", ylabel="coefficient value")
    return _save(fig, path)


def alm_history_figure(path: Path, histories: Mapping[int, np.ndarray]) -> Path:
    plt = _pyplot()
    fig, axis = plt.subplots()
    for round_number, values in sorted(histories.items()):
        axis.plot(values, label=str(round_number))
    axis.set(xlabel="period", ylabel="aggregate capital")
    axis.legend(title="round", ncol=2)
    return _save(fig, path)


def consumption_capital_figure(
    path: Path,
    dp_consumption: np.ndarray,
    dp_capital: np.ndarray,
    ddpg_consumption: np.ndarray,
    ddpg_capital: np.ndarray,
) -> Path:
    plt = _pyplot()
    fig, axis = plt.subplots()
    axis.plot(dp_consumption, "r-", label="consumption - dynamic programming")
    axis.plot(dp_capital, "r--", label="capital - dynamic programming")
    axis.plot(ddpg_consumption, "b-", label="consumption - DDPG")
    axis.plot(ddpg_capital, "b--", label="capital - DDPG")
    axis.set(xlabel="period", ylabel="consumption/capital")
    axis.legend(fontsize=8)
    return _save(fig, path)


def comparison_figure(
    path: Path,
    dp_values: np.ndarray,
    ddpg_values: np.ndarray,
    *,
    ylabel: str,
) -> Path:
    plt = _pyplot()
    fig, axis = plt.subplots()
    axis.plot(dp_values, "r-", label="dynamic programming")
    axis.plot(ddpg_values, "b--", label="DDPG")
    axis.set(xlabel="period", ylabel=ylabel)
    axis.legend()
    return _save(fig, path)


def utility_density_figure(
    path: Path,
    dp_utility: np.ndarray,
    ddpg_utility: np.ndarray,
    bins: int = 120,
) -> Path:
    """Plot a stable histogram-density approximation without seaborn/scipy."""

    dp_utility = np.asarray(dp_utility, dtype=np.float64)
    ddpg_utility = np.asarray(ddpg_utility, dtype=np.float64)
    finite_dp = dp_utility[np.isfinite(dp_utility)]
    finite_ddpg = ddpg_utility[np.isfinite(ddpg_utility)]
    if finite_dp.size == 0 or finite_ddpg.size == 0:
        raise ValueError("utility samples contain no finite values")
    lower = min(float(finite_dp.min()), float(finite_ddpg.min()))
    upper = max(float(finite_dp.max()), float(finite_ddpg.max()))
    edges = np.linspace(lower, upper, bins + 1)
    dp_density, _ = np.histogram(finite_dp, bins=edges, density=True)
    ddpg_density, _ = np.histogram(finite_ddpg, bins=edges, density=True)
    # A small fixed convolution mimics the original KDE while remaining O(bins).
    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
    dp_density = np.convolve(dp_density, kernel, mode="same")
    ddpg_density = np.convolve(ddpg_density, kernel, mode="same")
    centres = 0.5 * (edges[:-1] + edges[1:])
    plt = _pyplot()
    fig, axis = plt.subplots()
    axis.plot(centres, dp_density, "r-", label="dynamic programming")
    axis.plot(centres, ddpg_density, "b-", label="DDPG")
    axis.set(xlabel="sum of discounted utility", ylabel="density")
    axis.legend()
    return _save(fig, path)


NUMERICAL_FIGURES = {
    "DP": (
        "DP_policy_good_state_eng.jpg",
        "DP_Equilibrium_ALM_eng.jpg",
        "DP_Equilibrium_K_path_eng.jpg",
    ),
    "round": (
        "policy_good_state_eng.jpg",
        "Equilibrium_ALM_eng.jpg",
        "Equilibrium_K_path_eng.jpg",
        "ALM_coeff_eng.jpg",
        "Equilibrium_ALM_after_each_round_selected_eng.jpg",
        "score_history_eng.jpg",
        "comparison_consumption_capital_eng.jpg",
        "comparison_income_eng.jpg",
        "comparison_utility_eng.jpg",
        "utility_eng.jpg",
    ),
}


def missing_figure_names(figure_root: Path, rounds: Iterable[int] = (5, 20)) -> list[str]:
    expected = [figure_root / "DP" / name for name in NUMERICAL_FIGURES["DP"]]
    for round_number in rounds:
        expected.extend(
            figure_root / f"round_{round_number}" / name
            for name in NUMERICAL_FIGURES["round"]
        )
    return [str(path.relative_to(figure_root)) for path in expected if not path.is_file()]
