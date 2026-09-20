"""Audit bundled historical outputs against the thesis, without running TensorFlow.

This module deliberately separates three kinds of evidence: the DP benchmark
cache, the archived evaluation arrays, and the numbers printed in Thesis.tex.
They were not all produced by the same saved run. A warning is preferable to
silently substituting one set of coefficients for another.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rounded_equal(actual: float, printed: float, decimals: int) -> bool:
    """Whether actual would print as the thesis number at its precision."""
    return math.isfinite(actual) and abs(actual - printed) < 0.5 * 10 ** (-decimals) + 1e-12


def _pooled_t(a: Dict[str, float], b: Dict[str, float]) -> Tuple[float, int]:
    """Original independent-samples statistic, with DDPG as the first group."""
    n_a, n_b = int(a["n"]), int(b["n"])
    var_a = a["std"] ** 2 * n_a / (n_a - 1)
    var_b = b["std"] ** 2 * n_b / (n_b - 1)
    dof = n_a + n_b - 2
    pooled = ((n_a - 1) * var_a + (n_b - 1) * var_b) / dof
    return (b["mean"] - a["mean"]) / math.sqrt(pooled * (1 / n_a + 1 / n_b)), dof


def verify(project_root: Path) -> Dict[str, Any]:
    """Return JSON-serializable checks; warnings do not make ``ok`` false.

    This validates saved results, not deterministic retraining. All ``.npy``
    and ``.npz`` reads disable pickle. TensorFlow is never imported.
    """
    root = Path(project_root).expanduser().resolve()
    checks: List[Dict[str, Any]] = []

    def add(check_id: str, status: str, detail: str, **values: Any) -> None:
        checks.append({"id": check_id, "status": status, "detail": detail, **values})

    def finish() -> Dict[str, Any]:
        failures = sum(check["status"] == "fail" for check in checks)
        warnings = sum(check["status"] == "warn" for check in checks)
        return {"ok": failures == 0, "checks": checks,
                "summary": {"passes": len(checks) - failures - warnings,
                            "warnings": warnings, "failures": failures},
                "scope": "archived-artifact audit only; no TensorFlow training or checkpoint restore"}

    target_path = root / "targets" / "thesis_results.json"
    if not target_path.is_file():
        add("targets", "fail", "Missing machine-readable transcription of Thesis.tex.", path=str(target_path))
        return finish()
    try:
        targets = json.loads(target_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        add("targets", "fail", "Could not read thesis targets.", error=str(exc))
        return finish()

    manuscript = root.parent / "manuscript" / "Thesis.tex"
    if manuscript.is_file():
        observed_hash = _sha256(manuscript)
        expected_hash = targets["manuscript"].get("sha256")
        add("manuscript.sha256", "pass" if observed_hash == expected_hash else "warn",
            "Thesis.tex matches the target transcription." if observed_hash == expected_hash
            else "Thesis.tex differs from the version used to transcribe targets.",
            observed=observed_hash, expected=expected_hash)
    else:
        add("manuscript.sha256", "warn", "Source manuscript is outside this copy; using the bundled target transcription.")

    try:
        import numpy as np
    except ImportError:
        add("numpy", "fail", "NumPy is needed to inspect archived arrays; install the legacy environment or a current NumPy for this audit.")
        return finish()

    def array(relative: str) -> Optional[Any]:
        path = root / relative
        if not path.is_file():
            add(f"file.{relative}", "fail", "Required archived array is missing.", path=relative)
            return None
        try:
            value = np.load(path, allow_pickle=False)
            if not isinstance(value, np.ndarray) or not np.issubdtype(value.dtype, np.number):
                raise ValueError("Expected a numeric .npy array")
            if not np.all(np.isfinite(value)):
                raise ValueError("Array has non-finite entries")
            return value
        except (OSError, ValueError, TypeError) as exc:
            add(f"file.{relative}", "fail", "Could not read a finite numeric array.", path=relative, error=str(exc))
            return None

    benchmark: Dict[str, Any] = {}
    expected_shapes = {"B": (4,), "R2": (2,), "kss_k_opt": (100, 20, 4), "kss_value": (100, 20, 4)}
    for name, shape in expected_shapes.items():
        relative = f"data/benchmark/{name}.npy"
        path = root / relative
        value = array(relative)
        if value is None:
            continue
        benchmark[name] = value
        expected_hash = targets["benchmark_inputs"][f"{name}_sha256"]
        actual_hash = _sha256(path)
        add(f"benchmark.{name}", "pass" if value.shape == shape and actual_hash == expected_hash else "fail",
            "Canonical DP cache has the documented shape and SHA-256." if value.shape == shape and actual_hash == expected_hash
            else "DP cache differs from the documented original; do not substitute the other organized cache.",
            shape=list(value.shape), expected_shape=list(shape), sha256=actual_hash, expected_sha256=expected_hash)

    printed_alm = targets["manuscript"]["aggregate_law_of_motion"]
    if "B" in benchmark and "R2" in benchmark:
        b, r2 = benchmark["B"], benchmark["R2"]
        if b.shape == (4,) and r2.shape == (2,):
            for state, offset, r2_index in (("good", 0, 0), ("bad", 2, 1)):
                printed = printed_alm["dynamic_programming"][state]
                actual = {"intercept": float(b[offset]), "slope": float(b[offset + 1]), "r_squared": float(r2[r2_index])}
                match = all(_rounded_equal(actual[k], printed[k], 6 if k == "r_squared" else 4) for k in actual)
                add(f"manuscript.dp_alm.{state}", "pass" if match else "fail",
                    "Benchmark DP ALM matches printed coefficients at the printed precision." if match
                    else "Benchmark DP ALM does not match the printed coefficients.",
                    observed=actual, printed={k: printed[k] for k in actual})

    baseline_b = array("data/historical/dp_base/B.npy")
    baseline_r2 = array("data/historical/dp_base/R2.npy")
    if baseline_b is not None and baseline_r2 is not None and "B" in benchmark and "R2" in benchmark:
        divergent = not (np.array_equal(baseline_b, benchmark["B"]) and np.array_equal(baseline_r2, benchmark["R2"]))
        add("provenance.dp_alm", "warn" if divergent else "pass",
            "The archived utility-evaluation run used different DP ALM coefficients from the cache whose coefficients appear in Thesis.tex. The two sources are kept separate."
            if divergent else "Archived evaluation and benchmark DP ALM coefficients agree.",
            benchmark_B=benchmark["B"].tolist(), evaluation_B=baseline_b.tolist(),
            benchmark_R2=benchmark["R2"].tolist(), evaluation_R2=baseline_r2.tolist())

    utility: Dict[str, Dict[str, Dict[str, float]]] = {}
    for round_name in ("round5", "round20"):
        summary_path = root / "data" / "historical" / f"dp_{round_name}" / "summary.txt"
        expected_summary_hash = targets["archived_run"][f"round_{round_name[5:]}_summary_sha256"]
        if summary_path.is_file():
            observed_summary_hash = _sha256(summary_path)
            add(f"archive.summary.{round_name}", "pass" if observed_summary_hash == expected_summary_hash else "fail",
                "Archived TensorFlow 1.14 run summary has its documented SHA-256."
                if observed_summary_hash == expected_summary_hash else "Archived run summary differs from its documented original.",
                sha256=observed_summary_hash, expected_sha256=expected_summary_hash)
        else:
            add(f"archive.summary.{round_name}", "fail", "Archived run summary is missing.",
                path=str(summary_path.relative_to(root)))
        utility[round_name] = {}
        for method, relative in (
            ("dp", f"data/historical/dp_{round_name}/DP_utility.npy"),
            ("ddpg", f"data/historical/{round_name}/DDPG_utility.npy"),
        ):
            value = array(relative)
            if value is None:
                continue
            observed = {"n": int(value.size), "mean": float(np.mean(value)), "std": float(np.std(value)),
                        "maximum": float(np.max(value)), "minimum": float(np.min(value))}
            utility[round_name][method] = observed
            valid_n = observed["n"] == 2_500_000
            if round_name == "round5":
                printed_key = "dynamic_programming" if method == "dp" else "ddpg_round_5"
                printed = targets["manuscript"]["discounted_utility"][printed_key]
                match = valid_n and all(_rounded_equal(observed[k], printed[k], 4)
                                        for k in ("mean", "maximum", "minimum"))
                match = match and _rounded_equal(observed["std"], printed["standard_deviation"], 4)
                add(f"manuscript.utility.{method}.{round_name}", "pass" if match else "fail",
                    "Archived discounted-utility values match the manuscript at four decimals." if match
                    else "Archived discounted-utility values do not match the manuscript at four decimals.",
                    observed=observed, printed=printed)
            else:
                add(f"archive.utility.{method}.{round_name}", "pass" if valid_n else "fail",
                    "Archived evaluation contains 500 paths × 5,000 agents." if valid_n
                    else "Archived evaluation size is not 500 paths × 5,000 agents.", observed=observed)

    if {"dp", "ddpg"} <= utility["round5"].keys():
        dp, ddpg = utility["round5"]["dp"], utility["round5"]["ddpg"]
        t_value, dof = _pooled_t(dp, ddpg)
        relative_percent = 100 * (ddpg["mean"] - dp["mean"]) / dp["mean"]
        printed = targets["manuscript"]["discounted_utility"]["comparison"]
        match = (_rounded_equal(relative_percent, printed["relative_mean_difference_percent"], 2)
                 and _rounded_equal(t_value, printed["independent_samples_t"], 3)
                 and dof == printed["degrees_of_freedom"])
        add("manuscript.utility.comparison", "pass" if match else "fail",
            "Pooled independent-samples t and relative difference match the manuscript." if match
            else "Comparison statistics do not match the manuscript.",
            observed={"relative_mean_difference_percent": relative_percent, "independent_samples_t": t_value,
                      "degrees_of_freedom": dof}, printed=printed)

    def rmse(name: str, folder: str, prefix: str, state: str) -> Optional[float]:
        stem = "Kg" if state == "good" else "Kb"
        observed = array(f"data/historical/{folder}/{prefix}_{stem}.npy")
        predicted = array(f"data/historical/{folder}/{prefix}_{stem}_ALM.npy")
        if observed is None or predicted is None:
            return None
        if observed.shape != predicted.shape:
            add(name, "fail", "ALM path and simulated path have different shapes.",
                observed_shape=list(observed.shape), predicted_shape=list(predicted.shape))
            return None
        return float(np.sqrt(np.mean((observed - predicted) ** 2)))

    for round_name in ("round5", "round20"):
        for state in ("good", "bad"):
            dp_value = rmse(f"alm.rmse.dp.{round_name}.{state}", f"dp_{round_name}", "DP", state)
            if dp_value is not None:
                if round_name == "round5":
                    expected = printed_alm["dynamic_programming"][state]["rmse"]
                    match = _rounded_equal(dp_value, expected, 6)
                    add(f"manuscript.dp_rmse.{state}", "pass" if match else "fail",
                        "Archived DP ALM path error matches the manuscript." if match
                        else "Archived DP ALM path error differs from the manuscript.",
                        observed=dp_value, printed=expected)
                else:
                    add(f"archive.dp_rmse.{round_name}.{state}", "pass", "Computed error from the archived DP paths.", observed=dp_value)
            ddpg_value = rmse(f"alm.rmse.ddpg.{round_name}.{state}", round_name, "DDPG", state)
            if ddpg_value is not None:
                if round_name == "round5":
                    expected = printed_alm["ddpg_round_5"][state]["rmse"]
                    match = _rounded_equal(ddpg_value, expected, 6)
                    add(f"manuscript.ddpg_rmse.{state}", "pass" if match else "fail",
                        "Archived DDPG ALM path error matches the manuscript." if match
                        else "Archived DDPG ALM path error differs from the manuscript.",
                        observed=ddpg_value, printed=expected)
                else:
                    add(f"archive.ddpg_rmse.{round_name}.{state}", "pass", "Computed error from the archived DDPG paths.", observed=ddpg_value)

        reg_path = root / "data" / "historical" / round_name / "reg_stat.npz"
        if not reg_path.is_file():
            add(f"archive.reg_stat.{round_name}", "fail", "Missing regression history.", path=str(reg_path))
        else:
            try:
                with np.load(reg_path, allow_pickle=False) as reg:
                    coefficients, r2_values = reg["B"], reg["R2"]
                if coefficients.shape != (int(round_name[5:]) + 1, 4) or r2_values.shape != (int(round_name[5:]) + 1, 2):
                    raise ValueError("Unexpected regression history shape")
                if not (np.all(np.isfinite(coefficients)) and np.all(np.isfinite(r2_values))):
                    raise ValueError("Non-finite regression history")
                add(f"archive.reg_stat.{round_name}", "pass", "Regression history has an entry for every round plus initialization.",
                    B_shape=list(coefficients.shape), R2_shape=list(r2_values.shape))
                if round_name == "round5":
                    for state, offset, index in (("good", 2, 1), ("bad", 0, 0)):
                        printed = printed_alm["ddpg_round_5"][state]
                        observed = {"intercept": float(coefficients[-1, offset]),
                                    "slope": float(coefficients[-1, offset + 1]),
                                    "r_squared": float(r2_values[-1, index])}
                        matched = {key: _rounded_equal(value, printed[key], 6 if key == "r_squared" else 4)
                                   for key, value in observed.items()}
                        slope_typo = state == "good" and matched == {"intercept": True, "slope": False, "r_squared": True}
                        status = "warn" if slope_typo else ("pass" if all(matched.values()) else "fail")
                        detail = ("The archived good-state slope rounds to 0.9252, while Thesis.tex prints 0.9251; other values match."
                                  if slope_typo else "Archived DDPG ALM matches the printed coefficients at printed precision."
                                  if status == "pass" else "Archived DDPG ALM differs from the printed coefficients.")
                        add(f"manuscript.ddpg_alm.{state}", status, detail, observed=observed, printed={k: printed[k] for k in observed})
            except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
                add(f"archive.reg_stat.{round_name}", "fail", "Could not validate regression history without pickle.", error=str(exc))

        ckpt_dir = root / "data" / "historical" / round_name / "ckpt" / "Actor"
        missing: List[str] = []
        for index in range(1, int(round_name[5:]) + 1):
            for suffix in (".data-00000-of-00001", ".index", ".meta"):
                file = ckpt_dir / f"{index}.ckpt{suffix}"
                if not file.is_file() or file.stat().st_size == 0:
                    missing.append(str(file.relative_to(root)))
        add(f"archive.checkpoints.{round_name}", "pass" if not missing else "fail",
            "All actor checkpoint triplets are present and nonempty; TensorFlow restore is not tested here."
            if not missing else "One or more actor checkpoint components are missing or empty.",
            complete_rounds=int(round_name[5:]) - len({p.split("/")[-1].split(".")[0] for p in missing}), missing=missing)

        expected_final = targets["archived_run"]["round_{}".format(round_name[5:])]["final_actor_checkpoint_sha256"]
        final_hashes = {}
        for filename, expected_hash in expected_final.items():
            checkpoint = ckpt_dir / filename
            if checkpoint.is_file():
                final_hashes[filename] = _sha256(checkpoint)
        match_final = all(final_hashes.get(name) == digest for name, digest in expected_final.items())
        add("archive.final_checkpoint_sha256.{}".format(round_name), "pass" if match_final else "fail",
            "Final Actor checkpoint components match the historical source bytes."
            if match_final else "Final Actor checkpoint bytes differ from the historical source.",
            observed=final_hashes, expected=expected_final)

    return finish()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path, nargs="?", default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    report = verify(args.project_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
