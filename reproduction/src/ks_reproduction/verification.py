"""Compare generated metrics and artifacts with explicit manuscript targets."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json, sha256


@dataclass(frozen=True)
class ScalarCheck:
    path: str
    status: str
    actual: float | int | None
    target: float | int
    absolute_error: float | None
    tolerance: float | None
    operator: str
    note: str | None = None


def _get_path(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError(".".join(keys))
        value = value[key]
    return value


def _target_leaves(
    value: Any,
    prefix: tuple[str, ...] = (),
) -> Iterator[tuple[tuple[str, ...], dict[str, Any]]]:
    if isinstance(value, Mapping) and "value" in value and (
        "absolute_tolerance" in value or "operator" in value
    ):
        yield prefix, dict(value)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _target_leaves(item, (*prefix, str(key)))


def _check_scalar(
    keys: tuple[str, ...], target: Mapping[str, Any], actual: Mapping[str, Any]
) -> ScalarCheck:
    path = ".".join(keys)
    expected = target["value"]
    operator = str(target.get("operator", "absolute_tolerance"))
    tolerance = target.get("absolute_tolerance")
    try:
        observed = _get_path(actual, keys)
    except KeyError:
        return ScalarCheck(
            path,
            "not_verifiable",
            None,
            expected,
            None,
            tolerance,
            operator,
            "missing metric",
        )
    if not isinstance(observed, (int, float)) or not math.isfinite(float(observed)):
        return ScalarCheck(
            path,
            "fail",
            observed if isinstance(observed, (int, float)) else None,
            expected,
            None,
            tolerance,
            operator,
            "metric is not a finite scalar",
        )
    error = abs(float(observed) - float(expected))
    if operator == "less_than":
        passed = float(observed) < float(expected)
    elif operator == "less_than_or_equal":
        passed = float(observed) <= float(expected)
    elif operator == "greater_than":
        passed = float(observed) > float(expected)
    elif operator == "absolute_tolerance":
        if tolerance is None:
            raise ValueError(f"target {path} lacks absolute_tolerance")
        passed = error <= float(tolerance)
    else:
        raise ValueError(f"unknown target operator {operator!r} at {path}")
    return ScalarCheck(
        path=path,
        status="pass" if passed else "fail",
        actual=observed,
        target=expected,
        absolute_error=error,
        tolerance=float(tolerance) if tolerance is not None else None,
        operator=operator,
    )


def verify(
    *,
    metrics_path: Path,
    targets_path: Path,
    run_root: Path,
    benchmark_directory: Path | None = None,
    numerical: bool = True,
) -> dict[str, Any]:
    targets = json.loads(targets_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    numerical_targets = targets.get("numerical_targets", {})
    checks = (
        [
            _check_scalar(keys, target, metrics)
            for keys, target in _target_leaves(numerical_targets)
        ]
        if numerical
        else []
    )
    required = targets.get("artifact_targets", {}).get("required_figures", [])
    artifacts = [
        {
            "path": relative,
            "status": "pass" if (run_root / relative).is_file() else "fail",
        }
        for relative in required
    ]
    benchmark: list[dict[str, str]] = []
    if benchmark_directory is not None:
        checksum_file = benchmark_directory / "SHA256SUMS"
        if checksum_file.is_file():
            for line in checksum_file.read_text(encoding="utf-8").splitlines():
                expected_hash, filename = line.split(maxsplit=1)
                filename = filename.strip()
                source = benchmark_directory / filename
                actual_hash = sha256(source) if source.is_file() else "missing"
                benchmark.append(
                    {
                        "path": filename,
                        "expected_sha256": expected_hash,
                        "actual_sha256": actual_hash,
                        "status": "pass" if actual_hash == expected_hash else "fail",
                    }
                )
        else:
            benchmark.append(
                {
                    "path": "SHA256SUMS",
                    "expected_sha256": "present",
                    "actual_sha256": "missing",
                    "status": "fail",
                }
            )
    required_checks = [check for check in checks if check.status != "not_verifiable"]
    numerically_verified = (
        bool(required_checks)
        and all(check.status == "pass" for check in required_checks)
        and not any(check.status == "not_verifiable" for check in checks)
        if numerical
        else None
    )
    artifacts_complete = all(item["status"] == "pass" for item in artifacts)
    inputs_verified = bool(benchmark) and all(
        item["status"] == "pass" for item in benchmark
    )
    report = {
        "profile": "thesis_numerical" if numerical else "structural",
        "pipeline_completed": True,
        "numerically_verified": numerically_verified,
        "artifacts_complete": artifacts_complete,
        "inputs_verified": inputs_verified,
        "verification_passed": bool(
            artifacts_complete
            and inputs_verified
            and (numerically_verified if numerical else True)
        ),
        "scalar_checks": [asdict(check) for check in checks],
        "artifact_checks": artifacts,
        "benchmark_checks": benchmark,
        "interpretation": (
            "A completed pipeline and verified inputs do not imply that the PyTorch run "
            "matches the manuscript. numerically_verified is true only when every printed "
            "scalar target passes its declared comparison."
            if numerical
            else "This reduced or robustness run checks pipeline structure, required figures, "
            "and benchmark integrity only; it does not claim numerical thesis reproduction."
        ),
    }
    return report


def report_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Reproduction verification",
        "",
        f"- Pipeline completed: **{report['pipeline_completed']}**",
        f"- Verification profile: **{report.get('profile', 'thesis_numerical')}**",
        f"- Verification passed: **{report.get('verification_passed', False)}**",
        f"- Numerical targets verified: **{report['numerically_verified']}**",
        f"- Figure set complete: **{report['artifacts_complete']}**",
        f"- Benchmark inputs verified: **{report['inputs_verified']}**",
        "",
        "## Scalar checks",
        "",
        "| Target | Status | Actual | Expected | Absolute error | Tolerance/operator |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for check in report.get("scalar_checks", []):
        criterion = (
            str(check.get("tolerance"))
            if check.get("operator") == "absolute_tolerance"
            else str(check.get("operator"))
        )
        lines.append(
            f"| `{check['path']}` | {check['status']} | {check.get('actual')} | "
            f"{check.get('target')} | {check.get('absolute_error')} | {criterion} |"
        )
    missing = [
        item["path"] for item in report.get("artifact_checks", []) if item["status"] != "pass"
    ]
    if missing:
        lines.extend(("", "## Missing figures", ""))
        lines.extend(f"- `{path}`" for path in missing)
    lines.extend(("", str(report.get("interpretation", "")), ""))
    return "\n".join(lines)


def write_verification(
    report: Mapping[str, Any], *, json_path: Path, markdown_path: Path
) -> None:
    atomic_write_json(json_path, report)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(report_markdown(report), encoding="utf-8")
