"""Command-line orchestration for auditable thesis reproduction runs."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifacts import RunLayout, atomic_write_json, provenance, sha256, write_manifest
from .config import ReproductionConfig, load_config
from .dp import DPReference, dp_metadata
from .evaluation import ThesisEvaluator
from .runtime import resolve_device
from .training import FictitiousPlayTrainer
from .verification import verify, write_verification

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "thesis.toml"
BENCHMARK_DIRECTORY = PROJECT_ROOT / "data" / "benchmark"
TARGETS_PATH = PROJECT_ROOT / "targets" / "thesis_results.json"


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _config_with_cli_overrides(
    config: ReproductionConfig,
    *,
    seed: int | None,
    device: str | None,
) -> ReproductionConfig:
    runtime = replace(
        config.runtime,
        seed=config.runtime.seed if seed is None else int(seed),
        device=config.runtime.device if device is None else str(device),
    )
    updated = replace(config, runtime=runtime)
    updated.validate()
    return updated


def _default_run_directory(config: ReproductionConfig) -> Path:
    return PROJECT_ROOT / "outputs" / config.experiment_name / f"seed_{config.runtime.seed:04d}"


def _run_directory(config: ReproductionConfig, requested: str | None) -> Path:
    return _default_run_directory(config) if requested is None else _project_path(requested)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return value


def _update_run(
    layout: RunLayout,
    *,
    stage: str | None = None,
    pipeline_status: str | None = None,
    verification_status: str | None = None,
    verification_profile: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = layout.root / "run.json"
    state = _read_json(path)
    old_pipeline = state.get("pipeline", {})
    stages: dict[str, Any] = {}
    if isinstance(old_pipeline, Mapping):
        old_stages = old_pipeline.get("stages", {})
        if isinstance(old_stages, Mapping):
            stages.update(old_stages)
        for key, value in old_pipeline.items():
            if key not in {"status", "stages"}:
                stages.setdefault(str(key), value)
    if stage is not None:
        stages[stage] = "completed"
    state["pipeline"] = {
        "status": pipeline_status or (
            old_pipeline.get("status", "in_progress")
            if isinstance(old_pipeline, Mapping)
            else "in_progress"
        ),
        "stages": stages,
    }
    old_verification = state.get("verification", {})
    verification_state = (
        dict(old_verification) if isinstance(old_verification, Mapping) else {}
    )
    if verification_status is not None:
        verification_state["status"] = verification_status
    else:
        verification_state.setdefault("status", "not_run")
    if verification_profile is not None:
        verification_state["profile"] = verification_profile
    state["verification"] = verification_state
    if extra:
        state.update(extra)
    atomic_write_json(path, state)
    return state


def _check_resolved_config(layout: RunLayout, config: ReproductionConfig) -> None:
    path = layout.root / "config.resolved.toml"
    if path.is_file() and load_config(path).to_dict() != config.to_dict():
        raise ValueError(
            f"requested configuration differs from {path}; use the run's resolved config"
        )


def _write_run_metadata(layout: RunLayout, config: ReproductionConfig) -> None:
    config.write_toml(layout.root / "config.resolved.toml")
    device = resolve_device(config.runtime.device)
    atomic_write_json(
        layout.root / "provenance.json",
        provenance(str(device), config.runtime.deterministic_torch),
    )


def _train(config: ReproductionConfig, layout: RunLayout, *, resume: bool) -> None:
    trainer = FictitiousPlayTrainer(config, layout, resume=resume)
    _write_run_metadata(layout, config)
    _update_run(layout, pipeline_status="in_progress")
    trainer.train()
    _update_run(
        layout,
        stage="training",
        pipeline_status="in_progress",
        extra={
            "completed_round": trainer.completed_round,
            "episodes": len(trainer.discounted_returns),
        },
    )


def _evaluate(config: ReproductionConfig, layout: RunLayout) -> dict[str, Any]:
    _check_resolved_config(layout, config)
    layout.create(allow_existing=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(layout.logs / "cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(layout.logs / "matplotlib-cache"))
    result = ThesisEvaluator(config, layout, BENCHMARK_DIRECTORY).evaluate()
    _update_run(layout, stage="evaluation", pipeline_status="in_progress")
    _update_run(layout, stage="plotting", pipeline_status="in_progress")
    return result


def _plot(config: ReproductionConfig, layout: RunLayout) -> None:
    _check_resolved_config(layout, config)
    os.environ.setdefault("XDG_CACHE_HOME", str(layout.logs / "cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(layout.logs / "matplotlib-cache"))
    ThesisEvaluator(config, layout, BENCHMARK_DIRECTORY).plot_saved()
    _update_run(layout, stage="plotting", pipeline_status="in_progress")


def _verify(
    config: ReproductionConfig,
    layout: RunLayout,
    *,
    numerical: bool,
) -> dict[str, Any]:
    _check_resolved_config(layout, config)
    report = verify(
        metrics_path=layout.metrics / "summary.json",
        targets_path=TARGETS_PATH,
        run_root=layout.root,
        benchmark_directory=BENCHMARK_DIRECTORY,
        numerical=numerical,
    )
    write_verification(
        report,
        json_path=layout.verification / "report.json",
        markdown_path=layout.verification / "report.md",
    )
    status = "passed" if report["verification_passed"] else "failed"
    _update_run(
        layout,
        stage="verification",
        pipeline_status="completed",
        verification_status=status,
        verification_profile=str(report["profile"]),
    )
    return report


def _reproduce(
    config: ReproductionConfig,
    layout: RunLayout,
    *,
    resume: bool,
    numerical: bool,
) -> dict[str, Any]:
    try:
        _train(config, layout, resume=resume)
        _evaluate(config, layout)
        report = _verify(config, layout, numerical=numerical)
        write_manifest(layout.root)
        return report
    except Exception as error:
        if layout.root.is_dir():
            _update_run(
                layout,
                pipeline_status="failed",
                extra={"error": f"{type(error).__name__}: {error}"},
            )
        raise


def _set_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    if len(parts) < 2:
        raise ValueError(f"suite override must name a section and field: {dotted!r}")
    current: dict[str, Any] = data
    for part in parts[:-1]:
        nested = current.get(part)
        if not isinstance(nested, dict):
            raise ValueError(f"suite override refers to an unknown section: {dotted!r}")
        current = nested
    if parts[-1] not in current:
        raise ValueError(f"suite override refers to an unknown field: {dotted!r}")
    current[parts[-1]] = value


def _apply_overrides(
    config: ReproductionConfig,
    overrides: Mapping[str, Any],
) -> ReproductionConfig:
    data = copy.deepcopy(config.to_dict())
    for key, value in overrides.items():
        _set_dotted(data, str(key), value)
    return ReproductionConfig.from_dict(data)


def _suite_variants(
    suite_path: Path,
    *,
    seed: int | None,
    device: str | None,
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], ReproductionConfig]]]:
    suite = _read_toml(suite_path)
    if suite.get("suite_type") != "robustness":
        raise ValueError(f"unsupported suite type in {suite_path}")
    base_path = _project_path(str(suite["base_config"]))
    base = _apply_overrides(load_config(base_path), suite.get("base_overrides", {}))
    variants: list[tuple[dict[str, Any], ReproductionConfig]] = []
    for raw_variant in suite.get("variants", []):
        if not isinstance(raw_variant, Mapping):
            raise ValueError("each robustness variant must be a table")
        variant = dict(raw_variant)
        configured = _apply_overrides(base, variant.get("overrides", {}))
        seeds = variant.get("seed_values", [configured.runtime.seed])
        if seed is not None:
            seeds = [seed]
        for variant_seed in seeds:
            runtime = replace(
                configured.runtime,
                seed=int(variant_seed),
                device=configured.runtime.device if device is None else device,
            )
            named = replace(
                configured,
                experiment_name=f"robustness_{variant['id']}",
                verification_profile="structural",
                runtime=runtime,
            )
            named.validate()
            variants.append((variant, named))
    return suite, variants


def _run_suite(args: argparse.Namespace, suite_path: Path) -> int:
    suite, variants = _suite_variants(
        suite_path,
        seed=args.seed,
        device=args.device,
    )
    output_root = (
        _project_path(args.run_dir)
        if args.run_dir is not None
        else _project_path(str(suite.get("output_root", "outputs/robustness")))
    )
    records: list[dict[str, Any]] = []
    for variant, config in variants:
        run_root = output_root / str(variant["id"]) / f"seed_{config.runtime.seed:04d}"
        record = {
            "id": variant["id"],
            "test_number": variant.get("test_number"),
            "seed": config.runtime.seed,
            "reported_outcome": variant.get("reported_outcome"),
            "run_directory": str(run_root),
        }
        print(f"[{variant['id']} seed={config.runtime.seed}] {run_root}", flush=True)
        try:
            report = _reproduce(
                config,
                RunLayout(run_root),
                resume=args.resume,
                numerical=False,
            )
            record["pipeline_status"] = "completed"
            record["verification_status"] = (
                "passed" if report["verification_passed"] else "failed"
            )
        except Exception as error:  # retain other variants and the failure evidence
            record["pipeline_status"] = "failed"
            record["error"] = f"{type(error).__name__}: {error}"
        records.append(record)
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        output_root / "suite_summary.json",
        {
            "suite_name": suite.get("suite_name"),
            "historical_classification": suite.get("historical_classification"),
            "runs": records,
            "historical_numerical_verification": "not_verifiable",
        },
    )
    failed = [record for record in records if record["pipeline_status"] != "completed"]
    print(f"Robustness suite finished: {len(records) - len(failed)}/{len(records)} completed")
    return 2 if failed else 0


def _checksum_report() -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    checksum_path = BENCHMARK_DIRECTORY / "SHA256SUMS"
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, filename = line.split(maxsplit=1)
        path = BENCHMARK_DIRECTORY / filename.strip()
        actual = sha256(path) if path.is_file() else None
        report.append(
            {
                "path": str(path),
                "expected": expected,
                "actual": actual,
                "status": "pass" if actual == expected else "fail",
            }
        )
    return report


def _doctor(config_path: Path, *, seed: int | None, device: str | None) -> int:
    raw = _read_toml(config_path)
    if raw.get("suite_type") == "robustness":
        _, variants = _suite_variants(config_path, seed=seed, device=device)
        configs_validated = len(variants)
        config = variants[0][1]
    else:
        config = _config_with_cli_overrides(
            load_config(config_path), seed=seed, device=device
        )
        configs_validated = 1
    checksums = _checksum_report()
    reference = DPReference.load(BENCHMARK_DIRECTORY)
    requested_device = resolve_device(config.runtime.device)
    with tempfile.NamedTemporaryFile(prefix=".doctor-", dir=PROJECT_ROOT):
        writable = True
    result = {
        "status": "pass" if all(item["status"] == "pass" for item in checksums) else "fail",
        "project_root": str(PROJECT_ROOT),
        "config": str(config_path),
        "configs_validated": configs_validated,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "requested_device": config.runtime.device,
        "resolved_device": str(requested_device),
        "project_root_writable": writable,
        "benchmark": dp_metadata(reference),
        "checksums": checksums,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


def _add_config_arguments(parser: argparse.ArgumentParser, *, run_dir: bool) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="TOML config or suite")
    if run_dir:
        parser.add_argument("--run-dir", help="output run directory, relative to project root")
    parser.add_argument("--seed", type=int, help="override runtime.seed")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        help="override runtime.device",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ks-reproduce",
        description="PyTorch reproduction of the Krusell-Smith thesis experiments",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="validate environment, config, and inputs")
    _add_config_arguments(doctor, run_dir=False)

    train = subparsers.add_parser("train", help="run DDPG/fictitious-play training")
    _add_config_arguments(train, run_dir=True)
    train.add_argument("--resume", action="store_true", help="resume from checkpoints/last.pt")

    evaluate = subparsers.add_parser("evaluate", help="evaluate checkpoints and make figures")
    _add_config_arguments(evaluate, run_dir=True)

    plot = subparsers.add_parser("plot", help="regenerate figures from saved evaluation arrays")
    _add_config_arguments(plot, run_dir=True)

    verify_parser = subparsers.add_parser("verify", help="verify metrics and artifacts")
    _add_config_arguments(verify_parser, run_dir=True)

    reproduce = subparsers.add_parser("reproduce", help="run train, evaluate, plot, and verify")
    _add_config_arguments(reproduce, run_dir=True)
    reproduce.add_argument(
        "--resume", action="store_true", help="resume a single run or every suite member"
    )
    return parser


def _single_config(args: argparse.Namespace, path: Path) -> ReproductionConfig:
    raw = _read_toml(path)
    if raw.get("suite_type") is not None:
        raise ValueError("suite configs are accepted only by doctor and reproduce")
    return _config_with_cli_overrides(
        load_config(path),
        seed=args.seed,
        device=args.device,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = _project_path(args.config)
    try:
        if args.command == "doctor":
            return _doctor(config_path, seed=args.seed, device=args.device)
        raw = _read_toml(config_path)
        if args.command == "reproduce" and raw.get("suite_type") == "robustness":
            return _run_suite(args, config_path)

        config = _single_config(args, config_path)
        layout = RunLayout(_run_directory(config, args.run_dir))
        print(f"Run directory: {layout.root}", flush=True)
        if args.command == "train":
            _train(config, layout, resume=args.resume)
            completed_round = _read_json(layout.root / "run.json").get("completed_round")
            print(f"Training completed through round {completed_round}")
            return 0
        if args.command == "evaluate":
            _evaluate(config, layout)
            print(f"Evaluation and figures written to {layout.root}")
            return 0
        if args.command == "plot":
            _plot(config, layout)
            print(f"Figures regenerated under {layout.figures}")
            return 0
        numerical = config.verification_profile == "thesis_numerical"
        if args.command == "verify":
            report = _verify(config, layout, numerical=numerical)
            print(f"Verification status: {'passed' if report['verification_passed'] else 'failed'}")
            return 0 if report["verification_passed"] else 2
        if args.command == "reproduce":
            report = _reproduce(
                config,
                layout,
                resume=args.resume,
                numerical=numerical,
            )
            print(f"Pipeline completed; verification profile: {report['profile']}")
            print(f"Verification status: {'passed' if report['verification_passed'] else 'failed'}")
            return 0 if report["verification_passed"] else 2
        raise AssertionError(f"unhandled command: {args.command}")
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
