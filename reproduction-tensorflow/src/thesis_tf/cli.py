"""Run the historical TensorFlow scripts in isolated experiment directories.

The computational scripts in :mod:`thesis_tf.legacy` are intentionally kept
close to the thesis-era source. This module only resolves configuration,
copies immutable reference inputs, and launches a script in a separate process.
Importing this module does not import TensorFlow.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROUND_FOLDERS = {
    5: "round_5_life_5000_step_30_pop_100_idx_30",
    20: "round_20_life_5000_step_30_pop_100_idx_50",
}
ROUND_INDEX = {5: 30, 20: 50}
SCRIPT_FOR_STAGE = {
    "train": "main.py",
    "evaluate": "main.py",
    "compare": "plot_ddpg_dp.py",
    "dp": "main_dp.py",
}


def _parser():
    parser = argparse.ArgumentParser(
        prog="python -m thesis_tf.cli",
        description="Reproduce thesis results with the historical TensorFlow 1.14 scripts.",
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for stage in ("verify", "train", "evaluate", "compare", "dp"):
        sub = subparsers.add_parser(stage)
        sub.add_argument(
            "--config",
            default="configs/thesis.json",
            help="JSON experiment configuration, relative to the project root by default",
        )
        sub.add_argument(
            "--rounds", type=int, choices=(5, 20),
            help="Use the five- or twenty-round thesis experiment",
        )
        if stage != "verify":
            sub.add_argument(
                "--run-dir",
                help="Output directory (default: runs/<stage>-round<N>)",
            )
            sub.add_argument(
                "--resimulate", action="store_true",
                help="Regenerate simulated DP/comparison paths instead of replaying saved paths",
            )
            labels = sub.add_mutually_exclusive_group()
            labels.add_argument("--english", dest="english", action="store_true",
                                help="Use English plot labels (the default)")
            labels.add_argument("--chinese", dest="english", action="store_false",
                                help="Use the original Chinese plot labels")
            sub.set_defaults(english=True)
    return parser


def _resolve_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _load_config(value):
    path = _resolve_path(value)
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object: {}".format(path))
    return config


def _copy_missing(source, destination):
    """Copy regular files without replacing any file in an existing run."""
    if not source.is_dir():
        return 0
    copied = 0
    for item in sorted(source.rglob("*")):
        if item.is_symlink() or not item.is_file():
            continue
        target = destination / item.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with item.open("rb") as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer)
            copied += 1
        except FileExistsError:
            continue
    return copied


def _stage_data(run_dir, stage, rounds, fresh_training=False):
    """Place historical inputs where the original cwd-relative scripts expect them."""
    data = PROJECT_ROOT / "data"
    test = run_dir / "test"
    dp = test / "dynamic_programming"
    copied = _copy_missing(data / "benchmark", dp)
    copied += _copy_missing(data / "historical" / "dp_base", dp)

    # A fresh training run must not start with archived actor weights or scores.
    if stage in ("train", "dp"):
        return copied

    if not fresh_training:
        copied += _copy_missing(
            data / "historical" / "round{}".format(rounds),
            test / ROUND_FOLDERS[rounds],
        )

    if stage == "compare" and not fresh_training:
        copied += _copy_missing(data / "historical" / "dp_round{}".format(rounds), dp)
    return copied


def _verify():
    from .verify import verify

    result = verify(PROJECT_ROOT)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    # The verifier marks documentary caveats as warnings; only failed checks
    # make ``ok`` false and set a nonzero exit status.
    return 0 if result.get("ok", False) else 1


def _run_stage(args, config):
    rounds = args.rounds if args.rounds is not None else int(config.get("n_round", 5))
    if rounds not in ROUND_FOLDERS and args.stage in ("evaluate", "compare"):
        raise ValueError("Archived evaluation is available only for 5 or 20 rounds")
    idx = int(config.get("idx", ROUND_INDEX.get(rounds, 0)))
    if args.rounds is not None:
        idx = ROUND_INDEX[rounds]

    run_dir = _resolve_path(args.run_dir or "runs/{}-round{}".format(args.stage, rounds))
    protected = (PROJECT_ROOT / "data", PROJECT_ROOT / "src",
                 PROJECT_ROOT / "configs", PROJECT_ROOT / "targets")
    if run_dir == PROJECT_ROOT or any(run_dir == p or p in run_dir.parents for p in protected):
        raise ValueError("Run directory must not be the project root or an input/source directory")

    script = PROJECT_ROOT / "src" / "thesis_tf" / "legacy" / SCRIPT_FOR_STAGE[args.stage]
    if not script.is_file():
        raise FileNotFoundError("Historical script is missing: {}".format(script))

    marker = run_dir / "run_metadata.json"
    provenance = None
    if marker.is_file():
        with marker.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        if not isinstance(metadata, dict):
            raise ValueError("Invalid run metadata: {}".format(marker))
        provenance = metadata.get("provenance")
        if int(metadata.get("n_round", -1)) != rounds:
            raise ValueError("Existing run has a different round count: {}".format(run_dir))
    fresh_training = provenance == "fresh_training"
    if args.stage == "compare" and fresh_training and not args.resimulate:
        raise ValueError("Comparing a fresh training run requires --resimulate; archived metrics must not be mixed in")
    if args.stage in ("evaluate", "compare") and not fresh_training:
        if provenance not in (None, "archival_replay"):
            raise ValueError("Run provenance is not suitable for archive replay: {}".format(run_dir))
        canonical = (int(config.get("n_life", 5000)) == 5000
                     and int(config.get("n_steps", 30)) == 30
                     and int(config.get("n_pop", 100)) == 100
                     and idx == ROUND_INDEX[rounds])
        if not canonical:
            raise ValueError("Archive replay requires historical n_life=5000, n_steps=30, n_pop=100 and canonical idx")
        historical_target = run_dir / "test" / ROUND_FOLDERS[rounds]
        if provenance is None and historical_target.exists() and any(historical_target.rglob("*")):
            raise ValueError("Existing unmarked results may be fresh; select a new --run-dir to avoid mixing them with archives")

    folder = "round_{}_life_{}_step_{}_pop_{}_idx_{}".format(
        rounds, int(config.get("n_life", 5000)), int(config.get("n_steps", 30)),
        int(config.get("n_pop", 100)), idx,
    )
    if args.stage == "train":
        target = run_dir / "test" / folder
        if marker.exists() or (target / "reg_stat.npz").exists() or (target / "score_history.npy").exists():
            raise ValueError("Training results already exist in {}; select a new --run-dir".format(run_dir))

    run_dir.mkdir(parents=True, exist_ok=True)
    copied = _stage_data(run_dir, args.stage, rounds, fresh_training)
    if args.stage in ("evaluate", "compare") and provenance is None:
        with marker.open("x", encoding="utf-8") as stream:
            json.dump({"provenance": "archival_replay", "n_round": rounds,
                       "configuration": config}, stream, indent=2, sort_keys=True)
            stream.write("\n")
    if args.stage == "train":
        (target / "ckpt" / "Actor").mkdir(parents=True, exist_ok=True)
        with marker.open("x", encoding="utf-8") as stream:
            json.dump({"provenance": "fresh_training", "n_round": rounds,
                       "configuration": config}, stream, indent=2, sort_keys=True)
            stream.write("\n")
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["THESIS_TF_N_ROUND"] = str(rounds)
    env["THESIS_TF_IDX"] = str(idx)
    env["THESIS_TF_TRAIN"] = "1" if args.stage == "train" else "0"
    env["THESIS_TF_RESIMULATE"] = "1" if args.resimulate else "0"
    env["THESIS_TF_ENGLISH"] = "1" if args.english else "0"
    for key in ("n_pop", "n_steps", "n_life", "n_life_0", "simu_pop",
                "simu_round", "K_0", "seed"):
        if key in config:
            env["THESIS_TF_{}".format(key.upper())] = str(config[key])

    print("Stage: {} ({} rounds)".format(args.stage, rounds), flush=True)
    print("Run directory: {}".format(run_dir), flush=True)
    print("Reference files staged: {}".format(copied), flush=True)
    return subprocess.call([sys.executable, str(script)], cwd=str(run_dir), env=env)


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        if args.stage == "verify":
            return _verify()
        return _run_stage(args, _load_config(args.config))
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print("Error: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
