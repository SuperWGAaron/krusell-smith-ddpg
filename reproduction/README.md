# PyTorch DDPG for the Krusell–Smith economy

This directory contains a modern, import-safe PyTorch implementation of reinforcement learning for household saving decisions in the Krusell–Smith economy. It packages the economic environment, the DDPG/fictitious-play solver, the dynamic-programming reference path, evaluation, plotting, and verification behind one CLI: `ks-reproduce`. The historical research experiment is represented by the bundled configuration, benchmark inputs, and machine-readable numerical targets; no manuscript files or external source checkout are required.

The project makes an important distinction:

- **Pipeline completed** means the requested stages ran and produced their required files.
- **Numerically verified** means the resulting metrics also match every required manuscript target in `targets/thesis_results.json` under its declared strict comparison.

A completed smoke run is not numerical reproduction of the historical experiment. A completed full run can also fail numerical verification. The published DP cache is bundled here. Historical TensorFlow Actor checkpoints, histories, and simulation samples are preserved separately in [the TensorFlow implementation](../reproduction-tensorflow/data/README.md); this PyTorch workflow trains new weights and does not automatically import or compare those checkpoints.

## Project structure

| Path | Purpose |
| --- | --- |
| `src/ks_reproduction/` | Economic environment, DDPG networks and agent, fictitious-play training, simulation, evaluation, and CLI |
| `configs/` | Full historical profile, fast smoke profile, and robustness experiments |
| `data/benchmark/` | Four DP reference arrays and their integrity checksums |
| `targets/thesis_results.json` | Historical numerical targets, comparison tolerances, and source provenance |
| `docs/` | Numerical compatibility notes |
| `outputs/` | Locally generated runs, checkpoints, metrics, and figures; excluded from version control |
| `pyproject.toml`, `uv.lock` | Package dependencies and locked environment specification |

The sibling `../reproduction-tensorflow/` provides the updated TensorFlow workflow and preserved historical evidence. It has its own environment and is not required to run this PyTorch package.

## Install

From this directory, create an isolated environment and install the package in editable mode:

```bash
python3 -m venv .venv && .venv/bin/python -m pip install --upgrade pip && .venv/bin/python -m pip install -e .
```

Keep this checkout in place: the CLI resolves bundled configurations, targets, and benchmark data relative to it. Editable installation is the supported repository workflow.

Activate it if desired:

```bash
source .venv/bin/activate
```

PyTorch accelerator wheels are platform-specific. The default dependency declaration installs the wheel selected by pip for the current platform. For a particular CUDA runtime, install the appropriate PyTorch build first and then install this project. Every run records the actual Python, PyTorch, NumPy, device, CUDA, and deterministic-runtime information used.

## One-command workflows

Run these commands from the reproduction directory after installation.

Check dependencies, configured inputs, devices, and write permissions without training:

```bash
ks-reproduce doctor --config configs/thesis.toml
```

Run a small end-to-end experiment to check the installation:

```bash
ks-reproduce reproduce --config configs/smoke.toml
```

Run the full thesis profile, including round-5 and round-20 snapshots, evaluation, plots, and strict verification:

```bash
ks-reproduce reproduce --config configs/thesis.toml
```

Run all eleven robustness variants recorded in the thesis:

```bash
ks-reproduce reproduce --config configs/robustness.toml
```

The full profile is intentionally expensive. The thesis reports approximately 8 hours through round 5 and 30 hours through round 20 on a Tesla K80; modern runtime varies substantially by device. The evaluation profile processes 2.5 million simulated agents per method and should be streamed rather than held entirely on an accelerator.

## Stage-by-stage use

The orchestrating `reproduce` command is the normal path. Individual stages are available for diagnosis or resuming work:

```bash
ks-reproduce train --config configs/thesis.toml
ks-reproduce evaluate --config configs/thesis.toml --run-dir outputs/thesis/seed_0000
ks-reproduce plot --config configs/thesis.toml --run-dir outputs/thesis/seed_0000
ks-reproduce verify --config configs/thesis.toml --run-dir outputs/thesis/seed_0000
```

Use `--resume` with `train` or `reproduce` to continue the configured run from its last complete checkpoint. Existing completed runs are not overwritten silently.

The full round-20 training pass captures complete snapshots at rounds 5 and 20. A snapshot includes the Actor ensemble needed by fictitious play, the associated ALM, optimizer and target-network state, replay state, counters, histories, OU-noise state, and random-number-generator state. Generating both thesis checkpoints from one run avoids repeating the first five rounds.

`training.alm_tolerance` is logged as a convergence diagnostic, not used to stop the canonical run early. The artifact contract requires both configured round-5 and round-20 checkpoints, so continuing through those rounds is an explicit orchestration choice rather than silent convergence behavior.

## Configurations

- `configs/thesis.toml` is the legacy-compatible paper profile: seed 0, `verification_profile = "thesis_numerical"`, discount factor 0.8, stage-specific initial aggregate states, 20 training rounds, round-5 and round-20 snapshots, and the manuscript-scale evaluation. Utility and comparison paths start good (`1`); training and evaluation ALM paths start bad (`0`); the generic transition fallback is good (`1`).
- `configs/smoke.toml` uses the same equations and artifact pipeline with tiny dimensions and `verification_profile = "structural"`. Its verification checks invariants and artifact structure only. The tiny profile uses seed 1 because seed 0 produces a degenerate all-zero Actor under the legacy initializer at this reduced scale; the thesis profile remains seed 0.
- `configs/robustness.toml` extends the thesis profile with the eleven variants from the robustness table. Every materialized variant explicitly receives `verification_profile = "structural"`. The thesis provides only qualitative `close` or `fail` outcomes, not underlying numbers or a classification threshold; the suite therefore reports new metrics without claiming automatic historical verification.

Robustness test 11 is explicitly an Actor-only operationalization of the manuscript: it pretrains toward a constant saving ratio of `0.9`. The retained TensorFlow helper instead returns `0.7`, and its legacy orchestration subsequently pretrains the Critic as well. Because neither the historical pretraining artifacts nor a complete stopping-state specification survives, this variant is not an exact replay of the original robustness run.

Paths in configuration files are resolved relative to this reproduction directory, never the caller's current working directory. CLI values override configuration values. Unknown keys and invalid parameter combinations should fail before a run directory is created.

Verification semantics come only from the explicit top-level `verification_profile`, whose allowed values are `thesis_numerical` and `structural`. `experiment_name` controls naming and output placement; renaming an experiment cannot silently switch its verification standard.

## Expected run output

The configured full run is written beneath `outputs/thesis/seed_0000/`:

```text
config.resolved.toml
run.json
provenance.json
logs/events.jsonl
checkpoints/round_0005.pt
checkpoints/round_0020.pt
checkpoints/last.pt
data/*.npz
metrics/summary.json
metrics/summary.csv
metrics/thesis_table.tex
figures/DP/*.jpg
figures/round_5/*.jpg
figures/round_20/*.jpg
verification/report.json
verification/report.md
manifest.sha256
```

`run.json` carries separate pipeline and verification states. The verification command returns a failing status when required numerical targets miss their tolerances, while preserving the generated evidence and report.

The 23 numerical figure filenames are part of the artifact contract. Exact JPEG hashes are not: Matplotlib, fonts, and JPEG encoders can change pixels without changing the underlying result. Verification operates on saved arrays and metrics; figure checks confirm required files and basic render properties.

## Reproducibility controls

The runtime seeds Python, NumPy, Torch CPU, and all CUDA devices. It uses named random streams for model initialization, environment shocks, replay sampling, exploration noise, ALM simulations, and evaluation so that diagnostic or plotting order cannot alter training. Full checkpoints store these states for exact continuation.

Initial aggregate Markov states are configured per stage. The retained evaluation imported separate stateful chains: the DP utility path used the chain created through `parameters_dp`, while ALM simulation used the chain owned by the `parameters_ddpg`-backed `MFG`. Each chain sampled one default `x0` before the evaluation script called `np.random.seed(0)`, then reused it. Neither pre-seed NumPy state nor either sampled `x0` was recorded. A full 2.5-million-observation DP audit strongly supports good (`1`) for utility paths, while the printed ALM RMSE pattern is consistent with a separate bad (`0`) start. The canonical profile therefore uses good for utility and comparison paths, bad for training and evaluation ALM paths, and good only as the generic transition fallback. These are evidence-based stage reconstructions, not recovered historical RNG states; strict statistics may still miss their manuscript tolerances.

Deterministic algorithms and deterministic cuDNN settings are requested by the thesis profile. CPU is the most portable reference device. Cross-device bitwise identity is not promised, particularly for CUDA and MPS, and the device is always part of run provenance.

Neural-network computation uses `float32`; economic simulation, interpolation, regression, and reported statistics use `float64`. This prevents GPU training defaults from silently reducing the precision of the numerical benchmark.

## What can and cannot be verified

The scalar values transcribed from the thesis include:

- discounted-utility mean, standard deviation, extrema, relative difference, and t statistic;
- good- and bad-state ALM intercepts, slopes, R² values, and RMSE values for DP and round-5 DDPG;
- required sample sizes, snapshots, and numerical figure paths;
- all eleven qualitative robustness outcomes.

`data/benchmark/` contains the recovered Julia-generated `(0.99, 1.01)` DP family used by the historical experiment. Its `B.npy` and `R2.npy` values round to the printed DP ALM coefficients and R² values, and `SHA256SUMS` pins all four inputs. A later `(0.9, 1.1)` cache family has different coefficients; its selected `B.npy` and `R2.npy` files are retained for audit in `../reproduction-tensorflow/data/historical/dp_base/` and are not inputs to this pipeline.

Selected historical TensorFlow Actor checkpoints, score and ALM histories, utility samples, and comparison paths are bundled in `../reproduction-tensorflow/data/historical/`, with their provenance documented there. They support historical replay through the TensorFlow workflow. This PyTorch rewrite produces independent training runs; no cross-framework checkpoint conversion or weight-for-weight equivalence has been established. A verified benchmark checksum establishes input provenance, not DDPG numerical verification.

Likewise, fixing the stage-specific initial states makes new evaluation deterministic but does not recover either legacy import-time draw. Good for utility/comparison and bad for ALM are the settings best supported by the audits, yet they do not guarantee that every strict published statistic will match.

See `docs/LEGACY_COMPATIBILITY.md` for preserved or corrected historical behavior.
