# Krusell–Smith DDPG: TensorFlow edition

This updated, structured Python 3.7 / TensorFlow 1.14 edition applies deep
deterministic policy gradients (DDPG) to household saving decisions and
aggregate equilibrium in a Krusell–Smith economy. It adds explicit experiment
configuration, command-line stages, isolated output directories, and artifact
verification around the later May–June 2020 TensorFlow implementation.
The Actor–Critic and mean-field economic update rules retain their historical
behavior. A separate [PyTorch implementation](../reproduction/README.md) is
included in the repository.

The research originated in *A Reinforcement Learning Approach to Krusell and
Smith (1998) from the Perspective of Mean Field Game*. All numerical inputs
needed by the documented commands are bundled here; the manuscript, PDFs,
and separate original source tree are not required or included.

The thesis results are targets, not a claim that a new training run has
already reproduced them. Historical checkpoints and samples are bundled as
read-only reference inputs. In particular, the published and archived ALM
records are not entirely consistent; see
[`docs/LEGACY_COMPATIBILITY.md`](docs/LEGACY_COMPATIBILITY.md).

## Project structure

| Path | Role |
| --- | --- |
| `src/thesis_tf/legacy/` | Later May–June 2020 TensorFlow 1.x code lineage, reorganized without a framework port |
| `src/thesis_tf/` | Explicit stage orchestration and verification |
| `configs/` | Historical round-5, round-20, and small smoke settings |
| `data/benchmark/` | Four original published-family DP cache arrays; immutable inputs |
| `data/historical/` | Selected original TensorFlow checkpoints, simulation arrays, and summaries; immutable evidence |
| `targets/thesis_results.json` | Separate manuscript and archived scalar values |
| `docs/` | Historical implementation choices and result provenance |
| `environment.yml`, `requirements/` | Python 3.7 environment and runtime dependencies |
| `runs/` | New generated files only (ignored by Git) |

The `legacy/` directory name identifies the TensorFlow 1.x runtime, not a
second, unmodified project. Its numerical engines `ddpg.py`,
`dynamic_programming.py`, and `utils.py` preserve the historical source bytes.
The updated parameter and entry-point scripts expose stage choice, historical
parameters, random seed, resimulation, and label language through environment
variables. The surrounding CLI adds configuration, data staging, output
protection, and verification without changing the TensorFlow graph or
economic update rules.

## Legacy environment

Use an x86-64 Linux or Intel macOS machine with a Python 3.7 interpreter.
TensorFlow 1.14.0 provides CPython 3.7 wheels for those platforms, not native
Apple Silicon wheels; see the historical
[TensorFlow 1.14.0 PyPI release](https://pypi.org/project/tensorflow/1.14.0/).
The default environment uses the CPU distribution. The original GPU experiment
ran on a Tesla K80 and would additionally need a period-compatible CUDA/cuDNN
installation. No complete original dependency lockfile survives, so the
non-TensorFlow version pins are a compatible-era reconstruction.

From this directory:

```bash
conda env create -f environment.yml
conda activate ks-thesis-tf114
python -m pip install -r requirements/legacy-py37.txt
PYTHONPATH=src python -m thesis_tf.cli --help
```

`requirements/legacy-py37.txt` deliberately pins the historically observed
TensorFlow 1.14.0 and NumPy 1.18.1. Do not substitute TensorFlow 2 or modern
SciPy for the baseline: the source uses `tf.Session`, `tf.layers`, and
`scipy.interpolate.interp2d`.

## Verify the bundled references

The reference audit also runs on a current Python with NumPy installed;
it does not import TensorFlow:

```bash
python -m pip install numpy
PYTHONPATH=src python -m thesis_tf.cli verify
```

The manuscript is intentionally absent, so the audit reports that its hash
cannot be checked locally and uses the bundled target transcription. The two
documented discrepancies in historical ALM records also produce warnings.
A successful audit validates the bundled records, not a new training run.

## Run stages

The command-line interface uses explicit verbs instead of the original
scripts' `test` and `simulate_data` booleans. Run each command from this
project directory with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python -m thesis_tf.cli verify --config configs/thesis.json
PYTHONPATH=src python -m thesis_tf.cli dp --config configs/thesis.json
PYTHONPATH=src python -m thesis_tf.cli train --config configs/thesis.json
PYTHONPATH=src python -m thesis_tf.cli evaluate --config configs/thesis.json
PYTHONPATH=src python -m thesis_tf.cli compare --config configs/thesis.json
```

`verify` checks the bundled references without training. By default, `dp`
plots the preserved DP policy and path cache; `evaluate` loads the archived
TensorFlow Actor/checkpoint and histories; `compare` reads the preserved
round-specific DP/DDPG evaluation arrays. These are **historical replay**
stages, not fresh numerical replications. `--resimulate` on `dp` or `compare`
asks the legacy scripts to regenerate their path/simulation inputs; for
`compare`, the manuscript-scale utility and ALM simulations are expensive.
English labels and manuscript filenames are the default; `--chinese` selects
the alternate original labels.
The archived DP path and evaluation arrays used a different B/R² cache than
`data/benchmark/`; plots or summaries that combine the two are
mixed-provenance, not one consistent fresh DP simulation. Use `--resimulate`
when a DP path should be computed under the selected benchmark policy.

`configs/thesis.json` describes the five-round main result (`idx=30`), and
`configs/thesis_round20.json` the 20-round appendix (`idx=50`). These are
separate historical run identities. The first training round uses 10,000
episodes, later rounds 5,000 each; each ALM regression is based on 100
simulations of 10,000 households. The full runs are costly—the manuscript
reports roughly 8 hours for round 5 and 30 hours for round 20 on its K80
machine. A CPU run may take longer.

For an inexpensive pipeline check, use `configs/smoke.json`. A smoke run does
not test the manuscript numbers; it is intentionally too small to reproduce
the thesis. The original source's replay batch size remains 1,024, so short
smoke training may not perform a gradient update.

New work belongs under `runs/`. Never overwrite `data/benchmark/` or
`data/historical/`; those inputs preserve original provenance. Inspect the
CLI's `--help` for stage-specific options before a long run. Use
`--run-dir` to keep independent experiments separate. A new `train` run does
not import archived weights or scores. The CLI marks that directory as fresh
and will not stage archived Actor weights or evaluation arrays into it. To
evaluate the new weights, reuse the same directory. A fresh `compare` also
requires `--resimulate`, so archived utility samples cannot be silently mixed
with the new Actor:

```bash
PYTHONPATH=src python -m thesis_tf.cli train --config configs/thesis.json --run-dir runs/fresh-r5
PYTHONPATH=src python -m thesis_tf.cli evaluate --config configs/thesis.json --run-dir runs/fresh-r5
PYTHONPATH=src python -m thesis_tf.cli compare --config configs/thesis.json --run-dir runs/fresh-r5 --resimulate
```

The last command performs manuscript-scale simulation and may take a long
time. Historical replay supports only the exact round-5 and round-20 folder
identities represented by the supplied thesis configs; a smoke or altered
setting is not an archived thesis result.

## Reading a result

The thesis utility table uses 500 aggregate paths, 5,000 people per path,
30 periods, and a discount factor of 0.8. The reported round-5 DDPG mean is
8.8478 versus a DP mean of 8.9583. The archived round-5 utility samples yield
these displayed values, but they are *historical samples*, not the result of
running the new commands. The printed DP ALM coefficients and an archived
round-5 DDPG slope disagree with the archived summary; the discrepancy is
recorded in [`targets/thesis_results.json`](targets/thesis_results.json), not
silently averaged away.

The bundled targets describe the reported statistics and expected figures.
A successful import, small run, or archived-array comparison establishes
pipeline health and source integrity only. A fresh
numerical claim requires a completed full training/evaluation run and a
separate comparison against manuscript targets; exact replay is not guaranteed
because TensorFlow and pre-seed random states were not saved.
