# Historical TensorFlow compatibility

This updated edition reorganizes the later May–June 2020 TensorFlow code
lineage under [`src/thesis_tf/legacy/`](../src/thesis_tf/legacy/). It preserves
the TensorFlow 1.14 Actor–Critic and implemented numerical choices while
making inputs, stages, and output directories explicit. The numerical
implementation is the behavioral reference when historical manuscript prose
and code conflict. The separate original source archive and manuscript are
not distributed or required.

## Environment and provenance

The manuscript identifies Python 3.7 and TensorFlow r1.14. The bundled
[round-5 summary](../data/historical/dp_round5/summary.txt) and
[round-20 summary](../data/historical/dp_round20/summary.txt) also print
TensorFlow 1.14.0 and NumPy 1.18.1. No original complete package lockfile, Docker image,
CUDA driver record, TensorFlow RNG state, or import-time NumPy RNG state was
retained. `environment.yml` and `requirements/legacy-py37.txt` are a
compatible-era reconstruction, not a claim to recover every original wheel.

PyPI published CPython 3.7 wheels for TensorFlow 1.14.0 on x86-64 Linux and
Intel macOS. Native Apple Silicon and current CPython versions are not
supported by that historical wheel. The default dependency file chooses the
CPU TensorFlow distribution; reproducing the original K80 GPU environment
additionally requires a compatible CUDA/cuDNN stack and may change numeric
behavior.

## Source-versus-manuscript decisions

| Issue | Historical evidence | Preservation rule |
| --- | --- | --- |
| Discount factor | Text and `parameters_ddpg.py` use `0.8`; a model-parameter table prints `0.99`. | Use `0.8` for the thesis base run. |
| Hidden widths | Architecture prose says `100, 100`; DDPG table and `parameters_ddpg.py` say `200, 100`. | Keep `200, 100`. |
| Round counts | Final `main.py` specifies five rounds, index 30; `parameters_ddpg.py` defaults to 20 rounds, index 50. | Provide distinct round-5 and round-20 configs; do not mix their checkpoints. |
| ALM order | DDPG regression returns `[bad intercept, bad slope, good intercept, good slope]`; DP cache/plot labels use good then bad. | Carry source-specific order explicitly at boundaries. |
| Training return | `MFG.play` discounts each reward despite prose calling the training score undiscounted. | Keep the computed discounted score. |
| Random seed | Entry points call `np.random.seed(0)` after importing parameter modules; there is no `tf.set_random_seed` in final code. | Record seed 0 but do not promise original neural weights or shock paths. |
| Initial aggregate state | Separate module-level Markov chains draw reusable defaults before the script seeds NumPy. | Record the state actually used by any new run; original pre-seed draws cannot be reconstructed. |

## Numerically consequential implementation details

- Actor and Critic use `tf.layers` graph/session APIs, Glorot-normal kernels
  and biases, and Adam with learning rates `1e-5`/`5e-5`. Keep TensorFlow
  1.14 variable scopes and checkpoint names for loading historical weights.
- `tf.layers.batch_normalization` is called without `training=True` and
  without update-operation dependencies. Treating it as active training-mode
  normalization would change the network.
- The target assignment operations are constructed with `tau=0.01`. The
  subsequent Python-side temporary change to `tau=1` in
  `update_target_parameters(first=True)` does not change the already-built
  graph, so the first update is soft rather than a hard copy.
- Deterministic Actor output is bounded to `[0,1]` as a saving ratio. OU noise
  (`theta=0.2`, `sigma=0.05`, `dt=0.01`) is added without clipping; the
  original training loop does not reset that noise at each episode.
- Replay has capacity one million and samples with replacement. `MFG` resets
  replay between fictitious-play rounds but retains the network weights.
- Round-`r` average policy is the unweighted mean of saved Actor checkpoints
  from rounds `1..r`, obtained in the source by repeatedly restoring each
  checkpoint. A single round-5 Actor checkpoint is not the complete round-5
  equilibrium policy.
- `MFG.solve_policy` checks an ALM coefficient difference below `1e-3` and
  above `1e-7` after round 1 and may stop before its maximum round count. The
  program must report the rounds actually achieved; it must not silently
  label an early-stopped run as round 20.
- Legacy simulation writes period outcomes for only `T-1` transitions; the
  last utility slot stays zero. DP and DDPG utility simulations use separate
  shock draws. Summary standard deviations use NumPy's `ddof=0` convention.

Changing any of these choices may be scientifically useful, but it defines a
new experiment rather than a same-method reorganization.

## Benchmark boundaries

The four arrays in `data/benchmark/` come from the historical archive location
`Thesis/code/test/dynamic_programming/`, whose `B.npy` and `R2.npy` round to
the reported DP ALM values. That path records provenance and is not a local
dependency. Similarly named arrays in a later version of the original archive
belong to a different productivity shock family and are not substitutes;
their B/R² values are retained under `data/historical/dp_base/` for comparison.
Loading cached policy/value arrays is
not a new dynamic-programming solve. The partial Python DP source has
known issues, including `regress_ALM` references to undefined `nb`/`ng`, and
is not an independent validated benchmark solver.

Archived data under `data/historical/` are immutable evidence from the 2020
run. Their presence can validate import, parsing, figure inventory, and
selected historical computations; it cannot establish that newly trained
weights would coincide with the archive.

The original comparison script prints DP ALM coefficients from its currently
loaded `B.npy`/`R2.npy` while computing RMSE from separately loaded
`DP_K*_ALM.npy` arrays. The bundled historical evaluation arrays were saved
under a later, different DP cache. Combining them with the printed-family
`data/benchmark/` coefficients produces a hybrid summary: each component has
provenance, but the combined report is not one internally consistent fresh
simulation. Regenerate paired ALM prediction arrays under the selected cache
before making a new joint numerical claim.

## Inconsistencies in preserved result records

The archived round-5 summary reproduces the manuscript's utility table after
rounding, but its DP ALM coefficients (`0.090042/0.895456` good and
`0.061485/0.896117` bad) differ from the printed manuscript values
(`0.0802/0.8930` and `0.0749/0.8926`). Its DDPG good-state ALM slope is
`0.925185971...`, which rounds to `0.9252`; `Thesis.tex` prints `0.9251`.
No verifier should coerce these contradictions into a pass by widening a
tolerance silently.

Historical figure pixels, hardware times, and qualitative robustness labels
are not exact numerical acceptance criteria. A fresh run should report
machine-checkable metric differences and separate structural success from
manuscript-value verification.
