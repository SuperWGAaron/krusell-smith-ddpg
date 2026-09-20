# Legacy compatibility and interpretation

The canonical `thesis.toml` profile aims to preserve the behavior that produced the selected thesis figures while replacing TensorFlow 1 session code and working-directory scripts with a structured PyTorch application. “Standard practice” here concerns packaging, configuration, state management, and verification; it does not authorize silent changes to the numerical experiment.

Where the legacy source is contradictory or defective, the resolved choice is explicit in configuration and provenance.

## Source contradictions

| Topic | Conflicting evidence | Canonical reproduction choice |
| --- | --- | --- |
| Discount factor | Parameter table says `0.99`; surrounding prose and final Python parameters use `0.8`. | `0.8`, with the table entry recorded as a manuscript inconsistency. |
| Hidden widths | Architecture prose says `100, 100`; the hyperparameter table and final code use `200, 100`. | `200, 100`. |
| ALM vector order | DDPG regression stores bad then good. DP comments and summary label good then bad, while some DP simulation branches interpret the first pair as bad. | Internally use named `good` and `bad` coefficients. Legacy inputs require an explicit source-specific converter. |
| Training score | Thesis prose calls it undiscounted; `MFG.play` multiplies rewards by successive powers of β. | Preserve the computed discounted return and label it accurately. An undiscounted diagnostic may be emitted separately. |
| Random seed | Thesis says the seed is fixed; final scripts seed NumPy but never TensorFlow. | Seed every modern RNG and record that original neural weights cannot be reconstructed from seed 0. |
| Initial aggregate states | The evaluation imports separate stateful chains: DP utility uses the chain created through `parameters_dp`, while ALM uses the `parameters_ddpg`-backed `MFG` chain. Each chooses and reuses an `x0` before the script calls `np.random.seed(0)`; neither pre-seed state nor draw was saved. | Use good (`1`) for utility and comparison, bad (`0`) for training/evaluation ALM, and good only as the generic fallback. The utility audit and ALM RMSE pattern support this split, but it remains an inference. |
| Pretraining | Robustness prose says to pretrain the Actor to return `0.9` and then train the Critic; the retained `actions_generator` helper returns `0.7`, and `MFG.pretrain` invokes both legacy stages. | Provide an Actor-only `0.9` hook as a manuscript operationalization, not an exact historical replay. |

## PyTorch compatibility choices

### Batch normalization

The TensorFlow code calls `tf.layers.batch_normalization` without `training=True` or update operations. Moving mean and variance therefore remain at their initialized values, while affine parameters are trainable. Normal PyTorch `BatchNorm1d` in training mode would change both activations and optimization.

The legacy profile uses frozen-statistics normalization with TensorFlow-compatible epsilon and trainable affine terms. A corrected experiment may enable ordinary training-mode batch normalization, but it is a different algorithm and must use a different profile name.

### Target-network initialization

The old code appears to request a hard target copy by temporarily setting `tau=1`. The TensorFlow assignment graph was built earlier with `tau=0.01`, so mutating the Python attribute did not change those operations. Targets consequently received a soft update rather than a true initial hard synchronization.

The legacy profile preserves this behavior. A standard-DDPG profile can hard-copy the online networks at initialization, but results from that profile are not directly compared to the thesis targets without being identified as corrected.

### Initialization and loss

The original dense kernels and biases use Glorot-normal initializers. PyTorch defaults differ, so the reproduction initializes explicitly. Cross-framework seed streams still do not produce identical initial tensors.

TensorFlow layers declare L2 regularizers, but the training losses optimize mean-squared error without adding the regularization collection. The legacy profile therefore uses zero effective weight decay. Enabling regularization is a new experiment.

### Actor output and exploration

The legacy Actor computes the equivalent of `clamp(relu(raw), 0, 2) / 2`, yielding a deterministic action in `[0, 1]`. Ornstein–Uhlenbeck noise is then added without clipping. Invalid capital choices are handled by the environment's `-20` terminal penalty rather than by clipping the exploratory action.

The legacy profile preserves unclipped exploration, OU parameters `sigma=0.05`, `theta=0.2`, `dt=0.01`, replay sampling with replacement, and a replay capacity of one million transitions.

### Fictitious-play policy

After each round the old implementation evaluates both the current Actor and the uniform average of saved Actor policies. It repeatedly restores TensorFlow checkpoints to compute that average. The PyTorch implementation may keep immutable in-memory or checkpointed snapshots, but the evaluated function must remain the uniform mean of rounds `1..r` for the legacy profile.

The full profile trains once through round 20 and captures complete round-5 and round-20 states. Merely loading the round-5 Actor after round 20 is insufficient: the associated ensemble and ALM must also come from round 5.

The configured ALM tolerance is recorded after each round as a convergence diagnostic. It does not terminate the canonical run because the thesis artifact contract requires complete snapshots at both rounds 5 and 20. Continuing to those checkpoints is an explicit orchestration rule, not an undocumented claim that the ALM had or had not converged.

### Optional pretraining

Robustness test 11 enables supervised Actor pretraining toward a constant saving ratio of `0.9`, as described in the manuscript. It is deliberately Actor-only. By contrast, the retained Python `actions_generator` returns `0.7`, and `MFG.pretrain` calls both `pretrain_actor` and `pretrain_critic`. The prose also describes the Critic stage, but no historical pretraining checkpoints, sampled training data, stopping state, or loss traces survive. Recreating that stage would therefore require new choices rather than recoverable parameters.

The configured hook is an explicit modern operationalization of the manuscript's Actor statement, not an exact reproduction of the legacy pretraining path. Its newly generated metrics may be compared with the base run, but the thesis's qualitative `close` label remains non-numerically-verifiable.

## Dynamic-programming compatibility

The historical Python DP source, preserved within the sibling TensorFlow package for compatibility, is not a complete, reliable solver:

- it uses removed `scipy.interpolate.interp2d`;
- saved arrays are Fortran-order `(100, 20, 4)` while some code expects `(100, 20, 2, 2)`;
- successor-state expectation indexing is inconsistent;
- the maximization routine stores the sign of the minimized negative objective;
- `regress_ALM` references undefined `nb` and `ng` and does not consistently implement the documented log-log regression;
- the Python driver does not implement the complete outer ALM fixed point represented by the Julia benchmark.

The reproduction treats the four recovered published arrays in `data/benchmark/` as provenance-checked legacy reference inputs and normalizes the joint-state axis explicitly. New interpolation is tested at every grid node. A recomputed DP solution must be labeled separately from a cache-loaded benchmark.

The recovered `(0.99, 1.01)` `B.npy` and `R2.npy` round to the historical reported values. Selected arrays from the later `(0.9, 1.1)` family are preserved under `../../reproduction-tensorflow/data/historical/dp_base/` and differ; they are not PyTorch reproduction inputs. Even for the recovered family, cache loading establishes benchmark provenance rather than an independent recomputation.

## Simulation and evaluation semantics

Legacy simulation fills period outcomes through `horizon - 1`, leaving the final utility slot zero. DP and DDPG utility samples are generated from separate shock streams despite prose that can be read as applying both policies to each shock path. Summary standard deviations use NumPy's population convention (`ddof=0`).

The retained evaluation did not have one universal initial aggregate state. Its DP utility paths drew from a `parameters_dp` chain, whereas ALM paths used the separately constructed `parameters_ddpg`/`MFG` chain. Both default states were selected during import before the evaluation entry point seeded NumPy, and each was reused for its paths. The exact import-time RNG states and selected values are unrecoverable.

Exhaustive evaluation of the 2.5-million-observation DP utility design strongly indicates a fixed good start. Separately, the printed ALM RMSE pattern is consistent with a fixed bad start. The canonical profile encodes that stage split explicitly: `evaluation.utility_initial_aggregate_state = 1`, `evaluation.comparison_initial_aggregate_state = 1`, `training.alm_initial_aggregate_state = 0`, and `evaluation.alm_initial_aggregate_state = 0`; `transitions.simulation_initial_aggregate_state = 1` is only the fallback when no stage override is supplied. This improves prospective reproducibility while preserving the uncertainty honestly: the values are inferred, and strict reported statistics may still fail.

The legacy profile preserves these choices for comparison with the printed targets. A corrected profile may compute the terminal period and use common random numbers, but its outputs must identify those changes and cannot silently replace legacy-profile results.

Economic simulation, interpolation, ALM regression, and final statistics use `float64`; networks use `float32`. Evaluation is chunked so manuscript-scale populations do not need to reside in accelerator memory.

## Determinism boundaries

The sibling TensorFlow package includes selected historical Actor checkpoints, training histories, and simulation samples; see [the preserved-input guide](../../reproduction-tensorflow/data/README.md). This PyTorch package does not import those checkpoints. Its deterministic training and numerical-target checks are independent of the TensorFlow historical replay, and no weight-for-weight equivalence between the frameworks has been established.

The modern run seeds Python, NumPy, Torch CPU, and every CUDA device; requests deterministic algorithms; disables cuDNN benchmarking; and stores all RNG states in resumable checkpoints. Named streams prevent evaluation or plotting from consuming training randomness. It records every stage-specific initial aggregate state, but cannot reconstruct the separate random `x0` values selected before seeding in the historical process.

These controls guarantee reproducibility only within the limits of the recorded runtime. PyTorch version, accelerator, driver, BLAS implementation, and deterministic-kernel availability can affect results. Cross-device bitwise equality is not promised. `run.json` and `provenance.json` make those conditions part of the evidence.

## Verification rules

Three kinds of checks remain separate:

1. **Input integrity:** legacy files exist and match recorded checksums and shapes.
2. **Pipeline integrity:** stages run, outputs are finite, checkpoints resume, and required artifacts exist.
3. **Thesis numerical verification:** generated statistics match the strict targets in `targets/thesis_results.json`.

Passing the first two does not imply the third. Qualitative robustness labels, visual similarity, or a permissive “approximately close” band cannot override a failed strict scalar target. Reports retain actual values and errors even when verification fails.

## Corrected experiments

The code may support scientifically useful corrections such as hard target synchronization, active batch normalization, action clipping, common random numbers, terminal-period utility, or a fully recomputed DP fixed point. Such runs should use a non-legacy compatibility profile, a separate output directory, and a target set that does not claim equivalence to the historical experiment.
