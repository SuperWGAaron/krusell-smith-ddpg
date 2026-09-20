# Preserved numerical inputs

All files in this directory are source inputs, not generated outputs from
`reproduction-tensorflow/`. New runs belong under `../runs/`. The source files
were copied without transformation unless an individual provenance note says
otherwise.

The original-source paths below identify locations in the historical research
archive, which is not included in this repository. They are provenance labels,
not paths needed to run the project; all selected inputs are present in the
destination folders.

| Destination | Original source | Interpretation |
| --- | --- | --- |
| `benchmark/{B,R2,kss_k_opt,kss_value}.npy` | `Thesis/code/test/dynamic_programming/` | DP cache family whose B/R² values round to the manuscript's printed ALM values |
| `historical/round5/` | Selected files from `Thesis/test/round_5_life_5000_step_30_pop_100_idx_30/` | Round-5 TensorFlow Actor checkpoints, histories, paths, and DDPG evaluation arrays |
| `historical/round20/` | Selected files from `Thesis/test/round_20_life_5000_step_30_pop_100_idx_50/` | Round-20 counterpart |
| `historical/dp_base/` | Selected cache and path files from `Thesis/test/dynamic_programming/` | Later-family B/R² and historical DP plot/comparison paths |
| `historical/dp_round5/` | Selected files from `Thesis/test/dynamic_programming/round_5_7/` | Historical round-5 DP evaluation arrays and summary |
| `historical/dp_round20/` | Selected files from `Thesis/test/dynamic_programming/round_20_7/` | Historical round-20 counterpart |

The checkpoint `checkpoint` pointer files, which embed obsolete absolute
paths, are omitted. The TensorFlow loader uses each explicit `N.ckpt` prefix;
its `.index`, `.meta`, and `.data-00000-of-00001` components are retained.

## Benchmark SHA-256

`benchmark/SHA256SUMS` records the four exact source hashes. The four arrays
are not interchangeable with same-named files from later versions of the
original archive. Those belong to another DP cache family, represented here
by the B/R² arrays in `historical/dp_base/`. In particular, the
archived `dp_round5/` summary lists DP ALM coefficients that do not match the
manuscript's printed DP coefficients. See
[`LEGACY_COMPATIBILITY.md`](../docs/LEGACY_COMPATIBILITY.md).
The CLI stages benchmark B/R² first for a run, preserving the later-family
copies only under `historical/dp_base/` for audit; it does not silently
overwrite the selected benchmark inputs.

Only archived inputs read by the documented commands or required by their
integrity checks are bundled. Temporary `10000.ckpt` snapshots and saved
aggregate-path/timing files that the code regenerates or never reads are
omitted. New runs recreate their own outputs under `runs/`.
