# Dynamic-programming benchmark

These four arrays are the Julia-generated benchmark used for the numerical
values reported in the original research experiment. Their original archive
location was `Thesis/code/test/dynamic_programming/`; that path records provenance
and is not a runtime dependency. The arrays are copied here so the reproduction
is self-contained. Their hashes are pinned in `SHA256SUMS`.

A later notebook run with productivity shocks `(0.9, 1.1)` produced similarly
named arrays that do **not** match the historical printed ALM coefficients.
Selected later-family `B.npy` and `R2.npy` files are preserved for audit in the
sibling TensorFlow package's `data/historical/dp_base/` directory. This directory
intentionally contains the earlier `(0.99, 1.01)` result family:

- `B.npy`: `(good intercept, good slope, bad intercept, bad slope)`;
- `R2.npy`: `(good, bad)`;
- `kss_k_opt.npy`: policy grid with Julia shock order `(g,e),(b,e),(g,u),(b,u)`;
- `kss_value.npy`: value grid in the same order.

The cached benchmark is an input with provenance, not evidence that a new
PyTorch DDPG run matches the thesis. Verification reports that comparison
separately.
