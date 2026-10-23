# Issue #15 — reproducible held-out training protocol

**Status:** experimental candidate benchmark machinery. No full training results have
been measured or accepted by this change. This is not a declaration that the
integer backend converges on full MNIST, beats FP32, or is hardware-efficient.

## Why a separate entrypoint?

The historical `examples/mnist/main.rs` CNN and `examples/mnist_nn/main.rs`
dense example both evaluate subsets or all of the **test** data during
training and use test performance for early stopping. `examples/iris.rs`
also evaluates its test split every epoch. Preserve these as historical
demonstrations, **not** #15 benchmark evidence.

`examples/held_out.rs` uses the integer-only `Sequential::new()` path
(Linear/ReLU/Linear/ReLU/Linear) and an independently implemented scalar FP32
reference with the **same dense architecture and identical initialization**,
permutations, batch size, initial targets, SGD learning rate and momentum
convention. The FP32 model uses continuous arithmetic rather than dyadic
rounding, saturation and STE; the two optimizers therefore need not produce
identical trajectories. Both use the unnormalized batch-accumulated squared
error derivative `output - target` (equivalent objective up to constant
factors). No Softmax/BatchNorm FP64 runtime path is enabled in integer mode.

## Data splits and quantization

| Dataset | Source split | Training | Validation | Final test |
|---|---|---|---|---|
| Bundled Iris | 130 previously published development rows + 20 historical test rows | 106 | 24, stratified 8/class | all 20 reserved rows, evaluated once |
| Canonical MNIST IDX | 60,000 official training rows | 55,000 | 5,000, stratified 500/digit | all 10,000 official test images, evaluated once |

Split seed is a fixed constant `0x0123456789ABCDEF`, *not* a training
hyperparameter. For every model seed the same indices are held out.
The independent test set is absent from splitting, fitting preprocessing
statistics, hyperparameter/epoch selection and the per-epoch learning curve.
Training samples are shuffled deterministically each epoch with a separate
seed-derived RNG stream. Initialization and stochastic rounding are seeded
explicitly per run. The loss target is 127/128 for the positive class.

The Iris `f32` feature-to-Q7 mapping is fitted only on **training indices**.
Validation and test receive that mapping without clipping to the fitted
min/max ranges, so out-of-range data is not silently compressed. MNIST uses
the existing fixed byte-to-Q7 mapping from `mnist_nn/mnist_loader.rs`
for all splits; this mapping has no fitted statistics. Inputs and targets
for FP32 are exactly the decoded dyadic values supplied to the integer model.

**Historical Iris caveat:** the bundled 20-row test split was examined in
earlier work. It is reserved from all computations in this experiment but
is not a genuinely *never-before-seen* test. The protocol is meaningful,
but its reported accuracy cannot be advertised as a pristine external
generalization estimate.

## Running

From the repository root, Rust stable (edition 2024), Python 3.9+, and
the same checked-out commit:

```sh
# No downloads; 4 small runs, reproducible fixture-smoke behavior.
python3 experiments/run_baselines.py --dataset iris --smoke --epochs 2 --seeds 1337 2027

# 3 repeated seeds, separate fresh process per seed/backend.
python3 experiments/run_baselines.py --dataset iris --epochs 100 --seeds 1337 2027 4242

# Download canonical MNIST IDX files under data/mnist (see data/mnist README).
# The runner REQUIRES 60,000 train + 10,000 test rows; it does not
# substitute small sampled training/testing subsets.
python3 experiments/run_baselines.py --dataset mnist --epochs 10 --seeds 1337 2027 4242
```

An explicit `--batch-size 32` is the default for both tasks. No early
stopping, validation-guided checkpoint selection or test-guided tuning is
performed: all epochs are specified **before** a run. If hyperparameters are
changed based on validation diagnostics, record a separate experiment;
never choose them by reading test accuracy. For publication of a final
locked protocol, freeze them before the final test run.

MNIST uses the existing `load_mnist_auto` which prefers IDX over Parquet;
the Python launcher **requires the canonical four IDX files** and hashes
them. Don't claim a comparison from an alternate dataset variant. The
standard runner checks the canonical train/test counts and dimensions.
The Iris runner requires the exact 130/20 bundled row counts.

## Machine-readable evidence

`reports/held_out/{iris,mnist}/{smoke,full}/` contains:
- `integer-seed-<n>.json` and `fp32-seed-<n>.json`: full epoch-by-epoch
  training and validation accuracies, sample counts, elapsed time,
  hyperparameters, validation size, test accuracy, and peak RSS (Linux only).
- `manifest.json`: exact source SHA-256 hashes, git commit, Rust/cargo
  versions, hardware/OS descriptor and execution timestamp.
- `summary.md`: repeated-seed means and min/max (accuracy, wall time, peak
  RSS) with the known comparability limitations.

Wall time includes training and one final evaluation, but **not** build time
or dataset-loading/preparation time. `peak_rss_kb` is Linux
`/proc/self/status` VmHWM for the **entire child process**, including the
dataset in RAM; when unavailable the report records JSON `null`.
Each seed/backend runs in its own process to avoid conflating peak RSS
between backends. The full learning curve contains validation accuracy
after every epoch, not an independently reported test curve.

A failed run must not be silently omitted when calculating mean/range:
the launcher stops on failure. Report missing runs rather than filling them
with guesses. Review and commit only lightweight reports and manifests,
never downloaded data or model checkpoints.

## Accuracy/latency/memory decisions

**ADR-004 remains pending.** The first full-data runs are baseline
measurements, not automatic pass/fail judgements. Propose numeric thresholds
only after reading measured multi-seed curves and accounting for hardware,
training budget and implementation maturity. The FP32 reference is written
for auditability, not optimized BLAS; do not read a speed difference as
proof that integer instructions are inherently faster or slower.

The CI smoke workflow executes two tiny Iris epochs with both backends,
verifies the protocol and report shapes, and never claims convergence.
Full MNIST training requires local dataset access and adequate resources
and is deliberately **not** triggered in default CI.
