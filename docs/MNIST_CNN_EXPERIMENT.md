# Issue #35 — reproducible MNIST CNN methodology and status

**Status:** implementation PR #36, **experiment incomplete pending measured feasibility and full runs**. This is not a claim of 94% CNN accuracy or hardware efficiency. Base commit: `795129207f8ca1d4be7b8ac4e5f590b91b633017` on `big-refactor`. Architecture and exploratory settings are frozen below before full CNN measurements.

## Model and arithmetic

Input is 1×28×28 row-major, channel-first; convolution is **valid** cross-correlation with unit stride and no padding. Pool is 2×2, stride 2, strict-`>` maximum (first element wins ties):

| Stage | Shape / elements |
| --- | --- |
| Input | 1×28×28 = 784 |
| Conv2D(1→4, 3×3), ReLU | 4×26×26 = 2,704 |
| MaxPool(2,2) | 4×13×13 = 676 |
| Conv2D(4→8, 3×3), ReLU | 8×11×11 = 968 |
| MaxPool(2,2), Flatten | 8×5×5 = 200 |
| Linear(200→64), ReLU | 64 |
| Linear(64→10), **raw logits** | 10 |

Both models minimize `L = 0.5 * sum_c (logit[c] - one_hot[c])²`, with derivative `dL/dlogit = logit - target`. They do **not** use cross-entropy or Softmax. This fixes a conceptual ambiguity in the legacy tiny-CNN example without modifying that historical example. We compare the same fixed model, with identical decoded integer-initialized parameters and training order, not independently drawn continuous FP32 weights.

All four parameterized integer layers have dyadic weight, input, bias and multiplication quantization shifts `s=7`, 31 output bits (`[-2^30, 2^30-1]`), `grad_clip=8192` mantissa at each upstream per-neuron backward gradient, learning rate `2^-7`, and momentum decay `2^-1`. The weight constructors use the existing seeded, fan-in based dyadic initialization. Inputs are pixel values normalized (in preparation only) to mantissas `[-127,127]` at shift 7. Target class mantissa is 127 at shift 7; other classes 0. The integer backend is `Sequential::new()` and **never** includes floating Softmax or BatchNorm. Dyadic operations use the existing stochastic rounding and signed saturation; forward clip decisions are cached for the backward STE (identity through rounding, blocked if actually saturated). Mean summed gradients are divided by the *actual number of examples* before the momentum/SGD update. We do not change the public precision representation.

The FP32 reference has its own plain Rust scalar implementations of channel-first valid convolution, strict-max pooling, dense transforms, ReLU and their backward equations, with independently allocated parameter-gradient and velocity buffers. It starts from `Dyadic` parameters decoded once **before training**; no integer conv, pool, or dyadic training helper is called by FP32 forward/backward/update. FP32 uses continuous arithmetic and true piecewise gradients for these operations; integer quantization, saturation and STE can diverge from the mathematical derivatives, and stochastic-rounding streams differ from the reference (which consumes none). Both references clip the upstream per-example gradient at the same decoded threshold, apply mean batch gradients **before** momentum, and use the same scalar optimizer rule `velocity = 0.5 velocity + mean_gradient; weight -= 2^-7 velocity`. The FP32 reference is not a BLAS/PyTorch speed baseline.

## Split, seeds and phases

Canonical MNIST training IDX has 60k examples. The #15 fixed stratified split seed `0x0123456789ABCDEF` reserves 500 per digit (5,000 total) for validation and leaves 55,000 training. The paired model/epoch seeds are predeclared **1337, 2027, 4242**. Identical per-epoch class-independent permutations, batch size 32, and short-last-batch handling are used. The exploratory CNN budget is **10 epochs** with a fixed `lr_shift=7`, `momentum_shift=1`, and no checkpoint selection by validation metrics. This is a reproducibility protocol, *not* an approved CNN-specific accuracy threshold; future thresholds require separate maintainer agreement.

A validation-only run loads and hashes **only** the canonical training images/labels. The `--train-limit` / `--validation-limit` options sample deterministically *after* the canonical 55k/5k split for feasibility and debugging, without accessing the official 10k test. Caps are explicitly recorded in the report and cannot be combined with final evaluation. In a full validation-only run the limits are omitted. The Python orchestration produces one JSON per backend per seed, per-epoch training/validation accuracy, validation confusion matrix and predicted class counts, whole-process VmHWM peak RSS on Linux, elapsed wall-time, source SHA-256 and compiler/OS/git metadata in the manifest. Per-class recall is calculated as `confusion[c][c]/sum(confusion[c])`; output extremes and FP32 nonfinite counts are included where observed. They are not per-operation overflow counters.

MNIST's official 10k test was already examined in #15 and cannot be called a new, untouched holdout. Optional `--final-test` is **an already-known benchmark/regression evaluation** after frozen CNN selection; it must not drive further CNN tuning. A new unbiased generalization claim would need a separately reserved source (e.g., MNIST-C as an explicitly out-of-distribution robustness study, with corruption choice/severity and normalization fixed in advance, or a completely independently collected digits dataset). Do **not** present reused MNIST scores as new independent evidence.

## Reproduction

Prerequisites: Rust/Cargo and Python 3, canonical files acquired and SHA256-verified using #15's fetch script. No dataset/checkpoints are committed.

```bash
# Fetch only the canonical training IDX files for validation-only development.
python3 experiments/fetch_mnist.py --help
python3 experiments/fetch_mnist.py --data-dir data/mnist --train-only

# Offline unit/integration and all target checks.
cargo test --locked
cargo test --locked --test cnn_integration
cargo check --locked --all-targets
cargo check --locked --all-targets --all-features

# Feasibility gate: each backend runs in its own process.
python3 experiments/run_baselines.py --dataset mnist --architecture cnn \
  --validation-only --epochs 1 --seeds 1337 --batch-size 32 --lr-shift 7 \
  --train-limit 128 --validation-limit 128 --output-dir reports/feasibility

# Only after reviewing estimated time and resources:
python3 experiments/run_baselines.py --dataset mnist --architecture cnn \
  --validation-only --epochs 10 --seeds 1337 2027 4242 \
  --batch-size 32 --lr-shift 7 --output-dir reports/cnn-validation
```

Note: the train-only fetch option and its checksum behavior must be verified against the live fetch script; the requested validation-only path itself does not open test IDX files. Do not run full three-seed jobs in routine CI.

## Feasibility and acceptance evidence

No CNN-specific full 55k/5k three-seed accuracy, numerical anomaly frequency, runtime or peak-RSS comparison has been established by the existing dense experiment. Do not extrapolate dense timings to CNN. To estimate expense, measure both backends on a representative post-split 128-example cohort, measure validation overhead separately, and extrapolate the **training-only component** to 55,000 examples per epoch; validation remains 5,000 per epoch. Multiply by ten epochs, three seeds and both backends, retaining an uncertainty margin for allocation, cache effects and hardware differences. Reports must include measured samples, profile, wall-time and VmHWM, not fabricated full-run estimates.

### Issue #35 acceptance snapshot (to be updated after CI)

- Implemented: integer CNN path through default ADR-002 backend, paired independently implemented scalar FP32 reference, architecture switch and shared #15 split/provenance/reporting, deterministic offline composition tests and batch/cache regressions.
- Verification pending: successful `cargo test --locked`, all-target checks and feature checks on the PR head.
- Not established: full three-seed CNN learning curves, full-data benchmark accuracy and parity, observed convergence, hardware efficiency, full feasibility extrapolation and detailed per-operation saturation/overflow telemetry.
- Validation-only guarantees apply to the explicitly isolated mode, not historical `--ci-sample` which uses the #15 four-file setup.
- ADR-004's 94%/2pp dense criteria are **not** CNN pass/fail gates. No unauthorized public API, dtype, licence or platform policy changes.

## Known limitations

The existing integer kernels allocate intermediate per-layer tensors and caches in the batch path. Without first measuring workload throughput we cannot infer whether this is feasible for the complete dataset. The reference's per-example forward cache allocations are intentionally simple, not performance optimized. Mixed-scale and saturation behavior is covered by historical #12/#14 oracle tests and should be expanded with CNN-specific analytic fixtures; a passing model test alone never proves numerical completeness. CPU/device energy and Raspberry Pi results remain #17/#18 work.
