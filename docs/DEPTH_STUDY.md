# #38 — paired MNIST depth feasibility: 4/8/12/16/20 convolutions

**Status:** 4/8-layer He-v2 validation was confirmed across three paired
independent seeds; 12/16/20 layers are now implemented for progressive
feasibility testing. Numeric behavior and convergence at deeper levels are
**not yet established**. Defer #37 per-layer diagnostics until experiments
identify a reproducible integer-specific deviation.

## Dataset decision for the first step

This remains an **MNIST depth study**; the CIFAR-10 stress test is explicitly deferred. It
extends the independently tested, seeded MNIST CNN from #35 while holding the
input, target convention, loss and source/preparation fixed. That isolates
depth as the immediate variable; adding a new CIFAR loader/data layout at the
same time would confound initial diagnosis. Follow up on CIFAR-10 after this
feasibility gate; do not close #38 based solely on MNIST smoke results.

The canonical 60k MNIST **training** IDX is split 55k/5k with the existing
fixed stratified seed. All runs here use `--validation-only`, which neither
opens, hashes nor evaluates the official MNIST test partition. The official
10k test has already been consumed in #15/#35 and is not a fresh holdout.

## Predeclared architecture

| Stage | 4-convolution variant | 8-convolution variant |
| --- | --- | --- |
| Input | 1×28×28 | 1×28×28 |
| Stage A | 3×3 Conv(1→4) → ReLU, 1×1 Conv(4→4) → ReLU | 3×3 Conv(1→4) → ReLU, then 3× repeated 1×1 Conv(4→4) → ReLU |
| Pool A | 2×2, stride 2 → 4×13×13 | Same |
| Stage B | 3×3 Conv(4→8) → ReLU, 1×1 Conv(8→8) → ReLU | 3×3 Conv(4→8) → ReLU, then 3× repeated 1×1 Conv(8→8) → ReLU |
| Pool B | 2×2, stride 2 → 8×5×5 | Same |
| Head | Flatten(200) → Linear(64) → ReLU → Linear(10) | Same |

All convolutions are valid, with unit stride and no padding. Extra 1×1
convolutions keep the spatial geometry and dense head constant while increasing
the number of backpropagated learnable stages. They test depth/rounding
propagation but do **not** independently test deeper spatial receptive fields.
There are no BatchNorm, Softmax, residual connections, augmentations or
float-backed runtime operations in the integer path.

Both backends start with decoded versions of the **same quantized weights**
and use identical samples and permutation schedules, `shift=7`,
`output_bits=31`, upstream gradient clipping mantissa 8192, `lr_shift=7`,
momentum decay `2^-1`, actual-size minibatch means, and the pre-existing
0.5×sum((raw logits − one-hot targets)²) objective. Float uses separate
continuous FP32 convolution, pooling, linear, backward and optimizer arithmetic
from `experiments/cnn_reference.rs`; no integer production helpers in its
training arithmetic.

## Phase D: 12/16/20 convolutional layers on MNIST

After **three independent He-v2 seeds** on 4/8 layers, the mean validation
accuracy was approximately 87.33% integer vs 88.55% FP32 at depth 4, and
83.11% integer vs 85.25% FP32 at depth 8. The largest single-seed deficit
was 3.74 pp (depth 8, seed 2027); all three matched seeds showed learning.
These are developer-uploaded, checksum-verified development-validation
results, not a pristine test set or controlled timing evidence.

Continue on MNIST for now. Do **not** introduce CIFAR-10, augmentation,
normalization, residual connections, or any optimizer/quantization change
while investigating the depth limit.

### Frozen deeper architecture

Extend the He-v2 paired CNN construction to **12, 16, and 20** convolutional
layers. There are always two identical spatial stages:

- Stage A: one 3x3 Conv(1→4) and `(D/2 − 1)` 1x1 Conv(4→4), each
  followed by ReLU, then a 2x2 MaxPool2D.
- Stage B: one 3x3 Conv(4→8) and `(D/2 − 1)` 1x1 Conv(8→8), each
  followed by ReLU, then another 2x2 MaxPool2D.
- The resulting `8×5×5` feature map is flattened and processed by
  `Linear(200→64) → ReLU → Linear(64→10)`, exactly as at depth 4/8.
- All hidden convolution and hidden linear weights retain the integer-only
  preparation-time ×5/2 He-style initialization. The independent FP32
  reference receives the decoded *same* initialized dyadic parameters.
- Shift 7, 31-bit outputs, raw-logit MSE, clipping, momentum, dataset
  split, learning-rate shift, and deterministic sample orders are unchanged.
- The model identities are `mnist_cnn_{12,16,20}conv_he_v2`.

This isolates **backpropagation depth**, not increased spatial receptive
field or model width. It is a plain, narrow network without skip connections;
lack of convergence in *both* backends is not an integer-specific bug.

### Execution: do not launch all training at once

The PR's new CI depth smoke runs **one epoch, 64 train / 64 validation**, and
one independent integer/FP32 trial for each new depth. It checks construction,
forward/backward, report and archive validity, but is **not convergence
evidence** and must not be evaluated against the 80% accuracy gate.

On an x86_64 machine, after the MNIST data and Rust release example are
prepared, first run *one seed for one depth* to identify which depths are
computationally feasible and learn. Each `plan` requires a fresh directory:

```sh
python3 experiments/depth_paired_abba.py plan \
  --dir depth-results/mnist-12conv-seed1337 \
  --single-pass --seeds 1337 --depths 12 --epochs 10 \
  --train-limit 8192 --validation-limit 1042 --batch-size 32 --lr-shift 7
python3 experiments/depth_paired_abba.py run \
  --dir depth-results/mnist-12conv-seed1337
python3 experiments/depth_paired_abba.py analyze \
  --dir depth-results/mnist-12conv-seed1337
```

Repeat with separate directories and `--depths 16` then `20` after
inspecting each FP32 learning curve and elapsed CPU cost. If the FP32 control
learns at a given depth and integer fails or reproducibly degrades,
investigate layerwise numerical diagnostics in #37 **before** modifying
quantization to improve headline accuracy. If the FP32 control fails,
the benchmark is inconclusive for integer numerical quality.

Only after both backends learn at a depth run the frozen 3-seed cohort
`--seeds 1337,2027,4242` there. Then, separately, use the existing
mirrored ABBA/BAAB protocol for timing on a low-load x86_64 or Pi5 host;
wall times gathered under competing load are not reliable comparisons.
No official MNIST test partition is used in these depth experiments.

## After the first longer experiment: isolate an initialization control

The [frozen 10-epoch v1 hosted x86_64 run](https://github.com/philsupertramp/integers/actions/runs/40928659717)
used 8,192 training / 1,042 validation rows and seed 1337:

| Model v1 (same default fan-in uniform initialization) | Integer | FP32 |
| --- | ---: | ---: |
| 4 convolutions | 74.280% | 79.367% |
| 8 convolutions | 8.637% | 9.693% |

The four-layer networks learn noticeably; neither 8-layer backend leaves
chance-level classification. This is a **failed FP32 learning control** for
eight layers, *not* evidence of a specifically integer-induced depth defect.
The run's full reports remain immutable in the linked Actions artifact.

To isolate a conventional source of deep plain-ReLU training failure, the
**new v2 experiment differs from v1 only in its initialization**:
before copying parameters to the independent FP32 model, all hidden
convolution and hidden dense weights are multiplied **once** by the exact
integer factor 5/2 (truncated toward zero). The baseline constructor samples
uniformly at approximately ±1/sqrt(fan-in); this produces approximately
±2.5/sqrt(fan-in) initial bounds, near He-uniform
±sqrt(6/fan-in). The output linear head is unchanged, as are model topology,
loss, quantization shifts, clipping, momentum, SGD and data selection.
Both numerical backends receive exactly the same already-scaled, quantized
weights. This preparation-only change does **not** alter production layer
constructors or introduce floats to runtime integer training.

New result identities are `mnist_cnn_4conv_he_v2` and
`mnist_cnn_8conv_he_v2` to prevent mixing the original v1 experiment
with v2. The same 10-epoch/8192/1042 job can be rerun and should be
interpreted independently. We must still see FP32 improve before drawing
conclusions about integer-specific numerical soundness; if not, revisit
plain-network width/initialization or learning-rate choices as new disclosed
experiments rather than simply raising the integer precision.

## Larger learning-control experiment

The 2026-10-29 2-epoch, 2,048-train/512-validation one-seed feasibility
experiment did not establish learning for *either* backend, with only
8–16% validation accuracy. Do not diagnose integer-specific instability
from this short run.

A second, predeclared validation-only CI experiment now trains **10 epochs on
8,192 examples**, validating on 1,042 fixed examples, batch 32, learning rate
shift 7. Its time budget is approximately 5–10 minutes on a hosted x86_64
runner, but duration depends on the runner. The separate workflow is
`.github/workflows/depth-learning-experiment.yml`.

We use `--single-pass` to run one independent A=integer/B=FP32 process
per depth for exploratory **learning quality**. Deterministic single-seed
repetitions would waste this training budget and do not add independent
learning evidence; the default **mirrored ABBA/BAAB mode is unchanged**
for repeatable performance comparison after convergence.

This is an exploratory **one-seed** control, not a final conclusion. Before
reading outcomes, the initial learning/quality classifications are frozen:
both backends must achieve **at least 80% mean validation accuracy**,
and the integer model must lag FP32 by **no more than 5 percentage points**
for the `initial_quality_gate_pass` label. If FP32 stays below 80%, the
result is `inconclusive_fp32_not_learned`, even when integer accuracy
is comparably low. These gates do not replace gradient or operator oracles,
or establish test-set generalization. Later three-seed matched runs and
the full mirrored Pi timing suite remain separate.

To reproduce this longer exploratory CI job locally:

```sh
cargo build --release --locked --example held_out
python3 experiments/depth_paired_abba.py plan \
  --dir depth-results/learning-10epoch-01 --single-pass \
  --seeds 1337 --depths 4,8 --epochs 10 \
  --train-limit 8192 --validation-limit 1042 --batch-size 32
python3 experiments/depth_paired_abba.py run --dir depth-results/learning-10epoch-01
python3 experiments/depth_paired_abba.py analyze --dir depth-results/learning-10epoch-01
```

## Execution

On an x86_64 host with Rust and Python 3 and canonical MNIST training IDX:

```sh
python3 experiments/fetch_mnist.py --data-dir data/mnist --train-only
cargo test --locked --all-targets
cargo build --locked --release --example held_out

# Bounded 1-seed feasibility first. This creates 4 mirrored trials per depth.
python3 experiments/depth_paired_abba.py plan \
  --dir depth-results/feasibility-01 \
  --seeds 1337 --depths 4,8 \
  --epochs 1 --train-limit 2048 --validation-limit 512
python3 experiments/depth_paired_abba.py run --dir depth-results/feasibility-01
python3 experiments/depth_paired_abba.py analyze --dir depth-results/feasibility-01

# ONLY if both backends are stable, learning and affordable:
# A fresh directory and at least 3 distinct model seeds. Never overwrite a run.
python3 experiments/depth_paired_abba.py plan \
  --dir depth-results/paired-01 \
  --seeds 1337,2027,4242 --depths 4,8 \
  --epochs 10 --train-limit 55000 --validation-limit 5000
python3 experiments/depth_paired_abba.py run --dir depth-results/paired-01
python3 experiments/depth_paired_abba.py analyze --dir depth-results/paired-01
```

**Do not launch the full run merely because the one-epoch plan succeeded.**
Inspect x86_64 feasibility timings/memory and check that the FP32 control
learns before choosing a practical follow-up budget. Any changed budget,
source commit, binary, seed set or hyperparameter requires a new immutable
plan and must be disclosed alongside the original trials.

The runner reuses `pi5_interleaved_ab.mirrored_arms` from #55 and the
`BENCHMARK_ARCHIVE_V1` checked archive envelope. For each seed and depth,
the order is ABBA or BAAB, where A = integer and B = FP32. The executable,
dataset IDX hashes, source commit, seed/depth order and hyperparameters are
frozen in `plan.json`. Individual results are archived with an independent
SHA-256 and log envelope; resumption validates existing trials and refuses
partial/tampered outputs.

### What to interpret

- Per-seed training and validation accuracy/curves, confusion and predicted
  classes, nonfinite/extreme output diagnostics, numerical gap vs FP32.
- Per-process training time and independently collected process peak-RSS
  from the underlying model JSON; the summary reports training-time ratios.
- Only **distinct initialization seeds** are independent observations of
  learning behavior. The two same-seed repeats per arm address run-order
  drift, not numerical sample size.
- ABBA/BAAB is a controlled local timing protocol, not evidence of physical
  Pi5 speedup or energy efficiency. Do not apply #55's ≥5% speed gate here.
- Comparable accuracy does not prove all operators or gradients numerically
  correct. Existing analytic layer oracles remain mandatory. If FP32 fails to
  learn at a depth, report **inconclusive**, not an integer-specific failure.

The CI feasibility test uses 2 epochs with 2,048 training / 512 validation
observations and one seed, with mirrored repeats. It tests execution, archive
validation, dataset isolation and an initial learning signal, **not full
convergence or independent generalization**. Further layerwise loss, zero-gradient, saturation and
rounding-to-zero observability belongs in #37 after reviewing actual results.
