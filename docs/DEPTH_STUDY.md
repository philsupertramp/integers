# #38 — paired 4/8-convolution integer-vs-FP32 feasibility

**Status:** implementation candidate only. No complete 3-seed feasibility/quality
results are asserted here. This is the *first*, inexpensive, x86_64-controlled
depth experiment. #37 instrumentation and 12/16/20-convolution studies are
deliberately deferred until the paired results are reviewed.

## Dataset decision for the first step

This is an **MNIST control**, not yet the proposed CIFAR-10 stress test. It
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

The CI smoke uses only 16 training / 16 validation observations and one
seed; it tests code execution, archive validation and dataset isolation, **not
convergence**. Further layerwise loss, zero-gradient, saturation and
rounding-to-zero observability belongs in #37 after reviewing actual results.
