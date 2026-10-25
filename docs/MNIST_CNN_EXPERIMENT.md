# Issue #35 — reproducible MNIST CNN methodology and status

**Status:** implementation PR #36 and the frozen full 55k/5k, three-seed **10-epoch validation experiment completed and archived** on 2026-10-24. An independently archived **final-test benchmark run** was subsequently completed on 2026-10-25 with the same frozen configuration; both archives are separate. This is an **already-known MNIST benchmark**, not a pristine untouched holdout (the official test was consumed in issue #15). This is not a claim of deep-network numerical correctness or hardware efficiency. Base commit: `795129207f8ca1d4be7b8ac4e5f590b91b633017` on `big-refactor`. Architecture and exploratory settings are frozen below before full CNN measurements.

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

The first measured feasibility run on the GitHub-hosted x86_64 Ubuntu runner ([CI 40127767359](https://github.com/philsupertramp/integers/actions/runs/40127767359), commit `c8cd44eb`) completed one epoch with **128 training and 128 validation images**, seed 1337, batch 32, release build, no test IDX files. Integer online training accuracy was 10.94%, validation 8.59%, whole epoch 275 ms, peak RSS 410.85 MiB and extreme output count zero. Scalar FP32 online training accuracy was 12.50%, validation 12.50%, whole epoch 55 ms, RSS 410.95 MiB and zero nonfinite parameters. These numbers do not show convergence; integer elapsed time was about 5× the scalar FP32 reference for this tiny cohort. The report artifact is attached to that run. A 128+128-sample timing cannot safely predict throughput at 55k+5k or with different batch/cache conditions; the runner now also records training and validation time separately for an improved subsequent estimate.

At the initial feasibility stage, no CNN-specific full 55k/5k three-seed accuracy, numerical anomaly frequency, runtime or peak-RSS comparison had been established by the existing dense experiment. The complete measured CNN validation results are recorded below. Do not extrapolate dense timings to CNN. To estimate expense, measure both backends on a representative post-split 128-example cohort, measure validation overhead separately, and extrapolate the **training-only component** to 55,000 examples per epoch; validation remains 5,000 per epoch. Multiply by ten epochs, three seeds and both backends, retaining an uncertainty margin for allocation, cache effects and hardware differences. Reports must include measured samples, profile, wall-time and VmHWM, not fabricated full-run estimates.

### Issue #35 acceptance snapshot (to be updated after CI)

- Implemented: integer CNN path through default ADR-002 backend, paired independently implemented scalar FP32 reference, architecture switch and shared #15 split/provenance/reporting, deterministic offline composition tests and batch/cache regressions. The small real-MNIST, validation-only feasibility gate passed on CI 40127767359.
- Verification pending: successful `cargo test --locked`, all-target checks and feature checks on the PR head.
- Established by the subsequent frozen validation run: full 55k/5k three-seed learning curves, measurable integer CNN learning, paired validation accuracy and measured per-backend wall time/process peak RSS. Not established: an optimization plateau, strict numerical parity, independent final-test generalization, Pi 5/hardware efficiency or detailed per-operation saturation/overflow telemetry.
- Validation-only guarantees apply to the explicitly isolated mode, not historical `--ci-sample` which uses the #15 four-file setup.
- ADR-004's 94%/2pp dense criteria are **not** CNN pass/fail gates. No unauthorized public API, dtype, licence or platform policy changes.

## Complete frozen 10-epoch three-seed validation experiment (measured 2026-10-24)

The maintainer executed the predeclared `--validation-only` experiment at training commit `ce1e9975b0d3ca7ca98e1a54548cb9f32be4cf02`, without changing architecture or hyperparameters. **All six complete JSON reports, original manifest, original summary, and original SVG are committed as-is** under [archived CNN validation results](../reports/held_out/mnist/cnn/validation-only/2026-10-24-10epoch-ce1e9975/README.md); a compressed companion archive retains the original training log. Git blob hashes of each extracted report match the originally submitted archive.

- Environment: Manjaro `Linux-6.12.95-1-MANJARO-x86_64-with-glibc2.43`, Rust `rustc 1.95.0 (59807616e 2026-04-14)`, Cargo 1.95.0, release profile. Source SHA-256 values are in the archived manifest and match the earlier dense validation protocol.
- Configuration: exact 55,000 train / 5,000 stratified validation observations; 10 fixed epochs per model; seeds 1337, 2027, 4242; batch 32, `lr_shift=7`, `momentum_shift=1`, dyadic shift 7; no validation-based early stopping.
- All six JSON reports record `test_size=0` and `test_evaluated=false`. The official 10k test partition was not loaded, hashed or evaluated in this validation experiment.

| Paired seed | Integer validation | Independent scalar FP32 validation | FP32 − integer | Integer wall | FP32 wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1337 | 96.14% | 97.12% | 0.98 percentage points | 477.68 s | 111.25 s |
| 2027 | 96.42% | 97.18% | 0.76 pp | 537.54 s | 116.03 s |
| 4242 | 96.54% | 96.98% | 0.44 pp | 550.99 s | 112.11 s |
| **Mean** | **96.37%** | **97.09%** | **0.73 pp** | **522.07 s** | **113.13 s** |

All three integer runs reached >96% validation accuracy at epoch 10 and showed sustained improvement over the ten-epoch budget, but this is **not** evidence of a final learning plateau. Mean integer process peak RSS was **410.80 MiB**, FP32 **410.65 MiB**; integer wall time was approximately **4.62×** the deliberately unoptimized scalar FP32 reference on this system. Every reported epoch had zero recorded `extreme_outputs` and `nonfinite_outputs`, but this is **not** a per-operation clipping/overflow or lost-gradient audit (see future research [#37](https://github.com/philsupertramp/integers/issues/37)). The same 5k validation examples were reused across three independently seeded models, not 15k independent observations.

**Boundary:** the archived validation experiment is distinct from the [completed CNN final benchmark](../reports/held_out/mnist/cnn/full/2026-10-25-locked-ce1e9975/README.md) and was produced before the latter. The final benchmark used the already-known MNIST test, since #15 had consumed it; do not tune further CNN hyperparameters using those scores.

## Frozen final 10-epoch MNIST benchmark (measured 2026-10-25)

The maintainer subsequently ran the explicitly opt-in `--final-test` CNN protocol with the **same locked architecture, hyperparameters and all three seeds** on original training commit `ce1e9975b0d3ca7ca98e1a54548cb9f32be4cf02`. The complete independent benchmark evidence is committed under [archived CNN final test results](../reports/held_out/mnist/cnn/full/2026-10-25-locked-ce1e9975/README.md), alongside a test-score SVG and losslessly packed original inputs. This was a fresh training pass (rather than reused checkpoints). All ten per-epoch training/validation curves and confusion matrices in the separately submitted final reports match the already archived validation experiment for every seed/backend, while the final reports additionally record the one-time official test score.

- **Dataset:** exactly **55,000 train / 5,000 development validation**, plus **10,000 official MNIST test** images evaluated only after epoch 10. The four canonical source IDX SHA-256 digests are recorded in the archived final `manifest.json`. Models share the 10k test cases across seeds.
- **Test disclosure:** #15 had already accessed the official MNIST test. This CNN evaluation is **a standardized already-known benchmark** and cannot support a fresh pristine independent generalization claim.
- **Environment:** Manjaro x86_64, Rust 1.95.0, Cargo 1.95.0, release build, exactly the same training commit. Predeclared seeds 1337/2027/4242, batch 32, `lr_shift=7`, `momentum_shift=1`, same 10-epoch budget, no test-driven epoch or hyperparameter selection.

| Paired seed | Integer official test | Scalar FP32 official test | FP32 − integer | Integer wall | FP32 wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1337 | 96.77% | 97.25% | 0.48 pp | 501.03 s | 117.16 s |
| 2027 | 96.53% | 97.37% | 0.84 pp | 549.25 s | 117.90 s |
| 4242 | 96.90% | 97.17% | 0.27 pp | 532.12 s | 111.25 s |
| **Mean** | **96.73%** | **97.26%** | **0.53 pp** | **527.47 s** | **115.44 s** |

- **Test-versus-validation:** final 10-epoch mean development validation is 96.37% integer and 97.09% FP32; official-test results are respectively +0.37 and +0.17 percentage points. This variation is consistent with different finite sample splits; it does not establish an independent new holdout.
- **Cost:** mean reported peak process VmHWM was 436.85 MiB integer and 434.10 MiB FP32. Integer training took ~4.57× this **unoptimized independent scalar FP32** baseline on the measured machine, not an intrinsic hardware performance comparison.
- **Diagnostics:** recorded nonfinite and extreme outputs = 0 in all 60 per-epoch reports; FP32 reports zero nonfinite parameters. Per-operation saturation/rounding-to-zero/gradient loss not instrumented ([research #37](https://github.com/philsupertramp/integers/issues/37)).
- **Interpretation:** #35 now has both a fully reproducible 55k/5k validation experiment and an explicitly labeled full **already-known official MNIST benchmark**. The sample size is **10k per seed** with shared examples, not 30k independent observations. This is strong evidence for the declared shallow CNN training path, not a claim that deep networks or all integer operators are verified. ADR-004's dense acceptance thresholds are not inherited by CNN.

Reproduction (training/test data downloaded separately; no raw data or model weights committed):

```sh
python3 experiments/fetch_mnist.py --data-dir data/mnist
python3 experiments/run_baselines.py \
  --dataset mnist --architecture cnn --final-test \
  --epochs 10 --seeds 1337 2027 4242 \
  --batch-size 32 --lr-shift 7 \
  --output-dir reports/cnn-final
```

## Known limitations

The existing integer kernels allocate intermediate per-layer tensors and caches in the batch path. Full-dataset runtime and peak RSS on one x86_64 Manjaro host have now been measured for both CNN protocols, but cannot establish memory/throughput feasibility on Raspberry Pi 5 or other hardware. The reference's per-example forward cache allocations are intentionally simple, not performance optimized. Mixed-scale and saturation behavior is covered by historical #12/#14 oracle tests and should be expanded with CNN-specific analytic fixtures; a passing model test alone never proves numerical completeness. CPU/device energy and Raspberry Pi results remain #17/#18 work.
