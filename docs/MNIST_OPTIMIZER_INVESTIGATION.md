# MNIST training diagnosis — shared minibatch optimizer scaling

**Observed:** 2026-10-24; [issue #15](https://github.com/philsupertramp/integers/issues/15); [draft PR #34](https://github.com/philsupertramp/integers/pull/34). This documents experimental evidence rather than a change to ADR-002's numerical representation or a new ADR-004 pass threshold.

## Code-level cause under test

In `Linear::backward_one`, parameter gradients accumulate **by sum** over individual examples. Both integer `Sequential::update` and the paired Rust FP32 `FloatMlp::update` originally applied `2^-7` to that sum for minibatches of 32. Neither divided by batch size. Thus the first-order update at batch size 32 is approximately **32 times the update from a mean gradient** with the same learning rate (different momentum, stochastic rounding and saturation can alter later trajectories).

With full-size power-of-two batches, changing the update from `lr_shift=7` to `lr_shift=12` multiplies the effective step by `2^-5 = 1/32`, while leaving the network, training samples, shuffled order, gradient shapes and layer quantization unchanged. This is an **effective learning-rate compensation**, not a newly implemented exact general-purpose batch-mean gradient operator. A partial batch (e.g. the 24-example tail of a full 55,000-image train split) still receives the fixed shift. Proper variable-size averaging and momentum scaling deserve independent follow-up if required.

The independent `tests/batch_gradient_scaling.rs` regression exercises a single dyadic Linear layer with exact integer inputs, verifying that 32 identical samples give a 32-fold unnormalized weight update and that adding five bits to the SGD shift restores the single-example step in that exact, no-momentum fixture.

## Controlled real-MNIST evidence

Source: canonical 60,000 training / 10,000 test IDX files, compressed SHA-256 pinned by `experiments/fetch_mnist.py`. **Only the original training pool is sampled**: 640 training examples (64 per class) and 200 validation examples (20 per class), deterministic and disjoint; the official 10,000-image test set is **not evaluated** in any ablation. Model is the same dense `784→128→ReLU→128→ReLU→10` MLP for both arithmetic backends, initialized from identical quantized weights. Squared-error derivative, per-example neuron-gradient clipping (64 decoded units), momentum decay 0.5, fixed 20 epochs, batch 32, seeds 1337 and 2027. Default layer Q7 shifts and 31-bit output bound remain unchanged.

No accuracy threshold was prescribed before collecting these results. The values below are **observations on the 200-image validation cohort**, not claims of full-MNIST generalization.

| Effective LR | Integer final validation, seed range | FP32 final validation, seed range | Interpretation |
| --- | ---: | ---: | --- |
| `2^-7` (historical) | **10.00%** (10.00–10.00) | **10.00%** (10.00–10.00) | Predictions collapsed to one class |
| `2^-12` (batch-compensated) | **75.00%** (72.00–78.00) | **76.75%** (75.50–78.00) | Both learn and predict all 10 classes |
| `2^-15` (smaller step) | **49.25%** (48.00–50.50) | **51.25%** (48.00–54.50) | Learning slower under fixed 20-epoch budget |

Run: [GitHub Actions 40073270293](https://github.com/philsupertramp/integers/actions/runs/40073270293), Rust stable from the Actions toolchain, Linux x86_64, release build. Each run has a full per-epoch JSON learning curve, prediction counts by digit, FP32 non-finite parameter/output counts, integer output-extreme counts, wall time, RSS, and a dataset/source manifest. The end-of-epoch `2^-7` baseline had **0 non-finite FP32 parameters/outputs** and **0 integer extreme final-layer output mantissas**, but all 200 validation predictions chose one class. At `2^-12` both backends produced predictions in every digit class.

For the `2^-12` measurements, mean elapsed time per seed was 28.32 s (integer) versus 10.45 s (unoptimized scalar FP32); whole-process peak RSS was about 434 MiB for either backend, including both IDX datasets. These are implementation/runtime comparisons, not arithmetic-efficiency claims. On this small cohort, the 1.75 percentage-point mean accuracy difference between the two backends is **not** sufficient to identify an integer-specific convergence defect.

**Conclusion:** This strongly establishes a shared, oversized **effective step on summed gradients** as the principal cause of the initial both-at-chance CI result. It does *not* prove that integer gradients are numerically identical to FP32, that no additional quantization/STE defects exist, or that full held-out MNIST converges. Leave the 55,000/5,000/10,000 experiment and ADR-004 criteria open.

## CI policy after diagnosis

Use `lr_shift=12` with the same two-seed, 20-epoch 640/200 diagnostic in ordinary PR CI, preserving the full learning curves without hard-coding an accuracy threshold before maintainer review. Avoid repeatedly paying for the historical failure (`2^-7`) and the deliberately slow (`2^-15`) ablations in every PR; their run and artifacts remain linked above. Use the standard Iris `2^-7` baseline unchanged, because its architecture and training cohort are different.


## Resolution: per-sample means at the optimizer boundary (new implementation)

The integer `Linear` and `Conv2D` modules now count individual gradient-contributing samples across single and batched backward calls, resetting counts on `zero_grad`. Their ordinary `update(lr_shift)` applies actual **sample-mean** gradients **before** momentum and SGD, including arbitrary final-batch sizes. No change to `Module` or `Sequential` signatures was needed.

- For power-of-two batch counts, the dyadic mean is **exact** (increase scale exponent), with no premature per-example rounding.
- For non-power-of-two sizes, integer-only division uses a wider `i128` intermediate, round-to-nearest ties-to-even and adaptive precision to fit the mantissa. The representational error is bounded by half a unit at the selected dyadic scale, excluding any gradient saturation during accumulation.
- Momentum aligns its decayed velocity to the **finer** dyadic gradient scale so initial velocity (mantissa 0, exponent 0) cannot discard sub-unit means. Runtime forward, backward and optimizer updates remain integer-only.
- Without accumulated examples, `update` is a no-op. `zero_grad` resets accumulated counts but retains velocity.
- The paired FP32 reference now divides its summed gradients by the **actual current batch count** before updating velocity.
- Legacy experimental BatchNorm layers, excluded from the ADR-002 integer path, retain their previous summed-gradient optimizer convention.

The shared default `lr_shift=7` is restored for both Iris and MNIST **after** implementing true averaging. The earlier `lr_shift=12` experiment compensated a *summed* 32-sample batch; it is not the same algorithm as the corrected mean-gradient code with `lr_shift=12`. The historic ablation is preserved as evidence, not reinterpreted as a new measurement. Numerical and end-to-end training checks of corrected code are reported separately.
