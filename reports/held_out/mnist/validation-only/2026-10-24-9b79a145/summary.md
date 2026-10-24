# Held-out experiment report

Dataset: **mnist**; smoke mode: **False**; CI sample: **False**.
Commit: `9b79a1450b51ab4cb2befb264bf8c1a8a4a75108`; Rust: `rustc 1.95.0 (59807616e 2026-04-14)`.
Target: `Linux-6.12.95-1-MANJARO-x86_64-with-glibc2.43`; generated at `2026-10-24T12:21:27.976478+00:00`.

Source SHA-256 (each exact input file):

- `data/mnist/train-images-idx3-ubyte`: `ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db`
- `data/mnist/train-labels-idx1-ubyte`: `65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5`

## Validation-only results (test NOT loaded or evaluated)

| Backend | Seeds | Final validation accuracy mean [min, max] | Wall time mean [min, max], s | Peak RSS mean [min, max], MiB |
| --- | --- | --- | --- | --- |
| fp32 | 1337, 2027, 4242 | 95.13 [94.90, 95.30]% | 182.86 [176.08, 189.28] | 410.60 [410.55, 410.66] |
| integer | 1337, 2027, 4242 | 93.53 [93.48, 93.56]% | 724.95 [715.42, 730.38] | 410.65 [410.57, 410.80] |

Each seed/backend JSON contains its **full per-epoch learning curve**, split sizes,
fixed hyperparameters, an explicit unevaluated-test marker, and process-local wall time / peak RSS.
Training accuracy is measured online (before parameter updates), while
validation is measured after each epoch; no test evaluation was performed.
Validation is used for diagnostics; epochs and hyperparameters are specified
in advance and never selected using test performance.

**Comparison limits:** FP32 is a transparent scalar Rust reference, not an
optimized BLAS/PyTorch implementation. It uses the identical dense architecture,
starting weights, sample order, target and squared-error objective, power-of-two
learning rate and momentum convention, but continuous arithmetic differs from
dyadic quantization, stochastic rounding, and STE. Latency reflects kernels and
implementation maturity, not intrinsic integer-versus-float hardware efficiency.

**VALIDATION ONLY:** the test partition was not opened, loaded, hashed, or evaluated. This score is used for selection, not an independent generalization estimate.