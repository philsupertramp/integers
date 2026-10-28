# Held-out experiment report

Dataset: **mnist**; architecture: **cnn**; smoke mode: **False**; CI sample: **False**.
Commit: `ce1e9975b0d3ca7ca98e1a54548cb9f32be4cf02`; Rust: `rustc 1.95.0 (59807616e 2026-04-14)`.
Target: `Linux-6.12.95-1-MANJARO-x86_64-with-glibc2.43`; generated at `2026-10-25T03:29:24.450956+00:00`.

Source SHA-256 (each exact input file):

- `data/mnist/train-images-idx3-ubyte`: `ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db`
- `data/mnist/train-labels-idx1-ubyte`: `65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5`
- `data/mnist/t10k-images-idx3-ubyte`: `0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7`
- `data/mnist/t10k-labels-idx1-ubyte`: `ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2`

## Results (held-out test evaluated once per run, after fixed epochs)

| Backend | Seeds | Test accuracy mean [min, max] | Wall time mean [min, max], s | Peak RSS mean [min, max], MiB |
| --- | --- | --- | --- | --- |
| fp32 | 1337, 2027, 4242 | 97.26 [97.17, 97.37]% | 115.44 [111.25, 117.90] | 434.10 [434.05, 434.14] |
| integer | 1337, 2027, 4242 | 96.73 [96.53, 96.90]% | 527.47 [501.02, 549.25] | 436.85 [436.84, 436.86] |

Each seed/backend JSON contains its **full per-epoch learning curve**, split sizes,
fixed hyperparameters, final test result, and process-local wall time / peak RSS.
Training accuracy is measured online (before parameter updates), while
validation is measured after each epoch. Final test is evaluated only once.
Validation is used for diagnostics; epochs and hyperparameters are specified
in advance and never selected using test performance.

**Comparison limits:** FP32 is a transparent scalar Rust reference, not an
optimized BLAS/PyTorch implementation. Both backends use identical topology,
decoded starting weights, sample order, target and 0.5·sum(squared-error) objective, power-of-two
learning rate and momentum convention, but continuous arithmetic differs from
dyadic quantization, stochastic rounding, and STE. Latency reflects kernels and
implementation maturity, not intrinsic integer-versus-float hardware efficiency.

**CNN baseline:** Conv(1→4,3) / ReLU / 2x2 pool / Conv(4→8,3) / ReLU / 2x2 pool / Flatten(200) / Dense(64) / ReLU / Dense(10). The FP32 path uses separate scalar kernels and continuous derivatives; the integer path uses quantization, stochastic rounding, clipping and STE. Both optimize 0.5·sum((logits-target)^2), not cross entropy.

**Prior test disclosure:** MNIST's official 10k test set was already used in issue #15. Any CNN final-test result is an already-known benchmark, not a pristine independent generalization test.
