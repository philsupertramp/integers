# Normalization-free residual block POC (#39)

This is **not a ResNet-20 training result**. It only demonstrates a
shape-preserving identity skip for an integer-only `Sequential` branch.

`ResidualBlock::new(branch)` implements `y = x + F(x)` and the corresponding
`dL/dx = dL/dy + dF/dx * dL/dy` using the existing dyadic arithmetic.
When nested in `Sequential`, it deliberately requests **full branch input
gradients** from `backward_batch_with_input_grad`; when it is itself the first
layer of ordinary training, it only accumulates the branch's parameter
gradients and discards the complete model-input derivative.

**Deliberate limits:** identity path only, matching input/output shapes,
no projection, same-size 3×3 padding, integer BatchNorm or CIFAR training.
Skip addition aligns to the finer dyadic exponent and **rejects**
unrepresentable alignments/sums instead of silently clipping. That makes
the arithmetic invariant explicit in this POC, but is not a production
residual-add saturation/STE policy.

Offline contract checks:

```sh
cargo test --locked --test residual_block
cargo test --locked --lib skip_add
cargo fmt --all -- --check
cargo check --locked --all-targets
```

The tests check hand-calculated convolutional skip values, the identity
gradient, branch parameter updates, nested propagation to earlier trainable
layers, mixed-scale signed addition and explicit overflow/shape rejection.
No dataset, GPU, Pi, or long training run is needed.

**Stopping point:** if these tests and CI pass, the POC is complete. A
follow-on ResNet-20-style model would require a separately reviewed,
fixed topology and an independent FP32 control (issue #39), with
normalization-free results kept distinct from any future integer BN model.
Do not reopen the completed deep-plain-CNN precision ablations (#38).
