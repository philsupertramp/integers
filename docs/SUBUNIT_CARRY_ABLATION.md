# Frozen sub-unit optimizer ablation — issue #37

**Status:** experimental; no production optimizer API or general numerical
correctness claim. This branch is intentionally stacked on draft PR #59
which is itself stacked on the unmerged depth study PR #58.

## Hypothesis and existing observations

At depth 16 / seed 4242 / MNIST, the last convolution of the first stage has
100% nonpositive cached outputs by the sampled epoch 5, exactly-zero stage-A
gradients at sampled epoch 20, and a validation score near chance even after
20 epochs. Depth 20 with the same seed recovers late. Earlier dual-observer
reports show ~99% of nonzero post-momentum integer optimizer steps in the
sampled batches round to zero, but this **cannot by itself explain genuinely
zero gradients** in the inactive stage A.

The causal intervention tests whether ***retaining sub-LSB optimizer update
residue before an activation pathway becomes dead*** prevents that collapse.
It does **not** test larger learning rates, changed weight quantization, extra
training epochs, wider architecture or CIFAR.

## Treatment versus control

- **Control:** exactly the existing per-parameter momentum, sample-averaging,
  stochastic SGD weight quantization, activations, ReLU, loss, architecture,
  integer numerical representation, and 20-epoch schedule.
- **Treatment (`--optimizer-mode subunit_carry24`):** keep the same
  post-momentum dyadic effective step `u = v 2^(-s)` and learning-rate shift
  `L`. For weight scale `s_w`, ideal stored-weight delta is
  `u * 2^(s_w-L)` in mantissa LSB units. Accumulate that desired update in
  signed i128 at **2^-24 of one stored-weight LSB**. When total magnitude
  crosses one LSB, truncate toward zero and apply that representable
  integer step; preserve fractional remainder across batches. This is
  deterministic *error feedback*; it preserves the nominal learning rate,
  but changes update timing and stochastic noise. Extremely tiny steps below
  2^-24 LSB truncate to zero; i32 saturation discards overflow residue.
- **RNG matched:** the treatment still performs the baseline stochastic
  rounding exactly once at the original optimizer site, deliberately
  discarding its sampled delta, so it consumes the same RNG draws and does
  not confound subsequent forward/backward streams. Momentum updates follow
  the baseline arithmetic, with no FP operations in integer runtime.
- **FP32:** separately executed, independently implemented baseline control,
  initialized from the same decoded integer weights.

A/B results are not statistical noninferiority. If carry rescues learning,
we establish that the **optimizer quantization policy contributes causally**
for this seed/workload, not that it is the only cause or that carry is the
right production optimizer. If not, the failure also depends on forward/
backward rounding, quantized activation bottlenecks, or dead ReLU paths.
A 20-layer matching-seed ablation is useful as a comparison but not required
to establish a 16-layer recovery.

## Evidence rules

- The experiment runner freezes the optimizer choice into its checksummed
  manifest, validates matching report metadata, and never loads the official
  test partition. Same MNIST development train=8192/validation=1042,
  batch32, shift7, seed4242, 20 epochs, He-v2 CNN architecture.
- Both control and treatment use separate predeclared output directories.
  The baseline control is retrained at the **same source commit**, so a
  relative comparison does not depend on old binaries/commits.
- Each variant optionally captures the first minibatch at epochs 1/5/10/20;
  carry-mode traces distinguish `subunit_deferred_steps` from the baseline's
  real stochastic `quantized_to_zero_updates`. Those are different events.
- Test control and treatment tracing on/off with identical per-epoch
  confusion matrices and updates at the same seed. Low-level tests verify
  momentum and RNG matching and controlled carry accumulation.
- The run is a research ablation, **not an improvement merged into the
  integer backend**. We require review of the curves and first-stage
  activation-zero progression before claiming a causal result.
