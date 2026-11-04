# Initial backward divergence probe (research issue #37)

This opt-in experiment investigates the frozen 16-layer CNN, seed 4242,
where a 20-epoch fractional-carry optimizer ablation did not recover learning.

The goal is **localization before changing the arithmetic**. No optimizer
intervention, learning-rate change, altered initialization, or extra FP32
forward pass is allowed. No official MNIST test data is loaded.

## Matched first minibatch

Use the standard He-v2 model initialized at seed 4242, same 8192 MNIST
development-training examples/1042 development-validation examples, batch32,
lr_shift7, original integer optimizer, one epoch. `--probe-first-backward`
is restricted to validation-only integer CNN experiments with `--trace-layers`
and exactly one epoch.

The first minibatch of epoch one uses the existing epoch-order selection.
The integer model performs its standard **one and only** forward and
backward operation. Its integer-only numerical boundary summary records,
for **every** layer (including ReLU, pools, Flatten), how many dyadic
gradients enter and leave the backward operation, exact zeros, maximum
absolute raw mantissa and nonzero scale bounds. The model's input gradient
remains discarded as in baseline training, and is deliberately reported
`null`.

The **independent FP32 reference** was initialized from identical decoded
integer weights. It executes its own forward and backward *once* for each
of the exact same minibatch image IDs, without any optimizer step. We
record corresponding FP32 gradient-boundary zeros, magnitudes and nonfinite
values. It does not consume integer RNG, nor does its execution change
the integer training process.

At each initial pre-ReLU Conv2D/hidden Linear, the probe also compares
**each activation's sign at the same sample and position**, counting
`both_active`, `integer_only`, `fp32_only`, and `both_inactive`.
This is stronger than comparing aggregate activation-zero percentages:
the latter can hide individual gate mismatches.

Accumulated trainable parameter-gradient zero counts are reported for
both independent backends before any optimizer step. This is a first-pass
numerical *diagnostic*, not a strict tensor arithmetic equivalence proof;
the independent model has continuous FP32 loss gradients while the integer
model has quantized loss gradients.

## Interpretation

Compare per-layer gate disagreements **forward** and the fraction of
nonzero gradient elements **backward**. The first sustained layer with
many `fp32_only` positive gates implicates forward quantization. If the
gates agree but the backward gradient entering a convolution is already
mostly or wholly zero in integer while FP32 is nonzero, prioritize
the preceding (later-in-forward) backward operation and its gradient
quantization/scale. A ReLU that loses most gradients only in one backend
is evidence of different forward gates, not necessarily a bug in ReLU.

With one minibatch and one seed, localization is descriptive, not proof
that changing that operation alone will rescue 20-epoch learning. Do not
merge draft PRs #58/#59/#60 or the foundational PR #9 based on this probe.
