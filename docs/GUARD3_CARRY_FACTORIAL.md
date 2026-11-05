# 16-layer 2 × 2 experiment: backward guard3 × optimizer carry24

**Status:** research-only ablation on draft PR #63, originally rooted in the
verified numerical observability and pooling/gradient-rounding studies.

## Hypothesis

Guard3 alone retains first-stage gradients and positive activations but
does not learn (8.83% MNIST validation versus 9.79% baseline). Its sampled
nonzero optimizer updates still almost all round to zero. Carry24 alone
also failed to restore training (8.83% validation in a previous study).

The distinct *interaction hypothesis* is that retaining the upstream
backpropagated gradient **and** carrying sub-LSB parameter updates can
jointly make early-layer training effective where either intervention
alone did not.

## Frozen factorial design

| Arm | Conv2D propagated gradient target | Integer optimizer |
| --- | --- | --- |
| baseline | original scale, usually s7 | original stochastic SGD |
| carry24 | original scale, usually s7 | deterministic 24-bit sub-LSB residual, normal RNG draws discarded |
| guard3 | fixed target scale s10 | original stochastic SGD |
| guard3_carry24 | fixed target scale s10 | deterministic 24-bit sub-LSB residual |

Each arm is run **at the same source SHA and binary hash**, same He-v2
16-convolution architecture, seed 4242, development MNIST train8192 /
validation1042, batch32, lr shift7, **20 epochs**, and identical
sample ordering. Each archive includes its own independent FP32 control.
Only the original two interventions are combined: no different learning
rate, clipping policy, batch size, forward quantization, wider network,
new architecture, or access to official MNIST test.

Stochastic draws are consumed at the original integer multiplication and
optimizer sites, including in carry24, preserving the seeded stochastic
forward/backward RNG order. The carry-only module is copied from draft
PR #60 rather than reimplemented. The original no-treatment paths must
remain bit-identical to prior reference results.

On selected first minibatches at epochs 1, 5, 10 and 20, independently
report stage-A gradient sparsity, positive first-stage Conv outputs,
changed parameters, and either true SGD round-to-zero steps or
carry-deferred effective steps (**not** mislabeled as stochastic rounding).
The report/plan records both arm identities, checksums binary/data/protocol,
and is validated before accepting any quality result.

## Interpretation

- **Combined rescue:** supports an interaction or joint sufficiency of
  these chosen backward and optimizer representations for this seed/model,
  not proof of a unique defect, generalization, or production suitability.
- **No rescue:** this pair of discrete interventions is inadequate.
  It does not show that integer training is impossible.
- **Negative guard or carry singles:** use the *same-commit* four-arm data;
  previous separately built results remain prior evidence only.
- **Important confound:** guard3 changes the decoded threshold of raw
  mantissa-based gradient clipping. Carry24 changes the stochastic update
  distribution and timing even when nominal LR is held fixed. Neither
  individual treatment isolates rounding independent of those effects.
- **Limited resolution:** carry is 24-bit sub-LSB, not exact arbitrary
  precision. Tiny steps may remain below the carry grid.

## Stop criterion

Once all four 20-epoch arms have passed correctness and generated verified
reports, compare accuracy and early-stage activation/gradient/update traces,
record a verdict in the issue/PR, and stop. No more numerical variants, seed
sweeps, or merging PRs without explicit maintainer approval.
