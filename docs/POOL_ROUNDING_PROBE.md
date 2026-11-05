# Paired pooling routing / exact convolution-backward rounding probe

This builds on the verified [first-backward localization](FIRST_BACKWARD_FINDINGS.md)
from [draft PR #61](https://github.com/philsupertramp/integers/pull/61).

## Narrow question

At depth 16 / seed 4242, why does the first convolution of stage two
show ~19% pre-ReLU gate disagreement from the matched FP32 model when
earlier first-stage convolutions disagree at fewer than 1% of positions?
And during the first true backward pass, where do exact nonzero integer
products become zero at the fixed `quant_shift=7`?

No optimizer or arithmetic formula changes. Both independent models use
the same decoded initial weights and the same 32 MNIST training IDs.
Observation occurs before the first optimizer update. The development
train/validation split and official test-set boundary remain frozen.

## What is actually measured

- **Pool winners:** first-minibatch cached MaxPool2D source indexes, compared
  position-for-position against the independent FP32 max-pool winners.
  Also count equal-valued tied maxima under each backend's original
  comparison semantics. Report mismatches **with a tie on either side**
  separately from mismatches with **neither side tied**. Ties do not imply
  an arithmetic error; a different input can cause one backend to see a
  tie but not the other.
- **Backward multiplication events:** for each integer Conv2D,
  count each actual weight-gradient `mul(g, x, q)` and propagated-gradient
  `mul(g, w, q)` call, including nonzero exact i64 input products
  and those whose **actual stochastic-rounded output mantissa is zero**.
  Count the exact nonzero products strictly below one output mantissa
  LSB, and count nonzero rounded terms whose accumulator remains unchanged.
  The latter is *not automatically* attributed to rounding: saturation or
  other causes can also leave an accumulator unchanged.
- These are `Conv2D` multiplications only. Other operations (`add`,
  `ste_requantize`, pooling scatter accumulation, dense backprop) have
  no per-operation round-event counters here. No claim that the counters
  describe all backward precision loss.
- Both sources execute each original operator **once** and consume the same
  RNG stream. Integer code performs only integer comparisons and counts;
  FP32 is confined to the independent oracle and diagnostic reporting.

## Diagnostic criteria

If MaxPool source index disagreement is rare except for tied windows,
concentrate on post-pool convolution's `mul`/accumulation and gradient
resolution; pooling itself has no extra numeric quantization operation.
If non-tie winner disagreements are large, examine pre-pool activation
magnitudes first. If actual Conv2D backward nonzero->zero rates are large,
test a separately isolated backward-only precision treatment later,
but do not assume it will fix 20-epoch learning.

Everything is a *single first-minibatch* observation with fixed seed,
not a new multi-seed learning-quality benchmark. No merges without
maintainer approval.
