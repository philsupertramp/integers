# Verified first-backward localization — 16/20 conv, seed 4242

Source: [successful native x86_64 workflow #41876562598](https://github.com/philsupertramp/integers/actions/runs/41876562598)
at diagnostic branch head `1da7edda4262eb1217e2f33a6b7b30381dd68855`.
[16-layer raw trace](https://github.com/philsupertramp/integers/actions/runs/41876562598/artifacts/13346044428),
[20-layer raw trace](https://github.com/philsupertramp/integers/actions/runs/41876562598/artifacts/13347805530).

The invariant gate passed and both 1-epoch jobs succeeded. Exact experiment:
He-v2 initialized integer + decoded-weight independent FP32 CNN, seed 4242,
MNIST development split train=8192 / validation=1042, batch=32,
lr shift=7, no official MNIST test data. Both backends observe the *same first
32 training IDs*, **before the very first optimizer update**. The model
continues one ordinary integer epoch after the comparison; probe-on/off
produces exactly matching learning outcomes and confusion matrices in
16/20-convolution smoke tests.

## Empirical localization

| First-minibatch numerical observation | 16 convolution | 20 convolution |
| --- | ---: | ---: |
| Maximum positional sign disagreement across first-stage conv outputs | 0.88% | 1.23% |
| Positional sign disagreement at *first convolution of stage 2*, after stage-1 MaxPool2D | **19.15%** (index 17) | **20.26%** (index 21) |
| FP32-active / integer-inactive at that stage-2 convolution | **5,831 / 30,976** | **6,156 / 30,976** |
| Integer-active / FP32-inactive at same convolution | 100 / 30,976 | 119 / 30,976 |
| Nonzero gradients entering stage-1 MaxPool2D, integer / FP32 | **1,080 / 10,383** | **1,282 / 10,636** |
| Nonzero FP32 gradients at that boundary with abs value **below 2^-7** | **7,694 / 10,383 (74.1%)** | **9,122 / 10,636 (85.8%)** |
| Nonzero loss gradients entering final classifier layer, integer / FP32 | **179 / 320** | **220 / 320** |
| FP32 loss gradients with abs value below 2^-7 | **288 / 320** | **288 / 320** |

At the first-stage pooling backward boundary, the integer dyadic
gradients' reported nonzero scale is `s=7`, meaning 1/128 is one
representable mantissa unit. Nonzero real magnitudes strictly below that
unit cannot be expressed as nonzero dyadics **at scale 7**. The large
proportion of independent FP32 gradient magnitudes below the unit,
combined with far fewer nonzero integer gradients, strongly implicates
*backward-path gradient resolution/underflow* at this boundary.
This is a representability observation, **not proof that each particular
FP32 gradient was rounded away**: the backends' forward activations and
backpropagated gradients are not identical.

The first major mismatch in **forward activation signs** emerges between
the first-stage last convolution and the first second-stage convolution,
*across a MaxPool2D + Conv2D transition*. The sign measurements alone
do not establish whether input magnitude changes, max-pool winner routing,
or the second-stage convolution's dyadic multiplications/accumulations
first create the divergence. Both stages can contribute.

Loss-head gradient sparsity is present even before any backpropagation:
integer MSE gradients are differences of already quantized output logits
and targets at scale 7, while FP32 gradients are continuous. This is a
separate early source of information loss, not an optimizer update
problem. The existing sub-LSB optimizer ablation [PR #60](https://github.com/philsupertramp/integers/pull/60)
already demonstrated no recovery of the 16-layer failure by 20 epochs.

## What we can and cannot conclude

**Supported:** the observed initial numerical bottleneck is upstream of
optimizer updates and spans (a) coarse gradient representation, with
sub-LSB FP32 magnitudes at the first-stage backprop boundary, and (b)
material forward activation-sign divergence immediately after the first
pooling transition. The 16/20 controls show this pattern across both
depths even though their later training outcomes differ.

**Not established:** that one specific arithmetic implementation is
incorrect; that every small FP32 gradient directly corresponds to an
integer zero at the same tensor position; that MaxPool2D itself, as
opposed to its quantized inputs or following Conv2D, is causal; or that
fixing this initial mismatch alone will restore 20-epoch accuracy.
This is one fixed seed, one minibatch, and one small architecture family.

## Targeted next discriminating probe

1. Record **MaxPool2D winners** for both backends at the same positions
   and compare their input and output values using a reference-only
   postprocessing step. If winner indices diverge, investigate
   quantized value ties/routing; if not, compare stage-2 Conv2D output
   magnitudes before ReLU.
2. For the first-stage backward interface, measure exact quantization
   events (nonzero pre-rounding input -> zero integer output) and compare
   a narrowly bounded **gradient-only scale-retention** arm against
   the default shift=7. Keep the forward model, optimizer, seed, and
   sample order unchanged for that diagnostic; do not immediately commit
   to another full 20-epoch ablation.

Do not merge PRs #61, #60, #59, #58 or foundation PR #9 without review.
