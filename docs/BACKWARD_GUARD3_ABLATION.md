# Fixed-target backward propagation precision ablation — #37

This research-only branch tests one variable after the verified [pooling and
backward multiplication study](POOL_ROUNDING_FINDINGS.md). Do not merge the
experimental arm into production or into the underlying draft PRs.

**Hypothesis:** the 16-convolution MNIST model with seed4242 fails partly
because stage-two backward products and upstream propagated gradients lose
sub-LSB information at dyadic scale `s=7` before the optimizer is invoked.
Preserving **three fractional guard bits only in Conv2D propagated-input
gradients** may keep earlier pathways trainable.

## Isolated treatment

- Baseline Conv2D backprop: `mul(g, w, q=7)`, output gradient shift typically
  `s=7`. The quantized parameter-gradient term `mul(g, input, q=7)`
  is unchanged.
- Treatment: `--backward-guard3` only for experimental 16/20-layer integer
  MNIST validation. Each Conv2D propagated-gradient product selects a `q`
  for **fixed target output exponent `s=10`** (`7+3`), instead of blindly
  decreasing `q` at each convolution (which would grow scale and overflow).
  Initially `g.s=7, w.s=7, q=4`, then subsequent layers typically have
  `g.s=10, w.s=7, q=7`. Both remain one-draw stochastic rounding. Original
  forward, trainable parameter-gradient multiply formula, update/SGD code,
  weight representation, activation bits, architecture, initialization and
  official-test exclusion remain untouched. Earlier layers may accumulate
  richer parameter gradients because their incoming gradient representation
  is finer, even though their own weight-gradient multiplication formula
  remains unchanged.
- **Important limitation:** the Conv2D gradient clip is still expressed as a
  raw mantissa bound and is not rescaled in this minimal experiment. Its
  decoded clipping threshold differs when incoming gradient exponents change.
  Therefore any learning improvement cannot be attributed exclusively to
  stochastic rounding precision unless clipping is independently shown
  inactive or normalized in a later experiment. This is disclosed rather
  than silently changing two numerical operations at once.
- The diagnostic regime is first-minibatch epochs 1/5/10/20, not full-epoch
  numerical telemetry. `--probe-first-backward` remains limited to one
  epoch and is used only in the small correctness smoke. Only research flags
  toggle the treatment; no new public optimizer architecture is introduced.

## Frozen paired trial

**20 epochs**, seed 4242, original He-v2 CNN with 16 convolutions, exactly
8192 development-training and 1042 development-validation examples,
batch32, lr-shift7, independent FP32 reference, no official test set. Both
baseline and guard3 runs are built from the **same exact commit** and
checked via the SHA256-frozen plan + recorded binary and MNIST hashes.
The comparator also requires trace-on/off prediction and confusion parity
*within each optimizer arm*. A successful CI job does not imply numerical
learning-quality success.

**POC stop boundary:** once both arms finish, compare stage-one exact-zero
gradients, changed parameters, positive-activation fraction and final
validation. If neither arm retains a working learning pathway, stop;
do not automatically tune a larger guard-bit count, retry 20 more epochs,
or expand into architecture redesign. The observed first-backward
multiplication product losses and pooling winners remain separate evidence.
