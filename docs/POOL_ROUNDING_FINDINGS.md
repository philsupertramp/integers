# Verified first-minibatch MaxPool routing and Conv2D backward rounding

Evidence: [successful native x86_64 run #42061755442](https://github.com/philsupertramp/integers/actions/runs/42061755442), with three successful jobs: formatting/Rust/Python regression and on/off equivalence, plus depth 16 and 20.
Raw full JSON: [16 layers](https://github.com/philsupertramp/integers/actions/runs/42061755442/artifacts/13430748318) and [20 layers](https://github.com/philsupertramp/integers/actions/runs/42061755442/artifacts/13431103642).

## Protocol

Fixed MNIST development-training split, 8192 train and 1042 validation examples, He-v2 initialized depth CNNs, seed 4242, batch32, learning shift 7, no official test partition. The observed work is the **real first training minibatch of epoch 1, before any parameter update**. The independent FP32 reference starts with exactly decoded integer initial weights and processes the same minibatch indexes. No operation is replayed, no additional RNG draws are added, and the trace-on/off smoke check verified identical learning outcomes/confusion matrices. Only Conv2D dyadic multiplications are instrumented for per-operator backward rounding; other operators remain outside the counter scope.

## MaxPool2D winner positions

| First-stage pool | Depth 16 | Depth 20 |
| --- | ---: | ---: |
| Pooled outputs | 21,632 | 21,632 |
| Same winner index | 21,201 (98.01%) | 21,432 (99.08%) |
| Different winner index | 431 (1.99%) | 200 (0.92%) |
| Different winner with a tie in at least one backend | 417 | 188 |
| Different winner without a tie in either backend | **14 (0.065%)** | **12 (0.055%)** |
| Integer tied-max windows | 21,555 | 21,552 |
| FP32 tied-max windows | 20,992 | 21,312 |

**Localization:** first-stage MaxPool winner routing agrees overwhelmingly; most disagreements reflect one or both backends having equal maximum candidates. Under the frozen 2x2 pool, the next convolution's ~19%/20% forward sign disagreement is **not a comparably large winner-index mismatch**. This does not demonstrate that post-pool *value magnitudes* agree (they are not yet measured). Even matched routing can lead to different values in the succeeding 3x3 convolution.

Later, the second pooling layer exhibits much larger winner disagreements (**1,332/6,400 = 20.81%** at depth16; **2,044/6,400 = 31.94%** at depth20), predominantly involving ties (1,179 and 1,724 respectively). This later discrepancy occurs **after** the previously localized stage-two divergence and must not be mistaken for its original cause.

## Actual Conv2D backward multiplication rounding

At each original `mul(g, input, quant_shift)` for weight gradients and `mul(g, weight, quant_shift)` for propagated gradients, the probe records the **exact i64 mantissa product**, then the **actual dyadic result of that single original stochastic-rounding operation**. No second operation or RNG draw is performed. Counters consider only *nonzero* exact products.

| Second convolutional stage | Depth 16 | Depth 20 |
| --- | ---: | ---: |
| Nonzero exact parameter-gradient products | 7,980 | 10,365 |
| Those rounding to zero | **7,695 (96.4%)** | **10,055 (97.0%)** |
| Nonzero exact propagated-gradient products | 26,589 | 39,317 |
| Those rounding to zero | **10,347 (38.9%)** | **16,177 (41.1%)** |

At the very first Conv2D **after stage-one pooling** (forward indices 17 and 21):

| Actual multiplication | Depth 16 | Depth 20 |
| --- | ---: | ---: |
| Weight-gradient nonzero product → zero | 311/314 (99.0%) | 315/329 (95.7%) |
| Input-gradient nonzero product → zero | 3,630/5,606 (64.8%) | 6,084/8,156 (74.6%) |

For the entire second stage, **all measured nonzero exact parameter-gradient products have magnitude below one mantissa unit before the shift by 7**. Stochastic rounding occasionally preserves them; most yield a zero output. This directly confirms substantial resolution loss at the Conv2D backward multiply sites. The first-stage Conv2D traces are very sparse in nonzero exact products because **incoming gradients are already zero** over large regions. One cannot attribute those zero-input products to stochastic rounding: their exact product is zero.

The optional `nonzero_term_unchanged_accumulator` field reports an unchanged destination after adding a nonzero rounded dyadic term; it does not by itself identify saturation or add-site rounding. We do **not** claim to have instrumented loss gradients, dense backpropagation, max-pool backward scatter, STE clipping, or every dyadic addition.

## Conclusions and boundary

1. **Original pooling winner selection is unlikely to be the primary reason for the initial forward activation-sign divergence**: first-stage matched winners exceed 98% at depth16 and 99% at depth20, with only 14/12 non-tie disagreements.
2. **A concrete backward precision limit is confirmed at the actual operator**: 96–97% of nonzero weight-gradient products in stage two round to zero at fixed shift7, and 39–41% of propagated-gradient products do as well.
3. This is *one first minibatch at a fixed initialization*. It is not an epoch-wide or multi-seed rate, a proof of a bug, nor proof that this alone causes the 16-layer 20-epoch training collapse. First-stage zero gradients can be downstream of multiple quantization sites. The 20-layer model previously recovered with longer training even though its first-minibatch rounding loss is comparably high.

**Research next step, not started:** if revisiting, isolate higher-precision **backward-only** multiplication/gradient propagation in a controlled arm without changing forward activations, optimizer, seed or schedule; separately inspect matched post-pool value magnitudes if forward divergence remains unresolved. Stop at these measurements before automatically implementing a new optimizer or full training experiment.

All PRs #62, #61, #60, #59, #58 and foundation PR #9 remain unmerged unless maintainer explicitly approves.
