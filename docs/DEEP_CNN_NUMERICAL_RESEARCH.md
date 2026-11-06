# Deep CNN numerical research — evidence and disposition

**Consolidated:** 2026-11-06 (documentation-only review; no numerical changes).
**Authority:** [research observability #37](https://github.com/philsupertramp/integers/issues/37),
[depth study #38](https://github.com/philsupertramp/integers/issues/38),
and their frozen GitHub Actions artifacts.
**Code status:** All relevant research PRs #58–#64 are **draft and unmerged**.
This document does **not** approve any merge, change the approved integer-only
training boundary, or propose a new public precision policy ([#16](https://github.com/philsupertramp/integers/issues/16)).

## Executive finding

The integer backend can train real models; its **deep plain CNN training
is unreliable** under the investigated settings. In a frozen
16-convolution-layer He-v2 model at seed 4242, independent FP32 learns,
while the integer baseline and all three evaluated intervention choices
(optimizer carry, backward guard bits, and their combination) remain
near chance after 20 epochs.

The experiments **directly demonstrate** distinct numerical bottlenecks:
different forward ReLU gates, sparse backward gradients, substantial actual
nonzero-product-to-zero rounding, and ineffective weight updates. They
**do not establish one unique root cause** of deep-training failure.
An independently observed 20-layer model later learns despite substantial
initial rounding losses, so a one-layer or one-event explanation would be
premature.

This is a **negative but completed proof of concept** for the proposed
precision rescues. No automatic next optimizer/precision variants or long
training sweep is authorized by the results.

## Baselines that must not be conflated

| Protocol | Integer | Independent FP32 | Interpretation |
| --- | ---: | ---: | --- |
| Dense MNIST, three seeds, 20 epochs, official 10k test | **95.41% mean** | **96.87% mean** | Verified integer-only end-to-end learning; official test already consumed; do not retune on it |
| Deep MNIST, 16 conv / seed 4242, 20 epochs, 8,192 train / 1,042 development validation | **9.788868%** | **83.877159%** | Frozen failing case |
| Deep MNIST, 20 conv / same seed and development protocol | **80.518%** | **85.701%** | Counterexample to monotone depth-based collapse; this model eventually learns |

Dense results: [#15](https://github.com/philsupertramp/integers/issues/15),
[ADR-004](ADR_004.md). The 16/20-layer paired controls and diagnostic
definitions: [#38](https://github.com/philsupertramp/integers/issues/38),
[dual-observer run](https://github.com/philsupertramp/integers/actions/runs/41143025458).
The depth numbers are **different architecture workloads**; their validation
results are not an architecture-independent accuracy guarantee.

## Verified 2 × 2 intervention outcome

**Single seed:** 4242; **same source commit and binary within this factorial
run**; He-v2 16-conv network; train=8,192, validation=1,042, batch=32,
`lr_shift=7`, 20 epochs. Official MNIST test images were **not** used.
Each arm was independently run with a matched FP32 reference; FP32 final
accuracy was 83.877159% in all four. Source/binary/data/report hashes,
checksummed frozen manifests and archived stdout envelopes were independently
verified in the research issue.

| Integer arm | Target propagated gradient precision | Optimizer | Validation accuracy (epoch 20) | Versus baseline |
| --- | --- | --- | ---: | ---: |
| Baseline | original scale, typically `s=7` | stochastic SGD | **9.788868%** | — |
| Carry24 | original `s=7` | 24-bit sub-weight-LSB residual | **8.829175%** | −0.959693 pp |
| Guard3 | target `s=10` in Conv2D propagated gradients | stochastic SGD | **8.829175%** | −0.959693 pp |
| Guard3 + Carry24 | target `s=10` | residual carry | **9.980806%** | +0.191938 pp |

[All five successful CI jobs and raw artifacts](https://github.com/philsupertramp/integers/actions/runs/42117167245):
[baseline](https://github.com/philsupertramp/integers/actions/runs/42117167245/artifacts/13457474865),
[carry-only](https://github.com/philsupertramp/integers/actions/runs/42117167245/artifacts/13457104617),
[guard-only](https://github.com/philsupertramp/integers/actions/runs/42117167245/artifacts/13459792528),
[combined](https://github.com/philsupertramp/integers/actions/runs/42117167245/artifacts/13458768872).

No integer arm comes close to 80% validation. The descriptive two-factor
contrast `both − guard − carry + baseline` is **+2.111324 percentage
points**, but **one fixed seed is not an estimate of a population interaction**.
A green CI job means correct experiment execution, **not** a successful
learning-quality result. Earlier stand-alone runs
[carry](https://github.com/philsupertramp/integers/actions/runs/41866736064)
and [guard3](https://github.com/philsupertramp/integers/actions/runs/42095053416)
agreed on the negative verdict; the same-commit factorial is the primary
comparison.

## Mechanisms actually observed

1. **Before optimizer updates:** On the first epoch's first minibatch,
   174/180 accumulated first-stage trainable gradients in the integer
   baseline were exactly zero, versus 109/180 for the independent FP32
   reference. At the classifier, **179/320 integer loss-gradient entries**
   were nonzero, versus **320/320 FP32**. Hence optimizer rounding
   cannot explain *all* initial information loss.
2. **Forward divergence:** Pre-ReLU activation-sign disagreement is under
   ~1% through the first stage of the failing 16-conv model but jumps to
   **19.15%** at the first Conv2D after first pooling. At that first
   MaxPool2D, **21,201/21,632 (98.01%)** winner indices match FP32;
   among 431 differences, 417 involve ties and **only 14** are non-tied.
   **Pool-index routing alone is not a convincing explanation** for the
   much larger following Conv sign divergence. Pool-output magnitudes and
   subsequent convolutions have not been fully isolated.
3. **Backward resolution at a real operator:** At the second convolution
   stage on the first minibatch, **7,695/7,980 (96.4%)** nonzero exact
   weight-gradient multiplication products and **10,347/26,589 (38.9%)**
   nonzero propagated-gradient products **actually rounded to zero**.
   The 20-conv comparator also shows large loss rates (97.0%, 41.1%)
   yet eventually learns. These counts do *not* cover every add, dense
   gradient, max-pool scatter, or clipping operation.
4. **Guard3 alters pathway survival but not task success:** At sampled
   epoch 20, the last stage-one Conv has **43,264/86,528 (50.0%)**
   positive outputs under guard3 alone versus **zero** in baseline;
   nevertheless guard3 accuracy is **8.83%**. Thus complete early-stage
   ReLU inactivity is **not necessary** for failure to learn.
5. **The combined arm changes the trajectory:** At sampled epoch 20,
   guard3+carry has **85/86,528 (~0.10%)** positive outputs at the same
   Conv, **122/180** exactly-zero accumulated stage-one gradients,
   **56** nonzero effective stage-one steps, and **zero changed
   stage-one parameters on that sampled update**. All 56 effective
   steps were *deferred* by carry (not stochastic-rounded-to-zero).
   Stage-one parameters changed **zero** times on that minibatch in
   every arm; **do not extrapolate sampled counts to the full epoch**.

Primary evidence:
[dual observer](https://github.com/philsupertramp/integers/actions/runs/41143025458),
[initial-backward probe](https://github.com/philsupertramp/integers/actions/runs/41876562598),
[pool/rounding probe](https://github.com/philsupertramp/integers/actions/runs/42061755442),
[guard study](https://github.com/philsupertramp/integers/actions/runs/42095053416),
[factorial study](https://github.com/philsupertramp/integers/actions/runs/42117167245).
Detailed readouts reside in the research branches:
[initial backward](https://github.com/philsupertramp/integers/blob/research/first-backward-gate-probe/docs/FIRST_BACKWARD_FINDINGS.md),
[pooling/rounding](https://github.com/philsupertramp/integers/blob/research/pool-routing-backward-rounding/docs/POOL_ROUNDING_FINDINGS.md),
[guard3](https://github.com/philsupertramp/integers/blob/research/backward-gradient-guard-bits/docs/BACKWARD_GUARD3_FINDINGS.md).

**Limitations:** The numerical probes observe the **first minibatch** of
specific epochs (1, 5, 10, 20), or the single initial minibatch, not all
training updates. Guard3's fixed raw-mantissa clipping bound changes its
decoded clipping threshold when the gradient exponent changes. Carry24
changes update timing/noise and has finite 2^-24 weight-LSB resolution.
Independent FP32 follows its own arithmetic and gradients, so every
FP32-small-gradient/INT-zero pair is not automatically attributable to
the *same* rounding event. Zero exact products must not be counted as
quantization loss. Single-seed results neither prove arithmetic bugs nor
establish universal integer/depth limitations.

## Research PR disposition — selective retention, **no merges**

| PR | Contribution | Current recommendation |
| --- | --- | --- |
| [#58](https://github.com/philsupertramp/integers/pull/58) | Reproducible paired depth-training harness and independent FP32 reference | **Candidate to retain selectively** after code/protocol review; do not merge the entire stack by implication |
| [#59](https://github.com/philsupertramp/integers/pull/59) | Opt-in layer activation/gradient/update observations, true SGD rounding counts | **Best general-purpose diagnostics to keep** in a small follow-up extraction if needed |
| [#60](https://github.com/philsupertramp/integers/pull/60) | 24-bit sub-LSB optimizer carry, negative A/B | **Research-only, do not promote** to default optimizer |
| [#61](https://github.com/philsupertramp/integers/pull/61) | Position-matched forward ReLU gates, full backprop boundary snapshots | **Useful targeted diagnostic**, opt-in and bounded; not a production default |
| [#62](https://github.com/philsupertramp/integers/pull/62) | MaxPool winner/tie comparison and actual Conv2D backward product rounding | **Useful targeted diagnostic**, opt-in and bounded; selective extraction only |
| [#63](https://github.com/philsupertramp/integers/pull/63) | Guard3 propagated-gradient scale treatment, negative A/B | **Research-only, do not promote** as validated precision solution |
| [#64](https://github.com/philsupertramp/integers/pull/64) | Same-commit 2×2 study and combined treatment, negative learning result | **Preserve frozen reports; do not promote** combined algorithm |

**Preserve now:** checksummed plans, original JSON reports and CI artifacts,
independent FP32 controls, opt-in observing definitions and invariance tests.
**Not integrated now:** any runtime instrumentation, new optimizer API or
precision policy. PR #59/#61/#62 are **proven useful as research observers**,
but a production-quality extraction still needs its own review for
performance overhead, schema compatibility, test coverage and opt-in default.
In particular [#37](https://github.com/philsupertramp/integers/issues/37)
still has unchecked instrumentation overhead/memory acceptance criteria.

Research branches remain available and draft PRs open as historical evidence.
Do not close them or merge stacked experiments automatically; the maintainer
can decide separately whether to close/archive individual PRs after this
document is reviewed.

## Open questions; explicitly deferred

- **Forward value/scale behavior** from first pool output to first stage-two
  convolution, including magnitudes and clipping—not just winning indices.
- **Gradient-scale and clipping contracts:** retain useful signal without
  altering raw-mantissa clip thresholds unintentionally; exact relative
  attribution of loss, STE, Conv, dense and optimizer remains unresolved.
- **Generalizability:** multiple architectures/seeds and a new independent
  dataset would be needed for broader claims. The official MNIST test is
  already consumed for the dense reference and was excluded from this depth
  research.
- **Operator policy:** adaptive power-of-two/layerwise scaling belongs to
  [backlog #41](https://github.com/philsupertramp/integers/issues/41);
  storage/accumulation contracts to [RFC #16](https://github.com/philsupertramp/integers/issues/16).
  Neither is authorized for implementation by this consolidation.

## Operational handoff and stop

**Research POC complete:** retain findings, do not repeat 20-epoch ablations
or tune guard-bit/carry settings merely because more knobs are available.
Numerical research can remain open as a backlog with its limitations clearly
recorded.

**Separate next engineering work (not started by this PR):** use the
*working dense model* and [physical Raspberry Pi profile #47](https://github.com/philsupertramp/integers/issues/47)
to optimize a measured portable `Linear` backward hotspot with numerical,
RNG and **actual Pi 5** before/after gates. The existing
[#18](https://github.com/philsupertramp/integers/issues/18)
already contains physical-device profiling evidence; do not claim Pi work
has never been measured. Keep PR #9 into `master` unmerged and retain
ADR-002/003 platform and integer-only runtime constraints.

**No architectural decision changed, no code merged, no new research run.**
