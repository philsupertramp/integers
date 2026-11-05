# Backward-only guard3 POC: verified negative learning result

Source: [successful x86_64 workflow #42095053416](https://github.com/philsupertramp/integers/actions/runs/42095053416).
Raw, checksum-frozen archives:
[baseline](https://github.com/philsupertramp/integers/actions/runs/42095053416/artifacts/13448116990) and
[guard3](https://github.com/philsupertramp/integers/actions/runs/42095053416/artifacts/13446804935).

## Frozen protocol

Both treatment and control use **the same source SHA** (`6a8222b49060f495326d83d97726e6a90e83d5d9`, the pull-request CI merge commit),
**the same built binary SHA256** (`f064c3911b622f6d2b0133281bfe30b3d5e4c86a850e0d9bb20b29fdd7361092`), the same canonical MNIST *development* training IDX file hashes, seed4242,
He-v2 16-convolution architecture, train8192, held-out validation1042,
batch32, lr-shift7, 20 epochs, no official test access. Each arm trains its
own independent FP32 reference. Both independent FP32 references have the
**same confusion matrices and accuracy** in every epoch.

The correctness gate passed Rust unit, plan-protocol and trace-on/off
regressions, verifying unchanged first-minibatch forward ReLU gates and
pooling winners. Both complete 20-epoch arms succeeded in CI.
Independent archive verification: canonical plan digests correct; all four
JSON reports' SHA256 match metadata; the compressed/chunked original stdout
envelopes decompress byte-for-byte to the same JSON reports.

## Learning quality

| Backend | Validation at epoch 20 | Verdict |
| --- | ---: | --- |
| Independent FP32 (each arm) | **83.877159%** | Learns |
| Original integer, Conv propagated gradient scale7 | **9.788868%** | Fails |
| Guard3 integer, Conv propagated gradient target scale10 | **8.829175%** | Fails |

Treatment is **-0.959693 percentage points** against baseline. Both integer
runs remain near random chance across twenty epochs. This specific
backward-only precision increase is **not sufficient to rescue learning**.

## First-minibatch numerical observations at sampled epochs

The existing trace captures only the first minibatch of epochs1,5,10,20.
*Stage A* means the first eight Conv2D trainable modules (180 parameters).
The last stage-A convolution has 86,528 observed output elements in each
sampled first minibatch.

| Diagnostic | Baseline epoch 1 | Guard3 epoch 1 | Baseline epoch 5 | Guard3 epoch 5 | Baseline epoch 20 | Guard3 epoch 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage-A exactly-zero accumulated parameter gradients | 174/180 | **150/180** | 180/180 | **124/180** | 180/180 | **162/180** |
| Positive last-stage-A convolution outputs | 85/86,528 | 85/86,528 | **0/86,528** | **60,458/86,528** | **0/86,528** | **43,264/86,528** |
| Stage-A parameters changed by sampled update | 0/180 | 0/180 | 0/180 | 0/180 | 0/180 | 0/180 |
| Stage-A nonzero effective optimizer steps | 6 | 30 | 2 | 58 | **0** | **18** |
| Stage-A nonzero effective steps stochastic-rounded to zero | 6 | 30 | 2 | 58 | **0** | **18** |

At epoch 10, guard3's sampled stage-A update changed 1/180 parameters,
while baseline changed 0/180; do not infer all other minibatches have zero
updates. At epoch 20, globally, the baseline stochastic-rounds 2,099/2,104
nonzero effective steps to zero on the sampled minibatch, and guard3
stochastic-rounds 2,549/2,553. No measured saturation-blocked optimizer
updates in the sampled epoch20 step.

**Mechanistic interpretation:**

- The intervention *does* improve retention of nonzero stage-A gradients
  before the first optimizer step: 30 vs 6 nonzero parameter gradients
  on the first sampled minibatch. This supports a **causal role of gradient
  representation in first-pass gradient sparsity** at this fixed seed/model.
- Unlike baseline, the guard3 treatment retains positive stage-A
  activations at epochs5 and20, even though it never learns a useful
  classifier. Therefore **full stage-A forward-activation collapse is
  not required for failure to learn** in this configuration.
- Nevertheless, at the sampled stage-A optimizer step, **all nonzero
  effective steps still round to zero at epochs1,5,20**. Improving
  propagated gradient resolution does not automatically supply
  weight updates representable at the unchanged optimizer scale.
- This is not a proof that the optimizer alone is the remaining root
  cause: a previous fractional-carry-only treatment also failed, and
  the *combination* of carry and guard3 has not been tested. The
  guard3 intervention changes the effective decoded gradient-clipping
  threshold because clip bounds use raw mantissas; it does not isolate
  rounding from clipping completely.

**Stop boundary:** negative POC recorded. Do not automatically combine
guard bits with optimizer error feedback, tune more scales, or launch
another 20-epoch study. The research issue remains open, the PR remains
draft, and no foundational PRs are merged without maintainer approval.
