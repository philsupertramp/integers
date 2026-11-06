# Architectural decision register

**Last updated:** 2026-10-24  
**Authority:** Project maintainer; see [issue #10](https://github.com/philsupertramp/integers/issues/10).  
**Status:** ADR-001 through ADR-004 approved by the maintainer as high-level direction and prospective numerical regression criteria. Correctness, implementation integration, independent test integrity, and explicit PR merge approval are **still required**; recommendations for other ADRs are not approvals.

| ID | Decision | Status | Source |
| --- | --- | --- | --- |
| ADR-001 | Use PR #9 `big-refactor` as the **candidate** numerical foundation, subject to correctness and build gates; don't merge it now | **Approved, conditional** | [#10](https://github.com/philsupertramp/integers/issues/10) |
| ADR-002 | Integer arithmetic for forward, backward, and optimizer/weight update computations; floats allowed in preparation/reference/support work | **Approved** | [#10](https://github.com/philsupertramp/integers/issues/10) |
| ADR-003 | x86_64 correctness first, then Raspberry Pi 5 (8 GB) AArch64 optimization; defer MCU and GPU training | **Approved** | [#10](https://github.com/philsupertramp/integers/issues/10) |
| ADR-004 | Numerical correctness and prospective dense-MNIST regression criteria: integer mean/worst ≥94.0%, ≤2.0 pp mean gap to FP32; time/RSS report-only | **Approved 2026-10-24** — explicitly *not* a pre-registered 2026 test claim or PR #9 merge authorization | [ADR-004](ADR_004.md), [#15](https://github.com/philsupertramp/integers/issues/15) |
| ADR-005 | Open-source license and external/public API policy | **Pending maintainer approval** | [#19](https://github.com/philsupertramp/integers/issues/19) |
| ADR-006 | Storage bit widths, accumulator precision, and which combinations become public API commitments | **Pending maintainer approval** | [#16](https://github.com/philsupertramp/integers/issues/16) |

## Verified partial integration (not approval to merge PR #9)

[PR #24](https://github.com/philsupertramp/integers/pull/24) has already merged into `big-refactor` at `3e7892e`. [Run 39847262737](https://github.com/philsupertramp/integers/actions/runs/39847262737) passed 28 library tests, 8 independent dyadic regression tests, 5 doctests and all-target compilation. This is partial #12/#13 progress only: arithmetic extremes, optional features, layer gradients [#14](https://github.com/philsupertramp/integers/issues/14) and integer-only runtime audit [#26](https://github.com/philsupertramp/integers/issues/26) remain open. Current project instructions and evidence are on the candidate branch by way of [#25](https://github.com/philsupertramp/integers/issues/25).

## ADR-001 — Candidate numerical foundation

**Decision:** Build the *evaluation and gated integration plan* around [open PR #9](https://github.com/philsupertramp/integers/pull/9), not a fresh numerical-core redesign or unconditional continuation of the old `Scalar`/`Numeric` API. PR #9 is a **candidate**, not accepted implementation.

**Rationale:** It contains meaningful layer, dyadic arithmetic, and benchmark work to salvage, but source inspection identified important correctness and integration risks. Choosing the candidate avoids duplicating implementation work while allowing revision or rejection if it fails validation.

**Required gates before acceptance/merge:**
1. [#12](https://github.com/philsupertramp/integers/issues/12): document and verify invariant `Dyadic(v,s) = v * 2^(-s)`; test and correct scale conversion, rounding, saturation/overflow, negative values and shape behavior against an independent oracle.
2. [#13](https://github.com/philsupertramp/integers/issues/13): run and fix library/all-target build, feature configurations, doctests, and stale examples/documentation.
3. [#14](https://github.com/philsupertramp/integers/issues/14): validate retained layer forward/backward computations and gradients; test explicit STE behavior separately.
4. Review actual resulting diff and receive a **separate explicit maintainer merge approval**. Approval of this ADR is **not** that approval.

**Consequences:** Agents can plan experiments and propose isolated correctness/build fixes around `big-refactor` after this record is written, but may not merge PR #9 or impose its present API as stable. If the gates reveal untenable flaws, bring an alternative foundation to the maintainer rather than choosing it unilaterally.

## ADR-002 — Integer training arithmetic boundary

**Decision:** The integer backend must perform **forward propagation, backward propagation, and parameter/optimizer updates using integer arithmetic**. This applies to every operation used by these training passes, including normalization and regularization layers when enabled. Don't hide floating-point calculations behind an integer-facing API.

**Allowed uses of floating point:** Dataset loading/preprocessing and quantization setup, parameter initialization before training, independent FP32/FP64 reference models and numerical test oracles, diagnostics, timing and reporting. Document the setup/runtime boundary and identify any temporary noncompliant experimental components. The goal is not a blanket prohibition on all floating-point instructions in the application.

**Known incompatibility in PR #9:** Its BatchNorm implementation currently computes statistics/normalization or gradients using floating-point operations. It must be replaced with integer computations or excluded from the integer training path until implemented and tested; allowing such calculations in the runtime path would require an additional maintainer decision. This is a source-audit finding, not a measured numerical result.

**Consequences:** Numerical correctness first, performance optimization later. Agents choose the implementation technique and references but don't weaken the agreed arithmetic boundary.

## ADR-003 — Platform and hardware focus

**Decision:** Verify numerical behavior using portable **x86_64** tests and independent reference models before tuning performance. The first optimized deployment/benchmark target is the **Raspberry Pi 5 (8 GB RAM), AArch64**.

**Consequences:** Retain a correct scalar implementation and compare optimized AArch64 kernels to it. Record device, compiler, build flags, accuracy, speed, peak memory and thermal state. Defer MCU and GPU compute backends; do not imply support merely because of shared code. Related: [#17](https://github.com/philsupertramp/integers/issues/17) and [#18](https://github.com/philsupertramp/integers/issues/18).

## ADR-004 — Approved measurement and numerical regression policy

**Decision (maintainer-approved 2026-10-24):** Adopt the [ADR-004 acceptance hierarchy](ADR_004.md): independent dyadic/layer oracle and integer runtime-boundary compliance remain correctness gates; a **future fixed 20-epoch, three-or-more-seed dense-MNIST reference workload** has prospective regression targets of **≥94.0% mean integer accuracy, ≥94.0% worst-seed integer accuracy and a ≤2.0 percentage-point deficit** versus matched independent scalar FP32. Fixed sample counts, seed schedules, dataset checksums, and complete diagnostics are required. **Do not conflate experimental floating-point BatchNorm/Softmax with integer-compliant execution** (see #26).

**Hardware policy:** report x86_64 accuracy, wall time, peak process RSS and reproducibility; do not impose hardware-independent speed or RAM limits before comparable controlled measurements on target hardware, especially Raspberry Pi 5 AArch64. The measured unoptimized scalar-FP32 ratio is a descriptive implementation-specific baseline, not an intrinsic hardware advantage claim.

**Scientific limitation:** these numeric targets were set **after** observing the October 24 held-out MNIST result (integer 95.41% vs scalar FP32 96.87%). They are **approved future regression criteria**, not retroactive pre-registration and not a license to tune on the already-consumed official MNIST test. PR #34 can proceed to review against `big-refactor`; PR #9 to `master` still requires separate explicit authorization. The scientific measurement portion of #15 is complete, but integration and release claims remain gated.

## Decisions intentionally left open
- **ADR-005:** Maintainer selects license and release/API policy; do not invent authorization to relicense or publish a stable API.
- **ADR-006:** Evaluate INT8/INT16 storage and wider accumulation as research options, but don't commit to public width or overflow policies without review.

## Approval protocol and operating rules

- The maintainer owns the decisions above, approval of significant departures, licensing, and merges. Agents own scoped implementation, tests, benchmarks, docs and internal refactors within those decisions.
- Link architecture-dependent changes to [#10](https://github.com/philsupertramp/integers/issues/10) and any relevant acceptance-gate issues. Record actual validation results, including failures; no undocumented reinterpretation of dyadic scale or integer-only semantics.
- No direct pushes to `master`. No merge of [PR #9](https://github.com/philsupertramp/integers/pull/9) without subsequent explicit approval. Use reviewable branches/PRs.
- The first approved record lives in [foundation PR #22](https://github.com/philsupertramp/integers/pull/22) against legacy `master`; this adapted copy is proposed for `big-refactor` through [#25](https://github.com/philsupertramp/integers/issues/25). Neither copy approves merging PR #9 or waiving its correctness and integration gates.
