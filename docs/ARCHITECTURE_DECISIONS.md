# Architectural decision register

**Status: awaiting maintainer choices.** Keep this file updated when decisions are recorded; implementation agents should link the corresponding GitHub issue and PR. No default below counts as approval.

| ID | Question for maintainer | Recommended direction (not approved) | Status |
| --- | --- | --- | --- |
| ADR-001 | Base: retain `master` Scalar/FP32 abstraction, adapt PR #9 Dyadic design, or develop a smaller hybrid? | Evaluate PR #9 as candidate; do not merge until arithmetic, compile, and regression gates pass | Pending |
| ADR-002 | What does `integer-only training` mean? No floating point anywhere, or integer forward/backward/update hot paths with float references, initialization, loading, and reporting permitted? | Integer hot path; float permitted outside it for data/reference/evaluation, clearly disclosed | Pending |
| ADR-003 | First-class hardware target? | x86_64 correctness CI plus Raspberry Pi 5 AArch64 reference/optimization; MCU support deferred | Pending |
| ADR-004 | Minimum evidence for a meaningful milestone? | Reproducible full-held-out MNIST experiment, numerical reference tests, accuracy/runtime/memory baseline; set pass thresholds *after* reference baselines | Pending |
| ADR-005 | Open-source distribution license? | Maintainer selects and adds a recognized LICENSE and Cargo package metadata before public release | Pending |

## Approval protocol
Architectural decisions are made by the maintainer in the linked GitHub issue or PR discussion. Once approved, replace `Pending` with the chosen rule, link the decision, and describe consequences. Agents may propose alternatives, benchmarks, and implementation details without treating recommendations as accepted.

## Constraints until approval
- Don't merge PR #9, rename core public APIs, remove FP32 oracle code, choose bit widths as permanent API commitments, introduce new supported targets, or claim fixed performance/accuracy targets.
- Permitted without architectural approval: issue triage, independent reference tests, CI/agent documentation, build investigations, reproducible benchmarks, and minimal correctness regressions on an isolated branch.
