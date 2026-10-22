# Project audit and working baseline — 2026-10-22

This is a **source inspection**, not a completed build or hardware benchmark. GitHub source files were inspected at `master` commit `d15c46240cb68ac122a45637fbc9418135cf76eb` and the open `big-refactor` PR #9. Local clone/build was unavailable in the inspecting environment. Revalidate after changes.

## Branches and work already started
| Area | `master` | PR #9 / `big-refactor` |
| --- | --- | --- |
| Scalar representation | `Scalar`/`Numeric`, `i32` and `f32` | `Dyadic {v: i32, s: u32}` |
| Neural-network layers | Linear, activations, RNN; Conv2D and Flatten placeholders | Linear, Conv2D, pooling, dropout, batch norm, softmax, flatten |
| Experiments | Iris, MNIST, parity | Iris, MNIST (including CNN), CIFAR-10, XOR |
| Benchmarks | None identified | Criterion benchmarks and memory benchmark |
| CI | Local `.githooks/post-receive` only; hard-coded machine paths | Same; no GitHub workflow found |
| Documented user-facing API | README describes `master` | README is not updated to match branch API |

### Critical items to investigate
1. **Dyadic scale preservation** on PR #9: `Dyadic(v,s) = v * 2^(-s)`, but `align(x,s_target)` appears to right-shift when `s_target > x.s`; value-preserving conversion would require increasing the mantissa (subject to overflow policy). A test named `align_exact_power_of_two` expects `(4,0) -> (1,2)`, which changes the represented value 4 to 0.25. `requantize` has a related discrepancy. Resolve definitions with an independent reference oracle before accepting PR #9.
2. **PR #9 API/module integration**: examples and docs refer to `integers::dataset_loaders` / generic `DatasetBuilder::<i32>`, whereas the new loader lives under `data::dataset_loaders` and `DatasetBuilder` is non-generic. Validate `cargo check --all-targets`, doctests, feature combinations, and README.
3. **Weak empirical evidence**: refactor MNIST example uses 256 training examples per epoch and 20 fixed test examples for early stopping; report complete held-out test accuracy with fixed seeds and avoid tuning on the test set.
4. **`master` ARM kernel**: `dot_product_neon_raw` takes `&[i32]` while using `vld1q_s8`, does not safely handle arbitrary lengths, and is unproven on the Pi 5. Don't optimize the discarded branch before architecture selection.
5. **Correctness and portability**: audit shifting/overflow bounds, negative rounding, shape checks, stochastic determinism, gradient checks, and integer vs FP32 reference behavior.

## Acceptance gates
- **G0 — agent ready:** maintainers can locate instructions, select an issue, review an architectural decision, and see automated compile/unit-test feedback. This is the scope of the agent-foundation PR.
- **G1 — architecture chosen:** maintainer approves representation and training arithmetic boundary, and decides disposition of PR #9.
- **G2 — correctness:** reproducible arithmetic property tests, layer forward/backward oracles, at least one full deterministic end-to-end training evaluation.
- **G3 — performance:** portable reference kernels established first; then optimize and benchmark on Raspberry Pi 5 (AArch64, 8 GB).

## Limitations / non-goals of this audit
No tests, profiler, or Pi experiments were executed. Code presence does not certify numerical correctness or model convergence. Exact scheduling and issues are tracked in GitHub; this file describes the rationale and checkpoints, not a second backlog.
