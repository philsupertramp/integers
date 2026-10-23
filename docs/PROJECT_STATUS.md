# Project status — candidate branch

**Updated:** 2026-10-23. This tracks the selected `big-refactor` **candidate**, not the legacy `master` API. Architectural authority remains with [issue #10](https://github.com/philsupertramp/integers/issues/10) and the [decision register](ARCHITECTURE_DECISIONS.md).

## Branch distinction and verified evidence

- `master` retains legacy `Scalar`/`Numeric` APIs and compilation debt. [PR #23](https://github.com/philsupertramp/integers/pull/23) addresses a syntax error, not the full legacy build.
- `big-refactor` implements `Dyadic { v: i32, s: u32 }` with `Tensor`/`TensorView`, `nn::Module`/`Sequential`, Linear, ReLU, Softmax, Conv2D, MaxPool2D, Flatten, Dropout and BatchNorm1D/2D, plus data and optional features. It is **not** an accepted public/stable API.
- [PR #24](https://github.com/philsupertramp/integers/pull/24) **already merged** to `big-refactor` as [`3e7892e`](https://github.com/philsupertramp/integers/commit/3e7892e01ec5fbbbff4f99c2e96b7874e966a3d3). Its [pre-merge run 39847262737](https://github.com/philsupertramp/integers/actions/runs/39847262737) **actually passed** 28 library tests, 8 independent dyadic regression tests, 5 doctests and `cargo check --locked --all-targets`. This is *previous* default-feature evidence, not current feature-branch or optional-feature CI evidence.
- [#25](https://github.com/philsupertramp/integers/issues/25) ports documentation from draft legacy-targeting PR #22, updates README, and adds a no-dataset public API test plus per-feature build checks. **Observed on draft PR #27 commit `be1cf21`:** [default workflow 39863734424](https://github.com/philsupertramp/integers/actions/runs/39863734424) passed 28 library tests, 8 independent dyadic regressions, **2 public API smoke tests**, 5 doctests and default all-target compilation; [feature workflow 39863734401](https://github.com/philsupertramp/integers/actions/runs/39863734401) passed separate **no-default**, **parquet-support**, **image-decode**, **bench**, and **all-features** all-target checks. These checks compile examples/benches but do not execute dataset-based training. Both workflows logged pre-existing warnings, including Rust 2024 unsafe-op-in-unsafe-fn warnings for the memory benchmark (addressed in this PR) and other unused imports/variables/naming warnings tracked under #20; do not claim the warning baseline is clean. Any subsequent PR commit requires its own current-head CI revalidation. PR #9 into `master` must **not** be merged without further independent maintainer approval.

## #12 dyadic invariants — partial

**Complete:** PR #24 repaired scale alignment direction and saturation, corrected the contradictory alignment unit test, and added eight independent oracle-backed tests. They cover mixed-scale conversion/add/sub, signed clipping, negative and large-scale equivalence, simple multiplication/division, stochastic determinism and signed rounding symmetry.

**Outstanding:** Specify the complete supported rounding/overflow/domain contract; examine division with extreme shifts and scale arithmetic (`div` uses `(i64) << p`), very large exponents, unbiased stochastic behavior outside ordinary shift ranges, and broader generated/property tests. Current tests do not prove arbitrary inputs or end-to-end model accuracy. Keep [#12](https://github.com/philsupertramp/integers/issues/12) open.

## #13 integration — partial

**Complete:** PR #24 repaired BatchNorm debug printing, stale tensor API examples and feature-gating for `mnist_to_parquet`. Default-feature unit, regression, doctest and all-target checks passed in the above Actions run. #25 adds candidate README/contribution docs and public import/one-step smoke coverage.

**Outstanding:** Revalidate CI on the final PR head after integration changes; default and all five optional feature checks have passed at `be1cf21` (see evidence above). Data-dependent example execution remains untested; compilation does not establish runtime correctness. Route feature integration failures to [#13](https://github.com/philsupertramp/integers/issues/13), and hygiene/MSRV/Clippy/rustfmt/full-test issues to [#20](https://github.com/philsupertramp/integers/issues/20). Keep #13 open until genuinely complete.

## Known training risks

- **#14:** Retained layer forward/backward gradients, shape errors, cache isolation, gradient accumulation, saturation/STE and independent FP64 oracles are not validated.
- **#26:** ADR-002 requires *all* actual integer-backend forward/backward/optimizer update calculations to be integer arithmetic. BatchNorm1D and BatchNorm2D currently perform **f64** during forward and backward; they are **noncompliant** for integer training until redesigned or excluded with enforcement tests. Float in initialization, dataset preparation, tests, independent oracles and reporting is allowed.
- The default MNIST CNN example trains on **256 sampled train images** and checks **20 sampled test images** per epoch. Do not present this as held-out convergence evidence. No accepted full-dataset experiment, x86_64 model-level accuracy proof or measured Pi 5 AArch64 performance yet.
- The current loss-gradient helpers use `zip` and/or assume shared scale; input length/shape assumptions and division overflow behavior need deliberate coverage before production claims.

## Handoff

1. For [#14](https://github.com/philsupertramp/integers/issues/14), work from current `big-refactor`; add tiny deterministic forward/backward fixtures for Linear, ReLU, Conv2D, pooling and losses; compare against independent oracles with documented quantization tolerances; exercise batch caches and shape validation. Separate STE approximations from analytic gradients.
2. For [#26](https://github.com/philsupertramp/integers/issues/26), inventory floats used inside all forward/backward/`update` paths, differentiate setup-only floats, and rework or explicitly exclude/guard BN1D/2D. Add automated integer-backend enforcement tests before asserting integer-only runtime training.
3. Keep #12/#13/#20 follow-up debt open and linked; don't change approved ADRs or PR #9's merge status as part of these handoffs.
