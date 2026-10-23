# Agent instructions — integers

Read this file, [CONTRIBUTING.md](CONTRIBUTING.md), [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md), and [docs/ARCHITECTURE_DECISIONS.md](docs/ARCHITECTURE_DECISIONS.md) before changing implementation code.

## Repository reality
- `master` is legacy `Scalar`/`Numeric`; `big-refactor` is the **approved candidate only** ([#10](https://github.com/philsupertramp/integers/issues/10)), NOT a merge approval for [PR #9](https://github.com/philsupertramp/integers/pull/9).
- Base candidate issue branches on the **current `big-refactor` HEAD** and target `big-refactor` with draft PRs. Do not cherry-pick legacy code from PR #22 or repeat fixes in [merged PR #24](https://github.com/philsupertramp/integers/pull/24).
- PR #24's commit `3e7892e` repaired initial arithmetic and compilation issues, with 28 library tests, 8 independent regressions, 5 doctests and an all-target check passing in [run 39847262737](https://github.com/philsupertramp/integers/actions/runs/39847262737). This does not prove full correctness or verify subsequent commits.
- The project targets research and reproducibility first; performance claims must be benchmarked.

## Responsibilities and approval
Agents own implementation choices, tests, documentation, and small refactors within an approved design. Follow approved ADR-001/002/003. Escalate changes to high-level numerical representation, training arithmetic boundary, public API compatibility, supported platforms, external dependencies that dictate architecture, license, or merging the refactor. Record decisions in `docs/ARCHITECTURE_DECISIONS.md`.
Never push to `master` or merge PR #9 autonomously; use focused feature branches and reviewable PRs.

## Numerical invariants
- If using `Dyadic { v, s }`, the represented real value is `v * 2^(-s)`. Cross-scale conversions must preserve this value up to explicitly bounded quantization error; increasing `s` means **finer** precision.
- Put a deterministic test before each arithmetic/gradient fix. Cover mixed scales, negative values, clipping, extreme shifts, saturation/overflow, RNG seeds, and empty/mismatched tensors.
- **ADR-002 is decided:** All actual integer-backend forward, backward, and optimizer/parameter-update computations must be integer-only. Floats are permitted for preprocessing, initialization before training, reference oracles, tests, reporting and diagnostics, not for hidden runtime training math. Current BatchNorm1D/2D use `f64` in runtime forward/backward and **must not be claimed compliant** until [#26](https://github.com/philsupertramp/integers/issues/26) reworks or excludes them.
- Validation requires small analytic examples, differential oracles, and independent held-out evaluation; don't infer correctness from training accuracy alone.
- Do not silently change quantization semantics to make tests pass.

## Local validation
From repository root:
```sh
cargo test --locked --lib
cargo test --locked --test dyadic_regression
cargo test --locked --test public_api_smoke
cargo test --locked --doc
cargo check --locked --all-targets
```
See CONTRIBUTING.md for per-feature smoke commands (no-default-features, parquet-support, image-decode, bench and all-features). Use `cargo fmt --all -- --check` and `cargo clippy --all-targets -- -D warnings` to inspect hygiene, but record pre-existing failures instead of mixing a large formatting sweep into functional work.
The bundled XOR and Iris examples have a **runtime smoke** workflow; it exercises the checked-in train/test Iris TSV splits but does not assert convergence or held-out generalization. MNIST/CIFAR require external datasets and are not executed in normal CI. Document data provenance, seeds and execution requirements for full experiments; do not commit data dumps, checkpoints, generated binaries or access tokens.

## PR handoff
For [#14](https://github.com/philsupertramp/integers/issues/14), provide independent seeded layer/gradient oracles including quantization and STE tolerances. For [#26](https://github.com/philsupertramp/integers/issues/26), inventory actual runtime float use across layers and updates and guard noncompliant components. Passing candidate builds, layer oracles, or bundled example smoke tests does not establish full held-out accuracy or Raspberry Pi performance (#15/#18).

Each PR must identify its base, linked issue, architectural decision (or `none`), numerical invariants touched, tests run (and tests not run), and any accuracy/performance results with device, compiler, flags, input, and baseline. Prefer small, reviewable changes. Treat stale docs or example programs as unverified until checked against the selected branch.
