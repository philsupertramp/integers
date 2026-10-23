# Contributing to integers

This is an experimental Rust research project, not yet a stable training library. Start with [AGENTS.md](AGENTS.md) and the [project audit](docs/PROJECT_STATUS.md). The selected candidate is `big-refactor`; `master` remains legacy. Its API is experimental. PR #9 must not be merged without separate maintainer approval.

## Workflow
1. Pick a GitHub issue and confirm dependencies / architectural gates are resolved.
2. For candidate issues, branch from the current `big-refactor` HEAD and target `big-refactor` with a draft PR; do not push implementation changes to `master`.
3. Add a regression test reproducing a failure or specify a deterministic acceptance test for new behavior.
4. Keep implementation PRs focused. State whether arithmetic is exact, stochastic, saturating, or intentionally approximate.
5. Report verification results and limitations in the PR; await review to merge.

## Baseline commands
```sh
cargo test --locked --lib
cargo test --locked --test dyadic_regression
cargo test --locked --test public_api_smoke
cargo test --locked --doc
cargo check --locked --all-targets
```
Feature-specific compilation checks (no external datasets downloaded):
```sh
cargo check --locked --all-targets --no-default-features
cargo check --locked --all-targets --no-default-features --features parquet-support
cargo check --locked --all-targets --no-default-features --features image-decode
cargo check --locked --all-targets --no-default-features --features bench
cargo check --locked --all-targets --all-features
```
`cifar` requires both `parquet-support` and `image-decode`, `mnist_to_parquet` requires `parquet-support`, and `tensor_bench` requires `bench`. Compilation is not an assertion that dataset-dependent examples train correctly. See the separate feature-smoke workflow and track failures under [#13](https://github.com/philsupertramp/integers/issues/13)/[#20](https://github.com/philsupertramp/integers/issues/20).

Formatting and lint diagnostics:
```sh
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
```
A failing pre-existing check should be noted, not concealed. CI presently targets compilation and unit-level validation; wider quality gates will follow once the baseline is known.

## Experiments and claims
Record Rust compiler version, git commit, target hardware, seed, dataset identity and split, hyperparameters, iterations, accuracy on a complete held-out test set, runtime, memory, and methodology. Distinguish arithmetic correctness from model convergence and runtime performance. When comparing to FP32, specify whether kernels and implementation quality are comparable. Don't commit licensed datasets without confirming redistribution rights.

## Decisions
Agents can decide code-level mechanics. The maintainer approves direction-setting tradeoffs documented in [docs/ARCHITECTURE_DECISIONS.md](docs/ARCHITECTURE_DECISIONS.md) before major rewrites. ADR-001 conditionally selects the candidate; ADR-002 requires integer-only runtime forward/backward/updates, with floats allowed in preparation, initialization, independent oracles, diagnostics and reports. BatchNorm1D/2D currently compute floating-point runtime normalization/gradients and are noncompliant pending [#26](https://github.com/philsupertramp/integers/issues/26). ADR-003 prioritizes x86_64 correctness then Raspberry Pi 5 AArch64. PR #9 remains unmerged.
