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
`cifar` requires both `parquet-support` and `image-decode`, `mnist_to_parquet` requires `parquet-support`, and `tensor_bench` requires `bench`. Compilation is not an assertion that dataset-dependent models learn. In addition to the separate feature-smoke workflow, the bundled-example workflow runs XOR and the checked-in Iris train/test splits as **runtime smoke tests**; MNIST/CIFAR full dataset runs remain outside CI (#15). Track feature and integration regressions under [#13](https://github.com/philsupertramp/integers/issues/13) and wider hygiene under [#20](https://github.com/philsupertramp/integers/issues/20).

Formatting and lint diagnostics:
```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
```
The issue #20 hygiene workflow on `issue-20-rust-hygiene` runs rustfmt, the locked full suite, all-target compilation, and strict library Clippy. Strict all-target Clippy and rustfmt are blocking checks, enabled after a successful GitHub Actions run on the hygiene branch. Other regression, example-runtime, and feature-smoke workflows remain independent. A failing pre-existing check should be noted, not concealed.

**Toolchain policy:** Rust edition 2024; current development and validation use stable Rust 1.95.0 on x86_64 Linux. An MSRV has not been adopted or tested; do not claim compatibility with an earlier compiler. Hardware priorities are x86_64 correctness and Raspberry Pi 5 AArch64 after parity tests. All default checks are dataset-free; dataset-dependent full MNIST/CIFAR runs are excluded from local fast gates.

## Experiments and claims
Record Rust compiler version, git commit, target hardware, seed, dataset identity and split, hyperparameters, iterations, accuracy on a complete held-out test set, runtime, memory, and methodology. Distinguish arithmetic correctness from model convergence and runtime performance. When comparing to FP32, specify whether kernels and implementation quality are comparable. Don't commit licensed datasets without confirming redistribution rights.

## Decisions
Agents can decide code-level mechanics. The maintainer approves direction-setting tradeoffs documented in [docs/ARCHITECTURE_DECISIONS.md](docs/ARCHITECTURE_DECISIONS.md) before major rewrites. ADR-001 conditionally selects the candidate; ADR-002 requires integer-only runtime forward/backward/updates, with floats allowed in preparation, initialization, independent oracles, diagnostics and reports. BatchNorm1D/2D and Softmax have previously documented runtime FP paths and remain noncompliant: use the explicit experimental mode for them, not the default fail-closed integer backend ([#26](https://github.com/philsupertramp/integers/issues/26)). ADR-003 prioritizes x86_64 correctness then Raspberry Pi 5 AArch64. PR #9 remains unmerged.
