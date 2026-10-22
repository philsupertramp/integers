# Contributing to integers

This is an experimental Rust research project, not yet a stable training library. Start with [AGENTS.md](AGENTS.md) and the [project audit](docs/PROJECT_STATUS.md). There are two divergent implementations: `master` and [PR #9](https://github.com/philsupertramp/integers/pull/9); do not assume their APIs are interchangeable.

## Workflow
1. Pick a GitHub issue and confirm dependencies / architectural gates are resolved.
2. Create a branch from the explicitly chosen base; don't push implementation changes to `master`.
3. Add a regression test reproducing a failure or specify a deterministic acceptance test for new behavior.
4. Keep implementation PRs focused. State whether arithmetic is exact, stochastic, saturating, or intentionally approximate.
5. Report verification results and limitations in the PR; await review to merge.

## Baseline commands
```sh
cargo test --locked --lib
cargo check --locked --all-targets
cargo test --locked --doc
```
Formatting and lint diagnostics:
```sh
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
```
A failing pre-existing check should be noted, not concealed. CI presently targets compilation and unit-level validation; wider quality gates will follow once the baseline is known.

## Experiments and claims
Record Rust compiler version, git commit, target hardware, seed, dataset identity and split, hyperparameters, iterations, accuracy on a complete held-out test set, runtime, memory, and methodology. Distinguish arithmetic correctness from model convergence and runtime performance. When comparing to FP32, specify whether kernels and implementation quality are comparable. Don't commit licensed datasets without confirming redistribution rights.

## Decisions
Agents can decide code-level mechanics. The maintainer approves direction-setting tradeoffs documented in [docs/ARCHITECTURE_DECISIONS.md](docs/ARCHITECTURE_DECISIONS.md) before major rewrites. Existing PR #9 is a proposal, not a ratified design.
