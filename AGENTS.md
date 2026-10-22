# Agent instructions — integers

Read this file, [CONTRIBUTING.md](CONTRIBUTING.md), [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md), and [docs/ARCHITECTURE_DECISIONS.md](docs/ARCHITECTURE_DECISIONS.md) before changing implementation code.

## Repository reality
- Default branch: `master`, based on `Scalar`/`Numeric`; `src/lib/nn/conv.rs` includes nonfunctional placeholders.
- Open [PR #9](https://github.com/philsupertramp/integers/pull/9) (`big-refactor`) replaces this with a dyadic arithmetic and layer design. It is **not accepted architecture**. Always identify which branch you are examining.
- Do not port, merge, discard, or build on top of PR #9's API without an explicit architectural decision by the maintainer.
- The project targets research and reproducibility first; performance claims must be benchmarked.

## Responsibilities and approval
Agents own implementation choices, tests, documentation, and small refactors within an approved design. Escalate high-level choices before acting: numerical representation, training arithmetic boundary, public API compatibility, supported platforms, external dependencies that dictate architecture, license, and merging the refactor. Record decisions in `docs/ARCHITECTURE_DECISIONS.md`.
Never push to `master` or merge pull requests autonomously; use focused feature branches and reviewable PRs.

## Numerical invariants
- If using `Dyadic { v, s }`, the represented real value is `v * 2^(-s)`. Cross-scale conversions must preserve this value up to explicitly bounded quantization error; increasing `s` means **finer** precision.
- Put a deterministic test before each arithmetic/gradient fix. Cover mixed scales, negative values, clipping, extreme shifts, saturation/overflow, RNG seeds, and empty/mismatched tensors.
- Keep integer-hot-path promises narrow and measurable; floats may be useful for reference implementations or reports until the maintainer chooses otherwise.
- Validation requires small analytic examples, differential oracles, and independent held-out evaluation; don't infer correctness from training accuracy alone.
- Do not silently change quantization semantics to make tests pass.

## Local validation
From repository root:
```sh
cargo test --locked --lib
cargo check --locked --all-targets
cargo test --locked --doc
```
Use `cargo fmt --all -- --check` and `cargo clippy --all-targets -- -D warnings` to inspect hygiene, but record pre-existing failures instead of mixing a large formatting sweep into functional work.
Dataset-dependent examples and benchmarks are **not** CI acceptance tests without explicitly documented data provenance, seeds, and execution requirements. Do not commit data dumps, checkpoints, generated binaries, or access tokens.

## PR handoff
Each PR must identify its base, linked issue, architectural decision (or `none`), numerical invariants touched, tests run (and tests not run), and any accuracy/performance results with device, compiler, flags, input, and baseline. Prefer small, reviewable changes. Treat stale docs or example programs as unverified until checked against the selected branch.
