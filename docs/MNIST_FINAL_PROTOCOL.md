# Locked MNIST final-test protocol — issue #15

**Frozen on 2026-10-24 using only training and validation data. At the time of this freeze, the official MNIST 10,000-image test partition had not been evaluated.** The actual test was subsequently executed on the frozen parameters. Do not adjust the protocol in response to any official test score.

## Evidence before freeze

Both completed paired three-seed studies used the same 60k training source (55k for training, 5k for validation), integer dyadic and independent FP32 dense MLP backends. Full original records:
- [10-epoch baseline](../reports/held_out/mnist/validation-only/2026-10-24-9b79a145/README.md): integer **93.53%**, FP32 **95.13%** mean final validation.
- [20-epoch diagnostic](../reports/held_out/mnist/validation-only/2026-10-24-20epoch-0dcf0a3a/README.md): integer **94.76% [94.60–94.86]**, FP32 **96.67% [96.52–96.90]**. The gap at epoch 20 is **1.91 pp**, with no non-finite/extreme outputs recorded.

The 20-epoch user-run executable commit was `0dcf0a3af6b106565ea801b22c1ade3b499fc613`, Rust 1.95.0, Manjaro x86_64. The candidate branch later adds only experiment evidence/documentation, not a different training algorithm. The exact training-source SHA-256 hashes in both manifests match:
- train-images-idx3-ubyte: `ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db`
- train-labels-idx1-ubyte: `65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5`

The 20-epoch budget was selected from a predeclared validation-only diagnostic, not by inspecting the test. Though mean integer validation peaked at epoch 19, **we freeze 20 fixed epochs**, rather than selecting the best observed per-epoch checkpoint.

## Locked configuration

| Setting | Value |
|---|---|
| Architecture | 784 → 128 → 128 → 10 dense MLP, same architecture for integer/FP32 |
| Backends | Integer-only dyadic training and independent scalar FP32 reference |
| Training budget | **20 fixed epochs**, no early stopping / checkpoint selection |
| Batch / learning rate | **32**, actual-sample mean gradients, LR shift **7** (`2^-7`) |
| Momentum / quantization | momentum shift **1**; input quantization shift **7**, output bits **31**, grad clip mantissa **8192** |
| Model seeds | **1337, 2027, 4242** |
| Data | Canonical official MNIST IDX, 60,000 train-source images and **10,000 official test images** |
| Validation | 55,000 train / 5,000 stratified validation from the training partition; fixed split seed `81985529216486895` |
| Test evaluation | Only after epoch 20 completes, **once per backend/seed**; no test-driven tuning. In final mode the test IDX is loaded during setup, but no predictions or metrics are computed until training ends |
| Measurements | Per-epoch train/validation curves and confusion matrices; one final 10k test accuracy per backend/seed, wall time, peak RSS, source checksums, commit/toolchain/platform |

The scalar FP32 reference is intentionally unoptimized, so the runtime ratio is **not** evidence of a hardware architectural advantage or disadvantage.

## One-off local official test

Run this with the issue branch checked out, from the repository root, *once* after freezing the above settings. The `--final-test` opt-in is required in both the Python launcher and Rust executable and cannot be combined with `--validation-only`, `--ci-sample` or `--smoke`. Use a fresh output directory.

```sh
git switch issue-15-heldout-baselines
git pull --ff-only
python3 experiments/fetch_mnist.py --data-dir data/mnist

python3 experiments/run_baselines.py \
    --dataset mnist \
    --final-test \
    --epochs 20 \
    --batch-size 32 \
    --lr-shift 7 \
    --seeds 1337 2027 4242 \
    --output-dir reports/mnist-final-locked-2026-10-24
```

The only files to return are:
`reports/mnist-final-locked-2026-10-24/mnist/full/{manifest.json,summary.md,integer-seed-1337.json,fp32-seed-1337.json,integer-seed-2027.json,fp32-seed-2027.json,integer-seed-4242.json,fp32-seed-4242.json}`.

**Do not upload datasets, weights, or checkpoints.** Each of the six executions retrains the model from scratch, evaluates validation every epoch, and evaluates the official test only at the end. Do not rerun with new hyperparameters in response to test scores.

## Release status

This is a predeclared *measurement protocol*, **not** an ADR-004 numerical quality gate. Issue #15 remains open until all three-seed official held-out results are recorded and evaluated. Propose quantitative acceptance thresholds to the maintainer only after reviewing those results. Do not merge PR #9 to master as part of this issue.


## Post-run audit — recorded after the one-shot evaluation

The official test was completed locally on 2026-10-24 using candidate commit `fa405deeab3d7e456f50a407930f2a7b46673eb4`, with the same 20 epochs, LR shift 7, batch 32 and three fixed model seeds. This appendix is **post-measurement** and does not retroactively modify the pre-registered parameters above.

- **Integer:** **95.41%** mean, [95.23–95.67]% across three seeds; **1387.95 seconds** mean wall time, **434.06 MiB** whole-process peak RSS.
- **Unoptimized scalar FP32:** **96.87%** mean, [96.78–96.97]%; **341.14 seconds** mean wall time, **434.10 MiB** whole-process peak RSS.
- **Mean FP32 advantage:** **1.45 percentage points**. Reported integer/FP32 runtime ratio **4.07×** on one x86_64 Manjaro machine, without any claim of intrinsic speed advantage.
- All six original JSONs, 20-epoch training-derived validation curves, the complete original manifest with **four canonical IDX SHA-256 values**, and the original summary were archived [here](../reports/held_out/mnist/full/2026-10-24-locked-fa405dee/README.md). No new hyperparameters were selected using the official test result.
- The final test has now been **consumed** for this experiment. Any future modification motivated by the official test needs new independent data for a fresh generalization claim; comparisons on the already-seen MNIST benchmark remain useful as regression diagnostics but not pristine hold-out experiments.

Proposal, not approved policy: [ADR-004 candidate metrics](ADR_004_PROPOSAL.md). Approval and the decision on merging draft PR #34 into `big-refactor` remain with the maintainer. PR #9 remains unmerged.
