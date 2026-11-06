# Iris hyperparameter optimization — validation only

**Evidence:** [GitHub Actions run 40090654565](https://github.com/philsupertramp/integers/actions/runs/40090654565), commit `cbe8f06c61236347de69dc617644b85b56fce9b7`, Linux x86_64, Rust 1.99.1. The `iris-validation-only-hpo` artifact contains 180 complete epoch-by-epoch learning-curve JSON records plus `selection.json` and `summary.md`, preserved for 30 days. This document preserves the full ranked grid beyond artifact expiry.

## Winner selected *only* on integer validation scores

**Batch size 32, learning-rate shift 6 (`2^-6`), fixed 200 epochs.**

- Integer: **98.61%** mean validation accuracy (range **87.50–100.00%**).
- Scalar FP32: **99.07%** paired mean validation accuracy (range **91.67–100.00%**).

The two backends use the same `4→8→8→3` MLP and independently implemented forward/backward/optimizer math, with the corrected actual-batch-mean gradient convention, fixed dyadic Q7 quantization, and momentum decay `2^-1`.

The selection criterion, fixed *before running the search*, was highest integer **mean validation accuracy**, then higher minimum integer validation accuracy, fewer epochs, smaller batch size, and higher learning-rate shift. FP32 results were collected for comparison only; **FP32 scores did not determine the winner**.

## Protocol and strict test isolation

The only data read were the 130 existing `data/iris_train.tsv` development rows, SHA-256 `8e8b26241ece0cfc5e942015db617dbcc9e3064bc2d5bc22b07c49369b0c00d3`. For each of **three fixed stratified split seeds**, 106 rows trained and 24 held out for validation (8 from each class). The preprocessing min/max were learned separately on the training subset for each split and applied to validation unchanged. Each split was trained with **three independent model seeds** (`1337`, `2027`, `4242`). For each pair of split and model seed, integer and FP32 models shared initialization and deterministic example ordering.

The predeclared grid comprised learning-rate shifts **4–8**, batch sizes **8 and 32**, and fixed evaluation checkpoints at epochs **100, 200, 400**. Thus 30 candidate configurations × 3 validation splits × 3 model seeds × 2 backends = **540 validation scores**. To avoid redundant training, each of the 180 actual training runs was trained once for 400 epochs and scored at the three prespecified checkpoints.

**The historically reused `data/iris_test.tsv` file was never opened, hashed, or evaluated by the HPO runner.** The Rust `--validation-only` mode also skips loading it and records `test_size=0`, `test_evaluated=false` and JSON-null test metrics. The routine Iris CI protocol likewise removes the test TSV from its checkout before running validation-only checks.

## Full ranked grid

| Rank | Batch | LR shift | Epochs | Integer mean [min, max] | FP32 mean [min, max] |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 6 | 200 | 98.61% [87.50, 100.00] | 99.07% [91.67, 100.00] |
| 2 | 8 | 8 | 200 | 98.15% [83.33, 100.00] | 98.15% [83.33, 100.00] |
| 3 | 32 | 7 | 400 | 97.69% [91.67, 100.00] | 98.61% [91.67, 100.00] |
| 4 | 8 | 7 | 200 | 97.69% [95.83, 100.00] | 96.76% [91.67, 100.00] |
| 5 | 8 | 8 | 400 | 97.69% [95.83, 100.00] | 96.76% [91.67, 100.00] |
| 6 | 8 | 7 | 100 | 97.22% [87.50, 100.00] | 98.15% [83.33, 100.00] |
| 7 | 32 | 4 | 100 | 97.22% [95.83, 100.00] | 97.69% [95.83, 100.00] |
| 8 | 8 | 6 | 200 | 97.22% [95.83, 100.00] | 97.22% [95.83, 100.00] |
| 9 | 8 | 4 | 200 | 97.22% [95.83, 100.00] | 97.22% [95.83, 100.00] |
| 10 | 32 | 5 | 200 | 97.22% [95.83, 100.00] | 96.76% [91.67, 100.00] |
| 11 | 32 | 4 | 200 | 97.22% [95.83, 100.00] | 97.22% [95.83, 100.00] |
| 12 | 8 | 5 | 400 | 97.22% [95.83, 100.00] | 97.22% [95.83, 100.00] |
| 13 | 32 | 5 | 400 | 97.22% [95.83, 100.00] | 97.22% [95.83, 100.00] |
| 14 | 8 | 6 | 100 | 96.76% [91.67, 100.00] | 97.69% [95.83, 100.00] |
| 15 | 8 | 5 | 200 | 96.76% [91.67, 100.00] | 97.22% [95.83, 100.00] |
| 16 | 8 | 7 | 400 | 96.76% [91.67, 100.00] | 96.76% [91.67, 100.00] |
| 17 | 8 | 6 | 400 | 96.76% [91.67, 100.00] | 97.22% [95.83, 100.00] |
| 18 | 8 | 4 | 400 | 96.76% [91.67, 100.00] | 97.22% [95.83, 100.00] |
| 19 | 32 | 6 | 400 | 96.76% [91.67, 100.00] | 96.76% [91.67, 100.00] |
| 20 | 32 | 4 | 400 | 96.76% [91.67, 100.00] | 97.22% [95.83, 100.00] |
| 21 | 8 | 5 | 100 | 96.30% [91.67, 100.00] | 96.30% [91.67, 100.00] |
| 22 | 8 | 4 | 100 | 95.83% [87.50, 100.00] | 94.91% [87.50, 100.00] |
| 23 | 32 | 5 | 100 | 95.37% [83.33, 100.00] | 98.61% [91.67, 100.00] |
| 24 | 32 | 7 | 200 | 86.57% [70.83, 100.00] | 86.57% [66.67, 100.00] |
| 25 | 32 | 6 | 100 | 85.19% [66.67, 100.00] | 87.04% [66.67, 100.00] |
| 26 | 32 | 8 | 400 | 85.19% [66.67, 100.00] | 86.57% [66.67, 100.00] |
| 27 | 8 | 8 | 100 | 83.33% [66.67, 100.00] | 84.26% [66.67, 100.00] |
| 28 | 32 | 7 | 100 | 78.70% [66.67, 100.00] | 78.70% [66.67, 100.00] |
| 29 | 32 | 8 | 200 | 76.39% [66.67, 100.00] | 76.39% [66.67, 100.00] |
| 30 | 32 | 8 | 100 | 75.46% [66.67, 95.83] | 72.69% [62.50, 95.83] |

## Repeat

```sh
cargo build --locked --release --example held_out
python3 experiments/iris_hpo.py \
    --lr-shifts 4 5 6 7 8 --batch-sizes 8 32 \
    --checkpoints 100 200 400 --seeds 1337 2027 4242
```

The validation-selected default **for subsequent Iris experiments** is now batch=32, `lr_shift=6`, epochs=200. HPO did not itself evaluate the test data. `experiments/run_baselines.py --dataset iris --validation-only` provides the same explicit no-test execution route.

## Limitations and outstanding decision

These are **model-selection validation scores**, not an external test result or a new ADR-004 acceptance threshold. The three stratified splits draw from the same tiny 130-example pool and can overlap; the nine split/seed scores per backend must not be treated as nine independent datasets. Searching 30 configurations can overfit these validation samples, and a 0.5–1 percentage point difference between candidates may not be meaningful. The 20-row historical Iris test subset was inspected in prior project experiments; even a future one-shot run cannot make it a pristine independent test set. Any final comparison must fix its settings in advance, avoid using its test score for further tuning, and clearly disclose that limitation.
