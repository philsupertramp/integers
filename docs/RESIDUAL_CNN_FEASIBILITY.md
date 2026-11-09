# Residual CNN paired feasibility (#39)

This extends the [single-block arithmetic POC](RESIDUAL_BLOCK_POC.md) with a
**fixed one-seed development-data feasibility experiment**, not ResNet-20.

## Frozen model and paired controls

Two network variants use the **same parameter initialization and compute
budget**. They only differ in whether the output of a two-convolution branch
is added to the branch's input.

| Part | Plain | Residual |
|---|---|---|
| Stem | Conv 1→4, 3×3, ReLU, 2×2 pool | same |
| Mid | Conv 4→4, 1×1, ReLU, Conv 4→4, 1×1 | identical branch, plus identity shortcut |
| Downstream | Conv 4→8, 3×3, ReLU, 2×2 pool | same |
| Head | Flatten, Linear 200→64, ReLU, Linear 64→10 | same |

No BatchNorm, convolutional padding, skip projection or extra tuning. Training
uses the same SGD/momentum, raw-logit MSE, target preparation, batch size,
initial dyadic mantissas and He-v2 preparation scaling (integer-only ×5/2)
as the relevant control in the previous depth study. FP32 decodes and copies
the initial weights only, then runs **independent scalar forward/backward/
update** routines, including its own identity skip and branch gradient.

The identity add uses exact, finer-scale dyadic alignment with explicit
overflow rejection. This proof of concept does not establish a general
saturation/STE policy. Both models use an equal total count of convolutions.

## Run from the PR branch

```sh
python3 experiments/fetch_mnist.py --data-dir data/mnist --train-only
cargo build --locked --release --example held_out
python3 experiments/residual_feasibility.py \
  --out-dir reports/residual-feasibility-01
```

Default single-seed exploratory budget: seed 1337, 2 epochs, 1024 training /
256 validation examples selected after the locked MNIST 55k/5k development
split, batch 32, learning-rate shift 7. The official 10k test files are
forbidden by the runner; no final generalization claim is made.

The runner records source SHA, release binary SHA, dataset IDX checksums,
all four raw learning curves/confusion matrices, complete stdout and
per-report SHA-256. An existing output directory is rejected, so experimental
evidence is never overwritten. The PR's CI workflow runs precisely this
bounded study and preserves its artifact.

**Stopping point:** a four-arm valid report without instability demonstrates
the residual training **plumbing**, not that residuals rescue the failing
16-conv model. If both FP32 controls do not learn under this short budget,
no integer precision conclusion is possible. A separate predeclared
multi-seed, longer development study would be required to judge depth
stability or whether residuals help at all. PR #9 stays untouched.
