# integers

Experimental Rust library researching neural-network training with **integer-stored dyadic rationals**, rather than only quantized inference.

**Candidate branch:** `big-refactor` (described here) is conditionally selected under [ADR-001](docs/ARCHITECTURE_DECISIONS.md). The default `master` still uses legacy `Scalar`/`Numeric` APIs; those types, `RNNCell` and old `src/lib/nn/` paths **do not describe this candidate**. [PR #9](https://github.com/philsupertramp/integers/pull/9) is still open and **not approved for merge**.

## Dyadic arithmetic

`Dyadic { v: i32, s: u32 }` represents `v × 2^(-s)`. For example `Dyadic::new(3, 2)` decodes to 0.75, equivalent to `Dyadic::new(6, 3)`. Larger `s` gives **finer** precision. The `dyadic` module provides `add`, `sub`, `mul`, `div`, `align`, `requantize`, clipping, stochastic rounding and STE helpers; lossy conversions may round or saturate. `Tensor` stores row-major `Vec<Dyadic>` and shape metadata; `view()` borrows a `TensorView`. `to_f64()` is for references/inspection.

## Public API: one training step

The candidate uses `nn::Module`, `Sequential` and concrete layers; `Linear::new(in, out, weight_shift, quant_shift, output_bits)`. This code demonstrates a **single step**, not verified convergence:

```rust
use integers::{Dyadic, Tensor, mse_grad};
use integers::nn::{Linear, ReLU, Sequential};

let s = 7u32;
let mut model = Sequential::new();
model.add(Linear::new(2, 4, s, s, 16).with_grad_clip(8192));
model.add(ReLU::new());
model.add(Linear::new(4, 1, s, s, 16));

let x = Tensor::from_vec(
    vec![Dyadic::new(64, s), Dyadic::new(-32, s)], vec![1, 2]
);
let target = Tensor::from_vec(vec![Dyadic::new(64, s)], vec![1, 1]);

model.zero_grad();
let y = model.forward(&x.view());
let g = Tensor::from_vec(mse_grad(&y.data, &target.data), y.shape.clone());
model.backward(&g.view());
model.update(6); // integer shift for learning rate
```

For mini-batches use `forward_batch` / `backward_batch`. `set_training(bool)` toggles supported stochastic/batch-dependent layers. Other candidate components are `ReLU`, `Softmax`, `Conv2D`, `MaxPool2D`, `Flatten`, `Dropout`, `BatchNorm1D` and `BatchNorm2D`; not all have verified gradients or integer-only implementations.

## Repository layout and examples

| Location | Contents |
| --- | --- |
| `src/lib/dyadic.rs` | `Dyadic`, `Tensor`, views, arithmetic, rounding, quantization helpers |
| `src/nn/` | `Module`, `Sequential`, Linear, Conv2D, activations, pooling, dropout, BatchNorm |
| `src/data/` | `Dataset`, shuffle/minibatches; `data::dataset_loaders::DatasetBuilder` |
| `src/lib/quant.rs` and `src/lib/rng.rs` | Preprocessing quantizers and seeded XorShift64 |
| `tests/` | Independent dyadic regressions and public API smoke test |
| `examples/xor.rs`, `examples/iris.rs` | XOR (built-in inputs) and Iris tabular experiment |
| `examples/mnist/main.rs`, `examples/cifar/main.rs` | Dataset-dependent convolutional experiments |
| `benches/` | Candidate benchmark experiments |

The dataset loader is **non-generic**, exported as `integers::data::dataset_loaders::{DatasetBuilder, QuantizationMethod, FileFormat}`. For an existing TSV:

```rust
use integers::data::dataset_loaders::{DatasetBuilder, FileFormat, QuantizationMethod};

let ds = DatasetBuilder::new_csv("data/iris_train.tsv")
    .format(FileFormat::TSV)
    .with_features(vec![0, 1, 2, 3])
    .with_label_column(4)
    .with_quantization(QuantizationMethod::MinMax)
    .load()?;
# Ok::<(), integers::data::DataError>(())
```

## Build and validation

Requires a Rust toolchain supporting **edition 2024** (minimum supported version not yet determined). Baseline commands:

```sh
cargo test --locked --lib
cargo test --locked --test dyadic_regression
cargo test --locked --test public_api_smoke
cargo test --locked --doc
cargo check --locked --all-targets
cargo run --example xor
cargo run --release --example iris
# For custom Iris input, supply separate no-header TSV train and test splits:
cargo run --release --example iris -- data/iris_train.tsv data/iris_test.tsv
```

The correctness and feature workflows validate default tests/doctests and all-target compilation under separate configurations. The dedicated `example-runtime.yml` workflow **executes** bundled XOR and Iris training using the checked-in Iris train/test TSV files, and checks that Iris rejects a train-only dataset path. Passing these smoke runs shows that the examples execute, **not** that model convergence, accuracy, or generalization has been independently validated. CI does **not** download MNIST/CIFAR datasets or execute those full training programs.

### Cargo features

| Feature | Description |
| --- | --- |
| Default / `--no-default-features` | Core dyadic library, plain-text data loaders |
| `parquet-support` | Arrow/Parquet support; enables `mnist_to_parquet` example |
| `image-decode` | PNG/JPEG support; needed with Parquet for the CIFAR example |
| `bench` | Optional Criterion benchmark (`tensor_bench`) |

```sh
cargo check --locked --all-targets --no-default-features
cargo check --locked --all-targets --features parquet-support
cargo check --locked --all-targets --features image-decode
cargo check --locked --all-targets --features bench
cargo check --locked --all-targets --all-features
# CIFAR requires both:
cargo run --release --example cifar --features parquet-support,image-decode
```

## Limitations and architectural boundary

**Integer-only training** ([ADR-002](docs/ARCHITECTURE_DECISIONS.md)): `Sequential::new()` enforces a fail-closed integer-only layer allowlist during forward, backward and parameter updates. `Linear`, `Conv2D`, `ReLU`, `MaxPool2D`, `Flatten` and `Dropout` explicitly declare compliant runtime arithmetic. Initialization/preprocessing, diagnostic output and FP reference tests may use floats. The known floating-point runtime paths in `Softmax` and `BatchNorm1D/2D` are excluded by default; opt in with `Sequential::new_experimental_float()` **only for historical float experiments**, not integer-only training. Custom modules require explicit source review/declaration. See [the audited boundary](docs/INTEGER_TRAINING_BOUNDARY.md) and [#26](https://github.com/philsupertramp/integers/issues/26). This is not a complete mathematical correctness or binary-level absence-of-FP proof.

PR #24 fixed major scale and integration regressions; additional numerical checks are tracked under [#12](https://github.com/philsupertramp/integers/issues/12) and layer/gradient oracle work under [#14](https://github.com/philsupertramp/integers/issues/14). Example execution is covered by [#13](https://github.com/philsupertramp/integers/issues/13); reproducible, independently held-out model comparison remains [#15](https://github.com/philsupertramp/integers/issues/15). The default MNIST CNN demo samples only 256 training and 20 evaluation images per epoch; this is **not** a full held-out benchmark. No verified deployment/benchmark evidence on Raspberry Pi 5 AArch64 yet.

Follow [AGENTS.md](AGENTS.md), [CONTRIBUTING.md](CONTRIBUTING.md) and [project status](docs/PROJECT_STATUS.md) for scoped contributions; do not merge PR #9 into `master` without separate maintainer approval.
