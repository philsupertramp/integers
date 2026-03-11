# integers

A Rust library for training and running neural networks using integer arithmetic.

`integers` implements feed-forward and recurrent neural networks where weights, activations,
and gradients are stored and computed as fixed-point integers (`i32`) rather than floating-point
values. An `f32` backend is also provided for reference, debugging, and baseline comparison.
Both backends share the same API through a unified `Scalar` trait.

## Motivation

Floating-point arithmetic is convenient but expensive in hardware. Quantised integer networks
can be significantly faster and more energy-efficient on inference hardware that lacks an FPU,
while still achieving competitive accuracy. This library explores how far you can push
**training entirely in integer arithmetic** — not just quantised inference after float training —
using stochastic rounding to keep the process unbiased.

## Key Concepts

### Fixed-Point Representation

Values are stored as scaled integers. A value `v` at shift `s` represents the real number
`v / 2^s`. All tensors carry their shift alongside them through the forward and backward passes.
Modules are responsible for tracking `input_shift` and `output_shift` so downstream layers
can correctly interpret the values they receive.

### Stochastic Rounding (Downcast)

Whenever precision must be reduced (e.g. after a dot product that accumulates into a wider
accumulator), `stochastic_downcast` is used instead of deterministic truncation. The fractional
bits are rounded up or down with probability proportional to their magnitude. This keeps the
expected value unbiased and prevents systematic quantisation drift during training.

### Scalar Trait

The `Scalar` trait abstracts over element types. Implementing `Scalar` for a type gives access
to the full layer and training infrastructure. Two implementations are provided:

| Type  | Accumulator | Use case |
|-------|-------------|----------|
| `f32` | `f32`       | Baseline / debugging |
| `i32` | `i32`       | Integer training & inference |


**Note**: The idea for the future is using different, bigger Accumulator types like `i32` for smaller
"base" types, e.g. quantize `i32` to `i8` for inference.

---

## Architecture

```
src/
├── lib.rs                  # Tensor, Scalar, Numeric traits; XorShift64 RNG; argmax / accuracy
└── lib/
    ├── nn.rs               # Module trait, Linear, Sequential, Params, HasWeights
    ├── nn/
    │   ├── activations.rs  # ReLU, Tanh
    │   ├── kernels.rs      # dot_product, stochastic_downcast, tanh LUT, isqrt, NEON kernels
    │   ├── losses.rs       # MSE, MAE
    │   ├── optim.rs        # SGD (with momentum), Adam
    │   ├── conv.rs         # Conv2d
    │   └── rnn.rs          # RNNCell
    ├── data.rs             # Dataset struct, minibatch, shuffled_indices
    ├── dataset_loaders.rs  # DatasetBuilder (CSV / TSV / Parquet), quantization
    ├── quant.rs            # minmax_quantize, standard_score_quantize
    └── debug.rs            # OverflowStats, thread-local counters (debug builds only)

examples/
├── iris.rs                 # Iris classification (4 features → 3 classes)
├── iris_f32.rs             # [FP32 version] Iris classification (4 features → 3 classes)
├── mnist.rs                # MNIST digit classification (784 → 128 → 128 → 10)
├── mnist_f32.rs            # [FP32 version] MNIST digit classification (784 → 128 → 128 → 10)
├── mnist_to_parquet.rs     # One-shot converter: MNIST IDX binary → Parquet
└── pairity.rs              # Binary parity with RNNCell (sequence classification)
```

---

## Layers & Components

### `Linear<S>`

A fully-connected layer. Weights are stored in transposed form `[out, in]` for cache-friendly
access. Xavier uniform initialisation is used by default. After initialisation the weight
quantisation shift is inferred automatically from the magnitude of the master weights.

```rust
let mut layer = Linear::<f32>::new(784, 128);
layer.init(&mut rng);
```

**Shift tracking.** Each `Linear` layer records `input_shift` and `output_shift`. The combined
output shift after a forward pass is `weights.quant_shift + input_shift + s_x`. Backward pass
uses the cached input and shift to compute correct weight and input gradients.

### `Sequential<S>`

A container that chains modules in order, threading the shift value through each one.

```rust
let mut model = Sequential::<f32>::new();
model
    .add(Linear::<f32>::new(4, 8))
    .add(ReLU::<f32>::new())
    .add(Linear::<f32>::new(8, 3));
```

### Activations: `ReLU<S>`, `Tanh<S>`

Both cache their input for the backward pass. `Tanh` for `i32` uses a pre-computed 256-entry
lookup table generated at first use (`OnceLock`), mapping `i8` inputs to `i8` outputs scaled
to `[-127, 127]`.

### `RNNCell<S>`

An Elman-style recurrent cell: `h_t = tanh(W_ih · x_t + W_hh · h_{t-1})`.

Hidden state is accumulated across time steps via `h_prev`. Backpropagation through time
(BPTT) is implemented by storing the carry gradient `d_h_next` between backward calls. Call
`reset_state()` between sequences.

### Losses: `MSE`, `MAE`

Both return `(scalar_loss, gradient_tensor)`. Gradient for MSE is the element-wise error;
for MAE it is the sign of the error.

### Optimisers: `SGDConfig`, `AdamConfig`

Learning rates and momentum/beta coefficients are expressed as bit-shifts (`1/2^shift`) to
stay in integer arithmetic. Helper constructors accept floating-point values and convert them:

```rust
let optim = SGDConfig::new()
    .with_learn_rate(0.01)        // → lr_shift = 7
    .with_momentum(0.9);          // → momentum_shift = 3
```

Adam maintains per-weight first and second moment estimates (`m`, `v`) in `OptimizerState`.

---

## Dataset Loading

`DatasetBuilder` provides a fluent API for loading labelled tabular datasets from CSV, TSV,
or Parquet files.

```rust
let train_ds = DatasetBuilder::<f32>::new("data/iris_train.tsv")
    .format(FileFormat::TSV)
    .with_features(vec![0, 1, 2, 3])
    .with_label_column(4)
    .with_quantization(QuantizationMethod::StandardScore)
    .load()?;
```

String labels are mapped to numeric class indices automatically. Two quantisation strategies
are supported:

| Method | Description | Output shift |
|--------|-------------|-------------|
| `MinMax` | Scales each feature column to `[-127, 127]` | 7 |
| `StandardScore` | Z-score normalised, then scaled by 32 | 5 |

The resulting `Dataset<S>` exposes `get_input(i)`, `get_target(i)`, and `minibatch(&indices)`
for efficient training loops.

---

## Quickstart

### Prerequisites

- Rust 1.85+ (edition 2024)
- For MNIST: download the raw IDX files and convert them to Parquet once with
  `cargo run --example mnist_to_parquet -- data/mnist/ data/`

### Run the Iris example

```bash
cargo run --example iris --release
```

### Run MNIST

```bash
cargo run --example mnist --release
```

### Run the parity RNN

```bash
cargo run --example pairity --release
```

### Run tests

```bash
cargo test
```

---

## Diagnostics (Debug Builds)

Building in debug mode enables thread-local overflow counters via `debug::OverflowStats`:

| Counter | Tracks |
|---------|--------|
| `forward_wraps` | Wrapping additions in forward loops |
| `backward_wraps` | Wrapping subtractions in weight updates |
| `downcast_clamps` | Stochastic downcast saturation events |

Call `reset_overflow_stats()` / `get_overflow_stats()` around epochs to monitor numerical
health during development.

---

## Recommended Hyperparameters

These have been empirically stable across the included examples.

| Setting | Conservative | Aggressive |
|---------|-------------|------------|
| `lr_shift` | 8 | 6–7 |
| `batch_size` | 16 | 32 |
| Optimiser | `SGDConfig` with momentum | `AdamConfig` |

For networks deeper than two hidden layers, increase `lr_shift` by 1–2 to compensate for
gradient cascade.

---

## CI/CD

A `post-receive` Git hook in `.githooks/` automates deployment and testing on push to `master`:
it checks out the working tree to the target directory and runs `cargo test`. Install it on
your remote with:

```bash
cp .githooks/post-receive <remote_repo>/hooks/post-receive
chmod +x <remote_repo>/hooks/post-receive
```

---

## Dependencies

| Crate | Purpose |
|-------|---------|
| `arrow 58` | Columnar memory format for Parquet I/O |
| `parquet 58` | Reading Parquet dataset files |

No external ML framework dependencies. The rest of the library is `std`-only.

---

## Contributing

Contributions are welcome. Key areas for future work:

- **Convolutional layers** (`Conv2d`)
- **Batch normalisation** in integer arithmetic
- **SIMD kernels** — an ARM NEON dot-product stub exists in `kernels::arm_neon`, we should extend our set of kernels further.
- **Serialisation** of trained model weights
- **Caching** for the Parquet loader (noted in source as a known bottleneck for "large" datasets; MNIST is already quite slow)
