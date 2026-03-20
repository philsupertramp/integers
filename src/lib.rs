//! `dyadic_nn` — Integer neural network training on dyadic rational arithmetic.
//!
//! # Module map
//!
//! | Module               | Contents |
//! |----------------------|----------|
//! | [`dyadic`]           | `Dyadic` type and all four arithmetic operations |
//! | [`nn`]               | `Layer` trait, `Sequential`, `Linear`, `ReLU`, `Softmax` |
//! | [`tensor`]           | Generic `Tensor<S>` and the `Scalar` trait |
//! | [`quant`]            | Per-column quantisation strategies |
//! | [`rng`]              | `XorShift64` — deterministic integer PRNG for shuffling |
//! | [`data`]             | `Dataset<S>` struct and `shuffled_indices` |
//! | [`dataset_loaders`]  | Fluent `DatasetBuilder` for CSV / TSV / Parquet |
//! | [`mnist_loader`]     | IDX binary format loader for MNIST |
//!
//! # Quick-start
//!
//! ```no_run
//! use dyadic_nn::nn::{Sequential, Linear, ReLU, Softmax};
//! use dyadic_nn::dataset_loaders::{DatasetBuilder, QuantizationMethod};
//! use dyadic_nn::{sample_to_dyadic, target_to_dyadic, cross_entropy_grad, argmax};
//! use dyadic_nn::data::shuffled_indices;
//! use dyadic_nn::rng::XorShift64;
//!
//! const S: u32 = 7;
//!
//! let ds = DatasetBuilder::<i32>::new_csv("iris.csv")
//!     .with_features(vec![0,1,2,3]).with_label_column(4)
//!     .with_quantization(QuantizationMethod::MinMax)
//!     .load().unwrap();
//!
//! let shift = ds.input_shift.max(0) as u32;
//!
//! let mut model = Sequential::new();
//! model.add(Linear::new(4, 8, S, S, 32).with_grad_clip(8192).with_momentum(1));
//! model.add(ReLU::new());
//! model.add(Linear::new(8, 3, S, S, 32).with_grad_clip(8192).with_momentum(1));
//! model.add(Softmax::new(S));
//!
//! let mut rng = XorShift64::new(42);
//! for _epoch in 0..200 {
//!     for &i in &shuffled_indices(ds.len(), &mut rng) {
//!         let x = sample_to_dyadic(&ds.get_input(i).data,  shift);
//!         let t = target_to_dyadic(&ds.get_target(i).data, shift);
//!         model.zero_grad();
//!         let y = model.forward(&x);
//!         let g = cross_entropy_grad(&y, &t, shift);
//!         model.backward(&g);
//!         model.update(7);
//!     }
//! }
//! ```
pub mod nn;
pub mod data;

#[path = "lib/dyadic.rs"]
pub mod dyadic;
#[path = "lib/tensor.rs"]
pub mod tensor;
#[path = "lib/quant.rs"]
pub mod quant;
#[path = "lib/rng.rs"]
pub mod rng;

// ─── Re-exports ───────────────────────────────────────────────────────────────

pub use dyadic::Dyadic;
pub use tensor::{Tensor, Scalar};
pub use rng::XorShift64;

// ─── Dataset → network bridge ─────────────────────────────────────────────────

/// Convert a flat `&[i32]` row from a `Dataset<i32>` into `Vec<Dyadic>`.
///
/// `shift = dataset.input_shift.max(0) as u32`
pub fn sample_to_dyadic(values: &[i32], shift: u32) -> Vec<Dyadic> {
    values.iter().map(|&v| Dyadic::new(v, shift)).collect()
}

/// Same as [`sample_to_dyadic`] but for one-hot target rows.
pub fn target_to_dyadic(values: &[i32], shift: u32) -> Vec<Dyadic> {
    values.iter().map(|&v| Dyadic::new(v, shift)).collect()
}

// ─── Loss gradients ───────────────────────────────────────────────────────────

/// MSE gradient: `∂L/∂yᵢ = yᵢ − tᵢ`.
///
/// Both slices must share the same dyadic scale.  The result is passed
/// directly to [`nn::Sequential::backward`].
pub fn mse_grad(output: &[Dyadic], target: &[Dyadic]) -> Vec<Dyadic> {
    output.iter().zip(target.iter())
        .map(|(&y, &t)| {
            debug_assert_eq!(y.s, t.s, "output and target must share scale for MSE grad");
            Dyadic::new(y.v.saturating_sub(t.v), y.s)
        })
        .collect()
}

/// Cross-entropy gradient for use after a [`nn::Softmax`] output layer.
///
/// When the network ends with Softmax + CE loss, the combined gradient of
/// `CE(softmax(z), t)` with respect to the pre-softmax logits `z` simplifies
/// to `p − t` (softmax probabilities minus one-hot targets).
///
/// Since [`nn::Softmax::backward`] is a straight-through estimator (identity),
/// you can pass this gradient directly into [`nn::Sequential::backward`] and
/// it will correctly flow back through Softmax without needing the full
/// softmax jacobian.
///
/// `output` — the Softmax layer output (quantised probabilities).  
/// `target` — one-hot target at the same shift.  
/// `shift`  — the dyadic scale of both tensors.
///
/// # Example
/// ```no_run
/// # use dyadic_nn::{Dyadic, cross_entropy_grad};
/// # let y: Vec<Dyadic> = vec![];
/// # let t: Vec<Dyadic> = vec![];
/// let g = cross_entropy_grad(&y, &t, 7);
/// ```
pub fn cross_entropy_grad(output: &[Dyadic], target: &[Dyadic], shift: u32) -> Vec<Dyadic> {
    // p − t: both at `shift`, result at `shift`.
    output.iter().zip(target.iter())
        .map(|(&p, &t)| {
            Dyadic::new(p.v.saturating_sub(t.v), shift)
        })
        .collect()
}

/// Argmax over a `&[Dyadic]` — returns the index of the largest mantissa.
///
/// Since all elements share the same scale `s`, comparing mantissas is
/// equivalent to comparing decoded values.
pub fn argmax(output: &[Dyadic]) -> usize {
    output.iter()
        .enumerate()
        .max_by_key(|(_, d)| d.v)
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ─── Training reporter ────────────────────────────────────────────────────────

/// A lightweight per-epoch training reporter.
///
/// Accumulates loss and accuracy statistics during a training epoch, then
/// formats them into a consistent table row.
///
/// # Usage
/// ```no_run
/// # use dyadic_nn::{TrainingReporter};
/// let mut reporter = TrainingReporter::new(500, 50);   // 500 epochs, log every 50
/// reporter.print_header();
///
/// for epoch in 0..500 {
///     reporter.reset();
///     // ... training loop ...
///     // reporter.record(sq_err_sum, n_correct, n_samples, test_acc_opt);
///     reporter.maybe_print(epoch, Some(test_accuracy));
/// }
/// reporter.print_footer(final_train_acc, final_test_acc);
/// ```
pub struct TrainingReporter {
    pub total_epochs:  u32,
    pub log_every:     u32,
    pub shift:         u32,   // for decoding squared-error to real MSE

    // per-epoch accumulators
    sq_err_sum: i64,
    n_correct:  usize,
    n_samples:  usize,

    pub best_train_acc: f64,
    pub best_test_acc:  f64,
}

impl TrainingReporter {
    /// `shift` should match `dataset.input_shift` (used to decode squared
    /// integer error back to floating-point MSE).
    pub fn new(total_epochs: u32, log_every: u32, shift: u32) -> Self {
        Self {
            total_epochs, log_every, shift,
            sq_err_sum: 0, n_correct: 0, n_samples: 0,
            best_train_acc: 0.0, best_test_acc: 0.0,
        }
    }

    /// Print the column header.  Call once before the epoch loop.
    pub fn print_header(&self) {
        println!("{:>7}  {:>14}  {:>10}  {:>12}",
            "Epoch", "Train Loss", "Train Acc", "Test Acc");
        println!("{}", "─".repeat(50));
    }

    /// Reset accumulators at the start of each epoch.
    pub fn reset(&mut self) {
        self.sq_err_sum = 0;
        self.n_correct  = 0;
        self.n_samples  = 0;
    }

    /// Record a single sample's contribution to epoch statistics.
    ///
    /// `sq_err` — sum of squared gradient mantissas (`g.v^2`) for this sample.  
    /// `correct` — whether the predicted class matched the true label.
    pub fn record(&mut self, sq_err: i64, correct: bool) {
        self.sq_err_sum += sq_err;
        if correct { self.n_correct += 1; }
        self.n_samples  += 1;
    }

    /// Compute the current epoch's training accuracy (0.0–100.0).
    pub fn train_accuracy(&self) -> f64 {
        if self.n_samples == 0 { return 0.0; }
        self.n_correct as f64 / self.n_samples as f64 * 100.0
    }

    /// Compute the current epoch's mean squared error (decoded to f64).
    pub fn train_mse(&self) -> f64 {
        if self.n_samples == 0 { return 0.0; }
        (self.sq_err_sum as f64 / self.n_samples as f64)
            * 2f64.powi(-2 * self.shift as i32)
    }

    /// Print a log row if this epoch should be logged.
    ///
    /// `test_acc` — optional test-set accuracy (0.0–100.0).  Pass `None` if
    ///              evaluation is skipped this epoch.
    pub fn maybe_print(&mut self, epoch: u32, test_acc: Option<f64>) {
        let train_acc = self.train_accuracy();
        if train_acc > self.best_train_acc { self.best_train_acc = train_acc; }
        if let Some(ta) = test_acc {
            if ta > self.best_test_acc { self.best_test_acc = ta; }
        }

        let is_last = epoch + 1 == self.total_epochs;
        if epoch % self.log_every == 0 || is_last {
            let test_str = test_acc
                .map(|a| format!("{:>11.2}%", a))
                .unwrap_or_else(|| "          —".to_string());
            println!("{:>7}  {:>14.6}  {:>9.2}%  {}",
                epoch, self.train_mse(), train_acc, test_str);
        }
    }

    /// Print a summary footer after training.
    pub fn print_footer(&self) {
        println!("{}", "─".repeat(50));
        println!("Best train accuracy : {:.2}%", self.best_train_acc);
        println!("Best test  accuracy : {:.2}%", self.best_test_acc);
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mse_grad_is_difference() {
        let y = vec![Dyadic::new(100, 7), Dyadic::new(-50, 7)];
        let t = vec![Dyadic::new(127, 7), Dyadic::new(  0, 7)];
        let g = mse_grad(&y, &t);
        assert_eq!(g[0].v, 100 - 127);
        assert_eq!(g[1].v, -50 -   0);
    }

    #[test]
    fn cross_entropy_grad_is_prob_minus_target() {
        // Predicted prob for class 0 ≈ 0.9 at shift 7 → mantissa ≈ 115
        let y = vec![Dyadic::new(115, 7), Dyadic::new(10, 7), Dyadic::new(3, 7)];
        let t = vec![Dyadic::new(127, 7), Dyadic::new( 0, 7), Dyadic::new(0, 7)];
        let g = cross_entropy_grad(&y, &t, 7);
        // Gradient for correct class should be negative (prob < target hot bit)
        assert!(g[0].v < 0);
        // Gradient for incorrect classes should be positive
        assert!(g[1].v > 0);
        assert!(g[2].v > 0);
    }

    #[test]
    fn argmax_returns_max_index() {
        let v = vec![
            Dyadic::new(-10, 7),
            Dyadic::new( 50, 7),
            Dyadic::new( 20, 7),
        ];
        assert_eq!(argmax(&v), 1);
    }

    #[test]
    fn reporter_accumulates_correctly() {
        let mut r = TrainingReporter::new(10, 2, 7);
        r.reset();
        r.record(1000, true);
        r.record(2000, false);
        assert_eq!(r.n_correct, 1);
        assert_eq!(r.n_samples, 2);
        assert!((r.train_accuracy() - 50.0).abs() < 1e-6);
    }
}
