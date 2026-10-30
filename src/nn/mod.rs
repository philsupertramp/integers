//! Neural network layers built on the [`crate::dyadic`] arithmetic system.
//!
//! # Module catalogue
//!
//! | Module          | Parameters      | Use for                        |
//! |-----------------|-----------------|--------------------------------|
//! | [`Linear`]      | weights, biases | fully-connected                |
//! | [`ReLU`]        | —               | activation                     |
//! | [`Dropout`]     | —               | regularisation                 |
//! | [`Softmax`]     | —               | output probabilities           |
//! | [`BatchNorm1D`] | γ, β            | normalise after Linear         |
//! | [`BatchNorm2D`] | γ, β            | normalise after Conv2D         |
//! | [`Conv2D`]      | kernels, biases | spatial feature extraction     |
//! | [`MaxPool2D`]   | —               | spatial downsampling           |
//! | [`Flatten`]     | —               | bridge conv → linear           |
//!
//! # Integer-training arithmetic boundary (ADR-002)
//!
//! `Sequential::new()` fails closed: only built-in or custom modules explicitly
//! declaring `TrainingArithmetic::IntegerOnly` may run its forward/backward
//! and update paths. Softmax and BatchNorm1D/2D are excluded because they
//! perform known FP64 runtime calculations. To reproduce those historical
//! experiments, use `Sequential::new_experimental_float()` and **do not**
//! describe that mode as integer-only. Direct calls to FP layers are likewise
//! outside the enforced integer backend.
//!
//! # Implementing a new layer
//!
//! 1. Implement the [`Module`] trait (see below).
//! 2. `model.add(your_layer)` into a [`Sequential`].
//!
//! The trait has two forward/backward APIs:
//! - **Per-sample** (`forward` / `backward`) — used for inference and layers
//!   that are stateless per-call.
//! - **Batch** (`forward_batch` / `backward_batch`) — used during training.
//!   Default: loop over samples calling the per-sample methods.  Override for
//!   layers that need all N samples at once (BatchNorm, or any layer with a
//!   per-sample cache that the backward depends on).
//!
//! # Why the batch API exists
//!
//! BatchNorm must compute μ and σ² over the whole mini-batch before it can
//! normalise any sample.  This forces a separation between "all forwards" and
//! "all backwards" in the training loop.  Modules that cache per-sample state
//! (Linear, ReLU, Conv2D, MaxPool2D, Dropout) must therefore store **all N
//! caches** during `forward_batch` so that `backward_batch` can retrieve the
//! correct cache for each sample.
//!
//! # Shift convention
//!
//! All built-in layers use a uniform `SHIFT` so that every multiply keeps the
//! output scale equal to the input scale:
//! `mul(w, x, SHIFT)` with `w.s = x.s = SHIFT` → `result.s = SHIFT`.
//!
//! BatchNorm re-quantises its output back to `SHIFT` after normalisation,
//! explicitly resetting any scale drift that accumulated across layers.

mod linear;
pub use linear::Linear;

mod activations;
pub use activations::{ReLU, Softmax};

mod regularization;
pub use regularization::{BatchNorm1D, BatchNorm2D, Dropout, MaxPool2D};

mod conv;
pub use conv::Conv2D;

use crate::dyadic::{Dyadic, Tensor, TensorView, add, stochastic_round};
use std::time::Instant;

/// An opt-in, read-only raw-dyadic snapshot used only for research reporting.
/// A snapshot must not modify weights, gradients, training state or RNG.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerNumericSnapshot {
    pub parameters: Vec<Dyadic>,
    pub gradients: Vec<Dyadic>,
    pub forward_outputs: usize,
    pub forward_zeros: usize,
    pub forward_saturated: usize,
}

// ─── Module trait ──────────────────────────────────────────────────────────────

/// Declares whether a layer may execute in the ADR-002 integer training backend.
///
/// This is an explicit code-review contract, not an automatic proof.
/// Third-party modules default to Unverified until explicitly audited.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrainingArithmetic {
    IntegerOnly,
    FloatingPoint,
    Unverified,
}

/// The extension point for all layer types.
pub trait Module {
    /// Default fail-closed: custom layers are not implicitly integer-safe.
    fn training_arithmetic(&self) -> TrainingArithmetic {
        TrainingArithmetic::Unverified
    }
    // ── Per-sample API (inference + default batch impl) ───────────────────────

    /// Forward pass for a single sample.  Must cache everything `backward` needs.
    fn forward(&mut self, input: &TensorView) -> Tensor;

    /// Backward pass for a single sample.  *Accumulates* `∂L/∂params`.
    fn backward(&mut self, grad_output: &TensorView) -> Tensor;

    // ── Batch API (training) ──────────────────────────────────────────────────

    /// Forward pass for a mini-batch.
    ///
    /// Default: calls `forward` once per sample.  Override when the layer
    /// needs to see all N inputs simultaneously (BatchNorm) or needs to store
    /// all N caches so that `backward_batch` can use them (Linear, ReLU, …).
    fn forward_batch(&mut self, inputs: &Tensor) -> Tensor {
        inputs.iter().map(|x| self.forward(&x)).collect()
    }

    /// Backward pass for a mini-batch.
    ///
    /// Default: calls `backward` once per grad.  Override whenever
    /// `forward_batch` is overridden.
    fn backward_batch(&mut self, grads: &Tensor) -> Tensor {
        grads.iter().map(|g| self.backward(&g)).collect()
    }

    /// Training-only backward when the caller does not need input gradients.
    /// The default still executes the complete backward pass to preserve
    /// gradient accumulation and stochastic-rounding RNG state.
    ///
    /// A specialized implementation must produce identical parameter
    /// gradients, updates, errors, and subsequent RNG draws. Never use this
    /// method if the layer's input gradients are required.
    fn backward_batch_discard_input_grad(&mut self, grads: &Tensor) {
        let _ = self.backward_batch(grads);
    }

    // ── Parameter management ──────────────────────────────────────────────────

    /// One optimiser step: `params -= lr * velocity`.
    fn update(&mut self, lr_shift: u32);

    /// Built-in trainable integer modules average summed per-sample gradients
    /// over the actual number of accumulated samples BEFORE momentum and SGD.
    /// Works for a full batch, a short last batch, and repeated backward calls.
    /// Reset accumulated gradients. Does **not** reset velocity.
    fn zero_grad(&mut self);

    /// Diagnostic snapshots are disabled unless an experiment explicitly calls
    /// this. New modules are never implicitly included in the report.
    fn numeric_snapshot(&self) -> Option<LayerNumericSnapshot> { None }

    // ── Metadata ─────────────────────────────────────────────────────────────

    fn name(&self) -> &'static str;
    fn describe(&self) -> String {
        self.name().to_string()
    }

    /// Toggle training / eval mode.  Only meaningful for stochastic layers
    /// (Dropout, BatchNorm).  Default: no-op.
    fn set_training(&mut self, _training: bool) {}
}

// ─── Sequential ───────────────────────────────────────────────────────────────

pub struct Sequential {
    /// Public for legacy compatibility; entrypoints revalidate to prevent bypass.
    pub layers: Vec<Box<dyn Module>>,
    allow_experimental_float: bool,
}

impl Sequential {
    /// Default backend: require integer-only forward/backward/update.
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            allow_experimental_float: false,
        }
    }

    /// Explicit opt-in to the historical floating-point experimental path.
    /// This is NOT the ADR-002 integer backend. Softmax and BatchNorm still
    /// perform FP64 runtime operations here.
    pub fn new_experimental_float() -> Self {
        Self {
            layers: Vec::new(),
            allow_experimental_float: true,
        }
    }

    fn check_integer_backend(&self) {
        if self.allow_experimental_float {
            return;
        }
        for layer in &self.layers {
            assert!(
                layer.training_arithmetic() == TrainingArithmetic::IntegerOnly,
                "Sequential integer backend rejects '{}' ({:?}); use the experimental float mode if intended",
                layer.name(),
                layer.training_arithmetic(),
            );
        }
    }

    pub fn add<L: Module + 'static>(&mut self, layer: L) {
        if !self.allow_experimental_float {
            assert!(
                layer.training_arithmetic() == TrainingArithmetic::IntegerOnly,
                "Sequential integer backend rejects '{}' ({:?}); use the experimental float mode if intended",
                layer.name(),
                layer.training_arithmetic(),
            );
        }
        self.layers.push(Box::new(layer));
    }

    /// Per-sample forward — use for inference.
    pub fn forward(&mut self, input: &TensorView) -> Tensor {
        self.check_integer_backend();
        let mut x = input.to_tensor();
        for layer in &mut self.layers {
            x = layer.forward(&x.view());
        }
        x
    }

    /// Per-sample backward.
    pub fn backward(&mut self, grad: &TensorView) -> Tensor {
        self.check_integer_backend();
        let mut g = grad.to_tensor();
        for layer in self.layers.iter_mut().rev() {
            g = layer.backward(&g.view());
        }
        g
    }

    /// Batch forward — use during training (required for BatchNorm correctness).
    pub fn forward_batch(&mut self, inputs: &Tensor) -> Tensor {
        self.check_integer_backend();
        let mut x: Tensor = inputs.clone();
        for layer in &mut self.layers {
            x = layer.forward_batch(&x);
        }
        x
    }

    /// Batch backward.
    pub fn backward_batch(&mut self, grads: &Tensor) -> Tensor {
        self.check_integer_backend();
        let mut g: Tensor = grads.clone();
        for layer in self.layers.iter_mut().rev() {
            g = layer.backward_batch(&g);
        }
        g
    }

    /// Backpropagate for a training step that never consumes model-input
    /// gradients. Only the first layer can safely omit them: all subsequent
    /// layer input gradients are needed for the preceding layer's parameters.
    /// Explicit opt-in keeps the original backward_batch API unchanged.
    pub fn backward_batch_discard_input_grad(&mut self, grads: &Tensor) {
        self.check_integer_backend();
        let mut g = grads.clone();
        for (index, layer) in self.layers.iter_mut().enumerate().rev() {
            if index == 0 {
                layer.backward_batch_discard_input_grad(&g);
            } else {
                g = layer.backward_batch(&g);
            }
        }
    }

    /// Profile individual backward layers, recording time in forward layer
    /// order. This method calls every layer exactly once and in the same
    /// sequence as backward_batch; no RNG draws are added, skipped or moved.
    /// Stopwatch overhead means these figures are diagnostic, not cycle counts.
    pub fn backward_batch_timed(&mut self, grads: &Tensor, layer_ns: &mut [u128]) -> Tensor {
        self.check_integer_backend();
        assert_eq!(
            layer_ns.len(),
            self.layers.len(),
            "layer timing count mismatch"
        );
        let mut g: Tensor = grads.clone();
        for (index, layer) in self.layers.iter_mut().enumerate().rev() {
            let started = Instant::now();
            g = layer.backward_batch(&g);
            layer_ns[index] += started.elapsed().as_nanos();
        }
        g
    }

    /// Same timing attribution as backward_batch_timed, but do not compute
    /// the model's unconsumed input gradients. Profile the first layer's
    /// specialized work rather than a full-backward surrogate.
    pub fn backward_batch_timed_discard_input_grad(
        &mut self,
        grads: &Tensor,
        layer_ns: &mut [u128],
    ) {
        self.check_integer_backend();
        assert_eq!(
            layer_ns.len(),
            self.layers.len(),
            "layer timing count mismatch"
        );
        let mut g = grads.clone();
        for (index, layer) in self.layers.iter_mut().enumerate().rev() {
            let started = Instant::now();
            if index == 0 {
                layer.backward_batch_discard_input_grad(&g);
            } else {
                g = layer.backward_batch(&g);
            }
            layer_ns[index] += started.elapsed().as_nanos();
        }
    }

    pub fn numeric_snapshots(&self) -> Vec<(usize, &'static str, LayerNumericSnapshot)> {
        self.layers.iter().enumerate()
            .filter_map(|(i, layer)| layer.numeric_snapshot().map(|s| (i, layer.name(), s)))
            .collect()
    }

    pub fn update(&mut self, lr_shift: u32) {
        self.check_integer_backend();
        for layer in &mut self.layers {
            layer.update(lr_shift);
        }
    }

    pub fn zero_grad(&mut self) {
        self.check_integer_backend();
        for layer in &mut self.layers {
            layer.zero_grad();
        }
    }

    pub fn set_training(&mut self, training: bool) {
        self.check_integer_backend();
        for layer in &mut self.layers {
            layer.set_training(training);
        }
    }

    pub fn summary(&self) {
        println!("Sequential(");
        for (i, layer) in self.layers.iter().enumerate() {
            println!("  ({i}): {}", layer.describe());
        }
        println!(")");
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Shared optimiser helpers ─────────────────────────────────────────────────

fn sgd_step(w: Dyadic, g: Dyadic, lr_shift: u32) -> Dyadic {
    let eff = (g.s as i64) + (lr_shift as i64) - (w.s as i64);
    let delta = if eff >= 0 {
        stochastic_round(g.v, eff as u32)
    } else {
        let k = (-eff).min(30) as u32;
        ((g.v as i64) << k).clamp(i32::MIN as i64, i32::MAX as i64) as i32
    };
    Dyadic::new(w.v.saturating_sub(delta), w.s)
}

fn momentum_step(w: Dyadic, g: Dyadic, v: &mut Dyadic, lr_shift: u32, m: u32) -> Dyadic {
    // Align to the FINER dyadic scale. Previously an initial zero velocity
    // at scale 0 quantized every sub-unit mean gradient away.
    let decayed = Dyadic::new(stochastic_round(v.v, m), v.s);
    *v = add(decayed, g);
    sgd_step(w, *v, lr_shift)
}

/// Integer-only dyadic approximation to a sample mean.
///
/// Power-of-two denominators are exact. Other denominators are divided
/// in i128 fixed point and rounded to nearest (ties to even); the guard
/// precision backs off until the signed i32 mantissa fits.
fn mean_gradient(g: Dyadic, samples: usize) -> Dyadic {
    assert!(samples > 0, "mean_gradient: sample count must be nonzero");
    if samples == 1 {
        return g;
    }
    if samples.is_power_of_two() {
        return Dyadic::new(
            g.v,
            g.s.checked_add(samples.trailing_zeros())
                .expect("mean_gradient: output scale overflow"),
        );
    }
    let ceil_log2 = usize::BITS - (samples - 1).leading_zeros();
    let mut bits = (ceil_log2 + 8).min(30).min(u32::MAX - g.s);
    let magnitude = (g.v as i128).abs();
    let divisor = samples as i128;
    loop {
        let numerator = magnitude << bits;
        let mut quotient = numerator / divisor;
        let remainder = numerator % divisor;
        if remainder * 2 > divisor || (remainder * 2 == divisor && quotient & 1 != 0) {
            quotient += 1;
        }
        let limit = if g.v < 0 {
            1i128 << 31
        } else {
            i32::MAX as i128
        };
        if quotient <= limit {
            let signed = if g.v < 0 { -quotient } else { quotient };
            return Dyadic::new(signed as i32, g.s + bits);
        }
        assert!(bits > 0, "mean_gradient: mean overflow after division");
        bits -= 1;
    }
}

// Compatibility path for experimental floating-point BatchNorm modules:
// preserves their existing SUM-gradient optimizer behavior. Not used by
// the ADR-002 integer-only trainable layers.
fn apply_updates(
    params: &mut [Dyadic],
    grads: &Tensor,
    vels: &mut [Dyadic],
    lr_shift: u32,
    momentum: Option<u32>,
) {
    match momentum {
        None => params
            .iter_mut()
            .zip(grads.data.iter())
            .for_each(|(w, &g)| *w = sgd_step(*w, g, lr_shift)),
        Some(m) => params
            .iter_mut()
            .zip(grads.data.iter())
            .zip(vels.iter_mut())
            .for_each(|((w, &g), v)| *w = momentum_step(*w, g, v, lr_shift, m)),
    }
}

fn apply_updates_mean(
    params: &mut [Dyadic],
    grads: &Tensor,
    vels: &mut [Dyadic],
    lr_shift: u32,
    momentum: Option<u32>,
    samples: usize,
) {
    assert!(samples > 0, "apply_updates: no accumulated examples");
    assert_eq!(
        params.len(),
        grads.data.len(),
        "apply_updates: param/gradient mismatch"
    );
    assert_eq!(
        params.len(),
        vels.len(),
        "apply_updates: param/velocity mismatch"
    );
    match momentum {
        None => params
            .iter_mut()
            .zip(grads.data.iter())
            .for_each(|(w, &g)| *w = sgd_step(*w, mean_gradient(g, samples), lr_shift)),
        Some(m) => params
            .iter_mut()
            .zip(grads.data.iter())
            .zip(vels.iter_mut())
            .for_each(|((w, &g), v)| {
                *w = momentum_step(*w, mean_gradient(g, samples), v, lr_shift, m)
            }),
    }
}

// ─── Flatten ──────────────────────────────────────────────────────────────────

/// Pass-through shape bridge (conv → linear). Exists for `summary()` clarity.
pub struct Flatten;
impl Module for Flatten {
    fn training_arithmetic(&self) -> TrainingArithmetic {
        TrainingArithmetic::IntegerOnly
    }
    fn name(&self) -> &'static str {
        "Flatten"
    }
    fn forward(&mut self, x: &TensorView) -> Tensor {
        x.to_tensor()
    }
    fn backward(&mut self, g: &TensorView) -> Tensor {
        g.to_tensor()
    }
    fn update(&mut self, _: u32) {}
    fn zero_grad(&mut self) {}
}

#[cfg(test)]
mod gradient_average_tests {
    use super::{Dyadic, mean_gradient};

    #[test]
    fn power_of_two_sample_means_are_exact() {
        assert_eq!(mean_gradient(Dyadic::new(7, 4), 32), Dyadic::new(7, 9));
        assert_eq!(mean_gradient(Dyadic::new(-7, 4), 8), Dyadic::new(-7, 7));
        assert_eq!(mean_gradient(Dyadic::new(19, 3), 1), Dyadic::new(19, 3));
    }

    #[test]
    fn non_power_of_two_means_match_reference_with_bounded_rounding() {
        for samples in [3usize, 7, 10, 24, 31] {
            for v in [-123, -23, -1, 0, 1, 23, 123] {
                let original = Dyadic::new(v, 7);
                let actual = mean_gradient(original, samples);
                let reference = original.to_f64() / samples as f64;
                let tolerance = 0.5 * 2f64.powi(-(actual.s as i32)) + 1e-12;
                assert!(
                    (actual.to_f64() - reference).abs() <= tolerance,
                    "{original:?}, samples={samples}, actual={actual:?}"
                );
            }
        }
    }

    #[test]
    fn extreme_mantissas_do_not_overflow_or_wrap() {
        for v in [i32::MIN, i32::MAX] {
            for samples in [3, 24, 31, 1000] {
                let actual = mean_gradient(Dyadic::new(v, 0), samples);
                let expected = v as f64 / samples as f64;
                assert!(
                    (actual.to_f64() - expected).abs() <= 1.0,
                    "v={v}, samples={samples}, actual={actual:?}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "sample count must be nonzero")]
    fn zero_samples_rejected() {
        mean_gradient(Dyadic::new(3, 0), 0);
    }
}
