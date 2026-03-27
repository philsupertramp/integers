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
pub use regularization::{Dropout, BatchNorm1D, BatchNorm2D, MaxPool2D};

mod conv;
pub use conv::Conv2D;

use crate::dyadic::{
    add, mul, requantize, signed_bounds, ste_requantize, stochastic_round, Dyadic,
    Tensor, TensorView
};
use crate::rng::rng_range;

// ─── Module trait ──────────────────────────────────────────────────────────────

/// The extension point for all layer types.
pub trait Module {
    // ── Per-sample API (inference + default batch impl) ───────────────────────

    /// Forward pass for a single sample.  Must cache everything `backward` needs.
    fn forward(&mut self, input: TensorView) -> Tensor;

    /// Backward pass for a single sample.  *Accumulates* `∂L/∂params`.
    fn backward(&mut self, grad_output: TensorView) -> Tensor;

    // ── Batch API (training) ──────────────────────────────────────────────────

    /// Forward pass for a mini-batch.
    ///
    /// Default: calls `forward` once per sample.  Override when the layer
    /// needs to see all N inputs simultaneously (BatchNorm) or needs to store
    /// all N caches so that `backward_batch` can use them (Linear, ReLU, …).
    fn forward_batch(&mut self, inputs: &Tensor) -> Tensor {
        inputs.iter().map(|x| self.forward(x)).collect()
    }

    /// Backward pass for a mini-batch.
    ///
    /// Default: calls `backward` once per grad.  Override whenever
    /// `forward_batch` is overridden.
    fn backward_batch(&mut self, grads: &Tensor) -> Tensor {
        grads.iter().map(|g| self.backward(g)).collect()
    }

    // ── Parameter management ──────────────────────────────────────────────────

    /// One optimiser step: `params -= lr * velocity`.
    fn update(&mut self, lr_shift: u32);

    /// Reset accumulated gradients.  Does **not** reset velocity.
    fn zero_grad(&mut self);

    // ── Metadata ─────────────────────────────────────────────────────────────

    fn name(&self) -> &'static str;
    fn describe(&self) -> String { self.name().to_string() }

    /// Toggle training / eval mode.  Only meaningful for stochastic layers
    /// (Dropout, BatchNorm).  Default: no-op.
    fn set_training(&mut self, _training: bool) {}
}

// ─── Sequential ───────────────────────────────────────────────────────────────

pub struct Sequential {
    pub layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new() -> Self { Self { layers: Vec::new() } }

    pub fn add<L: Module + 'static>(&mut self, layer: L) {
        self.layers.push(Box::new(layer));
    }

    /// Per-sample forward — use for inference.
    pub fn forward(&mut self, input: TensorView) -> Tensor {
        let mut x = input.to_tensor();
        for layer in &mut self.layers { x = layer.forward(x.view()); }
        x
    }

    /// Per-sample backward.
    pub fn backward(&mut self, grad: TensorView) -> Tensor {
        let mut g = grad.to_tensor();
        for layer in self.layers.iter_mut().rev() { g = layer.backward(g.view()); }
        g
    }

    /// Batch forward — use during training (required for BatchNorm correctness).
    pub fn forward_batch(&mut self, inputs: &Tensor) -> Tensor {
        let mut x: Tensor = inputs.clone();
        for layer in &mut self.layers {
            x = layer.forward_batch(&x);
        }
        x
    }

    /// Batch backward.
    pub fn backward_batch(&mut self, grads: &Tensor) -> Tensor {
        let mut g: Tensor = grads.clone();
        for layer in self.layers.iter_mut().rev() {
            g = layer.backward_batch(&g);
        }
        g
    }

    pub fn update(&mut self, lr_shift: u32) {
        for layer in &mut self.layers { layer.update(lr_shift); }
    }

    pub fn zero_grad(&mut self) {
        for layer in &mut self.layers { layer.zero_grad(); }
    }

    pub fn set_training(&mut self, training: bool) {
        for layer in &mut self.layers { layer.set_training(training); }
    }

    pub fn summary(&self) {
        println!("Sequential(");
        for (i, layer) in self.layers.iter().enumerate() {
            println!("  ({i}): {}", layer.describe());
        }
        println!(")");
    }
}

impl Default for Sequential { fn default() -> Self { Self::new() } }

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
    let vd = Dyadic::new(stochastic_round(v.v, m), v.s);
    let (vs, vn) = if g.s >= v.s {
        let ga = Dyadic::new(stochastic_round(g.v, g.s - v.s), v.s);
        (v.s, Dyadic::new(vd.v.saturating_add(ga.v), v.s))
    } else {
        let shift = v.s - g.s;
        let vda = Dyadic::new(stochastic_round(vd.v, shift), g.s);
        (g.s, Dyadic::new(vda.v.saturating_add(g.v), g.s))
    };
    *v = Dyadic::new(vn.v, vs);
    sgd_step(w, *v, lr_shift)
}

fn apply_updates(
    params: &mut [Dyadic], grads: &Tensor, vels: &mut [Dyadic],
    lr_shift: u32, momentum: Option<u32>,
) {
    match momentum {
        None    => params.iter_mut().zip(grads.data.iter())
            .for_each(|(w, &g)| *w = sgd_step(*w, g, lr_shift)),
        Some(m) => params.iter_mut().zip(grads.data.iter()).zip(vels.iter_mut())
            .for_each(|((w, &g), v)| *w = momentum_step(*w, g, v, lr_shift, m)),
    }
}


// ─── Flatten ──────────────────────────────────────────────────────────────────

/// Pass-through shape bridge (conv → linear). Exists for `summary()` clarity.
pub struct Flatten;
impl Module for Flatten {
    fn name(&self) -> &'static str { "Flatten" }
    fn forward(&mut self, x: TensorView) -> Tensor { x.to_tensor() }
    fn backward(&mut self, g: TensorView) -> Tensor { g.to_tensor() }
    fn update(&mut self, _: u32) {}
    fn zero_grad(&mut self) {}
}
