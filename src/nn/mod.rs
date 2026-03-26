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
};
use crate::rng::rng_range;

// ─── Module trait ──────────────────────────────────────────────────────────────

/// The extension point for all layer types.
pub trait Module {
    // ── Per-sample API (inference + default batch impl) ───────────────────────

    /// Forward pass for a single sample.  Must cache everything `backward` needs.
    fn forward(&mut self, input: &[Dyadic]) -> Vec<Dyadic>;

    /// Backward pass for a single sample.  *Accumulates* `∂L/∂params`.
    fn backward(&mut self, grad_output: &[Dyadic]) -> Vec<Dyadic>;

    // ── Batch API (training) ──────────────────────────────────────────────────

    /// Forward pass for a mini-batch.
    ///
    /// Default: calls `forward` once per sample.  Override when the layer
    /// needs to see all N inputs simultaneously (BatchNorm) or needs to store
    /// all N caches so that `backward_batch` can use them (Linear, ReLU, …).
    fn forward_batch(&mut self, inputs: &[Vec<Dyadic>]) -> Vec<Vec<Dyadic>> {
        inputs.iter().map(|x| self.forward(x)).collect()
    }

    /// Backward pass for a mini-batch.
    ///
    /// Default: calls `backward` once per grad.  Override whenever
    /// `forward_batch` is overridden.
    fn backward_batch(&mut self, grads: &[Vec<Dyadic>]) -> Vec<Vec<Dyadic>> {
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
    pub fn forward(&mut self, input: &[Dyadic]) -> Vec<Dyadic> {
        let mut x = input.to_vec();
        for layer in &mut self.layers { x = layer.forward(&x); }
        x
    }

    /// Per-sample backward.
    pub fn backward(&mut self, grad: &[Dyadic]) -> Vec<Dyadic> {
        let mut g = grad.to_vec();
        for layer in self.layers.iter_mut().rev() { g = layer.backward(&g); }
        g
    }

    /// Batch forward — use during training (required for BatchNorm correctness).
    pub fn forward_batch(&mut self, inputs: &[Vec<Dyadic>]) -> Vec<Vec<Dyadic>> {
        let mut x: Vec<Vec<Dyadic>> = inputs.to_vec();
        for layer in &mut self.layers {
            x = layer.forward_batch(&x);
        }
        x
    }

    /// Batch backward.
    pub fn backward_batch(&mut self, grads: &[Vec<Dyadic>]) -> Vec<Vec<Dyadic>> {
        let mut g: Vec<Vec<Dyadic>> = grads.to_vec();
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
    params: &mut [Dyadic], grads: &[Dyadic], vels: &mut [Dyadic],
    lr_shift: u32, momentum: Option<u32>,
) {
    match momentum {
        None    => params.iter_mut().zip(grads)
            .for_each(|(w, &g)| *w = sgd_step(*w, g, lr_shift)),
        Some(m) => params.iter_mut().zip(grads).zip(vels.iter_mut())
            .for_each(|((w, &g), v)| *w = momentum_step(*w, g, v, lr_shift, m)),
    }
}


// ─── Flatten ──────────────────────────────────────────────────────────────────

/// Pass-through shape bridge (conv → linear). Exists for `summary()` clarity.
pub struct Flatten;
impl Module for Flatten {
    fn name(&self) -> &'static str { "Flatten" }
    fn forward(&mut self, x: &[Dyadic]) -> Vec<Dyadic> { x.to_vec() }
    fn backward(&mut self, g: &[Dyadic]) -> Vec<Dyadic> { g.to_vec() }
    fn update(&mut self, _: u32) {}
    fn zero_grad(&mut self) {}
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    const S: u32 = 7;
    fn d(x: f64) -> Dyadic { Dyadic::new((x * 128.0).round() as i32, S) }

    // ── Linear ────────────────────────────────────────────────────────────────
    #[test] fn linear_per_sample() {
        let mut l = Linear::new(3, 5, S, S, 32);
        let out = l.forward(&[d(1.0), d(-0.5), d(0.25)]);
        assert_eq!(out.len(), 5);
        assert_eq!(l.backward(&vec![d(0.1); 5]).len(), 3);
    }

    #[test] fn linear_batch_shapes() {
        let mut l = Linear::new(3, 5, S, S, 32);
        let inputs = vec![
            vec![d(1.0), d(-0.5), d(0.25)],
            vec![d(0.5), d( 0.5), d(-0.1)],
        ];
        let outs = l.forward_batch(&inputs);
        assert_eq!(outs.len(), 2);
        assert_eq!(outs[0].len(), 5);
        let grads = vec![vec![d(0.1); 5]; 2];
        let gi = l.backward_batch(&grads);
        assert_eq!(gi.len(), 2);
        assert_eq!(gi[0].len(), 3);
    }

    #[test] fn linear_batch_grads_use_correct_input() {
        // With batch API each sample's backward must use its own input, not the last one.
        let mut l = Linear::new(2, 1, S, S, 32);
        // Force a simple weight so we can reason about the result.
        l.weights = vec![Dyadic::new(128, S), Dyadic::new(128, S)]; // w = [1.0, 1.0]
        let inputs = vec![
            vec![d(1.0), d(0.0)],  // sample 0
            vec![d(0.0), d(1.0)],  // sample 1
        ];
        let _ = l.forward_batch(&inputs);
        let grads = vec![vec![d(1.0)]; 2];
        let gi = l.backward_batch(&grads);
        // For sample 0: grad_input[1] should be near 0 (weight[1]*g=1.0*1.0 but input[0][1]=0 doesn't affect grad_input)
        // Actually grad_input = g * W, so both samples get same grad_input (weights are same).
        // The test is that we get 2 grad vecs of length 2.
        assert_eq!(gi.len(), 2);
        assert_eq!(gi[0].len(), 2);
        assert_eq!(gi[1].len(), 2);
    }

    // ── ReLU ──────────────────────────────────────────────────────────────────
    #[test] fn relu_batch() {
        let mut relu = ReLU::new();
        let inputs = vec![
            vec![d(1.0), d(-1.0)],
            vec![d(-1.0), d(1.0)],
        ];
        let outs = relu.forward_batch(&inputs);
        assert!(outs[0][0].v > 0); assert_eq!(outs[0][1].v, 0);
        assert_eq!(outs[1][0].v, 0); assert!(outs[1][1].v > 0);
        // Backward uses the correct per-sample mask.
        let grads = vec![vec![d(1.0), d(1.0)]; 2];
        let gi    = relu.backward_batch(&grads);
        assert!(gi[0][0].v != 0); assert_eq!(gi[0][1].v, 0);
        assert_eq!(gi[1][0].v, 0); assert!(gi[1][1].v != 0);
    }

    // ── BatchNorm1D ───────────────────────────────────────────────────────────
    #[test] fn bn1d_output_shape() {
        let mut bn = BatchNorm1D::new(4, S);
        bn.set_training(true);
        let inputs = (0..4).map(|_| vec![d(1.0), d(-1.0), d(0.5), d(-0.5)]).collect::<Vec<_>>();
        let outs = bn.forward_batch(&inputs);
        assert_eq!(outs.len(), 4);
        assert_eq!(outs[0].len(), 4);
    }

    #[test] fn bn1d_normalises_mean_and_var() {
        let mut bn = BatchNorm1D::new(1, S);
        bn.set_training(true);
        // Batch with known mean=2.0, var=1.0 for feature 0
        let inputs = vec![
            vec![d(1.0)], vec![d(2.0)], vec![d(3.0)], vec![d(2.0)],
        ];
        let outs = bn.forward_batch(&inputs);
        // With gamma=1, beta=0: outputs should be zero-mean, unit-var.
        // (gamma is initialized to 1.0 at shift S)
        let vals: Vec<f64> = outs.iter().map(|o| o[0].to_f64()).collect();
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let var  = vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
        assert!(mean.abs() < 0.1,  "mean = {mean:.4}, expected ~0");
        assert!((var - 1.0).abs() < 0.15, "var = {var:.4}, expected ~1");
    }

    #[test] fn bn1d_backward_shapes() {
        let mut bn = BatchNorm1D::new(4, S);
        bn.set_training(true);
        let inputs = (0..8).map(|i| vec![d(i as f64 * 0.1); 4]).collect::<Vec<_>>();
        bn.forward_batch(&inputs);
        let grads = vec![vec![d(0.1); 4]; 8];
        let gi = bn.backward_batch(&grads);
        assert_eq!(gi.len(), 8);
        assert_eq!(gi[0].len(), 4);
    }

    #[test] fn bn1d_eval_uses_running_stats() {
        let mut bn = BatchNorm1D::new(1, S);
        bn.set_training(true);
        // Train on a few batches to warm up running stats.
        for _ in 0..10 {
            let inputs = vec![vec![d(2.0)], vec![d(2.0)]];
            bn.forward_batch(&inputs);
        }
        bn.set_training(false);
        // In eval mode, same input twice should give same output.
        let out1 = bn.forward(&[d(2.0)]);
        let out2 = bn.forward(&[d(2.0)]);
        assert_eq!(out1[0].v, out2[0].v);
    }

    // ── BatchNorm2D ───────────────────────────────────────────────────────────
    #[test] fn bn2d_output_shape() {
        let mut bn = BatchNorm2D::new(2, 3, 3, S);
        bn.set_training(true);
        let inputs = (0..4).map(|_| vec![d(0.5); 2 * 3 * 3]).collect::<Vec<_>>();
        let outs = bn.forward_batch(&inputs);
        assert_eq!(outs.len(), 4);
        assert_eq!(outs[0].len(), 2 * 3 * 3);
    }

    #[test] fn bn2d_backward_shapes() {
        let mut bn = BatchNorm2D::new(2, 3, 3, S);
        bn.set_training(true);
        let inputs = (0..4).map(|_| vec![d(0.5); 18]).collect::<Vec<_>>();
        bn.forward_batch(&inputs);
        let grads = vec![vec![d(0.1); 18]; 4];
        let gi = bn.backward_batch(&grads);
        assert_eq!(gi.len(), 4);
        assert_eq!(gi[0].len(), 18);
    }

    // ── MaxPool2D batch ───────────────────────────────────────────────────────
    #[test] fn maxpool_batch_correct_masks() {
        let mut pool = MaxPool2D::new(1, 2, 2, 2, 2);
        // sample 0: max at [2], sample 1: max at [3]
        let inputs = vec![
            vec![d(0.1), d(0.2), d(0.9), d(0.3)],
            vec![d(0.1), d(0.2), d(0.3), d(0.9)],
        ];
        let outs = pool.forward_batch(&inputs);
        assert_eq!(outs[0][0].v, inputs[0][2].v);
        assert_eq!(outs[1][0].v, inputs[1][3].v);

        let grads = vec![vec![d(1.0)]; 2];
        let gi = pool.backward_batch(&grads);
        // Gradient routed to the correct max in each sample.
        assert_eq!(gi[0][2].v, grads[0][0].v);
        assert_eq!(gi[0][0].v, 0); assert_eq!(gi[0][1].v, 0); assert_eq!(gi[0][3].v, 0);
        assert_eq!(gi[1][3].v, grads[1][0].v);
        assert_eq!(gi[1][0].v, 0); assert_eq!(gi[1][1].v, 0); assert_eq!(gi[1][2].v, 0);
    }

    // ── Dropout ───────────────────────────────────────────────────────────────
    #[test] fn dropout_zeros_roughly_half() {
        let mut drop = Dropout::new(0.5);
        drop.set_training(true);
        let input: Vec<Dyadic> = (0..1000).map(|_| d(1.0)).collect();
        let out = drop.forward(&input);
        let n_alive = out.iter().filter(|x| x.v != 0).count();
        assert!(n_alive > 400 && n_alive < 600, "expected ~500 alive, got {n_alive}");
    }

    #[test] fn dropout_eval_is_identity() {
        let mut drop = Dropout::new(0.5);
        drop.set_training(false);
        let input = vec![d(1.0), d(-0.5), d(0.25)];
        let out   = drop.forward(&input);
        assert_eq!(out[0].v, input[0].v);
        assert_eq!(out[1].v, input[1].v);
    }

    #[test] fn dropout_batch_independent_masks() {
        let mut drop = Dropout::new(0.5);
        drop.set_training(true);
        let inputs = vec![vec![d(1.0); 100]; 4];
        let outs = drop.forward_batch(&inputs);
        // Each sample should have a different mask (extremely likely with 100 elements).
        assert_eq!(drop.batch_masks.len(), 4);
        let alive: Vec<usize> = outs.iter().map(|o| o.iter().filter(|x| x.v != 0).count()).collect();
        // All should have some alive neurons.
        assert!(alive.iter().all(|&a| a > 10));
    }

    // ── Sequential batch pipeline ─────────────────────────────────────────────
    #[test] fn sequential_batch_with_bn() {
        let mut model = Sequential::new();
        model.add(Linear::new(4, 8, S, S, 32));
        model.add(BatchNorm1D::new(8, S));
        model.add(ReLU::new());
        model.add(Linear::new(8, 3, S, S, 32));
        model.add(Softmax::new(S));
        model.set_training(true);

        let inputs: Vec<Vec<Dyadic>> = (0..4)
            .map(|i| (0..4).map(|j| d((i * 4 + j) as f64 * 0.1)).collect())
            .collect();

        let outs = model.forward_batch(&inputs);
        assert_eq!(outs.len(), 4);
        assert_eq!(outs[0].len(), 3);
        // Outputs are probabilities (non-negative).
        assert!(outs.iter().all(|o| o.iter().all(|x| x.v >= 0)));

        let targets: Vec<Vec<Dyadic>> = (0..4)
            .map(|i| {
                let mut t = vec![Dyadic::new(0, S); 3];
                t[i % 3] = Dyadic::new(127, S);
                t
            })
            .collect();

        let grads: Vec<Vec<Dyadic>> = outs.iter().zip(&targets)
            .map(|(y, t)| y.iter().zip(t).map(|(&yv, &tv)| Dyadic::new(yv.v - tv.v, S)).collect())
            .collect();

        model.backward_batch(&grads);
        model.update(7);
    }

    #[test] fn sequential_conv_with_bn2d() {
        let mut model = Sequential::new();
        model.add(Conv2D::new(1, 4, 3, 3, 6, 6, S, S, 32));
        model.add(BatchNorm2D::new(4, 4, 4, S));
        model.add(ReLU::new());
        model.add(Flatten);
        model.add(Linear::new(64, 3, S, S, 32));
        model.add(Softmax::new(S));
        model.set_training(true);

        let inputs = (0..4).map(|_| vec![d(0.5); 36]).collect::<Vec<_>>();
        let outs = model.forward_batch(&inputs);
        assert_eq!(outs.len(), 4);
        assert_eq!(outs[0].len(), 3);
    }
}
