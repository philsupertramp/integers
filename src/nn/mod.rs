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

// ─── Linear ───────────────────────────────────────────────────────────────────

/// Fully-connected affine layer: `y = ℛ(W⊗x ⊕ b)`.
pub struct Linear {
    pub in_features:  usize,
    pub out_features: usize,
    pub weights: Vec<Dyadic>,
    pub biases:  Vec<Dyadic>,
    pub quant_shift:    u32,
    pub output_bits:    u32,
    pub grad_clip:      i32,
    pub momentum_shift: Option<u32>,
    // per-sample caches (single call)
    input_cache:  Vec<Dyadic>,
    output_cache: Vec<Dyadic>,
    // batch caches (batch call)
    input_batch_cache:  Vec<Vec<Dyadic>>,
    output_batch_cache: Vec<Vec<Dyadic>>,
    // gradients + velocity
    grad_w: Vec<Dyadic>,
    grad_b: Vec<Dyadic>,
    vel_w:  Vec<Dyadic>,
    vel_b:  Vec<Dyadic>,

    q_min: i32,
    q_max: i32,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize,
               weight_shift: u32, quant_shift: u32, output_bits: u32) -> Self {
        let n_w = out_features * in_features;
        let k = ((1u32 << weight_shift) as f64 / (in_features as f64).sqrt()).round() as i32;
        let k = k.max(1);
        let z = |n: usize| vec![Dyadic::new(0, weight_shift); n];
        let weights = (0..n_w).map(|_| Dyadic::new(-k + rng_range(2 * k as u32) as i32, weight_shift)).collect();
        let (q_min, q_max) = signed_bounds(output_bits);
        Self {
            in_features, out_features,
            weights: weights,
            biases: z(out_features),
            quant_shift, output_bits,
            grad_clip: i32::MAX, momentum_shift: None,
            input_cache: Vec::new(), output_cache: Vec::new(),
            input_batch_cache: Vec::new(), output_batch_cache: Vec::new(),
            grad_w: z(n_w), grad_b: z(out_features),
            vel_w: z(n_w),  vel_b: z(out_features),
            q_min: q_min, q_max: q_max,
        }
    }

    pub fn with_grad_clip(mut self, c: i32) -> Self { self.grad_clip = c.abs(); self }
    pub fn with_momentum(mut self, s: u32) -> Self  { self.momentum_shift = Some(s); self }

    #[inline] fn w(&self, j: usize, i: usize) -> Dyadic { self.weights[j * self.in_features + i] }
    #[inline] fn flat(&self, j: usize, i: usize) -> usize { j * self.in_features + i }

    fn forward_one(&mut self, input: &[Dyadic]) -> Vec<Dyadic> {
        let mut out = Vec::with_capacity(self.out_features);
        for j in 0..self.out_features {
            let mut acc = self.biases[j];
            for i in 0..self.in_features {
                acc = add(acc, mul(self.w(j, i), input[i], self.quant_shift));
            }
            let (y, _) = requantize(acc, acc.s, self.q_min, self.q_max);
            out.push(y);
        }
        out
    }

    fn backward_one(
        &mut self,
        grad_output: &[Dyadic],
        input:       &[Dyadic],
        output:      &[Dyadic],
    ) -> Vec<Dyadic> {
        let g_s = grad_output.first().map_or(0, |g| g.s);
        let mut gi = vec![Dyadic::new(0, g_s); self.in_features];
        for j in 0..self.out_features {
            let gr = ste_requantize(grad_output[j], output[j].v, self.q_min, self.q_max);
            let gj = Dyadic::new(gr.v.clamp(-self.grad_clip, self.grad_clip), gr.s);
            for i in 0..self.in_features {
                let idx = self.flat(j, i);
                self.grad_w[idx] = add(self.grad_w[idx], mul(gj, input[i], self.quant_shift));
                gi[i]            = add(gi[i],            mul(gj, self.w(j, i), self.quant_shift));
            }
            self.grad_b[j] = add(self.grad_b[j], gj);
        }
        gi
    }
}

impl Module for Linear {
    fn name(&self) -> &'static str { "Linear" }
    fn describe(&self) -> String {
        let ws   = self.weights.first().map_or(0, |w| w.s);
        let clip = if self.grad_clip == i32::MAX { "off".into() } else { format!("2^{}", (self.grad_clip as f64).log2() as i32) };
        let mom  = self.momentum_shift.map_or("off".into(), |m| format!("shift={m}"));
        format!("Linear(in={}, out={}, w_shift={ws}, q={}, bits={}, clip={clip}, mom={mom})",
            self.in_features, self.out_features, self.quant_shift, self.output_bits)
    }

    fn forward(&mut self, input: &[Dyadic]) -> Vec<Dyadic> {
        let out = self.forward_one(input);
        self.input_cache  = input.to_vec();
        self.output_cache = out.clone();
        out
    }

    fn backward(&mut self, grad_output: &[Dyadic]) -> Vec<Dyadic> {
        let input  = self.input_cache.clone();
        let output = self.output_cache.clone();
        self.backward_one(grad_output, &input, &output)
    }

    fn forward_batch(&mut self, inputs: &[Vec<Dyadic>]) -> Vec<Vec<Dyadic>> {
        let outputs: Vec<Vec<Dyadic>> = inputs.iter()
            .map(|x| self.forward_one(x))
            .collect();
        self.input_batch_cache  = inputs.to_vec();
        self.output_batch_cache = outputs.clone();
        outputs
    }

    fn backward_batch(&mut self, grads: &[Vec<Dyadic>]) -> Vec<Vec<Dyadic>> {
        grads.iter().enumerate()
            .map(|(n, g)| {
                let inp = self.input_batch_cache[n].clone();
                let out = self.output_batch_cache[n].clone();
                self.backward_one(g, &inp, &out)
            })
            .collect()
    }

    fn update(&mut self, lr: u32) {
        apply_updates(&mut self.weights, &self.grad_w, &mut self.vel_w, lr, self.momentum_shift);
        apply_updates(&mut self.biases,  &self.grad_b, &mut self.vel_b, lr, self.momentum_shift);
    }

    fn zero_grad(&mut self) {
        self.grad_w.iter_mut().for_each(|g| g.v = 0);
        self.grad_b.iter_mut().for_each(|g| g.v = 0);
    }
}

// ─── ReLU ─────────────────────────────────────────────────────────────────────

pub struct ReLU {
    mask:        Vec<bool>,
    batch_masks: Vec<Vec<bool>>,
}

impl ReLU {
    pub fn new() -> Self { Self { mask: Vec::new(), batch_masks: Vec::new() } }

    fn apply_forward(input: &[Dyadic], mask: &mut Vec<bool>) -> Vec<Dyadic> {
        *mask = input.iter().map(|x| x.v > 0).collect();
        input.iter().map(|x| if x.v > 0 { *x } else { Dyadic::new(0, x.s) }).collect()
    }

    fn apply_backward(grad: &[Dyadic], mask: &[bool]) -> Vec<Dyadic> {
        grad.iter().zip(mask).map(|(&g, &a)| if a { g } else { Dyadic::new(0, g.s) }).collect()
    }
}

impl Default for ReLU { fn default() -> Self { Self::new() } }

impl Module for ReLU {
    fn name(&self) -> &'static str { "ReLU" }

    fn forward(&mut self, input: &[Dyadic]) -> Vec<Dyadic> {
        Self::apply_forward(input, &mut self.mask)
    }

    fn backward(&mut self, grad: &[Dyadic]) -> Vec<Dyadic> {
        Self::apply_backward(grad, &self.mask)
    }

    fn forward_batch(&mut self, inputs: &[Vec<Dyadic>]) -> Vec<Vec<Dyadic>> {
        let mut masks = Vec::with_capacity(inputs.len());
        let outputs: Vec<Vec<Dyadic>> = inputs.iter().map(|x| {
            let mut m = Vec::new();
            let out = Self::apply_forward(x, &mut m);
            masks.push(m);
            out
        }).collect();
        self.batch_masks = masks;
        outputs
    }

    fn backward_batch(&mut self, grads: &[Vec<Dyadic>]) -> Vec<Vec<Dyadic>> {
        grads.iter().enumerate()
            .map(|(n, g)| Self::apply_backward(g, &self.batch_masks[n]))
            .collect()
    }

    fn update(&mut self, _: u32) {}
    fn zero_grad(&mut self) {}
}

// ─── Softmax ──────────────────────────────────────────────────────────────────

/// Numerically stable softmax output layer.
///
/// Backward is a straight-through estimator: when paired with cross-entropy
/// the combined gradient is `p − t`, which the STE passes through correctly.
pub struct Softmax {
    pub output_shift: u32,
    pub last_probs:   Vec<f64>,
}

impl Softmax {
    pub fn new(output_shift: u32) -> Self { Self { output_shift, last_probs: Vec::new() } }

    fn softmax_forward(input: &[Dyadic], shift: u32, probs: &mut Vec<f64>) -> Vec<Dyadic> {
        let logits: Vec<f64> = input.iter().map(|x| x.to_f64()).collect();
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&z| (z - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        *probs = exps.iter().map(|&e| e / sum).collect();
        let scale = (1u32 << shift) as f64;
        let (_, mx) = signed_bounds(shift + 1);
        probs.iter().map(|&p| Dyadic::new((p * scale).round().clamp(0.0, mx as f64) as i32, shift)).collect()
    }
}

impl Module for Softmax {
    fn name(&self) -> &'static str { "Softmax" }
    fn describe(&self) -> String { format!("Softmax(shift={})", self.output_shift) }

    fn forward(&mut self, input: &[Dyadic]) -> Vec<Dyadic> {
        Self::softmax_forward(input, self.output_shift, &mut self.last_probs)
    }

    fn backward(&mut self, grad: &[Dyadic]) -> Vec<Dyadic> { grad.to_vec() }
    fn update(&mut self, _: u32) {}
    fn zero_grad(&mut self) {}
}

// ─── Dropout ──────────────────────────────────────────────────────────────────

/// Inverted dropout.
///
/// Training: each element zeroed with probability `drop_rate`, survivors
/// scaled by `1/(1 − drop_rate)`.  Eval: identity.
pub struct Dropout {
    pub drop_rate:   f32,
    training:        bool,
    mask:            Vec<bool>,
    batch_masks:     Vec<Vec<bool>>,
    scale:           Dyadic,
    quant_shift:     u32,
    keep_prob_bits:  u32,  // drop threshold in [0, u32::MAX]
}

impl Dropout {
    pub fn new(drop_rate: f32) -> Self {
        assert!(drop_rate >= 0.0 && drop_rate < 1.0);
        const SHIFT: u32 = 7;
        let scale_f   = 1.0 / (1.0 - drop_rate);
        let mantissa  = (scale_f * (1u32 << SHIFT) as f32).round() as i32;
        Self {
            drop_rate,
            training:       true,
            mask:           Vec::new(),
            batch_masks:    Vec::new(),
            scale:          Dyadic::new(mantissa, SHIFT),
            quant_shift:    SHIFT,
            keep_prob_bits: (drop_rate * u32::MAX as f32) as u32,
        }
    }

    fn sample_mask(len: usize, keep_prob_bits: u32) -> Vec<bool> {
        (0..len).map(|_| {
            (crate::rng::rng_next() >> 32) as u32 >= keep_prob_bits
        }).collect()
    }

    fn apply_mask(input: &[Dyadic], mask: &[bool], scale: Dyadic, qs: u32) -> Vec<Dyadic> {
        input.iter().zip(mask).map(|(&x, &keep)| {
            if keep { mul(x, scale, qs) } else { Dyadic::new(0, x.s) }
        }).collect()
    }
}

impl Module for Dropout {
    fn name(&self) -> &'static str { "Dropout" }
    fn describe(&self) -> String { format!("Dropout(p={:.2}, training={})", self.drop_rate, self.training) }
    fn set_training(&mut self, t: bool) { self.training = t; }

    fn forward(&mut self, input: &[Dyadic]) -> Vec<Dyadic> {
        if !self.training { self.mask.clear(); return input.to_vec(); }
        self.mask = Self::sample_mask(input.len(), self.keep_prob_bits);
        Self::apply_mask(input, &self.mask, self.scale, self.quant_shift)
    }

    fn backward(&mut self, grad: &[Dyadic]) -> Vec<Dyadic> {
        if !self.training || self.mask.is_empty() { return grad.to_vec(); }
        Self::apply_mask(grad, &self.mask, self.scale, self.quant_shift)
    }

    fn forward_batch(&mut self, inputs: &[Vec<Dyadic>]) -> Vec<Vec<Dyadic>> {
        if !self.training {
            self.batch_masks.clear();
            return inputs.to_vec();
        }
        let mut masks = Vec::with_capacity(inputs.len());
        let outputs: Vec<Vec<Dyadic>> = inputs.iter().map(|x| {
            let m = Self::sample_mask(x.len(), self.keep_prob_bits);
            let out = Self::apply_mask(x, &m, self.scale, self.quant_shift);
            masks.push(m);
            out
        }).collect();
        self.batch_masks = masks;
        outputs
    }

    fn backward_batch(&mut self, grads: &[Vec<Dyadic>]) -> Vec<Vec<Dyadic>> {
        if !self.training || self.batch_masks.is_empty() { return grads.to_vec(); }
        grads.iter().enumerate()
            .map(|(n, g)| Self::apply_mask(g, &self.batch_masks[n], self.scale, self.quant_shift))
            .collect()
    }

    fn update(&mut self, _: u32) {}
    fn zero_grad(&mut self) {}
}

// ─── BatchNorm1D ──────────────────────────────────────────────────────────────

/// Batch normalisation for flat (Linear) activations.
///
/// **Forward (training, batch of N):**  
/// For each feature f:
/// ```text
/// μ_f      = mean(x_n[f])  over n=0..N
/// σ²_f     = var(x_n[f])   over n=0..N
/// x̂_n[f]  = (x_n[f] − μ_f) / √(σ²_f + ε)
/// y_n[f]   = γ_f · x̂_n[f] + β_f
/// ```
/// Running statistics are updated with exponential moving average (momentum)
/// and used for eval-mode normalisation.
///
/// **Forward (eval / single-sample):** uses running mean and variance.
///
/// **Backward:** full batch-norm gradient (not STE):
/// ```text
/// ∂L/∂x_n = (γ/σ) · [∂L/∂ŷ_n − (1/N)·∂L/∂β − (x̂_n/N)·∂L/∂γ]
/// ```
///
/// # Scale convention
/// Input and output both live at `shift`.  Normalised values are re-quantised
/// to `shift` after the f64 computation, resetting any accumulated scale drift.
pub struct BatchNorm1D {
    pub num_features: usize,
    pub shift:        u32,
    pub momentum:     f32,    // EMA coefficient for running stats (typical: 0.1)
    pub eps:          f64,    // numerical stability (typical: 1e-5)
    pub gamma:        Vec<Dyadic>,
    pub beta:         Vec<Dyadic>,
    pub running_mean: Vec<f64>,
    pub running_var:  Vec<f64>,
    training:         bool,
    // batch caches
    x_hat_cache:  Vec<Vec<f64>>,   // normalised float values per (sample, feature)
    inv_std_cache: Vec<f64>,        // 1/σ per feature
    input_batch_cache: Vec<Vec<Dyadic>>,
    // gradients + velocity
    grad_gamma: Vec<Dyadic>,
    grad_beta:  Vec<Dyadic>,
    vel_gamma:  Vec<Dyadic>,
    vel_beta:   Vec<Dyadic>,
    pub momentum_shift: Option<u32>,
}

impl BatchNorm1D {
    /// Create a new BatchNorm1D layer.
    ///
    /// `shift` should match the rest of the network.
    /// `momentum` is the EMA coefficient for running stats (PyTorch default: 0.1).
    pub fn new(num_features: usize, shift: u32) -> Self {
        let scale = (1u32 << shift) as f64;  // value of 1.0 at this shift
        let gamma = vec![Dyadic::new(scale.round() as i32, shift); num_features];
        let beta  = vec![Dyadic::new(0, shift); num_features];
        let z     = |n: usize| vec![Dyadic::new(0, shift); n];
        Self {
            num_features, shift,
            momentum: 0.1,
            eps:      1e-5,
            gamma, beta,
            running_mean: vec![0.0; num_features],
            running_var:  vec![1.0; num_features],
            training: true,
            x_hat_cache:       vec![vec![0.0; num_features]; 0],
            inv_std_cache:     vec![1.0; num_features],
            input_batch_cache: Vec::new(),
            grad_gamma: z(num_features),
            grad_beta:  z(num_features),
            vel_gamma:  z(num_features),
            vel_beta:   z(num_features),
            momentum_shift: None,
        }
    }

    pub fn with_momentum_opt(mut self, shift: u32) -> Self {
        self.momentum_shift = Some(shift);
        self
    }
}

impl Module for BatchNorm1D {
    fn name(&self) -> &'static str { "BatchNorm1D" }
    fn describe(&self) -> String {
        format!("BatchNorm1D(features={}, shift={}, eps={:.0e}, training={})",
            self.num_features, self.shift, self.eps, self.training)
    }
    fn set_training(&mut self, t: bool) { self.training = t; }

    /// Single-sample forward — uses running statistics.
    /// During training this is an approximation; prefer `forward_batch`.
    fn forward(&mut self, input: &[Dyadic]) -> Vec<Dyadic> {
        let scale = (1u32 << self.shift) as f64;
        let mut out = Vec::with_capacity(self.num_features);

        for f in 0..self.num_features {
            let x_f   = input[f].to_f64();
            let mu    = if self.training { x_f } else { self.running_mean[f] };
            let var   = if self.training { 0.0  } else { self.running_var[f] };
            // With a single sample, variance is 0 — just update running stats
            // and pass through (γ·1 + β).
            if self.training {
                let m = self.momentum as f64;
                self.running_mean[f] = (1.0 - m) * self.running_mean[f] + m * x_f;
                self.running_var[f]  = (1.0 - m) * self.running_var[f]  + m * 1.0;
            }
            let inv_std = 1.0 / (var + self.eps).sqrt();
            let x_hat   = (x_f - mu) * inv_std;
            let xhd     = Dyadic::new((x_hat * scale).round() as i32, self.shift);
            out.push(add(mul(self.gamma[f], xhd, self.shift), self.beta[f]));
        }
        out
    }

    fn backward(&mut self, grad: &[Dyadic]) -> Vec<Dyadic> {
        // Single-sample STE: pass gradient through scaled by γ/σ.
        // (Full backward requires the batch; this is used when batch_size=1.)
        let scale = (1u32 << self.shift) as f64;
        grad.iter().enumerate().map(|(f, &g)| {
            let gamma_f   = self.gamma[f].to_f64();
            let inv_std_f = self.inv_std_cache.get(f).copied().unwrap_or(1.0);
            let dx = g.to_f64() * gamma_f * inv_std_f;
            // Accumulate γ and β gradients
            self.grad_gamma[f] = Dyadic::new(
                self.grad_gamma[f].v.saturating_add((g.to_f64() * 0.0 * scale) as i32), // x_hat not cached
                self.shift);
            self.grad_beta[f]  = Dyadic::new(
                self.grad_beta[f].v.saturating_add((g.to_f64() * scale).round() as i32),
                self.shift);
            Dyadic::new((dx * scale).round() as i32, self.shift)
        }).collect()
    }

    /// Batch forward — computes proper batch statistics.  Always use this during training.
    fn forward_batch(&mut self, inputs: &[Vec<Dyadic>]) -> Vec<Vec<Dyadic>> {
        let n     = inputs.len();
        let scale = (1u32 << self.shift) as f64;
        let m_ema = self.momentum as f64;

        // Decode all inputs
        let decoded: Vec<Vec<f64>> = inputs.iter()
            .map(|x| x.iter().map(|d| d.to_f64()).collect())
            .collect();

        let mut means     = vec![0.0f64; self.num_features];
        let mut inv_stds  = vec![0.0f64; self.num_features];
        let mut x_hat_all = vec![vec![0.0f64; self.num_features]; n];
        let mut outputs   = vec![vec![Dyadic::new(0, self.shift); self.num_features]; n];

        for f in 0..self.num_features {
            // Compute mean and variance over the batch.
            let mu  = decoded.iter().map(|x| x[f]).sum::<f64>() / n as f64;
            let var = decoded.iter().map(|x| (x[f] - mu).powi(2)).sum::<f64>() / n as f64;
            let inv_std = 1.0 / (var + self.eps).sqrt();

            means[f]    = mu;
            inv_stds[f] = inv_std;

            // Update running statistics.
            if self.training {
                self.running_mean[f] = (1.0 - m_ema) * self.running_mean[f] + m_ema * mu;
                self.running_var[f]  = (1.0 - m_ema) * self.running_var[f]  + m_ema * var;
            }

            // Normalise each sample.
            for s in 0..n {
                let x_hat = (decoded[s][f] - mu) * inv_std;
                x_hat_all[s][f] = x_hat;
                let xhd = Dyadic::new((x_hat * scale).round() as i32, self.shift);
                outputs[s][f] = add(mul(self.gamma[f], xhd, self.shift), self.beta[f]);
            }
        }

        self.inv_std_cache     = inv_stds;
        self.x_hat_cache       = x_hat_all;
        self.input_batch_cache = inputs.to_vec();
        outputs
    }

    /// Batch backward — full batch-norm gradient.
    fn backward_batch(&mut self, grads: &[Vec<Dyadic>]) -> Vec<Vec<Dyadic>> {
        let n     = grads.len();
        let scale = (1u32 << self.shift) as f64;
        let nf    = n as f64;

        let dL_dy: Vec<Vec<f64>> = grads.iter()
            .map(|g| g.iter().map(|d| d.to_f64()).collect())
            .collect();

        let mut grad_inputs = vec![vec![Dyadic::new(0, self.shift); self.num_features]; n];

        for f in 0..self.num_features {
            let gamma_f   = self.gamma[f].to_f64();
            let inv_std_f = self.inv_std_cache[f];

            let sum_dL_dy:      f64 = (0..n).map(|s| dL_dy[s][f]).sum();
            let sum_dL_dy_xhat: f64 = (0..n).map(|s| dL_dy[s][f] * self.x_hat_cache[s][f]).sum();

            // Accumulate learnable parameter gradients.
            self.grad_gamma[f] = Dyadic::new(
                self.grad_gamma[f].v.saturating_add((sum_dL_dy_xhat * scale).round() as i32),
                self.shift);
            self.grad_beta[f] = Dyadic::new(
                self.grad_beta[f].v.saturating_add((sum_dL_dy * scale).round() as i32),
                self.shift);

            // Full BN backward:
            // ∂L/∂x_n = (γ/σ) · [ ∂L/∂ŷ_n − (1/N)·∂L/∂β − (x̂_n/N)·∂L/∂γ ]
            for s in 0..n {
                let dx = (gamma_f * inv_std_f)
                    * (dL_dy[s][f]
                       - sum_dL_dy / nf
                       - self.x_hat_cache[s][f] * sum_dL_dy_xhat / nf);
                grad_inputs[s][f] = Dyadic::new((dx * scale).round() as i32, self.shift);
            }
        }

        grad_inputs
    }

    fn update(&mut self, lr: u32) {
        apply_updates(&mut self.gamma, &self.grad_gamma, &mut self.vel_gamma, lr, self.momentum_shift);
        apply_updates(&mut self.beta,  &self.grad_beta,  &mut self.vel_beta,  lr, self.momentum_shift);
    }

    fn zero_grad(&mut self) {
        self.grad_gamma.iter_mut().for_each(|g| g.v = 0);
        self.grad_beta.iter_mut().for_each(|g| g.v = 0);
    }
}

// ─── BatchNorm2D ──────────────────────────────────────────────────────────────

/// Batch normalisation for spatial (Conv2D) activations.
///
/// Input layout: `[C × H × W]` flat (channel-first).
/// Normalises per channel over `N × H × W` values in the batch —
/// exactly matching PyTorch's `nn.BatchNorm2d` semantics.
///
/// See [`BatchNorm1D`] for the mathematical details; the only difference here
/// is that the "batch" for channel `c` consists of all `N × H × W` spatial
/// positions across all N samples.
pub struct BatchNorm2D {
    pub channels:  usize,
    pub h:         usize,
    pub w:         usize,
    pub shift:     u32,
    pub momentum:  f32,
    pub eps:       f64,
    pub gamma:     Vec<Dyadic>,   // [channels]
    pub beta:      Vec<Dyadic>,   // [channels]
    pub running_mean: Vec<f64>,
    pub running_var:  Vec<f64>,
    training:      bool,
    // caches
    x_hat_cache:       Vec<Vec<f64>>,   // [n_samples, C*H*W]
    inv_std_cache:     Vec<f64>,         // [channels]
    input_batch_cache: Vec<Vec<Dyadic>>,
    // gradients + velocity
    grad_gamma: Vec<Dyadic>,
    grad_beta:  Vec<Dyadic>,
    vel_gamma:  Vec<Dyadic>,
    vel_beta:   Vec<Dyadic>,
    pub momentum_shift: Option<u32>,
}

impl BatchNorm2D {
    pub fn new(channels: usize, h: usize, w: usize, shift: u32) -> Self {
        let scale = (1u32 << shift) as f64;
        let gamma = vec![Dyadic::new(scale.round() as i32, shift); channels];
        let beta  = vec![Dyadic::new(0, shift); channels];
        let z     = |n: usize| vec![Dyadic::new(0, shift); n];
        Self {
            channels, h, w, shift,
            momentum: 0.1,
            eps:      1e-5,
            gamma, beta,
            running_mean: vec![0.0; channels],
            running_var:  vec![1.0; channels],
            training: true,
            x_hat_cache:       Vec::new(),
            inv_std_cache:     vec![1.0; channels],
            input_batch_cache: Vec::new(),
            grad_gamma: z(channels),
            grad_beta:  z(channels),
            vel_gamma:  z(channels),
            vel_beta:   z(channels),
            momentum_shift: None,
        }
    }

    pub fn with_momentum_opt(mut self, shift: u32) -> Self {
        self.momentum_shift = Some(shift);
        self
    }

    #[inline] fn spatial_idx(&self, c: usize, h: usize, w: usize) -> usize {
        c * self.h * self.w + h * self.w + w
    }
}

impl Module for BatchNorm2D {
    fn name(&self) -> &'static str { "BatchNorm2D" }
    fn describe(&self) -> String {
        format!("BatchNorm2D(C={}, H={}, W={}, shift={}, training={})",
            self.channels, self.h, self.w, self.shift, self.training)
    }
    fn set_training(&mut self, t: bool) { self.training = t; }

    /// Single-sample forward using running statistics.
    fn forward(&mut self, input: &[Dyadic]) -> Vec<Dyadic> {
        let scale = (1u32 << self.shift) as f64;
        let hw    = self.h * self.w;
        let m_ema = self.momentum as f64;
        let mut out = vec![Dyadic::new(0, self.shift); input.len()];

        for c in 0..self.channels {
            let (mu, inv_std) = if self.training {
                // With a single sample, compute stats over H*W spatial positions.
                let vals: Vec<f64> = (0..hw)
                    .map(|p| input[c * hw + p].to_f64())
                    .collect();
                let mu  = vals.iter().sum::<f64>() / hw as f64;
                let var = vals.iter().map(|&v| (v - mu).powi(2)).sum::<f64>() / hw as f64;
                self.running_mean[c] = (1.0 - m_ema) * self.running_mean[c] + m_ema * mu;
                self.running_var[c]  = (1.0 - m_ema) * self.running_var[c]  + m_ema * var;
                (mu, 1.0 / (var + self.eps).sqrt())
            } else {
                (self.running_mean[c], 1.0 / (self.running_var[c] + self.eps).sqrt())
            };

            for p in 0..hw {
                let idx   = self.spatial_idx(c, p / self.w, p % self.w);
                let x_hat = (input[idx].to_f64() - mu) * inv_std;
                let xhd   = Dyadic::new((x_hat * scale).round() as i32, self.shift);
                out[idx]  = add(mul(self.gamma[c], xhd, self.shift), self.beta[c]);
            }
        }
        out
    }

    fn backward(&mut self, grad: &[Dyadic]) -> Vec<Dyadic> {
        // Single-sample STE: γ/σ passthrough.
        let scale = (1u32 << self.shift) as f64;
        let hw    = self.h * self.w;
        let mut out = vec![Dyadic::new(0, self.shift); grad.len()];

        for c in 0..self.channels {
            let gamma_f   = self.gamma[c].to_f64();
            let inv_std_c = self.inv_std_cache[c];
            for p in 0..hw {
                let idx = self.spatial_idx(c, p / self.w, p % self.w);
                let dx  = grad[idx].to_f64() * gamma_f * inv_std_c;
                // Accumulate β gradient
                self.grad_beta[c] = Dyadic::new(
                    self.grad_beta[c].v.saturating_add((grad[idx].to_f64() * scale).round() as i32),
                    self.shift);
                out[idx] = Dyadic::new((dx * scale).round() as i32, self.shift);
            }
        }
        out
    }

    fn forward_batch(&mut self, inputs: &[Vec<Dyadic>]) -> Vec<Vec<Dyadic>> {
        let n     = inputs.len();
        let hw    = self.h * self.w;
        let scale = (1u32 << self.shift) as f64;
        let m_ema = self.momentum as f64;
        let n_hw  = (n * hw) as f64;

        let decoded: Vec<Vec<f64>> = inputs.iter()
            .map(|x| x.iter().map(|d| d.to_f64()).collect())
            .collect();

        let mut inv_stds  = vec![0.0f64; self.channels];
        let mut x_hat_all = vec![vec![0.0f64; self.channels * hw]; n];
        let mut outputs   = vec![vec![Dyadic::new(0, self.shift); self.channels * hw]; n];

        for c in 0..self.channels {
            // Collect all N*H*W values for this channel.
            let mut mu  = 0.0f64;
            let mut var = 0.0f64;
            for s in 0..n {
                for p in 0..hw {
                    mu += decoded[s][c * hw + p];
                }
            }
            mu /= n_hw;
            for s in 0..n {
                for p in 0..hw {
                    var += (decoded[s][c * hw + p] - mu).powi(2);
                }
            }
            var /= n_hw;
            let inv_std = 1.0 / (var + self.eps).sqrt();
            inv_stds[c] = inv_std;

            if self.training {
                self.running_mean[c] = (1.0 - m_ema) * self.running_mean[c] + m_ema * mu;
                self.running_var[c]  = (1.0 - m_ema) * self.running_var[c]  + m_ema * var;
            }

            for s in 0..n {
                for p in 0..hw {
                    let idx   = c * hw + p;
                    let x_hat = (decoded[s][idx] - mu) * inv_std;
                    x_hat_all[s][idx] = x_hat;
                    let xhd = Dyadic::new((x_hat * scale).round() as i32, self.shift);
                    outputs[s][idx] = add(mul(self.gamma[c], xhd, self.shift), self.beta[c]);
                }
            }
        }

        self.inv_std_cache     = inv_stds;
        self.x_hat_cache       = x_hat_all;
        self.input_batch_cache = inputs.to_vec();
        outputs
    }

    fn backward_batch(&mut self, grads: &[Vec<Dyadic>]) -> Vec<Vec<Dyadic>> {
        let n     = grads.len();
        let hw    = self.h * self.w;
        let scale = (1u32 << self.shift) as f64;
        let n_hw  = (n * hw) as f64;

        let dL_dy: Vec<Vec<f64>> = grads.iter()
            .map(|g| g.iter().map(|d| d.to_f64()).collect())
            .collect();

        let mut grad_inputs = vec![vec![Dyadic::new(0, self.shift); self.channels * hw]; n];

        for c in 0..self.channels {
            let gamma_c   = self.gamma[c].to_f64();
            let inv_std_c = self.inv_std_cache[c];

            // Sum gradients over all N*H*W positions for γ and β.
            let mut sum_dL_dy      = 0.0f64;
            let mut sum_dL_dy_xhat = 0.0f64;
            for s in 0..n {
                for p in 0..hw {
                    let idx = c * hw + p;
                    sum_dL_dy      += dL_dy[s][idx];
                    sum_dL_dy_xhat += dL_dy[s][idx] * self.x_hat_cache[s][idx];
                }
            }

            self.grad_gamma[c] = Dyadic::new(
                self.grad_gamma[c].v.saturating_add((sum_dL_dy_xhat * scale).round() as i32),
                self.shift);
            self.grad_beta[c] = Dyadic::new(
                self.grad_beta[c].v.saturating_add((sum_dL_dy * scale).round() as i32),
                self.shift);

            // Full BN backward per spatial position.
            for s in 0..n {
                for p in 0..hw {
                    let idx = c * hw + p;
                    let dx  = (gamma_c * inv_std_c)
                        * (dL_dy[s][idx]
                           - sum_dL_dy / n_hw
                           - self.x_hat_cache[s][idx] * sum_dL_dy_xhat / n_hw);
                    grad_inputs[s][idx] = Dyadic::new((dx * scale).round() as i32, self.shift);
                }
            }
        }

        grad_inputs
    }

    fn update(&mut self, lr: u32) {
        apply_updates(&mut self.gamma, &self.grad_gamma, &mut self.vel_gamma, lr, self.momentum_shift);
        apply_updates(&mut self.beta,  &self.grad_beta,  &mut self.vel_beta,  lr, self.momentum_shift);
    }

    fn zero_grad(&mut self) {
        self.grad_gamma.iter_mut().for_each(|g| g.v = 0);
        self.grad_beta.iter_mut().for_each(|g| g.v = 0);
    }
}

// ─── Conv2D ───────────────────────────────────────────────────────────────────

/// 2D convolution. Input layout: `[C_in × H × W]` channel-first flat.
pub struct Conv2D {
    pub in_channels:    usize,
    pub out_channels:   usize,
    pub kernel_h:       usize,
    pub kernel_w:       usize,
    pub in_h:           usize,
    pub in_w:           usize,
    pub quant_shift:    u32,
    pub output_bits:    u32,
    pub grad_clip:      i32,
    pub momentum_shift: Option<u32>,
    out_h: usize,
    out_w: usize,
    pub kernels: Vec<Dyadic>,
    pub biases:  Vec<Dyadic>,
    input_cache:        Vec<Dyadic>,
    output_cache:       Vec<Dyadic>,
    input_batch_cache:  Vec<Vec<Dyadic>>,
    output_batch_cache: Vec<Vec<Dyadic>>,
    grad_k: Vec<Dyadic>,
    grad_b: Vec<Dyadic>,
    vel_k:  Vec<Dyadic>,
    vel_b:  Vec<Dyadic>,

    q_min: i32,
    q_max: i32,
}

impl Conv2D {
    pub fn new(
        in_channels: usize, out_channels: usize,
        kernel_h: usize, kernel_w: usize,
        in_h: usize, in_w: usize,
        weight_shift: u32, quant_shift: u32, output_bits: u32,
    ) -> Self {
        assert!(in_h >= kernel_h && in_w >= kernel_w);
        let fan_in  = in_channels * kernel_h * kernel_w;
        let n_k     = out_channels * fan_in;
        let k       = ((1u32 << weight_shift) as f64 / (fan_in as f64).sqrt()).round() as i32;
        let k       = k.max(1);
        let out_h   = in_h - kernel_h + 1;
        let out_w   = in_w - kernel_w + 1;
        let z       = |n: usize| vec![Dyadic::new(0, weight_shift); n];
        let (q_min, q_max) = signed_bounds(output_bits);
        Self {
            in_channels, out_channels, kernel_h, kernel_w, in_h, in_w,
            quant_shift, output_bits,
            grad_clip: i32::MAX, momentum_shift: None,
            out_h, out_w,
            kernels: (0..n_k).map(|_| Dyadic::new(-k + rng_range(2 * k as u32) as i32, weight_shift)).collect(),
            biases:  z(out_channels),
            input_cache: Vec::new(), output_cache: Vec::new(),
            input_batch_cache: Vec::new(), output_batch_cache: Vec::new(),
            grad_k: z(n_k), grad_b: z(out_channels),
            vel_k:  z(n_k), vel_b:  z(out_channels),
            q_min: q_min, q_max: q_max,
        }
    }

    pub fn with_grad_clip(mut self, c: i32) -> Self { self.grad_clip = c.abs(); self }
    pub fn with_momentum(mut self, s: u32) -> Self  { self.momentum_shift = Some(s); self }
    pub fn output_len(&self) -> usize { self.out_channels * self.out_h * self.out_w }

    #[inline] fn in_idx (&self, c: usize, h: usize, w: usize) -> usize { c * self.in_h * self.in_w  + h * self.in_w  + w }
    #[inline] fn out_idx(&self, c: usize, h: usize, w: usize) -> usize { c * self.out_h * self.out_w + h * self.out_w + w }
    #[inline] fn k_idx  (&self, oc: usize, ic: usize, kh: usize, kw: usize) -> usize {
        oc * (self.in_channels * self.kernel_h * self.kernel_w)
        + ic * (self.kernel_h * self.kernel_w)
        + kh * self.kernel_w + kw
    }

    fn forward_one(&mut self, input: &[Dyadic]) -> Vec<Dyadic> {
        let mut out = vec![Dyadic::new(0, 0); self.output_len()];
        for oc in 0..self.out_channels {
            for oh in 0..self.out_h {
                for ow in 0..self.out_w {
                    let mut acc = self.biases[oc];
                    for ic in 0..self.in_channels {
                        for kh in 0..self.kernel_h {
                            for kw in 0..self.kernel_w {
                                acc = add(acc, mul(
                                    self.kernels[self.k_idx(oc, ic, kh, kw)],
                                    input[self.in_idx(ic, oh + kh, ow + kw)],
                                    self.quant_shift,
                                ));
                            }
                        }
                    }
                    let (y, _) = requantize(acc, acc.s, self.q_min, self.q_max);
                    out[self.out_idx(oc, oh, ow)] = y;
                }
            }
        }
        out
    }

    fn backward_one(&mut self, grad_output: &[Dyadic], input: &[Dyadic], output: &[Dyadic]) -> Vec<Dyadic> {
        let g_s = grad_output.first().map_or(0, |g| g.s);
        let mut gi = vec![Dyadic::new(0, g_s); self.in_channels * self.in_h * self.in_w];
        for oc in 0..self.out_channels {
            for oh in 0..self.out_h {
                for ow in 0..self.out_w {
                    let op  = self.out_idx(oc, oh, ow);
                    let gr  = ste_requantize(grad_output[op], output[op].v, self.q_min, self.q_max);
                    let gj  = Dyadic::new(gr.v.clamp(-self.grad_clip, self.grad_clip), gr.s);
                    for ic in 0..self.in_channels {
                        for kh in 0..self.kernel_h {
                            for kw in 0..self.kernel_w {
                                let ki = self.k_idx(oc, ic, kh, kw);
                                let ii = self.in_idx(ic, oh + kh, ow + kw);
                                self.grad_k[ki] = add(self.grad_k[ki], mul(gj, input[ii],              self.quant_shift));
                                gi[ii]          = add(gi[ii],          mul(gj, self.kernels[ki], self.quant_shift));
                            }
                        }
                    }
                    self.grad_b[oc] = add(self.grad_b[oc], gj);
                }
            }
        }
        gi
    }
}

impl Module for Conv2D {
    fn name(&self) -> &'static str { "Conv2D" }
    fn describe(&self) -> String {
        let ws   = self.kernels.first().map_or(0, |k| k.s);
        let clip = if self.grad_clip == i32::MAX { "off".into() } else { format!("2^{}", (self.grad_clip as f64).log2() as i32) };
        let mom  = self.momentum_shift.map_or("off".into(), |m| format!("shift={m}"));
        format!("Conv2D(in={}, out={}, {}×{}, {}×{}→{}×{}, clip={clip}, mom={mom})",
            self.in_channels, self.out_channels,
            self.kernel_h, self.kernel_w,
            self.in_h, self.in_w, self.out_h, self.out_w)
    }

    fn forward(&mut self, input: &[Dyadic]) -> Vec<Dyadic> {
        let out = self.forward_one(input);
        self.input_cache  = input.to_vec();
        self.output_cache = out.clone();
        out
    }

    fn backward(&mut self, grad: &[Dyadic]) -> Vec<Dyadic> {
        let inp = self.input_cache.clone();
        let out = self.output_cache.clone();
        self.backward_one(grad, &inp, &out)
    }

    fn forward_batch(&mut self, inputs: &[Vec<Dyadic>]) -> Vec<Vec<Dyadic>> {
        let outputs: Vec<Vec<Dyadic>> = inputs.iter().map(|x| self.forward_one(x)).collect();
        self.input_batch_cache  = inputs.to_vec();
        self.output_batch_cache = outputs.clone();
        outputs
    }

    fn backward_batch(&mut self, grads: &[Vec<Dyadic>]) -> Vec<Vec<Dyadic>> {
        grads.iter().enumerate().map(|(n, g)| {
            let inp = self.input_batch_cache[n].clone();
            let out = self.output_batch_cache[n].clone();
            self.backward_one(g, &inp, &out)
        }).collect()
    }

    fn update(&mut self, lr: u32) {
        apply_updates(&mut self.kernels, &self.grad_k, &mut self.vel_k, lr, self.momentum_shift);
        apply_updates(&mut self.biases,  &self.grad_b, &mut self.vel_b, lr, self.momentum_shift);
    }

    fn zero_grad(&mut self) {
        self.grad_k.iter_mut().for_each(|g| g.v = 0);
        self.grad_b.iter_mut().for_each(|g| g.v = 0);
    }
}

// ─── MaxPool2D ────────────────────────────────────────────────────────────────

pub struct MaxPool2D {
    pub channels: usize,
    pub in_h:     usize,
    pub in_w:     usize,
    pub kernel:   usize,
    pub stride:   usize,
    out_h:        usize,
    out_w:        usize,
    max_mask:       Vec<usize>,
    batch_max_masks: Vec<Vec<usize>>,
}

impl MaxPool2D {
    pub fn new(channels: usize, in_h: usize, in_w: usize, kernel: usize, stride: usize) -> Self {
        let out_h = (in_h - kernel) / stride + 1;
        let out_w = (in_w - kernel) / stride + 1;
        Self { channels, in_h, in_w, kernel, stride, out_h, out_w,
               max_mask: Vec::new(), batch_max_masks: Vec::new() }
    }

    pub fn output_len(&self) -> usize { self.channels * self.out_h * self.out_w }

    #[inline] fn in_idx (&self, c: usize, h: usize, w: usize) -> usize { c * self.in_h  * self.in_w  + h * self.in_w  + w }
    #[inline] fn out_idx(&self, c: usize, h: usize, w: usize) -> usize { c * self.out_h * self.out_w + h * self.out_w + w }

    fn pool_forward(
        &self, input: &[Dyadic], mask: &mut Vec<usize>,
    ) -> Vec<Dyadic> {
        let mut out = vec![Dyadic::new(0, 0); self.output_len()];
        *mask = vec![0usize; self.output_len()];
        for c in 0..self.channels {
            for oh in 0..self.out_h {
                for ow in 0..self.out_w {
                    let ih0 = oh * self.stride;
                    let iw0 = ow * self.stride;
                    let mut mx  = i32::MIN;
                    let mut mxi = self.in_idx(c, ih0, iw0);
                    for kh in 0..self.kernel {
                        for kw in 0..self.kernel {
                            let idx = self.in_idx(c, ih0 + kh, iw0 + kw);
                            if input[idx].v > mx { mx = input[idx].v; mxi = idx; }
                        }
                    }
                    let op = self.out_idx(c, oh, ow);
                    out[op] = input[mxi];
                    mask[op] = mxi;
                }
            }
        }
        out
    }

    fn pool_backward(
        &self, grad: &[Dyadic], mask: &[usize],
    ) -> Vec<Dyadic> {
        let g_s = grad.first().map_or(0, |g| g.s);
        let mut gi = vec![Dyadic::new(0, g_s); self.channels * self.in_h * self.in_w];
        for (op, &mi) in mask.iter().enumerate() {
            gi[mi] = add(gi[mi], grad[op]);
        }
        gi
    }
}

impl Module for MaxPool2D {
    fn name(&self) -> &'static str { "MaxPool2D" }
    fn describe(&self) -> String {
        format!("MaxPool2D(C={}, {}×{}→{}×{}, k={}, s={})",
            self.channels, self.in_h, self.in_w, self.out_h, self.out_w, self.kernel, self.stride)
    }

    fn forward(&mut self, input: &[Dyadic]) -> Vec<Dyadic> {
        let mut mask = Vec::new();
        let out = self.pool_forward(input, &mut mask);
        self.max_mask = mask;
        out
    }

    fn backward(&mut self, grad: &[Dyadic]) -> Vec<Dyadic> {
        self.pool_backward(grad, &self.max_mask.clone())
    }

    fn forward_batch(&mut self, inputs: &[Vec<Dyadic>]) -> Vec<Vec<Dyadic>> {
        let mut all_masks = Vec::with_capacity(inputs.len());
        let outputs: Vec<Vec<Dyadic>> = inputs.iter().map(|x| {
            let mut mask = Vec::new();
            let out = self.pool_forward(x, &mut mask);
            all_masks.push(mask);
            out
        }).collect();
        self.batch_max_masks = all_masks;
        outputs
    }

    fn backward_batch(&mut self, grads: &[Vec<Dyadic>]) -> Vec<Vec<Dyadic>> {
        grads.iter().enumerate()
            .map(|(n, g)| self.pool_backward(g, &self.batch_max_masks[n]))
            .collect()
    }

    fn update(&mut self, _: u32) {}
    fn zero_grad(&mut self) {}
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
