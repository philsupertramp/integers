//! Neural network layers built on the [`crate::dyadic`] arithmetic system.
//!
//! # Layer catalogue
//!
//! | Layer         | Parameters | Use for |
//! |---------------|-----------|---------|
//! | [`Linear`]    | weights, biases | fully-connected |
//! | [`ReLU`]      | — | activation |
//! | [`Softmax`]   | — | output probabilities |
//! | [`Conv2D`]    | kernels, biases | spatial feature extraction |
//! | [`MaxPool2D`] | — | spatial downsampling |
//! | [`Flatten`]   | — | bridge conv→linear |
//!
//! # Implementing a new layer
//!
//! Implement the [`Layer`] trait:
//! - `forward`  — compute output, **cache** everything `backward` needs.
//! - `backward` — *accumulate* `∂L/∂params`, return `∂L/∂input`.
//! - `update`   — one optimizer step: `params -= lr * velocity`.
//! - `zero_grad`— reset accumulated gradients (NOT velocity/momentum).
//! - `name`     — `&'static str` label.
//! - `describe` — richer one-line summary (default = `name()`).
//!
//! Then `model.add(your_layer)` into a [`Sequential`].
//!
//! # Shift convention
//!
//! All built-in layers work with a uniform `SHIFT` so that output scale = input scale:
//! `mul(w, x, SHIFT)` with `w.s = x.s = SHIFT` → `result.s = SHIFT`.
//!
//! # Momentum SGD
//!
//! Enable on `Linear` and `Conv2D` via `.with_momentum(shift)`.  With `shift = 1`:
//! ```text
//! v  ←  SR(v, shift)  +  grad_accum     (decay old velocity, add new)
//! w  ←  w  −  SR(v,  lr_shift)          (apply damped velocity)
//! ```

use crate::dyadic::{
    add, mul, requantize, signed_bounds, ste_requantize, stochastic_round, Dyadic,
};
use rand::Rng;

// ─── Layer trait ──────────────────────────────────────────────────────────────

pub trait Layer {
    fn forward(&mut self, input: &[Dyadic]) -> Vec<Dyadic>;
    fn backward(&mut self, grad_output: &[Dyadic]) -> Vec<Dyadic>;
    fn update(&mut self, lr_shift: u32);
    fn zero_grad(&mut self);
    fn name(&self) -> &'static str;
    fn describe(&self) -> String { self.name().to_string() }
}

// ─── Sequential ───────────────────────────────────────────────────────────────

pub struct Sequential {
    pub layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    pub fn new() -> Self { Self { layers: Vec::new() } }

    pub fn add<L: Layer + 'static>(&mut self, layer: L) {
        self.layers.push(Box::new(layer));
    }

    pub fn forward(&mut self, input: &[Dyadic]) -> Vec<Dyadic> {
        let mut x = input.to_vec();
        for layer in &mut self.layers { x = layer.forward(&x); }
        x
    }

    pub fn backward(&mut self, grad: &[Dyadic]) -> Vec<Dyadic> {
        let mut g = grad.to_vec();
        for layer in self.layers.iter_mut().rev() { g = layer.backward(&g); }
        g
    }

    pub fn update(&mut self, lr_shift: u32) {
        for layer in &mut self.layers { layer.update(lr_shift); }
    }

    pub fn zero_grad(&mut self) {
        for layer in &mut self.layers { layer.zero_grad(); }
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

// ─── Shared optimizer helpers ──────────────────────────────────────────────────

/// Vanilla SGD: `w ← w − 2^(−lr_shift) · g`.
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

/// Momentum SGD: `v ← SR(v, m) + g;  w ← w − SR(v, lr_shift)`.
fn momentum_step(w: Dyadic, g: Dyadic, v: &mut Dyadic, lr_shift: u32, m: u32) -> Dyadic {
    let v_decayed = Dyadic::new(stochastic_round(v.v, m), v.s);
    let (v_s, v_new) = if g.s >= v.s {
        let g_aligned = Dyadic::new(stochastic_round(g.v, g.s - v.s), v.s);
        (v.s, Dyadic::new(v_decayed.v.saturating_add(g_aligned.v), v.s))
    } else {
        let shift = v.s - g.s;
        let vd_aligned = Dyadic::new(stochastic_round(v_decayed.v, shift), g.s);
        (g.s, Dyadic::new(vd_aligned.v.saturating_add(g.v), g.s))
    };
    *v = Dyadic::new(v_new.v, v_s);
    sgd_step(w, *v, lr_shift)
}

/// Apply one optimizer step to a slice of (param, grad, velocity) triples.
fn apply_updates(
    params: &mut [Dyadic],
    grads:  &[Dyadic],
    vels:   &mut [Dyadic],
    lr_shift: u32,
    momentum: Option<u32>,
) {
    match momentum {
        None    => params.iter_mut().zip(grads).for_each(|(w, &g)| *w = sgd_step(*w, g, lr_shift)),
        Some(m) => params.iter_mut().zip(grads).zip(vels.iter_mut())
            .for_each(|((w, &g), v)| *w = momentum_step(*w, g, v, lr_shift, m)),
    }
}

// ─── Linear ───────────────────────────────────────────────────────────────────

/// Fully-connected affine layer with optional momentum SGD and gradient clipping.
pub struct Linear {
    pub in_features:  usize,
    pub out_features: usize,
    pub weights: Vec<Dyadic>,
    pub biases:  Vec<Dyadic>,
    pub quant_shift:    u32,
    pub output_bits:    u32,
    pub grad_clip:      i32,
    pub momentum_shift: Option<u32>,
    input_cache:  Vec<Dyadic>,
    output_cache: Vec<Dyadic>,
    grad_w: Vec<Dyadic>,
    grad_b: Vec<Dyadic>,
    vel_w:  Vec<Dyadic>,
    vel_b:  Vec<Dyadic>,
}

impl Linear {
    /// Construct with He-style initialisation: `W ~ U(−k, k)`,
    /// `k = ⌊2^weight_shift / √fan_in⌋` (min 1).
    pub fn new(in_features: usize, out_features: usize, weight_shift: u32, quant_shift: u32, output_bits: u32) -> Self {
        let mut rng = rand::thread_rng();
        let n_w = out_features * in_features;
        let k   = ((1u32 << weight_shift) as f64 / (in_features as f64).sqrt()).round() as i32;
        let k   = k.max(1);
        let mk  = |n: usize| vec![Dyadic::new(0, weight_shift); n];

        Self {
            in_features, out_features,
            weights: (0..n_w).map(|_| Dyadic::new(rng.gen_range(-k..=k), weight_shift)).collect(),
            biases:  mk(out_features),
            quant_shift, output_bits,
            grad_clip:      i32::MAX,
            momentum_shift: None,
            input_cache:  Vec::new(),
            output_cache: Vec::new(),
            grad_w: mk(n_w),
            grad_b: mk(out_features),
            vel_w:  mk(n_w),
            vel_b:  mk(out_features),
        }
    }

    pub fn with_grad_clip(mut self, clip: i32) -> Self { self.grad_clip = clip.abs(); self }
    pub fn with_momentum(mut self, shift: u32) -> Self { self.momentum_shift = Some(shift); self }

    #[inline] fn w   (&self, j: usize, i: usize) -> Dyadic { self.weights[j * self.in_features + i] }
    #[inline] fn flat(&self, j: usize, i: usize) -> usize  { j * self.in_features + i }
}

impl Layer for Linear {
    fn name(&self) -> &'static str { "Linear" }
    fn describe(&self) -> String {
        let ws    = self.weights.first().map_or(0, |w| w.s);
        let clip  = if self.grad_clip == i32::MAX { "off".into() } else { format!("2^{}", (self.grad_clip as f64).log2() as i32) };
        let mom   = self.momentum_shift.map_or("off".into(), |m| format!("shift={m}"));
        format!("Linear(in={}, out={}, w_shift={ws}, q_shift={}, bits={}, clip={clip}, mom={mom})",
            self.in_features, self.out_features, self.quant_shift, self.output_bits)
    }

    fn forward(&mut self, input: &[Dyadic]) -> Vec<Dyadic> {
        assert_eq!(input.len(), self.in_features);
        self.input_cache = input.to_vec();
        let (q_min, q_max) = signed_bounds(self.output_bits);
        let mut out = Vec::with_capacity(self.out_features);
        for j in 0..self.out_features {
            let mut acc = self.biases[j];
            for i in 0..self.in_features { acc = add(acc, mul(self.w(j, i), input[i], self.quant_shift)); }
            let (y, _) = requantize(acc, acc.s, q_min, q_max);
            out.push(y);
        }
        self.output_cache = out.clone();
        out
    }

    fn backward(&mut self, grad_output: &[Dyadic]) -> Vec<Dyadic> {
        assert_eq!(grad_output.len(), self.out_features);
        let (q_min, q_max) = signed_bounds(self.output_bits);
        let g_s = grad_output.first().map_or(0, |g| g.s);
        let mut grad_input = vec![Dyadic::new(0, g_s); self.in_features];
        for j in 0..self.out_features {
            let g_raw = ste_requantize(grad_output[j], self.output_cache[j].v, q_min, q_max);
            let g_j   = Dyadic::new(g_raw.v.clamp(-self.grad_clip, self.grad_clip), g_raw.s);
            for i in 0..self.in_features {
                let idx = self.flat(j, i);
                self.grad_w[idx] = add(self.grad_w[idx], mul(g_j, self.input_cache[i], self.quant_shift));
                grad_input[i]    = add(grad_input[i],    mul(g_j, self.w(j, i),        self.quant_shift));
            }
            self.grad_b[j] = add(self.grad_b[j], g_j);
        }
        grad_input
    }

    fn update(&mut self, lr_shift: u32) {
        apply_updates(&mut self.weights, &self.grad_w, &mut self.vel_w, lr_shift, self.momentum_shift);
        apply_updates(&mut self.biases,  &self.grad_b, &mut self.vel_b, lr_shift, self.momentum_shift);
    }

    fn zero_grad(&mut self) {
        self.grad_w.iter_mut().for_each(|g| g.v = 0);
        self.grad_b.iter_mut().for_each(|g| g.v = 0);
    }
}

// ─── ReLU ─────────────────────────────────────────────────────────────────────

pub struct ReLU { mask: Vec<bool> }

impl ReLU { pub fn new() -> Self { Self { mask: Vec::new() } } }
impl Default for ReLU { fn default() -> Self { Self::new() } }

impl Layer for ReLU {
    fn name(&self) -> &'static str { "ReLU" }

    fn forward(&mut self, input: &[Dyadic]) -> Vec<Dyadic> {
        self.mask = input.iter().map(|x| x.v > 0).collect();
        input.iter().map(|x| if x.v > 0 { *x } else { Dyadic::new(0, x.s) }).collect()
    }

    fn backward(&mut self, grad_output: &[Dyadic]) -> Vec<Dyadic> {
        grad_output.iter().zip(&self.mask)
            .map(|(&g, &a)| if a { g } else { Dyadic::new(0, g.s) })
            .collect()
    }

    fn update(&mut self, _: u32) {}
    fn zero_grad(&mut self) {}
}

// ─── Softmax ──────────────────────────────────────────────────────────────────

/// Numerically stable softmax output layer.
///
/// Computes softmax in f64 then requantises to `output_shift`.
/// Backward is a straight-through estimator: when paired with cross-entropy
/// the combined gradient is simply `p − t`, so no jacobian is needed.
pub struct Softmax {
    pub output_shift: u32,
    pub last_probs:   Vec<f64>,
}

impl Softmax { pub fn new(output_shift: u32) -> Self { Self { output_shift, last_probs: Vec::new() } } }

impl Layer for Softmax {
    fn name(&self) -> &'static str { "Softmax" }
    fn describe(&self) -> String { format!("Softmax(shift={})", self.output_shift) }

    fn forward(&mut self, input: &[Dyadic]) -> Vec<Dyadic> {
        let logits: Vec<f64> = input.iter().map(|x| x.to_f64()).collect();
        let max  = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&z| (z - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        self.last_probs = exps.iter().map(|&e| e / sum).collect();
        let scale = (1u32 << self.output_shift) as f64;
        let (_, max_v) = signed_bounds(self.output_shift + 1);
        self.last_probs.iter()
            .map(|&p| Dyadic::new((p * scale).round().clamp(0.0, max_v as f64) as i32, self.output_shift))
            .collect()
    }

    fn backward(&mut self, grad_output: &[Dyadic]) -> Vec<Dyadic> { grad_output.to_vec() }
    fn update(&mut self, _: u32) {}
    fn zero_grad(&mut self) {}
}

// ─── Conv2D ───────────────────────────────────────────────────────────────────

/// 2D convolution layer.  Input layout: `[in_channels, height, width]` (channel-first flat).
///
/// # Forward
/// ```text
/// for (oc, oh, ow): y[oc,oh,ow] = ℛ(b[oc] ⊕ Σ_{ic,kh,kw} K[oc,ic,kh,kw] ⊗ x[ic, oh+kh, ow+kw])
/// ```
///
/// # Backward (STE through ℛ)
/// ```text
/// ∂L/∂K[oc,ic,kh,kw] += Σ_{oh,ow} g̃[oc,oh,ow] ⊗ x[ic, oh+kh, ow+kw]
/// ∂L/∂x[ic,ih,iw]     = Σ_{oc,kh,kw} g̃[oc,ih−kh, iw−kw] ⊗ K[oc,ic,kh,kw]
///                                       (where oh = ih−kh, ow = iw−kw are valid)
/// ```
///
/// # Spatial dimensions
/// Must be provided at construction time so gradient buffers can be allocated
/// correctly.  `out_h = in_h − kernel_h + 1`,  `out_w = in_w − kernel_w + 1`.
pub struct Conv2D {
    // config
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
    // derived
    out_h: usize,
    out_w: usize,
    // params: kernels[out_c, in_c, kH, kW], biases[out_c]
    pub kernels: Vec<Dyadic>,
    pub biases:  Vec<Dyadic>,
    // caches
    input_cache:  Vec<Dyadic>,
    output_cache: Vec<Dyadic>,
    // gradients + velocity
    grad_k: Vec<Dyadic>,
    grad_b: Vec<Dyadic>,
    vel_k:  Vec<Dyadic>,
    vel_b:  Vec<Dyadic>,
}

impl Conv2D {
    /// Construct with He-style kernel initialisation.
    ///
    /// # Arguments
    /// * `in_channels`, `out_channels` — channel depths
    /// * `kernel_h`, `kernel_w` — kernel spatial size
    /// * `in_h`, `in_w` — input spatial size (fixed at construction time)
    /// * `weight_shift`, `quant_shift`, `output_bits` — same as [`Linear`]
    pub fn new(
        in_channels:  usize,
        out_channels: usize,
        kernel_h:     usize,
        kernel_w:     usize,
        in_h:         usize,
        in_w:         usize,
        weight_shift: u32,
        quant_shift:  u32,
        output_bits:  u32,
    ) -> Self {
        assert!(in_h >= kernel_h && in_w >= kernel_w, "kernel larger than input");

        let mut rng  = rand::thread_rng();
        let fan_in   = in_channels * kernel_h * kernel_w;
        let n_k      = out_channels * fan_in;
        let k        = ((1u32 << weight_shift) as f64 / (fan_in as f64).sqrt()).round() as i32;
        let k        = k.max(1);
        let out_h    = in_h - kernel_h + 1;
        let out_w    = in_w - kernel_w + 1;
        let mk       = |n: usize| vec![Dyadic::new(0, weight_shift); n];

        Self {
            in_channels, out_channels, kernel_h, kernel_w, in_h, in_w,
            quant_shift, output_bits,
            grad_clip:      i32::MAX,
            momentum_shift: None,
            out_h, out_w,
            kernels: (0..n_k).map(|_| Dyadic::new(rng.gen_range(-k..=k), weight_shift)).collect(),
            biases:  mk(out_channels),
            input_cache:  Vec::new(),
            output_cache: Vec::new(),
            grad_k: mk(n_k),
            grad_b: mk(out_channels),
            vel_k:  mk(n_k),
            vel_b:  mk(out_channels),
        }
    }

    pub fn with_grad_clip(mut self, clip: i32) -> Self { self.grad_clip = clip.abs(); self }
    pub fn with_momentum(mut self, shift: u32) -> Self { self.momentum_shift = Some(shift); self }

    /// Output slice length: `out_channels * out_h * out_w`.
    pub fn output_len(&self) -> usize { self.out_channels * self.out_h * self.out_w }

    // ── Index helpers ─────────────────────────────────────────────────────────

    #[inline] fn in_idx (&self, c: usize, h: usize, w: usize) -> usize { c * self.in_h  * self.in_w  + h * self.in_w  + w }
    #[inline] fn out_idx(&self, c: usize, h: usize, w: usize) -> usize { c * self.out_h * self.out_w + h * self.out_w + w }
    #[inline] fn k_idx  (&self, oc: usize, ic: usize, kh: usize, kw: usize) -> usize {
        oc * (self.in_channels * self.kernel_h * self.kernel_w)
        + ic * (self.kernel_h * self.kernel_w)
        + kh * self.kernel_w + kw
    }
}

impl Layer for Conv2D {
    fn name(&self) -> &'static str { "Conv2D" }

    fn describe(&self) -> String {
        let ws   = self.kernels.first().map_or(0, |k| k.s);
        let clip = if self.grad_clip == i32::MAX { "off".into() } else { format!("2^{}", (self.grad_clip as f64).log2() as i32) };
        let mom  = self.momentum_shift.map_or("off".into(), |m| format!("shift={m}"));
        format!(
            "Conv2D(in={}, out={}, kernel={}×{}, spatial={}×{}→{}×{}, clip={clip}, mom={mom})",
            self.in_channels, self.out_channels,
            self.kernel_h, self.kernel_w,
            self.in_h, self.in_w, self.out_h, self.out_w,
        )
    }

    fn forward(&mut self, input: &[Dyadic]) -> Vec<Dyadic> {
        debug_assert_eq!(input.len(), self.in_channels * self.in_h * self.in_w,
            "Conv2D forward: input length mismatch");

        self.input_cache = input.to_vec();
        let (q_min, q_max) = signed_bounds(self.output_bits);
        let mut output = vec![Dyadic::new(0, 0); self.output_len()];

        for oc in 0..self.out_channels {
            for oh in 0..self.out_h {
                for ow in 0..self.out_w {
                    let mut acc = self.biases[oc];
                    for ic in 0..self.in_channels {
                        for kh in 0..self.kernel_h {
                            for kw in 0..self.kernel_w {
                                let inp = input[self.in_idx(ic, oh + kh, ow + kw)];
                                let ker = self.kernels[self.k_idx(oc, ic, kh, kw)];
                                acc = add(acc, mul(ker, inp, self.quant_shift));
                            }
                        }
                    }
                    let (y, _) = requantize(acc, acc.s, q_min, q_max);
                    output[self.out_idx(oc, oh, ow)] = y;
                }
            }
        }

        self.output_cache = output.clone();
        output
    }

    fn backward(&mut self, grad_output: &[Dyadic]) -> Vec<Dyadic> {
        debug_assert_eq!(grad_output.len(), self.output_len());

        let (q_min, q_max) = signed_bounds(self.output_bits);
        let g_s = grad_output.first().map_or(0, |g| g.s);
        let mut grad_input = vec![Dyadic::new(0, g_s); self.in_channels * self.in_h * self.in_w];

        for oc in 0..self.out_channels {
            for oh in 0..self.out_h {
                for ow in 0..self.out_w {
                    // STE gate through ℛ.
                    let g_raw = ste_requantize(
                        grad_output[self.out_idx(oc, oh, ow)],
                        self.output_cache[self.out_idx(oc, oh, ow)].v,
                        q_min, q_max,
                    );
                    let g = Dyadic::new(g_raw.v.clamp(-self.grad_clip, self.grad_clip), g_raw.s);

                    for ic in 0..self.in_channels {
                        for kh in 0..self.kernel_h {
                            for kw in 0..self.kernel_w {
                                let in_pos  = self.in_idx(ic, oh + kh, ow + kw);
                                let k_pos   = self.k_idx(oc, ic, kh, kw);

                                // ∂L/∂K[oc,ic,kh,kw] += g ⊗ x[ic, oh+kh, ow+kw]
                                self.grad_k[k_pos] = add(
                                    self.grad_k[k_pos],
                                    mul(g, self.input_cache[in_pos], self.quant_shift),
                                );

                                // ∂L/∂x[ic, oh+kh, ow+kw] += g ⊗ K[oc,ic,kh,kw]
                                grad_input[in_pos] = add(
                                    grad_input[in_pos],
                                    mul(g, self.kernels[k_pos], self.quant_shift),
                                );
                            }
                        }
                    }
                    // ∂L/∂b[oc] += g
                    self.grad_b[oc] = add(self.grad_b[oc], g);
                }
            }
        }

        grad_input
    }

    fn update(&mut self, lr_shift: u32) {
        apply_updates(&mut self.kernels, &self.grad_k, &mut self.vel_k, lr_shift, self.momentum_shift);
        apply_updates(&mut self.biases,  &self.grad_b, &mut self.vel_b, lr_shift, self.momentum_shift);
    }

    fn zero_grad(&mut self) {
        self.grad_k.iter_mut().for_each(|g| g.v = 0);
        self.grad_b.iter_mut().for_each(|g| g.v = 0);
    }
}

// ─── MaxPool2D ────────────────────────────────────────────────────────────────

/// 2D max-pooling with configurable kernel and stride.
///
/// Input layout: `[channels, height, width]` (channel-first flat).
///
/// **Forward:** for each pooling window, take the element with the largest mantissa.  
/// **Backward:** route gradient to the max-position only (standard max-pool STE).
///
/// No learnable parameters; `update` and `zero_grad` are no-ops.
pub struct MaxPool2D {
    pub channels: usize,
    pub in_h:     usize,
    pub in_w:     usize,
    pub kernel:   usize,
    pub stride:   usize,
    out_h:    usize,
    out_w:    usize,
    max_mask: Vec<usize>,   // flat input index of the max element per output position
}

impl MaxPool2D {
    /// `kernel × kernel` pooling window with the given stride.
    /// `in_h` and `in_w` are the spatial dimensions of the input.
    ///
    /// Typical usage: `MaxPool2D::new(channels, in_h, in_w, 2, 2)` — 2×2 halving.
    pub fn new(channels: usize, in_h: usize, in_w: usize, kernel: usize, stride: usize) -> Self {
        assert!(kernel > 0 && stride > 0);
        let out_h = (in_h - kernel) / stride + 1;
        let out_w = (in_w - kernel) / stride + 1;
        Self { channels, in_h, in_w, kernel, stride, out_h, out_w, max_mask: Vec::new() }
    }

    /// Output slice length: `channels * out_h * out_w`.
    pub fn output_len(&self) -> usize { self.channels * self.out_h * self.out_w }

    #[inline] fn in_idx (&self, c: usize, h: usize, w: usize) -> usize { c * self.in_h  * self.in_w  + h * self.in_w  + w }
    #[inline] fn out_idx(&self, c: usize, h: usize, w: usize) -> usize { c * self.out_h * self.out_w + h * self.out_w + w }
}

impl Layer for MaxPool2D {
    fn name(&self) -> &'static str { "MaxPool2D" }

    fn describe(&self) -> String {
        format!("MaxPool2D(channels={}, {}×{}→{}×{}, kernel={}, stride={})",
            self.channels, self.in_h, self.in_w, self.out_h, self.out_w, self.kernel, self.stride)
    }

    fn forward(&mut self, input: &[Dyadic]) -> Vec<Dyadic> {
        debug_assert_eq!(input.len(), self.channels * self.in_h * self.in_w);
        let mut output   = vec![Dyadic::new(0, 0); self.output_len()];
        self.max_mask = vec![0usize; self.output_len()];

        for c in 0..self.channels {
            for oh in 0..self.out_h {
                for ow in 0..self.out_w {
                    let ih0 = oh * self.stride;
                    let iw0 = ow * self.stride;
                    let mut max_v   = i32::MIN;
                    let mut max_idx = self.in_idx(c, ih0, iw0);

                    for kh in 0..self.kernel {
                        for kw in 0..self.kernel {
                            let idx = self.in_idx(c, ih0 + kh, iw0 + kw);
                            if input[idx].v > max_v {
                                max_v   = input[idx].v;
                                max_idx = idx;
                            }
                        }
                    }

                    let out_pos = self.out_idx(c, oh, ow);
                    output[out_pos]        = input[max_idx];
                    self.max_mask[out_pos] = max_idx;
                }
            }
        }
        output
    }

    fn backward(&mut self, grad_output: &[Dyadic]) -> Vec<Dyadic> {
        let g_s = grad_output.first().map_or(0, |g| g.s);
        let mut grad_input = vec![Dyadic::new(0, g_s); self.channels * self.in_h * self.in_w];

        // Accumulate gradient at max positions (STE: zero elsewhere).
        for (out_pos, &max_idx) in self.max_mask.iter().enumerate() {
            grad_input[max_idx] = add(grad_input[max_idx], grad_output[out_pos]);
        }
        grad_input
    }

    fn update(&mut self, _: u32) {}
    fn zero_grad(&mut self) {}
}

// ─── Flatten ──────────────────────────────────────────────────────────────────

/// Shape-only layer that bridges conv → linear.
///
/// Passes data through unchanged — the `[channels, H, W]` flat layout that
/// conv layers produce is already the format that `Linear` expects.
/// This layer exists for documentation and `summary()` clarity.
pub struct Flatten;

impl Layer for Flatten {
    fn name(&self) -> &'static str { "Flatten" }
    fn forward(&mut self, input: &[Dyadic]) -> Vec<Dyadic> { input.to_vec() }
    fn backward(&mut self, grad: &[Dyadic]) -> Vec<Dyadic> { grad.to_vec() }
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
    #[test] fn linear_shapes() {
        let mut l = Linear::new(3, 5, S, S, 32);
        let out = l.forward(&[d(1.0), d(-0.5), d(0.25)]);
        assert_eq!(out.len(), 5);
        let g = l.backward(&vec![d(0.1); 5]);
        assert_eq!(g.len(), 3);
    }

    #[test] fn linear_momentum_runs() {
        let mut l = Linear::new(2, 2, S, S, 32).with_momentum(1);
        l.forward(&[d(1.0), d(-1.0)]);
        l.backward(&[d(0.5), d(-0.5)]);
        l.update(7);
        assert!(l.vel_w.iter().any(|v| v.v != 0));
    }

    // ── ReLU ──────────────────────────────────────────────────────────────────
    #[test] fn relu_gates() {
        let mut relu = ReLU::new();
        let out = relu.forward(&[d(1.0), d(-0.5), d(0.0)]);
        assert!(out[0].v > 0); assert_eq!(out[1].v, 0); assert_eq!(out[2].v, 0);
        let g = relu.backward(&[d(1.0); 3]);
        assert!(g[0].v != 0); assert_eq!(g[1].v, 0); assert_eq!(g[2].v, 0);
    }

    // ── Softmax ───────────────────────────────────────────────────────────────
    #[test] fn softmax_sums_to_one() {
        let mut sm = Softmax::new(S);
        sm.forward(&[d(1.0), d(0.5), d(-0.5)]);
        assert!((sm.last_probs.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    // ── Conv2D ────────────────────────────────────────────────────────────────
    #[test] fn conv2d_output_shape() {
        // 1×5×5 input, 2 output channels, 3×3 kernel → 2×3×3 = 18 outputs
        let mut conv = Conv2D::new(1, 2, 3, 3, 5, 5, S, S, 32);
        let input = vec![d(0.5); 1 * 5 * 5];
        let out = conv.forward(&input);
        assert_eq!(out.len(), 2 * 3 * 3);
    }

    #[test] fn conv2d_backward_grad_input_shape() {
        let mut conv = Conv2D::new(1, 2, 3, 3, 5, 5, S, S, 32);
        let input = vec![d(0.5); 25];
        conv.forward(&input);
        let g = conv.backward(&vec![d(0.1); 2 * 3 * 3]);
        assert_eq!(g.len(), 25);
    }

    #[test] fn conv2d_grad_accumulates() {
        let mut conv = Conv2D::new(1, 1, 3, 3, 5, 5, S, S, 32);
        let input = vec![d(1.0); 25];
        conv.forward(&input);
        conv.backward(&vec![d(0.5); 9]);
        assert!(conv.grad_k.iter().any(|g| g.v != 0) || conv.grad_b.iter().any(|g| g.v != 0));
    }

    #[test] fn conv2d_with_momentum_updates() {
        let mut conv = Conv2D::new(1, 1, 3, 3, 5, 5, S, S, 32).with_momentum(1);
        let input = vec![d(0.5); 25];
        conv.forward(&input);
        conv.backward(&vec![d(0.1); 9]);
        conv.update(7);
        assert!(conv.vel_k.iter().any(|v| v.v != 0));
    }

    // ── MaxPool2D ─────────────────────────────────────────────────────────────
    #[test] fn maxpool_output_shape() {
        // 1×4×4 → MaxPool(2,2) → 1×2×2
        let mut pool = MaxPool2D::new(1, 4, 4, 2, 2);
        let input = vec![d(0.5); 16];
        let out = pool.forward(&input);
        assert_eq!(out.len(), 4);
    }

    #[test] fn maxpool_selects_max() {
        let mut pool = MaxPool2D::new(1, 2, 2, 2, 2);
        // 2×2 input, one pool window. Largest value at position 2.
        let input = vec![d(0.1), d(0.2), d(0.9), d(0.3)];
        let out = pool.forward(&input);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].v, input[2].v); // max is at index 2
    }

    #[test] fn maxpool_backward_routes_to_max() {
        let mut pool = MaxPool2D::new(1, 2, 2, 2, 2);
        let input = vec![d(0.1), d(0.2), d(0.9), d(0.3)];
        pool.forward(&input);
        let grad_out = vec![d(1.0)];
        let grad_in  = pool.backward(&grad_out);
        assert_eq!(grad_in.len(), 4);
        assert_eq!(grad_in[2].v, grad_out[0].v); // gradient routed to max position
        assert_eq!(grad_in[0].v, 0);
        assert_eq!(grad_in[1].v, 0);
        assert_eq!(grad_in[3].v, 0);
    }

    // ── Sequential with conv ──────────────────────────────────────────────────
    #[test] fn full_conv_pipeline() {
        // Simulate a tiny "image": 1×6×6
        let mut model = Sequential::new();
        model.add(Conv2D::new(1, 2, 3, 3, 6, 6, S, S, 32).with_momentum(1));
        model.add(ReLU::new());
        model.add(MaxPool2D::new(2, 4, 4, 2, 2));  // 2×4×4 → 2×2×2 = 8
        model.add(Flatten);
        model.add(Linear::new(8, 3, S, S, 32).with_momentum(1));
        model.add(Softmax::new(S));
        model.summary();

        let input = vec![d(0.5); 36];
        let out   = model.forward(&input);
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|x| x.v >= 0));

        let t   = vec![Dyadic::new(127, S), Dyadic::new(0, S), Dyadic::new(0, S)];
        let g: Vec<_> = out.iter().zip(&t).map(|(&y, &ti)| Dyadic::new(y.v - ti.v, S)).collect();
        model.backward(&g);
        model.update(7);
    }
}
