use crate::nn::{Module, apply_updates};
use crate::dyadic::{
    add, mul, requantize, signed_bounds, ste_requantize, stochastic_round, Dyadic,
    Tensor, TensorView
};
use crate::rng::rng_range;

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

    fn apply_mask(input: TensorView, mask: &[bool], scale: Dyadic, qs: u32) -> Tensor {
        Tensor::from_vec(
            input.data.iter().zip(mask).map(|(&x, &keep)| {
                if keep { mul(x, scale, qs) } else { Dyadic::new(0, x.s) }
            }).collect(),
            input.shape.to_vec()
        )
    }
}

impl Module for Dropout {
    fn name(&self) -> &'static str { "Dropout" }
    fn describe(&self) -> String { format!("Dropout(p={:.2}, training={})", self.drop_rate, self.training) }
    fn set_training(&mut self, t: bool) { self.training = t; }

    fn forward(&mut self, input: TensorView) -> Tensor {
        if !self.training { self.mask.clear(); return input.to_tensor(); }
        self.mask = Self::sample_mask(input.data.len(), self.keep_prob_bits);
        Self::apply_mask(input, &self.mask, self.scale, self.quant_shift)
    }

    fn backward(&mut self, grad: TensorView) -> Tensor {
        if !self.training || self.mask.is_empty() { return grad.to_tensor(); }
        Self::apply_mask(grad, &self.mask, self.scale, self.quant_shift)
    }

    fn forward_batch(&mut self, inputs: &Tensor) -> Tensor {
        if !self.training {
            self.batch_masks.clear();
            return inputs.clone();
        }
        let mut masks = Vec::with_capacity(inputs.len());
        let outputs: Tensor = inputs.iter().map(|x| {
            let m = Self::sample_mask(x.data.len(), self.keep_prob_bits);
            let out = Self::apply_mask(x, &m, self.scale, self.quant_shift);
            masks.push(m);
            out
        }).collect();
        self.batch_masks = masks;
        outputs
    }

    fn backward_batch(&mut self, grads: &Tensor) -> Tensor {
        if !self.training || self.batch_masks.is_empty() { return grads.clone(); }
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
    pub gamma:        Tensor,
    pub beta:         Tensor,
    pub running_mean: Vec<f64>,
    pub running_var:  Vec<f64>,
    training:         bool,
    // batch caches
    x_hat_cache:  Vec<Vec<f64>>,   // normalised float values per (sample, feature)
    inv_std_cache: Vec<f64>,        // 1/σ per feature
    input_batch_cache: Tensor,
    // gradients + velocity
    grad_gamma: Tensor,
    grad_beta:  Tensor,
    vel_gamma:  Tensor,
    vel_beta:   Tensor,
    pub momentum_shift: Option<u32>,
}

impl BatchNorm1D {
    /// Create a new BatchNorm1D layer.
    ///
    /// `shift` should match the rest of the network.
    /// `momentum` is the EMA coefficient for running stats (PyTorch default: 0.1).
    pub fn new(num_features: usize, shift: u32) -> Self {
        let scale = (1u32 << shift) as f64;  // value of 1.0 at this shift
        let gamma = Tensor::from_vec(vec![Dyadic::new(scale.round() as i32, shift); num_features], vec![num_features]);
        let beta  = Tensor::from_vec(vec![Dyadic::new(0, shift); num_features], vec![num_features]);
        let z     = |n: usize| vec![Dyadic::new(0, shift); n];
        Self {
            num_features, shift,
            momentum: 0.1,
            eps:      1e-5,
            gamma,
            beta,
            running_mean: vec![0.0; num_features],
            running_var:  vec![1.0; num_features],
            training: true,
            x_hat_cache:       vec![vec![0.0; num_features]; 0],
            inv_std_cache:     vec![1.0; num_features],
            input_batch_cache: Tensor::new(),
            grad_gamma: Tensor::zeros(vec![num_features]),
            grad_beta:  Tensor::zeros(vec![num_features]),
            vel_gamma:  Tensor::zeros(vec![num_features]),
            vel_beta:   Tensor::zeros(vec![num_features]),
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
    fn forward(&mut self, input: TensorView) -> Tensor {
        let scale = (1u32 << self.shift) as f64;
        let mut out = Vec::with_capacity(self.num_features);

        for f in 0..self.num_features {
            let x_f   = input.data[f].to_f64();
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
            out.push(add(mul(self.gamma.data[f], xhd, self.shift), self.beta.data[f]));
        }
        Tensor::from_vec(out, vec![self.num_features])
    }

    fn backward(&mut self, grad: TensorView) -> Tensor {
        // Single-sample STE: pass gradient through scaled by γ/σ.
        // (Full backward requires the batch; this is used when batch_size=1.)
        let scale = (1u32 << self.shift) as f64;
        let gout = grad.data.iter().enumerate().map(|(f, &g)| {
            let gamma_f   = self.gamma.data[f].to_f64();
            let inv_std_f = self.inv_std_cache.get(f).copied().unwrap_or(1.0);
            let dx = g.to_f64() * gamma_f * inv_std_f;
            // Accumulate γ and β gradients
            self.grad_gamma.data[f] = Dyadic::new(
                self.grad_gamma.data[f].v.saturating_add((g.to_f64() * 0.0 * scale) as i32), // x_hat not cached
                self.shift);
            self.grad_beta.data[f]  = Dyadic::new(
                self.grad_beta.data[f].v.saturating_add((g.to_f64() * scale).round() as i32),
                self.shift);
            Dyadic::new((dx * scale).round() as i32, self.shift)
        }).collect();
        Tensor::from_vec(gout, grad.shape.to_vec())
    }

    /// Batch forward — computes proper batch statistics.  Always use this during training.
    fn forward_batch(&mut self, inputs: &Tensor) -> Tensor {
        let n     = inputs.len();
        let scale = (1u32 << self.shift) as f64;
        let m_ema = self.momentum as f64;

        // Decode all inputs
        let decoded: Vec<Vec<f64>> = inputs.iter()
            .map(|x| x.data.iter().map(|d| d.to_f64()).collect())
            .collect();

        let mut means     = vec![0.0f64; self.num_features];
        let mut inv_stds  = vec![0.0f64; self.num_features];
        let mut x_hat_all = vec![vec![0.0f64; self.num_features]; n];
        let mut outputs   = vec![Dyadic::new(0, self.shift); self.num_features * n];

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
                outputs[s + (f * self.num_features)] = add(mul(self.gamma.data[f], xhd, self.shift), self.beta.data[f]);
            }
        }

        self.inv_std_cache     = inv_stds;
        self.x_hat_cache       = x_hat_all;
        self.input_batch_cache = inputs.clone();
        Tensor::from_vec(outputs, vec![n, self.num_features])
    }

    /// Batch backward — full batch-norm gradient.
    fn backward_batch(&mut self, grads: &Tensor) -> Tensor {
        let n     = grads.len();
        let scale = (1u32 << self.shift) as f64;
        let nf    = n as f64;

        let dL_dy: Vec<Vec<f64>> = grads.iter()
            .map(|g| g.data.iter().map(|d| d.to_f64()).collect())
            .collect();

        let mut grad_inputs = vec![Dyadic::new(0, self.shift); self.num_features * n];

        for f in 0..self.num_features {
            let gamma_f   = self.gamma.data[f].to_f64();
            let inv_std_f = self.inv_std_cache[f];

            let sum_dL_dy:      f64 = (0..n).map(|s| dL_dy[s][f]).sum();
            let sum_dL_dy_xhat: f64 = (0..n).map(|s| dL_dy[s][f] * self.x_hat_cache[s][f]).sum();

            // Accumulate learnable parameter gradients.
            self.grad_gamma.data[f] = Dyadic::new(
                self.grad_gamma.data[f].v.saturating_add((sum_dL_dy_xhat * scale).round() as i32),
                self.shift);
            self.grad_beta.data[f] = Dyadic::new(
                self.grad_beta.data[f].v.saturating_add((sum_dL_dy * scale).round() as i32),
                self.shift);

            // Full BN backward:
            // ∂L/∂x_n = (γ/σ) · [ ∂L/∂ŷ_n − (1/N)·∂L/∂β − (x̂_n/N)·∂L/∂γ ]
            for s in 0..n {
                let dx = (gamma_f * inv_std_f)
                    * (dL_dy[s][f]
                       - sum_dL_dy / nf
                       - self.x_hat_cache[s][f] * sum_dL_dy_xhat / nf);
                grad_inputs[s + (f * self.num_features)] = Dyadic::new((dx * scale).round() as i32, self.shift);
            }
        }

        Tensor::from_vec(grad_inputs, vec![n, self.num_features])
    }

    fn update(&mut self, lr: u32) {
        apply_updates(&mut self.gamma.data, &self.grad_gamma, &mut self.vel_gamma.data, lr, self.momentum_shift);
        apply_updates(&mut self.beta.data,  &self.grad_beta,  &mut self.vel_beta.data,  lr, self.momentum_shift);
    }

    fn zero_grad(&mut self) {
        self.grad_gamma.data.iter_mut().for_each(|g| g.v = 0);
        self.grad_beta.data.iter_mut().for_each(|g| g.v = 0);
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
    pub gamma:     Tensor,   // [channels]
    pub beta:      Tensor,   // [channels]
    pub running_mean: Vec<f64>,
    pub running_var:  Vec<f64>,
    training:      bool,
    // caches
    x_hat_cache:       Vec<Vec<f64>>,   // [n_samples, C*H*W]
    inv_std_cache:     Vec<f64>,         // [channels]
    input_batch_cache: Tensor,
    // gradients + velocity
    grad_gamma: Tensor,
    grad_beta:  Tensor,
    vel_gamma:  Tensor,
    vel_beta:   Tensor,

    pub momentum_shift: Option<u32>,
}

impl BatchNorm2D {
    pub fn new(channels: usize, h: usize, w: usize, shift: u32) -> Self {
        let scale = (1u32 << shift) as f64;
        let gamma = Tensor::from_vec(vec![Dyadic::new(scale.round() as i32, shift); channels], vec![channels]);
        let beta  = Tensor::from_vec(vec![Dyadic::new(0, shift); channels], vec![channels]);

        let z     = |n: usize| vec![Dyadic::new(0, shift); n];
        Self {
            channels, h, w, shift,
            momentum: 0.1,
            eps:      1e-5,
            gamma,
            beta,
            running_mean: vec![0.0; channels],
            running_var:  vec![1.0; channels],
            training: true,
            x_hat_cache:       Vec::new(),
            inv_std_cache:     vec![1.0; channels],
            input_batch_cache: Tensor::new(),
            grad_gamma: Tensor::zeros(vec![channels]),
            grad_beta:  Tensor::zeros(vec![channels]),
            vel_gamma:  Tensor::zeros(vec![channels]),
            vel_beta:   Tensor::zeros(vec![channels]),
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
    fn forward(&mut self, input: TensorView) -> Tensor {
        let scale = (1u32 << self.shift) as f64;
        let hw    = self.h * self.w;
        let m_ema = self.momentum as f64;
        let mut out = Tensor::from_vec(
            vec![Dyadic::new(0, self.shift); input.data.len()],
            vec![input.data.len()]
        );

        for c in 0..self.channels {
            let (mu, inv_std) = if self.training {
                // With a single sample, compute stats over H*W spatial positions.
                let vals: Vec<f64> = (0..hw)
                    .map(|p| input.data[c * hw + p].to_f64())
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
                let x_hat = (input.data[idx].to_f64() - mu) * inv_std;
                let xhd   = Dyadic::new((x_hat * scale).round() as i32, self.shift);
                out.data[idx]  = add(mul(self.gamma.data[c], xhd, self.shift), self.beta.data[c]);
            }
        }
        out
    }

    fn backward(&mut self, grad: TensorView) -> Tensor {
        // Single-sample STE: γ/σ passthrough.
        let scale = (1u32 << self.shift) as f64;
        let hw    = self.h * self.w;
        let mut out = Tensor::from_vec(
            vec![Dyadic::new(0, self.shift); grad.data.len()],
            vec![grad.data.len()]
        );

        for c in 0..self.channels {
            let gamma_f   = self.gamma.data[c].to_f64();
            let inv_std_c = self.inv_std_cache[c];
            for p in 0..hw {
                let idx = self.spatial_idx(c, p / self.w, p % self.w);
                let dx  = grad.data[idx].to_f64() * gamma_f * inv_std_c;
                // Accumulate β gradient
                self.grad_beta.data[c] = Dyadic::new(
                    self.grad_beta.data[c].v.saturating_add((grad.data[idx].to_f64() * scale).round() as i32),
                    self.shift);
                out.data[idx] = Dyadic::new((dx * scale).round() as i32, self.shift);
            }
        }
        out
    }

    fn forward_batch(&mut self, inputs: &Tensor) -> Tensor {
        let n     = inputs.len();
        let hw    = self.h * self.w;
        let scale = (1u32 << self.shift) as f64;
        let m_ema = self.momentum as f64;
        let n_hw  = (n * hw) as f64;

        let decoded: Vec<Vec<f64>> = inputs.iter()
            .map(|x| x.data.iter().map(|d| d.to_f64()).collect())
            .collect();

        let mut inv_stds  = vec![0.0f64; self.channels];
        let mut x_hat_all = vec![vec![0.0f64; self.channels * hw]; n];
        let mut outputs   = Tensor::from_vec(
            vec![Dyadic::new(0, self.shift); self.channels * hw * n],
            vec![n, self.channels * hw]
        );

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
                    // original: outputs[s][idx]
                    outputs.data[s * hw + idx] = add(mul(self.gamma.data[c], xhd, self.shift), self.beta.data[c]);
                }
            }
        }

        self.inv_std_cache     = inv_stds;
        self.x_hat_cache       = x_hat_all;
        self.input_batch_cache = inputs.clone();
        outputs
    }

    fn backward_batch(&mut self, grads: &Tensor) -> Tensor {
        let n     = grads.len();
        let hw    = self.h * self.w;
        let scale = (1u32 << self.shift) as f64;
        let n_hw  = (n * hw) as f64;

        let dL_dy: Vec<Vec<f64>> = grads.iter()
            .map(|g| g.data.iter().map(|d| d.to_f64()).collect())
            .collect();

        let mut grad_inputs = Tensor::from_vec(
            vec![Dyadic::new(0, self.shift); self.channels * hw * n],
            vec![n, self.channels*hw]
        );

        for c in 0..self.channels {
            let gamma_c   = self.gamma.data[c].to_f64();
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

            self.grad_gamma.data[c] = Dyadic::new(
                self.grad_gamma.data[c].v.saturating_add((sum_dL_dy_xhat * scale).round() as i32),
                self.shift);
            self.grad_beta.data[c] = Dyadic::new(
                self.grad_beta.data[c].v.saturating_add((sum_dL_dy * scale).round() as i32),
                self.shift);

            // Full BN backward per spatial position.
            for s in 0..n {
                for p in 0..hw {
                    let idx = c * hw + p;
                    let dx  = (gamma_c * inv_std_c)
                        * (dL_dy[s][idx]
                           - sum_dL_dy / n_hw
                           - self.x_hat_cache[s][idx] * sum_dL_dy_xhat / n_hw);
                    grad_inputs.data[s * hw + idx] = Dyadic::new((dx * scale).round() as i32, self.shift);
                }
            }
        }

        grad_inputs
    }

    fn update(&mut self, lr: u32) {
        apply_updates(&mut self.gamma.data, &self.grad_gamma, &mut self.vel_gamma.data, lr, self.momentum_shift);
        apply_updates(&mut self.beta.data,  &self.grad_beta,  &mut self.vel_beta.data,  lr, self.momentum_shift);
    }

    fn zero_grad(&mut self) {
        self.grad_gamma.data.iter_mut().for_each(|g| g.v = 0);
        self.grad_beta.data.iter_mut().for_each(|g| g.v = 0);
    }
}


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
        &self, input: TensorView, mask: &mut Vec<usize>,
    ) -> Tensor {
        let mut out = Tensor::from_vec(
            vec![Dyadic::new(0, 0); self.output_len()],
            vec![self.output_len()]
        );
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
                            if input.data[idx].v > mx { mx = input.data[idx].v; mxi = idx; }
                        }
                    }
                    let op = self.out_idx(c, oh, ow);
                    out.data[op] = input.data[mxi];
                    mask[op] = mxi;
                }
            }
        }
        out
    }

    fn pool_backward(
        &self, grad: TensorView, mask: &[usize],
    ) -> Tensor {
        let g_s = grad.data.first().map_or(0, |g| g.s);
        let mut gi = Tensor::from_vec(
            vec![Dyadic::new(0, g_s); self.channels * self.in_h * self.in_w],
            vec![self.channels * self.in_h * self.in_w]
        );
        for (op, &mi) in mask.iter().enumerate() {
            gi.data[mi] = add(gi.data[mi], grad.data[op]);
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

    fn forward(&mut self, input: TensorView) -> Tensor {
        let mut mask = Vec::new();
        let out = self.pool_forward(input, &mut mask);
        self.max_mask = mask;
        out
    }

    fn backward(&mut self, grad: TensorView) -> Tensor {
        self.pool_backward(grad, &self.max_mask.clone())
    }

    fn forward_batch(&mut self, inputs: &Tensor) -> Tensor {
        let mut all_masks = Vec::with_capacity(inputs.len());
        let outputs: Tensor = inputs.iter().map(|x| {
            let mut mask = Vec::new();
            let out = self.pool_forward(x, &mut mask);
            all_masks.push(mask);
            out
        }).collect();
        self.batch_max_masks = all_masks;
        outputs
    }

    fn backward_batch(&mut self, grads: &Tensor) -> Tensor {
        grads.iter().enumerate()
            .map(|(n, g)| self.pool_backward(g, &self.batch_max_masks[n]))
            .collect()
    }

    fn update(&mut self, _: u32) {}
    fn zero_grad(&mut self) {}
}

