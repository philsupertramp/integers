use crate::nn::{Module, apply_updates};
use crate::dyadic::{
    add, mul, requantize, signed_bounds, ste_requantize, stochastic_round, Dyadic,
    Tensor, TensorView
};
use crate::rng::rng_range;

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
    pub kernels: Tensor,
    pub biases:  Tensor,
    input_cache:        Tensor,
    output_cache:       Tensor,
    input_batch_cache:  Tensor,
    output_batch_cache: Tensor,
    grad_k: Tensor,
    grad_b: Tensor,
    vel_k:  Tensor,
    vel_b:  Tensor,

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
        let kernels = Tensor::from_vec(
            (0..n_k).map(|_| Dyadic::new(-k + rng_range(2 * k as u32) as i32, weight_shift)).collect(),
            vec![out_channels, fan_in] // TODO: might be fan_in, out_channels instead
        );
        Self {
            in_channels, out_channels, kernel_h, kernel_w, in_h, in_w,
            quant_shift, output_bits,
            grad_clip: i32::MAX, momentum_shift: None,
            out_h, out_w,
            kernels: kernels,
            biases:  Tensor::from_vec(z(out_channels), vec![out_channels]),
            input_cache: Tensor::new(),
            output_cache: Tensor::new(),
            input_batch_cache: Tensor::new(),
            output_batch_cache: Tensor::new(),
            grad_k: Tensor::zeros(vec![n_k]),
            grad_b: Tensor::zeros(vec![out_channels]),
            vel_k:  Tensor::zeros(vec![n_k]),
            vel_b:  Tensor::zeros(vec![out_channels]),
            q_min: q_min,
            q_max: q_max,
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

    fn forward_one(&mut self, input: TensorView) -> Tensor {
        let mut out = vec![Dyadic::new(0, 0); self.output_len()];
        for oc in 0..self.out_channels {
            for oh in 0..self.out_h {
                for ow in 0..self.out_w {
                    let mut acc = self.biases.data[oc];
                    for ic in 0..self.in_channels {
                        for kh in 0..self.kernel_h {
                            for kw in 0..self.kernel_w {
                                acc = add(acc, mul(
                                    self.kernels.data[self.k_idx(oc, ic, kh, kw)],
                                    input.data[self.in_idx(ic, oh + kh, ow + kw)],
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
        Tensor::from_vec(out, vec![self.output_len()])
    }

    fn backward_one(&mut self, grad_output: TensorView, input: TensorView, output: TensorView) -> Tensor {
        let g_s = grad_output.data.first().map_or(0, |g| g.s);
        let mut gi = vec![Dyadic::new(0, g_s); self.in_channels * self.in_h * self.in_w];
        for oc in 0..self.out_channels {
            for oh in 0..self.out_h {
                for ow in 0..self.out_w {
                    let op  = self.out_idx(oc, oh, ow);
                    let gr  = ste_requantize(grad_output.data[op], output.data[op].v, self.q_min, self.q_max);
                    let gj  = Dyadic::new(gr.v.clamp(-self.grad_clip, self.grad_clip), gr.s);
                    for ic in 0..self.in_channels {
                        for kh in 0..self.kernel_h {
                            for kw in 0..self.kernel_w {
                                let ki = self.k_idx(oc, ic, kh, kw);
                                let ii = self.in_idx(ic, oh + kh, ow + kw);
                                self.grad_k.data[ki] = add(self.grad_k.data[ki], mul(gj, input.data[ii],        self.quant_shift));
                                gi[ii]          = add(gi[ii],          mul(gj, self.kernels.data[ki], self.quant_shift));
                            }
                        }
                    }
                    self.grad_b.data[oc] = add(self.grad_b.data[oc], gj);
                }
            }
        }
        Tensor::from_vec(gi, vec![self.in_channels * self.in_h * self.in_w])
    }
}

impl Module for Conv2D {
    fn name(&self) -> &'static str { "Conv2D" }
    fn describe(&self) -> String {
        let ws   = self.kernels.data.first().map_or(0, |k| k.s);
        let clip = if self.grad_clip == i32::MAX { "off".into() } else { format!("2^{}", (self.grad_clip as f64).log2() as i32) };
        let mom  = self.momentum_shift.map_or("off".into(), |m| format!("shift={m}"));
        format!("Conv2D(in={}, out={}, {}×{}, {}×{}→{}×{}, clip={clip}, mom={mom})",
            self.in_channels, self.out_channels,
            self.kernel_h, self.kernel_w,
            self.in_h, self.in_w, self.out_h, self.out_w)
    }

    fn forward(&mut self, input: TensorView) -> Tensor {
        let out = self.forward_one(input.clone());
        self.input_cache  = input.to_tensor();
        self.output_cache = out.clone();
        out
    }

    fn backward(&mut self, grad: TensorView) -> Tensor {
        let inp = self.input_cache.clone();
        let out = self.output_cache.clone();
        self.backward_one(grad, inp.view(), out.view())
    }

    fn forward_batch(&mut self, inputs: &Tensor) -> Tensor {
        let outputs: Tensor = inputs.iter().map(|x| self.forward_one(x)).collect();
        self.input_batch_cache  = inputs.clone();
        self.output_batch_cache = outputs.clone();
        outputs
    }

    fn backward_batch(&mut self, grads: &Tensor) -> Tensor {
        let inputs = std::mem::take(&mut self.input_batch_cache);
        let outputs = std::mem::take(&mut self.output_batch_cache);
        grads.iter().zip(inputs.iter().zip(outputs.iter()))
            .map(|(g, (inp, out))| {
                self.backward_one(g, inp, out)
            })
            .collect()
    }

    fn update(&mut self, lr: u32) {
        apply_updates(&mut self.kernels.data, &self.grad_k, &mut self.vel_k.data, lr, self.momentum_shift);
        apply_updates(&mut self.biases.data,  &self.grad_b, &mut self.vel_b.data, lr, self.momentum_shift);
    }

    fn zero_grad(&mut self) {
        self.grad_k.data.iter_mut().for_each(|g| g.v = 0);
        self.grad_b.data.iter_mut().for_each(|g| g.v = 0);
    }
}

