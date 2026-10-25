use crate::dyadic::{
    Dyadic, Tensor, TensorView, add, mul, requantize, signed_bounds, ste_requantize,
};
use crate::nn::{Module, TrainingArithmetic, apply_updates_mean};
use crate::rng::rng_range;

/// Fully-connected affine layer: `y = ℛ(W⊗x ⊕ b)`.
pub struct Linear {
    pub in_features: usize,
    pub out_features: usize,
    pub weights: Tensor,
    pub biases: Tensor,
    pub quant_shift: u32,
    pub output_bits: u32,
    pub grad_clip: i32,
    pub momentum_shift: Option<u32>,
    // per-sample caches (single call)
    input_cache: Tensor,
    output_cache: Tensor,
    // batch caches (batch call)
    input_batch_cache: Tensor,
    output_batch_cache: Tensor,
    clip_cache: Vec<bool>,
    clip_batch_cache: Vec<Vec<bool>>,
    // gradients + velocity
    grad_w: Tensor,
    grad_b: Tensor,
    vel_w: Tensor,
    vel_b: Tensor,
    grad_samples: usize,

    q_min: i32,
    q_max: i32,
}

impl Linear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        weight_shift: u32,
        quant_shift: u32,
        output_bits: u32,
    ) -> Self {
        let n_w = out_features * in_features;
        let k = ((1u32 << weight_shift) as f64 / (in_features as f64).sqrt()).round() as i32;
        let k = k.max(1);
        let z = |n: usize| vec![Dyadic::new(0, weight_shift); n];
        let weights = Tensor::from_vec(
            (0..n_w)
                .map(|_| Dyadic::new(-k + rng_range(2 * k as u32) as i32, weight_shift))
                .collect(),
            vec![out_features, in_features],
        );
        let biases = Tensor::from_vec(z(out_features), vec![out_features]);
        let (q_min, q_max) = signed_bounds(output_bits);
        Self {
            in_features,
            out_features,
            weights,
            biases,
            quant_shift,
            output_bits,
            grad_clip: i32::MAX,
            momentum_shift: None,
            input_cache: Tensor::new(),
            output_cache: Tensor::new(),
            input_batch_cache: Tensor::new(),
            output_batch_cache: Tensor::new(),
            clip_cache: Vec::new(),
            clip_batch_cache: Vec::new(),
            grad_w: Tensor::zeros(vec![n_w]),
            grad_b: Tensor::zeros(vec![out_features]),
            vel_w: Tensor::zeros(vec![n_w]),
            vel_b: Tensor::zeros(vec![out_features]),
            grad_samples: 0,
            q_min,
            q_max,
        }
    }

    pub fn with_grad_clip(mut self, c: i32) -> Self {
        self.grad_clip = c.abs();
        self
    }
    pub fn with_momentum(mut self, s: u32) -> Self {
        self.momentum_shift = Some(s);
        self
    }

    #[inline]
    fn w(&self, j: usize, i: usize) -> Dyadic {
        self.weights.data[j * self.in_features + i]
    }
    #[inline]
    fn flat(&self, j: usize, i: usize) -> usize {
        j * self.in_features + i
    }

    fn forward_one(&mut self, input: &TensorView) -> (Tensor, Vec<bool>) {
        assert_eq!(
            input.data.len(),
            self.in_features,
            "Linear: input feature count mismatch"
        );
        let mut out = Vec::with_capacity(self.out_features);
        let mut clipped = Vec::with_capacity(self.out_features);
        for j in 0..self.out_features {
            let mut acc = self.biases.data[j];
            for i in 0..self.in_features {
                acc = add(acc, mul(self.w(j, i), input.data[i], self.quant_shift));
            }
            let (y, was_clipped) = requantize(acc, acc.s, self.q_min, self.q_max);
            out.push(y);
            clipped.push(was_clipped);
        }
        (Tensor::from_vec(out, vec![self.out_features]), clipped)
    }

    fn backward_one(
        &mut self,
        grad_output: &TensorView,
        input: &TensorView,
        output: &TensorView,
        clipped: &[bool],
    ) -> Tensor {
        assert_eq!(
            grad_output.data.len(),
            self.out_features,
            "Linear: output gradient count mismatch"
        );
        assert_eq!(
            input.data.len(),
            self.in_features,
            "Linear: missing or invalid forward cache"
        );
        assert_eq!(
            output.data.len(),
            self.out_features,
            "Linear: missing output cache"
        );
        assert_eq!(
            clipped.len(),
            self.out_features,
            "Linear: missing saturation cache"
        );
        let g_s = grad_output.data.first().map_or(0, |g| g.s);
        let mut gi = vec![Dyadic::new(0, g_s); self.in_features];
        for (j, &is_clipped) in clipped.iter().enumerate() {
            let gr = if is_clipped {
                Dyadic::new(0, grad_output.data[j].s)
            } else {
                ste_requantize(
                    grad_output.data[j],
                    output.data[j].v,
                    self.q_min,
                    self.q_max,
                )
            };
            let gj = Dyadic::new(gr.v.clamp(-self.grad_clip, self.grad_clip), gr.s);
            for (i, gi_value) in gi.iter_mut().enumerate() {
                let idx = self.flat(j, i);

                self.grad_w.data[idx] = add(
                    self.grad_w.data[idx],
                    mul(gj, input.data[i], self.quant_shift),
                );

                *gi_value = add(*gi_value, mul(gj, self.w(j, i), self.quant_shift));
            }
            self.grad_b.data[j] = add(self.grad_b.data[j], gj);
        }
        self.grad_samples = self
            .grad_samples
            .checked_add(1)
            .expect("Linear: sample counter overflow");
        Tensor::from_vec(gi, vec![self.in_features])
    }
}

impl Module for Linear {
    fn training_arithmetic(&self) -> TrainingArithmetic {
        TrainingArithmetic::IntegerOnly
    }
    fn name(&self) -> &'static str {
        "Linear"
    }
    fn describe(&self) -> String {
        let ws = self.weights.data.first().map_or(0, |w| w.s);
        let clip = if self.grad_clip == i32::MAX {
            "off".into()
        } else {
            format!("2^{}", (self.grad_clip as f64).log2() as i32)
        };
        let mom = self
            .momentum_shift
            .map_or("off".into(), |m| format!("shift={m}"));
        format!(
            "Linear(in={}, out={}, w_shift={ws}, q={}, bits={}, clip={clip}, mom={mom})",
            self.in_features, self.out_features, self.quant_shift, self.output_bits
        )
    }

    fn forward(&mut self, input: &TensorView) -> Tensor {
        let (out, clipped) = self.forward_one(input);
        self.input_cache = input.to_tensor();
        self.output_cache = out.clone();
        self.clip_cache = clipped;
        out
    }

    fn backward(&mut self, grad_output: &TensorView) -> Tensor {
        let input = self.input_cache.clone();
        let output = self.output_cache.clone();
        let clipped = self.clip_cache.clone();
        self.backward_one(grad_output, &input.view(), &output.view(), &clipped)
    }

    fn forward_batch(&mut self, inputs: &Tensor) -> Tensor {
        assert!(
            inputs.shape.len() >= 2,
            "Linear: expected [batch, features]"
        );
        assert_eq!(
            inputs.shape[1..].iter().product::<usize>(),
            self.in_features,
            "Linear: batch feature shape mismatch"
        );
        assert_eq!(
            inputs.data.len(),
            inputs.shape[0] * self.in_features,
            "Linear: batch buffer length mismatch"
        );
        let mut clips = Vec::with_capacity(inputs.shape[0]);
        let outputs: Tensor = inputs
            .iter()
            .map(|x| {
                let (out, clipped) = self.forward_one(&x);
                clips.push(clipped);
                out
            })
            .collect();
        self.clip_batch_cache = clips;
        self.input_batch_cache = inputs.clone();
        self.output_batch_cache = outputs.clone();
        outputs
    }

    fn backward_batch(&mut self, grads: &Tensor) -> Tensor {
        let n = self
            .input_batch_cache
            .shape
            .first()
            .copied()
            .expect("Linear: missing batch forward cache");
        assert!(grads.shape.len() >= 2, "Linear: expected batched gradients");
        assert_eq!(grads.shape[0], n, "Linear: gradient batch size mismatch");
        assert_eq!(
            grads.shape[1..].iter().product::<usize>(),
            self.out_features,
            "Linear: gradient feature shape mismatch"
        );
        assert_eq!(
            grads.data.len(),
            n * self.out_features,
            "Linear: gradient buffer mismatch"
        );
        assert_eq!(
            self.clip_batch_cache.len(),
            n,
            "Linear: missing batch saturation cache"
        );
        let inputs = std::mem::take(&mut self.input_batch_cache);
        let outputs = std::mem::take(&mut self.output_batch_cache);
        let clips = std::mem::take(&mut self.clip_batch_cache);
        grads
            .iter()
            .zip(inputs.iter().zip(outputs.iter()))
            .zip(clips.iter())
            .map(|((g, (inp, out)), clipped)| self.backward_one(&g, &inp, &out, clipped))
            .collect()
    }

    fn update(&mut self, lr: u32) {
        if self.grad_samples == 0 {
            return;
        }
        apply_updates_mean(
            &mut self.weights.data,
            &self.grad_w,
            &mut self.vel_w.data,
            lr,
            self.momentum_shift,
            self.grad_samples,
        );
        apply_updates_mean(
            &mut self.biases.data,
            &self.grad_b,
            &mut self.vel_b.data,
            lr,
            self.momentum_shift,
            self.grad_samples,
        );
    }

    fn zero_grad(&mut self) {
        self.grad_samples = 0;
        self.grad_w.data.iter_mut().for_each(|g| g.v = 0);
        self.grad_b.data.iter_mut().for_each(|g| g.v = 0);
    }
}
