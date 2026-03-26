use crate::nn::{Module, apply_updates};
use crate::dyadic::{
    add, mul, requantize, signed_bounds, ste_requantize, stochastic_round, Dyadic,
};
use crate::rng::rng_range;


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
            in_features,
            out_features,
            weights: weights,
            biases: z(out_features),
            quant_shift,
            output_bits,
            grad_clip: i32::MAX,
            momentum_shift: None,
            input_cache: Vec::new(),
            output_cache: Vec::new(),
            input_batch_cache: Vec::new(),
            output_batch_cache: Vec::new(),
            grad_w: z(n_w),
            grad_b: z(out_features),
            vel_w: z(n_w),
            vel_b: z(out_features),
            q_min: q_min,
            q_max: q_max,
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

