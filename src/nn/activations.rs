use crate::nn::{Module};
use crate::dyadic::{
    signed_bounds, Dyadic,
};
use crate::rng::rng_range;


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

