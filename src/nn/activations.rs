use crate::nn::{Module};
use crate::dyadic::{
    signed_bounds, Dyadic, Tensor, TensorView
};
use crate::rng::rng_range;


pub struct ReLU {
    mask:        Vec<bool>,
    batch_masks: Vec<Vec<bool>>,
}

impl ReLU {
    pub fn new() -> Self { Self { mask: Vec::new(), batch_masks: Vec::new() } }

    fn apply_forward(input: TensorView, mask: &mut Vec<bool>) -> Tensor {
        *mask = input.data.iter().map(|x| x.v > 0).collect();
        let output = input.data.iter().map(|x| if x.v > 0 { *x } else { Dyadic::new(0, x.s) }).collect();
        Tensor::from_vec(output, input.shape.to_vec())
    }

    fn apply_backward(grad: TensorView, mask: &[bool]) -> Tensor {
        let grad_output = grad.data.iter().zip(mask).map(|(&g, &a)| if a { g } else { Dyadic::new(0, g.s) }).collect();
        Tensor::from_vec(grad_output, grad.shape.to_vec())
    }
}

impl Default for ReLU { fn default() -> Self { Self::new() } }

impl Module for ReLU {
    fn name(&self) -> &'static str { "ReLU" }

    fn forward(&mut self, input: TensorView) -> Tensor {
        Self::apply_forward(input, &mut self.mask)
    }

    fn backward(&mut self, grad: TensorView) -> Tensor {
        Self::apply_backward(grad, &self.mask)
    }

    fn forward_batch(&mut self, inputs: &Tensor) -> Tensor {
        let mut masks = Vec::with_capacity(inputs.len());
        let outputs: Tensor = inputs.iter().map(|x| {
            let mut m = Vec::new();
            let out = Self::apply_forward(x, &mut m);
            masks.push(m);
            out
        }).collect();
        self.batch_masks = masks;
        outputs
    }

    fn backward_batch(&mut self, grads: &Tensor) -> Tensor {
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

    fn softmax_forward(input: TensorView, shift: u32, probs: &mut Vec<f64>) -> Tensor {
        let logits: Vec<f64> = input.data.iter().map(|x| x.to_f64()).collect();
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&z| (z - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        *probs = exps.iter().map(|&e| e / sum).collect();
        let scale = (1u32 << shift) as f64;
        let (_, mx) = signed_bounds(shift + 1);
        let output = probs.iter().map(|&p| Dyadic::new((p * scale).round().clamp(0.0, mx as f64) as i32, shift)).collect();
        Tensor::from_vec(output, input.shape.to_vec())
    }
}

impl Module for Softmax {
    fn name(&self) -> &'static str { "Softmax" }
    fn describe(&self) -> String { format!("Softmax(shift={})", self.output_shift) }

    fn forward(&mut self, input: TensorView) -> Tensor {
        Self::softmax_forward(input, self.output_shift, &mut self.last_probs)
    }

    fn backward(&mut self, grad: TensorView) -> Tensor { grad.to_tensor() }
    fn update(&mut self, _: u32) {}
    fn zero_grad(&mut self) {}
}

