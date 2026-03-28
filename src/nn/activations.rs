use crate::nn::{Module};
use crate::dyadic::{
    signed_bounds, Dyadic, Tensor, TensorView
};
use crate::rng::rng_range;


pub struct ReLU {
    mask_bits: Vec<u64>,
    batch_size: usize,
    sample_size: usize,
}

impl ReLU {
    pub fn new() -> Self { 
        Self { 
            mask_bits: Vec::new(), 
            batch_size: 0,
            sample_size: 0,
        } 
    }
}

impl Module for ReLU {
    /// Forward pass for a single sample
    fn forward(&mut self, input: &TensorView) -> Tensor {
        let sample_size = input.data.len();
        
        self.batch_size = 1;
        self.sample_size = sample_size;
        self.mask_bits.clear();
        self.mask_bits.reserve((sample_size + 63) / 64);

        let mut current_u64 = 0u64;
        let mut bit_pos = 0;

        let output: Vec<Dyadic> = input.data.iter().map(|x| {
            if x.v > 0 {
                current_u64 |= 1u64 << bit_pos;
            }
            bit_pos += 1;
            if bit_pos == 64 {
                self.mask_bits.push(current_u64);
                current_u64 = 0;
                bit_pos = 0;
            }
            if x.v > 0 { *x } else { Dyadic::new(0, x.s) }
        }).collect();

        if bit_pos > 0 {
            self.mask_bits.push(current_u64);
        }

        Tensor::from_vec(output, input.shape.to_vec())
    }

    /// Backward pass for a single sample
    fn backward(&mut self, grad: &TensorView) -> Tensor {
        assert_eq!(self.batch_size, 1, "Use backward_batch for batched gradients");
        assert_eq!(grad.data.len(), self.sample_size);
        
        let grad_output: Vec<Dyadic> = grad.data.iter().enumerate().map(|(i, &g)| {
            let bit_idx = i % 64;
            let u64_idx = i / 64;
            let is_active = (self.mask_bits[u64_idx] >> bit_idx) & 1 == 1;
            
            if is_active { g } else { Dyadic::new(0, g.s) }
        }).collect();
        
        Tensor::from_vec(grad_output, grad.shape.to_vec())
    }

    /// Forward pass for a batch [B, N]
    fn forward_batch(&mut self, batch: &Tensor) -> Tensor {
        assert!(batch.shape.len() >= 1, "Expected batched input");
        
        let batch_size = batch.shape[0];
        let sample_size = batch.data.len() / batch_size;
        
        self.batch_size = batch_size;
        self.sample_size = sample_size;
        
        let bits_per_sample = (sample_size + 63) / 64;
        self.mask_bits.clear();
        self.mask_bits.reserve(batch_size * bits_per_sample);

        let mut output_data = Vec::with_capacity(batch.data.len());
        
        for sample_idx in 0..batch_size {
            let start = sample_idx * sample_size;
            let end = start + sample_size;
            let sample = &batch.data[start..end];
            
            let mut current_u64 = 0u64;
            let mut bit_pos = 0;
            
            for &x in sample {
                if x.v > 0 {
                    current_u64 |= 1u64 << bit_pos;
                    output_data.push(x);
                } else {
                    output_data.push(Dyadic::new(0, x.s));
                }
                
                bit_pos += 1;
                if bit_pos == 64 {
                    self.mask_bits.push(current_u64);
                    current_u64 = 0;
                    bit_pos = 0;
                }
            }
            
            if bit_pos > 0 {
                self.mask_bits.push(current_u64);
            }
        }
        
        Tensor::from_vec(output_data, batch.shape.to_vec())
    }

    /// Backward pass for a batch
    fn backward_batch(&mut self, grad: &Tensor) -> Tensor {
        assert_eq!(grad.data.len(), self.batch_size * self.sample_size);
        
        let bits_per_sample = (self.sample_size + 63) / 64;
        let mut grad_output = Vec::with_capacity(grad.data.len());
        
        for sample_idx in 0..self.batch_size {
            let grad_start = sample_idx * self.sample_size;
            let mask_start = sample_idx * bits_per_sample;
            
            for elem_idx in 0..self.sample_size {
                let grad_val = grad.data[grad_start + elem_idx];
                
                let bit_idx = elem_idx % 64;
                let u64_idx = mask_start + (elem_idx / 64);
                let is_active = (self.mask_bits[u64_idx] >> bit_idx) & 1 == 1;
                
                grad_output.push(if is_active { 
                    grad_val 
                } else { 
                    Dyadic::new(0, grad_val.s) 
                });
            }
        }
        
        Tensor::from_vec(grad_output, grad.shape.to_vec())
    }

    fn name(&self) -> &'static str { "ReLU" }

    fn update(&mut self, _: u32) {}
    fn zero_grad(&mut self) {}
}

impl Default for ReLU { fn default() -> Self { Self::new() } }


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

    fn softmax_forward(input: &TensorView, shift: u32, probs: &mut Vec<f64>) -> Tensor {
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

    fn forward(&mut self, input: &TensorView) -> Tensor {
        Self::softmax_forward(input, self.output_shift, &mut self.last_probs)
    }

    fn backward(&mut self, grad: &TensorView) -> Tensor { grad.to_tensor() }
    fn update(&mut self, _: u32) {}
    fn zero_grad(&mut self) {}
}

