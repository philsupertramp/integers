#[path = "nn/losses.rs"]
pub mod losses;
#[path = "nn/kernels.rs"]
pub mod kernels;
#[path = "nn/optim.rs"]
pub mod optim;
#[path = "nn/rnn.rs"]
pub mod rnn;
#[path = "nn/activations.rs"]
pub mod activations;

use crate::{Tensor, XorShift64};
use crate::nn::optim::{OptimizerConfig, OptimizerState};


// Module trait
// Every building block needs to implement this.
// Containers delegate; primitives do work.
//
#[derive(PartialEq, Debug)]
pub struct ModuleInfo {
    pub name: &'static str,
    pub params: usize,
    pub static_bytes: usize,
    pub children: Vec<ModuleInfo>,
}

impl ModuleInfo {
    pub fn total(&self) -> (usize, usize) {
        let child_totals = self
            .children
            .iter()
            .map(|c| c.total())
            .fold((0, 0), |(p, b), (cp, cb)| (p + cp, b + cb));
        (
            self.params + child_totals.0,
            self.static_bytes + child_totals.1,
        )
    }
}
pub trait Module {
    /// Implements a forward pass for the module.
    ///
    /// Must keep track of internal scale shifts
    /// 1. track `input_shift` for backward pass
    /// 2. track `output_shift` to reconstruct the actual result values in i32
    /// 
    /// # Arguments
    ///
    /// * `input`: Input vector scaled to i8 using `input_shift`
    /// * `input_shift`: The shift scale used to compress the input to i8
    /// * `rng`: RNG state
    ///
    ///
    /// # Examples
    /// ```
    /// use integers::nn::{Module};
    /// use integers::{Tensor, XorShift64};
    ///
    /// struct XPlusN {
    ///     n: i8,
    ///     input_shift: Option<u32>,
    ///     output_shift: Option<u32>,
    /// }
    ///
    /// impl Module for XPlusN {
    ///     fn backward(&mut self, grad: &Tensor<i16>, _grad_shift: Option<u32>) -> Tensor<i16> {
    ///         grad.clone()
    ///     }
    ///     fn forward(&mut self, input: &Tensor<i8>, input_shift: u32, _rng: &mut XorShift64) -> Tensor<i8>{
    ///         self.input_shift = Some(input_shift);
    ///         self.output_shift = Some(input_shift);
    ///
    ///         let mut output = Tensor::new(input.shape.clone());
    ///
    ///         for (val, o) in input.data.iter().zip(output.data.iter_mut()) {
    ///             *o = val + self.n;
    ///         }
    ///         output
    ///     }
    /// }
    ///
    /// let mut mymod = XPlusN{n: 1, input_shift: None, output_shift: None};
    /// let input = Tensor::from_vec(vec![1, 2, 3], vec![3, 1]);
    /// let mut rng = XorShift64::new(420);
    ///
    ///
    /// let output = mymod.forward(&input, 0, &mut rng);
    ///
    /// assert_eq!(output.shape, vec![3, 1]);
    /// assert_eq!(output.data[0], 2);
    /// assert_eq!(output.data[1], 3);
    /// assert_eq!(output.data[2], 4);
    /// ```
    fn forward(&mut self, input: &Tensor<i8>, input_shift: u32, rng: &mut XorShift64) -> Tensor<i8>;
    fn backward(&mut self, grad: &Tensor<i16>, grad_shift: Option<u32>) -> Tensor<i16>;

    /// quantize i32 master weights -> i8 storage. No-op for non-parameterized modules.
    fn sync_weights(&mut self, _rng: &mut XorShift64) {}

    fn init(&mut self, _rng: &mut XorShift64) {}

    /// Apply one optimizer step using internal grad cache. No-op if no grads cached.
    fn step(&mut self, _optim: &dyn OptimizerConfig) {}

    fn get_output_shift(&self) -> u32 {
        0
    }

    fn memory_report(&self) -> (usize, usize) {
        (0, 0)
    }

    fn describe(&self) -> ModuleInfo {
        ModuleInfo {
            name: "unknown",
            params: 0,
            static_bytes: 0,
            children: vec![],
        }
    }

    fn print_summary(&self, info: &ModuleInfo, depth: usize) {
        let indent = "  ".repeat(depth);
        let (total_params, total_bytes) = info.total();
        println!(
            "{}{}  params={} static={}B",
            indent, info.name, total_params, total_bytes
        );
        for child in &info.children {
            self.print_summary(child, depth + 1);
        }
    }
}

pub struct Params {
    /// Weights used for training
    pub master: Tensor<i32>,

    /// Actual weights for inference
    pub storage: Tensor<i8>,

    /// Accumulated gradient from backward pass
    pub grads: Option<Tensor<i32>>,

    /// State of optimizer
    pub state: Option<OptimizerState>,

    /// Determines the quantization bits used when scaling from master -> storage
    /// Typical range: 3-7 (shift values 3-7)
    ///
    /// Examples:
    ///     shift = 4 // for single-layer networks
    ///     shift = 5 // for 2-3 layer networks
    ///     shift = 6 // for deeper networks
    pub quant_shift: u32,

    /// Used to track what shift was applied to inputs
    pub input_shift: Option<u32>,

    /// Used to track what shift was applied to outputs before returning them
    pub output_shift: Option<u32>,
}

impl Params {
    pub fn new(shape: Vec<usize>, shift: u32) -> Self {
        // shift >= 0 given due to u32
        assert!(shift <= 8, "Weight shift must be 1-8, got {}", shift);
        Self {
            master: Tensor::new(shape.clone()),
            storage: Tensor::new(shape),
            grads: None,
            state: None,
            quant_shift: shift,
            input_shift: None,
            output_shift: None,
        }
    }

    pub fn with_shift(mut self, shift: u32) -> Self {
        self.quant_shift = shift;
        self
    }

    pub fn init_uniform(&mut self, rng: &mut XorShift64, range: i32) {
        let range = range.max(1);
        let spread = (2 * range) as u32;
        for w in self.master.data.iter_mut() {
            *w = (rng.gen_range(spread) as i32) - range;
        }
    }

    pub fn init_xavier_uniform(&mut self, rng: &mut XorShift64, fan_in: usize, fan_out: usize) {
        let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
        let limit_i8 = (limit * 127.0).round() as i32;
        self.init_uniform(rng, limit_i8.max(1));
    }

    pub fn sync(&mut self, rng: &mut XorShift64) {
        for (m, s) in self.master.data.iter().zip(self.storage.data.iter_mut()) {
            *s = kernels::stochastic_downcast(*m, self.quant_shift, rng);
        }
    }

    pub fn accumulate_grads(&mut self, new_grads: Tensor<i16>) {
        match self.grads.as_mut() {
            Some(existing) => {
                // Saturating add to avoid wrapping on large accumulated signals
                for (e, n) in existing.data.iter_mut().zip(new_grads.data.iter()) {
                    *e = e.saturating_add(*n as i32);
                }
            }
            None => {
                let mut i32_grads = Tensor::<i32>::new(new_grads.shape.clone());
                for (i, &n) in new_grads.data.iter().enumerate() {
                    i32_grads.data[i] = n as i32;
                }
                self.grads = Some(i32_grads);
            }
        }
    }

    pub fn zero_grads(&mut self) {
        self.grads = None;
    }

    pub fn step(&mut self, optim: &dyn OptimizerConfig) {
        if let Some(grads) = self.grads.take() {
            let state = self
                .state
                .get_or_insert_with(|| optim.init_state(self.master.len()));
            optim.update(&mut self.master.data, &grads.data, state);
        }
    }

    pub fn memory_bytes(&self) -> usize {
        self.master.memory_bytes() + self.storage.memory_bytes()
    }
}

fn compute_shift_for_max(max_magnitude: u32) -> u32 {
    let mut shift: u32 = 0;
    while (max_magnitude >> shift) > 127 && shift < 8 {
        shift += 1;
    }
    shift.max(0).min(8)
}

pub trait HasWeights {
    /// Get all weight parameters (master i32 tensors) in this module.
    /// Useful for analysis, initialization, and automatic shift detection.
    fn get_all_weights(&self) -> Vec<&Tensor<i32>>;
    
    /// Infer an appropriate scale_shift based on weight magnitudes.
    /// Returns a shift value in the range [0, 8].
    fn infer_scale_shift(&self) -> u32 {
        let all_weights = self.get_all_weights();
        if all_weights.is_empty() {
            return 4; // default middle ground
        }
        
        // Find max magnitude across all weight tensors
        let max_magnitude = all_weights
            .iter()
            .flat_map(|t| &t.data)
            .map(|&w| w.abs() as u32)
            .max()
            .unwrap_or(1);
        
        if max_magnitude == 0 {
            return 4; // weights not yet initialized
        }
        
        compute_shift_for_max(max_magnitude)
    }
}

pub struct Linear {
    /// all weights are of Transposed Form [out, in] for memory reasons
    pub weights: Params,
    pub bias: Params,
    pub cache: Vec<Tensor<i8>>,
}

impl Linear {
    pub fn new(input_dim: usize, output_dim: usize, scale_shift: u32) -> Self {
        Self {
            weights: Params::new(vec![output_dim, input_dim], scale_shift),
            bias: Params::new(vec![output_dim], scale_shift),
            cache: Vec::new(),
        }
    }

    pub fn init_xavier(&mut self, rng: &mut XorShift64) {
        let fan_in = self.weights.master.shape[1];
        let fan_out = self.weights.master.shape[0];
        self.weights.init_xavier_uniform(rng, fan_in, fan_out);
        for b in self.bias.master.data.iter_mut() {
            *b = 0;
        }
    }
}

impl Module for Linear {
    fn sync_weights(&mut self, rng: &mut XorShift64) {
        self.weights.sync(rng);
        self.bias.sync(rng);
    }

    fn get_output_shift(&self) -> u32 {
        self.weights.output_shift.unwrap_or(0)
    }

    fn forward(&mut self, raw_input: &Tensor<i8>, input_shift: u32, rng: &mut XorShift64) -> Tensor<i8> {
        assert_eq!(
            raw_input.shape[1], self.weights.storage.shape[1],
            "Linear::forward: Input in wrong dimension for weights! {} vs {}",
            raw_input.shape[1], self.weights.storage.shape[1]
        );
        let batch = raw_input.shape[0];
        let input_dim = raw_input.shape[1];
        let output_dim = self.weights.storage.shape[0];

        let input = raw_input << input_shift;

        self.cache.push(input.clone());
        let mut out = Tensor::new(vec![batch, output_dim]);
        let mut out_raw = Tensor::<i32>::new(vec![batch, output_dim]);

        for b in 0..batch {
            let in_row = &input.data[b * input_dim..(b + 1) * input_dim];
            for o in 0..output_dim {
                let w_row = &self.weights.storage.data[o * input_dim..(o + 1) * input_dim];
                #[cfg(target_arch = "aarch64")]
                let raw_val = unsafe {
                    if input_dim % 16 == 0 {
                        kernels::arm_neon::dot_product_neon_raw(in_row, w_row)
                    } else {
                        kernels::dot_product_scalar(in_row, w_row)
                    }
                };
                #[cfg(not(target_arch = "aarch64"))]
                let raw_val = kernels::dot_product_scalar(in_row, w_row);
                let acc = raw_val + self.bias.storage.data[o] as i32;
                out_raw.data[b * output_dim + o] = acc;
            }
        }

        let max_val = out_raw.data
            .iter()
            .map(|x| x.abs() as u32)
            .max()
            // TODO: (SHIFT#1) Potentially an issue. Empty input is shifted?
            .unwrap_or(0);

        let output_shift = compute_shift_for_max(max_val);
        self.weights.input_shift = Some(input_shift);
        self.weights.output_shift = Some(output_shift);

        for (raw_out_val, out_val) in out_raw.data.iter().zip(&mut out.data) {
            *out_val = kernels::stochastic_downcast(*raw_out_val, output_shift, rng);
        }

        out
    }

    fn backward(&mut self, shifted_grad_output: &Tensor<i16>, gradient_shift: Option<u32>) -> Tensor<i16> {
        // not required to shift anymore, we did that in the forward pass already
        let input = self.cache.pop().expect("Linear::backward: Backward called without forward");

        // gradient_shift: The shift applied to the incoming gradient (`grad_out`)
        // input_shift: The shift applied to `input`
        // output_shift: The shift applied to the processed `input`
        // TODO:
        //      - rescale grad_out with gradient_shift
        // used to shift back into the original input space
        let input_shift = self.weights.input_shift.expect("Linear::backward: Backward called without forward.");

        let output_shift = self.weights.output_shift.expect("Linear::backward: Backward called without forward.");

        // Mathematically required for the chain rule:
        let gshift = gradient_shift.unwrap_or(0);

        let batch = input.shape[0];
        let input_dim = input.shape[1];
        let output_dim = self.weights.storage.shape[0];

        let mut grad_input = Tensor::<i16>::new(vec![batch, input_dim]);
        let mut grad_weights = Tensor::<i16>::new(vec![output_dim, input_dim]);
        let mut grad_bias = Tensor::<i16>::new(vec![output_dim]);

        let grad_output = shifted_grad_output >> output_shift;

        for b in 0..batch {
            for o in 0..output_dim {
                let g = grad_output.data[b * output_dim + o];
                if g == 0 {
                    continue;
                }

                // Bias update is an accumulator, so apply gshift
                grad_bias.data[o] = grad_bias.data[o].saturating_add(g);

                for i in 0..input_dim {
                    let x = input.data[b * input_dim + i];

                    let dw = kernels::mul_mixed_scalar(g, x);

                    // --- Weight Gradient ---
                    // dL/dw = grad_output * input_i8
                    // Both terms are quantized:
                    //  grad_output is quantized by output_shift
                    //  input_i8 is quantized by input_shift
                    // Weight update is an accumulator, so apply gshift
                    grad_weights.data[o * input_dim + i] =
                        grad_weights.data[o * input_dim + i].saturating_add(dw as i16);

                    let w = self.weights.storage.data[o * input_dim + i];

                    // --- Input Gradient ---
                    // dL/d(input) = grad_output * weight_i8
                    // Both terms are quantized:
                    //  grad_output is quantized by output_shift
                    //  input_i8 is quantized by weight_shift
                    let dx = kernels::mul_mixed_scalar(g, w);

                    // Input gradient flows back through the network, apply wshift!
                    grad_input.data[b * input_dim + i] =
                        grad_input.data[b * input_dim + i].saturating_add(dx as i16);
                }
            }
        }

        let grad_input_shifted = grad_input << input_shift;
        self.weights.accumulate_grads(grad_weights << gshift);
        self.bias.accumulate_grads(grad_bias << gshift);

        grad_input_shifted
    }

    fn step(&mut self, optim: &dyn OptimizerConfig) {
        self.weights.step(optim);
        self.bias.step(optim);
    }

    fn memory_report(&self) -> (usize, usize) {
        let stat = self.weights.memory_bytes() + self.bias.memory_bytes();
        let dyn_ = self.cache.iter().map(|t| t.memory_bytes()).sum();
        (stat, dyn_)
    }

    fn describe(&self) -> ModuleInfo {
        let in_dim = self.weights.storage.shape[1];
        let out_dim = self.weights.storage.shape[0];
        ModuleInfo {
            name: "Linear",
            params: (in_dim * out_dim) + out_dim, // weights + bias
            static_bytes: self.weights.memory_bytes() + self.bias.memory_bytes(),
            children: vec![],
        }
    }

    fn init(&mut self, rng: &mut XorShift64) {
        self.init_xavier(rng);

        // After initialization, infer the appropriate shift
        let inferred_shift = self.infer_scale_shift();
        self.weights.quant_shift = inferred_shift;
        self.bias.quant_shift = inferred_shift;
    }
}

impl HasWeights for Linear {
    fn get_all_weights(&self) -> Vec<&Tensor<i32>> {
        vec![&self.weights.master, &self.bias.master]
    }
}


pub struct Sequential {
    pub modules: Vec<Box<dyn Module>>,
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Sequential {
    pub fn new() -> Self {
        Self { modules: vec![] }
    }

    pub fn add(&mut self, m: impl Module + 'static) -> &mut Self {
        self.modules.push(Box::new(m));
        self
    }

    pub fn init_all(&mut self, rng: &mut XorShift64) {
        for module in &mut self.modules {
            module.init(rng);
        }
    }

    pub fn analyze_scale_shifts(&self) {

    }
}

impl Module for Sequential {
    fn forward(&mut self, input: &Tensor<i8>, mut input_shift: u32, rng: &mut XorShift64) -> Tensor<i8> {
        let mut output = input.clone();
        for m in self.modules.iter_mut() {
            output = m.forward(&output, input_shift, rng);
            input_shift = m.get_output_shift();
        }
        output
    }
    fn backward(&mut self, grad_output: &Tensor<i16>, grad_shift: Option<u32>) -> Tensor<i16> {
        let mut output = grad_output.clone();
        for m in self.modules.iter_mut().rev() {
            output = m.backward(&output, grad_shift);
        }
        output
    }
    fn sync_weights(&mut self, rng: &mut XorShift64) {
        for m in self.modules.iter_mut() {
            m.sync_weights(rng)
        }
    }
    fn step(&mut self, optim: &dyn OptimizerConfig) {
        for m in self.modules.iter_mut() {
            m.step(optim);
        }
    }

    fn memory_report(&self) -> (usize, usize) {
        self.modules
            .iter()
            .map(|m| m.memory_report())
            .fold((0, 0), |(s, d), (ms, md)| (s + ms, d + md))
    }

    fn describe(&self) -> ModuleInfo {
        ModuleInfo {
            name: "Sequential",
            params: 0,
            static_bytes: 0,
            children: self.modules.iter().map(|m| m.describe()).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::optim::*;

    // Linear tests
    #[test]
    fn test_linear_new() {
        let lin = Linear::new(4, 8, 2);

        assert_eq!(lin.weights.master.len(), 4 * 8);
        assert_eq!(lin.weights.storage.len(), 4 * 8);
        assert_eq!(lin.bias.master.len(), 8);
        assert_eq!(lin.weights.quant_shift, 2);
        assert_eq!(lin.cache, Vec::new());
        assert_eq!(lin.weights.grads, None);
        assert_eq!(lin.bias.grads, None);
    }

    #[test]
    fn test_linear_sync_weights() {
        let mut lin = Linear::new(2, 2, 0);
        lin.weights.master.data[0] = 10;
        lin.weights.master.data[3] = 10;
        let mut rng = XorShift64 { state: 420 };
        lin.sync_weights(&mut rng);

        assert_eq!(lin.weights.storage.data[0], 10);
        assert_eq!(lin.weights.storage.data[1], 0);
        assert_eq!(lin.weights.storage.data[2], 0);
        assert_eq!(lin.weights.storage.data[3], 10);

        lin.weights.quant_shift = 1;
        lin.sync_weights(&mut rng);

        assert_eq!(lin.weights.storage.data[0], 5);
        assert_eq!(lin.weights.storage.data[1], 0);
        assert_eq!(lin.weights.storage.data[2], 0);
        assert_eq!(lin.weights.storage.data[3], 5);

        lin.weights.quant_shift = 2;
        lin.sync_weights(&mut rng);

        assert_eq!(lin.weights.storage.data[0], 2);
        assert_eq!(lin.weights.storage.data[1], 0);
        assert_eq!(lin.weights.storage.data[2], 0);
        assert_eq!(lin.weights.storage.data[3], 2);

        lin.weights.quant_shift = 3;
        lin.sync_weights(&mut rng);

        assert_eq!(lin.weights.storage.data[0], 2);
        assert_eq!(lin.weights.storage.data[1], 0);
        assert_eq!(lin.weights.storage.data[2], 0);
        assert_eq!(lin.weights.storage.data[3], 1);
    }
    #[test]
    fn test_auto_scale_shift_detection() {
        let mut rng = XorShift64::new(42);
        let mut layer = Linear::new(10, 5, 0); // start with shift=0
        
        // Initialize with Xavier distribution
        layer.init_xavier(&mut rng);
        
        // Auto-detect appropriate shift
        let detected_shift = layer.infer_scale_shift();
        println!("Detected shift: {}", detected_shift);
        
        layer.weights.quant_shift = detected_shift;
        layer.bias.quant_shift = detected_shift;
        
        // Now safely quantize
        layer.sync_weights(&mut rng);
        
        // Verify all weights fit in i8 range
        for &w in &layer.weights.storage.data {
            assert!(w >= -128 && w <= 127);
        }
    }
    #[test]
    fn test_linear_forward() {
        let mut lin = Linear::new(2, 2, 0);
        lin.weights.master.data[0] = 1;
        lin.weights.master.data[3] = 1;
        let mut rng = XorShift64 { state: 420 };
        lin.sync_weights(&mut rng);

        let mut input = Tensor::new(vec![1, 2]);
        input.data[0] = 10;
        input.data[1] = 20;
        let out = lin.forward(&input, 0, &mut rng);
        let output_shift = lin.weights.output_shift.expect("Does not exist.");

        assert_eq!(out.data[0] << output_shift, 10);
        assert_eq!(out.data[1] << output_shift, 20);
    }

    #[test]
    fn test_linear_forward_saturation() {
        let mut lin = Linear::new(2, 2, 0);
        lin.weights.master.data[0] = 126;
        lin.weights.master.data[3] = 126;
        let mut rng = XorShift64 { state: 420 };
        lin.sync_weights(&mut rng);

        let mut input = Tensor::new([1, 2].to_vec());
        input.data[0] = 126;
        input.data[1] = 126;
        let out = lin.forward(&input, 0, &mut rng);
        let output_shift = lin.weights.output_shift.expect("Does not exist.");

        // clamping around 127
        assert_eq!(out.data[0] << output_shift, 0);
        assert_eq!(out.data[1] << output_shift, 0);
        assert_eq!(output_shift, 7);
    }

    #[test]
    fn test_linear_forward_shape() {
        let mut lin = Linear::new(2, 3, 0);
        let mut rng = XorShift64 { state: 420 };
        lin.sync_weights(&mut rng);

        let input = Tensor::new([10, 2].to_vec());
        let out = lin.forward(&input, 0, &mut rng);

        assert_eq!(out.shape[0], 10);
        assert_eq!(out.shape[1], 3);
    }

    #[test]
    fn test_linear_forward_identity() {
        // Linear 2->2, Scale 0 (Identity mapping potential)
        let mut lin = Linear::new(2, 2, 0);

        // Identity Matrix in Master Weights [1, 0] / [0, 1]
        lin.weights.master.data[0] = 1;
        lin.weights.master.data[3] = 1;

        let mut rng = XorShift64::new(420);
        lin.sync_weights(&mut rng);

        let mut input = Tensor::from_vec(vec![10, 20], vec![1, 2]);

        let out = lin.forward(&input, 0, &mut rng);
        let output_shift = lin.weights.output_shift.expect("Does not exist.");

        assert_eq!(out.data[0] << output_shift, 10);
        assert_eq!(out.data[1] << output_shift, 20);
        assert_eq!(out.data[0], 10);
        assert_eq!(out.data[1], 20);
    }

    #[test]
    fn test_linear_forward_identity_with_shift() {
        // Linear 2->2, Scale 0 (Identity mapping potential)
        let mut lin = Linear::new(2, 2, 0);

        // Identity Matrix in Master Weights [2, 0] / [0, 2]
        lin.weights.master.data[0] = 2;
        lin.weights.master.data[3] = 2;

        let mut rng = XorShift64::new(420);
        lin.sync_weights(&mut rng);

        let mut input = Tensor::from_vec(vec![127, -128], vec![1, 2]);

        let out = lin.forward(&input, 0, &mut rng);
        let output_shift = lin.weights.output_shift.expect("Does not exist.");

        assert_eq!(out.data[0] << output_shift, 0);
        assert_eq!(out.data[1] << output_shift, 0);
        assert_eq!(out.data[0], 64);
        assert_eq!(out.data[1], -64);

        input = Tensor::from_vec(vec![127, 64], vec![1, 2]);

        let out = lin.forward(&input, 0, &mut rng);
        let output_shift = lin.weights.output_shift.expect("Does not exist.");

        assert_eq!(out.data[0] << output_shift, -2);
        assert_eq!(out.data[1] << output_shift, -128);
        assert_eq!(out.data[0], 127);
        assert_eq!(out.data[1], 64);
    }

    #[test]
    fn test_linear_forward_shape_batch() {
        let mut lin = Linear::new(2, 3, 0);
        let mut rng = XorShift64::new(420);
        lin.sync_weights(&mut rng);

        let input = Tensor::new(vec![10, 2]);
        let out = lin.forward(&input, 0, &mut rng);

        assert_eq!(out.shape[0], 10);
        assert_eq!(out.shape[1], 3);
        assert_eq!(out.data.len(), 30);
    }

    #[test]
    fn test_linear_backward_shapes() {
        // Batch 2, Input 4 -> Output 3
        let mut lin = Linear::new(4, 3, 0);
        let mut rng = XorShift64::new(420);

        let input = Tensor::new(vec![2, 4]); // [Batch, In]
        let grad_out = Tensor::new(vec![2, 3]); // [Batch, Out]

        lin.forward(&input, 0, &mut rng);
        lin.cache = vec![input.clone()];

        let d_x = lin.backward(&grad_out, None);
        let d_w = lin.weights.grads.unwrap().clone();

        // Check shapes
        assert_eq!(d_x.shape, vec![2, 4]); // [Batch, In]
        assert_eq!(d_w.shape, vec![3, 4]); // [Out, In]
    }

    #[test]
    #[should_panic(
        expected = "assertion `left == right` failed: Linear::forward: Input in wrong dimension for weights! 1 vs 2\n  left: 1\n right: 2"
    )]
    fn test_linear_forward_wrong_dimensional_input() {
        let mut lin = Linear::new(2, 3, 0);
        let mut rng = XorShift64::new(420);
        lin.sync_weights(&mut rng);

        let input = Tensor::from_vec(vec![2; 10], vec![10, 1]);
        lin.forward(&input, 0, &mut rng);
    }

    #[test]
    fn test_linear_memory_report() {
        let mut rng = XorShift64::new(777);
        let mut l1 = Linear::new(2, 1, 2);
        let x = Tensor::from_vec(vec![0, 0], vec![1, 2]);
        l1.forward(&x, 2, &mut rng);

        let (a, b) = l1.memory_report();
        assert_eq!(a, 15);
        assert_eq!(b, 2);

        let mut l2 = Linear::new(2, 10, 2);
        l2.forward(&x, 2, &mut rng);

        let (a2, b2) = l2.memory_report();
        assert_eq!(a2, 150);
        assert_eq!(b2, 2);
    }

    #[test]
    fn test_linear_step() {
        let mut rng = XorShift64::new(777);
        let mut l1 = Linear::new(2, 1, 1);
        let mut optim = SGDConfig::new().with_learn_rate(1.0);
        l1.weights.master.data[0] = 111;
        l1.weights.master.data[1] = 222;
        l1.sync_weights(&mut rng);

        assert_eq!(l1.weights.master.data[0], 111);
        assert_eq!(l1.weights.master.data[1], 222);
        assert_eq!(l1.weights.storage.data[0], 55);
        assert_eq!(l1.weights.storage.data[1], 111);

        // no gradient -> no change
        l1.step(&mut optim);
        assert_eq!(l1.weights.master.data[0], 111);
        assert_eq!(l1.weights.master.data[1], 222);
        assert_eq!(l1.weights.storage.data[0], 55);
        assert_eq!(l1.weights.storage.data[1], 111);

        let x = Tensor::from_vec(vec![1, 1], vec![1, 2]);

        l1.forward(&x, 1, &mut rng);
        l1.step(&mut optim);

        assert_eq!(l1.weights.master.data[0], 111);
        assert_eq!(l1.weights.master.data[1], 222);
        assert_eq!(l1.weights.storage.data[0], 55);
        assert_eq!(l1.weights.storage.data[1], 111);

        let x2 = Tensor::from_vec(vec![2, 2], vec![1, 2]);

        l1.sync_weights(&mut rng);
        let _preds = l1.forward(&x2, 1, &mut rng);

        // 3. Backpropagate "a" gradient 
        let grad_out = Tensor::<i32>::from_vec(vec![126, 2], vec![2, 1]);
        l1.weights.grads = Some(grad_out);

        // with a gradient we have change
        l1.step(&mut optim);

        assert_eq!(l1.weights.master.data[0], -15);
        assert_eq!(l1.weights.master.data[1], 220);
        assert_eq!(l1.weights.storage.data[0], 56);
        assert_eq!(l1.weights.storage.data[1], 111);
    }

    #[test]
    fn test_linear_step_with_shifting() {
        let mut l1 = Linear::new(2, 1, 2);
        let mut optim = SGDConfig::new().with_learn_rate(0.75);
        l1.weights.master.data[0] = 111;
        l1.weights.master.data[1] = 222;

        l1.step(&mut optim);

        assert_eq!(l1.weights.master.data[0], 111);
        assert_eq!(l1.weights.master.data[1], 222);
        assert_eq!(l1.weights.storage.data[0], 0);
        assert_eq!(l1.weights.storage.data[1], 0);
    }
    // Examples

    #[test]
    fn test_summary() {
        let model = Sequential {
            modules: vec![
                Box::new(Linear::new(2, 8, 2)),
                Box::new(activations::ReLU::new()),
                Box::new(Linear::new(8, 1, 2)),
            ],
        };

        model.print_summary(&model.describe(), 1);
    }

}


