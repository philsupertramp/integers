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

use crate::{Tensor, XorShift64, Scalar, Numeric};
use crate::nn::optim::{OptimizerConfig, OptimizerState};
use crate::{checked_add_counting};

use std::any::Any;


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

pub trait Module<S: Scalar>: Any {
    /// Implements a forward pass for the module.
    ///
    /// Must keep track of internal scale shifts
    /// 1. track `input_shift` for backward pass
    /// 2. track `output_shift` to reconstruct the actual result values in i32
    /// 
    /// # Arguments
    ///
    /// * `input`: Input vector scaled to i32 using `input_shift`
    /// * `input_shift`: The shift scale used to compress the input to i32
    /// * `rng`: RNG state
    ///
    ///
    /// # Examples
    /// ```
    /// use integers::nn::{Module};
    /// use integers::{Tensor, XorShift64, Scalar, Numeric};
    ///
    /// use std::any::Any;
    ///
    /// struct XPlusN {
    ///     n: i32,
    ///     input_shift: Option<u32>,
    ///     output_shift: Option<u32>,
    /// }
    ///
    /// impl<S: Scalar + 'static> Module<S> for XPlusN {
    ///     fn backward(&mut self, grad: &Tensor<i32>, s_g: u32) -> (Tensor<i32>, u32) {
    ///         (grad.clone(), s_g)
    ///     }
    ///     fn forward(&mut self, input: &Tensor<i32>, s_x: u32, _rng: &mut XorShift64) -> (Tensor<i32>, u32) {
    ///         self.input_shift = Some(s_x);
    ///         self.output_shift = Some(s_x);
    ///
    ///         let mut output = Tensor::new(input.shape.clone());
    ///
    ///         for (val, o) in input.data.iter().zip(output.data.iter_mut()) {
    ///             *o = val + self.n;
    ///         }
    ///         (output, s_x)
    ///     }
    ///     fn as_any(&self) -> &dyn Any { self }
    ///     fn as_any_mut(&mut self) -> &mut dyn Any { self }
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
    fn forward(&mut self, input: &Tensor<S>, s_x: u32, rng: &mut XorShift64) -> (Tensor<S>, u32);
    fn backward(&mut self, grad: &Tensor<S::Acc>, s_g: u32) -> (Tensor<S::Acc>, u32);

    /// quantize i32 master weights -> i32 storage. No-op for non-parameterized modules.
    fn sync_weights(&mut self, _rng: &mut XorShift64) {}

    fn init(&mut self, _rng: &mut XorShift64) {}

    /// sets all gradients to zero, e.g. at the beginning of a training iteration
    fn zero_grads(&mut self) {}

    /// Apply one optimizer step using internal grad cache. No-op if no grads cached.
    fn step(&mut self, _optim: &dyn OptimizerConfig<S>) {}

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

    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

#[derive(Debug)]
pub struct Params<S: Scalar> {
    /// Weights used for training
    pub master: Tensor<S::Acc>,

    /// Actual weights for inference
    pub storage: Tensor<S>,

    /// Accumulated gradient from backward pass
    pub grads: Option<Tensor<S::Acc>>,

    /// State of optimizer
    pub state: Option<OptimizerState<S>>,

    /// Determines the quantization bits used when scaling from master -> storage
    /// Typical range: 3-7 (shift values 3-7)
    ///
    /// Examples:
    ///     shift = 4 // for single-layer networks
    ///     shift = 5 // for 2-3 layer networks
    ///     shift = 6 // for deeper networks
    pub quant_shift: u32,
}

impl<S: Scalar> Params<S> {
    pub fn new(shape: Vec<usize>, shift: u32) -> Self {
        // shift >= 0 given due to u32
        assert!(shift <= 8, "Weight shift must be 0-8, got {}", shift);
        Self {
            master: Tensor::<S::Acc>::new(shape.clone()),
            storage: Tensor::<S>::new(shape),
            grads: None,
            state: None,
            quant_shift: shift,
        }
    }

    pub fn with_shift(mut self, shift: u32) -> Self {
        self.quant_shift = shift;
        self
    }

    pub fn init_uniform(&mut self, rng: &mut XorShift64, range: S::Acc) {
        for w in self.master.data.iter_mut() {
            *w = S::random_uniform(rng, range);
        }
    }

    pub fn init_xavier_uniform(&mut self, rng: &mut XorShift64, fan_in: usize, fan_out: usize) {
        let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
        let range = S::from_f64(limit);
        self.init_uniform(rng, range);
    }

    pub fn sync(&mut self, rng: &mut XorShift64) {
        for (m, s) in self.master.data.iter().zip(self.storage.data.iter_mut()) {
            *s = S::downcast(*m, self.quant_shift, rng);
        }
    }

    pub fn accumulate_grads(&mut self, new_grads: &Tensor<S::Acc>) {
        match self.grads.as_mut() {
            Some(existing) => {
                // Saturating add to avoid wrapping on large accumulated signals
                for (e, n) in existing.data.iter_mut().zip(new_grads.data.iter()) {
                    *e = e.add(*n);
                }
            }
            None => {
                let mut grads = Tensor::<S::Acc>::new(new_grads.shape.clone());
                for (i, &n) in new_grads.data.iter().enumerate() {
                    grads.data[i] = n;
                }
                self.grads = Some(grads);
            }
        }
    }

    pub fn zero_grads(&mut self) {
        self.grads = None;
        self.state = None;
    }

    pub fn step(&mut self, optim: &dyn OptimizerConfig<S>) {
        if let Some(grads) = self.grads.take() {
            let state = self
                .state
                .get_or_insert_with(|| optim.init_state(self.master.len()));
            optim.update(&mut self.master.data, &grads.data, state, self.quant_shift);
        }
    }

    pub fn memory_bytes(&self) -> usize {
        self.master.memory_bytes() + self.storage.memory_bytes()
    }
}

pub fn compute_shift_for_max(max_magnitude: u32) -> u32 {
    let mut shift: u32 = 0;
    while (max_magnitude >> shift) > 127 && shift < 8 {
        shift += 1;
    }
    shift.max(0).min(8)
}

pub trait HasWeights<S: Scalar> {
    /// Implements trait to dynamically infer current shift of weights
    ///
    /// Must implement
    ///  * get_all_weights: returns all weight values
    ///
    ///
    /// Examples:
    /// ```
    /// use integers::{Tensor};
    /// use integers::nn::{HasWeights, Params};
    ///
    /// struct MyModule {
    ///     weights: Params<i32>,
    ///     bias: Params<i32>,
    /// }
    ///
    /// impl HasWeights<i32> for MyModule {
    ///     fn get_all_weights(&self) -> Vec<&Tensor<i32>> {
    ///         vec![&self.weights.master, &self.bias.master]
    ///     }
    /// }
    ///
    /// let module = MyModule {
    ///     weights: Params::<i32>::new(vec![2, 1], 0),
    ///     bias: Params::<i32>::new(vec![1, 1], 0),
    /// };
    ///
    /// // magnitude is 0, so we fall back to 4
    /// assert_eq!(module.infer_scale_shift(), 4);
    /// ```

    /// Get all weight parameters (master i32 tensors) in this module.
    /// Useful for analysis, initialization, and automatic shift detection.
    fn get_all_weights(&self) -> Vec<&Tensor<S::Acc>>;
    
    /// Infer an appropriate scale_shift based on weight magnitudes.
    /// Returns a shift value in the range [0, 8].
    fn infer_scale_shift(&self) -> u32 {
        let all_weights = self.get_all_weights();
        if all_weights.is_empty() {
            return 4;
        }

        let max_magnitude = all_weights
            .iter()
            .flat_map(|t| &t.data)
            .map(|&w| w.abs())
            .fold(S::Acc::zero(), |acc, x| if x.gt(acc) { x } else { acc });

        if max_magnitude == S::Acc::zero() {
            return 4;
        }

        compute_shift_for_max(max_magnitude.to_u32())
    }
}

#[derive(Debug)]
pub struct Linear<S: Scalar> {
    /// all weights are of Transposed Form [out, in] for memory reasons
    pub weights: Params<S>,
    pub bias: Params<S>,
    pub cache: Vec<Tensor<S>>,
    pub s_x_cache: Vec<u32>,


    /// Used to track what shift was applied to inputs
    pub input_shift: u32,
    /// Used to track what shift was applied to outputs before returning them
    pub output_shift: u32,
}

impl<S: Scalar> Linear<S> {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let acc_shift = S::acc_shift(input_dim); // ceil(log2)
        let back_acc_shift = S::acc_shift(output_dim);
        Self {
            weights: Params::new(vec![output_dim, input_dim], 0),
            bias: Params::new(vec![output_dim], 0),
            cache: Vec::new(),
            s_x_cache: Vec::new(),
            input_shift: acc_shift,
            output_shift: back_acc_shift,
        }
    }

    pub fn init_xavier(&mut self, rng: &mut XorShift64) {
        let fan_in = self.weights.master.shape[1];
        let fan_out = self.weights.master.shape[0];
        self.weights.init_xavier_uniform(rng, fan_in, fan_out);
        for b in self.bias.master.data.iter_mut() {
            *b = S::Acc::zero();
        }
    }

    pub fn with_shift(mut self, shift: u32) -> Self {
        self.weights.quant_shift = shift;
        self.bias.quant_shift = shift;
        self
    }
}

impl<S: Scalar + 'static> Module<S> for Linear<S> {
    fn zero_grads(&mut self){
        self.weights.zero_grads();
        self.bias.zero_grads();
    }

    fn sync_weights(&mut self, rng: &mut XorShift64) {
        self.weights.sync(rng);
        self.bias.sync(rng);
    }

    fn forward(&mut self, raw_input: &Tensor<S>, s_x: u32, rng: &mut XorShift64) -> (Tensor<S>, u32) {
        assert_eq!(
            raw_input.shape[1], self.weights.storage.shape[1],
            "Linear::forward: Input in wrong dimension for weights! {} vs {}",
            raw_input.shape[1], self.weights.storage.shape[1]
        );
        let batch = raw_input.shape[0];
        let input_dim = raw_input.shape[1];
        let output_dim = self.weights.storage.shape[0];

        let input = raw_input.clone();

        self.cache.push(input.clone());
        self.s_x_cache.push(s_x);
        let mut out_raw = Tensor::<S>::new(vec![batch, output_dim]);
        let out_shift = self.weights.quant_shift + self.input_shift + s_x;

        for b in 0..batch {
            let in_row = &input.data[b * input_dim..(b + 1) * input_dim];
            for o in 0..output_dim {
                let w_row = &self.weights.storage.data[o * input_dim..(o + 1) * input_dim];
                let mut raw_val = kernels::dot_product_scalar::<S>(in_row, w_row);
                raw_val = raw_val.add(self.bias.storage.data[o].into_acc());
                out_raw.data[b * output_dim + o] = S::downcast(raw_val, out_shift, rng);
            }
        }

        (out_raw, out_shift)
    }

    fn backward(&mut self, shifted_grad_output: &Tensor<S::Acc>, s_g: u32) -> (Tensor<S::Acc>, u32) {
        // not required to shift anymore, we did that in the forward pass already
        let input = self.cache.pop().expect("Linear::backward: Backward called without forward.");
        let s_x = self.s_x_cache.pop().expect("Linear::backward: Backward called without forward.");

        let local_delta_w = s_g + s_x;
        let local_delta_x = self.weights.quant_shift + self.output_shift;

        let batch = input.shape[0];
        let input_dim = input.shape[1];
        let output_dim = self.weights.storage.shape[0];

        let mut grad_input = Tensor::<S::Acc>::new(vec![batch, input_dim]);
        let mut grad_weights = Tensor::<S::Acc>::new(vec![output_dim, input_dim]);
        let mut grad_bias = Tensor::<S::Acc>::new(vec![output_dim]);

        let grad_output = shifted_grad_output; //shifted_grad_output >> output_shift;
                                               //
        for b in 0..batch {
            for o in 0..output_dim {
                let g = grad_output.data[b * output_dim + o];
                if g == S::Acc::zero() {
                    continue;
                }

                // Bias update is an accumulator, so apply gshift
                grad_bias.data[o] = grad_bias.data[o].add(
                    g
                );

                for i in 0..input_dim {
                    let x = input.data[b * input_dim + i];
                    let w = self.weights.storage.data[o * input_dim + i];

                    // --- Weight Gradient ---
                    // dL/dw = grad_output * input_i32
                    let dw = g.mul(x.into_acc());
                    grad_weights.data[o * input_dim + i] = grad_weights.data[o * input_dim + i].add(
                        dw.div(S::Acc::from_i32(1 << local_delta_w))
                    );

                    // --- Input Gradient ---
                    // dL/d(input) = grad_output * weight_i32
                    let dx = g.mul(w.into_acc());
                    grad_input.data[b * input_dim + i] = grad_input.data[b * input_dim + i].add(
                        dx.div(S::Acc::from_i32(1 << local_delta_x))
                    );
                }
            }
        }

        let grad_input_shifted = grad_input;// << input_shift;
        self.weights.accumulate_grads(&grad_weights);// << gshift);
        self.bias.accumulate_grads(&grad_bias);// << gshift);
                                               //
        let s_g_prev = s_g;//.saturating_sub(local_delta_x);

        (grad_input_shifted, s_g_prev)
    }

    fn step(&mut self, optim: &dyn OptimizerConfig<S>) {
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
        let shift = if S::unit_shift() == 0 { 0 } else { self.infer_scale_shift() };
        self.weights.quant_shift = shift;
        self.bias.quant_shift = shift;
    }

    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

impl<S: Scalar> HasWeights<S> for Linear<S> {
    fn get_all_weights(&self) -> Vec<&Tensor<S::Acc>> {
        vec![&self.weights.master, &self.bias.master]
    }
}


pub struct Sequential<S: Scalar> {
    pub modules: Vec<Box<dyn Module<S>>>,
}

impl<S: Scalar + 'static> Default for Sequential<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Scalar + 'static> Sequential<S> {
    pub fn new() -> Self {
        Self { modules: vec![] }
    }

    pub fn add(&mut self, m: impl Module<S> + 'static) -> &mut Self {
        self.modules.push(Box::new(m));
        self
    }

    pub fn init_all(&mut self, rng: &mut XorShift64) {
        for module in &mut self.modules {
            module.init(rng);
        }
    }
}

impl<S: Scalar + 'static> Module<S> for Sequential<S> {
    fn forward(&mut self, input: &Tensor<S>, s_x: u32, rng: &mut XorShift64) -> (Tensor<S>, u32) {
        let mut output = input.clone();
        let mut shift = s_x;
        for m in self.modules.iter_mut() {
            let (out, s_o) = m.forward(&output, shift, rng);
            output = out;
            shift = s_o;
        }
        (output, shift)
    }
    fn backward(&mut self, grad_output: &Tensor<S::Acc>, s_g: u32) -> (Tensor<S::Acc>, u32) {
        let mut output = grad_output.clone();
        let mut shift = s_g;
        for m in self.modules.iter_mut().rev() {
            let (out, s_g_prev) = m.backward(&output, shift);
            output = out;
            shift = s_g_prev;
        }
        (output, shift)
    }
    fn sync_weights(&mut self, rng: &mut XorShift64) {
        for m in self.modules.iter_mut() { m.sync_weights(rng); }
    }
    fn step(&mut self, optim: &dyn OptimizerConfig<S>) {
        for m in self.modules.iter_mut() { m.step(optim); }
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

    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::optim::*;

    // Module info tests
    #[test]
    fn test_module_info_total(){
        let info = ModuleInfo {
            name: "MyModule",
            params: 10,
            static_bytes: 0,
            children: vec![
                ModuleInfo{
                    name: "MyChild",
                    params: 5,
                    static_bytes: 12,
                    children: vec![]
                }
            ]
        };
        let (params, static_bytes) = info.total();

        assert_eq!(params, 15);
        assert_eq!(static_bytes, 12);
    }

    // Module trait tests
    // Testing Helper class implementation
    struct TestXPlusN {
        n: i32,
        input_shift: Option<u32>,
        output_shift: Option<u32>,
    }
   
    impl Module for TestXPlusN {
        fn backward(&mut self, grad: &Tensor<i32>) -> Tensor<i32> {
            grad.clone()
        }
        fn forward(&mut self, input: &Tensor<i32>, input_shift: u32, _rng: &mut XorShift64) -> Tensor<i32>{
            self.input_shift = Some(input_shift);
            self.output_shift = Some(input_shift);

            let mut output = Tensor::new(input.shape.clone());

            for (val, o) in input.data.iter().zip(output.data.iter_mut()) {
                *o = val + self.n;
            }
            output
        }

        fn as_any(&self) -> &dyn Any { self }
        fn as_any_mut(&mut self) -> &mut dyn Any { self }
    }

    #[test]
    fn test_testxplusn_contract() {
        let mut module = TestXPlusN {
            n: 1,
            input_shift: None,
            output_shift: None,
        };

        let input = Tensor::from_vec(vec![1, 2, 3], vec![3, 1]);
        let mut rng = XorShift64::new(42);

        let out = module.forward(&input, 0, &mut rng);

        // Contract: shape must match
        assert_eq!(out.shape, input.shape);

        // Contract: output shift must be set consistently
    }

    #[test]
    fn test_module_default_behaviour(){
        let mut module = TestXPlusN {
            n: 1,
            input_shift: None,
            output_shift: None,
        };
        let mut rng = XorShift64::new(420);

        let t = Tensor::from_vec(vec![1, 2], vec![2, 1]);

        assert_eq!(t.data, vec![2, 4]);

        module.sync_weights(&mut rng);
        assert_eq!(rng.state, 420);

        module.init(&mut rng);
        assert_eq!(rng.state, 420);

        let mut optim = SGDConfig::new();
        module.step(&mut optim);
        assert_eq!(rng.state, 420);

        assert_eq!(module.memory_report(), (0, 0));
        assert_eq!(module.describe(), ModuleInfo{name: "unknown", params: 0, static_bytes: 0, children: vec![]});
    }

    // Params tests
    #[test]
    fn test_params_new(){
        let shape = vec![1, 1];
        let params = Params::new(shape.clone(), 0);

        assert_eq!(params.master.len(), 1);
        assert_eq!(params.master.shape, shape);
        assert_eq!(params.storage.len(), 1);
        assert_eq!(params.storage.shape, shape);
        assert_eq!(params.grads, None);
        assert_eq!(params.state, None);
        assert_eq!(params.quant_shift, 0);
    }

    #[test]
    #[should_panic(
        expected="Weight shift must be 0-8, got 9"
    )]
    fn test_params_wrong_shift() {
        Params::new(vec![1, 1], 9);
    }

    #[test]
    fn test_params_with_shift() {
        let mut params = Params::new(vec![1, 1], 1);

        assert_eq!(params.quant_shift, 1);

        params = params.with_shift(2);

        assert_eq!(params.quant_shift, 2);
    }

    #[test]
    fn test_params_init_uniform() {
        let mut params = Params::new(vec![100000, 1], 0);
        let mut rng = XorShift64::new(420);

        params.init_uniform(&mut rng, 10);

        for w in params.master.data.iter() {
            assert!(*w <= 10);
        }
    }

    #[test]
    fn test_params_init_xavier_uniform() {
        let mut params = Params::new(vec![100000, 1], 0);
        let mut rng = XorShift64::new(420);

        // creates data in range 127 * sqrt(6 / 10) \approx 98
        params.init_xavier_uniform(&mut rng, 0, 10);

        for w in params.master.data.iter() {
            assert!(*w <= 98);
        }
    }

    #[test]
    fn test_params_sync(){
        let mut params = Params::new(vec![1, 1], 0);
        let mut rng = XorShift64::new(420);
        params.master.data[0] = 127;

        assert_eq!(params.storage.data[0], 0);

        params.sync(&mut rng);

        assert_eq!(params.storage.data[0], 127);

        params = params.with_shift(1);
        params.sync(&mut rng);

        assert_eq!(params.storage.data[0], 64);
    }

    #[test]
    fn test_params_accumulate_grads() {
        let mut params = Params::new(vec![1, 1], 0);

        let grad = Tensor::from_vec(vec![10; 1], vec![1, 1]);

        params.accumulate_grads(&grad);

        let stored_grads = params.grads.as_ref().unwrap();
        assert_eq!(stored_grads.data[0], 10);

        params.accumulate_grads(&grad);

        let accumulated_grads = params.grads.as_ref().unwrap();
        assert_eq!(accumulated_grads.data[0], 20);

        // but we never exceed i32::MAX
        let max_grad = Tensor::from_vec(vec![i32::MAX; 1], vec![1, 1]);
        params.accumulate_grads(&max_grad);

        let final_accumulate_grads = params.grads.as_ref().unwrap();
        assert_eq!(final_accumulate_grads.data[0], i32::MAX);
    }

    #[test]
    fn test_params_zero_grads(){
        let mut params = Params::new(vec![1, 1], 0);

        let grad = Tensor::from_vec(vec![10; 1], vec![1, 1]);

        params.accumulate_grads(&grad);

        let stored_grads = params.grads.as_ref().unwrap();
        assert_eq!(stored_grads.data[0], 10);

        params.zero_grads();

        assert!(params.grads.as_ref().is_none());
    }

    #[test]
    fn test_params_step_without_grad_is_okay(){
        let mut params = Params::new(vec![1, 1], 0);
        let optim = SGDConfig::new();

        params.step(&optim);

        assert!(matches!(
            params.state,
            None
        ))
    }

    #[test]
    fn test_params_step_initializes_state(){
        let mut params = Params::new(vec![1, 1], 0);
        let optim = SGDConfig::new();

        let grad = Tensor::from_vec(vec![10; 1], vec![1, 1]);

        params.grads = Some(grad);
        params.step(&optim);

        assert!(matches!(
            params.state.as_ref().unwrap(),
            OptimizerState<S>::None,
        ))
    }

    #[test]
    fn test_params_memory_bytes() {
        let params = Params::new(vec![1, 1], 0);

        assert_eq!(params.memory_bytes(), 8);

        // twice the size, twice the memory required
        let other_params = Params::new(vec![2, 1], 0);

        assert_eq!(other_params.memory_bytes(), 16);
    }

    #[test]
    fn test_compute_shift_for_max() {
        let vals = [
            0, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
        ];
        let shifts = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 8 // we don't exceed 8
        ];

        for (val, shift) in vals.iter().zip(shifts.iter()) {
            assert_eq!(compute_shift_for_max(*val), *shift);
        }
    }

    // HasWeights tests
    struct TestHasWeightsStruct {
        w: Params,
        b: Params,
    }

    impl HasWeights for TestHasWeightsStruct {
        fn get_all_weights(&self) -> Vec<&Tensor<i32>> {
            vec![&self.w.master, &self.b.master]
        }
    }

    struct TestHasWeightsEmptyStruct {

    }
    impl HasWeights for TestHasWeightsEmptyStruct {
        fn get_all_weights(&self) -> Vec<&Tensor<i32>> {
            vec![]
        }
    }

    #[test]
    fn test_has_weights_infer_scale(){
        let mut module = TestHasWeightsStruct{
            w: Params::new(vec![2, 1], 0),
            b: Params::new(vec![1, 1], 0),
        };

        // uninitialized, fallback shift 4 [middle ground]
        assert_eq!(module.infer_scale_shift(), 4);

        module.w.master.data[0] = 127;
        module.w.master.data[1] = -127;
        module.b.master.data[0] = 0;

        assert_eq!(module.infer_scale_shift(), 0);

        module.w.master.data[0] = 127;
        module.w.master.data[1] = -128;
        module.b.master.data[0] = 0;

        assert_eq!(module.infer_scale_shift(), 1);

        module.w.master.data = vec![];
        module.b.master.data = vec![];

        assert_eq!(module.infer_scale_shift(), 4);
    }

    #[test]
    fn test_has_weights_infer_scale_no_weights() {
        let module = TestHasWeightsEmptyStruct{};

        assert_eq!(module.infer_scale_shift(), 4);
    }

    // Linear tests
    #[test]
    fn test_linear_new() {
        let lin = Linear::new(4, 8);

        assert_eq!(lin.weights.master.len(), 4 * 8);
        assert_eq!(lin.weights.storage.len(), 4 * 8);
        assert_eq!(lin.bias.master.len(), 8);
        assert_eq!(lin.weights.quant_shift, 0);
        assert_eq!(lin.cache, Vec::new());
        assert_eq!(lin.weights.grads, None);
        assert_eq!(lin.bias.grads, None);
    }

    #[test]
    fn test_linear_init_xavier(){
        let mut lin = Linear::new(2, 2);
        let mut rng = XorShift64::new(420);

        lin.init_xavier(&mut rng);

        for w in lin.weights.master.data.iter() {
            assert!(*w <= 155);
        }
        for b in lin.bias.master.data.iter() {
            assert_eq!(*b, 0);
        }
    }

    #[test]
    fn test_linear_sync_weights() {
        let mut lin = Linear::new(2, 2);
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
        let mut layer = Linear::new(10, 5); // start with shift=0
        
        // Initialize with Xavier distribution
        layer.init_xavier(&mut rng);
        
        // Auto-detect appropriate shift
        let detected_shift = layer.infer_scale_shift();
        println!("Detected shift: {}", detected_shift);
        
        layer.weights.quant_shift = detected_shift;
        layer.bias.quant_shift = detected_shift;
        
        // Now safely quantize
        layer.sync_weights(&mut rng);
        
        // Verify all weights fit in i32 range
        for &w in &layer.weights.storage.data {
            assert!(w >= -128);  // w <= 127 given by data type
        }
    }
    #[test]
    fn test_linear_forward() {
        let mut lin = Linear::new(2, 2);
        lin.weights.master.data[0] = 1;
        lin.weights.master.data[3] = 1;
        let mut rng = XorShift64 { state: 420 };
        lin.sync_weights(&mut rng);

        let mut input = Tensor::new(vec![1, 2]);
        input.data[0] = 10;
        input.data[1] = 20;
        let out = lin.forward(&input, 0, &mut rng);
        let output_shift = lin.output_shift.expect("Does not exist.");

        assert_eq!(out.data[0] << output_shift, 10);
        assert_eq!(out.data[1] << output_shift, 20);
    }

    #[test]
    fn test_linear_forward_saturation() {
        let mut lin = Linear::new(2, 2);
        lin.weights.master.data[0] = 33292288;
        lin.weights.master.data[3] = 33292288;
        let mut rng = XorShift64 { state: 420 };
        lin.sync_weights(&mut rng);

        let mut input = Tensor::new([1, 2].to_vec());
        input.data[0] = 33292288;
        input.data[1] = 33292288;
        let out = lin.forward(&input, 0, &mut rng);
        let output_shift = lin.output_shift.expect("Does not exist.");

        // clamping around 127
        assert_eq!(output_shift, 8);
        assert_eq!(out.data[0] << output_shift, -256);
        assert_eq!(out.data[1] << output_shift, -256);
        assert_eq!(out.data[0], 2147483647);
        assert_eq!(out.data[1], 2147483647);
    }

    #[test]
    fn test_linear_forward_shape() {
        let mut lin = Linear::new(2, 3);
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
        let mut lin = Linear::new(2, 2);

        // Identity Matrix in Master Weights [1, 0] / [0, 1]
        lin.weights.master.data[0] = 1;
        lin.weights.master.data[3] = 1;

        let mut rng = XorShift64::new(420);
        lin.sync_weights(&mut rng);

        let input = Tensor::from_vec(vec![10, 20], vec![1, 2]);

        let out = lin.forward(&input, 0, &mut rng);
        let output_shift = lin.output_shift.expect("Does not exist.");

        assert_eq!(out.data[0] << output_shift, 10);
        assert_eq!(out.data[1] << output_shift, 20);
        assert_eq!(out.data[0], 10);
        assert_eq!(out.data[1], 20);
    }

    #[test]
    fn test_linear_forward_identity_with_shift() {
        // Linear 2->2, Scale 0 (Identity mapping potential)
        let mut lin = Linear::new(2, 2);

        // Identity Matrix in Master Weights [2, 0] / [0, 2]
        lin.weights.master.data[0] = 2;
        lin.weights.master.data[3] = 2;

        let mut rng = XorShift64::new(420);
        lin.sync_weights(&mut rng);

        let mut input = Tensor::from_vec(vec![127, -128], vec![1, 2]);

        let out = lin.forward(&input, 0, &mut rng);
        let output_shift = lin.output_shift.expect("Does not exist.");

        assert_eq!(out.data[0] << output_shift, 1016);
        assert_eq!(out.data[1] << output_shift, -1024);
        assert_eq!(out.data[0], 254);
        assert_eq!(out.data[1], -256);

        input = Tensor::from_vec(vec![127, 64], vec![1, 2]);

        let out = lin.forward(&input, 0, &mut rng);
        let output_shift = lin.output_shift.expect("Does not exist.");

        assert_eq!(out.data[0] << output_shift, 508);
        assert_eq!(out.data[1] << output_shift, 256);
        assert_eq!(out.data[0], 254);
        assert_eq!(out.data[1], 128);
    }

    #[test]
    fn test_linear_forward_shape_batch() {
        let mut lin = Linear::new(2, 3);
        let mut rng = XorShift64::new(420);
        lin.sync_weights(&mut rng);

        let input = Tensor::new(vec![10, 2]);
        let out = lin.forward(&input, 0, &mut rng);

        assert_eq!(out.shape[0], 10);
        assert_eq!(out.shape[1], 3);
        assert_eq!(out.data.len(), 30);
    }

    #[test]
    #[should_panic(
        expected = "assertion `left == right` failed: Linear::forward: Input in wrong dimension for weights! 1 vs 2\n  left: 1\n right: 2"
    )]
    fn test_linear_forward_wrong_dimensional_input() {
        let mut lin = Linear::new(2, 3);
        let mut rng = XorShift64::new(420);
        lin.sync_weights(&mut rng);

        let input = Tensor::from_vec(vec![2; 10], vec![10, 1]);
        lin.forward(&input, 0, &mut rng);
    }

    #[test]
    fn test_linear_backward_shapes() {
        // Batch 2, Input 4 -> Output 3
        let mut lin = Linear::new(4, 3);
        let mut rng = XorShift64::new(420);

        let input = Tensor::new(vec![2, 4]); // [Batch, In]
        let grad_out = Tensor::new(vec![2, 3]); // [Batch, Out]

        lin.forward(&input, 0, &mut rng);
        lin.cache = vec![input.clone()];

        let d_x = lin.backward(&grad_out);
        let d_w = lin.weights.grads.unwrap().clone();

        // Check shapes
        assert_eq!(d_x.shape, vec![2, 4]); // [Batch, In]
        assert_eq!(d_w.shape, vec![3, 4]); // [Out, In]
    }

    #[test]
    fn test_linear_backward() {
        let mut lin = Linear::new(2, 3);
        let mut rng = XorShift64::new(420);

        for w in lin.weights.master.data.iter_mut() {
            *w = 120;
        }

        let input = Tensor::from_vec(vec![10; 4], vec![2, 2]); // [Batch, In]
        let grad_out = Tensor::from_vec(vec![22; 6], vec![2, 3]); // [Batch, Out]

        lin.forward(&input, 0, &mut rng);
        lin.cache = vec![input.clone()];

        let d_x = lin.backward(&grad_out);
        let d_w = lin.weights.grads.unwrap().clone();
        let d_b = lin.bias.grads.unwrap().clone();

        // Check shapes
        assert_eq!(d_x.shape, vec![2, 2]); // [Batch, In]
        assert_eq!(d_w.shape, vec![3, 2]); // [Out, In]
        assert_eq!(d_b.shape, vec![3]);

        assert_eq!(d_x.data[0], 0);
        assert_eq!(d_x.data[1], 0);
        assert_eq!(d_x.data[2], 0);
        assert_eq!(d_x.data[3], 0);

        assert_eq!(d_w.data[0], 440);
        assert_eq!(d_w.data[1], 440);
        assert_eq!(d_w.data[2], 440);
        assert_eq!(d_w.data[3], 440);
        assert_eq!(d_w.data[4], 440);
        assert_eq!(d_w.data[5], 440);
        
        assert_eq!(d_b.data[0], 44);
        assert_eq!(d_b.data[1], 44);
        assert_eq!(d_b.data[2], 44);
    }

    #[test]
    fn test_linear_memory_report() {
        let mut rng = XorShift64::new(777);
        let mut l1 = Linear::new(2, 1)
            .with_shift(2);
        let x = Tensor::from_vec(vec![0, 0], vec![1, 2]);
        l1.forward(&x, 2, &mut rng);

        let (a, b) = l1.memory_report();
        assert_eq!(a, 24);
        assert_eq!(b, 8);

        let mut l2 = Linear::new(2, 10)
            .with_shift(2);
        l2.forward(&x, 2, &mut rng);

        let (a2, b2) = l2.memory_report();
        assert_eq!(a2, 240);
        assert_eq!(b2, 8);
    }

    #[test]
    fn test_linear_step() {
        let mut rng = XorShift64::new(777);
        let mut l1 = Linear::new(2, 1)
            .with_shift(1);
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
        assert_eq!(l1.weights.storage.data[0], 55);
        assert_eq!(l1.weights.storage.data[1], 111);
    }

    #[test]
    fn test_linear_describe(){
        let l1 = Linear::new(2, 1);

        assert_eq!(l1.describe(), ModuleInfo{
            name: "Linear",
            params: 3,
            static_bytes: 24,
            children: vec![]
        });
    }

    #[test]
    fn test_linear_init_sets_weights_and_shifts(){
        let mut l1 = Linear::new(1, 1)
            .with_shift(2);
        let mut rng = XorShift64::new(420);

        l1.init(&mut rng);

        assert_ne!(l1.weights.master.data[0], 0);
        assert_eq!(l1.weights.quant_shift, 0);
        assert_eq!(l1.bias.quant_shift, 0);
    }

    #[test]
    fn test_linear_get_all_weights(){
        let l1 = Linear::new(1, 1);

        assert_eq!(l1.get_all_weights(), vec![&l1.weights.master, &l1.bias.master]);

    }

    // Sequential
    #[test]
    fn test_sequential_default(){
        let model = Sequential::default();
        let model2 = Sequential::new();

        assert!(model.modules.is_empty());
        assert!(model2.modules.is_empty());
    }

    #[test]
    fn test_sequential_add(){
        let mut model = Sequential::new();
        let l1 = Linear::new(1, 1);

        model.add(l1);

        assert_eq!(model.modules.len(), 1);
    }
    #[test]
    fn test_sequential_init_all(){
        let mut model = Sequential::new();
        let l1 = Linear::new(1, 1);
        let mut rng = XorShift64::new(420);

        model.add(l1);

        model.init_all(&mut rng);

        let l1_ref = model.modules[0]
            .as_any()
            .downcast_ref::<Linear>()
            .expect("Expected linear layer");

        assert_ne!(l1_ref.weights.master.data[0], 0);
    }

    #[test]
    fn test_sequential_forward(){
        let mut rng = XorShift64::new(420);
        let mut rng2 = XorShift64::new(420);
        let mut l1 = Linear::new(2, 2);
        let mut model = Sequential {
            modules: vec![
                Box::new(Linear::new(2, 2)),
            ],
        };

        let input = Tensor::from_vec(vec![10, 5], vec![1, 2]);

        let f1_out = l1.forward(&input, 0, &mut rng);
        let model_out = model.forward(&input, 0, &mut rng2);

        assert_eq!(f1_out, model_out);
    }

    #[test]
    fn test_sequential_backward(){
        let mut rng = XorShift64::new(420);
        let mut rng2 = XorShift64::new(420);
        let mut l1 = Linear::new(2, 2);
        let mut model = Sequential {
            modules: vec![
                Box::new(Linear::new(2, 2)),
            ],
        };

        let input = Tensor::from_vec(vec![10, 5], vec![1, 2]);

        let _ = l1.forward(&input, 0, &mut rng);
        let _ = model.forward(&input, 0, &mut rng2);

        let grad = Tensor::from_vec(vec![-1, -2], vec![1, 2]);

        let f1_grad_out = l1.backward(&grad);
        let grad_out = model.backward(&grad);

        assert_eq!(f1_grad_out, grad_out);
    }

    #[test]
    fn test_sequential_sync_weights(){
        let mut rng = XorShift64::new(420);
        let mut model = Sequential {
            modules: vec![
                Box::new(Linear::new(2, 2)),
            ],
        };

        model.init_all(&mut rng);

        model.sync_weights(&mut rng);

        let l1_ref = model.modules[0]
            .as_any()
            .downcast_ref::<Linear>()
            .expect("Expected linear layer");

        assert_eq!(l1_ref.weights.master.data, vec![139, -30, -53, 98]);
        assert_eq!(l1_ref.weights.storage.data, vec![70, -15, -26, 49]);

    }

    #[test]
    fn test_sequential_step(){
        // basically the same as just a linear layer
        let mut rng = XorShift64::new(777);
        let mut l1 = Linear::new(2, 1)
            .with_shift(1);
        let mut model = Sequential::new();

        let mut optim = SGDConfig::new().with_learn_rate(1.0);
        l1.weights.master.data[0] = 111;
        l1.weights.master.data[1] = 222;

        model.add(l1);

        model.sync_weights(&mut rng);
        
        let l1_ref = model.modules[0]
            .as_any()
            .downcast_ref::<Linear>()
            .expect("Expected linear layer");

        assert_eq!(l1_ref.weights.master.data[0], 111);
        assert_eq!(l1_ref.weights.master.data[1], 222);

        // Backpropagate "a" gradient 
        let grad_out = Tensor::<i32>::from_vec(vec![126, 2], vec![2, 1]);
        let mut_ref = model.modules[0]
            .as_any_mut()
            .downcast_mut::<Linear>()
            .expect("Expected linear layer");
        mut_ref.weights.grads = Some(grad_out);

        // with a gradient we have change
        model.step(&mut optim);

        let new_l1_ref = model.modules[0]
            .as_any()
            .downcast_ref::<Linear>()
            .expect("Expected linear layer");

        assert_eq!(new_l1_ref.weights.master.data[0], -15);
        assert_eq!(new_l1_ref.weights.master.data[1], 220);

    }

    #[test]
    fn test_sequential_memory_report(){
        let model = Sequential {
            modules: vec![
                Box::new(Linear::new(2, 8)),
                Box::new(activations::ReLU::new()),
                Box::new(Linear::new(8, 1)),
            ],
        };

        let (s, d) = model.memory_report();

        assert_eq!(s, 264);
        assert_eq!(d, 0);
    }

    #[test]
    fn test_summary() {
        let model = Sequential {
            modules: vec![
                Box::new(Linear::new(2, 8)),
                Box::new(activations::ReLU::new()),
                Box::new(Linear::new(8, 1)),
            ],
        };

        model.print_summary(&model.describe(), 1);
    }

}


