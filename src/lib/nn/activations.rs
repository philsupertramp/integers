use crate::{Tensor, XorShift64};
use crate::nn::{ModuleInfo, Module, compute_shift_for_max};
use crate::nn::kernels;

pub struct ReLU {
    pub cache: Vec<Tensor<i8>>,
    /// Automatically tracks what shift was applied to inputs
    pub input_shift: Option<u32>,
    pub output_shift: Option<u32>,
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl ReLU {
    pub fn new() -> Self {
        Self { cache: Vec::new(), input_shift: None, output_shift: None }
    }
}

impl Module for ReLU {
    fn get_output_shift(&self) -> u32 {
        self.output_shift.expect("ReLU::get_output_shift: Didn't call forward!")
    }

    fn forward(&mut self, input: &Tensor<i8>, input_shift: u32, rng: &mut XorShift64) -> Tensor<i8> {
        self.input_shift = Some(input_shift);
        self.output_shift = Some(input_shift);

        self.cache.push(input.clone());
        let mut output = Tensor::<i8>::new(input.shape.clone());
        for idx in 0..input.data.len() {
            output.data[idx] = if input.data[idx] > 0 {
                input.data[idx]
            } else {
                0
            };
        }
        output
    }
    fn backward(&mut self, grad_output: &Tensor<i16>, _grad_shift: Option<u32>) -> Tensor<i16> {
        let input = self
            .cache
            .pop()
            .expect("ReLU::backward: No state registered. Perform forward pass first!");
        let mut output = Tensor::<i16>::new(grad_output.shape.clone());
        for o in 0..grad_output.data.len() {
            output.data[o] = if input.data[o] > 0 {
                grad_output.data[o]
            } else {
                0
            };
        }
        output
    }
    fn memory_report(&self) -> (usize, usize) {
        let dyn_ = self.cache.iter().map(|t| t.memory_bytes()).sum();
        (0, dyn_)
    }

    fn describe(&self) -> ModuleInfo {
        ModuleInfo {
            name: "ReLU",
            params: 0,
            static_bytes: 0,
            children: vec![],
        }
    }
}

pub struct Tanh {
    pub cache: Vec<Tensor<i8>>,
    /// Automatically tracks what shift was applied to inputs
    pub input_shift: Option<u32>,
    pub output_shift: Option<u32>,
}

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl Tanh {
    pub fn new() -> Self {
        Self { cache: Vec::new(), input_shift: None, output_shift: None }
    }
}

impl Module for Tanh {
    fn forward(&mut self, input: &Tensor<i8>, input_shift: u32, rng: &mut XorShift64) -> Tensor<i8> {
        self.input_shift = Some(input_shift);

        self.cache.push(input.clone());
        let mut output = Tensor::<i8>::new(input.shape.clone());
        for (o, &x) in output.data.iter_mut().zip(&input.data) {
            *o = kernels::tanh_i8(x);
        }

        let max_magnitude = output.data
            .iter()
            .map(|x| x.abs() as u32)
            .max()
            .unwrap_or(1);

        let output_shift = compute_shift_for_max(max_magnitude);
        self.output_shift = Some(output_shift);

        output
    }
    fn backward(&mut self, grad_output: &Tensor<i16>, _grad_shift: Option<u32>) -> Tensor<i16> {
        let input = self
            .cache
            .pop()
            .expect("Tanh::backward: No state registered. Perform forward pass first!");

        let input_shift = self.input_shift.expect("Tanh::backward: No state registered. Perform forward pass first!");
        let output_shift = self.output_shift
            .expect("Tanh::backward: No state registered. Perform forward pass first.");


        let mut output = Tensor::<i16>::new(grad_output.shape.clone());
        for o in 0..grad_output.data.len() {
            let t = kernels::tanh_i8(input.data[o]) as i32;
            let dtanh = ((127 * 127) - (t * t)) / 127;

            let grad = grad_output.data[o] as i32;
            let dout = grad * dtanh / (127 * 127);
            let adjusted = dout >> (output_shift.saturating_sub(input_shift));
            output.data[o] = adjusted.clamp(-32768, 32767) as i16;
        }
        output
    }
    fn memory_report(&self) -> (usize, usize) {
        let dyn_ = self.cache.iter().map(|t| t.memory_bytes()).sum();
        (0, dyn_)
    }

    fn describe(&self) -> ModuleInfo {
        ModuleInfo {
            name: "Tanh",
            params: 0,
            static_bytes: 0,
            children: vec![],
        }
    }

    fn get_output_shift(&self) -> u32 {
        self.output_shift
            .expect("No forward call.")
    }
}

