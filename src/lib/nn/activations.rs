use crate::{Tensor, XorShift64};
use crate::nn::{ModuleInfo, Module};
use crate::nn::kernels;

pub struct ReLU {
    pub cache: Vec<Tensor<i8>>,
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl ReLU {
    pub fn new() -> Self {
        Self { cache: Vec::new() }
    }
}

impl Module for ReLU {
    fn forward(&mut self, input: &Tensor<i8>, input_shift: u32, rng: &mut XorShift64) -> Tensor<i8> {
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
}

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl Tanh {
    pub fn new() -> Self {
        Self { cache: Vec::new() }
    }
}

impl Module for Tanh {
    fn forward(&mut self, input: &Tensor<i8>, input_shift: u32, rng: &mut XorShift64) -> Tensor<i8> {
        self.cache.push(input.clone());
        let mut output = Tensor::<i8>::new(input.shape.clone());
        for (o, &x) in output.data.iter_mut().zip(&input.data) {
            *o = kernels::tanh_i8(x);
        }
        output
    }
    fn backward(&mut self, grad_output: &Tensor<i16>, _grad_shift: Option<u32>) -> Tensor<i16> {
        let input = self
            .cache
            .pop()
            .expect("Tanh::backward: No state registered. Perform forward pass first!");
        let mut output = Tensor::<i16>::new(grad_output.shape.clone());
        for o in 0..grad_output.data.len() {
            let t = kernels::tanh_i8(input.data[o]) as i32;
            let dtanh = (127 * 127 - t * t) / 127;
            output.data[o] =
                ((grad_output.data[o] as i32 * dtanh) / 128).clamp(-32768, 32767) as i16;
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
}

