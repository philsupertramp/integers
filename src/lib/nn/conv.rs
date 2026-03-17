use crate::{Tensor, XorShift64, checked_add_counting, Scalar, Numeric};
use crate::nn::{Module, ModuleInfo, Params};
use std::any::Any;

pub struct Flatten {}

impl<S: Scalar + 'static> Module<S> for Flatten {
    fn forward(&mut self, input: &Tensor<S>, s_x: i32, rng: &mut XorShift64) -> (Tensor<S>, i32){
        // TODO: Reshape 4D to 2D
        // [batch_size, channels, height, width] to [batch_size, channels * height * width]
        let output = Tensor::<S>::new(input.shape.clone());
        
        (output, s_x)
    }
    fn backward(&mut self, grad: &Tensor<S::Acc>, s_g: i32) -> (Tensor<S::Acc>, i32){
        // TODO: reshape 2D back to 4D
        let output = Tensor::<S::Acc>::new(grad.shape.clone());
        (output, s_g)
    }

    fn describe(&self) -> ModuleInfo {
        ModuleInfo {
            name: "Flatten",
            params: 0,
            static_bytes: 0,
            children: vec![],
        }
    }

    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}


pub struct Conv2D<S: Scalar> {
    pub weights: Params<S>,
    pub bias: Params<S>,

    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,

    pub input_shift: u32,
    pub output_shift: u32,

    // caches
    cache: Vec<Tensor<S>>,
    s_x_cache: Vec<u32>,
}

impl<S: Scalar> Conv2D<S> {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            weights: Params::new(vec![out_channels, in_channels, kernel_size, kernel_size], 0),
            bias: Params::new(vec![], 0),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            input_shift: 0,
            output_shift: 0,
            cache: Vec::new(),
            s_x_cache: Vec::new()
        }
    }
}

impl<S: Scalar + 'static> Module<S> for Conv2D<S> {
    fn forward(&mut self, input: &Tensor<S>, s_x: i32, rng: &mut XorShift64) -> (Tensor<S>, i32){
        // TODO: 
        let output = Tensor::<S>::new(input.shape.clone());
        
        (output, s_x)
    }
    fn backward(&mut self, grad: &Tensor<S::Acc>, s_g: i32) -> (Tensor<S::Acc>, i32){
        // TODO:
        let output = Tensor::<S::Acc>::new(grad.shape.clone());
        (output, s_g)
    }

    fn describe(&self) -> ModuleInfo {
        ModuleInfo {
            name: "Conv2D",
            params: 0,
            static_bytes: 0,
            children: vec![],
        }
    }

    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}
