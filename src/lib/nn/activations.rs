use crate::{Tensor, XorShift64};
use crate::nn::{ModuleInfo, Module, compute_shift_for_max};
use crate::nn::kernels;

pub struct ReLU {
    pub cache: Vec<Tensor<i32>>,
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

    fn forward(&mut self, input: &Tensor<i32>, input_shift: u32, _rng: &mut XorShift64) -> Tensor<i32> {
        self.input_shift = Some(input_shift);
        self.output_shift = Some(input_shift);

        self.cache.push(input.clone());
        let mut output = Tensor::<i32>::new(input.shape.clone());
        for idx in 0..input.data.len() {
            output.data[idx] = if input.data[idx] > 0 {
                input.data[idx]
            } else {
                0
            };
        }
        output
    }
    fn backward(&mut self, grad_output: &Tensor<i32>, _grad_shift: Option<u32>) -> Tensor<i32> {
        let input = self
            .cache
            .pop()
            .expect("ReLU::backward: No state registered. Perform forward pass first!");
        let mut output = Tensor::<i32>::new(grad_output.shape.clone());
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
    pub cache: Vec<Tensor<i32>>,
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
    fn forward(&mut self, input: &Tensor<i32>, input_shift: u32, _rng: &mut XorShift64) -> Tensor<i32> {
        self.input_shift = Some(input_shift);

        self.cache.push(input.clone());
        let mut output = Tensor::<i32>::new(input.shape.clone());
        for (o, &x) in output.data.iter_mut().zip(&input.data) {
            *o = kernels::tanh_i8(x as i8) as i32;
        }

        let max_magnitude = output.data
            .iter()
            .map(|x| x.abs() as u32)
            .max()
            // TODO: (SHIFT#1) Potentially an issue. Empty input is shifted?
            .unwrap_or(0);

        let output_shift = compute_shift_for_max(max_magnitude);
        self.output_shift = Some(output_shift);

        output
    }
    fn backward(&mut self, grad_output: &Tensor<i32>, _grad_shift: Option<u32>) -> Tensor<i32> {
        let input = self
            .cache
            .pop()
            .expect("Tanh::backward: No state registered. Perform forward pass first.");

        let input_shift = self.input_shift.expect("Tanh::backward: No state registered. Perform forward pass first.");
        let output_shift = self.output_shift
            .expect("Tanh::backward: No state registered. Perform forward pass first.");


        let mut output = Tensor::<i32>::new(grad_output.shape.clone());
        for o in 0..grad_output.data.len() {
            let t = kernels::tanh_i8(input.data[o] as i8) as i64;
            let dtanh_num = (127 * 127) - (t * t); // [0, 16129]
            let grad = grad_output.data[o] as i64;
            // TODO: For more accurate results we can divide by 16129
            //       This divides by 16384
            let dout = (grad * dtanh_num) >> 14;
            let shift = output_shift.saturating_sub(input_shift);
            let adjusted = dout;// >> shift;
            output.data[o] = adjusted.clamp(-32768, 32767) as i32;
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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_new() {
        let relu = ReLU::new();

        assert_eq!(relu.cache, Vec::new());
        assert_eq!(relu.input_shift, None);
        assert_eq!(relu.output_shift, None);
    }

    #[test]
    fn test_relu_default() {
        let relu = ReLU::default();

        assert_eq!(relu.cache, Vec::new());
        assert_eq!(relu.input_shift, None);
        assert_eq!(relu.output_shift, None);
    }

    #[test]
    fn test_relu_input_shift_sets_shifts(){
        let mut rng = XorShift64::new(420);
        let mut relu = ReLU::new();
        let input = Tensor::from_vec(vec![1], vec![1, 1]);

        assert_eq!(relu.input_shift, None);
        assert_eq!(relu.output_shift, None);

        relu.forward(&input, 8, &mut rng);

        assert_eq!(relu.input_shift, Some(8));
        assert_eq!(relu.output_shift, Some(8));
    }

    #[test]
    fn test_relu_get_output_shift(){
        let mut rng = XorShift64::new(420);
        let mut relu = ReLU::new();
        let input = Tensor::from_vec(vec![1], vec![1, 1]);

        relu.forward(&input, 0, &mut rng);

        assert_eq!(relu.get_output_shift(), 0);
    }

    #[test]
    #[should_panic(
        expected="ReLU::get_output_shift: Didn't call forward!"
    )]
    fn test_relu_get_output_shift_no_forward_call(){
        let relu = ReLU::new();

        relu.get_output_shift();
    }

    #[test]
    fn test_relu_forward(){
        let mut rng = XorShift64::new(420);
        let mut relu = ReLU::new();
        let input = Tensor::from_vec(vec![-128, -1, 0, 1, 127], vec![5, 1]);
        let expected_values = Tensor::from_vec(vec![0, 0, 0, 1, 127], vec![5, 1]);

        let computed = relu.forward(&input, 8, &mut rng);

        for (calculated, expected) in computed.data.into_iter().zip(expected_values.data.into_iter()) {
            assert_eq!(calculated, expected);
        }
    }

    #[test]
    #[should_panic(
        expected="ReLU::backward: No state registered. Perform forward pass first!"
    )]
    fn test_relu_backward_no_cache(){
        let mut relu = ReLU::new();
        let grad = Tensor::from_vec(vec![1], vec![1, 1]);

        relu.backward(&grad, Some(0));
    }

    #[test]
    fn test_relu_backward(){
        let mut relu = ReLU::new();
        let input = Tensor::from_vec(vec![-128, -1, 0, 1, 127], vec![5, 1]);
        let grad = Tensor::from_vec(vec![1, 0, -10, 20, 100], vec![5, 1]);
        let expected_grad = Tensor::from_vec(vec![0, 0, 0, 20, 100], vec![5, 1]);
        relu.cache.push(input.clone());

        let grad_out = relu.backward(&grad, Some(0));

        for (calculated, expected) in grad_out.data.into_iter().zip(expected_grad.data.into_iter()) {
            assert_eq!(calculated, expected);
        }
    }

    #[test]
    fn test_relu_memory_report(){
        let mut rng = XorShift64::new(420);
        let mut relu = ReLU::new();
        let mut dyn_;
        let mut sta_;

        (sta_, dyn_) = relu.memory_report();

        assert_eq!(sta_, 0);
        assert_eq!(dyn_, 0);

        let input = Tensor::from_vec(vec![-128, -1, 0, 1, 127], vec![5, 1]);
        relu.forward(&input, 0, &mut rng);

        (sta_, dyn_) = relu.memory_report();

        assert_eq!(sta_, 0);
        assert_eq!(dyn_, 20);

        // we accumulate inputs
        relu.forward(&input, 0, &mut rng);

        (sta_, dyn_) = relu.memory_report();

        assert_eq!(sta_, 0);
        assert_eq!(dyn_, 40);

        // and can free them by calling the backward pass
        let grad = Tensor::from_vec(vec![1, 0, -10, 20, 100], vec![5, 1]);
        relu.backward(&grad, None);

        (sta_, dyn_) = relu.memory_report();

        assert_eq!(sta_, 0);
        assert_eq!(dyn_, 20);

        relu.backward(&grad, None);

        (sta_, dyn_) = relu.memory_report();

        assert_eq!(sta_, 0);
        assert_eq!(dyn_, 0);
    }

    #[test]
    fn test_relu_describe() {
        let relu = ReLU::new();
        let info = ModuleInfo {
            name: "ReLU",
            params: 0,
            static_bytes: 0,
            children: vec![],
        };

        assert_eq!(relu.describe(), info);
    }

    #[test]
    fn test_tanh_default() {
        let tanh = Tanh::default();

        assert_eq!(tanh.cache, Vec::new());
        assert_eq!(tanh.input_shift, None);
        assert_eq!(tanh.output_shift, None);
    }

    #[test]
    fn test_tanh_new() {
        let tanh = Tanh::new();

        assert_eq!(tanh.cache, Vec::new());
        assert_eq!(tanh.input_shift, None);
        assert_eq!(tanh.output_shift, None);
    }

    #[test]
    fn test_tanh_forward_empty_input() {
        let mut rng = XorShift64::new(420);
        let mut tanh = Tanh::new();

        let input = Tensor::from_vec(vec![], vec![]);

        let output = tanh.forward(&input, 0, &mut rng);

        assert_eq!(output.shape, vec![]);
        assert_eq!(output.data, vec![]);
    }

    #[test]
    fn test_tanh_forward() {
        let mut rng = XorShift64::new(420);
        let mut tanh = Tanh::new();

        let input = Tensor::from_vec(vec![-128, -1, 0, 1, 127], vec![5, 1]);

        let output = tanh.forward(&input, 0, &mut rng);

        assert_eq!(output.shape, vec![5, 1]);
        assert_eq!(output.data, vec![-97, -1, 0, 1, 97]);
        assert_eq!(tanh.input_shift, Some(0));
        assert_eq!(tanh.output_shift, Some(0));
    }

    #[test]
    fn test_tanh_forward_sets_cache_and_shifts() {
        let mut rng = XorShift64::new(420);
        let mut tanh = Tanh::new();

        assert_eq!(tanh.input_shift, None);
        assert_eq!(tanh.output_shift, None);
        assert!(tanh.cache.is_empty());

        let input = Tensor::from_vec(vec![], vec![]);

        tanh.forward(&input, 0, &mut rng);

        assert_eq!(tanh.input_shift, Some(0));
        assert_eq!(tanh.output_shift, Some(0));
    }

    #[test]
    #[should_panic(
        expected="Tanh::backward: No state registered. Perform forward pass first."
    )]
    fn test_tanh_backward_without_forward_cache() {
        let mut tanh = Tanh::new();

        assert_eq!(tanh.input_shift, None);
        assert_eq!(tanh.output_shift, None);
        assert!(tanh.cache.is_empty());

        let input = Tensor::from_vec(vec![], vec![]);

        tanh.backward(&input, Some(0));

    }

    #[test]
    #[should_panic(
        expected="Tanh::backward: No state registered. Perform forward pass first."
    )]
    fn test_tanh_backward_without_input_shift() {
        let mut tanh = Tanh::new();

        assert_eq!(tanh.input_shift, None);
        assert_eq!(tanh.output_shift, None);
        assert!(tanh.cache.is_empty());

        let input = Tensor::from_vec(vec![], vec![]);
        let grad = Tensor::from_vec(vec![], vec![]);
        tanh.cache.push(input);

        tanh.backward(&grad, Some(0));
    }

    #[test]
    #[should_panic(
        expected="Tanh::backward: No state registered. Perform forward pass first."
    )]
    fn test_tanh_backward_without_output_shift() {
        let mut tanh = Tanh::new();

        assert_eq!(tanh.input_shift, None);
        assert_eq!(tanh.output_shift, None);
        assert!(tanh.cache.is_empty());

        let input = Tensor::from_vec(vec![], vec![]);
        let grad = Tensor::from_vec(vec![], vec![]);
        tanh.cache.push(input);
        tanh.input_shift = Some(0);

        tanh.backward(&grad, Some(0));
    }

    #[test]
    fn test_tanh_backward_with_empty_input() {
        let mut tanh = Tanh::new();

        assert_eq!(tanh.input_shift, None);
        assert_eq!(tanh.output_shift, None);
        assert!(tanh.cache.is_empty());

        let input = Tensor::from_vec(vec![], vec![]);
        let grad = Tensor::from_vec(vec![], vec![]);
        tanh.cache.push(input);
        tanh.input_shift = Some(0);
        tanh.output_shift = Some(0);

        let out = tanh.backward(&grad, Some(0));
        assert_eq!(out.shape, vec![]);
        assert_eq!(out.data, vec![]);
    }

    #[test]
    fn test_tanh_backward() {
        let mut tanh = Tanh::new();

        assert_eq!(tanh.input_shift, None);
        assert_eq!(tanh.output_shift, None);
        assert!(tanh.cache.is_empty());

        let input = Tensor::from_vec(vec![-128, -1, 0, 1, 127], vec![5, 1]);
        let grad = Tensor::from_vec(vec![0, 96, 0, -96, 0], vec![5, 1]);
        tanh.cache.push(input);
        tanh.input_shift = Some(0);
        tanh.output_shift = Some(0);

        let out = tanh.backward(&grad, Some(0));
        assert_eq!(out.shape, vec![5, 1]);
        assert_eq!(out.data, vec![0, 94, 0, -95, 0]);
    }

    #[test]
    fn test_tanh_memory_report(){
        let mut tanh = Tanh::new();
        let mut rng = XorShift64::new(420);

        let (stat_, dyn_) = tanh.memory_report();

        assert_eq!(stat_, 0);
        assert_eq!(dyn_, 0);

        let input = Tensor::from_vec(vec![-128, -1, 0, 1, 127], vec![5, 1]);

        tanh.forward(&input, 0, &mut rng);

        let (stat_, dyn_) = tanh.memory_report();

        assert_eq!(stat_, 0);
        assert_eq!(dyn_, 20);

        tanh.forward(&input, 0, &mut rng);

        let (stat_, dyn_) = tanh.memory_report();

        assert_eq!(stat_, 0);
        assert_eq!(dyn_, 40);

        // and the backward pass removes objects from the cache
        tanh.backward(&input, Some(0));

        let (stat_, dyn_) = tanh.memory_report();

        assert_eq!(stat_, 0);
        assert_eq!(dyn_, 20);
    }

    #[test]
    fn test_tanh_describe(){
        let tanh = Tanh::new();

        let info = tanh.describe();

        assert_eq!(info.name, "Tanh");
        assert_eq!(info.params, 0);
        assert_eq!(info.static_bytes, 0);
        assert_eq!(info.children, vec![]);
    }

    #[test]
    #[should_panic(
        expected="No forward call."
    )]
    fn test_tanh_get_output_shift_no_shift(){
        let tanh = Tanh::new();

        tanh.get_output_shift();
    }

    #[test]
    fn test_tanh_get_output_shift(){
        let mut tanh = Tanh::new();
        tanh.output_shift = Some(1);

        assert_eq!(tanh.get_output_shift(), 1u32);
    }
}
