#[path = "lib/debug.rs"]
pub mod debug;
#[path = "lib/nn.rs"]
pub mod nn;
#[path = "lib/data.rs"]
pub mod data;
#[path = "lib/dataset_loaders.rs"]
pub mod dataset_loaders;
#[path = "lib/quant.rs"]
pub mod quant;

use std::fmt;
use std::ops::{Shr, Shl};

use crate::nn::kernels;


pub trait Numeric: Copy + Clone + Default + fmt::Debug + PartialEq {
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
    fn div(self, other: Self) -> Self;
    fn sqrt(self) -> Self;
    fn zero() -> Self { Self::default() }
    fn abs(self) -> Self;
    fn from_i32(val: i32) -> Self;
    fn gt(self, other: Self) -> bool;
    fn to_u32(self) -> u32;
    fn signum(self) -> i32;
    fn shr(self, shift: u32) -> Self;
    fn shl(self, shift: u32) -> Self;
    fn max(self, other: Self) -> Self;
}

impl Numeric for f32 {
    fn add(self, other: Self) -> Self { self + other }
    fn sub(self, other: Self) -> Self { self - other }
    fn mul(self, other: Self) -> Self { self * other }
    fn div(self, other: Self) -> Self { self / other }
    fn sqrt(self) -> Self { f32::sqrt(self) }
    fn from_i32(val: i32) -> f32 { val as f32 }
    fn abs(self) -> f32 { self.abs() }
    fn gt(self, other: f32) -> bool { self > other }
    fn to_u32(self) -> u32 { self as u32 }
    fn signum(self) -> i32 { if self > 0.0 { 1 } else if self == 0.0 { 0 } else { -1 } }
    fn shr(self, shift: u32) -> f32 { self / (1u32 << shift) as f32 }
    fn shl(self, shift: u32) -> f32 { self * (1u32 << shift) as f32 }
    fn max(self, other: f32) -> f32 { if self > other { self } else { other }}
}

impl Numeric for i32 {
    fn add(self, other: Self) -> Self { self.saturating_add(other) }
    fn sub(self, other: Self) -> Self { self.saturating_sub(other) }
    fn mul(self, other: Self) -> Self { self.saturating_mul(other) }
    fn div(self, other: Self) -> Self { if other == 0 { 0 } else { self / other } }
    fn sqrt(self) -> Self { let v = if self > 0 { self as u64 } else { 0u64 }; kernels::isqrt_64(v) as i32 }
    fn from_i32(val: i32) -> i32 { val }
    fn abs(self) -> i32 { if self > 0 { self } else { -1 * self }}
    fn gt(self, other: i32) -> bool { self > other }
    fn to_u32(self) -> u32 { if self < 0i32 { 0u32 } else { self as u32 } }
    fn signum(self) -> i32 { if self > 0 { 1 } else if self == 0 { 0 } else { -1 } }
    fn shr(self, shift: u32) -> i32 { self >> shift }
    fn shl(self, shift: u32) -> i32 { self.saturating_mul(1i32 << shift) }
    fn max(self, other: i32) -> i32 { if self > other { self } else { other }}
}

pub trait Scalar: Copy + Clone + Default + fmt::Debug + PartialOrd {
    type Acc: Numeric;

    /// Casts `acc` (accumulated value) to desired quantization using `shift` (and `rng` for
    /// stochastic downcasting)
    fn downcast(acc: Self::Acc, shift: u32, rng: &mut XorShift64) -> Self;

    /// Casts own value to accumulator type
    fn into_acc(self) -> Self::Acc;

    fn random_uniform(rng: &mut XorShift64, range: Self::Acc) -> Self::Acc;

    fn from_i32(val: i32) -> Self::Acc;
    fn from_f64(val: f64) -> Self::Acc;
    fn from_quantized(val: i32) -> Self;
    fn dataset_input_shift(quantizer_shift: u32) -> u32;

    fn to_u32(self) -> u32;

    /// Multiplies two scalars and produces Acc
    /// for instance two i8 values produce a i32 value
    fn mul(self, other: Self) -> Self::Acc;
    
    fn sub(self, other: Self) -> Self::Acc;

    /// Absolute value of scalar
    fn abs(self) -> Self;

    fn relu(value: Self) -> Self;

    fn tanh(value: Self) -> Self;
    fn dtanh(value: Self) -> Self::Acc;

    fn is_positive(value: Self) -> bool;

    /// default shift when converting into base representation,
    /// only required for integer types, otherwise 0
    fn unit_shift() -> u32;

    fn acc_shift(fan_in: usize) -> u32;
}

impl Scalar for f32 {
    type Acc = f32;

    fn downcast(acc: f32, shift: u32, rng: &mut XorShift64) -> Self { acc / (1 << shift) as f32 }

    fn random_uniform(rng: &mut XorShift64, range: f32) -> f32 {
        // generate float in [-range, range] properly
        let u = rng.next() as f32 / u64::MAX as f32; // [0, 1)
        u * 2.0 * range - range
    }

    fn to_u32(self) -> u32 { self as u32 }
    fn from_i32(val: i32) -> f32 { val as f32 }
    fn from_f64(val: f64) -> f32 { val as f32 }
    fn into_acc(self) -> f32 { self }
    fn mul(self, other: f32) -> f32 { self * other }
    fn sub(self, other: Self) -> f32 { self - other }
    fn abs(self) -> f32 { if self > 0.0 { self } else { -1.0 * self }}
    fn unit_shift() -> u32 { 0u32 }
    fn acc_shift(fan_in: usize) -> u32 { 0u32 }
    
    fn from_quantized(val: i32) -> f32 { val as f32 / 127.0 }
    fn dataset_input_shift(_: u32) -> u32 { 0 }

    fn is_positive(value: f32) -> bool { value > 0.0 }
    fn relu(value: f32) -> f32 {
        if value > 0.0 {
            value
        } else {
            0.0
        }
    }
    fn tanh(value: f32) -> f32 { value.tanh() }
    fn dtanh(value: f32) -> f32 { 1.0 - value * value }
}

impl Scalar for i32 {
    type Acc = i32;

    fn downcast(acc: i32, shift: u32, rng: &mut XorShift64) -> Self { kernels::stochastic_downcast(acc, shift, rng) }

    fn random_uniform(rng: &mut XorShift64, range: Self::Acc) -> i32 {
        let range: Self::Acc = Numeric::max(range, <i32 as Numeric>::from_i32(1i32));
        let spread = (2 * range as u32) as u32;
        rng.gen_range(spread) as i32 - range
    }

    fn to_u32(self) -> u32 { if self < 0i32 { 0u32 } else { self as u32 } }
    fn from_i32(val: i32) -> i32 { val as i32 }
    fn from_f64(val: f64) -> i32 { (val * 127.0).round() as i32 }
    fn into_acc(self) -> i32 { self }
    fn mul(self, other: i32) -> i32 { self * other }
    fn sub(self, other: i32) -> i32 { self.saturating_sub(other) }
    fn abs(self) -> i32 { if self > 0 { self } else { -1 * self }}
    
    fn from_quantized(val: i32) -> i32 { val }
    fn dataset_input_shift(quantizer_shift: u32) -> u32 { quantizer_shift }

    fn unit_shift() -> u32 { 15u32 }
    fn acc_shift(fan_in: usize) -> u32 { (usize::BITS - fan_in.leading_zeros()) as u32 }
    fn is_positive(value: i32) -> bool { value > 0 }
    fn relu(value: i32) -> i32 {
        if value > 0 {
            value
        } else {
            0
        }
    }
    fn tanh(value: i32) -> i32 { kernels::tanh_i8(value as i8) as i32 }
    fn dtanh(value: i32) -> i32 { (127 * 127) - (value * value) }
}


#[derive(Clone, Debug, PartialEq)]
pub struct Tensor<T>
where
    T: Clone + Copy + fmt::Debug + Default,
{
    /// Flattened data storage (Row-Major contiguous layout)
    pub data: Vec<T>,
    /// Dimension of the tensor, e.g. [batch, input_dim]
    pub shape: Vec<usize>,
}

impl<T> Tensor<T>
where
    T: Clone + Copy + fmt::Debug + Default,
{
    pub fn new(shape: Vec<usize>) -> Self {
        let mut total_elements: usize = 0;

        if !shape.is_empty() {
            total_elements = shape.iter().product();
        }
        Self {
            data: vec![T::default(); total_elements],
            shape,
        }
    }

    pub fn from_vec(data: Vec<T>, shape: Vec<usize>) -> Self {
        let mut expected = 0;
        if !shape.is_empty() {
            expected = shape.iter().product();
        }
        assert_eq!(
            data.len(),
            expected,
            "Tensor::from_vec: Shape {:?} does not match data len {}",
            shape,
            data.len()
        );
        Self { data, shape }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn memory_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<T>()
    }
}

///
/// Right shift operator for Tensors
///
/// Example:
/// ```
/// use integers::Tensor;
/// let t: Tensor<i32> = Tensor::from_vec(vec![4], vec![1, 1]);
/// let o = t >> 1u32;
/// assert_eq!(o.len(), 1);
/// assert_eq!(o.data[0], 2);
/// ```
impl<S: Scalar + Numeric> Shr<u32> for Tensor<S>
{
    type Output = Tensor<S>;

    fn shr(self, shift: u32) -> Tensor<S> {
        let shifted_data = self
            .data
            .into_iter()
            .map(|val| val.shr(shift))
            .collect();

        Tensor {
            data: shifted_data,
            shape: self.shape,
        }
    }
}

///
/// Right shift operator for Tensors [by reference]
///
/// Example:
/// ```
/// use integers::Tensor;
/// let t: Tensor<i32> = Tensor::from_vec(vec![4], vec![1, 1]);
/// let o = &t >> 1u32;
/// assert_eq!(o.len(), 1);
/// assert_eq!(o.data[0], 2);
/// assert_eq!(t.data[0], 4);
/// ```
impl<S: Scalar + Numeric> Shr<u32> for &Tensor<S>
{
    type Output = Tensor<S>;

    fn shr(self, shift: u32) -> Tensor<S> {
        let shifted_data = self
            .data
            .iter()
            .map(|&val| val.shr(shift))
            .collect();

        Tensor {
            data: shifted_data,
            shape: self.shape.clone(),
        }
    }
}

///
/// Left shift operator for Tensors
///
/// Example:
/// ```
/// use integers::Tensor;
/// let t: Tensor<i32> = Tensor::from_vec(vec![4], vec![1, 1]);
/// let o = t << 1u32;
/// assert_eq!(o.len(), 1);
/// assert_eq!(o.data[0], 8);
/// ```
impl<T> Shl<u32> for Tensor<T>
where
    T: Clone + Copy + fmt::Debug + Default + Shl<u32, Output = T>,
{
    type Output = Tensor<T>;

    fn shl(self, shift: u32) -> Tensor<T> {
        let shifted_data = self
            .data
            .into_iter()
            .map(|val| val << shift)
            .collect();

        Tensor {
            data: shifted_data,
            shape: self.shape,
        }
    }
}

///
/// Left shift operator for Tensors [by reference]
///
/// Example:
/// ```
/// use integers::Tensor;
/// let t: Tensor<i32> = Tensor::from_vec(vec![4], vec![1, 1]);
/// let o = &t << 1u32;
/// assert_eq!(o.len(), 1);
/// assert_eq!(o.data[0], 8);
/// assert_eq!(t.data[0], 4);
/// ```
impl<T> Shl<u32> for &Tensor<T>
where
    T: Clone + Copy + fmt::Debug + Default + Shl<u32, Output = T>,
{
    type Output = Tensor<T>;

    fn shl(self, shift: u32) -> Tensor<T> {
        let shifted_data = self
            .data
            .iter()
            .map(|&val| val << shift)
            .collect();

        Tensor {
            data: shifted_data,
            shape: self.shape.clone(),
        }
    }
}


pub fn argmax<S: Scalar>(tensor: &Tensor<S>, axis: Option<usize>) -> Vec<usize> {
    let axis = axis.unwrap_or(1);
    
    // Only supports 2D tensors
    if tensor.shape.len() != 2 {
        panic!("argmax requires a 2D tensor, got shape: {:?}", tensor.shape);
    }
    
    let (rows, cols) = (tensor.shape[0], tensor.shape[1]);
    let mut result = Vec::new();
    
    if axis == 1 {
        // Find argmax along columns (per row)
        // For each row, find the column index with max value
        for row_idx in 0..rows {
            let start = row_idx * cols;
            let row = &tensor.data[start..start + cols];
            
            // Find index of maximum value in this row
            let (max_idx, _) = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &S::default()));
            
            result.push(max_idx);
        }
    } else if axis == 0 {
        // Find argmax along rows (per column)
        // For each column, find the row index with max value
        for col_idx in 0..cols {
            let (max_idx, _) = (0..rows)
                .map(|r| (r, tensor.data[r * cols + col_idx]))
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, S::default()));
            
            result.push(max_idx);
        }
    } else {
        panic!("Invalid axis {}. Expected 0 or 1.", axis);
    }
    
    result
}

pub fn accuracy(predictions: &[usize], ground_truth: &[u8]) -> f32 {
    let correct = predictions
        .iter()
        .zip(ground_truth)
        .filter(|(pred, truth)| **pred == **truth as usize)
        .count();
    
    correct as f32 / predictions.len() as f32
}


/// RNG State
pub struct XorShift64 {
    pub state: u64,
}

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        // edge case handleing for state = 0
        let state = if seed == 0 { 0xCAFEBABE } else { seed };
        Self { state }
    }

    pub fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Random value generator for value in range [0, range)
    #[inline(always)]
    pub fn gen_range(&mut self, range: u32) -> u32 {
        (self.next() as u32) % range
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    // Tensor<T> test suite
    #[test]
    fn test_tensor_new() {
        let t: Tensor<i32> = Tensor::new(vec![2, 4]);

        assert_eq!(t.shape[0], 2);
        assert_eq!(t.shape[1], 4);
        assert_eq!(t.data.len(), 8);
    }

    #[test]
    fn test_tensor_new_empty_vec() {
        let t: Tensor<i32> = Tensor::new(vec![]);

        assert_eq!(t.shape.len(), 0);
        assert_eq!(t.data.len(), 0);
    }

    #[test]
    fn test_tensor_from_vec() {
        let t: Tensor<i32> = Tensor::from_vec(vec![2, 1], vec![2, 1]);
        assert_eq!(t.data.len(), t.shape[0] * t.shape[1]);
    }

    #[test]
    fn test_tensor_from_vec_empty_vec() {
        let t: Tensor<i32> = Tensor::from_vec(vec![], vec![]);
        assert_eq!(t.shape.len(), 0);
        assert_eq!(t.data.len(), 0);
    }

    #[test]
    #[should_panic(
        expected = "assertion `left == right` failed: Tensor::from_vec: Shape [2, 1] does not match data len 3\n  left: 3\n right: 2"
    )]
    fn test_tensor_from_vec_wrong_shape_for_data() {
        let _t: Tensor<i32> = Tensor::from_vec(vec![2, 1, 3], vec![2, 1]);
    }

    #[test]
    fn test_tensor_len() {
        let t: Tensor<i32> = Tensor::from_vec(vec![2, 1], vec![2, 1]);

        assert_eq!(t.len(), 2);
    }
    #[test]
    fn test_tensor_len_empty() {
        let t: Tensor<i32> = Tensor::new(vec![]);

        assert_eq!(t.len(), 0, "{:?}", t);
    }

    #[test]
    fn test_tensor_memory_bytes() {
        let t: Tensor<i32> = Tensor::from_vec(vec![2], vec![1, 1]);

        assert_eq!(t.len(), 1);
        assert_eq!(t.memory_bytes(), 4);
    }

    #[test]
    fn test_tensor_shl_borrow(){
        let t: Tensor<i32> = Tensor::from_vec(vec![4], vec![1, 1]);

        let o = t << 1u32;
        assert_eq!(o.len(), 1);
        assert_eq!(o.data[0], 8);
        // not allowed
        // assert_eq!(t.data[0], 4);
    }

    #[test]
    fn test_tensor_shl_reference(){
        let t: Tensor<i32> = Tensor::from_vec(vec![4], vec![1, 1]);

        let o = &t << 1u32;
        assert_eq!(o.len(), 1);
        assert_eq!(o.data[0], 8);
        assert_eq!(t.data[0], 4);
    }

    #[test]
    fn test_tensor_shr_borrow(){
        let t: Tensor<i32> = Tensor::from_vec(vec![4], vec![1, 1]);

        let o = t >> 1u32;
        assert_eq!(o.len(), 1);
        assert_eq!(o.data[0], 2);
        // not allowed
        // assert_eq!(t.data[0], 4);
    }

    #[test]
    fn test_tensor_shr_reference(){
        let t: Tensor<i32> = Tensor::from_vec(vec![4], vec![1, 1]);

        let o = &t >> 1u32;
        assert_eq!(o.len(), 1);
        assert_eq!(o.data[0], 2);
        assert_eq!(t.data[0], 4);
    }

    #[test]
    fn test_argmax_batch() {
        // Batch of 3 samples, 4 classes each
        let data = vec![
            10i32, 5, 3, 2,      // Sample 0: max at index 0
            2, 15, 8, 1,        // Sample 1: max at index 1
            1, 2, 20, 5,        // Sample 2: max at index 2
        ];
        let tensor = Tensor::from_vec(data, vec![3, 4]);
        
        let result = argmax(&tensor, Some(1));
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn test_argmax_axis_0() {
        // 3 rows, 2 columns
        let data = vec![
            10i32, 2,
            5, 15,
            3, 1,
        ];
        let tensor = Tensor::from_vec(data, vec![3, 2]);
        
        let result = argmax(&tensor, Some(0));
        // Column 0: max is 10 at row 0
        // Column 1: max is 15 at row 1
        assert_eq!(result, vec![0, 1]);
    }

    #[test]
    fn test_argmax_single_sample() {
        // Single sample [1, 5]
        let data = vec![2i32, 8, 5, 3, 1];
        let tensor = Tensor::from_vec(data, vec![1, 5]);
        
        let result = argmax(&tensor, Some(1));
        assert_eq!(result, vec![1]);  // Max is 8 at index 1
    }

    // accuracy tests
    // TODO
    #[test]
    fn test_accuracy(){
        let preds: &[usize] = &[0, 1, 0, 1];
        let ground_truth: &[u8] = &[0, 1, 0, 1];

        assert_eq!(accuracy(preds, ground_truth), 1.0);
    }


    // XorShift64 tests
    #[test]
    fn test_xorshift_new() {
        let rng1 = XorShift64::new(12345);
        assert_eq!(rng1.state, 12345);
        let rng2 = XorShift64::new(0);
        assert_eq!(rng2.state, 0xCAFEBABE);
    }

    #[test]
    fn test_xorshift_next() {
        let mut rng = XorShift64::new(1);

        assert_eq!(rng.next(), 1082269761);
        assert_eq!(rng.state, 1082269761);
    }

    #[test]
    fn test_xorshift_determinism() {
        let mut rng1 = XorShift64::new(12345);
        let mut rng2 = XorShift64::new(12345);

        assert_eq!(rng1.next(), rng2.next());
        assert_eq!(rng1.next(), rng2.next());
        assert_eq!(rng1.next(), rng2.next());
    }

    #[test]
    fn test_xorshift_gen_range() {
        let mut rng = XorShift64::new(1);

        for _i in 0..1000 {
            let val = rng.gen_range(12) + 1;
            assert!(val <= 12);
            assert!(val > 0);
        }
    }

    #[test]
    fn test_scalar_tensor(){
        let tensor = Tensor::<f32>::new(vec![1, 1]);
    }

}
