//! Small, normalization-free identity-skip proof of concept for #39.
//!
//! This deliberately only supports shape-preserving branches and refuses
//! overflowing additions rather than introducing a production clipping/STE
//! policy. No projection, padded convolution or BatchNorm is implied.

use crate::dyadic::{Dyadic, Tensor, TensorView, requantize};
use crate::nn::{Module, Sequential, TrainingArithmetic};

/// Compute a lossless, shape-preserving skip addition at the finer exponent.
///
/// An overflow is a blocked experimental configuration, not silently clipped
/// training data. No stochastic draws are consumed by this alignment.
fn checked_skip_add(a: Dyadic, b: Dyadic) -> Dyadic {
    let shift = a.s.max(b.s);
    let (a, a_overflow) = requantize(a, shift, i32::MIN, i32::MAX);
    let (b, b_overflow) = requantize(b, shift, i32::MIN, i32::MAX);
    assert!(
        !a_overflow && !b_overflow,
        "ResidualBlock: skip-branch scale alignment overflow"
    );
    let total = i64::from(a.v) + i64::from(b.v);
    let mantissa = i32::try_from(total).expect("ResidualBlock: skip addition overflow");
    Dyadic::new(mantissa, shift)
}

fn combine(a: &[Dyadic], b: &[Dyadic], shape: &[usize]) -> Tensor {
    assert_eq!(a.len(), b.len(), "ResidualBlock: branch buffer mismatch");
    Tensor::from_vec(
        a.iter()
            .zip(b.iter())
            .map(|(&left, &right)| checked_skip_add(left, right))
            .collect(),
        shape.to_vec(),
    )
}

/// Identity shortcut: `y = x + F(x)` for a shape-preserving integer branch.
///
/// This research POC has no output clipping or automatic projection. A branch
/// whose output shape changes, or an unrepresentable skip sum, fails closed.
/// The branch is a regular `Sequential` of integer-only modules.
pub struct ResidualBlock {
    pub branch: Sequential,
    input_shape: Option<Vec<usize>>,
}

impl ResidualBlock {
    pub fn new(branch: Sequential) -> Self {
        assert!(!branch.layers.is_empty(), "ResidualBlock: empty branch");
        assert!(
            branch
                .layers
                .iter()
                .all(|layer| layer.training_arithmetic() == TrainingArithmetic::IntegerOnly),
            "ResidualBlock: branch must use integer-only modules"
        );
        Self {
            branch,
            input_shape: None,
        }
    }

    fn check_grad_shape(&self, shape: &[usize]) {
        assert_eq!(
            self.input_shape.as_deref(),
            Some(shape),
            "ResidualBlock: backward shape differs from cached input"
        );
    }
}

impl Module for ResidualBlock {
    fn training_arithmetic(&self) -> TrainingArithmetic {
        if self
            .branch
            .layers
            .iter()
            .all(|layer| layer.training_arithmetic() == TrainingArithmetic::IntegerOnly)
        {
            TrainingArithmetic::IntegerOnly
        } else {
            TrainingArithmetic::Unverified
        }
    }

    fn name(&self) -> &'static str {
        "ResidualBlock"
    }

    fn describe(&self) -> String {
        format!("ResidualBlock(identity + {} layers)", self.branch.layers.len())
    }

    fn forward(&mut self, input: &TensorView) -> Tensor {
        let branch_output = self.branch.forward(input);
        assert_eq!(
            branch_output.shape.as_slice(),
            input.shape,
            "ResidualBlock: branch must preserve the input shape"
        );
        self.input_shape = Some(input.shape.to_vec());
        combine(input.data, &branch_output.data, input.shape)
    }

    fn backward(&mut self, grads: &TensorView) -> Tensor {
        self.check_grad_shape(grads.shape);
        let branch_dx = self.branch.backward(grads);
        assert_eq!(branch_dx.shape.as_slice(), grads.shape);
        combine(grads.data, &branch_dx.data, grads.shape)
    }

    fn forward_batch(&mut self, inputs: &Tensor) -> Tensor {
        let branch_output = self.branch.forward_batch(inputs);
        assert_eq!(
            branch_output.shape, inputs.shape,
            "ResidualBlock: branch must preserve the batch shape"
        );
        self.input_shape = Some(inputs.shape.clone());
        combine(&inputs.data, &branch_output.data, &inputs.shape)
    }

    fn backward_batch(&mut self, grads: &Tensor) -> Tensor {
        self.check_grad_shape(&grads.shape);
        // Unlike ordinary top-level training, the branch input derivative is
        // needed to add to the shortcut derivative.
        let branch_dx = self.branch.backward_batch_with_input_grad(grads);
        assert_eq!(branch_dx.shape, grads.shape);
        combine(&grads.data, &branch_dx.data, &grads.shape)
    }

    fn backward_batch_discard_input_grad(&mut self, grads: &Tensor) {
        self.check_grad_shape(&grads.shape);
        // Outer Sequential has discarded the complete block input derivative.
        // Branch parameter gradients are still required.
        self.branch.backward_batch(grads);
    }

    fn update(&mut self, lr_shift: u32) {
        self.branch.update(lr_shift);
    }

    fn zero_grad(&mut self) {
        self.branch.zero_grad();
    }

    fn set_training(&mut self, training: bool) {
        self.branch.set_training(training);
    }
}

#[cfg(test)]
mod tests {
    use super::checked_skip_add;
    use crate::Dyadic;

    #[test]
    fn skip_add_has_exact_mixed_scale_and_signed_oracles() {
        assert_eq!(
            checked_skip_add(Dyadic::new(3, 1), Dyadic::new(4, 2)),
            Dyadic::new(10, 2)
        );
        assert_eq!(
            checked_skip_add(Dyadic::new(-6, 2), Dyadic::new(1, 1)),
            Dyadic::new(-4, 2)
        );
    }

    #[test]
    fn skip_add_rejects_unrepresentable_results() {
        assert!(std::panic::catch_unwind(|| {
            checked_skip_add(Dyadic::new(i32::MAX, 0), Dyadic::new(1, 0))
        }).is_err());
        assert!(std::panic::catch_unwind(|| {
            checked_skip_add(Dyadic::new(i32::MAX, 0), Dyadic::new(1, 2))
        }).is_err());
    }
}
