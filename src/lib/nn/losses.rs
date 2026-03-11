use crate::{Tensor, Scalar, Numeric};

pub trait Loss<S: Scalar> {
    fn forward(&self, preds: &Tensor<S>, targets: &Tensor<S>) -> (S::Acc, Tensor<S::Acc>);
}

pub struct MSE;
pub struct MAE;

impl<S: Scalar> Loss<S> for MSE {
    fn forward(&self, preds: &Tensor<S>, targets: &Tensor<S>) -> (S::Acc, Tensor<S::Acc>) {
        assert_eq!(
            preds.shape, targets.shape,
            "MSE::forward: Tensor shapes don't match."
        );
        assert_eq!(
            preds.shape.len(), 2,
            "MSE::forward: Expected 2D tensors [batch, dim]."
        );

        let batch_size = preds.shape[0];
        let dim = preds.shape[1];

        let mut total_loss: S::Acc = S::Acc::zero();
        let mut grad = Tensor::<S::Acc>::new(preds.shape.clone());

        if batch_size == 0 {
            return (total_loss, grad);
        }

        // Standard reduction='mean' divides by total elements (batch_size * dim)
        let n_elements = S::Acc::from_i32((batch_size * dim) as i32);
        
        // For the gradient, we scale the error by the batch size (or total elements depending on convention)
        let grad_divisor = S::Acc::from_i32(batch_size as i32);

        for b in 0..batch_size {
            for d in 0..dim {
                let i = b * dim + d;
                let error = preds.data[i].sub(targets.data[i]);
                
                // Accumulate squared error for the loss
                total_loss = total_loss.add(error.mul(error));
                
                // The gradient of MSE with respect to the output is (pred - target) / batch_size
                grad.data[i] = error;//.div(grad_divisor); 
            }
        }

        (total_loss.div(n_elements), grad)
    }
}

impl<S: Scalar> Loss<S> for MAE {
    fn forward(&self, preds: &Tensor<S>, targets: &Tensor<S>) -> (S::Acc, Tensor<S::Acc>) {
        assert_eq!(
            preds.shape, targets.shape,
            "MAE::forward: Tensor shapes don't match."
        );
        assert_eq!(
            preds.shape.len(), 2,
            "MAE::forward: Expected 2D tensors [batch, dim]."
        );

        let batch_size = preds.shape[0];
        let dim = preds.shape[1];

        let mut total_loss: S::Acc = S::Acc::zero();
        let mut grad = Tensor::<S::Acc>::new(preds.shape.clone());

        if batch_size == 0 {
            return (total_loss, grad);
        }

        let n_elements = S::Acc::from_i32((batch_size * dim) as i32);
        let grad_divisor = S::Acc::from_i32(batch_size as i32);

        for b in 0..batch_size {
            for d in 0..dim {
                let i = b * dim + d;
                let error = preds.data[i].sub(targets.data[i]);
                
                total_loss = total_loss.add(error.abs());
                
                // Derivative of |x| is sign(x)
                let sign = S::Acc::from_i32(error.signum());
                grad.data[i] = sign;//sign.div(grad_divisor);
            }
        }

        (total_loss.div(n_elements), grad)
    }
}
