use crate::{Tensor, Scalar, Numeric};


pub trait Loss<S: Scalar> {
    fn forward(&self, preds: &Tensor<S>, targets: &Tensor<S>) -> (S::Acc, Tensor<S::Acc>);
}

pub struct MSE;
pub struct MAE;

impl<S: Scalar> Loss<S> for MSE {
    fn forward(&self, preds: &Tensor<S>, targets: &Tensor<S>) -> (S::Acc, Tensor<S::Acc>) {
        assert_eq!(
            preds.len(),
            targets.len(),
            "MSE::forward: vector sizes don't match."
        );
        let mut loss: S::Acc = S::Acc::zero();
        let mut grad = Tensor::<S::Acc>::new(preds.shape.clone());

        if preds.data.len() == 0 {
            return (S::Acc::zero(), grad);
        }
        for i in 0..preds.data.len() {
            let error = preds.data[i].sub(targets.data[i]);
            // Cast to i32 BEFORE multiplying
            loss = loss.add((error).mul(error));
            grad.data[i] = error;
        }
        (loss.div(S::Acc::from_i32(preds.data.len() as i32)), grad)
    }
}

impl<S: Scalar> Loss<S> for MAE {
    fn forward(&self, preds: &Tensor<S>, targets: &Tensor<S>) -> (S::Acc, Tensor<S::Acc>) {
        assert_eq!(
            preds.len(),
            targets.len(),
            "MAE::forward: vector sizes don't match."
        );
        let mut loss: S::Acc = S::Acc::zero();
        let mut grad = Tensor::<S::Acc>::new(preds.shape.clone());

        if preds.data.len() == 0 {
            return (S::Acc::zero(), grad);
        }
        for i in 0..preds.data.len() {
            let error = preds.data[i].sub(targets.data[i]);
            loss = loss.add(error.abs());
            grad.data[i] = S::Acc::from_i32(error.signum());
        }
        (loss.div(S::Acc::from_i32(preds.data.len() as i32)), grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse(){
        let loss = MSE;

        let pred = Tensor::from_vec(vec![1, 1], vec![2, 1]);
        let targets = Tensor::from_vec(vec![0, 1], vec![2, 1]);

        let (loss, grad) = loss.forward(&pred, &targets);

        assert_eq!(loss, 0);
        assert_eq!(grad, Tensor::from_vec(vec![1, 0], vec![2, 1]));
    }

    #[test]
    #[should_panic(
        expected = "assertion `left == right` failed:"
    )]
    fn test_mse_wrong_shapes(){
        let loss = MSE;

        let pred = Tensor::from_vec(vec![1, 1], vec![2, 1]);
        let targets = Tensor::from_vec(vec![0], vec![1, 1]);

        loss.forward(&pred, &targets);
    }

    #[test]
    fn test_mse_empty_data(){
        let loss = MSE;

        let pred = Tensor::from_vec(vec![], vec![]);
        let targets = Tensor::from_vec(vec![], vec![]);

        let (loss, grad) = loss.forward(&pred, &targets);

        assert_eq!(loss, 0);
        assert_eq!(grad, Tensor::from_vec(vec![], vec![]));
    }

    #[test]
    fn test_mae(){
        let loss = MAE;

        let pred = Tensor::from_vec(vec![1, 1], vec![2, 1]);
        let targets = Tensor::from_vec(vec![0, 1], vec![2, 1]);

        let (loss, grad) = loss.forward(&pred, &targets);

        assert_eq!(loss, 0);
        assert_eq!(grad, Tensor::from_vec(vec![1, 0], vec![2, 1]));
    }

    #[test]
    fn test_mae_empty_data(){
        let loss = MAE;

        let pred = Tensor::from_vec(vec![], vec![]);
        let targets = Tensor::from_vec(vec![], vec![]);

        let (loss, grad) = loss.forward(&pred, &targets);

        assert_eq!(loss, 0);
        assert_eq!(grad, Tensor::from_vec(vec![], vec![]));
    }

    #[test]
    #[should_panic(
        expected = "assertion `left == right` failed:"
    )]
    fn test_mae_wrong_shapes(){
        let loss = MAE;

        let pred = Tensor::from_vec(vec![1, 1], vec![2, 1]);
        let targets = Tensor::from_vec(vec![0], vec![1, 1]);

        loss.forward(&pred, &targets);
    }

}
