use crate::Tensor;


pub trait Loss {
    fn forward(&self, preds: &Tensor<i32>, targets: &Tensor<i32>) -> (i32, Tensor<i32>);
}

pub struct MSE;
pub struct MAE;

impl Loss for MSE {
    fn forward(&self, preds: &Tensor<i32>, targets: &Tensor<i32>) -> (i32, Tensor<i32>) {
        assert_eq!(
            preds.len(),
            targets.len(),
            "MSE::forward: vector sizes don't match."
        );
        let mut loss: i64 = 0;
        let mut grad = Tensor::<i32>::new(preds.shape.clone());

        if preds.data.len() == 0 {
            return (0, grad);
        }
        for i in 0..preds.data.len() {
            let error = preds.data[i] as i64 - targets.data[i] as i64;
            // Cast to i32 BEFORE multiplying
            loss = loss.saturating_add((error).saturating_mul(error));
            grad.data[i] = error as i32;
        }
        (loss.clamp(i32::MIN as i64, i32::MAX as i64) as i32 / preds.data.len() as i32, grad)
    }
}

impl Loss for MAE {
    fn forward(&self, preds: &Tensor<i32>, targets: &Tensor<i32>) -> (i32, Tensor<i32>) {
        assert_eq!(
            preds.len(),
            targets.len(),
            "MAE::forward: vector sizes don't match."
        );
        let mut loss: i64 = 0;
        let mut grad = Tensor::<i32>::new(preds.shape.clone());

        if preds.data.len() == 0 {
            return (0i32, grad);
        }
        for i in 0..preds.data.len() {
            let error = preds.data[i] as i32 - targets.data[i] as i32;
            loss += error.abs() as i64;
            grad.data[i] = error.signum();
        }
        (loss.clamp(i32::MIN as i64, i32::MAX as i64) as i32 / (preds.data.len() as i32), grad)
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
