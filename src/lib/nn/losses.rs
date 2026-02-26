use crate::Tensor;


pub trait Loss {
    fn forward(&self, preds: &Tensor<i8>, targets: &Tensor<i8>) -> (i32, Tensor<i16>);
}

pub struct MSE;
pub struct MAE;

impl Loss for MSE {
    fn forward(&self, preds: &Tensor<i8>, targets: &Tensor<i8>) -> (i32, Tensor<i16>) {
        assert_eq!(
            preds.len(),
            targets.len(),
            "MSE::forward: vector sizes don't match."
        );
        let mut loss: i32 = 0;
        let mut grad = Tensor::<i16>::new(preds.shape.clone());

        for i in 0..preds.data.len() {
            let error = preds.data[i] as i16 - targets.data[i] as i16;
            // Cast to i32 BEFORE multiplying
            let error_i32 = error as i32;
            loss += error_i32 * error_i32;
            grad.data[i] = error;
        }
        (loss, grad)
    }
}

impl Loss for MAE {
    fn forward(&self, preds: &Tensor<i8>, targets: &Tensor<i8>) -> (i32, Tensor<i16>) {
        assert_eq!(
            preds.len(),
            targets.len(),
            "MAE::forward: vector sizes don't match."
        );
        let mut loss: i32 = 0;
        let mut grad = Tensor::<i16>::new(preds.shape.clone());

        for i in 0..preds.data.len() {
            let error = preds.data[i] as i16 - targets.data[i] as i16;
            loss += error.abs() as i32;
            // dL/dy = 2*(y - t), dropping the 2 it's absorbed by lr
            grad.data[i] = error;
        }
        (loss / (preds.data.len() as i32), grad)
    }
}


