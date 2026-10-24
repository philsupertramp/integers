//! Regress the effective learning-rate discrepancy caused by summed minibatch
//! gradients. No change to the integer arithmetic contract is implied.
use integers::dyadic::{Dyadic, Tensor};
use integers::nn::{Linear, Module};
use integers::mse_grad;

fn one_layer(batch: usize, lr_shift: u32) -> Dyadic {
    let mut layer = Linear::new(1, 1, 0, 0, 31);
    layer.weights.data[0] = Dyadic::new(1, 0);
    layer.biases.data[0] = Dyadic::new(0, 0);
    let inputs = Tensor::from_vec(vec![Dyadic::new(1, 0); batch], vec![batch, 1]);
    let targets = Tensor::from_vec(vec![Dyadic::new(0, 0); batch], vec![batch, 1]);
    layer.zero_grad();
    let outputs = layer.forward_batch(&inputs);
    let gradients: Tensor = outputs.iter().zip(targets.iter())
        .map(|(y, t)| Tensor::from_vec(mse_grad(y.data, t.data), y.shape.to_vec()))
        .collect();
    layer.backward_batch(&gradients);
    layer.update(lr_shift);
    layer.weights.data[0]
}

#[test]
fn summed_gradients_make_batch_32_update_32_times_larger() {
    assert_eq!(one_layer(1, 0), Dyadic::new(0, 0));
    assert_eq!(one_layer(32, 0), Dyadic::new(-31, 0));
}

#[test]
fn batch_size_shift_reproduces_per_sample_step_for_power_of_two_batch() {
    assert_eq!(one_layer(1, 0), one_layer(32, 5));
}
