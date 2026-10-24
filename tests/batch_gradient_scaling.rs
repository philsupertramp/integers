//! Regression: built-in integer trainable modules must apply the same
//! per-example mean gradient for full and short batches, before momentum.
use integers::dyadic::{Dyadic, Tensor};
use integers::nn::{Conv2D, Linear, Module};
use integers::mse_grad;

fn linear(batch: usize, momentum: bool, steps: usize) -> (Dyadic, Dyadic) {
    let mut layer = Linear::new(1, 1, 0, 0, 31);
    if momentum { layer = layer.with_momentum(1); }
    layer.weights.data[0] = Dyadic::new(4, 0);
    layer.biases.data[0] = Dyadic::new(0, 0);
    for _ in 0..steps {
        let inputs = Tensor::from_vec(vec![Dyadic::new(1, 0); batch], vec![batch, 1]);
        let targets = Tensor::from_vec(vec![Dyadic::new(0, 0); batch], vec![batch, 1]);
        layer.zero_grad();
        let outputs = layer.forward_batch(&inputs);
        let gradients: Tensor = outputs.iter().zip(targets.iter())
            .map(|(y, t)| Tensor::from_vec(mse_grad(y.data, t.data), y.shape.to_vec()))
            .collect();
        layer.backward_batch(&gradients);
        layer.update(0);
    }
    (layer.weights.data[0], layer.biases.data[0])
}

fn conv(batch: usize, momentum: bool, steps: usize) -> (Dyadic, Dyadic) {
    let mut layer = Conv2D::new(1, 1, 1, 1, 1, 1, 0, 0, 31);
    if momentum { layer = layer.with_momentum(1); }
    layer.kernels.data[0] = Dyadic::new(4, 0);
    layer.biases.data[0] = Dyadic::new(0, 0);
    for _ in 0..steps {
        let inputs = Tensor::from_vec(vec![Dyadic::new(1, 0); batch], vec![batch, 1]);
        let targets = Tensor::from_vec(vec![Dyadic::new(0, 0); batch], vec![batch, 1]);
        layer.zero_grad();
        let outputs = layer.forward_batch(&inputs);
        let gradients: Tensor = outputs.iter().zip(targets.iter())
            .map(|(y, t)| Tensor::from_vec(mse_grad(y.data, t.data), y.shape.to_vec()))
            .collect();
        layer.backward_batch(&gradients);
        layer.update(0);
    }
    (layer.kernels.data[0], layer.biases.data[0])
}

#[test]
fn linear_means_are_invariant_across_power_of_two_and_partial_batches() {
    let expected = linear(1, false, 1);
    assert_eq!(expected, (Dyadic::new(0, 0), Dyadic::new(-4, 0)));
    for n in [3, 7, 24, 31, 32, 64] {
        assert_eq!(linear(n, false, 1), expected, "batch={n}");
    }
}

#[test]
fn conv_means_are_invariant_across_power_of_two_and_partial_batches() {
    let expected = conv(1, false, 1);
    for n in [3, 7, 24, 31, 32, 64] {
        assert_eq!(conv(n, false, 1), expected, "batch={n}");
    }
}

#[test]
fn linear_momentum_uses_mean_gradients() {
    let expected = linear(1, true, 2);
    for n in [3, 24, 32] {
        assert_eq!(linear(n, true, 2), expected, "batch={n}");
    }
}

#[test]
fn conv_momentum_uses_mean_gradients() {
    let expected = conv(1, true, 2);
    for n in [3, 24, 32] {
        assert_eq!(conv(n, true, 2), expected, "batch={n}");
    }
}

#[test]
fn repeated_backward_calls_use_total_sample_count() {
    let mut layer = Linear::new(1, 1, 0, 0, 31);
    layer.weights.data[0] = Dyadic::new(4, 0);
    layer.biases.data[0] = Dyadic::new(0, 0);
    layer.zero_grad();
    for _ in 0..7 {
        let input = Tensor::from_vec(vec![Dyadic::new(1, 0)], vec![1]);
        let output = layer.forward(&input.view());
        let targets = [Dyadic::new(0, 0)];
        let gradient = Tensor::from_vec(mse_grad(&output.data, &targets), vec![1]);
        layer.backward(&gradient.view());
    }
    layer.update(0);
    assert_eq!((layer.weights.data[0], layer.biases.data[0]), linear(1, false, 1));
}

#[test]
fn update_after_zero_grad_with_no_examples_is_no_op() {
    let mut layer = Linear::new(1, 1, 0, 0, 31).with_momentum(1);
    layer.weights.data[0] = Dyadic::new(4, 0);
    layer.zero_grad();
    layer.update(0);
    assert_eq!(layer.weights.data[0], Dyadic::new(4, 0));
}
