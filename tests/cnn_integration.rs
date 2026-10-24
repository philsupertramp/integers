//! Offline issue #35 integrated CNN regression fixtures.
//! Exact hand-calculated outputs are independent of the production dyadic
//! arithmetic helper implementations. No MNIST files required.
use integers::dyadic::{Dyadic, Tensor};
use integers::nn::{Conv2D, Flatten, Linear, MaxPool2D, Module, ReLU};
use integers::{mse_grad};
use integers::rng::seed_rng;

#[path = "../experiments/cnn_reference.rs"]
mod cnn_reference;

fn d(x: i32) -> Dyadic { Dyadic::new(x, 0) }

#[test]
fn independent_two_channel_cnn_oracle_through_dense_backward_and_update() {
    // Two independent 3x3 channels, 2x2 valid kernels, ReLU, 2x2 pool,
    // Flatten, Linear(2->1). Channel first, first-index pool ties.
    let mut conv = Conv2D::new(2, 2, 2, 2, 3, 3, 0, 0, 31);
    conv.kernels.data.fill(d(0));
    for p in 0..4 { conv.kernels.data[p] = d(1); }
    for p in 12..16 { conv.kernels.data[p] = d(1); }
    conv.biases.data.fill(d(0));
    let mut relu = ReLU::new();
    let mut pool = MaxPool2D::new(2, 2, 2, 2, 2);
    let mut flatten = Flatten;
    let mut dense = Linear::new(2, 1, 0, 0, 31);
    dense.weights.data = vec![d(1), d(2)];
    dense.biases.data = vec![d(0)];

    let input = Tensor::from_vec((1..=18).map(d).collect(), vec![18]);
    let a = conv.forward(&input.view());
    assert_eq!(a.data.iter().map(|v| v.v).collect::<Vec<_>>(),
        [12,16,24,28,48,52,60,64]);
    let b = relu.forward(&a.view());
    let c = pool.forward(&b.view());
    assert_eq!(c.data.iter().map(|v| v.v).collect::<Vec<_>>(), [28,64]);
    let flat = flatten.forward(&c.view());
    assert_eq!(flat.data, c.data);
    let y = dense.forward(&flat.view());
    assert_eq!(y.data, vec![d(156)]);
    // 1/2 * (156 - 155)^2; gradient is 1, not cross-entropy p-t.
    assert_eq!(mse_grad(&y.data, &[d(155)]), vec![d(1)]);
    let g = Tensor::from_vec(vec![d(1)], vec![1]);
    let dc = dense.backward(&g.view());
    assert_eq!(dc.data, vec![d(1),d(2)]);
    let db = flatten.backward(&dc.view());
    let da = pool.backward(&db.view());
    assert_eq!(da.data.iter().map(|v|v.v).collect::<Vec<_>>(), [0,0,0,1,0,0,0,2]);
    let di = relu.backward(&da.view());
    let dx = conv.backward(&di.view());
    let expected = [0,0,0,0,1,1,0,1,1,0,0,0,0,2,2,0,2,2];
    assert_eq!(dx.data.iter().map(|v|v.v).collect::<Vec<_>>(), expected);

    conv.update(0);
    dense.update(0);
    assert_eq!(dense.weights.data, vec![d(-27),d(-62)]);
    assert_eq!(dense.biases.data, vec![d(-1)]);
    assert_eq!(conv.kernels.data[0..4], [d(-4),d(-5),d(-7),d(-8)]);
    assert_eq!(conv.kernels.data[12..16], [d(-27),d(-29),d(-33),d(-35)]);
    assert_eq!(conv.biases.data, vec![d(-1),d(-2)]);
}

fn one_step(count: usize) -> (Vec<Dyadic>, Vec<Dyadic>, Vec<Dyadic>, Vec<Dyadic>) {
    seed_rng(0x3500);
    let mut conv = Conv2D::new(1, 1, 1, 1, 1, 1, 0, 0, 31).with_momentum(1);
    let mut dense = Linear::new(1, 1, 0, 0, 31).with_momentum(1);
    conv.kernels.data[0] = d(1); conv.biases.data[0] = d(0);
    dense.weights.data[0] = d(1); dense.biases.data[0] = d(0);
    let input = Tensor::from_vec(vec![d(1);count], vec![count,1]);
    let targets = Tensor::from_vec(vec![d(0);count], vec![count,1]);
    let a = conv.forward_batch(&input);
    let y = dense.forward_batch(&a);
    let gradients: Tensor = y.iter().zip(targets.iter())
        .map(|(out, target)| Tensor::from_vec(mse_grad(out.data,target.data),vec![1]))
        .collect();
    let ga = dense.backward_batch(&gradients);
    let dx = conv.backward_batch(&ga);
    assert_eq!(dx.shape, vec![count,1]);
    conv.update(0); dense.update(0);
    (conv.kernels.data, conv.biases.data, dense.weights.data, dense.biases.data)
}

#[test]
fn cnn_uses_actual_batch_count_before_momentum_for_full_and_partial_batches() {
    let expected = one_step(1);
    for count in [3,7,31,32] {
        assert_eq!(one_step(count), expected, "batch count {count}");
    }
}

#[test]
fn cache_replacement_mismatch_and_zero_gradient_updates() {
    let mut conv = Conv2D::new(2, 1, 1, 1, 2, 2, 0, 0, 31);
    conv.kernels.data.fill(d(1));
    let batch = Tensor::from_vec(vec![d(1);16], vec![2,8]);
    let output = conv.forward_batch(&batch);
    assert_eq!(output.shape, vec![2,4]);
    let grad = Tensor::from_vec(vec![d(0);8], vec![2,4]);
    let before = conv.kernels.data.clone();
    conv.backward_batch(&grad);
    conv.update(0);
    assert_eq!(conv.kernels.data, before);
    conv.zero_grad();
    conv.update(0);
    assert_eq!(conv.kernels.data, before);
    let smaller = Tensor::from_vec(vec![d(2);8], vec![1,8]);
    conv.forward_batch(&smaller); // replaces cache of a 2-sample forward
    let stale = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        conv.backward_batch(&grad);
    }));
    assert!(stale.is_err(), "stale batch gradients must not be accepted");
}

#[test]
fn fixed_cnn_topology_produces_ten_logits_and_784_input_gradients() {
    let (mut model, mut reference) = cnn_reference::build_models(1337);
    let input = Tensor::from_vec(vec![Dyadic::new(0, 7);784],vec![784]);
    let output = model.forward(&input.view());
    assert_eq!(output.shape, vec![10]);
    assert_eq!(reference.forward(&vec![0.0;784]).len(),10);
    let grad = Tensor::from_vec(vec![Dyadic::new(0,7);10],vec![10]);
    let dx = model.backward(&grad.view());
    assert_eq!(dx.shape,vec![784]);
    assert!(dx.data.iter().all(|x|x.v == 0));
}
