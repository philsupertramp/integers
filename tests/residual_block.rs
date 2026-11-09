//! Small exact integer fixtures for the #39 identity-shortcut POC.
//! They do not establish deep ResNet convergence.
use integers::nn::{Conv2D, Linear, Module, ResidualBlock, Sequential};
use integers::{Dyadic, Tensor};
use integers::rng::seed_rng;

fn d(v: i32) -> Dyadic {
    Dyadic::new(v, 0)
}

fn branch() -> Sequential {
    seed_rng(39);
    let mut conv = Conv2D::new(1, 1, 1, 1, 1, 1, 0, 0, 31);
    conv.kernels.data[0] = d(2);
    conv.biases.data[0] = d(0);
    let mut branch = Sequential::new();
    branch.add(conv);
    branch
}

#[test]
fn single_sample_identity_skip_preserves_gradient_and_updates_branch() {
    let mut block = ResidualBlock::new(branch());
    let x = Tensor::from_vec(vec![d(3)], vec![1]);
    let output = block.forward(&x.view());
    // F(x) = 2x, hence x + F(x) = 3x.
    assert_eq!(output.data, vec![d(9)]);
    let gradient = Tensor::from_vec(vec![d(1)], vec![1]);
    assert_eq!(block.backward(&gradient.view()).data, vec![d(3)]);
    block.update(0);
    // The branch kernel update is −3 and its bias update is −1.
    // New output: x + ((2−3)x − 1) = −1.
    assert_eq!(block.forward(&x.view()).data, vec![d(-1)]);
}

fn nested_model() -> Sequential {
    seed_rng(39);
    let mut lead = Linear::new(1, 1, 0, 0, 31);
    lead.weights.data[0] = d(1);
    lead.biases.data[0] = d(0);
    let mut model = Sequential::new();
    model.add(lead);
    model.add(ResidualBlock::new(branch()));
    model
}

#[test]
fn nested_block_propagates_input_gradients_to_earlier_trainable_layers() {
    let mut train = nested_model();
    let mut full = nested_model();
    let input = Tensor::from_vec(vec![d(3), d(5)], vec![2, 1]);
    let grads = Tensor::from_vec(vec![d(1), d(1)], vec![2, 1]);
    for model in [&mut train, &mut full] {
        model.zero_grad();
        assert_eq!(model.forward_batch(&input).data, vec![d(9), d(15)]);
    }
    train.backward_batch(&grads);
    let dx = full.backward_batch_with_input_grad(&grads);
    assert_eq!(dx.shape, input.shape);
    assert_eq!(dx.data, vec![d(3), d(3)]);

    train.update(0);
    full.update(0);
    let train_y = train.forward_batch(&input);
    let full_y = full.forward_batch(&input);
    assert_eq!(train_y.data, full_y.data, "training and full backward changed parameters differently");
    assert_ne!(train_y.data, vec![d(9), d(15)], "earlier parameter update was lost");
}

#[test]
fn residual_rejects_shape_changing_branch_and_stale_gradients() {
    let mut branch = Sequential::new();
    branch.add(Linear::new(1, 2, 0, 0, 31));
    let mut block = ResidualBlock::new(branch);
    let input = Tensor::from_vec(vec![d(1)], vec![1, 1]);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        block.forward_batch(&input);
    })).is_err());

    let mut valid = ResidualBlock::new(self::branch());
    valid.forward_batch(&input);
    let wrong = Tensor::from_vec(vec![d(1), d(1)], vec![2, 1]);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        valid.backward_batch(&wrong);
    })).is_err());
}
