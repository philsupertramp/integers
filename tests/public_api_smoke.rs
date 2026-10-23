//! Exercises the current public dyadic API without loading external datasets.
use integers::{Dyadic, Tensor, mse_grad};
use integers::data::dataset_loaders::{DatasetBuilder, FileFormat, QuantizationMethod};
use integers::nn::{Linear, ReLU, Sequential};

#[test]
fn public_layer_forward_backward_and_update() {
    let s = 7;
    let x = Tensor::from_vec(
        vec![Dyadic::new(64, s), Dyadic::new(-32, s)],
        vec![1, 2],
    );
    let target = Tensor::from_vec(vec![Dyadic::new(64, s)], vec![1, 1]);
    let mut model = Sequential::new();
    model.add(Linear::new(2, 4, s, s, 16).with_grad_clip(8192));
    model.add(ReLU::new());
    model.add(Linear::new(4, 1, s, s, 16));
    model.zero_grad();
    let output = model.forward(&x.view());
    assert_eq!(output.data.len(), 1);
    let gradient = Tensor::from_vec(
        mse_grad(&output.data, &target.data),
        output.shape.clone(),
    );
    let backprop = model.backward(&gradient.view());
    assert_eq!(backprop.data.len(), 2);
    model.update(6);
}

#[test]
fn public_non_generic_builder_path() {
    // Compile the downstream import/method path without touching a dataset.
    let _builder = DatasetBuilder::new_csv("not-loaded.tsv")
        .format(FileFormat::TSV)
        .with_features(vec![0, 1, 2, 3])
        .with_label_column(4)
        .with_quantization(QuantizationMethod::MinMax);
}
