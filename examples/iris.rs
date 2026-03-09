use integers::*;
use integers::nn::*;
use integers::nn::losses::*;
use integers::nn::activations::{ReLU, Tanh};
use integers::dataset_loaders::*;
use integers::data::{shuffled_indices};
use integers::debug::*;
use integers::nn::optim::{SGDConfig, AdamConfig};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = XorShift64::new(42);
    let epochs: i32 = 30000;
    let batch_size: usize = 32;
    let mut optim = SGDConfig::new();
    optim.lr_shift = 10;
    optim.momentum_shift = Some(0);

    let mut l1 = Linear::<i32>::new(4, 8);
    let mut l2 = Linear::<i32>::new(8, 8);
    let mut l3 = Linear::<i32>::new(8, 3);

    l1.init(&mut rng);
    l2.init(&mut rng);
    l3.init(&mut rng);

    println!("L1 {}: {} -> {}", l1.weights.quant_shift, l1.input_shift, l1.output_shift);
    println!("L2 {}: {} -> {}", l2.weights.quant_shift, l2.input_shift, l2.output_shift);
    println!("L3 {}: {} -> {}", l3.weights.quant_shift, l3.input_shift, l3.output_shift);
    
    // Build model for Iris: 4 input features → 3 output classes
    let mut model = Sequential::<i32>::new();
    model
        .add(l1)
        .add(ReLU::<i32>::new())
        .add(l2)
        .add(ReLU::<i32>::new())
        .add(l3);

    // Load datasets (unwrap Results with ?)
    let train_ds = DatasetBuilder::<i32>::new("data/iris_train.tsv")
        .format(FileFormat::TSV)
        .with_features(vec![0, 1, 2, 3])
        .with_label_column(4)
        .with_quantization(QuantizationMethod::StandardScore)
        .load()?;  // ← Unwrap Result<Dataset, DataError>
    
    let test_ds = DatasetBuilder::<i32>::new("data/iris_test.tsv")
        .format(FileFormat::TSV)
        .with_features(vec![0, 1, 2, 3])
        .with_label_column(4)
        .with_quantization(QuantizationMethod::StandardScore)
        .load()?;   // ← Unwrap Result<Dataset, DataError>

    println!("Train set: [Shift: {}] {} samples, {} features", train_ds.input_shift, train_ds.len(), train_ds.n_features());
    println!("Test set: [Shift: {}]  {} samples, {} features", test_ds.input_shift, test_ds.len(), test_ds.n_features());
    println!("Classes:   {}", train_ds.n_classes);
    
    for epoch in 0..epochs {
        let epoch_start = Instant::now();
        #[cfg(debug_assertions)]
        reset_overflow_stats();

        // Shuffle
        let indices = shuffled_indices(train_ds.len(), &mut rng);

        let mut epoch_loss: i64 = 0;
        let mut batches_processed = 0;

        let last_batch_start = batch_size * ((train_ds.len() / batch_size) - 1);
        // Minibatches
        for batch_start in (0..train_ds.len()).step_by(batch_size) {
            model.sync_weights(&mut rng);
            let batch_end = (batch_start + batch_size).min(train_ds.len());
            let batch_indices = &indices[batch_start..batch_end];
            let (batch_inputs, batch_targets) = train_ds.minibatch(batch_indices);
            let (preds, shift) = model.forward(&batch_inputs, 0, &mut rng);
            let (loss, grad_out) = MSE.forward(&preds, &batch_targets);

            model.zero_grads();
            epoch_loss += loss as i64;
            batches_processed += 1;

            model.backward(&grad_out, shift);
            model.step(&mut optim);

            if batches_processed == 1 && epoch % 100 == 0 {
                //println!("L3 Gradients: {:?}", l3.weights.grads); // Ensure these aren't all 0
            }
        }
        if epoch % 100 == 0 {
            #[cfg(debug_assertions)]
            {
                get_overflow_stats();
                reset_overflow_stats();
            }
            println!("Epoch {:>4}: loss = {}", epoch, epoch_loss / batches_processed as i64);
        }
    }
    #[cfg(debug_assertions)]
    get_overflow_stats();

    // ── Evaluation ────────────────────────────────────────────────────────────
    model.sync_weights(&mut rng);

    println!(
        "\n{:<6} {:>10} {:>10} {:>8}",
        "t", "target", "pred", "error"
    );
    let mut eval_loss: i64 = 0;
    for t in 0..test_ds.len() {
        let x_t = test_ds.get_input(t);
        let target = test_ds.get_target(t);
        let (pred, shift) = model.forward(&x_t, 0, &mut rng);
        let (loss, grad_out) = MSE.forward(&pred, &target);

        eval_loss += loss as i64;
        println!("{:<6} {:>10} {:>10} {:>8}", t, argmax(&target, Some(1))[0], argmax(&pred, Some(1))[0], loss);
    }
    println!("Eval  total  MSE : {}", eval_loss / test_ds.len() as i64);

    // Get a batch of test samples
    let test_indices: Vec<usize> = (0..test_ds.len()).collect();
    let (test_inputs, _test_targets) = test_ds.minibatch(&test_indices);
    
    // Forward pass
    let (predictions_tensor, shift) = model.forward(&test_inputs, 0, &mut rng);

    // Get predicted classes [batch_size]
    let predicted_classes = argmax(&predictions_tensor, Some(1));
    
    // Get ground truth labels [batch_size]
    let true_labels: Vec<u8> = test_indices
        .iter()
        .map(|&i| test_ds.labels[i])
        .collect();

    // Compute accuracy
    let accuracy = accuracy(&predicted_classes, &true_labels);

    println!("PREDICTED CLASSES: {:?}", predictions_tensor);
    println!("\nTest batch accuracy: {:.2}%", accuracy * 100.0);

    // Print predictions vs ground truth
    for (i, &idx) in test_indices.iter().enumerate() {
        println!("Sample {}: predicted class {}, true class {}",
            idx, predicted_classes[i], true_labels[i]);
    }


    Ok(())
}
