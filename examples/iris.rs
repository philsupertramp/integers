use integers::*;
use integers::nn::*;
use integers::nn::losses::*;
use integers::nn::activations::{ReLU};
use integers::dataset_loaders::*;
use integers::debug::*;
use integers::nn::optim::{SGDConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = XorShift64::new(42);
    let mut sync_rng = XorShift64::new(42);
    const EPOCHS: i32 = 200;
    const SCALE_SHIFT: u32 = 0;
    const GRAD_SHIFT: u32 = 0;
    let optim = SGDConfig::new().with_learn_rate(0.0125).with_momentum(0.8);  // lr_shift=2, momentum_shift=2

    let mut l1 = Linear::new(4, 8, SCALE_SHIFT);
    let mut l2 = Linear::new(8, 3, SCALE_SHIFT);
    
    // Build model for Iris: 4 input features → 3 output classes
    let mut model = Sequential::new();
    model
        .add(l1)
        .add(ReLU::new())
        .add(l2);

    model.init(&mut rng);

    // Load datasets (unwrap Results with ?)
    let train_ds = DatasetBuilder::new("data/iris_train.tsv")
        .format(FileFormat::TSV)
        .with_features(vec![0, 1, 2, 3])
        .with_label_column(4)
        .with_quantization(QuantizationMethod::StandardScore)
        .load()?;  // ← Unwrap Result<Dataset, DataError>
    
    let test_ds = DatasetBuilder::new("data/iris_test.tsv")
        .format(FileFormat::TSV)
        .with_features(vec![0, 1, 2, 3])
        .with_label_column(4)
        .with_quantization(QuantizationMethod::StandardScore)
        .load()?;   // ← Unwrap Result<Dataset, DataError>

    println!("Train set: {} samples, {} features", train_ds.len(), train_ds.n_features());
    println!("Test set:  {} samples, {} features", test_ds.len(), test_ds.n_features());
    println!("Classes:   {}", train_ds.n_classes);
    
    let mse = MSE;

    for epoch in 0..EPOCHS {
        model.sync_weights(&mut sync_rng);
        let mut epoch_loss: i64 = 0;

        for t in 0..(train_ds.len()) {
            model.zero_grads();
            let x_t = train_ds.get_input(t);
            let target = train_ds.get_target(t);

            let pred = model.forward(&x_t, SCALE_SHIFT, &mut rng);

            let (loss, grad_out) = mse.forward(&pred, &target);
            epoch_loss += loss as i64;

            if t % 100 == 0 {
                println!("T={}: {:?} => {:?}", t, x_t, pred);
                println!("{:?}", target);
                println!("Loss: {:?}; Grad: {:?}", loss, grad_out);
            }

            model.backward(&grad_out, Some(GRAD_SHIFT));
            model.step(&optim);
        }
        if epoch % 100 == 0 {
            #[cfg(debug_assertions)]
            {
                get_overflow_stats();
                reset_overflow_stats();
            }
            println!("Epoch {:>4}: loss = {}", epoch, epoch_loss);
        }
    }
    #[cfg(debug_assertions)]
    get_overflow_stats();

    // ── Evaluation ────────────────────────────────────────────────────────────
    model.sync_weights(&mut sync_rng);

    println!(
        "\n{:<6} {:>10} {:>10} {:>8}",
        "t", "target", "pred", "error"
    );
    let mut eval_loss: i64 = 0;
    for t in 0..test_ds.len() {
        let x_t = test_ds.get_input(t);
        let target = test_ds.get_target(t);
        let pred = model.forward(&x_t, SCALE_SHIFT, &mut rng);
        let pred_cls = argmax(&pred, Some(1));
        let error = pred_cls[0] as i32 - target.data[0];
        eval_loss += (error as i64) * (error as i64);
    }
    println!("Eval  total  MSE : {}", eval_loss);

    // Get a batch of test samples
    let test_indices: Vec<usize> = (0..test_ds.len()).collect();
    let (test_inputs, _test_targets) = test_ds.minibatch(&test_indices);
    
    // Forward pass
    let predictions_tensor = model.forward(&test_inputs, SCALE_SHIFT, &mut rng);
    
    // Get predicted classes [batch_size]
    let predicted_classes = argmax(&predictions_tensor, Some(1));
    
    // Get ground truth labels [batch_size]
    let true_labels: Vec<u8> = test_indices
        .iter()
        .map(|&i| test_ds.labels[i])
        .collect();
    
    // Compute accuracy
    let accuracy = accuracy(&predicted_classes, &true_labels);
    
    println!("\nTest batch accuracy: {:.2}%", accuracy * 100.0);
    
    // Print predictions vs ground truth
    for (i, &idx) in test_indices.iter().enumerate() {
        println!("Sample {}: predicted class {}, true class {}",
            idx, predicted_classes[i], true_labels[i]);
    }


    Ok(())
}
