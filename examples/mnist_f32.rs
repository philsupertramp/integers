use integers::*;
use integers::nn::*;
use integers::nn::losses::*;
use integers::nn::activations::{ReLU};
use integers::data::{shuffled_indices};
#[cfg(debug_assertions)]
use integers::debug::{reset_overflow_stats, get_overflow_stats};
use integers::nn::optim::{AdamConfig, SGDConfig};
use integers::dataset_loaders::{QuantizationMethod, DatasetBuilder, FileFormat};

use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║               MNIST FLOAT NEURAL NETWORK                  ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // ─── Load Data ─────────────────────────────────────────────────────────
    println!("Loading datasets...");
    let train_start = Instant::now();
    let train_ds = DatasetBuilder::<f32>::new("data/mnist_train.parquet")
        .format(FileFormat::Parquet)
        .with_features((0..784).collect())
        .with_label_column(784)
        .with_num_classes(10)
        .with_quantization(QuantizationMethod::StandardScore)
        .load()?;
    let test_ds = DatasetBuilder::<f32>::new("data/mnist_test.parquet")
        .format(FileFormat::Parquet)
        .with_features((0..784).collect())
        .with_label_column(784)
        .with_num_classes(10)
        .with_quantization(QuantizationMethod::StandardScore)
        .load()?;
    println!("✓ Loaded in {:.2}s", train_start.elapsed().as_secs_f32());
    println!("  Train: {} samples, {} features", train_ds.len(), train_ds.n_features());
    println!("  Test:  {} samples, {} features\n", test_ds.len(), test_ds.n_features());

    // ─── RECOMMENDED HYPERPARAMETERS ──────────────────────────────────────
    // 
    // CONSERVATIVE (most reliable):
    //   grad_shift = 8   (increased from 6, handles 3-layer gradient cascade)
    //   batch_size = 16  (reduced from 32, higher variance helps exploration)
    //   epochs = 100     (increased from 50, integer training is noisier)
    //   optimizer = AdamConfig::new(4)  (slower learning, more stable)
    //
    // AGGRESSIVE (faster, needs monitoring):
    //   grad_shift = 7
    //   batch_size = 32
    //   epochs = 150
    //   optimizer = AdamConfig::new(2)
    //
    // SGD+MOMENTUM (often best for integer nets):
    //   grad_shift = 7
    //   batch_size = 32
    //   epochs = 150
    //   optimizer = SGDConfig::new(4, Some(2))
    //
    // Start with CONSERVATIVE, monitor diagnostics below ↓
    
    let batch_size: usize = 32;    // ← REDUCED from 32
    let epochs = 150i32;         // ← INCREASED from 50

    println!("Model Configuration (RECOMMENDED):");
    println!("  batch_size = {}", batch_size);
    println!("  epochs = {}\n", epochs);
    let mut rng = XorShift64::new(42);

    let mut l1 = Linear::<f32>::new(784, 128);
    let mut l2 = Linear::<f32>::new(128, 128);  // ← second hidden layer
    let mut l3 = Linear::<f32>::new(128, 10);   // ← output layer
    let mut model = Sequential::<f32>::new();

    l1.init(&mut rng);
    l2.init(&mut rng);
    l3.init(&mut rng);

    println!("L1 {}: {} -> {}", l1.weights.quant_shift, l1.input_shift, l1.output_shift);
    println!("L2 {}: {} -> {}", l2.weights.quant_shift, l2.input_shift, l2.output_shift);
    println!("L3 {}: {} -> {}", l3.weights.quant_shift, l3.input_shift, l3.output_shift);


    model
        .add(l1)
        .add(ReLU::<f32>::new())
        .add(l2)
        .add(ReLU::<f32>::new())
        .add(l3);
    model.init_all(&mut rng);

    let mut optim = SGDConfig::new();
    optim.lr_shift = 6;

    // Print architecture
    println!("Architecture:");
    model.print_summary(&model.describe(), 0);
    println!();

    // ─── Training Loop ────────────────────────────────────────────────────
    println!("{:>6} {:>12} {:>12} {:>10} {:>8}",
        "Epoch", "Loss", "Accuracy", "Time(s)", "Clamps");
    println!("{}", "─".repeat(60));

    let training_start = Instant::now();

    for epoch in 0..epochs {
        let epoch_start = Instant::now();
        #[cfg(debug_assertions)]
        reset_overflow_stats();

        // Shuffle
        let indices = shuffled_indices(train_ds.len(), &mut rng);

        let mut epoch_loss: f64 = 0.0;
        let mut batches_processed = 0;

        // Minibatches
        for batch_start in (0..train_ds.len()).step_by(batch_size) {
            model.sync_weights(&mut rng);
            let batch_end = (batch_start + batch_size).min(train_ds.len());
            let batch_indices = &indices[batch_start..batch_end];
            let (batch_inputs, batch_targets) = train_ds.minibatch(batch_indices);

            let (preds, shift) = model.forward(&batch_inputs, train_ds.input_shift, &mut rng);
            let (loss, grad_out) = MSE.forward(&preds, &batch_targets);

            if batch_start == 0 {
                println!("with shift {}", shift);
                //println!("Loss: {} for GRAD {:?}", loss, grad_out);
            }
            epoch_loss += loss as f64;
            batches_processed += 1;

            model.zero_grads();
            model.backward(&grad_out, shift);
            model.step(&mut optim);
        }

        // ─── Evaluation ────────────────────────────────────────────────────
        model.sync_weights(&mut rng);
        let mut correct: usize = 0;
        for t in 0..test_ds.len().min(1000) {
            let x = test_ds.get_input(t);
            let target_cls = test_ds.labels[t];
            let (pred, shift) = model.forward(&x, test_ds.input_shift, &mut rng);
            let pred_cls = argmax(&pred, Some(1))[0] as u8;
            if pred_cls == target_cls {
                correct += 1;
            }
        }
        let accuracy = correct as f32 / 1000.0;

        #[cfg(debug_assertions)]
        {
            let overflow_stats = get_overflow_stats();
            let elapsed = epoch_start.elapsed().as_secs_f32();

            println!("{:>6} {:>12} {:>11.1}% {:>10.2} {:>8}",
                epoch,
                epoch_loss / (batches_processed as f64),
                accuracy * 100.0,
                elapsed,
                overflow_stats.downcast_clamps
            );
        }
        #[cfg(not(debug_assertions))]
        {
            let elapsed = epoch_start.elapsed().as_secs_f32();

            println!("{:>6} {:>12} {:>11.1}% {:>10.2} {:>8}",
                epoch,
                epoch_loss / (batches_processed as f64),
                accuracy * 100.0,
                elapsed,
                -1
            );
        }

        // Early stopping
        if epoch > 5 && accuracy > 0.95 {
            println!("\n✓ Early stopping at epoch {} (95% accuracy reached)", epoch);
            break;
        }
    }

    println!("{}", "─".repeat(60));
    let total_time = training_start.elapsed().as_secs_f32();
    println!("✓ Training completed in {:.2}s\n", total_time);

    // ─── Final Test Evaluation ────────────────────────────────────────────
    println!("Final Test Set Evaluation:");
    println!("{}", "─".repeat(60));

    let mut correct: usize = 0;
    let mut confusion = vec![vec![0usize; 10]; 10];  // [true][pred]

    for t in 0..test_ds.len() {
        let x = test_ds.get_input(t);
        let target_cls = test_ds.labels[t] as usize;
        model.sync_weights(&mut rng);
        let (pred, shift) = model.forward(&x, test_ds.input_shift, &mut rng);
        let pred_cls = argmax(&pred, Some(1))[0] as usize;

        if pred_cls == target_cls {
            correct += 1;
        }
        confusion[target_cls][pred_cls] += 1;
    }

    let final_accuracy = correct as f32 / test_ds.len() as f32;
    println!("Accuracy: {:.2}% ({}/{})", 
        final_accuracy * 100.0, correct, test_ds.len());
    println!();

    // Print per-class accuracy
    println!("Per-class accuracy:");
    for digit in 0..10 {
        let total = confusion[digit].iter().sum::<usize>();
        let correct = confusion[digit][digit];
        let acc = if total > 0 {
            correct as f32 / total as f32 * 100.0
        } else {
            0.0
        };
        println!("  {}: {:.1}% ({}/{})", digit, acc, correct, total);
    }

    println!("\n✓ Done!");
    Ok(())
}

