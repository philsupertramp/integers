use integers::data::{load_mnist, shuffled_indices};
use integers::nn::*;
use integers::debug::{reset_overflow_stats, get_overflow_stats};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║               MNIST INTEGER NEURAL NETWORK                 ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // ─── Load Data ─────────────────────────────────────────────────────────
    println!("Loading datasets...");
    let train_start = Instant::now();
    let train_ds = load_mnist("data/mnist", "train", Some(10000))?;
    let test_ds = load_mnist("data/mnist", "test", None)?;
    println!("✓ Loaded in {:.2}s", train_start.elapsed().as_secs_f32());
    println!("  Train: {} samples, {} features", train_ds.len(), train_ds.n_features());
    println!("  Test:  {} samples, {} features\n", test_ds.len(), test_ds.n_features());

    // ─── Build Model ──────────────────────────────────────────────────────
    let scale_shift = 4u32;
    let grad_shift = 6u32;
    let batch_size = 32usize;
    let epochs = 50i32;
    
    println!("Model Configuration:");
    println!("  scale_shift = {}", scale_shift);
    println!("  grad_shift = {}", grad_shift);
    println!("  batch_size = {}", batch_size);
    println!("  epochs = {}\n", epochs);

    let mut model = Sequential::new();
    model
        .add(Linear::new(784, 128, scale_shift))
        .add(ReLU::new())
        .add(Linear::new(128, 10, scale_shift));

    let optim = AdamConfig::new(2);
    let mut rng = XorShift64::new(42);
    
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
        reset_overflow_stats();

        // Shuffle
        let indices = shuffled_indices(train_ds.len(), &mut rng);

        let mut epoch_loss: i64 = 0;
        let mut batches_processed = 0;

        // Minibatches
        for batch_start in (0..train_ds.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(train_ds.len());
            let batch_indices = &indices[batch_start..batch_end];
            let (batch_inputs, batch_targets) = train_ds.minibatch(batch_indices);

            model.sync_weights(&mut rng);
            let preds = model.forward(&batch_inputs, &mut rng);
            let (loss, grad_out) = MSE.forward(&preds, &batch_targets);

            epoch_loss += loss as i64;
            batches_processed += 1;

            model.backward(&grad_out, Some(grad_shift));
            model.step(&optim);
        }

        // ─── Evaluation ────────────────────────────────────────────────────
        let mut correct: usize = 0;
        for t in 0..test_ds.len().min(1000) {
            let x = test_ds.get_input(t);
            let target_cls = test_ds.labels[t];
            model.sync_weights(&mut rng);
            let pred = model.forward(&x, &mut rng);
            let pred_cls = argmax(&pred, Some(1))[0] as u8;
            if pred_cls == target_cls {
                correct += 1;
            }
        }
        let accuracy = correct as f32 / 1000.0;

        let overflow_stats = get_overflow_stats();
        let elapsed = epoch_start.elapsed().as_secs_f32();

        println!("{:>6} {:>12} {:>11.1}% {:>10.2} {:>8}",
            epoch,
            epoch_loss / batches_processed as i64,
            accuracy * 100.0,
            elapsed,
            overflow_stats.downcast_clamps
        );

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
        let pred = model.forward(&x, &mut rng);
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
