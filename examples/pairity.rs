use integers::*;
use integers::nn::*;
use integers::nn::losses::*;
use integers::nn::rnn::RNNCell;
use integers::nn::optim::SGDConfig;
use integers::data::shuffled_indices;

// ── Dataset ───────────────────────────────────────────────────────────────────
//
// Binary parity: given a sequence of 0s and 1s, predict whether the number
// of 1s is even (label=0) or odd (label=1).
//
// This is the classic RNN stress test — it requires maintaining a running
// parity bit across the full sequence, so the network cannot cheat by looking
// at only recent tokens.

fn generate_parity(
    n: usize,
    seq_len: usize,
    rng: &mut XorShift64,
) -> (Vec<Vec<f32>>, Vec<u8>) {
    let mut seqs = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    for _ in 0..n {
        let seq: Vec<f32> = (0..seq_len).map(|_| rng.gen_range(2) as f32).collect();
        let ones: u32 = seq.iter().map(|&x| x as u32).sum();
        labels.push((ones % 2) as u8);
        seqs.push(seq);
    }
    (seqs, labels)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const SEQ_LEN: usize = 10;
    const HIDDEN: usize = 32;
    const N_TRAIN: usize = 2000;
    const N_TEST: usize = 500;
    const EPOCHS: i32 = 100;

    let mut rng = XorShift64::new(42);
    let mut sync_rng = XorShift64::new(42);

    // ── Generate Dataset ──────────────────────────────────────────────────────
    println!("╔═══════════════════════════════════════╗");
    println!("║        PARITY RNN EXAMPLE              ║");
    println!("╚═══════════════════════════════════════╝\n");
    println!("Task:    predict parity of a binary sequence");
    println!("Seq len: {}", SEQ_LEN);
    println!("Train:   {} samples", N_TRAIN);
    println!("Test:    {} samples\n", N_TEST);

    let (train_seqs, train_labels) = generate_parity(N_TRAIN, SEQ_LEN, &mut rng);
    let (test_seqs, test_labels) = generate_parity(N_TEST, SEQ_LEN, &mut rng);

    // ── Build Model ───────────────────────────────────────────────────────────
    //
    // RNNCell(1 -> HIDDEN) + Linear(HIDDEN -> 2)
    // One bit at a time, binary classification at the final step.
    let mut rnn_cell = RNNCell::<f32>::new(1, HIDDEN);
    let mut out_layer = Linear::<f32>::new(HIDDEN, 2);

    rnn_cell.init(&mut sync_rng);
    out_layer.init(&mut sync_rng);

    println!("Architecture:");
    rnn_cell.print_summary(&rnn_cell.describe(), 1);
    out_layer.print_summary(&out_layer.describe(), 1);
    println!();

    // SGD + momentum works well for RNNs: momentum smooths the noisy
    // single-sample gradients from BPTT.
    let optim = SGDConfig {
        lr_shift: 8,       // lr ≈ 1/128
        momentum_shift: Some(3),// Some(3), // momentum ≈ 1 - 1/8 = 0.875
    };

    println!("{:>6} {:>12} {:>10}", "Epoch", "Loss", "Accuracy");
    println!("{}", "─".repeat(32));

    // ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in 0..EPOCHS {
        let indices = shuffled_indices(N_TRAIN, &mut rng);
        let mut epoch_loss = 0.0f64;

        for &idx in &indices {
            // Reset hidden state between samples — each sequence is independent
            rnn_cell.reset_state();
            rnn_cell.sync_weights(&mut sync_rng);
            out_layer.sync_weights(&mut sync_rng);

            // ── Forward through sequence ──────────────────────────────────────
            let mut h_last = Tensor::<f32>::new(vec![1, HIDDEN]);
            let mut s_last = 0u32;

            for t in 0..SEQ_LEN {
                let x_t = Tensor::from_vec(vec![train_seqs[idx][t]], vec![1, 1]);
                let (h, s) = rnn_cell.forward(&x_t, 0, &mut rng);
                h_last = h;
                s_last = s;
            }

            // Classification head on the final hidden state only
            let (pred, s_out) = out_layer.forward(&h_last, s_last, &mut rng);

            // ── Loss ──────────────────────────────────────────────────────────
            let label = train_labels[idx];
            let target = if label == 0 {
                Tensor::from_vec(vec![1.0f32, 0.0], vec![1, 2])
            } else {
                Tensor::from_vec(vec![0.0f32, 1.0], vec![1, 2])
            };

            // For f32: pred and target are both at shift 0, no alignment needed
            let (loss, grad_out) = MSE.forward(&pred, &target);
            epoch_loss += loss as f64;

            // ── Backward (BPTT) ───────────────────────────────────────────────
            rnn_cell.zero_grads();
            out_layer.zero_grads();

            // Output layer backward: returns gradient w.r.t. h_last at s_g = s_out
            let (grad_h, s_g) = out_layer.backward(&grad_out, s_out);

            // Last timestep: gradient from output layer flows into RNN
            // RNNCell::backward internally stores d_h_next for the carry
            rnn_cell.backward(&grad_h, s_g);

            // Earlier timesteps: no direct output loss, only d_h_next carry
            // The zero tensor means "no external gradient at this step"
            let zero_grad = Tensor::<f32>::new(grad_h.shape.clone());
            for _ in 1..SEQ_LEN {
                rnn_cell.backward(&zero_grad, s_g);
            }

            rnn_cell.step(&optim);
            out_layer.step(&optim);
        }

        // ── Evaluation ────────────────────────────────────────────────────────
        if epoch % 10 == 0 || epoch == EPOCHS - 1 {
            rnn_cell.sync_weights(&mut sync_rng);
            out_layer.sync_weights(&mut sync_rng);

            let mut correct = 0usize;
            for i in 0..N_TEST {
                rnn_cell.reset_state();

                let mut h_last = Tensor::<f32>::new(vec![1, HIDDEN]);
                let mut s_last = 0u32;

                for t in 0..SEQ_LEN {
                    let x_t = Tensor::from_vec(vec![test_seqs[i][t]], vec![1, 1]);
                    let (h, s) = rnn_cell.forward(&x_t, 0, &mut rng);
                    h_last = h;
                    s_last = s;
                }

                let (pred, _) = out_layer.forward(&h_last, s_last, &mut rng);
                let pred_cls = argmax(&pred, Some(1))[0] as u8;
                if pred_cls == test_labels[i] {
                    correct += 1;
                }
            }

            let accuracy = correct as f32 / N_TEST as f32 * 100.0;
            println!(
                "{:>6} {:>12.4} {:>9.1}%",
                epoch,
                epoch_loss / N_TRAIN as f64,
                accuracy
            );

            // Early stopping — parity is solvable to ~100% with enough capacity
            if accuracy > 95.0 && epoch > 10 {
                println!("\n✓ Early stopping at epoch {} (>95% accuracy reached)", epoch);
                break;
            }
        }
    }

    println!("\n✓ Done!");
    Ok(())
}
