use integers::{Sequential, Tensor, Linear, RNNCell, XorShift64, AdamConfig, SGDConfig, Module, RNN};


// RNNCell is just another module — drop it into Sequential for
// a simple sequence classifier
#[test]
fn test_rnn_in_sequential() {
    let mut model = Sequential::new();
    model.add(RNNCell::new(4, 8, 2))   // composite module, no special handling
         .add(Linear::new(8, 1, 2));

    let mut rng   = XorShift64::new(42);
    let mut optim = AdamConfig::new(2);

    let x = Tensor::new(vec![1, 4]);   // single step, batch=1, input_dim=4

    model.sync_weights(&mut rng);
    let out = model.forward(&x, &mut rng);
    assert_eq!(out.shape, vec![1, 1]);
}

// Full sequence with RNN struct (multi-step BPTT)
#[test]
fn test_rnn_sequence() {
    let mut rnn   = RNN::new(4, 8, 2, 5);
    let mut rng   = XorShift64::new(42);
    let mut optim = SGDConfig { lr_shift: 4, momentum_shift: None };

    rnn.cell.sync_weights(&mut rng);
    rnn.reset_state();

    let seq: Vec<Tensor<i8>> = (0..10).map(|_| Tensor::new(vec![2, 4])).collect();
    let outputs = rnn.forward_seq(&seq, &mut rng);
    assert_eq!(outputs.len(), 10);
    assert_eq!(outputs[0].shape, vec![2, 8]);

    let grads: Vec<Tensor<i16>> = outputs.iter()
        .map(|o| Tensor::new(o.shape.iter().map(|&d| d).collect()))
        .collect();

    let _ = rnn.backward_seq(&grads, Some(0));
    rnn.cell.step(&optim);
}


// Example test: predict the next value in a sin wave using an RNN.
//
// Task: given sin(t), predict sin(t+1)  — a one-step-ahead forecast.
//
// Data representation:
//   sin is in [-1.0, 1.0]. We scale by SCALE=100 to fit in i8 [-100, 100].
//   Inputs and targets are therefore i8 values in that range.
//
// Model:
//   RNNCell(input_dim=1, hidden_dim=16) -> Linear(16, 1)
//   The RNN reads one scaled sin sample per timestep.
//   The Linear head maps the hidden state to the predicted next sample.
//
// Training:
//   One full pass through the sequence = one epoch.
//   At each step t we:
//     1. feed x_t = sin(t) into the RNN
//     2. project hidden state -> predicted y
//     3. compare y to x_{t+1} (next sample)
//     4. backprop and step
//   We reset hidden state between epochs (not between steps within an epoch).

#[test]
fn test_rnn_sin_prediction() {
    use std::f64::consts::PI;

    // ── Hyperparameters ───────────────────────────────────────────────────────
    const SCALE:       f64   = 100.0; // float → i8 scaling factor
    const SEQ_LEN:     usize = 128;    // one full period sampled at 64 points
    const HIDDEN_DIM:  usize = 16;
    const SCALE_SHIFT: u32   = 2;     // weight quantization shift
    const GRAD_SHIFT:  u32   = 0;     // keep gradients full-resolution (small numbers)
    const EPOCHS:      usize = 500;
    const BPTT_STEPS:  usize = 8;     // truncate BPTT to last 8 steps

    let mut rng = XorShift64::new(42);

    // ── Generate one period of sin, scaled to i8 ─────────────────────────────
    // samples[t] = round(sin(2π * t / SEQ_LEN) * SCALE)
    let samples: Vec<i8> = (0..=SEQ_LEN)   // SEQ_LEN+1 so we have a target for the last input
        .map(|t| {
            let v = (2.0 * PI * t as f64 / SEQ_LEN as f64).sin() * SCALE;
            v.round().clamp(-128.0, 127.0) as i8
        })
        .collect();

    // ── Model: RNNCell + Linear projection head ───────────────────────────────
    let mut rnn  = RNNCell::new(1, HIDDEN_DIM, SCALE_SHIFT);
    let mut head = Linear::new(HIDDEN_DIM, 1, SCALE_SHIFT);

    // Random init — small values to stay well within i8 range
    rnn.init_weights(&mut rng);
    head.init_xavier(&mut rng);

    let mut optim = AdamConfig::new(4);

    let mut first_epoch_loss = 0i64;
    let mut last_epoch_loss  = 0i64;

    // ── Training loop ─────────────────────────────────────────────────────────
    for epoch in 0..EPOCHS {
        // Sync i32 master weights → i8 storage once per epoch
        rnn.sync_weights(&mut rng);
        head.sync_weights(&mut rng);

        // Reset hidden state at the start of each epoch
        rnn.reset_state();

        let mut epoch_loss: i64 = 0;

        // Collect (h_t, grad) pairs for BPTT — ring buffer of last BPTT_STEPS
        // For simplicity here we do online (step-by-step) updates:
        // forward one step → compute loss → backward → step.
        // This is "online BPTT-1" — equivalent to bptt_steps=1.
        // For proper truncated BPTT, accumulate BPTT_STEPS then unroll backward.
        for t in 0..(SEQ_LEN - 1) {
            // Input: x_t as [batch=1, input_dim=1]
            let x_t = Tensor::from_vec(vec![samples[t]], vec![1, 1]);

            // Target: x_{t+1}
            let target = samples[t + 1] as i16;

            // Forward through RNN cell → hidden state h_t  [1, HIDDEN_DIM]
            let h_t = rnn.forward(&x_t, &mut rng);

            // Forward through projection head → prediction  [1, 1]
            let pred = head.forward(&h_t, &mut rng);

            // MSE-flavoured loss and gradient (error = pred - target)
            let error = pred.data[0] as i16 - target;
            epoch_loss += (error as i64) * (error as i64);

            // Backward through head
            let grad_head_out = Tensor::from_vec(vec![error], vec![1, 1]);
            let grad_h        = head.backward(&grad_head_out, Some(GRAD_SHIFT));

            // Backward through RNN cell
            rnn.backward(&grad_h, Some(GRAD_SHIFT));

            // Step both modules
            head.step(&optim);
            rnn.step(&optim);
        }

        if epoch == 0          { first_epoch_loss = epoch_loss; }
        if epoch == EPOCHS - 1 { last_epoch_loss  = epoch_loss; }

        if epoch % 100 == 0 {
            println!("Epoch {:>4}: loss = {}", epoch, epoch_loss);
        }
    }

    // ── Evaluation ────────────────────────────────────────────────────────────
    rnn.sync_weights(&mut rng);
    head.sync_weights(&mut rng);
    rnn.reset_state();

    println!("\n{:<6} {:>10} {:>10} {:>8}", "t", "target", "pred", "error");
    let mut eval_loss: i64 = 0;
    for t in 0..(SEQ_LEN - 1) {
        let x_t   = Tensor::from_vec(vec![samples[t]], vec![1, 1]);
        let target = samples[t + 1] as i16;
        let h_t    = rnn.forward(&x_t, &mut rng);
        let pred   = head.forward(&h_t, &mut rng);
        let error  = pred.data[0] as i16 - target;
        eval_loss += (error as i64) * (error as i64);
        println!("{:<6} {:>10} {:>10} {:>8}", t, target, pred.data[0], error);
    }

    println!("\nFirst epoch loss : {}", first_epoch_loss);
    println!("Final epoch loss : {}", last_epoch_loss);
    println!("Eval  total  MSE : {}", eval_loss);

    // Loss should have decreased over training
    assert!(
        last_epoch_loss < first_epoch_loss,
        "Loss did not decrease: first={} last={}",
        first_epoch_loss, last_epoch_loss
    );

    // Predictions should be in a plausible range — sin scaled to 100 sits in
    // [-100, 100], so total squared error over 63 steps should be well under
    // the trivial baseline of predicting 0 every time (= sum of squares of targets).
    let trivial_baseline: i64 = samples[1..SEQ_LEN]
        .iter()
        .map(|&v| (v as i64) * (v as i64))
        .sum();

    assert!(
        eval_loss < trivial_baseline,
        "Model worse than predicting zero: eval_loss={} baseline={}",
        eval_loss, trivial_baseline
    );
}

// ── Bonus: cos variant ────────────────────────────────────────────────────────
// Identical structure — just swap sin for cos.
// Useful as a sanity check that the model isn't overfitting to sin's phase.

#[test]
fn test_rnn_cos_prediction() {
    use std::f64::consts::PI;

    const SCALE:       f64   = 100.0;
    const SEQ_LEN:     usize = 128;
    const HIDDEN_DIM:  usize = 16;
    const SCALE_SHIFT: u32   = 2;
    const GRAD_SHIFT:  u32   = 0;
    const EPOCHS:      usize = 500;

    let mut rng = XorShift64::new(123);

    let samples: Vec<i8> = (0..=SEQ_LEN)
        .map(|t| {
            let v = (2.0 * PI * t as f64 / SEQ_LEN as f64).cos() * SCALE;
            v.round().clamp(-128.0, 127.0) as i8
        })
        .collect();

    let mut rnn  = RNNCell::new(1, HIDDEN_DIM, SCALE_SHIFT);
    let mut head = Linear::new(HIDDEN_DIM, 1, SCALE_SHIFT);

    for w in rnn.w_ih.weights.master.data.iter_mut() { *w = (rng.gen_range(8) as i32) - 10; }
    for w in rnn.w_hh.weights.master.data.iter_mut() { *w = (rng.gen_range(20) as i32) - 10; }
    for w in head.weights.master.data.iter_mut()     { *w = (rng.gen_range(20) as i32) - 10; }

    let mut optim = AdamConfig::new(3);

    let mut first_epoch_loss = 0i64;
    let mut last_epoch_loss  = 0i64;

    for epoch in 0..EPOCHS {
        rnn.sync_weights(&mut rng);
        head.sync_weights(&mut rng);
        rnn.reset_state();

        let mut epoch_loss: i64 = 0;

        for t in 0..(SEQ_LEN - 1) {
            let x_t    = Tensor::from_vec(vec![samples[t]], vec![1, 1]);
            let target = samples[t + 1] as i16;
            let h_t    = rnn.forward(&x_t, &mut rng);
            let pred   = head.forward(&h_t, &mut rng);

            let error         = pred.data[0] as i16 - target;
            epoch_loss       += (error as i64) * (error as i64);

            let grad_head_out = Tensor::from_vec(vec![error], vec![1, 1]);
            let grad_h        = head.backward(&grad_head_out, Some(GRAD_SHIFT));
            rnn.backward(&grad_h, Some(GRAD_SHIFT));
            head.step(&optim);
            rnn.step(&optim);
        }

        if epoch == 0          { first_epoch_loss = epoch_loss; }
        if epoch == EPOCHS - 1 { last_epoch_loss  = epoch_loss; }

        if epoch % 100 == 0 {
            println!("Epoch {:>4}: loss = {}", epoch, epoch_loss);
        }
    }

    let trivial_baseline: i64 = samples[1..SEQ_LEN]
        .iter()
        .map(|&v| (v as i64) * (v as i64))
        .sum();

    println!("cos | first={} last={} baseline={}", first_epoch_loss, last_epoch_loss, trivial_baseline);

    assert!(last_epoch_loss < first_epoch_loss);
    assert!(last_epoch_loss < trivial_baseline);
}

