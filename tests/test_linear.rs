use integers::{Tensor, XorShift64};
use integers::nn::{Linear, Module, Sequential};
use integers::nn::optim::{AdamConfig, SGDConfig};
use integers::nn::activations::{ReLU};

#[test]
fn test_train_linear_regression_sgd() {
    // We want to learn the function: y = 3x + 1
    // 1 Input, 1 Output. Scale shift = 0.
    let mut layer = Linear::new(1, 1, 0);

    // Initialize Master Weight poorly (start at 10)
    layer.weights.master.data[0] = 10;
    let mut rng = XorShift64::new(42);

    // Dataset
    let x = Tensor::from_vec(vec![1, 2, 3, 4], vec![4, 1]); // Inputs
    let y_target = vec![4, 7, 10, 13]; // Targets

    // Hyperparameters
    let epochs = 8;
    let lr_shift = 0.125; // Shift right by 4 (approx learning rate of 1/16 = 0.0625)
    let grad_shift = 2; // Don't shrink the gradients here, numbers are small

    let mut optim = SGDConfig::new().with_learn_rate(lr_shift);
    println!("--- Starting Integer Training ---");
    for epoch in 0..epochs {
        // 1. Sync weights from i32 -> i8
        layer.sync_weights(&mut rng);

        // 2. Forward Pass
        let preds = layer.forward(&x, 0, &mut rng);

        // 3. Compute Loss & Gradients (Error = Pred - Target)
        let mut grad_out = Tensor::<i16>::new(vec![4, 1]);
        let mut loss = 0;

        for i in 0..4 {
            let error = preds.data[i] as i16 - y_target[i] as i16;
            grad_out.data[i] = error; // The "Gradient" is just the error
            loss += (error as i32) * (error as i32); // MSE
        }

        // 4. Backward Pass
        let _d_x = layer.backward(&grad_out, Some(grad_shift));

        // 5. Optimizer Step: w = w - (dW >> lr_shift)
        layer.step(&mut optim);

        println!(
            "Epoch {:02}: Loss = {:04}, Grad = {:?}, Master Weight = {}, Storage Weight = {}",
            epoch, loss, grad_out, layer.weights.master.data[0], layer.weights.storage.data[0]
        );

        if loss == 0 {
            println!("Converged early! Epoch {}", epoch);
            break;
        }
    }

    layer.sync_weights(&mut rng);
    assert_eq!(
        layer.weights.storage.data[0], 4,
        "Model failed to learn y=3x + 1"
    );
}

#[test]
fn test_train_linear_regression_no_bias_adam() {
    // We want to learn the function: y = 3x
    // 1 Input, 1 Output. Scale shift = 0.
    let mut layer = Linear::new(1, 1, 0);

    // Initialize Master Weight poorly (start at 10)
    let mut rng = XorShift64::new(42);
    layer.weights.master.data[0] = 10;

    // Dataset
    let x = Tensor::from_vec(vec![1, 2, 3, 4], vec![4, 1]); // Inputs
    let y_target = vec![3, 6, 9, 12]; // Targets

    // Hyperparameters
    let epochs = 300;
    let lr_shift = 0; // Shift right by 4 (approx learning rate of 1/16 = 0.0625)
    let grad_shift = 0; // Don't shrink the gradients here, numbers are small

    let optim = AdamConfig {
        lr_shift: lr_shift,
        b1_shift: 3,
        b2_shift: 4,
        eps: 2,
    };
    println!("--- Starting Integer Training ---");
    println!(
        "Epoch -1: Weight = {}, Bias = {}",
        layer.weights.master.data[0], layer.bias.master.data[0]
    );
    for epoch in 0..epochs {
        // 1. Sync weights from i32 -> i8
        layer.sync_weights(&mut rng);

        // 2. Forward Pass
        let preds = layer.forward(&x, 0, &mut rng);

        // 3. Compute Loss & Gradients (Error = Pred - Target)
        let mut grad_out = Tensor::<i16>::new(vec![4, 1]);
        let mut loss = 0;

        for i in 0..4 {
            let error = preds.data[i] as i16 - y_target[i] as i16;
            grad_out.data[i] = error; // The "Gradient" is just the error
            loss += (error as i32) * (error as i32); // MSE
        }

        // 4. Backward Pass
        let _d_x = layer.backward(&grad_out, Some(grad_shift));

        // 5. Optimizer Step: w = w - (dW >> lr_shift)
        layer.step(&optim);

        println!(
            "Epoch {:02}: Loss = {:04}, Grad = {:?}, Weight = {}, Bias = {}",
            epoch, loss, grad_out, layer.weights.master.data[0], layer.bias.master.data[0]
        );

        if loss == 0 {
            println!("Converged early! Epoch {}", epoch);
            break;
        }
    }

    layer.sync_weights(&mut rng);
    let inp = Tensor::from_vec(vec![1], vec![1, 1]);
    assert_eq!(
        layer.forward(&inp, 0, &mut rng).data[0] as i16,
        3,
        "Model failed to learn y=3x"
    );
}

#[test]
fn test_train_linear_regression_with_bias_adam() {
    // We want to learn the function: y = 3x + 1
    // 1 Input, 1 Output. Scale shift = 0.
    let mut layer = Linear::new(1, 1, 0);

    // Initialize Master Weight poorly (start at 10)
    let mut rng = XorShift64::new(42);
    layer.weights.master.data[0] = 11;

    // Dataset
    let x = Tensor::from_vec(vec![1, 2, 3, 4], vec![4, 1]); // Inputs
    let y_target = vec![4, 7, 10, 13]; // Targets

    // Hyperparameters
    let epochs = 300;
    let lr_shift = 0; // Shift right by 4 (approx learning rate of 1/16 = 0.0625)
    let grad_shift = 0; // Don't shrink the gradients here, numbers are small

    let optim = AdamConfig {
        lr_shift: lr_shift,
        b1_shift: 3,
        b2_shift: 4,
        eps: 3,
    };
    println!("--- Starting Integer Training ---");
    for epoch in 0..epochs {
        // 1. Sync weights from i32 -> i8
        layer.sync_weights(&mut rng);

        // 2. Forward Pass
        let preds = layer.forward(&x, 0, &mut rng);

        // 3. Compute Loss & Gradients (Error = Pred - Target)
        let mut grad_out = Tensor::<i16>::new(vec![4, 1]);
        let mut loss = 0;

        for i in 0..4 {
            let error = preds.data[i] as i16 - y_target[i] as i16;
            grad_out.data[i] = error; // The "Gradient" is just the error
            loss += (error as i32) * (error as i32); // MSE
        }

        println!(
            "Epoch {:02}: Loss = {:04}, Weight = {}, Bias = {}",
            epoch, loss, layer.weights.master.data[0], layer.bias.master.data[0]
        );

        if loss == 0 {
            println!("Converged early! Epoch {}", epoch);
            break;
        }

        // 4. Backward Pass
        let _d_x = layer.backward(&grad_out, Some(grad_shift));

        // 5. Optimizer Step: w = w - (dW >> lr_shift)
        layer.step(&optim);
    }

    layer.sync_weights(&mut rng);
    let inp = Tensor::from_vec(vec![1], vec![1, 1]);
    assert_eq!(
        layer.forward(&inp, 0, &mut rng).data[0] as i16,
        4,
        "Model failed to learn y=3x + 1"
    );
}

#[test]
fn test_train_linear_xor_sgd_momentum() {
    let mut rng = XorShift64::new(777);

    let lr_shift: u32 = 2;
    let grad_shift: u32 = 3;
    let mut l1 = Linear::new(2, 8, lr_shift);
    let mut l2 = Linear::new(8, 1, lr_shift);

    for w in l1.weights.master.data.iter_mut() {
        *w = rng.gen_range(10) as i32 - 5;
    }

    for w in l2.weights.master.data.iter_mut() {
        *w = rng.gen_range(10) as i32 - 5;
    }
    let mut model = Sequential::new();
    model
        .add(l1)
        .add(ReLU::new())
        .add(l2);


    let mut optim = SGDConfig::new().with_learn_rate(0.9);//.with_momentum(0.8);

    let x = Tensor::from_vec(vec![0, 0, 0, 1, 1, 0, 1, 1], vec![4, 2]);

    let y_target = vec![0, 20, 20, 0];
    let epochs = 800;

    println!("--- Starting Integer XOR Training with SGD + Momentum ---");

    for epoch in 0..epochs {
        model.sync_weights(&mut rng);
        let preds = model.forward(&x, 0, &mut rng);

        let mut grad_out = Tensor::<i16>::new(vec![4, 1]);
        let mut loss = 0;

        for i in 0..4 {
            let error = preds.data[i] as i16 - y_target[i] as i16;
            grad_out.data[i] = error;
            loss += (error as i32) * (error as i32);
        }

        if epoch % 50 == 0 {
            println!(
                "Epoch {:03}: Loss = {:05}, Grad = {:?}, Preds: [{}, {}, {}, {}]",
                epoch, loss, grad_out, preds.data[0], preds.data[1], preds.data[2], preds.data[3]
            );
        }

        if loss == 0 {
            println!("Converged early at epoch {}!", epoch);
            break;
        }
        model.backward(&grad_out, Some(grad_shift));
        model.step(&mut optim);
    }

    model.sync_weights(&mut rng);
    let final_preds = model.forward(&x, 0, &mut rng);

    assert!(final_preds.data[0] < 10, "{}", final_preds.data[0]);
    assert!(final_preds.data[3] < 10, "{}", final_preds.data[3]);
    assert!(final_preds.data[1] > 10, "{}", final_preds.data[1]);
    assert!(final_preds.data[2] > 10, "{}", final_preds.data[2]);
}

#[test]
fn test_train_linear_xor_adam() {
    let mut rng = XorShift64::new(777);

    let lr_shift: u32 = 0;
    let grad_shift: u32 = 0;
    let mut l1 = Linear::new(2, 8, lr_shift);
    let mut l2 = Linear::new(8, 1, lr_shift);

    for w in l1.weights.master.data.iter_mut() {
        *w = rng.gen_range(15) as i32 - 5;
    }

    for w in l2.weights.master.data.iter_mut() {
        *w = rng.gen_range(15) as i32 - 5;
    }

    let mut model = Sequential::new();
    model
        .add(l1)
        .add(ReLU::new())
        .add(l2);

    // NEW: Instantiate our Integer Adam!
    // We set the learning rate multiplier to 2.
    let mut optim = AdamConfig::new().with_learn_rate(1.0);

    let mut x = Tensor::from_vec(vec![0, 0, 0, 1, 1, 0, 1, 1], vec![4, 2]);

    let y_target = vec![0, 20, 20, 0];
    let epochs = 100;

    println!("--- Starting Integer XOR Training with ADAM ---");

    for epoch in 0..epochs {
        model.sync_weights(&mut rng);
        let preds = model.forward(&x, 0, &mut rng);

        let mut grad_out = Tensor::<i16>::new(vec![4, 1]);
        let mut loss = 0;

        for i in 0..4 {
            let error = preds.data[i] as i16 - y_target[i] as i16;
            grad_out.data[i] = error;
            loss += (error as i32) * (error as i32);
        }

        if epoch % 50 == 0 {
            println!(
                "Epoch {:03}: Loss = {:05}, Preds: [{}, {}, {}, {}]",
                epoch, loss, preds.data[0], preds.data[1], preds.data[2], preds.data[3]
            );
        }

        if loss == 0 {
            println!("Converged early at epoch {}!", epoch);
            break;
        }

        model.backward(&grad_out, Some(grad_shift));

        // NEW: Pass the optimizer to the model step
        model.step(&mut optim);
    }

    model.sync_weights(&mut rng);
    let final_preds = model.forward(&x, 0, &mut rng);
    let p00 = final_preds.data[0];
    let p01 = final_preds.data[1];
    let p10 = final_preds.data[2];
    let p11 = final_preds.data[3];

    println!(
        "Final XOR Evaluation: 0,0->{} | 0,1->{} | 1,0->{} | 1,1->{}",
        p00, p01, p10, p11
    );

    assert!(p00 < 10, "0,0 failed: expected low, got {}", p00);
    assert!(p11 < 10, "1,1 failed: expected low, got {}", p11);
    assert!(p01 > 10, "0,1 failed: expected high, got {}", p01);
    assert!(p10 > 10, "1,0 failed: expected high, got {}", p10);
}

