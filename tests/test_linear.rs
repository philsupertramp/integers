use integers::{Tensor, XorShift64};
use integers::nn::{Linear, Module};
use integers::nn::optim::{AdamConfig, SGDConfig};

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
