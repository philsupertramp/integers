pub fn none_quantize(values: &[f32]) -> (Vec<i32>, i32) {
    if values.is_empty() {
        return (Vec::new(), 0);
    }
    return (values
        .iter()
        .map(|&v| {
            v.round() as i32
        })
        .collect(), 0);
}

pub fn minmax_quantize(values: &[f32]) -> (Vec<i32>, i32) {
    if values.is_empty() {
        return (Vec::new(), 0);
    }
    let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(1e-6);
    return (values
        .iter()
        .map(|&v| {
            let norm = (v - min) / range; // [0, 1]
            let scaled = norm * 254.0 - 127.0; // [-127, 127]
            scaled.round().clamp(-127.0, 127.0) as i32
        })
        .collect(), 7);
}

pub fn standard_score_quantize(values: &[f32]) -> (Vec<i32>, i32) {
    if values.is_empty() {
        return (Vec::new(), 0);
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
    let std = variance.sqrt().max(1e-6);
    return (values
        .iter()
        .map(|&v| {
            let zscore = (v - mean) / std;
            let scaled = zscore * 32.0;
            scaled.round().clamp(-127.0, 127.0) as i32
        })
        .collect(), 5);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minmax_quantize_empty_vec() {
        let res = minmax_quantize(&Vec::new());

        assert_eq!(res, Vec::new());
    }

    #[test]
    fn test_minmax_quantize() {
        let mut res = minmax_quantize(&vec![0.; 4]);
        assert_eq!(res, vec![-127; 4]);
        res = minmax_quantize(&vec![127.; 4]);
        assert_eq!(res, vec![-127; 4]);
        res = minmax_quantize(&vec![1., 2., 3., 4.]);
        // [-127, 127] split into 4: x_0 + ((x_N - x_0) * i / (n-1)) starting with i = 0
        // x_N - x_0 = 254
        // i = 0 => -127 + (254 * 0 / 3) = -127
        // i = 1 => -127 + (254 * 1 / 3) = -42
        // i = 2 => -127 + (254 * 2 / 3) = 42
        // i = 3 => -127 + (254 * 3 / 3) = 127
        assert_eq!(res, vec![-127, -42, 42, 127]);

        // i = 0 => -127 + (254 * 0 / 4) = -127
        // i = 1 => -127 + (254 * 1 / 4) = -64
        // i = 2 => -127 + (254 * 2 / 4) = 0
        // i = 3 => -127 + (254 * 3 / 4) = 64
        // i = 4 => -127 + (254 * 4 / 4) = 127
        res = minmax_quantize(&vec![1., 2., 3., 4., 5.]);
        assert_eq!(res, vec![-127, -64, 0, 64, 127]);
    }

    #[test]
    fn test_standard_score_quantize_empty_values() {
        let res = standard_score_quantize(&Vec::new());

        assert_eq!(res, Vec::new());
    }

    #[test]
    fn test_standard_score_quantize() {
        let mut res = standard_score_quantize(&vec![0.; 4]);
        assert_eq!(res, vec![0; 4]);

        res = standard_score_quantize(&vec![127.; 4]);
        assert_eq!(res, vec![0; 4]);

        res = standard_score_quantize(&vec![1., 2., 3., 4.]);

        assert_eq!(res, vec![-1, 0, 0, 1]);

        res = standard_score_quantize(&vec![1., 2., 3., 4., 5.]);
        assert_eq!(res, vec![-1, -1, 0, 1, 1]);

        res = standard_score_quantize(&(-128..127).map(|x| x as f32).collect::<Vec<f32>>());
        let expected: Vec<i32> = vec![
            -2, -2, -2, -2, -2, -2, -2, -2, -2,
            -2, -2, -2, -2, -2, -2, -2, -2, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  1,  1,  1,  1,  1,  1,  1,
             1,  1,  1,  1,  1,  1,  1,  1,  1,
             1,  1,  1,  1,  1,  1,  1,  1,  1,
             1,  1,  1,  1,  1,  1,  1,  1,  1,
             1,  1,  1,  1,  1,  1,  1,  1,  1,
             1,  1,  1,  1,  1,  1,  1,  1,  1,
             1,  1,  1,  1,  1,  1,  1,  1,  1,
             1,  1,  1,  1,  1,  1,  1,  1,  1,
             1,  1,  1,  1,  2,  2,  2,  2,  2,
             2,  2,  2,  2,  2,  2,  2,  2,  2,
             2,  2,  2
        ];
        assert_eq!(res, expected);
    }


}
