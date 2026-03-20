//! Core dataset type and shuffle utility.
//!
//! `Dataset<S>` holds pre-quantised inputs, integer labels, and one-hot
//! targets as flat row-major tensors.  It is produced by the loaders in
//! [`crate::dataset_loaders`] and [`crate::mnist_loader`].
//!
//! To feed a sample into the neural network, use the bridge helpers in
//! [`crate`] (`sample_to_dyadic` / `target_to_dyadic`).

use std::io;

use crate::{Scalar, Tensor};

// ─── Error type ───────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum DataError {
    Io(io::Error),
    InvalidMagic { expected: u32, found: u32 },
    DimensionMismatch { images: usize, labels: usize },
    ParseError(String),
    EmptyDataset,
}

impl std::error::Error for DataError {}

impl std::fmt::Display for DataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataError::Io(e) =>
                write!(f, "IO error: {e}"),
            DataError::InvalidMagic { expected, found } =>
                write!(f, "Invalid magic number: expected {expected:#010x}, found {found:#010x}"),
            DataError::DimensionMismatch { images, labels } =>
                write!(f, "Image count ({images}) does not match label count ({labels})"),
            DataError::ParseError(s) =>
                write!(f, "Parse error: {s}"),
            DataError::EmptyDataset =>
                write!(f, "Dataset is empty"),
        }
    }
}

impl From<io::Error> for DataError {
    fn from(e: io::Error) -> Self { DataError::Io(e) }
}

/// Convenience alias.
pub type DataResult<T> = Result<T, DataError>;

// ─── Dataset ──────────────────────────────────────────────────────────────────

/// A labelled dataset split.
///
/// | Field        | Shape                    | Notes                          |
/// |--------------|--------------------------|--------------------------------|
/// | `inputs`     | `[n_samples, n_features]`| row-major                      |
/// | `labels`     | `[n_samples]`            | class index `0..n_classes`     |
/// | `targets`    | `[n_samples, n_classes]` | one-hot, hot bit = 127         |
/// | `input_shift`| scalar                   | dyadic scale exponent for inputs |
pub struct Dataset<S: Scalar> {
    pub inputs:      Tensor<S>,
    pub labels:      Vec<u8>,
    pub targets:     Tensor<S>,
    pub n_classes:   usize,
    /// Dyadic scale exponent `s` such that decoded value = mantissa · 2⁻ˢ.
    pub input_shift: i32,
}

impl<S: Scalar> Dataset<S> {
    pub fn len(&self) -> usize { self.labels.len() }
    pub fn is_empty(&self) -> bool { self.labels.is_empty() }

    pub fn n_features(&self) -> usize {
        self.inputs.shape.get(1).copied().unwrap_or(0)
    }

    /// Return a single input row as a `[1, n_features]` tensor.
    pub fn get_input(&self, idx: usize) -> Tensor<S> {
        let nf    = self.n_features();
        let start = idx * nf;
        Tensor::<S>::from_vec(
            self.inputs.data[start..start + nf].to_vec(),
            vec![1, nf],
        )
    }

    /// Return a single one-hot target row as a `[1, n_classes]` tensor.
    pub fn get_target(&self, idx: usize) -> Tensor<S> {
        let nc    = self.n_classes;
        let start = idx * nc;
        Tensor::<S>::from_vec(
            self.targets.data[start..start + nc].to_vec(),
            vec![1, nc],
        )
    }

    /// Predicted class from a `[1, n_classes]` output tensor (argmax).
    pub fn argmax(output: &Tensor<S>) -> u8 {
        let (max_idx, _) = output.data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or((0, &S::default()));
        max_idx as u8
    }

    /// Build a mini-batch from the given sample indices.
    ///
    /// Returns `(inputs [batch, n_features], targets [batch, n_classes])`.
    pub fn minibatch(&self, indices: &[usize]) -> (Tensor<S>, Tensor<S>) {
        let nf = self.n_features();
        let nc = self.n_classes;
        let b  = indices.len();

        let mut inp_data = Vec::<S>::with_capacity(b * nf);
        let mut tgt_data = Vec::<S>::with_capacity(b * nc);

        for &i in indices {
            inp_data.extend_from_slice(&self.inputs.data[i * nf..i * nf + nf]);
            tgt_data.extend_from_slice(&self.targets.data[i * nc..i * nc + nc]);
        }

        (
            Tensor::<S>::from_vec(inp_data, vec![b, nf]),
            Tensor::<S>::from_vec(tgt_data, vec![b, nc]),
        )
    }
}

// ─── Shuffle utility ──────────────────────────────────────────────────────────

/// Build a Fisher-Yates shuffled index array of length `n` using [`crate::XorShift64`].
///
/// Entirely in integer arithmetic — no floating-point RNG needed.
///
/// # Example
/// ```no_run
/// let mut rng = dyadic_nn::rng::XorShift64::new(42);
/// let idx = dyadic_nn::data::shuffled_indices(1000, &mut rng);
/// for batch in idx.chunks(32) { /* train on batch */ }
/// ```
pub fn shuffled_indices(n: usize, rng: &mut crate::XorShift64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.gen_range((i + 1) as u32) as usize;
        idx.swap(i, j);
    }
    idx
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Tensor, XorShift64};

    #[test]
    fn shuffled_indices_length_and_contents() {
        let mut rng = XorShift64::new(42);
        let idx = shuffled_indices(10, &mut rng);
        assert_eq!(idx.len(), 10);
        let mut sorted = idx.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn shuffled_indices_is_not_identity() {
        let mut rng = XorShift64::new(42);
        let idx = shuffled_indices(100, &mut rng);
        assert_ne!(idx, (0..100).collect::<Vec<_>>(),
            "shuffle returned identity permutation");
    }

    #[test]
    fn argmax_picks_max() {
        let t = Tensor::from_vec(vec![-10i32, 50, 20], vec![1, 3]);
        assert_eq!(Dataset::argmax(&t), 1);
    }
}
