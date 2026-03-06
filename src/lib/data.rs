use std::io::{self};

use crate::{Tensor, Scalar};

// ─── Error type ──────────────────────────────────────────────────────────────

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
            DataError::Io(e) => write!(f, "IO error: {}", e),
            DataError::InvalidMagic { expected, found } =>
                write!(f, "Invalid magic number: expected {:#010x}, found {:#010x}", expected, found),
            DataError::DimensionMismatch { images, labels } =>
                write!(f, "Image count ({}) does not match label count ({})", images, labels),
            DataError::ParseError(s) => write!(f, "Parse error: {}", s),
            DataError::EmptyDataset => write!(f, "Dataset is empty"),
        }
    }
}

impl From<io::Error> for DataError {
    fn from(e: io::Error) -> Self {
        DataError::Io(e)
    }
}

// ─── Shared result type ───────────────────────────────────────────────────────

pub type DataResult<T> = Result<T, DataError>;

// ─── Dataset struct ───────────────────────────────────────────────────────────

/// A split of input tensors and class labels.
///
/// `inputs`  shape: [n_samples, n_features]
/// `labels`  length: n_samples  (class index 0..n_classes)
/// `targets` shape: [n_samples, n_classes]  (one-hot i32, see module docs)
pub struct Dataset<S: Scalar> {
    pub inputs:   Tensor<S>,
    pub labels:   Vec<u8>,
    pub targets:  Tensor<S>,
    pub n_classes: usize,
    pub input_shift: u32,
}

impl<S: Scalar> Dataset<S> {
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    pub fn n_features(&self) -> usize {
        if self.inputs.shape.len() < 2 { 0 } else { self.inputs.shape[1] }
    }

    /// Return a single sample as a [1, n_features] tensor (no allocation on inputs slice).
    pub fn get_input(&self, idx: usize) -> Tensor<S> {
        let nf = self.n_features();
        let start = idx * nf;
        Tensor::<S>::from_vec(self.inputs.data[start..start + nf].to_vec(), vec![1, nf])
    }

    /// Return a single one-hot target as a [1, n_classes] tensor.
    pub fn get_target(&self, idx: usize) -> Tensor<S> {
        let nc = self.n_classes;
        let start = idx * nc;
        Tensor::from_vec(self.targets.data[start..start + nc].to_vec(), vec![1, nc])
    }

    /// Convenience: predicted class from a [1, n_classes] output tensor.
    pub fn argmax(output: &Tensor<S>) -> u8 {
        let (max_idx, _) = output.data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &S::default()));
        return max_idx as u8;
    }

    /// Build a random mini-batch of `batch_size` samples.
    ///
    /// Returns `(inputs [batch, nf], targets [batch, nc])`.
    pub fn minibatch(
        &self,
        indices: &[usize],
    ) -> (Tensor<S>, Tensor<S>) {
        let nf = self.n_features();
        let nc = self.n_classes;
        let b  = indices.len();

        let mut inp_data  = Vec::<S>::with_capacity(b * nf);
        let mut tgt_data  = Vec::<S>::with_capacity(b * nc);

        for &i in indices {
            let base_i = i * nf;
            inp_data.extend_from_slice(&self.inputs.data[base_i..base_i + nf]);
            let base_t = i * nc;
            tgt_data.extend_from_slice(&self.targets.data[base_t..base_t + nc]);
        }

        (
            Tensor::<S>::from_vec(inp_data, vec![b, nf]),
            Tensor::<S>::from_vec(tgt_data, vec![b, nc]),
        )
    }
}


// ─── Shuffle utility ─────────────────────────────────────────────────────────

/// Build a shuffled index array of length `n` using XorShift64.
///
/// A Fisher-Yates shuffle entirely in integer arithmetic — no float RNG needed.
/// Use this to get mini-batch indices:
///
/// ```no_run
/// use integers::data::shuffled_indices;
/// use integers::XorShift64;
/// let mut rng = XorShift64::new(42);
/// let idx = shuffled_indices(1000, &mut rng);
/// for batch in idx.chunks(32) {
///     // train on batch
/// }
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
    use crate::XorShift64;

    // ── Mini-batch + shuffle ─────────────────────────────────────────────────

    #[test]
    fn test_shuffled_indices_length_and_contents() {
        let mut rng = XorShift64::new(42);
        let idx = shuffled_indices(10, &mut rng);
        assert_eq!(idx.len(), 10);
        let mut sorted = idx.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_shuffled_indices_is_shuffled() {
        let mut rng = XorShift64::new(42);
        let idx = shuffled_indices(100, &mut rng);
        // With overwhelming probability a 100-element shuffle is not identity
        let identity: Vec<usize> = (0..100).collect();
        assert_ne!(idx, identity, "shuffle returned identity permutation");
    }

    #[test]
    fn test_argmax() {
        let t = Tensor::from_vec(vec![-10i32, 50, 20], vec![1, 3]);
        assert_eq!(Dataset::argmax(&t), 1);
    }
}
