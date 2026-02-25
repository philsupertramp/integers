/// Data loading utilities for real-world datasets.
///
/// Supports:
///   - MNIST (IDX binary format, from http://yann.lecun.com/exdb/mnist/)
///   - Iris / "Petals" dataset (CSV, from UCI or sklearn-compatible exports)
///
/// All loaders return `Tensor<i8>` inputs and either `Tensor<i8>` for
/// regression targets or `Vec<u8>` label indices for classification.
///
/// # Quantization convention
///
/// Pixel values [0, 255] → i8 by `(v as i32 - 128) as i8`
///   so 0 → -128, 128 → 0, 255 → 127.
///
/// Floating-point features are min-max scaled to [-127, 127].
///
/// One-hot targets use +96 for the true class and 0 elsewhere,
/// giving a comfortable margin without saturating tanh.

use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};
use std::path::Path;

use crate::nn::Tensor;

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
/// `targets` shape: [n_samples, n_classes]  (one-hot i8, see module docs)
pub struct Dataset {
    pub inputs:   Tensor<i8>,
    pub labels:   Vec<u8>,
    pub targets:  Tensor<i8>,
    pub n_classes: usize,
}

impl Dataset {
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
    pub fn get_input(&self, idx: usize) -> Tensor<i8> {
        let nf = self.n_features();
        let start = idx * nf;
        Tensor::from_vec(self.inputs.data[start..start + nf].to_vec(), vec![1, nf])
    }

    /// Return a single one-hot target as a [1, n_classes] tensor.
    pub fn get_target(&self, idx: usize) -> Tensor<i8> {
        let nc = self.n_classes;
        let start = idx * nc;
        Tensor::from_vec(self.targets.data[start..start + nc].to_vec(), vec![1, nc])
    }

    /// Convenience: predicted class from a [1, n_classes] output tensor.
    pub fn argmax(output: &Tensor<i8>) -> u8 {
        output.data
            .iter()
            .enumerate()
            .max_by_key(|&(_, &v)| v)
            .map(|(i, _)| i as u8)
            .unwrap_or(0)
    }

    /// Build a random mini-batch of `batch_size` samples.
    ///
    /// Returns `(inputs [batch, nf], targets [batch, nc])`.
    pub fn minibatch(
        &self,
        indices: &[usize],
    ) -> (Tensor<i8>, Tensor<i8>) {
        let nf = self.n_features();
        let nc = self.n_classes;
        let b  = indices.len();

        let mut inp_data  = Vec::with_capacity(b * nf);
        let mut tgt_data  = Vec::with_capacity(b * nc);

        for &i in indices {
            let base_i = i * nf;
            inp_data.extend_from_slice(&self.inputs.data[base_i..base_i + nf]);
            let base_t = i * nc;
            tgt_data.extend_from_slice(&self.targets.data[base_t..base_t + nc]);
        }

        (
            Tensor::from_vec(inp_data, vec![b, nf]),
            Tensor::from_vec(tgt_data, vec![b, nc]),
        )
    }
}

// ─── MNIST ────────────────────────────────────────────────────────────────────

/// Reads a big-endian u32 from a reader.
fn read_u32_be(r: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

/// Load MNIST images from a raw IDX3-ubyte file.
///
/// Returns a flat `Vec<u8>` of pixel values, plus (n_images, rows, cols).
fn load_mnist_images_raw(path: &Path) -> DataResult<(Vec<u8>, usize, usize, usize)> {
    let mut f = BufReader::new(File::open(path)?);

    let magic = read_u32_be(&mut f)?;
    if magic != 0x0000_0803 {
        return Err(DataError::InvalidMagic { expected: 0x0000_0803, found: magic });
    }
    let n      = read_u32_be(&mut f)? as usize;
    let rows   = read_u32_be(&mut f)? as usize;
    let cols   = read_u32_be(&mut f)? as usize;

    let mut pixels = vec![0u8; n * rows * cols];
    f.read_exact(&mut pixels)?;

    Ok((pixels, n, rows, cols))
}

/// Load MNIST labels from a raw IDX1-ubyte file.
fn load_mnist_labels_raw(path: &Path) -> DataResult<Vec<u8>> {
    let mut f = BufReader::new(File::open(path)?);

    let magic = read_u32_be(&mut f)?;
    if magic != 0x0000_0801 {
        return Err(DataError::InvalidMagic { expected: 0x0000_0801, found: magic });
    }
    let n = read_u32_be(&mut f)? as usize;

    let mut labels = vec![0u8; n];
    f.read_exact(&mut labels)?;

    Ok(labels)
}

/// Load an MNIST split (train or test) from a directory.
///
/// The directory must contain the four standard MNIST files:
/// ```text
/// train-images-idx3-ubyte
/// train-labels-idx1-ubyte
/// t10k-images-idx3-ubyte
/// t10k-labels-idx1-ubyte
/// ```
/// (gzip decompression is NOT performed — decompress first with `gunzip`.)
///
/// # Arguments
///
/// * `dir`   - directory containing the IDX files
/// * `split` - `"train"` or `"test"`
/// * `max_samples` - optional cap; useful for quick smoke-tests
///
/// # Returns
///
/// A [`Dataset`] where inputs are flattened 28×28 images quantized to i8.
///
/// # Example
///
/// ```no_run
/// use integers::data::load_mnist;
/// let train = load_mnist("data/mnist", "train", Some(1000)).unwrap();
/// println!("{} samples, {} features", train.len(), train.n_features());
/// ```
pub fn load_mnist(
    dir: impl AsRef<Path>,
    split: &str,
    max_samples: Option<usize>,
) -> DataResult<Dataset> {
    let dir = dir.as_ref();

    let (img_file, lbl_file) = match split {
        "train" => (
            dir.join("train-images-idx3-ubyte"),
            dir.join("train-labels-idx1-ubyte"),
        ),
        "test" => (
            dir.join("t10k-images-idx3-ubyte"),
            dir.join("t10k-labels-idx1-ubyte"),
        ),
        other => return Err(DataError::ParseError(
            format!("Unknown split '{}'. Expected 'train' or 'test'.", other)
        )),
    };

    let (pixels, n_images, rows, cols) = load_mnist_images_raw(&img_file)?;
    let raw_labels = load_mnist_labels_raw(&lbl_file)?;

    if n_images != raw_labels.len() {
        return Err(DataError::DimensionMismatch {
            images: n_images,
            labels: raw_labels.len(),
        });
    }
    if n_images == 0 {
        return Err(DataError::EmptyDataset);
    }

    let n = max_samples.map(|m| m.min(n_images)).unwrap_or(n_images);
    let n_features = rows * cols;    // 784 for standard MNIST
    let n_classes  = 10usize;

    // Quantize pixels: [0,255] → i8 via centre-shift.
    // 0 → -128,  127/128 → ~0,  255 → 127
    let mut inp_data = Vec::with_capacity(n * n_features);
    for px in &pixels[..n * n_features] {
        inp_data.push((*px as i32 - 128) as i8);
    }

    let labels: Vec<u8> = raw_labels[..n].to_vec();

    // One-hot targets: true class → +96, others → 0
    // (leaves room for the head to output the right direction without saturation)
    let mut tgt_data = vec![0i8; n * n_classes];
    for (i, &lbl) in labels.iter().enumerate() {
        tgt_data[i * n_classes + lbl as usize] = 96;
    }

    Ok(Dataset {
        inputs:    Tensor::from_vec(inp_data, vec![n, n_features]),
        labels,
        targets:   Tensor::from_vec(tgt_data, vec![n, n_classes]),
        n_classes,
    })
}

// ─── Iris / Petals ────────────────────────────────────────────────────────────

/// Map a species string to a label index.
fn iris_class_index(s: &str) -> Option<u8> {
    // Accepts both full names and sklearn-style integers.
    // Common CSV spellings from UCI and Kaggle exports.
    match s.trim().trim_matches('"').to_ascii_lowercase().as_str() {
        "iris-setosa"     | "setosa"     | "0" => Some(0),
        "iris-versicolor" | "versicolor" | "1" => Some(1),
        "iris-virginica"  | "virginica"  | "2" => Some(2),
        _ => None,
    }
}

/// Min-max scale a column of f32 values to [-127, 127] → i8.
fn minmax_quantize(values: &[f32]) -> Vec<i8> {
    let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(1e-6);
    values.iter().map(|&v| {
        let norm = (v - min) / range;          // [0, 1]
        let scaled = norm * 254.0 - 127.0;    // [-127, 127]
        scaled.round().clamp(-127.0, 127.0) as i8
    }).collect()
}

/// Load the Iris dataset from a CSV file.
///
/// The CSV may contain an optional header row. The expected column order is:
/// ```text
/// sepal_length, sepal_width, petal_length, petal_width, species
/// ```
/// Species may be a string name (`Iris-setosa`) or integer (0/1/2).
/// An id column before the features is detected and skipped automatically
/// if the first field parses as a pure integer that doesn't fit the species
/// vocabulary.
///
/// Downloads / pre-bundled copies:
///   - UCI: <https://archive.ics.uci.edu/ml/datasets/iris>
///   - Kaggle "IRIS.csv": has header `sepal_length_cm,...,species`
///   - sklearn `load_iris()` CSV export
///
/// # Example
///
/// ```no_run
/// use integers::data::load_iris;
/// let ds = load_iris("data/iris.csv").unwrap();
/// println!("{} samples, {} features, {} classes", ds.len(), ds.n_features(), ds.n_classes);
/// ```
pub fn load_iris(path: impl AsRef<Path>) -> DataResult<Dataset> {
    let f = BufReader::new(File::open(path.as_ref())?);
    let n_classes = 3usize;
    let n_features = 4usize;

    let mut raw_features: Vec<Vec<f32>> = Vec::new(); // [sample][feature]
    let mut labels: Vec<u8> = Vec::new();

    for line_result in f.lines() {
        let line = line_result?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split(',').collect();

        // Skip header rows: only if ALL fields fail to parse as f32 and
        // aren't known species names (i.e., actual feature name headers).
        let is_header = fields.iter().all(|s| {
            let s = s.trim().trim_matches('"');
            s.parse::<f32>().is_err() && iris_class_index(s).is_none()
        });
        if is_header {
            continue;
        }

        // Auto-detect optional leading id column:
        // if we have 6 fields and field[0] is an integer, skip it.
        let start = if fields.len() == 6 && fields[0].trim().parse::<u64>().is_ok() {
            1
        } else {
            0
        };

        if fields.len() - start < n_features + 1 {
            return Err(DataError::ParseError(
                format!("Expected {} feature columns + 1 label column, got {} fields in: '{}'",
                    n_features, fields.len() - start, line)
            ));
        }

        let feat_end = start + n_features;
        let mut row = Vec::with_capacity(n_features);
        for &f in &fields[start..feat_end] {
            let v = f.trim().parse::<f32>().map_err(|_| {
                DataError::ParseError(format!("Cannot parse '{}' as f32 in line: {}", f, line))
            })?;
            row.push(v);
        }

        let species_field = fields[feat_end].trim();
        let label = iris_class_index(species_field).ok_or_else(|| {
            DataError::ParseError(format!(
                "Unknown species '{}' in line: {}", species_field, line
            ))
        })?;

        raw_features.push(row);
        labels.push(label);
    }

    if labels.is_empty() {
        return Err(DataError::EmptyDataset);
    }

    let n = labels.len();

    // Quantize per-column so each feature spans the full [-127, 127] range.
    let mut columns: Vec<Vec<i8>> = Vec::with_capacity(n_features);
    for feat_idx in 0..n_features {
        let col: Vec<f32> = raw_features.iter().map(|r| r[feat_idx]).collect();
        columns.push(minmax_quantize(&col));
    }

    // Re-interleave into row-major layout [n, n_features]
    let mut inp_data = vec![0i8; n * n_features];
    for (sample_idx, row) in inp_data.chunks_exact_mut(n_features).enumerate() {
        for (feat_idx, cell) in row.iter_mut().enumerate() {
            *cell = columns[feat_idx][sample_idx];
        }
    }

    // One-hot targets
    let mut tgt_data = vec![0i8; n * n_classes];
    for (i, &lbl) in labels.iter().enumerate() {
        tgt_data[i * n_classes + lbl as usize] = 96;
    }

    Ok(Dataset {
        inputs:    Tensor::from_vec(inp_data,  vec![n, n_features]),
        labels,
        targets:   Tensor::from_vec(tgt_data, vec![n, n_classes]),
        n_classes,
    })
}

// ─── Shuffle utility ─────────────────────────────────────────────────────────

/// Build a shuffled index array of length `n` using XorShift64.
///
/// A Fisher-Yates shuffle entirely in integer arithmetic — no float RNG needed.
/// Use this to get mini-batch indices:
///
/// ```no_run
/// use integers::data::shuffled_indices;
/// use integers::nn::XorShift64;
/// let mut rng = XorShift64::new(42);
/// let idx = shuffled_indices(1000, &mut rng);
/// for batch in idx.chunks(32) {
///     // train on batch
/// }
/// ```
pub fn shuffled_indices(n: usize, rng: &mut crate::nn::XorShift64) -> Vec<usize> {
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
    use crate::nn::XorShift64;

    // ── Iris ─────────────────────────────────────────────────────────────────

    /// Write a minimal CSV to a tempfile and round-trip it through the loader.
    fn iris_csv_roundtrip(csv: &str) -> DataResult<Dataset> {
        use std::io::Write;
        let dir = std::env::temp_dir();
        let path = dir.join(format!("test_iris_roundtrip_{:?}.csv", std::thread::current().id()));
        let mut f = File::create(&path).unwrap();
        f.write_all(csv.as_bytes()).unwrap();
        load_iris(&path)
    }

    #[test]
    fn test_iris_minimal_no_header() {
        let csv = "5.1,3.5,1.4,0.2,Iris-setosa\n
                   7.0,3.2,4.7,1.4,Iris-versicolor\n
                   6.3,3.3,6.0,2.5,Iris-virginica\n";
        let ds = iris_csv_roundtrip(csv).unwrap();
        assert_eq!(ds.len(), 3);
        assert_eq!(ds.n_features(), 4);
        assert_eq!(ds.n_classes, 3);
        assert_eq!(ds.labels, vec![0, 1, 2]);
    }

    #[test]
    fn test_iris_with_header() {
        let csv = "sepal_length,sepal_width,petal_length,petal_width,species\n
                   5.1,3.5,1.4,0.2,Iris-setosa\n
                   7.0,3.2,4.7,1.4,Iris-versicolor\n";
        let ds = iris_csv_roundtrip(csv).unwrap();
        assert_eq!(ds.len(), 2);
        assert_eq!(ds.labels[0], 0);
        assert_eq!(ds.labels[1], 1);
    }

    #[test]
    fn test_iris_kaggle_format_with_id() {
        // Kaggle IRIS.csv sometimes has an extra leading id column
        let csv = "Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species\n
                   1,5.1,3.5,1.4,0.2,Iris-setosa\n
                   2,7.0,3.2,4.7,1.4,Iris-versicolor\n
                   3,6.3,3.3,6.0,2.5,Iris-virginica\n";
        let ds = iris_csv_roundtrip(csv).unwrap();
        assert_eq!(ds.len(), 3);
        assert_eq!(ds.labels, vec![0, 1, 2]);
    }

    #[test]
    fn test_iris_integer_labels() {
        let csv = "5.1,3.5,1.4,0.2,0\n
                   7.0,3.2,4.7,1.4,1\n
                   6.3,3.3,6.0,2.5,2\n";
        let ds = iris_csv_roundtrip(csv).unwrap();
        assert_eq!(ds.labels, vec![0, 1, 2]);
    }

    #[test]
    fn test_iris_quantization_bounds() {
        let csv = "5.1,3.5,1.4,0.2,Iris-setosa\n
                   7.0,3.2,4.7,1.4,Iris-versicolor\n
                   6.3,3.3,6.0,2.5,Iris-virginica\n";
        let ds = iris_csv_roundtrip(csv).unwrap();
        for &v in &ds.inputs.data {
            assert!(v >= -127 && v <= 127, "value {} out of i8 range", v);
        }
        // Min and max of each column should hit the extremes
        for feat in 0..4 {
            let col: Vec<i8> = (0..ds.len())
                .map(|i| ds.inputs.data[i * 4 + feat])
                .collect();
            let has_min = col.contains(&-127);
            let has_max = col.contains(&127);
            assert!(has_min || has_max,
                "feature {} never hit quantization extremes: {:?}", feat, col);
        }
    }

    #[test]
    fn test_iris_one_hot_targets() {
        let csv = "5.1,3.5,1.4,0.2,Iris-setosa\n
                   7.0,3.2,4.7,1.4,Iris-versicolor\n
                   6.3,3.3,6.0,2.5,Iris-virginica\n";
        let ds = iris_csv_roundtrip(csv).unwrap();

        // Each target row: exactly one slot is 96, rest are 0
        for i in 0..ds.len() {
            let tgt = ds.get_target(i);
            let hot_count = tgt.data.iter().filter(|&&x| x == 96).count();
            let zero_count = tgt.data.iter().filter(|&&x| x == 0).count();
            assert_eq!(hot_count, 1, "sample {} should have exactly one hot entry", i);
            assert_eq!(zero_count, 2, "sample {} should have two zero entries", i);
            // The hot slot matches the label
            let expected_slot = ds.labels[i] as usize;
            assert_eq!(tgt.data[expected_slot], 96);
        }
    }

    #[test]
    fn test_iris_get_input_shape() {
        let csv = "5.1,3.5,1.4,0.2,Iris-setosa\n
                   7.0,3.2,4.7,1.4,Iris-versicolor\n";
        let ds = iris_csv_roundtrip(csv).unwrap();
        let inp = ds.get_input(0);
        assert_eq!(inp.shape, vec![1, 4]);
        assert_eq!(inp.data.len(), 4);
    }

    #[test]
    fn test_iris_empty_error() {
        let csv = "sepal_length,sepal_width,petal_length,petal_width,species\n";
        let result = iris_csv_roundtrip(csv);
        assert!(matches!(result, Err(DataError::EmptyDataset)));
    }

    #[test]
    fn test_iris_unknown_species_error() {
        let csv = "5.1,3.5,1.4,0.2,UnknownFlower\n";
        let result = iris_csv_roundtrip(csv);
        assert!(matches!(result, Err(DataError::ParseError(_))));
    }

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
    fn test_minibatch_shapes() {
        let csv = "5.1,3.5,1.4,0.2,Iris-setosa\n\
                   7.0,3.2,4.7,1.4,Iris-versicolor\n\
                   6.3,3.3,6.0,2.5,Iris-virginica\n";
        let ds = iris_csv_roundtrip(csv).unwrap();
        let (inp, tgt) = ds.minibatch(&[0, 2]);
        assert_eq!(inp.shape, vec![2, 4]);
        assert_eq!(tgt.shape, vec![2, 3]);
    }

    #[test]
    fn test_argmax() {
        let t = Tensor::from_vec(vec![-10i8, 50, 20], vec![1, 3]);
        assert_eq!(Dataset::argmax(&t), 1);
    }

    // ── MNIST ─────────────────────────────────────────────────────────────────
    // Full MNIST requires the downloaded files. These tests check the logic
    // with hand-crafted binary blobs so no network access is needed.

    fn write_idx3(path: &Path, images: &[u8], n: u32, rows: u32, cols: u32) {
        use std::io::Write;
        let mut f = File::create(path).unwrap();
        f.write_all(&0x0000_0803u32.to_be_bytes()).unwrap();
        f.write_all(&n.to_be_bytes()).unwrap();
        f.write_all(&rows.to_be_bytes()).unwrap();
        f.write_all(&cols.to_be_bytes()).unwrap();
        f.write_all(images).unwrap();
    }

    fn write_idx1(path: &Path, labels: &[u8]) {
        use std::io::Write;
        let mut f = File::create(path).unwrap();
        f.write_all(&0x0000_0801u32.to_be_bytes()).unwrap();
        f.write_all(&(labels.len() as u32).to_be_bytes()).unwrap();
        f.write_all(labels).unwrap();
    }

    fn make_fake_mnist(dir: &Path, n: usize) {
        let pixels: Vec<u8> = (0..n * 4).map(|i| (i % 256) as u8).collect();
        write_idx3(&dir.join("train-images-idx3-ubyte"), &pixels, n as u32, 2, 2);
        write_idx1(&dir.join("train-labels-idx1-ubyte"), &(0..n).map(|i| (i % 10) as u8).collect::<Vec<_>>());
    }

    #[test]
    fn test_mnist_pixel_quantization() {
        let dir = std::env::temp_dir().join("fake_mnist_quant");
        std::fs::create_dir_all(&dir).unwrap();
        // Single image: pixels 0 and 255 must map to -128 and 127
        write_idx3(&dir.join("train-images-idx3-ubyte"), &[0u8, 128, 255, 64], 1, 2, 2);
        write_idx1(&dir.join("train-labels-idx1-ubyte"), &[3u8]);

        let ds = load_mnist(&dir, "train", None).unwrap();
        assert_eq!(ds.inputs.data[0], -128i8);  // 0   → -128
        assert_eq!(ds.inputs.data[1], 0i8);     // 128 → 0
        assert_eq!(ds.inputs.data[2], 127i8);   // 255 → 127
    }

    #[test]
    fn test_mnist_load_shape_and_labels() {
        let dir = std::env::temp_dir().join("fake_mnist_shape");
        std::fs::create_dir_all(&dir).unwrap();
        make_fake_mnist(&dir, 5);

        let ds = load_mnist(&dir, "train", None).unwrap();
        assert_eq!(ds.len(), 5);
        assert_eq!(ds.n_features(), 4);   // 2×2 images
        assert_eq!(ds.n_classes, 10);
        assert_eq!(ds.inputs.shape, vec![5, 4]);
    }

    #[test]
    fn test_mnist_max_samples_cap() {
        let dir = std::env::temp_dir().join("fake_mnist_cap");
        std::fs::create_dir_all(&dir).unwrap();
        make_fake_mnist(&dir, 10);

        let ds = load_mnist(&dir, "train", Some(3)).unwrap();
        assert_eq!(ds.len(), 3);
    }

    #[test]
    fn test_mnist_one_hot_targets() {
        let dir = std::env::temp_dir().join("fake_mnist_hot");
        std::fs::create_dir_all(&dir).unwrap();
        write_idx3(&dir.join("train-images-idx3-ubyte"), &[0u8; 4], 1, 2, 2);
        write_idx1(&dir.join("train-labels-idx1-ubyte"), &[7u8]);

        let ds = load_mnist(&dir, "train", None).unwrap();
        let tgt = ds.get_target(0);
        assert_eq!(tgt.data[7], 96);
        for (i, &v) in tgt.data.iter().enumerate() {
            if i != 7 { assert_eq!(v, 0); }
        }
    }

    #[test]
    fn test_mnist_invalid_magic() {
        use std::io::Write;
        let dir = std::env::temp_dir().join("fake_mnist_magic");
        std::fs::create_dir_all(&dir).unwrap();

        let path = dir.join("train-images-idx3-ubyte");
        let mut f = File::create(&path).unwrap();
        f.write_all(&0xDEAD_BEEFu32.to_be_bytes()).unwrap();
        write_idx1(&dir.join("train-labels-idx1-ubyte"), &[0u8]);

        let result = load_mnist(&dir, "train", None);
        assert!(matches!(result, Err(DataError::InvalidMagic { .. })));
    }

    #[test]
    fn test_mnist_unknown_split() {
        let dir = std::env::temp_dir();
        let result = load_mnist(&dir, "validation", None);
        assert!(matches!(result, Err(DataError::ParseError(_))));
    }
}
