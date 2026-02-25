/// Generalized dataset loader supporting multiple formats and flexible configuration.
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::nn::Tensor;

// ─── Configuration ────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub enum FileFormat {
    CSV,
    TSV,
}

#[derive(Clone, Debug)]
pub enum QuantizationMethod {
    /// Min-max scale each column to [-127, 127]
    MinMax,
    /// (value - mean) / std, clamped to [-127, 127]
    StandardScore,
    /// Custom quantization function
    Custom,
}

/// Configuration for loading a dataset from any supported format.
#[derive(Clone, Debug)]
pub struct DatasetConfig {
    /// CSV, TSV, etc.
    pub format: FileFormat,
    /// Which columns contain features (0-indexed). None = all except label.
    pub feature_columns: Option<Vec<usize>>,
    /// Index of the label/target column
    pub label_column: usize,
    /// Whether the first row is a header
    pub has_header: bool,
    /// How to map string labels to numeric class indices
    pub class_mapping: Option<HashMap<String, u8>>,
    /// Quantization strategy for floats
    pub quantization: QuantizationMethod,
    /// Number of classes (inferred from data if None)
    pub num_classes: Option<usize>,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            format: FileFormat::CSV,
            feature_columns: None,
            label_column: 4, // Common for Iris-like datasets
            has_header: false,
            class_mapping: None,
            quantization: QuantizationMethod::MinMax,
            num_classes: None,
        }
    }
}
impl DatasetConfig {
    fn default_format(format: FileFormat) -> Self {
        Self {
            format,
            feature_columns: None,
            label_column: 4, // Common for Iris-like datasets
            has_header: false,
            class_mapping: None,
            quantization: QuantizationMethod::MinMax,
            num_classes: None,
        }
    }
}

// ─── Builder API ──────────────────────────────────────────────────────────────

/// Fluent builder for dataset loading.
pub struct DatasetBuilder {
    path: std::path::PathBuf,
    config: DatasetConfig,
}

impl DatasetBuilder {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            config: DatasetConfig::default(),
        }
    }
    pub fn new_csv(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            config: DatasetConfig::default_format(FileFormat::CSV),
        }
    }
    pub fn new_tsv(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            config: DatasetConfig::default_format(FileFormat::TSV),
        }
    }

    pub fn format(mut self, format: FileFormat) -> Self {
        self.config.format = format;
        self
    }

    pub fn with_features(mut self, columns: Vec<usize>) -> Self {
        self.config.feature_columns = Some(columns);
        self
    }

    pub fn with_label_column(mut self, col: usize) -> Self {
        self.config.label_column = col;
        self
    }

    pub fn with_header(mut self, has_header: bool) -> Self {
        self.config.has_header = has_header;
        self
    }

    pub fn with_quantization(mut self, method: QuantizationMethod) -> Self {
        self.config.quantization = method;
        self
    }

    pub fn with_class_mapping(mut self, mapping: HashMap<String, u8>) -> Self {
        self.config.class_mapping = Some(mapping);
        self
    }

    pub fn with_num_classes(mut self, n: usize) -> Self {
        self.config.num_classes = Some(n);
        self
    }

    pub fn load(self) -> crate::data::DataResult<crate::data::Dataset> {
        match self.config.format {
            FileFormat::CSV | FileFormat::TSV => load_csv(&self.path, self.config),
        }
    }
}

// ─── Quantization Helpers ─────────────────────────────────────────────────────

fn minmax_quantize(values: &[f32]) -> Vec<i8> {
    if values.is_empty() {
        return Vec::new();
    }
    let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(1e-6);
    values
        .iter()
        .map(|&v| {
            let norm = (v - min) / range; // [0, 1]
            let scaled = norm * 254.0 - 127.0; // [-127, 127]
            scaled.round().clamp(-127.0, 127.0) as i8
        })
        .collect()
}

fn standard_score_quantize(values: &[f32]) -> Vec<i8> {
    if values.is_empty() {
        return Vec::new();
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
    let std = variance.sqrt().max(1e-6);
    values
        .iter()
        .map(|&v| {
            let zscore = (v - mean) / std;
            zscore.round().clamp(-127.0, 127.0) as i8
        })
        .collect()
}

// ─── CSV Loader ───────────────────────────────────────────────────────────────

fn load_csv(
    path: &Path,
    config: DatasetConfig,
) -> crate::data::DataResult<crate::data::Dataset> {
    let f = BufReader::new(File::open(path)?);
    let delimiter = match config.format {
        FileFormat::CSV => ',',
        FileFormat::TSV => '\t',
    };

    let mut raw_features: Vec<Vec<f32>> = Vec::new();
    let mut labels: Vec<u8> = Vec::new();
    let mut label_string_map: HashMap<String, u8> = config.class_mapping.unwrap_or_default();
    let mut next_class_id = label_string_map.len() as u8;

    let mut line_count = 0;
    for line_result in f.lines() {
        let line = line_result?;
        let line = line.trim();

        // Skip empty lines
        if line.is_empty() {
            continue;
        }

        // Skip header
        if config.has_header && line_count == 0 {
            line_count += 1;
            continue;
        }

        let fields: Vec<&str> = line.split(delimiter).collect();

        // Determine feature columns
        let feature_cols: Vec<usize> = if let Some(ref cols) = config.feature_columns {
            cols.clone()
        } else {
            // All columns except label
            (0..fields.len())
                .filter(|i| *i != config.label_column)
                .collect()
        };

        // Parse features
        let mut row = Vec::with_capacity(feature_cols.len());
        for &col_idx in &feature_cols {
            if col_idx >= fields.len() {
                return Err(crate::data::DataError::ParseError(
                    format!(
                        "Feature column {} out of bounds in line: {}",
                        col_idx, line
                    ),
                ));
            }
            let val = fields[col_idx]
                .trim()
                .parse::<f32>()
                .map_err(|_| {
                    crate::data::DataError::ParseError(format!(
                        "Cannot parse '{}' as f32 in line: {}",
                        fields[col_idx], line
                    ))
                })?;
            row.push(val);
        }

        // Parse label
        if config.label_column >= fields.len() {
            return Err(crate::data::DataError::ParseError(format!(
                "Label column {} out of bounds in line: {}",
                config.label_column, line
            )));
        }

        let label_field = fields[config.label_column].trim();
        let label = if let Ok(num) = label_field.parse::<u8>() {
            // Numeric label
            num
        } else {
            // String label — map to id
            if !label_string_map.contains_key(label_field) {
                label_string_map.insert(label_field.to_string(), next_class_id);
                next_class_id += 1;
            }
            label_string_map[label_field]
        };

        raw_features.push(row);
        labels.push(label);
        line_count += 1;
    }

    if labels.is_empty() {
        return Err(crate::data::DataError::EmptyDataset);
    }

    let n_samples = labels.len();
    let n_features = if raw_features.is_empty() {
        0
    } else {
        raw_features[0].len()
    };
    let n_classes = config
        .num_classes
        .unwrap_or(labels.iter().max().map(|&x| x as usize + 1).unwrap_or(0));

    // Quantize per-column
    let mut columns: Vec<Vec<i8>> = Vec::with_capacity(n_features);
    for feat_idx in 0..n_features {
        let col: Vec<f32> = raw_features.iter().map(|r| r[feat_idx]).collect();
        let quantized = match config.quantization {
            QuantizationMethod::MinMax => minmax_quantize(&col),
            QuantizationMethod::StandardScore => standard_score_quantize(&col),
            QuantizationMethod::Custom => {
                return Err(crate::data::DataError::ParseError(
                    "Custom quantization not yet implemented".to_string(),
                ))
            }
        };
        columns.push(quantized);
    }

    // Re-interleave into row-major layout [n, n_features]
    let mut inp_data = vec![0i8; n_samples * n_features];
    for (sample_idx, row) in inp_data.chunks_exact_mut(n_features).enumerate() {
        for (feat_idx, cell) in row.iter_mut().enumerate() {
            *cell = columns[feat_idx][sample_idx];
        }
    }

    // One-hot targets
    let mut tgt_data = vec![0i8; n_samples * n_classes];
    for (i, &lbl) in labels.iter().enumerate() {
        tgt_data[i * n_classes + lbl as usize] = 96;
    }

    Ok(crate::data::Dataset {
        inputs: Tensor::from_vec(inp_data, vec![n_samples, n_features]),
        labels,
        targets: Tensor::from_vec(tgt_data, vec![n_samples, n_classes]),
        n_classes,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_csv_builder_basic() {
        let csv = "5.1,3.5,1.4,0.2,setosa\n\
                   7.0,3.2,4.7,1.4,versicolor\n\
                   6.3,3.3,6.0,2.5,virginica\n";

        let dir = std::env::temp_dir();
        let path = dir.join(format!("test_builder_{:?}.csv", std::thread::current().id()));
        let mut f = File::create(&path).unwrap();
        f.write_all(csv.as_bytes()).unwrap();

        let ds = DatasetBuilder::new_csv(&path)
            .with_features(vec![0, 1, 2, 3])
            .with_label_column(4)
            .load()
            .unwrap();

        assert_eq!(ds.len(), 3);
        assert_eq!(ds.n_features(), 4);
        assert_eq!(ds.n_classes, 3);
    }

    #[test]
    fn test_csv_with_header() {
        let csv = "sepal_length,sepal_width,petal_length,petal_width,species\n\
                   5.1,3.5,1.4,0.2,setosa\n\
                   7.0,3.2,4.7,1.4,versicolor\n";

        let dir = std::env::temp_dir();
        let path = dir.join(format!("test_header_{:?}.csv", std::thread::current().id()));
        let mut f = File::create(&path).unwrap();
        f.write_all(csv.as_bytes()).unwrap();

        let ds = DatasetBuilder::new_csv(&path)
            .with_header(true)
            .with_features(vec![0, 1, 2, 3])
            .with_label_column(4)
            .load()
            .unwrap();

        assert_eq!(ds.len(), 2);
    }

    #[test]
    fn test_numeric_labels() {
        let csv = "5.1,3.5,1.4,0.2,0\n\
                   7.0,3.2,4.7,1.4,1\n\
                   6.3,3.3,6.0,2.5,2\n";

        let dir = std::env::temp_dir();
        let path = dir.join(format!("test_numeric_{:?}.csv", std::thread::current().id()));
        let mut f = File::create(&path).unwrap();
        f.write_all(csv.as_bytes()).unwrap();

        let ds = DatasetBuilder::new_csv(&path)
            .with_features(vec![0, 1, 2, 3])
            .with_label_column(4)
            .load()
            .unwrap();

        assert_eq!(ds.labels, vec![0, 1, 2]);
    }

    #[test]
    fn test_standard_score_quantization() {
        let csv = "5.1,3.5,1.4,0.2,0\n\
                   7.0,3.2,4.7,1.4,1\n\
                   6.3,3.3,6.0,2.5,2\n";

        let dir = std::env::temp_dir();
        let path = dir.join(format!("test_zscore_{:?}.csv", std::thread::current().id()));
        let mut f = File::create(&path).unwrap();
        f.write_all(csv.as_bytes()).unwrap();

        let ds = DatasetBuilder::new_csv(&path)
            .with_features(vec![0, 1, 2, 3])
            .with_label_column(4)
            .with_quantization(QuantizationMethod::StandardScore)
            .load()
            .unwrap();

        assert_eq!(ds.len(), 3);
        // Values should be in i8 range
        for &v in &ds.inputs.data {
            assert!(v >= -127 && v <= 127);
        }
    }
}
