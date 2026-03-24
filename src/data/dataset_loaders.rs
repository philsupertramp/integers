//! Fluent builder for loading CSV, TSV, and (optionally) Parquet datasets.
//!
//! # Quick start
//! ```no_run
//! use dyadic_nn::dataset_loaders::{DatasetBuilder, QuantizationMethod};
//!
//! let ds = DatasetBuilder::<i32>::new_csv("iris.csv")
//!     .with_header(true)
//!     .with_features(vec![0, 1, 2, 3])
//!     .with_label_column(4)
//!     .with_quantization(QuantizationMethod::MinMax)
//!     .load()
//!     .unwrap();
//! ```
//!
//! Parquet support requires the `parquet-support` Cargo feature:
//! ```sh
//! cargo run --example iris --features parquet-support
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::quant::{minmax_quantize, none_quantize, standard_score_quantize};
use crate::dyadic::{Dyadic, Tensor};

// ─── Configuration ────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub enum FileFormat { CSV, TSV, Parquet }

#[derive(Clone, Debug)]
pub enum QuantizationMethod {
    /// No rescaling — cast `f32` directly to `i32`.
    None,
    /// Scale each column linearly to `[-127, 127]` (shift = 7).
    MinMax,
    /// Z-score each column then scale ±3σ to ±127 (shift = 7).
    StandardScore,
    /// Plug in a custom `fn(&[f32]) -> (Vec<i32>, i32)` — not yet wired up.
    Custom,
}

/// Full configuration for a dataset load.
#[derive(Clone, Debug)]
pub struct DatasetConfig {
    pub format:          FileFormat,
    /// Which 0-indexed columns are features.  `None` = all except label.
    pub feature_columns: Option<Vec<usize>>,
    pub label_column:    usize,
    pub has_header:      bool,
    pub class_mapping:   Option<HashMap<String, u8>>,
    pub quantization:    QuantizationMethod,
    /// Override inferred class count.
    pub num_classes:     Option<usize>,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            format:          FileFormat::CSV,
            feature_columns: None,
            label_column:    4,     // common for Iris-like datasets
            has_header:      false,
            class_mapping:   None,
            quantization:    QuantizationMethod::None,
            num_classes:     None,
        }
    }
}

impl DatasetConfig {
    fn with_format(format: FileFormat) -> Self {
        Self { format, ..Self::default() }
    }
}

// ─── Builder ──────────────────────────────────────────────────────────────────

/// Fluent builder — mirrors `torch.utils.data` conventions.
pub struct DatasetBuilder {
    path:     std::path::PathBuf,
    config:   DatasetConfig,
}

impl DatasetBuilder {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self { path: path.as_ref().to_path_buf(), config: DatasetConfig::default(), }
    }
    pub fn new_csv(path: impl AsRef<Path>) -> Self {
        Self { path: path.as_ref().to_path_buf(), config: DatasetConfig::with_format(FileFormat::CSV), }
    }
    pub fn new_tsv(path: impl AsRef<Path>) -> Self {
        Self { path: path.as_ref().to_path_buf(), config: DatasetConfig::with_format(FileFormat::TSV), }
    }
    pub fn new_parquet(path: impl AsRef<Path>) -> Self {
        Self { path: path.as_ref().to_path_buf(), config: DatasetConfig::with_format(FileFormat::Parquet), }
    }

    pub fn format(mut self, fmt: FileFormat) -> Self { self.config.format = fmt; self }
    pub fn with_features(mut self, cols: Vec<usize>) -> Self { self.config.feature_columns = Some(cols); self }
    pub fn with_label_column(mut self, col: usize) -> Self { self.config.label_column = col; self }
    pub fn with_header(mut self, h: bool) -> Self { self.config.has_header = h; self }
    pub fn with_quantization(mut self, q: QuantizationMethod) -> Self { self.config.quantization = q; self }
    pub fn with_class_mapping(mut self, m: HashMap<String, u8>) -> Self { self.config.class_mapping = Some(m); self }
    pub fn with_num_classes(mut self, n: usize) -> Self { self.config.num_classes = Some(n); self }

    /// Load the dataset according to the current configuration.
    pub fn load(self) -> crate::data::DataResult<crate::data::Dataset> {
        match self.config.format {
            FileFormat::CSV | FileFormat::TSV =>
                load_csv(&self.path, self.config),

            FileFormat::Parquet => {
                #[cfg(feature = "parquet-support")]
                { return load_parquet(&self.path, self.config); }

                #[cfg(not(feature = "parquet-support"))]
                Err(crate::data::DataError::ParseError(
                    "Parquet support is not compiled in. \
                     Add `--features parquet-support` to your cargo command.".to_string()
                ))
            }
        }
    }
}

// ─── CSV / TSV loader ─────────────────────────────────────────────────────────

fn load_csv(
    path: &Path,
    config: DatasetConfig,
) -> crate::data::DataResult<crate::data::Dataset> {
    use crate::data::DataError;

    let delimiter = match config.format {
        FileFormat::CSV => ',',
        FileFormat::TSV => '\t',
        FileFormat::Parquet => unreachable!(),
    };

    let reader = BufReader::new(File::open(path)?);
    let mut raw_features: Vec<Vec<f32>> = Vec::new();
    let mut labels: Vec<u8> = Vec::new();
    let mut label_map: HashMap<String, u8> = config.class_mapping.clone().unwrap_or_default();
    let mut next_id = label_map.len() as u8;
    let mut line_no = 0usize;

    for line_result in reader.lines() {
        let line = line_result?;
        let line = line.trim();
        if line.is_empty() { continue; }

        if config.has_header && line_no == 0 { line_no += 1; continue; }

        let fields: Vec<&str> = line.split(delimiter).collect();

        // Decide which columns are features.
        let feat_cols: Vec<usize> = config.feature_columns.clone().unwrap_or_else(|| {
            (0..fields.len()).filter(|&i| i != config.label_column).collect()
        });

        // Parse feature values.
        let mut row = Vec::with_capacity(feat_cols.len());
        for &col in &feat_cols {
            if col >= fields.len() {
                return Err(DataError::ParseError(
                    format!("Feature column {col} out of bounds in: {line}")));
            }
            let v = fields[col].trim().parse::<f32>().map_err(|_| {
                DataError::ParseError(
                    format!("Cannot parse '{}' as f32 in: {line}", fields[col]))
            })?;
            row.push(v);
        }

        // Parse label (numeric or string).
        if config.label_column >= fields.len() {
            return Err(DataError::ParseError(
                format!("Label column {} out of bounds in: {line}", config.label_column)));
        }
        let lf = fields[config.label_column].trim();
        let label = if let Ok(n) = lf.parse::<u8>() {
            n
        } else {
            if !label_map.contains_key(lf) {
                label_map.insert(lf.to_string(), next_id);
                next_id += 1;
            }
            label_map[lf]
        };

        raw_features.push(row);
        labels.push(label);
        line_no += 1;
    }

    if labels.is_empty() { return Err(crate::data::DataError::EmptyDataset); }

    finalize_dataset(raw_features, labels, config)
}

// ─── Parquet loader (feature-gated) ──────────────────────────────────────────

#[cfg(feature = "parquet-support")]
fn load_parquet(
    path: &Path,
    config: DatasetConfig,
) -> crate::data::DataResult<crate::data::Dataset> {
    use parquet::file::reader::{FileReader, SerializedFileReader};
    use parquet::record::Field;
    use crate::data::DataError;

    let file   = File::open(path)?;
    let reader = SerializedFileReader::new(file)
        .map_err(|e| DataError::ParseError(format!("Failed to open parquet: {e}")))?;

    let mut raw_features: Vec<Vec<f32>> = Vec::new();
    let mut labels: Vec<u8> = Vec::new();
    let mut label_map: HashMap<String, u8> = config.class_mapping.clone().unwrap_or_default();
    let mut next_id = label_map.len() as u8;

    let row_iter = reader.get_row_iter(None)
        .map_err(|e| DataError::ParseError(format!("Failed to get row iterator: {e}")))?;

    for rec in row_iter {
        let rec = rec.map_err(|e| DataError::ParseError(format!("Row error: {e}")))?;

        let feat_cols: Vec<usize> = config.feature_columns.clone().unwrap_or_else(|| {
            (0..rec.len()).filter(|&i| i != config.label_column).collect()
        });

        let mut row = Vec::with_capacity(feat_cols.len());
        let mut label_field = &Field::Null;

        for (idx, (_name, value)) in rec.get_column_iter().enumerate() {
            if idx == config.label_column {
                label_field = value;
            } else if feat_cols.contains(&idx) {
                let fv = match value {
                    Field::Float(f)  => *f as f32,
                    Field::Double(d) => *d as f32,
                    Field::Int(i)    => *i as f32,
                    Field::Long(l)   => *l as f32,
                    Field::UByte(u)  => *u as f32,
                    Field::UInt(u)   => *u as f32,
                    Field::ULong(u)  => *u as f32,
                    other => return Err(DataError::ParseError(
                        format!("Unsupported feature field type: {other:?}"))),
                };
                row.push(fv);
            }
        }

        let label = match label_field {
            Field::Str(s) => {
                if !label_map.contains_key(s.as_str()) {
                    label_map.insert(s.clone(), next_id);
                    next_id += 1;
                }
                label_map[s.as_str()]
            }
            Field::UByte(u) => *u,
            Field::Int(i)   => *i as u8,
            Field::Long(l)  => *l as u8,
            Field::UInt(u)  => *u as u8,
            Field::ULong(u) => *u as u8,
            other => return Err(DataError::ParseError(
                format!("Unsupported label field type: {other:?}"))),
        };

        raw_features.push(row);
        labels.push(label);
    }

    if labels.is_empty() { return Err(crate::data::DataError::EmptyDataset); }

    finalize_dataset(raw_features, labels, config)
}

// ─── Shared finalization ──────────────────────────────────────────────────────

fn finalize_dataset(
    raw_features: Vec<Vec<f32>>,
    labels: Vec<u8>,
    config: DatasetConfig,
) -> crate::data::DataResult<crate::data::Dataset> {
    let n_samples  = labels.len();
    let n_features = raw_features.first().map_or(0, |r| r.len());
    let n_classes  = config.num_classes
        .unwrap_or_else(|| labels.iter().max().map_or(0, |&x| x as usize + 1));

    // Quantise column by column, then re-interleave into row-major layout.
    let mut columns: Vec<Vec<i32>> = Vec::with_capacity(n_features);
    let mut input_shift: i32 = 0;

    for fi in 0..n_features {
        let col: Vec<f32> = raw_features.iter().map(|r| r[fi]).collect();
        let (quantized, shift) = match &config.quantization {
            QuantizationMethod::None          => none_quantize(&col),
            QuantizationMethod::MinMax        => minmax_quantize(&col),
            QuantizationMethod::StandardScore => standard_score_quantize(&col),
            QuantizationMethod::Custom        => return Err(
                crate::data::DataError::ParseError(
                    "Custom quantization not yet implemented".to_string())),
        };
        input_shift = shift;    // all columns share one strategy, so shift is uniform
        columns.push(quantized);
    }

    let shift_u32 = input_shift.max(0) as u32;

    // Row-major input tensor [n_samples × n_features]
    let mut inp_data = vec![Dyadic::new(0, shift_u32); n_samples * n_features];
    for (si, row) in inp_data.chunks_exact_mut(n_features).enumerate() {
        for (fi, cell) in row.iter_mut().enumerate() {
            *cell = Dyadic::new(columns[fi][si], shift_u32);
        }
    }

    // One-hot target tensor [n_samples × n_classes], hot bit = 127
    let mut tgt_data = vec![Dyadic::new(0, shift_u32); n_samples * n_classes];
    for (i, &lbl) in labels.iter().enumerate() {
        tgt_data[i * n_classes + lbl as usize] = Dyadic::new(127, shift_u32);
    }

    Ok(crate::data::Dataset {
        inputs:      Tensor::from_vec(inp_data, vec![n_samples, n_features]),
        labels,
        targets:     Tensor::from_vec(tgt_data, vec![n_samples, n_classes]),
        n_classes,
        input_shift: input_shift,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_tmp(name: &str, contents: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(name);
        let mut f = File::create(&p).unwrap();
        f.write_all(contents.as_bytes()).unwrap();
        p
    }

    #[test]
    fn csv_string_labels() {
        let p = write_tmp("test_iris_str.csv",
            "5.1,3.5,1.4,0.2,setosa\n\
             7.0,3.2,4.7,1.4,versicolor\n\
             6.3,3.3,6.0,2.5,virginica\n");
        let ds = DatasetBuilder::<i32>::new_csv(&p)
            .with_features(vec![0, 1, 2, 3])
            .with_label_column(4)
            .load().unwrap();
        assert_eq!(ds.len(), 3);
        assert_eq!(ds.n_features(), 4);
        assert_eq!(ds.n_classes, 3);
    }

    #[test]
    fn csv_with_header() {
        let p = write_tmp("test_iris_hdr.csv",
            "sl,sw,pl,pw,species\n\
             5.1,3.5,1.4,0.2,setosa\n\
             7.0,3.2,4.7,1.4,versicolor\n");
        let ds = DatasetBuilder::<i32>::new_csv(&p)
            .with_header(true)
            .with_features(vec![0, 1, 2, 3])
            .with_label_column(4)
            .load().unwrap();
        assert_eq!(ds.len(), 2);
    }

    #[test]
    fn csv_numeric_labels() {
        let p = write_tmp("test_iris_num.csv",
            "5.1,3.5,1.4,0.2,0\n\
             7.0,3.2,4.7,1.4,1\n\
             6.3,3.3,6.0,2.5,2\n");
        let ds = DatasetBuilder::<i32>::new_csv(&p)
            .with_features(vec![0, 1, 2, 3])
            .with_label_column(4)
            .load().unwrap();
        assert_eq!(ds.labels, vec![0, 1, 2]);
    }

    #[test]
    fn csv_minmax_quantization() {
        let p = write_tmp("test_mm.csv",
            "0.0,0.0,0\n1.0,0.5,1\n");
        let ds = DatasetBuilder::<i32>::new_csv(&p)
            .with_features(vec![0, 1])
            .with_label_column(2)
            .with_quantization(QuantizationMethod::MinMax)
            .load().unwrap();
        assert_eq!(ds.input_shift, 7);
        for &v in &ds.inputs.data {
            assert!(v >= -127 && v <= 127);
        }
    }
}
