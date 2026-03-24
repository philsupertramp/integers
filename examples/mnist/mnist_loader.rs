//! MNIST dataset loaders (IDX binary and Parquet).
//!
//! Parquet support requires the `parquet-support` Cargo feature.
//! Auto-detection: IDX is always preferred when both formats are present.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::time::Instant;

use integers::data::{DataError, DataResult, Dataset};
use integers::tensor::Tensor;

// ─── Shared helpers ───────────────────────────────────────────────────────────

#[inline]
fn normalize_pixel(p: u8) -> i32 {
    let scaled = (p as f32 / 255.0) * 254.0 - 127.0;
    scaled.round().clamp(-127.0, 127.0) as i32
}

fn build_targets(labels: &[u8], n_classes: usize) -> Vec<i32> {
    let mut tgt = vec![0i32; labels.len() * n_classes];
    for (i, &lbl) in labels.iter().enumerate() {
        tgt[i * n_classes + lbl as usize] = 127;
    }
    tgt
}

// ─── IDX binary loader ────────────────────────────────────────────────────────

const IMAGE_MAGIC: u32 = 0x0000_0803;
const LABEL_MAGIC: u32 = 0x0000_0801;

fn read_u32_be(r: &mut impl Read) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

/// Load an MNIST split from a pair of IDX binary files.
///
/// `normalize = true`  → pixels mapped to `[-127, 127]` at `input_shift = 7`.
/// `normalize = false` → raw bytes `[0, 255]` at `input_shift = 0`.
pub fn load_mnist_idx(
    images_path: &Path,
    labels_path: &Path,
    normalize:   bool,
) -> DataResult<Dataset<i32>> {
    let t0 = Instant::now();

    let mut img = BufReader::new(File::open(images_path)?);
    let magic   = read_u32_be(&mut img)?;
    if magic != IMAGE_MAGIC {
        return Err(DataError::InvalidMagic { expected: IMAGE_MAGIC, found: magic });
    }
    let n_images = read_u32_be(&mut img)? as usize;
    let n_rows   = read_u32_be(&mut img)? as usize;
    let n_cols   = read_u32_be(&mut img)? as usize;
    let n_pixels = n_rows * n_cols;

    let mut pixel_bytes = vec![0u8; n_images * n_pixels];
    img.read_exact(&mut pixel_bytes)?;

    let mut lbl    = BufReader::new(File::open(labels_path)?);
    let lmagic     = read_u32_be(&mut lbl)?;
    if lmagic != LABEL_MAGIC {
        return Err(DataError::InvalidMagic { expected: LABEL_MAGIC, found: lmagic });
    }
    let n_labels = read_u32_be(&mut lbl)? as usize;
    if n_images != n_labels {
        return Err(DataError::DimensionMismatch { images: n_images, labels: n_labels });
    }
    let mut label_bytes = vec![0u8; n_labels];
    lbl.read_exact(&mut label_bytes)?;

    let (inp_data, input_shift) = if normalize {
        (pixel_bytes.iter().map(|&p| normalize_pixel(p)).collect(), 7)
    } else {
        (pixel_bytes.iter().map(|&p| p as i32).collect(), 0)
    };

    let tgt_data = build_targets(&label_bytes, 10);
    eprintln!("  [IDX] loaded {n_images} samples in {}ms", t0.elapsed().as_millis());

    Ok(Dataset::<i32> {
        inputs:      Tensor::from_vec(inp_data,  vec![n_images, n_pixels]),
        labels:      label_bytes,
        targets:     Tensor::from_vec(tgt_data,  vec![n_labels, 10]),
        n_classes:   10,
        input_shift,
    })
}

// ─── Parquet loader (feature-gated) ───────────────────────────────────────────

/// Load an MNIST split from a single Parquet file using the column reader API.
///
/// Expected schema: a BYTE_ARRAY `image` column (PNG bytes) and an INT32/INT64
/// `label` column. This matches the HuggingFace `ylecun/mnist` dataset format.
///
/// Requires the `parquet-support` Cargo feature.
#[cfg(feature = "parquet-support")]
pub fn load_mnist_parquet(
    path:      &Path,
    normalize: bool,
) -> DataResult<Dataset<i32>> {
    use parquet::column::reader::ColumnReader;
    use parquet::file::reader::{FileReader, SerializedFileReader};

    let t0   = Instant::now();
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file)
        .map_err(|e| DataError::ParseError(format!("Cannot open parquet: {e}")))?;

    let meta         = reader.metadata();
    let n_rows_total = meta.file_metadata().num_rows() as usize;
    let fields       = meta.file_metadata().schema().get_fields();

    // Locate image and label columns by name.
    let img_col = fields.iter().position(|f| matches!(f.name(), "image" | "img"))
        .ok_or_else(|| DataError::ParseError(
            format!("No image column found. Columns: {:?}",
                fields.iter().map(|f| f.name()).collect::<Vec<_>>())))?;
    let lbl_col = fields.iter().position(|f| matches!(f.name(), "label" | "labels"))
        .ok_or_else(|| DataError::ParseError(
            format!("No label column found. Columns: {:?}",
                fields.iter().map(|f| f.name()).collect::<Vec<_>>())))?;

    let mut inp_data   = Vec::<i32>::with_capacity(n_rows_total * 784);
    let mut label_data = Vec::<u8>::with_capacity(n_rows_total);

    for rg_idx in 0..meta.num_row_groups() {
        // get_row_group returns Box<dyn RowGroupReader>; call methods directly on it.
        let rg      = reader.get_row_group(rg_idx)
            .map_err(|e| DataError::ParseError(format!("Row group {rg_idx}: {e}")))?;
        let rg_rows = meta.row_group(rg_idx).num_rows() as usize;

        // ── Image column ──────────────────────────────────────────────────────
        // v58: read_records(max, Option<&mut Vec<i16>>, Option<&mut Vec<i16>>, &mut Vec<T::T>)
        let mut col = rg.get_column_reader(img_col)
            .map_err(|e| DataError::ParseError(format!("Image col reader: {e}")))?;

        match &mut col {
            ColumnReader::ByteArrayColumnReader(r) => {
                let mut values:     Vec<parquet::data_type::ByteArray> = Vec::new();
                let mut def_levels: Vec<i16> = Vec::new();
                let (n, _, _) = r.read_records(rg_rows, Some(&mut def_levels), None, &mut values)
                    .map_err(|e| DataError::ParseError(format!("Image read: {e}")))?;

                for ba in values.iter().take(n) {
                    // Raw pixel bytes — 784 bytes per image.
                    let bytes = ba.data();
                    if bytes.len() != 784 {
                        return Err(DataError::ParseError(
                            format!("Expected 784 bytes per image, got {}", bytes.len())));
                    }
                    inp_data.extend(bytes.iter().map(|&p| {
                        if normalize { normalize_pixel(p) } else { p as i32 }
                    }));
                }
            }
            _ => return Err(DataError::ParseError(
                "Image column is not BYTE_ARRAY. Check schema.".to_string())),
        }

        // ── Label column ──────────────────────────────────────────────────────
        let mut col = rg.get_column_reader(lbl_col)
            .map_err(|e| DataError::ParseError(format!("Label col reader: {e}")))?;

        match &mut col {
            ColumnReader::Int32ColumnReader(r) => {
                let mut values:     Vec<i32>  = Vec::new();
                let mut def_levels: Vec<i16>  = Vec::new();
                let (n, _, _) = r.read_records(rg_rows, Some(&mut def_levels), None, &mut values)
                    .map_err(|e| DataError::ParseError(format!("Label read: {e}")))?;
                label_data.extend(values.iter().take(n).map(|&l| l as u8));
            }
            ColumnReader::Int64ColumnReader(r) => {
                let mut values:     Vec<i64>  = Vec::new();
                let mut def_levels: Vec<i16>  = Vec::new();
                let (n, _, _) = r.read_records(rg_rows, Some(&mut def_levels), None, &mut values)
                    .map_err(|e| DataError::ParseError(format!("Label read: {e}")))?;
                label_data.extend(values.iter().take(n).map(|&l| l as u8));
            }
            _ => return Err(DataError::ParseError(
                "Label column is not INT32 or INT64.".to_string())),
        }
    }

    if label_data.is_empty() { return Err(DataError::EmptyDataset); }

    let n        = label_data.len();
    let n_pixels = inp_data.len() / n;
    let input_shift = if normalize { 7 } else { 0 };
    let tgt_data = build_targets(&label_data, 10);

    eprintln!("  [Parquet] loaded {n} samples in {}ms", t0.elapsed().as_millis());

    Ok(Dataset::<i32> {
        inputs:      Tensor::from_vec(inp_data,  vec![n, n_pixels]),
        labels:      label_data,
        targets:     Tensor::from_vec(tgt_data,  vec![n, 10]),
        n_classes:   10,
        input_shift,
    })
}

// ─── Format auto-detection ────────────────────────────────────────────────────

#[derive(Debug)]
pub enum MnistFormat {
    Idx {
        train_images: std::path::PathBuf,
        train_labels: std::path::PathBuf,
        test_images:  std::path::PathBuf,
        test_labels:  std::path::PathBuf,
    },
    #[cfg(feature = "parquet-support")]
    Parquet {
        train: std::path::PathBuf,
        test:  std::path::PathBuf,
    },
}

/// Probe `dir` for MNIST data files. IDX is always preferred over Parquet.
pub fn probe_format(dir: &Path) -> Option<MnistFormat> {
    let ti = dir.join("train-images-idx3-ubyte");
    let tl = dir.join("train-labels-idx1-ubyte");
    let ei = dir.join("t10k-images-idx3-ubyte");
    let el = dir.join("t10k-labels-idx1-ubyte");
    if ti.exists() && tl.exists() && ei.exists() && el.exists() {
        return Some(MnistFormat::Idx {
            train_images: ti, train_labels: tl,
            test_images:  ei, test_labels:  el,
        });
    }

    #[cfg(feature = "parquet-support")]
    {
        for (train_name, test_name) in &[
            ("mnist_train.parquet", "mnist_test.parquet"),
            ("train.parquet", "test.parquet"),
            ("train-00000-of-00001.parquet", "test-00000-of-00001.parquet"),
        ] {
            let tp = dir.join(train_name);
            let ep = dir.join(test_name);
            if tp.exists() && ep.exists() {
                return Some(MnistFormat::Parquet { train: tp, test: ep });
            }
        }
    }

    None
}

/// Load train + test splits from `dir`, auto-detecting IDX or Parquet format.
pub fn load_mnist_auto(
    dir:       &Path,
    normalize: bool,
) -> DataResult<(Dataset<i32>, Dataset<i32>)> {
    match probe_format(dir) {
        Some(MnistFormat::Idx { train_images, train_labels, test_images, test_labels }) => {
            println!("  Format: IDX binary");
            let train = load_mnist_idx(&train_images, &train_labels, normalize)?;
            let test  = load_mnist_idx(&test_images,  &test_labels,  normalize)?;
            Ok((train, test))
        }

        #[cfg(feature = "parquet-support")]
        Some(MnistFormat::Parquet { train, test }) => {
            println!("  Format: Parquet");
            let train = load_mnist_parquet(&train, normalize)?;
            let test  = load_mnist_parquet(&test,  normalize)?;
            Ok((train, test))
        }

        None => Err(DataError::ParseError(format!(
            "No MNIST data found in '{}'.\n\
             IDX: train-images-idx3-ubyte, train-labels-idx1-ubyte, \
                  t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte\n\
             Parquet (--features parquet-support): train.parquet, test.parquet",
            dir.display()
        ))),
    }
}
