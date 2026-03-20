//! CIFAR-10 loader for HuggingFace-style Parquet files.
//!
//! # Schema handling
//!
//! Parquet column readers index into **leaf columns** — the primitive fields
//! at the bottom of the type tree.  Top-level `schema().get_fields()` includes
//! group (struct) nodes which have no physical type and panic if you call
//! `get_physical_type()` on them.  This loader uses `schema_descr().columns()`
//! throughout, which gives only the leaf columns with their full dotted path.
//!
//! HuggingFace `uoft-cs-toronto/cifar10` has this leaf layout:
//!
//! ```text
//! leaf 0: path="img.bytes"    physical=BYTE_ARRAY
//! leaf 1: path="label"        physical=INT32
//! ```
//!
//! The loader matches by path component (any segment containing "img"/"image"
//! or "bytes" on a BYTE_ARRAY leaf, and "label" for the int column).
//!
//! Requires features: `parquet-support` and `image-decode`.

use std::fs::File;
use std::io::Cursor;
use std::path::Path;
use std::time::Instant;

use parquet::column::reader::ColumnReader;
use parquet::file::reader::{FileReader, SerializedFileReader};

use integers::data::{DataError, DataResult, Dataset};
use integers::tensor::Tensor;

// ─── Schema detection ─────────────────────────────────────────────────────────

/// Detected column layout.
#[derive(Debug, Clone)]
pub enum CifarSchema {
    /// BYTE_ARRAY image leaf + INT label leaf (HuggingFace format).
    ByteArrayImage { image_leaf: usize, label_leaf: usize },
    /// Flat raw pixel INT columns + INT label leaf.
    FlatPixels { first_leaf: usize, n_pixels: usize, label_leaf: usize },
}

/// Inspect the parquet **leaf columns** and return the best-matching schema.
///
/// Always prints the full leaf column list to stderr so mismatches are easy
/// to diagnose without guessing.
pub fn detect_schema(path: &Path) -> DataResult<CifarSchema> {
    let reader  = open_reader(path)?;
    let meta    = reader.metadata();
    let descr   = meta.file_metadata().schema_descr();
    let columns = descr.columns();

    eprintln!("  Leaf columns ({}):", columns.len());
    for (i, col) in columns.iter().enumerate() {
        eprintln!("    [{i:>2}]  {:?}  path=\"{}\"",
            col.self_type().get_physical_type(),
            col.path());
    }

    // Match by dotted path. Examples:
    //   HuggingFace:  "img.bytes" → BYTE_ARRAY,  "label" → INT32
    //   Flat export:  "pixel_0" … "pixel_3071",  "label"

    let mut image_leaf  = None;
    let mut label_leaf  = None;
    let mut pixel_leaves: Vec<usize> = Vec::new();

    let label_keywords = ["\"label\"", "\"labels\"", "\"class\"", "\"target\""];
    let image_keywords = ["\"img\"", "\"image\"", "\"bytes\"", "\"img.bytes\""];  // matches img.bytes or image
    let pixel_prefix   = "pixel_";

    for (i, col) in columns.iter().enumerate() {
        // Use the leaf name (last path component) for matching.
        let path_str  = col.path().to_string().to_lowercase();
        let leaf_name = path_str.split('.').last().unwrap_or(&path_str);

        if label_keywords.iter().any(|&k| leaf_name == k) {
            label_leaf = Some(i);
        } else if image_keywords.iter().any(|&k| path_str.contains(k)) {
            image_leaf = Some(i);
        } else if leaf_name.starts_with(pixel_prefix) {
            pixel_leaves.push(i);
        }
    }

    let label_leaf = label_leaf.ok_or_else(|| DataError::ParseError(format!(
        "No label leaf found (tried {label_keywords:?}). \
         Leaf paths: {:?}", columns.iter().map(|c| c.path().to_string()).collect::<Vec<_>>())))?;

    if let Some(il) = image_leaf {
        eprintln!("  → ByteArrayImage  (image leaf {il}, label leaf {label_leaf})");
        return Ok(CifarSchema::ByteArrayImage { image_leaf: il, label_leaf });
    }

    if !pixel_leaves.is_empty() {
        pixel_leaves.sort_unstable();
        let first = pixel_leaves[0];
        let n     = pixel_leaves.len();
        eprintln!("  → FlatPixels  ({n} pixel leaves from {first}, label leaf {label_leaf})");
        return Ok(CifarSchema::FlatPixels { first_leaf: first, n_pixels: n, label_leaf });
    }

    Err(DataError::ParseError(format!(
        "Cannot identify CIFAR-10 schema from leaf columns:\n  {:?}",
        columns.iter().map(|c| c.path().to_string()).collect::<Vec<_>>()
    )))
}

// ─── Public entry point ───────────────────────────────────────────────────────

/// Load a CIFAR-10 split from a parquet file, auto-detecting the schema.
///
/// Returns a `Dataset<i32>` with:
/// - `inputs`  `[n, 3072]` — channel-first (R plane, G plane, B plane)
/// - `labels`  class indices `0..10`
/// - `targets` one-hot, hot bit = 127
/// - `input_shift = 7`
pub fn load_cifar10_parquet(path: &Path) -> DataResult<Dataset<i32>> {
    let schema = detect_schema(path)?;
    match schema {
        CifarSchema::ByteArrayImage { image_leaf, label_leaf } =>
            load_byte_array(path, image_leaf, label_leaf),
        CifarSchema::FlatPixels { first_leaf, n_pixels, label_leaf } =>
            load_flat(path, first_leaf, n_pixels, label_leaf),
    }
}

// ─── ByteArray (HuggingFace PNG) path ─────────────────────────────────────────

fn load_byte_array(path: &Path, img_leaf: usize, lbl_leaf: usize) -> DataResult<Dataset<i32>> {
    let t0     = Instant::now();
    let reader = open_reader(path)?;
    let meta   = reader.metadata();
    let n_rows = meta.file_metadata().num_rows() as usize;

    let mut inp_data   = Vec::<i32>::with_capacity(n_rows * 3072);
    let mut label_data = Vec::<u8>::with_capacity(n_rows);

    for rg_idx in 0..meta.num_row_groups() {
        let rg      = reader.get_row_group(rg_idx)
            .map_err(|e| DataError::ParseError(format!("Row group {rg_idx}: {e}")))?;
        let rg_rows = meta.row_group(rg_idx).num_rows() as usize;

        // ── Image leaf ────────────────────────────────────────────────────────
        let mut col = rg.get_column_reader(img_leaf)
            .map_err(|e| DataError::ParseError(format!("Image leaf {img_leaf}: {e}")))?;

        match &mut col {
            ColumnReader::ByteArrayColumnReader(r) => {
                let mut values:     Vec<parquet::data_type::ByteArray> = Vec::new();
                let mut def_levels: Vec<i16> = Vec::new();
                let (n, _, _) = r.read_records(rg_rows, Some(&mut def_levels), None, &mut values)
                    .map_err(|e| DataError::ParseError(format!("Image read: {e}")))?;
                for ba in values.iter().take(n) {
                    inp_data.extend_from_slice(&decode_png(ba.data())?);
                }
            }
            _ => return Err(DataError::ParseError(
                format!("Image leaf {img_leaf} is not BYTE_ARRAY."))),
        }

        // ── Label leaf ────────────────────────────────────────────────────────
        read_int_labels(&rg, lbl_leaf, rg_rows, &mut label_data)?;
    }

    finish(inp_data, label_data, t0, "ByteArray/PNG")
}

// ─── Flat pixel path ──────────────────────────────────────────────────────────

fn load_flat(
    path:       &Path,
    first_leaf: usize,
    n_pixels:   usize,
    lbl_leaf:   usize,
) -> DataResult<Dataset<i32>> {
    let t0     = Instant::now();
    let reader = open_reader(path)?;
    let meta   = reader.metadata();
    let n_rows = meta.file_metadata().num_rows() as usize;

    let mut channels: Vec<Vec<i32>> = Vec::with_capacity(n_pixels);

    for px_leaf in first_leaf..(first_leaf + n_pixels) {
        let mut col_data = Vec::<i32>::with_capacity(n_rows);

        for rg_idx in 0..meta.num_row_groups() {
            let rg      = reader.get_row_group(rg_idx)
                .map_err(|e| DataError::ParseError(format!("Row group {rg_idx}: {e}")))?;
            let rg_rows = meta.row_group(rg_idx).num_rows() as usize;

            let mut col = rg.get_column_reader(px_leaf)
                .map_err(|e| DataError::ParseError(format!("Pixel leaf {px_leaf}: {e}")))?;

            match &mut col {
                ColumnReader::Int32ColumnReader(r) => {
                    let mut values: Vec<i32>  = Vec::new();
                    let mut def:    Vec<i16>  = Vec::new();
                    let (n, _, _) = r.read_records(rg_rows, Some(&mut def), None, &mut values)
                        .map_err(|e| DataError::ParseError(format!("{e}")))?;
                    col_data.extend(values.iter().take(n).map(|&v| norm(v.clamp(0, 255) as u8)));
                }
                ColumnReader::Int64ColumnReader(r) => {
                    let mut values: Vec<i64>  = Vec::new();
                    let mut def:    Vec<i16>  = Vec::new();
                    let (n, _, _) = r.read_records(rg_rows, Some(&mut def), None, &mut values)
                        .map_err(|e| DataError::ParseError(format!("{e}")))?;
                    col_data.extend(values.iter().take(n).map(|&v| norm(v.clamp(0, 255) as u8)));
                }
                _ => return Err(DataError::ParseError(
                    format!("Pixel leaf {px_leaf} is not INT32/INT64."))),
            }
        }
        channels.push(col_data);
    }

    let mut label_data = Vec::<u8>::with_capacity(n_rows);
    for rg_idx in 0..meta.num_row_groups() {
        let rg      = reader.get_row_group(rg_idx)
            .map_err(|e| DataError::ParseError(format!("Row group {rg_idx}: {e}")))?;
        let rg_rows = meta.row_group(rg_idx).num_rows() as usize;
        read_int_labels(&rg, lbl_leaf, rg_rows, &mut label_data)?;
    }

    let n = label_data.len();
    let mut inp_data = vec![0i32; n * n_pixels];
    for (ci, col) in channels.iter().enumerate() {
        for (ri, &v) in col.iter().enumerate() {
            inp_data[ri * n_pixels + ci] = v;
        }
    }

    finish(inp_data, label_data, t0, "flat pixels")
}

// ─── Label helper ─────────────────────────────────────────────────────────────

fn read_int_labels(
    rg:      &Box<dyn parquet::file::reader::RowGroupReader + '_>,
    lbl_leaf: usize,
    rg_rows:  usize,
    out:      &mut Vec<u8>,
) -> DataResult<()> {
    let mut col = rg.get_column_reader(lbl_leaf)
        .map_err(|e| DataError::ParseError(format!("Label leaf {lbl_leaf}: {e}")))?;

    match &mut col {
        ColumnReader::Int32ColumnReader(r) => {
            let mut values: Vec<i32>  = Vec::new();
            let mut def:    Vec<i16>  = Vec::new();
            let (n, _, _) = r.read_records(rg_rows, Some(&mut def), None, &mut values)
                .map_err(|e| DataError::ParseError(format!("Label read: {e}")))?;
            out.extend(values.iter().take(n).map(|&l| l as u8));
        }
        ColumnReader::Int64ColumnReader(r) => {
            let mut values: Vec<i64>  = Vec::new();
            let mut def:    Vec<i16>  = Vec::new();
            let (n, _, _) = r.read_records(rg_rows, Some(&mut def), None, &mut values)
                .map_err(|e| DataError::ParseError(format!("Label read: {e}")))?;
            out.extend(values.iter().take(n).map(|&l| l as u8));
        }
        ColumnReader::ByteArrayColumnReader(r) => {
            // Some exports store label as string "3".
            let mut values: Vec<parquet::data_type::ByteArray> = Vec::new();
            let mut def:    Vec<i16> = Vec::new();
            let (n, _, _) = r.read_records(rg_rows, Some(&mut def), None, &mut values)
                .map_err(|e| DataError::ParseError(format!("Label read: {e}")))?;
            for ba in values.iter().take(n) {
                let s = std::str::from_utf8(ba.data())
                    .map_err(|_| DataError::ParseError("Non-UTF8 label".into()))?;
                let v = s.trim().parse::<u8>()
                    .map_err(|_| DataError::ParseError(format!("Cannot parse label '{s}'")))?;
                out.push(v);
            }
        }
        _ => return Err(DataError::ParseError(
            format!("Label leaf {lbl_leaf} is not INT32, INT64, or BYTE_ARRAY."))),
    }
    Ok(())
}

// ─── Pixel normalisation ──────────────────────────────────────────────────────

#[inline]
fn norm(p: u8) -> i32 {
    let scaled = (p as f32 / 255.0) * 254.0 - 127.0;
    scaled.round().clamp(-127.0, 127.0) as i32
}

/// Decode PNG bytes → channel-first `[R…, G…, B…]` normalised i32 (3072 values).
fn decode_png(bytes: &[u8]) -> DataResult<Vec<i32>> {
    #[cfg(feature = "image-decode")]
    {
        use image::ImageReader;
        let img = ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .map_err(|e| DataError::ParseError(format!("Image format: {e}")))?
            .decode()
            .map_err(|e| DataError::ParseError(format!("Image decode: {e}")))?
            .to_rgb8();

        if img.width() != 32 || img.height() != 32 {
            return Err(DataError::ParseError(format!(
                "Expected 32×32, got {}×{}", img.width(), img.height())));
        }

        // PNG is HWC [R,G,B, R,G,B, …] → convert to CHW.
        let raw  = img.as_raw();
        let npix = 1024usize;
        let mut out = vec![0i32; 3072];
        for i in 0..npix {
            out[i]              = norm(raw[i * 3]);
            out[npix + i]       = norm(raw[i * 3 + 1]);
            out[2 * npix + i]   = norm(raw[i * 3 + 2]);
        }
        Ok(out)
    }

    #[cfg(not(feature = "image-decode"))]
    {
        let _ = bytes;
        Err(DataError::ParseError(
            "PNG decoding requires the `image-decode` feature. \
             Add `--features image-decode` to your cargo command.".to_string()))
    }
}

// ─── Shared utilities ─────────────────────────────────────────────────────────

fn open_reader(path: &Path) -> DataResult<SerializedFileReader<File>> {
    let file = File::open(path)?;
    SerializedFileReader::new(file)
        .map_err(|e| DataError::ParseError(format!("Cannot open parquet: {e}")))
}

fn finish(
    inp_data:   Vec<i32>,
    label_data: Vec<u8>,
    t0:         Instant,
    kind:       &str,
) -> DataResult<Dataset<i32>> {
    if label_data.is_empty() { return Err(DataError::EmptyDataset); }

    let n         = label_data.len();
    let n_pixels  = inp_data.len() / n;
    let n_classes = 10;

    let mut tgt = vec![0i32; n * n_classes];
    for (i, &lbl) in label_data.iter().enumerate() {
        tgt[i * n_classes + lbl as usize] = 127;
    }

    eprintln!("  [CIFAR-10 {kind}] loaded {n} samples in {}ms",
        t0.elapsed().as_millis());

    Ok(Dataset::<i32> {
        inputs:      Tensor::from_vec(inp_data, vec![n, n_pixels]),
        labels:      label_data,
        targets:     Tensor::from_vec(tgt, vec![n, n_classes]),
        n_classes,
        input_shift: 7,
    })
}
