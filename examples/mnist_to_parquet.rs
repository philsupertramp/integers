/// Helper script to convert MNIST IDX binary format to Parquet
/// 
/// Run this once to convert your MNIST data:
/// 
/// ```bash
/// cargo run --example mnist_convert_to_parquet -- data/mnist data/
/// ```

use std::fs::File;
use std::io::{self, Read, BufReader};
use std::path::Path;

use arrow::array::{UInt8Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;

fn read_u32_be(r: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

fn load_mnist_images_raw(path: &Path) -> io::Result<(Vec<u8>, usize, usize, usize)> {
    let mut f = BufReader::new(File::open(path)?);

    let magic = read_u32_be(&mut f)?;
    if magic != 0x0000_0803 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Invalid magic: {:#010x}", magic),
        ));
    }
    let n = read_u32_be(&mut f)? as usize;
    let rows = read_u32_be(&mut f)? as usize;
    let cols = read_u32_be(&mut f)? as usize;

    let mut pixels = vec![0u8; n * rows * cols];
    f.read_exact(&mut pixels)?;

    Ok((pixels, n, rows, cols))
}

fn load_mnist_labels_raw(path: &Path) -> io::Result<Vec<u8>> {
    let mut f = BufReader::new(File::open(path)?);

    let magic = read_u32_be(&mut f)?;
    if magic != 0x0000_0801 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Invalid magic: {:#010x}", magic),
        ));
    }
    let n = read_u32_be(&mut f)? as usize;

    let mut labels = vec![0u8; n];
    f.read_exact(&mut labels)?;

    Ok(labels)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <mnist_dir> <output_dir>", args[0]);
        eprintln!("Example: {} data/mnist data/", args[0]);
        std::process::exit(1);
    }

    let mnist_dir = &args[1];
    let output_dir = &args[2];

    std::fs::create_dir_all(output_dir)?;

    // Convert training set
    println!("Converting MNIST train set...");
    convert_split(
        &format!("{}/train-images-idx3-ubyte", mnist_dir),
        &format!("{}/train-labels-idx1-ubyte", mnist_dir),
        &format!("{}/mnist_train.parquet", output_dir),
    )?;

    // Convert test set
    println!("Converting MNIST test set...");
    convert_split(
        &format!("{}/t10k-images-idx3-ubyte", mnist_dir),
        &format!("{}/t10k-labels-idx1-ubyte", mnist_dir),
        &format!("{}/mnist_test.parquet", output_dir),
    )?;

    println!("✓ Done! Parquet files created in {}", output_dir);
    Ok(())
}

fn convert_split(
    img_file: &str,
    lbl_file: &str,
    out_file: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let (pixels, n_images, rows, cols) = load_mnist_images_raw(Path::new(img_file))?;
    let labels = load_mnist_labels_raw(Path::new(lbl_file))?;

    if n_images != labels.len() {
        return Err("Image count mismatch".into());
    }

    let n_features = rows * cols; // 784 for 28x28

    // Build columns: 784 pixel features + 1 label
    let mut pixel_columns: Vec<UInt8Array> = Vec::new();
    
    // For each pixel position, extract all values across samples
    for pixel_idx in 0..n_features {
        let mut values = Vec::new();
        for sample_idx in 0..n_images {
            let pixel_val = pixels[sample_idx * n_features + pixel_idx];
            values.push(Some(pixel_val));
        }
        pixel_columns.push(UInt8Array::from(values));
    }

    // Label column
    let label_column = UInt8Array::from(
        labels.iter().map(|&l| Some(l)).collect::<Vec<_>>()
    );

    // Build schema
    let mut fields = Vec::new();
    for i in 0..n_features {
        fields.push(Field::new(format!("pixel_{}", i), DataType::UInt8, false));
    }
    fields.push(Field::new("label", DataType::UInt8, false));

    let schema = Schema::new(fields);

    // Build record batch
    let mut columns: Vec<std::sync::Arc<dyn arrow::array::Array>> = pixel_columns
        .into_iter()
        .map(|a| std::sync::Arc::new(a) as std::sync::Arc<dyn arrow::array::Array>)
        .collect();
    columns.push(std::sync::Arc::new(label_column));

    let batch = RecordBatch::try_new(std::sync::Arc::new(schema), columns)?;

    // Write parquet
    let file = File::create(out_file)?;
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None)?;
    writer.write(&batch)?;
    writer.close()?;

    println!("  ✓ {} ({} samples)", out_file, n_images);

    Ok(())
}
