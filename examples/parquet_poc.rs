//! Parquet Proof-of-Concept for Index Storage
//!
//! This example benchmarks Parquet vs the current custom format for both
//! inverted index and main index storage.
//!
//! Usage:
//!   cargo run --release --features parquet --example parquet_poc -- <index_path>
//!
//! Examples:
//!   # Test inverted index shard
//!   cargo run --release --features parquet --example parquet_poc -- \
//!     perf-data/n100-w200-fixed.ryxdi.shard.0
//!
//!   # Test main index
//!   cargo run --release --features parquet --example parquet_poc -- \
//!     perf-data/n100-w200.ryidx

use anyhow::{Context, Result};
use arrow::array::{ArrayRef, ListArray, UInt32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, Encoding};
use parquet::file::properties::{WriterProperties, WriterVersion};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::schema::types::ColumnPath;
use rayon::prelude::*;
use rype::InvertedIndex;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

const ROW_GROUP_SIZE: usize = 100_000;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <shard_path>", args[0]);
        eprintln!(
            "Example: {} perf-data/n100-w200-fixed.ryxdi.shard.0",
            args[0]
        );
        std::process::exit(1);
    }

    let shard_path = Path::new(&args[1]);
    println!("=== Parquet PoC for Inverted Index ===\n");

    // 1. Load current format (baseline)
    println!("1. Loading current format: {}", shard_path.display());
    let t_load = Instant::now();
    let index = InvertedIndex::load_shard(shard_path)
        .with_context(|| format!("Failed to load shard: {}", shard_path.display()))?;
    let load_time_ms = t_load.elapsed().as_millis();
    println!(
        "   Loaded: {} minimizers, {} bucket_ids in {}ms",
        index.minimizers().len(),
        index.bucket_ids().len(),
        load_time_ms
    );

    let current_file_size = std::fs::metadata(shard_path)?.len();
    println!(
        "   File size: {:.2} MB",
        current_file_size as f64 / 1_000_000.0
    );

    // 2. Write nested schema Parquet
    let nested_path = shard_path.with_extension("nested.parquet");
    println!(
        "\n2. Writing nested schema Parquet: {}",
        nested_path.display()
    );
    let t_write_nested = Instant::now();
    write_nested_parquet(&index, &nested_path)?;
    let write_nested_ms = t_write_nested.elapsed().as_millis();
    let nested_size = std::fs::metadata(&nested_path)?.len();
    println!(
        "   Written in {}ms, size: {:.2} MB ({:.1}% of original)",
        write_nested_ms,
        nested_size as f64 / 1_000_000.0,
        nested_size as f64 / current_file_size as f64 * 100.0
    );

    // 3. Write flattened schema Parquet
    let flat_path = shard_path.with_extension("flat.parquet");
    println!(
        "\n3. Writing flattened schema Parquet: {}",
        flat_path.display()
    );
    let t_write_flat = Instant::now();
    write_flat_parquet(&index, &flat_path)?;
    let write_flat_ms = t_write_flat.elapsed().as_millis();
    let flat_size = std::fs::metadata(&flat_path)?.len();
    println!(
        "   Written in {}ms, size: {:.2} MB ({:.1}% of original)",
        write_flat_ms,
        flat_size as f64 / 1_000_000.0,
        flat_size as f64 / current_file_size as f64 * 100.0
    );

    // 4. Read nested Parquet (sequential)
    println!("\n4. Reading nested Parquet (sequential)");
    let t_read_nested_seq = Instant::now();
    let (mins_nested, bids_nested) = read_nested_parquet_sequential(&nested_path)?;
    let read_nested_seq_ms = t_read_nested_seq.elapsed().as_millis();
    println!(
        "   Read {} minimizers, {} bucket_ids in {}ms",
        mins_nested.len(),
        bids_nested.len(),
        read_nested_seq_ms
    );

    // 5. Read nested Parquet (parallel row groups)
    println!("\n5. Reading nested Parquet (parallel row groups)");
    let t_read_nested_par = Instant::now();
    let (mins_nested_par, bids_nested_par) = read_nested_parquet_parallel(&nested_path)?;
    let read_nested_par_ms = t_read_nested_par.elapsed().as_millis();
    println!(
        "   Read {} minimizers, {} bucket_ids in {}ms",
        mins_nested_par.len(),
        bids_nested_par.len(),
        read_nested_par_ms
    );

    // 6. Read flat Parquet (sequential)
    println!("\n6. Reading flat Parquet (sequential)");
    let t_read_flat_seq = Instant::now();
    let (mins_flat, bids_flat) = read_flat_parquet_sequential(&flat_path)?;
    let read_flat_seq_ms = t_read_flat_seq.elapsed().as_millis();
    println!("   Read {} rows in {}ms", mins_flat.len(), read_flat_seq_ms);

    // 7. Read flat Parquet (parallel row groups)
    println!("\n7. Reading flat Parquet (parallel row groups)");
    let t_read_flat_par = Instant::now();
    let (mins_flat_par, bids_flat_par) = read_flat_parquet_parallel(&flat_path)?;
    let read_flat_par_ms = t_read_flat_par.elapsed().as_millis();
    println!(
        "   Read {} rows in {}ms",
        mins_flat_par.len(),
        read_flat_par_ms
    );

    // 8. Inspect Parquet metadata (encodings, compression)
    println!("\n8. Parquet file metadata (flat schema):");
    inspect_parquet_metadata(&flat_path)?;

    // 9. Verify correctness
    println!("\n9. Verifying correctness...");
    verify_nested(&index, &mins_nested, &bids_nested)?;
    verify_nested(&index, &mins_nested_par, &bids_nested_par)?;
    verify_flat(&index, &mins_flat, &bids_flat)?;
    verify_flat(&index, &mins_flat_par, &bids_flat_par)?;
    println!("   All verifications passed!");

    // Summary
    println!("\n=== Inverted Index Summary ===");
    println!("| Metric                  | Current | Nested  | Flat    |");
    println!("|-------------------------|---------|---------|---------|");
    println!(
        "| File size (MB)          | {:7.2} | {:7.2} | {:7.2} |",
        current_file_size as f64 / 1_000_000.0,
        nested_size as f64 / 1_000_000.0,
        flat_size as f64 / 1_000_000.0
    );
    println!(
        "| Write time (ms)         |     N/A | {:7} | {:7} |",
        write_nested_ms, write_flat_ms
    );
    println!(
        "| Read time seq (ms)      | {:7} | {:7} | {:7} |",
        load_time_ms, read_nested_seq_ms, read_flat_seq_ms
    );
    println!(
        "| Read time parallel (ms) |     N/A | {:7} | {:7} |",
        read_nested_par_ms, read_flat_par_ms
    );

    // Cleanup
    std::fs::remove_file(&nested_path).ok();
    std::fs::remove_file(&flat_path).ok();

    Ok(())
}

/// Write inverted index as nested Parquet: (minimizer: u64, bucket_ids: list<u32>)
fn write_nested_parquet(index: &InvertedIndex, path: &Path) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("minimizer", DataType::UInt64, false),
        Field::new(
            "bucket_ids",
            DataType::List(Arc::new(Field::new("item", DataType::UInt32, false))),
            false,
        ),
    ]));

    let props = WriterProperties::builder()
        .set_writer_version(WriterVersion::PARQUET_2_0)
        .set_compression(Compression::ZSTD(Default::default()))
        .set_column_encoding(ColumnPath::from("minimizer"), Encoding::DELTA_BINARY_PACKED)
        .set_column_dictionary_enabled(ColumnPath::from("minimizer"), false)
        .set_max_row_group_size(ROW_GROUP_SIZE)
        .build();

    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

    let minimizers = index.minimizers();
    let offsets = index.offsets();
    let bucket_ids = index.bucket_ids();

    // Write in row groups (nested schema)
    for chunk_start in (0..minimizers.len()).step_by(ROW_GROUP_SIZE) {
        let chunk_end = (chunk_start + ROW_GROUP_SIZE).min(minimizers.len());
        let chunk_mins = &minimizers[chunk_start..chunk_end];

        // Build list array for this chunk
        let mut list_offsets: Vec<i32> = Vec::with_capacity(chunk_mins.len() + 1);
        list_offsets.push(0);

        let bucket_start = offsets[chunk_start] as usize;
        let bucket_end = offsets[chunk_end] as usize;

        for i in chunk_start..chunk_end {
            let relative_end = (offsets[i + 1] - offsets[chunk_start]) as i32;
            list_offsets.push(relative_end);
        }

        let chunk_bucket_ids = &bucket_ids[bucket_start..bucket_end];

        let minimizer_array: ArrayRef = Arc::new(UInt64Array::from(chunk_mins.to_vec()));
        let bucket_values = UInt32Array::from(chunk_bucket_ids.to_vec());
        let list_array = ListArray::try_new(
            Arc::new(Field::new("item", DataType::UInt32, false)),
            arrow::buffer::OffsetBuffer::new(arrow::buffer::ScalarBuffer::from(list_offsets)),
            Arc::new(bucket_values),
            None,
        )?;

        let batch =
            RecordBatch::try_new(schema.clone(), vec![minimizer_array, Arc::new(list_array)])?;

        writer.write(&batch)?;
    }

    writer.close()?;
    Ok(())
}

/// Write inverted index as flat Parquet: (minimizer: u64, bucket_id: u32)
fn write_flat_parquet(index: &InvertedIndex, path: &Path) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("minimizer", DataType::UInt64, false),
        Field::new("bucket_id", DataType::UInt32, false),
    ]));

    let props = WriterProperties::builder()
        .set_writer_version(WriterVersion::PARQUET_2_0)
        .set_compression(Compression::ZSTD(Default::default()))
        .set_column_encoding(ColumnPath::from("minimizer"), Encoding::DELTA_BINARY_PACKED)
        .set_column_dictionary_enabled(ColumnPath::from("minimizer"), false)
        .set_column_encoding(ColumnPath::from("bucket_id"), Encoding::DELTA_BINARY_PACKED)
        .set_column_dictionary_enabled(ColumnPath::from("bucket_id"), false)
        .set_max_row_group_size(ROW_GROUP_SIZE)
        .build();

    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

    let minimizers = index.minimizers();
    let offsets = index.offsets();
    let bucket_ids = index.bucket_ids();

    // Pre-expand to flat format
    let mut flat_mins = Vec::with_capacity(bucket_ids.len());
    for (i, &min) in minimizers.iter().enumerate() {
        let count = (offsets[i + 1] - offsets[i]) as usize;
        for _ in 0..count {
            flat_mins.push(min);
        }
    }

    // Write in row group chunks
    for chunk_start in (0..bucket_ids.len()).step_by(ROW_GROUP_SIZE) {
        let chunk_end = (chunk_start + ROW_GROUP_SIZE).min(bucket_ids.len());

        let chunk_mins = &flat_mins[chunk_start..chunk_end];
        let chunk_bids = &bucket_ids[chunk_start..chunk_end];

        let minimizer_array: ArrayRef = Arc::new(UInt64Array::from(chunk_mins.to_vec()));
        let bucket_id_array: ArrayRef = Arc::new(UInt32Array::from(chunk_bids.to_vec()));

        let batch = RecordBatch::try_new(schema.clone(), vec![minimizer_array, bucket_id_array])?;
        writer.write(&batch)?;
    }

    writer.close()?;
    Ok(())
}

/// Read nested Parquet sequentially
fn read_nested_parquet_sequential(path: &Path) -> Result<(Vec<u64>, Vec<u32>)> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut all_minimizers = Vec::new();
    let mut all_bucket_ids = Vec::new();

    for batch in reader {
        let batch = batch?;
        let minimizers = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let bucket_lists = batch
            .column(1)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();

        for i in 0..minimizers.len() {
            all_minimizers.push(minimizers.value(i));
            let list = bucket_lists.value(i);
            let bucket_arr = list.as_any().downcast_ref::<UInt32Array>().unwrap();
            for j in 0..bucket_arr.len() {
                all_bucket_ids.push(bucket_arr.value(j));
            }
        }
    }

    Ok((all_minimizers, all_bucket_ids))
}

/// Read nested Parquet with parallel row groups
fn read_nested_parquet_parallel(path: &Path) -> Result<(Vec<u64>, Vec<u32>)> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let metadata = builder.metadata().clone();
    let num_row_groups = metadata.num_row_groups();

    // Read row groups in parallel
    let results: Vec<(Vec<u64>, Vec<u32>)> = (0..num_row_groups)
        .into_par_iter()
        .map(|rg_idx| {
            let file = File::open(path).unwrap();
            let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
            let reader = builder.with_row_groups(vec![rg_idx]).build().unwrap();

            let mut minimizers = Vec::new();
            let mut bucket_ids = Vec::new();

            for batch in reader {
                let batch = batch.unwrap();
                let mins = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap();
                let lists = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .unwrap();

                for i in 0..mins.len() {
                    minimizers.push(mins.value(i));
                    let list = lists.value(i);
                    let arr = list.as_any().downcast_ref::<UInt32Array>().unwrap();
                    for j in 0..arr.len() {
                        bucket_ids.push(arr.value(j));
                    }
                }
            }

            (minimizers, bucket_ids)
        })
        .collect();

    // Merge results (in order)
    let mut all_minimizers = Vec::new();
    let mut all_bucket_ids = Vec::new();
    for (mins, bids) in results {
        all_minimizers.extend(mins);
        all_bucket_ids.extend(bids);
    }

    Ok((all_minimizers, all_bucket_ids))
}

/// Read flat Parquet sequentially
fn read_flat_parquet_sequential(path: &Path) -> Result<(Vec<u64>, Vec<u32>)> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut all_minimizers = Vec::new();
    let mut all_bucket_ids = Vec::new();

    for batch in reader {
        let batch = batch?;
        let minimizers = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let bucket_ids = batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();

        all_minimizers.extend(minimizers.values().iter().copied());
        all_bucket_ids.extend(bucket_ids.values().iter().copied());
    }

    Ok((all_minimizers, all_bucket_ids))
}

/// Read flat Parquet with parallel row groups
fn read_flat_parquet_parallel(path: &Path) -> Result<(Vec<u64>, Vec<u32>)> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let metadata = builder.metadata().clone();
    let num_row_groups = metadata.num_row_groups();

    let results: Vec<(Vec<u64>, Vec<u32>)> = (0..num_row_groups)
        .into_par_iter()
        .map(|rg_idx| {
            let file = File::open(path).unwrap();
            let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
            let reader = builder.with_row_groups(vec![rg_idx]).build().unwrap();

            let mut minimizers = Vec::new();
            let mut bucket_ids = Vec::new();

            for batch in reader {
                let batch = batch.unwrap();
                let mins = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap();
                let bids = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .unwrap();

                minimizers.extend(mins.values().iter().copied());
                bucket_ids.extend(bids.values().iter().copied());
            }

            (minimizers, bucket_ids)
        })
        .collect();

    let mut all_minimizers = Vec::new();
    let mut all_bucket_ids = Vec::new();
    for (mins, bids) in results {
        all_minimizers.extend(mins);
        all_bucket_ids.extend(bids);
    }

    Ok((all_minimizers, all_bucket_ids))
}

/// Verify nested format matches original
fn verify_nested(index: &InvertedIndex, minimizers: &[u64], bucket_ids: &[u32]) -> Result<()> {
    assert_eq!(
        minimizers.len(),
        index.minimizers().len(),
        "Minimizer count mismatch"
    );
    assert_eq!(
        bucket_ids.len(),
        index.bucket_ids().len(),
        "Bucket ID count mismatch"
    );

    for (i, (&expected, &actual)) in index.minimizers().iter().zip(minimizers.iter()).enumerate() {
        assert_eq!(expected, actual, "Minimizer mismatch at index {}", i);
    }

    for (i, (&expected, &actual)) in index.bucket_ids().iter().zip(bucket_ids.iter()).enumerate() {
        assert_eq!(expected, actual, "Bucket ID mismatch at index {}", i);
    }

    Ok(())
}

/// Verify flat format matches original (expanded)
fn verify_flat(index: &InvertedIndex, minimizers: &[u64], bucket_ids: &[u32]) -> Result<()> {
    assert_eq!(
        bucket_ids.len(),
        index.bucket_ids().len(),
        "Bucket ID count mismatch"
    );

    // Expand original to flat format for comparison
    let mut expected_mins = Vec::with_capacity(index.bucket_ids().len());
    for (i, &min) in index.minimizers().iter().enumerate() {
        let count = (index.offsets()[i + 1] - index.offsets()[i]) as usize;
        for _ in 0..count {
            expected_mins.push(min);
        }
    }

    assert_eq!(
        minimizers.len(),
        expected_mins.len(),
        "Flat minimizer count mismatch"
    );

    for (i, (&expected, &actual)) in expected_mins.iter().zip(minimizers.iter()).enumerate() {
        assert_eq!(expected, actual, "Flat minimizer mismatch at index {}", i);
    }

    for (i, (&expected, &actual)) in index.bucket_ids().iter().zip(bucket_ids.iter()).enumerate() {
        assert_eq!(expected, actual, "Flat bucket ID mismatch at index {}", i);
    }

    Ok(())
}

/// Inspect Parquet file metadata to show encodings and compression
fn inspect_parquet_metadata(path: &Path) -> Result<()> {
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file)?;
    let metadata = reader.metadata();
    let file_metadata = metadata.file_metadata();

    println!("   Schema: {:?}", file_metadata.schema_descr().name());
    println!("   Num row groups: {}", metadata.num_row_groups());
    println!("   Total rows: {}", file_metadata.num_rows());

    // Inspect first row group for column details
    if metadata.num_row_groups() > 0 {
        let rg = metadata.row_group(0);
        println!("   Row group 0 details:");
        for col_idx in 0..rg.num_columns() {
            let col = rg.column(col_idx);
            let col_path = col.column_path().string();
            let encodings: Vec<_> = col.encodings().collect();
            println!("     Column '{}': compression={:?}, encodings={:?}, total_compressed={}, total_uncompressed={}",
                col_path,
                col.compression(),
                encodings,
                col.compressed_size(),
                col.uncompressed_size(),
            );
        }
    }

    Ok(())
}
