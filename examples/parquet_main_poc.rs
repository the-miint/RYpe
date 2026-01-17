//! Parquet Proof-of-Concept for Main Index Storage
//!
//! This example benchmarks Parquet vs the current custom format for main index storage.
//!
//! Usage:
//!   cargo run --release --features parquet --example parquet_main_poc -- <index_path>
//!
//! Example:
//!   cargo run --release --features parquet --example parquet_main_poc -- \
//!     perf-data/n100-w200.ryidx

use anyhow::{Context, Result};
use arrow::array::{
    Array, ArrayRef, ListArray, ListBuilder, StringArray, UInt32Array, UInt64Array, UInt64Builder,
};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, Encoding};
use parquet::file::properties::{WriterProperties, WriterVersion};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::schema::types::ColumnPath;
use rayon::prelude::*;
use rype::Index;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <index_path>", args[0]);
        eprintln!("Example: {} perf-data/n100-w200.ryidx", args[0]);
        std::process::exit(1);
    }

    let index_path = Path::new(&args[1]);
    println!("=== Parquet PoC for Main Index ===\n");

    // 1. Load current format (baseline)
    println!("1. Loading current format: {}", index_path.display());
    let t_load = Instant::now();
    let index = Index::load(index_path)
        .with_context(|| format!("Failed to load index: {}", index_path.display()))?;
    let load_time_ms = t_load.elapsed().as_millis();

    let total_minimizers: usize = index.buckets.values().map(|v| v.len()).sum();
    println!(
        "   Loaded: {} buckets, {} total minimizers in {}ms",
        index.buckets.len(),
        total_minimizers,
        load_time_ms
    );

    let current_file_size = std::fs::metadata(index_path)?.len();
    println!(
        "   File size: {:.2} MB",
        current_file_size as f64 / 1_000_000.0
    );

    // 2. Write simple flat Parquet (bucket_id, minimizer) - no bucket_name
    let simple_path = index_path.with_extension("simple.parquet");
    println!(
        "\n2. Writing simple flat Parquet (bucket_id, minimizer): {}",
        simple_path.display()
    );
    let t_write_simple = Instant::now();
    write_simple_flat_parquet(&index, &simple_path)?;
    let write_simple_ms = t_write_simple.elapsed().as_millis();
    let simple_size = std::fs::metadata(&simple_path)?.len();
    println!(
        "   Written in {}ms, size: {:.2} MB ({:.1}% of original)",
        write_simple_ms,
        simple_size as f64 / 1_000_000.0,
        simple_size as f64 / current_file_size as f64 * 100.0
    );

    // 3. Verify encodings are correct
    println!("\n3. Verifying Parquet encodings:");
    verify_parquet_encodings(
        &simple_path,
        &[
            ("bucket_id", "DELTA_BINARY_PACKED"),
            ("minimizer", "DELTA_BINARY_PACKED"),
        ],
    )?;

    // 4. Inspect full Parquet metadata
    println!("\n4. Simple flat Parquet file metadata:");
    inspect_parquet_metadata(&simple_path)?;

    // 5. Read simple flat Parquet (parallel)
    println!("\n5. Reading simple flat Parquet (parallel row groups)");
    let t_read_simple_par = Instant::now();
    let (bucket_ids_simple, mins_simple) = read_simple_flat_parquet_parallel(&simple_path)?;
    let read_simple_par_ms = t_read_simple_par.elapsed().as_millis();
    println!(
        "   Read {} rows in {}ms",
        mins_simple.len(),
        read_simple_par_ms
    );

    // 6. Verify simple flat format correctness
    println!("\n6. Verifying simple flat format correctness...");
    verify_flat_main(&index, &bucket_ids_simple, &mins_simple)?;
    println!("   Simple flat format verification passed!");

    // Store results for summary
    let flat_size = simple_size;
    let write_flat_ms = write_simple_ms;
    let read_flat_par_ms = read_simple_par_ms;

    // 7. Write nested schema Parquet (bucket_id, bucket_name, minimizers: list<u64>)
    let nested_path = index_path.with_extension("nested.parquet");
    println!(
        "\n7. Writing nested schema Parquet: {}",
        nested_path.display()
    );
    let t_write_nested = Instant::now();
    write_nested_main_parquet(&index, &nested_path)?;
    let write_nested_ms = t_write_nested.elapsed().as_millis();
    let nested_size = std::fs::metadata(&nested_path)?.len();
    println!(
        "   Written in {}ms, size: {:.2} MB ({:.1}% of original)",
        write_nested_ms,
        nested_size as f64 / 1_000_000.0,
        nested_size as f64 / current_file_size as f64 * 100.0
    );

    // 8. Inspect nested Parquet metadata
    println!("\n8. Nested Parquet file metadata:");
    inspect_parquet_metadata(&nested_path)?;

    // 9. Read nested Parquet
    println!("\n9. Reading nested Parquet");
    let t_read_nested = Instant::now();
    let nested_buckets = read_nested_main_parquet(&nested_path)?;
    let read_nested_ms = t_read_nested.elapsed().as_millis();
    let nested_total_mins: usize = nested_buckets.iter().map(|(_, _, m)| m.len()).sum();
    println!(
        "   Read {} buckets, {} total minimizers in {}ms",
        nested_buckets.len(),
        nested_total_mins,
        read_nested_ms
    );

    // 10. Verify nested format correctness
    println!("\n10. Verifying nested format correctness...");
    verify_nested_main(&index, &nested_buckets)?;
    println!("   Nested format verification passed!");

    // Summary
    println!("\n=== Main Index Summary ===");
    println!("| Metric                  | Current | Simple  | Nested  |");
    println!("|-------------------------|---------|---------|---------|");
    println!(
        "| File size (MB)          | {:7.2} | {:7.2} | {:7.2} |",
        current_file_size as f64 / 1_000_000.0,
        flat_size as f64 / 1_000_000.0,
        nested_size as f64 / 1_000_000.0
    );
    println!(
        "| Write time (ms)         |     N/A | {:7} | {:7} |",
        write_flat_ms, write_nested_ms
    );
    println!(
        "| Read time (ms)          | {:7} | {:7} | {:7} |",
        load_time_ms, read_flat_par_ms, read_nested_ms
    );

    // Cleanup
    std::fs::remove_file(&simple_path).ok();
    std::fs::remove_file(&nested_path).ok();

    Ok(())
}

/// Write main index as simple flat Parquet: (bucket_id: u32, minimizer: u64)
/// No bucket_name - can be stored separately. Sorted by bucket_id, then minimizer.
fn write_simple_flat_parquet(index: &Index, path: &Path) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("bucket_id", DataType::UInt32, false),
        Field::new("minimizer", DataType::UInt64, false),
    ]));

    let props = WriterProperties::builder()
        .set_writer_version(WriterVersion::PARQUET_2_0)
        .set_compression(Compression::ZSTD(Default::default()))
        // Delta encoding for bucket_id (sorted, so consecutive rows often have same bucket)
        .set_column_encoding(ColumnPath::from("bucket_id"), Encoding::DELTA_BINARY_PACKED)
        .set_column_dictionary_enabled(ColumnPath::from("bucket_id"), false)
        // Delta encoding for minimizer (sorted within each bucket)
        .set_column_encoding(ColumnPath::from("minimizer"), Encoding::DELTA_BINARY_PACKED)
        .set_column_dictionary_enabled(ColumnPath::from("minimizer"), false)
        .set_max_row_group_size(100_000)
        .build();

    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

    // Collect all (bucket_id, minimizer) tuples, sorted by bucket_id then minimizer
    let mut rows: Vec<(u32, u64)> = Vec::new();
    for (&bucket_id, minimizers) in &index.buckets {
        for &min in minimizers {
            rows.push((bucket_id, min));
        }
    }
    rows.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    // Write in chunks
    let chunk_size = 100_000;
    for chunk in rows.chunks(chunk_size) {
        let bucket_ids: Vec<u32> = chunk.iter().map(|&(bid, _)| bid).collect();
        let minimizers: Vec<u64> = chunk.iter().map(|&(_, min)| min).collect();

        let bucket_id_array: ArrayRef = Arc::new(UInt32Array::from(bucket_ids));
        let minimizer_array: ArrayRef = Arc::new(UInt64Array::from(minimizers));

        let batch = RecordBatch::try_new(schema.clone(), vec![bucket_id_array, minimizer_array])?;
        writer.write(&batch)?;
    }

    writer.close()?;
    Ok(())
}

/// Read simple flat Parquet (bucket_id, minimizer) in parallel
fn read_simple_flat_parquet_parallel(path: &Path) -> Result<(Vec<u32>, Vec<u64>)> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let metadata = builder.metadata().clone();
    let num_row_groups = metadata.num_row_groups();

    let results: Vec<(Vec<u32>, Vec<u64>)> = (0..num_row_groups)
        .into_par_iter()
        .map(|rg_idx| {
            let file = File::open(path).unwrap();
            let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
            let reader = builder.with_row_groups(vec![rg_idx]).build().unwrap();

            let mut bucket_ids = Vec::new();
            let mut minimizers = Vec::new();

            for batch in reader {
                let batch = batch.unwrap();
                let bids = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .unwrap();
                let mins = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap();
                bucket_ids.extend(bids.values().iter().copied());
                minimizers.extend(mins.values().iter().copied());
            }

            (bucket_ids, minimizers)
        })
        .collect();

    let mut all_bucket_ids = Vec::new();
    let mut all_minimizers = Vec::new();
    for (bids, mins) in results {
        all_bucket_ids.extend(bids);
        all_minimizers.extend(mins);
    }

    Ok((all_bucket_ids, all_minimizers))
}

/// Inspect and verify Parquet encodings match expectations
fn verify_parquet_encodings(path: &Path, expected_encodings: &[(&str, &str)]) -> Result<()> {
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file)?;
    let metadata = reader.metadata();

    if metadata.num_row_groups() == 0 {
        anyhow::bail!("No row groups in file");
    }

    let rg = metadata.row_group(0);
    println!("   Verifying encodings for {} columns:", rg.num_columns());

    for (col_name, expected_enc) in expected_encodings {
        let mut found = false;
        for col_idx in 0..rg.num_columns() {
            let col = rg.column(col_idx);
            if col.column_path().string() == *col_name {
                found = true;
                let encodings: Vec<_> = col.encodings().collect();
                let has_expected = encodings
                    .iter()
                    .any(|e| format!("{:?}", e).contains(expected_enc));
                let compression_ratio =
                    col.compressed_size() as f64 / col.uncompressed_size() as f64;

                println!(
                    "     '{}': encodings={:?}, compression={:?}, ratio={:.1}%, {}",
                    col_name,
                    encodings,
                    col.compression(),
                    compression_ratio * 100.0,
                    if has_expected {
                        "✓"
                    } else {
                        "✗ MISSING EXPECTED ENCODING"
                    }
                );

                if !has_expected {
                    println!(
                        "       WARNING: Expected '{}' encoding not found!",
                        expected_enc
                    );
                }
                break;
            }
        }
        if !found {
            println!("     '{}': NOT FOUND in metadata", col_name);
        }
    }

    Ok(())
}

/// Write main index as flat Parquet: (bucket_id: u32, bucket_name: str, minimizer: u64)
/// Sorted by bucket_id, then minimizer (matches current index structure)
fn write_flat_main_parquet(index: &Index, path: &Path) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("bucket_id", DataType::UInt32, false),
        Field::new("bucket_name", DataType::Utf8, false),
        Field::new("minimizer", DataType::UInt64, false),
    ]));

    let props = WriterProperties::builder()
        .set_writer_version(WriterVersion::PARQUET_2_0)
        .set_compression(Compression::ZSTD(Default::default()))
        // Dictionary encoding for bucket_id (high repetition - same ID for all mins in bucket)
        .set_column_dictionary_enabled(ColumnPath::from("bucket_id"), true)
        // Dictionary encoding for bucket_name (high repetition - same name for all mins in bucket)
        .set_column_dictionary_enabled(ColumnPath::from("bucket_name"), true)
        // Delta encoding for minimizer (sorted within each bucket)
        .set_column_encoding(ColumnPath::from("minimizer"), Encoding::DELTA_BINARY_PACKED)
        .set_column_dictionary_enabled(ColumnPath::from("minimizer"), false)
        .set_max_row_group_size(100_000)
        .build();

    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

    // Collect all (bucket_id, bucket_name, minimizer) tuples
    // Sort by bucket_id, then minimizer (matches current index structure)
    let mut rows: Vec<(u32, &str, u64)> = Vec::new();
    for (&bucket_id, minimizers) in &index.buckets {
        let bucket_name = index
            .bucket_names
            .get(&bucket_id)
            .map(|s| s.as_str())
            .unwrap_or("");
        for &min in minimizers {
            rows.push((bucket_id, bucket_name, min));
        }
    }
    // Sort by bucket_id, then minimizer
    rows.sort_by(|a, b| a.0.cmp(&b.0).then(a.2.cmp(&b.2)));

    // Write in chunks
    let chunk_size = 100_000;
    for chunk in rows.chunks(chunk_size) {
        let bucket_ids: Vec<u32> = chunk.iter().map(|&(bid, _, _)| bid).collect();
        let bucket_names: Vec<&str> = chunk.iter().map(|&(_, name, _)| name).collect();
        let minimizers: Vec<u64> = chunk.iter().map(|&(_, _, min)| min).collect();

        let bucket_id_array: ArrayRef = Arc::new(UInt32Array::from(bucket_ids));
        let bucket_name_array: ArrayRef = Arc::new(StringArray::from(bucket_names));
        let minimizer_array: ArrayRef = Arc::new(UInt64Array::from(minimizers));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![bucket_id_array, bucket_name_array, minimizer_array],
        )?;
        writer.write(&batch)?;
    }

    writer.close()?;
    Ok(())
}

/// Read flat main Parquet sequentially
fn read_flat_main_parquet_sequential(path: &Path) -> Result<(Vec<u32>, Vec<String>, Vec<u64>)> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut all_bucket_ids = Vec::new();
    let mut all_bucket_names = Vec::new();
    let mut all_minimizers = Vec::new();

    for batch in reader {
        let batch = batch?;
        let bucket_ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let bucket_names = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let minimizers = batch
            .column(2)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();

        all_bucket_ids.extend(bucket_ids.values().iter().copied());
        for i in 0..bucket_names.len() {
            all_bucket_names.push(bucket_names.value(i).to_string());
        }
        all_minimizers.extend(minimizers.values().iter().copied());
    }

    Ok((all_bucket_ids, all_bucket_names, all_minimizers))
}

/// Read flat main Parquet with parallel row groups
/// NOTE: Skips bucket_name column to avoid 11GB+ String allocation overhead
fn read_flat_main_parquet_parallel(path: &Path) -> Result<(Vec<u32>, Vec<u64>)> {
    use parquet::arrow::ProjectionMask;

    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let metadata = builder.metadata().clone();
    let num_row_groups = metadata.num_row_groups();
    let schema = builder.parquet_schema().clone();

    // Project only bucket_id (col 0) and minimizer (col 2), skip bucket_name (col 1)
    let projection = ProjectionMask::leaves(&schema, [0, 2]);

    let results: Vec<(Vec<u32>, Vec<u64>)> = (0..num_row_groups)
        .into_par_iter()
        .map(|rg_idx| {
            let file = File::open(path).unwrap();
            let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
            let schema = builder.parquet_schema().clone();
            let projection = ProjectionMask::leaves(&schema, [0, 2]);
            let reader = builder
                .with_projection(projection)
                .with_row_groups(vec![rg_idx])
                .build()
                .unwrap();

            let mut bucket_ids = Vec::new();
            let mut minimizers = Vec::new();

            for batch in reader {
                let batch = batch.unwrap();
                let bids = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .unwrap();
                let mins = batch
                    .column(1)  // After projection, minimizer is column 1
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap();

                bucket_ids.extend(bids.values().iter().copied());
                minimizers.extend(mins.values().iter().copied());
            }

            (bucket_ids, minimizers)
        })
        .collect();

    let mut all_bucket_ids = Vec::new();
    let mut all_minimizers = Vec::new();
    for (bids, mins) in results {
        all_bucket_ids.extend(bids);
        all_minimizers.extend(mins);
    }

    Ok((all_bucket_ids, all_minimizers))
}

/// Verify flat format matches original
fn verify_flat_main(index: &Index, bucket_ids: &[u32], minimizers: &[u64]) -> Result<()> {
    // Count total minimizers
    let expected_total: usize = index.buckets.values().map(|v| v.len()).sum();
    assert_eq!(
        minimizers.len(),
        expected_total,
        "Total minimizer count mismatch: got {}, expected {}",
        minimizers.len(),
        expected_total
    );
    assert_eq!(
        bucket_ids.len(),
        minimizers.len(),
        "Bucket ID count mismatch"
    );

    // Verify each bucket's minimizers are present (order may differ due to sorting)
    use std::collections::HashMap;
    let mut bucket_mins: HashMap<u32, Vec<u64>> = HashMap::new();
    for (&bid, &min) in bucket_ids.iter().zip(minimizers.iter()) {
        bucket_mins.entry(bid).or_default().push(min);
    }

    for (&bucket_id, expected_mins) in &index.buckets {
        let got_mins = bucket_mins
            .get(&bucket_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        assert_eq!(
            got_mins.len(),
            expected_mins.len(),
            "Minimizer count mismatch for bucket {}",
            bucket_id
        );

        // Sort both for comparison (Parquet data is sorted by minimizer globally)
        let mut expected_sorted = expected_mins.clone();
        expected_sorted.sort();
        let mut got_sorted = got_mins.to_vec();
        got_sorted.sort();
        assert_eq!(
            expected_sorted, got_sorted,
            "Minimizers mismatch for bucket {}",
            bucket_id
        );
    }

    Ok(())
}

/// Inspect Parquet file metadata
fn inspect_parquet_metadata(path: &Path) -> Result<()> {
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file)?;
    let metadata = reader.metadata();
    let file_metadata = metadata.file_metadata();

    println!("   Num row groups: {}", metadata.num_row_groups());
    println!("   Total rows: {}", file_metadata.num_rows());

    if metadata.num_row_groups() > 0 {
        let rg = metadata.row_group(0);
        println!("   Row group 0 details:");
        for col_idx in 0..rg.num_columns() {
            let col = rg.column(col_idx);
            let col_path = col.column_path().string();
            let encodings: Vec<_> = col.encodings().collect();
            println!(
                "     Column '{}': compression={:?}, encodings={:?}, compressed={}, uncompressed={}",
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

/// Write main index as nested Parquet: (bucket_id: u32, bucket_name: str, minimizers: list<u64>)
/// One row per bucket (matches current index structure)
fn write_nested_main_parquet(index: &Index, path: &Path) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("bucket_id", DataType::UInt32, false),
        Field::new("bucket_name", DataType::Utf8, false),
        Field::new(
            "minimizers",
            DataType::List(Arc::new(Field::new("item", DataType::UInt64, true))), // nullable inner field to match ListBuilder
            false,
        ),
    ]));

    let props = WriterProperties::builder()
        .set_writer_version(WriterVersion::PARQUET_2_0)
        .set_compression(Compression::ZSTD(Default::default()))
        // Delta encoding for minimizers (sorted within each bucket)
        // Column path for nested list is "minimizers.list.item" not "minimizers.item"
        .set_column_encoding(ColumnPath::from("minimizers.list.item"), Encoding::DELTA_BINARY_PACKED)
        .set_column_dictionary_enabled(ColumnPath::from("minimizers.list.item"), false)
        .build();

    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

    // Collect buckets sorted by bucket_id
    let mut buckets: Vec<(u32, &str, &[u64])> = index
        .buckets
        .iter()
        .map(|(&id, mins)| {
            let name = index
                .bucket_names
                .get(&id)
                .map(|s| s.as_str())
                .unwrap_or("");
            (id, name, mins.as_slice())
        })
        .collect();
    buckets.sort_by_key(|(id, _, _)| *id);

    // Build arrays
    let bucket_ids: Vec<u32> = buckets.iter().map(|(id, _, _)| *id).collect();
    let bucket_names: Vec<&str> = buckets.iter().map(|(_, name, _)| *name).collect();

    let bucket_id_array: ArrayRef = Arc::new(UInt32Array::from(bucket_ids));
    let bucket_name_array: ArrayRef = Arc::new(StringArray::from(bucket_names));

    // Build list array for minimizers
    let mut list_builder = ListBuilder::new(UInt64Builder::new());
    for (_, _, mins) in &buckets {
        let values_builder = list_builder.values();
        for &m in *mins {
            values_builder.append_value(m);
        }
        list_builder.append(true);
    }
    let minimizers_array: ArrayRef = Arc::new(list_builder.finish());

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![bucket_id_array, bucket_name_array, minimizers_array],
    )?;
    writer.write(&batch)?;

    writer.close()?;
    Ok(())
}

/// Read nested main Parquet
fn read_nested_main_parquet(path: &Path) -> Result<Vec<(u32, String, Vec<u64>)>> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut result = Vec::new();

    for batch in reader {
        let batch = batch?;
        let bucket_ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let bucket_names = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let minimizers_list = batch
            .column(2)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();

        for i in 0..batch.num_rows() {
            let bid = bucket_ids.value(i);
            let name = bucket_names.value(i).to_string();
            let mins_array = minimizers_list.value(i);
            let mins = mins_array.as_any().downcast_ref::<UInt64Array>().unwrap();
            let mins_vec: Vec<u64> = mins.values().iter().copied().collect();
            result.push((bid, name, mins_vec));
        }
    }

    Ok(result)
}

/// Verify nested format matches original
fn verify_nested_main(index: &Index, buckets: &[(u32, String, Vec<u64>)]) -> Result<()> {
    assert_eq!(
        buckets.len(),
        index.buckets.len(),
        "Bucket count mismatch: got {}, expected {}",
        buckets.len(),
        index.buckets.len()
    );

    for (bid, _name, mins) in buckets {
        let expected = index.buckets.get(bid).expect("Missing bucket");
        assert_eq!(
            mins.len(),
            expected.len(),
            "Minimizer count mismatch for bucket {}",
            bid
        );
        // Compare sorted (in case order differs)
        let mut expected_sorted = expected.clone();
        expected_sorted.sort();
        let mut got_sorted = mins.clone();
        got_sorted.sort();
        assert_eq!(
            expected_sorted, got_sorted,
            "Minimizers mismatch for bucket {}",
            bid
        );
    }

    Ok(())
}
