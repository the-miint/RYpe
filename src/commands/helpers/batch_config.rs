//! Batch size configuration and computation for classification commands.
//!
//! This module provides shared logic for computing optimal batch sizes based on
//! available memory, index metadata, and input file characteristics.

use std::path::Path;

use anyhow::Result;

use rype::memory::{
    calculate_batch_config, detect_available_memory, estimate_shard_reservation, format_bytes,
    InputFormat, MemoryConfig, MemorySource, ReadMemoryProfile,
};

use super::load_index_metadata;

/// Configuration for computing effective batch size.
pub struct BatchSizeConfig<'a> {
    /// User-specified batch size override (None = auto-detect)
    pub batch_size_override: Option<usize>,
    /// Maximum memory to use (0 = auto-detect)
    pub max_memory: usize,
    /// Path to R1 input file
    pub r1_path: &'a Path,
    /// Optional path to R2 input file for paired-end
    pub r2_path: Option<&'a Path>,
    /// Whether input is Parquet format
    pub is_parquet_input: bool,
    /// Path to the index
    pub index_path: &'a Path,
    /// Optional trim-to length (caps read lengths for memory estimation)
    pub trim_to: Option<usize>,
    /// Optional minimum read length filter
    pub minimum_length: Option<usize>,
}

/// Result of batch size computation with logging metadata.
pub struct BatchSizeResult {
    /// The computed batch size
    pub batch_size: usize,
    /// Estimated peak memory (for logging)
    pub peak_memory: usize,
    /// Input format used for estimation
    pub input_format: InputFormat,
    /// Memory reserved for shard loading (for logging)
    pub shard_reservation: usize,
}

/// Determine the correct `InputFormat` for memory estimation.
///
/// For Parquet input with trim/filter active, the reader thread converts Arrow batches
/// to `OwnedFastxRecord`, so memory estimation should use owned-record sizing.
fn determine_input_format(config: &BatchSizeConfig, is_paired: bool) -> InputFormat {
    if config.is_parquet_input {
        let trimmed_in_reader = config.trim_to.is_some() || config.minimum_length.is_some();
        InputFormat::Parquet {
            is_paired,
            trimmed_in_reader,
        }
    } else {
        InputFormat::Fastx { is_paired }
    }
}

/// Compute effective batch size with memory-aware auto-sizing.
///
/// If `config.batch_size_override` is Some, returns that value directly.
/// Otherwise, uses adaptive batch sizing based on:
/// - Available system memory (auto-detected or from `max_memory`)
/// - Index metadata (k, w, bucket counts)
/// - Sampled read lengths from input files
/// - Input format (Parquet vs FASTX)
///
/// # Arguments
/// * `config` - Batch size configuration parameters
///
/// # Returns
/// A `BatchSizeResult` containing the computed batch size and metadata for logging.
pub fn compute_effective_batch_size(config: &BatchSizeConfig) -> Result<BatchSizeResult> {
    // For FASTX, paired-end is determined by having an R2 file.
    // For Parquet, paired-end data lives in the sequence2 column of the same file,
    // so r2_path is always None. We detect pairing from the file itself later.
    let is_paired_hint = config.r2_path.is_some();

    if let Some(bs) = config.batch_size_override {
        log::info!("Using user-specified batch size: {}", bs);
        // For user-specified batch, return minimal metadata
        let input_format = determine_input_format(config, is_paired_hint);
        return Ok(BatchSizeResult {
            batch_size: bs,
            peak_memory: 0, // Unknown for user-specified
            input_format,
            shard_reservation: 0, // Unknown for user-specified
        });
    }

    // Load index metadata to get k, w, num_buckets
    let metadata = load_index_metadata(config.index_path)?;

    // Detect or use specified memory limit (0 = auto)
    let mem_limit = if config.max_memory == 0 {
        let detected = detect_available_memory();
        if detected.source == MemorySource::Fallback {
            log::warn!(
                "Could not detect available memory, using 8GB fallback. \
                Consider specifying --max-memory explicitly."
            );
        } else {
            log::info!(
                "Auto-detected available memory: {} (source: {:?})",
                format_bytes(detected.bytes),
                detected.source
            );
        }
        detected.bytes
    } else {
        config.max_memory
    };

    // Log trim_to setting
    let effective_trim_to = match config.trim_to {
        Some(0) => {
            log::warn!("--trim-to 0 specified, treating as no trimming");
            None
        }
        Some(n) => {
            log::info!("Read trimming enabled: --trim-to {}", n);
            Some(n)
        }
        None => None,
    };

    // Sample read lengths from input files
    let read_profile = ReadMemoryProfile::from_files(
        config.r1_path,
        config.r2_path,
        1000, // sample size
        metadata.k,
        metadata.w,
        config.is_parquet_input,
        effective_trim_to,
    )
    .unwrap_or_else(|| {
        log::warn!("Could not sample read lengths, using default profile");
        ReadMemoryProfile::default_profile(is_paired_hint, metadata.k, metadata.w)
    });

    // Use is_paired from the read profile, which correctly detects pairing
    // from the Parquet sequence2 column (not just from r2_path).
    let is_paired = read_profile.is_paired;

    log::debug!(
        "Read profile: avg_read_length={}, avg_query_length={}, minimizers_per_query={}, is_paired={}",
        read_profile.avg_read_length,
        read_profile.avg_query_length,
        read_profile.minimizers_per_query,
        is_paired
    );

    // Estimate index memory from metadata
    let estimated_index_mem = metadata.bucket_minimizer_counts.values().sum::<usize>() * 8;
    let num_buckets = metadata.bucket_names.len();

    // Determine input format for accurate memory estimation
    let input_format = determine_input_format(config, is_paired);

    // Estimate shard loading memory from largest shard size
    let shard_reservation =
        estimate_shard_reservation(metadata.largest_shard_entries, rayon::current_num_threads());

    let mem_config = MemoryConfig {
        max_memory: mem_limit,
        num_threads: rayon::current_num_threads(),
        index_memory: estimated_index_mem,
        shard_reservation,
        read_profile,
        num_buckets,
        input_format,
    };

    let batch_config = calculate_batch_config(&mem_config);

    Ok(BatchSizeResult {
        batch_size: batch_config.batch_size,
        peak_memory: batch_config.peak_memory,
        input_format,
        shard_reservation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{NamedTempFile, TempDir};

    /// Create a minimal Parquet index for testing.
    fn create_test_index() -> TempDir {
        use std::fs;

        let dir = TempDir::new().unwrap();
        let index_path = dir.path().join("test.ryxdi");
        fs::create_dir(&index_path).unwrap();

        // Create minimal manifest.toml
        let manifest = r#"magic = "RYPE_PARQUET_V1"
format_version = 1
k = 64
w = 50
salt = "0x5555555555555555"
source_hash = "0xDEADBEEF"
num_buckets = 2
total_minimizers = 1000

[inverted]
num_shards = 1
total_entries = 3
has_overlapping_shards = false

[[inverted.shards]]
shard_id = 0
min_minimizer = "0x0000000000000001"
max_minimizer = "0x0000000000000003"
num_entries = 3
"#;
        fs::write(index_path.join("manifest.toml"), manifest).unwrap();

        // Create minimal buckets.parquet
        use arrow::array::{
            ArrayRef, LargeListBuilder, LargeStringArray, LargeStringBuilder, UInt32Array,
        };
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("bucket_id", DataType::UInt32, false),
            Field::new("bucket_name", DataType::LargeUtf8, false),
            Field::new(
                "sources",
                DataType::LargeList(Arc::new(Field::new("item", DataType::LargeUtf8, true))),
                false,
            ),
        ]));

        // Build list array for sources
        let mut list_builder = LargeListBuilder::new(LargeStringBuilder::new());
        list_builder.values().append_value("source0");
        list_builder.append(true);
        list_builder.values().append_value("source1");
        list_builder.append(true);
        let sources_array: ArrayRef = Arc::new(list_builder.finish());

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![0, 1])) as ArrayRef,
                Arc::new(LargeStringArray::from(vec!["bucket0", "bucket1"])) as ArrayRef,
                sources_array,
            ],
        )
        .unwrap();

        let file = fs::File::create(index_path.join("buckets.parquet")).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        // Create inverted directory with minimal shard
        let inverted_path = index_path.join("inverted");
        fs::create_dir(&inverted_path).unwrap();

        let shard_schema = Arc::new(Schema::new(vec![
            Field::new("minimizer", DataType::UInt64, false),
            Field::new("bucket_id", DataType::UInt32, false),
        ]));

        let shard_batch = RecordBatch::try_new(
            shard_schema.clone(),
            vec![
                Arc::new(arrow::array::UInt64Array::from(vec![1u64, 2, 3])) as ArrayRef,
                Arc::new(UInt32Array::from(vec![0, 0, 1])) as ArrayRef,
            ],
        )
        .unwrap();

        let shard_file = fs::File::create(inverted_path.join("shard.0.parquet")).unwrap();
        let mut shard_writer = ArrowWriter::try_new(shard_file, shard_schema, None).unwrap();
        shard_writer.write(&shard_batch).unwrap();
        shard_writer.close().unwrap();

        dir
    }

    /// Create a temporary FASTQ file with known read lengths.
    fn create_test_fastq(read_length: usize, num_reads: usize) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        for i in 0..num_reads {
            writeln!(file, "@read{}", i).unwrap();
            writeln!(file, "{}", "A".repeat(read_length)).unwrap();
            writeln!(file, "+").unwrap();
            writeln!(file, "{}", "I".repeat(read_length)).unwrap();
        }
        file.flush().unwrap();
        file
    }

    #[test]
    fn test_user_specified_batch_size_used_directly() {
        let index_dir = create_test_index();
        let index_path = index_dir.path().join("test.ryxdi");
        let r1_file = create_test_fastq(150, 10);

        let config = BatchSizeConfig {
            batch_size_override: Some(5000),
            max_memory: 0,
            r1_path: r1_file.path(),
            r2_path: None,
            is_parquet_input: false,
            index_path: &index_path,
            trim_to: None,
            minimum_length: None,
        };

        let result = compute_effective_batch_size(&config).unwrap();
        assert_eq!(result.batch_size, 5000);
    }

    #[test]
    fn test_auto_batch_size_returns_reasonable_value() {
        let index_dir = create_test_index();
        let index_path = index_dir.path().join("test.ryxdi");
        let r1_file = create_test_fastq(150, 10);

        let config = BatchSizeConfig {
            batch_size_override: None,
            max_memory: 4 * 1024 * 1024 * 1024, // 4GB
            r1_path: r1_file.path(),
            r2_path: None,
            is_parquet_input: false,
            index_path: &index_path,
            trim_to: None,
            minimum_length: None,
        };

        let result = compute_effective_batch_size(&config).unwrap();

        // Should return a reasonable batch size (at least MIN_BATCH_SIZE)
        assert!(
            result.batch_size >= 1000,
            "Batch size {} should be >= 1000",
            result.batch_size
        );
        assert!(result.peak_memory > 0);
    }

    #[test]
    fn test_paired_end_detected_correctly() {
        let index_dir = create_test_index();
        let index_path = index_dir.path().join("test.ryxdi");
        let r1_file = create_test_fastq(150, 10);
        let r2_file = create_test_fastq(150, 10);

        let config = BatchSizeConfig {
            batch_size_override: None,
            max_memory: 4 * 1024 * 1024 * 1024,
            r1_path: r1_file.path(),
            r2_path: Some(r2_file.path()),
            is_parquet_input: false,
            index_path: &index_path,
            trim_to: None,
            minimum_length: None,
        };

        let result = compute_effective_batch_size(&config).unwrap();

        // Should detect paired-end format
        assert!(matches!(
            result.input_format,
            InputFormat::Fastx { is_paired: true }
        ));
    }

    #[test]
    fn test_parquet_format_detected() {
        let index_dir = create_test_index();
        let index_path = index_dir.path().join("test.ryxdi");
        let r1_file = create_test_fastq(150, 10);

        let config = BatchSizeConfig {
            batch_size_override: None,
            max_memory: 4 * 1024 * 1024 * 1024,
            r1_path: r1_file.path(),
            r2_path: None,
            is_parquet_input: true,
            index_path: &index_path,
            trim_to: None,
            minimum_length: None,
        };

        let result = compute_effective_batch_size(&config).unwrap();

        assert!(matches!(
            result.input_format,
            InputFormat::Parquet {
                is_paired: false,
                trimmed_in_reader: false
            }
        ));
    }

    // =========================================================================
    // Regression tests for batch sizing with real sharded indices.
    //
    // These tests use the perf-assessment data (real 8-shard index with 160
    // buckets, ~485M total entries; real long-read Parquet queries).
    // They are #[ignore]d because the data is local-only (not in git).
    //
    // Run with: cargo test batch_config -- --ignored --nocapture
    // =========================================================================

    /// Regression: with real sharded index and real query data, batch sizing
    /// should produce a reasonable batch size (not crippled by over-reservation).
    ///
    /// Before fix: batch_count=num_threads reserved ~8x too much memory,
    /// producing batch sizes ~4-8x smaller than necessary.
    #[test]
    #[ignore]
    fn test_real_sharded_index_batch_size_is_reasonable() {
        let index_path = std::path::Path::new("perf-assessment/parquet-index/n100-w200.ryxdi");
        let query_path = std::path::Path::new("perf-assessment/query-files/long_read.parquet");

        if !index_path.exists() || !query_path.exists() {
            eprintln!("Skipping: perf-assessment data not available");
            return;
        }

        let config = BatchSizeConfig {
            batch_size_override: None,
            max_memory: 64 * 1024 * 1024 * 1024, // 64GB
            r1_path: query_path,
            r2_path: None,
            is_parquet_input: true,
            index_path,
            trim_to: None,
            minimum_length: None,
        };

        let result = compute_effective_batch_size(&config).unwrap();

        eprintln!(
            "Real index: batch_size={}, peak_memory={:.2}GB, shard_reservation={:.2}MB",
            result.batch_size,
            result.peak_memory as f64 / (1024.0 * 1024.0 * 1024.0),
            result.shard_reservation as f64 / (1024.0 * 1024.0)
        );

        // With batch_count=1 fix, batch_size should be well above 1M for 64GB
        assert!(
            result.batch_size > 1_000_000,
            "batch_size should be > 1M for 64GB with sequential batches, got {}",
            result.batch_size
        );
    }

    /// Regression: shard reservation must be nonzero for a sharded index.
    ///
    /// The 8-shard index has largest shard ~62M entries. Loading a shard
    /// involves concurrent row-group decoding + filtered CSR output.
    /// This memory must be accounted for.
    ///
    /// We verify: load_index_metadata populates largest_shard_entries,
    /// and estimate_shard_reservation returns a nonzero value that feeds
    /// into the batch sizing. The shard reservation for the 8-shard index
    /// (largest shard ≈ 62M entries) should be substantial (>100MB).
    #[test]
    #[ignore]
    fn test_real_index_shard_reservation_affects_batch_size() {
        let sharded_index = std::path::Path::new("perf-assessment/parquet-index/n100-w200.ryxdi");

        if !sharded_index.exists() {
            eprintln!("Skipping: perf-assessment data not available");
            return;
        }

        let metadata = load_index_metadata(sharded_index).unwrap();

        // The 8-shard index has shards with ~60-62M entries each
        eprintln!("largest_shard_entries: {}", metadata.largest_shard_entries);
        assert!(
            metadata.largest_shard_entries > 50_000_000,
            "8-shard index should have largest shard > 50M entries, got {}",
            metadata.largest_shard_entries
        );

        // Shard reservation should be substantial
        let reservation = estimate_shard_reservation(metadata.largest_shard_entries, 8);
        let reservation_mb = reservation / (1024 * 1024);
        eprintln!("shard_reservation: {}MB", reservation_mb);
        assert!(
            reservation_mb > 100,
            "Shard reservation should be > 100MB for 62M-entry shard, got {}MB",
            reservation_mb
        );
    }

    /// Regression: with the test helper index (has shards with known entries),
    /// verify that shard info flows through to batch sizing.
    ///
    /// The test index has 1 shard with 3 entries. After fix, load_index_metadata
    /// should populate largest_shard_entries=3, and compute_effective_batch_size
    /// should use it for shard_reservation.
    #[test]
    fn test_shard_info_flows_to_batch_sizing() {
        let index_dir = create_test_index();
        let index_path = index_dir.path().join("test.ryxdi");

        // Load metadata and verify shard info is captured
        let metadata = load_index_metadata(&index_path).unwrap();

        // After fix: largest_shard_entries should be 3 (from manifest)
        // Before fix: field doesn't exist or is always 0
        // We use a compile-time check: if the field exists, it must be 3
        assert_eq!(
            metadata.largest_shard_entries, 3,
            "load_index_metadata should populate largest_shard_entries from manifest"
        );
    }
}
