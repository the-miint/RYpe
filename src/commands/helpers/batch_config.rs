//! Batch size configuration and computation for classification commands.
//!
//! This module provides shared logic for computing optimal batch sizes based on
//! available memory, index metadata, and input file characteristics.

use std::path::Path;

use anyhow::Result;

use rype::memory::{
    calculate_batch_config, detect_available_memory, estimate_shard_reservation, format_bytes,
    safety_margin_bytes, InputFormat, MemoryConfig, MemorySource, ReadMemoryProfile,
    MIN_BATCH_SIZE,
};

use super::load_index_metadata;

/// Upper bound on one input batch's read bytes (~340k short-read pairs, ~9k
/// HiFi reads), for callers whose passes span many input batches (`classify
/// run` without a negative index, `classify filter`): there, smaller batches
/// don't add index scans. Capping them keeps the per-batch costs the
/// reservation doesn't model (the batch's extracted minimizers, per-read
/// allocation overhead) from scaling with --max-memory. On the full 100M-pair
/// short-read holdout at 24G, uncapped 2.8M-pair batches peaked at 27.1 GiB
/// vs 18.3 GiB with ~450k.
pub const MAX_INPUT_BATCH_BYTES: usize = 256 << 20;

/// `batch_size` capped at `MAX_INPUT_BATCH_BYTES` of reads costing
/// `read_bytes` each, but never below `MIN_BATCH_SIZE`.
pub fn cap_input_batch(batch_size: usize, read_bytes: usize) -> usize {
    (MAX_INPUT_BATCH_BYTES / read_bytes)
        .max(MIN_BATCH_SIZE)
        .min(batch_size)
}

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
    /// Whether this is a log-ratio classification (adds deferred buffer memory)
    pub is_log_ratio: bool,
    /// Optional path to denominator index (log-ratio only).
    /// When set, shard reservation uses the larger of numerator/denominator shards.
    pub denominator_index_path: Option<&'a Path>,
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
    /// Byte budget for the `QueryAccumulator` that drives classification pass
    /// count — independent of `batch_size`, which now only bounds input
    /// (sequence-byte) residency. `usize::MAX` when unconstrained (a caller
    /// gave an explicit `batch_size_override`; see `classify_max_reads`).
    pub classify_byte_budget: usize,
    /// When set, the accumulator must also flush after this many reads,
    /// regardless of byte budget. Set only when the caller passed an explicit
    /// `batch_size_override`, to preserve read-count semantics for that case.
    pub classify_max_reads: Option<usize>,
    /// Bytes of `max_memory` set aside for input batches in flight, i.e. what
    /// `classify_byte_budget` was computed net of. 0 for a user-specified
    /// batch size.
    pub input_batch_residency: usize,
    /// Sampled average query length (R1 + R2 for pairs). 0 for a
    /// user-specified batch size.
    pub avg_query_length: usize,
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
/// One-time deduction from `classify_byte_budget` covering the parallel CSR
/// merge-join's per-thread accumulator fan-out (`classify::merge_join`, up
/// to ~2x `num_threads` full-size `HitAccumulator`s instead of one), which
/// is only live while a pass's final read count could still land at or
/// under `FOLD_REDUCE_MAX_READS`.
///
/// A one-time deduction, not a per-read multiplier: a per-read multiplier
/// was tried first and reverted (see the note at the
/// `QueryAccumulator::with_accumulator_cost_per_read` call site in
/// `commands/classify.rs`) because it creates a self-reinforcing trap —
/// scaling every read's reservation by the fan-out factor forces
/// `should_flush()` to flush well under `FOLD_REDUCE_MAX_READS` reads,
/// which "confirms" the pass is still in the risky zone, so the inflated
/// rate keeps applying and the pass never gets the chance to grow past the
/// threshold into the safe sparse-hit-collection regime it would naturally
/// reach with a realistic per-read rate. Measured to turn a real 1-pass,
/// ~15s run on a 97-bucket index into 12+ passes and 170s+.
///
/// Sized for the worst case that could ever land in the risky zone
/// (`FOLD_REDUCE_MAX_READS` reads, each costing up to `fanout - 1` extra
/// accumulator-sized allocations beyond the one the per-read rate already
/// covers), capped to at most 1/8 of `mem_limit` — the uncapped figure can
/// itself reach the multi-GB range on a many-bucket index, consuming most
/// or all of a modest `--max-memory` budget on its own (measured: zeroed
/// `classify_byte_budget` entirely on a 16 GB run against a 97-bucket
/// index). This cap is deliberately not a tight bound on the real risk — a
/// long-read/many-bucket pass small enough to hit the fold-reduce path
/// could still, in principle, exceed 1/8 of the budget in real per-thread
/// accumulator allocation — it trades some of that protection for
/// guaranteeing this term alone can never zero out the budget for every
/// other workload.
/// Floor `uncapped_budget` at `mem_limit / 8` (the same fraction used to cap
/// `fold_reduce_budget_reserve`, chosen for consistency rather than
/// independent tuning). Logs a warning when the floor binds, since a
/// silently-floored budget is otherwise indistinguishable in the logs from
/// a run that fit comfortably — see the caller for why zero is reachable
/// and what it does to `should_flush()`.
fn floor_classify_byte_budget(uncapped_budget: usize, mem_limit: usize) -> usize {
    let budget_floor = mem_limit / 8;
    if uncapped_budget < budget_floor {
        log::warn!(
            "Classification pass budget computed to {} (of {} max-memory), which is \
             below the {} floor — flooring it. Peak memory for this run may exceed \
             --max-memory; consider raising --max-memory for this index.",
            format_bytes(uncapped_budget as u64),
            format_bytes(mem_limit as u64),
            format_bytes(budget_floor as u64),
        );
        budget_floor
    } else {
        uncapped_budget
    }
}

fn fold_reduce_budget_reserve(
    mem_limit: usize,
    single_accumulator_bytes_per_read: usize,
    num_threads: usize,
) -> usize {
    let fanout = rype::memory::fold_reduce_accumulator_fanout(num_threads);
    rype::memory::fold_reduce_max_reads()
        .saturating_mul(single_accumulator_bytes_per_read)
        .saturating_mul(fanout.saturating_sub(1))
        .min(mem_limit / 8)
}

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
            // An explicit batch size is a read-count request, not a memory
            // budget: preserve it as a read-count flush trigger rather than
            // guessing at a byte budget with no read-length sample.
            classify_byte_budget: usize::MAX,
            classify_max_reads: Some(bs),
            input_batch_residency: 0,
            avg_query_length: 0,
        });
    }

    // Load index metadata to get k, w, num_buckets
    let metadata = load_index_metadata(config.index_path)?;

    // For log-ratio, use the larger shard from numerator or denominator
    let largest_shard_entries = if let Some(denom_path) = config.denominator_index_path {
        let denom_meta = load_index_metadata(denom_path)?;
        metadata
            .largest_shard_entries
            .max(denom_meta.largest_shard_entries)
    } else {
        metadata.largest_shard_entries
    };

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
                format_bytes(detected.bytes as u64),
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

    // Estimate shard loading memory from largest shard size.
    // TODO: When has_overlapping_shards is true, classify_shard_loop maintains a `seen`
    // Vec<(u64, u32)> that grows up to min(total_ref_entries, batch_unique_minimizers) × 12
    // bytes. This is not yet accounted for here. For most workloads the existing headroom
    // is sufficient, but extreme cases (400k+ long reads, 30 overlapping shards) may
    // exceed --max-memory. See PLAN-overlapping-shards-bug.md Step 3.
    let shard_reservation =
        estimate_shard_reservation(largest_shard_entries, rayon::current_num_threads());

    let mem_config = MemoryConfig {
        max_memory: mem_limit,
        num_threads: rayon::current_num_threads(),
        index_memory: estimated_index_mem,
        shard_reservation,
        read_profile,
        num_buckets,
        input_format,
        is_log_ratio: config.is_log_ratio,
    };

    let batch_config = calculate_batch_config(&mem_config);

    // What's actually resident, concurrently with the accumulator, while
    // reading and extracting one input batch (raw records + I/O prefetch
    // buffers — not the CSR/accumulator terms `calculate_batch_config`'s own
    // `peak_memory` still models, which no longer materialize per batch on
    // this path; see `estimate_accumulator_path_input_residency`'s doc).
    let input_batch_residency = rype::memory::estimate_accumulator_path_input_residency(
        batch_config.batch_size,
        &mem_config.read_profile,
        &mem_config,
    )
    .unwrap_or(0);

    let max_bucket_id = metadata.bucket_names.keys().max().copied().unwrap_or(0);
    let single_accumulator_bytes_per_read =
        rype::memory::estimate_accumulator_bytes_per_read(max_bucket_id as usize).unwrap_or(0);
    let fold_reduce_reserve = fold_reduce_budget_reserve(
        mem_limit,
        single_accumulator_bytes_per_read,
        rayon::current_num_threads(),
    );

    // Classification-pass budget: what's left of max_memory after everything
    // else concurrently resident during a pass — the shard reservation
    // (index residency during a shard scan), the safety margin, one input
    // batch's worth of raw records, and the parallel-merge-join fan-out
    // reserve above — available for the QueryAccumulator's flat COO entries.
    // This is independent of `batch_size`, which now only bounds input-batch
    // (sequence-byte) residency ahead of extraction.
    //
    // Deliberately does NOT subtract `estimated_index_mem`: for a sharded
    // index the full index is never resident at once — shards stream one at
    // a time, which `shard_reservation` above already covers. Subtracting
    // total index size would double-count the part that's resident and
    // count a great deal that never is. (It was also a no-op in practice —
    // `load_index_metadata` returns an empty `bucket_minimizer_counts` for
    // every `.ryxdi`, the only format this path accepts — so removing it
    // changes nothing about what real runs computed.) If a real index-
    // residency cost is identified later, it needs its own accounting, not
    // a revival of this term.
    let uncapped_budget = mem_limit
        .saturating_sub(shard_reservation)
        .saturating_sub(safety_margin_bytes(mem_limit))
        .saturating_sub(input_batch_residency)
        .saturating_sub(fold_reduce_reserve);

    // Floor: the five reservations above are each individually bounded, but
    // nothing bounds their sum, and for a large sharded index at a modest
    // `--max-memory` (e.g. this crate's own n97-w50 fixture at 2-4G) they
    // can consume the entire budget, driving it to exactly 0. At 0,
    // `should_flush()` is true against an *empty* accumulator, so every
    // input batch becomes its own classification pass — this is Finding 1's
    // flush-degeneration failure mode reached by a different route. Floor
    // at mem_limit/8 (the same fraction already used to cap
    // `fold_reduce_budget_reserve`) so a pass always has room to hold a
    // meaningful number of reads; this may mean actual peak RSS exceeds
    // `--max-memory` on undersized budgets, which is why it's logged.
    let classify_byte_budget = floor_classify_byte_budget(uncapped_budget, mem_limit);

    Ok(BatchSizeResult {
        batch_size: batch_config.batch_size,
        peak_memory: batch_config.peak_memory,
        input_format,
        shard_reservation,
        classify_byte_budget,
        classify_max_reads: None,
        input_batch_residency,
        avg_query_length: mem_config.read_profile.avg_query_length,
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
            is_log_ratio: false,
            denominator_index_path: None,
        };

        let result = compute_effective_batch_size(&config).unwrap();
        assert_eq!(result.batch_size, 5000);
    }

    /// Regression: `fold_reduce_budget_reserve` must never consume more
    /// than 1/8 of `mem_limit`, even when the uncapped worst-case figure
    /// (FOLD_REDUCE_MAX_READS reads x per-read accumulator cost x fan-out)
    /// would exceed the entire budget on its own. Before this cap existed,
    /// a many-bucket index at a modest --max-memory zeroed
    /// classify_byte_budget completely — verified on a real 97-bucket
    /// index at 16 GB, where the uncapped reserve alone exceeded 9 GB.
    #[test]
    fn test_fold_reduce_budget_reserve_capped_at_one_eighth_of_mem_limit() {
        let mem_limit = 16 * 1024 * 1024 * 1024; // 16GB
                                                 // A large per-read accumulator cost (many buckets) and many threads
                                                 // — deliberately sized so the uncapped product would exceed mem_limit
                                                 // entirely: FOLD_REDUCE_MAX_READS (500_000) * 1000 * 23 ~= 11.5 TB.
        let reserve = fold_reduce_budget_reserve(mem_limit, 1000, 12);
        assert_eq!(
            reserve,
            mem_limit / 8,
            "reserve should be capped at mem_limit/8 when the uncapped figure would exceed it"
        );
        assert!(
            reserve < mem_limit,
            "capped reserve must leave room for everything else"
        );
    }

    /// Direct check on the floor helper: below the floor, clamps up to it;
    /// at or above, passes through unchanged.
    #[test]
    fn test_floor_classify_byte_budget_clamps_below_floor() {
        let mem_limit = 4 * 1024 * 1024 * 1024; // 4GB -> floor = 512MB
        assert_eq!(floor_classify_byte_budget(0, mem_limit), mem_limit / 8);
        assert_eq!(
            floor_classify_byte_budget(mem_limit / 8 - 1, mem_limit),
            mem_limit / 8
        );
        assert_eq!(
            floor_classify_byte_budget(mem_limit / 8, mem_limit),
            mem_limit / 8,
            "exactly at the floor should pass through, not be treated as below it"
        );
        assert_eq!(
            floor_classify_byte_budget(mem_limit, mem_limit),
            mem_limit,
            "well above the floor should pass through unchanged"
        );
    }

    /// Reproduces the re-review's reported collapse: `classify_byte_budget`
    /// reaching exactly 0 for realistic `(max_memory, largest_shard_entries)`
    /// combinations, including this crate's own `n97-w50` fixture at
    /// `--max-memory` values CLAUDE.md itself uses as a worked example.
    /// Exercises the real budget-arithmetic functions directly (as the
    /// review did) rather than driving a multi-gigabyte synthetic index
    /// through `compute_effective_batch_size`. `input_batch_residency` is
    /// omitted, matching the review's own reproduction — including it only
    /// shrinks the uncapped budget further, so this is the conservative
    /// (largest-uncapped-budget) case for each row.
    ///
    /// Before the fix (no floor on `classify_byte_budget`'s sum), several of
    /// these rows compute to exactly 0.
    #[test]
    fn test_classify_byte_budget_floor_holds_across_reported_collapse_scenarios() {
        let num_threads = 12;
        // (max_memory, largest_shard_entries, max_bucket_id) — the three
        // combinations the review reported, using this PR's own n97-w50
        // shape (97 buckets) as the bucket count for all rows.
        let scenarios: &[(usize, u64, u32)] = &[
            (2 * 1024 * 1024 * 1024, 989_000_000, 97),
            (3 * 1024 * 1024 * 1024, 989_000_000, 97),
            (4 * 1024 * 1024 * 1024, 989_000_000, 97),
            (8 * 1024 * 1024 * 1024, 989_000_000, 97),
            (4 * 1024 * 1024 * 1024, 1_500_000_000, 97),
            (8 * 1024 * 1024 * 1024, 2_500_000_000, 97),
        ];

        for &(mem_limit, largest_shard_entries, max_bucket_id) in scenarios {
            let shard_reservation = estimate_shard_reservation(largest_shard_entries, num_threads);
            let single_accumulator_bytes_per_read =
                rype::memory::estimate_accumulator_bytes_per_read(max_bucket_id as usize)
                    .unwrap_or(0);
            let fold_reduce_reserve = fold_reduce_budget_reserve(
                mem_limit,
                single_accumulator_bytes_per_read,
                num_threads,
            );
            let uncapped_budget = mem_limit
                .saturating_sub(shard_reservation)
                .saturating_sub(safety_margin_bytes(mem_limit))
                .saturating_sub(fold_reduce_reserve);
            let budget = floor_classify_byte_budget(uncapped_budget, mem_limit);

            assert!(
                budget >= mem_limit / 8,
                "max_memory={mem_limit}, largest_shard_entries={largest_shard_entries}: \
                 budget {budget} fell below the mem_limit/8 floor (uncapped was \
                 {uncapped_budget})"
            );
            assert!(
                budget > 0,
                "max_memory={mem_limit}, largest_shard_entries={largest_shard_entries}: \
                 budget must never be 0 — that reintroduces Finding 1's one-scan-per-batch \
                 failure via an empty accumulator"
            );
        }
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
            is_log_ratio: false,
            denominator_index_path: None,
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
            is_log_ratio: false,
            denominator_index_path: None,
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
            is_log_ratio: false,
            denominator_index_path: None,
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
    // These tests use the perf-assessment data documented in CLAUDE.md's
    // "Local-Only Performance Test Data" section: `n100-w200.ryxdi` (8-shard,
    // 160-bucket, ~486M minimizers) and `n97-w50.ryxdi` (11-shard, 97-bucket,
    // ~10.5B minimizers). Both are covered here — different contributors have
    // whichever one they built locally, and running against both gives real
    // coverage from whichever is present rather than silently skipping for
    // anyone who only has the other. They are #[ignore]d because this data
    // is local-only (not in git).
    //
    // Run with: cargo test batch_config -- --ignored --nocapture
    // =========================================================================

    /// Regression: with a real sharded index and real query data, batch
    /// sizing should produce a reasonable classification byte budget (not
    /// crippled by over-reservation).
    ///
    /// Before fix: batch_count=num_threads reserved ~8x too much memory,
    /// producing batch sizes ~4-8x smaller than necessary. `batch_size` now
    /// only bounds input (sequence-byte) residency (see `QueryAccumulator`),
    /// so this regression check moved to `classify_byte_budget`, the field
    /// that now drives classification pass count.
    ///
    /// The threshold is a fraction of `max_memory`, not an absolute GB
    /// figure: `classify_byte_budget` now honestly subtracts the reference
    /// index, the shard reservation, and one input batch's real raw-record
    /// residency (see `estimate_accumulator_path_input_residency`) — for a
    /// query file of very long reads, that last term alone can be tens of
    /// GB, so a fixed "> 32GB of a 64GB budget" assumption doesn't hold
    /// across read-length regimes. A generous 10% floor still catches the
    /// actual bug this test guards against (the shard-count over-reservation
    /// crippling the budget to near-zero) without depending on how large the
    /// query reads happen to be.
    fn check_real_sharded_index_batch_size_is_reasonable(index_path: &std::path::Path) {
        let query_path = std::path::Path::new("perf-assessment/query-files/long_read.parquet");

        if !index_path.exists() || !query_path.exists() {
            eprintln!(
                "Skipping: perf-assessment data not available at {}",
                index_path.display()
            );
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
            is_log_ratio: false,
            denominator_index_path: None,
        };

        let result = compute_effective_batch_size(&config).unwrap();

        eprintln!(
            "{}: batch_size={}, peak_memory={:.2}GB, shard_reservation={:.2}MB, classify_byte_budget={:.2}GB",
            index_path.display(),
            result.batch_size,
            result.peak_memory as f64 / (1024.0 * 1024.0 * 1024.0),
            result.shard_reservation as f64 / (1024.0 * 1024.0),
            result.classify_byte_budget as f64 / (1024.0 * 1024.0 * 1024.0)
        );

        // With batch_count=1 fix, classify_byte_budget should be a healthy
        // fraction of max_memory, not crippled by shard-count
        // over-reservation. A 10% floor (not the honest-accounting fraction
        // itself, which varies with read length — see the doc comment
        // above) still catches that specific regression.
        let min_expected = config.max_memory / 10;
        assert!(
            result.classify_byte_budget > min_expected,
            "{}: classify_byte_budget should be > 10% of max_memory ({} bytes) for a {}GB \
             max-memory run, got {} bytes",
            index_path.display(),
            min_expected,
            config.max_memory / (1024 * 1024 * 1024),
            result.classify_byte_budget
        );
    }

    #[test]
    #[ignore]
    fn test_real_sharded_index_batch_size_is_reasonable_n100_w200() {
        check_real_sharded_index_batch_size_is_reasonable(std::path::Path::new(
            "perf-assessment/parquet-index/n100-w200.ryxdi",
        ));
    }

    #[test]
    #[ignore]
    fn test_real_sharded_index_batch_size_is_reasonable_n97_w50() {
        check_real_sharded_index_batch_size_is_reasonable(std::path::Path::new(
            "perf-assessment/parquet-index/n97-w50.ryxdi",
        ));
    }

    /// Regression: shard reservation must be nonzero for a sharded index.
    ///
    /// Loading a shard involves concurrent row-group decoding + filtered CSR
    /// output; this memory must be accounted for. We verify:
    /// `load_index_metadata` populates `largest_shard_entries`, and
    /// `estimate_shard_reservation` returns a nonzero value that feeds into
    /// batch sizing. Thresholds (>50M entries, >100MB reservation) are
    /// scale-independent and hold for both fixtures: `n100-w200.ryxdi`'s
    /// largest shard is ~62M entries; `n97-w50.ryxdi`'s is ~865-989M.
    fn check_real_index_shard_reservation_affects_batch_size(sharded_index: &std::path::Path) {
        if !sharded_index.exists() {
            eprintln!(
                "Skipping: perf-assessment data not available at {}",
                sharded_index.display()
            );
            return;
        }

        let metadata = load_index_metadata(sharded_index).unwrap();

        eprintln!(
            "{}: largest_shard_entries: {}",
            sharded_index.display(),
            metadata.largest_shard_entries
        );
        assert!(
            metadata.largest_shard_entries > 50_000_000,
            "{}: largest shard should be > 50M entries, got {}",
            sharded_index.display(),
            metadata.largest_shard_entries
        );

        let reservation = estimate_shard_reservation(metadata.largest_shard_entries, 8);
        let reservation_mb = reservation / (1024 * 1024);
        eprintln!(
            "{}: shard_reservation: {}MB",
            sharded_index.display(),
            reservation_mb
        );
        assert!(
            reservation_mb > 100,
            "{}: shard reservation should be > 100MB, got {}MB",
            sharded_index.display(),
            reservation_mb
        );
    }

    #[test]
    #[ignore]
    fn test_real_index_shard_reservation_affects_batch_size_n100_w200() {
        check_real_index_shard_reservation_affects_batch_size(std::path::Path::new(
            "perf-assessment/parquet-index/n100-w200.ryxdi",
        ));
    }

    #[test]
    #[ignore]
    fn test_real_index_shard_reservation_affects_batch_size_n97_w50() {
        check_real_index_shard_reservation_affects_batch_size(std::path::Path::new(
            "perf-assessment/parquet-index/n97-w50.ryxdi",
        ));
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

    // =========================================================================
    // Log-ratio denominator shard reservation tests
    // =========================================================================

    /// Create a minimal Parquet index with a configurable number of shard entries.
    fn create_test_index_with_entries(num_entries: usize) -> TempDir {
        use std::fs;

        let dir = TempDir::new().unwrap();
        let index_path = dir.path().join("test.ryxdi");
        fs::create_dir(&index_path).unwrap();

        // Create manifest with parameterized num_entries
        let manifest = format!(
            r#"magic = "RYPE_PARQUET_V1"
format_version = 1
k = 64
w = 50
salt = "0x5555555555555555"
source_hash = "0xDEADBEEF"
num_buckets = 1
total_minimizers = {num_entries}

[inverted]
num_shards = 1
total_entries = {num_entries}
has_overlapping_shards = false

[[inverted.shards]]
shard_id = 0
min_minimizer = "0x0000000000000001"
max_minimizer = "0xFFFFFFFFFFFFFFFF"
num_entries = {num_entries}
"#
        );
        fs::write(index_path.join("manifest.toml"), manifest).unwrap();

        // Create minimal buckets.parquet (1 bucket)
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

        let mut list_builder = LargeListBuilder::new(LargeStringBuilder::new());
        list_builder.values().append_value("source0");
        list_builder.append(true);
        let sources_array: ArrayRef = Arc::new(list_builder.finish());

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![0])) as ArrayRef,
                Arc::new(LargeStringArray::from(vec!["bucket0"])) as ArrayRef,
                sources_array,
            ],
        )
        .unwrap();

        let file = fs::File::create(index_path.join("buckets.parquet")).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        // Create inverted directory with minimal shard (actual parquet data not needed
        // for metadata loading, but the directory must exist)
        let inverted_path = index_path.join("inverted");
        fs::create_dir(&inverted_path).unwrap();

        let shard_schema = Arc::new(Schema::new(vec![
            Field::new("minimizer", DataType::UInt64, false),
            Field::new("bucket_id", DataType::UInt32, false),
        ]));

        // Write a minimal shard (just 1 entry — the manifest num_entries is what matters)
        let shard_batch = RecordBatch::try_new(
            shard_schema.clone(),
            vec![
                Arc::new(arrow::array::UInt64Array::from(vec![1u64])) as ArrayRef,
                Arc::new(UInt32Array::from(vec![0])) as ArrayRef,
            ],
        )
        .unwrap();

        let shard_file = fs::File::create(inverted_path.join("shard.0.parquet")).unwrap();
        let mut shard_writer = ArrowWriter::try_new(shard_file, shard_schema, None).unwrap();
        shard_writer.write(&shard_batch).unwrap();
        shard_writer.close().unwrap();

        dir
    }

    #[test]
    fn test_log_ratio_uses_larger_denominator_shard() {
        // When denominator has a much larger shard than numerator, the shard
        // reservation must use the denominator's shard size.
        let num_dir = create_test_index_with_entries(3); // tiny numerator
        let denom_dir = create_test_index_with_entries(1_000_000); // large denominator
        let num_path = num_dir.path().join("test.ryxdi");
        let denom_path = denom_dir.path().join("test.ryxdi");
        let r1_file = create_test_fastq(150, 10);

        let config = BatchSizeConfig {
            batch_size_override: None,
            max_memory: 4 * 1024 * 1024 * 1024, // 4GB
            r1_path: r1_file.path(),
            r2_path: None,
            is_parquet_input: false,
            index_path: &num_path,
            trim_to: None,
            minimum_length: None,
            is_log_ratio: true,
            denominator_index_path: Some(&denom_path),
        };

        let result = compute_effective_batch_size(&config).unwrap();
        let threads = rayon::current_num_threads();
        let expected_reservation = estimate_shard_reservation(1_000_000, threads);

        assert!(
            result.shard_reservation >= expected_reservation,
            "shard_reservation {} should be >= {} (from denominator's 1M-entry shard)",
            result.shard_reservation,
            expected_reservation
        );
    }

    #[test]
    fn test_log_ratio_batch_size_shrinks_for_large_denominator() {
        // With a large denominator shard, the batch size should be smaller than
        // without a denominator (more memory reserved for shard loading).
        let num_dir = create_test_index_with_entries(3);
        let denom_dir = create_test_index_with_entries(1_000_000);
        let num_path = num_dir.path().join("test.ryxdi");
        let denom_path = denom_dir.path().join("test.ryxdi");
        let r1_file = create_test_fastq(150, 10);

        let config_no_denom = BatchSizeConfig {
            batch_size_override: None,
            max_memory: 4 * 1024 * 1024 * 1024,
            r1_path: r1_file.path(),
            r2_path: None,
            is_parquet_input: false,
            index_path: &num_path,
            trim_to: None,
            minimum_length: None,
            is_log_ratio: true,
            denominator_index_path: None,
        };

        let config_with_denom = BatchSizeConfig {
            batch_size_override: None,
            max_memory: 4 * 1024 * 1024 * 1024,
            r1_path: r1_file.path(),
            r2_path: None,
            is_parquet_input: false,
            index_path: &num_path,
            trim_to: None,
            minimum_length: None,
            is_log_ratio: true,
            denominator_index_path: Some(&denom_path),
        };

        let result_no_denom = compute_effective_batch_size(&config_no_denom).unwrap();
        let result_with_denom = compute_effective_batch_size(&config_with_denom).unwrap();

        assert!(
            result_with_denom.batch_size < result_no_denom.batch_size,
            "Batch size with large denom {} should be < without denom {} \
             (more memory reserved for shard loading)",
            result_with_denom.batch_size,
            result_no_denom.batch_size
        );
    }
}
