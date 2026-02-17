//! Classify command handlers and helper functions.
//!
//! This module contains the implementation logic for classification commands.

use anyhow::{anyhow, Result};
use arrow::record_batch::RecordBatch;
use std::collections::HashMap;
use std::path::PathBuf;

use rype::memory::format_bytes;
use rype::{
    classify_batch_sharded_merge_join, classify_batch_sharded_parallel_rg,
    classify_with_sharded_negative, filter_best_hits, log_timing, partition_by_numerator_score,
    validate_compatible_indices, validate_single_bucket_index, PartitionResult,
    ShardedInvertedIndex,
};

use super::helpers::seq_writer::rewalk_and_write_passing;
use super::helpers::{
    accumulate_owned_batches, compute_effective_batch_size, create_input_reader,
    format_classification_results, format_log_ratio_bucket_name, format_log_ratio_output,
    is_parquet_input, load_index_for_classification, stacked_batches_to_records,
    validate_input_config, BatchSizeConfig, ClassificationInput, DeferredDenomBuffer,
    IndexLoadOptions, InputReaderConfig, OutputFormat, OutputWriter, PassingReadTracker,
};

/// Common arguments shared between classify run and log-ratio commands.
pub struct CommonClassifyArgs {
    pub index: PathBuf,
    pub r1: PathBuf,
    pub r2: Option<PathBuf>,
    pub threshold: f64,
    pub max_memory: usize,
    pub batch_size: Option<usize>,
    pub output: Option<PathBuf>,
    pub parallel_rg: bool,
    pub use_bloom_filter: bool,
    pub parallel_input_rg: usize,
    pub trim_to: Option<usize>,
    pub minimum_length: Option<usize>,
}

/// Arguments for the classify run command.
pub struct ClassifyRunArgs {
    pub common: CommonClassifyArgs,
    pub negative_index: Option<PathBuf>,
    pub best_hit: bool,
    pub wide: bool,
}

/// Default threshold value for classification.
const DEFAULT_THRESHOLD: f64 = 0.1;

/// Tolerance for floating-point threshold comparison.
/// This is generous enough to handle typical floating-point representation issues
/// while still catching intentional user-specified threshold values.
const THRESHOLD_TOLERANCE: f64 = 1e-9;

/// Run the classify command with the given arguments.
pub fn run_classify(args: ClassifyRunArgs) -> Result<()> {
    // Validate --wide incompatibility with --threshold
    if args.wide && (args.common.threshold - DEFAULT_THRESHOLD).abs() > THRESHOLD_TOLERANCE {
        return Err(anyhow!(
            "--wide is incompatible with --threshold.\n\
             Wide format requires all bucket scores, so no threshold filtering can be applied.\n\
             Use --wide without --threshold, or omit --wide to use threshold filtering."
        ));
    }

    // Load negative index if provided (memory-efficient sharded filtering)
    let negative_sharded: Option<ShardedInvertedIndex> = if let Some(ref neg_path) =
        args.negative_index
    {
        if !rype::is_parquet_index(neg_path) {
            return Err(anyhow!(
                "Negative index not found or not in Parquet format: {}\n\
                 Create a negative index with: rype index create -o negative.ryxdi -r contaminants.fasta",
                neg_path.display()
            ));
        }
        log::info!("Loading negative index from {:?}", neg_path);
        let neg = ShardedInvertedIndex::open(neg_path)?;
        log::info!(
            "Negative index: {} shards, {} total minimizers (memory-efficient filtering enabled)",
            neg.num_shards(),
            neg.total_minimizers()
        );
        Some(neg)
    } else {
        None
    };

    // Check for Parquet input early (needed for memory estimation)
    let input_is_parquet = is_parquet_input(&args.common.r1);

    // Determine effective batch size: user override or adaptive
    let batch_result = compute_effective_batch_size(&BatchSizeConfig {
        batch_size_override: args.common.batch_size,
        max_memory: args.common.max_memory,
        r1_path: &args.common.r1,
        r2_path: args.common.r2.as_deref(),
        is_parquet_input: input_is_parquet,
        index_path: &args.common.index,
        trim_to: args.common.trim_to,
        minimum_length: args.common.minimum_length,
        is_log_ratio: false,
        denominator_index_path: None,
    })?;
    let effective_batch_size = batch_result.batch_size;

    // Log adaptive batch sizing details (only for auto-computed batch sizes)
    if args.common.batch_size.is_none() {
        log::info!(
            "Adaptive batch sizing: batch_size={}, threads={}, estimated peak memory={}, shard_reservation={}, format={:?}",
            batch_result.batch_size,
            rayon::current_num_threads(),
            format_bytes(batch_result.peak_memory),
            format_bytes(batch_result.shard_reservation),
            batch_result.input_format
        );
    }
    // Validate input configuration
    validate_input_config(input_is_parquet, args.common.r2.as_ref())?;

    // Load index (validates, loads metadata, sharded index, and read options)
    let loaded_index = load_index_for_classification(
        &args.common.index,
        &IndexLoadOptions {
            use_bloom_filter: args.common.use_bloom_filter,
            parallel_rg: args.common.parallel_rg,
        },
    )?;
    let metadata = loaded_index.metadata;
    let sharded = loaded_index.sharded;
    let read_options = loaded_index.read_options;

    // Set up I/O based on input format
    let output_format = OutputFormat::detect(args.common.output.as_ref());

    // For wide format: build header and bucket_ids once (reused for both writer and formatting)
    let (wide_header, wide_bucket_ids): (Option<Vec<u8>>, Option<Vec<u32>>) = if args.wide {
        let (header, bucket_ids) = build_wide_header(&metadata.bucket_names);
        (Some(header), Some(bucket_ids))
    } else {
        (None, None)
    };

    // Create output writer and write header
    let mut out_writer = if args.wide {
        let mut writer = OutputWriter::new_wide(
            output_format,
            args.common.output.as_ref(),
            &metadata.bucket_names,
            None,
        )?;
        writer.write_header(wide_header.as_ref().unwrap())?;
        writer
    } else {
        let mut writer = OutputWriter::new(output_format, args.common.output.as_ref(), None)?;
        writer.write_header(b"read_id\tbucket_name\tscore\n")?;
        writer
    };

    // For wide format or best-hit, use threshold 0.0 to get all scores.
    // Best-hit needs all scores so it can select the highest per query.
    let effective_threshold = if args.wide || args.best_hit {
        0.0
    } else {
        args.common.threshold
    };

    // Create input reader (Parquet or FASTX)
    // Note: For FASTX, trim_to is handled at read time. For Parquet, it's handled
    // during batch conversion (see Parquet processing path below).
    let mut input_reader = create_input_reader(
        &InputReaderConfig {
            r1_path: &args.common.r1,
            r2_path: args.common.r2.as_ref(),
            batch_size: effective_batch_size,
            parallel_input_rg: args.common.parallel_input_rg,
            is_parquet: input_is_parquet,
            trim_to: args.common.trim_to,
            minimum_length: args.common.minimum_length,
        },
        false, // Not writing sequences in run command
    )?;

    let mut total_reads = 0;
    let mut batch_num = 0;

    if args.common.parallel_rg {
        log::info!(
            "Starting parallel row group classification (batch_size={})",
            effective_batch_size
        );
    } else {
        log::info!(
            "Starting merge-join classification with sequential shard loading (batch_size={})",
            effective_batch_size
        );
    }

    // Log which I/O path will be used for Parquet input
    let parquet_needs_owned = args.common.trim_to.is_some() || args.common.minimum_length.is_some();
    if input_is_parquet {
        if parquet_needs_owned {
            log::info!(
                "Using owned-copy Parquet path (trim_to={:?}, minimum_length={:?})",
                args.common.trim_to,
                args.common.minimum_length
            );
        } else {
            log::debug!("Using zero-copy Parquet path for maximum performance");
        }
    }

    // Helper closure for classification
    // Note: uses effective_threshold (0.0 for wide format, args.common.threshold otherwise)
    // Sequences are pre-trimmed at read time when --trim-to is specified.
    let classify_records = |batch_refs: &[rype::QueryRecord]| -> Result<Vec<rype::HitResult>> {
        // If negative index is provided, use memory-efficient sharded filtering
        if let Some(ref neg) = negative_sharded {
            // Negative filtering not supported with parallel-rg
            if args.common.parallel_rg {
                return Err(anyhow!(
                    "Negative index filtering is not supported with --parallel-rg."
                ));
            }
            classify_with_sharded_negative(
                &sharded,
                Some(neg),
                batch_refs,
                effective_threshold,
                read_options.as_ref(),
            )
        } else if args.common.parallel_rg {
            classify_batch_sharded_parallel_rg(
                &sharded,
                None,
                batch_refs,
                effective_threshold,
                read_options.as_ref(),
            )
        } else {
            classify_batch_sharded_merge_join(
                &sharded,
                None,
                batch_refs,
                effective_threshold,
                read_options.as_ref(),
            )
        }
    };

    loop {
        let t_io_read = std::time::Instant::now();

        // Handle Parquet and FASTX differently for zero-copy optimization
        match &mut input_reader {
            ClassificationInput::Parquet(reader) => {
                // Parquet path: use owned-copy when trim/filter active,
                // otherwise use zero-copy for better performance with short reads.
                if parquet_needs_owned {
                    // Owned-copy path: reader thread trims/filters, we accumulate
                    let result = accumulate_owned_batches(reader, effective_batch_size)?;

                    log_timing("batch: io_read+trim", t_io_read.elapsed().as_millis());

                    if result.records.is_empty() {
                        break;
                    }

                    let is_final_batch = result.reached_end;
                    batch_num += 1;
                    let batch_read_count = result.records.len();
                    total_reads += batch_read_count;

                    let t_convert = std::time::Instant::now();
                    let batch_refs: Vec<rype::QueryRecord> = result
                        .records
                        .iter()
                        .map(|rec| (rec.query_id, rec.seq1.as_slice(), rec.seq2.as_deref()))
                        .collect();
                    log_timing("batch: convert_refs", t_convert.elapsed().as_millis());

                    log::debug!(
                        "Converted {} row groups into {} trimmed records",
                        result.rg_count,
                        batch_refs.len()
                    );

                    let results = classify_records(&batch_refs)?;
                    let results = if args.best_hit {
                        filter_best_hits(results)
                    } else {
                        results
                    };

                    let t_format = std::time::Instant::now();
                    let chunk_out = if let Some(ref bucket_ids) = wide_bucket_ids {
                        format_results_wide(&results, &result.headers, bucket_ids)
                    } else {
                        format_classification_results(
                            &results,
                            &result.headers,
                            &metadata.bucket_names,
                        )
                    };
                    log_timing("batch: format_output", t_format.elapsed().as_millis());

                    let t_write = std::time::Instant::now();
                    out_writer.write_chunk(chunk_out)?;
                    log_timing("batch: io_write", t_write.elapsed().as_millis());

                    log::info!(
                        "Processed batch {} ({} row groups): {} reads ({} total)",
                        batch_num,
                        result.rg_count,
                        batch_read_count,
                        total_reads
                    );

                    if is_final_batch {
                        break;
                    }
                } else {
                    // Zero-copy Parquet path with batch stacking (no trimming)
                    let mut stacked_batches: Vec<(RecordBatch, Vec<String>)> = Vec::new();
                    let mut stacked_rows = 0usize;
                    let mut reached_end = false;

                    // Accumulate batches until we have enough rows or run out of data
                    loop {
                        let batch_opt = reader.next_batch()?;
                        let Some(parquet_batch) = batch_opt else {
                            reached_end = true;
                            break;
                        };
                        let (record_batch, headers) = parquet_batch.into_arrow();

                        let batch_rows = record_batch.num_rows();
                        stacked_rows += batch_rows;
                        stacked_batches.push((record_batch, headers));

                        if stacked_rows >= effective_batch_size {
                            break;
                        }
                    }

                    log_timing("batch: io_read", t_io_read.elapsed().as_millis());

                    if stacked_batches.is_empty() {
                        break;
                    }

                    let is_final_batch = reached_end;
                    batch_num += 1;
                    total_reads += stacked_rows;

                    let t_convert = std::time::Instant::now();
                    let (batch_refs, headers) = stacked_batches_to_records(&stacked_batches)?;
                    log_timing("batch: convert_refs", t_convert.elapsed().as_millis());

                    log::debug!(
                        "Stacked {} row groups into {} records",
                        stacked_batches.len(),
                        batch_refs.len()
                    );

                    // already_trimmed=false since we're using zero-copy (no trimming at read)
                    let results = classify_records(&batch_refs)?;
                    let results = if args.best_hit {
                        filter_best_hits(results)
                    } else {
                        results
                    };

                    let t_format = std::time::Instant::now();
                    let chunk_out = if let Some(ref bucket_ids) = wide_bucket_ids {
                        format_results_wide_ref(&results, &headers, bucket_ids)
                    } else {
                        format_classification_results(&results, &headers, &metadata.bucket_names)
                    };
                    log_timing("batch: format_output", t_format.elapsed().as_millis());

                    let t_write = std::time::Instant::now();
                    out_writer.write_chunk(chunk_out)?;
                    log_timing("batch: io_write", t_write.elapsed().as_millis());

                    log::info!(
                        "Processed batch {} ({} batches stacked): {} reads ({} total)",
                        batch_num,
                        stacked_batches.len(),
                        stacked_rows,
                        total_reads
                    );

                    if is_final_batch {
                        break;
                    }
                }
            }
            ClassificationInput::Fastx(io) => {
                // FASTX path (copies sequences, trimmed at read time if trim_to set)
                let batch_opt = io.next_batch()?;
                log_timing("batch: io_read", t_io_read.elapsed().as_millis());

                let Some((owned_records, headers)) = batch_opt else {
                    break;
                };

                batch_num += 1;
                let batch_read_count = owned_records.len();
                total_reads += batch_read_count;

                let t_convert = std::time::Instant::now();
                let batch_refs: Vec<rype::QueryRecord> = owned_records
                    .iter()
                    .map(|rec| (rec.query_id, rec.seq1.as_slice(), rec.seq2.as_deref()))
                    .collect();
                log_timing("batch: convert_refs", t_convert.elapsed().as_millis());

                // already_trimmed=true since FASTX reader now trims at read time
                let results = classify_records(&batch_refs)?;
                let results = if args.best_hit {
                    filter_best_hits(results)
                } else {
                    results
                };

                let t_format = std::time::Instant::now();
                let chunk_out = if let Some(ref bucket_ids) = wide_bucket_ids {
                    format_results_wide(&results, &headers, bucket_ids)
                } else {
                    format_classification_results(&results, &headers, &metadata.bucket_names)
                };
                log_timing("batch: format_output", t_format.elapsed().as_millis());

                let t_write = std::time::Instant::now();
                out_writer.write_chunk(chunk_out)?;
                log_timing("batch: io_write", t_write.elapsed().as_millis());

                log::info!(
                    "Processed batch {}: {} reads ({} total)",
                    batch_num,
                    batch_read_count,
                    total_reads
                );
            }
        }
    }

    log::info!("Classification complete: {} reads processed", total_reads);
    out_writer.finish()?;
    input_reader.finish()?;

    Ok(())
}

/// Arguments for the classify aggregate command.
#[allow(dead_code)] // Fields will be used when aggregate is implemented
pub struct ClassifyAggregateArgs {
    pub index: PathBuf,
    pub negative_index: Option<PathBuf>,
    pub r1: PathBuf,
    pub r2: Option<PathBuf>,
    pub threshold: f64,
    pub max_memory: usize,
    pub batch_size: Option<usize>,
    pub output: Option<PathBuf>,
}

/// Run the aggregate classify command with the given arguments.
pub fn run_aggregate(_args: ClassifyAggregateArgs) -> Result<()> {
    Err(anyhow!(
        "aggregate command is not yet supported with Parquet indices.\n\
         This feature is pending development. Use 'classify run' for per-read classification."
    ))
}

/// Arguments for the classify log-ratio command.
///
/// Uses two single-bucket indices (numerator and denominator) instead of a
/// single two-bucket index.
pub struct ClassifyLogRatioArgs {
    pub numerator: PathBuf,
    pub denominator: PathBuf,
    pub r1: PathBuf,
    pub r2: Option<PathBuf>,
    pub max_memory: usize,
    pub batch_size: Option<usize>,
    pub output: Option<PathBuf>,
    pub parallel_rg: bool,
    pub use_bloom_filter: bool,
    pub parallel_input_rg: usize,
    pub trim_to: Option<usize>,
    pub minimum_length: Option<usize>,
    /// Output path for passing sequences (gzipped FASTA/FASTQ).
    pub output_sequences: Option<PathBuf>,
    /// If true, pass sequences with POSITIVE log-ratio (default: pass NEGATIVE/zero).
    pub passing_is_positive: bool,
    /// If set, reads with numerator score >= this value skip denominator classification
    /// and are assigned +inf (fast path).
    pub numerator_skip_threshold: Option<f64>,
}

/// Validate sequence output configuration.
///
/// # Errors
/// Returns an error if:
/// - `--trim-to` is used with `--output-sequences` (would output incomplete sequences)
fn validate_seq_output(
    _is_parquet: bool,
    has_trim_to: bool,
    output_sequences: Option<&std::path::Path>,
) -> Result<()> {
    let Some(_path) = output_sequences else {
        return Ok(());
    };

    if has_trim_to {
        return Err(anyhow!(
            "--output-sequences is not supported with --trim-to.\n\
             Trimmed sequences would be incomplete. Remove --trim-to to output full sequences."
        ));
    }

    Ok(())
}

/// Run the log-ratio classify command with the given arguments.
///
/// Uses two single-bucket indices to compute log10(numerator_score / denominator_score)
/// for each read, with fast-path optimizations for reads that only need one index.
///
/// Flow per batch:
/// 1. Classify all reads against numerator (threshold=0.0)
/// 2. Partition: num_score >= skip_threshold → +inf (NumHigh); all others need denominator
/// 3. Classify remaining reads against denominator
/// 4. Merge fast-path + exact results, format and write
pub fn run_log_ratio(args: ClassifyLogRatioArgs) -> Result<()> {
    // Check for Parquet input early (needed for memory estimation and validation)
    let input_is_parquet = is_parquet_input(&args.r1);

    // Validate numerator_skip_threshold range
    if let Some(thresh) = args.numerator_skip_threshold {
        if thresh <= 0.0 || thresh > 1.0 {
            return Err(anyhow!(
                "--numerator-skip-threshold must be between 0.0 (exclusive) and 1.0 (inclusive), got: {}",
                thresh
            ));
        }
    }

    // Validate input configuration
    validate_input_config(input_is_parquet, args.r2.as_ref())?;
    validate_seq_output(
        input_is_parquet,
        args.trim_to.is_some(),
        args.output_sequences.as_deref(),
    )?;

    // Load both indices
    let load_opts = IndexLoadOptions {
        use_bloom_filter: args.use_bloom_filter,
        parallel_rg: args.parallel_rg,
    };
    let num_loaded = load_index_for_classification(&args.numerator, &load_opts)?;
    let denom_loaded = load_index_for_classification(&args.denominator, &load_opts)?;

    // Validate each index has exactly 1 bucket
    let (_num_bucket_id, num_bucket_name) =
        validate_single_bucket_index(&num_loaded.metadata.bucket_names)?;
    let (_denom_bucket_id, denom_bucket_name) =
        validate_single_bucket_index(&denom_loaded.metadata.bucket_names)?;

    // Validate compatible k/w/salt
    validate_compatible_indices(&num_loaded.metadata, &denom_loaded.metadata)?;

    // Compute effective batch size (use max of numerator/denominator shard sizes)
    let batch_result = compute_effective_batch_size(&BatchSizeConfig {
        batch_size_override: args.batch_size,
        max_memory: args.max_memory,
        r1_path: &args.r1,
        r2_path: args.r2.as_deref(),
        is_parquet_input: input_is_parquet,
        index_path: &args.numerator,
        trim_to: args.trim_to,
        minimum_length: args.minimum_length,
        is_log_ratio: true,
        denominator_index_path: Some(&args.denominator),
    })?;
    let effective_batch_size = batch_result.batch_size;

    if args.batch_size.is_none() {
        log::info!(
            "Adaptive batch sizing: batch_size={}, threads={}, estimated peak memory={}, shard_reservation={}, format={:?}",
            batch_result.batch_size,
            rayon::current_num_threads(),
            format_bytes(batch_result.peak_memory),
            format_bytes(batch_result.shard_reservation),
            batch_result.input_format
        );
    }

    // Format bucket name for output: "log10([num_name] / [denom_name])"
    let ratio_bucket_name = format_log_ratio_bucket_name(&num_bucket_name, &denom_bucket_name);

    // Set up output writer and write header
    let output_format = OutputFormat::detect(args.output.as_ref());
    let mut out_writer = OutputWriter::new_long(output_format, args.output.as_ref(), None, true)?;
    out_writer.write_header(b"read_id\tbucket_name\tscore\tfast_path\n")?;

    // Set up passing read tracker if --output-sequences
    // Sequences are written post-classification by re-walking the input file,
    // so we only need a compact bitset during classification.
    let mut passing_tracker = if args.output_sequences.is_some() {
        // Initial capacity is zero; the bitset grows as reads are marked.
        // 100M reads = ~12.5MB, so growth overhead is negligible.
        Some(PassingReadTracker::with_capacity(0))
    } else {
        None
    };

    // Create input reader
    // No quality capture needed — sequences are written via post-classification re-walk
    let mut input_reader = create_input_reader(
        &InputReaderConfig {
            r1_path: &args.r1,
            r2_path: args.r2.as_ref(),
            batch_size: effective_batch_size,
            parallel_input_rg: args.parallel_input_rg,
            is_parquet: input_is_parquet,
            trim_to: args.trim_to,
            minimum_length: args.minimum_length,
        },
        false,
    )?;

    let num_sharded = &num_loaded.sharded;
    let denom_sharded = &denom_loaded.sharded;
    let num_read_options = num_loaded.read_options.as_ref();
    let denom_read_options = denom_loaded.read_options.as_ref();
    let parallel_rg = args.parallel_rg;
    let numerator_skip_threshold = args.numerator_skip_threshold;
    let passing_is_positive = args.passing_is_positive;

    let mut total_reads = 0;
    let mut batch_num = 0;

    // Deferred denominator buffer: accumulate needs-denom reads across batches
    // and only classify against denom when buffer is large enough to amortize I/O cost.
    //
    // Memory overhead: the buffer stores cached minimizers (not sequences), so memory
    // is proportional to minimizer count (~8 bytes × ~6 minimizers/read for short reads)
    // rather than sequence length. Much smaller than the previous OwnedFastxRecord approach.
    let deferred_threshold = effective_batch_size;
    let mut deferred_buffer = DeferredDenomBuffer::new(deferred_threshold.max(1));
    let mut global_read_offset: usize = 0;

    log::info!(
        "Starting log-ratio classification: numerator={}, denominator={} (batch_size={}, deferred_threshold={})",
        num_bucket_name,
        denom_bucket_name,
        effective_batch_size,
        deferred_threshold.max(1)
    );

    // Context for log-ratio batch processing (immutable configuration)
    let ctx = LogRatioContext {
        num_sharded,
        num_read_options,
        denom_sharded,
        denom_read_options,
        parallel_rg,
        numerator_skip_threshold,
        passing_is_positive,
        ratio_bucket_name: &ratio_bucket_name,
    };

    loop {
        let t_io_read = std::time::Instant::now();

        match &mut input_reader {
            ClassificationInput::Parquet(reader) => {
                let log_ratio_needs_owned = args.trim_to.is_some() || args.minimum_length.is_some();
                if log_ratio_needs_owned {
                    // Owned-copy path: reader thread trims/filters, we accumulate
                    let result = accumulate_owned_batches(reader, effective_batch_size)?;

                    log_timing("batch: io_read+trim", t_io_read.elapsed().as_millis());

                    if result.records.is_empty() {
                        break;
                    }

                    let is_final_batch = result.reached_end;
                    batch_num += 1;
                    let batch_read_count = result.records.len();
                    total_reads += batch_read_count;

                    let batch_refs: Vec<rype::QueryRecord> = result
                        .records
                        .iter()
                        .map(|rec| (rec.query_id, rec.seq1.as_slice(), rec.seq2.as_deref()))
                        .collect();

                    let (fast_path_count, needs_denom_count) = process_log_ratio_batch(
                        &ctx,
                        &batch_refs,
                        &result.headers,
                        batch_read_count,
                        &mut deferred_buffer,
                        &mut out_writer,
                        &mut passing_tracker,
                        &mut global_read_offset,
                        batch_num,
                    )?;

                    log::info!(
                        "Processed batch {} ({} row groups): {} reads ({} fast-path, {} deferred, {} total)",
                        batch_num, result.rg_count, batch_read_count,
                        fast_path_count, needs_denom_count, total_reads
                    );

                    if is_final_batch {
                        break;
                    }
                } else {
                    // Zero-copy Parquet path (no trimming)
                    let mut stacked_batches: Vec<(RecordBatch, Vec<String>)> = Vec::new();
                    let mut stacked_rows = 0usize;
                    let mut reached_end = false;

                    loop {
                        let batch_opt = reader.next_batch()?;
                        let Some(parquet_batch) = batch_opt else {
                            reached_end = true;
                            break;
                        };
                        let (record_batch, headers) = parquet_batch.into_arrow();

                        let batch_rows = record_batch.num_rows();
                        stacked_rows += batch_rows;
                        stacked_batches.push((record_batch, headers));

                        if stacked_rows >= effective_batch_size {
                            break;
                        }
                    }

                    log_timing("batch: io_read", t_io_read.elapsed().as_millis());

                    if stacked_batches.is_empty() {
                        break;
                    }

                    let is_final_batch = reached_end;
                    let num_stacked = stacked_batches.len();
                    batch_num += 1;
                    total_reads += stacked_rows;

                    let t_convert = std::time::Instant::now();
                    let (batch_refs, headers) = stacked_batches_to_records(&stacked_batches)?;
                    log_timing("batch: convert_refs", t_convert.elapsed().as_millis());

                    let (fast_path_count, needs_denom_count) = process_log_ratio_batch(
                        &ctx,
                        &batch_refs,
                        &headers,
                        stacked_rows,
                        &mut deferred_buffer,
                        &mut out_writer,
                        &mut passing_tracker,
                        &mut global_read_offset,
                        batch_num,
                    )?;

                    log::info!(
                        "Processed batch {} ({} batches stacked): {} reads ({} fast-path, {} deferred, {} total)",
                        batch_num, num_stacked, stacked_rows,
                        fast_path_count, needs_denom_count, total_reads
                    );

                    if is_final_batch {
                        break;
                    }
                }
            }
            ClassificationInput::Fastx(io) => {
                let batch_opt = io.next_batch()?;
                log_timing("batch: io_read", t_io_read.elapsed().as_millis());

                let Some((owned_records, headers)) = batch_opt else {
                    break;
                };

                batch_num += 1;
                let batch_read_count = owned_records.len();
                total_reads += batch_read_count;

                let t_convert = std::time::Instant::now();
                let batch_refs: Vec<rype::QueryRecord> = owned_records
                    .iter()
                    .map(|rec| (rec.query_id, rec.seq1.as_slice(), rec.seq2.as_deref()))
                    .collect();
                log_timing("batch: convert_refs", t_convert.elapsed().as_millis());

                let (fast_path_count, needs_denom_count) = process_log_ratio_batch(
                    &ctx,
                    &batch_refs,
                    &headers,
                    batch_read_count,
                    &mut deferred_buffer,
                    &mut out_writer,
                    &mut passing_tracker,
                    &mut global_read_offset,
                    batch_num,
                )?;

                log::info!(
                    "Processed batch {}: {} reads ({} fast-path, {} deferred, {} total)",
                    batch_num,
                    batch_read_count,
                    fast_path_count,
                    needs_denom_count,
                    total_reads
                );
            }
        }
    }

    // Flush any remaining deferred reads after all batches
    if !deferred_buffer.is_empty() {
        log::info!(
            "Flushing {} remaining deferred reads (~{}) after final batch (batch {})",
            deferred_buffer.len(),
            format_bytes(deferred_buffer.approx_bytes()),
            batch_num
        );
        let t_drain = std::time::Instant::now();
        let (entries, metadata) = deferred_buffer.drain();
        log_timing("deferred: drain_sort", t_drain.elapsed().as_millis());
        flush_deferred_denom(
            entries,
            metadata,
            ctx.denom_sharded,
            ctx.denom_read_options,
            ctx.parallel_rg,
            ctx.ratio_bucket_name,
            &mut out_writer,
            passing_tracker.as_mut(),
            ctx.passing_is_positive,
        )?;
    }

    log::info!(
        "Log-ratio classification complete: {} reads processed",
        total_reads
    );
    out_writer.finish()?;

    // Post-classification: re-walk input file and write passing sequences
    if let (Some(tracker), Some(ref output_seq_path)) = (passing_tracker, &args.output_sequences) {
        let paired = args.r2.is_some();
        log::info!(
            "Writing {} passing sequences to {:?}",
            tracker.count(),
            output_seq_path
        );
        let written = rewalk_and_write_passing(
            &args.r1,
            args.r2.as_deref(),
            input_is_parquet,
            &tracker,
            output_seq_path,
            paired,
            total_reads,
        )?;
        log::info!("Wrote {} passing sequences", written);
    }

    input_reader.finish()?;

    Ok(())
}

/// Result of `classify_numerator_and_partition`, including cached minimizers.
pub struct NumeratorResult {
    pub partition: PartitionResult,
    /// Extracted minimizers for all reads in the batch, indexed by query_id.
    /// Minimizers for needs-denom reads can be moved out via `std::mem::take`.
    pub extracted: Vec<(Vec<u64>, Vec<u64>)>,
}

/// Classify all reads against the numerator index and partition into fast-path vs needs-denom.
///
/// Extracts minimizers once and caches them in the returned `NumeratorResult`.
/// Needs-denom reads can reuse these cached minimizers for denominator classification,
/// avoiding redundant re-extraction.
pub fn classify_numerator_and_partition(
    num_sharded: &ShardedInvertedIndex,
    batch_refs: &[rype::QueryRecord],
    total_reads: usize,
    num_read_options: Option<&rype::ParquetReadOptions>,
    parallel_rg: bool,
    numerator_skip_threshold: Option<f64>,
) -> Result<NumeratorResult> {
    let manifest = num_sharded.manifest();

    // Extract minimizers once — cached for reuse
    let t_extract = std::time::Instant::now();
    let extracted =
        rype::extract_batch_minimizers(manifest.k, manifest.w, manifest.salt, None, batch_refs);
    log_timing("batch: extraction", t_extract.elapsed().as_millis());

    let query_ids: Vec<i64> = batch_refs.iter().map(|(id, _, _)| *id).collect();

    // Classify from extracted minimizers
    let t_num = std::time::Instant::now();
    let num_results = if parallel_rg {
        rype::classify_from_extracted_minimizers_parallel_rg(
            num_sharded,
            &extracted,
            &query_ids,
            0.0,
            num_read_options,
        )?
    } else {
        rype::classify_from_extracted_minimizers(
            num_sharded,
            &extracted,
            &query_ids,
            0.0,
            num_read_options,
        )?
    };
    log_timing("batch: classify_numerator", t_num.elapsed().as_millis());

    let partition =
        partition_by_numerator_score(&num_results, total_reads, numerator_skip_threshold);

    log::debug!(
        "Partitioned {} reads: {} fast-path, {} need denominator",
        total_reads,
        partition.fast_path_results.len(),
        partition.needs_denom_query_ids.len()
    );

    Ok(NumeratorResult {
        partition,
        extracted,
    })
}

/// Flush a batch of deferred-denom reads: classify against denominator using cached
/// minimizers, merge results, format output, and write. Optionally marks passing reads
/// in the `PassingReadTracker` for post-classification sequence output.
///
/// Each deferred read has its numerator score and pre-extracted minimizers stored.
/// This function:
/// 1. Assigns fresh 0-based query IDs to the buffered reads
/// 2. Classifies them against the denominator index using cached minimizers (no re-extraction)
/// 3. Computes log-ratios from numerator + denominator scores
/// 4. Formats and writes log-ratio output
/// 5. If `passing_tracker` is provided, marks passing reads in the bitset
///
/// Returns the number of reads flushed.
#[allow(clippy::too_many_arguments)]
pub fn flush_deferred_denom(
    entries: Vec<(u64, u32)>,
    metadata: Vec<super::helpers::DeferredMeta>,
    denom_sharded: &ShardedInvertedIndex,
    denom_read_options: Option<&rype::ParquetReadOptions>,
    parallel_rg: bool,
    ratio_bucket_name: &str,
    out_writer: &mut super::helpers::OutputWriter,
    passing_tracker: Option<&mut super::helpers::PassingReadTracker>,
    passing_is_positive: bool,
) -> Result<usize> {
    use super::helpers::{compute_log_ratio, format_log_ratio_output, FastPath, LogRatioResult};

    if metadata.is_empty() {
        return Ok(0);
    }

    let t_flush_total = std::time::Instant::now();
    let count = metadata.len();

    // Build QueryInvertedIndex directly from sorted COO entries
    let fwd_counts: Vec<u32> = metadata.iter().map(|m| m.fwd_count).collect();
    let rc_counts: Vec<u32> = metadata.iter().map(|m| m.rc_count).collect();

    let t_build = std::time::Instant::now();
    let query_idx = rype::QueryInvertedIndex::from_sorted_coo(entries, fwd_counts, rc_counts);
    log_timing("deferred: build_query_index", t_build.elapsed().as_millis());

    let query_ids: Vec<i64> = (0..count as i64).collect();

    // Classify against denominator using pre-built query index (no re-extraction!)
    let t_denom = std::time::Instant::now();
    let denom_results = if parallel_rg {
        rype::classify_from_query_index_parallel_rg(
            denom_sharded,
            &query_idx,
            &query_ids,
            0.0,
            denom_read_options,
        )?
    } else {
        rype::classify_from_query_index(
            denom_sharded,
            &query_idx,
            &query_ids,
            0.0,
            denom_read_options,
        )?
    };
    log_timing(
        "deferred: classify_denominator",
        t_denom.elapsed().as_millis(),
    );

    // Build denom score map
    let mut denom_score_map: HashMap<i64, f64> = HashMap::with_capacity(denom_results.len());
    for hit in &denom_results {
        denom_score_map.insert(hit.query_id, hit.score);
    }

    // Compute log-ratio for each deferred read
    let mut results: Vec<LogRatioResult> = Vec::with_capacity(count);
    for (i, meta) in metadata.iter().enumerate() {
        let denom_score = denom_score_map.get(&(i as i64)).copied().unwrap_or(0.0);
        let log_ratio = compute_log_ratio(meta.num_score, denom_score);
        results.push(LogRatioResult {
            query_id: i as i64,
            log_ratio,
            fast_path: FastPath::None,
        });
    }

    // Mark passing reads in tracker for post-classification sequence output
    if let Some(tracker) = passing_tracker {
        for (i, lr) in results.iter().enumerate() {
            let passes = if passing_is_positive {
                lr.log_ratio > 0.0 || lr.log_ratio.is_nan()
            } else {
                lr.log_ratio <= 0.0 || lr.log_ratio.is_nan()
            };
            if passes {
                tracker.mark(metadata[i].global_index);
            }
        }
    }

    // Build header refs for formatting
    let header_refs: Vec<&str> = metadata.iter().map(|m| m.header.as_str()).collect();

    // Format and write output
    let t_format = std::time::Instant::now();
    let chunk = format_log_ratio_output(&results, &header_refs, ratio_bucket_name);
    log_timing("deferred: format_output", t_format.elapsed().as_millis());

    let t_write = std::time::Instant::now();
    out_writer.write_chunk(chunk)?;
    log_timing("deferred: io_write", t_write.elapsed().as_millis());

    log_timing("deferred: flush_total", t_flush_total.elapsed().as_millis());
    log::info!("Flushed {} deferred-denom reads", count);

    Ok(count)
}

/// Immutable configuration for log-ratio batch processing.
///
/// Groups the many parameters that stay constant across batches into a single
/// struct, reducing argument count in `process_log_ratio_batch`.
struct LogRatioContext<'a> {
    num_sharded: &'a ShardedInvertedIndex,
    num_read_options: Option<&'a rype::ParquetReadOptions>,
    denom_sharded: &'a ShardedInvertedIndex,
    denom_read_options: Option<&'a rype::ParquetReadOptions>,
    parallel_rg: bool,
    numerator_skip_threshold: Option<f64>,
    passing_is_positive: bool,
    ratio_bucket_name: &'a str,
}

/// Process a single batch for log-ratio classification.
///
/// Classifies against numerator, partitions into fast-path/needs-denom,
/// marks passing reads in tracker, writes fast-path results immediately,
/// and buffers needs-denom reads for later denominator classification.
///
/// Returns (fast_path_count, needs_denom_count).
#[allow(clippy::too_many_arguments)]
fn process_log_ratio_batch<S: AsRef<str>>(
    ctx: &LogRatioContext,
    batch_refs: &[rype::QueryRecord],
    headers: &[S],
    batch_read_count: usize,
    deferred_buffer: &mut DeferredDenomBuffer,
    out_writer: &mut super::helpers::OutputWriter,
    passing_tracker: &mut Option<super::helpers::PassingReadTracker>,
    global_read_offset: &mut usize,
    batch_num: usize,
) -> Result<(usize, usize)> {
    let NumeratorResult {
        partition,
        mut extracted,
    } = classify_numerator_and_partition(
        ctx.num_sharded,
        batch_refs,
        batch_read_count,
        ctx.num_read_options,
        ctx.parallel_rg,
        ctx.numerator_skip_threshold,
    )?;

    let fast_path_count = partition.fast_path_results.len();
    let needs_denom_count = partition.needs_denom_query_ids.len();

    // Mark passing fast-path reads in tracker
    if let Some(ref mut tracker) = passing_tracker {
        for lr in &partition.fast_path_results {
            let passes = if ctx.passing_is_positive {
                lr.log_ratio > 0.0 || lr.log_ratio.is_nan()
            } else {
                lr.log_ratio <= 0.0 || lr.log_ratio.is_nan()
            };
            if passes {
                tracker.mark(*global_read_offset + lr.query_id as usize);
            }
        }
    }

    // Write fast-path results immediately
    if !partition.fast_path_results.is_empty() {
        let t_format = std::time::Instant::now();
        let chunk =
            format_log_ratio_output(&partition.fast_path_results, headers, ctx.ratio_bucket_name);
        log_timing("batch: format_fast_path", t_format.elapsed().as_millis());

        let t_write = std::time::Instant::now();
        out_writer.write_chunk(chunk)?;
        log_timing("batch: io_write_fast_path", t_write.elapsed().as_millis());
    }

    // Push needs-denom reads into deferred buffer (cache minimizers, not sequences)
    for &qid in &partition.needs_denom_query_ids {
        let idx = qid as usize;
        debug_assert!(
            idx < headers.len() && idx < partition.num_scores.len() && idx < extracted.len(),
            "query_id {} out of bounds (headers={}, scores={}, extracted={})",
            idx,
            headers.len(),
            partition.num_scores.len(),
            extracted.len()
        );
        let (fwd, rc) = std::mem::take(&mut extracted[idx]);
        deferred_buffer.push(
            headers[idx].as_ref().to_string(),
            partition.num_scores[idx],
            *global_read_offset + idx,
            fwd,
            rc,
        );
    }

    *global_read_offset += batch_read_count;

    // Flush deferred buffer if threshold reached
    if deferred_buffer.should_flush() {
        log::info!(
            "Deferred buffer reached threshold ({} reads, ~{}), flushing (triggered by batch {})",
            deferred_buffer.len(),
            format_bytes(deferred_buffer.approx_bytes()),
            batch_num
        );
        let t_drain = std::time::Instant::now();
        let (entries, metadata) = deferred_buffer.drain();
        log_timing("deferred: drain_sort", t_drain.elapsed().as_millis());
        flush_deferred_denom(
            entries,
            metadata,
            ctx.denom_sharded,
            ctx.denom_read_options,
            ctx.parallel_rg,
            ctx.ratio_bucket_name,
            out_writer,
            passing_tracker.as_mut(),
            ctx.passing_is_positive,
        )?;
    }

    Ok((fast_path_count, needs_denom_count))
}

/// Build the header line and sorted bucket IDs for wide-format output.
///
/// Returns a tuple of:
/// - Header bytes: "read_id\tBucket1\tBucket2\t...\n" (tab-separated, newline-terminated)
/// - Sorted bucket IDs: Vec<u32> in ascending order (for formatting results)
///
/// Bucket columns are ordered by bucket_id ascending but display bucket_name.
pub fn build_wide_header(bucket_names: &HashMap<u32, String>) -> (Vec<u8>, Vec<u32>) {
    // Sort bucket IDs ascending
    let mut bucket_ids: Vec<u32> = bucket_names.keys().copied().collect();
    bucket_ids.sort_unstable();

    // Build header: "read_id\tBucket1\tBucket2\t...\n"
    let mut header = Vec::with_capacity(256);
    header.extend_from_slice(b"read_id");
    for &bucket_id in &bucket_ids {
        header.push(b'\t');
        if let Some(name) = bucket_names.get(&bucket_id) {
            header.extend_from_slice(name.as_bytes());
        }
    }
    header.push(b'\n');

    (header, bucket_ids)
}

/// Format classification results in wide format for TSV output.
///
/// Each **processed** read produces one row with scores for all buckets (0.0 if no hit).
/// Scores are formatted to 4 decimal places.
///
/// # Output Behavior
///
/// Only reads that were actually processed by classification are output.
/// Reads that were skipped (e.g., too short after `--trim-to`) are omitted entirely.
///
/// With wide format (threshold=0.0), any processed read will have results for all
/// buckets. Reads with no results at all were skipped and are not included.
///
/// # Arguments
/// * `results` - Classification results (may have multiple entries per read)
/// * `headers` - Read names/IDs indexed by query_id (accepts `&[String]` or `&[&str]`)
/// * `bucket_ids` - Sorted bucket IDs defining column order
///
/// # Returns
/// Formatted bytes: "read_id\tscore1\tscore2\t...\n" for each processed read
pub fn format_results_wide<S: AsRef<str>>(
    results: &[rype::HitResult],
    headers: &[S],
    bucket_ids: &[u32],
) -> Vec<u8> {
    use std::io::Write;

    // Group results by query_id: query_id -> (bucket_id -> score)
    let mut scores_by_query: HashMap<i64, HashMap<u32, f64>> = HashMap::new();
    for res in results {
        scores_by_query
            .entry(res.query_id)
            .or_default()
            .insert(res.bucket_id, res.score);
    }

    let num_buckets = bucket_ids.len();
    let mut output = Vec::with_capacity(headers.len() * (num_buckets * 8 + 32));

    // Output one row per processed read (skip reads with no results - they were skipped)
    for (query_id, header) in headers.iter().enumerate() {
        let Some(query_scores) = scores_by_query.get(&(query_id as i64)) else {
            // Read was skipped (e.g., too short after trim_to) - omit from output
            continue;
        };

        output.extend_from_slice(header.as_ref().as_bytes());

        for &bucket_id in bucket_ids {
            output.push(b'\t');
            let score = query_scores.get(&bucket_id).copied().unwrap_or(0.0);
            write!(&mut output, "{:.4}", score).unwrap();
        }
        output.push(b'\n');
    }

    output
}

/// Format classification results in wide format for TSV output (borrowed headers variant).
///
/// This is an alias for `format_results_wide` that accepts `&[&str]` directly.
/// Kept for backwards compatibility and explicit type annotation.
#[inline]
pub fn format_results_wide_ref(
    results: &[rype::HitResult],
    headers: &[&str],
    bucket_ids: &[u32],
) -> Vec<u8> {
    format_results_wide(results, headers, bucket_ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_wide_header_produces_correct_header() {
        let mut bucket_names = HashMap::new();
        bucket_names.insert(1, "Bucket_A".to_string());
        bucket_names.insert(2, "Bucket_B".to_string());
        bucket_names.insert(3, "Bucket_C".to_string());

        let (header, bucket_ids) = build_wide_header(&bucket_names);

        let header_str = String::from_utf8(header).unwrap();
        assert_eq!(header_str, "read_id\tBucket_A\tBucket_B\tBucket_C\n");
        assert_eq!(bucket_ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_build_wide_header_orders_by_bucket_id_ascending() {
        let mut bucket_names = HashMap::new();
        // Insert in non-sorted order
        bucket_names.insert(10, "Z_last".to_string());
        bucket_names.insert(1, "A_first".to_string());
        bucket_names.insert(5, "M_middle".to_string());

        let (header, bucket_ids) = build_wide_header(&bucket_names);

        let header_str = String::from_utf8(header).unwrap();
        // Columns should be ordered by bucket_id (1, 5, 10), not alphabetically
        assert_eq!(header_str, "read_id\tA_first\tM_middle\tZ_last\n");
        assert_eq!(bucket_ids, vec![1, 5, 10]);
    }

    #[test]
    fn test_build_wide_header_empty_bucket_names() {
        let bucket_names = HashMap::new();

        let (header, bucket_ids) = build_wide_header(&bucket_names);

        let header_str = String::from_utf8(header).unwrap();
        assert_eq!(header_str, "read_id\n");
        assert!(bucket_ids.is_empty());
    }

    #[test]
    fn test_build_wide_header_single_bucket() {
        let mut bucket_names = HashMap::new();
        bucket_names.insert(42, "OnlyBucket".to_string());

        let (header, bucket_ids) = build_wide_header(&bucket_names);

        let header_str = String::from_utf8(header).unwrap();
        assert_eq!(header_str, "read_id\tOnlyBucket\n");
        assert_eq!(bucket_ids, vec![42]);
    }

    // Phase 3: format_results_wide tests

    #[test]
    fn test_format_results_wide_all_buckets_have_scores() {
        use rype::HitResult;

        let results = vec![
            HitResult {
                query_id: 0,
                bucket_id: 1,
                score: 0.85,
            },
            HitResult {
                query_id: 0,
                bucket_id: 2,
                score: 0.75,
            },
            HitResult {
                query_id: 0,
                bucket_id: 3,
                score: 0.65,
            },
        ];
        let headers = vec!["read_1".to_string()];
        let bucket_ids = vec![1, 2, 3];

        let output = format_results_wide(&results, &headers, &bucket_ids);
        let output_str = String::from_utf8(output).unwrap();

        assert_eq!(output_str, "read_1\t0.8500\t0.7500\t0.6500\n");
    }

    #[test]
    fn test_format_results_wide_partial_results_fills_zeros() {
        use rype::HitResult;

        // Read only has scores for buckets 1 and 3, missing bucket 2
        let results = vec![
            HitResult {
                query_id: 0,
                bucket_id: 1,
                score: 0.85,
            },
            HitResult {
                query_id: 0,
                bucket_id: 3,
                score: 0.32,
            },
        ];
        let headers = vec!["read_1".to_string()];
        let bucket_ids = vec![1, 2, 3];

        let output = format_results_wide(&results, &headers, &bucket_ids);
        let output_str = String::from_utf8(output).unwrap();

        // Bucket 2 should have 0.0000
        assert_eq!(output_str, "read_1\t0.8500\t0.0000\t0.3200\n");
    }

    #[test]
    fn test_format_results_wide_no_results_skips_read() {
        // A read with no results was skipped (e.g., too short after trim_to)
        // and should NOT appear in the output
        use rype::HitResult;

        let results: Vec<HitResult> = vec![];
        let headers = vec!["read_1".to_string()];
        let bucket_ids = vec![1, 2, 3];

        let output = format_results_wide(&results, &headers, &bucket_ids);
        let output_str = String::from_utf8(output).unwrap();

        // Skipped reads produce empty output
        assert_eq!(output_str, "");
    }

    #[test]
    fn test_format_results_wide_multiple_reads() {
        use rype::HitResult;

        let results = vec![
            HitResult {
                query_id: 0,
                bucket_id: 1,
                score: 0.85,
            },
            HitResult {
                query_id: 0,
                bucket_id: 3,
                score: 0.32,
            },
            HitResult {
                query_id: 1,
                bucket_id: 2,
                score: 0.91,
            },
        ];
        let headers = vec!["read_1".to_string(), "read_2".to_string()];
        let bucket_ids = vec![1, 2, 3];

        let output = format_results_wide(&results, &headers, &bucket_ids);
        let output_str = String::from_utf8(output).unwrap();

        let expected = "read_1\t0.8500\t0.0000\t0.3200\nread_2\t0.0000\t0.9100\t0.0000\n";
        assert_eq!(output_str, expected);
    }

    #[test]
    fn test_format_results_wide_scores_formatted_to_4_decimals() {
        use rype::HitResult;

        let results = vec![HitResult {
            query_id: 0,
            bucket_id: 1,
            score: 0.123456789,
        }];
        let headers = vec!["read_1".to_string()];
        let bucket_ids = vec![1];

        let output = format_results_wide(&results, &headers, &bucket_ids);
        let output_str = String::from_utf8(output).unwrap();

        // Should be rounded to 4 decimal places
        assert_eq!(output_str, "read_1\t0.1235\n");
    }

    #[test]
    fn test_format_results_wide_ref_works_with_str_refs() {
        use rype::HitResult;

        let results = vec![
            HitResult {
                query_id: 0,
                bucket_id: 1,
                score: 0.85,
            },
            HitResult {
                query_id: 0,
                bucket_id: 2,
                score: 0.75,
            },
        ];
        let headers: Vec<&str> = vec!["read_1"];
        let bucket_ids = vec![1, 2];

        let output = format_results_wide_ref(&results, &headers, &bucket_ids);
        let output_str = String::from_utf8(output).unwrap();

        assert_eq!(output_str, "read_1\t0.8500\t0.7500\n");
    }

    // validate_seq_output tests

    #[test]
    fn test_validate_seq_output_accepts_parquet_input() {
        use std::path::Path;

        // Parquet input is now supported via post-classification re-walk
        let result = validate_seq_output(true, false, Some(Path::new("out.fastq.gz")));
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_seq_output_rejects_trim_to() {
        use std::path::Path;

        let result = validate_seq_output(false, true, Some(Path::new("out.fastq.gz")));
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("--trim-to"));
    }

    #[test]
    fn test_validate_seq_output_accepts_valid_config() {
        use std::path::Path;

        let result = validate_seq_output(false, false, Some(Path::new("out.fastq.gz")));
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_seq_output_accepts_none_output() {
        // No output_sequences is always valid
        let result = validate_seq_output(true, true, None);
        assert!(result.is_ok());
    }
}
