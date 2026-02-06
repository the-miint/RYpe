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
    classify_with_sharded_negative, filter_best_hits, log_timing, ShardedInvertedIndex,
};

use super::helpers::{
    compute_effective_batch_size, compute_log_ratio_from_hits, create_input_reader,
    filter_log_ratios_by_threshold, format_classification_results, format_log_ratio_bucket_name,
    format_log_ratio_output, is_parquet_input, load_index_for_classification,
    stacked_batches_to_records, validate_input_config, BatchSizeConfig, ClassificationInput,
    IndexLoadOptions, InputReaderConfig, OutputFormat, OutputWriter,
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
    })?;
    let effective_batch_size = batch_result.batch_size;

    // Log adaptive batch sizing details (only for auto-computed batch sizes)
    if args.common.batch_size.is_none() {
        log::info!(
            "Adaptive batch sizing: batch_size={}, parallel_batches={}, threads={}, estimated peak memory={}, format={:?}",
            batch_result.batch_size,
            batch_result.batch_count,
            rayon::current_num_threads(),
            format_bytes(batch_result.peak_memory),
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

    // For wide format, use threshold 0.0 to get all bucket scores
    let effective_threshold = if args.wide {
        0.0
    } else {
        args.common.threshold
    };

    // Create input reader (Parquet or FASTX)
    let mut input_reader = create_input_reader(&InputReaderConfig {
        r1_path: &args.common.r1,
        r2_path: args.common.r2.as_ref(),
        batch_size: effective_batch_size,
        parallel_input_rg: args.common.parallel_input_rg,
        is_parquet: input_is_parquet,
    })?;

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

    // Helper closure for classification
    // Note: uses effective_threshold (0.0 for wide format, args.common.threshold otherwise)
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
                args.common.trim_to,
            )
        } else if args.common.parallel_rg {
            classify_batch_sharded_parallel_rg(
                &sharded,
                None,
                batch_refs,
                effective_threshold,
                read_options.as_ref(),
                args.common.trim_to,
            )
        } else {
            classify_batch_sharded_merge_join(
                &sharded,
                None,
                batch_refs,
                effective_threshold,
                read_options.as_ref(),
                args.common.trim_to,
            )
        }
    };

    loop {
        let t_io_read = std::time::Instant::now();

        // Handle Parquet and FASTX differently for zero-copy optimization
        match &mut input_reader {
            ClassificationInput::Parquet(reader) => {
                // Zero-copy Parquet path with batch stacking
                let mut stacked_batches: Vec<(RecordBatch, Vec<String>)> = Vec::new();
                let mut stacked_rows = 0usize;
                let mut reached_end = false;

                // Accumulate batches until we have enough rows or run out of data
                loop {
                    let batch_opt = reader.next_batch()?;
                    let Some((record_batch, headers)) = batch_opt else {
                        reached_end = true;
                        break;
                    };

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
            ClassificationInput::Fastx(io) => {
                // FASTX path (copies sequences)
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
                    .map(|(id, s1, s2)| (*id, s1.as_slice(), s2.as_deref()))
                    .collect();
                log_timing("batch: convert_refs", t_convert.elapsed().as_millis());

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
pub struct ClassifyLogRatioArgs {
    pub common: CommonClassifyArgs,
    pub swap_buckets: bool,
}

/// Validate that the index has exactly two buckets and return their IDs and names.
///
/// Returns (numerator_id, denominator_id, numerator_name, denominator_name).
/// By default, the lower bucket_id is the numerator.
fn validate_two_bucket_index(
    bucket_names: &HashMap<u32, String>,
) -> Result<(u32, u32, String, String)> {
    if bucket_names.len() != 2 {
        return Err(anyhow!(
            "log-ratio mode requires an index with exactly 2 buckets, but found {}.\n\
             Use 'rype index stats -i <index>' to see bucket information.",
            bucket_names.len()
        ));
    }

    // Get bucket IDs and sort them
    let mut ids: Vec<u32> = bucket_names.keys().copied().collect();
    ids.sort_unstable();

    let num_id = ids[0];
    let denom_id = ids[1];
    let num_name = bucket_names.get(&num_id).unwrap().clone();
    let denom_name = bucket_names.get(&denom_id).unwrap().clone();

    Ok((num_id, denom_id, num_name, denom_name))
}

/// Run the log-ratio classify command with the given arguments.
///
/// Computes log10(numerator_score / denominator_score) for each read.
/// Requires an index with exactly 2 buckets.
pub fn run_log_ratio(args: ClassifyLogRatioArgs) -> Result<()> {
    // Check for Parquet input early (needed for memory estimation)
    let input_is_parquet = is_parquet_input(&args.common.r1);

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

    // Validate exactly 2 buckets and get their IDs/names
    let (mut num_id, mut denom_id, mut num_name, mut denom_name) =
        validate_two_bucket_index(&metadata.bucket_names)?;

    // Swap buckets if requested
    if args.swap_buckets {
        std::mem::swap(&mut num_id, &mut denom_id);
        std::mem::swap(&mut num_name, &mut denom_name);
        log::info!(
            "Buckets swapped: numerator={} (id={}), denominator={} (id={})",
            num_name,
            num_id,
            denom_name,
            denom_id
        );
    } else {
        log::info!(
            "Buckets: numerator={} (id={}), denominator={} (id={})",
            num_name,
            num_id,
            denom_name,
            denom_id
        );
    }

    // Format the bucket name for output
    let ratio_bucket_name = format_log_ratio_bucket_name(&num_name, &denom_name);

    // Determine effective batch size
    let batch_result = compute_effective_batch_size(&BatchSizeConfig {
        batch_size_override: args.common.batch_size,
        max_memory: args.common.max_memory,
        r1_path: &args.common.r1,
        r2_path: args.common.r2.as_deref(),
        is_parquet_input: input_is_parquet,
        index_path: &args.common.index,
        trim_to: args.common.trim_to,
    })?;
    let effective_batch_size = batch_result.batch_size;

    // Log adaptive batch sizing details (only for auto-computed batch sizes)
    if args.common.batch_size.is_none() {
        log::info!(
            "Adaptive batch sizing: batch_size={}, estimated peak memory={}",
            batch_result.batch_size,
            format_bytes(batch_result.peak_memory)
        );
    }

    // Set up output writer
    let output_format = OutputFormat::detect(args.common.output.as_ref());
    let mut out_writer = OutputWriter::new(output_format, args.common.output.as_ref(), None)?;
    out_writer.write_header(b"read_id\tbucket_name\tscore\n")?;

    // Create input reader
    let mut input_reader = create_input_reader(&InputReaderConfig {
        r1_path: &args.common.r1,
        r2_path: args.common.r2.as_ref(),
        batch_size: effective_batch_size,
        parallel_input_rg: args.common.parallel_input_rg,
        is_parquet: input_is_parquet,
    })?;

    log::info!(
        "Starting log-ratio classification (batch_size={}, threshold={})",
        effective_batch_size,
        args.common.threshold
    );

    let mut total_reads = 0usize;
    let mut batch_num = 0usize;

    // Classification closure - always use threshold 0.0 to get all scores
    let classify_records = |batch_refs: &[rype::QueryRecord]| -> Result<Vec<rype::HitResult>> {
        if args.common.parallel_rg {
            classify_batch_sharded_parallel_rg(
                &sharded,
                None,
                batch_refs,
                0.0, // Always get all scores for log-ratio
                read_options.as_ref(),
                args.common.trim_to,
            )
        } else {
            classify_batch_sharded_merge_join(
                &sharded,
                None,
                batch_refs,
                0.0, // Always get all scores for log-ratio
                read_options.as_ref(),
                args.common.trim_to,
            )
        }
    };

    loop {
        let t_io_read = std::time::Instant::now();

        match &mut input_reader {
            ClassificationInput::Parquet(reader) => {
                // Parquet path with batch stacking
                let mut stacked_batches: Vec<(RecordBatch, Vec<String>)> = Vec::new();
                let mut stacked_rows = 0usize;
                let mut reached_end = false;

                loop {
                    let batch_opt = reader.next_batch()?;
                    let Some((record_batch, headers)) = batch_opt else {
                        reached_end = true;
                        break;
                    };

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

                // Classify to get all scores
                let results = classify_records(&batch_refs)?;

                // Compute log ratios
                let t_ratio = std::time::Instant::now();
                let mut log_ratios = compute_log_ratio_from_hits(&results, num_id, denom_id);
                log_timing("batch: compute_log_ratio", t_ratio.elapsed().as_millis());

                // Filter by threshold if needed (filter if BOTH original scores < threshold)
                if args.common.threshold > 0.0 {
                    filter_log_ratios_by_threshold(
                        &mut log_ratios,
                        &results,
                        num_id,
                        denom_id,
                        args.common.threshold,
                    );
                }

                // Format and write output
                let t_format = std::time::Instant::now();
                let chunk_out = format_log_ratio_output(&log_ratios, &headers, &ratio_bucket_name);
                log_timing("batch: format_output", t_format.elapsed().as_millis());

                let t_write = std::time::Instant::now();
                out_writer.write_chunk(chunk_out)?;
                log_timing("batch: io_write", t_write.elapsed().as_millis());

                log::info!(
                    "Processed batch {}: {} reads ({} total), {} log-ratio results",
                    batch_num,
                    stacked_rows,
                    total_reads,
                    log_ratios.len()
                );

                if is_final_batch {
                    break;
                }
            }
            ClassificationInput::Fastx(io) => {
                // FASTX path
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
                    .map(|(id, s1, s2)| (*id, s1.as_slice(), s2.as_deref()))
                    .collect();
                log_timing("batch: convert_refs", t_convert.elapsed().as_millis());

                // Classify to get all scores
                let results = classify_records(&batch_refs)?;

                // Compute log ratios
                let t_ratio = std::time::Instant::now();
                let mut log_ratios = compute_log_ratio_from_hits(&results, num_id, denom_id);
                log_timing("batch: compute_log_ratio", t_ratio.elapsed().as_millis());

                // Filter by threshold if needed
                if args.common.threshold > 0.0 {
                    filter_log_ratios_by_threshold(
                        &mut log_ratios,
                        &results,
                        num_id,
                        denom_id,
                        args.common.threshold,
                    );
                }

                // Format and write output
                let t_format = std::time::Instant::now();
                let chunk_out = format_log_ratio_output(&log_ratios, &headers, &ratio_bucket_name);
                log_timing("batch: format_output", t_format.elapsed().as_millis());

                let t_write = std::time::Instant::now();
                out_writer.write_chunk(chunk_out)?;
                log_timing("batch: io_write", t_write.elapsed().as_millis());

                log::info!(
                    "Processed batch {}: {} reads ({} total), {} log-ratio results",
                    batch_num,
                    batch_read_count,
                    total_reads,
                    log_ratios.len()
                );
            }
        }
    }

    log::info!(
        "Log-ratio classification complete: {} reads processed",
        total_reads
    );
    out_writer.finish()?;
    input_reader.finish()?;

    Ok(())
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

    // Phase 3: validate_two_bucket_index tests

    #[test]
    fn test_validate_two_bucket_index_passes() {
        let mut bucket_names = HashMap::new();
        bucket_names.insert(0, "BucketA".to_string());
        bucket_names.insert(1, "BucketB".to_string());

        let result = validate_two_bucket_index(&bucket_names);
        assert!(result.is_ok());

        let (num_id, denom_id, num_name, denom_name) = result.unwrap();
        // Lower ID (0) is numerator by default
        assert_eq!(num_id, 0);
        assert_eq!(denom_id, 1);
        assert_eq!(num_name, "BucketA");
        assert_eq!(denom_name, "BucketB");
    }

    #[test]
    fn test_validate_two_bucket_index_fails_one_bucket() {
        let mut bucket_names = HashMap::new();
        bucket_names.insert(0, "OnlyBucket".to_string());

        let result = validate_two_bucket_index(&bucket_names);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("exactly 2 buckets"));
        assert!(err.contains("found 1"));
    }

    #[test]
    fn test_validate_two_bucket_index_fails_three_buckets() {
        let mut bucket_names = HashMap::new();
        bucket_names.insert(0, "A".to_string());
        bucket_names.insert(1, "B".to_string());
        bucket_names.insert(2, "C".to_string());

        let result = validate_two_bucket_index(&bucket_names);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("exactly 2 buckets"));
        assert!(err.contains("found 3"));
    }

    #[test]
    fn test_validate_two_bucket_index_orders_by_id() {
        let mut bucket_names = HashMap::new();
        // Insert with higher ID first
        bucket_names.insert(10, "HigherID".to_string());
        bucket_names.insert(5, "LowerID".to_string());

        let result = validate_two_bucket_index(&bucket_names);
        assert!(result.is_ok());

        let (num_id, denom_id, num_name, denom_name) = result.unwrap();
        // Lower ID (5) should be numerator
        assert_eq!(num_id, 5);
        assert_eq!(denom_id, 10);
        assert_eq!(num_name, "LowerID");
        assert_eq!(denom_name, "HigherID");
    }

    #[test]
    fn test_validate_two_bucket_index_fails_empty() {
        let bucket_names: HashMap<u32, String> = HashMap::new();

        let result = validate_two_bucket_index(&bucket_names);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("exactly 2 buckets"));
        assert!(err.contains("found 0"));
    }
}
