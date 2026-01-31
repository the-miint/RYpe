//! Classify command handlers and helper functions.
//!
//! This module contains the implementation logic for classification commands.

use anyhow::{anyhow, Result};
use arrow::record_batch::RecordBatch;
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;

use rype::memory::{
    calculate_batch_config, detect_available_memory, format_bytes, InputFormat, MemoryConfig,
    MemorySource, ReadMemoryProfile,
};
use rype::parquet_index;
use rype::{
    classify_batch_sharded_merge_join, classify_batch_sharded_parallel_rg,
    classify_with_sharded_negative, filter_best_hits, log_timing, IndexMetadata,
    ShardedInvertedIndex,
};

use super::helpers::{
    is_parquet_input, load_index_metadata, stacked_batches_to_records, OutputFormat, OutputWriter,
    PrefetchingIoHandler, PrefetchingParquetReader,
};

/// Arguments for the classify run command.
pub struct ClassifyRunArgs {
    pub index: PathBuf,
    pub negative_index: Option<PathBuf>,
    pub r1: PathBuf,
    pub r2: Option<PathBuf>,
    pub threshold: f64,
    pub max_memory: usize,
    pub batch_size: Option<usize>,
    pub output: Option<PathBuf>,
    pub parallel_rg: bool,
    pub use_bloom_filter: bool,
    pub parallel_input_rg: usize,
    pub best_hit: bool,
}

/// Run the classify command with the given arguments.
pub fn run_classify(args: ClassifyRunArgs) -> Result<()> {
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
    let input_is_parquet = is_parquet_input(&args.r1);
    let is_paired = args.r2.is_some();

    // Determine effective batch size: user override or adaptive
    let effective_batch_size = if let Some(bs) = args.batch_size {
        log::info!("Using user-specified batch size: {}", bs);
        bs
    } else {
        // Load index metadata to get k, w, num_buckets
        let metadata = load_index_metadata(&args.index)?;

        // Detect or use specified memory limit (0 = auto)
        let mem_limit = if args.max_memory == 0 {
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
            args.max_memory
        };

        // Sample read lengths from input files
        let read_profile = ReadMemoryProfile::from_files(
            &args.r1,
            args.r2.as_deref(),
            1000, // sample size
            metadata.k,
            metadata.w,
        )
        .unwrap_or_else(|| {
            log::warn!("Could not sample read lengths, using default profile");
            ReadMemoryProfile::default_profile(is_paired, metadata.k, metadata.w)
        });

        log::debug!(
            "Read profile: avg_read_length={}, avg_query_length={}, minimizers_per_query={}",
            read_profile.avg_read_length,
            read_profile.avg_query_length,
            read_profile.minimizers_per_query
        );

        // For now, use a simple heuristic since we don't have index loaded yet
        // We'll estimate index memory from metadata
        let estimated_index_mem = metadata.bucket_minimizer_counts.values().sum::<usize>() * 8;
        let num_buckets = metadata.bucket_names.len();

        // Determine input format for accurate memory estimation
        // FASTX uses 2 prefetch slots, Parquet uses 4
        let input_format = if input_is_parquet {
            // For Parquet, paired-end is determined by presence of sequence2 column
            // which we don't know yet - assume based on r2 argument
            InputFormat::Parquet { is_paired }
        } else {
            InputFormat::Fastx { is_paired }
        };

        let mem_config = MemoryConfig {
            max_memory: mem_limit,
            num_threads: rayon::current_num_threads(),
            index_memory: estimated_index_mem,
            shard_reservation: 0, // Will be updated after loading index
            read_profile,
            num_buckets,
            input_format,
        };

        let batch_config = calculate_batch_config(&mem_config);
        log::info!(
            "Adaptive batch sizing: batch_size={}, parallel_batches={}, threads={}, estimated peak memory={}, format={:?}",
            batch_config.batch_size,
            batch_config.batch_count,
            rayon::current_num_threads(),
            format_bytes(batch_config.peak_memory),
            input_format
        );
        batch_config.batch_size
    };
    if input_is_parquet && args.r2.is_some() {
        return Err(anyhow!(
            "Parquet input with separate R2 file is not supported. \
             Use a Parquet file with 'sequence2' column for paired-end data."
        ));
    }

    // Verify index is Parquet format
    if !rype::is_parquet_index(&args.index) {
        return Err(anyhow!(
            "Index not found or not in Parquet format: {}\n\
             Create an index with: rype index create -o index.ryxdi -r refs.fasta",
            args.index.display()
        ));
    }

    // Load metadata from Parquet index
    log::info!("Detected Parquet index at {:?}", args.index);
    let manifest = rype::ParquetManifest::load(&args.index)?;
    let (bucket_names, bucket_sources) = rype::parquet_index::read_buckets_parquet(&args.index)?;
    let metadata = IndexMetadata {
        k: manifest.k,
        w: manifest.w,
        salt: manifest.salt,
        bucket_names,
        bucket_sources,
        bucket_minimizer_counts: HashMap::new(), // Not stored in Parquet manifest
    };

    log::info!("Metadata loaded: {} buckets", metadata.bucket_names.len());

    // Set up I/O based on input format
    let output_format = OutputFormat::detect(args.output.as_ref());
    let mut out_writer = OutputWriter::new(output_format, args.output.as_ref(), None)?;
    out_writer.write_header(b"read_id\tbucket_name\tscore\n")?;

    // Create input reader (Parquet or FASTX)
    let mut parquet_reader = if input_is_parquet {
        let parallel_rg_opt = if args.parallel_input_rg > 0 {
            Some(args.parallel_input_rg)
        } else {
            None
        };
        log::info!(
            "Using prefetching Parquet input reader (zero-copy, batch_size={}, parallel_rg={:?}) for {:?}",
            effective_batch_size,
            parallel_rg_opt,
            args.r1
        );
        Some(PrefetchingParquetReader::with_parallel_row_groups(
            &args.r1,
            effective_batch_size,
            parallel_rg_opt,
        )?)
    } else {
        None
    };
    let mut fastx_io = if !input_is_parquet {
        Some(PrefetchingIoHandler::new(
            &args.r1,
            args.r2.as_ref(),
            None,
            effective_batch_size,
        )?)
    } else {
        None
    };

    let mut total_reads = 0;
    let mut batch_num = 0;

    // Load Parquet inverted index
    log::info!("Loading Parquet inverted index from {:?}", args.index);
    let sharded = ShardedInvertedIndex::open(&args.index)?;

    log::info!(
        "Sharded index: {} shards, {} total minimizers",
        sharded.num_shards(),
        sharded.total_minimizers()
    );

    // Advise kernel to prefetch Parquet shard files
    if args.parallel_rg {
        let available = detect_available_memory();
        let prefetch_budget = available.bytes / 2;
        let advised = sharded.advise_prefetch(Some(prefetch_budget));
        if advised > 0 {
            log::info!(
                "Advised kernel to prefetch {} of index data",
                format_bytes(advised)
            );
        }
    }

    // Build read options for Parquet indices
    let read_options = if args.use_bloom_filter {
        log::info!("Bloom filter row group filtering enabled");
        Some(parquet_index::ParquetReadOptions::with_bloom_filter())
    } else {
        None
    };

    if args.parallel_rg {
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
    let classify_records = |batch_refs: &[rype::QueryRecord]| -> Result<Vec<rype::HitResult>> {
        // If negative index is provided, use memory-efficient sharded filtering
        if let Some(ref neg) = negative_sharded {
            // Negative filtering not supported with parallel-rg
            if args.parallel_rg {
                return Err(anyhow!(
                    "Negative index filtering is not supported with --parallel-rg."
                ));
            }
            classify_with_sharded_negative(
                &sharded,
                Some(neg),
                batch_refs,
                args.threshold,
                read_options.as_ref(),
            )
        } else if args.parallel_rg {
            classify_batch_sharded_parallel_rg(
                &sharded,
                None,
                batch_refs,
                args.threshold,
                read_options.as_ref(),
            )
        } else {
            classify_batch_sharded_merge_join(
                &sharded,
                None,
                batch_refs,
                args.threshold,
                read_options.as_ref(),
            )
        }
    };

    // Helper closure for output formatting - works with &str headers
    let format_results_ref = |results: &[rype::HitResult], headers: &[&str]| {
        let mut chunk_out = Vec::with_capacity(1024);
        for res in results {
            let header = headers[res.query_id as usize];
            let bucket_name = metadata
                .bucket_names
                .get(&res.bucket_id)
                .map(|s| s.as_str())
                .unwrap_or("unknown");
            writeln!(chunk_out, "{}\t{}\t{:.4}", header, bucket_name, res.score).unwrap();
        }
        chunk_out
    };

    // Helper closure for output formatting - works with String headers
    let format_results = |results: &[rype::HitResult], headers: &[String]| {
        let mut chunk_out = Vec::with_capacity(1024);
        for res in results {
            let header = &headers[res.query_id as usize];
            let bucket_name = metadata
                .bucket_names
                .get(&res.bucket_id)
                .map(|s| s.as_str())
                .unwrap_or("unknown");
            writeln!(chunk_out, "{}\t{}\t{:.4}", header, bucket_name, res.score).unwrap();
        }
        chunk_out
    };

    loop {
        let t_io_read = std::time::Instant::now();

        // Handle Parquet and FASTX differently for zero-copy optimization
        if let Some(ref mut reader) = parquet_reader {
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
            let chunk_out = format_results_ref(&results, &headers);
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
        } else if let Some(ref mut io) = fastx_io {
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
            let chunk_out = format_results(&results, &headers);
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
        } else {
            break;
        }
    }

    log::info!("Classification complete: {} reads processed", total_reads);
    out_writer.finish()?;
    if let Some(ref mut reader) = parquet_reader {
        reader.finish()?;
    }
    if let Some(ref mut io) = fastx_io {
        io.finish()?;
    }

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
