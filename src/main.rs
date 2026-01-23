use anyhow::{anyhow, Result};
use arrow::record_batch::RecordBatch;
use clap::Parser;
use std::collections::{HashMap, HashSet};
use std::io::Write;

use rype::memory::{
    calculate_batch_config, detect_available_memory, format_bytes, MemoryConfig, MemorySource,
    ReadMemoryProfile,
};
use rype::parquet_index;
use rype::{
    classify_batch_sharded_merge_join, classify_batch_sharded_parallel_rg,
    classify_batch_sharded_sequential, log_timing, IndexMetadata, ShardedInvertedIndex,
    ENABLE_TIMING,
};

mod commands;
mod logging;

use commands::{
    build_parquet_index_from_config, create_parquet_index_from_refs, inspect_matches,
    is_parquet_input, load_index_metadata, stacked_batches_to_records, ClassifyCommands, Cli,
    Commands, IndexCommands, InspectCommands, OutputFormat, OutputWriter, PrefetchingIoHandler,
    PrefetchingParquetReader,
};

// CLI argument definitions moved to commands/args.rs
// Index command handlers moved to commands/index.rs
// Inspect command handlers moved to commands/inspect.rs
// IoHandler moved to commands/helpers.rs

fn main() -> Result<()> {
    let args = Cli::parse();

    // Initialize logging based on verbose flag
    logging::init_logger(args.verbose);

    match args.command {
        Commands::Index(index_cmd) => match index_cmd {
            IndexCommands::Create {
                output,
                reference,
                kmer_size,
                window,
                salt,
                separate_buckets,
                max_shard_size,
                row_group_size,
                zstd,
                bloom_filter,
                bloom_fpp,
                timing,
            } => {
                // Enable timing diagnostics if requested
                if timing {
                    ENABLE_TIMING.store(true, std::sync::atomic::Ordering::Relaxed);
                }

                if !matches!(kmer_size, 16 | 32 | 64) {
                    return Err(anyhow!("K must be 16, 32, or 64 (got {})", kmer_size));
                }

                // Create Parquet inverted index directly
                let parquet_options = parquet_index::ParquetWriteOptions {
                    row_group_size,
                    compression: if zstd {
                        parquet_index::ParquetCompression::Zstd
                    } else {
                        parquet_index::ParquetCompression::Snappy
                    },
                    bloom_filter_enabled: bloom_filter,
                    bloom_filter_fpp: bloom_fpp,
                    write_page_statistics: true,
                };

                create_parquet_index_from_refs(
                    &output,
                    &reference,
                    kmer_size,
                    window,
                    salt,
                    separate_buckets,
                    max_shard_size,
                    Some(&parquet_options),
                )?;
            }

            IndexCommands::Stats { index } => {
                // Load Parquet index stats
                if !rype::is_parquet_index(&index) {
                    return Err(anyhow!(
                        "Index not found or not in Parquet format: {}\n\
                         Create an index with: rype index create -o index.ryxdi -r refs.fasta",
                        index.display()
                    ));
                }

                let sharded = ShardedInvertedIndex::open(&index)?;
                let manifest = sharded.manifest();

                println!("Index Stats for {:?}", index);
                println!("  K: {}", manifest.k);
                println!("  Window (w): {}", manifest.w);
                println!("  Salt: 0x{:x}", manifest.salt);
                println!("  Buckets: {}", manifest.bucket_names.len());
                println!("  Shards: {}", manifest.shards.len());
                println!("  Total minimizers: {}", manifest.total_minimizers);
                println!("  Total bucket references: {}", manifest.total_bucket_ids);
                if manifest.total_minimizers > 0 {
                    println!(
                        "  Avg buckets per minimizer: {:.2}",
                        manifest.total_bucket_ids as f64 / manifest.total_minimizers as f64
                    );
                }

                println!("  ------------------------------------------------");
                println!("  Shard distribution:");
                for shard in &manifest.shards {
                    println!(
                        "    Shard {}: {} minimizers, {} bucket refs",
                        shard.shard_id, shard.num_minimizers, shard.num_bucket_ids
                    );
                }

                println!("------------------------------------------------");
                let mut sorted_ids: Vec<_> = manifest.bucket_names.keys().collect();
                sorted_ids.sort();
                for id in sorted_ids {
                    let name = manifest
                        .bucket_names
                        .get(id)
                        .map(|s| s.as_str())
                        .unwrap_or("unknown");
                    let sources = manifest
                        .bucket_sources
                        .get(id)
                        .map(|v| v.len())
                        .unwrap_or(0);
                    println!("  Bucket {}: '{}' ({} sources)", id, name, sources);
                }
            }

            IndexCommands::BucketSourceDetail {
                index,
                bucket,
                paths,
                ids,
            } => {
                let metadata = load_index_metadata(&index)?;
                let sources = metadata.bucket_sources.get(&bucket).unwrap();

                if paths && ids {
                    return Err(anyhow!("Cannot have --paths and --ids"));
                }

                if paths {
                    let mut all_paths: HashSet<String> = HashSet::new();
                    for source in sources {
                        let parts: Vec<_> = source.split(rype::BUCKET_SOURCE_DELIM).collect();
                        let path = parts.first().unwrap().to_string();
                        all_paths.insert(path.clone());
                    }

                    for path in all_paths {
                        println!("{}", path);
                    }
                } else if ids {
                    let mut sorted_ids: Vec<_> = metadata.bucket_names.keys().collect();
                    sorted_ids.sort();
                    for id in sorted_ids {
                        println!("{}", id);
                    }
                } else {
                    for source in sources {
                        println!("{}", source);
                    }
                }
            }

            IndexCommands::BucketAdd {
                index: _,
                reference: _,
            } => {
                return Err(anyhow!(
                    "bucket-add is not yet implemented for the Parquet index format.\n\
                     This feature is pending development. For now, re-create the index \
                     from scratch with all reference files."
                ));
            }

            IndexCommands::FromConfig {
                config,
                max_shard_size,
                row_group_size,
                bloom_filter,
                bloom_fpp,
                timing,
            } => {
                // Enable timing diagnostics if requested
                if timing {
                    ENABLE_TIMING.store(true, std::sync::atomic::Ordering::Relaxed);
                }

                // Create parquet inverted index directly
                let options = parquet_index::ParquetWriteOptions {
                    row_group_size,
                    bloom_filter_enabled: bloom_filter,
                    bloom_filter_fpp: bloom_fpp,
                    ..Default::default()
                };
                build_parquet_index_from_config(&config, max_shard_size, Some(&options))?;
            }

            IndexCommands::BucketAddConfig { config: _ } => {
                return Err(anyhow!(
                    "bucket-add-config is not yet implemented for the Parquet index format.\n\
                     This feature is pending development."
                ));
            }

            IndexCommands::Summarize { index: _ } => {
                return Err(anyhow!(
                    "summarize command is not available for Parquet indices.\n\
                     Use 'rype index stats -i <index>' to view index statistics."
                ));
            }
        },

        Commands::Classify(classify_cmd) => match classify_cmd {
            ClassifyCommands::Run {
                index,
                negative_index,
                r1,
                r2,
                threshold,
                max_memory,
                batch_size,
                output,
                merge_join,
                parallel_rg,
                use_bloom_filter,
                parallel_input_rg,
                timing,
            } => {
                // Enable timing diagnostics if requested
                if timing {
                    ENABLE_TIMING.store(true, std::sync::atomic::Ordering::Relaxed);
                }

                // Negative index not yet supported with Parquet format
                if negative_index.is_some() {
                    return Err(anyhow!(
                        "Negative index filtering is not yet supported with Parquet indices.\n\
                         This feature is pending development."
                    ));
                }
                let neg_mins: Option<HashSet<u64>> = None;

                // Determine effective batch size: user override or adaptive
                let effective_batch_size = if let Some(bs) = batch_size {
                    log::info!("Using user-specified batch size: {}", bs);
                    bs
                } else {
                    // Load index metadata to get k, w, num_buckets
                    let metadata = load_index_metadata(&index)?;

                    // Detect or use specified memory limit (0 = auto)
                    let mem_limit = if max_memory == 0 {
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
                        max_memory
                    };

                    // Sample read lengths from input files
                    let is_paired = r2.is_some();
                    let read_profile = ReadMemoryProfile::from_files(
                        &r1,
                        r2.as_deref(),
                        1000, // sample size
                        metadata.k,
                        metadata.w,
                    )
                    .unwrap_or_else(|| {
                        log::warn!("Could not sample read lengths, using default profile");
                        ReadMemoryProfile::default_profile(is_paired, metadata.k, metadata.w)
                    });

                    log::debug!("Read profile: avg_read_length={}, avg_query_length={}, minimizers_per_query={}",
                        read_profile.avg_read_length, read_profile.avg_query_length, read_profile.minimizers_per_query);

                    // For now, use a simple heuristic since we don't have index loaded yet
                    // We'll estimate index memory from metadata
                    let estimated_index_mem =
                        metadata.bucket_minimizer_counts.values().sum::<usize>() * 8;
                    let num_buckets = metadata.bucket_names.len();

                    let mem_config = MemoryConfig {
                        max_memory: mem_limit,
                        num_threads: rayon::current_num_threads(),
                        index_memory: estimated_index_mem,
                        shard_reservation: 0, // Will be updated after loading index
                        read_profile,
                        num_buckets,
                    };

                    let batch_config = calculate_batch_config(&mem_config);
                    log::info!("Adaptive batch sizing: batch_size={}, parallel_batches={}, threads={}, estimated peak memory={}",
                        batch_config.batch_size, batch_config.batch_count, rayon::current_num_threads(), format_bytes(batch_config.peak_memory));
                    batch_config.batch_size
                };

                // Check for Parquet input
                let input_is_parquet = is_parquet_input(&r1);
                if input_is_parquet && r2.is_some() {
                    return Err(anyhow!(
                        "Parquet input with separate R2 file is not supported. \
                         Use a Parquet file with 'sequence2' column for paired-end data."
                    ));
                }

                // Verify index is Parquet format
                if !rype::is_parquet_index(&index) {
                    return Err(anyhow!(
                        "Index not found or not in Parquet format: {}\n\
                         Create an index with: rype index create -o index.ryxdi -r refs.fasta",
                        index.display()
                    ));
                }

                // Load metadata from Parquet index
                log::info!("Detected Parquet index at {:?}", index);
                let manifest = rype::ParquetManifest::load(&index)?;
                let (bucket_names, bucket_sources) =
                    rype::parquet_index::read_buckets_parquet(&index)?;
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
                let output_format = OutputFormat::detect(output.as_ref());
                let mut out_writer = OutputWriter::new(output_format, output.as_ref(), None)?;
                out_writer.write_header(b"read_id\tbucket_name\tscore\n")?;

                // Create input reader (Parquet or FASTX)
                let mut parquet_reader = if input_is_parquet {
                    let parallel_rg_opt = if parallel_input_rg > 0 {
                        Some(parallel_input_rg)
                    } else {
                        None
                    };
                    log::info!(
                        "Using prefetching Parquet input reader (zero-copy, batch_size={}, parallel_rg={:?}) for {:?}",
                        effective_batch_size,
                        parallel_rg_opt,
                        r1
                    );
                    Some(PrefetchingParquetReader::with_parallel_row_groups(
                        &r1,
                        effective_batch_size,
                        parallel_rg_opt,
                    )?)
                } else {
                    None
                };
                let mut fastx_io = if !input_is_parquet {
                    Some(PrefetchingIoHandler::new(
                        &r1,
                        r2.as_ref(),
                        None,
                        effective_batch_size,
                    )?)
                } else {
                    None
                };

                let mut total_reads = 0;
                let mut batch_num = 0;

                // Load Parquet inverted index
                log::info!("Loading Parquet inverted index from {:?}", index);
                let sharded = ShardedInvertedIndex::open(&index)?;

                log::info!(
                    "Sharded index: {} shards, {} total minimizers",
                    sharded.num_shards(),
                    sharded.total_minimizers()
                );

                // Advise kernel to prefetch Parquet shard files
                if parallel_rg {
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
                let read_options = if use_bloom_filter {
                    log::info!("Bloom filter row group filtering enabled");
                    Some(parquet_index::ParquetReadOptions::with_bloom_filter())
                } else {
                    None
                };

                if parallel_rg {
                    log::info!(
                        "Starting parallel row group classification (batch_size={})",
                        effective_batch_size
                    );
                } else if merge_join {
                    log::info!("Starting merge-join classification with sequential shard loading (batch_size={})", effective_batch_size);
                } else {
                    log::info!(
                        "Starting classification with sequential shard loading (batch_size={})",
                        effective_batch_size
                    );
                }

                // Helper closure for classification
                let classify_records =
                    |batch_refs: &[rype::QueryRecord]| -> Result<Vec<rype::HitResult>> {
                        if parallel_rg {
                            classify_batch_sharded_parallel_rg(
                                &sharded,
                                neg_mins.as_ref(),
                                batch_refs,
                                threshold,
                                read_options.as_ref(),
                            )
                        } else if merge_join {
                            classify_batch_sharded_merge_join(
                                &sharded,
                                neg_mins.as_ref(),
                                batch_refs,
                                threshold,
                                read_options.as_ref(),
                            )
                        } else {
                            classify_batch_sharded_sequential(
                                &sharded,
                                neg_mins.as_ref(),
                                batch_refs,
                                threshold,
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
                        writeln!(chunk_out, "{}\t{}\t{:.4}", header, bucket_name, res.score)
                            .unwrap();
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
                        writeln!(chunk_out, "{}\t{}\t{:.4}", header, bucket_name, res.score)
                            .unwrap();
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

                        let t_format = std::time::Instant::now();
                        let chunk_out = format_results_ref(&results, &headers);
                        log_timing("batch: format_output", t_format.elapsed().as_millis());

                        let t_write = std::time::Instant::now();
                        out_writer.write_chunk(chunk_out)?;
                        log_timing("batch: io_write", t_write.elapsed().as_millis());

                        log::info!(
                            "Processed batch {} ({} row groups stacked): {} reads ({} total)",
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
            }

            ClassifyCommands::Aggregate {
                index: _,
                negative_index: _,
                r1: _,
                r2: _,
                threshold: _,
                max_memory: _,
                batch_size: _,
                output: _,
            } => {
                return Err(anyhow!(
                    "aggregate command is not yet supported with Parquet indices.\n\
                     This feature is pending development. Use 'classify run' for per-read classification."
                ));
            }
        },

        Commands::Inspect(inspect_cmd) => match inspect_cmd {
            InspectCommands::Matches {
                index,
                queries,
                ids,
                buckets,
            } => {
                inspect_matches(&index, &queries, &ids, &buckets)?;
            }
        },
    }

    Ok(())
}
