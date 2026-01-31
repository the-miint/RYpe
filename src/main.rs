use anyhow::{anyhow, Result};
use clap::Parser;
use std::collections::HashSet;

use rype::parquet_index;
use rype::{ShardedInvertedIndex, ENABLE_TIMING};

mod commands;
mod logging;

use commands::{
    build_parquet_index_from_config, create_parquet_index_from_refs, inspect_matches,
    load_index_metadata, resolve_bucket_id, run_aggregate, run_classify, ClassifyAggregateArgs,
    ClassifyCommands, ClassifyRunArgs, Cli, Commands, IndexCommands, InspectCommands,
};

// CLI argument definitions moved to commands/args.rs
// Index command handlers moved to commands/index.rs
// Classify command handlers moved to commands/classify.rs
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
                let bucket_id = resolve_bucket_id(&bucket, &metadata.bucket_names)?;
                let sources = metadata
                    .bucket_sources
                    .get(&bucket_id)
                    .ok_or_else(|| anyhow!("Bucket {} not found in index", bucket_id))?;

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
                orient,
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
                build_parquet_index_from_config(&config, max_shard_size, Some(&options), orient)?;
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
                parallel_rg,
                use_bloom_filter,
                parallel_input_rg,
                timing,
                best_hit,
                trim_to,
                wide,
            } => {
                // Enable timing diagnostics if requested
                if timing {
                    ENABLE_TIMING.store(true, std::sync::atomic::Ordering::Relaxed);
                }

                run_classify(ClassifyRunArgs {
                    index,
                    negative_index,
                    r1,
                    r2,
                    threshold,
                    max_memory,
                    batch_size,
                    output,
                    parallel_rg,
                    use_bloom_filter,
                    parallel_input_rg,
                    best_hit,
                    trim_to,
                    wide,
                })?;
            }

            ClassifyCommands::Aggregate {
                index,
                negative_index,
                r1,
                r2,
                threshold,
                max_memory,
                batch_size,
                output,
            } => {
                run_aggregate(ClassifyAggregateArgs {
                    index,
                    negative_index,
                    r1,
                    r2,
                    threshold,
                    max_memory,
                    batch_size,
                    output,
                })?;
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
