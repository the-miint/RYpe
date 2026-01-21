use anyhow::{anyhow, Context, Result};
use clap::Parser;
use needletail::parse_fastx_file;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use rype::memory::{
    calculate_batch_config, detect_available_memory, format_bytes, MemoryConfig, MemorySource,
    ReadMemoryProfile,
};
use rype::parquet_index;
use rype::{
    aggregate_batch, classify_batch, classify_batch_sharded_main,
    classify_batch_sharded_merge_join, classify_batch_sharded_sequential, extract_into, log_timing,
    Index, IndexMetadata, InvertedIndex, MainIndexManifest, MainIndexShard, MinimizerWorkspace,
    QueryRecord, ShardFormat, ShardManifest, ShardedInvertedIndex, ShardedMainIndex, ENABLE_TIMING,
};

mod commands;
mod logging;

use commands::{
    add_reference_file_to_index, bucket_add_from_config, build_index_from_config,
    build_parquet_index_from_config, create_parquet_index_from_refs, inspect_matches,
    load_index_metadata, sanitize_bucket_name, ClassifyCommands, Cli, Commands, IndexCommands,
    InspectCommands, IoHandler,
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
                parquet,
                parquet_row_group_size,
                parquet_zstd,
                parquet_bloom_filter,
                parquet_bloom_fpp,
                timing,
            } => {
                // Enable timing diagnostics if requested
                if timing {
                    ENABLE_TIMING.store(true, std::sync::atomic::Ordering::Relaxed);
                }

                if !matches!(kmer_size, 16 | 32 | 64) {
                    return Err(anyhow!("K must be 16, 32, or 64 (got {})", kmer_size));
                }

                if parquet {
                    // Create Parquet inverted index directly
                    // Build ParquetWriteOptions from CLI flags
                    let parquet_options = parquet_index::ParquetWriteOptions {
                        row_group_size: parquet_row_group_size,
                        compression: if parquet_zstd {
                            parquet_index::ParquetCompression::Zstd
                        } else {
                            parquet_index::ParquetCompression::Snappy
                        },
                        bloom_filter_enabled: parquet_bloom_filter,
                        bloom_filter_fpp: parquet_bloom_fpp,
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
                } else {
                    // Legacy: create main index
                    let mut index = Index::new(kmer_size, window, salt)?;
                    let mut next_id = 1;

                    for ref_file in reference {
                        add_reference_file_to_index(
                            &mut index,
                            &ref_file,
                            separate_buckets,
                            &mut next_id,
                        )?;
                    }

                    if let Some(max_bytes) = max_shard_size {
                        log::info!(
                            "Saving sharded index to {:?} (max {} bytes/shard)...",
                            output,
                            max_bytes
                        );
                        let manifest = index.save_sharded(&output, max_bytes)?;
                        log::info!(
                            "Created {} shards with {} total minimizers.",
                            manifest.shards.len(),
                            manifest.total_minimizers
                        );
                    } else {
                        log::info!("Saving index to {:?}...", output);
                        index.save(&output)?;
                    }
                    log::info!("Done.");
                }
            }

            IndexCommands::Stats { index, inverted } => {
                if inverted {
                    // Show inverted index stats - check for sharded vs single-file
                    let inverted_path = index.with_extension("ryxdi");
                    let manifest_path = ShardManifest::manifest_path(&inverted_path);

                    if manifest_path.exists() {
                        // Sharded inverted index - manifest is already lightweight
                        let manifest = ShardManifest::load(&manifest_path)?;
                        println!("Sharded Inverted Index Stats for {:?}", inverted_path);
                        println!("  K: {}", manifest.k);
                        println!("  Window (w): {}", manifest.w);
                        println!("  Salt: 0x{:x}", manifest.salt);
                        println!("  Shards: {}", manifest.shards.len());
                        println!("  Unique minimizers: {}", manifest.total_minimizers);
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
                    } else if inverted_path.exists() {
                        // Legacy single-file inverted index (RYXI) no longer supported
                        return Err(anyhow!(
                            "Legacy single-file inverted index format (RYXI) is no longer supported.\n\
                             Re-create the inverted index with:\n  \
                             rype index invert -i {}",
                            index.display()
                        ));
                    } else {
                        return Err(anyhow!(
                            "Inverted index not found: {:?}. Create it with 'rype index invert -i {:?}'",
                            inverted_path, index
                        ));
                    }
                } else {
                    // Show primary index stats - detect sharded vs single-file
                    let main_manifest_path = MainIndexManifest::manifest_path(&index);

                    let metadata = if main_manifest_path.exists() {
                        // Sharded main index
                        let manifest = MainIndexManifest::load(&main_manifest_path)?;
                        println!("Sharded Index Stats for {:?}", index);
                        println!("  K: {}", manifest.k);
                        println!("  Window (w): {}", manifest.w);
                        println!("  Salt: 0x{:x}", manifest.salt);
                        println!("  Buckets: {}", manifest.bucket_names.len());
                        println!("  Shards: {}", manifest.shards.len());
                        println!("  Total minimizers: {}", manifest.total_minimizers);

                        println!("  ------------------------------------------------");
                        println!("  Shard distribution:");
                        for shard in &manifest.shards {
                            println!(
                                "    Shard {}: {} buckets, {} minimizers, {} bytes",
                                shard.shard_id,
                                shard.bucket_ids.len(),
                                shard.num_minimizers,
                                shard.compressed_size
                            );
                        }

                        manifest.to_metadata()
                    } else {
                        // Single-file main index
                        let metadata = Index::load_metadata(&index)?;
                        println!("Index Stats for {:?}", index);
                        println!("  K: {}", metadata.k);
                        println!("  Window (w): {}", metadata.w);
                        println!("  Salt: 0x{:x}", metadata.salt);
                        println!("  Buckets: {}", metadata.bucket_names.len());
                        metadata
                    };

                    // Check if inverted index exists
                    let inverted_path = index.with_extension("ryxdi");
                    if inverted_path.exists() {
                        println!(
                            "  Inverted index: {:?} (use -I to show stats)",
                            inverted_path
                        );
                    }

                    println!("------------------------------------------------");
                    let mut sorted_ids: Vec<_> = metadata.bucket_names.keys().collect();
                    sorted_ids.sort();
                    for id in sorted_ids {
                        let name = metadata
                            .bucket_names
                            .get(id)
                            .map(|s| s.as_str())
                            .unwrap_or("unknown");
                        let count = metadata
                            .bucket_minimizer_counts
                            .get(id)
                            .copied()
                            .unwrap_or(0);
                        let sources = metadata
                            .bucket_sources
                            .get(id)
                            .map(|v| v.len())
                            .unwrap_or(0);
                        println!(
                            "  Bucket {}: '{}' ({} minimizers, {} sources)",
                            id, name, count, sources
                        );
                    }
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
                        let parts: Vec<_> = source.split(Index::BUCKET_SOURCE_DELIM).collect();
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

            IndexCommands::BucketAdd { index, reference } => {
                let main_manifest_path = MainIndexManifest::manifest_path(&index);

                if main_manifest_path.exists() {
                    // Sharded main index - add as new shard
                    let mut sharded = ShardedMainIndex::open(&index)?;
                    let next_id = sharded.next_id()?;
                    log::info!(
                        "Adding {:?} as new bucket ID {} (sharded)",
                        reference,
                        next_id
                    );

                    // Extract minimizers from reference file
                    let mut reader =
                        parse_fastx_file(&reference).context("Failed to open reference file")?;
                    let mut ws = MinimizerWorkspace::new();
                    let filename = reference
                        .canonicalize()
                        .unwrap()
                        .to_string_lossy()
                        .to_string();
                    let mut sources = Vec::new();
                    let mut all_minimizers = Vec::new();

                    while let Some(record) = reader.next() {
                        let rec = record.context("Invalid record")?;
                        let seq = rec.seq();
                        let name = String::from_utf8_lossy(rec.id()).to_string();
                        let source_label =
                            format!("{}{}{}", filename, Index::BUCKET_SOURCE_DELIM, name);
                        sources.push(source_label);

                        extract_into(&seq, sharded.k(), sharded.w(), sharded.salt(), &mut ws);
                        all_minimizers.extend_from_slice(&ws.buffer);
                    }

                    // Sort and deduplicate
                    all_minimizers.sort_unstable();
                    all_minimizers.dedup();
                    sources.sort_unstable();
                    sources.dedup();

                    let minimizer_count = all_minimizers.len();
                    sharded.add_bucket(
                        next_id,
                        &sanitize_bucket_name(&filename),
                        sources,
                        all_minimizers,
                    )?;

                    log::info!(
                        "Done. Added {} minimizers to bucket {} (new shard {}).",
                        minimizer_count,
                        next_id,
                        sharded.num_shards() - 1
                    );
                } else {
                    // Single-file main index
                    let mut idx = Index::load(&index)?;
                    let next_id = idx.next_id()?;
                    log::info!("Adding {:?} as new bucket ID {}", reference, next_id);

                    let mut reader =
                        parse_fastx_file(&reference).context("Failed to open reference file")?;
                    let mut ws = MinimizerWorkspace::new();
                    let filename = reference
                        .canonicalize()
                        .unwrap()
                        .to_string_lossy()
                        .to_string();

                    idx.bucket_names
                        .insert(next_id, sanitize_bucket_name(&filename));

                    while let Some(record) = reader.next() {
                        let rec = record.context("Invalid record")?;
                        let seq = rec.seq();
                        let name = String::from_utf8_lossy(rec.id()).to_string();
                        let source_label =
                            format!("{}{}{}", filename, Index::BUCKET_SOURCE_DELIM, name);
                        idx.add_record(next_id, &source_label, &seq, &mut ws);
                    }

                    idx.finalize_bucket(next_id);
                    idx.save(&index)?;
                    log::info!(
                        "Done. Added {} minimizers to bucket {}.",
                        idx.buckets.get(&next_id).map(|v| v.len()).unwrap_or(0),
                        next_id
                    );
                }
            }

            IndexCommands::Merge { output, inputs } => {
                // Logic: Load first index, then merge others into it.
                // Warning: Salt/W must match.
                if inputs.is_empty() {
                    return Err(anyhow!("No input indexes provided"));
                }

                log::info!("Loading base index: {:?}", inputs[0]);
                let mut base_idx = Index::load(&inputs[0])?;

                for path in &inputs[1..] {
                    log::info!("Merging index: {:?}", path);
                    let other_idx = Index::load(path)?;

                    if other_idx.k != base_idx.k
                        || other_idx.w != base_idx.w
                        || other_idx.salt != base_idx.salt
                    {
                        return Err(anyhow!(
                            "Index parameters mismatch: expected K={}, W={}, Salt=0x{:x}, got K={}, W={}, Salt=0x{:x}",
                            base_idx.k, base_idx.w, base_idx.salt,
                            other_idx.k, other_idx.w, other_idx.salt
                        ));
                    }

                    // Naive merge strategy: Re-map IDs of 'other' to not collide, then insert
                    // Simple version: just append buckets with new IDs
                    for (old_id, vec) in other_idx.buckets {
                        let new_id = base_idx.next_id()?;
                        base_idx.buckets.insert(new_id, vec);

                        if let Some(name) = other_idx.bucket_names.get(&old_id) {
                            base_idx
                                .bucket_names
                                .insert(new_id, sanitize_bucket_name(name));
                        }
                        if let Some(srcs) = other_idx.bucket_sources.get(&old_id) {
                            base_idx.bucket_sources.insert(new_id, srcs.clone());
                        }
                    }
                }
                base_idx.save(&output)?;
                log::info!("Merged index saved to {:?}", output);
            }

            IndexCommands::FromConfig {
                config,
                max_shard_size,
                invert,
                parquet,
                parquet_bloom_filter,
                parquet_bloom_fpp,
                timing,
            } => {
                // Enable timing diagnostics if requested
                if timing {
                    ENABLE_TIMING.store(true, std::sync::atomic::Ordering::Relaxed);
                }

                if parquet {
                    // Create parquet inverted index directly (bypasses main index)
                    let options = parquet_index::ParquetWriteOptions {
                        bloom_filter_enabled: parquet_bloom_filter,
                        bloom_filter_fpp: parquet_bloom_fpp,
                        ..Default::default()
                    };
                    build_parquet_index_from_config(&config, max_shard_size, Some(&options))?;
                } else {
                    build_index_from_config(&config, max_shard_size, invert)?;
                }
            }

            IndexCommands::BucketAddConfig { config } => {
                bucket_add_from_config(&config)?;
            }

            IndexCommands::Invert {
                index,
                format,
                parquet_row_group_size,
                parquet_zstd,
                parquet_bloom_filter,
                parquet_bloom_fpp,
                timing,
            } => {
                // Enable timing diagnostics if requested
                if timing {
                    ENABLE_TIMING.store(true, std::sync::atomic::Ordering::Relaxed);
                }

                let t_total = Instant::now();
                let output_path = index.with_extension("ryxdi");
                let use_parquet = format == "parquet";

                // Build ParquetWriteOptions from CLI flags (used when format=parquet)
                let parquet_options = parquet_index::ParquetWriteOptions {
                    row_group_size: parquet_row_group_size,
                    compression: if parquet_zstd {
                        parquet_index::ParquetCompression::Zstd
                    } else {
                        parquet_index::ParquetCompression::Snappy
                    },
                    bloom_filter_enabled: parquet_bloom_filter,
                    bloom_filter_fpp: parquet_bloom_fpp,
                    write_page_statistics: true,
                };

                // For Parquet format, create the inverted directory
                if use_parquet {
                    let inverted_dir = output_path.join("inverted");
                    std::fs::create_dir_all(&inverted_dir).with_context(|| {
                        format!(
                            "Failed to create inverted directory: {}",
                            inverted_dir.display()
                        )
                    })?;
                    log::info!("Using Parquet format, output to {:?}", output_path);
                }

                // Track files we create for cleanup on error
                let mut created_files: Vec<PathBuf> = Vec::new();

                let result: Result<ShardManifest> = if MainIndexManifest::is_sharded(&index) {
                    // Sharded main index: create 1:1 inverted shards
                    log::info!("Detected sharded main index, creating 1:1 inverted shards");
                    let t_load_manifest = Instant::now();
                    let main_manifest =
                        MainIndexManifest::load(&MainIndexManifest::manifest_path(&index))?;
                    log_timing(
                        "invert: load_manifest",
                        t_load_manifest.elapsed().as_millis(),
                    );
                    log::info!(
                        "Main index has {} shards, {} buckets",
                        main_manifest.shards.len(),
                        main_manifest.bucket_names.len()
                    );

                    (|| -> Result<ShardManifest> {
                        let mut inv_shards = Vec::new();
                        let mut total_minimizers = 0usize;
                        let mut total_bucket_ids = 0usize;
                        let num_shards = main_manifest.shards.len();

                        for (idx, shard_info) in main_manifest.shards.iter().enumerate() {
                            let shard_path =
                                MainIndexManifest::shard_path(&index, shard_info.shard_id);
                            log::info!(
                                "Processing main shard {}: {} buckets, {} minimizers (raw)",
                                shard_info.shard_id,
                                shard_info.bucket_ids.len(),
                                shard_info.num_minimizers
                            );

                            // Build inverted index from shard, dropping main shard immediately after
                            let t_shard_load = Instant::now();
                            let inverted = {
                                let main_shard = MainIndexShard::load(&shard_path)?;
                                log_timing(
                                    &format!("invert: shard_{}_load", shard_info.shard_id),
                                    t_shard_load.elapsed().as_millis(),
                                );
                                let t_build = Instant::now();
                                let inv = InvertedIndex::build_from_shard(&main_shard);
                                log_timing(
                                    &format!("invert: shard_{}_build", shard_info.shard_id),
                                    t_build.elapsed().as_millis(),
                                );
                                inv
                            };

                            log::info!(
                                "  Built inverted: {} unique minimizers, {} bucket entries",
                                inverted.num_minimizers(),
                                inverted.num_bucket_entries()
                            );

                            // Save as inverted shard with same ID
                            let t_save = Instant::now();
                            let is_last = idx == num_shards - 1;
                            let inv_shard_info = if use_parquet {
                                let inv_shard_path = ShardManifest::shard_path_parquet(
                                    &output_path,
                                    shard_info.shard_id,
                                );
                                let info = inverted.save_shard_parquet(
                                    &inv_shard_path,
                                    shard_info.shard_id,
                                    Some(&parquet_options),
                                )?;
                                created_files.push(inv_shard_path);
                                info
                            } else {
                                let inv_shard_path =
                                    ShardManifest::shard_path(&output_path, shard_info.shard_id);
                                let info = inverted.save_shard(
                                    &inv_shard_path,
                                    shard_info.shard_id,
                                    0,
                                    inverted.num_minimizers(),
                                    is_last,
                                )?;
                                created_files.push(inv_shard_path);
                                info
                            };
                            log_timing(
                                &format!("invert: shard_{}_save", shard_info.shard_id),
                                t_save.elapsed().as_millis(),
                            );

                            total_minimizers += inv_shard_info.num_minimizers;
                            total_bucket_ids += inv_shard_info.num_bucket_ids;
                            inv_shards.push(inv_shard_info);
                        }

                        // Create inverted manifest with bucket metadata from main index
                        let main_metadata = main_manifest.to_metadata();
                        let inv_manifest = ShardManifest {
                            k: main_manifest.k,
                            w: main_manifest.w,
                            salt: main_manifest.salt,
                            source_hash: InvertedIndex::compute_metadata_hash(&main_metadata),
                            total_minimizers,
                            total_bucket_ids,
                            has_overlapping_shards: true,
                            shard_format: if use_parquet {
                                ShardFormat::Parquet
                            } else {
                                ShardFormat::Legacy
                            },
                            shards: inv_shards,
                            bucket_names: main_metadata.bucket_names,
                            bucket_sources: main_metadata.bucket_sources,
                            bucket_minimizer_counts: main_metadata.bucket_minimizer_counts,
                        };

                        let t_manifest_save = Instant::now();
                        let manifest_path = ShardManifest::manifest_path(&output_path);
                        inv_manifest.save(&manifest_path)?;
                        created_files.push(manifest_path);
                        log_timing(
                            "invert: manifest_save",
                            t_manifest_save.elapsed().as_millis(),
                        );

                        Ok(inv_manifest)
                    })()
                } else {
                    // Non-sharded main index: treat as single shard, create 1-shard inverted
                    log::info!("Loading single-file index from {:?}", index);
                    let t_load = Instant::now();
                    let idx = Index::load(&index)?;
                    log_timing("invert: load_main_index", t_load.elapsed().as_millis());
                    log::info!("Index loaded: {} buckets", idx.buckets.len());

                    log::info!("Building inverted index...");
                    let t_build = Instant::now();
                    let inverted = InvertedIndex::build_from_index(&idx);
                    log_timing("invert: build_inverted", t_build.elapsed().as_millis());
                    log::info!(
                        "Inverted index built: {} unique minimizers, {} bucket entries",
                        inverted.num_minimizers(),
                        inverted.num_bucket_entries()
                    );

                    (|| -> Result<ShardManifest> {
                        // Save as single shard (shard_id = 0)
                        let t_save_shard = Instant::now();
                        let inv_shard_info = if use_parquet {
                            let shard_path = ShardManifest::shard_path_parquet(&output_path, 0);
                            let info = inverted.save_shard_parquet(
                                &shard_path,
                                0,
                                Some(&parquet_options),
                            )?;
                            created_files.push(shard_path);
                            info
                        } else {
                            let shard_path = ShardManifest::shard_path(&output_path, 0);
                            let info = inverted.save_shard(
                                &shard_path,
                                0, // shard_id
                                0, // min_start
                                inverted.num_minimizers(),
                                true, // is_last (and only) shard
                            )?;
                            created_files.push(shard_path);
                            info
                        };
                        log_timing("invert: save_shard", t_save_shard.elapsed().as_millis());

                        // Create manifest with single shard and bucket metadata
                        let bucket_minimizer_counts: std::collections::HashMap<u32, usize> = idx
                            .buckets
                            .iter()
                            .map(|(&id, mins)| (id, mins.len()))
                            .collect();
                        let metadata = IndexMetadata {
                            k: idx.k,
                            w: idx.w,
                            salt: idx.salt,
                            bucket_names: idx.bucket_names.clone(),
                            bucket_sources: idx.bucket_sources.clone(),
                            bucket_minimizer_counts: bucket_minimizer_counts.clone(),
                        };
                        let inv_manifest = ShardManifest {
                            k: idx.k,
                            w: idx.w,
                            salt: idx.salt,
                            source_hash: InvertedIndex::compute_metadata_hash(&metadata),
                            total_minimizers: inv_shard_info.num_minimizers,
                            total_bucket_ids: inv_shard_info.num_bucket_ids,
                            has_overlapping_shards: true,
                            shard_format: if use_parquet {
                                ShardFormat::Parquet
                            } else {
                                ShardFormat::Legacy
                            },
                            shards: vec![inv_shard_info],
                            bucket_names: idx.bucket_names.clone(),
                            bucket_sources: idx.bucket_sources.clone(),
                            bucket_minimizer_counts,
                        };

                        let t_manifest_save = Instant::now();
                        let manifest_path = ShardManifest::manifest_path(&output_path);
                        inv_manifest.save(&manifest_path)?;
                        created_files.push(manifest_path);
                        log_timing(
                            "invert: manifest_save",
                            t_manifest_save.elapsed().as_millis(),
                        );

                        Ok(inv_manifest)
                    })()
                };

                match result {
                    Ok(inv_manifest) => {
                        log::info!("Created {} inverted shard(s):", inv_manifest.shards.len());
                        for shard in &inv_manifest.shards {
                            log::info!(
                                "  Shard {}: {} unique minimizers, {} bucket entries",
                                shard.shard_id,
                                shard.num_minimizers,
                                shard.num_bucket_ids
                            );
                        }
                        log_timing("invert: total", t_total.elapsed().as_millis());
                        log::info!("Done.");
                    }
                    Err(e) => {
                        // Clean up partial files on error
                        for path in &created_files {
                            if path.exists() {
                                let _ = std::fs::remove_file(path);
                            }
                        }
                        return Err(e);
                    }
                }
            }

            IndexCommands::Summarize { index } => {
                log::info!("Loading index from {:?}", index);
                let idx = Index::load(&index)?;

                // Basic info
                println!("=== Index Summary ===");
                println!("File: {:?}", index);
                println!("K: {}", idx.k);
                println!("Window (w): {}", idx.w);
                println!("Salt: 0x{:x}", idx.salt);
                println!("Buckets: {}", idx.buckets.len());

                // Collect all minimizers from all buckets (they're already sorted within buckets)
                let mut all_minimizers: Vec<u64> = Vec::new();
                let mut per_bucket_counts: Vec<(u32, usize)> = Vec::new();

                for (&id, mins) in &idx.buckets {
                    per_bucket_counts.push((id, mins.len()));
                    all_minimizers.extend(mins.iter().copied());
                }

                let total_minimizers = all_minimizers.len();
                println!(
                    "Total minimizers (with duplicates across buckets): {}",
                    total_minimizers
                );

                if total_minimizers == 0 {
                    println!("No minimizers to analyze.");
                    return Ok(());
                }

                // Sort all minimizers for global analysis
                all_minimizers.sort_unstable();

                // Deduplicate to get unique minimizers
                all_minimizers.dedup();
                let unique_minimizers = all_minimizers.len();
                println!("Unique minimizers: {}", unique_minimizers);
                println!(
                    "Duplication ratio: {:.2}x",
                    total_minimizers as f64 / unique_minimizers as f64
                );

                // Value range
                let min_val = *all_minimizers.first().unwrap();
                let max_val = *all_minimizers.last().unwrap();
                println!("\n=== Minimizer Value Statistics ===");
                println!("Min value: {}", min_val);
                println!("Max value: {}", max_val);
                println!("Value range: {}", max_val - min_val);

                // Bits needed for raw values
                let bits_for_max = if max_val == 0 {
                    1
                } else {
                    64 - max_val.leading_zeros()
                };
                println!("Bits needed for max value: {}", bits_for_max);

                // Compute deltas
                if unique_minimizers > 1 {
                    println!("\n=== Delta Statistics (for compression analysis) ===");

                    let mut deltas: Vec<u64> = Vec::with_capacity(unique_minimizers - 1);
                    for i in 1..unique_minimizers {
                        deltas.push(all_minimizers[i] - all_minimizers[i - 1]);
                    }

                    let min_delta = *deltas.iter().min().unwrap();
                    let max_delta = *deltas.iter().max().unwrap();
                    let sum_delta: u128 = deltas.iter().map(|&d| d as u128).sum();
                    let mean_delta = sum_delta as f64 / deltas.len() as f64;

                    // Median delta
                    let mut sorted_deltas = deltas.clone();
                    sorted_deltas.sort_unstable();
                    let median_delta = sorted_deltas[sorted_deltas.len() / 2];

                    println!("Min delta: {}", min_delta);
                    println!("Max delta: {}", max_delta);
                    println!("Mean delta: {:.2}", mean_delta);
                    println!("Median delta: {}", median_delta);

                    // Bits needed for deltas
                    let bits_for_max_delta = if max_delta == 0 {
                        1
                    } else {
                        64 - max_delta.leading_zeros()
                    };
                    println!("Bits needed for max delta: {}", bits_for_max_delta);

                    // Distribution of bits needed per delta
                    let mut bit_distribution = [0usize; 65]; // 0-64 bits
                    for &d in &deltas {
                        let bits = if d == 0 { 1 } else { 64 - d.leading_zeros() } as usize;
                        bit_distribution[bits] += 1;
                    }

                    println!("\nDelta bit-width distribution:");
                    let mut cumulative = 0usize;
                    #[allow(clippy::needless_range_loop)]
                    for bits in 1..=64 {
                        if bit_distribution[bits] > 0 {
                            cumulative += bit_distribution[bits];
                            let pct = 100.0 * cumulative as f64 / deltas.len() as f64;
                            println!(
                                "  <= {} bits: {} deltas ({:.1}% cumulative)",
                                bits, bit_distribution[bits], pct
                            );
                        }
                    }

                    // Estimate compression potential
                    println!("\n=== Compression Estimates ===");
                    let raw_bytes = unique_minimizers * 8;
                    println!(
                        "Raw storage (8 bytes/minimizer): {} bytes ({:.2} GB)",
                        raw_bytes,
                        raw_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                    );

                    // Estimate varint-encoded delta size
                    // Varint uses 1 byte per 7 bits, roughly
                    let mut estimated_varint_bytes: usize = 8; // First value stored raw
                    for &d in &deltas {
                        let bits = if d == 0 { 1 } else { 64 - d.leading_zeros() } as usize;
                        estimated_varint_bytes += (bits + 6) / 7; // ceil(bits/7)
                    }
                    let varint_ratio = estimated_varint_bytes as f64 / raw_bytes as f64;
                    println!(
                        "Estimated delta+varint: {} bytes ({:.1}% of raw)",
                        estimated_varint_bytes,
                        varint_ratio * 100.0
                    );

                    // With zstd on top (rough estimate: 50-70% of varint size for sorted data)
                    let estimated_zstd = (estimated_varint_bytes as f64 * 0.6) as usize;
                    let zstd_ratio = estimated_zstd as f64 / raw_bytes as f64;
                    println!(
                        "Estimated delta+varint+zstd: ~{} bytes (~{:.1}% of raw)",
                        estimated_zstd,
                        zstd_ratio * 100.0
                    );
                }

                // Per-bucket summary
                println!("\n=== Per-Bucket Statistics ===");
                per_bucket_counts.sort_by_key(|(id, _)| *id);
                let total_in_buckets: usize = per_bucket_counts.iter().map(|(_, c)| c).sum();
                println!("Total minimizers across all buckets: {}", total_in_buckets);

                if per_bucket_counts.len() <= 20 {
                    for (id, count) in &per_bucket_counts {
                        let name = idx
                            .bucket_names
                            .get(id)
                            .map(|s| s.as_str())
                            .unwrap_or("unknown");
                        println!("  Bucket {}: {} minimizers ({})", id, count, name);
                    }
                } else {
                    println!(
                        "  (showing first 10 and last 10 of {} buckets)",
                        per_bucket_counts.len()
                    );
                    for (id, count) in per_bucket_counts.iter().take(10) {
                        let name = idx
                            .bucket_names
                            .get(id)
                            .map(|s| s.as_str())
                            .unwrap_or("unknown");
                        println!("  Bucket {}: {} minimizers ({})", id, count, name);
                    }
                    println!("  ...");
                    for (id, count) in per_bucket_counts
                        .iter()
                        .rev()
                        .take(10)
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                    {
                        let name = idx
                            .bucket_names
                            .get(id)
                            .map(|s| s.as_str())
                            .unwrap_or("unknown");
                        println!("  Bucket {}: {} minimizers ({})", id, count, name);
                    }
                }
            }

            IndexCommands::Shard {
                input,
                output,
                max_shard_size,
            } => {
                // Validate that input is not already sharded
                if MainIndexManifest::is_sharded(&input) {
                    return Err(anyhow!(
                        "Input index is already sharded: {}\n\
                         To re-shard with different settings, first merge shards into a single file,\n\
                         or use the original single-file index.",
                        input.display()
                    ));
                }

                log::info!("Loading single-file index from {:?}", input);
                let idx = Index::load(&input)?;

                log::info!(
                    "Sharding index with max shard size {}...",
                    format_bytes(max_shard_size)
                );
                let manifest = idx.save_sharded(&output, max_shard_size)?;

                log::info!(
                    "Created {} shards at {}",
                    manifest.shards.len(),
                    output.display()
                );
                eprintln!("Sharded index created:");
                eprintln!("  Manifest: {}.manifest", output.display());
                for shard_info in manifest.shards.iter() {
                    eprintln!(
                        "  Shard {}: {} buckets, {} minimizers",
                        shard_info.shard_id,
                        shard_info.bucket_ids.len(),
                        shard_info.num_minimizers
                    );
                }
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
                use_inverted,
                merge_join,
                use_bloom_filter,
                timing,
            }
            | ClassifyCommands::Batch {
                index,
                negative_index,
                r1,
                r2,
                threshold,
                max_memory,
                batch_size,
                output,
                use_inverted,
                merge_join,
                use_bloom_filter,
                timing,
            } => {
                if merge_join && !use_inverted {
                    return Err(anyhow!("--merge-join requires --use-inverted"));
                }

                if use_bloom_filter && !use_inverted {
                    return Err(anyhow!("--use-bloom-filter requires --use-inverted"));
                }

                // Enable timing diagnostics if requested
                if timing {
                    ENABLE_TIMING.store(true, std::sync::atomic::Ordering::Relaxed);
                }

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

                // Load negative index if provided, validate parameters, and build minimizer set
                let neg_mins: Option<HashSet<u64>> = match &negative_index {
                    None => None,
                    Some(neg_path) => {
                        log::info!("Loading negative index from {:?}", neg_path);
                        let neg = Index::load(neg_path)?;

                        // Load positive index metadata to validate parameters match
                        let pos_metadata = load_index_metadata(&index)?;
                        if neg.k != pos_metadata.k {
                            return Err(anyhow!(
                                "Negative index K ({}) does not match positive index K ({})",
                                neg.k,
                                pos_metadata.k
                            ));
                        }
                        if neg.w != pos_metadata.w {
                            return Err(anyhow!(
                                "Negative index W ({}) does not match positive index W ({})",
                                neg.w,
                                pos_metadata.w
                            ));
                        }
                        if neg.salt != pos_metadata.salt {
                            return Err(anyhow!(
                                "Negative index salt (0x{:x}) does not match positive index salt (0x{:x})",
                                neg.salt, pos_metadata.salt
                            ));
                        }

                        let total_mins: usize = neg.buckets.values().map(|v| v.len()).sum();
                        log::info!(
                            "Negative index loaded: {} buckets, {} minimizers",
                            neg.buckets.len(),
                            total_mins
                        );
                        let neg_set: HashSet<u64> = neg
                            .buckets
                            .values()
                            .flat_map(|v| v.iter().copied())
                            .collect();
                        log::info!(
                            "Built negative minimizer set: {} unique minimizers",
                            neg_set.len()
                        );
                        Some(neg_set)
                    }
                };

                if use_inverted {
                    // Use inverted index path - detect format:
                    // 1. Parquet format: directory with manifest.toml
                    // 2. Legacy RYXS: .ryxdi.manifest file
                    let is_parquet = rype::is_parquet_index(&index);

                    let (inverted_path, metadata) = if is_parquet {
                        // Parquet index passed directly - load metadata from it
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
                        (index.clone(), metadata)
                    } else {
                        // Legacy: derive inverted path from main index
                        let inverted_path = index.with_extension("ryxdi");
                        log::info!("Loading index metadata from {:?}", index);
                        let metadata = load_index_metadata(&index)?;
                        (inverted_path, metadata)
                    };

                    log::info!("Metadata loaded: {} buckets", metadata.bucket_names.len());

                    let mut io = IoHandler::new(&r1, r2.as_ref(), output)?;
                    io.write(b"read_id\tbucket_name\tscore\n".to_vec())?;

                    let mut total_reads = 0;
                    let mut batch_num = 0;

                    // Load sharded inverted index - use format-appropriate loader
                    let sharded = if is_parquet {
                        log::info!("Loading Parquet inverted index from {:?}", inverted_path);
                        ShardedInvertedIndex::open_parquet(&inverted_path)?
                    } else {
                        let manifest_path = ShardManifest::manifest_path(&inverted_path);
                        if !manifest_path.exists() {
                            return Err(anyhow!(
                                "Inverted index not found: {}. Create it with 'rype index invert -i {}' or use --parquet for direct creation.",
                                inverted_path.display(), index.display()
                            ));
                        }
                        log::info!(
                            "Loading sharded inverted index manifest from {:?}",
                            inverted_path
                        );
                        ShardedInvertedIndex::open(&inverted_path)?
                    };

                    log::info!(
                        "Sharded index: {} shards, {} total minimizers",
                        sharded.num_shards(),
                        sharded.total_minimizers()
                    );

                    // Skip validation for Parquet indices (metadata is embedded)
                    if !is_parquet {
                        sharded.validate_against_metadata(&metadata)?;
                        log::info!("Sharded index validated successfully");
                    }

                    // Build read options for Parquet indices
                    let read_options = if use_bloom_filter {
                        log::info!("Bloom filter row group filtering enabled");
                        Some(parquet_index::ParquetReadOptions::with_bloom_filter())
                    } else {
                        None
                    };

                    if merge_join {
                        log::info!("Starting merge-join classification with sequential shard loading (batch_size={})", effective_batch_size);
                    } else {
                        log::info!(
                            "Starting classification with sequential shard loading (batch_size={})",
                            effective_batch_size
                        );
                    }

                    while let Some((owned_records, headers)) =
                        io.next_batch_records(effective_batch_size)?
                    {
                        batch_num += 1;
                        let batch_read_count = owned_records.len();
                        total_reads += batch_read_count;

                        let batch_refs: Vec<QueryRecord> = owned_records
                            .iter()
                            .map(|(id, s1, s2)| (*id, s1.as_slice(), s2.as_deref()))
                            .collect();

                        let results = if merge_join {
                            classify_batch_sharded_merge_join(
                                &sharded,
                                neg_mins.as_ref(),
                                &batch_refs,
                                threshold,
                                read_options.as_ref(),
                            )?
                        } else {
                            classify_batch_sharded_sequential(
                                &sharded,
                                neg_mins.as_ref(),
                                &batch_refs,
                                threshold,
                                read_options.as_ref(),
                            )?
                        };

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
                        io.write(chunk_out)?;

                        log::info!(
                            "Processed batch {}: {} reads ({} total)",
                            batch_num,
                            batch_read_count,
                            total_reads
                        );
                    }

                    log::info!("Classification complete: {} reads processed", total_reads);
                    io.finish()?;
                } else {
                    // Standard path - detect sharded vs single-file main index
                    let main_manifest_path = MainIndexManifest::manifest_path(&index);

                    let mut io = IoHandler::new(&r1, r2.as_ref(), output)?;
                    io.write(b"read_id\tbucket_name\tscore\n".to_vec())?;

                    let mut total_reads = 0;
                    let mut batch_num = 0;

                    if main_manifest_path.exists() {
                        // Sharded main index - use sequential shard loading
                        log::info!("Loading sharded main index from {:?}", index);
                        let sharded = ShardedMainIndex::open(&index)?;
                        log::info!(
                            "Sharded main index: {} shards, {} buckets, {} total minimizers",
                            sharded.num_shards(),
                            sharded.manifest().bucket_names.len(),
                            sharded.total_minimizers()
                        );

                        let metadata = sharded.metadata();

                        log::info!("Starting classification with sequential main shard loading (batch_size={})", effective_batch_size);

                        while let Some((owned_records, headers)) =
                            io.next_batch_records(effective_batch_size)?
                        {
                            batch_num += 1;
                            let batch_read_count = owned_records.len();
                            total_reads += batch_read_count;

                            let batch_refs: Vec<QueryRecord> = owned_records
                                .iter()
                                .map(|(id, s1, s2)| (*id, s1.as_slice(), s2.as_deref()))
                                .collect();

                            let results = classify_batch_sharded_main(
                                &sharded,
                                neg_mins.as_ref(),
                                &batch_refs,
                                threshold,
                            )?;

                            let mut chunk_out = Vec::with_capacity(1024);
                            for res in results {
                                let header = &headers[res.query_id as usize];
                                let bucket_name = metadata
                                    .bucket_names
                                    .get(&res.bucket_id)
                                    .map(|s| s.as_str())
                                    .unwrap_or("unknown");
                                writeln!(
                                    chunk_out,
                                    "{}\t{}\t{:.4}",
                                    header, bucket_name, res.score
                                )
                                .unwrap();
                            }
                            io.write(chunk_out)?;

                            log::info!(
                                "Processed batch {}: {} reads ({} total)",
                                batch_num,
                                batch_read_count,
                                total_reads
                            );
                        }
                    } else {
                        // Single-file main index
                        log::info!("Loading index from {:?}", index);
                        let engine = Index::load(&index)?;
                        log::info!("Index loaded: {} buckets", engine.buckets.len());

                        log::info!(
                            "Starting classification (batch_size={})",
                            effective_batch_size
                        );

                        while let Some((owned_records, headers)) =
                            io.next_batch_records(effective_batch_size)?
                        {
                            batch_num += 1;
                            let batch_read_count = owned_records.len();
                            total_reads += batch_read_count;

                            let batch_refs: Vec<QueryRecord> = owned_records
                                .iter()
                                .map(|(id, s1, s2)| (*id, s1.as_slice(), s2.as_deref()))
                                .collect();

                            let results =
                                classify_batch(&engine, neg_mins.as_ref(), &batch_refs, threshold);

                            let mut chunk_out = Vec::with_capacity(1024);
                            for res in results {
                                let header = &headers[res.query_id as usize];
                                let bucket_name = engine
                                    .bucket_names
                                    .get(&res.bucket_id)
                                    .map(|s| s.as_str())
                                    .unwrap_or("unknown");
                                writeln!(
                                    chunk_out,
                                    "{}\t{}\t{:.4}",
                                    header, bucket_name, res.score
                                )
                                .unwrap();
                            }
                            io.write(chunk_out)?;

                            log::info!(
                                "Processed batch {}: {} reads ({} total)",
                                batch_num,
                                batch_read_count,
                                total_reads
                            );
                        }
                    }

                    log::info!("Classification complete: {} reads processed", total_reads);
                    io.finish()?;
                }
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
                log::info!("Loading index from {:?}", index);
                let engine = Index::load(&index)?;
                log::info!("Index loaded: {} buckets", engine.buckets.len());

                // Determine effective batch size: user override or adaptive
                let effective_batch_size = if let Some(bs) = batch_size {
                    log::info!("Using user-specified batch size: {}", bs);
                    bs
                } else {
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

                    let is_paired = r2.is_some();
                    let read_profile =
                        ReadMemoryProfile::from_files(&r1, r2.as_deref(), 1000, engine.k, engine.w)
                            .unwrap_or_else(|| {
                                log::warn!("Could not sample read lengths, using default profile");
                                ReadMemoryProfile::default_profile(is_paired, engine.k, engine.w)
                            });

                    let estimated_index_mem =
                        engine.buckets.values().map(|v| v.len() * 8).sum::<usize>();
                    let num_buckets = engine.buckets.len();

                    let mem_config = MemoryConfig {
                        max_memory: mem_limit,
                        num_threads: rayon::current_num_threads(),
                        index_memory: estimated_index_mem,
                        shard_reservation: 0,
                        read_profile,
                        num_buckets,
                    };

                    let batch_config = calculate_batch_config(&mem_config);
                    log::info!("Adaptive batch sizing: batch_size={}, parallel_batches={}, threads={}, estimated peak memory={}",
                        batch_config.batch_size, batch_config.batch_count, rayon::current_num_threads(), format_bytes(batch_config.peak_memory));
                    batch_config.batch_size
                };

                // Load negative index if provided, validate parameters, and build minimizer set
                let neg_mins: Option<HashSet<u64>> = match &negative_index {
                    None => None,
                    Some(neg_path) => {
                        log::info!("Loading negative index from {:?}", neg_path);
                        let neg = Index::load(neg_path)?;

                        // Validate parameters match
                        if neg.k != engine.k {
                            return Err(anyhow!(
                                "Negative index K ({}) does not match positive index K ({})",
                                neg.k,
                                engine.k
                            ));
                        }
                        if neg.w != engine.w {
                            return Err(anyhow!(
                                "Negative index W ({}) does not match positive index W ({})",
                                neg.w,
                                engine.w
                            ));
                        }
                        if neg.salt != engine.salt {
                            return Err(anyhow!(
                                "Negative index salt (0x{:x}) does not match positive index salt (0x{:x})",
                                neg.salt, engine.salt
                            ));
                        }

                        let total_mins: usize = neg.buckets.values().map(|v| v.len()).sum();
                        log::info!(
                            "Negative index loaded: {} buckets, {} minimizers",
                            neg.buckets.len(),
                            total_mins
                        );
                        let neg_set: HashSet<u64> = neg
                            .buckets
                            .values()
                            .flat_map(|v| v.iter().copied())
                            .collect();
                        log::info!(
                            "Built negative minimizer set: {} unique minimizers",
                            neg_set.len()
                        );
                        Some(neg_set)
                    }
                };

                let mut io = IoHandler::new(&r1, r2.as_ref(), output)?;
                io.write(b"query_name\tbucket_name\tscore\n".to_vec())?;

                let mut total_reads = 0;
                let mut batch_num = 0;

                log::info!(
                    "Starting aggregate classification (batch_size={})",
                    effective_batch_size
                );

                while let Some((owned_records, _)) = io.next_batch_records(effective_batch_size)? {
                    batch_num += 1;
                    let batch_read_count = owned_records.len();
                    total_reads += batch_read_count;

                    let batch_refs: Vec<QueryRecord> = owned_records
                        .iter()
                        .map(|(id, s1, s2)| (*id, s1.as_slice(), s2.as_deref()))
                        .collect();

                    let results =
                        aggregate_batch(&engine, neg_mins.as_ref(), &batch_refs, threshold);

                    let mut chunk_out = Vec::with_capacity(1024);
                    for res in results {
                        let bucket_name = engine
                            .bucket_names
                            .get(&res.bucket_id)
                            .map(|s| s.as_str())
                            .unwrap_or("unknown");
                        writeln!(chunk_out, "global\t{}\t{:.4}", bucket_name, res.score).unwrap();
                    }
                    io.write(chunk_out)?;

                    log::info!(
                        "Processed batch {}: {} reads ({} total)",
                        batch_num,
                        batch_read_count,
                        total_reads
                    );
                }

                log::info!(
                    "Aggregate classification complete: {} reads processed",
                    total_reads
                );
                io.finish()?;
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
