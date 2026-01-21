//! Index command handlers and helper functions.
//!
//! This module contains all the implementation logic for index-related commands:
//! create, stats, invert, merge, bucket-add, from-config, etc.

use anyhow::{anyhow, Context, Result};
use needletail::parse_fastx_file;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use std::time::Instant;

use rype::config::{
    parse_bucket_add_config, parse_config, resolve_path, validate_bucket_add_config,
    validate_config, AssignmentSettings, BestBinFallback,
};
use rype::parquet_index;
use rype::{
    extract_into, log_timing, Index, IndexMetadata, InvertedIndex, MainIndexManifest,
    MainIndexShard, MinimizerWorkspace, ShardFormat, ShardManifest, ShardedMainIndex,
    ShardedMainIndexBuilder,
};

use super::helpers::sanitize_bucket_name;

// ============================================================================
// Reference File Processing
// ============================================================================

/// Add sequences from a reference file to an index.
///
/// If `separate_buckets` is true, each sequence gets its own bucket.
/// Otherwise, all sequences go into bucket 1.
pub fn add_reference_file_to_index(
    index: &mut Index,
    path: &Path,
    separate_buckets: bool,
    next_id: &mut u32,
) -> Result<()> {
    log::info!("Adding reference: {}", path.display());
    let mut reader = parse_fastx_file(path).context("Failed to open reference file")?;
    let mut ws = MinimizerWorkspace::new();
    let filename = path.canonicalize().unwrap().to_string_lossy().to_string();
    let mut record_count = 0;

    while let Some(record) = reader.next() {
        let rec = record.context("Invalid record")?;
        let seq = rec.seq();
        let name = String::from_utf8_lossy(rec.id()).to_string();

        let bucket_id = if separate_buckets {
            let id = *next_id;
            *next_id += 1;
            index.bucket_names.insert(id, sanitize_bucket_name(&name));
            id
        } else {
            1
        };

        if !separate_buckets {
            index
                .bucket_names
                .entry(1)
                .or_insert_with(|| sanitize_bucket_name(&filename));
        }

        let source_label = format!("{}{}{}", filename, Index::BUCKET_SOURCE_DELIM, name);
        index.add_record(bucket_id, &source_label, &seq, &mut ws);

        record_count += 1;
        if record_count % 100_000 == 0 {
            log::info!("Processed {} records from {}", record_count, path.display());
        }
    }

    log::info!(
        "Finalized bucket processing for {}: {} total records",
        path.display(),
        record_count
    );

    if separate_buckets {
        let ids: Vec<u32> = index.buckets.keys().copied().collect();
        for id in ids {
            index.finalize_bucket(id);
        }
    } else {
        index.finalize_bucket(1);
    }

    Ok(())
}

/// Create Parquet inverted index directly from reference files.
#[allow(clippy::too_many_arguments)]
pub fn create_parquet_index_from_refs(
    output: &Path,
    references: &[PathBuf],
    k: usize,
    w: usize,
    salt: u64,
    separate_buckets: bool,
    max_shard_bytes: Option<usize>,
    options: Option<&parquet_index::ParquetWriteOptions>,
) -> Result<()> {
    use rype::{create_parquet_inverted_index, BucketData};

    log::info!(
        "Creating Parquet inverted index at {:?} (K={}, W={}, salt={:#x})",
        output,
        k,
        w,
        salt
    );

    let mut buckets: Vec<BucketData> = Vec::new();
    let mut next_id: u32 = 1;
    let mut ws = MinimizerWorkspace::new();

    for ref_path in references {
        log::info!("Processing reference: {}", ref_path.display());
        let mut reader = parse_fastx_file(ref_path).context("Failed to open reference file")?;
        let filename = ref_path
            .canonicalize()
            .unwrap()
            .to_string_lossy()
            .to_string();

        if separate_buckets {
            while let Some(record) = reader.next() {
                let rec = record.context("Invalid record")?;
                let seq = rec.seq();
                let name = String::from_utf8_lossy(rec.id()).to_string();
                let bucket_id = next_id;
                next_id += 1;

                extract_into(&seq, k, w, salt, &mut ws);
                let mut minimizers = std::mem::take(&mut ws.buffer);
                minimizers.sort_unstable();
                minimizers.dedup();

                let source_label = format!("{}{}{}", filename, Index::BUCKET_SOURCE_DELIM, name);
                buckets.push(BucketData {
                    bucket_id,
                    bucket_name: sanitize_bucket_name(&name),
                    sources: vec![source_label],
                    minimizers,
                });
            }
        } else {
            let bucket_id = next_id;
            next_id += 1;

            let mut all_minimizers: Vec<u64> = Vec::new();
            let mut sources: Vec<String> = Vec::new();

            while let Some(record) = reader.next() {
                let rec = record.context("Invalid record")?;
                let seq = rec.seq();
                let name = String::from_utf8_lossy(rec.id()).to_string();

                extract_into(&seq, k, w, salt, &mut ws);
                all_minimizers.extend_from_slice(&ws.buffer);

                let source_label = format!("{}{}{}", filename, Index::BUCKET_SOURCE_DELIM, name);
                sources.push(source_label);
            }

            all_minimizers.sort_unstable();
            all_minimizers.dedup();

            buckets.push(BucketData {
                bucket_id,
                bucket_name: sanitize_bucket_name(&filename),
                sources,
                minimizers: all_minimizers,
            });
        }
    }

    let total_minimizers: usize = buckets.iter().map(|b| b.minimizers.len()).sum();
    log::info!(
        "Extracted minimizers from {} buckets ({} total)",
        buckets.len(),
        total_minimizers
    );

    let manifest =
        create_parquet_inverted_index(output, buckets, k, w, salt, max_shard_bytes, options)?;

    log::info!("Created Parquet inverted index:");
    log::info!("  Buckets: {}", manifest.num_buckets);
    if let Some(ref inv) = manifest.inverted {
        log::info!("  Shards: {}", inv.num_shards);
        log::info!("  Total entries: {}", inv.total_entries);
    }
    log::info!("Done.");

    Ok(())
}

/// Create Parquet inverted index directly from a TOML config file.
/// This bypasses the main index entirely, creating only the parquet inverted index.
///
/// Reuses `build_single_bucket` for minimizer extraction and `create_parquet_inverted_index`
/// for the actual parquet file creation.
pub fn build_parquet_index_from_config(
    config_path: &Path,
    cli_max_shard_size: Option<usize>,
    options: Option<&parquet_index::ParquetWriteOptions>,
) -> Result<()> {
    use rype::{create_parquet_inverted_index, BucketData};

    let t_total = Instant::now();

    log::info!(
        "Building Parquet index from config: {}",
        config_path.display()
    );

    let cfg = parse_config(config_path)?;
    let config_dir = config_path
        .parent()
        .ok_or_else(|| anyhow!("Invalid config path"))?;

    log::info!("Validating file paths...");
    validate_config(&cfg, config_dir)?;
    log::info!("Validation successful.");

    let max_shard_size = cli_max_shard_size.or(cfg.index.max_shard_size);

    // Change output extension from .ryidx to .ryxdi for parquet inverted index
    let output_path = cfg.index.output.with_extension("ryxdi");
    let output_path = resolve_path(config_dir, &output_path);

    log::info!(
        "Creating Parquet inverted index at {:?} (K={}, W={}, salt={:#x})",
        output_path,
        cfg.index.k,
        cfg.index.window,
        cfg.index.salt
    );

    let mut bucket_names: Vec<_> = cfg.buckets.keys().cloned().collect();
    bucket_names.sort();

    const MAX_BUCKETS: usize = 100_000;
    if bucket_names.len() > MAX_BUCKETS {
        return Err(anyhow!(
            "Too many buckets: {} exceeds maximum {}",
            bucket_names.len(),
            MAX_BUCKETS
        ));
    }

    // Build buckets in parallel - reusing build_single_bucket
    let t_build = Instant::now();
    let bucket_results: Vec<_> = bucket_names
        .par_iter()
        .map(|bucket_name| {
            build_single_bucket(
                bucket_name,
                &cfg.buckets[bucket_name].files,
                config_dir,
                cfg.index.k,
                cfg.index.window,
                cfg.index.salt,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    log_timing(
        "parquet_index: bucket_building",
        t_build.elapsed().as_millis(),
    );

    // Convert to BucketData format for create_parquet_inverted_index
    let buckets: Vec<BucketData> = bucket_results
        .into_iter()
        .enumerate()
        .map(|(idx, (name, minimizers, sources))| BucketData {
            bucket_id: (idx + 1) as u32,
            bucket_name: sanitize_bucket_name(&name),
            sources,
            minimizers,
        })
        .collect();

    let total_minimizers: usize = buckets.iter().map(|b| b.minimizers.len()).sum();
    log::info!(
        "Extracted minimizers from {} buckets ({} total)",
        buckets.len(),
        total_minimizers
    );

    // Create parquet index - reusing create_parquet_inverted_index
    let t_write = Instant::now();
    let manifest = create_parquet_inverted_index(
        &output_path,
        buckets,
        cfg.index.k,
        cfg.index.window,
        cfg.index.salt,
        max_shard_size,
        options,
    )?;
    log_timing("parquet_index: write", t_write.elapsed().as_millis());

    log::info!("Created Parquet inverted index:");
    log::info!("  Buckets: {}", manifest.num_buckets);
    if let Some(ref inv) = manifest.inverted {
        log::info!("  Shards: {}", inv.num_shards);
        log::info!("  Total entries: {}", inv.total_entries);
    }

    log_timing("parquet_index: total", t_total.elapsed().as_millis());
    log::info!("Done.");

    Ok(())
}

// ============================================================================
// Config-based Index Building
// ============================================================================

/// Build a single bucket from its files, returning the name, minimizers, and sources.
pub fn build_single_bucket(
    bucket_name: &str,
    files: &[PathBuf],
    config_dir: &Path,
    k: usize,
    w: usize,
    salt: u64,
) -> Result<(String, Vec<u64>, Vec<String>)> {
    log::info!("Processing bucket '{}'...", bucket_name);
    let mut idx = Index::new(k, w, salt)?;
    let mut ws = MinimizerWorkspace::new();

    for file_path in files {
        let abs_path = resolve_path(config_dir, file_path);
        let mut reader = parse_fastx_file(&abs_path).context(format!(
            "Failed to open file {} for bucket '{}'",
            abs_path.display(),
            bucket_name
        ))?;

        let filename = file_path
            .canonicalize()
            .unwrap_or_else(|_| file_path.to_path_buf())
            .to_string_lossy()
            .to_string();

        while let Some(record) = reader.next() {
            let rec = record.context(format!(
                "Invalid record in file {} (bucket '{}')",
                abs_path.display(),
                bucket_name
            ))?;
            let seq_name = String::from_utf8_lossy(rec.id()).to_string();
            let source_label = format!("{}{}{}", filename, Index::BUCKET_SOURCE_DELIM, seq_name);
            idx.add_record(1, &source_label, &rec.seq(), &mut ws);
        }
    }

    idx.finalize_bucket(1);
    let minimizer_count = idx.buckets.get(&1).map(|v| v.len()).unwrap_or(0);
    log::info!(
        "Completed bucket '{}': {} minimizers",
        bucket_name,
        minimizer_count
    );

    let minimizers = idx.buckets.remove(&1).unwrap_or_default();
    let sources = idx.bucket_sources.remove(&1).unwrap_or_default();

    Ok((bucket_name.to_string(), minimizers, sources))
}

/// Build index from a TOML configuration file.
pub fn build_index_from_config(
    config_path: &Path,
    cli_max_shard_size: Option<usize>,
    cli_invert: bool,
) -> Result<()> {
    log::info!("Building index from config: {}", config_path.display());

    let cfg = parse_config(config_path)?;
    let config_dir = config_path
        .parent()
        .ok_or_else(|| anyhow!("Invalid config path"))?;

    log::info!("Validating file paths...");
    validate_config(&cfg, config_dir)?;
    log::info!("Validation successful.");

    let max_shard_size = cli_max_shard_size.or(cfg.index.max_shard_size);
    let should_invert = cli_invert || cfg.index.invert.unwrap_or(false);

    let mut bucket_names: Vec<_> = cfg.buckets.keys().cloned().collect();
    bucket_names.sort();

    const MAX_BUCKETS: usize = 100_000;
    if bucket_names.len() > MAX_BUCKETS {
        return Err(anyhow!(
            "Too many buckets: {} exceeds maximum {}",
            bucket_names.len(),
            MAX_BUCKETS
        ));
    }

    for name in &bucket_names {
        let file_count = cfg.buckets[name].files.len();
        log::info!(
            "  - {}: {} file{}",
            name,
            file_count,
            if file_count == 1 { "" } else { "s" }
        );
    }

    let num_threads = rayon::current_num_threads();
    if bucket_names.len() < num_threads {
        log::warn!(
            "Only {} bucket(s) but {} threads available. \
             Parallelism is over buckets, not files within a bucket.",
            bucket_names.len(),
            num_threads
        );
    }

    if let Some(max_bytes) = max_shard_size {
        build_sharded_index_from_config(&cfg, config_dir, &bucket_names, max_bytes, should_invert)
    } else {
        build_single_index_from_config(&cfg, config_dir, &bucket_names, should_invert)
    }
}

fn build_sharded_index_from_config(
    cfg: &rype::config::ConfigFile,
    config_dir: &Path,
    bucket_names: &[String],
    max_bytes: usize,
    should_invert: bool,
) -> Result<()> {
    let t_total = Instant::now();
    let batch_size = rayon::current_num_threads();
    log::info!(
        "Building {} buckets in batches of {} (memory-efficient mode)...",
        bucket_names.len(),
        batch_size
    );

    let mut builder = ShardedMainIndexBuilder::new(
        cfg.index.k,
        cfg.index.window,
        cfg.index.salt,
        &cfg.index.output,
        max_bytes,
    )?;

    let mut bucket_id = 1u32;
    let mut total_build_ms = 0u128;
    for (batch_idx, chunk) in bucket_names.chunks(batch_size).enumerate() {
        let t_batch = Instant::now();
        let batch_results: Vec<_> = chunk
            .par_iter()
            .map(|bucket_name| {
                build_single_bucket(
                    bucket_name,
                    &cfg.buckets[bucket_name].files,
                    config_dir,
                    cfg.index.k,
                    cfg.index.window,
                    cfg.index.salt,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        for (name, minimizers, sources) in batch_results {
            builder.add_bucket(bucket_id, &sanitize_bucket_name(&name), sources, minimizers)?;
            bucket_id += 1;
        }

        let batch_ms = t_batch.elapsed().as_millis();
        total_build_ms += batch_ms;
        log_timing(
            &format!("index_sharded: batch_{}_building", batch_idx),
            batch_ms,
        );

        log::info!(
            "Completed batch, {} buckets processed so far",
            bucket_id - 1
        );
    }
    log_timing("index_sharded: all_batches_building", total_build_ms);

    let t_finish = Instant::now();
    let manifest = builder.finish()?;
    log_timing("index_sharded: finish", t_finish.elapsed().as_millis());

    log::info!("Done! Sharded index saved successfully.");
    log::info!("\nFinal statistics:");
    log::info!("  Buckets: {}", bucket_id - 1);
    log::info!("  Shards: {}", manifest.shards.len());
    log::info!("  Total minimizers: {}", manifest.total_minimizers);

    if should_invert {
        let t_invert = Instant::now();
        create_inverted_from_sharded_main(&cfg.index.output, &manifest)?;
        log_timing("index_sharded: invert", t_invert.elapsed().as_millis());
    }

    log_timing("index_sharded: total", t_total.elapsed().as_millis());

    Ok(())
}

fn build_single_index_from_config(
    cfg: &rype::config::ConfigFile,
    config_dir: &Path,
    bucket_names: &[String],
    should_invert: bool,
) -> Result<()> {
    let t_total = Instant::now();
    log::info!("Building {} buckets in parallel...", bucket_names.len());

    let t_build = Instant::now();
    let bucket_results: Vec<_> = bucket_names
        .par_iter()
        .map(|bucket_name| {
            build_single_bucket(
                bucket_name,
                &cfg.buckets[bucket_name].files,
                config_dir,
                cfg.index.k,
                cfg.index.window,
                cfg.index.salt,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    log_timing("index: bucket_building", t_build.elapsed().as_millis());

    log::info!("Processing complete. Merging buckets...");

    let t_merge = Instant::now();
    let mut final_index = Index::new(cfg.index.k, cfg.index.window, cfg.index.salt)?;
    for (bucket_id, (bucket_name, minimizers, sources)) in bucket_results.into_iter().enumerate() {
        let new_id = (bucket_id + 1) as u32;
        final_index
            .bucket_names
            .insert(new_id, sanitize_bucket_name(&bucket_name));
        final_index.buckets.insert(new_id, minimizers);
        final_index.bucket_sources.insert(new_id, sources);
    }
    log_timing("index: bucket_merging", t_merge.elapsed().as_millis());

    log::info!("Saving index to {}...", cfg.index.output.display());
    let t_save = Instant::now();
    final_index.save(&cfg.index.output)?;
    log_timing("index: save", t_save.elapsed().as_millis());

    log::info!("Done! Index saved successfully.");
    log::info!("\nFinal statistics:");
    log::info!("  Buckets: {}", final_index.buckets.len());
    let total_minimizers: usize = final_index.buckets.values().map(|v| v.len()).sum();
    log::info!("  Total minimizers: {}", total_minimizers);

    if should_invert {
        let t_invert = Instant::now();
        create_inverted_from_single_index(&cfg.index.output, &final_index)?;
        log_timing("index: invert", t_invert.elapsed().as_millis());
    }

    log_timing("index: total", t_total.elapsed().as_millis());

    Ok(())
}

fn create_inverted_from_sharded_main(output: &Path, manifest: &MainIndexManifest) -> Result<()> {
    let inverted_path = output.with_extension("ryxdi");
    log::info!("Creating inverted index with 1:1 shard correspondence...");

    let mut inv_shards = Vec::new();
    let mut total_inv_minimizers = 0usize;
    let mut total_inv_bucket_ids = 0usize;
    let num_shards = manifest.shards.len();

    for (idx, shard_info) in manifest.shards.iter().enumerate() {
        let shard_path = MainIndexManifest::shard_path(output, shard_info.shard_id);
        log::info!(
            "Processing main shard {} for inversion...",
            shard_info.shard_id
        );

        let inverted = {
            let main_shard = MainIndexShard::load(&shard_path)?;
            InvertedIndex::build_from_shard(&main_shard)
        };

        log::info!(
            "  Built inverted: {} unique minimizers, {} bucket entries",
            inverted.num_minimizers(),
            inverted.num_bucket_entries()
        );

        let inv_shard_path = ShardManifest::shard_path(&inverted_path, shard_info.shard_id);
        let is_last = idx == num_shards - 1;
        let inv_shard_info = inverted.save_shard(
            &inv_shard_path,
            shard_info.shard_id,
            0,
            inverted.num_minimizers(),
            is_last,
        )?;

        total_inv_minimizers += inv_shard_info.num_minimizers;
        total_inv_bucket_ids += inv_shard_info.num_bucket_ids;
        inv_shards.push(inv_shard_info);
    }

    let main_metadata = manifest.to_metadata();
    let inv_manifest = ShardManifest {
        k: manifest.k,
        w: manifest.w,
        salt: manifest.salt,
        source_hash: InvertedIndex::compute_metadata_hash(&main_metadata),
        total_minimizers: total_inv_minimizers,
        total_bucket_ids: total_inv_bucket_ids,
        has_overlapping_shards: true,
        shard_format: ShardFormat::Legacy,
        shards: inv_shards,
        bucket_names: main_metadata.bucket_names,
        bucket_sources: main_metadata.bucket_sources,
        bucket_minimizer_counts: main_metadata.bucket_minimizer_counts,
    };

    let manifest_path = ShardManifest::manifest_path(&inverted_path);
    inv_manifest.save(&manifest_path)?;

    log::info!(
        "Created {} inverted shards (1:1 correspondence):",
        inv_manifest.shards.len()
    );
    for shard in &inv_manifest.shards {
        log::info!(
            "  Shard {}: {} unique minimizers, {} bucket entries",
            shard.shard_id,
            shard.num_minimizers,
            shard.num_bucket_ids
        );
    }

    Ok(())
}

fn create_inverted_from_single_index(output: &Path, final_index: &Index) -> Result<()> {
    let inverted_path = output.with_extension("ryxdi");
    log::info!("Building inverted index...");
    let inverted = InvertedIndex::build_from_index(final_index);
    log::info!(
        "Inverted index built: {} unique minimizers, {} bucket entries",
        inverted.num_minimizers(),
        inverted.num_bucket_entries()
    );

    let shard_path = ShardManifest::shard_path(&inverted_path, 0);
    let inv_shard_info = inverted.save_shard(&shard_path, 0, 0, inverted.num_minimizers(), true)?;

    let bucket_minimizer_counts: HashMap<u32, usize> = final_index
        .buckets
        .iter()
        .map(|(&id, mins)| (id, mins.len()))
        .collect();
    let metadata = IndexMetadata {
        k: final_index.k,
        w: final_index.w,
        salt: final_index.salt,
        bucket_names: final_index.bucket_names.clone(),
        bucket_sources: final_index.bucket_sources.clone(),
        bucket_minimizer_counts: bucket_minimizer_counts.clone(),
    };
    let inv_manifest = ShardManifest {
        k: final_index.k,
        w: final_index.w,
        salt: final_index.salt,
        source_hash: InvertedIndex::compute_metadata_hash(&metadata),
        total_minimizers: inv_shard_info.num_minimizers,
        total_bucket_ids: inv_shard_info.num_bucket_ids,
        has_overlapping_shards: true,
        shard_format: ShardFormat::Legacy,
        shards: vec![inv_shard_info],
        bucket_names: final_index.bucket_names.clone(),
        bucket_sources: final_index.bucket_sources.clone(),
        bucket_minimizer_counts,
    };

    let manifest_path = ShardManifest::manifest_path(&inverted_path);
    inv_manifest.save(&manifest_path)?;

    log::info!("Created 1 inverted shard:");
    log::info!(
        "  Shard 0: {} unique minimizers, {} bucket entries",
        inv_manifest.shards[0].num_minimizers,
        inv_manifest.shards[0].num_bucket_ids
    );

    Ok(())
}

// ============================================================================
// Bucket Add from Config
// ============================================================================

/// Represents a file assignment during bucket-add-config processing
#[derive(Debug)]
pub struct FileAssignment {
    pub file_path: PathBuf,
    pub bucket_id: u32,
    pub bucket_name: String,
    pub mode: &'static str,
    pub score: Option<f64>,
}

/// Data extracted from a single file in one pass
pub struct FileData {
    pub minimizers: Vec<u64>,
    pub sources: Vec<String>,
}

/// Extract minimizers and source labels from a file in a single pass
pub fn extract_file_data(path: &Path, k: usize, w: usize, salt: u64) -> Result<FileData> {
    let mut reader =
        parse_fastx_file(path).context(format!("Failed to open file {}", path.display()))?;

    let filename = path
        .canonicalize()
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .into_owned();

    let mut ws = MinimizerWorkspace::new();
    let mut all_mins = Vec::new();
    let mut sources = Vec::new();

    while let Some(record) = reader.next() {
        let rec = record.context(format!("Invalid record in file {}", path.display()))?;
        let seq_name = String::from_utf8_lossy(rec.id()).to_string();
        let source_label = format!("{}{}{}", filename, Index::BUCKET_SOURCE_DELIM, seq_name);
        sources.push(source_label);

        extract_into(&rec.seq(), k, w, salt, &mut ws);
        all_mins.extend_from_slice(&ws.buffer);
    }

    all_mins.sort_unstable();
    all_mins.dedup();

    Ok(FileData {
        minimizers: all_mins,
        sources,
    })
}

/// Count intersection using two-pointer merge - O(Q + B)
pub fn count_intersection_merge(query: &[u64], bucket: &[u64]) -> usize {
    let mut i = 0;
    let mut j = 0;
    let mut hits = 0;

    while i < query.len() && j < bucket.len() {
        match query[i].cmp(&bucket[j]) {
            std::cmp::Ordering::Equal => {
                hits += 1;
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    hits
}

/// Find bucket ID by name
pub fn find_bucket_by_name(bucket_names: &HashMap<u32, String>, name: &str) -> Result<u32> {
    for (&id, bucket_name) in bucket_names {
        if bucket_name == name {
            return Ok(id);
        }
    }
    let available: Vec<_> = bucket_names.values().collect();
    Err(anyhow!(
        "Bucket '{}' not found. Available buckets: {:?}",
        name,
        available
    ))
}

/// Index parameters for validation
pub struct IndexParams {
    pub k: usize,
    pub w: usize,
    pub salt: u64,
}

/// Handle bucket-add-config command
pub fn bucket_add_from_config(config_path: &Path) -> Result<()> {
    log::info!(
        "Adding files to index from config: {}",
        config_path.display()
    );

    let cfg = parse_bucket_add_config(config_path)?;
    let config_dir = config_path
        .parent()
        .ok_or_else(|| anyhow!("Invalid config path"))?;

    log::info!("Validating file paths...");
    validate_bucket_add_config(&cfg, config_dir)?;
    log::info!("Validation successful.");

    let index_path = resolve_path(config_dir, &cfg.target.index);

    if MainIndexManifest::is_sharded(&index_path) {
        bucket_add_sharded(&index_path, &cfg, config_dir)
    } else {
        bucket_add_single(&index_path, &cfg, config_dir)
    }
}

/// Handle bucket-add-config for single-file indices
pub fn bucket_add_single(
    index_path: &Path,
    cfg: &rype::config::BucketAddConfig,
    config_dir: &Path,
) -> Result<()> {
    let mut index = Index::load(index_path)?;
    let mut assignments: Vec<FileAssignment> = Vec::new();
    let params = IndexParams {
        k: index.k,
        w: index.w,
        salt: index.salt,
    };

    log::info!(
        "Loaded index with {} existing buckets (K={}, W={}, salt={:#x})",
        index.buckets.len(),
        params.k,
        params.w,
        params.salt
    );

    let resolved_files: Vec<PathBuf> = cfg
        .files
        .paths
        .iter()
        .map(|p| resolve_path(config_dir, p))
        .collect();

    match &cfg.assignment {
        AssignmentSettings::NewBucket { bucket_name } => {
            let new_id = index.buckets.keys().max().map(|m| m + 1).unwrap_or(1);
            // Use provided bucket name or derive from first file
            let effective_bucket_name = bucket_name.clone().unwrap_or_else(|| {
                resolved_files
                    .first()
                    .map(|p| {
                        p.file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .to_string()
                    })
                    .unwrap_or_else(|| format!("bucket_{}", new_id))
            });
            log::info!(
                "Creating new bucket '{}' (ID={})",
                effective_bucket_name,
                new_id
            );

            for file_path in &resolved_files {
                let data = extract_file_data(file_path, params.k, params.w, params.salt)?;
                for source in data.sources {
                    index.bucket_sources.entry(new_id).or_default().push(source);
                }
                for m in data.minimizers {
                    index.buckets.entry(new_id).or_default().push(m);
                }
                assignments.push(FileAssignment {
                    file_path: file_path.clone(),
                    bucket_id: new_id,
                    bucket_name: effective_bucket_name.clone(),
                    mode: "new_bucket",
                    score: None,
                });
            }

            index.bucket_names.insert(new_id, effective_bucket_name);
            index.finalize_bucket(new_id);
        }

        AssignmentSettings::ExistingBucket { bucket_name } => {
            let bucket_id = find_bucket_by_name(&index.bucket_names, bucket_name)?;
            log::info!(
                "Adding to existing bucket '{}' (ID={})",
                bucket_name,
                bucket_id
            );

            for file_path in &resolved_files {
                let data = extract_file_data(file_path, params.k, params.w, params.salt)?;
                for source in data.sources {
                    index
                        .bucket_sources
                        .entry(bucket_id)
                        .or_default()
                        .push(source);
                }
                for m in data.minimizers {
                    index.buckets.entry(bucket_id).or_default().push(m);
                }
                assignments.push(FileAssignment {
                    file_path: file_path.clone(),
                    bucket_id,
                    bucket_name: bucket_name.clone(),
                    mode: "existing_bucket",
                    score: None,
                });
            }

            index.finalize_bucket(bucket_id);
        }

        AssignmentSettings::BestBin {
            threshold,
            fallback,
        } => {
            log::info!(
                "Best-bin assignment: threshold={}, fallback={:?}",
                threshold,
                fallback
            );
            best_bin_assign(
                &mut index,
                &resolved_files,
                *threshold,
                fallback,
                &params,
                &mut assignments,
            )?;
        }
    }

    log::info!("Saving updated index...");
    index.save(index_path)?;
    print_assignment_summary(&assignments);

    Ok(())
}

/// Handle bucket-add-config for sharded indices
pub fn bucket_add_sharded(
    index_path: &Path,
    cfg: &rype::config::BucketAddConfig,
    config_dir: &Path,
) -> Result<()> {
    let sharded = ShardedMainIndex::open(index_path)?;
    let manifest = sharded.manifest();
    let params = IndexParams {
        k: manifest.k,
        w: manifest.w,
        salt: manifest.salt,
    };

    log::info!(
        "Loaded sharded index with {} buckets across {} shards",
        manifest.bucket_names.len(),
        manifest.shards.len()
    );

    let resolved_files: Vec<PathBuf> = cfg
        .files
        .paths
        .iter()
        .map(|p| resolve_path(config_dir, p))
        .collect();

    let mut assignments: Vec<FileAssignment> = Vec::new();

    match &cfg.assignment {
        AssignmentSettings::BestBin {
            threshold,
            fallback,
        } => {
            best_bin_assign_sharded(
                &sharded,
                index_path,
                &resolved_files,
                *threshold,
                fallback,
                &params,
                &mut assignments,
            )?;
        }
        _ => {
            return Err(anyhow!(
                "Only 'best_bin' assignment mode is supported for sharded indices"
            ));
        }
    }

    print_assignment_summary(&assignments);
    Ok(())
}

/// Best-bin assignment for single-file indices
fn best_bin_assign(
    index: &mut Index,
    files: &[PathBuf],
    threshold: f64,
    fallback: &BestBinFallback,
    params: &IndexParams,
    assignments: &mut Vec<FileAssignment>,
) -> Result<()> {
    for file_path in files {
        let data = extract_file_data(file_path, params.k, params.w, params.salt)?;

        if data.minimizers.is_empty() {
            handle_no_buckets(file_path, fallback, assignments)?;
            continue;
        }

        let mut best_id = None;
        let mut best_score = 0.0;
        let mut best_name = String::new();

        for (&bucket_id, bucket_mins) in &index.buckets {
            let hits = count_intersection_merge(&data.minimizers, bucket_mins);
            let score = hits as f64 / data.minimizers.len() as f64;

            if score > best_score {
                best_score = score;
                best_id = Some(bucket_id);
                best_name = index
                    .bucket_names
                    .get(&bucket_id)
                    .cloned()
                    .unwrap_or_else(|| format!("bucket_{}", bucket_id));
            }
        }

        if best_score >= threshold {
            let bucket_id = best_id.unwrap();
            log::info!(
                "File {} -> bucket '{}' (score={:.3})",
                file_path.display(),
                best_name,
                best_score
            );

            for source in &data.sources {
                index
                    .bucket_sources
                    .entry(bucket_id)
                    .or_default()
                    .push(source.clone());
            }
            for m in &data.minimizers {
                index.buckets.entry(bucket_id).or_default().push(*m);
            }
            index.finalize_bucket(bucket_id);

            assignments.push(FileAssignment {
                file_path: file_path.clone(),
                bucket_id,
                bucket_name: best_name,
                mode: "matched",
                score: Some(best_score),
            });
        } else {
            handle_below_threshold(
                file_path,
                &data,
                index,
                fallback,
                best_score,
                &best_name,
                assignments,
            )?;
        }
    }

    Ok(())
}

/// Handle files below threshold in best-bin assignment
fn handle_below_threshold(
    file_path: &Path,
    data: &FileData,
    index: &mut Index,
    fallback: &BestBinFallback,
    best_score: f64,
    best_name: &str,
    assignments: &mut Vec<FileAssignment>,
) -> Result<()> {
    match fallback {
        BestBinFallback::CreateNew => {
            let new_id = index.buckets.keys().max().map(|m| m + 1).unwrap_or(1);
            let new_name = get_file_stem(file_path);
            log::info!(
                "File {} below threshold (best={:.3} to '{}'), creating new bucket '{}'",
                file_path.display(),
                best_score,
                best_name,
                new_name
            );

            for source in &data.sources {
                index
                    .bucket_sources
                    .entry(new_id)
                    .or_default()
                    .push(source.clone());
            }
            index.buckets.insert(new_id, data.minimizers.clone());
            index.bucket_names.insert(new_id, new_name.clone());
            index.finalize_bucket(new_id);

            assignments.push(FileAssignment {
                file_path: file_path.to_path_buf(),
                bucket_id: new_id,
                bucket_name: new_name,
                mode: "created",
                score: Some(best_score),
            });
        }
        BestBinFallback::Skip => {
            log::info!(
                "File {} below threshold (best={:.3}), skipping",
                file_path.display(),
                best_score
            );
        }
        BestBinFallback::Error => {
            return Err(anyhow!(
                "File {} below threshold (best score={:.3} to '{}')",
                file_path.display(),
                best_score,
                best_name
            ));
        }
    }

    Ok(())
}

/// Handle files with no matching buckets
fn handle_no_buckets(
    file_path: &Path,
    fallback: &BestBinFallback,
    _assignments: &mut [FileAssignment],
) -> Result<()> {
    match fallback {
        BestBinFallback::CreateNew => {
            log::warn!(
                "File {} produced no minimizers, cannot create bucket",
                file_path.display()
            );
        }
        BestBinFallback::Skip => {
            log::info!(
                "File {} produced no minimizers, skipping",
                file_path.display()
            );
        }
        BestBinFallback::Error => {
            return Err(anyhow!(
                "File {} produced no minimizers",
                file_path.display()
            ));
        }
    }

    Ok(())
}

/// Get file stem for bucket naming
pub fn get_file_stem(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string()
}

/// Best-bin assignment for sharded indices
fn best_bin_assign_sharded(
    sharded: &ShardedMainIndex,
    index_path: &Path,
    files: &[PathBuf],
    threshold: f64,
    fallback: &BestBinFallback,
    params: &IndexParams,
    assignments: &mut Vec<FileAssignment>,
) -> Result<()> {
    let manifest = sharded.manifest();

    for file_path in files {
        let data = extract_file_data(file_path, params.k, params.w, params.salt)?;

        if data.minimizers.is_empty() {
            handle_no_buckets(file_path, fallback, assignments)?;
            continue;
        }

        let mut best_id = None;
        let mut best_score = 0.0;
        let mut best_name = String::new();
        let mut best_shard_id = 0u32;

        for shard_info in &manifest.shards {
            let shard = sharded.load_shard(shard_info.shard_id)?;

            for (&bucket_id, bucket_mins) in &shard.buckets {
                let hits = count_intersection_merge(&data.minimizers, bucket_mins);
                let score = hits as f64 / data.minimizers.len() as f64;

                if score > best_score {
                    best_score = score;
                    best_id = Some(bucket_id);
                    best_shard_id = shard_info.shard_id;
                    best_name = manifest
                        .bucket_names
                        .get(&bucket_id)
                        .cloned()
                        .unwrap_or_else(|| format!("bucket_{}", bucket_id));
                }
            }
        }

        if best_score >= threshold {
            let bucket_id = best_id.unwrap();
            log::info!(
                "File {} -> bucket '{}' in shard {} (score={:.3})",
                file_path.display(),
                best_name,
                best_shard_id,
                best_score
            );

            // Load shard, modify, and save
            let shard_path = MainIndexManifest::shard_path(index_path, best_shard_id);
            let mut shard = MainIndexShard::load(&shard_path)?;

            for source in &data.sources {
                shard
                    .bucket_sources
                    .entry(bucket_id)
                    .or_default()
                    .push(source.clone());
            }
            if let Some(mins) = shard.buckets.get_mut(&bucket_id) {
                mins.extend(&data.minimizers);
                mins.sort_unstable();
                mins.dedup();
            }

            shard.save(&shard_path)?;

            assignments.push(FileAssignment {
                file_path: file_path.clone(),
                bucket_id,
                bucket_name: best_name,
                mode: "matched",
                score: Some(best_score),
            });
        } else {
            match fallback {
                BestBinFallback::Skip => {
                    log::info!(
                        "File {} below threshold (best={:.3}), skipping",
                        file_path.display(),
                        best_score
                    );
                }
                BestBinFallback::Error => {
                    return Err(anyhow!(
                        "File {} below threshold (best score={:.3} to '{}')",
                        file_path.display(),
                        best_score,
                        best_name
                    ));
                }
                BestBinFallback::CreateNew => {
                    return Err(anyhow!(
                        "create_new fallback not supported for sharded indices"
                    ));
                }
            }
        }
    }

    Ok(())
}

/// Print summary of file assignments
pub fn print_assignment_summary(assignments: &[FileAssignment]) {
    if assignments.is_empty() {
        log::info!("No files were assigned.");
        return;
    }

    let mut by_bucket: HashMap<u32, Vec<&FileAssignment>> = HashMap::new();
    for a in assignments {
        by_bucket.entry(a.bucket_id).or_default().push(a);
    }

    println!("\n========== ASSIGNMENT SUMMARY ==========");
    let mut bucket_ids: Vec<_> = by_bucket.keys().collect();
    bucket_ids.sort();

    for bucket_id in bucket_ids {
        let bucket_assignments = &by_bucket[bucket_id];
        let bucket_name = &bucket_assignments[0].bucket_name;
        println!(
            "\nBucket '{}' (ID={}) - {} file(s):",
            bucket_name,
            bucket_id,
            bucket_assignments.len()
        );

        for a in bucket_assignments {
            let score_str = a
                .score
                .map(|s| format!(" (score={:.3})", s))
                .unwrap_or_default();
            println!("  [{}] {}{}", a.mode, a.file_path.display(), score_str);
        }
    }

    println!(
        "\n{} files assigned to {} bucket(s)",
        assignments.len(),
        by_bucket.len()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use rype::BucketData;
    use std::fs::File;
    use std::io::Write;
    use tempfile::TempDir;

    /// Helper to create a simple FASTA file with one sequence
    fn create_fasta_file(dir: &Path, name: &str, seq: &[u8]) -> PathBuf {
        let path = dir.join(name);
        let mut file = File::create(&path).unwrap();
        writeln!(file, ">seq1").unwrap();
        file.write_all(seq).unwrap();
        writeln!(file).unwrap();
        path
    }

    /// Helper to create a config file for testing
    fn create_test_config(
        dir: &Path,
        output_name: &str,
        buckets: &[(&str, &[&str])],
        k: usize,
        window: usize,
    ) -> PathBuf {
        let config_path = dir.join("config.toml");
        let mut content = format!(
            r#"[index]
k = {}
window = {}
salt = 0x5555555555555555
output = "{}"

"#,
            k, window, output_name
        );

        for (bucket_name, files) in buckets {
            let files_str: Vec<String> = files.iter().map(|f| format!("\"{}\"", f)).collect();
            content.push_str(&format!(
                "[buckets.{}]\nfiles = [{}]\n\n",
                bucket_name,
                files_str.join(", ")
            ));
        }

        let mut file = File::create(&config_path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
        config_path
    }

    #[test]
    fn test_build_single_bucket_extracts_minimizers() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // Create a FASTA file with a sequence long enough for k=32, w=10
        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let fasta_path = create_fasta_file(dir, "test.fa", seq);

        let (name, minimizers, sources) = build_single_bucket(
            "TestBucket",
            &[fasta_path],
            dir,
            32, // k
            10, // w
            0x5555555555555555,
        )
        .unwrap();

        assert_eq!(name, "TestBucket");
        assert!(!minimizers.is_empty(), "Should extract some minimizers");
        assert!(!sources.is_empty(), "Should have source labels");

        // Verify minimizers are sorted and deduplicated
        let mut sorted = minimizers.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            minimizers, sorted,
            "Minimizers should be sorted and deduplicated"
        );
    }

    #[test]
    fn test_bucket_result_to_bucket_data_conversion() {
        // Test that build_single_bucket output can be converted to BucketData
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let fasta_path = create_fasta_file(dir, "test.fa", seq);

        let (name, minimizers, sources) =
            build_single_bucket("TestBucket", &[fasta_path], dir, 32, 10, 0x5555555555555555)
                .unwrap();

        // Convert to BucketData (this is the reuse we want)
        let bucket_data = BucketData {
            bucket_id: 1,
            bucket_name: sanitize_bucket_name(&name),
            sources,
            minimizers,
        };

        assert_eq!(bucket_data.bucket_id, 1);
        assert!(!bucket_data.minimizers.is_empty());
        assert!(bucket_data.validate().is_ok(), "BucketData should be valid");
    }

    #[test]
    fn test_build_parquet_index_from_config_creates_index() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // Create test FASTA files
        let seq1 = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let seq2 = b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA";

        create_fasta_file(dir, "ref1.fa", seq1);
        create_fasta_file(dir, "ref2.fa", seq2);

        // Create config
        let config_path = create_test_config(
            dir,
            "test_index.ryidx",
            &[("Bucket1", &["ref1.fa"]), ("Bucket2", &["ref2.fa"])],
            32,
            10,
        );

        // Build parquet index
        let result = build_parquet_index_from_config(&config_path, None, None);
        assert!(result.is_ok(), "Should succeed: {:?}", result);

        // Verify the parquet index was created
        let output_path = dir.join("test_index.ryxdi");
        assert!(output_path.exists(), "Parquet index directory should exist");

        // Verify manifest exists
        let manifest_path = output_path.join("manifest.toml");
        assert!(manifest_path.exists(), "Manifest should exist");
    }

    #[test]
    fn test_build_parquet_index_from_config_with_bloom_filter() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        create_fasta_file(dir, "ref.fa", seq);

        let config_path = create_test_config(
            dir,
            "bloom_test.ryidx",
            &[("TestBucket", &["ref.fa"])],
            32,
            10,
        );

        let options = parquet_index::ParquetWriteOptions {
            bloom_filter_enabled: true,
            bloom_filter_fpp: 0.05,
            ..Default::default()
        };

        let result = build_parquet_index_from_config(&config_path, None, Some(&options));
        assert!(
            result.is_ok(),
            "Should succeed with bloom filter: {:?}",
            result
        );

        let output_path = dir.join("bloom_test.ryxdi");
        assert!(output_path.exists());
    }

    #[test]
    fn test_build_parquet_index_from_config_invalid_file() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // Create config referencing non-existent file
        let config_path = create_test_config(
            dir,
            "test.ryidx",
            &[("TestBucket", &["nonexistent.fa"])],
            32,
            10,
        );

        let result = build_parquet_index_from_config(&config_path, None, None);
        assert!(result.is_err(), "Should fail with missing file");
    }
}
