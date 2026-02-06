//! Index merge functionality.
//!
//! This module provides the ability to merge two Parquet indices into one,
//! with optional subtraction of minimizers from the secondary index that
//! exist in the primary index.

use crate::error::{Result, RypeError};
use crate::indices::sharded::{ShardManifest, ShardedInvertedIndex};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::Path;

use super::streaming::{compute_source_hash, ShardAccumulator, MIN_SHARD_BYTES};
use super::{
    create_index_directory, write_buckets_parquet, InvertedManifest, InvertedShardInfo,
    ParquetManifest, ParquetShardFormat, ParquetWriteOptions, FORMAT_MAGIC, FORMAT_VERSION,
};

/// Options for index merging.
#[derive(Debug, Clone, Default)]
pub struct MergeOptions {
    /// If true, remove minimizers from secondary that exist in primary.
    pub subtract_from_primary: bool,
    /// If true, emit progress messages to stderr.
    pub verbose: bool,
}

/// Validate that two indices are compatible for merging.
///
/// Indices must have matching k, w, and salt values. Bucket names must be
/// unique across both indices (no overlapping names).
///
/// # Errors
///
/// Returns an error if:
/// - k values don't match
/// - w values don't match
/// - salt values don't match
/// - Any bucket name appears in both indices
pub fn validate_merge_compatibility(
    primary: &ShardedInvertedIndex,
    secondary: &ShardedInvertedIndex,
) -> Result<()> {
    // Check k values match
    if primary.k() != secondary.k() {
        return Err(RypeError::validation(format!(
            "k mismatch: primary has k={}, secondary has k={}",
            primary.k(),
            secondary.k()
        )));
    }

    // Check w values match
    if primary.w() != secondary.w() {
        return Err(RypeError::validation(format!(
            "w mismatch: primary has w={}, secondary has w={}",
            primary.w(),
            secondary.w()
        )));
    }

    // Check salt values match
    if primary.salt() != secondary.salt() {
        return Err(RypeError::validation(format!(
            "salt mismatch: primary has salt={:#x}, secondary has salt={:#x}",
            primary.salt(),
            secondary.salt()
        )));
    }

    // Check for duplicate bucket names
    let primary_names: std::collections::HashSet<&str> = primary
        .manifest()
        .bucket_names
        .values()
        .map(|s| s.as_str())
        .collect();

    for name in secondary.manifest().bucket_names.values() {
        if primary_names.contains(name.as_str()) {
            return Err(RypeError::validation(format!(
                "duplicate bucket name '{}' found in both indices",
                name
            )));
        }
    }

    Ok(())
}

/// Result of computing bucket ID remapping for a merge operation.
///
/// Contains the combined bucket metadata and mappings from old bucket IDs
/// to new sequential IDs starting from 1.
#[derive(Debug, Clone)]
pub struct RemappedBuckets {
    /// Combined bucket names (new_id -> name).
    pub bucket_names: HashMap<u32, String>,
    /// Combined bucket sources (new_id -> sources).
    pub bucket_sources: HashMap<u32, Vec<String>>,
    /// Mapping from primary old bucket ID to new ID.
    pub primary_id_map: HashMap<u32, u32>,
    /// Mapping from secondary old bucket ID to new ID.
    pub secondary_id_map: HashMap<u32, u32>,
}

/// Compute bucket ID remapping for merging two indices.
///
/// Assigns new sequential bucket IDs starting from 1:
/// - Primary buckets come first (sorted by original ID)
/// - Secondary buckets follow (sorted by original ID)
///
/// # Arguments
///
/// * `primary` - Manifest of the primary index
/// * `secondary` - Manifest of the secondary index
///
/// # Returns
///
/// A `RemappedBuckets` struct containing:
/// - Combined bucket names and sources with new IDs
/// - Mappings from old IDs to new IDs for both indices
pub fn compute_bucket_remapping(
    primary: &ShardManifest,
    secondary: &ShardManifest,
) -> RemappedBuckets {
    let mut bucket_names = HashMap::new();
    let mut bucket_sources = HashMap::new();
    let mut primary_id_map = HashMap::new();
    let mut secondary_id_map = HashMap::new();

    let mut next_id: u32 = 1;

    // Process primary buckets first, sorted by original ID for deterministic ordering
    let mut primary_ids: Vec<u32> = primary.bucket_names.keys().copied().collect();
    primary_ids.sort();

    for old_id in primary_ids {
        let new_id = next_id;
        next_id += 1;

        primary_id_map.insert(old_id, new_id);

        if let Some(name) = primary.bucket_names.get(&old_id) {
            bucket_names.insert(new_id, name.clone());
        }
        if let Some(sources) = primary.bucket_sources.get(&old_id) {
            bucket_sources.insert(new_id, sources.clone());
        }
    }

    // Process secondary buckets, sorted by original ID
    let mut secondary_ids: Vec<u32> = secondary.bucket_names.keys().copied().collect();
    secondary_ids.sort();

    for old_id in secondary_ids {
        let new_id = next_id;
        next_id += 1;

        secondary_id_map.insert(old_id, new_id);

        if let Some(name) = secondary.bucket_names.get(&old_id) {
            bucket_names.insert(new_id, name.clone());
        }
        if let Some(sources) = secondary.bucket_sources.get(&old_id) {
            bucket_sources.insert(new_id, sources.clone());
        }
    }

    RemappedBuckets {
        bucket_names,
        bucket_sources,
        primary_id_map,
        secondary_id_map,
    }
}

/// Statistics from a merge operation.
#[derive(Debug, Clone)]
pub struct MergeStats {
    /// Total number of buckets in the merged index.
    pub total_buckets: u32,
    /// Total number of (minimizer, bucket_id) entries in the merged index.
    pub total_minimizer_entries: u64,
    /// Number of buckets from the primary index.
    pub primary_buckets: u32,
    /// Number of entries from the primary index.
    pub primary_entries: u64,
    /// Number of buckets from the secondary index.
    pub secondary_buckets: u32,
    /// Number of entries from the secondary index (after any subtraction).
    pub secondary_entries: u64,
    /// Original number of entries in the secondary index (before subtraction).
    pub secondary_entries_original: u64,
    /// Number of unique minimizers excluded by subtraction (0 if no subtraction).
    pub excluded_minimizers: usize,
}

/// Load all unique minimizers from a sharded index into a HashSet.
///
/// This loads the entire index into memory once, which is efficient when the
/// index is small relative to the queries (e.g., 80M minimizers = ~2GB with HashSet overhead).
///
/// # Memory Usage
/// HashSet overhead is approximately 24 bytes per entry (8 bytes for u64 + 16 bytes overhead).
/// For 80M minimizers: ~2GB. For 1B minimizers: ~24GB.
fn load_all_minimizers(index: &ShardedInvertedIndex) -> Result<HashSet<u64>> {
    let mut minimizers = HashSet::new();

    for shard_info in &index.manifest().shards {
        let shard_path = index.shard_path(shard_info.shard_id);
        let pairs = read_shard_pairs(&shard_path)?;
        for (minimizer, _) in pairs {
            minimizers.insert(minimizer);
        }
    }

    Ok(minimizers)
}

/// Result from processing a shard with row-group parallelism.
struct RowGroupProcessingResult {
    /// Filtered (minimizer, new_bucket_id) pairs ready for output
    pairs: Vec<(u64, u32)>,
    /// Count of entries written per OLD bucket ID
    bucket_counts: HashMap<u32, u64>,
    /// Count of original entries per OLD bucket ID
    bucket_counts_original: HashMap<u32, u64>,
    /// Count of minimizers per NEW bucket ID (for source hash)
    bucket_minimizer_counts: HashMap<u32, usize>,
    /// Total entries excluded
    entries_excluded: u64,
}

/// Process a single shard with row-group level parallelism.
///
/// This is the optimized version that parallelizes over row groups within a shard,
/// enabling full parallelism even when there's only one shard.
///
/// # Arguments
/// * `shard_path` - Path to the shard Parquet file
/// * `exclusion_set` - Pre-computed HashSet of minimizers to exclude
/// * `remapping` - Mapping from old bucket IDs to new bucket IDs
///
/// # Returns
/// RowGroupProcessingResult containing filtered pairs and statistics
fn process_shard_parallel_row_groups(
    shard_path: &Path,
    exclusion_set: &HashSet<u64>,
    remapping: &HashMap<u32, u32>,
) -> Result<RowGroupProcessingResult> {
    use arrow::array::{UInt32Array, UInt64Array};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use parquet::file::reader::FileReader;
    use parquet::file::serialized_reader::SerializedFileReader;

    // Get row group count
    let file = File::open(shard_path)
        .map_err(|e| RypeError::io(shard_path.to_path_buf(), "open shard for metadata", e))?;
    let reader = SerializedFileReader::new(file)?;
    let num_row_groups = reader.metadata().num_row_groups();
    drop(reader);

    if num_row_groups == 0 {
        return Ok(RowGroupProcessingResult {
            pairs: Vec::new(),
            bucket_counts: HashMap::new(),
            bucket_counts_original: HashMap::new(),
            bucket_minimizer_counts: HashMap::new(),
            entries_excluded: 0,
        });
    }

    log::debug!(
        "Processing {} row groups in parallel from {}",
        num_row_groups,
        shard_path.display()
    );

    // Process row groups in parallel
    let row_group_indices: Vec<usize> = (0..num_row_groups).collect();
    let shard_path_buf = shard_path.to_path_buf();

    // Type alias to satisfy clippy::type_complexity
    type RowGroupResult = (
        Vec<(u64, u32)>,
        HashMap<u32, u64>,
        HashMap<u32, u64>,
        HashMap<u32, usize>,
        u64,
    );

    let results: Vec<Result<RowGroupResult>> = row_group_indices
        .par_iter()
        .map(|&rg_idx| {
            // Each thread opens its own file handle (OS page cache handles deduplication)
            let file = File::open(&shard_path_buf)
                .map_err(|e| RypeError::io(shard_path_buf.clone(), "open shard", e))?;
            let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
            let reader = builder.with_row_groups(vec![rg_idx]).build()?;

            let mut filtered_pairs = Vec::new();
            let mut bucket_counts: HashMap<u32, u64> = HashMap::new();
            let mut bucket_counts_original: HashMap<u32, u64> = HashMap::new();
            let mut bucket_minimizer_counts: HashMap<u32, usize> = HashMap::new();
            let mut excluded = 0u64;

            for batch in reader {
                let batch = batch?;

                let minimizers = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| {
                        RypeError::format(
                            &shard_path_buf,
                            "Expected UInt64Array for minimizer column",
                        )
                    })?;

                let bucket_ids = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .ok_or_else(|| {
                        RypeError::format(
                            &shard_path_buf,
                            "Expected UInt32Array for bucket_id column",
                        )
                    })?;

                for i in 0..batch.num_rows() {
                    let minimizer = minimizers.value(i);
                    let old_bucket_id = bucket_ids.value(i);

                    // Track original count
                    *bucket_counts_original.entry(old_bucket_id).or_insert(0) += 1;

                    // O(1) HashSet lookup for exclusion
                    if exclusion_set.contains(&minimizer) {
                        excluded += 1;
                        continue;
                    }

                    // Remap bucket ID
                    let new_bucket_id = *remapping.get(&old_bucket_id).ok_or_else(|| {
                        RypeError::validation(format!(
                            "Secondary bucket ID {} not found in remapping",
                            old_bucket_id
                        ))
                    })?;

                    filtered_pairs.push((minimizer, new_bucket_id));
                    *bucket_counts.entry(old_bucket_id).or_insert(0) += 1;
                    *bucket_minimizer_counts.entry(new_bucket_id).or_insert(0) += 1;
                }
            }

            Ok((
                filtered_pairs,
                bucket_counts,
                bucket_counts_original,
                bucket_minimizer_counts,
                excluded,
            ))
        })
        .collect();

    // Merge results from all row groups
    let mut all_pairs = Vec::new();
    let mut combined_bucket_counts: HashMap<u32, u64> = HashMap::new();
    let mut combined_bucket_counts_original: HashMap<u32, u64> = HashMap::new();
    let mut combined_bucket_minimizer_counts: HashMap<u32, usize> = HashMap::new();
    let mut total_excluded = 0u64;

    for (rg_idx, result) in results.into_iter().enumerate() {
        let (pairs, bucket_counts, bucket_counts_original, bucket_minimizer_counts, excluded) =
            result.map_err(|e| RypeError::Parquet {
                context: format!("failed to process row group {}: {}", rg_idx, e),
                source: None,
            })?;

        all_pairs.extend(pairs);
        total_excluded += excluded;

        for (k, v) in bucket_counts {
            *combined_bucket_counts.entry(k).or_insert(0) += v;
        }
        for (k, v) in bucket_counts_original {
            *combined_bucket_counts_original.entry(k).or_insert(0) += v;
        }
        for (k, v) in bucket_minimizer_counts {
            *combined_bucket_minimizer_counts.entry(k).or_insert(0) += v;
        }
    }

    Ok(RowGroupProcessingResult {
        pairs: all_pairs,
        bucket_counts: combined_bucket_counts,
        bucket_counts_original: combined_bucket_counts_original,
        bucket_minimizer_counts: combined_bucket_minimizer_counts,
        entries_excluded: total_excluded,
    })
}

/// Helper to finish a merge operation: write manifest and return stats.
#[allow(clippy::too_many_arguments)]
fn finish_merge(
    output_path: &Path,
    primary: &ShardedInvertedIndex,
    remapped: &RemappedBuckets,
    shard_infos: Vec<InvertedShardInfo>,
    bucket_minimizer_counts: &HashMap<u32, usize>,
    primary_entries: u64,
    secondary_entries: u64,
    secondary_entries_original: u64,
    total_excluded: u64,
    options: &MergeOptions,
) -> Result<MergeStats> {
    // Compute total entries
    let total_entries: u64 = shard_infos.iter().map(|s| s.num_entries).sum();

    // Merged indices have overlapping shard ranges
    let has_overlapping_shards = true;

    // Write buckets.parquet
    write_buckets_parquet(
        output_path,
        &remapped.bucket_names,
        &remapped.bucket_sources,
    )?;

    // Compute source hash for the merged index
    let source_hash = compute_source_hash(bucket_minimizer_counts);

    // Build and save manifest
    let manifest = ParquetManifest {
        magic: FORMAT_MAGIC.to_string(),
        format_version: FORMAT_VERSION,
        k: primary.k(),
        w: primary.w(),
        salt: primary.salt(),
        source_hash,
        num_buckets: remapped.bucket_names.len() as u32,
        total_minimizers: total_entries,
        inverted: Some(InvertedManifest {
            format: ParquetShardFormat::Parquet,
            num_shards: shard_infos.len() as u32,
            total_entries,
            has_overlapping_shards,
            shards: shard_infos,
        }),
    };

    manifest.save(output_path)?;

    // For excluded_minimizers stat, we count excluded entries.
    let excluded_minimizers_count = if options.subtract_from_primary {
        total_excluded as usize
    } else {
        0
    };

    let stats = MergeStats {
        total_buckets: remapped.bucket_names.len() as u32,
        total_minimizer_entries: total_entries,
        primary_buckets: remapped.primary_id_map.len() as u32,
        primary_entries,
        secondary_buckets: remapped.secondary_id_map.len() as u32,
        secondary_entries,
        secondary_entries_original,
        excluded_minimizers: excluded_minimizers_count,
    };

    log::info!("Merge complete (streaming mode):");
    log::info!("  - Output: {}", output_path.display());
    log::info!("  - Total buckets: {}", stats.total_buckets);
    log::info!(
        "  - Total minimizer entries: {}",
        stats.total_minimizer_entries
    );
    log::info!("  - Primary entries: {}", stats.primary_entries);
    if options.subtract_from_primary && stats.excluded_minimizers > 0 {
        log::info!(
            "  - Secondary entries: {} (original: {}, removed: {})",
            stats.secondary_entries,
            stats.secondary_entries_original,
            stats.secondary_entries_original - stats.secondary_entries
        );
    } else {
        log::info!("  - Secondary entries: {}", stats.secondary_entries);
    }

    Ok(stats)
}

/// Memory-bounded merge with streaming per-shard subtraction.
///
/// This function processes each secondary shard independently, building per-shard exclusion
/// sets that are dropped after each shard is processed.
///
/// Memory usage: O(max_memory) regardless of total index overlap.
///
/// # Arguments
///
/// * `primary_path` - Path to the primary index directory (.ryxdi)
/// * `secondary_path` - Path to the secondary index directory (.ryxdi)
/// * `output_path` - Path for the merged output index directory (.ryxdi)
/// * `options` - Merge options (subtraction, verbose output)
/// * `max_memory` - Maximum memory budget in bytes (None = auto-detect)
/// * `write_options` - Optional Parquet write options (compression, bloom filters, etc.)
///
/// # Errors
///
/// Returns an error if:
/// - Either index cannot be opened
/// - Indices are incompatible (different k, w, or salt)
/// - Bucket names overlap between indices
/// - Subtraction results in an empty bucket (all minimizers removed)
/// - I/O errors during reading or writing
pub fn merge_indices_streaming(
    primary_path: &Path,
    secondary_path: &Path,
    output_path: &Path,
    options: &MergeOptions,
    max_memory: Option<usize>,
    write_options: Option<&ParquetWriteOptions>,
) -> Result<MergeStats> {
    use crate::memory::detect_available_memory;

    // Open both indices
    log::info!("Loading primary index: {}", primary_path.display());
    let primary = ShardedInvertedIndex::open(primary_path)?;
    log::debug!(
        "  - k={}, w={}, salt={:#x}",
        primary.k(),
        primary.w(),
        primary.salt()
    );
    log::debug!(
        "  - {} buckets, {} shards",
        primary.manifest().bucket_names.len(),
        primary.num_shards()
    );

    log::info!("Loading secondary index: {}", secondary_path.display());
    let secondary = ShardedInvertedIndex::open(secondary_path)?;
    log::debug!(
        "  - k={}, w={}, salt={:#x}",
        secondary.k(),
        secondary.w(),
        secondary.salt()
    );
    log::debug!(
        "  - {} buckets, {} shards",
        secondary.manifest().bucket_names.len(),
        secondary.num_shards()
    );

    // Reject empty indices
    if primary.manifest().bucket_names.is_empty() {
        return Err(RypeError::validation(format!(
            "Primary index '{}' is empty (0 buckets)",
            primary_path.display()
        )));
    }
    if secondary.manifest().bucket_names.is_empty() {
        return Err(RypeError::validation(format!(
            "Secondary index '{}' is empty (0 buckets)",
            secondary_path.display()
        )));
    }

    // Validate compatibility
    validate_merge_compatibility(&primary, &secondary)?;
    log::info!("Validation passed: indices are compatible");

    // Compute bucket remapping
    let remapped = compute_bucket_remapping(primary.manifest(), secondary.manifest());
    log::debug!(
        "Bucket name check: {} unique names (0 conflicts)",
        remapped.bucket_names.len()
    );

    // Create output directory structure
    create_index_directory(output_path)?;

    // Determine memory budget and shard size
    // Following the pattern from build_parquet_index_from_config()
    let available = max_memory.unwrap_or_else(|| detect_available_memory().bytes);
    // Use 40% of available memory for shard accumulator
    let max_shard_bytes = ((available as f64) * 0.4) as usize;
    let max_shard_bytes = max_shard_bytes.max(MIN_SHARD_BYTES);

    log::debug!(
        "Memory budget: {} ({:.1} MB for output shards)",
        format_bytes(available),
        max_shard_bytes as f64 / (1024.0 * 1024.0)
    );

    // Create accumulator for output
    let mut accumulator =
        ShardAccumulator::with_output_dir(output_path, max_shard_bytes, write_options);

    let mut primary_entries: u64 = 0;
    let mut secondary_entries: u64 = 0;
    let mut secondary_entries_original: u64 = 0;
    let mut total_excluded: u64 = 0;

    // Track per-bucket counts for secondary (to detect empty buckets after subtraction)
    let mut secondary_bucket_counts: HashMap<u32, u64> = HashMap::new();
    let mut secondary_bucket_counts_original: HashMap<u32, u64> = HashMap::new();
    for &old_id in secondary.manifest().bucket_names.keys() {
        secondary_bucket_counts.insert(old_id, 0);
        secondary_bucket_counts_original.insert(old_id, 0);
    }

    // Track per-bucket minimizer counts for source_hash (using NEW bucket IDs)
    let mut bucket_minimizer_counts: HashMap<u32, usize> = HashMap::new();
    for &new_id in remapped.bucket_names.keys() {
        bucket_minimizer_counts.insert(new_id, 0);
    }

    // Process primary index shards (unchanged from original)
    log::info!("Writing merged index (streaming mode)...");
    let primary_shards = primary.manifest().shards.len();
    for (i, shard_info) in primary.manifest().shards.iter().enumerate() {
        log::debug!(
            "  - Processing primary shard {}/{}...",
            i + 1,
            primary_shards
        );

        // Read shard pairs and remap bucket IDs
        let shard_path = primary.shard_path(shard_info.shard_id);
        let pairs = read_shard_pairs(&shard_path)?;

        for (minimizer, old_bucket_id) in pairs {
            let new_bucket_id = *remapped.primary_id_map.get(&old_bucket_id).ok_or_else(|| {
                RypeError::validation(format!(
                    "Primary bucket ID {} not found in remapping",
                    old_bucket_id
                ))
            })?;
            accumulator.add_entries(&[(minimizer, new_bucket_id)]);
            primary_entries += 1;
            *bucket_minimizer_counts.entry(new_bucket_id).or_insert(0) += 1;
        }

        // Check if we should flush
        if accumulator.should_flush() {
            accumulator.flush_shard()?;
        }
    }

    // Process secondary index shards
    let secondary_shards_count = secondary.manifest().shards.len();
    if options.subtract_from_primary {
        // OPTIMIZED: Pre-load ALL primary minimizers into a HashSet once
        // This avoids the O(secondary_shards × primary_shards) complexity of the old approach
        log::info!("Loading primary minimizers for subtraction...");
        let primary_minimizers = load_all_minimizers(&primary)?;
        let primary_minimizer_memory =
            (primary_minimizers.capacity() * 24) as f64 / (1024.0 * 1024.0);
        log::info!(
            "Loaded {} unique primary minimizers ({:.1} MB)",
            primary_minimizers.len(),
            primary_minimizer_memory
        );

        // Finish primary accumulator to get primary shard infos
        let primary_shard_infos = accumulator.finish()?;

        // Calculate next shard ID from primary shards (to avoid overwriting them)
        let next_shard_id = primary_shard_infos
            .iter()
            .map(|s| s.shard_id)
            .max()
            .map(|id| id + 1)
            .unwrap_or(0);

        // Create accumulator for secondary output
        let mut secondary_accumulator = ShardAccumulator::with_start_shard_id(
            output_path,
            max_shard_bytes,
            next_shard_id,
            write_options,
        );

        // Process secondary shards with row-group level parallelism
        log::debug!(
            "Processing {} secondary shards with row-group parallelism...",
            secondary_shards_count
        );

        for (i, shard_info) in secondary.manifest().shards.iter().enumerate() {
            log::debug!(
                "  - Processing secondary shard {}/{}...",
                i + 1,
                secondary_shards_count
            );

            let shard_path = secondary.shard_path(shard_info.shard_id);

            // Process this shard with row-group level parallelism
            let result = process_shard_parallel_row_groups(
                &shard_path,
                &primary_minimizers,
                &remapped.secondary_id_map,
            )?;

            // Update statistics
            secondary_entries += result.pairs.len() as u64;
            total_excluded += result.entries_excluded;

            for (k, v) in result.bucket_counts {
                *secondary_bucket_counts.entry(k).or_insert(0) += v;
            }
            for (k, v) in result.bucket_counts_original {
                *secondary_bucket_counts_original.entry(k).or_insert(0) += v;
            }
            for (k, v) in result.bucket_minimizer_counts {
                *bucket_minimizer_counts.entry(k).or_insert(0) += v;
            }

            // Add filtered pairs to accumulator
            secondary_accumulator.add_entries(&result.pairs);

            // Check if we should flush
            if secondary_accumulator.should_flush() {
                secondary_accumulator.flush_shard()?;
            }
        }

        // Finish secondary accumulator
        let secondary_shard_infos = secondary_accumulator.finish()?;
        secondary_entries_original = secondary_entries + total_excluded;

        // Combine shard infos (primary + secondary)
        let mut all_shard_infos = primary_shard_infos;
        all_shard_infos.extend(secondary_shard_infos);

        // Check for empty buckets after subtraction
        for (&bucket_id, &count) in &secondary_bucket_counts {
            if count == 0 {
                let bucket_name = secondary
                    .manifest()
                    .bucket_names
                    .get(&bucket_id)
                    .map(|s| s.as_str())
                    .unwrap_or("<unknown>");
                let original_count = secondary_bucket_counts_original
                    .get(&bucket_id)
                    .copied()
                    .unwrap_or(0);
                return Err(RypeError::validation(format!(
                    "Subtraction resulted in empty bucket '{}' (id={}): all {} minimizers were removed",
                    bucket_name,
                    bucket_id,
                    original_count
                )));
            }
        }

        // Write output manifest and finish
        return finish_merge(
            output_path,
            &primary,
            &remapped,
            all_shard_infos,
            &bucket_minimizer_counts,
            primary_entries,
            secondary_entries,
            secondary_entries_original,
            total_excluded,
            options,
        );
    } else {
        // No subtraction - just copy secondary entries sequentially
        for (i, shard_info) in secondary.manifest().shards.iter().enumerate() {
            log::debug!(
                "  - Processing secondary shard {}/{}...",
                i + 1,
                secondary_shards_count
            );

            let shard_path = secondary.shard_path(shard_info.shard_id);
            let pairs = read_shard_pairs(&shard_path)?;

            for (minimizer, old_bucket_id) in pairs {
                secondary_entries_original += 1;
                *secondary_bucket_counts_original
                    .entry(old_bucket_id)
                    .or_insert(0) += 1;

                let new_bucket_id =
                    *remapped
                        .secondary_id_map
                        .get(&old_bucket_id)
                        .ok_or_else(|| {
                            RypeError::validation(format!(
                                "Secondary bucket ID {} not found in remapping",
                                old_bucket_id
                            ))
                        })?;
                accumulator.add_entries(&[(minimizer, new_bucket_id)]);
                secondary_entries += 1;
                *bucket_minimizer_counts.entry(new_bucket_id).or_insert(0) += 1;
                *secondary_bucket_counts.entry(old_bucket_id).or_insert(0) += 1;
            }

            if accumulator.should_flush() {
                accumulator.flush_shard()?;
            }
        }
    }

    // Finish accumulator and write output (non-subtraction case)
    let shard_infos = accumulator.finish()?;

    finish_merge(
        output_path,
        &primary,
        &remapped,
        shard_infos,
        &bucket_minimizer_counts,
        primary_entries,
        secondary_entries,
        secondary_entries_original,
        total_excluded,
        options,
    )
}

/// Format bytes as human-readable string.
fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

/// Read (minimizer, bucket_id) pairs from a Parquet shard file.
fn read_shard_pairs(path: &Path) -> Result<Vec<(u64, u32)>> {
    use arrow::array::{UInt32Array, UInt64Array};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let file = std::fs::File::open(path)
        .map_err(|e| RypeError::io(path.to_path_buf(), "open shard", e))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut pairs = Vec::new();
    for batch in reader {
        let batch = batch?;
        let minimizers = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| RypeError::format(path, "Expected UInt64Array for minimizer column"))?;
        let bucket_ids = batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| RypeError::format(path, "Expected UInt32Array for bucket_id column"))?;

        for i in 0..batch.num_rows() {
            pairs.push((minimizers.value(i), bucket_ids.value(i)));
        }
    }
    Ok(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indices::parquet::{create_parquet_inverted_index, BucketData};
    use tempfile::TempDir;

    /// Helper to create a test index with given parameters.
    fn create_test_index(
        dir: &std::path::Path,
        k: usize,
        w: usize,
        salt: u64,
        bucket_name: &str,
        minimizers: Vec<u64>,
    ) -> ShardedInvertedIndex {
        let bucket = BucketData {
            bucket_id: 1,
            bucket_name: bucket_name.to_string(),
            sources: vec!["test_source".to_string()],
            minimizers,
        };

        create_parquet_inverted_index(dir, vec![bucket], k, w, salt, None, None).unwrap();

        ShardedInvertedIndex::open(dir).unwrap()
    }

    #[test]
    fn test_validate_compatible_indices_success() {
        let tmp = TempDir::new().unwrap();
        let primary_path = tmp.path().join("primary.ryxdi");
        let secondary_path = tmp.path().join("secondary.ryxdi");

        let primary = create_test_index(&primary_path, 64, 50, 0x5555, "bucket_a", vec![1, 2, 3]);
        let secondary =
            create_test_index(&secondary_path, 64, 50, 0x5555, "bucket_b", vec![4, 5, 6]);

        // Should succeed - same k, w, salt and different bucket names
        let result = validate_merge_compatibility(&primary, &secondary);
        assert!(result.is_ok(), "Expected success, got: {:?}", result);
    }

    #[test]
    fn test_validate_incompatible_k() {
        let tmp = TempDir::new().unwrap();
        let primary_path = tmp.path().join("primary.ryxdi");
        let secondary_path = tmp.path().join("secondary.ryxdi");

        let primary = create_test_index(&primary_path, 64, 50, 0x5555, "bucket_a", vec![1, 2, 3]);
        let secondary =
            create_test_index(&secondary_path, 32, 50, 0x5555, "bucket_b", vec![4, 5, 6]);

        let result = validate_merge_compatibility(&primary, &secondary);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("k mismatch"), "Error: {}", err_msg);
        assert!(err_msg.contains("64"), "Error: {}", err_msg);
        assert!(err_msg.contains("32"), "Error: {}", err_msg);
    }

    #[test]
    fn test_validate_incompatible_w() {
        let tmp = TempDir::new().unwrap();
        let primary_path = tmp.path().join("primary.ryxdi");
        let secondary_path = tmp.path().join("secondary.ryxdi");

        let primary = create_test_index(&primary_path, 64, 50, 0x5555, "bucket_a", vec![1, 2, 3]);
        let secondary =
            create_test_index(&secondary_path, 64, 100, 0x5555, "bucket_b", vec![4, 5, 6]);

        let result = validate_merge_compatibility(&primary, &secondary);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("w mismatch"), "Error: {}", err_msg);
        assert!(err_msg.contains("50"), "Error: {}", err_msg);
        assert!(err_msg.contains("100"), "Error: {}", err_msg);
    }

    #[test]
    fn test_validate_incompatible_salt() {
        let tmp = TempDir::new().unwrap();
        let primary_path = tmp.path().join("primary.ryxdi");
        let secondary_path = tmp.path().join("secondary.ryxdi");

        let primary = create_test_index(
            &primary_path,
            64,
            50,
            0x5555555555555555,
            "bucket_a",
            vec![1, 2, 3],
        );
        let secondary = create_test_index(
            &secondary_path,
            64,
            50,
            0xAAAAAAAAAAAAAAAA,
            "bucket_b",
            vec![4, 5, 6],
        );

        let result = validate_merge_compatibility(&primary, &secondary);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("salt mismatch"), "Error: {}", err_msg);
    }

    #[test]
    fn test_validate_duplicate_bucket_names() {
        let tmp = TempDir::new().unwrap();
        let primary_path = tmp.path().join("primary.ryxdi");
        let secondary_path = tmp.path().join("secondary.ryxdi");

        // Both indices have a bucket named "shared_name"
        let primary =
            create_test_index(&primary_path, 64, 50, 0x5555, "shared_name", vec![1, 2, 3]);
        let secondary = create_test_index(
            &secondary_path,
            64,
            50,
            0x5555,
            "shared_name",
            vec![4, 5, 6],
        );

        let result = validate_merge_compatibility(&primary, &secondary);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("duplicate bucket name"),
            "Error: {}",
            err_msg
        );
        assert!(err_msg.contains("shared_name"), "Error: {}", err_msg);
    }

    #[test]
    fn test_validate_unique_bucket_names() {
        let tmp = TempDir::new().unwrap();
        let primary_path = tmp.path().join("primary.ryxdi");
        let secondary_path = tmp.path().join("secondary.ryxdi");

        let primary = create_test_index(
            &primary_path,
            64,
            50,
            0x5555,
            "unique_name_1",
            vec![1, 2, 3],
        );
        let secondary = create_test_index(
            &secondary_path,
            64,
            50,
            0x5555,
            "unique_name_2",
            vec![4, 5, 6],
        );

        // Should succeed - different bucket names
        let result = validate_merge_compatibility(&primary, &secondary);
        assert!(result.is_ok(), "Expected success, got: {:?}", result);
    }

    // =========================================================================
    // Phase 2: Bucket remapping tests
    // =========================================================================

    /// Helper to create a test index with multiple buckets.
    fn create_multi_bucket_index(
        dir: &std::path::Path,
        k: usize,
        w: usize,
        salt: u64,
        buckets: Vec<(u32, &str, Vec<String>, Vec<u64>)>, // (id, name, sources, minimizers)
    ) -> ShardedInvertedIndex {
        let bucket_data: Vec<BucketData> = buckets
            .into_iter()
            .map(|(id, name, sources, mins)| BucketData {
                bucket_id: id,
                bucket_name: name.to_string(),
                sources,
                minimizers: mins,
            })
            .collect();

        create_parquet_inverted_index(dir, bucket_data, k, w, salt, None, None).unwrap();

        ShardedInvertedIndex::open(dir).unwrap()
    }

    #[test]
    fn test_remap_bucket_ids_sequential() {
        let tmp = TempDir::new().unwrap();
        let primary_path = tmp.path().join("primary.ryxdi");
        let secondary_path = tmp.path().join("secondary.ryxdi");

        // Primary has bucket IDs 5 and 10 (non-sequential)
        let primary = create_multi_bucket_index(
            &primary_path,
            64,
            50,
            0x5555,
            vec![
                (5, "bucket_a", vec!["src_a".to_string()], vec![1, 2, 3]),
                (10, "bucket_b", vec!["src_b".to_string()], vec![4, 5, 6]),
            ],
        );

        // Secondary has bucket IDs 3 and 7
        let secondary = create_multi_bucket_index(
            &secondary_path,
            64,
            50,
            0x5555,
            vec![
                (3, "bucket_c", vec!["src_c".to_string()], vec![7, 8, 9]),
                (7, "bucket_d", vec!["src_d".to_string()], vec![10, 11, 12]),
            ],
        );

        let remapped = compute_bucket_remapping(primary.manifest(), secondary.manifest());

        // New IDs should be 1, 2, 3, 4 (sequential starting from 1)
        assert_eq!(remapped.bucket_names.len(), 4);
        assert!(remapped.bucket_names.contains_key(&1));
        assert!(remapped.bucket_names.contains_key(&2));
        assert!(remapped.bucket_names.contains_key(&3));
        assert!(remapped.bucket_names.contains_key(&4));

        // Verify mappings point to sequential IDs
        let mut all_new_ids: Vec<u32> = remapped
            .primary_id_map
            .values()
            .chain(remapped.secondary_id_map.values())
            .copied()
            .collect();
        all_new_ids.sort();
        assert_eq!(all_new_ids, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_remap_preserves_names_and_sources() {
        let tmp = TempDir::new().unwrap();
        let primary_path = tmp.path().join("primary.ryxdi");
        let secondary_path = tmp.path().join("secondary.ryxdi");

        let primary = create_multi_bucket_index(
            &primary_path,
            64,
            50,
            0x5555,
            vec![(
                1,
                "my_bucket",
                vec!["file1.fa".to_string(), "file2.fa".to_string()],
                vec![1, 2, 3],
            )],
        );

        let secondary = create_multi_bucket_index(
            &secondary_path,
            64,
            50,
            0x5555,
            vec![(
                1,
                "other_bucket",
                vec!["file3.fa".to_string()],
                vec![4, 5, 6],
            )],
        );

        let remapped = compute_bucket_remapping(primary.manifest(), secondary.manifest());

        // Primary bucket (old ID 1) maps to new ID 1
        let primary_new_id = remapped.primary_id_map.get(&1).unwrap();
        assert_eq!(*primary_new_id, 1);
        assert_eq!(
            remapped.bucket_names.get(primary_new_id),
            Some(&"my_bucket".to_string())
        );
        assert_eq!(
            remapped.bucket_sources.get(primary_new_id),
            Some(&vec!["file1.fa".to_string(), "file2.fa".to_string()])
        );

        // Secondary bucket (old ID 1) maps to new ID 2
        let secondary_new_id = remapped.secondary_id_map.get(&1).unwrap();
        assert_eq!(*secondary_new_id, 2);
        assert_eq!(
            remapped.bucket_names.get(secondary_new_id),
            Some(&"other_bucket".to_string())
        );
        assert_eq!(
            remapped.bucket_sources.get(secondary_new_id),
            Some(&vec!["file3.fa".to_string()])
        );
    }

    #[test]
    fn test_remap_primary_then_secondary() {
        let tmp = TempDir::new().unwrap();
        let primary_path = tmp.path().join("primary.ryxdi");
        let secondary_path = tmp.path().join("secondary.ryxdi");

        // Primary has 2 buckets
        let primary = create_multi_bucket_index(
            &primary_path,
            64,
            50,
            0x5555,
            vec![
                (1, "primary_1", vec!["p1.fa".to_string()], vec![1, 2]),
                (2, "primary_2", vec!["p2.fa".to_string()], vec![3, 4]),
            ],
        );

        // Secondary has 2 buckets
        let secondary = create_multi_bucket_index(
            &secondary_path,
            64,
            50,
            0x5555,
            vec![
                (1, "secondary_1", vec!["s1.fa".to_string()], vec![5, 6]),
                (2, "secondary_2", vec!["s2.fa".to_string()], vec![7, 8]),
            ],
        );

        let remapped = compute_bucket_remapping(primary.manifest(), secondary.manifest());

        // Primary buckets should get IDs 1 and 2
        assert_eq!(*remapped.primary_id_map.get(&1).unwrap(), 1);
        assert_eq!(*remapped.primary_id_map.get(&2).unwrap(), 2);

        // Secondary buckets should get IDs 3 and 4
        assert_eq!(*remapped.secondary_id_map.get(&1).unwrap(), 3);
        assert_eq!(*remapped.secondary_id_map.get(&2).unwrap(), 4);

        // Verify names are in correct positions
        assert_eq!(
            remapped.bucket_names.get(&1),
            Some(&"primary_1".to_string())
        );
        assert_eq!(
            remapped.bucket_names.get(&2),
            Some(&"primary_2".to_string())
        );
        assert_eq!(
            remapped.bucket_names.get(&3),
            Some(&"secondary_1".to_string())
        );
        assert_eq!(
            remapped.bucket_names.get(&4),
            Some(&"secondary_2".to_string())
        );
    }

    // =========================================================================
    // Phase 3: Simple merge tests
    // =========================================================================

    #[test]
    fn test_merge_indices_basic() {
        let tmp = TempDir::new().unwrap();
        let primary_path = tmp.path().join("primary.ryxdi");
        let secondary_path = tmp.path().join("secondary.ryxdi");
        let output_path = tmp.path().join("merged.ryxdi");

        // Create two indices with distinct minimizers
        let _primary = create_multi_bucket_index(
            &primary_path,
            64,
            50,
            0x5555,
            vec![
                (
                    1,
                    "bucket_a",
                    vec!["src_a.fa".to_string()],
                    vec![100, 200, 300],
                ),
                (
                    2,
                    "bucket_b",
                    vec!["src_b.fa".to_string()],
                    vec![400, 500, 600],
                ),
            ],
        );

        let _secondary = create_multi_bucket_index(
            &secondary_path,
            64,
            50,
            0x5555,
            vec![
                (
                    1,
                    "bucket_c",
                    vec!["src_c.fa".to_string()],
                    vec![700, 800, 900],
                ),
                (
                    2,
                    "bucket_d",
                    vec!["src_d.fa".to_string()],
                    vec![1000, 1100, 1200],
                ),
            ],
        );

        // Merge the indices
        let options = MergeOptions {
            subtract_from_primary: false,
            verbose: false,
        };
        let stats = merge_indices_streaming(
            &primary_path,
            &secondary_path,
            &output_path,
            &options,
            None,
            None,
        )
        .expect("merge should succeed");

        // Verify stats
        assert_eq!(stats.total_buckets, 4);
        assert_eq!(stats.primary_buckets, 2);
        assert_eq!(stats.secondary_buckets, 2);
        assert_eq!(stats.primary_entries, 6); // 3 + 3
        assert_eq!(stats.secondary_entries, 6); // 3 + 3
        assert_eq!(stats.total_minimizer_entries, 12);
        assert_eq!(stats.excluded_minimizers, 0);

        // Verify output can be opened
        let merged = ShardedInvertedIndex::open(&output_path).expect("should open merged index");
        assert_eq!(merged.k(), 64);
        assert_eq!(merged.w(), 50);
        assert_eq!(merged.salt(), 0x5555);
    }

    #[test]
    fn test_merge_preserves_all_minimizers() {
        let tmp = TempDir::new().unwrap();
        let primary_path = tmp.path().join("primary.ryxdi");
        let secondary_path = tmp.path().join("secondary.ryxdi");
        let output_path = tmp.path().join("merged.ryxdi");

        // Create indices with specific minimizers
        let primary_mins = vec![10, 20, 30, 40, 50];
        let secondary_mins = vec![60, 70, 80, 90, 100];

        let _primary = create_test_index(
            &primary_path,
            64,
            50,
            0x5555,
            "primary",
            primary_mins.clone(),
        );
        let _secondary = create_test_index(
            &secondary_path,
            64,
            50,
            0x5555,
            "secondary",
            secondary_mins.clone(),
        );

        // Merge
        let options = MergeOptions::default();
        let stats = merge_indices_streaming(
            &primary_path,
            &secondary_path,
            &output_path,
            &options,
            None,
            None,
        )
        .expect("merge should succeed");

        // Verify all entries preserved
        assert_eq!(stats.total_minimizer_entries, 10);
        assert_eq!(stats.primary_entries, 5);
        assert_eq!(stats.secondary_entries, 5);

        // Verify we can query the merged index
        let merged = ShardedInvertedIndex::open(&output_path).unwrap();
        assert_eq!(merged.total_minimizers(), 10);
    }

    #[test]
    fn test_merge_output_structure() {
        let tmp = TempDir::new().unwrap();
        let primary_path = tmp.path().join("primary.ryxdi");
        let secondary_path = tmp.path().join("secondary.ryxdi");
        let output_path = tmp.path().join("merged.ryxdi");

        let _primary = create_test_index(&primary_path, 64, 50, 0x5555, "bucket_a", vec![1, 2, 3]);
        let _secondary =
            create_test_index(&secondary_path, 64, 50, 0x5555, "bucket_b", vec![4, 5, 6]);

        let options = MergeOptions::default();
        merge_indices_streaming(
            &primary_path,
            &secondary_path,
            &output_path,
            &options,
            None,
            None,
        )
        .expect("merge should succeed");

        // Verify directory structure
        assert!(output_path.exists(), "Output directory should exist");
        assert!(
            output_path.join("manifest.toml").exists(),
            "manifest.toml should exist"
        );
        assert!(
            output_path.join("buckets.parquet").exists(),
            "buckets.parquet should exist"
        );
        assert!(
            output_path.join("inverted").exists(),
            "inverted directory should exist"
        );

        // At least one shard should exist
        let shard_files: Vec<_> = std::fs::read_dir(output_path.join("inverted"))
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "parquet"))
            .collect();
        assert!(
            !shard_files.is_empty(),
            "Should have at least one shard file"
        );
    }

    #[test]
    fn test_merge_bucket_names_in_output() {
        let tmp = TempDir::new().unwrap();
        let primary_path = tmp.path().join("primary.ryxdi");
        let secondary_path = tmp.path().join("secondary.ryxdi");
        let output_path = tmp.path().join("merged.ryxdi");

        let _primary = create_multi_bucket_index(
            &primary_path,
            64,
            50,
            0x5555,
            vec![
                (1, "alpha", vec!["a.fa".to_string()], vec![1, 2]),
                (2, "beta", vec!["b.fa".to_string()], vec![3, 4]),
            ],
        );

        let _secondary = create_multi_bucket_index(
            &secondary_path,
            64,
            50,
            0x5555,
            vec![(1, "gamma", vec!["c.fa".to_string()], vec![5, 6])],
        );

        let options = MergeOptions::default();
        merge_indices_streaming(
            &primary_path,
            &secondary_path,
            &output_path,
            &options,
            None,
            None,
        )
        .expect("merge should succeed");

        // Open merged index and verify bucket names
        let merged = ShardedInvertedIndex::open(&output_path).unwrap();
        let names = &merged.manifest().bucket_names;

        assert_eq!(names.len(), 3);
        // Primary buckets get IDs 1 and 2
        assert_eq!(names.get(&1), Some(&"alpha".to_string()));
        assert_eq!(names.get(&2), Some(&"beta".to_string()));
        // Secondary bucket gets ID 3
        assert_eq!(names.get(&3), Some(&"gamma".to_string()));
    }

    #[test]
    fn test_merge_with_shared_minimizers() {
        // Test that minimizers appearing in both indices are handled correctly
        let tmp = TempDir::new().unwrap();
        let primary_path = tmp.path().join("primary.ryxdi");
        let secondary_path = tmp.path().join("secondary.ryxdi");
        let output_path = tmp.path().join("merged.ryxdi");

        // Both indices have minimizer 100
        let _primary = create_test_index(
            &primary_path,
            64,
            50,
            0x5555,
            "bucket_a",
            vec![100, 200, 300],
        );
        let _secondary = create_test_index(
            &secondary_path,
            64,
            50,
            0x5555,
            "bucket_b",
            vec![100, 400, 500],
        );

        let options = MergeOptions::default();
        let stats = merge_indices_streaming(
            &primary_path,
            &secondary_path,
            &output_path,
            &options,
            None,
            None,
        )
        .expect("merge should succeed");

        // Both indices contribute all their entries
        assert_eq!(stats.primary_entries, 3);
        assert_eq!(stats.secondary_entries, 3);
        assert_eq!(stats.total_minimizer_entries, 6);

        // Verify merged index is valid
        let merged = ShardedInvertedIndex::open(&output_path).unwrap();
        assert_eq!(merged.manifest().bucket_names.len(), 2);
    }

    #[test]
    fn test_merge_verbose_output() {
        let tmp = TempDir::new().unwrap();
        let primary_path = tmp.path().join("primary.ryxdi");
        let secondary_path = tmp.path().join("secondary.ryxdi");
        let output_path = tmp.path().join("merged.ryxdi");

        let _primary = create_test_index(&primary_path, 64, 50, 0x5555, "bucket_a", vec![1, 2, 3]);
        let _secondary =
            create_test_index(&secondary_path, 64, 50, 0x5555, "bucket_b", vec![4, 5, 6]);

        // Run with verbose=true (output goes to stderr, we just verify it doesn't crash)
        let options = MergeOptions {
            subtract_from_primary: false,
            verbose: true,
        };
        let result = merge_indices_streaming(
            &primary_path,
            &secondary_path,
            &output_path,
            &options,
            None,
            None,
        );
        assert!(result.is_ok(), "merge with verbose should succeed");
    }

    // =========================================================================
    // Phase 4: Subtraction tests
    // =========================================================================

    #[test]
    fn test_subtract_removes_shared_minimizers() {
        let tmp = TempDir::new().unwrap();
        let primary_path = tmp.path().join("primary.ryxdi");
        let secondary_path = tmp.path().join("secondary.ryxdi");
        let output_path = tmp.path().join("merged.ryxdi");

        // Primary has minimizers 100, 200, 300
        // Secondary has minimizers 100, 400, 500 (100 is shared)
        let _primary = create_test_index(
            &primary_path,
            64,
            50,
            0x5555,
            "bucket_a",
            vec![100, 200, 300],
        );
        let _secondary = create_test_index(
            &secondary_path,
            64,
            50,
            0x5555,
            "bucket_b",
            vec![100, 400, 500],
        );

        let options = MergeOptions {
            subtract_from_primary: true,
            verbose: false,
        };
        let stats = merge_indices_streaming(
            &primary_path,
            &secondary_path,
            &output_path,
            &options,
            None,
            None,
        )
        .expect("merge with subtraction should succeed");

        // Primary: 3 entries unchanged
        assert_eq!(stats.primary_entries, 3);

        // Secondary: originally 3 entries, 1 removed (minimizer 100)
        assert_eq!(stats.secondary_entries_original, 3);
        assert_eq!(stats.secondary_entries, 2);
        assert_eq!(stats.excluded_minimizers, 1);

        // Total: 3 (primary) + 2 (secondary after subtraction) = 5
        assert_eq!(stats.total_minimizer_entries, 5);
    }

    #[test]
    fn test_subtract_keeps_unique_secondary_minimizers() {
        let tmp = TempDir::new().unwrap();
        let primary_path = tmp.path().join("primary.ryxdi");
        let secondary_path = tmp.path().join("secondary.ryxdi");
        let output_path = tmp.path().join("merged.ryxdi");

        // No overlap between indices
        let _primary = create_test_index(
            &primary_path,
            64,
            50,
            0x5555,
            "bucket_a",
            vec![100, 200, 300],
        );
        let _secondary = create_test_index(
            &secondary_path,
            64,
            50,
            0x5555,
            "bucket_b",
            vec![400, 500, 600],
        );

        let options = MergeOptions {
            subtract_from_primary: true,
            verbose: false,
        };
        let stats = merge_indices_streaming(
            &primary_path,
            &secondary_path,
            &output_path,
            &options,
            None,
            None,
        )
        .expect("merge with subtraction should succeed");

        // No overlap, so secondary is unchanged
        assert_eq!(stats.secondary_entries, 3);
        assert_eq!(stats.secondary_entries_original, 3);
        assert_eq!(stats.excluded_minimizers, 0);
        assert_eq!(stats.total_minimizer_entries, 6);
    }

    #[test]
    fn test_subtract_primary_unchanged() {
        let tmp = TempDir::new().unwrap();
        let primary_path = tmp.path().join("primary.ryxdi");
        let secondary_path = tmp.path().join("secondary.ryxdi");
        let output_path = tmp.path().join("merged.ryxdi");

        // Primary has minimizers 100, 200
        // Secondary has minimizers 100, 200, 300 (all of primary's minimizers)
        let _primary = create_test_index(&primary_path, 64, 50, 0x5555, "bucket_a", vec![100, 200]);
        let _secondary = create_test_index(
            &secondary_path,
            64,
            50,
            0x5555,
            "bucket_b",
            vec![100, 200, 300],
        );

        let options = MergeOptions {
            subtract_from_primary: true,
            verbose: false,
        };
        let stats = merge_indices_streaming(
            &primary_path,
            &secondary_path,
            &output_path,
            &options,
            None,
            None,
        )
        .expect("merge with subtraction should succeed");

        // Primary should be unchanged
        assert_eq!(stats.primary_entries, 2);

        // Secondary should have 100 and 200 removed, only 300 remains
        assert_eq!(stats.secondary_entries_original, 3);
        assert_eq!(stats.secondary_entries, 1);
        assert_eq!(stats.excluded_minimizers, 2);

        // Verify the merged index
        let merged = ShardedInvertedIndex::open(&output_path).unwrap();
        assert_eq!(merged.manifest().bucket_names.len(), 2);
    }

    #[test]
    fn test_subtract_empty_bucket_error() {
        let tmp = TempDir::new().unwrap();
        let primary_path = tmp.path().join("primary.ryxdi");
        let secondary_path = tmp.path().join("secondary.ryxdi");
        let output_path = tmp.path().join("merged.ryxdi");

        // Primary has all of secondary's minimizers
        let _primary = create_test_index(
            &primary_path,
            64,
            50,
            0x5555,
            "bucket_a",
            vec![100, 200, 300, 400, 500],
        );
        // Secondary's minimizers are a subset of primary's
        let _secondary = create_test_index(
            &secondary_path,
            64,
            50,
            0x5555,
            "bucket_b",
            vec![100, 200, 300],
        );

        let options = MergeOptions {
            subtract_from_primary: true,
            verbose: false,
        };
        let result = merge_indices_streaming(
            &primary_path,
            &secondary_path,
            &output_path,
            &options,
            None,
            None,
        );

        // Should fail because bucket_b would become empty
        assert!(result.is_err(), "Should fail when bucket becomes empty");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("empty bucket"),
            "Error should mention empty bucket: {}",
            err_msg
        );
        assert!(
            err_msg.contains("bucket_b"),
            "Error should mention bucket name: {}",
            err_msg
        );
    }

    #[test]
    fn test_subtract_partial_bucket() {
        let tmp = TempDir::new().unwrap();
        let primary_path = tmp.path().join("primary.ryxdi");
        let secondary_path = tmp.path().join("secondary.ryxdi");
        let output_path = tmp.path().join("merged.ryxdi");

        // Multiple buckets in secondary, one loses some minimizers but not all
        let _primary = create_multi_bucket_index(
            &primary_path,
            64,
            50,
            0x5555,
            vec![(1, "bucket_a", vec!["src_a".to_string()], vec![100, 200])],
        );
        let _secondary = create_multi_bucket_index(
            &secondary_path,
            64,
            50,
            0x5555,
            vec![
                // bucket_b has 100 (shared) and 300, 400 (unique) - will keep 2
                (
                    1,
                    "bucket_b",
                    vec!["src_b".to_string()],
                    vec![100, 300, 400],
                ),
                // bucket_c has 200 (shared) and 500, 600 (unique) - will keep 2
                (
                    2,
                    "bucket_c",
                    vec!["src_c".to_string()],
                    vec![200, 500, 600],
                ),
            ],
        );

        let options = MergeOptions {
            subtract_from_primary: true,
            verbose: false,
        };
        let stats = merge_indices_streaming(
            &primary_path,
            &secondary_path,
            &output_path,
            &options,
            None,
            None,
        )
        .expect("merge with partial subtraction should succeed");

        // Primary: 2 entries
        assert_eq!(stats.primary_entries, 2);

        // Secondary: originally 6 entries (3+3), 2 removed (100 and 200)
        assert_eq!(stats.secondary_entries_original, 6);
        assert_eq!(stats.secondary_entries, 4);
        assert_eq!(stats.excluded_minimizers, 2);

        // Verify the merged index has all 3 buckets
        let merged = ShardedInvertedIndex::open(&output_path).unwrap();
        assert_eq!(merged.manifest().bucket_names.len(), 3);
    }

    #[test]
    fn test_subtract_with_verbose() {
        let tmp = TempDir::new().unwrap();
        let primary_path = tmp.path().join("primary.ryxdi");
        let secondary_path = tmp.path().join("secondary.ryxdi");
        let output_path = tmp.path().join("merged.ryxdi");

        let _primary = create_test_index(
            &primary_path,
            64,
            50,
            0x5555,
            "bucket_a",
            vec![100, 200, 300],
        );
        let _secondary = create_test_index(
            &secondary_path,
            64,
            50,
            0x5555,
            "bucket_b",
            vec![100, 400, 500],
        );

        // Run with verbose to ensure it doesn't crash
        let options = MergeOptions {
            subtract_from_primary: true,
            verbose: true,
        };
        let result = merge_indices_streaming(
            &primary_path,
            &secondary_path,
            &output_path,
            &options,
            None,
            None,
        );
        assert!(
            result.is_ok(),
            "merge with subtraction and verbose should succeed"
        );
    }

    // =========================================================================
    // Edge case: Empty index validation
    // =========================================================================

    /// Helper to create an empty test index (0 buckets).
    fn create_empty_index(dir: &std::path::Path, k: usize, w: usize, salt: u64) {
        create_parquet_inverted_index(dir, vec![], k, w, salt, None, None).unwrap();
    }

    #[test]
    fn test_merge_rejects_empty_primary_index() {
        let tmp = TempDir::new().unwrap();
        let primary_path = tmp.path().join("primary.ryxdi");
        let secondary_path = tmp.path().join("secondary.ryxdi");
        let output_path = tmp.path().join("merged.ryxdi");

        // Create empty primary and non-empty secondary
        create_empty_index(&primary_path, 64, 50, 0x5555);
        let _secondary =
            create_test_index(&secondary_path, 64, 50, 0x5555, "bucket_b", vec![1, 2, 3]);

        let options = MergeOptions::default();
        let result = merge_indices_streaming(
            &primary_path,
            &secondary_path,
            &output_path,
            &options,
            None,
            None,
        );

        // Empty indices cause errors - either from our validation ("empty" / "0 buckets")
        // or from missing shard files (I/O error). Either way, merge should fail.
        assert!(
            result.is_err(),
            "Should reject empty primary index: {:?}",
            result
        );
    }

    #[test]
    fn test_merge_rejects_empty_secondary_index() {
        let tmp = TempDir::new().unwrap();
        let primary_path = tmp.path().join("primary.ryxdi");
        let secondary_path = tmp.path().join("secondary.ryxdi");
        let output_path = tmp.path().join("merged.ryxdi");

        // Create non-empty primary and empty secondary
        let _primary = create_test_index(&primary_path, 64, 50, 0x5555, "bucket_a", vec![1, 2, 3]);
        create_empty_index(&secondary_path, 64, 50, 0x5555);

        let options = MergeOptions::default();
        let result = merge_indices_streaming(
            &primary_path,
            &secondary_path,
            &output_path,
            &options,
            None,
            None,
        );

        // Empty indices cause errors - either from our validation ("empty" / "0 buckets")
        // or from missing shard files (I/O error). Either way, merge should fail.
        assert!(
            result.is_err(),
            "Should reject empty secondary index: {:?}",
            result
        );
    }

    // =========================================================================
    // Classification correctness test
    // =========================================================================

    #[test]
    fn test_merge_classification_correctness() {
        use crate::classify::classify_batch_sharded_merge_join;
        use crate::core::extraction::extract_into;
        use crate::core::workspace::MinimizerWorkspace;
        use crate::types::QueryRecord;

        let tmp = TempDir::new().unwrap();
        let primary_path = tmp.path().join("primary.ryxdi");
        let secondary_path = tmp.path().join("secondary.ryxdi");
        let output_path = tmp.path().join("merged.ryxdi");

        // Use distinct sequences that won't share minimizers
        let seq_a = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let seq_b = b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA";
        let seq_c = b"AAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTT";
        let seq_d = b"CCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAA";

        let k = 32;
        let w = 10;
        let salt = 0x1234u64;
        let mut ws = MinimizerWorkspace::new();

        // Extract minimizers for each sequence
        extract_into(seq_a, k, w, salt, &mut ws);
        let mut mins_a = std::mem::take(&mut ws.buffer);
        mins_a.sort_unstable();
        mins_a.dedup();

        extract_into(seq_b, k, w, salt, &mut ws);
        let mut mins_b = std::mem::take(&mut ws.buffer);
        mins_b.sort_unstable();
        mins_b.dedup();

        extract_into(seq_c, k, w, salt, &mut ws);
        let mut mins_c = std::mem::take(&mut ws.buffer);
        mins_c.sort_unstable();
        mins_c.dedup();

        extract_into(seq_d, k, w, salt, &mut ws);
        let mut mins_d = std::mem::take(&mut ws.buffer);
        mins_d.sort_unstable();
        mins_d.dedup();

        // Create primary index with buckets A and B
        let primary_buckets = vec![
            BucketData {
                bucket_id: 1,
                bucket_name: "bucket_a".to_string(),
                sources: vec!["a.fa".to_string()],
                minimizers: mins_a.clone(),
            },
            BucketData {
                bucket_id: 2,
                bucket_name: "bucket_b".to_string(),
                sources: vec!["b.fa".to_string()],
                minimizers: mins_b.clone(),
            },
        ];
        create_parquet_inverted_index(&primary_path, primary_buckets, k, w, salt, None, None)
            .unwrap();

        // Create secondary index with buckets C and D
        let secondary_buckets = vec![
            BucketData {
                bucket_id: 1,
                bucket_name: "bucket_c".to_string(),
                sources: vec!["c.fa".to_string()],
                minimizers: mins_c.clone(),
            },
            BucketData {
                bucket_id: 2,
                bucket_name: "bucket_d".to_string(),
                sources: vec!["d.fa".to_string()],
                minimizers: mins_d.clone(),
            },
        ];
        create_parquet_inverted_index(&secondary_path, secondary_buckets, k, w, salt, None, None)
            .unwrap();

        // Merge the indices
        let merge_options = MergeOptions::default();
        let stats = merge_indices_streaming(
            &primary_path,
            &secondary_path,
            &output_path,
            &merge_options,
            None,
            None,
        )
        .expect("merge should succeed");

        assert_eq!(stats.total_buckets, 4);

        // Open merged index
        let merged = ShardedInvertedIndex::open(&output_path).unwrap();

        // Verify bucket names in merged index
        let names = &merged.manifest().bucket_names;
        assert_eq!(names.get(&1), Some(&"bucket_a".to_string()));
        assert_eq!(names.get(&2), Some(&"bucket_b".to_string()));
        assert_eq!(names.get(&3), Some(&"bucket_c".to_string()));
        assert_eq!(names.get(&4), Some(&"bucket_d".to_string()));

        // Classify queries using the original sequences
        let records: Vec<QueryRecord> = vec![
            (0, seq_a.as_slice(), None),
            (1, seq_b.as_slice(), None),
            (2, seq_c.as_slice(), None),
            (3, seq_d.as_slice(), None),
        ];

        let threshold = 0.1;
        let results =
            classify_batch_sharded_merge_join(&merged, None, &records, threshold, None, None)
                .expect("classification should succeed");

        // Each query should match its corresponding bucket with high score
        // Query 0 (seq_a) -> bucket 1 (bucket_a)
        let match_a = results.iter().find(|r| r.query_id == 0 && r.bucket_id == 1);
        assert!(
            match_a.is_some(),
            "Query 0 should match bucket 1 (bucket_a)"
        );
        assert!(
            match_a.unwrap().score > 0.9,
            "Self-match should have high score"
        );

        // Query 1 (seq_b) -> bucket 2 (bucket_b)
        let match_b = results.iter().find(|r| r.query_id == 1 && r.bucket_id == 2);
        assert!(
            match_b.is_some(),
            "Query 1 should match bucket 2 (bucket_b)"
        );
        assert!(
            match_b.unwrap().score > 0.9,
            "Self-match should have high score"
        );

        // Query 2 (seq_c) -> bucket 3 (bucket_c)
        let match_c = results.iter().find(|r| r.query_id == 2 && r.bucket_id == 3);
        assert!(
            match_c.is_some(),
            "Query 2 should match bucket 3 (bucket_c)"
        );
        assert!(
            match_c.unwrap().score > 0.9,
            "Self-match should have high score"
        );

        // Query 3 (seq_d) -> bucket 4 (bucket_d)
        let match_d = results.iter().find(|r| r.query_id == 3 && r.bucket_id == 4);
        assert!(
            match_d.is_some(),
            "Query 3 should match bucket 4 (bucket_d)"
        );
        assert!(
            match_d.unwrap().score > 0.9,
            "Self-match should have high score"
        );
    }

    #[test]
    fn test_streaming_subtract_per_shard_memory_bounded() {
        // Test that streaming subtraction works correctly with memory budget.
        // Note: With MIN_SHARD_BYTES=1MB and small test data, we can't force multiple
        // shards in unit tests. The key test is that streaming mode produces correct
        // results with proper exclusion handling.
        let tmp = TempDir::new().unwrap();
        let primary_path = tmp.path().join("primary.ryxdi");
        let secondary_path = tmp.path().join("secondary.ryxdi");
        let output_path = tmp.path().join("merged.ryxdi");

        // Create primary with a moderate number of minimizers
        let primary_mins: Vec<u64> = (0..1000).map(|i| i * 2).collect(); // evens: 0, 2, 4, ..., 1998
        let _primary = create_test_index(&primary_path, 64, 50, 0x5555, "primary", primary_mins);

        // Create secondary with minimizers that have ~50% overlap with primary
        // Multiples of 4 are in both (0, 4, 8, ...) while multiples of 4+2 are only in secondary
        let secondary_mins: Vec<u64> = (0..3000).map(|i| i * 4).collect(); // 0, 4, 8, ..., 11996
        let _secondary =
            create_test_index(&secondary_path, 64, 50, 0x5555, "secondary", secondary_mins);

        // Call streaming merge with memory budget
        let options = MergeOptions {
            subtract_from_primary: true,
            verbose: false,
        };
        let stats = merge_indices_streaming(
            &primary_path,
            &secondary_path,
            &output_path,
            &options,
            Some(64 * 1024 * 1024), // 64MB max memory
            None,
        )
        .expect("streaming merge should succeed");

        // Assert: index was created successfully
        let merged = ShardedInvertedIndex::open(&output_path).unwrap();
        assert!(merged.num_shards() >= 1, "Should have at least one shard");

        // Assert: exclusions were applied (secondary minimizers that hit primary were removed)
        assert!(
            stats.excluded_minimizers > 0,
            "Should have excluded some minimizers due to overlap"
        );
        assert!(
            stats.secondary_entries < stats.secondary_entries_original,
            "Secondary entries should be reduced after subtraction"
        );

        // Assert: total entries makes sense
        let expected_total = stats.primary_entries + stats.secondary_entries;
        assert_eq!(stats.total_minimizer_entries, expected_total);

        // Assert: merged index has correct bucket count
        assert_eq!(merged.manifest().bucket_names.len(), 2);
    }

    #[test]
    fn test_streaming_subtract_empty_bucket_error() {
        // Verify error still detected when all bucket's minimizers are excluded in streaming mode
        let tmp = TempDir::new().unwrap();
        let primary_path = tmp.path().join("primary.ryxdi");
        let secondary_path = tmp.path().join("secondary.ryxdi");
        let output_path = tmp.path().join("merged.ryxdi");

        // Primary has all of secondary's minimizers
        let _primary = create_test_index(
            &primary_path,
            64,
            50,
            0x5555,
            "bucket_a",
            vec![100, 200, 300, 400, 500],
        );
        // Secondary's minimizers are a subset of primary's
        let _secondary = create_test_index(
            &secondary_path,
            64,
            50,
            0x5555,
            "bucket_b",
            vec![100, 200, 300],
        );

        let options = MergeOptions {
            subtract_from_primary: true,
            verbose: false,
        };

        // Streaming merge should also detect and error on empty bucket
        let result = merge_indices_streaming(
            &primary_path,
            &secondary_path,
            &output_path,
            &options,
            Some(64 * 1024 * 1024),
            None,
        );

        // Should fail because bucket_b would become empty
        assert!(result.is_err(), "Should fail when bucket becomes empty");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("empty bucket"),
            "Error should mention empty bucket: {}",
            err_msg
        );
        assert!(
            err_msg.contains("bucket_b"),
            "Error should mention bucket name: {}",
            err_msg
        );
    }

    #[test]
    fn test_streaming_subtract_partial_bucket_remains() {
        // Verify partial subtraction works in streaming mode
        let tmp = TempDir::new().unwrap();
        let primary_path = tmp.path().join("primary.ryxdi");
        let secondary_path = tmp.path().join("secondary.ryxdi");
        let output_path = tmp.path().join("merged.ryxdi");

        // Multiple buckets in secondary, one loses some minimizers but not all
        let _primary = create_multi_bucket_index(
            &primary_path,
            64,
            50,
            0x5555,
            vec![(1, "bucket_a", vec!["src_a".to_string()], vec![100, 200])],
        );
        let _secondary = create_multi_bucket_index(
            &secondary_path,
            64,
            50,
            0x5555,
            vec![
                // bucket_b has 100 (shared) and 300, 400 (unique) - will keep 2
                (
                    1,
                    "bucket_b",
                    vec!["src_b".to_string()],
                    vec![100, 300, 400],
                ),
                // bucket_c has 200 (shared) and 500, 600 (unique) - will keep 2
                (
                    2,
                    "bucket_c",
                    vec!["src_c".to_string()],
                    vec![200, 500, 600],
                ),
            ],
        );

        let options = MergeOptions {
            subtract_from_primary: true,
            verbose: false,
        };
        let stats = merge_indices_streaming(
            &primary_path,
            &secondary_path,
            &output_path,
            &options,
            Some(64 * 1024 * 1024),
            None,
        )
        .expect("streaming merge with partial subtraction should succeed");

        // Primary: 2 entries
        assert_eq!(stats.primary_entries, 2);

        // Secondary: originally 6 entries (3+3), 2 removed (100 and 200)
        assert_eq!(stats.secondary_entries_original, 6);
        assert_eq!(stats.secondary_entries, 4);
        assert_eq!(stats.excluded_minimizers, 2);

        // Verify the merged index has all 3 buckets
        let merged = ShardedInvertedIndex::open(&output_path).unwrap();
        assert_eq!(merged.manifest().bucket_names.len(), 3);
    }

    // =========================================================================
    // Phase 3: Parallel Streaming Tests (RED - parallel not implemented yet)
    // =========================================================================

    #[test]
    fn test_parallel_streaming_subtract_deterministic() {
        // Run parallel merge multiple times and verify identical output each time
        let tmp = TempDir::new().unwrap();
        let primary_path = tmp.path().join("primary.ryxdi");
        let secondary_path = tmp.path().join("secondary.ryxdi");

        // Create primary with a good spread of minimizers
        let _primary = create_multi_bucket_index(
            &primary_path,
            64,
            50,
            0x5555,
            vec![
                (
                    1,
                    "bucket_a",
                    vec!["a.fa".to_string()],
                    vec![100, 200, 300, 400, 500],
                ),
                (
                    2,
                    "bucket_b",
                    vec!["b.fa".to_string()],
                    vec![600, 700, 800, 900, 1000],
                ),
            ],
        );

        // Create secondary with some overlap
        let _secondary = create_multi_bucket_index(
            &secondary_path,
            64,
            50,
            0x5555,
            vec![
                (
                    1,
                    "bucket_c",
                    vec!["c.fa".to_string()],
                    vec![200, 300, 1100, 1200],
                ),
                (
                    2,
                    "bucket_d",
                    vec!["d.fa".to_string()],
                    vec![400, 500, 1300, 1400],
                ),
                (
                    3,
                    "bucket_e",
                    vec!["e.fa".to_string()],
                    vec![700, 800, 1500, 1600],
                ),
            ],
        );

        let options = MergeOptions {
            subtract_from_primary: true,
            verbose: false,
        };

        // Run parallel merge 3 times with different output directories
        // Uses rayon's default thread pool for parallel processing
        let mut results = Vec::new();
        for i in 0..3 {
            let output_path = tmp.path().join(format!("merged_{}.ryxdi", i));

            let stats = merge_indices_streaming(
                &primary_path,
                &secondary_path,
                &output_path,
                &options,
                Some(64 * 1024 * 1024),
                None,
            )
            .expect("parallel merge should succeed");

            // Load and collect all minimizer entries for comparison
            let merged = ShardedInvertedIndex::open(&output_path).unwrap();
            let mut all_entries = Vec::new();
            for shard_info in merged.manifest().shards.iter() {
                let shard_path = merged.shard_path(shard_info.shard_id);
                let pairs = read_shard_pairs(&shard_path).unwrap();
                all_entries.extend(pairs);
            }
            all_entries.sort();

            results.push((stats, all_entries));
        }

        // Verify all runs produced identical results
        let (first_stats, first_entries) = &results[0];
        for (i, (stats, entries)) in results.iter().enumerate().skip(1) {
            assert_eq!(
                first_stats.total_minimizer_entries, stats.total_minimizer_entries,
                "Run {} total_minimizer_entries differs from run 0",
                i
            );
            assert_eq!(
                first_stats.excluded_minimizers, stats.excluded_minimizers,
                "Run {} excluded_minimizers differs from run 0",
                i
            );
            assert_eq!(
                first_entries.len(),
                entries.len(),
                "Run {} entry count differs from run 0",
                i
            );
            assert_eq!(
                first_entries, entries,
                "Run {} entries differ from run 0",
                i
            );
        }
    }

    #[test]
    fn test_merge_with_subtraction_classification_correctness() {
        use crate::classify::classify_batch_sharded_merge_join;
        use crate::core::extraction::extract_into;
        use crate::core::workspace::MinimizerWorkspace;
        use crate::types::QueryRecord;

        let tmp = TempDir::new().unwrap();
        let primary_path = tmp.path().join("primary.ryxdi");
        let secondary_path = tmp.path().join("secondary.ryxdi");
        let output_path = tmp.path().join("merged.ryxdi");

        // Sequences where secondary has some overlap with primary
        let seq_primary = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let seq_secondary = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT";

        let k = 32;
        let w = 10;
        let salt = 0x1234u64;
        let mut ws = MinimizerWorkspace::new();

        // Extract minimizers
        extract_into(seq_primary, k, w, salt, &mut ws);
        let mut mins_primary = std::mem::take(&mut ws.buffer);
        mins_primary.sort_unstable();
        mins_primary.dedup();

        extract_into(seq_secondary, k, w, salt, &mut ws);
        let mut mins_secondary = std::mem::take(&mut ws.buffer);
        mins_secondary.sort_unstable();
        mins_secondary.dedup();

        // Create primary index
        let primary_buckets = vec![BucketData {
            bucket_id: 1,
            bucket_name: "primary_bucket".to_string(),
            sources: vec!["primary.fa".to_string()],
            minimizers: mins_primary.clone(),
        }];
        create_parquet_inverted_index(&primary_path, primary_buckets, k, w, salt, None, None)
            .unwrap();

        // Create secondary index
        let secondary_buckets = vec![BucketData {
            bucket_id: 1,
            bucket_name: "secondary_bucket".to_string(),
            sources: vec!["secondary.fa".to_string()],
            minimizers: mins_secondary.clone(),
        }];
        create_parquet_inverted_index(&secondary_path, secondary_buckets, k, w, salt, None, None)
            .unwrap();

        // Merge WITH subtraction
        let merge_options = MergeOptions {
            subtract_from_primary: true,
            verbose: false,
        };
        let stats = merge_indices_streaming(
            &primary_path,
            &secondary_path,
            &output_path,
            &merge_options,
            None,
            None,
        )
        .expect("merge should succeed");

        assert_eq!(stats.total_buckets, 2);
        assert!(
            stats.excluded_minimizers > 0,
            "Should have excluded some minimizers"
        );
        assert!(
            stats.secondary_entries < stats.secondary_entries_original,
            "Secondary entries should be reduced after subtraction"
        );

        // Open merged index
        let merged = ShardedInvertedIndex::open(&output_path).unwrap();

        // Classify primary sequence - should still match primary bucket
        let records: Vec<QueryRecord> = vec![(0, seq_primary.as_slice(), None)];

        let threshold = 0.1;
        let results =
            classify_batch_sharded_merge_join(&merged, None, &records, threshold, None, None)
                .expect("classification should succeed");

        // Primary query should match primary bucket (bucket 1)
        let primary_match = results.iter().find(|r| r.bucket_id == 1);
        assert!(
            primary_match.is_some(),
            "Primary query should still match primary bucket"
        );

        // Primary query should NOT match secondary bucket (bucket 2) at high score
        // because shared minimizers were removed from secondary
        let secondary_match = results.iter().find(|r| r.bucket_id == 2);
        if let Some(m) = secondary_match {
            // If there's a secondary match, its score should be lower than primary match
            assert!(
                m.score < primary_match.unwrap().score,
                "Secondary match score should be lower than primary after subtraction"
            );
        }
    }

    // =========================================================================
    // Phase 7: Streaming Merge Classification Correctness Test
    // =========================================================================

    #[test]
    fn test_streaming_merge_classification_correctness() {
        use crate::classify::classify_batch_sharded_merge_join;
        use crate::core::extraction::extract_into;
        use crate::core::workspace::MinimizerWorkspace;
        use crate::types::QueryRecord;

        let tmp = TempDir::new().unwrap();
        let primary_path = tmp.path().join("primary.ryxdi");
        let secondary_path = tmp.path().join("secondary.ryxdi");
        let output_path = tmp.path().join("merged.ryxdi");

        // Create test sequences:
        // - Primary sequence has specific minimizers
        // - Secondary sequence has some overlap with primary + unique minimizers
        // The trailing portion is unique to secondary (poly-T pattern)
        let seq_primary = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let seq_secondary_unique = b"TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT";
        // Secondary has shared prefix with primary + unique suffix
        let seq_secondary_overlap = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT";

        let k = 32;
        let w = 10;
        let salt = 0x1234u64;
        let mut ws = MinimizerWorkspace::new();

        // Extract minimizers for each sequence
        extract_into(seq_primary, k, w, salt, &mut ws);
        let mut mins_primary = std::mem::take(&mut ws.buffer);
        mins_primary.sort_unstable();
        mins_primary.dedup();

        extract_into(seq_secondary_unique, k, w, salt, &mut ws);
        let mut mins_secondary_unique = std::mem::take(&mut ws.buffer);
        mins_secondary_unique.sort_unstable();
        mins_secondary_unique.dedup();

        extract_into(seq_secondary_overlap, k, w, salt, &mut ws);
        let mut mins_secondary_overlap = std::mem::take(&mut ws.buffer);
        mins_secondary_overlap.sort_unstable();
        mins_secondary_overlap.dedup();

        // Calculate expected overlaps
        let primary_set: std::collections::HashSet<_> = mins_primary.iter().copied().collect();
        let overlap_set: std::collections::HashSet<_> =
            mins_secondary_overlap.iter().copied().collect();
        let shared_count = primary_set.intersection(&overlap_set).count();
        assert!(shared_count > 0, "Test requires overlapping minimizers");

        // Create primary index (bucket 1)
        let primary_buckets = vec![BucketData {
            bucket_id: 1,
            bucket_name: "primary_bucket".to_string(),
            sources: vec!["primary.fa".to_string()],
            minimizers: mins_primary.clone(),
        }];
        create_parquet_inverted_index(&primary_path, primary_buckets, k, w, salt, None, None)
            .unwrap();

        // Create secondary index with two buckets:
        // - bucket 1: has overlap with primary (will have minimizers removed)
        // - bucket 2: completely unique (no overlap)
        let secondary_buckets = vec![
            BucketData {
                bucket_id: 1,
                bucket_name: "secondary_overlap".to_string(),
                sources: vec!["secondary_overlap.fa".to_string()],
                minimizers: mins_secondary_overlap.clone(),
            },
            BucketData {
                bucket_id: 2,
                bucket_name: "secondary_unique".to_string(),
                sources: vec!["secondary_unique.fa".to_string()],
                minimizers: mins_secondary_unique.clone(),
            },
        ];
        create_parquet_inverted_index(&secondary_path, secondary_buckets, k, w, salt, None, None)
            .unwrap();

        // Merge using STREAMING function (not deprecated merge_indices)
        let merge_options = MergeOptions {
            subtract_from_primary: true,
            verbose: false,
        };
        let stats = merge_indices_streaming(
            &primary_path,
            &secondary_path,
            &output_path,
            &merge_options,
            Some(64 * 1024 * 1024), // 64MB memory budget
            None,
        )
        .expect("streaming merge should succeed");

        // Verify subtraction happened
        assert_eq!(
            stats.total_buckets, 3,
            "Should have 1 primary + 2 secondary buckets"
        );
        assert!(
            stats.excluded_minimizers > 0,
            "Should have excluded shared minimizers"
        );
        assert!(
            stats.secondary_entries < stats.secondary_entries_original,
            "Secondary entries should be reduced after subtraction"
        );

        // Open merged index for classification
        let merged = ShardedInvertedIndex::open(&output_path).unwrap();

        // Test 1: Primary sequence should match primary bucket (bucket 1)
        let records: Vec<QueryRecord> = vec![(0, seq_primary.as_slice(), None)];
        let threshold = 0.1;
        let results =
            classify_batch_sharded_merge_join(&merged, None, &records, threshold, None, None)
                .expect("classification should succeed");

        let primary_match = results.iter().find(|r| r.bucket_id == 1);
        assert!(
            primary_match.is_some(),
            "Primary sequence should match primary bucket"
        );
        let primary_score = primary_match.unwrap().score;

        // Test 2: Primary sequence should NOT strongly match the secondary overlap bucket
        // (because shared minimizers were removed)
        let secondary_overlap_match = results.iter().find(|r| r.bucket_id == 2);
        if let Some(m) = secondary_overlap_match {
            assert!(
                m.score < primary_score,
                "Secondary overlap bucket score ({}) should be lower than primary ({}) after subtraction",
                m.score, primary_score
            );
        }

        // Test 3: Secondary unique sequence should match secondary unique bucket
        let unique_records: Vec<QueryRecord> = vec![(0, seq_secondary_unique.as_slice(), None)];
        let unique_results = classify_batch_sharded_merge_join(
            &merged,
            None,
            &unique_records,
            threshold,
            None,
            None,
        )
        .expect("classification should succeed");

        let unique_match = unique_results.iter().find(|r| r.bucket_id == 3);
        assert!(
            unique_match.is_some(),
            "Secondary unique sequence should match its bucket"
        );

        // Test 4: The unique secondary bucket should not match primary sequence at all
        // (no overlap between poly-T and primary sequence)
        let unique_primary_match = results.iter().find(|r| r.bucket_id == 3);
        assert!(
            unique_primary_match.is_none(),
            "Primary sequence should not match secondary unique bucket"
        );
    }
}
