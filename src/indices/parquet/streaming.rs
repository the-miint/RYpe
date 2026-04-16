//! Streaming Parquet index creation.
//!
//! This module provides efficient streaming creation of Parquet inverted indices
//! using k-way merge, supporting both sequential and parallel sharding.

use crate::error::{Result, RypeError};
use arrow::array::ArrayRef;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::constants::{MIN_ENTRIES_PER_PARALLEL_PARTITION, PARQUET_BATCH_SIZE};

use super::buckets::write_buckets_parquet;
use super::manifest::{
    create_index_directory, BucketData, InvertedManifest, InvertedShardInfo, ParquetManifest,
    ParquetShardFormat,
};
use super::options::ParquetWriteOptions;
use super::{files, FORMAT_MAGIC, FORMAT_VERSION};

/// K-way merge heap entry: (Reverse((minimizer, bucket_id)), bucket_index, position)
type MergeHeapEntry = (Reverse<(u64, u32)>, usize, usize);

/// Create a Parquet inverted index directly from bucket data.
///
/// This streams (minimizer, bucket_id) pairs through a k-way merge
/// directly to Parquet, avoiding the intermediate CSR representation.
///
/// # Arguments
/// * `output_dir` - Directory to create (e.g., "index.ryxdi")
/// * `buckets` - Bucket data with sorted, deduplicated minimizers (VALIDATED)
/// * `k` - K-mer size
/// * `w` - Window size
/// * `salt` - Hash salt
/// * `max_shard_bytes` - Optional max shard size in bytes (target, not exact limit)
/// * `options` - Optional Parquet write options (compression, bloom filters, etc.)
///
/// # Returns
/// The manifest describing the created index.
///
/// # Errors
/// Returns an error if any bucket has unsorted or duplicate minimizers.
#[allow(clippy::too_many_arguments)]
pub fn create_parquet_inverted_index(
    output_dir: &Path,
    buckets: Vec<BucketData>,
    k: usize,
    w: usize,
    salt: u64,
    max_shard_bytes: Option<usize>,
    options: Option<&ParquetWriteOptions>,
    bucket_file_stats: Option<&HashMap<u32, crate::types::BucketFileStats>>,
) -> Result<ParquetManifest> {
    let opts = options.cloned().unwrap_or_default();
    opts.validate()?;

    // Validate bucket data upfront - buckets must be sorted and deduplicated
    for bucket in &buckets {
        bucket.validate().map_err(|e| {
            RypeError::validation(format!(
                "invalid bucket data for bucket '{}' (id={}): {}",
                bucket.bucket_name, bucket.bucket_id, e
            ))
        })?;
    }

    // Create directory structure
    create_index_directory(output_dir)?;

    // Collect bucket metadata
    let mut bucket_names: HashMap<u32, String> = HashMap::new();
    let mut bucket_sources: HashMap<u32, Vec<String>> = HashMap::new();
    let mut bucket_minimizer_counts: HashMap<u32, usize> = HashMap::new();
    let mut total_minimizers: u64 = 0;

    for bucket in &buckets {
        bucket_names.insert(bucket.bucket_id, bucket.bucket_name.clone());
        bucket_sources.insert(bucket.bucket_id, bucket.sources.clone());
        bucket_minimizer_counts.insert(bucket.bucket_id, bucket.minimizers.len());
        total_minimizers += bucket.minimizers.len() as u64;
    }

    // Write bucket metadata
    write_buckets_parquet(
        output_dir,
        &bucket_names,
        &bucket_sources,
        bucket_file_stats,
    )?;

    // Stream inverted pairs to Parquet shards
    let (shard_infos, has_overlapping_shards) = stream_to_parquet_shards(
        output_dir,
        buckets,
        max_shard_bytes.unwrap_or(usize::MAX),
        &opts,
    )?;

    // Compute total entries
    let total_entries: u64 = shard_infos.iter().map(|s| s.num_entries).sum();

    // Compute source hash for validation
    let source_hash = compute_source_hash(&bucket_minimizer_counts);

    // Build manifest with explicit format field
    let manifest = ParquetManifest {
        magic: FORMAT_MAGIC.to_string(),
        format_version: FORMAT_VERSION,
        k,
        w,
        salt,
        source_hash,
        num_buckets: bucket_names.len() as u32,
        total_minimizers,
        inverted: Some(InvertedManifest {
            format: ParquetShardFormat::Parquet,
            num_shards: shard_infos.len() as u32,
            total_entries,
            has_overlapping_shards,
            shards: shard_infos,
        }),
    };

    manifest.save(output_dir)?;

    Ok(manifest)
}

/// Compute a hash from bucket minimizer counts for validation.
pub fn compute_source_hash(counts: &HashMap<u32, usize>) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut pairs: Vec<(u32, usize)> = counts.iter().map(|(&id, &c)| (id, c)).collect();
    pairs.sort_by_key(|(id, _)| *id);

    let mut hasher = DefaultHasher::new();
    for (id, count) in pairs {
        id.hash(&mut hasher);
        count.hash(&mut hasher);
    }
    hasher.finish()
}

/// Stream (minimizer, bucket_id) pairs to Parquet shards using k-way merge.
///
/// Returns a tuple of (shard_infos, has_overlapping_shards):
/// - Sequential mode: shards may have overlapping minimizer ranges (buckets share minimizers)
/// - Parallel mode: shards have non-overlapping minimizer ranges (range-partitioned)
fn stream_to_parquet_shards(
    output_dir: &Path,
    buckets: Vec<BucketData>,
    max_shard_bytes: usize,
    options: &ParquetWriteOptions,
) -> Result<(Vec<InvertedShardInfo>, bool)> {
    if buckets.is_empty() || buckets.iter().all(|b| b.minimizers.is_empty()) {
        // Empty index - create single empty shard info
        return Ok((
            vec![InvertedShardInfo {
                shard_id: 0,
                min_minimizer: 0,
                max_minimizer: 0,
                num_entries: 0,
            }],
            false, // Single empty shard has no overlap
        ));
    }

    // For large indices, use parallel range partitioning
    let total_entries: usize = buckets.iter().map(|b| b.minimizers.len()).sum();
    let num_cpus = rayon::current_num_threads();

    // Use parallel sharding if we have enough data and multiple cores
    let use_parallel =
        total_entries > MIN_ENTRIES_PER_PARALLEL_PARTITION * num_cpus && num_cpus > 1;

    if use_parallel && max_shard_bytes < usize::MAX {
        // Parallel: partition minimizer space and process ranges in parallel
        // Shards are range-partitioned, so they do NOT overlap
        stream_to_shards_parallel(output_dir, buckets, max_shard_bytes, num_cpus, options)
            .map(|shards| (shards, false))
    } else {
        // Sequential: single k-way merge
        // Shards are created by size threshold, so buckets may share minimizers across shards
        stream_to_shards_sequential(output_dir, buckets, max_shard_bytes, options)
            .map(|shards| (shards, true))
    }
}

/// Sequential k-way merge to Parquet shards.
fn stream_to_shards_sequential(
    output_dir: &Path,
    buckets: Vec<BucketData>,
    max_shard_bytes: usize,
    options: &ParquetWriteOptions,
) -> Result<Vec<InvertedShardInfo>> {
    // Prepare bucket data for k-way merge
    let bucket_slices: Vec<(u32, &[u64])> = buckets
        .iter()
        .filter(|b| !b.minimizers.is_empty())
        .map(|b| (b.bucket_id, b.minimizers.as_slice()))
        .collect();

    // K-way merge heap: (Reverse((minimizer, bucket_id)), bucket_index, position)
    let mut heap: BinaryHeap<MergeHeapEntry> = BinaryHeap::with_capacity(bucket_slices.len());

    // Initialize heap with first element from each bucket
    for (idx, &(bucket_id, mins)) in bucket_slices.iter().enumerate() {
        heap.push((Reverse((mins[0], bucket_id)), idx, 0));
    }

    // Shard writer state
    let mut shard_infos: Vec<InvertedShardInfo> = Vec::new();
    let mut current_shard_id: u32 = 0;
    let mut current_writer: Option<ShardWriter> = None;
    let mut current_shard_entries: u64 = 0;
    let mut current_shard_min: u64 = 0;
    let mut current_shard_max: u64 = 0;

    // Batching buffers
    let mut minimizer_batch: Vec<u64> = Vec::with_capacity(PARQUET_BATCH_SIZE);
    let mut bucket_id_batch: Vec<u32> = Vec::with_capacity(PARQUET_BATCH_SIZE);

    while let Some((Reverse((minimizer, bucket_id)), bucket_idx, pos)) = heap.pop() {
        // Check if we need a new shard (use actual file size)
        let need_new_shard = if let Some(ref writer) = current_writer {
            // Use actual bytes written for accurate shard size decisions
            let current_bytes = writer.bytes_written();
            current_bytes >= max_shard_bytes && !minimizer_batch.is_empty()
        } else {
            true // No writer yet, need to create one
        };

        if need_new_shard {
            // Flush current writer if exists
            if let Some(mut writer) = current_writer.take() {
                // Flush remaining batch
                if !minimizer_batch.is_empty() {
                    writer.write_batch(&minimizer_batch, &bucket_id_batch)?;
                    minimizer_batch.clear();
                    bucket_id_batch.clear();
                }
                writer.finish()?;
                shard_infos.push(InvertedShardInfo {
                    shard_id: current_shard_id,
                    min_minimizer: current_shard_min,
                    max_minimizer: current_shard_max,
                    num_entries: current_shard_entries,
                });
                current_shard_id += 1;
            }

            // Start new shard
            let shard_path = output_dir
                .join(files::INVERTED_DIR)
                .join(files::inverted_shard(current_shard_id));
            current_writer = Some(ShardWriter::new(&shard_path, options)?);
            current_shard_entries = 0;
            current_shard_min = minimizer;
        }

        // Add to batch
        minimizer_batch.push(minimizer);
        bucket_id_batch.push(bucket_id);
        current_shard_entries += 1;
        current_shard_max = minimizer; // Sorted order guarantees this is always >= previous

        // Flush batch if full
        if minimizer_batch.len() >= PARQUET_BATCH_SIZE {
            if let Some(ref mut writer) = current_writer {
                writer.write_batch(&minimizer_batch, &bucket_id_batch)?;
            }
            minimizer_batch.clear();
            bucket_id_batch.clear();
        }

        // Advance to next element in this bucket
        let (_, mins) = bucket_slices[bucket_idx];
        let next_pos = pos + 1;
        if next_pos < mins.len() {
            let next_bucket_id = bucket_slices[bucket_idx].0;
            heap.push((
                Reverse((mins[next_pos], next_bucket_id)),
                bucket_idx,
                next_pos,
            ));
        }
    }

    // Flush final shard
    if let Some(mut writer) = current_writer.take() {
        if !minimizer_batch.is_empty() {
            writer.write_batch(&minimizer_batch, &bucket_id_batch)?;
        }
        writer.finish()?;
        shard_infos.push(InvertedShardInfo {
            shard_id: current_shard_id,
            min_minimizer: current_shard_min,
            max_minimizer: current_shard_max,
            num_entries: current_shard_entries,
        });
    }

    Ok(shard_infos)
}

/// Parallel range-partitioned sharding for large indices.
///
/// Partitions the minimizer space into ranges and processes each range in parallel.
fn stream_to_shards_parallel(
    output_dir: &Path,
    buckets: Vec<BucketData>,
    max_shard_bytes: usize,
    num_partitions: usize,
    options: &ParquetWriteOptions,
) -> Result<Vec<InvertedShardInfo>> {
    // Compute range boundaries by sampling minimizers
    let range_boundaries = compute_range_boundaries(&buckets, num_partitions);

    // Process each range in parallel
    let partition_results: Vec<Result<Vec<(InvertedShardInfo, PathBuf)>>> = range_boundaries
        .par_windows(2)
        .enumerate()
        .map(|(partition_idx, window)| {
            let range_start = window[0];
            let range_end = window[1];

            // Filter bucket data to this range using binary search (O(log n) per bucket)
            let filtered_buckets: Vec<(u32, &[u64])> = buckets
                .iter()
                .filter_map(|b| {
                    // Binary search for range bounds (buckets are sorted)
                    let start = b.minimizers.partition_point(|&m| m < range_start);
                    let end = b.minimizers.partition_point(|&m| m < range_end);
                    if start < end {
                        Some((b.bucket_id, &b.minimizers[start..end]))
                    } else {
                        None
                    }
                })
                .collect();

            if filtered_buckets.is_empty() {
                return Ok(vec![]);
            }

            // Process this partition
            process_partition(
                output_dir,
                &filtered_buckets,
                partition_idx as u32,
                num_partitions as u32,
                max_shard_bytes,
                options,
            )
        })
        .collect();

    // Collect shard infos with their file paths
    let mut all_shards_with_paths: Vec<(InvertedShardInfo, PathBuf)> = Vec::new();
    for result in partition_results {
        let partition_shards = result?;
        all_shards_with_paths.extend(partition_shards);
    }

    // Sort by min_minimizer to ensure consistent ordering
    all_shards_with_paths.sort_by_key(|(s, _)| s.min_minimizer);

    // Rename files to canonical names (shard.0.parquet, shard.1.parquet, ...)
    // and update shard IDs to match
    let mut final_shards: Vec<InvertedShardInfo> = Vec::with_capacity(all_shards_with_paths.len());
    for (new_id, (mut shard_info, old_path)) in all_shards_with_paths.into_iter().enumerate() {
        let new_id = new_id as u32;
        let new_path = output_dir
            .join(files::INVERTED_DIR)
            .join(files::inverted_shard(new_id));

        // Rename file from partition-specific name to canonical name
        if old_path != new_path {
            std::fs::rename(&old_path, &new_path).map_err(|e| {
                RypeError::io(
                    old_path.clone(),
                    "rename shard",
                    std::io::Error::new(
                        e.kind(),
                        format!("{} -> {}: {}", old_path.display(), new_path.display(), e),
                    ),
                )
            })?;
        }

        shard_info.shard_id = new_id;
        final_shards.push(shard_info);
    }

    Ok(final_shards)
}

/// Compute range boundaries for parallel partitioning.
fn compute_range_boundaries(buckets: &[BucketData], num_partitions: usize) -> Vec<u64> {
    // Sample minimizers to estimate distribution
    let sample_size = 10000;
    let mut samples: Vec<u64> = Vec::with_capacity(sample_size);

    let total_mins: usize = buckets.iter().map(|b| b.minimizers.len()).sum();
    let sample_rate = (sample_size as f64 / total_mins as f64).min(1.0);

    for bucket in buckets {
        for &min in &bucket.minimizers {
            if rand_sample(min, sample_rate) {
                samples.push(min);
            }
        }
    }

    samples.sort_unstable();

    // Compute quantile boundaries
    let mut boundaries = vec![0u64]; // Start at 0
    for i in 1..num_partitions {
        let idx = (samples.len() * i) / num_partitions;
        if idx < samples.len() {
            boundaries.push(samples[idx]);
        }
    }
    boundaries.push(u64::MAX); // End at max

    boundaries
}

/// Deterministic sampling based on the minimizer value itself.
///
/// This is thread-safe and produces consistent results: the same minimizer
/// will always be sampled or not sampled, regardless of which thread processes it.
/// This is critical for correct range boundary computation in parallel sharding.
///
/// Uses integer arithmetic to avoid floating-point precision issues.
fn rand_sample(minimizer: u64, rate: f64) -> bool {
    // Handle edge cases to avoid floating-point comparison issues
    if rate <= 0.0 {
        return false;
    }
    if rate >= 1.0 {
        return true;
    }

    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Hash the minimizer value to get a pseudo-random but deterministic decision
    let mut hasher = DefaultHasher::new();
    minimizer.hash(&mut hasher);
    let hash = hasher.finish();

    // Use integer comparison to avoid floating-point precision loss
    // Convert rate to a threshold in the u64 space
    let threshold = (u64::MAX as f64 * rate) as u64;
    hash < threshold
}

/// Process a single partition of the minimizer space.
/// Returns shard infos paired with their file paths (for later renaming).
fn process_partition(
    output_dir: &Path,
    filtered_buckets: &[(u32, &[u64])],
    partition_idx: u32,
    _total_partitions: u32,
    max_shard_bytes: usize,
    options: &ParquetWriteOptions,
) -> Result<Vec<(InvertedShardInfo, PathBuf)>> {
    // K-way merge for this partition (buckets already sliced)
    let bucket_slices: Vec<(u32, &[u64])> = filtered_buckets.to_vec();

    let mut heap: BinaryHeap<MergeHeapEntry> = BinaryHeap::with_capacity(bucket_slices.len());

    for (idx, &(bucket_id, mins)) in bucket_slices.iter().enumerate() {
        if !mins.is_empty() {
            heap.push((Reverse((mins[0], bucket_id)), idx, 0));
        }
    }

    let mut shard_infos: Vec<(InvertedShardInfo, PathBuf)> = Vec::new();
    let mut local_shard_id: u32 = 0;
    let mut current_writer: Option<ShardWriter> = None;
    let mut current_shard_path: PathBuf = PathBuf::new();
    let mut current_shard_entries: u64 = 0;
    let mut current_shard_min: u64 = 0;
    let mut current_shard_max: u64 = 0;

    let mut minimizer_batch: Vec<u64> = Vec::with_capacity(PARQUET_BATCH_SIZE);
    let mut bucket_id_batch: Vec<u32> = Vec::with_capacity(PARQUET_BATCH_SIZE);

    while let Some((Reverse((minimizer, bucket_id)), bucket_idx, pos)) = heap.pop() {
        let need_new_shard = if let Some(ref writer) = current_writer {
            writer.bytes_written() >= max_shard_bytes && !minimizer_batch.is_empty()
        } else {
            true
        };

        if need_new_shard {
            if let Some(mut writer) = current_writer.take() {
                if !minimizer_batch.is_empty() {
                    writer.write_batch(&minimizer_batch, &bucket_id_batch)?;
                    minimizer_batch.clear();
                    bucket_id_batch.clear();
                }
                writer.finish()?;
                shard_infos.push((
                    InvertedShardInfo {
                        shard_id: local_shard_id,
                        min_minimizer: current_shard_min,
                        max_minimizer: current_shard_max,
                        num_entries: current_shard_entries,
                    },
                    current_shard_path.clone(),
                ));
                local_shard_id += 1;
            }

            // Use partition index in filename to avoid conflicts during parallel writes
            let shard_name = format!("shard.{}.{}.parquet", partition_idx, local_shard_id);
            current_shard_path = output_dir.join(files::INVERTED_DIR).join(&shard_name);
            current_writer = Some(ShardWriter::new(&current_shard_path, options)?);
            current_shard_entries = 0;
            current_shard_min = minimizer;
        }

        minimizer_batch.push(minimizer);
        bucket_id_batch.push(bucket_id);
        current_shard_entries += 1;
        current_shard_max = minimizer; // Sorted order guarantees this is always >= previous

        if minimizer_batch.len() >= PARQUET_BATCH_SIZE {
            if let Some(ref mut writer) = current_writer {
                writer.write_batch(&minimizer_batch, &bucket_id_batch)?;
            }
            minimizer_batch.clear();
            bucket_id_batch.clear();
        }

        let (_, mins) = bucket_slices[bucket_idx];
        let next_pos = pos + 1;
        if next_pos < mins.len() {
            let next_bucket_id = bucket_slices[bucket_idx].0;
            heap.push((
                Reverse((mins[next_pos], next_bucket_id)),
                bucket_idx,
                next_pos,
            ));
        }
    }

    if let Some(mut writer) = current_writer.take() {
        if !minimizer_batch.is_empty() {
            writer.write_batch(&minimizer_batch, &bucket_id_batch)?;
        }
        writer.finish()?;
        shard_infos.push((
            InvertedShardInfo {
                shard_id: local_shard_id,
                min_minimizer: current_shard_min,
                max_minimizer: current_shard_max,
                num_entries: current_shard_entries,
            },
            current_shard_path,
        ));
    }

    Ok(shard_infos)
}

/// Accumulates (minimizer, bucket_id) entries and tracks when to flush to a shard.
///
/// This struct provides size-aware accumulation for streaming shard creation,
/// triggering flushes when the estimated in-memory size exceeds the configured threshold.
///
/// # Memory Estimation
///
/// The size threshold (`max_shard_bytes`) controls **resident memory usage**, not on-disk
/// shard size. The actual Parquet files will typically be significantly smaller due to:
/// - DELTA_BINARY_PACKED encoding for sorted integers
/// - Compression (ZSTD/Snappy)
///
/// The estimate uses [`BYTES_PER_ENTRY`](Self::BYTES_PER_ENTRY) as an upper bound for
/// in-memory representation (12 bytes data + padding/overhead).
///
/// # Thread Safety
///
/// `ShardAccumulator` is `Send` but not `Sync`. A single accumulator should only be
/// used from one thread at a time. For parallel index building, use separate
/// accumulators per thread/partition.
///
/// # Shard ID Limits
///
/// The shard ID is a `u32`, supporting up to 4 billion shards. Attempting to create
/// more shards will return an error.
pub struct ShardAccumulator {
    /// Accumulated entries: (minimizer, bucket_id)
    entries: Vec<(u64, u32)>,
    /// Maximum bytes per shard before flush
    max_shard_bytes: usize,
    /// Output directory for shards
    output_dir: PathBuf,
    /// Current shard ID (increments on each flush)
    current_shard_id: u32,
    /// Parquet write options
    options: ParquetWriteOptions,
    /// Accumulated shard infos from completed flushes
    shard_infos: Vec<InvertedShardInfo>,
}

/// Minimum allowed value for max_shard_bytes to prevent pathological behavior.
/// Set to 1MB - smaller shards are inefficient due to Parquet overhead.
pub const MIN_SHARD_BYTES: usize = 1024 * 1024;

impl ShardAccumulator {
    /// Upper bound estimate for bytes per entry in resident memory.
    ///
    /// This is used to estimate when to flush based on **in-memory** buffer size,
    /// NOT on-disk Parquet size. Actual Parquet files will be smaller due to
    /// DELTA_BINARY_PACKED encoding and compression.
    ///
    /// Calculation:
    /// - 8 bytes for u64 minimizer
    /// - 4 bytes for u32 bucket_id
    /// - 4 bytes for Vec overhead/alignment per entry
    ///
    /// This is intentionally conservative to ensure we flush before exceeding
    /// the memory target.
    pub const BYTES_PER_ENTRY: usize = 16;

    /// Create a new accumulator for size tracking only (no file writing).
    ///
    /// Use `with_output_dir()` for an accumulator that can flush to files.
    ///
    /// # Panics
    ///
    /// Panics if `max_shard_bytes` is less than [`MIN_SHARD_BYTES`] (1MB).
    /// Use `try_new()` for a fallible version.
    pub fn new(max_shard_bytes: usize) -> Self {
        assert!(
            max_shard_bytes >= MIN_SHARD_BYTES,
            "max_shard_bytes ({}) must be at least MIN_SHARD_BYTES ({})",
            max_shard_bytes,
            MIN_SHARD_BYTES
        );
        Self {
            entries: Vec::new(),
            max_shard_bytes,
            output_dir: PathBuf::new(),
            current_shard_id: 0,
            options: ParquetWriteOptions::default(),
            shard_infos: Vec::new(),
        }
    }

    /// Create a new accumulator that writes shards to the specified directory.
    ///
    /// # Arguments
    /// * `output_dir` - Directory where shards will be written (e.g., "index.ryxdi")
    /// * `max_shard_bytes` - Maximum bytes per shard before triggering a flush
    /// * `options` - Optional Parquet write options (compression, bloom filters, etc.)
    ///
    /// # Panics
    ///
    /// Panics if `max_shard_bytes` is less than [`MIN_SHARD_BYTES`] (1MB).
    pub fn with_output_dir(
        output_dir: &Path,
        max_shard_bytes: usize,
        options: Option<&ParquetWriteOptions>,
    ) -> Self {
        assert!(
            max_shard_bytes >= MIN_SHARD_BYTES,
            "max_shard_bytes ({}) must be at least MIN_SHARD_BYTES ({})",
            max_shard_bytes,
            MIN_SHARD_BYTES
        );
        Self {
            entries: Vec::new(),
            max_shard_bytes,
            output_dir: output_dir.to_path_buf(),
            current_shard_id: 0,
            options: options.cloned().unwrap_or_default(),
            shard_infos: Vec::new(),
        }
    }

    /// Create an accumulator for parallel processing with a specific starting shard ID.
    ///
    /// This is used when multiple accumulators are used in parallel, each needing
    /// non-overlapping shard ID ranges.
    ///
    /// # Arguments
    /// * `output_dir` - Directory where shards will be written (e.g., "index.ryxdi")
    /// * `max_shard_bytes` - Maximum bytes per shard before triggering a flush
    /// * `start_shard_id` - Starting shard ID for this accumulator
    /// * `options` - Optional Parquet write options (compression, bloom filters, etc.)
    ///
    /// # Panics
    ///
    /// Panics if `max_shard_bytes` is less than [`MIN_SHARD_BYTES`] (1MB).
    pub fn with_start_shard_id(
        output_dir: &Path,
        max_shard_bytes: usize,
        start_shard_id: u32,
        options: Option<&ParquetWriteOptions>,
    ) -> Self {
        assert!(
            max_shard_bytes >= MIN_SHARD_BYTES,
            "max_shard_bytes ({}) must be at least MIN_SHARD_BYTES ({})",
            max_shard_bytes,
            MIN_SHARD_BYTES
        );
        Self {
            entries: Vec::new(),
            max_shard_bytes,
            output_dir: output_dir.to_path_buf(),
            current_shard_id: start_shard_id,
            options: options.cloned().unwrap_or_default(),
            shard_infos: Vec::new(),
        }
    }

    /// Returns the current shard ID (the ID that will be used for the next shard).
    pub fn current_shard_id(&self) -> u32 {
        self.current_shard_id
    }

    /// Returns the configured maximum shard size in bytes.
    pub fn max_shard_bytes(&self) -> usize {
        self.max_shard_bytes
    }

    /// Returns the estimated current size in bytes.
    ///
    /// Uses `capacity()` rather than `len()` to account for the actual memory
    /// allocated by the Vec, which may be up to 2x the number of elements.
    pub fn current_size_bytes(&self) -> usize {
        self.entries.capacity() * Self::BYTES_PER_ENTRY
    }

    /// Returns the number of accumulated entries.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Add entries to the accumulator.
    pub fn add_entries(&mut self, entries: &[(u64, u32)]) {
        self.entries.extend_from_slice(entries);
    }

    /// Add entries from minimizers directly without intermediate allocation.
    ///
    /// This is more efficient than creating a temporary Vec of (minimizer, bucket_id)
    /// pairs and calling `add_entries`.
    pub fn add_entries_from_minimizers(&mut self, minimizers: &[u64], bucket_id: u32) {
        self.entries.reserve(minimizers.len());
        for &m in minimizers {
            self.entries.push((m, bucket_id));
        }
    }

    /// Returns true if the accumulator should be flushed (size exceeds threshold).
    pub fn should_flush(&self) -> bool {
        self.current_size_bytes() >= self.max_shard_bytes
    }

    /// Flush current entries to a new shard file.
    ///
    /// Sorts entries by (minimizer, bucket_id), writes to Parquet, clears buffer,
    /// and increments the shard ID for the next flush.
    ///
    /// Returns `Ok(None)` if the accumulator is empty (no-op).
    /// Returns `Ok(Some(shard_info))` on successful flush.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The output directory was not configured (created with `new()` instead of `with_output_dir()`)
    /// - The shard ID would overflow (> 4 billion shards)
    /// - I/O errors during Parquet writing
    pub fn flush_shard(&mut self) -> Result<Option<InvertedShardInfo>> {
        // Empty flush is a no-op, not an error
        if self.entries.is_empty() {
            return Ok(None);
        }

        // Validate output_dir was configured
        if self.output_dir.as_os_str().is_empty() {
            return Err(RypeError::validation(format!(
                "Cannot flush {} entries: accumulator created with new() has no output directory. \
                 Use with_output_dir() to create an accumulator that can write files.",
                self.entries.len()
            )));
        }

        // Check for shard ID overflow before incrementing
        if self.current_shard_id == u32::MAX {
            return Err(RypeError::validation(format!(
                "Shard ID overflow: cannot create shard {} (max is {}). \
                 {} entries pending, {} shards already written.",
                self.current_shard_id,
                u32::MAX - 1,
                self.entries.len(),
                self.shard_infos.len()
            )));
        }

        // Sort entries by (minimizer, bucket_id) and remove duplicates
        self.entries.sort_unstable();
        self.entries.dedup();

        // Extract min/max for shard info (after dedup for correct count)
        let min_minimizer = self.entries.first().unwrap().0;
        let max_minimizer = self.entries.last().unwrap().0;
        let num_entries = self.entries.len() as u64;

        // Write to parquet file
        let shard_path = self
            .output_dir
            .join(files::INVERTED_DIR)
            .join(files::inverted_shard(self.current_shard_id));

        write_shard_from_pairs(&shard_path, &self.entries, &self.options)?;

        // Build shard info - only after successful write
        let shard_info = InvertedShardInfo {
            shard_id: self.current_shard_id,
            min_minimizer,
            max_minimizer,
            num_entries,
        };

        // Store shard info and increment ID only after successful write
        self.shard_infos.push(shard_info);
        self.current_shard_id = self.current_shard_id.checked_add(1).ok_or_else(|| {
            RypeError::validation("Shard ID overflow after successful write".to_string())
        })?;

        // Clear buffer and release memory
        self.entries.clear();
        self.entries.shrink_to_fit();

        Ok(Some(shard_info))
    }

    /// Finish accumulation, flushing any remaining entries.
    ///
    /// Returns all shard infos from this accumulator. If the accumulator is empty
    /// and no shards were ever flushed, returns an empty Vec.
    pub fn finish(mut self) -> Result<Vec<InvertedShardInfo>> {
        // flush_shard handles empty case gracefully (returns Ok(None))
        self.flush_shard()?;
        Ok(self.shard_infos)
    }
}

/// Write (minimizer, bucket_id) pairs to a Parquet shard file.
///
/// Entries must be sorted by (minimizer, bucket_id) before calling.
fn write_shard_from_pairs(
    path: &Path,
    entries: &[(u64, u32)],
    options: &ParquetWriteOptions,
) -> Result<()> {
    let mut writer = ShardWriter::new(path, options)?;

    // Preallocate batch buffers outside the loop to avoid repeated allocations
    let mut minimizers: Vec<u64> = Vec::with_capacity(PARQUET_BATCH_SIZE);
    let mut bucket_ids: Vec<u32> = Vec::with_capacity(PARQUET_BATCH_SIZE);

    // Write in batches
    for chunk in entries.chunks(PARQUET_BATCH_SIZE) {
        minimizers.clear();
        bucket_ids.clear();
        for &(m, b) in chunk {
            minimizers.push(m);
            bucket_ids.push(b);
        }
        writer.write_batch(&minimizers, &bucket_ids)?;
    }

    writer.finish()
}

/// Helper struct for writing a single Parquet shard.
struct ShardWriter {
    writer: ArrowWriter<File>,
    schema: Arc<Schema>,
}

impl ShardWriter {
    fn new(path: &Path, options: &ParquetWriteOptions) -> Result<Self> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("minimizer", DataType::UInt64, false),
            Field::new("bucket_id", DataType::UInt32, false),
        ]));

        // DRY: Use ParquetWriteOptions::to_writer_properties() as single source of truth
        let props = options.to_writer_properties();

        let file =
            File::create(path).map_err(|e| RypeError::io(path.to_path_buf(), "create shard", e))?;
        let writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

        Ok(Self { writer, schema })
    }

    /// Return actual bytes written to the file so far.
    fn bytes_written(&self) -> usize {
        self.writer.bytes_written()
    }

    fn write_batch(&mut self, minimizers: &[u64], bucket_ids: &[u32]) -> Result<()> {
        let minimizer_array: ArrayRef =
            Arc::new(arrow::array::UInt64Array::from(minimizers.to_vec()));
        let bucket_id_array: ArrayRef =
            Arc::new(arrow::array::UInt32Array::from(bucket_ids.to_vec()));

        let batch =
            RecordBatch::try_new(self.schema.clone(), vec![minimizer_array, bucket_id_array])?;
        self.writer.write(&batch)?;
        Ok(())
    }

    fn finish(self) -> Result<()> {
        self.writer.close()?;
        Ok(())
    }
}

/// Streaming reader that yields `(minimizer, bucket_id)` pairs from a Parquet shard
/// one batch at a time.
///
/// Unlike `read_shard_pairs` which materializes the entire shard into a `Vec`,
/// this reader holds at most one `RecordBatch` in memory
/// (~`PARQUET_BATCH_SIZE × 12 B`). It is intended for streaming k-way merges
/// where multiple shards are iterated in parallel.
struct StreamingShardReader {
    iter: parquet::arrow::arrow_reader::ParquetRecordBatchReader,
    current: Option<RecordBatch>,
    row_idx: usize,
    #[cfg(debug_assertions)]
    last_pair: Option<(u64, u32)>,
}

impl StreamingShardReader {
    fn open(path: &Path) -> Result<Self> {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let file = std::fs::File::open(path)
            .map_err(|e| RypeError::io(path.to_path_buf(), "open shard", e))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let iter = builder.build()?;
        Ok(Self {
            iter,
            current: None,
            row_idx: 0,
            #[cfg(debug_assertions)]
            last_pair: None,
        })
    }

    fn next_pair(&mut self) -> Result<Option<(u64, u32)>> {
        use arrow::array::{UInt32Array, UInt64Array};

        loop {
            if let Some(batch) = self.current.as_ref() {
                if self.row_idx < batch.num_rows() {
                    let minimizers = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<UInt64Array>()
                        .ok_or_else(|| {
                            RypeError::validation("Expected UInt64Array for minimizer column")
                        })?;
                    let bucket_ids = batch
                        .column(1)
                        .as_any()
                        .downcast_ref::<UInt32Array>()
                        .ok_or_else(|| {
                            RypeError::validation("Expected UInt32Array for bucket_id column")
                        })?;
                    let pair = (
                        minimizers.value(self.row_idx),
                        bucket_ids.value(self.row_idx),
                    );
                    self.row_idx += 1;
                    #[cfg(debug_assertions)]
                    {
                        // Sort order of a shard is on `minimizer` only; within a
                        // minimizer the bucket_ids may appear in any order.
                        if let Some(prev) = self.last_pair {
                            debug_assert!(
                                prev.0 <= pair.0,
                                "shard minimizers not monotone: {} followed by {}",
                                prev.0,
                                pair.0
                            );
                        }
                        self.last_pair = Some(pair);
                    }
                    return Ok(Some(pair));
                }
                // Current batch exhausted.
                self.current = None;
                self.row_idx = 0;
            }

            match self.iter.next() {
                Some(batch_result) => {
                    let batch = batch_result?;
                    if batch.num_rows() == 0 {
                        continue;
                    }
                    self.current = Some(batch);
                    self.row_idx = 0;
                }
                None => return Ok(None),
            }
        }
    }
}

/// Rename final shard files from `shard.{offset+i}.parquet` → `shard.{i}.parquet`
/// and rewrite the `shard_id` in each `InvertedShardInfo` to match.
///
/// Used by streaming consolidation, which writes finals at an offset to avoid
/// colliding with still-alive intermediate file paths.
fn rename_final_shards(
    output_dir: &Path,
    shard_infos: &mut [InvertedShardInfo],
    offset: u32,
) -> Result<()> {
    if offset == 0 {
        return Ok(());
    }
    let inverted_dir = output_dir.join(files::INVERTED_DIR);
    for info in shard_infos.iter_mut() {
        let old_id = info.shard_id;
        let new_id = old_id
            .checked_sub(offset)
            .expect("final shard id below offset; streaming consolidation invariant violated");
        let old_path = inverted_dir.join(files::inverted_shard(old_id));
        let new_path = inverted_dir.join(files::inverted_shard(new_id));

        // Refuse to clobber an existing file. If we get here and new_path
        // already exists, Phase 4 (intermediate deletion) did not complete —
        // likely a retry after a partial crash. `fs::rename` is atomic on
        // POSIX and would destroy the surviving file without warning.
        if new_path.exists() {
            return Err(RypeError::validation(format!(
                "rename final shard {}: destination {} already exists \
                 (index directory is in an inconsistent state; rebuild required)",
                new_id,
                new_path.display(),
            )));
        }

        std::fs::rename(&old_path, &new_path)
            .map_err(|e| RypeError::io(new_path, "rename final shard", e))?;
        info.shard_id = new_id;
    }
    Ok(())
}

/// Streaming consolidation of intermediate shards into final deduplicated,
/// non-overlapping shards via a k-way merge.
///
/// Unlike [`consolidate_shards`] (the legacy implementation), this function
/// streams `(minimizer, bucket_id)` pairs through a `BinaryHeap` and pushes
/// them directly into a `ShardAccumulator`. Peak memory is
/// `O(num_intermediate_shards × PARQUET_BATCH_SIZE × 12 B + max_shard_bytes)`,
/// independent of the total unique minimizer count.
///
/// # Shard-id handling
///
/// Intermediate shards live at `shard.{0..N-1}.parquet` and must stay on disk
/// until the merge completes. The accumulator therefore starts at
/// `start_shard_id = N`; once the merge is done and intermediates are deleted,
/// final files are renamed back to `shard.{0..M-1}.parquet`.
pub fn consolidate_shards_streaming(
    output_dir: &Path,
    intermediate_shards: &[InvertedShardInfo],
    bucket_id: u32,
    max_shard_bytes: usize,
    options: &ParquetWriteOptions,
) -> Result<(Vec<InvertedShardInfo>, u64)> {
    if intermediate_shards.is_empty() {
        return Ok((vec![], 0));
    }

    let inverted_dir = output_dir.join(files::INVERTED_DIR);
    let n = intermediate_shards.len();
    let offset: u32 = n
        .try_into()
        .map_err(|_| RypeError::validation("too many intermediate shards for u32 offset"))?;

    let input_total: u64 = intermediate_shards.iter().map(|s| s.num_entries).sum();
    let input_min = intermediate_shards
        .iter()
        .map(|s| s.min_minimizer)
        .min()
        .unwrap_or(0);
    let input_max = intermediate_shards
        .iter()
        .map(|s| s.max_minimizer)
        .max()
        .unwrap_or(0);
    log::info!(
        "Streaming consolidation: {} intermediate shards, {} total entries \
         (range [{}, {}]), shard budget {} MiB",
        n,
        input_total,
        input_min,
        input_max,
        max_shard_bytes / (1024 * 1024),
    );

    // Phase 1: open streaming readers; seed heap with first pair of each.
    let mut readers: Vec<StreamingShardReader> = Vec::with_capacity(n);
    let mut heap: BinaryHeap<Reverse<(u64, u32, usize)>> = BinaryHeap::new();
    for (idx, info) in intermediate_shards.iter().enumerate() {
        let path = inverted_dir.join(files::inverted_shard(info.shard_id));
        let mut reader = StreamingShardReader::open(&path)?;
        if let Some((m, b)) = reader.next_pair()? {
            heap.push(Reverse((m, b, idx)));
        }
        readers.push(reader);
    }
    log::info!("  Opened {} streaming shard readers", n);

    // Phase 2: k-way merge → write via ShardAccumulator at offset shard ids.
    let mut accumulator =
        ShardAccumulator::with_start_shard_id(output_dir, max_shard_bytes, offset, Some(options));

    /// How often (in unique pairs merged) to emit a progress log. Tuned to
    /// one log line per ~1-5s at typical merge throughput.
    const MERGE_PROGRESS_INTERVAL: u64 = 50_000_000;

    let mut last_written: Option<(u64, u32)> = None;
    let mut total_unique: u64 = 0;
    let mut total_popped: u64 = 0;
    let t_merge = std::time::Instant::now();

    while let Some(Reverse((m, b, idx))) = heap.pop() {
        total_popped += 1;
        if last_written != Some((m, b)) {
            // Push a single pair directly — ShardAccumulator's internal Vec is
            // already the canonical batching buffer; a second pending Vec here
            // just duplicates that layer without benefit.
            accumulator.add_entries(std::slice::from_ref(&(m, b)));
            last_written = Some((m, b));
            total_unique += 1;
            while accumulator.should_flush() {
                if let Some(info) = accumulator.flush_shard()? {
                    log::info!(
                        "  Flushed final shard #{} ({} entries, range [{}, {}])",
                        info.shard_id,
                        info.num_entries,
                        info.min_minimizer,
                        info.max_minimizer
                    );
                }
            }
            if total_unique % MERGE_PROGRESS_INTERVAL == 0 {
                let elapsed = t_merge.elapsed();
                let rate = total_popped as f64 / elapsed.as_secs_f64().max(0.001);
                log::info!(
                    "  Merge progress: {} unique / {} popped ({:.1}% dedup, {:.1}M pairs/s)",
                    total_unique,
                    total_popped,
                    100.0 * (1.0 - total_unique as f64 / total_popped.max(1) as f64),
                    rate / 1_000_000.0,
                );
            }
        }
        if let Some(next) = readers[idx].next_pair()? {
            heap.push(Reverse((next.0, next.1, idx)));
        }
    }

    let merge_elapsed = t_merge.elapsed();
    log::info!(
        "  Merge phase complete: {} unique pairs from {} popped in {:.2}s ({:.1}M pairs/s input)",
        total_unique,
        total_popped,
        merge_elapsed.as_secs_f64(),
        total_popped as f64 / merge_elapsed.as_secs_f64().max(0.001) / 1_000_000.0,
    );

    // accumulator.finish() flushes the trailing partial shard; the per-flush
    // log above covers every shard produced inside the loop, and the summary
    // "Consolidation complete" log below covers the final shard count.
    let mut offset_shard_infos = accumulator.finish()?;

    // Phase 3: drop readers (close files) before touching intermediate paths.
    // On Unix, deleting a file with open descriptors is legal; on Windows it
    // can fail with a sharing violation. Dropping the readers explicitly here
    // makes the ordering portable rather than relying on end-of-scope drop.
    drop(readers);

    // Phase 4: delete intermediates.
    // ORDERING DEPENDENCY: all final writes above must be complete before any
    // intermediate is removed; all intermediates must be gone before renames
    // below, otherwise a rename could collide with an intermediate file name.
    for info in intermediate_shards {
        let path = inverted_dir.join(files::inverted_shard(info.shard_id));
        std::fs::remove_file(&path)
            .map_err(|e| RypeError::io(path, "delete intermediate shard", e))?;
    }
    log::info!("  Deleted {} intermediate shard files", n);

    // Phase 5: rename shard.{offset+i}.parquet → shard.{i}.parquet.
    rename_final_shards(output_dir, &mut offset_shard_infos, offset)?;
    log::info!(
        "  Renamed {} final shards to canonical IDs (0..{})",
        offset_shard_infos.len(),
        offset_shard_infos.len().saturating_sub(1),
    );

    log::info!(
        "Consolidation complete (streaming): {} final shards, {} unique entries, bucket_id={}",
        offset_shard_infos.len(),
        total_unique,
        bucket_id,
    );

    debug_assert!(
        offset_shard_infos
            .windows(2)
            .all(|w| w[0].max_minimizer < w[1].min_minimizer),
        "streaming consolidation produced overlapping shards"
    );
    debug_assert!(
        offset_shard_infos
            .iter()
            .enumerate()
            .all(|(i, s)| s.shard_id as usize == i),
        "streaming consolidation produced non-contiguous shard ids"
    );

    Ok((offset_shard_infos, total_unique))
}

/// Consolidate intermediate shards into final deduplicated, non-overlapping shards.
///
/// Thin wrapper around [`consolidate_shards_streaming`]. Peak memory is
/// `O(num_intermediate_shards × PARQUET_BATCH_SIZE × 12 B + max_shard_bytes)`,
/// independent of the total unique minimizer count.
///
/// This is only needed for single-bucket streaming builds where the same
/// minimizer can appear in multiple chunks (and thus multiple intermediate shards).
pub fn consolidate_shards(
    output_dir: &Path,
    intermediate_shards: &[InvertedShardInfo],
    bucket_id: u32,
    max_shard_bytes: usize,
    options: &ParquetWriteOptions,
) -> Result<(Vec<InvertedShardInfo>, u64)> {
    consolidate_shards_streaming(
        output_dir,
        intermediate_shards,
        bucket_id,
        max_shard_bytes,
        options,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indices::parquet::manifest::is_parquet_index;
    use tempfile::TempDir;

    /// Test that streaming Parquet creation produces correct classification results.
    #[test]
    fn test_streaming_parquet_classification() {
        use crate::classify::classify_batch_sharded_merge_join;
        use crate::core::extraction::extract_into;
        use crate::core::workspace::MinimizerWorkspace;
        use crate::indices::sharded::ShardedInvertedIndex;
        use crate::types::QueryRecord;

        let tmp = TempDir::new().unwrap();
        let streaming_dir = tmp.path().join("streaming.ryxdi");

        // Test sequences - long enough for K=32, W=10
        let seq1 = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let seq2 = b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA";
        let seq3 = b"AAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTT";

        let k = 32;
        let w = 10;
        let salt = 0x1234u64;

        // === Create streaming Parquet index ===
        let mut ws = MinimizerWorkspace::new();

        // Extract minimizers for each bucket
        extract_into(seq1, k, w, salt, &mut ws);
        let mut mins1 = std::mem::take(&mut ws.buffer);
        mins1.sort_unstable();
        mins1.dedup();

        extract_into(seq2, k, w, salt, &mut ws);
        let mut mins2 = std::mem::take(&mut ws.buffer);
        mins2.sort_unstable();
        mins2.dedup();

        extract_into(seq3, k, w, salt, &mut ws);
        let mut mins3 = std::mem::take(&mut ws.buffer);
        mins3.sort_unstable();
        mins3.dedup();

        let buckets = vec![
            BucketData {
                bucket_id: 1,
                bucket_name: "bucket1".to_string(),
                sources: vec!["ref1.fa".to_string()],
                minimizers: mins1.clone(),
            },
            BucketData {
                bucket_id: 2,
                bucket_name: "bucket2".to_string(),
                sources: vec!["ref2.fa".to_string()],
                minimizers: mins2.clone(),
            },
            BucketData {
                bucket_id: 3,
                bucket_name: "bucket3".to_string(),
                sources: vec!["ref3.fa".to_string()],
                minimizers: mins3.clone(),
            },
        ];

        let _streaming_manifest =
            create_parquet_inverted_index(&streaming_dir, buckets, k, w, salt, None, None, None)
                .unwrap();

        // Open the created index
        let streaming_sharded = ShardedInvertedIndex::open(&streaming_dir).unwrap();

        // Query sequences (using the same sequences we indexed)
        let query1: &[u8] = seq1;
        let query2: &[u8] = seq2;
        let query3: &[u8] = seq3;
        let records: Vec<QueryRecord> =
            vec![(0, query1, None), (1, query2, None), (2, query3, None)];

        let threshold = 0.1;

        let results =
            classify_batch_sharded_merge_join(&streaming_sharded, None, &records, threshold, None)
                .unwrap();

        // Verify we got expected matches (each query should match its own bucket perfectly)
        assert!(
            results.iter().any(|r| r.query_id == 0 && r.bucket_id == 1),
            "Query 0 should match bucket 1"
        );
        assert!(
            results.iter().any(|r| r.query_id == 1 && r.bucket_id == 2),
            "Query 1 should match bucket 2"
        );
        assert!(
            results.iter().any(|r| r.query_id == 2 && r.bucket_id == 3),
            "Query 2 should match bucket 3"
        );

        // Verify perfect matches have high scores (self-match should be ~1.0)
        let bucket1_match = results.iter().find(|r| r.query_id == 0 && r.bucket_id == 1);
        assert!(
            bucket1_match.is_some() && bucket1_match.unwrap().score > 0.9,
            "Query 0 should have high score for bucket 1"
        );
    }

    #[test]
    fn test_parquet_write_with_zstd() {
        use crate::core::extraction::extract_into;
        use crate::core::workspace::MinimizerWorkspace;
        use crate::indices::parquet::options::ParquetCompression;

        let tmp = TempDir::new().unwrap();
        let index_dir = tmp.path().join("zstd_test.ryxdi");

        // Create test data
        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let k = 32;
        let w = 10;
        let salt = 0x1234u64;

        let mut ws = MinimizerWorkspace::new();
        extract_into(seq, k, w, salt, &mut ws);
        let mut mins = std::mem::take(&mut ws.buffer);
        mins.sort_unstable();
        mins.dedup();

        let buckets = vec![BucketData {
            bucket_id: 1,
            bucket_name: "test".to_string(),
            sources: vec!["test.fa".to_string()],
            minimizers: mins,
        }];

        // Create with zstd compression
        let options = ParquetWriteOptions {
            compression: ParquetCompression::Zstd,
            ..Default::default()
        };

        let manifest = create_parquet_inverted_index(
            &index_dir,
            buckets.clone(),
            k,
            w,
            salt,
            None,
            Some(&options),
            None,
        )
        .unwrap();

        // Verify we can read it back (data integrity)
        assert!(manifest.inverted.as_ref().unwrap().num_shards > 0);
        assert!(is_parquet_index(&index_dir));
    }

    #[test]
    fn test_parquet_write_with_bloom_filter() {
        use crate::core::extraction::extract_into;
        use crate::core::workspace::MinimizerWorkspace;
        use parquet::file::reader::FileReader;

        let tmp = TempDir::new().unwrap();
        let index_dir = tmp.path().join("bloom_test.ryxdi");

        // Create test data
        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let k = 32;
        let w = 10;
        let salt = 0x1234u64;

        let mut ws = MinimizerWorkspace::new();
        extract_into(seq, k, w, salt, &mut ws);
        let mut mins = std::mem::take(&mut ws.buffer);
        mins.sort_unstable();
        mins.dedup();

        let buckets = vec![BucketData {
            bucket_id: 1,
            bucket_name: "test".to_string(),
            sources: vec!["test.fa".to_string()],
            minimizers: mins,
        }];

        // Create with bloom filter enabled
        let options = ParquetWriteOptions {
            bloom_filter_enabled: true,
            bloom_filter_fpp: 0.01,
            ..Default::default()
        };

        let manifest = create_parquet_inverted_index(
            &index_dir,
            buckets.clone(),
            k,
            w,
            salt,
            None,
            Some(&options),
            None,
        )
        .unwrap();

        assert!(manifest.inverted.as_ref().unwrap().num_shards > 0);

        // Verify bloom filter metadata exists in the parquet file
        // We check this by reading the file metadata
        let shard_path = index_dir.join("inverted").join("shard.0.parquet");
        let file = std::fs::File::open(&shard_path).unwrap();
        let reader = parquet::file::reader::SerializedFileReader::new(file).unwrap();
        let metadata = reader.metadata();

        // Check that at least one row group has bloom filter metadata
        let mut has_bloom_filter = false;
        for rg in 0..metadata.num_row_groups() {
            let rg_meta = metadata.row_group(rg);
            for col in 0..rg_meta.num_columns() {
                if rg_meta.column(col).bloom_filter_offset().is_some() {
                    has_bloom_filter = true;
                    break;
                }
            }
        }
        assert!(
            has_bloom_filter,
            "Bloom filter should be present in Parquet file"
        );
    }

    #[test]
    fn test_sampling_is_deterministic() {
        // rand_sample should return the same result for the same minimizer
        let rate = 0.5;
        for m in [0u64, 1, 100, 1000, u64::MAX, 0xDEADBEEF] {
            let result1 = rand_sample(m, rate);
            let result2 = rand_sample(m, rate);
            assert_eq!(result1, result2, "rand_sample should be deterministic");
        }
    }

    // ========== Phase 1: ShardAccumulator Tests ==========

    #[test]
    fn test_shard_accumulator_creation() {
        let max_bytes = MIN_SHARD_BYTES; // Use minimum valid size
        let accumulator = ShardAccumulator::new(max_bytes);

        assert_eq!(accumulator.max_shard_bytes(), max_bytes);
        assert_eq!(accumulator.current_size_bytes(), 0);
        assert_eq!(accumulator.entry_count(), 0);
    }

    #[test]
    fn test_shard_accumulator_size_tracking() {
        let max_bytes = MIN_SHARD_BYTES;
        let mut accumulator = ShardAccumulator::new(max_bytes);

        // Add some entries
        let entries: Vec<(u64, u32)> = vec![(100, 1), (200, 2), (300, 1), (400, 3)];
        accumulator.add_entries(&entries);

        // Size is based on capacity, not length (accounts for Vec over-allocation)
        assert_eq!(accumulator.entry_count(), 4);
        let min_size = 4 * ShardAccumulator::BYTES_PER_ENTRY;
        assert!(
            accumulator.current_size_bytes() >= min_size,
            "current_size_bytes ({}) should be >= entry_count * BYTES_PER_ENTRY ({})",
            accumulator.current_size_bytes(),
            min_size
        );

        // Add more entries
        let more_entries: Vec<(u64, u32)> = vec![(500, 2), (600, 1)];
        accumulator.add_entries(&more_entries);

        assert_eq!(accumulator.entry_count(), 6);
        let min_size = 6 * ShardAccumulator::BYTES_PER_ENTRY;
        assert!(
            accumulator.current_size_bytes() >= min_size,
            "current_size_bytes ({}) should be >= entry_count * BYTES_PER_ENTRY ({})",
            accumulator.current_size_bytes(),
            min_size
        );
    }

    #[test]
    fn test_shard_accumulator_should_flush() {
        // Use minimum valid size for testing
        let max_bytes = MIN_SHARD_BYTES; // 1MB
        let mut accumulator = ShardAccumulator::new(max_bytes);

        // Initially should not need flush
        assert!(!accumulator.should_flush());

        // Add entries that are under threshold
        let entries: Vec<(u64, u32)> = vec![(100, 1), (200, 2), (300, 1)];
        accumulator.add_entries(&entries);
        assert!(!accumulator.should_flush());

        // Add enough entries to exceed MIN_SHARD_BYTES (1MB / 16 bytes = 65536 entries)
        let entries_needed = MIN_SHARD_BYTES / ShardAccumulator::BYTES_PER_ENTRY;
        let many_entries: Vec<(u64, u32)> = (0..entries_needed as u64)
            .map(|i| (i * 1000, (i % 10) as u32))
            .collect();
        accumulator.add_entries(&many_entries);
        assert!(accumulator.should_flush());
    }

    // ========== Phase 2: ShardAccumulator Flush Tests ==========

    #[test]
    fn test_shard_accumulator_flush_sorts_entries() {
        let tmp = TempDir::new().unwrap();
        let output_dir = tmp.path().join("test.ryxdi");
        std::fs::create_dir_all(output_dir.join("inverted")).unwrap();

        let max_bytes = MIN_SHARD_BYTES;
        let mut accumulator = ShardAccumulator::with_output_dir(&output_dir, max_bytes, None);

        // Add entries in unsorted order
        let entries: Vec<(u64, u32)> = vec![
            (500, 2),
            (100, 1),
            (300, 3),
            (100, 2), // Same minimizer, different bucket
            (200, 1),
        ];
        accumulator.add_entries(&entries);

        // Flush and verify entries are sorted by (minimizer, bucket_id)
        let shard_info = accumulator
            .flush_shard()
            .unwrap()
            .expect("should have flushed");

        // Verify shard info has correct min/max
        assert_eq!(shard_info.min_minimizer, 100);
        assert_eq!(shard_info.max_minimizer, 500);
        assert_eq!(shard_info.num_entries, 5);

        // Read back the parquet file and verify sorted order
        let shard_path = output_dir.join("inverted").join("shard.0.parquet");
        let pairs = read_shard_pairs(&shard_path).unwrap();

        let expected: Vec<(u64, u32)> = vec![(100, 1), (100, 2), (200, 1), (300, 3), (500, 2)];
        assert_eq!(
            pairs, expected,
            "Entries should be sorted by (minimizer, bucket_id)"
        );
    }

    #[test]
    fn test_flush_shard_deduplicates_entries() {
        let tmp = TempDir::new().unwrap();
        let output_dir = tmp.path().join("test.ryxdi");
        std::fs::create_dir_all(output_dir.join("inverted")).unwrap();

        let mut accumulator = ShardAccumulator::with_output_dir(&output_dir, MIN_SHARD_BYTES, None);

        // Add entries with duplicates (same (minimizer, bucket_id) pairs)
        let entries: Vec<(u64, u32)> = vec![
            (100, 1),
            (200, 1),
            (100, 1), // duplicate of (100,1)
            (300, 1),
            (200, 1), // duplicate of (200,1)
            (100, 2), // different bucket_id — NOT a duplicate
            (100, 2), // duplicate of (100,2)
        ];
        accumulator.add_entries(&entries);

        let shard_info = accumulator
            .flush_shard()
            .unwrap()
            .expect("should have flushed");

        // After dedup: (100,1), (100,2), (200,1), (300,1) = 4 unique entries
        assert_eq!(
            shard_info.num_entries, 4,
            "num_entries should reflect deduplicated count, got {}",
            shard_info.num_entries
        );

        // Verify the actual file contents
        let shard_path = output_dir.join("inverted").join("shard.0.parquet");
        let pairs = read_shard_pairs(&shard_path).unwrap();
        assert_eq!(pairs, vec![(100, 1), (100, 2), (200, 1), (300, 1)]);
    }

    #[test]
    fn test_shard_accumulator_flush_writes_parquet() {
        let tmp = TempDir::new().unwrap();
        let output_dir = tmp.path().join("test.ryxdi");
        std::fs::create_dir_all(output_dir.join("inverted")).unwrap();

        let max_bytes = MIN_SHARD_BYTES;
        let mut accumulator = ShardAccumulator::with_output_dir(&output_dir, max_bytes, None);

        let entries: Vec<(u64, u32)> = vec![(100, 1), (200, 2), (300, 1)];
        accumulator.add_entries(&entries);

        let shard_info = accumulator
            .flush_shard()
            .unwrap()
            .expect("should have flushed");

        // Verify parquet file was created
        let shard_path = output_dir.join("inverted").join("shard.0.parquet");
        assert!(shard_path.exists(), "Shard parquet file should exist");

        // Verify shard info
        assert_eq!(shard_info.shard_id, 0);
        assert_eq!(shard_info.num_entries, 3);
    }

    #[test]
    fn test_shard_accumulator_flush_clears_buffer() {
        let tmp = TempDir::new().unwrap();
        let output_dir = tmp.path().join("test.ryxdi");
        std::fs::create_dir_all(output_dir.join("inverted")).unwrap();

        let max_bytes = MIN_SHARD_BYTES;
        let mut accumulator = ShardAccumulator::with_output_dir(&output_dir, max_bytes, None);

        let entries: Vec<(u64, u32)> = vec![(100, 1), (200, 2)];
        accumulator.add_entries(&entries);
        assert_eq!(accumulator.entry_count(), 2);

        accumulator.flush_shard().unwrap();

        // Buffer should be cleared after flush
        assert_eq!(accumulator.entry_count(), 0);
        assert_eq!(accumulator.current_size_bytes(), 0);
    }

    #[test]
    fn test_shard_accumulator_increments_shard_id() {
        let tmp = TempDir::new().unwrap();
        let output_dir = tmp.path().join("test.ryxdi");
        std::fs::create_dir_all(output_dir.join("inverted")).unwrap();

        let max_bytes = MIN_SHARD_BYTES;
        let mut accumulator = ShardAccumulator::with_output_dir(&output_dir, max_bytes, None);

        // First flush
        accumulator.add_entries(&[(100, 1), (200, 2)]);
        let info1 = accumulator.flush_shard().unwrap().expect("should flush");
        assert_eq!(info1.shard_id, 0);

        // Second flush
        accumulator.add_entries(&[(300, 1), (400, 2)]);
        let info2 = accumulator.flush_shard().unwrap().expect("should flush");
        assert_eq!(info2.shard_id, 1);

        // Third flush
        accumulator.add_entries(&[(500, 1)]);
        let info3 = accumulator.flush_shard().unwrap().expect("should flush");
        assert_eq!(info3.shard_id, 2);

        // Verify all shard files exist
        assert!(output_dir.join("inverted/shard.0.parquet").exists());
        assert!(output_dir.join("inverted/shard.1.parquet").exists());
        assert!(output_dir.join("inverted/shard.2.parquet").exists());
    }

    // ========== Phase 3: Failure/Edge Case Tests ==========

    #[test]
    fn test_shard_accumulator_empty_flush_returns_none() {
        let tmp = TempDir::new().unwrap();
        let output_dir = tmp.path().join("test.ryxdi");
        std::fs::create_dir_all(output_dir.join("inverted")).unwrap();

        let mut accumulator = ShardAccumulator::with_output_dir(&output_dir, MIN_SHARD_BYTES, None);

        // Flushing empty accumulator should return Ok(None), not an error
        let result = accumulator.flush_shard().unwrap();
        assert!(result.is_none(), "Empty flush should return None");

        // No shard file should be created
        assert!(!output_dir.join("inverted/shard.0.parquet").exists());
    }

    #[test]
    fn test_shard_accumulator_finish_empty_returns_empty_vec() {
        let tmp = TempDir::new().unwrap();
        let output_dir = tmp.path().join("test.ryxdi");
        std::fs::create_dir_all(output_dir.join("inverted")).unwrap();

        let accumulator = ShardAccumulator::with_output_dir(&output_dir, MIN_SHARD_BYTES, None);

        // Finishing empty accumulator should return empty vec
        let shard_infos = accumulator.finish().unwrap();
        assert!(
            shard_infos.is_empty(),
            "finish() on empty should return empty vec"
        );
    }

    #[test]
    fn test_shard_accumulator_flush_without_output_dir_errors() {
        let mut accumulator = ShardAccumulator::new(MIN_SHARD_BYTES);
        accumulator.add_entries(&[(100, 1), (200, 2)]);

        // Flushing without output_dir should error
        let result = accumulator.flush_shard();
        assert!(result.is_err(), "flush without output_dir should error");

        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("no output directory"),
            "Error should mention missing output directory: {}",
            err_msg
        );
    }

    #[test]
    #[should_panic(expected = "must be at least MIN_SHARD_BYTES")]
    fn test_shard_accumulator_rejects_zero_max_bytes() {
        let _ = ShardAccumulator::new(0);
    }

    #[test]
    #[should_panic(expected = "must be at least MIN_SHARD_BYTES")]
    fn test_shard_accumulator_rejects_small_max_bytes() {
        let _ = ShardAccumulator::new(1024); // 1KB is too small
    }

    #[test]
    fn test_shard_accumulator_accepts_min_shard_bytes() {
        // Should not panic
        let accumulator = ShardAccumulator::new(MIN_SHARD_BYTES);
        assert_eq!(accumulator.max_shard_bytes(), MIN_SHARD_BYTES);
    }

    /// Helper to read (minimizer, bucket_id) pairs from a shard parquet file.
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
                .unwrap();
            let bucket_ids = batch
                .column(1)
                .as_any()
                .downcast_ref::<UInt32Array>()
                .unwrap();

            for i in 0..batch.num_rows() {
                pairs.push((minimizers.value(i), bucket_ids.value(i)));
            }
        }
        Ok(pairs)
    }

    // ========== StreamingShardReader Tests (Cycle 1) ==========

    #[test]
    fn test_streaming_shard_reader_yields_pairs_in_order() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("shard.parquet");
        let opts = ParquetWriteOptions::default();

        let pairs: Vec<(u64, u32)> = vec![
            (10, 1),
            (20, 1),
            (30, 1),
            (40, 2),
            (50, 1),
            (60, 3),
            (70, 1),
        ];
        write_shard_from_pairs(&path, &pairs, &opts).unwrap();

        let mut reader = StreamingShardReader::open(&path).unwrap();
        let mut collected: Vec<(u64, u32)> = Vec::new();
        while let Some(pair) = reader.next_pair().unwrap() {
            collected.push(pair);
        }
        assert_eq!(collected, pairs);

        assert!(reader.next_pair().unwrap().is_none());
    }

    /// A shard can legitimately contain the same minimizer with *different*
    /// bucket_ids in strictly-ascending minimizer order; the ordering
    /// invariant the reader tracks is on minimizers alone, not on the full
    /// (u64, u32) tuple. (Today's consolidation path is single-bucket, but
    /// the reader is a reusable primitive and must not false-positive.)
    #[test]
    fn test_streaming_shard_reader_debug_assert_ignores_bucket_id_order() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("shard.parquet");
        let opts = ParquetWriteOptions::default();

        // Minimizers are monotone non-decreasing. Within a minimizer, bucket
        // ids are NOT required to be sorted.
        let pairs: Vec<(u64, u32)> = vec![(10, 5), (10, 3), (10, 9), (20, 1)];
        write_shard_from_pairs(&path, &pairs, &opts).unwrap();

        let mut reader = StreamingShardReader::open(&path).unwrap();
        let mut collected: Vec<(u64, u32)> = Vec::new();
        while let Some(pair) = reader.next_pair().unwrap() {
            collected.push(pair);
        }
        // Must not panic in debug; must yield inputs in stored order.
        assert_eq!(collected, pairs);
    }

    #[test]
    fn test_streaming_shard_reader_empty_shard() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("shard.parquet");
        let opts = ParquetWriteOptions::default();

        write_shard_from_pairs(&path, &[], &opts).unwrap();

        let mut reader = StreamingShardReader::open(&path).unwrap();
        assert!(reader.next_pair().unwrap().is_none());
        assert!(reader.next_pair().unwrap().is_none());
    }

    // ========== Streaming consolidation (Cycle 2) ==========

    /// Build the same 3-overlapping-shard input used by the legacy test, then
    /// assert that the streaming implementation produces byte-identical output.
    #[test]
    fn test_consolidate_shards_streaming_matches_legacy() {
        let opts = ParquetWriteOptions::default();

        let intermediate_shards = vec![
            InvertedShardInfo {
                shard_id: 0,
                min_minimizer: 10,
                max_minimizer: 30,
                num_entries: 3,
            },
            InvertedShardInfo {
                shard_id: 1,
                min_minimizer: 20,
                max_minimizer: 40,
                num_entries: 3,
            },
            InvertedShardInfo {
                shard_id: 2,
                min_minimizer: 30,
                max_minimizer: 50,
                num_entries: 3,
            },
        ];
        let shard_contents: [&[(u64, u32)]; 3] = [
            &[(10, 1), (20, 1), (30, 1)],
            &[(20, 1), (30, 1), (40, 1)],
            &[(30, 1), (40, 1), (50, 1)],
        ];

        // Run legacy and streaming into two separate tempdirs.
        let legacy_tmp = TempDir::new().unwrap();
        let legacy_dir = legacy_tmp.path().join("legacy.ryxdi");
        std::fs::create_dir_all(legacy_dir.join("inverted")).unwrap();
        for (i, content) in shard_contents.iter().enumerate() {
            write_shard_from_pairs(
                &legacy_dir
                    .join("inverted")
                    .join(format!("shard.{}.parquet", i)),
                content,
                &opts,
            )
            .unwrap();
        }
        let (legacy_shards, legacy_unique) =
            consolidate_shards(&legacy_dir, &intermediate_shards, 1, MIN_SHARD_BYTES, &opts)
                .unwrap();

        let streaming_tmp = TempDir::new().unwrap();
        let streaming_dir = streaming_tmp.path().join("streaming.ryxdi");
        std::fs::create_dir_all(streaming_dir.join("inverted")).unwrap();
        for (i, content) in shard_contents.iter().enumerate() {
            write_shard_from_pairs(
                &streaming_dir
                    .join("inverted")
                    .join(format!("shard.{}.parquet", i)),
                content,
                &opts,
            )
            .unwrap();
        }
        let (streaming_shards, streaming_unique) = consolidate_shards_streaming(
            &streaming_dir,
            &intermediate_shards,
            1,
            MIN_SHARD_BYTES,
            &opts,
        )
        .unwrap();

        assert_eq!(legacy_unique, streaming_unique);
        assert_eq!(legacy_shards.len(), streaming_shards.len());
        for (l, s) in legacy_shards.iter().zip(streaming_shards.iter()) {
            assert_eq!(l.shard_id, s.shard_id);
            assert_eq!(l.min_minimizer, s.min_minimizer);
            assert_eq!(l.max_minimizer, s.max_minimizer);
            assert_eq!(l.num_entries, s.num_entries);

            let l_pairs = read_shard_pairs(
                &legacy_dir
                    .join("inverted")
                    .join(files::inverted_shard(l.shard_id)),
            )
            .unwrap();
            let s_pairs = read_shard_pairs(
                &streaming_dir
                    .join("inverted")
                    .join(files::inverted_shard(s.shard_id)),
            )
            .unwrap();
            assert_eq!(l_pairs, s_pairs);
        }
    }

    // ========== Streaming consolidation edge cases (Cycle 4) ==========

    #[test]
    fn test_consolidate_streaming_empty_intermediates() {
        let tmp = TempDir::new().unwrap();
        let output_dir = tmp.path().join("x.ryxdi");
        std::fs::create_dir_all(output_dir.join("inverted")).unwrap();

        let (shards, unique) = consolidate_shards_streaming(
            &output_dir,
            &[],
            1,
            MIN_SHARD_BYTES,
            &ParquetWriteOptions::default(),
        )
        .unwrap();
        assert!(shards.is_empty());
        assert_eq!(unique, 0);
    }

    #[test]
    fn test_consolidate_streaming_single_intermediate_passthrough() {
        let tmp = TempDir::new().unwrap();
        let output_dir = tmp.path().join("x.ryxdi");
        std::fs::create_dir_all(output_dir.join("inverted")).unwrap();
        let opts = ParquetWriteOptions::default();

        let input_pairs: Vec<(u64, u32)> = vec![(10, 1), (20, 1), (30, 1), (40, 1)];
        write_shard_from_pairs(
            &output_dir.join("inverted/shard.0.parquet"),
            &input_pairs,
            &opts,
        )
        .unwrap();
        let intermediate = vec![InvertedShardInfo {
            shard_id: 0,
            min_minimizer: 10,
            max_minimizer: 40,
            num_entries: 4,
        }];

        let (shards, unique) =
            consolidate_shards_streaming(&output_dir, &intermediate, 1, MIN_SHARD_BYTES, &opts)
                .unwrap();

        assert_eq!(unique, 4);
        assert_eq!(shards.len(), 1);
        // After rename, final file is at shard.0.parquet
        let out_pairs = read_shard_pairs(&output_dir.join("inverted/shard.0.parquet")).unwrap();
        assert_eq!(out_pairs, input_pairs);
    }

    #[test]
    fn test_consolidate_streaming_all_duplicates_across_shards() {
        let tmp = TempDir::new().unwrap();
        let output_dir = tmp.path().join("x.ryxdi");
        std::fs::create_dir_all(output_dir.join("inverted")).unwrap();
        let opts = ParquetWriteOptions::default();

        let pairs = vec![(100u64, 1u32), (200, 1)];
        let mut intermediate = Vec::new();
        for i in 0..5u32 {
            write_shard_from_pairs(
                &output_dir.join(format!("inverted/shard.{}.parquet", i)),
                &pairs,
                &opts,
            )
            .unwrap();
            intermediate.push(InvertedShardInfo {
                shard_id: i,
                min_minimizer: 100,
                max_minimizer: 200,
                num_entries: 2,
            });
        }

        let (shards, unique) =
            consolidate_shards_streaming(&output_dir, &intermediate, 1, MIN_SHARD_BYTES, &opts)
                .unwrap();

        assert_eq!(unique, 2);
        assert_eq!(shards.len(), 1);
        let out_pairs = read_shard_pairs(&output_dir.join("inverted/shard.0.parquet")).unwrap();
        assert_eq!(out_pairs, pairs);
    }

    #[test]
    fn test_consolidate_streaming_preserves_bucket_id_diversity() {
        // Dedup key must be (u64, u32), not u64 alone — same minimizer can appear
        // with multiple bucket_ids in a multi-bucket downstream path.
        let tmp = TempDir::new().unwrap();
        let output_dir = tmp.path().join("x.ryxdi");
        std::fs::create_dir_all(output_dir.join("inverted")).unwrap();
        let opts = ParquetWriteOptions::default();

        let shard_a: Vec<(u64, u32)> = vec![(100, 1), (100, 2), (200, 1)];
        let shard_b: Vec<(u64, u32)> = vec![(100, 2), (100, 3), (200, 2)];
        write_shard_from_pairs(
            &output_dir.join("inverted/shard.0.parquet"),
            &shard_a,
            &opts,
        )
        .unwrap();
        write_shard_from_pairs(
            &output_dir.join("inverted/shard.1.parquet"),
            &shard_b,
            &opts,
        )
        .unwrap();
        let intermediate = vec![
            InvertedShardInfo {
                shard_id: 0,
                min_minimizer: 100,
                max_minimizer: 200,
                num_entries: 3,
            },
            InvertedShardInfo {
                shard_id: 1,
                min_minimizer: 100,
                max_minimizer: 200,
                num_entries: 3,
            },
        ];

        let (shards, unique) =
            consolidate_shards_streaming(&output_dir, &intermediate, 1, MIN_SHARD_BYTES, &opts)
                .unwrap();
        // Unique (minimizer, bucket_id) pairs: (100,1), (100,2), (100,3), (200,1), (200,2) = 5
        assert_eq!(unique, 5);
        let mut all_pairs: Vec<(u64, u32)> = Vec::new();
        for s in &shards {
            all_pairs.extend(
                read_shard_pairs(
                    &output_dir
                        .join("inverted")
                        .join(files::inverted_shard(s.shard_id)),
                )
                .unwrap(),
            );
        }
        assert_eq!(
            all_pairs,
            vec![(100, 1), (100, 2), (100, 3), (200, 1), (200, 2)]
        );
    }

    #[test]
    fn test_consolidate_streaming_skewed_sizes() {
        let tmp = TempDir::new().unwrap();
        let output_dir = tmp.path().join("x.ryxdi");
        std::fs::create_dir_all(output_dir.join("inverted")).unwrap();
        let opts = ParquetWriteOptions::default();

        // 1 large shard (100_000 entries) + 3 tiny shards (10 entries each).
        // Disjoint minimizer ranges so cross-shard dedup doesn't kick in.
        let large: Vec<(u64, u32)> = (0..100_000u64).map(|i| (i, 1u32)).collect();
        let small_a: Vec<(u64, u32)> = (200_000..200_010u64).map(|i| (i, 1u32)).collect();
        let small_b: Vec<(u64, u32)> = (300_000..300_010u64).map(|i| (i, 1u32)).collect();
        let small_c: Vec<(u64, u32)> = (400_000..400_010u64).map(|i| (i, 1u32)).collect();

        write_shard_from_pairs(&output_dir.join("inverted/shard.0.parquet"), &large, &opts)
            .unwrap();
        write_shard_from_pairs(
            &output_dir.join("inverted/shard.1.parquet"),
            &small_a,
            &opts,
        )
        .unwrap();
        write_shard_from_pairs(
            &output_dir.join("inverted/shard.2.parquet"),
            &small_b,
            &opts,
        )
        .unwrap();
        write_shard_from_pairs(
            &output_dir.join("inverted/shard.3.parquet"),
            &small_c,
            &opts,
        )
        .unwrap();

        let intermediate = vec![
            InvertedShardInfo {
                shard_id: 0,
                min_minimizer: 0,
                max_minimizer: 99_999,
                num_entries: 100_000,
            },
            InvertedShardInfo {
                shard_id: 1,
                min_minimizer: 200_000,
                max_minimizer: 200_009,
                num_entries: 10,
            },
            InvertedShardInfo {
                shard_id: 2,
                min_minimizer: 300_000,
                max_minimizer: 300_009,
                num_entries: 10,
            },
            InvertedShardInfo {
                shard_id: 3,
                min_minimizer: 400_000,
                max_minimizer: 400_009,
                num_entries: 10,
            },
        ];

        let (shards, unique) =
            consolidate_shards_streaming(&output_dir, &intermediate, 1, MIN_SHARD_BYTES, &opts)
                .unwrap();
        assert_eq!(unique, 100_030);

        // shard_infos are range-partitioned contiguous from 0
        for (i, s) in shards.iter().enumerate() {
            assert_eq!(s.shard_id as usize, i);
        }
        for w in shards.windows(2) {
            assert!(w[0].max_minimizer < w[1].min_minimizer);
        }

        let total_entries: u64 = shards.iter().map(|s| s.num_entries).sum();
        assert_eq!(total_entries, unique);
    }

    /// Consolidation must produce more output shards than input when the
    /// deduplicated unique minimizer count exceeds what fits in a single
    /// final shard. Exercises the rename path with M > N and verifies
    /// final shard IDs are still contiguous 0..M-1.
    #[test]
    fn test_consolidate_streaming_more_output_shards_than_input() {
        let tmp = TempDir::new().unwrap();
        let output_dir = tmp.path().join("x.ryxdi");
        std::fs::create_dir_all(output_dir.join("inverted")).unwrap();
        let opts = ParquetWriteOptions::default();

        // 2 intermediate shards × 50_000 entries each = 100_000 unique entries
        // (disjoint ranges so no cross-shard dedup). At MIN_SHARD_BYTES = 1 MiB
        // and 16 B/entry, each final shard holds ~65k entries, so we expect
        // ~2 final shards — but the streaming accumulator's flush cadence can
        // yield 2-3 depending on alignment. Assert M >= N and contiguous IDs.
        let shard_a: Vec<(u64, u32)> = (0..50_000u64).map(|i| (i, 1)).collect();
        let shard_b: Vec<(u64, u32)> = (1_000_000..1_050_000u64).map(|i| (i, 1)).collect();
        write_shard_from_pairs(
            &output_dir.join("inverted/shard.0.parquet"),
            &shard_a,
            &opts,
        )
        .unwrap();
        write_shard_from_pairs(
            &output_dir.join("inverted/shard.1.parquet"),
            &shard_b,
            &opts,
        )
        .unwrap();

        let intermediate = vec![
            InvertedShardInfo {
                shard_id: 0,
                min_minimizer: 0,
                max_minimizer: 49_999,
                num_entries: 50_000,
            },
            InvertedShardInfo {
                shard_id: 1,
                min_minimizer: 1_000_000,
                max_minimizer: 1_049_999,
                num_entries: 50_000,
            },
        ];

        let (shards, unique) =
            consolidate_shards_streaming(&output_dir, &intermediate, 1, MIN_SHARD_BYTES, &opts)
                .unwrap();

        assert_eq!(unique, 100_000);
        assert!(
            shards.len() >= intermediate.len(),
            "expected M >= N (got M={}, N={})",
            shards.len(),
            intermediate.len()
        );

        // Contiguous IDs 0..M-1.
        for (i, s) in shards.iter().enumerate() {
            assert_eq!(s.shard_id as usize, i);
            let path = output_dir
                .join("inverted")
                .join(files::inverted_shard(s.shard_id));
            assert!(
                path.exists(),
                "final shard file must exist at contiguous id"
            );
        }

        // Range-partitioned (non-overlapping).
        for w in shards.windows(2) {
            assert!(w[0].max_minimizer < w[1].min_minimizer);
        }

        // No leftover intermediate file.
        assert!(
            !output_dir.join("inverted/shard.0.parquet").exists()
                || shards.iter().any(|s| s.shard_id == 0),
            "shard.0.parquet must either be a final shard or be absent"
        );
    }

    #[test]
    fn test_rename_final_shards_refuses_to_clobber_existing_file() {
        // If a previous consolidation attempt crashed mid-Phase-4 (delete
        // intermediates), some intermediates remain on disk at the ids that
        // the rename phase wants to move onto. The rename MUST refuse, not
        // silently overwrite — `fs::rename` on Linux is atomic and would
        // destroy the surviving intermediate's data otherwise.
        let tmp = TempDir::new().unwrap();
        let output_dir = tmp.path().join("x.ryxdi");
        std::fs::create_dir_all(output_dir.join("inverted")).unwrap();

        // Simulate: intermediate shard.0.parquet survived a partial crash, AND
        // a final shard at offset 3 wants to rename to shard.0.parquet.
        std::fs::write(output_dir.join("inverted/shard.0.parquet"), b"SURVIVOR").unwrap();
        std::fs::write(output_dir.join("inverted/shard.3.parquet"), b"FINAL").unwrap();

        let mut infos = vec![InvertedShardInfo {
            shard_id: 3,
            min_minimizer: 0,
            max_minimizer: 10,
            num_entries: 1,
        }];

        let err = rename_final_shards(&output_dir, &mut infos, 3).unwrap_err();
        // Survivor data must be untouched.
        let survivor = std::fs::read(output_dir.join("inverted/shard.0.parquet")).unwrap();
        assert_eq!(
            survivor, b"SURVIVOR",
            "rename must not clobber existing file"
        );
        let _ = err; // accept any RypeError variant; the invariant above is what matters.
    }

    #[test]
    fn test_rename_final_shards_adjusts_ids() {
        let tmp = TempDir::new().unwrap();
        let output_dir = tmp.path().join("x.ryxdi");
        std::fs::create_dir_all(output_dir.join("inverted")).unwrap();

        // Pre-populate offset-named shard files.
        for i in 3u32..6 {
            std::fs::File::create(output_dir.join(format!("inverted/shard.{}.parquet", i)))
                .unwrap();
        }
        let mut infos: Vec<InvertedShardInfo> = (3u32..6)
            .map(|i| InvertedShardInfo {
                shard_id: i,
                min_minimizer: i as u64 * 10,
                max_minimizer: i as u64 * 10 + 9,
                num_entries: 10,
            })
            .collect();

        rename_final_shards(&output_dir, &mut infos, 3).unwrap();

        for (i, info) in infos.iter().enumerate() {
            assert_eq!(info.shard_id as usize, i);
            assert!(output_dir
                .join(format!("inverted/shard.{}.parquet", i))
                .exists());
            assert!(!output_dir
                .join(format!("inverted/shard.{}.parquet", i + 3))
                .exists());
        }
    }

    // ========== Phase 4: Shard Consolidation Tests ==========

    #[test]
    fn test_consolidate_shards_removes_cross_shard_duplicates() {
        let tmp = TempDir::new().unwrap();
        let output_dir = tmp.path().join("test.ryxdi");
        std::fs::create_dir_all(output_dir.join("inverted")).unwrap();

        let opts = ParquetWriteOptions::default();

        // 3 intermediate shards with overlapping minimizer ranges
        let intermediate_shards = vec![
            InvertedShardInfo {
                shard_id: 0,
                min_minimizer: 10,
                max_minimizer: 30,
                num_entries: 3,
            },
            InvertedShardInfo {
                shard_id: 1,
                min_minimizer: 20,
                max_minimizer: 40,
                num_entries: 3,
            },
            InvertedShardInfo {
                shard_id: 2,
                min_minimizer: 30,
                max_minimizer: 50,
                num_entries: 3,
            },
        ];

        write_shard_from_pairs(
            &output_dir.join("inverted/shard.0.parquet"),
            &[(10, 1), (20, 1), (30, 1)],
            &opts,
        )
        .unwrap();
        write_shard_from_pairs(
            &output_dir.join("inverted/shard.1.parquet"),
            &[(20, 1), (30, 1), (40, 1)],
            &opts,
        )
        .unwrap();
        write_shard_from_pairs(
            &output_dir.join("inverted/shard.2.parquet"),
            &[(30, 1), (40, 1), (50, 1)],
            &opts,
        )
        .unwrap();

        let (new_shards, unique_count) =
            consolidate_shards(&output_dir, &intermediate_shards, 1, MIN_SHARD_BYTES, &opts)
                .unwrap();

        // Should have 5 unique minimizers: 10, 20, 30, 40, 50
        assert_eq!(unique_count, 5);

        let total_entries: u64 = new_shards.iter().map(|s| s.num_entries).sum();
        assert_eq!(total_entries, 5);

        // Read back all new shards and verify no duplicates
        let mut all_pairs: Vec<(u64, u32)> = Vec::new();
        for shard in &new_shards {
            let path = output_dir
                .join("inverted")
                .join(super::files::inverted_shard(shard.shard_id));
            let pairs = read_shard_pairs(&path).unwrap();
            all_pairs.extend(pairs);
        }

        assert_eq!(all_pairs, vec![(10, 1), (20, 1), (30, 1), (40, 1), (50, 1)]);
    }

    #[test]
    fn test_consolidate_shards_empty() {
        let tmp = TempDir::new().unwrap();
        let output_dir = tmp.path().join("test.ryxdi");
        std::fs::create_dir_all(output_dir.join("inverted")).unwrap();

        let (new_shards, unique_count) = consolidate_shards(
            &output_dir,
            &[],
            1,
            MIN_SHARD_BYTES,
            &ParquetWriteOptions::default(),
        )
        .unwrap();

        assert_eq!(unique_count, 0);
        assert!(new_shards.is_empty());
    }

    #[test]
    fn test_consolidate_shards_single_shard_no_duplicates() {
        let tmp = TempDir::new().unwrap();
        let output_dir = tmp.path().join("test.ryxdi");
        std::fs::create_dir_all(output_dir.join("inverted")).unwrap();

        let opts = ParquetWriteOptions::default();
        let intermediate = vec![InvertedShardInfo {
            shard_id: 0,
            min_minimizer: 10,
            max_minimizer: 30,
            num_entries: 3,
        }];
        write_shard_from_pairs(
            &output_dir.join("inverted/shard.0.parquet"),
            &[(10, 1), (20, 1), (30, 1)],
            &opts,
        )
        .unwrap();

        let (new_shards, unique_count) =
            consolidate_shards(&output_dir, &intermediate, 1, MIN_SHARD_BYTES, &opts).unwrap();

        assert_eq!(unique_count, 3);
        let total: u64 = new_shards.iter().map(|s| s.num_entries).sum();
        assert_eq!(total, 3);
    }

    #[test]
    fn test_consolidate_shards_all_duplicates() {
        let tmp = TempDir::new().unwrap();
        let output_dir = tmp.path().join("test.ryxdi");
        std::fs::create_dir_all(output_dir.join("inverted")).unwrap();

        let opts = ParquetWriteOptions::default();
        // Two shards with identical content
        let intermediate = vec![
            InvertedShardInfo {
                shard_id: 0,
                min_minimizer: 10,
                max_minimizer: 30,
                num_entries: 3,
            },
            InvertedShardInfo {
                shard_id: 1,
                min_minimizer: 10,
                max_minimizer: 30,
                num_entries: 3,
            },
        ];
        write_shard_from_pairs(
            &output_dir.join("inverted/shard.0.parquet"),
            &[(10, 1), (20, 1), (30, 1)],
            &opts,
        )
        .unwrap();
        write_shard_from_pairs(
            &output_dir.join("inverted/shard.1.parquet"),
            &[(10, 1), (20, 1), (30, 1)],
            &opts,
        )
        .unwrap();

        let (new_shards, unique_count) =
            consolidate_shards(&output_dir, &intermediate, 1, MIN_SHARD_BYTES, &opts).unwrap();

        assert_eq!(unique_count, 3); // only 3 unique minimizers
        let total: u64 = new_shards.iter().map(|s| s.num_entries).sum();
        assert_eq!(total, 3);
    }
}
