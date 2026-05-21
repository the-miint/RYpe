//! Streaming writer for the `.ryci` cluster-index format.
//!
//! Phase 3: sequential k-way merge over `(minimizer, bucket_id, position)`
//! triples with optional shard rollover when `max_shard_bytes` is set.
//! The parallel range-partitioned path from `.ryxdi` is intentionally not
//! ported here (the cluster scale — ≤10K contigs, single-digit GB — does
//! not require it; see plan §"Deferred").

use crate::error::{Result, RypeError};
use arrow::array::{ArrayRef, UInt32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use crate::constants::PARQUET_BATCH_SIZE;
use crate::indices::parquet::write_buckets_parquet;

use super::manifest::{
    create_cluster_index_directory, ClusterBucketData, ClusterInvertedManifest,
    ClusterInvertedShardInfo, ClusterParquetManifest,
};
use super::options::ClusterParquetWriteOptions;
use super::{files, FORMAT_MAGIC_CLUSTER, FORMAT_VERSION_CLUSTER};

/// Compute a hash from bucket minimizer counts for source-change detection.
///
/// Copied verbatim from `crate::indices::parquet::streaming::compute_source_hash`
/// per the plan: the two index formats share an identical hash convention so a
/// downstream caller building both from the same inputs gets matching hashes.
fn compute_source_hash(counts: &HashMap<u32, usize>) -> u64 {
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

/// Build a `.ryci` cluster index from validated bucket data.
///
/// Convenience entry: single-shard, default options. Use
/// [`create_cluster_parquet_index_with_options`] for shard rollover or
/// non-default compression / row group sizes.
///
/// # Arguments
/// * `output_dir` — path to create (e.g. `cluster.ryci`).
/// * `buckets` — per-bucket minimizer/position parallel arrays; each must satisfy
///   [`ClusterBucketData::validate`] (sorted-unique minimizers, parallel arrays).
/// * `k` / `w` / `salt` — index parameters recorded in the manifest.
pub fn create_cluster_parquet_index(
    output_dir: &Path,
    buckets: Vec<ClusterBucketData>,
    k: usize,
    w: usize,
    salt: u64,
) -> Result<ClusterParquetManifest> {
    create_cluster_parquet_index_with_options(output_dir, buckets, k, w, salt, None, None)
}

/// As [`create_cluster_parquet_index`] but with caller-supplied options.
///
/// When `max_shard_bytes` is `Some(n)`, the writer rolls over to a new shard once
/// the current shard's on-disk size reaches `n` bytes (checked between batches),
/// matching the `.ryxdi` sequential mode contract. When `None`, all triples go
/// into a single shard.
pub fn create_cluster_parquet_index_with_options(
    output_dir: &Path,
    buckets: Vec<ClusterBucketData>,
    k: usize,
    w: usize,
    salt: u64,
    max_shard_bytes: Option<usize>,
    options: Option<&ClusterParquetWriteOptions>,
) -> Result<ClusterParquetManifest> {
    let opts = options.cloned().unwrap_or_default();
    opts.validate()?;

    // Validate every bucket up front. Lifting this to the writer boundary means
    // a malformed bucket is caught before we touch the filesystem.
    for bucket in &buckets {
        bucket.validate().map_err(|e| {
            RypeError::validation(format!(
                "invalid cluster bucket '{}' (id={}): {}",
                bucket.bucket_name, bucket.bucket_id, e
            ))
        })?;
    }

    // Reject duplicate bucket_id across input. Without this, a HashMap insert
    // below silently overwrites bucket metadata while every triple from the
    // overwritten bucket still reaches the shard — total_minimizers would no
    // longer match num_buckets * mean-bucket-size, and the bucket_id column in
    // the shard would refer to a name the manifest doesn't have.
    let mut seen_ids: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for bucket in &buckets {
        if !seen_ids.insert(bucket.bucket_id) {
            return Err(RypeError::validation(format!(
                "duplicate bucket_id {} in input (bucket name '{}')",
                bucket.bucket_id, bucket.bucket_name
            )));
        }
    }

    create_cluster_index_directory(output_dir)?;

    // Collect bucket metadata (same layout as .ryxdi — reuses write_buckets_parquet).
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
    write_buckets_parquet(output_dir, &bucket_names, &bucket_sources, None)?;

    // K-way merge over (minimizer, bucket_id, position) triples, rolling over
    // to a new shard when the current shard hits `max_shard_bytes`.
    let shard_infos = stream_to_shards_sequential(output_dir, &buckets, max_shard_bytes, &opts)?;

    let source_hash = compute_source_hash(&bucket_minimizer_counts);
    let total_entries: u64 = shard_infos.iter().map(|s| s.num_entries).sum();

    let manifest = ClusterParquetManifest {
        magic: FORMAT_MAGIC_CLUSTER.to_string(),
        format_version: FORMAT_VERSION_CLUSTER,
        k,
        w,
        salt,
        source_hash,
        num_buckets: bucket_names.len() as u32,
        total_minimizers,
        inverted: Some(ClusterInvertedManifest {
            num_shards: shard_infos.len() as u32,
            total_entries,
            // Sequential k-way merge emits size-bounded shards that may share
            // minimizers across shard boundaries (each bucket's minimizers are
            // unique within its own bucket, but two buckets can both contribute
            // the same minimizer value to adjacent shards). Matches .ryxdi
            // sequential semantics; readers must dedupe across shards.
            has_overlapping_shards: true,
            shards: shard_infos,
        }),
    };
    manifest.save(output_dir)?;

    Ok(manifest)
}

/// K-way merge heap entry: `(Reverse((minimizer, bucket_id)), bucket_index, pos_in_bucket)`.
/// Sort key is `(minimizer, bucket_id)`; position is looked up from the bucket
/// when the entry is popped.
type MergeHeapEntry = (Reverse<(u64, u32)>, usize, usize);

/// Sequential k-way merge over per-bucket sorted minimizer streams, with
/// shard rollover once a shard's on-disk size reaches `max_shard_bytes`.
///
/// Mirrors `crate::indices::parquet::streaming::stream_to_shards_sequential`
/// but reads one extra column (positions) from each bucket and writes one
/// extra column (position) into each shard.
fn stream_to_shards_sequential(
    output_dir: &Path,
    buckets: &[ClusterBucketData],
    max_shard_bytes: Option<usize>,
    options: &ClusterParquetWriteOptions,
) -> Result<Vec<ClusterInvertedShardInfo>> {
    let rollover_bytes = max_shard_bytes.unwrap_or(usize::MAX);

    // Empty input still emits a single empty shard so the reader has something
    // to open. Empty-range sentinel: min > max signals "covers no minimizers".
    if buckets.is_empty() || buckets.iter().all(|b| b.minimizers.is_empty()) {
        let shard_path = output_dir
            .join(files::INVERTED_DIR)
            .join(files::inverted_shard(0));
        let writer = ClusterShardWriter::new(&shard_path, options)?;
        writer.finish()?;
        return Ok(vec![ClusterInvertedShardInfo {
            shard_id: 0,
            min_minimizer: u64::MAX,
            max_minimizer: 0,
            num_entries: 0,
        }]);
    }

    // Index buckets by their non-empty subset; the heap stores indices into
    // this slice, not into the original `buckets` Vec.
    let bucket_data: Vec<(u32, &[u64], &[u32])> = buckets
        .iter()
        .filter(|b| !b.minimizers.is_empty())
        .map(|b| (b.bucket_id, b.minimizers.as_slice(), b.positions.as_slice()))
        .collect();

    let mut heap: BinaryHeap<MergeHeapEntry> = BinaryHeap::with_capacity(bucket_data.len());
    for (idx, &(bid, mins, _)) in bucket_data.iter().enumerate() {
        heap.push((Reverse((mins[0], bid)), idx, 0));
    }

    let mut shard_infos: Vec<ClusterInvertedShardInfo> = Vec::new();
    let mut current_shard_id: u32 = 0;
    let mut current_writer: Option<ClusterShardWriter> = None;
    let mut current_shard_entries: u64 = 0;
    let mut current_shard_min: u64 = u64::MAX;
    let mut current_shard_max: u64 = 0;

    // Flush the in-memory batch on row-group boundaries, not just at
    // PARQUET_BATCH_SIZE. Otherwise a small test (or any workload smaller than
    // PARQUET_BATCH_SIZE rows) never causes ArrowWriter to finalize a row group
    // during the merge, so `bytes_written` stays near zero and `max_shard_bytes`
    // can't fire. Capping at row_group_size guarantees one row group flushes
    // per batch — bytes_written then advances on a cadence the rollover check
    // can observe.
    let batch_cap = std::cmp::min(PARQUET_BATCH_SIZE, options.row_group_size);
    let mut min_batch: Vec<u64> = Vec::with_capacity(batch_cap);
    let mut bid_batch: Vec<u32> = Vec::with_capacity(batch_cap);
    let mut pos_batch: Vec<u32> = Vec::with_capacity(batch_cap);

    while let Some((Reverse((minimizer, bucket_id)), bucket_idx, pos_in_bucket)) = heap.pop() {
        // Rollover check: only meaningful after a writer exists AND at least
        // one batch has flushed (so bytes_written reflects real data, not
        // just the header). The `!min_batch.is_empty()` guard avoids
        // spinning on the same boundary if bytes_written already passed it.
        let need_new_shard = match current_writer.as_ref() {
            None => true,
            Some(w) => w.bytes_written() >= rollover_bytes && !min_batch.is_empty(),
        };

        if need_new_shard {
            if let Some(mut w) = current_writer.take() {
                if !min_batch.is_empty() {
                    w.write_batch(&min_batch, &bid_batch, &pos_batch)?;
                    min_batch.clear();
                    bid_batch.clear();
                    pos_batch.clear();
                }
                w.finish()?;
                shard_infos.push(ClusterInvertedShardInfo {
                    shard_id: current_shard_id,
                    min_minimizer: current_shard_min,
                    max_minimizer: current_shard_max,
                    num_entries: current_shard_entries,
                });
                current_shard_id += 1;
            }
            let shard_path = output_dir
                .join(files::INVERTED_DIR)
                .join(files::inverted_shard(current_shard_id));
            current_writer = Some(ClusterShardWriter::new(&shard_path, options)?);
            current_shard_entries = 0;
            current_shard_min = minimizer;
        }

        let position = bucket_data[bucket_idx].2[pos_in_bucket];
        min_batch.push(minimizer);
        bid_batch.push(bucket_id);
        pos_batch.push(position);
        current_shard_entries += 1;
        // The heap pop order guarantees `minimizer` is non-decreasing within
        // a shard, so this is a real max.
        current_shard_max = minimizer;

        if min_batch.len() >= batch_cap {
            if let Some(ref mut w) = current_writer {
                w.write_batch(&min_batch, &bid_batch, &pos_batch)?;
            }
            min_batch.clear();
            bid_batch.clear();
            pos_batch.clear();
        }

        let mins = bucket_data[bucket_idx].1;
        let next_pos = pos_in_bucket + 1;
        if next_pos < mins.len() {
            heap.push((Reverse((mins[next_pos], bucket_id)), bucket_idx, next_pos));
        }
    }

    if let Some(mut w) = current_writer.take() {
        if !min_batch.is_empty() {
            w.write_batch(&min_batch, &bid_batch, &pos_batch)?;
        }
        w.finish()?;
        shard_infos.push(ClusterInvertedShardInfo {
            shard_id: current_shard_id,
            min_minimizer: current_shard_min,
            max_minimizer: current_shard_max,
            num_entries: current_shard_entries,
        });
    }

    Ok(shard_infos)
}

/// Helper for writing a single `.ryci` shard file.
///
/// 3-column schema `(minimizer u64, bucket_id u32, position u32)`. Schema lives
/// alongside the writer so callers don't have to reconstruct it.
pub(crate) struct ClusterShardWriter {
    writer: ArrowWriter<File>,
    schema: Arc<Schema>,
}

impl ClusterShardWriter {
    pub(crate) fn new(path: &Path, options: &ClusterParquetWriteOptions) -> Result<Self> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("minimizer", DataType::UInt64, false),
            Field::new("bucket_id", DataType::UInt32, false),
            Field::new("position", DataType::UInt32, false),
        ]));
        let props = options.to_writer_properties();
        let file = File::create(path)
            .map_err(|e| RypeError::io(path.to_path_buf(), "create cluster shard", e))?;
        let writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;
        Ok(Self { writer, schema })
    }

    /// Bytes physically flushed to the underlying file so far. Used by the
    /// sequential merge to decide when to roll over to a new shard. Note: the
    /// in-progress row group buffer is NOT counted here, so the value only
    /// advances on row-group flush — callers expecting precise byte budgets
    /// should size `row_group_size` accordingly.
    pub(crate) fn bytes_written(&self) -> usize {
        self.writer.bytes_written()
    }

    pub(crate) fn write_batch(
        &mut self,
        minimizers: &[u64],
        bucket_ids: &[u32],
        positions: &[u32],
    ) -> Result<()> {
        debug_assert_eq!(minimizers.len(), bucket_ids.len());
        debug_assert_eq!(minimizers.len(), positions.len());

        let minimizer_array: ArrayRef = Arc::new(UInt64Array::from(minimizers.to_vec()));
        let bucket_id_array: ArrayRef = Arc::new(UInt32Array::from(bucket_ids.to_vec()));
        let position_array: ArrayRef = Arc::new(UInt32Array::from(positions.to_vec()));
        let batch = RecordBatch::try_new(
            self.schema.clone(),
            vec![minimizer_array, bucket_id_array, position_array],
        )?;
        self.writer.write(&batch)?;
        Ok(())
    }

    pub(crate) fn finish(self) -> Result<()> {
        self.writer.close()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // WHY: a writer hidden behind compute_source_hash being identical to .ryxdi's
    // is part of the format contract — two builders fed the same inputs must
    // produce matching hashes so callers can pair indices by hash.
    #[test]
    fn source_hash_matches_classify_side_for_identical_input() {
        let mut counts: HashMap<u32, usize> = HashMap::new();
        counts.insert(0, 5);
        counts.insert(1, 8);
        // Same content via the .ryxdi entry point.
        let ryxdi_hash = crate::indices::parquet::compute_source_hash(&counts);
        let ryci_hash = super::compute_source_hash(&counts);
        assert_eq!(ryci_hash, ryxdi_hash);
    }

    // WHY: the writer must reject malformed input at the boundary, BEFORE any
    // files have been created. A half-written .ryci directory would confuse
    // is_cluster_parquet_index (which only checks the manifest).
    #[test]
    fn create_rejects_unsorted_input_before_writing() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("unsorted.ryci");
        let bad = ClusterBucketData {
            bucket_id: 0,
            bucket_name: "x".to_string(),
            sources: vec![],
            minimizers: vec![5, 3, 9], // unsorted
            positions: vec![1, 2, 3],
        };
        let err = create_cluster_parquet_index(&dir, vec![bad], 64, 50, 0)
            .expect_err("unsorted bucket must be rejected");
        assert!(format!("{}", err).contains("unsorted"));
        // No directory should have been created — we validate before touching disk.
        assert!(!dir.exists(), "writer leaked a directory on rejected input");
    }
}
