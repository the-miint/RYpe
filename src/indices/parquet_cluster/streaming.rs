//! Streaming writer for the `.ryci` cluster-index format.
//!
//! Phase 2 ships only the single-shard sequential entry point and the
//! `ClusterShardWriter`. Multi-shard k-way merge lands in Phase 3; the
//! parallel range-partitioned path from `.ryxdi` is intentionally not
//! ported here (the cluster scale — ≤10K contigs, single-digit GB — does
//! not require it; see plan §"Deferred").

use crate::error::{Result, RypeError};
use arrow::array::{ArrayRef, UInt32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use std::collections::HashMap;
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

/// Build a `.ryci` cluster index from validated bucket data, writing a single shard.
///
/// Phase 2 entry point: input is held in memory, sorted by `(minimizer, bucket_id)`,
/// and streamed into one shard. Phase 3 will add a `max_shard_bytes` parameter and
/// switch to a k-way-merge path; for now everything fits in one shard.
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
    create_cluster_parquet_index_with_options(output_dir, buckets, k, w, salt, None)
}

/// As [`create_cluster_parquet_index`] but with caller-supplied write options.
pub fn create_cluster_parquet_index_with_options(
    output_dir: &Path,
    buckets: Vec<ClusterBucketData>,
    k: usize,
    w: usize,
    salt: u64,
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

    // Sort (minimizer, bucket_id, position) triples and write a single shard.
    let shard_info = write_single_shard(output_dir, &buckets, &opts)?;

    let source_hash = compute_source_hash(&bucket_minimizer_counts);
    let total_entries = shard_info.num_entries;

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
            num_shards: 1,
            total_entries,
            // Single shard cannot overlap with itself; phase 3 sets this true for
            // sequential multi-shard mode.
            has_overlapping_shards: false,
            shards: vec![shard_info],
        }),
    };
    manifest.save(output_dir)?;

    Ok(manifest)
}

/// Materialize all triples, sort by `(minimizer, bucket_id)`, write one shard.
fn write_single_shard(
    output_dir: &Path,
    buckets: &[ClusterBucketData],
    options: &ClusterParquetWriteOptions,
) -> Result<ClusterInvertedShardInfo> {
    let total: usize = buckets.iter().map(|b| b.minimizers.len()).sum();

    let shard_path = output_dir
        .join(files::INVERTED_DIR)
        .join(files::inverted_shard(0));

    if total == 0 {
        // Empty input still gets a shard file so the reader has something to open.
        let writer = ClusterShardWriter::new(&shard_path, options)?;
        writer.finish()?;
        // Empty-range sentinel: max < min. Lets a future range-skip
        // optimization treat an empty shard as covering nothing rather than
        // as covering minimizer 0 (which is a valid value).
        return Ok(ClusterInvertedShardInfo {
            shard_id: 0,
            min_minimizer: u64::MAX,
            max_minimizer: 0,
            num_entries: 0,
        });
    }

    // Build a flat triple list then sort. Phase 3 will switch to a k-way merge
    // for memory-bounded operation; Phase 2 trades that for simplicity.
    let mut triples: Vec<(u64, u32, u32)> = Vec::with_capacity(total);
    for b in buckets {
        // ClusterBucketData::validate already enforced parallel-array length.
        for (i, &m) in b.minimizers.iter().enumerate() {
            triples.push((m, b.bucket_id, b.positions[i]));
        }
    }
    // Sort by (minimizer, bucket_id) — same merge-join contract as .ryxdi.
    // The position tiebreak is unreachable in well-formed input (validate()
    // rejects duplicate minimizers within a bucket, so (minimizer, bucket_id)
    // is already unique) but we include it to keep the sort total.
    triples.sort_unstable_by_key(|&(m, b, p)| (m, b, p));

    let min_minimizer = triples.first().map(|t| t.0).unwrap_or(0);
    let max_minimizer = triples.last().map(|t| t.0).unwrap_or(0);
    let num_entries = triples.len() as u64;

    let mut writer = ClusterShardWriter::new(&shard_path, options)?;
    let mut mins: Vec<u64> = Vec::with_capacity(PARQUET_BATCH_SIZE);
    let mut bids: Vec<u32> = Vec::with_capacity(PARQUET_BATCH_SIZE);
    let mut poss: Vec<u32> = Vec::with_capacity(PARQUET_BATCH_SIZE);
    for chunk in triples.chunks(PARQUET_BATCH_SIZE) {
        mins.clear();
        bids.clear();
        poss.clear();
        for &(m, b, p) in chunk {
            mins.push(m);
            bids.push(b);
            poss.push(p);
        }
        writer.write_batch(&mins, &bids, &poss)?;
    }
    writer.finish()?;

    Ok(ClusterInvertedShardInfo {
        shard_id: 0,
        min_minimizer,
        max_minimizer,
        num_entries,
    })
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
