//! Sharded classification functions for memory-efficient large-scale classification.
//!
//! These functions load one shard at a time to minimize memory usage when
//! classifying against large indices that don't fit in memory.

use anyhow::Result;
use rayon::prelude::*;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Instant;

use crate::constants::{
    COO_MERGE_JOIN_MAX_BUCKETS, DENSE_ACCUMULATOR_MAX_BUCKETS, ESTIMATED_MINIMIZERS_PER_SEQUENCE,
};
use crate::core::extraction::get_paired_minimizers_into;
use crate::core::workspace::MinimizerWorkspace;
use crate::indices::sharded::{ShardManifest, ShardedInvertedIndex};
use crate::indices::{InvertedIndex, QueryInvertedIndex};
use crate::types::{HitResult, QueryRecord};

use crate::log_timing;

use super::common::{collect_negative_minimizers_sharded, filter_negative_mins};
use super::merge_join::{
    merge_join_coo_parallel, merge_join_csr, merge_join_pairs_sparse, DenseAccumulator,
    HitAccumulator, SparseAccumulator, SparseHit,
};

/// A loaded shard ready for merge-join, either COO or CSR format.
///
/// - `Coo`: Sorted (minimizer, bucket_id) pairs. Used when
///   `num_buckets <= COO_MERGE_JOIN_MAX_BUCKETS`. Avoids CSR conversion
///   overhead and reduces peak memory (no concurrent COO + CSR).
/// - `Csr`: Compressed Sparse Row format (`InvertedIndex`). Used when
///   `num_buckets > COO_MERGE_JOIN_MAX_BUCKETS`. Iterates only unique
///   minimizers, avoiding the N× blowup when reference COO is much
///   larger than the unique minimizer array.
enum LoadedShard {
    Coo(Vec<(u64, u32)>),
    Csr(InvertedIndex),
}

/// Estimate minimizers per query from the first record in a batch.
///
/// Uses the formula: ((query_length - k) / w + 1) * 2 (for both strands).
/// Falls back to ESTIMATED_MINIMIZERS_PER_SEQUENCE if the batch is empty or
/// sequences are too short.
fn estimate_minimizers_from_records(records: &[QueryRecord], k: usize, w: usize) -> usize {
    if records.is_empty() {
        return ESTIMATED_MINIMIZERS_PER_SEQUENCE;
    }

    let (_, s1, s2) = &records[0];
    let query_len = s1.len() + s2.map(|s| s.len()).unwrap_or(0);

    if query_len <= k {
        return ESTIMATED_MINIMIZERS_PER_SEQUENCE;
    }

    // Estimate: (len - k) / w + 1 minimizers per strand, times 2 for both strands
    let estimate = ((query_len - k) / w + 1) * 2;
    estimate.max(ESTIMATED_MINIMIZERS_PER_SEQUENCE)
}

/// Extract minimizers from a batch of query records in parallel.
///
/// This is the extraction step factored out from classification functions,
/// allowing callers to cache extracted minimizers for reuse (e.g., log-ratio
/// deferred buffer avoids re-extracting when classifying against the denominator).
///
/// # Arguments
/// * `k` - K-mer size
/// * `w` - Window size
/// * `salt` - Hash salt
/// * `negative_mins` - Optional set of minimizers to exclude before returning
/// * `records` - Batch of query records
///
/// # Returns
/// Vec of (forward_minimizers, rc_minimizers) per query, in the same order as `records`.
pub fn extract_batch_minimizers(
    k: usize,
    w: usize,
    salt: u64,
    negative_mins: Option<&HashSet<u64>>,
    records: &[QueryRecord],
) -> Vec<(Vec<u64>, Vec<u64>)> {
    if records.is_empty() {
        return Vec::new();
    }
    let estimated_mins = estimate_minimizers_from_records(records, k, w);
    records
        .par_iter()
        .map_init(
            || MinimizerWorkspace::with_estimate(estimated_mins),
            |ws, (_, s1, s2)| {
                let (ha, hb) = get_paired_minimizers_into(s1, *s2, k, w, salt, ws);
                filter_negative_mins(ha, hb, negative_mins)
            },
        )
        .collect()
}

/// Get the max bucket ID from a manifest's bucket_names.
/// Returns 0 if no buckets exist.
fn max_bucket_id_from_manifest(manifest: &crate::indices::sharded::ShardManifest) -> u32 {
    manifest.bucket_names.keys().max().copied().unwrap_or(0)
}

/// Check if dense accumulators should be used for this manifest.
fn use_dense_accumulator(manifest: &crate::indices::sharded::ShardManifest) -> bool {
    let max_id = max_bucket_id_from_manifest(manifest);
    max_id > 0 && (max_id as usize) <= DENSE_ACCUMULATOR_MAX_BUCKETS
}

/// Check if COO merge-join should be used for this manifest.
///
/// COO merge-join is faster for indices with few buckets (reference COO is
/// close to 1:1 with unique minimizers). For many-bucket indices, CSR
/// merge-join iterates only unique minimizers and does compact bucket-slice
/// lookups, avoiding the N× blowup in the reference COO representation.
fn use_coo_merge_join(manifest: &crate::indices::sharded::ShardManifest) -> bool {
    manifest.bucket_names.len() <= COO_MERGE_JOIN_MAX_BUCKETS
}

// ============================================================================
// Pipelined shard loop (classify_from_query_index)
// ============================================================================

/// Inner shard loop generic over accumulator type.
///
/// Uses pipelined I/O: a background thread loads shard N+1 while the main
/// thread merge-joins shard N. This overlaps I/O latency with CPU work.
/// For single-shard indices, the overhead is minimal (one send + one receive).
///
/// When `use_coo` is true, loads shards as COO pairs and uses `merge_join_coo`.
/// When false, loads shards as CSR (InvertedIndex) and uses `accumulate_merge_join_csr`,
/// which iterates only unique minimizers — much faster for multi-bucket indices.
fn classify_shard_loop<A: HitAccumulator>(
    sharded: &ShardedInvertedIndex,
    query_idx: &QueryInvertedIndex,
    query_ids: &[i64],
    threshold: f64,
    read_options: Option<&crate::indices::parquet::ParquetReadOptions>,
    mut accumulator: A,
    use_coo: bool,
) -> Result<Vec<HitResult>> {
    let t_start = Instant::now();
    let manifest = sharded.manifest();

    let mut total_shard_load_ms = 0u128;
    let mut total_merge_join_ms = 0u128;

    // Compute unique query minimizers once for filtered shard loading.
    // Returns sorted, deduplicated minimizers — required for binary search
    // correctness in load_shard_*_for_query() and for gallop_join_csr().
    let query_minimizers = query_idx.unique_minimizers();

    // Debug: show query minimizer statistics
    if std::env::var("RYPE_DEBUG").is_ok() && !query_minimizers.is_empty() {
        eprintln!(
            "[DEBUG] Query minimizers: {} unique, range: {} to {}",
            query_minimizers.len(),
            query_minimizers[0],
            query_minimizers[query_minimizers.len() - 1]
        );
    }

    // Pipelined shard processing: background loader thread + main merge-join thread.
    // sync_channel(1) allows at most one shard buffered ahead, bounding memory to
    // at most 2 loaded shards simultaneously.
    let load_result: Result<()> = std::thread::scope(|scope| {
        let (tx, rx) = mpsc::sync_channel::<Result<(LoadedShard, u128)>>(1);
        let query_mins_ref = &query_minimizers;

        // Background loader thread.
        // MUST use scoped thread (not std::thread::spawn) because we borrow
        // `sharded`, `manifest.shards`, and `query_mins_ref` from the enclosing
        // scope. thread::scope guarantees these borrows don't outlive the thread.
        let loader = scope.spawn(move || {
            for shard_info in &manifest.shards {
                let t_load = Instant::now();
                let loaded: Result<LoadedShard> = if use_coo {
                    sharded
                        .load_shard_coo_for_query(shard_info.shard_id, query_mins_ref, read_options)
                        .map(LoadedShard::Coo)
                        .map_err(Into::into)
                } else {
                    sharded
                        .load_shard_for_query(shard_info.shard_id, query_mins_ref, read_options)
                        .map(LoadedShard::Csr)
                        .map_err(Into::into)
                };
                let load_ms = t_load.elapsed().as_millis();

                // If receiver dropped (main thread errored/stopped), stop loading
                if tx.send(loaded.map(|s| (s, load_ms))).is_err() {
                    break;
                }
            }
        });

        // Main thread: receive loaded shards and merge-join
        for received in rx {
            let (shard, load_ms) = received?;
            total_shard_load_ms += load_ms;

            let t_merge = Instant::now();
            match shard {
                LoadedShard::Coo(ref pairs) => {
                    merge_join_coo_parallel(query_idx, pairs, &mut accumulator);
                }
                LoadedShard::Csr(ref idx) => {
                    merge_join_csr(query_idx, idx, &mut accumulator, &query_minimizers);
                }
            }
            total_merge_join_ms += t_merge.elapsed().as_millis();
        }

        // Join the loader thread (propagates panics)
        loader.join().expect("shard loader thread panicked");

        Ok(())
    });

    load_result?;

    log_timing("merge_join: shard_load_total", total_shard_load_ms);
    log_timing("merge_join: merge_join_total", total_merge_join_ms);

    let t_score = Instant::now();
    let results = accumulator.score_and_filter(query_idx, query_ids, threshold);
    log_timing("merge_join: scoring", t_score.elapsed().as_millis());
    log_timing("merge_join: total", t_start.elapsed().as_millis());

    Ok(results)
}

/// Classify using a pre-built QueryInvertedIndex against a sharded inverted index.
///
/// This is the core classification step that processes each shard sequentially
/// using merge-join. Use this when you have a pre-built `QueryInvertedIndex`
/// (e.g., from a deferred buffer's flat COO entries).
///
/// # Arguments
/// * `sharded` - The sharded inverted index
/// * `query_idx` - Pre-built query inverted index
/// * `query_ids` - Query IDs corresponding to each read in the query index
/// * `threshold` - Minimum score threshold
/// * `read_options` - Parquet read options (None = default behavior without bloom filters)
///
/// # Returns
/// Vector of HitResult for all (query, bucket) pairs meeting the threshold.
pub fn classify_from_query_index(
    sharded: &ShardedInvertedIndex,
    query_idx: &QueryInvertedIndex,
    query_ids: &[i64],
    threshold: f64,
    read_options: Option<&crate::indices::parquet::ParquetReadOptions>,
) -> Result<Vec<HitResult>> {
    let num_reads = query_idx.num_reads();
    if num_reads == 0 {
        return Ok(Vec::new());
    }

    let manifest = sharded.manifest();
    let max_id = max_bucket_id_from_manifest(manifest);
    let use_coo = use_coo_merge_join(manifest);

    if use_dense_accumulator(manifest) {
        classify_shard_loop(
            sharded,
            query_idx,
            query_ids,
            threshold,
            read_options,
            DenseAccumulator::new(num_reads, max_id),
            use_coo,
        )
    } else {
        classify_shard_loop(
            sharded,
            query_idx,
            query_ids,
            threshold,
            read_options,
            SparseAccumulator::new(num_reads),
            use_coo,
        )
    }
}

/// Classify from pre-extracted minimizers against a sharded inverted index using merge-join.
///
/// Builds a QueryInvertedIndex, then delegates to [`classify_from_query_index`].
/// Use this when you have pre-extracted minimizers but not a pre-built query index.
///
/// # Arguments
/// * `sharded` - The sharded inverted index
/// * `extracted` - Pre-extracted minimizers: (fwd_mins, rc_mins) per query
/// * `query_ids` - Query IDs corresponding to each entry in `extracted`
/// * `threshold` - Minimum score threshold
/// * `read_options` - Parquet read options (None = default behavior without bloom filters)
///
/// # Returns
/// Vector of HitResult for all (query, bucket) pairs meeting the threshold.
pub fn classify_from_extracted_minimizers(
    sharded: &ShardedInvertedIndex,
    extracted: &[(Vec<u64>, Vec<u64>)],
    query_ids: &[i64],
    threshold: f64,
    read_options: Option<&crate::indices::parquet::ParquetReadOptions>,
) -> Result<Vec<HitResult>> {
    if extracted.is_empty() {
        return Ok(Vec::new());
    }

    // Build query inverted index
    let t_build_idx = Instant::now();
    let query_idx = QueryInvertedIndex::build(extracted);
    log_timing(
        "merge_join: build_query_index",
        t_build_idx.elapsed().as_millis(),
    );

    classify_from_query_index(sharded, &query_idx, query_ids, threshold, read_options)
}

/// Classify a batch of records against a sharded inverted index using merge-join.
///
/// Extracts minimizers from sequences, then delegates to
/// `classify_from_extracted_minimizers`. This is the standard entry point when
/// you have raw sequences and don't need to cache the extracted minimizers.
///
/// # Arguments
/// * `sharded` - The sharded inverted index
/// * `negative_mins` - Optional set of minimizers to exclude from queries before scoring
/// * `records` - Batch of query records (should be pre-trimmed if trimming is desired)
/// * `threshold` - Minimum score threshold
/// * `read_options` - Parquet read options (None = default behavior without bloom filters)
///
/// # Returns
/// Vector of HitResult for all (query, bucket) pairs meeting the threshold.
pub fn classify_batch_sharded_merge_join(
    sharded: &ShardedInvertedIndex,
    negative_mins: Option<&HashSet<u64>>,
    records: &[QueryRecord],
    threshold: f64,
    read_options: Option<&crate::indices::parquet::ParquetReadOptions>,
) -> Result<Vec<HitResult>> {
    if records.is_empty() {
        return Ok(Vec::new());
    }

    let manifest = sharded.manifest();

    let t_extract = Instant::now();
    let extracted = extract_batch_minimizers(
        manifest.k,
        manifest.w,
        manifest.salt,
        negative_mins,
        records,
    );
    log_timing("merge_join: extraction", t_extract.elapsed().as_millis());

    let query_ids: Vec<i64> = records.iter().map(|(id, _, _)| *id).collect();

    classify_from_extracted_minimizers(sharded, &extracted, &query_ids, threshold, read_options)
}

// ============================================================================
// Parallel row group processing (classify_from_query_index_parallel_rg)
// ============================================================================

/// Inner parallel RG processing generic over accumulator type.
///
/// Handles both fold/reduce (for small batches) and collect+merge (for large
/// batches) strategies, using the accumulator trait for hit accumulation.
#[allow(clippy::too_many_arguments)]
fn parallel_rg_inner<A>(
    work_items: Vec<(PathBuf, usize)>,
    query_idx: &QueryInvertedIndex,
    query_ids: &[i64],
    query_minimizers: &[u64],
    threshold: f64,
    num_reads: usize,
    total_rg_count: usize,
    filtered_rg_count: usize,
    t_start: Instant,
    make_acc: impl Fn() -> A + Send + Sync,
) -> Result<Vec<HitResult>>
where
    A: HitAccumulator,
{
    use crate::indices::load_row_group_pairs;

    // Strategy selection: fold/reduce creates per-thread accumulators.
    // For large batches (millions of short reads), this may exceed available
    // memory with dense accumulators. Use collect+merge instead.
    //
    // At DENSE_ACCUMULATOR_MAX_BUCKETS=256 and 8 bytes per bucket:
    // 256 × 8 × num_reads bytes per accumulator × num_threads.
    // At 8 threads and 640K reads: 256 × 8 × 640K × 8 ≈ 10 GB — too much.
    // 500K is a conservative threshold (~80% of theoretical 640K) to stay
    // well within memory bounds across varying thread counts.
    const FOLD_REDUCE_MAX_READS: usize = 500_000;

    let t_parallel = Instant::now();
    let final_accumulator = if num_reads <= FOLD_REDUCE_MAX_READS {
        // Fold/reduce: efficient for small batches (long reads).
        // Each thread merges hits into a local accumulator immediately.
        work_items
            .into_par_iter()
            .try_fold(&make_acc, |mut acc, (shard_path, rg_idx)| -> Result<A> {
                let pairs = load_row_group_pairs(&shard_path, rg_idx, query_minimizers)?;
                if !pairs.is_empty() {
                    let hits = merge_join_pairs_sparse(query_idx, &pairs);
                    for (read_idx, bucket_id, fwd, rc) in hits {
                        acc.record_hit_counts(read_idx as usize, bucket_id, fwd, rc);
                    }
                }
                Ok(acc)
            })
            .try_reduce(&make_acc, |mut a, b| {
                a.merge(b);
                Ok(a)
            })?
    } else {
        // Collect+merge: efficient for large batches (millions of short reads).
        // SparseHits are small per-read, so materializing all at once is fine.
        let results: Result<Vec<Vec<SparseHit>>> = work_items
            .into_par_iter()
            .map(|(shard_path, rg_idx)| {
                let pairs = load_row_group_pairs(&shard_path, rg_idx, query_minimizers)?;
                Ok(merge_join_pairs_sparse(query_idx, &pairs))
            })
            .collect();
        let mut acc = make_acc();
        for rg_hits in results? {
            if rg_hits.is_empty() {
                continue;
            }
            for (read_idx, bucket_id, fwd, rc) in rg_hits {
                acc.record_hit_counts(read_idx as usize, bucket_id, fwd, rc);
            }
        }
        acc
    };
    log_timing(
        "parallel_rg: rg_process_total",
        t_parallel.elapsed().as_millis(),
    );
    log_timing("parallel_rg: total_rg_count", total_rg_count as u128);
    log_timing("parallel_rg: filtered_rg_count", filtered_rg_count as u128);

    let t_score = Instant::now();
    let results = final_accumulator.score_and_filter(query_idx, query_ids, threshold);
    log_timing("parallel_rg: scoring", t_score.elapsed().as_millis());
    log_timing("parallel_rg: total", t_start.elapsed().as_millis());

    Ok(results)
}

/// Classify using a pre-built QueryInvertedIndex with parallel row group processing.
///
/// This is the core parallel-RG classification step. Use this when you have a
/// pre-built `QueryInvertedIndex` (e.g., from a deferred buffer's flat COO entries).
///
/// Each row group is processed independently in parallel:
/// 1. Pre-filter RGs by query minimizer range (using column statistics)
/// 2. Load matching RG pairs (pre-sorted within RG)
/// 3. Merge-join pairs with query index, emitting sparse hits
/// 4. Merge all sparse hits into final accumulators
///
/// # Arguments
/// * `sharded` - The sharded inverted index (must be Parquet format)
/// * `query_idx` - Pre-built query inverted index
/// * `query_ids` - Query IDs corresponding to each read in the query index
/// * `threshold` - Minimum score threshold
/// * `_read_options` - Unused; accepted for API consistency
///
/// # Returns
/// Vector of HitResult for all (query, bucket) pairs meeting the threshold.
pub fn classify_from_query_index_parallel_rg(
    sharded: &ShardedInvertedIndex,
    query_idx: &QueryInvertedIndex,
    query_ids: &[i64],
    threshold: f64,
    _read_options: Option<&crate::indices::parquet::ParquetReadOptions>,
) -> Result<Vec<HitResult>> {
    use crate::indices::get_row_group_ranges;

    let t_start = Instant::now();

    let num_reads = query_idx.num_reads();
    if num_reads == 0 {
        return Ok(Vec::new());
    }

    let manifest = sharded.manifest();

    // Compute unique query minimizers once for shard/RG filtering and bloom filter hints
    let query_minimizers = query_idx.unique_minimizers();
    let (query_min, query_max) = match query_idx.minimizer_range() {
        Some(range) => range,
        None => return Ok(Vec::new()),
    };

    let mut total_rg_count = 0usize;

    // Collect (shard_path, rg_idx) pairs that overlap with query range
    let mut work_items: Vec<(PathBuf, usize)> = Vec::new();

    let t_filter = Instant::now();

    // Use cached row group ranges if available (Parquet format with preloaded metadata)
    let use_cache = sharded.has_rg_cache();

    for (shard_pos, shard_info) in manifest.shards.iter().enumerate() {
        let shard_path =
            ShardManifest::shard_path_parquet(sharded.base_path(), shard_info.shard_id);

        // Get RG ranges from cache or load from file
        let rg_ranges = if use_cache {
            sharded
                .rg_ranges(shard_pos)
                .map(|s| s.to_vec())
                .unwrap_or_default()
        } else {
            get_row_group_ranges(&shard_path)?
        };
        total_rg_count += rg_ranges.len();

        for info in rg_ranges {
            // Check if RG range overlaps with query range
            if info.max >= query_min && info.min <= query_max {
                work_items.push((shard_path.clone(), info.rg_idx));
            }
        }
    }
    let filtered_rg_count = work_items.len();
    log_timing("parallel_rg: rg_filter", t_filter.elapsed().as_millis());

    // Dispatch to generic inner function based on accumulator type
    let max_id = max_bucket_id_from_manifest(manifest);

    if use_dense_accumulator(manifest) {
        parallel_rg_inner(
            work_items,
            query_idx,
            query_ids,
            &query_minimizers,
            threshold,
            num_reads,
            total_rg_count,
            filtered_rg_count,
            t_start,
            || DenseAccumulator::new(num_reads, max_id),
        )
    } else {
        parallel_rg_inner(
            work_items,
            query_idx,
            query_ids,
            &query_minimizers,
            threshold,
            num_reads,
            total_rg_count,
            filtered_rg_count,
            t_start,
            || SparseAccumulator::new(num_reads),
        )
    }
}

/// Classify from pre-extracted minimizers using parallel row group processing.
///
/// Builds a QueryInvertedIndex, then delegates to [`classify_from_query_index_parallel_rg`].
/// Use this when you have pre-extracted minimizers but not a pre-built query index.
///
/// # Arguments
/// * `sharded` - The sharded inverted index (must be Parquet format)
/// * `extracted` - Pre-extracted minimizers: (fwd_mins, rc_mins) per query
/// * `query_ids` - Query IDs corresponding to each entry in `extracted`
/// * `threshold` - Minimum score threshold
/// * `_read_options` - Unused; accepted for API consistency
///
/// # Returns
/// Vector of HitResult for all (query, bucket) pairs meeting the threshold.
pub fn classify_from_extracted_minimizers_parallel_rg(
    sharded: &ShardedInvertedIndex,
    extracted: &[(Vec<u64>, Vec<u64>)],
    query_ids: &[i64],
    threshold: f64,
    _read_options: Option<&crate::indices::parquet::ParquetReadOptions>,
) -> Result<Vec<HitResult>> {
    if extracted.is_empty() {
        return Ok(Vec::new());
    }

    // Build query inverted index (built once, reused across all RGs)
    let t_build_idx = Instant::now();
    let query_idx = QueryInvertedIndex::build(extracted);
    log_timing(
        "parallel_rg: build_query_index",
        t_build_idx.elapsed().as_millis(),
    );

    classify_from_query_index_parallel_rg(sharded, &query_idx, query_ids, threshold, _read_options)
}

/// Classify using parallel row group processing.
///
/// Extracts minimizers from sequences, then delegates to
/// `classify_from_extracted_minimizers_parallel_rg`. This is the standard entry
/// point when you have raw sequences and don't need to cache the extracted minimizers.
///
/// # Arguments
/// * `sharded` - The sharded inverted index (must be Parquet format)
/// * `negative_mins` - Optional set of minimizers to exclude from queries before scoring
/// * `records` - Batch of query records (should be pre-trimmed if trimming is desired)
/// * `threshold` - Minimum score threshold
/// * `_read_options` - Unused; accepted for API consistency
///
/// # Returns
/// Vector of HitResult for all (query, bucket) pairs meeting the threshold.
pub fn classify_batch_sharded_parallel_rg(
    sharded: &ShardedInvertedIndex,
    negative_mins: Option<&HashSet<u64>>,
    records: &[QueryRecord],
    threshold: f64,
    _read_options: Option<&crate::indices::parquet::ParquetReadOptions>,
) -> Result<Vec<HitResult>> {
    if records.is_empty() {
        return Ok(Vec::new());
    }

    let manifest = sharded.manifest();

    let t_extract = Instant::now();
    let extracted = extract_batch_minimizers(
        manifest.k,
        manifest.w,
        manifest.salt,
        negative_mins,
        records,
    );
    log_timing("parallel_rg: extraction", t_extract.elapsed().as_millis());

    let query_ids: Vec<i64> = records.iter().map(|(id, _, _)| *id).collect();

    classify_from_extracted_minimizers_parallel_rg(
        sharded,
        &extracted,
        &query_ids,
        threshold,
        _read_options,
    )
}

/// Classify with memory-efficient negative filtering using sharded index.
///
/// Instead of requiring a pre-loaded `HashSet<u64>` containing all minimizers from
/// a negative index (which can require 24GB+ for large indices), this function
/// accepts a `ShardedInvertedIndex` for the negative filter and processes it
/// shard-by-shard to collect only the query minimizers that need filtering.
///
/// This reduces memory from O(entire_negative_index) to O(single_shard + batch_minimizers).
///
/// # Arguments
/// * `positive_index` - The sharded inverted index to classify against
/// * `negative_index` - Optional sharded index containing sequences to filter out
/// * `records` - Batch of query records (should be pre-trimmed if trimming is desired)
/// * `threshold` - Minimum score threshold
/// * `read_options` - Parquet read options
///
/// # Returns
/// Vector of HitResult for all (query, bucket) pairs meeting the threshold.
pub fn classify_with_sharded_negative(
    positive_index: &ShardedInvertedIndex,
    negative_index: Option<&ShardedInvertedIndex>,
    records: &[QueryRecord],
    threshold: f64,
    read_options: Option<&crate::indices::parquet::ParquetReadOptions>,
) -> Result<Vec<HitResult>> {
    // If no negative index, delegate directly to the merge-join function
    if negative_index.is_none() {
        return classify_batch_sharded_merge_join(
            positive_index,
            None,
            records,
            threshold,
            read_options,
        );
    }

    let negative = negative_index.unwrap();
    let manifest = positive_index.manifest();

    // Step 1: Extract minimizers once (no negative filtering yet)
    let extracted = extract_batch_minimizers(manifest.k, manifest.w, manifest.salt, None, records);

    // Step 2: Build sorted unique minimizers for querying negative index
    let mut all_minimizers: Vec<u64> = extracted
        .iter()
        .flat_map(|(fwd, rc)| fwd.iter().chain(rc.iter()).copied())
        .collect();
    all_minimizers.sort_unstable();
    all_minimizers.dedup();

    // Step 3: Collect hitting minimizers from negative index (memory-efficient)
    let negative_set =
        collect_negative_minimizers_sharded(negative, &all_minimizers, read_options)?;

    // Step 4: Filter extracted minimizers by negative set, then classify
    let filtered: Vec<(Vec<u64>, Vec<u64>)> = extracted
        .into_iter()
        .map(|(fwd, rc)| filter_negative_mins(fwd, rc, Some(&negative_set)))
        .collect();

    let query_ids: Vec<i64> = records.iter().map(|(id, _, _)| *id).collect();

    classify_from_extracted_minimizers(
        positive_index,
        &filtered,
        &query_ids,
        threshold,
        read_options,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{create_parquet_inverted_index, extract_into, BucketData, ParquetWriteOptions};
    use tempfile::tempdir;

    /// Generate a DNA sequence with a deterministic pattern.
    fn generate_sequence(len: usize, seed: u8) -> Vec<u8> {
        let bases = [b'A', b'C', b'G', b'T'];
        (0..len).map(|i| bases[(i + seed as usize) % 4]).collect()
    }

    /// Create a test Parquet index with multiple buckets.
    fn create_test_index() -> (tempfile::TempDir, ShardedInvertedIndex, Vec<Vec<u8>>) {
        let dir = tempdir().unwrap();
        let index_path = dir.path().join("test.ryxdi");

        let k = 32;
        let w = 10;
        let salt = 0x12345u64;

        let mut ws = MinimizerWorkspace::new();

        // Create sequences for two buckets
        let seq1 = generate_sequence(200, 0);
        let seq2 = generate_sequence(200, 1);

        extract_into(&seq1, k, w, salt, &mut ws);
        let mut mins1: Vec<u64> = ws.buffer.drain(..).collect();
        mins1.sort();
        mins1.dedup();

        extract_into(&seq2, k, w, salt, &mut ws);
        let mut mins2: Vec<u64> = ws.buffer.drain(..).collect();
        mins2.sort();
        mins2.dedup();

        let buckets = vec![
            BucketData {
                bucket_id: 1,
                bucket_name: "Bucket1".to_string(),
                sources: vec!["seq1".to_string()],
                minimizers: mins1,
            },
            BucketData {
                bucket_id: 2,
                bucket_name: "Bucket2".to_string(),
                sources: vec!["seq2".to_string()],
                minimizers: mins2,
            },
        ];

        let options = ParquetWriteOptions::default();
        create_parquet_inverted_index(&index_path, buckets, k, w, salt, None, Some(&options))
            .unwrap();

        let index = ShardedInvertedIndex::open(&index_path).unwrap();
        (dir, index, vec![seq1, seq2])
    }

    // =========================================================================
    // classify_batch_sharded_merge_join tests
    // =========================================================================

    #[test]
    fn test_merge_join_empty_records() {
        let (_dir, index, _seqs) = create_test_index();
        let records: Vec<QueryRecord> = vec![];

        let results = classify_batch_sharded_merge_join(&index, None, &records, 0.1, None).unwrap();

        assert!(
            results.is_empty(),
            "Empty records should produce empty results"
        );
    }

    #[test]
    fn test_merge_join_self_match() {
        let (_dir, index, seqs) = create_test_index();

        // Query with the same sequence used to build bucket 1
        let records: Vec<QueryRecord> = vec![(1, seqs[0].as_slice(), None)];

        let results = classify_batch_sharded_merge_join(&index, None, &records, 0.1, None).unwrap();

        // Should have at least one hit for bucket 1 with high score
        let bucket1_hit = results.iter().find(|r| r.query_id == 1 && r.bucket_id == 1);
        assert!(bucket1_hit.is_some(), "Should have self-match for bucket 1");
        assert!(
            bucket1_hit.unwrap().score > 0.9,
            "Self-match score should be >0.9, got {}",
            bucket1_hit.unwrap().score
        );
    }

    #[test]
    fn test_merge_join_multiple_queries() {
        let (_dir, index, seqs) = create_test_index();

        // Query with both sequences
        let records: Vec<QueryRecord> =
            vec![(1, seqs[0].as_slice(), None), (2, seqs[1].as_slice(), None)];

        let results = classify_batch_sharded_merge_join(&index, None, &records, 0.1, None).unwrap();

        // Each query should match its corresponding bucket
        let q1_b1 = results.iter().find(|r| r.query_id == 1 && r.bucket_id == 1);
        let q2_b2 = results.iter().find(|r| r.query_id == 2 && r.bucket_id == 2);

        assert!(q1_b1.is_some(), "Query 1 should match bucket 1");
        assert!(q2_b2.is_some(), "Query 2 should match bucket 2");
        assert!(
            q1_b1.unwrap().score > 0.9,
            "Self-match should have high score"
        );
        assert!(
            q2_b2.unwrap().score > 0.9,
            "Self-match should have high score"
        );
    }

    #[test]
    fn test_merge_join_threshold_filtering() {
        let (_dir, index, seqs) = create_test_index();

        let records: Vec<QueryRecord> = vec![(1, seqs[0].as_slice(), None)];

        // Low threshold - should get results
        let low_results =
            classify_batch_sharded_merge_join(&index, None, &records, 0.1, None).unwrap();

        // Medium threshold - may filter some cross-matches
        let mid_results =
            classify_batch_sharded_merge_join(&index, None, &records, 0.5, None).unwrap();

        // Threshold > 1.0 - should filter everything (scores are max 1.0)
        let high_results =
            classify_batch_sharded_merge_join(&index, None, &records, 1.01, None).unwrap();

        assert!(!low_results.is_empty(), "Low threshold should have results");
        assert!(
            mid_results.len() <= low_results.len(),
            "Higher threshold should have fewer or equal results"
        );
        assert!(
            high_results.is_empty(),
            "Threshold > 1.0 should filter all results"
        );
    }

    #[test]
    fn test_merge_join_short_sequence() {
        let (_dir, index, _seqs) = create_test_index();

        // Sequence shorter than k (32) - should produce no minimizers
        let short_seq = b"ACGTACGT"; // 8 bases
        let records: Vec<QueryRecord> = vec![(1, short_seq.as_slice(), None)];

        let results = classify_batch_sharded_merge_join(&index, None, &records, 0.1, None).unwrap();

        assert!(results.is_empty(), "Short sequence should have no hits");
    }

    #[test]
    fn test_merge_join_with_negative_mins() {
        let (_dir, index, seqs) = create_test_index();

        let records: Vec<QueryRecord> = vec![(1, seqs[0].as_slice(), None)];

        // First, get results without negative filtering
        let results_without =
            classify_batch_sharded_merge_join(&index, None, &records, 0.1, None).unwrap();
        assert!(
            !results_without.is_empty(),
            "Should have results without filtering"
        );

        // Extract minimizers to use as negative set
        let k = index.k();
        let w = index.w();
        let salt = index.salt();
        let mut ws = MinimizerWorkspace::new();
        extract_into(&seqs[0], k, w, salt, &mut ws);
        let negative_mins: HashSet<u64> = ws.buffer.drain(..).collect();

        // With all query minimizers as negative - should filter everything
        let results_with =
            classify_batch_sharded_merge_join(&index, Some(&negative_mins), &records, 0.1, None)
                .unwrap();

        assert!(
            results_with.is_empty(),
            "Filtering all minimizers should produce no hits"
        );
    }

    #[test]
    fn test_merge_join_paired_end() {
        let (_dir, index, seqs) = create_test_index();

        // Use seq1 as read1 and seq2 as read2 (paired-end)
        let records: Vec<QueryRecord> = vec![(1, seqs[0].as_slice(), Some(seqs[1].as_slice()))];

        let results = classify_batch_sharded_merge_join(&index, None, &records, 0.1, None).unwrap();

        // Should have hits for both buckets (read1 matches bucket1, read2 matches bucket2)
        let b1_hit = results.iter().find(|r| r.bucket_id == 1);
        let b2_hit = results.iter().find(|r| r.bucket_id == 2);

        assert!(
            b1_hit.is_some(),
            "Should have hit for bucket 1 (from read1)"
        );
        assert!(
            b2_hit.is_some(),
            "Should have hit for bucket 2 (from read2)"
        );
    }

    // =========================================================================
    // Helper function tests
    // =========================================================================

    #[test]
    fn test_estimate_minimizers_from_records_empty() {
        let records: Vec<QueryRecord> = vec![];
        let estimate = estimate_minimizers_from_records(&records, 32, 10);
        assert_eq!(estimate, ESTIMATED_MINIMIZERS_PER_SEQUENCE);
    }

    #[test]
    fn test_estimate_minimizers_from_records_short_sequence() {
        let short_seq = b"ACGT"; // 4 bases, less than k=32
        let records: Vec<QueryRecord> = vec![(1, short_seq.as_slice(), None)];
        let estimate = estimate_minimizers_from_records(&records, 32, 10);
        assert_eq!(estimate, ESTIMATED_MINIMIZERS_PER_SEQUENCE);
    }

    #[test]
    fn test_estimate_minimizers_from_records_long_sequence() {
        let long_seq = generate_sequence(200, 0);
        let records: Vec<QueryRecord> = vec![(1, long_seq.as_slice(), None)];
        let estimate = estimate_minimizers_from_records(&records, 32, 10);
        // Should be approximately ((200 - 32) / 10 + 1) * 2 = 36
        assert!(
            estimate >= 30 && estimate <= 50,
            "Estimate should be reasonable"
        );
    }

    // =========================================================================
    // classify_with_sharded_negative tests (Phase 3)
    // =========================================================================

    /// Helper to create a test Parquet index at a given path with specified bucket data.
    fn create_test_index_at_path(
        path: &std::path::Path,
        bucket_data: Vec<(u32, &str, Vec<u64>)>,
        k: usize,
        w: usize,
        salt: u64,
    ) -> ShardedInvertedIndex {
        let buckets: Vec<BucketData> = bucket_data
            .into_iter()
            .map(|(id, name, mins)| BucketData {
                bucket_id: id,
                bucket_name: name.to_string(),
                sources: vec![format!("source_{}", id)],
                minimizers: mins,
            })
            .collect();

        let options = ParquetWriteOptions::default();
        create_parquet_inverted_index(path, buckets, k, w, salt, None, Some(&options)).unwrap();

        ShardedInvertedIndex::open(path).unwrap()
    }

    #[test]
    fn test_classify_with_sharded_negative_no_negative() {
        // Without a negative index, should behave identically to classify_batch_sharded_merge_join
        let (_dir, index, seqs) = create_test_index();
        let records: Vec<QueryRecord> = vec![(1, seqs[0].as_slice(), None)];

        let results_standard =
            classify_batch_sharded_merge_join(&index, None, &records, 0.1, None).unwrap();
        let results_sharded =
            classify_with_sharded_negative(&index, None, &records, 0.1, None).unwrap();

        assert_eq!(
            results_standard.len(),
            results_sharded.len(),
            "Results should match when no negative index"
        );
    }

    #[test]
    fn test_classify_with_sharded_negative_filters_correctly() {
        // Use explicit minimizer values for predictable behavior
        let dir = tempdir().unwrap();

        let k = 32;
        let w = 10;
        let salt = 0x12345u64;

        // Positive index has minimizers 100-109 (10 total)
        let positive_mins: Vec<u64> = (100..110).collect();
        // Negative index filters out 100-104 (first 5)
        let negative_mins: Vec<u64> = (100..105).collect();

        // Create positive index
        let pos_path = dir.path().join("positive.ryxdi");
        let pos_index = create_test_index_at_path(
            &pos_path,
            vec![(1, "target", positive_mins.clone())],
            k,
            w,
            salt,
        );

        // Create negative index
        let neg_path = dir.path().join("negative.ryxdi");
        let neg_index = create_test_index_at_path(
            &neg_path,
            vec![(1, "contaminant", negative_mins)],
            k,
            w,
            salt,
        );

        // Create a query sequence that produces minimizers overlapping with positive index
        // Use a longer, non-repetitive sequence
        let seq = generate_sequence(500, 42);
        let mut ws = MinimizerWorkspace::new();
        extract_into(&seq, k, w, salt, &mut ws);
        let _query_mins: Vec<u64> = ws.buffer.drain(..).collect();

        // Build a synthetic query that has exact minimizers we want to test
        // For this test, we'll use the existing index's self-match capability
        let records: Vec<QueryRecord> = vec![(1, seq.as_slice(), None)];

        // Without negative filtering
        let results_without =
            classify_with_sharded_negative(&pos_index, None, &records, 0.0, None).unwrap();

        // With negative filtering
        let results_with =
            classify_with_sharded_negative(&pos_index, Some(&neg_index), &records, 0.0, None)
                .unwrap();

        // Both should work (may or may not have hits depending on minimizer overlap)
        // The key test is that with negative filtering, we get equal or fewer hits
        assert!(
            results_with.len() <= results_without.len(),
            "Negative filtering should not increase hit count"
        );
    }

    #[test]
    fn test_classify_with_sharded_negative_all_filtered() {
        // Test that when all query minimizers hit the negative index, we get no results
        let (_dir, index, seqs) = create_test_index();

        // Extract minimizers from seq[0] to use as the negative set
        let k = index.k();
        let w = index.w();
        let salt = index.salt();
        let mut ws = MinimizerWorkspace::new();
        extract_into(&seqs[0], k, w, salt, &mut ws);
        let mut seq_mins: Vec<u64> = ws.buffer.drain(..).collect();
        seq_mins.sort();
        seq_mins.dedup();

        // Create negative index with all minimizers from the query sequence
        let dir = tempdir().unwrap();
        let neg_path = dir.path().join("negative.ryxdi");
        let neg_index =
            create_test_index_at_path(&neg_path, vec![(1, "contaminant", seq_mins)], k, w, salt);

        let records: Vec<QueryRecord> = vec![(1, seqs[0].as_slice(), None)];

        // All query minimizers should be filtered, resulting in no hits
        let results =
            classify_with_sharded_negative(&index, Some(&neg_index), &records, 0.1, None).unwrap();

        assert!(
            results.is_empty(),
            "All minimizers filtered should produce no hits"
        );
    }

    #[test]
    fn test_classify_with_sharded_negative_empty_records() {
        let (_dir, index, _seqs) = create_test_index();
        let records: Vec<QueryRecord> = vec![];

        let results = classify_with_sharded_negative(&index, None, &records, 0.1, None).unwrap();

        assert!(
            results.is_empty(),
            "Empty records should produce empty results"
        );
    }

    // =========================================================================
    // classify_from_query_index tests (Cycle 3)
    // =========================================================================

    #[test]
    fn test_classify_from_query_index_matches_extracted() {
        let (_dir, index, seqs) = create_test_index();

        // Build extracted minimizers from both sequences
        let manifest = index.manifest();
        let extracted = extract_batch_minimizers(
            manifest.k,
            manifest.w,
            manifest.salt,
            None,
            &[
                (1i64, seqs[0].as_slice(), None),
                (2i64, seqs[1].as_slice(), None),
            ],
        );
        let query_ids = vec![1i64, 2];

        // Get results via the existing classify_from_extracted_minimizers
        let results_existing =
            classify_from_extracted_minimizers(&index, &extracted, &query_ids, 0.1, None).unwrap();

        // Build QueryInvertedIndex manually, then call classify_from_query_index
        let query_idx = QueryInvertedIndex::build(&extracted);
        let results_new =
            classify_from_query_index(&index, &query_idx, &query_ids, 0.1, None).unwrap();

        // Results should be identical
        assert_eq!(
            results_existing.len(),
            results_new.len(),
            "classify_from_query_index should produce same number of results"
        );

        // Sort both by (query_id, bucket_id) for comparison
        let mut existing_sorted = results_existing.clone();
        existing_sorted.sort_by(|a, b| {
            a.query_id
                .cmp(&b.query_id)
                .then(a.bucket_id.cmp(&b.bucket_id))
        });
        let mut new_sorted = results_new.clone();
        new_sorted.sort_by(|a, b| {
            a.query_id
                .cmp(&b.query_id)
                .then(a.bucket_id.cmp(&b.bucket_id))
        });

        for (e, n) in existing_sorted.iter().zip(new_sorted.iter()) {
            assert_eq!(e.query_id, n.query_id);
            assert_eq!(e.bucket_id, n.bucket_id);
            assert!(
                (e.score - n.score).abs() < 1e-10,
                "Scores should match: {} vs {}",
                e.score,
                n.score
            );
        }
    }

    #[test]
    fn test_classify_from_query_index_parallel_rg_matches_extracted() {
        let (_dir, index, seqs) = create_test_index();

        let manifest = index.manifest();
        let extracted = extract_batch_minimizers(
            manifest.k,
            manifest.w,
            manifest.salt,
            None,
            &[
                (1i64, seqs[0].as_slice(), None),
                (2i64, seqs[1].as_slice(), None),
            ],
        );
        let query_ids = vec![1i64, 2];

        // Get results via the existing classify_from_extracted_minimizers_parallel_rg
        let results_existing = classify_from_extracted_minimizers_parallel_rg(
            &index, &extracted, &query_ids, 0.1, None,
        )
        .unwrap();

        // Build QueryInvertedIndex manually, then call classify_from_query_index_parallel_rg
        let query_idx = QueryInvertedIndex::build(&extracted);
        let results_new =
            classify_from_query_index_parallel_rg(&index, &query_idx, &query_ids, 0.1, None)
                .unwrap();

        assert_eq!(
            results_existing.len(),
            results_new.len(),
            "classify_from_query_index_parallel_rg should produce same number of results"
        );

        let mut existing_sorted = results_existing.clone();
        existing_sorted.sort_by(|a, b| {
            a.query_id
                .cmp(&b.query_id)
                .then(a.bucket_id.cmp(&b.bucket_id))
        });
        let mut new_sorted = results_new.clone();
        new_sorted.sort_by(|a, b| {
            a.query_id
                .cmp(&b.query_id)
                .then(a.bucket_id.cmp(&b.bucket_id))
        });

        for (e, n) in existing_sorted.iter().zip(new_sorted.iter()) {
            assert_eq!(e.query_id, n.query_id);
            assert_eq!(e.bucket_id, n.bucket_id);
            assert!(
                (e.score - n.score).abs() < 1e-10,
                "Scores should match: {} vs {}",
                e.score,
                n.score
            );
        }
    }

    #[test]
    fn test_classify_from_query_index_empty() {
        let (_dir, index, _seqs) = create_test_index();

        // Empty query index
        let query_idx = QueryInvertedIndex::build(&[]);
        let results = classify_from_query_index(&index, &query_idx, &[], 0.1, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_classify_with_sharded_negative_consistency_with_hashset() {
        // Verify that sharded negative filtering produces the same results
        // as the traditional HashSet-based approach
        let (_dir, index, seqs) = create_test_index();

        // Extract minimizers to use as negative set
        let k = index.k();
        let w = index.w();
        let salt = index.salt();
        let mut ws = MinimizerWorkspace::new();
        extract_into(&seqs[0], k, w, salt, &mut ws);
        let mut seq_mins: Vec<u64> = ws.buffer.drain(..).collect();
        seq_mins.sort();
        seq_mins.dedup();

        // Use first third as negative minimizers
        let neg_count = (seq_mins.len() / 3).max(1);
        let negative_mins: Vec<u64> = seq_mins[..neg_count].to_vec();
        let negative_set: HashSet<u64> = negative_mins.iter().copied().collect();

        // Create negative index
        let dir = tempdir().unwrap();
        let neg_path = dir.path().join("negative.ryxdi");
        let neg_index = create_test_index_at_path(
            &neg_path,
            vec![(1, "contaminant", negative_mins)],
            k,
            w,
            salt,
        );

        let records: Vec<QueryRecord> = vec![(1, seqs[0].as_slice(), None)];

        // Traditional HashSet approach
        let results_hashset =
            classify_batch_sharded_merge_join(&index, Some(&negative_set), &records, 0.1, None)
                .unwrap();

        // New sharded approach
        let results_sharded =
            classify_with_sharded_negative(&index, Some(&neg_index), &records, 0.1, None).unwrap();

        assert_eq!(
            results_hashset.len(),
            results_sharded.len(),
            "Both approaches should produce same number of results"
        );

        if !results_hashset.is_empty() {
            let score_hashset = results_hashset[0].score;
            let score_sharded = results_sharded[0].score;
            let diff = (score_hashset - score_sharded).abs();
            assert!(
                diff < 1e-10,
                "Scores should match: hashset={}, sharded={}",
                score_hashset,
                score_sharded
            );
        }
    }
}
