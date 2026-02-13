//! Merge-join classification algorithms.
//!
//! Provides efficient classification using sorted merge-join between query and
//! reference inverted indices.
//!
//! ## Accumulator Strategy
//!
//! Two accumulator implementations are available:
//! - `DenseAccumulator`: Flat array indexed by `read_idx * stride + bucket_id`.
//!   Optimal for indices with few buckets (≤ DENSE_ACCUMULATOR_MAX_BUCKETS).
//!   Eliminates HashMap overhead (hashing, probing, allocation) in the hot path.
//! - `SparseAccumulator`: Per-read HashMaps. Used for indices with many buckets
//!   where a dense array would waste memory.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::constants::{ESTIMATED_BUCKETS_PER_READ, GALLOP_THRESHOLD, MIN_PARALLEL_SHARD_SIZE};
use crate::core::gallop_for_each;
use crate::indices::{InvertedIndex, QueryInvertedIndex};
use crate::types::HitResult;

use super::scoring::compute_score;

// ============================================================================
// HitAccumulator trait and implementations
// ============================================================================

/// Trait for accumulating merge-join hits.
///
/// Provides an abstraction over dense (array-based) and sparse (HashMap-based)
/// accumulation strategies. Dense accumulators are faster for indices with few
/// buckets, while sparse accumulators handle arbitrary bucket counts.
pub(super) trait HitAccumulator: Sized + Send {
    /// Record a single hit for a read against a bucket.
    fn record_hit(&mut self, read_idx: usize, bucket_id: u32, is_rc: bool);

    /// Record pre-counted hits (from sparse hit merging).
    fn record_hit_counts(&mut self, read_idx: usize, bucket_id: u32, fwd: u32, rc: u32);

    /// Merge another accumulator into this one (for parallel reduce).
    fn merge(&mut self, other: Self);

    /// Score all accumulated hits and filter by threshold.
    fn score_and_filter(
        self,
        query_idx: &QueryInvertedIndex,
        query_ids: &[i64],
        threshold: f64,
    ) -> Vec<HitResult>;
}

/// Dense accumulator using a flat array indexed by `read_idx * stride + bucket_id`.
///
/// Optimal for indices with few buckets (≤ DENSE_ACCUMULATOR_MAX_BUCKETS).
/// Single-bucket log-ratio gets 16 bytes/read instead of ~200 bytes/read HashMaps.
pub(super) struct DenseAccumulator {
    data: Vec<(u32, u32)>,
    stride: usize,
    num_reads: usize,
    max_bucket_id: u32,
}

impl DenseAccumulator {
    pub(super) fn new(num_reads: usize, max_bucket_id: u32) -> Self {
        let stride = max_bucket_id as usize + 1;
        Self {
            data: vec![(0, 0); num_reads * stride],
            stride,
            num_reads,
            max_bucket_id,
        }
    }
}

impl HitAccumulator for DenseAccumulator {
    #[inline]
    fn record_hit(&mut self, read_idx: usize, bucket_id: u32, is_rc: bool) {
        let idx = read_idx * self.stride + bucket_id as usize;
        let entry = &mut self.data[idx];
        if is_rc {
            entry.1 = entry.1.wrapping_add(1);
        } else {
            entry.0 = entry.0.wrapping_add(1);
        }
    }

    #[inline]
    fn record_hit_counts(&mut self, read_idx: usize, bucket_id: u32, fwd: u32, rc: u32) {
        let idx = read_idx * self.stride + bucket_id as usize;
        let entry = &mut self.data[idx];
        entry.0 = entry.0.wrapping_add(fwd);
        entry.1 = entry.1.wrapping_add(rc);
    }

    fn merge(&mut self, other: Self) {
        assert_eq!(
            self.data.len(),
            other.data.len(),
            "DenseAccumulator merge: mismatched lengths ({} vs {})",
            self.data.len(),
            other.data.len()
        );
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            a.0 = a.0.wrapping_add(b.0);
            a.1 = a.1.wrapping_add(b.1);
        }
    }

    fn score_and_filter(
        self,
        query_idx: &QueryInvertedIndex,
        query_ids: &[i64],
        threshold: f64,
    ) -> Vec<HitResult> {
        let mut results = Vec::new();
        for (read_idx, &query_id) in query_ids.iter().enumerate().take(self.num_reads) {
            let fwd_total = query_idx.fwd_counts[read_idx] as usize;
            let rc_total = query_idx.rc_counts[read_idx] as usize;
            let base = read_idx * self.stride;
            for bucket_id in 1..=self.max_bucket_id {
                let (fwd_hits, rc_hits) = self.data[base + bucket_id as usize];
                if fwd_hits > 0 || rc_hits > 0 {
                    let score =
                        compute_score(fwd_hits as usize, fwd_total, rc_hits as usize, rc_total);
                    if score >= threshold {
                        results.push(HitResult {
                            query_id,
                            bucket_id,
                            score,
                        });
                    }
                }
            }
        }
        results
    }
}

/// Sparse accumulator wrapping per-read HashMaps.
///
/// Used for indices with many buckets (> DENSE_ACCUMULATOR_MAX_BUCKETS) where
/// a dense array would waste memory. Also serves as the fallback for edge cases.
pub(super) struct SparseAccumulator {
    accumulators: Vec<HashMap<u32, (u32, u32)>>,
}

impl SparseAccumulator {
    pub(super) fn new(num_reads: usize) -> Self {
        Self {
            accumulators: (0..num_reads)
                .map(|_| HashMap::with_capacity(ESTIMATED_BUCKETS_PER_READ))
                .collect(),
        }
    }
}

impl HitAccumulator for SparseAccumulator {
    #[inline]
    fn record_hit(&mut self, read_idx: usize, bucket_id: u32, is_rc: bool) {
        let entry = self.accumulators[read_idx]
            .entry(bucket_id)
            .or_insert((0, 0));
        if is_rc {
            entry.1 = entry.1.wrapping_add(1);
        } else {
            entry.0 = entry.0.wrapping_add(1);
        }
    }

    #[inline]
    fn record_hit_counts(&mut self, read_idx: usize, bucket_id: u32, fwd: u32, rc: u32) {
        let entry = self.accumulators[read_idx]
            .entry(bucket_id)
            .or_insert((0, 0));
        entry.0 += fwd;
        entry.1 += rc;
    }

    fn merge(&mut self, other: Self) {
        for (i, map) in other.accumulators.into_iter().enumerate() {
            for (bucket_id, (fwd, rc)) in map {
                let entry = self.accumulators[i].entry(bucket_id).or_insert((0, 0));
                entry.0 += fwd;
                entry.1 += rc;
            }
        }
    }

    fn score_and_filter(
        self,
        query_idx: &QueryInvertedIndex,
        query_ids: &[i64],
        threshold: f64,
    ) -> Vec<HitResult> {
        let mut results = Vec::new();
        for (read_idx, buckets) in self.accumulators.into_iter().enumerate() {
            let fwd_total = query_idx.fwd_counts[read_idx] as usize;
            let rc_total = query_idx.rc_counts[read_idx] as usize;
            let query_id = query_ids[read_idx];

            for (bucket_id, (fwd_hits, rc_hits)) in buckets {
                let score = compute_score(fwd_hits as usize, fwd_total, rc_hits as usize, rc_total);
                if score >= threshold {
                    results.push(HitResult {
                        query_id,
                        bucket_id,
                        score,
                    });
                }
            }
        }
        results
    }
}

// ============================================================================
// Generic merge-join functions
// ============================================================================

// ============================================================================
// COO merge-join
// ============================================================================

/// Merge-join query COO entries against reference COO pairs using both-side run detection.
///
/// Both inputs are sorted by minimizer. The algorithm detects runs (consecutive
/// entries with the same minimizer) on both sides, then cross-products matching
/// runs: each query entry × each ref bucket. This is O(Q + R) outer comparisons
/// plus O(hits) for the cross-product.
///
/// Eliminates the need for CSR conversion, galloping dispatch, and partition_point
/// calls that the old `accumulate_merge_join` required.
///
/// # Arguments
/// * `query_idx` - Query inverted index (COO format)
/// * `ref_pairs` - Sorted (minimizer, bucket_id) pairs from a shard or InvertedIndex
/// * `accumulator` - Hit accumulator (dense or sparse)
pub(super) fn merge_join_coo<A: HitAccumulator>(
    query_idx: &QueryInvertedIndex,
    ref_pairs: &[(u64, u32)],
    accumulator: &mut A,
) {
    let entries = &query_idx.entries;
    if entries.is_empty() || ref_pairs.is_empty() {
        return;
    }

    let mut qi = 0usize;
    let mut ri = 0usize;

    while qi < entries.len() && ri < ref_pairs.len() {
        let q_min = entries[qi].0;
        let r_min = ref_pairs[ri].0;

        if q_min < r_min {
            // Skip query run
            let run_end = qi + entries[qi..].partition_point(|e| e.0 == q_min);
            qi = run_end;
        } else if q_min > r_min {
            // Skip ref run
            let run_end = ri + ref_pairs[ri..].partition_point(|e| e.0 == r_min);
            ri = run_end;
        } else {
            // Match! Find both runs.
            let q_run_end = qi + entries[qi..].partition_point(|e| e.0 == q_min);
            let r_run_end = ri + ref_pairs[ri..].partition_point(|e| e.0 == r_min);

            // Cross-product: each query entry × each ref bucket
            for &(_, packed) in &entries[qi..q_run_end] {
                let (read_idx, is_rc) = QueryInvertedIndex::unpack_read_id(packed);
                for &(_, bucket_id) in &ref_pairs[ri..r_run_end] {
                    accumulator.record_hit(read_idx as usize, bucket_id, is_rc);
                }
            }

            qi = q_run_end;
            ri = r_run_end;
        }
    }
}

// ============================================================================
// Parallel COO merge-join
// ============================================================================

/// Merge-join a slice of query COO entries against a slice of reference COO pairs.
///
/// Same algorithm as `merge_join_coo` but operates on raw slices (not
/// `QueryInvertedIndex`) and produces `SparseHit` vectors instead of writing
/// to an accumulator. Used by `merge_join_coo_parallel` for per-chunk processing.
fn merge_join_coo_slice(entries: &[(u64, u32)], ref_pairs: &[(u64, u32)]) -> Vec<SparseHit> {
    if entries.is_empty() || ref_pairs.is_empty() {
        return Vec::new();
    }

    let mut hits = Vec::new();
    let mut qi = 0usize;
    let mut ri = 0usize;

    while qi < entries.len() && ri < ref_pairs.len() {
        let q_min = entries[qi].0;
        let r_min = ref_pairs[ri].0;

        if q_min < r_min {
            let run_end = qi + entries[qi..].partition_point(|e| e.0 == q_min);
            qi = run_end;
        } else if q_min > r_min {
            let run_end = ri + ref_pairs[ri..].partition_point(|e| e.0 == r_min);
            ri = run_end;
        } else {
            let q_run_end = qi + entries[qi..].partition_point(|e| e.0 == q_min);
            let r_run_end = ri + ref_pairs[ri..].partition_point(|e| e.0 == r_min);

            for &(_, packed) in &entries[qi..q_run_end] {
                let (read_idx, is_rc) = QueryInvertedIndex::unpack_read_id(packed);
                for &(_, bucket_id) in &ref_pairs[ri..r_run_end] {
                    if is_rc {
                        hits.push((read_idx, bucket_id, 0, 1));
                    } else {
                        hits.push((read_idx, bucket_id, 1, 0));
                    }
                }
            }

            qi = q_run_end;
            ri = r_run_end;
        }
    }

    hits
}

/// Compute chunk ranges that split entries at minimizer boundaries.
///
/// Returns `(start, end)` pairs covering all entries. No run of identical
/// minimizers is split across chunks. May return fewer chunks than requested
/// if entries have few unique minimizers.
fn compute_chunk_ranges(entries: &[(u64, u32)], num_chunks: usize) -> Vec<(usize, usize)> {
    if entries.is_empty() || num_chunks == 0 {
        return Vec::new();
    }
    if num_chunks == 1 {
        return vec![(0, entries.len())];
    }

    let target_size = entries.len() / num_chunks;
    if target_size == 0 {
        return vec![(0, entries.len())];
    }

    let mut ranges = Vec::with_capacity(num_chunks);
    let mut start = 0;

    for i in 1..num_chunks {
        let target = i * target_size;
        if target >= entries.len() {
            break;
        }

        // Find the end of the run at the target position
        let min_at_target = entries[target].0;
        let end = target + entries[target..].partition_point(|e| e.0 == min_at_target);

        if end > start && end < entries.len() {
            ranges.push((start, end));
            start = end;
        }
    }

    // Last chunk covers remaining entries
    if start < entries.len() {
        ranges.push((start, entries.len()));
    }

    ranges
}

/// Parallel merge-join of query COO entries against reference COO pairs.
///
/// Splits query entries into chunks by minimizer range, processes each chunk
/// in parallel using rayon, then merges sparse hits into the accumulator.
/// Falls back to single-threaded `merge_join_coo` when the shard is too small
/// or only one thread is available.
///
/// # Arguments
/// * `query_idx` - Query inverted index (COO format)
/// * `ref_pairs` - Sorted (minimizer, bucket_id) pairs from a shard
/// * `accumulator` - Hit accumulator (dense or sparse)
pub(super) fn merge_join_coo_parallel<A: HitAccumulator>(
    query_idx: &QueryInvertedIndex,
    ref_pairs: &[(u64, u32)],
    accumulator: &mut A,
) {
    let entries = &query_idx.entries;
    if entries.is_empty() || ref_pairs.is_empty() {
        return;
    }

    let num_threads = rayon::current_num_threads();

    // Fall back to single-threaded for small shards or single thread
    if num_threads <= 1 || ref_pairs.len() < MIN_PARALLEL_SHARD_SIZE {
        merge_join_coo(query_idx, ref_pairs, accumulator);
        return;
    }

    // Split query entries into chunks at minimizer boundaries
    let ranges = compute_chunk_ranges(entries, num_threads);

    if ranges.len() <= 1 {
        // Single chunk (e.g., all entries share one minimizer) — no parallelism
        merge_join_coo(query_idx, ref_pairs, accumulator);
        return;
    }

    // Parallel merge-join: each chunk binary-searches into ref_pairs
    let all_hits: Vec<Vec<SparseHit>> = ranges
        .into_par_iter()
        .map(|(q_start, q_end)| {
            let chunk = &entries[q_start..q_end];
            let min_min = chunk[0].0;
            let max_min = chunk[chunk.len() - 1].0;

            // Binary search into ref_pairs for the matching range
            let r_start = ref_pairs.partition_point(|e| e.0 < min_min);
            let r_end = ref_pairs.partition_point(|e| e.0 <= max_min);

            if r_start >= r_end {
                return Vec::new();
            }

            merge_join_coo_slice(chunk, &ref_pairs[r_start..r_end])
        })
        .collect();

    // Merge sparse hits into accumulator (single-threaded)
    for chunk_hits in all_hits {
        for (read_idx, bucket_id, fwd, rc) in chunk_hits {
            accumulator.record_hit_counts(read_idx as usize, bucket_id, fwd, rc);
        }
    }
}

// ============================================================================
// CSR merge-join (for multi-bucket indices where COO iterates too many pairs)
// ============================================================================

/// Accumulate hits from a query COO run against a CSR bucket slice.
///
/// Cross-products each query entry with each bucket in the slice, recording
/// hits via the accumulator trait.
#[inline]
fn accumulate_coo_run_csr<A: HitAccumulator>(
    entries: &[(u64, u32)],
    bucket_slice: &[u32],
    accumulator: &mut A,
) {
    for &(_, packed) in entries {
        let (read_idx, is_rc) = QueryInvertedIndex::unpack_read_id(packed);
        for &bucket_id in bucket_slice {
            accumulator.record_hit(read_idx as usize, bucket_id, is_rc);
        }
    }
}

/// CSR linear merge-join for similar-sized query and reference indices.
///
/// Walks query COO entries and reference CSR minimizers in parallel.
/// O(Q_unique + R) outer comparisons where R is the number of unique
/// minimizers in the reference (much smaller than the COO representation
/// for multi-bucket indices).
fn merge_join_csr_linear<A: HitAccumulator>(
    query_idx: &QueryInvertedIndex,
    ref_idx: &InvertedIndex,
    accumulator: &mut A,
) {
    let entries = &query_idx.entries;
    let mut qi = 0usize;
    let mut ri = 0usize;

    while qi < entries.len() && ri < ref_idx.minimizers.len() {
        let q_min = entries[qi].0;
        let r_min = ref_idx.minimizers[ri];

        if q_min < r_min {
            // Skip entire COO run — advance qi past all entries with q_min
            qi = entries[qi..].partition_point(|e| e.0 == q_min) + qi;
        } else if q_min > r_min {
            ri += 1;
        } else {
            // Match! Find end of COO run for this minimizer.
            let run_end = entries[qi..].partition_point(|e| e.0 == q_min) + qi;

            // Get ref bucket slice for this minimizer.
            let r_start = ref_idx.offsets[ri] as usize;
            let r_end = ref_idx.offsets[ri + 1] as usize;
            let bucket_slice = &ref_idx.bucket_ids[r_start..r_end];

            // Cross-product: each COO entry × each ref bucket
            accumulate_coo_run_csr(&entries[qi..run_end], bucket_slice, accumulator);

            qi = run_end;
            ri += 1;
        }
    }
}

/// CSR galloping search for skewed size ratios.
///
/// Uses pre-computed `unique_mins` for the galloping outer loop,
/// then finds COO runs via `partition_point` on match.
fn gallop_join_csr<A: HitAccumulator>(
    query_idx: &QueryInvertedIndex,
    ref_idx: &InvertedIndex,
    accumulator: &mut A,
    unique_mins: &[u64],
    query_smaller: bool,
) {
    let (smaller, larger) = if query_smaller {
        (unique_mins, &ref_idx.minimizers[..])
    } else {
        (&ref_idx.minimizers[..], unique_mins)
    };

    gallop_for_each(smaller, larger, |smaller_idx, larger_idx| {
        let (qi_unique, ri) = if query_smaller {
            (smaller_idx, larger_idx)
        } else {
            (larger_idx, smaller_idx)
        };

        // Find COO run for unique_mins[qi_unique] using two partition_points
        let target = unique_mins[qi_unique];
        let run_start = query_idx.entries.partition_point(|e| e.0 < target);
        let run_end = query_idx.entries.partition_point(|e| e.0 <= target);

        let r_start = ref_idx.offsets[ri] as usize;
        let r_end = ref_idx.offsets[ri + 1] as usize;
        let bucket_slice = &ref_idx.bucket_ids[r_start..r_end];

        accumulate_coo_run_csr(
            &query_idx.entries[run_start..run_end],
            bucket_slice,
            accumulator,
        );
    });
}

/// CSR merge-join dispatcher: chooses between linear merge-join and galloping
/// based on size ratio.
///
/// Used for multi-bucket indices where CSR's compact unique-minimizer
/// iteration is faster than COO's pair-by-pair iteration.
///
/// # Arguments
/// * `unique_mins` - Pre-computed sorted unique minimizers from the query index.
///   Computed once per classification call and reused across shards.
pub(super) fn merge_join_csr<A: HitAccumulator>(
    query_idx: &QueryInvertedIndex,
    ref_idx: &InvertedIndex,
    accumulator: &mut A,
    unique_mins: &[u64],
) {
    if query_idx.num_entries() == 0 || ref_idx.num_minimizers() == 0 {
        return;
    }

    let q_len = unique_mins.len();
    let r_len = ref_idx.minimizers.len();

    if q_len * GALLOP_THRESHOLD < r_len {
        // Query much smaller: gallop through reference
        gallop_join_csr(query_idx, ref_idx, accumulator, unique_mins, true);
    } else if r_len * GALLOP_THRESHOLD < q_len {
        // Reference much smaller: gallop through query
        gallop_join_csr(query_idx, ref_idx, accumulator, unique_mins, false);
    } else {
        // Similar sizes: pure merge-join
        merge_join_csr_linear(query_idx, ref_idx, accumulator);
    }
}

// ============================================================================
// Sparse hit types for parallel row group processing
// ============================================================================

/// A sparse hit: (read_idx, bucket_id, fwd_hits, rc_hits).
///
/// Used for memory-efficient parallel row group processing.
pub type SparseHit = (u32, u32, u32, u32);

/// Merge-join query index against sorted pairs, returning sparse hits.
///
/// Optimized for parallel row group processing. Instead of writing to dense
/// per-read accumulators (O(num_reads) HashMaps per RG), returns only actual
/// hits as a compact vector.
///
/// # Range-Bounded Query Filtering
///
/// Since each row group covers only a small slice of the minimizer space, we
/// narrow query entries to only those that could match using `partition_point`
/// on the COO entries.
///
/// # Arguments
/// * `query_idx` - Query inverted index (COO format, built once per batch)
/// * `ref_pairs` - Sorted (minimizer, bucket_id) pairs from a single row group
///
/// # Returns
/// Sparse hits vector. May contain multiple entries for the same (read, bucket)
/// pair; the merge step accumulates them.
pub fn merge_join_pairs_sparse(
    query_idx: &QueryInvertedIndex,
    ref_pairs: &[(u64, u32)],
) -> Vec<SparseHit> {
    debug_assert!(
        ref_pairs.windows(2).all(|w| w[0].0 <= w[1].0),
        "ref_pairs must be sorted by minimizer"
    );

    if query_idx.num_entries() == 0 || ref_pairs.is_empty() {
        return Vec::new();
    }

    // Get row group min/max from sorted pairs
    let rg_min = ref_pairs[0].0;
    let rg_max = ref_pairs[ref_pairs.len() - 1].0;

    // Binary search on COO entries to find bounded range
    let q_start = query_idx.entries.partition_point(|e| e.0 < rg_min);
    let q_end = query_idx.entries.partition_point(|e| e.0 <= rg_max);

    if q_start >= q_end {
        return Vec::new();
    }

    let bounded = &query_idx.entries[q_start..q_end];
    let bounded_count = bounded.len();
    let mut hits = Vec::with_capacity(bounded_count);

    let mut qi = 0usize;
    let mut ri = 0usize;

    while qi < bounded.len() && ri < ref_pairs.len() {
        let q_min = bounded[qi].0;
        let (r_min, bucket_id) = ref_pairs[ri];

        if q_min < r_min {
            // Skip entire COO run for this query minimizer
            qi = bounded[qi..].partition_point(|e| e.0 == q_min) + qi;
        } else if q_min > r_min {
            ri += 1;
        } else {
            // Match! Find COO run for this minimizer
            let run_end = bounded[qi..].partition_point(|e| e.0 == q_min) + qi;

            // Emit hit for each read with this query minimizer
            for &(_, packed) in &bounded[qi..run_end] {
                let (read_idx, is_rc) = QueryInvertedIndex::unpack_read_id(packed);
                if is_rc {
                    hits.push((read_idx, bucket_id, 0, 1));
                } else {
                    hits.push((read_idx, bucket_id, 1, 0));
                }
            }

            // Advance ref only - multiple ref pairs may have the same minimizer
            ri += 1;
        }
    }

    hits
}

/// Merge sparse hits from row groups into dense accumulators.
///
/// Takes sparse hit vectors from parallel RG processing and accumulates them
/// into per-read HashMaps suitable for scoring.
///
/// # Arguments
/// * `sparse_hits_list` - Vector of sparse hit vectors from each row group
/// * `num_reads` - Number of reads in the batch
///
/// # Returns
/// Dense accumulator: Vec<HashMap<bucket_id, (fwd_total, rc_total)>>
#[cfg(test)]
fn merge_sparse_hits(
    sparse_hits_list: Vec<Vec<SparseHit>>,
    num_reads: usize,
) -> Vec<HashMap<u32, (u32, u32)>> {
    let mut accumulators: Vec<HashMap<u32, (u32, u32)>> = (0..num_reads)
        .map(|_| HashMap::with_capacity(ESTIMATED_BUCKETS_PER_READ))
        .collect();

    for rg_hits in sparse_hits_list {
        for (read_idx, bucket_id, fwd, rc) in rg_hits {
            debug_assert!(
                (read_idx as usize) < num_reads,
                "read_idx {} >= num_reads {}",
                read_idx,
                num_reads
            );
            let entry = accumulators[read_idx as usize]
                .entry(bucket_id)
                .or_insert((0, 0));
            entry.0 += fwd;
            entry.1 += rc;
        }
    }

    accumulators
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build sorted COO reference pairs from bucket specifications.
    fn build_ref_pairs(buckets: Vec<(u32, Vec<u64>)>) -> Vec<(u64, u32)> {
        let mut pairs = Vec::new();
        for (bucket_id, minimizers) in buckets {
            for m in minimizers {
                pairs.push((m, bucket_id));
            }
        }
        pairs.sort_unstable();
        pairs
    }

    // Tests for merge_join_pairs_sparse

    #[test]
    fn test_merge_join_pairs_sparse_basic() {
        // Query with fwd=[100, 200, 300], rc=[150, 250]
        let queries = vec![(vec![100, 200, 300], vec![150, 250])];
        let query_idx = QueryInvertedIndex::build(&queries);

        // Sorted pairs from a "row group" - bucket 1 has 100, 200; bucket 2 has 150
        let ref_pairs: Vec<(u64, u32)> = vec![
            (100, 1), // minimizer 100 -> bucket 1
            (150, 2), // minimizer 150 -> bucket 2
            (200, 1), // minimizer 200 -> bucket 1
        ];

        let hits = merge_join_pairs_sparse(&query_idx, &ref_pairs);

        // Merge sparse hits to verify
        let accumulators = merge_sparse_hits(vec![hits], 1);

        // Read 0 should have: bucket 1 -> (2 fwd, 0 rc), bucket 2 -> (0 fwd, 1 rc)
        assert_eq!(accumulators[0].get(&1), Some(&(2, 0)));
        assert_eq!(accumulators[0].get(&2), Some(&(0, 1)));
    }

    #[test]
    fn test_merge_join_pairs_sparse_range_bounded() {
        // Query spans wide range [100..900], but row group only covers [400..600]
        let queries = vec![(vec![100, 200, 300, 400, 500, 600, 700, 800, 900], vec![])];
        let query_idx = QueryInvertedIndex::build(&queries);

        // Row group only has minimizers in [400..600]
        let ref_pairs: Vec<(u64, u32)> = vec![(400, 1), (500, 1), (600, 1)];

        let hits = merge_join_pairs_sparse(&query_idx, &ref_pairs);
        let accumulators = merge_sparse_hits(vec![hits], 1);

        // Should only count hits for 400, 500, 600 (3 hits)
        assert_eq!(accumulators[0].get(&1), Some(&(3, 0)));
    }

    #[test]
    fn test_merge_join_pairs_sparse_no_overlap() {
        // Query minimizers don't overlap with row group range
        let queries = vec![(vec![100, 200, 300], vec![])];
        let query_idx = QueryInvertedIndex::build(&queries);

        // Row group has minimizers outside query range
        let ref_pairs: Vec<(u64, u32)> = vec![(500, 1), (600, 1)];

        let hits = merge_join_pairs_sparse(&query_idx, &ref_pairs);

        // No hits
        assert!(hits.is_empty());
    }

    #[test]
    fn test_merge_join_pairs_sparse_duplicate_minimizers() {
        // Multiple pairs with same minimizer (different buckets)
        let queries = vec![(vec![100, 200], vec![])];
        let query_idx = QueryInvertedIndex::build(&queries);

        // Minimizer 100 appears in buckets 1 and 2
        let ref_pairs: Vec<(u64, u32)> = vec![(100, 1), (100, 2), (200, 1)];

        let hits = merge_join_pairs_sparse(&query_idx, &ref_pairs);
        let accumulators = merge_sparse_hits(vec![hits], 1);

        // bucket 1: 2 hits (100, 200), bucket 2: 1 hit (100)
        assert_eq!(accumulators[0].get(&1), Some(&(2, 0)));
        assert_eq!(accumulators[0].get(&2), Some(&(1, 0)));
    }

    #[test]
    fn test_merge_join_pairs_sparse_multiple_reads() {
        // Two reads with overlapping minimizers
        let queries = vec![
            (vec![100, 200], vec![150]), // read 0
            (vec![100, 300], vec![150]), // read 1
        ];
        let query_idx = QueryInvertedIndex::build(&queries);

        let ref_pairs: Vec<(u64, u32)> = vec![(100, 1), (150, 2)];

        let hits = merge_join_pairs_sparse(&query_idx, &ref_pairs);
        let accumulators = merge_sparse_hits(vec![hits], 2);

        // Read 0: bucket 1 -> (1, 0), bucket 2 -> (0, 1)
        assert_eq!(accumulators[0].get(&1), Some(&(1, 0)));
        assert_eq!(accumulators[0].get(&2), Some(&(0, 1)));

        // Read 1: bucket 1 -> (1, 0), bucket 2 -> (0, 1)
        assert_eq!(accumulators[1].get(&1), Some(&(1, 0)));
        assert_eq!(accumulators[1].get(&2), Some(&(0, 1)));
    }

    #[test]
    fn test_merge_join_pairs_sparse_empty_inputs() {
        let queries: Vec<(Vec<u64>, Vec<u64>)> = vec![];
        let query_idx = QueryInvertedIndex::build(&queries);
        let ref_pairs: Vec<(u64, u32)> = vec![];

        // Should not panic on empty inputs
        let hits = merge_join_pairs_sparse(&query_idx, &ref_pairs);
        assert!(hits.is_empty());
    }

    // Tests for merge_sparse_hits

    #[test]
    fn test_merge_sparse_hits_basic() {
        // Two RGs with sparse hits for 2 reads
        let rg1_hits = vec![
            (0, 1, 2, 0), // read 0, bucket 1, 2 fwd
            (0, 2, 1, 0), // read 0, bucket 2, 1 fwd
            (1, 1, 1, 0), // read 1, bucket 1, 1 fwd
        ];
        let rg2_hits = vec![
            (0, 1, 1, 0), // read 0, bucket 1, 1 more fwd
            (0, 3, 0, 1), // read 0, bucket 3, 1 rc
            (1, 2, 0, 2), // read 1, bucket 2, 2 rc
        ];

        let merged = merge_sparse_hits(vec![rg1_hits, rg2_hits], 2);

        // Read 0: bucket 1 -> (3, 0), bucket 2 -> (1, 0), bucket 3 -> (0, 1)
        assert_eq!(merged[0].get(&1), Some(&(3, 0)));
        assert_eq!(merged[0].get(&2), Some(&(1, 0)));
        assert_eq!(merged[0].get(&3), Some(&(0, 1)));

        // Read 1: bucket 1 -> (1, 0), bucket 2 -> (0, 2)
        assert_eq!(merged[1].get(&1), Some(&(1, 0)));
        assert_eq!(merged[1].get(&2), Some(&(0, 2)));
    }

    #[test]
    fn test_merge_sparse_hits_single_rg() {
        let hits = vec![(0, 1, 2, 1)];

        let merged = merge_sparse_hits(vec![hits], 1);

        assert_eq!(merged[0].get(&1), Some(&(2, 1)));
    }

    #[test]
    fn test_merge_sparse_hits_empty() {
        let merged = merge_sparse_hits(vec![], 3);

        assert_eq!(merged.len(), 3);
        assert!(merged[0].is_empty());
        assert!(merged[1].is_empty());
        assert!(merged[2].is_empty());
    }

    // =========================================================================
    // Accumulator trait tests (using COO merge-join)
    // =========================================================================

    /// Sort results by (query_id, bucket_id) for deterministic comparison.
    fn sort_results(results: &mut [HitResult]) {
        results.sort_by(|a, b| {
            a.query_id
                .cmp(&b.query_id)
                .then(a.bucket_id.cmp(&b.bucket_id))
        });
    }

    #[test]
    fn test_dense_sparse_identical_single_bucket() {
        let queries = vec![
            (vec![100, 200, 300], vec![150, 250]), // read 0
        ];
        let query_idx = QueryInvertedIndex::build(&queries);
        let ref_pairs = build_ref_pairs(vec![
            (1, vec![100, 200, 150]), // shares 100, 200 fwd + 150 rc
        ]);
        let query_ids = vec![1i64];

        let mut dense = classify_with_coo(
            &query_idx,
            &ref_pairs,
            &query_ids,
            0.0,
            DenseAccumulator::new(1, 1),
        );
        let mut sparse = classify_with_coo(
            &query_idx,
            &ref_pairs,
            &query_ids,
            0.0,
            SparseAccumulator::new(1),
        );

        sort_results(&mut dense);
        sort_results(&mut sparse);

        assert_eq!(dense.len(), sparse.len(), "Same number of results");
        for (d, s) in dense.iter().zip(sparse.iter()) {
            assert_eq!(d.query_id, s.query_id);
            assert_eq!(d.bucket_id, s.bucket_id);
            assert!(
                (d.score - s.score).abs() < 1e-10,
                "Scores match: {} vs {}",
                d.score,
                s.score
            );
        }
    }

    #[test]
    fn test_dense_sparse_identical_multi_bucket() {
        let queries = vec![
            (vec![100, 200, 300], vec![150, 250]),
            (vec![100, 400], vec![150, 350]),
        ];
        let query_idx = QueryInvertedIndex::build(&queries);
        let ref_pairs = build_ref_pairs(vec![
            (1, vec![100, 200, 400]),
            (2, vec![150, 250, 350]),
            (3, vec![300, 500]),
        ]);
        let query_ids = vec![1i64, 2];

        let mut dense = classify_with_coo(
            &query_idx,
            &ref_pairs,
            &query_ids,
            0.0,
            DenseAccumulator::new(2, 3),
        );
        let mut sparse = classify_with_coo(
            &query_idx,
            &ref_pairs,
            &query_ids,
            0.0,
            SparseAccumulator::new(2),
        );

        sort_results(&mut dense);
        sort_results(&mut sparse);

        assert_eq!(
            dense.len(),
            sparse.len(),
            "Same number of results: dense={}, sparse={}",
            dense.len(),
            sparse.len()
        );
        for (d, s) in dense.iter().zip(sparse.iter()) {
            assert_eq!(d.query_id, s.query_id);
            assert_eq!(d.bucket_id, s.bucket_id);
            assert!(
                (d.score - s.score).abs() < 1e-10,
                "Scores match: {} vs {}",
                d.score,
                s.score
            );
        }
    }

    #[test]
    fn test_dense_sparse_identical_no_overlap() {
        let queries = vec![(vec![100, 200], vec![150])];
        let query_idx = QueryInvertedIndex::build(&queries);
        let ref_pairs = build_ref_pairs(vec![(1, vec![500, 600])]);
        let query_ids = vec![1i64];

        let dense = classify_with_coo(
            &query_idx,
            &ref_pairs,
            &query_ids,
            0.0,
            DenseAccumulator::new(1, 1),
        );
        let sparse = classify_with_coo(
            &query_idx,
            &ref_pairs,
            &query_ids,
            0.0,
            SparseAccumulator::new(1),
        );

        assert!(dense.is_empty());
        assert!(sparse.is_empty());
    }

    #[test]
    fn test_dense_sparse_identical_empty() {
        let queries: Vec<(Vec<u64>, Vec<u64>)> = vec![];
        let query_idx = QueryInvertedIndex::build(&queries);
        let ref_pairs = build_ref_pairs(vec![(1, vec![100])]);
        let query_ids: Vec<i64> = vec![];

        let dense = classify_with_coo(
            &query_idx,
            &ref_pairs,
            &query_ids,
            0.0,
            DenseAccumulator::new(0, 1),
        );
        let sparse = classify_with_coo(
            &query_idx,
            &ref_pairs,
            &query_ids,
            0.0,
            SparseAccumulator::new(0),
        );

        assert!(dense.is_empty());
        assert!(sparse.is_empty());
    }

    #[test]
    fn test_dense_sparse_identical_all_hits() {
        // Every query minimizer matches the reference
        let queries = vec![(vec![100, 200, 300], vec![])];
        let query_idx = QueryInvertedIndex::build(&queries);
        let ref_pairs = build_ref_pairs(vec![(1, vec![100, 200, 300])]);
        let query_ids = vec![1i64];

        let dense = classify_with_coo(
            &query_idx,
            &ref_pairs,
            &query_ids,
            0.0,
            DenseAccumulator::new(1, 1),
        );
        let sparse = classify_with_coo(
            &query_idx,
            &ref_pairs,
            &query_ids,
            0.0,
            SparseAccumulator::new(1),
        );

        assert_eq!(dense.len(), 1);
        assert_eq!(sparse.len(), 1);
        assert_eq!(dense[0].score, 1.0);
        assert_eq!(sparse[0].score, 1.0);
    }

    #[test]
    fn test_dense_sparse_identical_skewed_sizes() {
        // Small query, large ref — tests both accumulator types with skewed sizes
        let queries = vec![(vec![500], vec![])];
        let query_idx = QueryInvertedIndex::build(&queries);
        let ref_pairs: Vec<(u64, u32)> = (0..100).map(|i| (i * 10, 1)).collect();
        let query_ids = vec![1i64];

        let mut dense = classify_with_coo(
            &query_idx,
            &ref_pairs,
            &query_ids,
            0.0,
            DenseAccumulator::new(1, 1),
        );
        let mut sparse = classify_with_coo(
            &query_idx,
            &ref_pairs,
            &query_ids,
            0.0,
            SparseAccumulator::new(1),
        );

        sort_results(&mut dense);
        sort_results(&mut sparse);

        assert_eq!(dense.len(), sparse.len());
        for (d, s) in dense.iter().zip(sparse.iter()) {
            assert_eq!(d.query_id, s.query_id);
            assert_eq!(d.bucket_id, s.bucket_id);
            assert!(
                (d.score - s.score).abs() < 1e-10,
                "Scores match: {} vs {}",
                d.score,
                s.score
            );
        }
    }

    #[test]
    fn test_dense_accumulator_merge() {
        let mut acc1 = DenseAccumulator::new(2, 2);
        let mut acc2 = DenseAccumulator::new(2, 2);

        // acc1: read 0 bucket 1 = (3, 0)
        acc1.record_hit_counts(0, 1, 3, 0);
        // acc2: read 0 bucket 1 = (2, 1)
        acc2.record_hit_counts(0, 1, 2, 1);

        acc1.merge(acc2);

        // Should be (5, 1) at index 0*3 + 1 = 1
        assert_eq!(acc1.data[1], (5, 1));
    }

    #[test]
    fn test_sparse_accumulator_merge() {
        let mut acc1 = SparseAccumulator::new(2);
        let mut acc2 = SparseAccumulator::new(2);

        acc1.record_hit_counts(0, 1, 3, 0);
        acc2.record_hit_counts(0, 1, 2, 1);
        acc2.record_hit_counts(0, 2, 1, 0);

        acc1.merge(acc2);

        assert_eq!(acc1.accumulators[0].get(&1), Some(&(5, 1)));
        assert_eq!(acc1.accumulators[0].get(&2), Some(&(1, 0)));
    }

    // =========================================================================
    // merge_join_coo tests
    // =========================================================================

    /// Helper: run merge_join_coo with a specific accumulator and score.
    fn classify_with_coo<A: HitAccumulator>(
        query_idx: &QueryInvertedIndex,
        ref_pairs: &[(u64, u32)],
        query_ids: &[i64],
        threshold: f64,
        mut acc: A,
    ) -> Vec<HitResult> {
        merge_join_coo(query_idx, ref_pairs, &mut acc);
        acc.score_and_filter(query_idx, query_ids, threshold)
    }

    #[test]
    fn test_merge_join_coo_basic() {
        let queries = vec![(vec![100, 200, 300], vec![150, 250])];
        let query_idx = QueryInvertedIndex::build(&queries);
        let ref_pairs = build_ref_pairs(vec![(1, vec![100, 200, 400]), (2, vec![150, 250, 500])]);
        let query_ids = vec![101i64];

        let mut results = classify_with_coo(
            &query_idx,
            &ref_pairs,
            &query_ids,
            0.0,
            DenseAccumulator::new(1, 2),
        );
        sort_results(&mut results);

        assert_eq!(results.len(), 2);
        let bucket1_hit = results.iter().find(|r| r.bucket_id == 1).unwrap();
        let bucket2_hit = results.iter().find(|r| r.bucket_id == 2).unwrap();

        // Bucket 1: 2 fwd hits (100, 200) out of 3 fwd minimizers = 0.667
        assert!((bucket1_hit.score - 2.0 / 3.0).abs() < 0.001);
        // Bucket 2: 2 rc hits (150, 250) out of 2 rc minimizers = 1.0
        assert!((bucket2_hit.score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_merge_join_coo_no_overlap() {
        let queries = vec![(vec![100, 200], vec![150])];
        let query_idx = QueryInvertedIndex::build(&queries);
        let ref_pairs: Vec<(u64, u32)> = vec![(500, 1), (600, 1), (700, 1)];

        let mut acc = DenseAccumulator::new(1, 1);
        merge_join_coo(&query_idx, &ref_pairs, &mut acc);
        let results = acc.score_and_filter(&query_idx, &[1], 0.0);
        assert!(results.is_empty(), "No overlap should produce no hits");
    }

    #[test]
    fn test_merge_join_coo_empty_inputs() {
        let empty_queries: Vec<(Vec<u64>, Vec<u64>)> = vec![];
        let query_idx = QueryInvertedIndex::build(&empty_queries);
        let ref_pairs: Vec<(u64, u32)> = vec![(100, 1)];

        let mut acc = SparseAccumulator::new(0);
        merge_join_coo(&query_idx, &ref_pairs, &mut acc);
        // Should not panic

        let queries = vec![(vec![100], vec![])];
        let query_idx2 = QueryInvertedIndex::build(&queries);
        let empty_ref: Vec<(u64, u32)> = vec![];
        let mut acc2 = DenseAccumulator::new(1, 1);
        merge_join_coo(&query_idx2, &empty_ref, &mut acc2);
        let results = acc2.score_and_filter(&query_idx2, &[1], 0.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_merge_join_coo_single_bucket() {
        let queries = vec![(vec![100, 200, 300], vec![])];
        let query_idx = QueryInvertedIndex::build(&queries);
        let ref_pairs: Vec<(u64, u32)> = vec![(100, 1), (200, 1), (300, 1)];

        let mut acc = DenseAccumulator::new(1, 1);
        merge_join_coo(&query_idx, &ref_pairs, &mut acc);
        let results = acc.score_and_filter(&query_idx, &[1], 0.0);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].score, 1.0); // 3/3 fwd
    }

    #[test]
    fn test_merge_join_coo_many_buckets_per_minimizer() {
        // Single minimizer maps to 3 buckets in the reference
        let queries = vec![(vec![100], vec![])];
        let query_idx = QueryInvertedIndex::build(&queries);
        let ref_pairs: Vec<(u64, u32)> = vec![(100, 1), (100, 2), (100, 3)];

        let mut acc = DenseAccumulator::new(1, 3);
        merge_join_coo(&query_idx, &ref_pairs, &mut acc);
        let results = acc.score_and_filter(&query_idx, &[1], 0.0);

        assert_eq!(results.len(), 3, "Should hit all 3 buckets");
        for r in &results {
            assert_eq!(r.score, 1.0);
        }
    }

    #[test]
    fn test_merge_join_coo_skewed_sizes() {
        // Small query, large ref — tests run skipping on ref side
        let queries = vec![(vec![500], vec![])];
        let query_idx = QueryInvertedIndex::build(&queries);
        let ref_pairs: Vec<(u64, u32)> = (0..100).map(|i| (i * 10, 1)).collect();

        let mut acc = DenseAccumulator::new(1, 1);
        merge_join_coo(&query_idx, &ref_pairs, &mut acc);
        let results = acc.score_and_filter(&query_idx, &[1], 0.0);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].score, 1.0);
    }

    #[test]
    fn test_merge_join_coo_multi_read_multi_bucket() {
        let queries = vec![
            (vec![100, 200, 300], vec![150, 250]),
            (vec![100, 400], vec![150, 350]),
        ];
        let query_idx = QueryInvertedIndex::build(&queries);
        let ref_pairs = build_ref_pairs(vec![
            (1, vec![100, 200, 400]),
            (2, vec![150, 250, 350]),
            (3, vec![300, 500]),
        ]);
        let query_ids = vec![1i64, 2];

        let mut results = classify_with_coo(
            &query_idx,
            &ref_pairs,
            &query_ids,
            0.0,
            DenseAccumulator::new(2, 3),
        );
        sort_results(&mut results);

        // Both reads should have hits
        assert!(!results.is_empty());
        // Read 1 should hit bucket 1 (shares 100, 200 fwd) and bucket 2 (shares 150, 250 rc)
        assert!(results.iter().any(|r| r.query_id == 1 && r.bucket_id == 1));
        assert!(results.iter().any(|r| r.query_id == 1 && r.bucket_id == 2));
    }

    #[test]
    fn test_merge_join_coo_with_sparse_accumulator() {
        // Verify COO path works with SparseAccumulator too
        let queries = vec![(vec![100, 200, 300], vec![150, 250])];
        let query_idx = QueryInvertedIndex::build(&queries);
        let ref_pairs = build_ref_pairs(vec![(1, vec![100, 200, 400]), (2, vec![150, 250, 500])]);
        let query_ids = vec![101i64];

        let mut dense_results = classify_with_coo(
            &query_idx,
            &ref_pairs,
            &query_ids,
            0.0,
            DenseAccumulator::new(1, 2),
        );
        let mut sparse_results = classify_with_coo(
            &query_idx,
            &ref_pairs,
            &query_ids,
            0.0,
            SparseAccumulator::new(1),
        );

        sort_results(&mut dense_results);
        sort_results(&mut sparse_results);

        assert_eq!(dense_results.len(), sparse_results.len());
        for (d, s) in dense_results.iter().zip(sparse_results.iter()) {
            assert_eq!(d.query_id, s.query_id);
            assert_eq!(d.bucket_id, s.bucket_id);
            assert!(
                (d.score - s.score).abs() < 1e-10,
                "Scores match: {} vs {}",
                d.score,
                s.score
            );
        }
    }

    // =========================================================================
    // Parallel COO merge-join tests
    // =========================================================================

    /// Helper: compare sequential and parallel merge-join results.
    fn assert_parallel_matches_sequential(
        queries: &[(Vec<u64>, Vec<u64>)],
        ref_pairs: &[(u64, u32)],
        query_ids: &[i64],
        max_bucket_id: u32,
    ) {
        let query_idx = QueryInvertedIndex::build(queries);

        // Sequential path
        let mut acc_seq = DenseAccumulator::new(queries.len(), max_bucket_id);
        merge_join_coo(&query_idx, ref_pairs, &mut acc_seq);
        let mut results_seq = acc_seq.score_and_filter(&query_idx, query_ids, 0.0);
        sort_results(&mut results_seq);

        // Parallel path
        let mut acc_par = DenseAccumulator::new(queries.len(), max_bucket_id);
        merge_join_coo_parallel(&query_idx, ref_pairs, &mut acc_par);
        let mut results_par = acc_par.score_and_filter(&query_idx, query_ids, 0.0);
        sort_results(&mut results_par);

        assert_eq!(
            results_seq.len(),
            results_par.len(),
            "Sequential and parallel should produce same number of results"
        );
        for (s, p) in results_seq.iter().zip(results_par.iter()) {
            assert_eq!(s.query_id, p.query_id);
            assert_eq!(s.bucket_id, p.bucket_id);
            assert!(
                (s.score - p.score).abs() < 1e-10,
                "Scores should match: seq={} vs par={}",
                s.score,
                p.score
            );
        }
    }

    #[test]
    fn test_parallel_coo_single_read_large_ref() {
        // Large ref to exceed MIN_PARALLEL_SHARD_SIZE and trigger parallel path
        let num_ref = 20_000;
        let ref_pairs: Vec<(u64, u32)> = (0..num_ref).map(|i| (i as u64 * 3, 1)).collect();

        let queries = vec![(vec![0, 6, 15, 99, 300, 600, 3000, 9000], vec![3, 9, 30])];
        let query_ids = vec![1i64];

        assert_parallel_matches_sequential(&queries, &ref_pairs, &query_ids, 1);
    }

    #[test]
    fn test_parallel_coo_many_reads_single_bucket() {
        // Many reads, single bucket, large ref
        let num_ref = 15_000;
        let ref_pairs: Vec<(u64, u32)> = (0..num_ref).map(|i| (i as u64 * 2, 1)).collect();

        let queries: Vec<(Vec<u64>, Vec<u64>)> = (0..50)
            .map(|r| {
                let fwd: Vec<u64> = (0..20).map(|j| (r * 100 + j * 5) as u64 * 2).collect();
                let rc: Vec<u64> = (0..10)
                    .map(|j| (r * 100 + j * 3 + 1000) as u64 * 2)
                    .collect();
                (fwd, rc)
            })
            .collect();
        let query_ids: Vec<i64> = (1..=50).collect();

        assert_parallel_matches_sequential(&queries, &ref_pairs, &query_ids, 1);
    }

    #[test]
    fn test_parallel_coo_many_buckets() {
        // Multiple buckets in the reference
        let num_ref_per_bucket = 5_000;
        let num_buckets = 3u32;
        let mut ref_pairs: Vec<(u64, u32)> = Vec::new();
        for bucket_id in 1..=num_buckets {
            for i in 0..num_ref_per_bucket {
                ref_pairs.push((i as u64 * 10 + bucket_id as u64, bucket_id));
            }
        }
        ref_pairs.sort_unstable();

        let queries = vec![
            (
                vec![11, 21, 31, 101, 201, 301, 1001, 2001],
                vec![12, 22, 32],
            ),
            (vec![11, 51, 91, 501], vec![13, 53]),
        ];
        let query_ids = vec![1i64, 2];

        assert_parallel_matches_sequential(&queries, &ref_pairs, &query_ids, num_buckets);
    }

    #[test]
    fn test_parallel_coo_empty_ref() {
        let queries = vec![(vec![100, 200], vec![150])];
        let query_idx = QueryInvertedIndex::build(&queries);
        let ref_pairs: Vec<(u64, u32)> = vec![];

        let mut acc = DenseAccumulator::new(1, 1);
        merge_join_coo_parallel(&query_idx, &ref_pairs, &mut acc);
        let results = acc.score_and_filter(&query_idx, &[1], 0.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parallel_coo_empty_query() {
        let queries: Vec<(Vec<u64>, Vec<u64>)> = vec![];
        let query_idx = QueryInvertedIndex::build(&queries);
        let ref_pairs: Vec<(u64, u32)> = (0..20_000).map(|i| (i as u64, 1)).collect();

        let mut acc = DenseAccumulator::new(0, 1);
        merge_join_coo_parallel(&query_idx, &ref_pairs, &mut acc);
        let results = acc.score_and_filter(&query_idx, &[], 0.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parallel_coo_no_overlap() {
        // Query minimizers don't overlap with ref
        let ref_pairs: Vec<(u64, u32)> = (0..20_000).map(|i| (i as u64 * 2, 1)).collect();
        let queries = vec![(vec![1, 3, 5, 7], vec![9, 11])]; // all odd, ref all even
        let query_idx = QueryInvertedIndex::build(&queries);

        let mut acc = DenseAccumulator::new(1, 1);
        merge_join_coo_parallel(&query_idx, &ref_pairs, &mut acc);
        let results = acc.score_and_filter(&query_idx, &[1], 0.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parallel_coo_small_ref_fallback() {
        // Small ref (below MIN_PARALLEL_SHARD_SIZE) should fall back to single-threaded
        let ref_pairs: Vec<(u64, u32)> = vec![(100, 1), (200, 1), (300, 1)];
        let queries = vec![(vec![100, 200, 300], vec![])];
        let query_ids = vec![1i64];

        assert_parallel_matches_sequential(&queries, &ref_pairs, &query_ids, 1);
    }

    // =========================================================================
    // compute_chunk_ranges tests
    // =========================================================================

    #[test]
    fn test_compute_chunk_ranges_empty() {
        assert!(compute_chunk_ranges(&[], 4).is_empty());
    }

    #[test]
    fn test_compute_chunk_ranges_single_chunk() {
        let entries: Vec<(u64, u32)> = vec![(100, 0), (200, 0), (300, 0)];
        let ranges = compute_chunk_ranges(&entries, 1);
        assert_eq!(ranges, vec![(0, 3)]);
    }

    #[test]
    fn test_compute_chunk_ranges_all_same_minimizer() {
        let entries: Vec<(u64, u32)> = vec![(100, 0), (100, 1), (100, 2), (100, 3)];
        let ranges = compute_chunk_ranges(&entries, 4);
        // Can't split — all same minimizer
        assert_eq!(ranges, vec![(0, 4)]);
    }

    #[test]
    fn test_compute_chunk_ranges_distinct_minimizers() {
        let entries: Vec<(u64, u32)> =
            vec![(100, 0), (200, 0), (300, 0), (400, 0), (500, 0), (600, 0)];
        let ranges = compute_chunk_ranges(&entries, 3);
        // target_size = 6/3 = 2
        // i=1: target=2, entries[2].0=300, end=3
        // i=2: target=4, entries[4].0=500, end=5
        // Remaining: (5, 6)
        assert_eq!(ranges, vec![(0, 3), (3, 5), (5, 6)]);
    }

    #[test]
    fn test_compute_chunk_ranges_with_runs() {
        // Entries with runs of same minimizer
        let entries: Vec<(u64, u32)> = vec![
            (100, 0),
            (100, 1), // run of 2
            (200, 0),
            (200, 1),
            (200, 2), // run of 3
            (300, 0), // run of 1
        ];
        let ranges = compute_chunk_ranges(&entries, 3);
        // target_size = 6/3 = 2
        // i=1: target=2, entries[2].0=200, end=2+pp(==200)+2 = 5, push (0,5)
        // i=2: target=4, entries[4].0=200, end=4+pp(==200)+4 = 5, but 5 == start=5 skip
        // Remaining: (5, 6)
        assert_eq!(ranges, vec![(0, 5), (5, 6)]);
    }

    #[test]
    fn test_compute_chunk_ranges_more_chunks_than_entries() {
        let entries: Vec<(u64, u32)> = vec![(100, 0)];
        let ranges = compute_chunk_ranges(&entries, 8);
        // target_size = 0, falls back to single chunk
        assert_eq!(ranges, vec![(0, 1)]);
    }

    // =========================================================================
    // merge_join_coo_slice tests
    // =========================================================================

    #[test]
    fn test_merge_join_coo_slice_basic() {
        let entries: Vec<(u64, u32)> = vec![
            (100, QueryInvertedIndex::pack_read_id(0, false)),
            (200, QueryInvertedIndex::pack_read_id(0, false)),
        ];
        let ref_pairs: Vec<(u64, u32)> = vec![(100, 1), (200, 1)];

        let hits = merge_join_coo_slice(&entries, &ref_pairs);
        assert_eq!(hits.len(), 2);
        // Both should be read 0, bucket 1, fwd
        for &(read_idx, bucket_id, fwd, rc) in &hits {
            assert_eq!(read_idx, 0);
            assert_eq!(bucket_id, 1);
            assert_eq!(fwd, 1);
            assert_eq!(rc, 0);
        }
    }

    #[test]
    fn test_merge_join_coo_slice_empty() {
        assert!(merge_join_coo_slice(&[], &[(100, 1)]).is_empty());
        assert!(merge_join_coo_slice(&[(100, 0)], &[]).is_empty());
        assert!(merge_join_coo_slice(&[], &[]).is_empty());
    }

    #[test]
    fn test_merge_join_coo_slice_no_overlap() {
        let entries: Vec<(u64, u32)> = vec![(100, QueryInvertedIndex::pack_read_id(0, false))];
        let ref_pairs: Vec<(u64, u32)> = vec![(200, 1)];

        assert!(merge_join_coo_slice(&entries, &ref_pairs).is_empty());
    }
}
