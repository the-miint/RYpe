//! Merge-join classification algorithms.
//!
//! Provides efficient classification using sorted merge-join between query and
//! reference inverted indices.

use std::collections::HashMap;

use crate::constants::{ESTIMATED_BUCKETS_PER_READ, GALLOP_THRESHOLD};
use crate::core::gallop_for_each;
use crate::indices::{InvertedIndex, QueryInvertedIndex};
use crate::types::HitResult;

use super::scoring::compute_score;

/// Classify using sorted merge-join between query and reference inverted indices.
///
/// This function performs a single-pass merge-join between sorted query and
/// reference minimizer arrays, achieving O(Q + R) complexity where Q and R are
/// unique minimizer counts. When one index is much smaller (>16:1 ratio), it
/// falls back to galloping search for O(Q * log(R/Q)) complexity.
///
/// # Arguments
/// * `query_idx` - Query inverted index built from extracted minimizers
/// * `ref_idx` - Reference inverted index
/// * `query_ids` - Original query IDs for results (parallel to query_idx read order)
/// * `threshold` - Minimum score threshold for reporting hits
///
/// # Returns
/// Vector of HitResult for all (query, bucket) pairs meeting the threshold.
pub fn classify_batch_merge_join(
    query_idx: &QueryInvertedIndex,
    ref_idx: &InvertedIndex,
    query_ids: &[i64],
    threshold: f64,
) -> Vec<HitResult> {
    let num_reads = query_idx.num_reads();
    if num_reads == 0 {
        return Vec::new();
    }

    // Per-read accumulator: bucket_id -> (fwd_hits, rc_hits)
    let mut accumulators: Vec<HashMap<u32, (u32, u32)>> = (0..num_reads)
        .map(|_| HashMap::with_capacity(ESTIMATED_BUCKETS_PER_READ))
        .collect();

    let unique_mins = query_idx.unique_minimizers();
    accumulate_merge_join(query_idx, ref_idx, &mut accumulators, &unique_mins);

    // Score and filter
    let mut results = Vec::new();
    for (read_idx, buckets) in accumulators.into_iter().enumerate() {
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

/// Accumulate hits into accumulators for all COO entries in `run`.
/// Each entry is cross-producted with each bucket in `bucket_slice`.
#[inline]
fn accumulate_coo_run(
    entries: &[(u64, u32)],
    bucket_slice: &[u32],
    accumulators: &mut [HashMap<u32, (u32, u32)>],
) {
    for &(_, packed) in entries {
        let (read_idx, is_rc) = QueryInvertedIndex::unpack_read_id(packed);
        for &bucket_id in bucket_slice {
            let entry = accumulators[read_idx as usize]
                .entry(bucket_id)
                .or_insert((0, 0));
            if is_rc {
                entry.1 = entry.1.saturating_add(1);
            } else {
                entry.0 = entry.0.saturating_add(1);
            }
        }
    }
}

/// Pure merge-join when query and reference have similar sizes.
/// Walks COO entries with run detection. O(Q_unique + R) outer comparisons.
pub(super) fn merge_join(
    query_idx: &QueryInvertedIndex,
    ref_idx: &InvertedIndex,
    accumulators: &mut [HashMap<u32, (u32, u32)>],
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
            accumulate_coo_run(&entries[qi..run_end], bucket_slice, accumulators);

            qi = run_end;
            ri += 1;
        }
    }
}

/// Galloping search for skewed size ratios.
///
/// Uses pre-computed `unique_mins` for the galloping outer loop,
/// then finds COO runs via `partition_point` on match.
///
/// # Arguments
/// * `unique_mins` - Pre-computed sorted unique minimizers (avoids per-shard allocation)
/// * `query_smaller` - if true, iterate query and search ref; else vice versa
pub(super) fn gallop_join(
    query_idx: &QueryInvertedIndex,
    ref_idx: &InvertedIndex,
    accumulators: &mut [HashMap<u32, (u32, u32)>],
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

        accumulate_coo_run(
            &query_idx.entries[run_start..run_end],
            bucket_slice,
            accumulators,
        );
    });
}

/// Accumulate hits from merge-join into existing accumulators.
///
/// This is the core accumulation logic extracted for reuse by sharded classification.
/// Chooses between pure merge-join and galloping based on size ratio.
///
/// # Arguments
/// * `unique_mins` - Pre-computed sorted unique minimizers from the query index.
///   Computed once per classification call and reused across shards.
pub(super) fn accumulate_merge_join(
    query_idx: &QueryInvertedIndex,
    ref_idx: &InvertedIndex,
    accumulators: &mut [HashMap<u32, (u32, u32)>],
    unique_mins: &[u64],
) {
    if query_idx.num_entries() == 0 || ref_idx.num_minimizers() == 0 {
        return;
    }

    let q_len = unique_mins.len();
    let r_len = ref_idx.minimizers.len();

    if q_len * GALLOP_THRESHOLD < r_len {
        // Query much smaller: gallop through reference
        gallop_join(query_idx, ref_idx, accumulators, unique_mins, true);
    } else if r_len * GALLOP_THRESHOLD < q_len {
        // Reference much smaller: gallop through query
        gallop_join(query_idx, ref_idx, accumulators, unique_mins, false);
    } else {
        // Similar sizes: pure merge-join
        merge_join(query_idx, ref_idx, accumulators);
    }
}

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
pub fn merge_sparse_hits(
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
    use crate::types::IndexMetadata;
    use std::collections::HashMap;

    /// Helper to build an InvertedIndex for testing from bucket minimizers.
    fn build_test_inverted_index(buckets: Vec<(u32, &str, Vec<u64>)>) -> InvertedIndex {
        let mut bucket_map: HashMap<u32, Vec<u64>> = HashMap::new();
        let mut bucket_names: HashMap<u32, String> = HashMap::new();
        let mut bucket_minimizer_counts: HashMap<u32, usize> = HashMap::new();

        for (id, name, mins) in buckets {
            bucket_minimizer_counts.insert(id, mins.len());
            bucket_map.insert(id, mins);
            bucket_names.insert(id, name.to_string());
        }

        let metadata = IndexMetadata {
            k: 64,
            w: 50,
            salt: 0,
            bucket_names,
            bucket_sources: HashMap::new(),
            bucket_minimizer_counts,
            largest_shard_entries: 0,
        };

        InvertedIndex::build_from_bucket_map(64, 50, 0, &bucket_map, &metadata)
    }

    #[test]
    fn test_merge_join_basic() {
        // Simple test with known minimizers
        let queries = vec![
            (vec![100, 200, 300], vec![150, 250]), // read 0
        ];
        let query_idx = QueryInvertedIndex::build(&queries);

        // Build reference with overlapping minimizers
        let ref_idx = build_test_inverted_index(vec![
            (1, "A", vec![100, 200, 400]), // shares 100, 200
            (2, "B", vec![150, 250, 500]), // shares 150, 250
        ]);
        let query_ids = vec![101i64];

        let results = classify_batch_merge_join(&query_idx, &ref_idx, &query_ids, 0.0);

        // Should have hits for both buckets
        assert_eq!(results.len(), 2);

        let bucket1_hit = results.iter().find(|r| r.bucket_id == 1).unwrap();
        let bucket2_hit = results.iter().find(|r| r.bucket_id == 2).unwrap();

        // Bucket 1: 2 fwd hits (100, 200) out of 3 fwd minimizers = 0.667
        assert!((bucket1_hit.score - 2.0 / 3.0).abs() < 0.001);
        // Bucket 2: 2 rc hits (150, 250) out of 2 rc minimizers = 1.0
        assert!((bucket2_hit.score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_merge_join_no_overlap() {
        let queries = vec![(vec![100, 200], vec![150])];
        let query_idx = QueryInvertedIndex::build(&queries);

        let ref_idx = build_test_inverted_index(vec![
            (1, "A", vec![500, 600, 700]), // no overlap
        ]);
        let query_ids = vec![101i64];

        let results = classify_batch_merge_join(&query_idx, &ref_idx, &query_ids, 0.0);

        assert!(
            results.is_empty(),
            "Should have no hits when no minimizers overlap"
        );
    }

    #[test]
    fn test_merge_join_empty_query() {
        let queries: Vec<(Vec<u64>, Vec<u64>)> = vec![];
        let query_idx = QueryInvertedIndex::build(&queries);

        let ref_idx = build_test_inverted_index(vec![(1, "A", vec![100, 200])]);
        let query_ids: Vec<i64> = vec![];

        let results = classify_batch_merge_join(&query_idx, &ref_idx, &query_ids, 0.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_galloping_search_small_query() {
        // Create scenario where query << ref to trigger galloping
        // Need query_len * 16 < ref_len
        let queries = vec![
            (vec![500], vec![]), // Single minimizer
        ];
        let query_idx = QueryInvertedIndex::build(&queries);

        // Large reference (need at least 17 minimizers to trigger gallop with 1 query)
        let minimizers: Vec<u64> = (0..100).map(|i| i * 10).collect();
        let ref_idx = build_test_inverted_index(vec![(1, "A", minimizers)]);
        let query_ids = vec![101i64];

        // 500 is in the reference (50 * 10 = 500)
        let results = classify_batch_merge_join(&query_idx, &ref_idx, &query_ids, 0.0);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].bucket_id, 1);
        assert_eq!(results[0].score, 1.0); // 1/1 fwd minimizers matched
    }

    #[test]
    fn test_galloping_search_boundary_case() {
        let queries = vec![
            (vec![10], vec![]), // Single minimizer at boundary position
        ];
        let query_idx = QueryInvertedIndex::build(&queries);

        // Large reference to trigger galloping (need q_len * 16 < r_len)
        let minimizers: Vec<u64> = (0..100).map(|i| i * 10).collect(); // [0, 10, 20, ...]
        let ref_idx = build_test_inverted_index(vec![(1, "A", minimizers)]);
        let query_ids = vec![101i64];

        let results = classify_batch_merge_join(&query_idx, &ref_idx, &query_ids, 0.0);

        assert_eq!(
            results.len(),
            1,
            "Minimizer at boundary position should be found"
        );
        assert_eq!(results[0].bucket_id, 1);
        assert_eq!(results[0].score, 1.0);
    }

    #[test]
    fn test_galloping_search_small_ref() {
        // Create scenario where ref << query to trigger galloping (rare case)
        let minimizers: Vec<u64> = (0..100).map(|i| i * 10).collect();
        let queries = vec![
            (minimizers.clone(), vec![]), // 100 minimizers
        ];
        let query_idx = QueryInvertedIndex::build(&queries);

        // Small reference (< 100/16 = 6 minimizers)
        let ref_idx = build_test_inverted_index(vec![
            (1, "A", vec![500]), // Single minimizer at position 50
        ]);
        let query_ids = vec![101i64];

        let results = classify_batch_merge_join(&query_idx, &ref_idx, &query_ids, 0.0);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].bucket_id, 1);
        // 1/100 fwd minimizers matched = 0.01
        assert!((results[0].score - 0.01).abs() < 0.001);
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
}
