//! Merge-join classification algorithms.
//!
//! Provides efficient classification using sorted merge-join between query and
//! reference inverted indices.

use std::collections::HashMap;

use crate::constants::{ESTIMATED_BUCKETS_PER_READ, GALLOP_THRESHOLD};
use crate::indices::inverted::{InvertedIndex, QueryInvertedIndex};
use crate::types::HitResult;

use super::scoring::compute_score;

/// Classify using sorted merge-join between query and reference inverted indices.
///
/// This function performs a single-pass merge-join between sorted query and
/// reference minimizer arrays, achieving O(Q + R) complexity where Q and R are
/// unique minimizer counts. When one index is much smaller (>16:1 ratio), it
/// falls back to galloping search for O(Q * log(R/Q)) complexity.
///
/// # Performance Characteristics
///
/// **Best for:**
/// - Large batches (>1000 reads) with high minimizer overlap
/// - Reads from similar genomic regions (amplicons, targeted sequencing)
/// - Query minimizer reuse >30%
///
/// **Use `classify_batch` instead when:**
/// - Small batches (<1000 reads)
/// - Diverse read origins (low minimizer reuse)
/// - Memory is extremely constrained (this builds an intermediate index)
///
/// **Complexity:**
/// - Time: O(Q + R) for merge-join, O(Q * log(R/Q)) for galloping
/// - Space: O(Q + num_reads * avg_buckets_hit)
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

    accumulate_merge_join(query_idx, ref_idx, &mut accumulators);

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

/// Pure merge-join when query and reference have similar sizes.
/// O(Q + R) complexity with excellent cache behavior.
pub(super) fn merge_join(
    query_idx: &QueryInvertedIndex,
    ref_idx: &InvertedIndex,
    accumulators: &mut [HashMap<u32, (u32, u32)>],
) {
    let mut qi = 0usize;
    let mut ri = 0usize;

    while qi < query_idx.minimizers.len() && ri < ref_idx.minimizers.len() {
        let q_min = query_idx.minimizers[qi];
        let r_min = ref_idx.minimizers[ri];

        if q_min < r_min {
            qi += 1;
        } else if q_min > r_min {
            ri += 1;
        } else {
            ref_idx.accumulate_hits_for_match(query_idx, qi, ri, accumulators);
            qi += 1;
            ri += 1;
        }
    }
}

/// Incremental binary search with exponential probing for skewed size ratios.
///
/// This is similar to the search pattern in `InvertedIndex::get_bucket_hits`,
/// but operates on two CSR structures simultaneously. When one index is much
/// larger, we iterate through the smaller one and use exponential probing
/// followed by binary search to find matches in the larger one.
///
/// # Algorithm
/// For each element in the smaller array:
/// 1. Exponential probe: jump 1, 2, 4, 8... positions until we overshoot
/// 2. Binary search: search within [current_pos, current_pos + jump]
/// 3. Advance position on match or miss (leveraging sorted order)
///
/// # Requirements
/// - Both `query_idx.minimizers` and `ref_idx.minimizers` must be sorted
/// - This is guaranteed by `QueryInvertedIndex::build` and `InvertedIndex::build_from_index`
///
/// # Arguments
/// * `query_smaller` - if true, iterate query and search ref; else vice versa
pub(super) fn gallop_join(
    query_idx: &QueryInvertedIndex,
    ref_idx: &InvertedIndex,
    accumulators: &mut [HashMap<u32, (u32, u32)>],
    query_smaller: bool,
) {
    let (smaller, larger) = if query_smaller {
        (&query_idx.minimizers[..], &ref_idx.minimizers[..])
    } else {
        (&ref_idx.minimizers[..], &query_idx.minimizers[..])
    };

    let mut larger_pos = 0usize;

    for (smaller_idx, &s_min) in smaller.iter().enumerate() {
        // Gallop: exponential search
        let mut jump = 1usize;
        while larger_pos + jump < larger.len() && larger[larger_pos + jump] < s_min {
            jump *= 2;
        }

        // Binary search in [larger_pos, larger_pos + jump] (inclusive)
        // Note: +1 because the match could be AT position larger_pos + jump
        let search_end = (larger_pos + jump + 1).min(larger.len());
        match larger[larger_pos..search_end].binary_search(&s_min) {
            Ok(rel_idx) => {
                let larger_idx = larger_pos + rel_idx;
                let (qi, ri) = if query_smaller {
                    (smaller_idx, larger_idx)
                } else {
                    (larger_idx, smaller_idx)
                };
                ref_idx.accumulate_hits_for_match(query_idx, qi, ri, accumulators);
                larger_pos = larger_idx + 1;
            }
            Err(rel_idx) => {
                larger_pos += rel_idx;
            }
        }
    }
}

/// Accumulate hits from merge-join into existing accumulators.
///
/// This is the core accumulation logic extracted for reuse by sharded classification.
/// Chooses between pure merge-join and galloping based on size ratio.
pub(super) fn accumulate_merge_join(
    query_idx: &QueryInvertedIndex,
    ref_idx: &InvertedIndex,
    accumulators: &mut [HashMap<u32, (u32, u32)>],
) {
    if query_idx.num_minimizers() == 0 || ref_idx.num_minimizers() == 0 {
        return;
    }

    let q_len = query_idx.minimizers.len();
    let r_len = ref_idx.minimizers.len();

    if q_len * GALLOP_THRESHOLD < r_len {
        // Query much smaller: gallop through reference
        gallop_join(query_idx, ref_idx, accumulators, true);
    } else if r_len * GALLOP_THRESHOLD < q_len {
        // Reference much smaller: gallop through query
        gallop_join(query_idx, ref_idx, accumulators, false);
    } else {
        // Similar sizes: pure merge-join
        merge_join(query_idx, ref_idx, accumulators);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indices::main::Index;

    #[test]
    fn test_merge_join_basic() {
        // Simple test with known minimizers
        let queries = vec![
            (vec![100, 200, 300], vec![150, 250]), // read 0
        ];
        let query_idx = QueryInvertedIndex::build(&queries);

        // Build reference with overlapping minimizers
        let mut index = Index::new(64, 50, 0).unwrap();
        index.buckets.insert(1, vec![100, 200, 400]); // shares 100, 200
        index.buckets.insert(2, vec![150, 250, 500]); // shares 150, 250
        index.bucket_names.insert(1, "A".into());
        index.bucket_names.insert(2, "B".into());

        let ref_idx = InvertedIndex::build_from_index(&index);
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

        let mut index = Index::new(64, 50, 0).unwrap();
        index.buckets.insert(1, vec![500, 600, 700]); // no overlap
        index.bucket_names.insert(1, "A".into());

        let ref_idx = InvertedIndex::build_from_index(&index);
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

        let mut index = Index::new(64, 50, 0).unwrap();
        index.buckets.insert(1, vec![100, 200]);
        index.bucket_names.insert(1, "A".into());

        let ref_idx = InvertedIndex::build_from_index(&index);
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
        let mut index = Index::new(64, 50, 0).unwrap();
        let minimizers: Vec<u64> = (0..100).map(|i| i * 10).collect();
        index.buckets.insert(1, minimizers);
        index.bucket_names.insert(1, "A".into());

        let ref_idx = InvertedIndex::build_from_index(&index);
        let query_ids = vec![101i64];

        // 500 is in the reference (50 * 10 = 500)
        let results = classify_batch_merge_join(&query_idx, &ref_idx, &query_ids, 0.0);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].bucket_id, 1);
        assert_eq!(results[0].score, 1.0); // 1/1 fwd minimizers matched
    }

    #[test]
    fn test_galloping_search_boundary_case() {
        // This test triggers an off-by-one bug in gallop_join where matches
        // at the gallop boundary position (larger_pos + jump) are missed.
        //
        // With query = [10] and ref = [0, 10, 20, ...], larger_pos=0:
        // - gallop: jump=1, larger[1]=10, 10 < 10? NO → exit
        // - search_end = min(0 + 1, len) = 1
        // - search larger[0..1] = [0] for 10 → NOT FOUND (bug!)
        //
        // The match at position 1 is excluded from the search range.
        let queries = vec![
            (vec![10], vec![]), // Single minimizer at boundary position
        ];
        let query_idx = QueryInvertedIndex::build(&queries);

        // Large reference to trigger galloping (need q_len * 16 < r_len)
        let mut index = Index::new(64, 50, 0).unwrap();
        let minimizers: Vec<u64> = (0..100).map(|i| i * 10).collect(); // [0, 10, 20, ...]
        index.buckets.insert(1, minimizers);
        index.bucket_names.insert(1, "A".into());

        let ref_idx = InvertedIndex::build_from_index(&index);
        let query_ids = vec![101i64];

        // 10 is at position 1 in the reference (1 * 10 = 10)
        let results = classify_batch_merge_join(&query_idx, &ref_idx, &query_ids, 0.0);

        // This assertion will FAIL with the bug (results.len() == 0)
        // and PASS after the fix (results.len() == 1)
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
        // Need ref_len * 16 < query_len
        let minimizers: Vec<u64> = (0..100).map(|i| i * 10).collect();
        let queries = vec![
            (minimizers.clone(), vec![]), // 100 minimizers
        ];
        let query_idx = QueryInvertedIndex::build(&queries);

        // Small reference (< 100/16 = 6 minimizers)
        let mut index = Index::new(64, 50, 0).unwrap();
        index.buckets.insert(1, vec![500]); // Single minimizer at position 50
        index.bucket_names.insert(1, "A".into());

        let ref_idx = InvertedIndex::build_from_index(&index);
        let query_ids = vec![101i64];

        let results = classify_batch_merge_join(&query_idx, &ref_idx, &query_ids, 0.0);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].bucket_id, 1);
        // 1/100 fwd minimizers matched = 0.01
        assert!((results[0].score - 0.01).abs() < 0.001);
    }
}
