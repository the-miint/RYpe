//! Batch classification functions for direct Index classification.

use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

use crate::constants::ESTIMATED_MINIMIZERS_PER_SEQUENCE;
use crate::core::extraction::{count_hits, get_paired_minimizers_into};
use crate::core::workspace::MinimizerWorkspace;
use crate::indices::main::Index;
use crate::types::{HitResult, QueryRecord};

use super::common::filter_negative_mins;

/// Estimate minimizers per query from the first record in a batch.
fn estimate_minimizers_from_records(records: &[QueryRecord], k: usize, w: usize) -> usize {
    if records.is_empty() {
        return ESTIMATED_MINIMIZERS_PER_SEQUENCE;
    }
    let (_, s1, s2) = &records[0];
    let query_len = s1.len() + s2.map(|s| s.len()).unwrap_or(0);
    if query_len <= k {
        return ESTIMATED_MINIMIZERS_PER_SEQUENCE;
    }
    let estimate = ((query_len - k) / w + 1) * 2;
    estimate.max(ESTIMATED_MINIMIZERS_PER_SEQUENCE)
}

/// Classify a batch of query records against an Index with optional negative filtering.
///
/// Uses parallel minimizer extraction and per-bucket binary search.
///
/// When negative minimizers are provided, they are removed from query minimizer sets
/// BEFORE scoring against the positive index. This allows filtering out known
/// contaminant or off-target sequences.
///
/// # Arguments
/// * `engine` - The index to classify against
/// * `negative_mins` - Optional set of minimizers to exclude from queries before scoring
/// * `records` - Batch of query records
/// * `threshold` - Minimum score threshold for reporting hits
///
/// # Returns
/// Vector of HitResult for all (query, bucket) pairs meeting threshold.
pub fn classify_batch(
    engine: &Index,
    negative_mins: Option<&HashSet<u64>>,
    records: &[QueryRecord],
    threshold: f64,
) -> Vec<HitResult> {
    let estimated_mins = estimate_minimizers_from_records(records, engine.k, engine.w);
    let processed: Vec<_> = records
        .par_iter()
        .map_init(
            || MinimizerWorkspace::with_estimate(estimated_mins),
            |ws, (id, s1, s2)| {
                let (ha, hb) =
                    get_paired_minimizers_into(s1, *s2, engine.k, engine.w, engine.salt, ws);
                let (fa, fb) = filter_negative_mins(ha, hb, negative_mins);
                (*id, fa, fb)
            },
        )
        .collect();

    let mut map_a: HashMap<u64, Vec<usize>> = HashMap::new();
    let mut map_b: HashMap<u64, Vec<usize>> = HashMap::new();
    let mut uniq_mins = HashSet::new();

    for (idx, (_, ma, mb)) in processed.iter().enumerate() {
        for &m in ma {
            map_a.entry(m).or_default().push(idx);
            uniq_mins.insert(m);
        }
        for &m in mb {
            map_b.entry(m).or_default().push(idx);
            uniq_mins.insert(m);
        }
    }
    let uniq_vec: Vec<u64> = uniq_mins.into_iter().collect();

    let results: Vec<HitResult> =
        engine
            .buckets
            .par_iter()
            .map(|(b_id, bucket)| {
                let mut hits = HashMap::new();

                for &m in &uniq_vec {
                    if bucket.binary_search(&m).is_ok() {
                        if let Some(rs) = map_a.get(&m) {
                            for &r in rs {
                                hits.entry(r).or_insert((0, 0)).0 += 1;
                            }
                        }
                        if let Some(rs) = map_b.get(&m) {
                            for &r in rs {
                                hits.entry(r).or_insert((0, 0)).1 += 1;
                            }
                        }
                    }
                }

                let mut bucket_results = Vec::new();
                for (r_idx, (hits_a, hits_b)) in hits {
                    let (qid, ha, hb) = &processed[r_idx];
                    let la = ha.len() as f64;
                    let lb = hb.len() as f64;

                    let score = (if la > 0.0 { hits_a as f64 / la } else { 0.0 })
                        .max(if lb > 0.0 { hits_b as f64 / lb } else { 0.0 });

                    if score >= threshold {
                        bucket_results.push(HitResult {
                            query_id: *qid,
                            bucket_id: *b_id,
                            score,
                        });
                    }
                }
                bucket_results
            })
            .flatten()
            .collect();

    results
}

/// Aggregate classification for paired-end reads.
///
/// Combines all minimizers from all records into a single query and
/// classifies against all buckets. Returns results with query_id = -1.
///
/// # Arguments
/// * `engine` - The index to classify against
/// * `negative_mins` - Optional set of minimizers to exclude from queries before scoring
/// * `records` - Batch of query records
/// * `threshold` - Minimum score threshold
///
/// # Returns
/// Vector of HitResult with query_id = -1 for all buckets meeting the threshold.
pub fn aggregate_batch(
    engine: &Index,
    negative_mins: Option<&HashSet<u64>>,
    records: &[QueryRecord],
    threshold: f64,
) -> Vec<HitResult> {
    let mut global = HashSet::new();

    let estimated_mins = estimate_minimizers_from_records(records, engine.k, engine.w);
    let batch_mins: Vec<Vec<u64>> = records
        .par_iter()
        .map_init(
            || MinimizerWorkspace::with_estimate(estimated_mins),
            |ws, (_, s1, s2)| {
                let (mut a, b) =
                    get_paired_minimizers_into(s1, *s2, engine.k, engine.w, engine.salt, ws);
                a.extend(b);
                if let Some(neg_set) = negative_mins {
                    a.retain(|m| !neg_set.contains(m));
                }
                a
            },
        )
        .collect();

    for v in batch_mins {
        for m in v {
            global.insert(m);
        }
    }

    let total = global.len() as f64;
    if total == 0.0 {
        return Vec::new();
    }

    let g_vec: Vec<u64> = global.into_iter().collect();

    engine
        .buckets
        .par_iter()
        .filter_map(|(id, b)| {
            let s = count_hits(&g_vec, b) / total;
            if s >= threshold {
                Some(HitResult {
                    query_id: -1,
                    bucket_id: *id,
                    score: s,
                })
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_batch_logic() {
        let mut index = Index::new(64, 10, 0).unwrap();
        index.buckets.insert(1, vec![10, 20, 30, 40, 50]);
        index.buckets.insert(2, vec![60, 70, 80, 90, 100]);

        let seq_a = vec![b'A'; 80];
        let mut ws = MinimizerWorkspace::new();

        index.add_record(1, "ref_a", &seq_a, &mut ws);
        index.finalize_bucket(1);

        let query_seq = seq_a.clone();
        let records: Vec<QueryRecord> = vec![(101, &query_seq, None)];

        let results = classify_batch(&index, None, &records, 0.5);

        assert!(!results.is_empty());
        let hit = &results[0];
        assert_eq!(hit.query_id, 101);
        assert_eq!(hit.bucket_id, 1);
        assert_eq!(hit.score, 1.0);
    }

    #[test]
    fn test_aggregate_batch_logic() {
        let mut index = Index::new(64, 10, 0).unwrap();
        let mut ws = MinimizerWorkspace::new();

        let seq_a = vec![b'A'; 100];
        index.add_record(1, "ref_a", &seq_a, &mut ws);
        index.finalize_bucket(1);

        let seq_at: Vec<u8> = (0..200)
            .map(|i| if i % 2 == 0 { b'A' } else { b'T' })
            .collect();
        index.add_record(2, "ref_at", &seq_at, &mut ws);
        index.finalize_bucket(2);

        let q1 = &seq_at[0..100];
        let q2 = &seq_at[100..200];

        let records: Vec<QueryRecord> = vec![(1, q1, None), (2, q2, None)];

        let results = aggregate_batch(&index, None, &records, 0.5);

        assert_eq!(results.len(), 1, "Should only match bucket 2");
        assert_eq!(results[0].bucket_id, 2);
        assert!(results[0].score > 0.9);
    }

    #[test]
    fn test_classify_batch_negative_filtering() {
        let mut index = Index::new(64, 10, 0).unwrap();
        let mut ws = MinimizerWorkspace::new();

        let seq_a = vec![b'A'; 80];
        index.add_record(1, "ref_a", &seq_a, &mut ws);
        index.finalize_bucket(1);

        let query_seq = seq_a.clone();
        let records: Vec<QueryRecord> = vec![(101, &query_seq, None)];

        // Without negative filtering: should match bucket 1
        let results_no_neg = classify_batch(&index, None, &records, 0.5);
        assert!(
            !results_no_neg.is_empty(),
            "Should have hits without negative filtering"
        );
        assert_eq!(results_no_neg[0].score, 1.0);

        // Extract the minimizers that would be in the query
        let (query_mins, _) = crate::core::extraction::get_paired_minimizers_into(
            &query_seq, None, index.k, index.w, index.salt, &mut ws,
        );

        // Create negative set containing all query minimizers
        let neg_set: HashSet<u64> = query_mins.into_iter().collect();

        // With full negative filtering: should have no hits (score = 0)
        let results_with_neg = classify_batch(&index, Some(&neg_set), &records, 0.5);
        assert!(
            results_with_neg.is_empty(),
            "Should have no hits above threshold when all minimizers filtered"
        );
    }

    #[test]
    fn test_aggregate_batch_negative_filtering() {
        // Use K=32 to allow shorter sequences to produce minimizers
        let mut index = Index::new(32, 10, 0).unwrap();
        let mut ws = MinimizerWorkspace::new();

        let seq_a = vec![b'A'; 100];
        index.add_record(1, "ref_a", &seq_a, &mut ws);
        index.finalize_bucket(1);

        let records: Vec<QueryRecord> = vec![(1, &seq_a[0..50], None), (2, &seq_a[50..100], None)];

        // Without negative filtering
        let results_no_neg = aggregate_batch(&index, None, &records, 0.5);
        assert!(
            !results_no_neg.is_empty(),
            "Should have results without negative filtering"
        );

        // Get all minimizers from the queries
        let (mins1, _) = crate::core::extraction::get_paired_minimizers_into(
            &seq_a[0..50],
            None,
            index.k,
            index.w,
            index.salt,
            &mut ws,
        );
        let (mins2, _) = crate::core::extraction::get_paired_minimizers_into(
            &seq_a[50..100],
            None,
            index.k,
            index.w,
            index.salt,
            &mut ws,
        );
        let mut full_neg: HashSet<u64> = mins1.into_iter().collect();
        full_neg.extend(mins2);

        // With full negative filtering
        let results_full_neg = aggregate_batch(&index, Some(&full_neg), &records, 0.5);
        assert!(
            results_full_neg.is_empty(),
            "Should have no hits when all minimizers filtered"
        );
    }

    #[test]
    fn test_negative_filtering_empty_set() {
        // Empty negative set should have no effect
        let mut index = Index::new(64, 10, 0).unwrap();
        let mut ws = MinimizerWorkspace::new();

        let seq_a = vec![b'A'; 80];
        index.add_record(1, "ref_a", &seq_a, &mut ws);
        index.finalize_bucket(1);

        let query_seq = seq_a.clone();
        let records: Vec<QueryRecord> = vec![(101, &query_seq, None)];

        let empty_neg: HashSet<u64> = HashSet::new();

        let results_none = classify_batch(&index, None, &records, 0.5);
        let results_empty = classify_batch(&index, Some(&empty_neg), &records, 0.5);

        assert_eq!(results_none.len(), results_empty.len());
        if !results_none.is_empty() {
            assert_eq!(results_none[0].bucket_id, results_empty[0].bucket_id);
            assert!((results_none[0].score - results_empty[0].score).abs() < 0.001);
        }
    }

    #[test]
    fn test_negative_filtering_no_overlap() {
        // Negative set with minimizers that don't appear in query should have no effect
        // Note: Use alternating AT sequence since all-A gives minimizer = u64::MAX
        let mut index = Index::new(64, 10, 0).unwrap();
        let mut ws = MinimizerWorkspace::new();

        // Use alternating sequence to get non-trivial minimizers
        let seq_at: Vec<u8> = (0..80)
            .map(|i| if i % 2 == 0 { b'A' } else { b'T' })
            .collect();
        index.add_record(1, "ref_at", &seq_at, &mut ws);
        index.finalize_bucket(1);

        let query_seq = seq_at.clone();
        let records: Vec<QueryRecord> = vec![(101, &query_seq[..], None)];

        // Create negative set with minimizers that won't match (use values far from typical hashes)
        // Value 1 is unlikely since it would require a very specific pattern
        let no_overlap_neg: HashSet<u64> = vec![1, 2, 3].into_iter().collect();

        let results_none = classify_batch(&index, None, &records, 0.5);
        let results_no_overlap = classify_batch(&index, Some(&no_overlap_neg), &records, 0.5);

        assert_eq!(
            results_none.len(),
            results_no_overlap.len(),
            "No-overlap negative filtering should not change result count"
        );
        if !results_none.is_empty() {
            assert_eq!(results_none[0].bucket_id, results_no_overlap[0].bucket_id);
            assert!((results_none[0].score - results_no_overlap[0].score).abs() < 0.001);
        }
    }
}
