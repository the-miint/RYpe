//! Classification functions for matching query sequences against indexed references.
//!
//! Provides multiple classification strategies:
//! - `classify_batch_sharded_merge_join`: Classification using merge-join algorithm (default)
//! - `classify_batch_sharded_parallel_rg`: Parallel row group processing

mod common;
pub mod log_ratio;
mod merge_join;
mod scoring;
mod sharded;

// Re-export for crate-internal use (c_api.rs)
#[allow(unused_imports)]
pub(crate) use common::collect_negative_minimizers_sharded;
pub use sharded::{
    classify_batch_sharded_merge_join, classify_batch_sharded_parallel_rg,
    classify_from_extracted_minimizers, classify_from_extracted_minimizers_parallel_rg,
    classify_from_query_index, classify_from_query_index_parallel_rg,
    classify_with_sharded_negative, extract_batch_minimizers,
};

// Re-export best-hit filtering
pub use best_hit::filter_best_hits;

mod best_hit;

#[cfg(test)]
mod best_hit_tests {
    use super::filter_best_hits;
    use crate::types::HitResult;

    #[test]
    fn test_filter_best_hits_basic() {
        // Query 1 has two hits, query 2 has one hit
        // Should keep the highest-scoring hit for each query
        let hits = vec![
            HitResult {
                query_id: 1,
                bucket_id: 10,
                score: 0.5,
            },
            HitResult {
                query_id: 1,
                bucket_id: 20,
                score: 0.9,
            }, // best for query 1
            HitResult {
                query_id: 2,
                bucket_id: 10,
                score: 0.8,
            }, // only hit for query 2
        ];
        let best = filter_best_hits(hits);
        assert_eq!(best.len(), 2); // one per query_id

        // Query 1 should have bucket 20 with score 0.9
        let q1 = best.iter().find(|h| h.query_id == 1).unwrap();
        assert_eq!(q1.bucket_id, 20);
        assert!((q1.score - 0.9).abs() < 1e-10);

        // Query 2 should have bucket 10 with score 0.8
        let q2 = best.iter().find(|h| h.query_id == 2).unwrap();
        assert_eq!(q2.bucket_id, 10);
        assert!((q2.score - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_filter_best_hits_tie_breaking() {
        // Query 1 has two hits with same score
        // Should keep the first one encountered (arbitrary but deterministic)
        let hits = vec![
            HitResult {
                query_id: 1,
                bucket_id: 10,
                score: 0.5,
            },
            HitResult {
                query_id: 1,
                bucket_id: 20,
                score: 0.5,
            }, // tie
        ];
        let best = filter_best_hits(hits);
        assert_eq!(best.len(), 1);
        // Should consistently pick the first one encountered (bucket_id 10)
        assert_eq!(best[0].bucket_id, 10);
    }

    #[test]
    fn test_filter_best_hits_empty() {
        let hits: Vec<HitResult> = vec![];
        let best = filter_best_hits(hits);
        assert!(best.is_empty());
    }

    #[test]
    fn test_filter_best_hits_unique_queries() {
        // All queries are unique - no filtering needed
        let hits = vec![
            HitResult {
                query_id: 1,
                bucket_id: 10,
                score: 0.5,
            },
            HitResult {
                query_id: 2,
                bucket_id: 20,
                score: 0.6,
            },
            HitResult {
                query_id: 3,
                bucket_id: 30,
                score: 0.7,
            },
        ];
        let best = filter_best_hits(hits);
        assert_eq!(best.len(), 3);
    }

    #[test]
    fn test_filter_best_hits_many_buckets_per_query() {
        // One query hits 5 buckets - should keep only the best
        let hits = vec![
            HitResult {
                query_id: 1,
                bucket_id: 1,
                score: 0.1,
            },
            HitResult {
                query_id: 1,
                bucket_id: 2,
                score: 0.3,
            },
            HitResult {
                query_id: 1,
                bucket_id: 3,
                score: 0.7,
            }, // best
            HitResult {
                query_id: 1,
                bucket_id: 4,
                score: 0.5,
            },
            HitResult {
                query_id: 1,
                bucket_id: 5,
                score: 0.2,
            },
        ];
        let best = filter_best_hits(hits);
        assert_eq!(best.len(), 1);
        assert_eq!(best[0].bucket_id, 3);
        assert!((best[0].score - 0.7).abs() < 1e-10);
    }
}
