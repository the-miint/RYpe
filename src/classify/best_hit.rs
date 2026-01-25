//! Best-hit filtering for classification results.
//!
//! Provides filtering to keep only the highest-scoring bucket per query.

use crate::types::HitResult;
use std::collections::hash_map::Entry;
use std::collections::HashMap;

/// Filter a list of HitResults to keep only the best hit per query.
///
/// When a query has multiple hits (to different buckets), this returns only
/// the hit with the highest score. If there's a tie, the first hit encountered
/// is kept (arbitrary but deterministic given input order).
///
/// # Arguments
/// * `hits` - Vector of all hits above threshold
///
/// # Returns
/// Vector with at most one HitResult per unique query_id
pub fn filter_best_hits(hits: Vec<HitResult>) -> Vec<HitResult> {
    // Estimate capacity: assume ~2 hits per query on average
    // This is better than hits.len() (over-allocates) or 0 (many resizes)
    let estimated_queries = (hits.len() / 2).max(1);
    let mut best_by_query: HashMap<i64, HitResult> = HashMap::with_capacity(estimated_queries);

    for hit in hits {
        match best_by_query.entry(hit.query_id) {
            Entry::Occupied(mut entry) => {
                // Only replace if strictly better (not equal - keeps first on tie)
                if hit.score > entry.get().score {
                    entry.insert(hit);
                }
            }
            Entry::Vacant(entry) => {
                entry.insert(hit);
            }
        }
    }
    best_by_query.into_values().collect()
}
