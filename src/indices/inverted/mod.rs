//! Inverted index for fast minimizer → bucket lookups.
//!
//! The inverted index maps each unique minimizer to the set of bucket IDs
//! that contain it, using a CSR (Compressed Sparse Row) format. This enables
//! O(Q * log(U)) lookups where Q = query minimizers and U = unique minimizers.
//!
//! This module is organized into submodules:
//! - `query`: Query-side inverted index for merge-join classification
//! - `shard_parquet`: Parquet format serialization
//! - `query_loading`: Query-aware Parquet loading with bloom filters

// The k-way merge heap type is complex but clear in context
#![allow(clippy::type_complexity)]

mod query;
mod query_loading;
mod shard_parquet;

// Re-export public types
pub use query::QueryInvertedIndex;
pub use query_loading::{get_row_group_ranges, load_row_group_pairs, RowGroupRangeInfo};

use crate::error::{Result, RypeError};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use crate::types::IndexMetadata;

/// CSR-format inverted index for fast minimizer → bucket lookups.
///
/// # Invariants
/// - `minimizers` is sorted in ascending order with no duplicates
/// - `offsets.len() == minimizers.len() + 1`
/// - `offsets[0] == 0`
/// - `offsets` is monotonically increasing
/// - `offsets[minimizers.len()] == bucket_ids.len()`
/// - For each minimizer at index i, the associated bucket IDs are `bucket_ids[offsets[i]..offsets[i+1]]`
///
/// # Thread Safety
/// InvertedIndex can be shared across threads for concurrent classification.
#[derive(Debug)]
pub struct InvertedIndex {
    pub k: usize,
    pub w: usize,
    pub salt: u64,
    pub(crate) source_hash: u64,
    pub(crate) minimizers: Vec<u64>,
    pub(crate) offsets: Vec<u32>,
    pub(crate) bucket_ids: Vec<u32>,
}

impl InvertedIndex {
    /// Compute a hash from index metadata for validation.
    /// Hash is computed from sorted (bucket_id, minimizer_count) pairs.
    pub fn compute_metadata_hash(metadata: &IndexMetadata) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut pairs: Vec<(u32, usize)> = metadata
            .bucket_minimizer_counts
            .iter()
            .map(|(&id, &count)| (id, count))
            .collect();
        pairs.sort_by_key(|(id, _)| *id);

        let mut hasher = DefaultHasher::new();
        for (id, count) in pairs {
            id.hash(&mut hasher);
            count.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Core k-way merge algorithm to build inverted index from bucket data.
    pub fn build_from_bucket_map(
        k: usize,
        w: usize,
        salt: u64,
        buckets: &HashMap<u32, Vec<u64>>,
        metadata: &IndexMetadata,
    ) -> Self {
        let total_entries: usize = buckets.values().map(|v| v.len()).sum();

        // Handle empty case
        if buckets.is_empty() || total_entries == 0 {
            return InvertedIndex {
                k,
                w,
                salt,
                source_hash: Self::compute_metadata_hash(metadata),
                minimizers: Vec::new(),
                offsets: vec![0],
                bucket_ids: Vec::new(),
            };
        }

        let estimated_unique = total_entries / 2;
        let mut minimizers: Vec<u64> = Vec::with_capacity(estimated_unique);
        let mut offsets: Vec<u32> = Vec::with_capacity(estimated_unique + 1);
        let mut bucket_ids_out: Vec<u32> = Vec::with_capacity(total_entries);

        offsets.push(0);

        // K-way merge using a min-heap
        // Each heap entry: (Reverse((minimizer_value, bucket_id)), data_index, position_in_bucket)
        let bucket_data: Vec<(u32, &[u64])> = buckets
            .iter()
            .filter(|(_, mins)| !mins.is_empty())
            .map(|(&id, mins)| (id, mins.as_slice()))
            .collect();

        let mut heap: BinaryHeap<(Reverse<(u64, u32)>, usize, usize)> =
            BinaryHeap::with_capacity(bucket_data.len());

        for (idx, &(bucket_id, mins)) in bucket_data.iter().enumerate() {
            heap.push((Reverse((mins[0], bucket_id)), idx, 0));
        }

        let mut current_min: Option<u64> = None;
        let mut current_bucket_ids: Vec<u32> = Vec::new();

        while let Some((Reverse((min_val, _)), data_idx, pos)) = heap.pop() {
            let (bucket_id, bucket_mins) = bucket_data[data_idx];

            if current_min != Some(min_val) {
                // Flush previous minimizer's bucket list
                if current_min.is_some() {
                    current_bucket_ids.sort_unstable();
                    current_bucket_ids.dedup();
                    bucket_ids_out.extend_from_slice(&current_bucket_ids);
                    offsets.push(
                        u32::try_from(bucket_ids_out.len())
                            .expect("CSR offset overflow: bucket_ids exceeded u32::MAX"),
                    );
                    current_bucket_ids.clear();
                }
                minimizers.push(min_val);
                current_min = Some(min_val);
            }

            current_bucket_ids.push(bucket_id);

            // Push next element from this bucket if available
            let next_pos = pos + 1;
            if next_pos < bucket_mins.len() {
                heap.push((
                    Reverse((bucket_mins[next_pos], bucket_id)),
                    data_idx,
                    next_pos,
                ));
            }
        }

        // Flush final minimizer's bucket list
        if current_min.is_some() {
            current_bucket_ids.sort_unstable();
            current_bucket_ids.dedup();
            bucket_ids_out.extend_from_slice(&current_bucket_ids);
            offsets.push(
                u32::try_from(bucket_ids_out.len())
                    .expect("CSR offset overflow: bucket_ids exceeded u32::MAX"),
            );
        }

        minimizers.shrink_to_fit();
        offsets.shrink_to_fit();
        bucket_ids_out.shrink_to_fit();

        InvertedIndex {
            k,
            w,
            salt,
            source_hash: Self::compute_metadata_hash(metadata),
            minimizers,
            offsets,
            bucket_ids: bucket_ids_out,
        }
    }

    /// Validate that this inverted index matches the given metadata.
    pub fn validate_against_metadata(&self, metadata: &IndexMetadata) -> Result<()> {
        if self.k != metadata.k || self.w != metadata.w || self.salt != metadata.salt {
            return Err(RypeError::validation(format!(
                "Inverted index parameters don't match source index.\n  \
                 Inverted: K={}, W={}, salt=0x{:x}\n  \
                 Source:   K={}, W={}, salt=0x{:x}",
                self.k, self.w, self.salt, metadata.k, metadata.w, metadata.salt
            )));
        }

        let expected_hash = Self::compute_metadata_hash(metadata);
        if self.source_hash != expected_hash {
            return Err(RypeError::validation(format!(
                "Inverted index is stale (hash 0x{:016x} != expected 0x{:016x}). \
                 The source index has been modified. Regenerate the index.",
                self.source_hash, expected_hash
            )));
        }

        Ok(())
    }

    /// Get bucket hit counts for a sorted query using hybrid binary search.
    ///
    /// # Arguments
    /// * `query` - A sorted, deduplicated slice of minimizer values
    ///
    /// # Returns
    /// HashMap of bucket_id -> hit_count for all buckets matching at least one query minimizer.
    pub fn get_bucket_hits(&self, query: &[u64]) -> HashMap<u32, usize> {
        let mut hits: HashMap<u32, usize> = HashMap::new();

        if query.is_empty() || self.minimizers.is_empty() {
            return hits;
        }

        let mut search_start = 0;

        for &q in query {
            if search_start >= self.minimizers.len() {
                break;
            }

            match self.minimizers[search_start..].binary_search(&q) {
                Ok(relative_idx) => {
                    let abs_idx = search_start + relative_idx;
                    let start = self.offsets[abs_idx] as usize;
                    let end = self.offsets[abs_idx + 1] as usize;
                    for &bid in &self.bucket_ids[start..end] {
                        *hits.entry(bid).or_insert(0) += 1;
                    }
                    search_start = abs_idx + 1;
                }
                Err(relative_idx) => {
                    search_start += relative_idx;
                }
            }
        }

        hits
    }

    /// Returns the number of unique minimizers in the index.
    pub fn num_minimizers(&self) -> usize {
        self.minimizers.len()
    }

    /// Returns the total number of bucket ID entries.
    pub fn num_bucket_entries(&self) -> usize {
        self.bucket_ids.len()
    }

    /// Get the sorted minimizer array (for debugging/inspection).
    pub fn minimizers(&self) -> &[u64] {
        &self.minimizers
    }

    /// Get the CSR offsets array (for debugging/inspection).
    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    /// Get the bucket IDs array (for debugging/inspection).
    pub fn bucket_ids(&self) -> &[u32] {
        &self.bucket_ids
    }

    /// Get query minimizers that exist in this index.
    ///
    /// Returns a Vec of minimizers (preserving query order) that hit anything
    /// in the index. This is useful for negative filtering where we need to
    /// know which query minimizers should be excluded.
    ///
    /// # Arguments
    /// * `query` - A sorted, deduplicated slice of minimizer values
    ///
    /// # Returns
    /// Vec of minimizers from query that exist in this index.
    pub fn get_hitting_minimizers(&self, query: &[u64]) -> Vec<u64> {
        let mut hits = Vec::new();

        if query.is_empty() || self.minimizers.is_empty() {
            return hits;
        }

        let mut search_start = 0;

        for &q in query {
            if search_start >= self.minimizers.len() {
                break;
            }

            match self.minimizers[search_start..].binary_search(&q) {
                Ok(relative_idx) => {
                    hits.push(q);
                    search_start += relative_idx + 1;
                }
                Err(relative_idx) => {
                    search_start += relative_idx;
                }
            }
        }

        hits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_inverted_index() -> InvertedIndex {
        let mut buckets: HashMap<u32, Vec<u64>> = HashMap::new();
        buckets.insert(1, vec![100, 200, 300]);
        buckets.insert(2, vec![200, 300, 400]);
        buckets.insert(3, vec![500, 600]);

        let metadata = IndexMetadata {
            k: 64,
            w: 50,
            salt: 0x1234,
            bucket_names: HashMap::new(),
            bucket_sources: HashMap::new(),
            bucket_minimizer_counts: buckets.iter().map(|(&id, v)| (id, v.len())).collect(),
            largest_shard_entries: 0,
        };

        InvertedIndex::build_from_bucket_map(64, 50, 0x1234, &buckets, &metadata)
    }

    #[test]
    fn test_inverted_index_build() {
        let inverted = build_test_inverted_index();

        assert_eq!(inverted.k, 64);
        assert_eq!(inverted.w, 50);
        assert_eq!(inverted.salt, 0x1234);
        assert_eq!(inverted.num_minimizers(), 6);
        assert_eq!(inverted.num_bucket_entries(), 8);
    }

    #[test]
    fn test_inverted_index_get_bucket_hits() {
        let inverted = build_test_inverted_index();

        let query = vec![200, 300, 500];
        let hits = inverted.get_bucket_hits(&query);

        assert_eq!(hits.get(&1), Some(&2));
        assert_eq!(hits.get(&2), Some(&2));
        assert_eq!(hits.get(&3), Some(&1));
    }

    #[test]
    fn test_inverted_index_get_bucket_hits_no_matches() {
        let inverted = build_test_inverted_index();

        let query = vec![999, 1000, 1001];
        let hits = inverted.get_bucket_hits(&query);

        assert!(hits.is_empty());
    }

    #[test]
    fn test_inverted_index_empty() {
        let buckets: HashMap<u32, Vec<u64>> = HashMap::new();
        let metadata = IndexMetadata {
            k: 64,
            w: 50,
            salt: 0,
            bucket_names: HashMap::new(),
            bucket_sources: HashMap::new(),
            bucket_minimizer_counts: HashMap::new(),
            largest_shard_entries: 0,
        };

        let inverted = InvertedIndex::build_from_bucket_map(64, 50, 0, &buckets, &metadata);

        assert_eq!(inverted.num_minimizers(), 0);
        assert_eq!(inverted.num_bucket_entries(), 0);

        let hits = inverted.get_bucket_hits(&[100, 200, 300]);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_inverted_index_hybrid_search_correctness() {
        let mut buckets: HashMap<u32, Vec<u64>> = HashMap::new();
        buckets.insert(1, (0..1000).map(|i| i * 10).collect());

        let metadata = IndexMetadata {
            k: 64,
            w: 50,
            salt: 0,
            bucket_names: HashMap::new(),
            bucket_sources: HashMap::new(),
            bucket_minimizer_counts: buckets.iter().map(|(&id, v)| (id, v.len())).collect(),
            largest_shard_entries: 0,
        };

        let inverted = InvertedIndex::build_from_bucket_map(64, 50, 0, &buckets, &metadata);

        let query: Vec<u64> = vec![50, 500, 5000, 9990];
        let hits = inverted.get_bucket_hits(&query);

        assert_eq!(hits.get(&1), Some(&4));
    }

    #[test]
    fn test_build_high_overlap() {
        // All buckets share all minimizers (maximum overlap)
        let mut buckets: HashMap<u32, Vec<u64>> = HashMap::new();
        let shared_mins = vec![100, 200, 300, 400, 500];
        buckets.insert(1, shared_mins.clone());
        buckets.insert(2, shared_mins.clone());
        buckets.insert(3, shared_mins.clone());
        buckets.insert(4, shared_mins.clone());

        let metadata = IndexMetadata {
            k: 64,
            w: 50,
            salt: 0,
            bucket_names: HashMap::new(),
            bucket_sources: HashMap::new(),
            bucket_minimizer_counts: buckets.iter().map(|(&id, v)| (id, v.len())).collect(),
            largest_shard_entries: 0,
        };

        let inverted = InvertedIndex::build_from_bucket_map(64, 50, 0, &buckets, &metadata);

        // Only 5 unique minimizers despite 20 total entries
        assert_eq!(inverted.num_minimizers(), 5);
        // Each minimizer maps to all 4 buckets = 20 bucket entries
        assert_eq!(inverted.num_bucket_entries(), 20);

        let hits = inverted.get_bucket_hits(&[100, 200, 300]);
        assert_eq!(hits.get(&1), Some(&3));
        assert_eq!(hits.get(&2), Some(&3));
        assert_eq!(hits.get(&3), Some(&3));
        assert_eq!(hits.get(&4), Some(&3));
    }

    // Tests for get_hitting_minimizers
    #[test]
    fn test_get_hitting_minimizers_basic() {
        // Index has minimizers: 100, 200, 300, 400, 500, 600
        let inverted = build_test_inverted_index();
        let query = vec![50, 100, 150, 200, 250];
        let hits = inverted.get_hitting_minimizers(&query);
        // Only 100, 200 exist in index
        assert_eq!(hits, vec![100, 200]);
    }

    #[test]
    fn test_get_hitting_minimizers_empty_query() {
        let inverted = build_test_inverted_index();
        let hits = inverted.get_hitting_minimizers(&[]);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_get_hitting_minimizers_no_matches() {
        let inverted = build_test_inverted_index();
        let hits = inverted.get_hitting_minimizers(&[50, 150, 250, 700]);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_get_hitting_minimizers_all_hit() {
        let inverted = build_test_inverted_index();
        // Query exactly the minimizers in the index
        let query = vec![100, 200, 300, 400, 500, 600];
        let hits = inverted.get_hitting_minimizers(&query);
        assert_eq!(hits, vec![100, 200, 300, 400, 500, 600]);
    }

    #[test]
    fn test_get_hitting_minimizers_partial_overlap() {
        let inverted = build_test_inverted_index();
        // Mix of hits and misses
        let query = vec![100, 150, 300, 450, 600, 700];
        let hits = inverted.get_hitting_minimizers(&query);
        assert_eq!(hits, vec![100, 300, 600]);
    }
}
