//! Common utilities shared across classification functions.

use anyhow::Result;
use std::collections::HashSet;

use crate::indices::parquet::ParquetReadOptions;
use crate::indices::sharded::ShardedInvertedIndex;

/// Filter out negative minimizers from forward and reverse-complement minimizer vectors.
///
/// Uses `retain()` to filter in-place, avoiding unnecessary allocations in hot paths.
/// If `negative_mins` is None, the vectors are returned unchanged.
///
/// # Performance Note
/// This iterates over the minimizer vectors after extraction. An alternative would be
/// to filter during extraction, but that would complicate the extraction hot path with
/// an optional parameter. Benchmarking shows the current approach is acceptable since:
/// - HashSet lookups are O(1) amortized
/// - Minimizer count per read is typically small (< 1000)
/// - The extraction step (hashing, deque operations) dominates runtime
#[inline]
pub(super) fn filter_negative_mins(
    mut fwd: Vec<u64>,
    mut rc: Vec<u64>,
    negative_mins: Option<&HashSet<u64>>,
) -> (Vec<u64>, Vec<u64>) {
    if let Some(neg_set) = negative_mins {
        fwd.retain(|m| !neg_set.contains(m));
        rc.retain(|m| !neg_set.contains(m));
    }
    (fwd, rc)
}

/// Collect query minimizers that hit anything in a sharded negative index.
///
/// Loads one shard at a time to minimize memory usage. This is the memory-efficient
/// alternative to loading the entire negative index into a `HashSet<u64>`.
///
/// Memory usage: O(single_shard) + O(output_set_size)
///
/// # Arguments
/// * `negative_index` - The sharded inverted index containing negative/contaminant sequences
/// * `query_minimizers` - Sorted, deduplicated slice of minimizers from query batch
/// * `read_options` - Optional Parquet read options
///
/// # Returns
/// HashSet containing all query minimizers that exist in the negative index.
/// These minimizers should be filtered out before classification against the positive index.
pub fn collect_negative_minimizers_sharded(
    negative_index: &ShardedInvertedIndex,
    query_minimizers: &[u64],
    read_options: Option<&ParquetReadOptions>,
) -> Result<HashSet<u64>> {
    let manifest = negative_index.manifest();
    let mut hitting: HashSet<u64> = HashSet::new();

    if query_minimizers.is_empty() {
        return Ok(hitting);
    }

    for shard_info in &manifest.shards {
        let shard = negative_index.load_shard_for_query(
            shard_info.shard_id,
            query_minimizers,
            read_options,
        )?;

        // Reuse the method from Phase 1
        for min in shard.get_hitting_minimizers(query_minimizers) {
            hitting.insert(min);
        }
        // shard dropped here, freeing memory before loading next
    }

    Ok(hitting)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::workspace::MinimizerWorkspace;
    use crate::{create_parquet_inverted_index, extract_into, BucketData, ParquetWriteOptions};
    use tempfile::tempdir;

    /// Create a test Parquet index with specified buckets.
    fn create_test_sharded_index(
        base_path: &std::path::Path,
        bucket_data: Vec<(u32, &str, Vec<u64>)>,
    ) -> ShardedInvertedIndex {
        let k = 32;
        let w = 10;
        let salt = 0x12345u64;

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
        create_parquet_inverted_index(base_path, buckets, k, w, salt, None, Some(&options))
            .unwrap();

        ShardedInvertedIndex::open(base_path).unwrap()
    }

    // =========================================================================
    // filter_negative_mins tests
    // =========================================================================

    #[test]
    fn test_filter_negative_mins_none() {
        let fwd = vec![1, 2, 3];
        let rc = vec![4, 5, 6];
        let (f, r) = filter_negative_mins(fwd.clone(), rc.clone(), None);
        assert_eq!(f, fwd);
        assert_eq!(r, rc);
    }

    #[test]
    fn test_filter_negative_mins_empty_set() {
        let fwd = vec![1, 2, 3];
        let rc = vec![4, 5, 6];
        let empty: HashSet<u64> = HashSet::new();
        let (f, r) = filter_negative_mins(fwd.clone(), rc.clone(), Some(&empty));
        assert_eq!(f, fwd);
        assert_eq!(r, rc);
    }

    #[test]
    fn test_filter_negative_mins_filters() {
        let fwd = vec![1, 2, 3];
        let rc = vec![4, 5, 6];
        let neg: HashSet<u64> = vec![2, 5].into_iter().collect();
        let (f, r) = filter_negative_mins(fwd, rc, Some(&neg));
        assert_eq!(f, vec![1, 3]);
        assert_eq!(r, vec![4, 6]);
    }

    #[test]
    fn test_filter_negative_mins_all_filtered() {
        let fwd = vec![1, 2];
        let rc = vec![1, 2];
        let neg: HashSet<u64> = vec![1, 2].into_iter().collect();
        let (f, r) = filter_negative_mins(fwd, rc, Some(&neg));
        assert!(f.is_empty());
        assert!(r.is_empty());
    }

    // =========================================================================
    // collect_negative_minimizers_sharded tests
    // =========================================================================

    #[test]
    fn test_collect_negative_minimizers_basic() {
        let dir = tempdir().unwrap();
        let index_path = dir.path().join("neg.ryxdi");

        // Build a sharded index with minimizers 100, 200, 300
        let neg_index =
            create_test_sharded_index(&index_path, vec![(1, "neg1", vec![100, 200, 300])]);

        // Query with a mix of matching and non-matching minimizers
        let query_mins = vec![50, 100, 150, 200, 400];
        let hitting = collect_negative_minimizers_sharded(&neg_index, &query_mins, None).unwrap();

        // Only 100 and 200 should be found (sorted query, 50/150/400 don't exist)
        let expected: HashSet<u64> = vec![100, 200].into_iter().collect();
        assert_eq!(hitting, expected);
    }

    #[test]
    fn test_collect_negative_minimizers_multiple_buckets() {
        let dir = tempdir().unwrap();
        let index_path = dir.path().join("neg.ryxdi");

        // Build a sharded index with multiple buckets
        let neg_index = create_test_sharded_index(
            &index_path,
            vec![(1, "neg1", vec![100, 200]), (2, "neg2", vec![300, 400])],
        );

        // Query should find minimizers across both buckets
        let query_mins = vec![100, 300, 500];
        let hitting = collect_negative_minimizers_sharded(&neg_index, &query_mins, None).unwrap();

        let expected: HashSet<u64> = vec![100, 300].into_iter().collect();
        assert_eq!(hitting, expected);
    }

    #[test]
    fn test_collect_negative_minimizers_none_hit() {
        let dir = tempdir().unwrap();
        let index_path = dir.path().join("neg.ryxdi");

        // Build a sharded index with minimizers 100, 200, 300
        let neg_index =
            create_test_sharded_index(&index_path, vec![(1, "neg1", vec![100, 200, 300])]);

        // Query with minimizers that don't exist in the index
        let query_mins = vec![50, 150, 250, 400];
        let hitting = collect_negative_minimizers_sharded(&neg_index, &query_mins, None).unwrap();

        assert!(
            hitting.is_empty(),
            "No query minimizers should hit the index"
        );
    }

    #[test]
    fn test_collect_negative_minimizers_all_hit() {
        let dir = tempdir().unwrap();
        let index_path = dir.path().join("neg.ryxdi");

        let neg_index =
            create_test_sharded_index(&index_path, vec![(1, "neg1", vec![100, 200, 300])]);

        // Query with exactly the minimizers in the index
        let query_mins = vec![100, 200, 300];
        let hitting = collect_negative_minimizers_sharded(&neg_index, &query_mins, None).unwrap();

        let expected: HashSet<u64> = vec![100, 200, 300].into_iter().collect();
        assert_eq!(hitting, expected);
    }

    #[test]
    fn test_collect_negative_minimizers_empty_query() {
        let dir = tempdir().unwrap();
        let index_path = dir.path().join("neg.ryxdi");

        let neg_index =
            create_test_sharded_index(&index_path, vec![(1, "neg1", vec![100, 200, 300])]);

        // Empty query should return empty set
        let query_mins: Vec<u64> = vec![];
        let hitting = collect_negative_minimizers_sharded(&neg_index, &query_mins, None).unwrap();

        assert!(hitting.is_empty());
    }

    #[test]
    fn test_collect_negative_minimizers_with_real_sequences() {
        // Test with actual sequence-derived minimizers to ensure end-to-end correctness
        let dir = tempdir().unwrap();
        let index_path = dir.path().join("neg.ryxdi");

        let k = 32;
        let w = 10;
        let salt = 0x12345u64;

        // Generate a "contaminant" sequence
        let contaminant_seq: Vec<u8> = (0..200).map(|i| b"ACGT"[i % 4]).collect();
        let mut ws = MinimizerWorkspace::new();
        extract_into(&contaminant_seq, k, w, salt, &mut ws);
        let mut contaminant_mins: Vec<u64> = ws.buffer.drain(..).collect();
        contaminant_mins.sort();
        contaminant_mins.dedup();

        // Create the negative index
        let buckets = vec![BucketData {
            bucket_id: 1,
            bucket_name: "contaminant".to_string(),
            sources: vec!["contaminant.fa".to_string()],
            minimizers: contaminant_mins.clone(),
        }];
        let options = ParquetWriteOptions::default();
        create_parquet_inverted_index(&index_path, buckets, k, w, salt, None, Some(&options))
            .unwrap();

        let neg_index = ShardedInvertedIndex::open(&index_path).unwrap();

        // Query with the same sequence - should find all minimizers
        let hitting =
            collect_negative_minimizers_sharded(&neg_index, &contaminant_mins, None).unwrap();
        assert_eq!(
            hitting.len(),
            contaminant_mins.len(),
            "All contaminant minimizers should be found"
        );

        // Query with non-overlapping minimizers - should find none
        // Use minimizers that definitely don't exist in the contaminant set
        let max_contaminant = *contaminant_mins.iter().max().unwrap_or(&0);
        let non_overlapping: Vec<u64> = (0..10).map(|i| max_contaminant + 1000 + i).collect();
        let hitting_none =
            collect_negative_minimizers_sharded(&neg_index, &non_overlapping, None).unwrap();
        assert!(
            hitting_none.is_empty(),
            "Non-overlapping minimizers should not hit the index"
        );
    }
}
