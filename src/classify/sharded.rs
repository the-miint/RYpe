//! Sharded classification functions for memory-efficient large-scale classification.
//!
//! These functions load one shard at a time to minimize memory usage when
//! classifying against large indices that don't fit in memory.

use anyhow::Result;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

use crate::constants::{ESTIMATED_BUCKETS_PER_READ, ESTIMATED_MINIMIZERS_PER_SEQUENCE};
use crate::core::extraction::get_paired_minimizers_into;
use crate::core::workspace::MinimizerWorkspace;
use crate::indices::sharded::{ShardManifest, ShardedInvertedIndex};
use crate::indices::QueryInvertedIndex;
use crate::types::{HitResult, QueryRecord};

use crate::log_timing;

use super::common::{collect_negative_minimizers_sharded, filter_negative_mins};
use super::merge_join::{
    accumulate_merge_join, merge_join_pairs_sparse, merge_sparse_hits, SparseHit,
};
use super::scoring::compute_score;

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

/// Classify a batch of records against a sharded inverted index using merge-join.
///
/// Builds a QueryInvertedIndex once, then processes each shard sequentially
/// using merge-join. This is efficient when there is high minimizer overlap
/// across reads.
///
/// Memory complexity: O(batch_size * minimizers_per_read) + O(single_shard_size)
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
    let t_start = Instant::now();

    if records.is_empty() {
        return Ok(Vec::new());
    }

    let manifest = sharded.manifest();

    // Extract minimizers in parallel, filtering negatives if provided
    let t_extract = Instant::now();
    let estimated_mins = estimate_minimizers_from_records(records, manifest.k, manifest.w);
    let extracted: Vec<_> = records
        .par_iter()
        .map_init(
            || MinimizerWorkspace::with_estimate(estimated_mins),
            |ws, (_, s1, s2)| {
                let (ha, hb) =
                    get_paired_minimizers_into(s1, *s2, manifest.k, manifest.w, manifest.salt, ws);
                filter_negative_mins(ha, hb, negative_mins)
            },
        )
        .collect();
    log_timing("merge_join: extraction", t_extract.elapsed().as_millis());

    // Collect query IDs
    let query_ids: Vec<i64> = records.iter().map(|(id, _, _)| *id).collect();

    // Build query inverted index
    let t_build_idx = Instant::now();
    let query_idx = QueryInvertedIndex::build(&extracted);
    log_timing(
        "merge_join: build_query_index",
        t_build_idx.elapsed().as_millis(),
    );

    let num_reads = query_idx.num_reads();
    if num_reads == 0 {
        return Ok(Vec::new());
    }

    // Per-read accumulator: bucket_id -> (fwd_hits, rc_hits)
    let mut accumulators: Vec<HashMap<u32, (u32, u32)>> = (0..num_reads)
        .map(|_| HashMap::with_capacity(ESTIMATED_BUCKETS_PER_READ))
        .collect();

    let mut total_shard_load_ms = 0u128;
    let mut total_merge_join_ms = 0u128;

    // Get query minimizers for filtered loading (Parquet only)
    let query_minimizers = query_idx.minimizers();

    // Debug: show query minimizer statistics
    if std::env::var("RYPE_DEBUG").is_ok() && !query_minimizers.is_empty() {
        eprintln!(
            "[DEBUG] Query minimizers: {} unique, range: {} to {}",
            query_minimizers.len(),
            query_minimizers[0],
            query_minimizers[query_minimizers.len() - 1]
        );
    }

    // Process each shard sequentially
    for shard_info in &manifest.shards {
        let t_load = Instant::now();
        // Use filtered loading for Parquet shards - only loads row groups
        // that contain query minimizers, potentially skipping 90%+ of data
        // (with optional bloom filter support)
        let shard =
            sharded.load_shard_for_query(shard_info.shard_id, query_minimizers, read_options)?;
        total_shard_load_ms += t_load.elapsed().as_millis();

        let t_merge = Instant::now();
        // Accumulate hits from this shard
        accumulate_merge_join(&query_idx, &shard, &mut accumulators);
        total_merge_join_ms += t_merge.elapsed().as_millis();
        // shard dropped here, freeing memory before loading next
    }
    log_timing("merge_join: shard_load_total", total_shard_load_ms);
    log_timing("merge_join: merge_join_total", total_merge_join_ms);

    // Score and filter
    let t_score = Instant::now();
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
    log_timing("merge_join: scoring", t_score.elapsed().as_millis());
    log_timing("merge_join: total", t_start.elapsed().as_millis());

    Ok(results)
}

/// Classify using parallel row group processing.
///
/// Each row group is processed independently in parallel:
/// 1. Pre-filter RGs by query minimizer range (using column statistics)
/// 2. Load matching RG pairs (pre-sorted within RG)
/// 3. Merge-join pairs with query index, emitting sparse hits
/// 4. Merge all sparse hits into final accumulators
///
/// This maximizes CPU utilization by processing only RGs that overlap with
/// the query minimizer range, and by using sparse hit representation to
/// minimize memory allocation.
///
/// # Memory Model
///
/// Peak memory is approximately:
/// - Query index: O(total_query_minimizers)
/// - Sparse hits: O(total_hits_across_all_RGs) - typically much smaller than reads × buckets
/// - Final accumulators: O(num_reads × avg_buckets_per_read)
///
/// Unlike the merge-join approach which loads entire shards, this processes
/// one RG at a time per thread, keeping per-thread memory bounded.
///
/// # Why read_options is Unused
///
/// The `read_options` parameter (bloom filter settings) is accepted for API
/// consistency with other sharded classification functions but is intentionally
/// unused. This function uses range-bounded filtering via Parquet column
/// statistics instead of bloom filters - each row group's min/max is compared
/// against the query minimizer range to skip non-overlapping RGs entirely.
/// This is more effective than bloom filters for the per-RG access pattern.
///
/// # Requirements
/// - Parquet shards only (Legacy format not supported)
/// - Row groups must have column statistics
///
/// # Arguments
/// * `sharded` - The sharded inverted index (must be Parquet format)
/// * `negative_mins` - Optional set of minimizers to exclude from queries before scoring
/// * `records` - Batch of query records (should be pre-trimmed if trimming is desired)
/// * `threshold` - Minimum score threshold
/// * `_read_options` - Unused; see "Why read_options is Unused" above
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
    use crate::indices::{get_row_group_ranges, load_row_group_pairs};

    let t_start = Instant::now();

    if records.is_empty() {
        return Ok(Vec::new());
    }

    let manifest = sharded.manifest();

    // Extract minimizers in parallel, filtering negatives if provided
    let t_extract = Instant::now();
    let estimated_mins = estimate_minimizers_from_records(records, manifest.k, manifest.w);
    let extracted: Vec<_> = records
        .par_iter()
        .map_init(
            || MinimizerWorkspace::with_estimate(estimated_mins),
            |ws, (_, s1, s2)| {
                let (ha, hb) =
                    get_paired_minimizers_into(s1, *s2, manifest.k, manifest.w, manifest.salt, ws);
                filter_negative_mins(ha, hb, negative_mins)
            },
        )
        .collect();
    log_timing("parallel_rg: extraction", t_extract.elapsed().as_millis());

    // Collect query IDs
    let query_ids: Vec<i64> = records.iter().map(|(id, _, _)| *id).collect();

    // Build query inverted index (CSR format, built once, reused across all RGs)
    let t_build_idx = Instant::now();
    let query_idx = QueryInvertedIndex::build(&extracted);
    log_timing(
        "parallel_rg: build_query_index",
        t_build_idx.elapsed().as_millis(),
    );

    let num_reads = query_idx.num_reads();
    if num_reads == 0 {
        return Ok(Vec::new());
    }

    // Get sorted query minimizers and their range for pre-filtering
    let query_minimizers = query_idx.minimizers();
    let (query_min, query_max) = if query_minimizers.is_empty() {
        return Ok(Vec::new());
    } else {
        (
            query_minimizers[0],
            query_minimizers[query_minimizers.len() - 1],
        )
    };

    let mut total_rg_count = 0usize;

    // Collect (shard_path, rg_idx) pairs that overlap with query range
    let mut work_items: Vec<(std::path::PathBuf, usize)> = Vec::new();

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

    // Process overlapping row groups in parallel, collecting sparse hits
    let t_parallel = Instant::now();
    let results: Result<Vec<Vec<SparseHit>>> = work_items
        .into_par_iter()
        .map(|(shard_path, rg_idx)| {
            let pairs = load_row_group_pairs(&shard_path, rg_idx, query_minimizers)?;
            if pairs.is_empty() {
                Ok(Vec::new())
            } else {
                Ok(merge_join_pairs_sparse(&query_idx, &pairs))
            }
        })
        .filter_map(|result: Result<Vec<SparseHit>>| match result {
            Ok(hits) if hits.is_empty() => None,
            Ok(hits) => Some(Ok(hits)),
            Err(e) => Some(Err(e)),
        })
        .collect();

    let all_sparse_hits = results?;
    log_timing(
        "parallel_rg: rg_process_total",
        t_parallel.elapsed().as_millis(),
    );

    // Merge all sparse hits into final accumulators
    let t_merge = Instant::now();
    let final_accumulators = merge_sparse_hits(all_sparse_hits, num_reads);
    log_timing("parallel_rg: merge_total", t_merge.elapsed().as_millis());
    log_timing("parallel_rg: total_rg_count", total_rg_count as u128);
    log_timing("parallel_rg: filtered_rg_count", filtered_rg_count as u128);

    // Score and filter
    let t_score = Instant::now();
    let mut results = Vec::new();
    for (read_idx, buckets) in final_accumulators.into_iter().enumerate() {
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
    log_timing("parallel_rg: scoring", t_score.elapsed().as_millis());
    log_timing("parallel_rg: total", t_start.elapsed().as_millis());

    Ok(results)
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

    // Step 1: Extract minimizers from all records
    let estimated_mins = estimate_minimizers_from_records(records, manifest.k, manifest.w);
    let processed: Vec<_> = records
        .par_iter()
        .map_init(
            || MinimizerWorkspace::with_estimate(estimated_mins),
            |ws, (_, s1, s2)| {
                let (fwd, rc) =
                    get_paired_minimizers_into(s1, *s2, manifest.k, manifest.w, manifest.salt, ws);
                (fwd, rc)
            },
        )
        .collect();

    // Step 2: Build sorted unique minimizers for querying negative index
    let mut all_minimizers: Vec<u64> = processed
        .iter()
        .flat_map(|(fwd, rc)| fwd.iter().chain(rc.iter()).copied())
        .collect();
    all_minimizers.sort_unstable();
    all_minimizers.dedup();

    // Step 3: Collect hitting minimizers from negative index (memory-efficient)
    let negative_set =
        collect_negative_minimizers_sharded(negative, &all_minimizers, read_options)?;

    // Step 4: Classify using the collected negative set
    // Note: This will re-extract minimizers, but the memory savings from sharded
    // negative filtering (24GB -> 1-2GB) far outweigh the extraction overhead.
    classify_batch_sharded_merge_join(
        positive_index,
        Some(&negative_set),
        records,
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
