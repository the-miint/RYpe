//! Sharded classification functions for memory-efficient large-scale classification.
//!
//! These functions load one shard at a time to minimize memory usage when
//! classifying against large indices that don't fit in memory.

use anyhow::Result;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::time::Instant;

use crate::constants::ESTIMATED_BUCKETS_PER_READ;
use crate::core::extraction::get_paired_minimizers_into;
use crate::core::workspace::MinimizerWorkspace;
use crate::indices::sharded::ShardedInvertedIndex;
use crate::indices::sharded_main::ShardedMainIndex;
use crate::indices::QueryInvertedIndex;
use crate::types::{HitResult, QueryRecord};

use crate::log_timing;

use super::common::filter_negative_mins;
use super::merge_join::accumulate_merge_join;
use super::scoring::compute_score;

/// Classify a batch of records against a sharded inverted index.
///
/// Loads one shard at a time to minimize memory usage. Memory complexity
/// is O(batch_size * minimizers_per_read) + O(single_shard_size).
///
/// # Arguments
/// * `sharded` - The sharded inverted index
/// * `negative_mins` - Optional set of minimizers to exclude from queries before scoring
/// * `records` - Batch of query records
/// * `threshold` - Minimum score threshold
/// * `read_options` - Parquet read options (None = default behavior without bloom filters)
///
/// # Returns
/// Vector of HitResult for all (query, bucket) pairs meeting the threshold.
pub fn classify_batch_sharded_sequential(
    sharded: &ShardedInvertedIndex,
    negative_mins: Option<&HashSet<u64>>,
    records: &[QueryRecord],
    threshold: f64,
    read_options: Option<&crate::indices::parquet::ParquetReadOptions>,
) -> Result<Vec<HitResult>> {
    let t_start = Instant::now();
    let manifest = sharded.manifest();

    let t_extract = Instant::now();
    let processed: Vec<_> = records
        .par_iter()
        .map_init(MinimizerWorkspace::new, |ws, (id, s1, s2)| {
            let (ha, hb) =
                get_paired_minimizers_into(s1, *s2, manifest.k, manifest.w, manifest.salt, ws);
            let (fa, fb) = filter_negative_mins(ha, hb, negative_mins);
            (*id, fa, fb)
        })
        .collect();
    log_timing("sequential: extraction", t_extract.elapsed().as_millis());

    let all_hits: Vec<Mutex<HashMap<u32, (usize, usize)>>> = (0..processed.len())
        .map(|_| Mutex::new(HashMap::new()))
        .collect();

    // Build sorted unique minimizers for filtered loading (Parquet only)
    let mut all_minimizers: Vec<u64> = processed
        .iter()
        .flat_map(|(_, fwd, rc)| fwd.iter().chain(rc.iter()).copied())
        .collect();
    all_minimizers.sort_unstable();
    all_minimizers.dedup();

    let mut total_shard_load_ms = 0u128;
    let mut total_shard_query_ms = 0u128;

    for shard_info in &manifest.shards {
        let t_load = Instant::now();
        // Use filtered loading for Parquet shards (with optional bloom filter support)
        let shard =
            sharded.load_shard_for_query(shard_info.shard_id, &all_minimizers, read_options)?;
        total_shard_load_ms += t_load.elapsed().as_millis();

        let t_query = Instant::now();
        processed
            .par_iter()
            .enumerate()
            .for_each(|(idx, (_, fwd_mins, rc_mins))| {
                let fwd_hits = shard.get_bucket_hits(fwd_mins);
                let rc_hits = shard.get_bucket_hits(rc_mins);

                let mut hits = all_hits[idx].lock().unwrap();
                for (bucket_id, count) in fwd_hits {
                    hits.entry(bucket_id).or_insert((0, 0)).0 += count;
                }
                for (bucket_id, count) in rc_hits {
                    hits.entry(bucket_id).or_insert((0, 0)).1 += count;
                }
            });
        total_shard_query_ms += t_query.elapsed().as_millis();
        // shard dropped here, freeing memory before loading next
    }
    log_timing("sequential: shard_load_total", total_shard_load_ms);
    log_timing("sequential: shard_query_total", total_shard_query_ms);

    let t_score = Instant::now();
    let results: Vec<HitResult> = processed
        .iter()
        .enumerate()
        .flat_map(|(idx, (query_id, fwd_mins, rc_mins))| {
            let fwd_total = fwd_mins.len();
            let rc_total = rc_mins.len();
            let hits = all_hits[idx].lock().unwrap();

            hits.iter()
                .filter_map(|(&bucket_id, &(fwd_count, rc_count))| {
                    let score = compute_score(fwd_count, fwd_total, rc_count, rc_total);
                    if score >= threshold {
                        Some(HitResult {
                            query_id: *query_id,
                            bucket_id,
                            score,
                        })
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();
    log_timing("sequential: scoring", t_score.elapsed().as_millis());
    log_timing("sequential: total", t_start.elapsed().as_millis());

    Ok(results)
}

/// Classify a batch of records against a sharded inverted index using merge-join.
///
/// Builds a QueryInvertedIndex once, then processes each shard sequentially
/// using merge-join. This is more efficient than `classify_batch_sharded_sequential`
/// when there is high minimizer overlap across reads.
///
/// Memory complexity: O(batch_size * minimizers_per_read) + O(single_shard_size)
///
/// # Arguments
/// * `sharded` - The sharded inverted index
/// * `negative_mins` - Optional set of minimizers to exclude from queries before scoring
/// * `records` - Batch of query records
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
    let extracted: Vec<_> = records
        .par_iter()
        .map_init(MinimizerWorkspace::new, |ws, (_, s1, s2)| {
            let (ha, hb) =
                get_paired_minimizers_into(s1, *s2, manifest.k, manifest.w, manifest.salt, ws);
            filter_negative_mins(ha, hb, negative_mins)
        })
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

/// Classify a batch of records against a sharded main index.
///
/// Loads one shard at a time to minimize memory usage. For each shard,
/// uses binary search against the bucket minimizers (same algorithm as
/// `classify_batch`).
///
/// Memory complexity: O(batch_size * minimizers_per_read) + O(single_shard_buckets)
///
/// # Arguments
/// * `sharded` - The sharded main index
/// * `negative_mins` - Optional set of minimizers to exclude from queries before scoring
/// * `records` - Batch of query records
/// * `threshold` - Minimum score threshold
///
/// # Returns
/// Vector of HitResult for all (query, bucket) pairs meeting the threshold.
pub fn classify_batch_sharded_main(
    sharded: &ShardedMainIndex,
    negative_mins: Option<&HashSet<u64>>,
    records: &[QueryRecord],
    threshold: f64,
) -> Result<Vec<HitResult>> {
    let manifest = sharded.manifest();

    // Extract minimizers for all queries (parallel), filtering negatives if provided
    let processed: Vec<_> = records
        .par_iter()
        .map_init(MinimizerWorkspace::new, |ws, (id, s1, s2)| {
            let (ha, hb) =
                get_paired_minimizers_into(s1, *s2, manifest.k, manifest.w, manifest.salt, ws);
            let (fa, fb) = filter_negative_mins(ha, hb, negative_mins);
            (*id, fa, fb)
        })
        .collect();

    // Build inverted maps: minimizer → query indices
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

    // Accumulator: query_idx → bucket_id → (fwd_hits, rc_hits)
    let all_hits: Vec<Mutex<HashMap<u32, (usize, usize)>>> = (0..processed.len())
        .map(|_| Mutex::new(HashMap::new()))
        .collect();

    // Process each shard sequentially (load one at a time)
    for shard_info in &manifest.shards {
        let shard = sharded.load_shard(shard_info.shard_id)?;

        // Process all buckets in this shard in parallel
        shard
            .buckets
            .par_iter()
            .for_each(|(bucket_id, bucket_minimizers)| {
                let mut bucket_hits: HashMap<usize, (usize, usize)> = HashMap::new();

                // Binary search each unique query minimizer against this bucket
                for &m in &uniq_vec {
                    if bucket_minimizers.binary_search(&m).is_ok() {
                        if let Some(rs) = map_a.get(&m) {
                            for &r in rs {
                                bucket_hits.entry(r).or_insert((0, 0)).0 += 1;
                            }
                        }
                        if let Some(rs) = map_b.get(&m) {
                            for &r in rs {
                                bucket_hits.entry(r).or_insert((0, 0)).1 += 1;
                            }
                        }
                    }
                }

                // Merge hits into per-query accumulators
                for (r_idx, (fwd_count, rc_count)) in bucket_hits {
                    let mut hits = all_hits[r_idx].lock().unwrap();
                    let entry = hits.entry(*bucket_id).or_insert((0, 0));
                    entry.0 += fwd_count;
                    entry.1 += rc_count;
                }
            });
        // shard dropped here, freeing memory before loading next
    }

    // Score and filter
    let results: Vec<HitResult> = processed
        .iter()
        .enumerate()
        .flat_map(|(idx, (query_id, fwd_mins, rc_mins))| {
            let fwd_total = fwd_mins.len();
            let rc_total = rc_mins.len();
            let hits = all_hits[idx].lock().unwrap();

            hits.iter()
                .filter_map(|(&bucket_id, &(fwd_count, rc_count))| {
                    let score = compute_score(fwd_count, fwd_total, rc_count, rc_total);
                    if score >= threshold {
                        Some(HitResult {
                            query_id: *query_id,
                            bucket_id,
                            score,
                        })
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classify::batch::classify_batch;
    use crate::indices::main::Index;
    use crate::indices::sharded::{ShardFormat, ShardManifest};
    use crate::indices::InvertedIndex;

    #[test]
    fn test_classify_batch_sharded_sequential_matches_regular() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let base_path = dir.path().join("test.ryxdi");

        let mut index = Index::new(32, 10, 0x1234).unwrap();
        let mut ws = MinimizerWorkspace::new();

        let seq1 = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let seq2 = b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA";

        index.add_record(1, "ref1", seq1, &mut ws);
        index.add_record(2, "ref2", seq2, &mut ws);
        index.bucket_names.insert(1, "A".into());
        index.bucket_names.insert(2, "B".into());
        index.finalize_bucket(1);
        index.finalize_bucket(2);

        let inverted = InvertedIndex::build_from_index(&index);
        assert!(
            inverted.num_minimizers() > 0,
            "Inverted index should not be empty"
        );

        // Save as single shard (like we do for non-sharded main index)
        let shard_path = ShardManifest::shard_path(&base_path, 0);
        let shard_info = inverted.save_shard(&shard_path, 0, 0, inverted.num_minimizers(), true)?;

        let manifest = ShardManifest {
            k: inverted.k,
            w: inverted.w,
            salt: inverted.salt,
            source_hash: 0,
            total_minimizers: inverted.num_minimizers(),
            total_bucket_ids: inverted.num_bucket_entries(),
            has_overlapping_shards: true,
            shard_format: ShardFormat::Legacy,
            shards: vec![shard_info],
            bucket_names: std::collections::HashMap::new(),
            bucket_sources: std::collections::HashMap::new(),
            bucket_minimizer_counts: std::collections::HashMap::new(),
        };
        let manifest_path = ShardManifest::manifest_path(&base_path);
        manifest.save(&manifest_path)?;

        let sharded = ShardedInvertedIndex::open(&base_path)?;

        let query1: &[u8] = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let query2: &[u8] = b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA";
        let records: Vec<QueryRecord> = vec![(0, query1, None), (1, query2, None)];

        let threshold = 0.1;

        let results_regular = classify_batch(&index, None, &records, threshold);
        let results_sequential =
            classify_batch_sharded_sequential(&sharded, None, &records, threshold, None)?;

        assert_eq!(
            results_regular.len(),
            results_sequential.len(),
            "Result counts should match: regular={}, sequential={}",
            results_regular.len(),
            results_sequential.len()
        );

        let mut sorted_regular = results_regular.clone();
        sorted_regular.sort_by(|a, b| (a.query_id, a.bucket_id).cmp(&(b.query_id, b.bucket_id)));

        let mut sorted_sequential = results_sequential.clone();
        sorted_sequential.sort_by(|a, b| (a.query_id, a.bucket_id).cmp(&(b.query_id, b.bucket_id)));

        for (reg, seq) in sorted_regular.iter().zip(sorted_sequential.iter()) {
            assert_eq!(reg.query_id, seq.query_id, "Query IDs should match");
            assert_eq!(reg.bucket_id, seq.bucket_id, "Bucket IDs should match");
            assert!(
                (reg.score - seq.score).abs() < 0.001,
                "Scores should match: {} vs {}",
                reg.score,
                seq.score
            );
        }

        Ok(())
    }

    #[test]
    fn test_classify_batch_sharded_merge_join_matches_sequential() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let base_path = dir.path().join("test.ryxdi");

        let mut index = Index::new(32, 10, 0x1234).unwrap();
        let mut ws = MinimizerWorkspace::new();

        let seq1 = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let seq2 = b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA";

        index.add_record(1, "ref1", seq1, &mut ws);
        index.add_record(2, "ref2", seq2, &mut ws);
        index.bucket_names.insert(1, "A".into());
        index.bucket_names.insert(2, "B".into());
        index.finalize_bucket(1);
        index.finalize_bucket(2);

        let inverted = InvertedIndex::build_from_index(&index);

        // Save as single shard
        let shard_path = ShardManifest::shard_path(&base_path, 0);
        let shard_info = inverted.save_shard(&shard_path, 0, 0, inverted.num_minimizers(), true)?;

        let manifest = ShardManifest {
            k: inverted.k,
            w: inverted.w,
            salt: inverted.salt,
            source_hash: 0,
            total_minimizers: inverted.num_minimizers(),
            total_bucket_ids: inverted.num_bucket_entries(),
            has_overlapping_shards: true,
            shard_format: ShardFormat::Legacy,
            shards: vec![shard_info],
            bucket_names: std::collections::HashMap::new(),
            bucket_sources: std::collections::HashMap::new(),
            bucket_minimizer_counts: std::collections::HashMap::new(),
        };
        let manifest_path = ShardManifest::manifest_path(&base_path);
        manifest.save(&manifest_path)?;

        let sharded = ShardedInvertedIndex::open(&base_path)?;

        let query1: &[u8] = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let query2: &[u8] = b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA";
        let records: Vec<QueryRecord> = vec![(0, query1, None), (1, query2, None)];

        let threshold = 0.1;

        let results_sequential =
            classify_batch_sharded_sequential(&sharded, None, &records, threshold, None)?;
        let results_merge_join =
            classify_batch_sharded_merge_join(&sharded, None, &records, threshold, None)?;

        assert_eq!(
            results_sequential.len(),
            results_merge_join.len(),
            "Result counts should match: sequential={}, merge_join={}",
            results_sequential.len(),
            results_merge_join.len()
        );

        let mut sorted_sequential = results_sequential.clone();
        sorted_sequential.sort_by(|a, b| (a.query_id, a.bucket_id).cmp(&(b.query_id, b.bucket_id)));

        let mut sorted_merge_join = results_merge_join.clone();
        sorted_merge_join.sort_by(|a, b| (a.query_id, a.bucket_id).cmp(&(b.query_id, b.bucket_id)));

        for (seq, mj) in sorted_sequential.iter().zip(sorted_merge_join.iter()) {
            assert_eq!(seq.query_id, mj.query_id, "Query IDs should match");
            assert_eq!(seq.bucket_id, mj.bucket_id, "Bucket IDs should match");
            assert!(
                (seq.score - mj.score).abs() < 0.001,
                "Scores should match: {} vs {}",
                seq.score,
                mj.score
            );
        }

        Ok(())
    }

    #[test]
    fn test_classify_batch_sharded_main_matches_regular() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let base_path = dir.path().join("test.ryidx");

        let mut index = Index::new(32, 10, 0x1234).unwrap();
        let mut ws = MinimizerWorkspace::new();

        let seq1 = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let seq2 = b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA";

        index.add_record(1, "ref1", seq1, &mut ws);
        index.add_record(2, "ref2", seq2, &mut ws);
        index.bucket_names.insert(1, "A".into());
        index.bucket_names.insert(2, "B".into());
        index.finalize_bucket(1);
        index.finalize_bucket(2);

        // Save as sharded (small budget to ensure multiple shards)
        index.save_sharded(&base_path, 100)?;
        let sharded = ShardedMainIndex::open(&base_path)?;

        let query1: &[u8] = seq1;
        let query2: &[u8] = seq2;
        let records: Vec<QueryRecord> = vec![(0, query1, None), (1, query2, None)];

        let threshold = 0.1;

        let results_regular = classify_batch(&index, None, &records, threshold);
        let results_sharded = classify_batch_sharded_main(&sharded, None, &records, threshold)?;

        assert_eq!(
            results_regular.len(),
            results_sharded.len(),
            "Result counts should match: regular={}, sharded={}",
            results_regular.len(),
            results_sharded.len()
        );

        let mut sorted_regular = results_regular.clone();
        sorted_regular.sort_by(|a, b| (a.query_id, a.bucket_id).cmp(&(b.query_id, b.bucket_id)));

        let mut sorted_sharded = results_sharded.clone();
        sorted_sharded.sort_by(|a, b| (a.query_id, a.bucket_id).cmp(&(b.query_id, b.bucket_id)));

        for (reg, shr) in sorted_regular.iter().zip(sorted_sharded.iter()) {
            assert_eq!(reg.query_id, shr.query_id, "Query IDs should match");
            assert_eq!(reg.bucket_id, shr.bucket_id, "Bucket IDs should match");
            assert!(
                (reg.score - shr.score).abs() < 0.001,
                "Scores should match: {} vs {}",
                reg.score,
                shr.score
            );
        }

        Ok(())
    }

    #[test]
    fn test_classify_batch_sharded_sequential_negative_filtering() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let base_path = dir.path().join("test.ryxdi");

        let mut index = Index::new(32, 10, 0x1234).unwrap();
        let mut ws = MinimizerWorkspace::new();

        let seq1 = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        index.add_record(1, "ref1", seq1, &mut ws);
        index.bucket_names.insert(1, "A".into());
        index.finalize_bucket(1);

        let inverted = InvertedIndex::build_from_index(&index);

        // Save as single shard
        let shard_path = ShardManifest::shard_path(&base_path, 0);
        let shard_info = inverted.save_shard(&shard_path, 0, 0, inverted.num_minimizers(), true)?;

        let manifest = ShardManifest {
            k: inverted.k,
            w: inverted.w,
            salt: inverted.salt,
            source_hash: 0,
            total_minimizers: inverted.num_minimizers(),
            total_bucket_ids: inverted.num_bucket_entries(),
            has_overlapping_shards: true,
            shard_format: ShardFormat::Legacy,
            shards: vec![shard_info],
            bucket_names: std::collections::HashMap::new(),
            bucket_sources: std::collections::HashMap::new(),
            bucket_minimizer_counts: std::collections::HashMap::new(),
        };
        let manifest_path = ShardManifest::manifest_path(&base_path);
        manifest.save(&manifest_path)?;

        let sharded = ShardedInvertedIndex::open(&base_path)?;

        let query: &[u8] = seq1;
        let records: Vec<QueryRecord> = vec![(0, query, None)];

        // Without negative filtering
        let results_no_neg =
            classify_batch_sharded_sequential(&sharded, None, &records, 0.5, None)?;
        assert!(!results_no_neg.is_empty());

        // Extract query minimizers for filtering
        let (query_mins, _) = crate::core::extraction::get_paired_minimizers_into(
            query,
            None,
            sharded.manifest().k,
            sharded.manifest().w,
            sharded.manifest().salt,
            &mut ws,
        );
        let full_neg: HashSet<u64> = query_mins.into_iter().collect();

        // With full negative filtering: no hits above threshold
        let results_full_neg =
            classify_batch_sharded_sequential(&sharded, Some(&full_neg), &records, 0.5, None)?;
        assert!(results_full_neg.is_empty());

        Ok(())
    }

    #[test]
    fn test_classify_batch_sharded_merge_join_negative_filtering() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let base_path = dir.path().join("test.ryxdi");

        let mut index = Index::new(32, 10, 0x1234).unwrap();
        let mut ws = MinimizerWorkspace::new();

        let seq1 = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        index.add_record(1, "ref1", seq1, &mut ws);
        index.bucket_names.insert(1, "A".into());
        index.finalize_bucket(1);

        let inverted = InvertedIndex::build_from_index(&index);

        // Save as single shard
        let shard_path = ShardManifest::shard_path(&base_path, 0);
        let shard_info = inverted.save_shard(&shard_path, 0, 0, inverted.num_minimizers(), true)?;

        let manifest = ShardManifest {
            k: inverted.k,
            w: inverted.w,
            salt: inverted.salt,
            source_hash: 0,
            total_minimizers: inverted.num_minimizers(),
            total_bucket_ids: inverted.num_bucket_entries(),
            has_overlapping_shards: true,
            shard_format: ShardFormat::Legacy,
            shards: vec![shard_info],
            bucket_names: std::collections::HashMap::new(),
            bucket_sources: std::collections::HashMap::new(),
            bucket_minimizer_counts: std::collections::HashMap::new(),
        };
        let manifest_path = ShardManifest::manifest_path(&base_path);
        manifest.save(&manifest_path)?;

        let sharded = ShardedInvertedIndex::open(&base_path)?;

        let query: &[u8] = seq1;
        let records: Vec<QueryRecord> = vec![(0, query, None)];

        // Without negative filtering
        let results_no_neg =
            classify_batch_sharded_merge_join(&sharded, None, &records, 0.5, None)?;
        assert!(!results_no_neg.is_empty());

        // Extract query minimizers for filtering
        let (query_mins, _) = crate::core::extraction::get_paired_minimizers_into(
            query,
            None,
            sharded.manifest().k,
            sharded.manifest().w,
            sharded.manifest().salt,
            &mut ws,
        );
        let full_neg: HashSet<u64> = query_mins.into_iter().collect();

        // With full negative filtering: no hits above threshold
        let results_full_neg =
            classify_batch_sharded_merge_join(&sharded, Some(&full_neg), &records, 0.5, None)?;
        assert!(results_full_neg.is_empty());

        Ok(())
    }

    #[test]
    fn test_classify_batch_sharded_main_negative_filtering() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let base_path = dir.path().join("test.ryidx");

        let mut index = Index::new(32, 10, 0x1234).unwrap();
        let mut ws = MinimizerWorkspace::new();

        let seq1 = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        index.add_record(1, "ref1", seq1, &mut ws);
        index.bucket_names.insert(1, "A".into());
        index.finalize_bucket(1);

        // Save as sharded (small budget to ensure multiple shards)
        index.save_sharded(&base_path, 100)?;
        let sharded = ShardedMainIndex::open(&base_path)?;

        let query: &[u8] = seq1;
        let records: Vec<QueryRecord> = vec![(0, query, None)];

        // Without negative filtering
        let results_no_neg = classify_batch_sharded_main(&sharded, None, &records, 0.5)?;
        assert!(!results_no_neg.is_empty());

        // Extract query minimizers for filtering
        let (query_mins, _) = crate::core::extraction::get_paired_minimizers_into(
            query,
            None,
            sharded.manifest().k,
            sharded.manifest().w,
            sharded.manifest().salt,
            &mut ws,
        );
        let full_neg: HashSet<u64> = query_mins.into_iter().collect();

        // With full negative filtering: no hits above threshold
        let results_full_neg =
            classify_batch_sharded_main(&sharded, Some(&full_neg), &records, 0.5)?;
        assert!(results_full_neg.is_empty());

        Ok(())
    }

    #[test]
    fn test_classify_batch_sharded_sequential_parquet() -> anyhow::Result<()> {
        use std::fs;

        let dir = tempfile::tempdir()?;
        let base_path = dir.path().join("test.ryxdi");

        // Create inverted directory for Parquet shards
        let inverted_dir = base_path.join("inverted");
        fs::create_dir_all(&inverted_dir)?;

        let mut index = Index::new(32, 10, 0x1234).unwrap();
        let mut ws = MinimizerWorkspace::new();

        let seq1 = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let seq2 = b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA";

        index.add_record(1, "ref1", seq1, &mut ws);
        index.add_record(2, "ref2", seq2, &mut ws);
        index.bucket_names.insert(1, "A".into());
        index.bucket_names.insert(2, "B".into());
        index.finalize_bucket(1);
        index.finalize_bucket(2);

        let inverted = InvertedIndex::build_from_index(&index);

        // Save as Parquet shard
        let shard_path = ShardManifest::shard_path_parquet(&base_path, 0);
        let shard_info = inverted.save_shard_parquet(&shard_path, 0, None)?;

        let manifest = ShardManifest {
            k: inverted.k,
            w: inverted.w,
            salt: inverted.salt,
            source_hash: 0,
            total_minimizers: inverted.num_minimizers(),
            total_bucket_ids: inverted.num_bucket_entries(),
            has_overlapping_shards: true,
            shard_format: ShardFormat::Parquet,
            shards: vec![shard_info],
            bucket_names: std::collections::HashMap::new(),
            bucket_sources: std::collections::HashMap::new(),
            bucket_minimizer_counts: std::collections::HashMap::new(),
        };
        let manifest_path = ShardManifest::manifest_path(&base_path);
        manifest.save(&manifest_path)?;

        // Open with auto-detection (should detect Parquet from manifest)
        let sharded = ShardedInvertedIndex::open(&base_path)?;
        assert_eq!(sharded.shard_format(), ShardFormat::Parquet);

        let query1: &[u8] = seq1;
        let query2: &[u8] = seq2;
        let records: Vec<QueryRecord> = vec![(0, query1, None), (1, query2, None)];

        let threshold = 0.1;

        // Classify using Parquet shards
        let results_parquet =
            classify_batch_sharded_sequential(&sharded, None, &records, threshold, None)?;

        // Should have results matching both sequences
        assert!(!results_parquet.is_empty());

        // Verify we get hits for both queries
        let query0_hits: Vec<_> = results_parquet.iter().filter(|r| r.query_id == 0).collect();
        let query1_hits: Vec<_> = results_parquet.iter().filter(|r| r.query_id == 1).collect();

        assert!(!query0_hits.is_empty(), "Query 0 should have hits");
        assert!(!query1_hits.is_empty(), "Query 1 should have hits");

        // Query 0 should match bucket 1 with high score
        let q0_b1 = query0_hits.iter().find(|r| r.bucket_id == 1);
        assert!(q0_b1.is_some(), "Query 0 should match bucket 1");
        assert!(
            q0_b1.unwrap().score > 0.9,
            "Query 0 should have high score for bucket 1"
        );

        Ok(())
    }

    #[test]
    fn test_classify_batch_sharded_merge_join_parquet() -> anyhow::Result<()> {
        use std::fs;

        let dir = tempfile::tempdir()?;
        let base_path = dir.path().join("test.ryxdi");

        // Create inverted directory for Parquet shards
        let inverted_dir = base_path.join("inverted");
        fs::create_dir_all(&inverted_dir)?;

        let mut index = Index::new(32, 10, 0x1234).unwrap();
        let mut ws = MinimizerWorkspace::new();

        let seq1 = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let seq2 = b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA";

        index.add_record(1, "ref1", seq1, &mut ws);
        index.add_record(2, "ref2", seq2, &mut ws);
        index.bucket_names.insert(1, "A".into());
        index.bucket_names.insert(2, "B".into());
        index.finalize_bucket(1);
        index.finalize_bucket(2);

        let inverted = InvertedIndex::build_from_index(&index);

        // Save as Parquet shard
        let shard_path = ShardManifest::shard_path_parquet(&base_path, 0);
        let shard_info = inverted.save_shard_parquet(&shard_path, 0, None)?;

        let manifest = ShardManifest {
            k: inverted.k,
            w: inverted.w,
            salt: inverted.salt,
            source_hash: 0,
            total_minimizers: inverted.num_minimizers(),
            total_bucket_ids: inverted.num_bucket_entries(),
            has_overlapping_shards: true,
            shard_format: ShardFormat::Legacy,
            shards: vec![shard_info],
            bucket_names: std::collections::HashMap::new(),
            bucket_sources: std::collections::HashMap::new(),
            bucket_minimizer_counts: std::collections::HashMap::new(),
        };
        let manifest_path = ShardManifest::manifest_path(&base_path);
        manifest.save(&manifest_path)?;

        let sharded = ShardedInvertedIndex::open(&base_path)?;
        assert_eq!(sharded.shard_format(), ShardFormat::Parquet);

        let query1: &[u8] = seq1;
        let query2: &[u8] = seq2;
        let records: Vec<QueryRecord> = vec![(0, query1, None), (1, query2, None)];

        let threshold = 0.1;

        // Classify using merge-join with Parquet shards
        let results_parquet =
            classify_batch_sharded_merge_join(&sharded, None, &records, threshold, None)?;

        assert!(!results_parquet.is_empty());

        // Verify we get hits for both queries
        let query0_hits: Vec<_> = results_parquet.iter().filter(|r| r.query_id == 0).collect();
        let query1_hits: Vec<_> = results_parquet.iter().filter(|r| r.query_id == 1).collect();

        assert!(!query0_hits.is_empty(), "Query 0 should have hits");
        assert!(!query1_hits.is_empty(), "Query 1 should have hits");

        Ok(())
    }

    #[test]
    fn test_classify_parquet_matches_legacy() -> anyhow::Result<()> {
        use std::fs;

        let dir = tempfile::tempdir()?;
        let legacy_base = dir.path().join("legacy.ryxdi");
        let parquet_base = dir.path().join("parquet.ryxdi");

        // Create inverted directory for Parquet shards
        let inverted_dir = parquet_base.join("inverted");
        fs::create_dir_all(&inverted_dir)?;

        let mut index = Index::new(32, 10, 0x1234).unwrap();
        let mut ws = MinimizerWorkspace::new();

        let seq1 = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let seq2 = b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA";

        index.add_record(1, "ref1", seq1, &mut ws);
        index.add_record(2, "ref2", seq2, &mut ws);
        index.bucket_names.insert(1, "A".into());
        index.bucket_names.insert(2, "B".into());
        index.finalize_bucket(1);
        index.finalize_bucket(2);

        let inverted = InvertedIndex::build_from_index(&index);

        // Save as legacy shard
        let legacy_shard_path = ShardManifest::shard_path(&legacy_base, 0);
        let legacy_shard_info =
            inverted.save_shard(&legacy_shard_path, 0, 0, inverted.num_minimizers(), true)?;

        let legacy_manifest = ShardManifest {
            k: inverted.k,
            w: inverted.w,
            salt: inverted.salt,
            source_hash: 0,
            total_minimizers: inverted.num_minimizers(),
            total_bucket_ids: inverted.num_bucket_entries(),
            has_overlapping_shards: true,
            shard_format: ShardFormat::Legacy,
            shards: vec![legacy_shard_info],
            bucket_names: std::collections::HashMap::new(),
            bucket_sources: std::collections::HashMap::new(),
            bucket_minimizer_counts: std::collections::HashMap::new(),
        };
        legacy_manifest.save(&ShardManifest::manifest_path(&legacy_base))?;

        // Save as Parquet shard
        let parquet_shard_path = ShardManifest::shard_path_parquet(&parquet_base, 0);
        let parquet_shard_info = inverted.save_shard_parquet(&parquet_shard_path, 0, None)?;

        let parquet_manifest = ShardManifest {
            k: inverted.k,
            w: inverted.w,
            salt: inverted.salt,
            source_hash: 0,
            total_minimizers: inverted.num_minimizers(),
            total_bucket_ids: inverted.num_bucket_entries(),
            has_overlapping_shards: true,
            shard_format: ShardFormat::Parquet,
            shards: vec![parquet_shard_info],
            bucket_names: std::collections::HashMap::new(),
            bucket_sources: std::collections::HashMap::new(),
            bucket_minimizer_counts: std::collections::HashMap::new(),
        };
        parquet_manifest.save(&ShardManifest::manifest_path(&parquet_base))?;

        let legacy_sharded = ShardedInvertedIndex::open(&legacy_base)?;
        let parquet_sharded = ShardedInvertedIndex::open(&parquet_base)?;

        let query1: &[u8] = seq1;
        let query2: &[u8] = seq2;
        let records: Vec<QueryRecord> = vec![(0, query1, None), (1, query2, None)];

        let threshold = 0.1;

        // Classify with both formats
        let results_legacy =
            classify_batch_sharded_sequential(&legacy_sharded, None, &records, threshold, None)?;
        let results_parquet =
            classify_batch_sharded_sequential(&parquet_sharded, None, &records, threshold, None)?;

        // Results should match
        assert_eq!(
            results_legacy.len(),
            results_parquet.len(),
            "Result counts should match: legacy={}, parquet={}",
            results_legacy.len(),
            results_parquet.len()
        );

        let mut sorted_legacy = results_legacy.clone();
        sorted_legacy.sort_by(|a, b| (a.query_id, a.bucket_id).cmp(&(b.query_id, b.bucket_id)));

        let mut sorted_parquet = results_parquet.clone();
        sorted_parquet.sort_by(|a, b| (a.query_id, a.bucket_id).cmp(&(b.query_id, b.bucket_id)));

        for (leg, pq) in sorted_legacy.iter().zip(sorted_parquet.iter()) {
            assert_eq!(leg.query_id, pq.query_id, "Query IDs should match");
            assert_eq!(leg.bucket_id, pq.bucket_id, "Bucket IDs should match");
            assert!(
                (leg.score - pq.score).abs() < 0.001,
                "Scores should match: {} vs {}",
                leg.score,
                pq.score
            );
        }

        Ok(())
    }
}
