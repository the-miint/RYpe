//! Classification functions for matching query sequences against indexed references.
//!
//! Provides multiple classification strategies:
//! - `classify_batch`: Direct classification against an Index
//! - `classify_batch_inverted`: Classification using an InvertedIndex
//! - `classify_batch_sharded_sequential`: Classification using a ShardedInvertedIndex
//! - `aggregate_batch`: Aggregated classification for paired-end reads

use anyhow::Result;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use crate::extraction::{count_hits, get_paired_minimizers_into};
use crate::index::Index;
use crate::inverted::InvertedIndex;
use crate::sharded::{ShardManifest, ShardedInvertedIndex};
use crate::types::{HitResult, QueryRecord};
use crate::workspace::MinimizerWorkspace;

/// Classify a batch of query records against an Index.
///
/// Uses parallel minimizer extraction and per-bucket binary search.
///
/// # Arguments
/// * `engine` - The index to classify against
/// * `records` - Batch of query records
/// * `threshold` - Minimum score threshold for reporting hits
///
/// # Returns
/// Vector of HitResult for all (query, bucket) pairs meeting the threshold.
pub fn classify_batch(
    engine: &Index,
    records: &[QueryRecord],
    threshold: f64
) -> Vec<HitResult> {

    let processed: Vec<_> = records.par_iter()
        .map_init(MinimizerWorkspace::new, |ws, (id, s1, s2)| {
            let (ha, hb) = get_paired_minimizers_into(s1, *s2, engine.k, engine.w, engine.salt, ws);
            (*id, ha, hb)
        }).collect();

    let mut map_a: HashMap<u64, Vec<usize>> = HashMap::new();
    let mut map_b: HashMap<u64, Vec<usize>> = HashMap::new();
    let mut uniq_mins = HashSet::new();

    for (idx, (_, ma, mb)) in processed.iter().enumerate() {
        for &m in ma { map_a.entry(m).or_default().push(idx); uniq_mins.insert(m); }
        for &m in mb { map_b.entry(m).or_default().push(idx); uniq_mins.insert(m); }
    }
    let uniq_vec: Vec<u64> = uniq_mins.into_iter().collect();

    let results: Vec<HitResult> = engine.buckets.par_iter().map(|(b_id, bucket)| {
        let mut hits = HashMap::new();

        for &m in &uniq_vec {
            if bucket.binary_search(&m).is_ok() {
                if let Some(rs) = map_a.get(&m) { for &r in rs { hits.entry(r).or_insert((0,0)).0 += 1; } }
                if let Some(rs) = map_b.get(&m) { for &r in rs { hits.entry(r).or_insert((0,0)).1 += 1; } }
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
                bucket_results.push(HitResult { query_id: *qid, bucket_id: *b_id, score });
            }
        }
        bucket_results
    }).flatten().collect();

    results
}

/// Classify a batch of query records using an inverted index.
///
/// Uses O(Q * log(U/Q)) lookups per query instead of O(B * Q * log(M)).
///
/// # Arguments
/// * `inverted` - The inverted index for minimizer → bucket lookups
/// * `records` - Batch of query records to classify
/// * `threshold` - Minimum score threshold for reporting hits
///
/// # Returns
/// Vector of HitResult for all (query, bucket) pairs meeting the threshold.
pub fn classify_batch_inverted(
    inverted: &InvertedIndex,
    records: &[QueryRecord],
    threshold: f64
) -> Vec<HitResult> {
    let processed: Vec<_> = records.par_iter()
        .map_init(MinimizerWorkspace::new, |ws, (id, s1, s2)| {
            let (ha, hb) = get_paired_minimizers_into(s1, *s2, inverted.k, inverted.w, inverted.salt, ws);
            (*id, ha, hb)
        }).collect();

    let results: Vec<HitResult> = processed.par_iter()
        .flat_map(|(query_id, fwd_mins, rc_mins)| {
            let fwd_hits = inverted.get_bucket_hits(fwd_mins);
            let fwd_len = fwd_mins.len() as f64;

            let rc_hits = inverted.get_bucket_hits(rc_mins);
            let rc_len = rc_mins.len() as f64;

            let capacity = fwd_hits.len() + rc_hits.len();
            let mut scores: HashMap<u32, (usize, usize)> = HashMap::with_capacity(capacity);
            for (&bucket_id, &count) in &fwd_hits {
                scores.entry(bucket_id).or_insert((0, 0)).0 = count;
            }
            for (&bucket_id, &count) in &rc_hits {
                scores.entry(bucket_id).or_insert((0, 0)).1 = count;
            }

            let mut query_results = Vec::with_capacity(scores.len());
            for (bucket_id, (fwd_count, rc_count)) in scores {
                let score = (if fwd_len > 0.0 { fwd_count as f64 / fwd_len } else { 0.0 })
                    .max(if rc_len > 0.0 { rc_count as f64 / rc_len } else { 0.0 });

                if score >= threshold {
                    query_results.push(HitResult {
                        query_id: *query_id,
                        bucket_id,
                        score,
                    });
                }
            }
            query_results
        })
        .collect();

    results
}

/// Classify a batch of records against a sharded inverted index.
///
/// Loads one shard at a time to minimize memory usage. Memory complexity
/// is O(batch_size * minimizers_per_read) + O(single_shard_size).
///
/// # Arguments
/// * `sharded` - The sharded inverted index
/// * `records` - Batch of query records
/// * `threshold` - Minimum score threshold
///
/// # Returns
/// Vector of HitResult for all (query, bucket) pairs meeting the threshold.
pub fn classify_batch_sharded_sequential(
    sharded: &ShardedInvertedIndex,
    records: &[QueryRecord],
    threshold: f64
) -> Result<Vec<HitResult>> {
    let manifest = sharded.manifest();
    let base_path = sharded.base_path();

    let processed: Vec<_> = records.par_iter()
        .map_init(MinimizerWorkspace::new, |ws, (id, s1, s2)| {
            let (ha, hb) = get_paired_minimizers_into(s1, *s2, manifest.k, manifest.w, manifest.salt, ws);
            (*id, ha, hb)
        }).collect();

    let all_hits: Vec<Mutex<HashMap<u32, (usize, usize)>>> =
        (0..processed.len()).map(|_| Mutex::new(HashMap::new())).collect();

    for shard_info in &manifest.shards {
        let shard_path = ShardManifest::shard_path(base_path, shard_info.shard_id);
        let shard = InvertedIndex::load_shard(&shard_path)?;

        processed.par_iter().enumerate().for_each(|(idx, (_, fwd_mins, rc_mins))| {
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
        // shard dropped here, freeing memory before loading next
    }

    let results: Vec<HitResult> = processed.iter().enumerate()
        .flat_map(|(idx, (query_id, fwd_mins, rc_mins))| {
            let fwd_len = fwd_mins.len() as f64;
            let rc_len = rc_mins.len() as f64;
            let hits = all_hits[idx].lock().unwrap();

            hits.iter().filter_map(|(&bucket_id, &(fwd_count, rc_count))| {
                let score = (if fwd_len > 0.0 { fwd_count as f64 / fwd_len } else { 0.0 })
                    .max(if rc_len > 0.0 { rc_count as f64 / rc_len } else { 0.0 });

                if score >= threshold {
                    Some(HitResult { query_id: *query_id, bucket_id, score })
                } else {
                    None
                }
            }).collect::<Vec<_>>()
        })
        .collect();

    Ok(results)
}

/// Aggregate classification for paired-end reads.
///
/// Combines all minimizers from all records into a single query and
/// classifies against all buckets. Returns results with query_id = -1.
///
/// # Arguments
/// * `engine` - The index to classify against
/// * `records` - Batch of query records
/// * `threshold` - Minimum score threshold
///
/// # Returns
/// Vector of HitResult with query_id = -1 for all buckets meeting the threshold.
pub fn aggregate_batch(
    engine: &Index,
    records: &[QueryRecord],
    threshold: f64
) -> Vec<HitResult> {
    let mut global = HashSet::new();

    let batch_mins: Vec<Vec<u64>> = records.par_iter()
        .map_init(MinimizerWorkspace::new, |ws, (_, s1, s2)| {
            let (mut a, b) = get_paired_minimizers_into(s1, *s2, engine.k, engine.w, engine.salt, ws);
            a.extend(b);
            a
        }).collect();

    for v in batch_mins { for m in v { global.insert(m); } }

    let total = global.len() as f64;
    if total == 0.0 { return Vec::new(); }

    let g_vec: Vec<u64> = global.into_iter().collect();

    engine.buckets.par_iter().filter_map(|(id, b)| {
        let s = count_hits(&g_vec, b) / total;
        if s >= threshold {
            Some(HitResult { query_id: -1, bucket_id: *id, score: s })
        } else {
            None
        }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::MinimizerWorkspace;

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
        let records: Vec<QueryRecord> = vec![
            (101, &query_seq, None)
        ];

        let results = classify_batch(&index, &records, 0.5);

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

        let seq_at: Vec<u8> = (0..200).map(|i| if i % 2 == 0 { b'A' } else { b'T' }).collect();
        index.add_record(2, "ref_at", &seq_at, &mut ws);
        index.finalize_bucket(2);

        let q1 = &seq_at[0..100];
        let q2 = &seq_at[100..200];

        let records: Vec<QueryRecord> = vec![
            (1, q1, None),
            (2, q2, None),
        ];

        let results = aggregate_batch(&index, &records, 0.5);

        assert_eq!(results.len(), 1, "Should only match bucket 2");
        assert_eq!(results[0].bucket_id, 2);
        assert!(results[0].score > 0.9);
    }

    #[test]
    fn test_classify_batch_inverted_matches_regular() -> anyhow::Result<()> {
        let mut index = Index::new(64, 10, 0).unwrap();
        let mut ws = MinimizerWorkspace::new();

        let seq_a = vec![b'A'; 80];
        index.add_record(1, "ref_a", &seq_a, &mut ws);
        index.finalize_bucket(1);
        index.bucket_names.insert(1, "BucketA".into());

        let seq_b: Vec<u8> = (0..80).map(|i| if i % 2 == 0 { b'A' } else { b'T' }).collect();
        index.add_record(2, "ref_b", &seq_b, &mut ws);
        index.finalize_bucket(2);
        index.bucket_names.insert(2, "BucketB".into());

        let inverted = InvertedIndex::build_from_index(&index);

        let query_seq = seq_a.clone();
        let records: Vec<QueryRecord> = vec![(101, &query_seq[..], None)];

        let results_regular = classify_batch(&index, &records, 0.5);
        let results_inverted = classify_batch_inverted(&inverted, &records, 0.5);

        assert_eq!(results_regular.len(), results_inverted.len());
        if !results_regular.is_empty() {
            assert_eq!(results_regular[0].bucket_id, results_inverted[0].bucket_id);
            assert!((results_regular[0].score - results_inverted[0].score).abs() < 0.001);
        }

        Ok(())
    }

    #[test]
    fn test_classify_batch_sharded_sequential_matches_inverted() -> anyhow::Result<()> {
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
        assert!(inverted.num_minimizers() > 0, "Inverted index should not be empty");

        inverted.save_sharded(&base_path, 4)?;
        let sharded = ShardedInvertedIndex::open(&base_path)?;

        let query1: &[u8] = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let query2: &[u8] = b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA";
        let records: Vec<QueryRecord> = vec![
            (0, query1, None),
            (1, query2, None),
        ];

        let threshold = 0.1;

        let results_inverted = classify_batch_inverted(&inverted, &records, threshold);
        let results_sequential = classify_batch_sharded_sequential(&sharded, &records, threshold)?;

        assert_eq!(results_inverted.len(), results_sequential.len(),
            "Result counts should match: inverted={}, sequential={}",
            results_inverted.len(), results_sequential.len());

        let mut sorted_inverted = results_inverted.clone();
        sorted_inverted.sort_by(|a, b| (a.query_id, a.bucket_id).cmp(&(b.query_id, b.bucket_id)));

        let mut sorted_sequential = results_sequential.clone();
        sorted_sequential.sort_by(|a, b| (a.query_id, a.bucket_id).cmp(&(b.query_id, b.bucket_id)));

        for (inv, seq) in sorted_inverted.iter().zip(sorted_sequential.iter()) {
            assert_eq!(inv.query_id, seq.query_id, "Query IDs should match");
            assert_eq!(inv.bucket_id, seq.bucket_id, "Bucket IDs should match");
            assert!((inv.score - seq.score).abs() < 0.001,
                "Scores should match: {} vs {}", inv.score, seq.score);
        }

        Ok(())
    }
}
