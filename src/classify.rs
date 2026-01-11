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
use crate::inverted::{InvertedIndex, QueryInvertedIndex};
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
            let rc_hits = inverted.get_bucket_hits(rc_mins);

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
                let score = compute_score(fwd_count, fwd_mins.len(), rc_count, rc_mins.len());
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
            let hits = all_hits[idx].lock().unwrap();

            hits.iter().filter_map(|(&bucket_id, &(fwd_count, rc_count))| {
                let score = compute_score(fwd_count, fwd_mins.len(), rc_count, rc_mins.len());
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

/// Threshold for switching from merge-join to galloping search.
///
/// When one index is more than GALLOP_THRESHOLD times larger than the other,
/// galloping search (exponential probe + binary search) is used instead of
/// pure merge-join. The value 16 is chosen because:
/// - Below 16:1, merge-join's sequential access beats galloping's binary searches
/// - Above 16:1, galloping's O(Q*log(R/Q)) is significantly faster than O(Q+R)
/// - 16 is a power of 2, making the threshold check cheap
///
/// This threshold is empirically reasonable but may benefit from tuning
/// based on actual workload characteristics (cache sizes, hit rates, etc.).
const GALLOP_THRESHOLD: usize = 16;

/// Estimated number of buckets each read will hit (for HashMap pre-allocation).
const ESTIMATED_BUCKETS_PER_READ: usize = 4;

/// Compute the dual-strand classification score.
///
/// Score is the maximum of forward and reverse-complement hit ratios.
#[inline]
fn compute_score(fwd_hits: usize, fwd_total: usize, rc_hits: usize, rc_total: usize) -> f64 {
    let fwd_score = if fwd_total > 0 { fwd_hits as f64 / fwd_total as f64 } else { 0.0 };
    let rc_score = if rc_total > 0 { rc_hits as f64 / rc_total as f64 } else { 0.0 };
    fwd_score.max(rc_score)
}

/// Accumulate hits for a single matching minimizer pair.
#[inline]
fn accumulate_match(
    query_idx: &QueryInvertedIndex,
    ref_idx: &InvertedIndex,
    qi: usize,
    ri: usize,
    accumulators: &mut [HashMap<u32, (u32, u32)>],
) {
    let q_start = query_idx.offsets[qi] as usize;
    let q_end = query_idx.offsets[qi + 1] as usize;
    let r_start = ref_idx.offsets[ri] as usize;
    let r_end = ref_idx.offsets[ri + 1] as usize;

    for &packed in &query_idx.read_ids[q_start..q_end] {
        let (read_idx, is_rc) = QueryInvertedIndex::unpack_read_id(packed);
        for &bucket_id in &ref_idx.bucket_ids[r_start..r_end] {
            let entry = accumulators[read_idx as usize].entry(bucket_id).or_insert((0, 0));
            if is_rc {
                entry.1 += 1;
            } else {
                entry.0 += 1;
            }
        }
    }
}

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
/// **Use `classify_batch_inverted` instead when:**
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
    if num_reads == 0 || query_idx.num_minimizers() == 0 || ref_idx.num_minimizers() == 0 {
        return Vec::new();
    }

    // Per-read accumulator: bucket_id -> (fwd_hits, rc_hits)
    // Pre-allocate with estimated capacity to reduce rehashing
    let mut accumulators: Vec<HashMap<u32, (u32, u32)>> = (0..num_reads)
        .map(|_| HashMap::with_capacity(ESTIMATED_BUCKETS_PER_READ))
        .collect();

    // Choose algorithm based on size ratio
    let q_len = query_idx.minimizers.len();
    let r_len = ref_idx.minimizers.len();

    if q_len * GALLOP_THRESHOLD < r_len {
        // Query much smaller: gallop through reference
        gallop_join(query_idx, ref_idx, &mut accumulators, true);
    } else if r_len * GALLOP_THRESHOLD < q_len {
        // Reference much smaller: gallop through query
        gallop_join(query_idx, ref_idx, &mut accumulators, false);
    } else {
        // Similar sizes: pure merge-join
        merge_join(query_idx, ref_idx, &mut accumulators);
    }

    // Score and filter
    let mut results = Vec::new();
    for (read_idx, buckets) in accumulators.into_iter().enumerate() {
        let fwd_total = query_idx.fwd_counts[read_idx] as usize;
        let rc_total = query_idx.rc_counts[read_idx] as usize;
        let query_id = query_ids[read_idx];

        for (bucket_id, (fwd_hits, rc_hits)) in buckets {
            let score = compute_score(fwd_hits as usize, fwd_total, rc_hits as usize, rc_total);
            if score >= threshold {
                results.push(HitResult { query_id, bucket_id, score });
            }
        }
    }

    results
}

/// Pure merge-join when query and reference have similar sizes.
/// O(Q + R) complexity with excellent cache behavior.
fn merge_join(
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
            accumulate_match(query_idx, ref_idx, qi, ri, accumulators);
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
fn gallop_join(
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

        // Binary search in [larger_pos, larger_pos + jump]
        let search_end = (larger_pos + jump).min(larger.len());
        match larger[larger_pos..search_end].binary_search(&s_min) {
            Ok(rel_idx) => {
                let larger_idx = larger_pos + rel_idx;
                let (qi, ri) = if query_smaller {
                    (smaller_idx, larger_idx)
                } else {
                    (larger_idx, smaller_idx)
                };
                accumulate_match(query_idx, ref_idx, qi, ri, accumulators);
                larger_pos = larger_idx + 1;
            }
            Err(rel_idx) => {
                larger_pos += rel_idx;
            }
        }
    }
}

/// Classify using merge-join with automatic query index construction.
///
/// This is a convenience wrapper that:
/// 1. Extracts minimizers from all query records (parallel)
/// 2. Builds a `QueryInvertedIndex` from the extracted minimizers
/// 3. Performs merge-join classification against the reference index
///
/// Use this when you have high minimizer overlap across reads (e.g., amplicon
/// sequencing, reads from similar regions). For diverse reads or small batches,
/// `classify_batch_inverted` may be more efficient.
///
/// See `classify_batch_merge_join` for detailed performance characteristics.
pub fn classify_batch_with_query_index(
    ref_idx: &InvertedIndex,
    records: &[QueryRecord],
    threshold: f64,
) -> Vec<HitResult> {
    if records.is_empty() {
        return Vec::new();
    }

    // Extract minimizers in parallel
    let extracted: Vec<_> = records.par_iter()
        .map_init(MinimizerWorkspace::new, |ws, (_, s1, s2)| {
            get_paired_minimizers_into(s1, *s2, ref_idx.k, ref_idx.w, ref_idx.salt, ws)
        }).collect();

    // Collect query IDs
    let query_ids: Vec<i64> = records.iter().map(|(id, _, _)| *id).collect();

    // Build query inverted index
    let query_idx = QueryInvertedIndex::build(&extracted);

    // Classify using merge-join
    classify_batch_merge_join(&query_idx, ref_idx, &query_ids, threshold)
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

    // ==================== Merge-Join Classification Tests ====================

    #[test]
    fn test_merge_join_basic() {
        // Simple test with known minimizers
        let queries = vec![
            (vec![100, 200, 300], vec![150, 250]),  // read 0
        ];
        let query_idx = QueryInvertedIndex::build(&queries);

        // Build reference with overlapping minimizers
        let mut index = Index::new(64, 50, 0).unwrap();
        index.buckets.insert(1, vec![100, 200, 400]);  // shares 100, 200
        index.buckets.insert(2, vec![150, 250, 500]);  // shares 150, 250
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
        assert!((bucket1_hit.score - 2.0/3.0).abs() < 0.001);
        // Bucket 2: 2 rc hits (150, 250) out of 2 rc minimizers = 1.0
        assert!((bucket2_hit.score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_merge_join_no_overlap() {
        let queries = vec![
            (vec![100, 200], vec![150]),
        ];
        let query_idx = QueryInvertedIndex::build(&queries);

        let mut index = Index::new(64, 50, 0).unwrap();
        index.buckets.insert(1, vec![500, 600, 700]);  // no overlap
        index.bucket_names.insert(1, "A".into());

        let ref_idx = InvertedIndex::build_from_index(&index);
        let query_ids = vec![101i64];

        let results = classify_batch_merge_join(&query_idx, &ref_idx, &query_ids, 0.0);

        assert!(results.is_empty(), "Should have no hits when no minimizers overlap");
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
    fn test_classify_merge_join_matches_inverted() -> anyhow::Result<()> {
        // Cross-validation: merge-join should produce same results as inverted
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

        let results_inverted = classify_batch_inverted(&inverted, &records, 0.5);
        let results_merge = classify_batch_with_query_index(&inverted, &records, 0.5);

        assert_eq!(results_inverted.len(), results_merge.len(),
            "Result counts should match: inverted={}, merge={}",
            results_inverted.len(), results_merge.len());

        // Sort and compare
        let mut sorted_inverted = results_inverted.clone();
        sorted_inverted.sort_by(|a, b| (a.query_id, a.bucket_id).cmp(&(b.query_id, b.bucket_id)));

        let mut sorted_merge = results_merge.clone();
        sorted_merge.sort_by(|a, b| (a.query_id, a.bucket_id).cmp(&(b.query_id, b.bucket_id)));

        for (inv, mrg) in sorted_inverted.iter().zip(sorted_merge.iter()) {
            assert_eq!(inv.query_id, mrg.query_id, "Query IDs should match");
            assert_eq!(inv.bucket_id, mrg.bucket_id, "Bucket IDs should match");
            assert!((inv.score - mrg.score).abs() < 0.001,
                "Scores should match: {} vs {}", inv.score, mrg.score);
        }

        Ok(())
    }

    #[test]
    fn test_classify_merge_multiple_reads_multiple_buckets() -> anyhow::Result<()> {
        // More comprehensive test with multiple reads and buckets
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

        let query1: &[u8] = seq1;
        let query2: &[u8] = seq2;
        let records: Vec<QueryRecord> = vec![
            (0, query1, None),
            (1, query2, None),
        ];

        let threshold = 0.1;
        let results_inverted = classify_batch_inverted(&inverted, &records, threshold);
        let results_merge = classify_batch_with_query_index(&inverted, &records, threshold);

        assert_eq!(results_inverted.len(), results_merge.len(),
            "Result counts should match");

        // Sort and compare all results
        let mut sorted_inverted = results_inverted.clone();
        sorted_inverted.sort_by(|a, b| (a.query_id, a.bucket_id).cmp(&(b.query_id, b.bucket_id)));

        let mut sorted_merge = results_merge.clone();
        sorted_merge.sort_by(|a, b| (a.query_id, a.bucket_id).cmp(&(b.query_id, b.bucket_id)));

        for (inv, mrg) in sorted_inverted.iter().zip(sorted_merge.iter()) {
            assert_eq!(inv.query_id, mrg.query_id);
            assert_eq!(inv.bucket_id, mrg.bucket_id);
            assert!((inv.score - mrg.score).abs() < 0.001,
                "Scores differ for query {} bucket {}: {} vs {}",
                inv.query_id, inv.bucket_id, inv.score, mrg.score);
        }

        Ok(())
    }

    #[test]
    fn test_galloping_search_small_query() {
        // Create scenario where query << ref to trigger galloping
        // Need query_len * 16 < ref_len
        let queries = vec![
            (vec![500], vec![]),  // Single minimizer
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
        assert_eq!(results[0].score, 1.0);  // 1/1 fwd minimizers matched
    }

    #[test]
    fn test_galloping_search_small_ref() {
        // Create scenario where ref << query to trigger galloping (rare case)
        // Need ref_len * 16 < query_len
        let minimizers: Vec<u64> = (0..100).map(|i| i * 10).collect();
        let queries = vec![
            (minimizers.clone(), vec![]),  // 100 minimizers
        ];
        let query_idx = QueryInvertedIndex::build(&queries);

        // Small reference (< 100/16 = 6 minimizers)
        let mut index = Index::new(64, 50, 0).unwrap();
        index.buckets.insert(1, vec![500]);  // Single minimizer at position 50
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
