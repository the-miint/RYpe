//! Classification functions for matching query sequences against indexed references.
//!
//! Provides multiple classification strategies:
//! - `classify_batch`: Direct classification against an Index
//! - `classify_batch_sharded_sequential`: Classification using a ShardedInvertedIndex
//! - `classify_batch_sharded_merge_join`: Classification using merge-join algorithm
//! - `aggregate_batch`: Aggregated classification for paired-end reads

use anyhow::Result;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::time::Instant;

use crate::extraction::{count_hits, get_paired_minimizers_into};
use crate::index::Index;
use crate::inverted::{InvertedIndex, QueryInvertedIndex};
use crate::sharded::ShardedInvertedIndex;
use crate::sharded_main::ShardedMainIndex;
use crate::types::{HitResult, QueryRecord};
use crate::workspace::MinimizerWorkspace;

/// Controls whether timing diagnostics are printed to stderr
pub static ENABLE_TIMING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Print timing info if enabled
#[inline]
fn log_timing(phase: &str, elapsed_ms: u128) {
    if ENABLE_TIMING.load(std::sync::atomic::Ordering::Relaxed) {
        eprintln!("[TIMING] {}: {}ms", phase, elapsed_ms);
    }
}

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
fn filter_negative_mins(
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
    let processed: Vec<_> = records
        .par_iter()
        .map_init(MinimizerWorkspace::new, |ws, (id, s1, s2)| {
            let (ha, hb) = get_paired_minimizers_into(s1, *s2, engine.k, engine.w, engine.salt, ws);
            let (fa, fb) = filter_negative_mins(ha, hb, negative_mins);
            (*id, fa, fb)
        })
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
///
/// # Returns
/// Vector of HitResult for all (query, bucket) pairs meeting the threshold.
pub fn classify_batch_sharded_sequential(
    sharded: &ShardedInvertedIndex,
    negative_mins: Option<&HashSet<u64>>,
    records: &[QueryRecord],
    threshold: f64,
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
        // Use filtered loading for Parquet shards
        let shard = sharded.load_shard_for_query(shard_info.shard_id, &all_minimizers)?;
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
///
/// # Returns
/// Vector of HitResult for all (query, bucket) pairs meeting the threshold.
pub fn classify_batch_sharded_merge_join(
    sharded: &ShardedInvertedIndex,
    negative_mins: Option<&HashSet<u64>>,
    records: &[QueryRecord],
    threshold: f64,
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
        let shard = sharded.load_shard_for_query(shard_info.shard_id, query_minimizers)?;
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
    negative_mins: Option<&HashSet<u64>>,
    records: &[QueryRecord],
    threshold: f64,
) -> Vec<HitResult> {
    let mut global = HashSet::new();

    let batch_mins: Vec<Vec<u64>> = records
        .par_iter()
        .map_init(MinimizerWorkspace::new, |ws, (_, s1, s2)| {
            let (mut a, b) =
                get_paired_minimizers_into(s1, *s2, engine.k, engine.w, engine.salt, ws);
            a.extend(b);
            if let Some(neg_set) = negative_mins {
                a.retain(|m| !neg_set.contains(m));
            }
            a
        })
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
    let fwd_score = if fwd_total > 0 {
        fwd_hits as f64 / fwd_total as f64
    } else {
        0.0
    };
    let rc_score = if rc_total > 0 {
        rc_hits as f64 / rc_total as f64
    } else {
        0.0
    };
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
            let entry = accumulators[read_idx as usize]
                .entry(bucket_id)
                .or_insert((0, 0));
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
                accumulate_match(query_idx, ref_idx, qi, ri, accumulators);
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
fn accumulate_merge_join(
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
    fn test_classify_batch_sharded_sequential_matches_regular() -> anyhow::Result<()> {
        use crate::sharded::ShardManifest;

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
            shards: vec![shard_info],
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
            classify_batch_sharded_sequential(&sharded, None, &records, threshold)?;

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
        use crate::sharded::ShardManifest;

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
            shards: vec![shard_info],
        };
        let manifest_path = ShardManifest::manifest_path(&base_path);
        manifest.save(&manifest_path)?;

        let sharded = ShardedInvertedIndex::open(&base_path)?;

        let query1: &[u8] = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let query2: &[u8] = b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA";
        let records: Vec<QueryRecord> = vec![(0, query1, None), (1, query2, None)];

        let threshold = 0.1;

        let results_sequential =
            classify_batch_sharded_sequential(&sharded, None, &records, threshold)?;
        let results_merge_join =
            classify_batch_sharded_merge_join(&sharded, None, &records, threshold)?;

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
        use crate::sharded_main::ShardedMainIndex;

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

    // ==================== Merge-Join Classification Tests ====================

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

    // ==================== Negative Filtering Tests ====================

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
        let (query_mins, _) = crate::extraction::get_paired_minimizers_into(
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
    fn test_classify_batch_sharded_sequential_negative_filtering() -> anyhow::Result<()> {
        use crate::sharded::ShardManifest;

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
            shards: vec![shard_info],
        };
        let manifest_path = ShardManifest::manifest_path(&base_path);
        manifest.save(&manifest_path)?;

        let sharded = ShardedInvertedIndex::open(&base_path)?;

        let query: &[u8] = seq1;
        let records: Vec<QueryRecord> = vec![(0, query, None)];

        // Without negative filtering
        let results_no_neg = classify_batch_sharded_sequential(&sharded, None, &records, 0.5)?;
        assert!(!results_no_neg.is_empty());

        // Extract query minimizers for filtering
        let (query_mins, _) = crate::extraction::get_paired_minimizers_into(
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
            classify_batch_sharded_sequential(&sharded, Some(&full_neg), &records, 0.5)?;
        assert!(results_full_neg.is_empty());

        Ok(())
    }

    #[test]
    fn test_classify_batch_sharded_merge_join_negative_filtering() -> anyhow::Result<()> {
        use crate::sharded::ShardManifest;

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
            shards: vec![shard_info],
        };
        let manifest_path = ShardManifest::manifest_path(&base_path);
        manifest.save(&manifest_path)?;

        let sharded = ShardedInvertedIndex::open(&base_path)?;

        let query: &[u8] = seq1;
        let records: Vec<QueryRecord> = vec![(0, query, None)];

        // Without negative filtering
        let results_no_neg = classify_batch_sharded_merge_join(&sharded, None, &records, 0.5)?;
        assert!(!results_no_neg.is_empty());

        // Extract query minimizers for filtering
        let (query_mins, _) = crate::extraction::get_paired_minimizers_into(
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
            classify_batch_sharded_merge_join(&sharded, Some(&full_neg), &records, 0.5)?;
        assert!(results_full_neg.is_empty());

        Ok(())
    }

    #[test]
    fn test_classify_batch_sharded_main_negative_filtering() -> anyhow::Result<()> {
        use crate::sharded_main::ShardedMainIndex;

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
        let (query_mins, _) = crate::extraction::get_paired_minimizers_into(
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
        let (mins1, _) = crate::extraction::get_paired_minimizers_into(
            &seq_a[0..50],
            None,
            index.k,
            index.w,
            index.salt,
            &mut ws,
        );
        let (mins2, _) = crate::extraction::get_paired_minimizers_into(
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

    // ==================== Edge Case Tests ====================

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

    // ==================== Parquet Shard Classification Tests ====================

    #[cfg(feature = "parquet")]
    #[test]
    fn test_classify_batch_sharded_sequential_parquet() -> anyhow::Result<()> {
        use crate::sharded::{ShardFormat, ShardManifest};
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
        let shard_info = inverted.save_shard_parquet(&shard_path, 0)?;

        let manifest = ShardManifest {
            k: inverted.k,
            w: inverted.w,
            salt: inverted.salt,
            source_hash: 0,
            total_minimizers: inverted.num_minimizers(),
            total_bucket_ids: inverted.num_bucket_entries(),
            has_overlapping_shards: true,
            shards: vec![shard_info],
        };
        let manifest_path = ShardManifest::manifest_path(&base_path);
        manifest.save(&manifest_path)?;

        // Open with auto-detection (should detect Parquet)
        let sharded = ShardedInvertedIndex::open(&base_path)?;
        assert_eq!(sharded.shard_format(), ShardFormat::Parquet);

        let query1: &[u8] = seq1;
        let query2: &[u8] = seq2;
        let records: Vec<QueryRecord> = vec![(0, query1, None), (1, query2, None)];

        let threshold = 0.1;

        // Classify using Parquet shards
        let results_parquet =
            classify_batch_sharded_sequential(&sharded, None, &records, threshold)?;

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

    #[cfg(feature = "parquet")]
    #[test]
    fn test_classify_batch_sharded_merge_join_parquet() -> anyhow::Result<()> {
        use crate::sharded::{ShardFormat, ShardManifest};
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
        let shard_info = inverted.save_shard_parquet(&shard_path, 0)?;

        let manifest = ShardManifest {
            k: inverted.k,
            w: inverted.w,
            salt: inverted.salt,
            source_hash: 0,
            total_minimizers: inverted.num_minimizers(),
            total_bucket_ids: inverted.num_bucket_entries(),
            has_overlapping_shards: true,
            shards: vec![shard_info],
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
            classify_batch_sharded_merge_join(&sharded, None, &records, threshold)?;

        assert!(!results_parquet.is_empty());

        // Verify we get hits for both queries
        let query0_hits: Vec<_> = results_parquet.iter().filter(|r| r.query_id == 0).collect();
        let query1_hits: Vec<_> = results_parquet.iter().filter(|r| r.query_id == 1).collect();

        assert!(!query0_hits.is_empty(), "Query 0 should have hits");
        assert!(!query1_hits.is_empty(), "Query 1 should have hits");

        Ok(())
    }

    #[cfg(feature = "parquet")]
    #[test]
    fn test_classify_parquet_matches_legacy() -> anyhow::Result<()> {
        use crate::sharded::ShardManifest;
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
            shards: vec![legacy_shard_info],
        };
        legacy_manifest.save(&ShardManifest::manifest_path(&legacy_base))?;

        // Save as Parquet shard
        let parquet_shard_path = ShardManifest::shard_path_parquet(&parquet_base, 0);
        let parquet_shard_info = inverted.save_shard_parquet(&parquet_shard_path, 0)?;

        let parquet_manifest = ShardManifest {
            k: inverted.k,
            w: inverted.w,
            salt: inverted.salt,
            source_hash: 0,
            total_minimizers: inverted.num_minimizers(),
            total_bucket_ids: inverted.num_bucket_entries(),
            has_overlapping_shards: true,
            shards: vec![parquet_shard_info],
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
            classify_batch_sharded_sequential(&legacy_sharded, None, &records, threshold)?;
        let results_parquet =
            classify_batch_sharded_sequential(&parquet_sharded, None, &records, threshold)?;

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
