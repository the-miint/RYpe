//! Log-ratio types and functions for two-index classification.
//!
//! This module provides the core types and logic for computing
//! log10(numerator_score / denominator_score) between two single-bucket indices.
//! It lives in the library crate so both the CLI and C API can use it.

use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, Result};

use crate::types::{HitResult, IndexMetadata, QueryRecord};
use crate::ShardedInvertedIndex;

/// Indicates whether a log-ratio result was determined via a fast path
/// (skipping the denominator classification) or computed exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastPath {
    /// Result was computed exactly (both numerator and denominator classified).
    None,
    /// Numerator score exceeded the skip threshold, so log-ratio is +inf without needing denominator.
    NumHigh,
}

impl FastPath {
    /// Return a short string label for TSV output.
    pub fn as_str(&self) -> &'static str {
        match self {
            FastPath::None => "none",
            FastPath::NumHigh => "num_high",
        }
    }
}

/// Result of log-ratio computation for a single query.
#[derive(Debug, Clone, PartialEq)]
pub struct LogRatioResult {
    pub query_id: i64,
    pub log_ratio: f64,
    pub fast_path: FastPath,
}

/// Compute log10(numerator / denominator) with special handling for edge cases.
///
/// Edge cases:
/// - numerator = 0, denominator > 0 → -infinity (read matches denom but not num)
/// - numerator > 0, denominator = 0 → +infinity (read matches num but not denom)
/// - both = 0 → NaN (no evidence for or against)
pub fn compute_log_ratio(numerator: f64, denominator: f64) -> f64 {
    if numerator == 0.0 && denominator == 0.0 {
        f64::NAN
    } else if numerator == 0.0 {
        f64::NEG_INFINITY
    } else if denominator == 0.0 {
        f64::INFINITY
    } else {
        (numerator / denominator).log10()
    }
}

/// Validate that the index has exactly one bucket and return its ID and name.
///
/// Used for the two-index log-ratio workflow where each index holds a single bucket.
pub fn validate_single_bucket_index(bucket_names: &HashMap<u32, String>) -> Result<(u32, String)> {
    if bucket_names.len() != 1 {
        return Err(anyhow!(
            "log-ratio mode requires each index to have exactly 1 bucket, but found {}.\n\
             Use 'rype index stats -i <index>' to see bucket information.",
            bucket_names.len()
        ));
    }

    let (&bucket_id, bucket_name) = bucket_names.iter().next().unwrap();
    Ok((bucket_id, bucket_name.clone()))
}

/// Validate that two indices are compatible for log-ratio computation.
///
/// Checks that k, w, and salt match between the numerator and denominator indices.
pub fn validate_compatible_indices(a: &IndexMetadata, b: &IndexMetadata) -> Result<()> {
    if a.k != b.k {
        return Err(anyhow!(
            "Numerator and denominator indices have different k values: {} vs {}.\n\
             Both indices must be built with the same k, w, and salt.",
            a.k,
            b.k
        ));
    }
    if a.w != b.w {
        return Err(anyhow!(
            "Numerator and denominator indices have different w values: {} vs {}.\n\
             Both indices must be built with the same k, w, and salt.",
            a.w,
            b.w
        ));
    }
    if a.salt != b.salt {
        return Err(anyhow!(
            "Numerator and denominator indices have different salt values: {:#x} vs {:#x}.\n\
             Both indices must be built with the same k, w, and salt.",
            a.salt,
            b.salt
        ));
    }
    Ok(())
}

/// Result of partitioning reads by numerator score into fast-path and needs-denominator groups.
pub struct PartitionResult {
    /// Reads resolved via fast path (NumHigh only) — no denominator needed.
    pub fast_path_results: Vec<LogRatioResult>,
    /// Query IDs of reads that need denominator classification.
    pub needs_denom_query_ids: Vec<i64>,
    /// Numerator scores indexed by query_id (0.0 for zero-score reads).
    /// Only entries for needs-denom reads are meaningful.
    pub num_scores: Vec<f64>,
}

/// Partition reads by numerator classification results into fast-path and needs-denominator groups.
///
/// For each read in 0..total_reads:
/// - If `skip_threshold` is set and the read's numerator score >= threshold, it gets
///   fast-path `+inf` (NumHigh).
/// - Otherwise (including score=0), the read needs denominator classification.
///   Score=0 reads need the denominator to distinguish -inf (denom>0) from NaN (denom=0).
///
/// `num_results` are the HitResults from classifying against the numerator index
/// (single bucket, threshold=0.0).
pub fn partition_by_numerator_score(
    num_results: &[HitResult],
    total_reads: usize,
    skip_threshold: Option<f64>,
) -> PartitionResult {
    // Dense score lookup: query_ids are sequential 0..total_reads
    let mut num_scores = vec![0.0_f64; total_reads];
    for hit in num_results {
        num_scores[hit.query_id as usize] = hit.score;
    }

    let mut fast_path_results = Vec::new();
    let mut needs_denom_query_ids = Vec::new();

    for query_id in 0..total_reads as i64 {
        let score = num_scores[query_id as usize];

        if let Some(thresh) = skip_threshold {
            if score >= thresh {
                // Strong numerator signal → +inf
                fast_path_results.push(LogRatioResult {
                    query_id,
                    log_ratio: f64::INFINITY,
                    fast_path: FastPath::NumHigh,
                });
            } else {
                needs_denom_query_ids.push(query_id);
            }
        } else {
            needs_denom_query_ids.push(query_id);
        }
    }

    PartitionResult {
        fast_path_results,
        needs_denom_query_ids,
        num_scores,
    }
}

/// Validate that two sharded indices are compatible for log-ratio classification.
///
/// Checks that both are single-bucket indices with matching k, w, and salt.
/// Returns `(k, w, salt)` on success.
pub fn validate_log_ratio_indices(
    numerator: &ShardedInvertedIndex,
    denominator: &ShardedInvertedIndex,
) -> Result<(usize, usize, u64)> {
    let num_manifest = numerator.manifest();
    let denom_manifest = denominator.manifest();

    validate_single_bucket_index(&num_manifest.bucket_names)
        .map_err(|e| anyhow!("numerator index: {}", e))?;
    validate_single_bucket_index(&denom_manifest.bucket_names)
        .map_err(|e| anyhow!("denominator index: {}", e))?;

    if num_manifest.k != denom_manifest.k {
        return Err(anyhow!(
            "Numerator and denominator indices have different k values: {} vs {}.\n\
             Both indices must be built with the same k, w, and salt.",
            num_manifest.k,
            denom_manifest.k
        ));
    }
    if num_manifest.w != denom_manifest.w {
        return Err(anyhow!(
            "Numerator and denominator indices have different w values: {} vs {}.\n\
             Both indices must be built with the same k, w, and salt.",
            num_manifest.w,
            denom_manifest.w
        ));
    }
    if num_manifest.salt != denom_manifest.salt {
        return Err(anyhow!(
            "Numerator and denominator indices have different salt values: {:#x} vs {:#x}.\n\
             Both indices must be built with the same k, w, and salt.",
            num_manifest.salt,
            denom_manifest.salt
        ));
    }

    Ok((num_manifest.k, num_manifest.w, num_manifest.salt))
}

/// Classify a batch of reads using log-ratio (numerator vs denominator).
///
/// This is the core log-ratio pipeline:
/// 1. Validate indices (single-bucket, compatible k/w/salt)
/// 2. Extract minimizers from all reads
/// 3. Classify all against numerator (threshold=0.0)
/// 4. Partition into fast-path (NumHigh) and needs-denom
/// 5. Classify needs-denom subset against denominator
/// 6. Compute log10(num_score / denom_score) for each read
///
/// Returns one `LogRatioResult` per input read, sorted by the original query IDs
/// from the input records.
///
/// Note: Internally uses sequential query IDs (0..N) for the partition step,
/// then maps back to the original IDs in the results. This avoids panics when
/// caller-provided query IDs are non-sequential (e.g., [100, 200, 300]).
pub fn classify_log_ratio_batch(
    numerator: &ShardedInvertedIndex,
    denominator: &ShardedInvertedIndex,
    records: &[QueryRecord],
    skip_threshold: Option<f64>,
) -> Result<Vec<LogRatioResult>> {
    let num_queries = records.len();
    if num_queries == 0 {
        return Ok(Vec::new());
    }

    let (k, w, salt) = validate_log_ratio_indices(numerator, denominator)?;

    // Save original query IDs, use sequential 0..N internally.
    // partition_by_numerator_score uses dense arrays indexed by query_id,
    // so query IDs must be sequential 0..N.
    let original_ids: Vec<i64> = records.iter().map(|r| r.0).collect();
    let sequential_ids: Vec<i64> = (0..num_queries as i64).collect();

    let extracted = crate::extract_batch_minimizers(k, w, salt, None, records);

    // Classify against numerator (threshold=0.0 to get all scores)
    let num_results = crate::classify_from_extracted_minimizers(
        numerator,
        &extracted,
        &sequential_ids,
        0.0,
        None,
    )?;

    // Partition: fast-path vs needs-denom (uses sequential IDs internally)
    let partition = partition_by_numerator_score(&num_results, num_queries, skip_threshold);

    // Build needs-denom subset
    let needs_denom_set: HashSet<i64> = partition.needs_denom_query_ids.iter().copied().collect();

    let mut denom_extracted = Vec::new();
    let mut denom_ids = Vec::new();
    for (i, ext) in extracted.iter().enumerate() {
        let seq_id = i as i64;
        if needs_denom_set.contains(&seq_id) {
            denom_extracted.push(ext.clone());
            denom_ids.push(seq_id);
        }
    }

    // Classify needs-denom subset against denominator
    let denom_results = if !denom_ids.is_empty() {
        crate::classify_from_extracted_minimizers(
            denominator,
            &denom_extracted,
            &denom_ids,
            0.0,
            None,
        )?
    } else {
        Vec::new()
    };

    // Build dense denominator score lookup (indexed by sequential ID)
    let mut denom_scores = vec![0.0_f64; num_queries];
    for hit in &denom_results {
        denom_scores[hit.query_id as usize] = hit.score;
    }

    // Merge fast-path + computed results, mapping back to original query IDs
    let mut results: Vec<LogRatioResult> = Vec::with_capacity(num_queries);

    for lr in &partition.fast_path_results {
        results.push(LogRatioResult {
            query_id: original_ids[lr.query_id as usize],
            log_ratio: lr.log_ratio,
            fast_path: lr.fast_path,
        });
    }

    for &seq_id in &partition.needs_denom_query_ids {
        let idx = seq_id as usize;
        let num_score = partition.num_scores[idx];
        let denom_score = denom_scores[idx];
        let log_ratio = compute_log_ratio(num_score, denom_score);
        results.push(LogRatioResult {
            query_id: original_ids[idx],
            log_ratio,
            fast_path: FastPath::None,
        });
    }

    // Sort by original query_id for deterministic output
    results.sort_by_key(|r| r.query_id);
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    // FastPath tests

    #[test]
    fn test_fast_path_as_str() {
        assert_eq!(FastPath::None.as_str(), "none");
        assert_eq!(FastPath::NumHigh.as_str(), "num_high");
    }

    #[test]
    fn test_log_ratio_result_with_fast_path() {
        let result = LogRatioResult {
            query_id: 7,
            log_ratio: f64::INFINITY,
            fast_path: FastPath::NumHigh,
        };
        assert_eq!(result.fast_path, FastPath::NumHigh);

        let result = LogRatioResult {
            query_id: 0,
            log_ratio: 1.5,
            fast_path: FastPath::None,
        };
        assert_eq!(result.fast_path, FastPath::None);
    }

    // compute_log_ratio tests

    #[test]
    fn test_compute_log_ratio_both_positive() {
        let result = compute_log_ratio(100.0, 10.0);
        assert!((result - 1.0).abs() < 1e-10);

        let result = compute_log_ratio(10.0, 100.0);
        assert!((result - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_compute_log_ratio_equal_scores() {
        let result = compute_log_ratio(50.0, 50.0);
        assert!((result - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_log_ratio_numerator_zero() {
        let result = compute_log_ratio(0.0, 50.0);
        assert!(result.is_infinite() && result.is_sign_negative());
    }

    #[test]
    fn test_compute_log_ratio_denominator_zero() {
        let result = compute_log_ratio(50.0, 0.0);
        assert!(result.is_infinite() && result.is_sign_positive());
    }

    #[test]
    fn test_compute_log_ratio_both_zero() {
        let result = compute_log_ratio(0.0, 0.0);
        assert!(result.is_nan());
    }

    // validate_single_bucket_index tests

    #[test]
    fn test_validate_single_bucket_index_passes() {
        let mut bucket_names = HashMap::new();
        bucket_names.insert(0, "MyBucket".to_string());

        let result = validate_single_bucket_index(&bucket_names);
        assert!(result.is_ok());

        let (bucket_id, bucket_name) = result.unwrap();
        assert_eq!(bucket_id, 0);
        assert_eq!(bucket_name, "MyBucket");
    }

    #[test]
    fn test_validate_single_bucket_index_fails_empty() {
        let bucket_names: HashMap<u32, String> = HashMap::new();

        let result = validate_single_bucket_index(&bucket_names);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("exactly 1 bucket"));
        assert!(err.contains("found 0"));
    }

    #[test]
    fn test_validate_single_bucket_index_fails_two_buckets() {
        let mut bucket_names = HashMap::new();
        bucket_names.insert(0, "A".to_string());
        bucket_names.insert(1, "B".to_string());

        let result = validate_single_bucket_index(&bucket_names);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("exactly 1 bucket"));
        assert!(err.contains("found 2"));
    }

    #[test]
    fn test_validate_single_bucket_index_preserves_id() {
        let mut bucket_names = HashMap::new();
        bucket_names.insert(42, "HighId".to_string());

        let result = validate_single_bucket_index(&bucket_names);
        assert!(result.is_ok());

        let (bucket_id, bucket_name) = result.unwrap();
        assert_eq!(bucket_id, 42);
        assert_eq!(bucket_name, "HighId");
    }

    // validate_compatible_indices tests

    fn make_metadata(k: usize, w: usize, salt: u64) -> IndexMetadata {
        IndexMetadata {
            k,
            w,
            salt,
            bucket_names: HashMap::new(),
            bucket_sources: HashMap::new(),
            bucket_minimizer_counts: HashMap::new(),
            largest_shard_entries: 0,
            bucket_file_stats: None,
        }
    }

    #[test]
    fn test_validate_compatible_indices_passes_when_matching() {
        let a = make_metadata(32, 10, 0x5555555555555555);
        let b = make_metadata(32, 10, 0x5555555555555555);

        assert!(validate_compatible_indices(&a, &b).is_ok());
    }

    #[test]
    fn test_validate_compatible_indices_fails_on_k_mismatch() {
        let a = make_metadata(32, 10, 0x5555555555555555);
        let b = make_metadata(64, 10, 0x5555555555555555);

        let result = validate_compatible_indices(&a, &b);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("different k values"));
    }

    #[test]
    fn test_validate_compatible_indices_fails_on_w_mismatch() {
        let a = make_metadata(32, 10, 0x5555555555555555);
        let b = make_metadata(32, 20, 0x5555555555555555);

        let result = validate_compatible_indices(&a, &b);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("different w values"));
    }

    #[test]
    fn test_validate_compatible_indices_fails_on_salt_mismatch() {
        let a = make_metadata(32, 10, 0x5555555555555555);
        let b = make_metadata(32, 10, 0xAAAAAAAAAAAAAAAA);

        let result = validate_compatible_indices(&a, &b);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("different salt values"));
    }

    // partition_by_numerator_score tests

    #[test]
    fn test_partition_all_zeros_goes_to_needs_denom() {
        let num_results: Vec<HitResult> = vec![];
        let result = partition_by_numerator_score(&num_results, 3, None);

        assert!(result.fast_path_results.is_empty());
        assert_eq!(result.needs_denom_query_ids.len(), 3);
        assert_eq!(result.needs_denom_query_ids, vec![0, 1, 2]);
        assert!(result.num_scores.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn test_partition_with_skip_threshold_creates_two_groups() {
        let num_results = vec![
            HitResult {
                query_id: 1,
                bucket_id: 0,
                score: 0.05,
            },
            HitResult {
                query_id: 2,
                bucket_id: 0,
                score: 0.5,
            },
            HitResult {
                query_id: 3,
                bucket_id: 0,
                score: 0.01,
            },
        ];

        let result = partition_by_numerator_score(&num_results, 4, Some(0.1));

        assert_eq!(result.fast_path_results.len(), 1);
        assert_eq!(result.fast_path_results[0].query_id, 2);
        assert_eq!(result.fast_path_results[0].fast_path, FastPath::NumHigh);
        assert!(result.fast_path_results[0].log_ratio == f64::INFINITY);

        assert_eq!(result.needs_denom_query_ids.len(), 3);
        assert!(result.needs_denom_query_ids.contains(&0));
        assert!(result.needs_denom_query_ids.contains(&1));
        assert!(result.needs_denom_query_ids.contains(&3));
    }

    #[test]
    fn test_partition_without_skip_threshold_no_fast_path() {
        let num_results = vec![
            HitResult {
                query_id: 1,
                bucket_id: 0,
                score: 0.5,
            },
            HitResult {
                query_id: 2,
                bucket_id: 0,
                score: 0.9,
            },
        ];

        let result = partition_by_numerator_score(&num_results, 3, None);

        assert!(result.fast_path_results.is_empty());
        assert_eq!(result.needs_denom_query_ids.len(), 3);
    }

    #[test]
    fn test_partition_skip_threshold_at_boundary() {
        let num_results = vec![HitResult {
            query_id: 0,
            bucket_id: 0,
            score: 0.1,
        }];

        let result = partition_by_numerator_score(&num_results, 1, Some(0.1));

        assert_eq!(result.fast_path_results.len(), 1);
        assert_eq!(result.fast_path_results[0].fast_path, FastPath::NumHigh);
        assert!(result.needs_denom_query_ids.is_empty());
    }

    #[test]
    fn test_partition_empty_batch() {
        let result = partition_by_numerator_score(&[], 0, None);

        assert!(result.fast_path_results.is_empty());
        assert!(result.needs_denom_query_ids.is_empty());
        assert!(result.num_scores.is_empty());
    }
}
