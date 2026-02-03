//! Log-ratio computation for two-bucket classification.
//!
//! This module provides functions for computing log10(score_A / score_B)
//! between exactly two buckets for each read.

use rype::HitResult;
use std::collections::HashMap;
use std::io::Write;

/// Result of log-ratio computation for a single query.
#[derive(Debug, Clone, PartialEq)]
pub struct LogRatioResult {
    pub query_id: i64,
    pub log_ratio: f64,
}

/// Compute log10(numerator / denominator) with special handling for edge cases.
///
/// Edge cases:
/// - numerator = 0 → 0.0
/// - denominator = 0 and numerator > 0 → +infinity
/// - both = 0 → 0.0
pub fn compute_log_ratio(numerator: f64, denominator: f64) -> f64 {
    if numerator == 0.0 {
        // Both zero or only numerator zero → 0.0
        0.0
    } else if denominator == 0.0 {
        // Numerator > 0, denominator = 0 → +infinity
        f64::INFINITY
    } else {
        (numerator / denominator).log10()
    }
}

/// Format the bucket name for log-ratio output.
///
/// Returns `log10([num_name] / [denom_name])`.
pub fn format_log_ratio_bucket_name(num_name: &str, denom_name: &str) -> String {
    format!("log10([{}] / [{}])", num_name, denom_name)
}

/// Compute log ratios from hit results for all queries.
///
/// Groups hits by query_id and computes log10(numerator_score / denominator_score)
/// for each query. Queries not present in results are treated as having score 0.0.
pub fn compute_log_ratio_from_hits(
    results: &[HitResult],
    num_id: u32,
    denom_id: u32,
) -> Vec<LogRatioResult> {
    use std::collections::HashMap;

    // Group scores by query_id
    let mut query_scores: HashMap<i64, (f64, f64)> = HashMap::new();

    for hit in results {
        let entry = query_scores.entry(hit.query_id).or_insert((0.0, 0.0));
        if hit.bucket_id == num_id {
            entry.0 = hit.score;
        } else if hit.bucket_id == denom_id {
            entry.1 = hit.score;
        }
    }

    // Compute log ratios
    let mut log_ratios: Vec<LogRatioResult> = query_scores
        .into_iter()
        .map(|(query_id, (num_score, denom_score))| LogRatioResult {
            query_id,
            log_ratio: compute_log_ratio(num_score, denom_score),
        })
        .collect();

    // Sort by query_id for deterministic output
    log_ratios.sort_by_key(|r| r.query_id);
    log_ratios
}

/// Format log-ratio results as TSV output bytes.
///
/// Works with any header type that implements `AsRef<str>` (e.g., `&str`, `String`).
/// Infinite values are formatted as "inf".
///
/// # Arguments
/// * `log_ratios` - The computed log-ratio results
/// * `headers` - Read names/IDs indexed by query_id
/// * `ratio_bucket_name` - The formatted bucket name (e.g., "log10([A] / [B])")
///
/// # Returns
/// Formatted bytes: "read_id\tbucket_name\tscore\n" for each result
pub fn format_log_ratio_output<S: AsRef<str>>(
    log_ratios: &[LogRatioResult],
    headers: &[S],
    ratio_bucket_name: &str,
) -> Vec<u8> {
    let mut output = Vec::with_capacity(log_ratios.len() * 64);
    for lr in log_ratios {
        let header = headers[lr.query_id as usize].as_ref();
        if lr.log_ratio.is_infinite() {
            writeln!(output, "{}\t{}\tinf", header, ratio_bucket_name).unwrap();
        } else {
            writeln!(
                output,
                "{}\t{}\t{:.4}",
                header, ratio_bucket_name, lr.log_ratio
            )
            .unwrap();
        }
    }
    output
}

/// Filter log-ratio results by threshold based on original classification scores.
///
/// Retains a log-ratio result only if EITHER the numerator or denominator score
/// meets or exceeds the threshold. This ensures we output reads that had meaningful
/// classification to at least one of the two buckets.
///
/// # Arguments
/// * `log_ratios` - Mutable vector of log-ratio results to filter in place
/// * `original_results` - The original HitResult scores from classification
/// * `num_id` - Bucket ID for the numerator
/// * `denom_id` - Bucket ID for the denominator
/// * `threshold` - Minimum score threshold (results kept if either score >= threshold)
pub fn filter_log_ratios_by_threshold(
    log_ratios: &mut Vec<LogRatioResult>,
    original_results: &[HitResult],
    num_id: u32,
    denom_id: u32,
    threshold: f64,
) {
    // Build a map of query_id -> (num_score, denom_score)
    let mut query_scores: HashMap<i64, (f64, f64)> = HashMap::new();
    for hit in original_results {
        let entry = query_scores.entry(hit.query_id).or_insert((0.0, 0.0));
        if hit.bucket_id == num_id {
            entry.0 = hit.score;
        } else if hit.bucket_id == denom_id {
            entry.1 = hit.score;
        }
    }

    log_ratios.retain(|lr| {
        if let Some(&(num_score, denom_score)) = query_scores.get(&lr.query_id) {
            // Keep if EITHER score is >= threshold
            num_score >= threshold || denom_score >= threshold
        } else {
            false
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_log_ratio_both_positive() {
        // 100 / 10 = 10, log10(10) = 1.0
        let result = compute_log_ratio(100.0, 10.0);
        assert!((result - 1.0).abs() < 1e-10);

        // 10 / 100 = 0.1, log10(0.1) = -1.0
        let result = compute_log_ratio(10.0, 100.0);
        assert!((result - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_compute_log_ratio_equal_scores() {
        // Equal scores → log10(1) = 0.0
        let result = compute_log_ratio(50.0, 50.0);
        assert!((result - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_log_ratio_numerator_zero() {
        // numerator = 0 → 0.0
        let result = compute_log_ratio(0.0, 50.0);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_compute_log_ratio_denominator_zero() {
        // denominator = 0, numerator > 0 → +infinity
        let result = compute_log_ratio(50.0, 0.0);
        assert!(result.is_infinite() && result.is_sign_positive());
    }

    #[test]
    fn test_compute_log_ratio_both_zero() {
        // both = 0 → 0.0
        let result = compute_log_ratio(0.0, 0.0);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_format_bucket_name() {
        let name = format_log_ratio_bucket_name("BucketA", "BucketB");
        assert_eq!(name, "log10([BucketA] / [BucketB])");
    }

    #[test]
    fn test_compute_from_hits_both_present() {
        let hits = vec![
            HitResult {
                query_id: 1,
                bucket_id: 0,
                score: 100.0,
            },
            HitResult {
                query_id: 1,
                bucket_id: 1,
                score: 10.0,
            },
        ];

        let results = compute_log_ratio_from_hits(&hits, 0, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].query_id, 1);
        assert!((results[0].log_ratio - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_from_hits_only_numerator() {
        let hits = vec![HitResult {
            query_id: 1,
            bucket_id: 0,
            score: 50.0,
        }];

        let results = compute_log_ratio_from_hits(&hits, 0, 1);
        assert_eq!(results.len(), 1);
        assert!(results[0].log_ratio.is_infinite());
    }

    #[test]
    fn test_compute_from_hits_only_denominator() {
        let hits = vec![HitResult {
            query_id: 1,
            bucket_id: 1,
            score: 50.0,
        }];

        let results = compute_log_ratio_from_hits(&hits, 0, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].log_ratio, 0.0); // numerator is 0
    }

    #[test]
    fn test_compute_from_hits_multiple_queries() {
        let hits = vec![
            HitResult {
                query_id: 1,
                bucket_id: 0,
                score: 100.0,
            },
            HitResult {
                query_id: 1,
                bucket_id: 1,
                score: 10.0,
            },
            HitResult {
                query_id: 2,
                bucket_id: 0,
                score: 10.0,
            },
            HitResult {
                query_id: 2,
                bucket_id: 1,
                score: 100.0,
            },
            HitResult {
                query_id: 3,
                bucket_id: 0,
                score: 50.0,
            },
            HitResult {
                query_id: 3,
                bucket_id: 1,
                score: 50.0,
            },
        ];

        let results = compute_log_ratio_from_hits(&hits, 0, 1);
        assert_eq!(results.len(), 3);

        // Results sorted by query_id
        assert_eq!(results[0].query_id, 1);
        assert!((results[0].log_ratio - 1.0).abs() < 1e-10); // 100/10

        assert_eq!(results[1].query_id, 2);
        assert!((results[1].log_ratio - (-1.0)).abs() < 1e-10); // 10/100

        assert_eq!(results[2].query_id, 3);
        assert!((results[2].log_ratio - 0.0).abs() < 1e-10); // 50/50
    }

    #[test]
    fn test_compute_from_hits_empty() {
        let hits: Vec<HitResult> = vec![];
        let results = compute_log_ratio_from_hits(&hits, 0, 1);
        assert!(results.is_empty());
    }

    #[test]
    fn test_compute_from_hits_swapped_ids() {
        // Same data but swap numerator/denominator IDs
        let hits = vec![
            HitResult {
                query_id: 1,
                bucket_id: 0,
                score: 100.0,
            },
            HitResult {
                query_id: 1,
                bucket_id: 1,
                score: 10.0,
            },
        ];

        // num_id=0, denom_id=1 → log10(100/10) = 1.0
        let results_normal = compute_log_ratio_from_hits(&hits, 0, 1);
        assert!((results_normal[0].log_ratio - 1.0).abs() < 1e-10);

        // num_id=1, denom_id=0 → log10(10/100) = -1.0
        let results_swapped = compute_log_ratio_from_hits(&hits, 1, 0);
        assert!((results_swapped[0].log_ratio - (-1.0)).abs() < 1e-10);
    }

    // Tests for format_log_ratio_output

    #[test]
    fn test_format_log_ratio_output_with_str_refs() {
        let log_ratios = vec![
            LogRatioResult {
                query_id: 0,
                log_ratio: 1.5,
            },
            LogRatioResult {
                query_id: 1,
                log_ratio: -0.5,
            },
        ];
        let headers: Vec<&str> = vec!["read_0", "read_1"];
        let ratio_name = "log10([A] / [B])";

        let output = format_log_ratio_output(&log_ratios, &headers, ratio_name);
        let output_str = String::from_utf8(output).unwrap();

        assert_eq!(
            output_str,
            "read_0\tlog10([A] / [B])\t1.5000\nread_1\tlog10([A] / [B])\t-0.5000\n"
        );
    }

    #[test]
    fn test_format_log_ratio_output_with_owned_strings() {
        let log_ratios = vec![LogRatioResult {
            query_id: 0,
            log_ratio: 2.0,
        }];
        let headers: Vec<String> = vec!["my_read".to_string()];
        let ratio_name = "log10([X] / [Y])";

        let output = format_log_ratio_output(&log_ratios, &headers, ratio_name);
        let output_str = String::from_utf8(output).unwrap();

        assert_eq!(output_str, "my_read\tlog10([X] / [Y])\t2.0000\n");
    }

    #[test]
    fn test_format_log_ratio_output_infinite_value() {
        let log_ratios = vec![LogRatioResult {
            query_id: 0,
            log_ratio: f64::INFINITY,
        }];
        let headers: Vec<&str> = vec!["read_inf"];
        let ratio_name = "ratio";

        let output = format_log_ratio_output(&log_ratios, &headers, ratio_name);
        let output_str = String::from_utf8(output).unwrap();

        assert_eq!(output_str, "read_inf\tratio\tinf\n");
    }

    #[test]
    fn test_format_log_ratio_output_empty_results() {
        let log_ratios: Vec<LogRatioResult> = vec![];
        let headers: Vec<&str> = vec!["read_0"];
        let ratio_name = "ratio";

        let output = format_log_ratio_output(&log_ratios, &headers, ratio_name);
        assert!(output.is_empty());
    }

    // Tests for filter_log_ratios_by_threshold

    #[test]
    fn test_filter_log_ratios_keeps_above_threshold() {
        let mut log_ratios = vec![
            LogRatioResult {
                query_id: 0,
                log_ratio: 1.0,
            },
            LogRatioResult {
                query_id: 1,
                log_ratio: 0.5,
            },
        ];
        let hits = vec![
            HitResult {
                query_id: 0,
                bucket_id: 0,
                score: 0.8,
            }, // num
            HitResult {
                query_id: 0,
                bucket_id: 1,
                score: 0.2,
            }, // denom
            HitResult {
                query_id: 1,
                bucket_id: 0,
                score: 0.3,
            }, // num
            HitResult {
                query_id: 1,
                bucket_id: 1,
                score: 0.1,
            }, // denom
        ];

        filter_log_ratios_by_threshold(&mut log_ratios, &hits, 0, 1, 0.5);

        // Query 0: num=0.8 >= 0.5, kept
        // Query 1: num=0.3 < 0.5 AND denom=0.1 < 0.5, filtered out
        assert_eq!(log_ratios.len(), 1);
        assert_eq!(log_ratios[0].query_id, 0);
    }

    #[test]
    fn test_filter_log_ratios_keeps_if_denom_above_threshold() {
        let mut log_ratios = vec![LogRatioResult {
            query_id: 0,
            log_ratio: -1.0,
        }];
        let hits = vec![
            HitResult {
                query_id: 0,
                bucket_id: 0,
                score: 0.1,
            }, // num below threshold
            HitResult {
                query_id: 0,
                bucket_id: 1,
                score: 0.9,
            }, // denom above threshold
        ];

        filter_log_ratios_by_threshold(&mut log_ratios, &hits, 0, 1, 0.5);

        // Denom score (0.9) >= threshold (0.5), so keep
        assert_eq!(log_ratios.len(), 1);
    }

    #[test]
    fn test_filter_log_ratios_removes_below_both_thresholds() {
        let mut log_ratios = vec![LogRatioResult {
            query_id: 0,
            log_ratio: 0.0,
        }];
        let hits = vec![
            HitResult {
                query_id: 0,
                bucket_id: 0,
                score: 0.4,
            },
            HitResult {
                query_id: 0,
                bucket_id: 1,
                score: 0.4,
            },
        ];

        filter_log_ratios_by_threshold(&mut log_ratios, &hits, 0, 1, 0.5);

        // Both scores below threshold
        assert!(log_ratios.is_empty());
    }

    #[test]
    fn test_filter_log_ratios_removes_missing_query() {
        let mut log_ratios = vec![LogRatioResult {
            query_id: 99,
            log_ratio: 1.0,
        }];
        let hits = vec![
            HitResult {
                query_id: 0,
                bucket_id: 0,
                score: 0.8,
            }, // Different query
        ];

        filter_log_ratios_by_threshold(&mut log_ratios, &hits, 0, 1, 0.1);

        // Query 99 not found in hits, filtered out
        assert!(log_ratios.is_empty());
    }

    #[test]
    fn test_filter_log_ratios_zero_threshold_keeps_all() {
        let mut log_ratios = vec![
            LogRatioResult {
                query_id: 0,
                log_ratio: 1.0,
            },
            LogRatioResult {
                query_id: 1,
                log_ratio: -1.0,
            },
        ];
        let hits = vec![
            HitResult {
                query_id: 0,
                bucket_id: 0,
                score: 0.001,
            },
            HitResult {
                query_id: 1,
                bucket_id: 1,
                score: 0.001,
            },
        ];

        filter_log_ratios_by_threshold(&mut log_ratios, &hits, 0, 1, 0.0);

        // Threshold 0.0 keeps everything with any score
        assert_eq!(log_ratios.len(), 2);
    }
}
