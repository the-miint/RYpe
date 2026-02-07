//! Log-ratio computation for two-bucket classification.
//!
//! This module provides functions for computing log10(score_A / score_B)
//! between exactly two buckets for each read.

use std::io::Write;

/// Indicates whether a log-ratio result was determined via a fast path
/// (skipping the denominator classification) or computed exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastPath {
    /// Result was computed exactly (both numerator and denominator classified).
    None,
    /// Numerator score was zero, so log-ratio is -inf without needing denominator.
    NumZero,
    /// Numerator score exceeded the skip threshold, so log-ratio is +inf without needing denominator.
    NumHigh,
}

impl FastPath {
    /// Return a short string label for TSV output.
    pub fn as_str(&self) -> &'static str {
        match self {
            FastPath::None => "none",
            FastPath::NumZero => "num_zero",
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
/// - numerator = 0 → -infinity (no numerator signal)
/// - denominator = 0 and numerator > 0 → +infinity
/// - both = 0 → -infinity (no numerator signal)
pub fn compute_log_ratio(numerator: f64, denominator: f64) -> f64 {
    if numerator == 0.0 {
        f64::NEG_INFINITY
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
        let fast_path = lr.fast_path.as_str();
        if lr.log_ratio == f64::NEG_INFINITY {
            writeln!(
                output,
                "{}\t{}\t-inf\t{}",
                header, ratio_bucket_name, fast_path
            )
            .unwrap();
        } else if lr.log_ratio.is_infinite() {
            writeln!(
                output,
                "{}\t{}\tinf\t{}",
                header, ratio_bucket_name, fast_path
            )
            .unwrap();
        } else {
            writeln!(
                output,
                "{}\t{}\t{:.4}\t{}",
                header, ratio_bucket_name, lr.log_ratio, fast_path
            )
            .unwrap();
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_path_as_str() {
        assert_eq!(FastPath::None.as_str(), "none");
        assert_eq!(FastPath::NumZero.as_str(), "num_zero");
        assert_eq!(FastPath::NumHigh.as_str(), "num_high");
    }

    #[test]
    fn test_log_ratio_result_with_fast_path() {
        let result = LogRatioResult {
            query_id: 42,
            log_ratio: f64::NEG_INFINITY,
            fast_path: FastPath::NumZero,
        };
        assert_eq!(result.query_id, 42);
        assert!(result.log_ratio.is_infinite() && result.log_ratio.is_sign_negative());
        assert_eq!(result.fast_path, FastPath::NumZero);

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
        // numerator = 0 → -inf
        let result = compute_log_ratio(0.0, 50.0);
        assert!(result.is_infinite() && result.is_sign_negative());
    }

    #[test]
    fn test_compute_log_ratio_denominator_zero() {
        // denominator = 0, numerator > 0 → +infinity
        let result = compute_log_ratio(50.0, 0.0);
        assert!(result.is_infinite() && result.is_sign_positive());
    }

    #[test]
    fn test_compute_log_ratio_both_zero() {
        // both = 0 → -inf (numerator is zero)
        let result = compute_log_ratio(0.0, 0.0);
        assert!(result.is_infinite() && result.is_sign_negative());
    }

    #[test]
    fn test_format_bucket_name() {
        let name = format_log_ratio_bucket_name("BucketA", "BucketB");
        assert_eq!(name, "log10([BucketA] / [BucketB])");
    }

    #[test]
    fn test_format_log_ratio_output_with_str_refs() {
        let log_ratios = vec![
            LogRatioResult {
                query_id: 0,
                log_ratio: 1.5,
                fast_path: FastPath::None,
            },
            LogRatioResult {
                query_id: 1,
                log_ratio: -0.5,
                fast_path: FastPath::None,
            },
        ];
        let headers: Vec<&str> = vec!["read_0", "read_1"];
        let ratio_name = "log10([A] / [B])";

        let output = format_log_ratio_output(&log_ratios, &headers, ratio_name);
        let output_str = String::from_utf8(output).unwrap();

        assert_eq!(
            output_str,
            "read_0\tlog10([A] / [B])\t1.5000\tnone\nread_1\tlog10([A] / [B])\t-0.5000\tnone\n"
        );
    }

    #[test]
    fn test_format_log_ratio_output_with_owned_strings() {
        let log_ratios = vec![LogRatioResult {
            query_id: 0,
            log_ratio: 2.0,
            fast_path: FastPath::None,
        }];
        let headers: Vec<String> = vec!["my_read".to_string()];
        let ratio_name = "log10([X] / [Y])";

        let output = format_log_ratio_output(&log_ratios, &headers, ratio_name);
        let output_str = String::from_utf8(output).unwrap();

        assert_eq!(output_str, "my_read\tlog10([X] / [Y])\t2.0000\tnone\n");
    }

    #[test]
    fn test_format_log_ratio_output_positive_infinity() {
        let log_ratios = vec![LogRatioResult {
            query_id: 0,
            log_ratio: f64::INFINITY,
            fast_path: FastPath::None,
        }];
        let headers: Vec<&str> = vec!["read_inf"];
        let ratio_name = "ratio";

        let output = format_log_ratio_output(&log_ratios, &headers, ratio_name);
        let output_str = String::from_utf8(output).unwrap();

        assert_eq!(output_str, "read_inf\tratio\tinf\tnone\n");
    }

    #[test]
    fn test_format_log_ratio_output_negative_infinity() {
        let log_ratios = vec![LogRatioResult {
            query_id: 0,
            log_ratio: f64::NEG_INFINITY,
            fast_path: FastPath::NumZero,
        }];
        let headers: Vec<&str> = vec!["read_neginf"];
        let ratio_name = "ratio";

        let output = format_log_ratio_output(&log_ratios, &headers, ratio_name);
        let output_str = String::from_utf8(output).unwrap();

        assert_eq!(output_str, "read_neginf\tratio\t-inf\tnum_zero\n");
    }

    #[test]
    fn test_format_log_ratio_output_mixed_fast_paths() {
        let log_ratios = vec![
            LogRatioResult {
                query_id: 0,
                log_ratio: f64::NEG_INFINITY,
                fast_path: FastPath::NumZero,
            },
            LogRatioResult {
                query_id: 1,
                log_ratio: f64::INFINITY,
                fast_path: FastPath::NumHigh,
            },
            LogRatioResult {
                query_id: 2,
                log_ratio: 0.3010,
                fast_path: FastPath::None,
            },
        ];
        let headers: Vec<&str> = vec!["read_0", "read_1", "read_2"];
        let ratio_name = "ratio";

        let output = format_log_ratio_output(&log_ratios, &headers, ratio_name);
        let output_str = String::from_utf8(output).unwrap();

        assert_eq!(
            output_str,
            "read_0\tratio\t-inf\tnum_zero\nread_1\tratio\tinf\tnum_high\nread_2\tratio\t0.3010\tnone\n"
        );
    }

    #[test]
    fn test_format_log_ratio_output_empty_results() {
        let log_ratios: Vec<LogRatioResult> = vec![];
        let headers: Vec<&str> = vec!["read_0"];
        let ratio_name = "ratio";

        let output = format_log_ratio_output(&log_ratios, &headers, ratio_name);
        assert!(output.is_empty());
    }
}
