//! Output formatting utilities for classification results.

use std::collections::HashMap;
use std::io::Write;

/// Format classification results as tab-separated output.
///
/// Each result is formatted as: `header\tbucket_name\tscore`
///
/// # Arguments
/// * `results` - Classification results to format
/// * `headers` - Sequence headers indexed by query_id
/// * `bucket_names` - Map from bucket_id to bucket name
///
/// # Returns
/// A `Vec<u8>` containing the formatted output
pub fn format_classification_results<S: AsRef<str>>(
    results: &[rype::HitResult],
    headers: &[S],
    bucket_names: &HashMap<u32, String>,
) -> Vec<u8> {
    let mut output = Vec::with_capacity(1024);
    for res in results {
        let header = headers[res.query_id as usize].as_ref();
        let bucket_name = bucket_names
            .get(&res.bucket_id)
            .map(|s| s.as_str())
            .unwrap_or("unknown");
        writeln!(output, "{}\t{}\t{:.4}", header, bucket_name, res.score).unwrap();
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_classification_results_basic() {
        let results = vec![
            rype::HitResult {
                query_id: 0,
                bucket_id: 1,
                score: 0.9567,
            },
            rype::HitResult {
                query_id: 1,
                bucket_id: 2,
                score: 0.1234,
            },
        ];
        let headers = vec!["read_A", "read_B"];
        let mut bucket_names = HashMap::new();
        bucket_names.insert(1, "bucket_one".to_string());
        bucket_names.insert(2, "bucket_two".to_string());

        let output = format_classification_results(&results, &headers, &bucket_names);
        let output_str = String::from_utf8(output).unwrap();

        assert_eq!(
            output_str,
            "read_A\tbucket_one\t0.9567\nread_B\tbucket_two\t0.1234\n"
        );
    }

    #[test]
    fn test_format_classification_results_with_string_headers() {
        let results = vec![rype::HitResult {
            query_id: 0,
            bucket_id: 1,
            score: 0.5,
        }];
        let headers: Vec<String> = vec!["seq1".to_string()];
        let mut bucket_names = HashMap::new();
        bucket_names.insert(1, "target".to_string());

        let output = format_classification_results(&results, &headers, &bucket_names);
        let output_str = String::from_utf8(output).unwrap();

        assert_eq!(output_str, "seq1\ttarget\t0.5000\n");
    }

    #[test]
    fn test_format_classification_results_unknown_bucket() {
        let results = vec![rype::HitResult {
            query_id: 0,
            bucket_id: 999,
            score: 0.75,
        }];
        let headers = vec!["read1"];
        let bucket_names = HashMap::new(); // Empty - no bucket names

        let output = format_classification_results(&results, &headers, &bucket_names);
        let output_str = String::from_utf8(output).unwrap();

        assert_eq!(output_str, "read1\tunknown\t0.7500\n");
    }

    #[test]
    fn test_format_classification_results_empty() {
        let results: Vec<rype::HitResult> = vec![];
        let headers: Vec<&str> = vec![];
        let bucket_names = HashMap::new();

        let output = format_classification_results(&results, &headers, &bucket_names);

        assert!(output.is_empty());
    }

    #[test]
    fn test_format_classification_results_score_precision() {
        let results = vec![rype::HitResult {
            query_id: 0,
            bucket_id: 1,
            score: 0.123456789,
        }];
        let headers = vec!["test"];
        let mut bucket_names = HashMap::new();
        bucket_names.insert(1, "bucket".to_string());

        let output = format_classification_results(&results, &headers, &bucket_names);
        let output_str = String::from_utf8(output).unwrap();

        // Score should be truncated to 4 decimal places
        assert_eq!(output_str, "test\tbucket\t0.1235\n");
    }
}
