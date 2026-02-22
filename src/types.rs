//! Core types used throughout the rype library.

use std::collections::HashMap;

/// ID (i64), Sequence Reference, Optional Pair Sequence Reference
pub type QueryRecord<'a> = (i64, &'a [u8], Option<&'a [u8]>);

/// Lightweight metadata-only view of an Index (without minimizer data)
#[derive(Debug, Clone)]
pub struct IndexMetadata {
    pub k: usize,
    pub w: usize,
    pub salt: u64,
    pub bucket_names: HashMap<u32, String>,
    pub bucket_sources: HashMap<u32, Vec<String>>,
    pub bucket_minimizer_counts: HashMap<u32, usize>,
    /// Number of entries in the largest inverted index shard (0 if no shards).
    /// Used for estimating shard loading memory during adaptive batch sizing.
    pub largest_shard_entries: u64,
    /// Per-bucket file-level sequence length statistics.
    /// `None` for indices built before this feature was added.
    pub bucket_file_stats: Option<HashMap<u32, BucketFileStats>>,
}

/// Per-bucket statistics about total sequence lengths of source files.
///
/// Each source file contributes one "file length" — the sum of all sequence
/// bases in that file.  These stats summarize the distribution of file lengths
/// within a single bucket.
#[derive(Debug, Clone, PartialEq)]
pub struct BucketFileStats {
    /// Mean of per-file total sequence lengths.
    pub mean: f64,
    /// Median of per-file total sequence lengths.
    pub median: f64,
    /// Population standard deviation of per-file total sequence lengths.
    pub stdev: f64,
    /// Minimum per-file total sequence length.
    pub min: f64,
    /// Maximum per-file total sequence length.
    pub max: f64,
}

impl BucketFileStats {
    /// Compute stats from a slice of per-file total sequence lengths.
    ///
    /// Returns `None` if the slice is empty.
    pub fn from_file_lengths(lengths: &[u64]) -> Option<Self> {
        if lengths.is_empty() {
            return None;
        }

        let n = lengths.len() as f64;
        let sum: f64 = lengths.iter().map(|&v| v as f64).sum();
        let mean = sum / n;

        // Population standard deviation (divide by N, not N-1)
        let variance = lengths
            .iter()
            .map(|&v| {
                let diff = v as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / n;
        let stdev = variance.sqrt();

        // Median: sort a copy
        let mut sorted = lengths.to_vec();
        sorted.sort_unstable();
        let median = if sorted.len() % 2 == 0 {
            let mid = sorted.len() / 2;
            (sorted[mid - 1] as f64 + sorted[mid] as f64) / 2.0
        } else {
            sorted[sorted.len() / 2] as f64
        };

        let min = *lengths.iter().min().unwrap() as f64;
        let max = *lengths.iter().max().unwrap() as f64;

        Some(BucketFileStats {
            mean,
            median,
            stdev,
            min,
            max,
        })
    }
}

/// Query ID, Bucket ID, Score
#[derive(Debug, Clone, PartialEq)]
pub struct HitResult {
    pub query_id: i64,
    pub bucket_id: u32,
    pub score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_file_lengths_single() {
        let stats = BucketFileStats::from_file_lengths(&[1000]).unwrap();
        assert_eq!(stats.mean, 1000.0);
        assert_eq!(stats.median, 1000.0);
        assert_eq!(stats.stdev, 0.0);
        assert_eq!(stats.min, 1000.0);
        assert_eq!(stats.max, 1000.0);
    }

    #[test]
    fn test_from_file_lengths_multiple_odd() {
        // 3 files: 100, 200, 600
        // mean = 300, median = 200
        // variance = ((100-300)^2 + (200-300)^2 + (600-300)^2) / 3
        //          = (40000 + 10000 + 90000) / 3 = 46666.667
        // stdev = sqrt(46666.667) ≈ 216.025
        let stats = BucketFileStats::from_file_lengths(&[100, 200, 600]).unwrap();
        assert!((stats.mean - 300.0).abs() < 1e-9);
        assert!((stats.median - 200.0).abs() < 1e-9);
        let expected_stdev = (46666.666666666666_f64).sqrt();
        assert!((stats.stdev - expected_stdev).abs() < 1e-6);
        assert_eq!(stats.min, 100.0);
        assert_eq!(stats.max, 600.0);
    }

    #[test]
    fn test_from_file_lengths_multiple_even() {
        // 4 files: 10, 20, 30, 40
        // mean = 25, median = (20+30)/2 = 25
        // variance = ((10-25)^2 + (20-25)^2 + (30-25)^2 + (40-25)^2)/4
        //          = (225 + 25 + 25 + 225) / 4 = 125
        // stdev = sqrt(125) ≈ 11.18
        let stats = BucketFileStats::from_file_lengths(&[10, 20, 30, 40]).unwrap();
        assert!((stats.mean - 25.0).abs() < 1e-9);
        assert!((stats.median - 25.0).abs() < 1e-9);
        assert!((stats.stdev - 11.180339887).abs() < 1e-6);
        assert_eq!(stats.min, 10.0);
        assert_eq!(stats.max, 40.0);
    }

    #[test]
    fn test_from_file_lengths_empty() {
        assert!(BucketFileStats::from_file_lengths(&[]).is_none());
    }

    #[test]
    fn test_from_file_lengths_identical() {
        let stats = BucketFileStats::from_file_lengths(&[500, 500, 500]).unwrap();
        assert_eq!(stats.mean, 500.0);
        assert_eq!(stats.median, 500.0);
        assert_eq!(stats.stdev, 0.0);
        assert_eq!(stats.min, 500.0);
        assert_eq!(stats.max, 500.0);
    }
}
