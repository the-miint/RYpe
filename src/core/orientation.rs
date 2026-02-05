//! Sequence orientation logic for bucket building.
//!
//! Uses sorted Vec with galloping search for memory-efficient overlap detection.
//! Provides reusable galloping iteration for merge-join operations.

/// Default number of minimizers to check for orientation.
/// Checking the first N sorted minimizers provides a sample for
/// orientation decisions without iterating the full bucket.
pub const ORIENTATION_FIRST_N: usize = 10_000;

/// Orientation of a sequence relative to the bucket baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    Forward,
    ReverseComplement,
}

/// Core galloping iteration over two sorted slices.
///
/// Calls `on_match(smaller_idx, larger_idx)` for each element in `smaller` found in `larger`.
/// Both slices must be sorted in ascending order.
///
/// # Algorithm
/// For each element in the smaller array:
/// 1. Exponential probe: jump 1, 2, 4, 8... positions until we overshoot
/// 2. Binary search: search within [current_pos, current_pos + jump + 1)
/// 3. Advance position on match or miss (leveraging sorted order)
///
/// # Complexity
/// O(smaller.len() * log(larger.len() / smaller.len()))
///
/// # Index Guarantees
/// The indices passed to `on_match` are guaranteed to be valid:
/// - `smaller_idx < smaller.len()`
/// - `larger_idx < larger.len()`
#[inline]
pub(crate) fn gallop_for_each<F>(smaller: &[u64], larger: &[u64], mut on_match: F)
where
    F: FnMut(usize, usize),
{
    let mut larger_pos = 0usize;

    for (smaller_idx, &s_val) in smaller.iter().enumerate() {
        // Gallop: exponential probe until we overshoot or find a value >= s_val
        let mut jump = 1usize;
        while larger_pos + jump < larger.len() && larger[larger_pos + jump] < s_val {
            jump *= 2;
        }

        // Binary search in bounded range [larger_pos, search_end)
        let search_end = (larger_pos + jump + 1).min(larger.len());
        match larger[larger_pos..search_end].binary_search(&s_val) {
            Ok(rel_idx) => {
                let larger_idx = larger_pos + rel_idx;
                on_match(smaller_idx, larger_idx);
                larger_pos = larger_idx + 1;
            }
            Err(rel_idx) => {
                larger_pos += rel_idx;
            }
        }
    }
}

/// Count how many elements in `needles` exist in `haystack`. Both must be sorted.
#[inline]
pub fn count_matches_gallop(haystack: &[u64], needles: &[u64]) -> usize {
    if haystack.is_empty() || needles.is_empty() {
        return 0;
    }
    let mut count = 0;
    gallop_for_each(needles, haystack, |_, _| count += 1);
    count
}

/// Compute overlap = |intersection| / |needles|. Both slices must be sorted.
pub fn compute_overlap(bucket: &[u64], seq_minimizers: &[u64]) -> f64 {
    if seq_minimizers.is_empty() {
        return 0.0;
    }
    let matches = count_matches_gallop(bucket, seq_minimizers);
    matches as f64 / seq_minimizers.len() as f64
}

/// Choose orientation with higher overlap. Ties favor Forward.
///
/// Returns the chosen orientation and its overlap score.
pub fn choose_orientation(
    bucket: &[u64],
    fwd_minimizers: &[u64],
    rc_minimizers: &[u64],
) -> (Orientation, f64) {
    let fwd_overlap = compute_overlap(bucket, fwd_minimizers);
    let rc_overlap = compute_overlap(bucket, rc_minimizers);

    if rc_overlap > fwd_overlap {
        (Orientation::ReverseComplement, rc_overlap)
    } else {
        (Orientation::Forward, fwd_overlap)
    }
}

/// Choose orientation by checking first N minimizers of each strand.
///
/// The first N sorted minimizers from each strand are checked against the bucket.
/// This provides a fast sample-based orientation decision without needing to
/// iterate through the entire bucket or sequences.
///
/// # Arguments
/// * `bucket` - The current bucket minimizers (sorted)
/// * `fwd_minimizers` - Forward strand minimizers (sorted)
/// * `rc_minimizers` - Reverse complement minimizers (sorted)
///
/// # Returns
/// The chosen orientation and overlap score (matches / N).
///
/// # Complexity
/// O(N × log(bucket_size)) per strand, where N = ORIENTATION_FIRST_N.
pub fn choose_orientation_sampled(
    bucket: &[u64],
    fwd_minimizers: &[u64],
    rc_minimizers: &[u64],
) -> (Orientation, f64) {
    if bucket.is_empty() {
        return (Orientation::Forward, 0.0);
    }

    // Use the smaller of N and the available minimizers
    let n = ORIENTATION_FIRST_N
        .min(fwd_minimizers.len())
        .min(rc_minimizers.len());

    if n == 0 {
        return (Orientation::Forward, 0.0);
    }

    let fwd_matches = count_matches_gallop(bucket, &fwd_minimizers[..n]);
    let rc_matches = count_matches_gallop(bucket, &rc_minimizers[..n]);

    if rc_matches > fwd_matches {
        (Orientation::ReverseComplement, rc_matches as f64 / n as f64)
    } else {
        (Orientation::Forward, fwd_matches as f64 / n as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== Galloping tests =====

    #[test]
    fn test_gallop_for_each_basic() {
        let mut matches = Vec::new();
        gallop_for_each(&[2, 4, 6], &[1, 2, 3, 4, 5, 6, 7], |si, li| {
            matches.push((si, li));
        });
        assert_eq!(matches, vec![(0, 1), (1, 3), (2, 5)]);
    }

    #[test]
    fn test_gallop_for_each_no_matches() {
        let mut matches = Vec::new();
        gallop_for_each(&[10, 20, 30], &[1, 2, 3, 4, 5], |si, li| {
            matches.push((si, li));
        });
        assert!(matches.is_empty());
    }

    #[test]
    fn test_gallop_for_each_empty_inputs() {
        let mut matches = Vec::new();
        gallop_for_each(&[], &[1, 2, 3], |si, li| {
            matches.push((si, li));
        });
        assert!(matches.is_empty());

        gallop_for_each(&[1, 2, 3], &[], |si, li| {
            matches.push((si, li));
        });
        assert!(matches.is_empty());
    }

    #[test]
    fn test_count_matches_gallop_identical() {
        assert_eq!(count_matches_gallop(&[1, 2, 3, 4, 5], &[1, 2, 3, 4, 5]), 5);
    }

    #[test]
    fn test_count_matches_gallop_disjoint() {
        assert_eq!(count_matches_gallop(&[1, 2, 3], &[4, 5, 6]), 0);
    }

    #[test]
    fn test_count_matches_gallop_partial() {
        assert_eq!(count_matches_gallop(&[1, 2, 3, 4], &[3, 4, 5, 6]), 2);
    }

    #[test]
    fn test_count_matches_gallop_empty() {
        assert_eq!(count_matches_gallop(&[], &[1, 2, 3]), 0);
        assert_eq!(count_matches_gallop(&[1, 2, 3], &[]), 0);
    }

    #[test]
    fn test_count_matches_gallop_large_haystack() {
        // Ensure galloping works correctly with large size differences
        let haystack: Vec<u64> = (0..1000).collect();
        let needles = vec![100, 500, 999];
        assert_eq!(count_matches_gallop(&haystack, &needles), 3);
    }

    #[test]
    fn test_count_matches_gallop_boundary_case() {
        // Test the boundary case that can trigger off-by-one bugs
        // With needles = [10] and haystack = [0, 10, 20, ...]:
        // - gallop: jump=1, haystack[1]=10, 10 < 10? NO → exit
        // - search_end must include position 1 for the match to be found
        let haystack: Vec<u64> = (0..100).map(|i| i * 10).collect();
        let needles = vec![10];
        assert_eq!(count_matches_gallop(&haystack, &needles), 1);
    }

    // ===== Overlap/orientation tests =====

    #[test]
    fn test_compute_overlap_identical() {
        assert!((compute_overlap(&[1, 2, 3, 4, 5], &[1, 2, 3, 4, 5]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_overlap_partial() {
        assert!((compute_overlap(&[1, 2, 3, 4], &[3, 4, 5, 6]) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_compute_overlap_empty_bucket() {
        // Empty bucket, non-empty seq: 0 matches / 3 = 0
        assert!((compute_overlap(&[], &[1, 2, 3]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_overlap_empty_seq() {
        // Empty seq: 0.0 by definition
        assert!((compute_overlap(&[1, 2, 3], &[]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_overlap_no_overlap() {
        assert!((compute_overlap(&[1, 2, 3], &[4, 5, 6]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_choose_orientation_forward_wins() {
        let bucket = vec![1, 2, 3, 4, 5];
        let fwd = vec![1, 2, 3]; // 3/3 = 100%
        let rc = vec![6, 7, 8]; // 0/3 = 0%
        let (orientation, overlap) = choose_orientation(&bucket, &fwd, &rc);
        assert_eq!(orientation, Orientation::Forward);
        assert!((overlap - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_choose_orientation_rc_wins() {
        let bucket = vec![10, 20, 30];
        let fwd = vec![1, 2, 3]; // 0/3 = 0%
        let rc = vec![10, 20, 30]; // 3/3 = 100%
        let (orientation, overlap) = choose_orientation(&bucket, &fwd, &rc);
        assert_eq!(orientation, Orientation::ReverseComplement);
        assert!((overlap - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_choose_orientation_tie_favors_forward() {
        let bucket = vec![1, 2, 10, 20];
        let fwd = vec![1, 2, 3, 4]; // 2/4 = 50%
        let rc = vec![10, 20, 5, 6]; // 2/4 = 50%
        let (orientation, overlap) = choose_orientation(&bucket, &fwd, &rc);
        assert_eq!(orientation, Orientation::Forward);
        assert!((overlap - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_choose_orientation_empty_bucket() {
        let bucket: Vec<u64> = vec![];
        let fwd = vec![1, 2, 3];
        let rc = vec![4, 5, 6];
        // Both have 0% overlap, forward wins tie
        let (orientation, overlap) = choose_orientation(&bucket, &fwd, &rc);
        assert_eq!(orientation, Orientation::Forward);
        assert!((overlap - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_choose_orientation_empty_minimizers() {
        // All inputs empty: should return Forward with 0.0 overlap
        let bucket: Vec<u64> = vec![];
        let fwd: Vec<u64> = vec![];
        let rc: Vec<u64> = vec![];
        let (orientation, overlap) = choose_orientation(&bucket, &fwd, &rc);
        assert_eq!(orientation, Orientation::Forward);
        assert_eq!(overlap, 0.0);
    }

    // ===== Sampled orientation tests =====

    #[test]
    fn test_choose_orientation_sampled_small_bucket() {
        // Small bucket should use full comparison
        let bucket = vec![1, 2, 3, 4, 5];
        let fwd = vec![1, 2, 3]; // All in bucket
        let rc = vec![6, 7, 8]; // None in bucket

        let (orientation, _) = choose_orientation_sampled(&bucket, &fwd, &rc);
        assert_eq!(orientation, Orientation::Forward);
    }

    #[test]
    fn test_choose_orientation_sampled_rc_wins() {
        // Test RC wins with small bucket
        let bucket = vec![10, 20, 30];
        let fwd = vec![1, 2, 3]; // None in bucket
        let rc = vec![10, 20, 30]; // All in bucket

        let (orientation, _) = choose_orientation_sampled(&bucket, &fwd, &rc);
        assert_eq!(orientation, Orientation::ReverseComplement);
    }

    #[test]
    fn test_choose_orientation_sampled_large_bucket() {
        // Large bucket triggers sampling
        let bucket: Vec<u64> = (0..200_000).collect();
        let fwd: Vec<u64> = (0..1000).collect(); // First 1000, all in bucket
        let rc: Vec<u64> = (500_000..501_000).collect(); // None in bucket

        let (orientation, overlap) = choose_orientation_sampled(&bucket, &fwd, &rc);
        assert_eq!(orientation, Orientation::Forward);
        assert!(overlap > 0.0);
    }

    #[test]
    fn test_choose_orientation_sampled_agrees_with_full_clear_cut() {
        // For clear-cut cases, sampled should agree with full
        let bucket: Vec<u64> = (0..200_000).collect();
        let fwd: Vec<u64> = (0..5000).collect(); // Clearly in bucket
        let rc: Vec<u64> = (1_000_000..1_005_000).collect(); // Clearly not in bucket

        let (sampled_orient, _) = choose_orientation_sampled(&bucket, &fwd, &rc);
        let (full_orient, _) = choose_orientation(&bucket, &fwd, &rc);

        assert_eq!(sampled_orient, full_orient);
    }

    #[test]
    fn test_choose_orientation_sampled_tail_heavy_sequences() {
        // Test case where sequence minimizers are in the tail of the bucket range.
        // The first-N approach checks first N sorted minimizers of the sequence
        // against the bucket, which handles this correctly.
        let bucket: Vec<u64> = (0..150_000).collect();
        let fwd: Vec<u64> = (120_000..130_000).collect(); // Matches tail of bucket
        let rc: Vec<u64> = (500_000..510_000).collect(); // Matches nothing

        let (sampled_orient, _) = choose_orientation_sampled(&bucket, &fwd, &rc);
        let (full_orient, _) = choose_orientation(&bucket, &fwd, &rc);

        assert_eq!(
            sampled_orient, full_orient,
            "First-N sampling should correctly orient tail-heavy sequences"
        );
    }

    #[test]
    fn test_choose_orientation_sampled_boundary_sizes() {
        // Test behavior at ORIENTATION_FIRST_N boundary
        let bucket: Vec<u64> = (0..100_000).collect();

        // Sequences shorter than ORIENTATION_FIRST_N - all minimizers used
        let fwd_short: Vec<u64> = (0..1000u64).collect();
        let rc_short: Vec<u64> = (500_000..501_000u64).collect();

        let (orient_short, _) = choose_orientation_sampled(&bucket, &fwd_short, &rc_short);
        assert_eq!(orient_short, Orientation::Forward);

        // Sequences longer than ORIENTATION_FIRST_N - only first N used
        let fwd_long: Vec<u64> = (0..20_000u64).collect(); // First 10K in bucket
        let rc_long: Vec<u64> = (500_000..520_000u64).collect(); // None in bucket

        let (orient_long, _) = choose_orientation_sampled(&bucket, &fwd_long, &rc_long);
        assert_eq!(orient_long, Orientation::Forward);
    }

    #[test]
    fn test_choose_orientation_sampled_close_overlap() {
        // Test with overlaps that are close but not equal
        // This validates that sampling doesn't flip close decisions incorrectly
        let bucket: Vec<u64> = (0..200_000).collect();
        // fwd has 60% of its elements in bucket
        let fwd: Vec<u64> = (0..10_000).chain(300_000..306_000).collect();
        // rc has 40% of its elements in bucket
        let rc: Vec<u64> = (0..6_000).chain(300_000..310_000).collect();

        // Both methods should choose fwd since it has higher bucket overlap
        let (sampled_orient, _) = choose_orientation_sampled(&bucket, &fwd, &rc);
        let (full_orient, _) = choose_orientation(&bucket, &fwd, &rc);

        // Note: We expect agreement for cases with meaningful overlap differences
        // Very close cases (e.g., 50.1% vs 49.9%) might differ, which is acceptable
        assert_eq!(sampled_orient, full_orient);
    }

    #[test]
    fn test_choose_orientation_sampled_empty_sequences() {
        let bucket: Vec<u64> = (0..200_000).collect();
        let fwd: Vec<u64> = vec![];
        let rc: Vec<u64> = vec![];

        let (orientation, overlap) = choose_orientation_sampled(&bucket, &fwd, &rc);
        // Both empty, should default to Forward with 0 overlap
        assert_eq!(orientation, Orientation::Forward);
        assert_eq!(overlap, 0.0);
    }

    // ===== First-N orientation tests (TDD Phase 1) =====
    // These tests define expected behavior for the first-N approach.
    // They may pass with the current implementation - that's okay for behavior tests.

    #[test]
    fn test_choose_orientation_first_n_forward_wins() {
        let bucket = vec![1, 2, 3, 4, 5, 10, 20, 30];
        let fwd = vec![1, 2, 3, 100, 200]; // 3 of first 5 match
        let rc = vec![500, 600, 700, 800, 900]; // 0 match
        let (orient, _) = choose_orientation_sampled(&bucket, &fwd, &rc);
        assert_eq!(orient, Orientation::Forward);
    }

    #[test]
    fn test_choose_orientation_first_n_rc_wins() {
        let bucket = vec![10, 20, 30, 40, 50];
        let fwd = vec![1, 2, 3, 4, 5]; // 0 match
        let rc = vec![10, 20, 30, 60, 70]; // 3 match
        let (orient, _) = choose_orientation_sampled(&bucket, &fwd, &rc);
        assert_eq!(orient, Orientation::ReverseComplement);
    }

    #[test]
    fn test_choose_orientation_first_n_tie_favors_forward() {
        let bucket = vec![1, 2, 10, 20];
        let fwd = vec![1, 2, 100, 200]; // 2 match
        let rc = vec![10, 20, 100, 200]; // 2 match
        let (orient, _) = choose_orientation_sampled(&bucket, &fwd, &rc);
        assert_eq!(orient, Orientation::Forward);
    }

    #[test]
    fn test_choose_orientation_first_n_empty() {
        let (orient, overlap) = choose_orientation_sampled(&[], &[], &[]);
        assert_eq!(orient, Orientation::Forward);
        assert_eq!(overlap, 0.0);
    }

    #[test]
    fn test_choose_orientation_first_n_short_sequences() {
        // Bucket has values 0..10000, fwd minimizers are in bucket, rc are not
        let bucket: Vec<u64> = (0..10000).collect();
        let fwd = vec![5, 10, 15]; // All in bucket
        let rc = vec![50000, 50001, 50002]; // None in bucket
        let (orient, _) = choose_orientation_sampled(&bucket, &fwd, &rc);
        assert_eq!(orient, Orientation::Forward);
    }

    #[test]
    fn test_choose_orientation_first_n_large_inputs() {
        // Large bucket and sequences - first N approach should handle efficiently
        let bucket: Vec<u64> = (0..1_000_000).collect();
        // fwd: first 5000 in bucket, rest not
        let fwd: Vec<u64> = (0..5000).chain(2_000_000..2_100_000).collect();
        // rc: none in bucket
        let rc: Vec<u64> = (3_000_000..3_100_000).collect();

        let (orient, _) = choose_orientation_sampled(&bucket, &fwd, &rc);
        assert_eq!(orient, Orientation::Forward);
    }

    #[test]
    fn test_choose_orientation_first_n_random_distributions() {
        // Validates behavior on pseudo-random data
        // Using deterministic patterns for reproducibility
        for seed in 0..10u64 {
            let bucket: Vec<u64> = (0u64..50_000)
                .filter(|&x| (x.wrapping_mul(31) ^ seed) % 2 == 0)
                .collect();

            // fwd has more matches than rc
            let fwd: Vec<u64> = (0u64..1000)
                .map(|x| x.wrapping_mul(2)) // Even numbers, high overlap with bucket
                .collect();
            let rc: Vec<u64> = (0u64..1000)
                .map(|x| x.wrapping_mul(2).wrapping_add(100_000)) // Far outside bucket
                .collect();

            let (orient, _) = choose_orientation_sampled(&bucket, &fwd, &rc);
            assert_eq!(
                orient,
                Orientation::Forward,
                "seed {} should choose Forward",
                seed
            );
        }
    }
}
