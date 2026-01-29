//! Sequence orientation logic for bucket building.
//!
//! Uses sorted Vec with galloping search for memory-efficient overlap detection.
//! Provides reusable galloping iteration for merge-join operations.

/// Orientation of a sequence relative to the bucket baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    Forward,
    ReverseComplement,
}

/// Core galloping iteration over two sorted slices.
///
/// Calls `on_match(smaller_idx, larger_idx)` for each element in `smaller` found in `larger`.
/// Both slices must be sorted.
///
/// # Algorithm
/// For each element in the smaller array:
/// 1. Exponential probe: jump 1, 2, 4, 8... positions until we overshoot
/// 2. Binary search: search within [current_pos, current_pos + jump + 1)
/// 3. Advance position on match or miss (leveraging sorted order)
///
/// # Complexity
/// O(smaller.len() * log(larger.len() / smaller.len()))
#[inline]
pub fn gallop_for_each<F>(smaller: &[u64], larger: &[u64], mut on_match: F)
where
    F: FnMut(usize, usize),
{
    let mut larger_pos = 0usize;

    for (smaller_idx, &s_val) in smaller.iter().enumerate() {
        // Gallop: exponential search with overflow protection
        let mut jump = 1usize;
        while larger_pos + jump < larger.len() && larger[larger_pos + jump] < s_val {
            jump = jump.saturating_mul(2);
            if jump >= larger.len() {
                break; // No point probing beyond array
            }
        }

        // Binary search in bounded range [larger_pos, search_end)
        // Note: +1 because the match could be AT position larger_pos + jump
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

/// Merge sorted `source` into sorted `target` in-place, deduplicating.
///
/// Uses swap to avoid copying target's contents. The result is a sorted,
/// deduplicated Vec containing all unique elements from both inputs.
///
/// # Complexity
/// O(target.len() + source.len()) - merges into a new Vec, then swaps.
/// The swap is O(1) as it just exchanges Vec internals (ptr, len, capacity).
pub fn merge_sorted_into(target: &mut Vec<u64>, source: &[u64]) {
    if source.is_empty() {
        return;
    }
    if target.is_empty() {
        target.extend_from_slice(source);
        target.dedup();
        return;
    }

    // Merge into temp, then swap (O(1) pointer swap, not O(n) copy)
    let mut merged = Vec::with_capacity(target.len() + source.len());

    let mut i = 0;
    let mut j = 0;
    let mut last_pushed: Option<u64> = None;

    while i < target.len() && j < source.len() {
        match target[i].cmp(&source[j]) {
            std::cmp::Ordering::Less => {
                if last_pushed != Some(target[i]) {
                    merged.push(target[i]);
                    last_pushed = Some(target[i]);
                }
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                if last_pushed != Some(source[j]) {
                    merged.push(source[j]);
                    last_pushed = Some(source[j]);
                }
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                if last_pushed != Some(target[i]) {
                    merged.push(target[i]);
                    last_pushed = Some(target[i]);
                }
                i += 1;
                j += 1;
            }
        }
    }

    // Remaining elements from target
    for &v in &target[i..] {
        if last_pushed != Some(v) {
            merged.push(v);
            last_pushed = Some(v);
        }
    }

    // Remaining elements from source
    for &v in &source[j..] {
        if last_pushed != Some(v) {
            merged.push(v);
            last_pushed = Some(v);
        }
    }

    // O(1) swap - just exchanges Vec internals (ptr, len, capacity)
    std::mem::swap(target, &mut merged);
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

    // ===== Merge tests =====

    #[test]
    fn test_merge_sorted_into_basic() {
        let mut target = vec![1, 3, 5];
        merge_sorted_into(&mut target, &[2, 4, 6]);
        assert_eq!(target, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_merge_sorted_into_overlapping() {
        let mut target = vec![1, 2, 3, 4];
        merge_sorted_into(&mut target, &[3, 4, 5, 6]);
        assert_eq!(target, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_merge_sorted_into_source_after_target() {
        let mut target = vec![1, 2, 3];
        merge_sorted_into(&mut target, &[10, 20, 30]);
        assert_eq!(target, vec![1, 2, 3, 10, 20, 30]);
    }

    #[test]
    fn test_merge_sorted_into_with_duplicates() {
        let mut target = vec![1, 1, 2, 3];
        merge_sorted_into(&mut target, &[2, 3, 3, 4]);
        assert_eq!(target, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_merge_sorted_into_empty_target() {
        let mut target: Vec<u64> = vec![];
        merge_sorted_into(&mut target, &[1, 2, 3]);
        assert_eq!(target, vec![1, 2, 3]);
    }

    #[test]
    fn test_merge_sorted_into_empty_source() {
        let mut target = vec![1, 2, 3];
        merge_sorted_into(&mut target, &[]);
        assert_eq!(target, vec![1, 2, 3]);
    }

    #[test]
    fn test_merge_sorted_into_source_before_target() {
        let mut target = vec![10, 20, 30];
        merge_sorted_into(&mut target, &[1, 2, 3]);
        assert_eq!(target, vec![1, 2, 3, 10, 20, 30]);
    }

    #[test]
    fn test_merge_sorted_into_interleaved() {
        let mut target = vec![1, 5, 9];
        merge_sorted_into(&mut target, &[2, 6, 10]);
        assert_eq!(target, vec![1, 2, 5, 6, 9, 10]);
    }

    #[test]
    fn test_merge_sorted_into_identical() {
        let mut target = vec![1, 2, 3];
        merge_sorted_into(&mut target, &[1, 2, 3]);
        assert_eq!(target, vec![1, 2, 3]);
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
}
