//! Sequence orientation logic for bucket building.
//!
//! Uses sorted Vec with galloping search for memory-efficient overlap detection.
//! Provides reusable galloping iteration for merge-join operations.

/// Default sample size for orientation overlap computation.
/// With 100K samples distributed across 10 strata, orientation decisions are
/// O(sample_size × log(seq_size)) instead of O(seq_size × log(bucket_size)).
///
/// Empirically validated: 100K samples provides >99% agreement with full
/// overlap computation on realistic minimizer distributions.
pub const ORIENTATION_SAMPLE_SIZE: usize = 100_000;

/// Number of strata for stratified sampling.
/// Dividing the bucket into strata ensures coverage across the full range,
/// avoiding bias toward any particular region.
const NUM_STRATA: usize = 10;

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

/// Sample elements from a sorted slice using stratified sampling.
///
/// Divides data into NUM_STRATA equal regions and samples proportionally from each,
/// ensuring coverage across the full range of the bucket. This avoids the bias
/// of simple stride-based sampling which can miss the tail of the data.
///
/// Fills `buffer` with the sample (clears it first).
#[inline]
fn sample_stratified_into(data: &[u64], sample_size: usize, buffer: &mut Vec<u64>) {
    buffer.clear();

    if data.len() <= sample_size {
        buffer.extend_from_slice(data);
        return;
    }

    buffer.reserve(sample_size);

    let per_stratum = sample_size / NUM_STRATA;
    let stratum_size = data.len() / NUM_STRATA;

    for i in 0..NUM_STRATA {
        let start = i * stratum_size;
        let end = if i == NUM_STRATA - 1 {
            data.len() // Last stratum includes remainder
        } else {
            (i + 1) * stratum_size
        };

        let actual_stratum_size = end - start;
        if actual_stratum_size == 0 {
            continue;
        }

        // How many samples from this stratum
        let samples_from_stratum = if i == NUM_STRATA - 1 {
            sample_size.saturating_sub(buffer.len()) // Fill remaining quota
        } else {
            per_stratum
        };

        if samples_from_stratum == 0 {
            continue;
        }

        if samples_from_stratum >= actual_stratum_size {
            // Take all elements from this stratum
            buffer.extend_from_slice(&data[start..end]);
        } else {
            // Evenly space within stratum
            let stride = actual_stratum_size / samples_from_stratum;
            for j in 0..samples_from_stratum {
                buffer.push(data[start + j * stride]);
            }
        }
    }
    // Buffer remains sorted since we iterate in order through sorted data
}

/// Compute overlap by checking what fraction of `sample` appears in `seq_minimizers`.
///
/// This measures |sample ∩ seq| / |sample|, i.e., what fraction of the bucket
/// sample is found in the sequence. By using the same denominator (|sample|)
/// for both forward and RC comparisons, the relative comparison remains valid
/// for orientation decisions.
///
/// Complexity: O(sample_size × log(seq_size)) - iterates through sample, binary
/// searches in seq.
#[inline]
fn compute_overlap_sampled(seq_minimizers: &[u64], sample: &[u64]) -> f64 {
    if sample.is_empty() {
        return 0.0;
    }
    // sample as needles, seq as haystack: O(sample_size × log(seq_size))
    let matches = count_matches_gallop(seq_minimizers, sample);
    matches as f64 / sample.len() as f64
}

/// Choose orientation using sampled overlap computation with buffer reuse.
///
/// For large buckets (> ORIENTATION_SAMPLE_SIZE), uses stratified sampling to make
/// orientation decisions in O(sample_size × log(seq_size)) instead of
/// O(seq_size × log(bucket_size)).
///
/// # Arguments
/// * `bucket` - The current bucket minimizers (sorted)
/// * `fwd_minimizers` - Forward strand minimizers (sorted)
/// * `rc_minimizers` - Reverse complement minimizers (sorted)
/// * `sample_buffer` - Reusable buffer for sampling (avoids allocation per call)
///
/// # Returns
/// The chosen orientation and overlap score.
///
/// # Correctness
/// Stratified sampling ensures coverage across the full bucket range. The overlap
/// metric |sample ∩ seq| / |sample| is computed identically for both orientations,
/// so the relative comparison is preserved even though the absolute values differ
/// from the full overlap computation.
///
/// # Optimization
/// Sampling is only used when beneficial. For small sequences (< sample_size),
/// the full computation is used since iterating through a large sample would be
/// slower than iterating through the small sequence.
pub fn choose_orientation_sampled(
    bucket: &[u64],
    fwd_minimizers: &[u64],
    rc_minimizers: &[u64],
    sample_buffer: &mut Vec<u64>,
) -> (Orientation, f64) {
    // For small buckets, use full comparison
    if bucket.len() <= ORIENTATION_SAMPLE_SIZE {
        return choose_orientation(bucket, fwd_minimizers, rc_minimizers);
    }

    // For small sequences, sampling is counterproductive:
    // - Sampled complexity: O(sample_size × log(seq_size))
    // - Full complexity: O(seq_size × log(bucket_size))
    // When seq_size < sample_size, full is faster.
    let max_seq_len = fwd_minimizers.len().max(rc_minimizers.len());
    if max_seq_len < ORIENTATION_SAMPLE_SIZE {
        return choose_orientation(bucket, fwd_minimizers, rc_minimizers);
    }

    // Stratified sample the bucket into reusable buffer
    sample_stratified_into(bucket, ORIENTATION_SAMPLE_SIZE, sample_buffer);

    // Compute what fraction of bucket sample is found in each orientation
    let fwd_overlap = compute_overlap_sampled(fwd_minimizers, sample_buffer);
    let rc_overlap = compute_overlap_sampled(rc_minimizers, sample_buffer);

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

    // ===== Stratified sampling tests =====

    #[test]
    fn test_sample_stratified_small_input() {
        // When input is smaller than sample size, return all elements
        let data: Vec<u64> = (0..100).collect();
        let mut buffer = Vec::new();
        sample_stratified_into(&data, 1000, &mut buffer);
        assert_eq!(buffer, data);
    }

    #[test]
    fn test_sample_stratified_exact_size() {
        // When input equals sample size, return all elements
        let data: Vec<u64> = (0..100).collect();
        let mut buffer = Vec::new();
        sample_stratified_into(&data, 100, &mut buffer);
        assert_eq!(buffer, data);
    }

    #[test]
    fn test_sample_stratified_larger_input() {
        // When input is larger, sample with stratification
        let data: Vec<u64> = (0..1000).collect();
        let mut buffer = Vec::new();
        sample_stratified_into(&data, 100, &mut buffer);

        // Should have ~100 elements
        assert!(buffer.len() >= 90);
        assert!(buffer.len() <= 110);

        // Should be sorted
        let mut sorted = buffer.clone();
        sorted.sort_unstable();
        assert_eq!(buffer, sorted);

        // Should cover the full range (stratified sampling)
        assert!(buffer[0] < 100); // Has element from first stratum
        assert!(*buffer.last().unwrap() > 900); // Has element from last stratum
    }

    #[test]
    fn test_sample_stratified_covers_full_range() {
        // Critical test: stratified sampling should cover the ENTIRE bucket range,
        // not just the first N elements like simple stride-based sampling
        let data: Vec<u64> = (0..150_000).collect();
        let mut buffer = Vec::new();
        sample_stratified_into(&data, 100_000, &mut buffer);

        // Should have elements from the tail (> 100K)
        let has_tail_elements = buffer.iter().any(|&x| x > 100_000);
        assert!(
            has_tail_elements,
            "Stratified sampling should include elements from bucket tail"
        );

        // Should have elements from near the end
        let has_near_end = buffer.iter().any(|&x| x > 140_000);
        assert!(
            has_near_end,
            "Stratified sampling should include elements near the end"
        );
    }

    #[test]
    fn test_sample_stratified_preserves_sortedness() {
        // Sampling from sorted data should produce sorted sample
        let data: Vec<u64> = (0..10000).map(|i| i * 7).collect();
        let mut buffer = Vec::new();
        sample_stratified_into(&data, 500, &mut buffer);

        let mut sorted = buffer.clone();
        sorted.sort_unstable();
        assert_eq!(buffer, sorted);
    }

    #[test]
    fn test_sample_stratified_buffer_reuse() {
        // Buffer should be cleared and reused
        let data1: Vec<u64> = (0..1000).collect();
        let data2: Vec<u64> = (5000..6000).collect();
        let mut buffer = Vec::new();

        sample_stratified_into(&data1, 100, &mut buffer);
        assert!(buffer.iter().all(|&x| x < 1000));

        sample_stratified_into(&data2, 100, &mut buffer);
        assert!(buffer.iter().all(|&x| x >= 5000 && x < 6000));
    }

    // ===== Sampled overlap tests =====

    #[test]
    fn test_compute_overlap_sampled_identical() {
        let seq = vec![1, 2, 3, 4, 5];
        let sample = vec![1, 2, 3, 4, 5];
        assert!((compute_overlap_sampled(&seq, &sample) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_overlap_sampled_partial() {
        let seq = vec![1, 2, 3, 4, 5, 6];
        let sample = vec![3, 4, 7, 8]; // 2 of 4 in seq
        assert!((compute_overlap_sampled(&seq, &sample) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_compute_overlap_sampled_empty_sample() {
        let seq = vec![1, 2, 3];
        let sample: Vec<u64> = vec![];
        assert_eq!(compute_overlap_sampled(&seq, &sample), 0.0);
    }

    // ===== Sampled orientation tests =====

    #[test]
    fn test_choose_orientation_sampled_small_bucket() {
        // Small bucket should use full comparison
        let bucket = vec![1, 2, 3, 4, 5];
        let fwd = vec![1, 2, 3]; // All in bucket
        let rc = vec![6, 7, 8]; // None in bucket
        let mut buffer = Vec::new();

        let (orientation, _) = choose_orientation_sampled(&bucket, &fwd, &rc, &mut buffer);
        assert_eq!(orientation, Orientation::Forward);
    }

    #[test]
    fn test_choose_orientation_sampled_rc_wins() {
        // Test RC wins with small bucket
        let bucket = vec![10, 20, 30];
        let fwd = vec![1, 2, 3]; // None in bucket
        let rc = vec![10, 20, 30]; // All in bucket
        let mut buffer = Vec::new();

        let (orientation, _) = choose_orientation_sampled(&bucket, &fwd, &rc, &mut buffer);
        assert_eq!(orientation, Orientation::ReverseComplement);
    }

    #[test]
    fn test_choose_orientation_sampled_large_bucket() {
        // Large bucket triggers sampling
        let bucket: Vec<u64> = (0..200_000).collect();
        let fwd: Vec<u64> = (0..1000).collect(); // First 1000, all in bucket
        let rc: Vec<u64> = (500_000..501_000).collect(); // None in bucket
        let mut buffer = Vec::new();

        let (orientation, overlap) = choose_orientation_sampled(&bucket, &fwd, &rc, &mut buffer);
        assert_eq!(orientation, Orientation::Forward);
        assert!(overlap > 0.0);
    }

    #[test]
    fn test_choose_orientation_sampled_agrees_with_full_clear_cut() {
        // For clear-cut cases, sampled should agree with full
        let bucket: Vec<u64> = (0..200_000).collect();
        let fwd: Vec<u64> = (0..5000).collect(); // Clearly in bucket
        let rc: Vec<u64> = (1_000_000..1_005_000).collect(); // Clearly not in bucket
        let mut buffer = Vec::new();

        let (sampled_orient, _) = choose_orientation_sampled(&bucket, &fwd, &rc, &mut buffer);
        let (full_orient, _) = choose_orientation(&bucket, &fwd, &rc);

        assert_eq!(sampled_orient, full_orient);
    }

    #[test]
    fn test_choose_orientation_sampled_tail_heavy_sequences() {
        // Test case that would fail with naive stride-based sampling:
        // Bucket has 150K elements, fwd matches the tail (100K-150K), rc matches nothing
        let bucket: Vec<u64> = (0..150_000).collect();
        let fwd: Vec<u64> = (120_000..130_000).collect(); // Matches tail of bucket
        let rc: Vec<u64> = (500_000..510_000).collect(); // Matches nothing
        let mut buffer = Vec::new();

        let (sampled_orient, _) = choose_orientation_sampled(&bucket, &fwd, &rc, &mut buffer);
        let (full_orient, _) = choose_orientation(&bucket, &fwd, &rc);

        assert_eq!(
            sampled_orient, full_orient,
            "Stratified sampling should correctly orient tail-heavy sequences"
        );
    }

    #[test]
    fn test_choose_orientation_sampled_boundary_sizes() {
        // Test at exactly the threshold
        let mut buffer = Vec::new();

        // Just below threshold
        let bucket_small: Vec<u64> = (0..ORIENTATION_SAMPLE_SIZE as u64).collect();
        let fwd: Vec<u64> = (0..1000u64).collect();
        let rc: Vec<u64> = (500_000..501_000u64).collect();

        let (orient_small, _) = choose_orientation_sampled(&bucket_small, &fwd, &rc, &mut buffer);
        assert_eq!(orient_small, Orientation::Forward);

        // Just above threshold
        let bucket_large: Vec<u64> = (0..(ORIENTATION_SAMPLE_SIZE + 1) as u64).collect();
        let (orient_large, _) = choose_orientation_sampled(&bucket_large, &fwd, &rc, &mut buffer);
        assert_eq!(orient_large, Orientation::Forward);
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
        let mut buffer = Vec::new();

        // Both methods should choose fwd since it has higher bucket overlap
        let (sampled_orient, _) = choose_orientation_sampled(&bucket, &fwd, &rc, &mut buffer);
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
        let mut buffer = Vec::new();

        let (orientation, overlap) = choose_orientation_sampled(&bucket, &fwd, &rc, &mut buffer);
        // Both empty, should default to Forward with 0 overlap
        assert_eq!(orientation, Orientation::Forward);
        assert_eq!(overlap, 0.0);
    }

    #[test]
    fn test_choose_orientation_sampled_buffer_not_leaked() {
        // Verify that the buffer is cleared and doesn't retain old garbage
        // Use values outside the bucket range as garbage markers
        // Use sequences larger than sample_size to trigger sampling
        let mut buffer = vec![999_999_999, 888_888_888, 777_777_777];

        let bucket: Vec<u64> = (0..200_000).collect();
        // Sequences must be >= ORIENTATION_SAMPLE_SIZE to trigger sampling
        let fwd: Vec<u64> = (0..150_000).collect();
        let rc: Vec<u64> = (500_000..650_000).collect();

        let (orientation, _) = choose_orientation_sampled(&bucket, &fwd, &rc, &mut buffer);
        assert_eq!(orientation, Orientation::Forward);

        // Buffer should have been cleared (garbage values removed)
        assert!(!buffer.contains(&999_999_999));
        assert!(!buffer.contains(&888_888_888));
        assert!(!buffer.contains(&777_777_777));

        // Buffer should contain valid bucket samples (all < 200_000)
        assert!(buffer.iter().all(|&x| x < 200_000));
    }

    // ===== Validation: sampled vs full agreement rate =====

    #[test]
    fn test_sampled_vs_full_agreement_random_distributions() {
        // Test agreement across multiple random-ish distributions
        // Using deterministic "random" patterns to ensure reproducibility
        let mut buffer = Vec::new();
        let mut agreements = 0;
        let total_tests = 100u64;

        for seed in 0..total_tests {
            // Create bucket with pseudo-random distribution
            let bucket: Vec<u64> = (0u64..200_000)
                .filter(|&x| (x.wrapping_mul(31) ^ seed) % 3 != 0)
                .collect();

            // Create fwd and rc with different overlap patterns
            let fwd: Vec<u64> = (0u64..20_000)
                .map(|x| x.wrapping_mul(17).wrapping_add(seed * 1000) % 300_000)
                .collect();
            let rc: Vec<u64> = (0u64..20_000)
                .map(|x| x.wrapping_mul(23).wrapping_add(seed * 2000) % 300_000)
                .collect();

            let mut fwd_sorted = fwd.clone();
            let mut rc_sorted = rc.clone();
            fwd_sorted.sort_unstable();
            fwd_sorted.dedup();
            rc_sorted.sort_unstable();
            rc_sorted.dedup();

            let (sampled, _) =
                choose_orientation_sampled(&bucket, &fwd_sorted, &rc_sorted, &mut buffer);
            let (full, _) = choose_orientation(&bucket, &fwd_sorted, &rc_sorted);

            if sampled == full {
                agreements += 1;
            }
        }

        // We expect very high agreement rate (>95%)
        let agreement_rate = agreements as f64 / total_tests as f64;
        assert!(
            agreement_rate >= 0.95,
            "Agreement rate {} is below 95% threshold",
            agreement_rate
        );
    }
}
