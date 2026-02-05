//! Merge utilities for sorted vectors.
//!
//! This module provides functions for merging multiple sorted, deduplicated
//! vectors efficiently. Used in index building and classification.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Merge multiple sorted, deduplicated vectors into one sorted, deduplicated vector.
///
/// Uses a min-heap for efficient k-way merge with O(n log k) complexity
/// where n is total elements and k is number of vectors.
///
/// # Arguments
/// * `sorted_vecs` - A vector of sorted, deduplicated vectors to merge
///
/// # Returns
/// A single sorted, deduplicated vector containing all unique elements.
///
/// # Examples
/// ```
/// use rype::kway_merge_dedup;
///
/// let input = vec![vec![1, 3, 5], vec![2, 4, 6], vec![1, 2, 7]];
/// let merged = kway_merge_dedup(input);
/// assert_eq!(merged, vec![1, 2, 3, 4, 5, 6, 7]);
/// ```
pub fn kway_merge_dedup(sorted_vecs: Vec<Vec<u64>>) -> Vec<u64> {
    if sorted_vecs.is_empty() {
        return Vec::new();
    }

    if sorted_vecs.len() == 1 {
        return sorted_vecs.into_iter().next().unwrap();
    }

    // Min-heap: (value, vec_idx, pos_in_vec)
    // Pre-allocate capacity for heap since we know max size
    let mut heap: BinaryHeap<Reverse<(u64, usize, usize)>> =
        BinaryHeap::with_capacity(sorted_vecs.len());

    // Initialize with first element of each non-empty vec
    for (i, v) in sorted_vecs.iter().enumerate() {
        if !v.is_empty() {
            heap.push(Reverse((v[0], i, 0)));
        }
    }

    let total_len: usize = sorted_vecs.iter().map(|v| v.len()).sum();
    let mut result = Vec::with_capacity(total_len);
    let mut last: Option<u64> = None;

    while let Some(Reverse((val, vec_idx, pos))) = heap.pop() {
        // Deduplicate across vectors
        if last != Some(val) {
            result.push(val);
            last = Some(val);
        }

        // Push next element from same vec
        let next_pos = pos + 1;
        if next_pos < sorted_vecs[vec_idx].len() {
            heap.push(Reverse((sorted_vecs[vec_idx][next_pos], vec_idx, next_pos)));
        }
    }

    result
}

/// Merge sorted `source` into sorted `target` in-place, deduplicating.
///
/// Uses swap to avoid copying target's contents. The result is a sorted,
/// deduplicated Vec containing all unique elements from both inputs.
///
/// # Complexity
/// O(target.len() + source.len()) - merges into a new Vec, then swaps.
/// The swap is O(1) as it just exchanges Vec internals (ptr, len, capacity).
///
/// # Examples
/// ```
/// use rype::merge_sorted_into;
///
/// let mut target = vec![1, 3, 5];
/// merge_sorted_into(&mut target, &[2, 4, 6]);
/// assert_eq!(target, vec![1, 2, 3, 4, 5, 6]);
/// ```
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
    let mut last_pushed: Option<u64> = None;

    // Helper closure to push unique values, avoiding DRY violation
    let mut push_unique = |val: u64| {
        if last_pushed != Some(val) {
            merged.push(val);
            last_pushed = Some(val);
        }
    };

    let mut i = 0;
    let mut j = 0;

    while i < target.len() && j < source.len() {
        match target[i].cmp(&source[j]) {
            std::cmp::Ordering::Less => {
                push_unique(target[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                push_unique(source[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                push_unique(target[i]);
                i += 1;
                j += 1;
            }
        }
    }

    // Remaining elements from target
    for &v in &target[i..] {
        push_unique(v);
    }

    // Remaining elements from source
    for &v in &source[j..] {
        push_unique(v);
    }

    // O(1) swap - just exchanges Vec internals (ptr, len, capacity)
    std::mem::swap(target, &mut merged);
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== kway_merge_dedup tests =====

    #[test]
    fn test_kway_merge_dedup_empty_input() {
        let result = kway_merge_dedup(vec![]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_kway_merge_dedup_single_vec() {
        let input = vec![vec![1, 2, 3, 4, 5]];
        let result = kway_merge_dedup(input);
        assert_eq!(result, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_kway_merge_dedup_multiple_vecs_no_overlap() {
        let input = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
        let result = kway_merge_dedup(input);
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_kway_merge_dedup_with_duplicates() {
        let input = vec![vec![1, 3, 5], vec![2, 3, 6], vec![1, 4, 5]];
        let result = kway_merge_dedup(input);
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_kway_merge_dedup_all_same() {
        let input = vec![vec![5, 5, 5], vec![5, 5], vec![5]];
        let result = kway_merge_dedup(input);
        assert_eq!(result, vec![5]);
    }

    #[test]
    fn test_kway_merge_dedup_with_empty_vecs() {
        let input = vec![vec![], vec![1, 2], vec![], vec![3, 4], vec![]];
        let result = kway_merge_dedup(input);
        assert_eq!(result, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_kway_merge_dedup_large_input() {
        // Test with many vectors to ensure heap operations work correctly
        let input: Vec<Vec<u64>> = (0..100).map(|i| vec![i as u64, i as u64 + 100]).collect();
        let result = kway_merge_dedup(input);
        assert_eq!(result.len(), 200); // 0..99 and 100..199
        assert!(result.windows(2).all(|w| w[0] < w[1]), "Should be sorted");
    }

    // ===== merge_sorted_into tests =====

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
}
