//! Common utilities shared across classification functions.

use std::collections::HashSet;

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
pub(super) fn filter_negative_mins(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_negative_mins_none() {
        let fwd = vec![1, 2, 3];
        let rc = vec![4, 5, 6];
        let (f, r) = filter_negative_mins(fwd.clone(), rc.clone(), None);
        assert_eq!(f, fwd);
        assert_eq!(r, rc);
    }

    #[test]
    fn test_filter_negative_mins_empty_set() {
        let fwd = vec![1, 2, 3];
        let rc = vec![4, 5, 6];
        let empty: HashSet<u64> = HashSet::new();
        let (f, r) = filter_negative_mins(fwd.clone(), rc.clone(), Some(&empty));
        assert_eq!(f, fwd);
        assert_eq!(r, rc);
    }

    #[test]
    fn test_filter_negative_mins_filters() {
        let fwd = vec![1, 2, 3];
        let rc = vec![4, 5, 6];
        let neg: HashSet<u64> = vec![2, 5].into_iter().collect();
        let (f, r) = filter_negative_mins(fwd, rc, Some(&neg));
        assert_eq!(f, vec![1, 3]);
        assert_eq!(r, vec![4, 6]);
    }

    #[test]
    fn test_filter_negative_mins_all_filtered() {
        let fwd = vec![1, 2];
        let rc = vec![1, 2];
        let neg: HashSet<u64> = vec![1, 2].into_iter().collect();
        let (f, r) = filter_negative_mins(fwd, rc, Some(&neg));
        assert!(f.is_empty());
        assert!(r.is_empty());
    }
}
