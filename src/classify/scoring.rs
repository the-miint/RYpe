//! Scoring utilities for classification.

/// Compute the dual-strand classification score.
///
/// Score is the maximum of forward and reverse-complement hit ratios.
#[inline]
pub(super) fn compute_score(
    fwd_hits: usize,
    fwd_total: usize,
    rc_hits: usize,
    rc_total: usize,
) -> f64 {
    let fwd_score = if fwd_total > 0 {
        fwd_hits as f64 / fwd_total as f64
    } else {
        0.0
    };
    let rc_score = if rc_total > 0 {
        rc_hits as f64 / rc_total as f64
    } else {
        0.0
    };
    fwd_score.max(rc_score)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_score_fwd_only() {
        assert!((compute_score(5, 10, 0, 0) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_compute_score_rc_only() {
        assert!((compute_score(0, 0, 8, 10) - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_compute_score_max_of_both() {
        // fwd = 3/10 = 0.3, rc = 7/10 = 0.7, max = 0.7
        assert!((compute_score(3, 10, 7, 10) - 0.7).abs() < 0.001);
        // fwd = 9/10 = 0.9, rc = 2/10 = 0.2, max = 0.9
        assert!((compute_score(9, 10, 2, 10) - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_compute_score_empty() {
        assert_eq!(compute_score(0, 0, 0, 0), 0.0);
    }

    #[test]
    fn test_compute_score_perfect() {
        assert_eq!(compute_score(10, 10, 10, 10), 1.0);
    }
}
