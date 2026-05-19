//! Containment math and per-edge threshold filtering.
//!
//! Rype's classify score is `shared(query, bucket) / |query_mins|`. For
//! greedy dereplication we instead want `containment_of_other_in_seed =
//! shared / |other_mins|` — "what fraction of the candidate-to-absorb's
//! minimizers are present in the seed it would join."

/// Containment of the candidate-to-absorb in the seed.
///
/// Returns 0.0 when `other_mins` is 0 (a contig with no minimizers cannot be
/// usefully absorbed). When `shared > other_mins` the value is clamped to
/// 1.0 — this should not happen in a correct pipeline (shared minimizers are
/// drawn from the candidate's own set), so a `debug_assert!` guards it.
pub fn containment(shared: u64, other_mins: u64) -> f64 {
    if other_mins == 0 {
        return 0.0;
    }
    debug_assert!(
        shared <= other_mins,
        "containment invariant violated: shared={} > other_mins={}",
        shared,
        other_mins
    );
    let c = (shared as f64) / (other_mins as f64);
    if c > 1.0 {
        1.0
    } else {
        c
    }
}

/// Decide whether a (candidate, seed) edge should trigger absorption.
///
/// Requires BOTH the normalized containment to clear `threshold` AND the
/// absolute shared-minimizer count to clear `min_shared`. The absolute
/// floor is the defense against small contigs being absorbed via a single
/// shared mobile element.
pub fn edge_qualifies(shared: u64, other_mins: u64, threshold: f64, min_shared: u64) -> bool {
    shared >= min_shared && containment(shared, other_mins) >= threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn containment_basic_ratio() {
        assert!((containment(100, 1000) - 0.10).abs() < 1e-12);
        assert!((containment(950, 1000) - 0.95).abs() < 1e-12);
        assert!((containment(1000, 1000) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn containment_zero_other_returns_zero() {
        assert_eq!(containment(0, 0), 0.0);
        // shared > 0 with other_mins=0 still returns 0.0 (defensive)
        assert_eq!(containment(50, 0), 0.0);
    }

    #[test]
    fn containment_zero_shared_returns_zero() {
        assert_eq!(containment(0, 1000), 0.0);
    }

    #[test]
    fn edge_qualifies_passes_when_both_satisfied() {
        // 900 / 1000 = 0.90, above T=0.85, above N_min=500
        assert!(edge_qualifies(900, 1000, 0.85, 500));
    }

    #[test]
    fn edge_qualifies_blocks_low_containment() {
        // 600 / 1000 = 0.60, below T=0.85, even though above N_min
        assert!(!edge_qualifies(600, 1000, 0.85, 500));
    }

    #[test]
    fn edge_qualifies_blocks_low_shared_count_high_containment() {
        // 100 / 100 = 1.0 (small contig, fully contained) but only 100
        // shared minimizers — below N_min=500 floor. This is the mobile-
        // element / tiny-plasmid defense.
        assert!(!edge_qualifies(100, 100, 0.85, 500));
    }

    #[test]
    fn edge_qualifies_threshold_boundary_is_inclusive() {
        // Containment exactly at threshold should qualify.
        assert!(edge_qualifies(850, 1000, 0.85, 500));
    }

    #[test]
    fn edge_qualifies_min_shared_boundary_is_inclusive() {
        // shared == min_shared and containment well above threshold should qualify.
        assert!(edge_qualifies(500, 500, 0.85, 500));
    }
}
