//! Banded-DP chaining after skani (Shaw & Yu 2023) — see
//! `localdocs/chaining-research.md` §3 for the algorithm reference.
//!
//! Plan 1.3 (`~/.claude/plans/banded-walking-shaw.md`) scopes this module to:
//!   - merge-join two hash-sorted minimizer lists into anchors
//!     ([`compute_anchors_into`], Phase 1);
//!   - run a banded DP over anchors to find the best colinear chain
//!     (`chain_anchors`, Phase 2);
//!   - return chain score + anchor count + query/reference span.
//!
//! Plan 1.4 wires this into `cluster::edges`; this plan ships only the
//! primitives.

/// Parameters for the banded-DP chaining algorithm.
///
/// No `Default` impl: per `chaining-research.md` §5.2 these starting values
/// are uncalibrated. Use [`ChainParams::starting_for_w`] to acknowledge that
/// you are picking research-doc starting values, not blessed defaults.
/// Plan 1.6 (calibration) will revise these on real MAG data.
#[derive(Debug, Clone)]
pub struct ChainParams {
    /// Constant credit per chained anchor. Net contribution per transition
    /// is `anchor_credit − gap`; `1.0` gives "score = count of well-aligned
    /// anchors". Plan 1.6 may revise to a fractional value if calibration
    /// shows a softer credit boundary improves discrimination.
    pub anchor_credit: f64,
    /// Max `|Δr − Δq|` for a transition to be allowed. Beyond this the
    /// candidate anchor pair is off-diagonal and rejected.
    pub max_gap_length: u32,
    /// Max `min(Δq, |Δr|)` for a transition. Beyond this the jump is too
    /// long even if the diagonal deviation is small — likely two unrelated
    /// chains being spuriously merged.
    pub max_lin_length: u32,
    /// Hard band on anchor-index lookback. `i − j ≤ band_anchors` to attempt
    /// the transition; beyond this the inner loop breaks.
    pub band_anchors: u32,
    /// Hard band on query-position lookback. `Δq ≤ band_bp` to attempt the
    /// transition; beyond this the inner loop breaks.
    pub band_bp: u32,
    /// Minimum number of chained anchors for the DP to return `Some` (Phase 2
    /// `chain_anchors`). Below this the chain is considered a false positive.
    pub min_anchors: u32,
}

impl ChainParams {
    /// Starting values from `localdocs/chaining-research.md` §5.2.
    ///
    /// These are *starting* values; the parameter sweep in Plan 1.6 will
    /// revise them. The name "starting_for_w" is deliberate — there is no
    /// `Default` impl because we have not yet calibrated these against real
    /// MAG data.
    pub fn starting_for_w(w: usize) -> Self {
        let w = w as u32;
        Self {
            anchor_credit: 1.0,
            max_gap_length: 4 * w,
            max_lin_length: 100 * w,
            band_anchors: 50,
            band_bp: 50 * w,
            min_anchors: 3,
        }
    }
}

/// Output of `chain_anchors` — the best colinear chain found.
///
/// `q_start`/`q_end` are query-position endpoints in traversal order, so
/// `q_start ≤ q_end` always. `r_start`/`r_end` are ref-position endpoints in
/// **traversal order**, which means `r_start ≤ r_end` on the forward strand
/// but `r_start ≥ r_end` on the reverse strand (rc-chain reads the reference
/// backwards as the query advances).
#[derive(Debug, Clone, PartialEq)]
pub struct ChainResult {
    /// Sum of `(anchor_credit − gap)` over chained transitions, plus the
    /// initial `anchor_credit` for the chain-start anchor.
    pub score: f64,
    /// Number of chained anchors (≥ `params.min_anchors`).
    pub anchors: u32,
    /// First chained query position (smallest, since the DP walks ascending).
    pub q_start: u32,
    /// Last chained query position (largest).
    pub q_end: u32,
    /// First chained reference position in chain-traversal order.
    pub r_start: u32,
    /// Last chained reference position in chain-traversal order.
    pub r_end: u32,
}

/// Reusable buffers for the chaining DP.
///
/// Cluster-time has O(N²) (query, target) pairs to score; allocating the DP
/// arrays per call would dominate. Thread one workspace through the loop
/// and pay the alloc cost once.
#[derive(Debug, Default)]
pub struct ChainWorkspace {
    // Phase 1 reserves these for Phase 2's DP. `allow(dead_code)` is
    // transient — Phase 2 uses both fields and removes the attribute.
    /// DP score array; `f[i]` = best chain score ending at anchor `i`.
    #[allow(dead_code)]
    pub(crate) f: Vec<f64>,
    /// DP predecessor pointer; `p[i] = j` means anchor `j` precedes `i` in
    /// the best chain ending at `i`, or `-1` if `i` is a chain start.
    #[allow(dead_code)]
    pub(crate) p: Vec<i32>,
}

impl ChainWorkspace {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Merge-join two hash-sorted minimizer arrays into an anchor list.
///
/// Both `q_hashes` and `t_hashes` must be sorted ascending and contain no
/// duplicates within their own array (the invariant enforced by
/// [`crate::pairs_into_cluster_bucket_arrays`] and
/// [`crate::ClusterBucketData`]'s `validate`). The `positions` arrays must
/// be index-parallel with their hash arrays.
///
/// `out` is **cleared on entry** before any anchors are appended; callers
/// reusing a workspace buffer don't need to clear themselves.
///
/// Each emitted `(q_pos, t_pos)` is the position pair where a hash is
/// shared between query and target. Output ordering follows hash-merge
/// order, **not** query-position order — callers running the DP must
/// position-sort first (Phase 2's `chain_anchors` does this).
///
/// # Panics
///
/// In debug builds, panics if either side's hash length differs from its
/// position length. Both pairs are an index-parallel contract.
pub fn compute_anchors_into(
    q_hashes: &[u64],
    q_positions: &[u32],
    t_hashes: &[u64],
    t_positions: &[u32],
    out: &mut Vec<(u32, u32)>,
) {
    debug_assert_eq!(
        q_hashes.len(),
        q_positions.len(),
        "query hashes and positions must be index-parallel"
    );
    debug_assert_eq!(
        t_hashes.len(),
        t_positions.len(),
        "target hashes and positions must be index-parallel"
    );

    out.clear();

    let mut qi = 0usize;
    let mut ti = 0usize;
    while qi < q_hashes.len() && ti < t_hashes.len() {
        let qh = q_hashes[qi];
        let th = t_hashes[ti];
        if qh == th {
            out.push((q_positions[qi], t_positions[ti]));
            qi += 1;
            ti += 1;
        } else if qh < th {
            qi += 1;
        } else {
            ti += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Plan 1.3 phase 1: ChainParams ----

    /// Pins the `chaining-research.md` §5.2 starting values for `w=50`.
    /// WHY: Plan 1.6 will revise these — this test catches silent drift in
    /// the interim. If `starting_for_w` intentionally changes, update this
    /// test in the same commit so the new values are committed to.
    #[test]
    fn chain_params_starting_for_w_pins_research_doc_values() {
        let p = ChainParams::starting_for_w(50);
        assert_eq!(p.anchor_credit, 1.0);
        assert_eq!(p.max_gap_length, 200, "4 × w");
        assert_eq!(p.max_lin_length, 5000, "100 × w");
        assert_eq!(p.band_anchors, 50);
        assert_eq!(p.band_bp, 2500, "50 × w");
        assert_eq!(p.min_anchors, 3);
    }

    /// `starting_for_w` must scale linearly in `w` for the bp-keyed
    /// parameters. WHY: if a future cluster run uses a different `w`
    /// (different sketch density), the gap bounds must track. Hardcoding
    /// research-doc values for `w=50` would silently break other-w callers.
    #[test]
    fn chain_params_starting_for_w_scales_with_w() {
        let p100 = ChainParams::starting_for_w(100);
        assert_eq!(p100.max_gap_length, 400);
        assert_eq!(p100.max_lin_length, 10_000);
        assert_eq!(p100.band_bp, 5000);
        // Anchor-count bands and min_anchors do NOT scale with w; they are
        // anchor-count bounds, independent of bp.
        assert_eq!(p100.band_anchors, 50);
        assert_eq!(p100.min_anchors, 3);
    }

    // ---- Plan 1.3 phase 1: compute_anchors_into ----

    /// WHY: the join must handle empty inputs gracefully — both empty bucket
    /// (target) and empty query are real (filter to min_length can drop one
    /// side to zero).
    #[test]
    fn compute_anchors_into_empty_inputs() {
        let mut out = Vec::new();

        compute_anchors_into(&[], &[], &[], &[], &mut out);
        assert!(out.is_empty());

        compute_anchors_into(&[1, 2], &[10, 20], &[], &[], &mut out);
        assert!(out.is_empty());

        compute_anchors_into(&[], &[], &[1, 2], &[10, 20], &mut out);
        assert!(out.is_empty());
    }

    /// WHY: when nothing matches, the DP must see an empty anchor list and
    /// return None. A bug here would silently feed garbage to the DP.
    #[test]
    fn compute_anchors_into_no_intersection() {
        let mut out = Vec::new();
        compute_anchors_into(
            &[1, 2, 3],
            &[10, 20, 30],
            &[4, 5, 6],
            &[100, 200, 300],
            &mut out,
        );
        assert!(out.is_empty());
    }

    /// WHY: identical query and target should yield one anchor per hash.
    /// Load-bearing identity case — `cluster::edges` excludes self-vs-self,
    /// but the DP must still handle correctly (e.g. near-identical assemblies).
    #[test]
    fn compute_anchors_into_full_intersection() {
        let mut out = Vec::new();
        compute_anchors_into(
            &[1, 2, 3],
            &[10, 20, 30],
            &[1, 2, 3],
            &[100, 200, 300],
            &mut out,
        );
        assert_eq!(out, vec![(10, 100), (20, 200), (30, 300)]);
    }

    /// WHY: the load-bearing real-data case is partial overlap. This test
    /// pins both the merge-order output AND the fact that anchors come out
    /// in hash-order, not query-position order — a downstream sort is
    /// required (Phase 2's `chain_anchors` does it).
    #[test]
    fn compute_anchors_into_partial_intersection() {
        let mut out = Vec::new();
        compute_anchors_into(
            &[1, 3, 5, 7, 9],
            &[100, 200, 300, 400, 500],
            &[2, 3, 5, 8],
            &[10, 20, 30, 40],
            &mut out,
        );
        // Hashes shared: 3 → (q_pos=200, t_pos=20), 5 → (q_pos=300, t_pos=30).
        assert_eq!(out, vec![(200, 20), (300, 30)]);
    }

    /// WHY: a reused workspace buffer must not accumulate anchors across
    /// calls — that would silently corrupt later chain DPs in the
    /// O(N²)-pair loop.
    #[test]
    fn compute_anchors_into_clears_out_on_entry() {
        let mut out = vec![(99, 99), (88, 88)];

        // Call 1: no intersection — pre-populated content must be cleared.
        compute_anchors_into(&[1], &[10], &[2], &[20], &mut out);
        assert!(out.is_empty());

        // Call 2: one anchor.
        compute_anchors_into(&[1], &[10], &[1], &[20], &mut out);
        assert_eq!(out, vec![(10, 20)]);

        // Call 3: new content. Prior anchor must be replaced, not appended to.
        compute_anchors_into(&[2], &[30], &[2], &[40], &mut out);
        assert_eq!(out, vec![(30, 40)]);
    }

    /// WHY: a length mismatch between hashes and positions is caller error
    /// that would silently truncate via the merge-join indexing in a release
    /// build. The `debug_assert` makes the contract enforceable in
    /// development; this test pins the assert message.
    #[test]
    #[should_panic(expected = "query hashes and positions must be index-parallel")]
    #[cfg(debug_assertions)]
    fn compute_anchors_into_panics_on_query_length_mismatch() {
        let mut out = Vec::new();
        compute_anchors_into(&[1, 2], &[10], &[1, 2], &[100, 200], &mut out);
    }

    /// WHY: same as above but for the target side.
    #[test]
    #[should_panic(expected = "target hashes and positions must be index-parallel")]
    #[cfg(debug_assertions)]
    fn compute_anchors_into_panics_on_target_length_mismatch() {
        let mut out = Vec::new();
        compute_anchors_into(&[1, 2], &[10, 20], &[1, 2], &[100], &mut out);
    }
}
