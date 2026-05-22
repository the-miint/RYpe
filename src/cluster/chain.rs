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
    /// DP score array; `f[i]` = best chain score ending at anchor `i`.
    pub(crate) f: Vec<f64>,
    /// DP predecessor pointer; `p[i] = j` means anchor `j` precedes `i` in
    /// the best chain ending at `i`, or `-1` if `i` is a chain start.
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

/// Banded-DP colinear chaining over an anchor list (skani-style).
///
/// `anchors` is sorted **in place** by `(q_pos, r_pos)` ascending before the
/// DP runs — callers reusing a workspace buffer get the position-sorted view
/// back. `compute_anchors_into` produces anchors in hash-order; this is the
/// re-sort step required to switch into chain-traversal order.
///
/// `is_rc` controls the sign of the reference-position delta in the score:
///   - `false` (fwd chain): valid predecessor when `r_pos[i] − r_pos[j] > 0`.
///   - `true`  (rc chain):  valid predecessor when `r_pos[j] − r_pos[i] > 0`.
///
/// Both modes require `q_pos[i] > q_pos[j]` (anchors are q-sorted, so
/// `dq == 0` at the same q-position is the only same-q case and is rejected).
///
/// The score for transition `j → i` is `anchor_credit − |Δr − Δq|`, where
/// `Δq = q_pos[i] − q_pos[j]` and `Δr` is the strand-signed delta. A
/// transition is rejected if `|Δr − Δq| > max_gap_length` (off-diagonal) or
/// `max(Δq, |Δr|) > max_lin_length` (either side jumps too long).
///
/// The `max` form matches skani's `lchain.rs` — a transition is too long if
/// EITHER side exceeds the limit. An earlier draft (research-doc §5.1) used
/// `min` (both sides must exceed); switching to `max` was an early-review
/// fix to match the reference algorithm.
///
/// The inner loop **breaks** (does not skip) when `i − j > band_anchors` or
/// `Δq > band_bp` — both monotonic-in-j cutoffs, no skip heuristic.
///
/// Returns `None` if the best chain has fewer than `params.min_anchors`
/// anchors; otherwise `Some(ChainResult)` describing the highest-scoring
/// chain. The chain's score is `f[argmax]`, never less than `anchor_credit`
/// (the chain-of-length-1 baseline).
pub fn chain_anchors(
    anchors: &mut [(u32, u32)],
    is_rc: bool,
    params: &ChainParams,
    ws: &mut ChainWorkspace,
) -> Option<ChainResult> {
    let n = anchors.len();
    if (n as u32) < params.min_anchors {
        return None;
    }

    // Sort into chain-traversal order. `compute_anchors_into` emits hash-
    // order; DP needs `q_pos` ascending. Secondary key `r_pos` makes the
    // sort deterministic when multiple anchors share a `q_pos` (possible
    // when different minimizers happen to start at the same base).
    anchors.sort_unstable();

    // Reset DP state. `clear() + resize()` is equivalent to a fresh Vec of
    // length `n` filled with the chain-length-1 baseline `anchor_credit`,
    // and reuses the existing allocation when `n` fits the prior capacity.
    ws.f.clear();
    ws.f.resize(n, params.anchor_credit);
    ws.p.clear();
    ws.p.resize(n, -1);

    for i in 1..n {
        let (qi, ri) = anchors[i];
        let mut best_f = params.anchor_credit;
        let mut best_j: i32 = -1;

        // Walk predecessors in reverse anchor-index order so both band
        // cutoffs (monotonic in `i − j` and in `Δq`) can break cleanly.
        for j in (0..i).rev() {
            let (qj, rj) = anchors[j];

            // Band on anchor-index lookback.
            if (i - j) as u32 > params.band_anchors {
                break;
            }
            // Band on query-position lookback. `qi >= qj` (sorted).
            let dq = qi - qj;
            if dq > params.band_bp {
                break;
            }
            // Same q-position cannot extend a chain; skip without breaking.
            if dq == 0 {
                continue;
            }

            // Strand-aware Δr. `i64` lets the sign of "out of order on the
            // chosen strand" be the simple `<= 0` test below.
            let dr_signed: i64 = if is_rc {
                rj as i64 - ri as i64
            } else {
                ri as i64 - rj as i64
            };
            if dr_signed <= 0 {
                continue;
            }
            let dr = dr_signed as u32;

            // Long-jump rejection: reject if EITHER side jumps farther than
            // `max_lin_length`. Matches skani's `chain.rs` semantics. Earlier
            // research-doc §5.1 had this as `min`; switched to `max` per
            // early-review feedback so a one-sided long jump can't sneak
            // past `max_gap_length` when the gap itself happens to be small.
            if dq.max(dr) > params.max_lin_length {
                continue;
            }
            // Off-diagonal rejection: `|Δr − Δq|`.
            let gap = dr.abs_diff(dq);
            if gap > params.max_gap_length {
                continue;
            }

            let new_f = ws.f[j] + params.anchor_credit - gap as f64;
            if new_f > best_f {
                best_f = new_f;
                best_j = j as i32;
            }
        }

        ws.f[i] = best_f;
        ws.p[i] = best_j;
    }

    // Argmax over f[]. With ties (rare with f64), the smallest index wins;
    // this is deterministic given the deterministic sort above.
    let mut argmax = 0usize;
    let mut max_f = ws.f[0];
    for i in 1..n {
        if ws.f[i] > max_f {
            max_f = ws.f[i];
            argmax = i;
        }
    }

    // Backtrack from `argmax` to chain start, counting length and capturing
    // the start anchor. No allocation: we don't store the chain, we just
    // need the endpoints and the count.
    let end_anchor = anchors[argmax];
    let mut start_idx = argmax;
    let mut chain_len: u32 = 1;
    while ws.p[start_idx] >= 0 {
        start_idx = ws.p[start_idx] as usize;
        chain_len += 1;
    }
    let start_anchor = anchors[start_idx];

    if chain_len < params.min_anchors {
        return None;
    }

    Some(ChainResult {
        score: max_f,
        anchors: chain_len,
        q_start: start_anchor.0,
        q_end: end_anchor.0,
        r_start: start_anchor.1,
        r_end: end_anchor.1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Plan 1.3 phase 1: ChainParams ----

    /// WHY: `ChainParams` deliberately has no `Default` impl — the starting
    /// values are uncalibrated and Plan 1.6 will revise them. A future PR
    /// adding `#[derive(Default)]` would silently bypass that intent. This
    /// compile-time assertion makes the absence machine-verifiable.
    #[test]
    fn chain_params_no_default_impl() {
        static_assertions::assert_not_impl_any!(ChainParams: Default);
    }

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

    // ---- Plan 1.3 phase 2: chain_anchors ----

    /// Build a ChainParams with sane defaults for the tests below. Each test
    /// overrides only the fields it cares about.
    fn test_params() -> ChainParams {
        ChainParams {
            anchor_credit: 10.0,
            max_gap_length: 100,
            max_lin_length: 100_000,
            band_anchors: 50,
            band_bp: 100_000,
            min_anchors: 3,
        }
    }

    /// WHY: empty anchors is a real case — disjoint genomes share no
    /// minimizers and the DP must not panic on n=0 indexing.
    #[test]
    fn chain_anchors_empty_returns_none() {
        let mut anchors = Vec::new();
        let mut ws = ChainWorkspace::new();
        assert!(chain_anchors(&mut anchors, false, &test_params(), &mut ws).is_none());
    }

    /// WHY: `min_anchors` is the false-positive gate. Two perfectly colinear
    /// anchors with `min_anchors=3` must return None — otherwise random
    /// matches in unrelated genomes could fake a chain.
    #[test]
    fn chain_anchors_below_min_returns_none() {
        let mut anchors = vec![(0u32, 0u32), (50, 50)];
        let mut ws = ChainWorkspace::new();
        assert!(chain_anchors(&mut anchors, false, &test_params(), &mut ws).is_none());
    }

    /// WHY (headline): score = anchor count when all gaps are zero. This
    /// is the contract that makes `chain_score` interpretable as
    /// "co-linear anchor count" downstream (chaining-research.md §5.2).
    #[test]
    fn chain_anchors_three_colinear_fwd() {
        let mut anchors = vec![(0u32, 0u32), (50, 50), (100, 100)];
        let mut ws = ChainWorkspace::new();
        let result = chain_anchors(&mut anchors, false, &test_params(), &mut ws).unwrap();
        assert_eq!(result.anchors, 3);
        assert_eq!(result.score, 30.0, "3 × anchor_credit when gaps are zero");
        assert_eq!(result.q_start, 0);
        assert_eq!(result.q_end, 100);
        assert_eq!(result.r_start, 0);
        assert_eq!(result.r_end, 100);
    }

    /// WHY: the rc strand reads ref positions backwards as q advances.
    /// `r_start`/`r_end` track query-traversal order, not r-value order —
    /// so `r_start ≥ r_end` on the rc strand, and any caller computing
    /// `r_end - r_start` must use signed arithmetic. Pinning this here
    /// prevents a silent regression where the rc chain reports endpoints
    /// in r-value order.
    #[test]
    fn chain_anchors_three_colinear_rc() {
        let mut anchors = vec![(0u32, 100u32), (50, 50), (100, 0)];
        let mut ws = ChainWorkspace::new();
        let result = chain_anchors(&mut anchors, true, &test_params(), &mut ws).unwrap();
        assert_eq!(result.anchors, 3);
        assert_eq!(result.score, 30.0);
        assert_eq!(result.q_start, 0);
        assert_eq!(result.q_end, 100);
        assert_eq!(result.r_start, 100, "rc chain starts at the largest r");
        assert_eq!(result.r_end, 0, "rc chain ends at the smallest r");
    }

    /// WHY: an off-diagonal anchor must isolate itself — both transitions
    /// touching it are rejected by `max_gap_length` (transition 0→1 has
    /// gap=450; transition 1→2 has dr_signed = 100−500 = −400 → wrong
    /// direction). The remaining valid transition 0→2 has gap=0 and forms
    /// a length-2 chain — below `min_anchors=3`, so the DP returns None.
    /// This is the central insight of chaining vs. set-containment: a weak
    /// off-diagonal anchor counted by set containment does NOT extend a
    /// chain, and a 2-anchor "chain" is rejected as a false positive.
    #[test]
    fn chain_anchors_off_diagonal_rejected() {
        let mut anchors = vec![(0u32, 0u32), (50, 500), (100, 100)];
        let mut ws = ChainWorkspace::new();
        assert!(chain_anchors(&mut anchors, false, &test_params(), &mut ws).is_none());
    }

    /// WHY: `band_bp` IS the algorithm — without the break, the inner loop
    /// is O(n²); without an effective break, the band is dead code. Test
    /// pins the break by constructing two anchors whose `dq` exceeds
    /// `band_bp` exactly, then sanity-checks that loosening the band lets
    /// the chain form.
    #[test]
    fn chain_anchors_band_bp_breaks_inner_loop() {
        let params_tight = ChainParams {
            band_bp: 500,
            min_anchors: 2,
            ..test_params()
        };
        let mut anchors = vec![(0u32, 0u32), (1000, 1000)];
        let mut ws = ChainWorkspace::new();
        // dq=1000 > band_bp=500 → break before scoring → no chain.
        assert!(chain_anchors(&mut anchors, false, &params_tight, &mut ws).is_none());

        let params_wide = ChainParams {
            band_bp: 1500,
            ..params_tight
        };
        let mut anchors_ok = vec![(0u32, 0u32), (1000, 1000)];
        let result = chain_anchors(&mut anchors_ok, false, &params_wide, &mut ws).unwrap();
        assert_eq!(result.anchors, 2);
    }

    /// WHY: `band_anchors` is the other half of the band — `i − j` cutoff.
    /// Same shape of test as `band_bp_breaks`: design four anchors so that
    /// the dominant predecessor is `j = i − 2` (forcing `band_anchors=1`
    /// to break before scoring it), and the immediate neighbor is an
    /// off-diagonal trap.
    #[test]
    fn chain_anchors_band_anchors_breaks_inner_loop() {
        // anchor 0 (q=0,   r=0)    — chain start
        // anchor 1 (q=100, r=100)  — colinear with 0
        // anchor 2 (q=110, r=10000) — off-diagonal trap (immediate predecessor of 3)
        // anchor 3 (q=200, r=200)  — colinear with 0,1
        let params_tight = ChainParams {
            band_anchors: 1,
            ..test_params()
        };
        let mut anchors = vec![(0u32, 0u32), (100, 100), (110, 10_000), (200, 200)];
        let mut ws = ChainWorkspace::new();
        // band_anchors=1: anchor 3 can only see anchor 2 → rejected → f[3]=10.
        // Chain length for anchor 3 stays 1. The best chain (anchor 1 from 0)
        // has length 2, which is below min_anchors=3 → None.
        assert!(chain_anchors(&mut anchors, false, &params_tight, &mut ws).is_none());

        let params_wide = ChainParams {
            band_anchors: 10,
            ..params_tight
        };
        let mut anchors_ok = vec![(0u32, 0u32), (100, 100), (110, 10_000), (200, 200)];
        let result = chain_anchors(&mut anchors_ok, false, &params_wide, &mut ws).unwrap();
        // anchor 3 can now reach anchor 1 (skipping the trap) → chain [0,1,3].
        assert_eq!(result.anchors, 3);
        assert_eq!(result.q_start, 0);
        assert_eq!(result.q_end, 200);
    }

    /// WHY: when two disjoint chains exist, argmax picks the higher-scoring
    /// one. A bug here (e.g. backtracking from the highest INDEX instead of
    /// the highest SCORE) would silently report the wrong chain.
    #[test]
    fn chain_anchors_two_chains_returns_best() {
        // Chain A (q=0..100):  gap=5 transitions → 3·anchor_credit − 2·gap
        //                      = 30 − 10 = 20. (f[0]=10, f[1]=15, f[2]=20.)
        // Chain B (q=10000..): gap=0 transitions → 3·anchor_credit
        //                      = 30. (f[3]=10, f[4]=20, f[5]=30.)
        // Inter-chain transitions: dq ≈ 9900, dr ≈ 4890 → gap ≈ 5010
        // > max_gap_length=100 → no cross-chain linkage. argmax = anchor 5.
        let mut anchors = vec![
            (0u32, 0u32),
            (50, 55),
            (100, 110),
            (10_000, 5_000),
            (10_050, 5_050),
            (10_100, 5_100),
        ];
        let mut ws = ChainWorkspace::new();
        let result = chain_anchors(&mut anchors, false, &test_params(), &mut ws).unwrap();
        assert_eq!(result.anchors, 3);
        assert_eq!(result.score, 30.0);
        assert_eq!(result.q_start, 10_000);
        assert_eq!(result.q_end, 10_100);
        assert_eq!(result.r_start, 5_000);
        assert_eq!(result.r_end, 5_100);
    }

    /// WHY: `max_lin_length` uses the `max(Δq, |Δr|)` form — reject if
    /// EITHER side jumps too far. An asymmetric anchor pair pins the
    /// choice: `dq=4900 < limit=5000`, `dr=5100 > limit`, `gap=200` at
    /// the gap limit (so gap rejection doesn't fire). Under the `min`
    /// form this would survive; under `max` (skani semantics) it is
    /// rejected. If someone "fixes" the `dq.max(dr)` line back to `min`,
    /// this test fails.
    #[test]
    fn chain_anchors_max_lin_length_rejects_asymmetric_jump() {
        let params = ChainParams {
            anchor_credit: 10.0,
            max_gap_length: 200,
            max_lin_length: 5_000,
            band_anchors: 10,
            band_bp: 100_000,
            min_anchors: 2,
        };
        let mut anchors = vec![(0u32, 0u32), (4_900, 5_100)];
        let mut ws = ChainWorkspace::new();
        // dq=4900 (≤ limit), dr=5100 (> limit), gap=200 (= max_gap_length, allowed).
        // max(4900, 5100) = 5100 > 5000 → reject. min(4900, 5100) = 4900 ≤ 5000
        // would NOT reject — proving the operator choice matters.
        assert!(chain_anchors(&mut anchors, false, &params, &mut ws).is_none());
    }

    /// WHY: `max_lin_length` prevents two distant but on-diagonal anchors
    /// from being merged into a single chain. Without it, a real chain at
    /// one end of the contig and a random match at the other end would
    /// spuriously join. The same anchor pair is tested with the limit
    /// raised, to prove the rejection is causal.
    #[test]
    fn chain_anchors_max_lin_length_rejects_long_jump() {
        let params_tight = ChainParams {
            max_lin_length: 5_000,
            min_anchors: 2,
            ..test_params()
        };
        let mut anchors = vec![(0u32, 0u32), (10_000, 10_000)];
        let mut ws = ChainWorkspace::new();
        // dq=dr=10_000 (gap=0, would be allowed), but max(dq,dr)=10_000 > 5_000 → reject.
        // This symmetric case can't distinguish `min` from `max` —
        // `chain_anchors_max_lin_length_rejects_asymmetric_jump` pins the
        // operator choice; this one only proves the threshold is checked.
        assert!(chain_anchors(&mut anchors, false, &params_tight, &mut ws).is_none());

        let params_wide = ChainParams {
            max_lin_length: 20_000,
            ..params_tight
        };
        let mut anchors_ok = vec![(0u32, 0u32), (10_000, 10_000)];
        let result = chain_anchors(&mut anchors_ok, false, &params_wide, &mut ws).unwrap();
        assert_eq!(result.anchors, 2);
    }

    /// WHY: the strand flag is the strand decision; calling with the wrong
    /// `is_rc` on real data produces silent zero-anchor chains. This test
    /// proves the flag is causal: same anchors, both flags, opposite
    /// outcomes.
    #[test]
    fn chain_anchors_strand_purity_via_is_rc() {
        // Forward-colinear: r ascends with q. fwd chain forms; rc rejects
        // every transition (dr_signed = rj - ri < 0).
        let anchors_fwd = vec![(0u32, 0u32), (50, 50), (100, 100)];
        let mut ws = ChainWorkspace::new();

        let mut fwd_copy = anchors_fwd.clone();
        let r_fwd = chain_anchors(&mut fwd_copy, false, &test_params(), &mut ws).unwrap();
        assert_eq!(r_fwd.anchors, 3);

        let mut rc_copy = anchors_fwd;
        assert!(
            chain_anchors(&mut rc_copy, true, &test_params(), &mut ws).is_none(),
            "rc chain on fwd-colinear anchors must not form"
        );
    }

    /// WHY: workspace reuse is the alloc-amortization story. If `f`/`p`
    /// leak state across calls, the second call's results are silently
    /// wrong. Test by running a 5-anchor case then a 3-anchor case and
    /// asserting the second's results are clean.
    #[test]
    fn chain_anchors_workspace_reuse_clears_dp_state() {
        let mut ws = ChainWorkspace::new();
        let params = test_params();

        let mut long_anchors = vec![(0u32, 0u32), (50, 50), (100, 100), (150, 150), (200, 200)];
        let r1 = chain_anchors(&mut long_anchors, false, &params, &mut ws).unwrap();
        assert_eq!(r1.anchors, 5);
        assert_eq!(r1.score, 50.0);

        let mut short_anchors = vec![(0u32, 0u32), (50, 50), (100, 100)];
        let r2 = chain_anchors(&mut short_anchors, false, &params, &mut ws).unwrap();
        assert_eq!(r2.anchors, 3);
        assert_eq!(r2.score, 30.0);

        assert_eq!(ws.f.len(), 3, "workspace must shrink to match second call");
        assert_eq!(ws.p.len(), 3);
    }

    /// WHY: `compute_anchors_into` emits hash-order; chain DP needs
    /// q_pos-order. The DP sorts in place. Test that passing anchors in
    /// hash-order (i.e. NOT q_pos-order) still produces the correct chain.
    #[test]
    fn chain_anchors_sorts_unsorted_input() {
        // Same 3 colinear anchors as chain_anchors_three_colinear_fwd, but
        // shuffled to simulate hash-order output from compute_anchors_into.
        let mut anchors = vec![(50u32, 50u32), (100, 100), (0, 0)];
        let mut ws = ChainWorkspace::new();
        let result = chain_anchors(&mut anchors, false, &test_params(), &mut ws).unwrap();
        assert_eq!(result.anchors, 3);
        assert_eq!(result.q_start, 0);
        assert_eq!(result.q_end, 100);
        // Side-effect: anchors are now sorted by (q,r) in place.
        assert_eq!(anchors, vec![(0, 0), (50, 50), (100, 100)]);
    }
}
