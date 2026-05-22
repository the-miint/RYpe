//! Length-sorted greedy dereplication.
//!
//! Pure function over a precomputed edge list and per-contig metadata; no
//! allocation of workspaces and no I/O.

use super::types::{ChainScore, ClusterEdge, ClusterRow, ContigInfo};

/// Per-target absorption candidate, projected from `ClusterEdge` into the
/// minimal tuple the greedy loop reads: `(query_idx, score, shared, chain)`.
/// `Option<ChainScore>` mirrors `ClusterEdge.chain`.
type Candidate = (u32, f64, u64, Option<ChainScore>);

/// Run length-sorted greedy dereplication.
///
/// One [`ClusterRow`] is emitted per input contig (complete partition).
/// Representatives appear with `rep_contig == member_contig` and
/// `containment == 1.0`. Within each representative, absorbed members are
/// sorted by their original `contigs` index for deterministic output.
///
/// Self-edges in `edges` are ignored. Duplicate `(query, target)` pairs
/// resolve to the first occurrence in iteration order.
///
/// `min_chain_containment` (Plan 1.4) is the optional chain-gate threshold:
///   - `None`: chain field on `ClusterEdge` is ignored (legacy behavior).
///   - `Some(m)`: an edge absorbs only if its `chain` is `Some(c)` with
///     `c.containment >= m`. Edges with `chain == None` are rejected.
pub fn greedy_dereplicate(
    edges: &[ClusterEdge],
    contigs: &[ContigInfo],
    threshold: f64,
    min_shared: u64,
    min_chain_containment: Option<f64>,
) -> Vec<ClusterRow> {
    let n = contigs.len();
    if n == 0 {
        return Vec::new();
    }

    let mut edges_by_target: Vec<Vec<Candidate>> = (0..n).map(|_| Vec::new()).collect();
    for e in edges {
        if e.query_idx == e.target_idx {
            continue;
        }
        let ti = e.target_idx as usize;
        if ti < n {
            edges_by_target[ti].push((e.query_idx, e.score, e.shared, e.chain));
        }
    }

    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by(|&a, &b| {
        let la = contigs[a as usize].length;
        let lb = contigs[b as usize].length;
        lb.cmp(&la).then(a.cmp(&b))
    });

    let mut processed = vec![false; n];
    let mut rows = Vec::with_capacity(n);

    for seed_idx in order {
        let si = seed_idx as usize;
        if processed[si] {
            continue;
        }
        processed[si] = true;

        let seed = &contigs[si];
        rows.push(ClusterRow {
            rep_contig: seed.id.clone(),
            member_contig: seed.id.clone(),
            source_mag: seed.source_mag.clone(),
            containment: 1.0,
            chain: None,
        });

        let candidates = &edges_by_target[si];
        if candidates.is_empty() {
            continue;
        }

        let mut absorbed: Vec<(u32, f64, Option<ChainScore>)> = candidates
            .iter()
            .copied()
            .filter(|&(q, _, _, _)| !processed[q as usize])
            .filter(|&(_, score, shared, chain)| {
                score >= threshold
                    && shared >= min_shared
                    && match (min_chain_containment, chain) {
                        (None, _) => true,
                        (Some(m), Some(c)) => c.containment >= m,
                        (Some(_), None) => false,
                    }
            })
            .map(|(q, score, _, chain)| (q, score, chain))
            .collect();
        absorbed.sort_by_key(|&(q, _, _)| q);

        for (q, score, chain) in absorbed {
            let qi = q as usize;
            if processed[qi] {
                continue;
            }
            processed[qi] = true;
            let member = &contigs[qi];
            rows.push(ClusterRow {
                rep_contig: seed.id.clone(),
                member_contig: member.id.clone(),
                source_mag: member.source_mag.clone(),
                containment: score,
                chain,
            });
        }
    }

    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ci(id: &str, length: u64) -> ContigInfo {
        ContigInfo {
            id: id.to_string(),
            source_mag: Some(format!("mag_{}", id)),
            length,
        }
    }

    fn edge(q: u32, t: u32, score: f64, shared: u64) -> ClusterEdge {
        ClusterEdge {
            query_idx: q,
            target_idx: t,
            score,
            shared,
            chain: None,
        }
    }

    fn edge_with_chain(
        q: u32,
        t: u32,
        score: f64,
        shared: u64,
        chain_containment: f64,
        chain_anchors: u32,
    ) -> ClusterEdge {
        ClusterEdge {
            query_idx: q,
            target_idx: t,
            score,
            shared,
            chain: Some(ChainScore {
                score: chain_anchors as f64,
                anchors: chain_anchors,
                containment: chain_containment,
                strand: crate::Strand::Forward,
            }),
        }
    }

    #[test]
    fn simple_absorption_into_longest() {
        let contigs = vec![ci("A", 5000), ci("B", 3000), ci("C", 2000), ci("D", 1500)];
        // B and C are well-contained in A; D unrelated.
        let edges = vec![
            edge(1, 0, 0.90, 270), // B->A
            edge(0, 1, 0.54, 270), // A->B (does not qualify; A is rep already anyway)
            edge(2, 0, 0.95, 190), // C->A
            edge(0, 2, 0.38, 190), // A->C
        ];

        let rows = greedy_dereplicate(&edges, &contigs, 0.85, 100, None);

        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0].rep_contig, "A");
        assert_eq!(rows[0].member_contig, "A");
        assert_eq!(rows[0].containment, 1.0);
        assert_eq!(rows[1].rep_contig, "A");
        assert_eq!(rows[1].member_contig, "B");
        assert!((rows[1].containment - 0.90).abs() < 1e-9);
        assert_eq!(rows[2].rep_contig, "A");
        assert_eq!(rows[2].member_contig, "C");
        assert!((rows[2].containment - 0.95).abs() < 1e-9);
        assert_eq!(rows[3].rep_contig, "D");
        assert_eq!(rows[3].member_contig, "D");
    }

    #[test]
    fn equal_length_tiebreak_is_index_ascending() {
        let contigs = vec![ci("A", 5000), ci("B", 5000), ci("C", 3000)];
        let edges = vec![edge(2, 0, 0.967, 290), edge(2, 1, 0.967, 290)];

        let rows = greedy_dereplicate(&edges, &contigs, 0.85, 100, None);

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].rep_contig, "A");
        assert_eq!(rows[0].member_contig, "A");
        assert_eq!(rows[1].rep_contig, "A");
        assert_eq!(rows[1].member_contig, "C");
        assert_eq!(rows[2].rep_contig, "B");
        assert_eq!(rows[2].member_contig, "B");
    }

    #[test]
    fn no_edges_makes_every_contig_its_own_rep() {
        let contigs = vec![ci("X", 5000), ci("Y", 3000), ci("Z", 1000)];
        let rows = greedy_dereplicate(&[], &contigs, 0.85, 100, None);

        assert_eq!(rows.len(), 3);
        for row in &rows {
            assert_eq!(row.rep_contig, row.member_contig);
            assert_eq!(row.containment, 1.0);
        }
        assert_eq!(rows[0].rep_contig, "X");
        assert_eq!(rows[1].rep_contig, "Y");
        assert_eq!(rows[2].rep_contig, "Z");
    }

    #[test]
    fn absorption_is_not_transitive_through_absorbed_targets() {
        let contigs = vec![ci("A", 5000), ci("B", 3000), ci("C", 2000)];
        let edges = vec![
            edge(1, 0, 0.90, 270), // B->A absorbs B
            edge(2, 1, 0.90, 180), // C->B qualifies but B no longer a rep
            edge(2, 0, 0.25, 50),  // C->A below threshold
        ];

        let rows = greedy_dereplicate(&edges, &contigs, 0.85, 100, None);

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].rep_contig, "A");
        assert_eq!(rows[1].member_contig, "B");
        assert_eq!(rows[2].rep_contig, "C");
        assert_eq!(rows[2].member_contig, "C");
    }

    #[test]
    fn below_threshold_does_not_absorb() {
        let contigs = vec![ci("A", 5000), ci("B", 3000)];
        let edges = vec![edge(1, 0, 0.667, 200)]; // score below T=0.85

        let rows = greedy_dereplicate(&edges, &contigs, 0.85, 100, None);

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].rep_contig, "A");
        assert_eq!(rows[1].rep_contig, "B");
    }

    #[test]
    fn below_min_shared_does_not_absorb_even_at_full_containment() {
        let contigs = vec![ci("A", 100_000), ci("B", 500)];
        let edges = vec![edge(1, 0, 1.0, 50)]; // shared=50 < min_shared=500

        let rows = greedy_dereplicate(&edges, &contigs, 0.85, 500, None);

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].rep_contig, "A");
        assert_eq!(rows[1].rep_contig, "B");
    }

    #[test]
    fn empty_input_returns_empty() {
        let rows = greedy_dereplicate(&[], &[], 0.85, 100, None);
        assert!(rows.is_empty());
    }

    #[test]
    fn self_edges_are_ignored() {
        let contigs = vec![ci("A", 5000)];
        let edges = vec![edge(0, 0, 1.0, 500)];

        let rows = greedy_dereplicate(&edges, &contigs, 0.85, 100, None);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].rep_contig, "A");
        assert_eq!(rows[0].member_contig, "A");
    }

    // ---- Plan 1.4 phase 3: chain gate tests ----
    //
    // WHY these tests: the chain gate is the new behavior Plan 1.4 layers on
    // top of set-containment. Each test pins one branch of the gate's
    // truth table (`min_chain_containment` × `edge.chain`) so a regression
    // in the filter expression can't slip past silently.

    #[test]
    fn greedy_chain_gate_disabled_passthrough() {
        // WHY: `min_chain_containment: None` must be a true no-op. Mixing
        // edges with and without chain data should match the legacy
        // (pre-Plan 1.4) absorption behavior — both get absorbed because the
        // chain field is ignored.
        let contigs = vec![ci("A", 5000), ci("B", 3000), ci("C", 2000)];
        let edges = vec![
            edge_with_chain(1, 0, 0.90, 270, 0.95, 100), // B->A with chain
            edge(2, 0, 0.95, 190),                       // C->A without chain
        ];

        let rows = greedy_dereplicate(&edges, &contigs, 0.85, 100, None);

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].member_contig, "A");
        assert_eq!(rows[1].rep_contig, "A");
        assert_eq!(rows[1].member_contig, "B");
        assert_eq!(rows[2].rep_contig, "A");
        assert_eq!(rows[2].member_contig, "C");
    }

    #[test]
    fn greedy_chain_gate_rejects_no_chain_present() {
        // WHY: When the gate is enabled, an edge with `chain: None` cannot
        // satisfy any containment threshold. The edge must be rejected even
        // though set-containment alone would have absorbed it.
        let contigs = vec![ci("A", 5000), ci("B", 3000)];
        let edges = vec![edge(1, 0, 0.99, 500)]; // would absorb under set-containment

        let rows = greedy_dereplicate(&edges, &contigs, 0.85, 100, Some(0.5));

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].rep_contig, "A");
        assert_eq!(rows[0].member_contig, "A");
        assert_eq!(rows[1].rep_contig, "B");
        assert_eq!(rows[1].member_contig, "B");
    }

    #[test]
    fn greedy_chain_gate_rejects_low_containment() {
        // WHY: set-containment may pass (score >= threshold) while chain
        // containment falls below the gate — the "shared minimizers
        // scattered without colinearity" case Plan 1.4 is designed to filter.
        let contigs = vec![ci("A", 5000), ci("B", 3000)];
        let edges = vec![edge_with_chain(1, 0, 0.99, 500, 0.30, 20)];

        let rows = greedy_dereplicate(&edges, &contigs, 0.85, 100, Some(0.5));

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].member_contig, "A");
        assert_eq!(rows[1].member_contig, "B");
        assert_eq!(rows[1].rep_contig, "B");
    }

    #[test]
    fn greedy_chain_gate_accepts_high_containment() {
        // WHY: When chain containment is above the gate, the edge is
        // absorbed — confirming the gate is a filter not a hard block.
        let contigs = vec![ci("A", 5000), ci("B", 3000)];
        let edges = vec![edge_with_chain(1, 0, 0.99, 500, 0.80, 100)];

        let rows = greedy_dereplicate(&edges, &contigs, 0.85, 100, Some(0.5));

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].member_contig, "A");
        assert_eq!(rows[1].rep_contig, "A");
        assert_eq!(rows[1].member_contig, "B");
        assert!((rows[1].containment - 0.99).abs() < 1e-9);
    }

    #[test]
    fn greedy_chain_does_not_change_tiebreak_order() {
        // WHY: The chain field is a filter, not a ranker. Two absorbed
        // members must still appear in `contigs`-index order, regardless of
        // their chain containment. Sorting by chain would silently break the
        // documented contract at the top of `greedy_dereplicate`.
        let contigs = vec![ci("A", 5000), ci("B", 3000), ci("C", 2000)];
        let edges = vec![
            edge_with_chain(1, 0, 0.90, 270, 0.55, 30), // B->A, low chain
            edge_with_chain(2, 0, 0.95, 190, 0.99, 200), // C->A, high chain
        ];

        let rows = greedy_dereplicate(&edges, &contigs, 0.85, 100, Some(0.5));

        // Both absorbed; order is by query_idx (B before C), not by chain.
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].member_contig, "A");
        assert_eq!(rows[1].member_contig, "B");
        assert_eq!(rows[2].member_contig, "C");
    }

    // ---- Plan 1.5 phase 1: chain forwarding into ClusterRow ----
    //
    // WHY these tests: Plan 1.4 left chain visible only on `ClusterEdge`;
    // Plan 1.5 forwards it through `greedy_dereplicate` into `ClusterRow`
    // so downstream surfaces (Arrow / Parquet CLI / C-API) can surface
    // chain per row. These tests pin the forwarding contract at the
    // single conversion site that downstream code depends on.

    #[test]
    fn greedy_forwards_chain_into_absorbed_rows() {
        // WHY: An edge with `chain: Some(_)` must propagate its chain
        // payload into the absorbed-member row. Without this forwarding,
        // every downstream surface would see `chain: None` on every row
        // regardless of whether chain DP ran.
        let contigs = vec![ci("A", 5000), ci("B", 3000)];
        let edges = vec![edge_with_chain(1, 0, 0.95, 270, 0.91, 88)];

        let rows = greedy_dereplicate(&edges, &contigs, 0.85, 100, None);

        // Find the absorbed row for B.
        let b_row = rows
            .iter()
            .find(|r| r.member_contig == "B" && r.rep_contig == "A")
            .expect("B should be absorbed into A");
        let chain = b_row
            .chain
            .as_ref()
            .expect("absorbed row must carry the edge's chain");
        assert_eq!(chain.anchors, 88);
        assert!((chain.containment - 0.91).abs() < 1e-9);
    }

    #[test]
    fn greedy_representative_rows_have_no_chain() {
        // WHY: A representative row (rep == member) has no edge driving
        // its absorption; chain is semantically inapplicable. Even when
        // ALL input edges carry `chain: Some(_)`, the rep row must emit
        // `chain: None`. Downstream surfaces rely on this to know which
        // rows can sensibly carry chain output.
        let contigs = vec![ci("A", 5000), ci("B", 3000)];
        let edges = vec![edge_with_chain(1, 0, 0.95, 270, 0.91, 88)];

        let rows = greedy_dereplicate(&edges, &contigs, 0.85, 100, None);

        let a_row = rows
            .iter()
            .find(|r| r.rep_contig == "A" && r.member_contig == "A")
            .expect("A should be its own rep");
        assert!(
            a_row.chain.is_none(),
            "representative row must have chain: None, got {:?}",
            a_row.chain
        );
    }

    #[test]
    fn greedy_rejected_absorption_does_not_leak_chain() {
        // WHY: When the chain gate rejects a candidate, the candidate
        // must NOT appear in the output rows at all (it becomes its own
        // representative with `chain: None`). Plan 1.4's gate is the
        // critical filter; a regression that "rejected but still emitted
        // the absorbed row with a chain payload" would silently break
        // the gate's purpose. Guards against future refactors that
        // accidentally treat gate-rejected candidates as absorbed.
        let contigs = vec![ci("A", 5000), ci("B", 3000)];
        // chain.containment = 0.30 < gate (0.5) → rejected.
        let edges = vec![edge_with_chain(1, 0, 0.99, 500, 0.30, 20)];

        let rows = greedy_dereplicate(&edges, &contigs, 0.85, 100, Some(0.5));

        // B must appear as its OWN rep, not absorbed into A.
        let b_row = rows
            .iter()
            .find(|r| r.member_contig == "B")
            .expect("B must appear in output");
        assert_eq!(
            b_row.rep_contig, "B",
            "rejected candidate must be its own rep, not absorbed"
        );
        assert!(
            b_row.chain.is_none(),
            "rejected-candidate row must have chain: None (it's a rep), got {:?}",
            b_row.chain
        );
    }
}
