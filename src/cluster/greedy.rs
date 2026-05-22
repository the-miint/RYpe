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
        });

        let candidates = &edges_by_target[si];
        if candidates.is_empty() {
            continue;
        }

        let mut absorbed: Vec<(u32, f64)> = candidates
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
            .map(|(q, score, _, _)| (q, score))
            .collect();
        absorbed.sort_by_key(|&(q, _)| q);

        for (q, score) in absorbed {
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

    #[allow(dead_code)] // Phase 3 tests use this. Allow transient unused-ness.
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
}
