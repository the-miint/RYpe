//! Length-sorted greedy dereplication.
//!
//! Pure function over a precomputed edge list and per-contig metadata.
//! Does not allocate workspaces or perform I/O.

use std::collections::HashMap;

use super::containment::{containment, edge_qualifies};
use super::types::{ClusterEdge, ClusterRow, ContigInfo};

/// Run length-sorted greedy dereplication.
///
/// # Arguments
/// * `edges` - Sparse (query, target, shared) edges. Self-edges are tolerated
///   but ignored. Order is irrelevant; the function indexes internally.
/// * `contigs` - All contigs participating in clustering. Indices into this
///   slice are referenced by [`ClusterEdge`].
/// * `threshold` - Minimum containment of a candidate in its seed to absorb.
/// * `min_shared` - Minimum absolute shared-minimizer count to absorb.
///
/// # Output
/// One [`ClusterRow`] per input contig (complete partition). Representatives
/// appear with `rep_contig == member_contig` and `containment == 1.0`.
/// Within each representative, absorbed members are sorted by their original
/// `contigs` index for deterministic output.
pub fn greedy_dereplicate(
    edges: &[ClusterEdge],
    contigs: &[ContigInfo],
    threshold: f64,
    min_shared: u64,
) -> Vec<ClusterRow> {
    let n = contigs.len();
    if n == 0 {
        return Vec::new();
    }

    let mut edges_by_target: HashMap<u32, Vec<(u32, u64)>> = HashMap::new();
    for e in edges {
        if e.query_idx == e.target_idx {
            continue;
        }
        edges_by_target
            .entry(e.target_idx)
            .or_default()
            .push((e.query_idx, e.shared));
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

        let Some(candidates) = edges_by_target.get(&seed_idx) else {
            continue;
        };

        let mut absorbed: Vec<(u32, u64)> = candidates
            .iter()
            .copied()
            .filter(|&(q, _)| !processed[q as usize])
            .filter(|&(q, shared)| {
                edge_qualifies(
                    shared,
                    contigs[q as usize].mins_count,
                    threshold,
                    min_shared,
                )
            })
            .collect();
        absorbed.sort_by_key(|&(q, _)| q);

        for (q, shared) in absorbed {
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
                containment: containment(shared, member.mins_count),
            });
        }
    }

    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ci(id: &str, length: u64, mins_count: u64) -> ContigInfo {
        ContigInfo {
            id: id.to_string(),
            source_mag: Some(format!("mag_{}", id)),
            length,
            mins_count,
        }
    }

    fn edge(q: u32, t: u32, shared: u64) -> ClusterEdge {
        ClusterEdge {
            query_idx: q,
            target_idx: t,
            shared,
        }
    }

    #[test]
    fn simple_absorption_into_longest() {
        // A=5000, B=3000, C=2000, D=1500
        let contigs = vec![
            ci("A", 5000, 500),
            ci("B", 3000, 300),
            ci("C", 2000, 200),
            ci("D", 1500, 150),
        ];
        // B and C are well-contained in A; D unrelated.
        let edges = vec![
            edge(1, 0, 270), // B->A: C(B,A) = 270/300 = 0.90
            edge(0, 1, 270), // A->B: C(A,B) = 270/500 = 0.54 (no absorb, A is rep already)
            edge(2, 0, 190), // C->A: C(C,A) = 190/200 = 0.95
            edge(0, 2, 190), // A->C: C(A,C) = 190/500 = 0.38
        ];

        let rows = greedy_dereplicate(&edges, &contigs, 0.85, 100);

        assert_eq!(rows.len(), 4);
        // A processed first: self-row, then absorbed B (idx 1) before C (idx 2)
        assert_eq!(rows[0].rep_contig, "A");
        assert_eq!(rows[0].member_contig, "A");
        assert_eq!(rows[0].containment, 1.0);
        assert_eq!(rows[1].rep_contig, "A");
        assert_eq!(rows[1].member_contig, "B");
        assert!((rows[1].containment - 0.90).abs() < 1e-9);
        assert_eq!(rows[2].rep_contig, "A");
        assert_eq!(rows[2].member_contig, "C");
        assert!((rows[2].containment - 0.95).abs() < 1e-9);
        // D becomes its own rep
        assert_eq!(rows[3].rep_contig, "D");
        assert_eq!(rows[3].member_contig, "D");
    }

    #[test]
    fn equal_length_tiebreak_is_index_ascending() {
        // A and B both length 5000; C is shorter and absorbed by whoever is processed first.
        let contigs = vec![ci("A", 5000, 500), ci("B", 5000, 500), ci("C", 3000, 300)];
        let edges = vec![
            edge(2, 0, 290), // C->A: 290/300 = 0.967
            edge(2, 1, 290), // C->B: 290/300 = 0.967
        ];

        let rows = greedy_dereplicate(&edges, &contigs, 0.85, 100);

        // Sort order: A (idx 0) before B (idx 1) -> A absorbs C
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
        let contigs = vec![ci("X", 5000, 500), ci("Y", 3000, 300), ci("Z", 1000, 100)];
        let rows = greedy_dereplicate(&[], &contigs, 0.85, 100);

        assert_eq!(rows.len(), 3);
        for row in &rows {
            assert_eq!(row.rep_contig, row.member_contig);
            assert_eq!(row.containment, 1.0);
        }
        // Order: X, Y, Z (length descending)
        assert_eq!(rows[0].rep_contig, "X");
        assert_eq!(rows[1].rep_contig, "Y");
        assert_eq!(rows[2].rep_contig, "Z");
    }

    #[test]
    fn absorption_is_not_transitive_through_absorbed_targets() {
        // A=5000 absorbs B=3000 (B contained in A). C=2000 is contained in B
        // but NOT in A directly. C should become its own rep, NOT chase
        // through the absorbed B to land in A's cluster.
        let contigs = vec![ci("A", 5000, 500), ci("B", 3000, 300), ci("C", 2000, 200)];
        let edges = vec![
            edge(1, 0, 270), // B->A: 0.90 (absorbs B into A)
            edge(2, 1, 180), // C->B: 0.90 (qualifies, but B is no longer a rep)
            edge(2, 0, 50),  // C->A: 0.25 (does not qualify; ignored)
        ];

        let rows = greedy_dereplicate(&edges, &contigs, 0.85, 100);

        assert_eq!(rows.len(), 3);
        // A absorbs B
        assert_eq!(rows[0].rep_contig, "A");
        assert_eq!(rows[0].member_contig, "A");
        assert_eq!(rows[1].rep_contig, "A");
        assert_eq!(rows[1].member_contig, "B");
        // C becomes its own rep — NOT absorbed by B (already absorbed) and
        // NOT chained transitively into A
        assert_eq!(rows[2].rep_contig, "C");
        assert_eq!(rows[2].member_contig, "C");
    }

    #[test]
    fn below_threshold_does_not_absorb() {
        let contigs = vec![ci("A", 5000, 500), ci("B", 3000, 300)];
        let edges = vec![edge(1, 0, 200)]; // C(B,A) = 0.667, below T=0.85

        let rows = greedy_dereplicate(&edges, &contigs, 0.85, 100);

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].rep_contig, "A");
        assert_eq!(rows[1].rep_contig, "B");
    }

    #[test]
    fn below_min_shared_does_not_absorb_even_at_full_containment() {
        // Tiny contig fully contained but with very few shared minimizers —
        // the mobile-element defense.
        let contigs = vec![ci("A", 100000, 10000), ci("B", 500, 50)];
        let edges = vec![edge(1, 0, 50)]; // C(B,A) = 1.0 but shared=50 < 500

        let rows = greedy_dereplicate(&edges, &contigs, 0.85, 500);

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].rep_contig, "A");
        assert_eq!(rows[1].rep_contig, "B"); // not absorbed
    }

    #[test]
    fn empty_input_returns_empty() {
        let rows = greedy_dereplicate(&[], &[], 0.85, 100);
        assert!(rows.is_empty());
    }

    #[test]
    fn self_edges_are_ignored() {
        let contigs = vec![ci("A", 5000, 500)];
        let edges = vec![edge(0, 0, 500)];

        let rows = greedy_dereplicate(&edges, &contigs, 0.85, 100);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].rep_contig, "A");
        assert_eq!(rows[0].member_contig, "A");
    }
}
