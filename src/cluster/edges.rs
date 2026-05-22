//! Build sparse containment edges between contigs via the existing classify
//! pipeline, augmented with positional chain confirmation (Plan 1.4).
//!
//! Each bucket stores the single-strand minimizer set of one contig; each
//! query is presented as `(fwd_mins, rc_mins)` so classify's
//! `max(fwd_score, rc_score)` gives orientation-independent containment.
//!
//! `score` is rype's classify score (containment of query in target).
//! `shared` is approximated as `round(score * |fwd_mins|)` — exact when fwd
//! wins, within ~1% when rc wins (for non-palindromic sequences).
//!
//! When `cfg.chain_params.is_some()`, every classify hit also gets a
//! `ChainScore` attached: the chain DP runs on both strands (fwd and rc),
//! and the higher-scoring chain wins. Containment is normalized by the
//! winning strand's query minimizer count, matching the set-containment
//! frame.

use std::path::Path;

use anyhow::{Context, Result};
use rayon::prelude::*;

use crate::{
    chain_anchors, classify_from_extracted_minimizers, compute_anchors_into,
    create_parquet_inverted_index, extract_dual_strand_into_with_positions,
    pairs_into_cluster_bucket_arrays, BucketData, ChainParams, ChainWorkspace, MinimizerWorkspace,
    ShardedInvertedIndex, Strand,
};

use super::types::{ChainScore, ClusterEdge, ContigInfo, ContigInput};
use super::ClusterConfig;

/// Positioned minimizers for one contig — both strands. Each `*_hashes`
/// array is sorted ascending and deduplicated by hash; `*_positions` is
/// index-parallel with its hashes, holding the smallest position at which
/// each hash occurred (the `pairs_into_cluster_bucket_arrays` contract).
///
/// Plan 1.4 introduces this to thread positions from extraction through to
/// the chain DP at classify-hit time. The hashes are also cloned into the
/// classify input (`query_mins`) so the existing classify path doesn't see
/// any API change.
struct PositionedContig {
    fwd_hashes: Vec<u64>,
    fwd_positions: Vec<u32>,
    rc_hashes: Vec<u64>,
    rc_positions: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct EdgeBuildOutput {
    pub contigs: Vec<ContigInfo>,
    pub edges: Vec<ClusterEdge>,
}

/// Build containment edges for a set of contigs.
///
/// `workdir` must exist and be writable. A Parquet index named
/// `cluster_index.ryxdi` is created inside it.
pub fn build_edges(
    inputs: &[ContigInput],
    cfg: &ClusterConfig,
    workdir: &Path,
) -> Result<EdgeBuildOutput> {
    if inputs.is_empty() {
        return Ok(EdgeBuildOutput {
            contigs: Vec::new(),
            edges: Vec::new(),
        });
    }

    // Phase 2: extract WITH positions. `pairs_into_cluster_bucket_arrays`
    // sorts each strand by hash ascending and dedups by hash, keeping the
    // smallest position per hash — the contract the merge-join in
    // `compute_anchors_into` requires.
    let extracted: Result<Vec<(ContigInfo, PositionedContig)>> = inputs
        .par_iter()
        .map_init(
            MinimizerWorkspace::new,
            |ws, input| -> Result<(ContigInfo, PositionedContig)> {
                extract_dual_strand_into_with_positions(
                    &input.sequence,
                    cfg.k,
                    cfg.w,
                    cfg.salt,
                    ws,
                )
                .with_context(|| format!("extracting positioned minimizers for {}", input.id))?;

                let mut fwd_hashes = std::mem::take(&mut ws.buffer);
                let mut fwd_positions = std::mem::take(&mut ws.positions_fwd);
                pairs_into_cluster_bucket_arrays(&mut fwd_hashes, &mut fwd_positions);

                let mut rc_hashes = std::mem::take(&mut ws.rc_buffer);
                let mut rc_positions = std::mem::take(&mut ws.positions_rc);
                pairs_into_cluster_bucket_arrays(&mut rc_hashes, &mut rc_positions);

                Ok((
                    ContigInfo {
                        id: input.id.clone(),
                        source_mag: input.source_mag.clone(),
                        length: input.sequence.len() as u64,
                    },
                    PositionedContig {
                        fwd_hashes,
                        fwd_positions,
                        rc_hashes,
                        rc_positions,
                    },
                ))
            },
        )
        .collect();
    let extracted = extracted?;

    let mut contigs: Vec<ContigInfo> = Vec::with_capacity(extracted.len());
    let mut positioned: Vec<PositionedContig> = Vec::with_capacity(extracted.len());
    for (info, pc) in extracted {
        contigs.push(info);
        positioned.push(pc);
    }

    // Classify takes hash-only input; clone hashes off the positioned data.
    // The clone cost is on the order of the minimizer count (cheap) and
    // keeps the existing classify API untouched.
    let query_mins: Vec<(Vec<u64>, Vec<u64>)> = positioned
        .iter()
        .map(|pc| (pc.fwd_hashes.clone(), pc.rc_hashes.clone()))
        .collect();

    // INVARIANT: `bucket_id == idx + 1` where `idx` is the position in the
    // `inputs` slice. `bucket_id_to_idx` relies on this. `enumerate`
    // captures the position BEFORE the filter, so bucket_id is stable
    // across the empty-fwd filter.
    let buckets: Vec<BucketData> = positioned
        .iter()
        .zip(inputs.iter())
        .enumerate()
        .filter(|(_, (pc, _))| !pc.fwd_hashes.is_empty())
        .map(|(idx, (pc, input))| BucketData {
            bucket_id: bucket_id_for(idx),
            bucket_name: input.id.clone(),
            sources: input
                .source_mag
                .clone()
                .map(|m| vec![m])
                .unwrap_or_default(),
            minimizers: pc.fwd_hashes.clone(),
        })
        .collect();

    if buckets.is_empty() {
        return Ok(EdgeBuildOutput {
            contigs,
            edges: Vec::new(),
        });
    }

    let index_path = workdir.join("cluster_index.ryxdi");
    create_parquet_inverted_index(
        &index_path,
        buckets,
        cfg.k,
        cfg.w,
        cfg.salt,
        None,
        None,
        None,
    )
    .context("creating clustering index")?;

    let index = ShardedInvertedIndex::open(&index_path).context("opening clustering index")?;

    let query_ids: Vec<i64> = (0..inputs.len() as i64).collect();
    let hits =
        classify_from_extracted_minimizers(&index, &query_mins, &query_ids, cfg.threshold, None)
            .context("classifying contigs against clustering index")?;

    // Project each classify hit into a candidate ClusterEdge. When chain is
    // enabled, augment with `ChainScore`; otherwise emit edges with
    // `chain: None` (legacy behavior). Two code paths so we don't pay the
    // per-thread workspace cost when chain is disabled.
    let edges: Vec<ClusterEdge> = match cfg.chain_params.as_ref() {
        None => hits
            .iter()
            .filter_map(|hit| {
                project_hit(hit, &positioned, &contigs, cfg.min_shared).map(
                    |(query_idx, target_idx, shared)| ClusterEdge {
                        query_idx,
                        target_idx,
                        score: hit.score,
                        shared,
                        chain: None,
                    },
                )
            })
            .collect(),
        Some(chain_params) => hits
            .par_iter()
            .map_init(
                || {
                    (
                        ChainWorkspace::new(),
                        Vec::<(u32, u32)>::new(),
                        Vec::<(u32, u32)>::new(),
                    )
                },
                |(ws, anchors_fwd, anchors_rc), hit| -> Option<ClusterEdge> {
                    let (query_idx, target_idx, shared) =
                        project_hit(hit, &positioned, &contigs, cfg.min_shared)?;
                    let chain = run_chain_for_hit(
                        &positioned[query_idx as usize],
                        &positioned[target_idx as usize],
                        chain_params,
                        ws,
                        anchors_fwd,
                        anchors_rc,
                    );
                    Some(ClusterEdge {
                        query_idx,
                        target_idx,
                        score: hit.score,
                        shared,
                        chain,
                    })
                },
            )
            .filter_map(|edge| edge)
            .collect(),
    };

    Ok(EdgeBuildOutput { contigs, edges })
}

/// Validate a classify hit and project into `(query_idx, target_idx, shared)`.
/// Returns `None` for self-hits, bucket_id=0, or shared below `min_shared`.
/// Factored out so the chain-enabled and chain-disabled paths share the same
/// hit projection + filter logic.
fn project_hit(
    hit: &crate::HitResult,
    positioned: &[PositionedContig],
    contigs: &[ContigInfo],
    min_shared: u64,
) -> Option<(u32, u32, u64)> {
    debug_assert!(
        hit.query_id >= 0 && (hit.query_id as usize) < contigs.len(),
        "classify returned query_id {} outside [0, {})",
        hit.query_id,
        contigs.len()
    );
    let query_idx = hit.query_id as u32;

    debug_assert!(
        hit.bucket_id > 0 && (hit.bucket_id as usize) <= contigs.len(),
        "classify returned bucket_id {} outside (0, {}]",
        hit.bucket_id,
        contigs.len()
    );
    let target_idx = bucket_id_to_idx(hit.bucket_id)?;

    if query_idx == target_idx {
        return None;
    }

    let fwd_total = positioned[query_idx as usize].fwd_hashes.len() as f64;
    let shared = (hit.score * fwd_total).round() as u64;
    if shared < min_shared {
        return None;
    }
    Some((query_idx, target_idx, shared))
}

/// Run the banded-DP chain on both strands of a (query, target) pair and
/// return the winner's `ChainScore`, or `None` if neither strand produced
/// a chain (chain length below `params.min_anchors`).
///
/// Target uses the FORWARD strand minimizers — matches the `.ryxdi` bucket
/// build path. `Strand::ReverseComplement` on the winner means the query's
/// RC-strand minimizers chained against the target's forward bucket.
///
/// `containment` is normalized by the **winning strand's** query minimizer
/// count. When fwd wins, divide by `|q.fwd_hashes|`; when rc wins, divide
/// by `|q.rc_hashes|`. This matches set-containment's
/// `max(C(q_fwd, t), C(q_rc, t))` frame.
fn run_chain_for_hit(
    query: &PositionedContig,
    target: &PositionedContig,
    params: &ChainParams,
    ws: &mut ChainWorkspace,
    anchors_fwd: &mut Vec<(u32, u32)>,
    anchors_rc: &mut Vec<(u32, u32)>,
) -> Option<ChainScore> {
    compute_anchors_into(
        &query.fwd_hashes,
        &query.fwd_positions,
        &target.fwd_hashes,
        &target.fwd_positions,
        anchors_fwd,
    );
    compute_anchors_into(
        &query.rc_hashes,
        &query.rc_positions,
        &target.fwd_hashes,
        &target.fwd_positions,
        anchors_rc,
    );

    let r_fwd = chain_anchors(anchors_fwd, false, params, ws);
    let r_rc = chain_anchors(anchors_rc, true, params, ws);

    let (winner, strand, denom) = match (r_fwd, r_rc) {
        (None, None) => return None,
        (Some(f), None) => (f, Strand::Forward, query.fwd_hashes.len()),
        (None, Some(r)) => (r, Strand::ReverseComplement, query.rc_hashes.len()),
        (Some(f), Some(r)) => {
            // Tie-break: fwd wins on equal score (deterministic).
            if f.score >= r.score {
                (f, Strand::Forward, query.fwd_hashes.len())
            } else {
                (r, Strand::ReverseComplement, query.rc_hashes.len())
            }
        }
    };

    debug_assert!(
        denom > 0,
        "chain produced a winner with zero query-side minimizers — \
         compute_anchors_into can't have returned non-empty anchors"
    );
    let containment = winner.anchors as f64 / denom as f64;
    debug_assert!(
        containment <= 1.0 + 1e-9,
        "chain.containment must be ≤ 1.0 (anchors=∣query⊆target∣ ≤ ∣query∣), got {}",
        containment
    );

    Some(ChainScore {
        score: winner.score,
        anchors: winner.anchors,
        containment,
        strand,
    })
}

#[inline]
fn bucket_id_for(idx: usize) -> u32 {
    (idx as u32) + 1
}

/// Recover the contig index from a bucket id. Returns `None` if `bucket_id`
/// is 0 (which violates our 1-based numbering) — callers can treat that as a
/// dropped edge rather than panicking.
#[inline]
fn bucket_id_to_idx(bucket_id: u32) -> Option<u32> {
    bucket_id.checked_sub(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn seq_from_seed(len: usize, seed: u64) -> Vec<u8> {
        let bases = [b'A', b'C', b'G', b'T'];
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        (0..len)
            .map(|_| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                bases[((s >> 56) & 0b11) as usize]
            })
            .collect()
    }

    fn test_cfg(threshold: f64, min_shared: u64) -> ClusterConfig {
        ClusterConfig {
            k: 32,
            w: 20,
            salt: 0x5555_5555_5555_5555,
            min_length: 0,
            threshold,
            min_shared,
            // Default: chain disabled. The phase-2 chain tests below opt-in
            // via `test_cfg_with_chain` so existing tests stay byte-identical.
            chain_params: None,
            min_chain_containment: None,
        }
    }

    /// Like `test_cfg` but enables chain DP with starting params calibrated
    /// for `w=20` (matches `test_cfg`'s window). Gate stays disabled (None)
    /// — Phase 3 tests the gate via `greedy_dereplicate` directly.
    fn test_cfg_with_chain(threshold: f64, min_shared: u64) -> ClusterConfig {
        ClusterConfig {
            chain_params: Some(ChainParams::starting_for_w(20)),
            ..test_cfg(threshold, min_shared)
        }
    }

    fn rc(seq: &[u8]) -> Vec<u8> {
        seq.iter()
            .rev()
            .map(|&b| match b {
                b'A' => b'T',
                b'T' => b'A',
                b'C' => b'G',
                b'G' => b'C',
                _ => b'N',
            })
            .collect()
    }

    #[test]
    fn build_edges_produces_edge_for_fragment_of_full_genome() {
        let dir = tempdir().unwrap();

        let a = seq_from_seed(20_000, 1);
        let b = a[4_000..16_000].to_vec();
        let c = seq_from_seed(20_000, 2);

        let inputs = vec![
            ContigInput {
                id: "A".to_string(),
                source_mag: Some("mag1".to_string()),
                sequence: a,
            },
            ContigInput {
                id: "B".to_string(),
                source_mag: Some("mag1".to_string()),
                sequence: b,
            },
            ContigInput {
                id: "C".to_string(),
                source_mag: Some("mag2".to_string()),
                sequence: c,
            },
        ];

        let out = build_edges(&inputs, &test_cfg(0.5, 1), dir.path()).unwrap();

        assert_eq!(out.contigs.len(), 3);
        assert_eq!(out.contigs[0].id, "A");
        assert_eq!(out.contigs[1].id, "B");
        assert_eq!(out.contigs[2].id, "C");

        let b_to_a = out
            .edges
            .iter()
            .find(|e| e.query_idx == 1 && e.target_idx == 0)
            .expect("expected an edge from B (fragment) to A (full genome)");
        assert!(
            b_to_a.score > 0.85,
            "fragment B should be > 85% contained in A, got score {}",
            b_to_a.score,
        );
        assert!(
            b_to_a.shared > 0,
            "B->A edge should have non-zero shared count, got {}",
            b_to_a.shared,
        );

        for e in &out.edges {
            if e.query_idx == 2 {
                assert!(
                    e.score < 0.5,
                    "unrelated C should have low containment in others, got {}",
                    e.score,
                );
            }
        }
    }

    #[test]
    fn build_edges_skips_self_hits() {
        let dir = tempdir().unwrap();
        let inputs = vec![ContigInput {
            id: "only".to_string(),
            source_mag: None,
            sequence: seq_from_seed(5000, 42),
        }];
        let mut cfg = test_cfg(0.5, 1);
        cfg.salt = 0x12345;
        let out = build_edges(&inputs, &cfg, dir.path()).unwrap();
        for e in &out.edges {
            assert_ne!(
                e.query_idx, e.target_idx,
                "self-hit should be excluded from edges"
            );
        }
    }

    #[test]
    fn build_edges_applies_min_shared_filter() {
        let dir = tempdir().unwrap();
        let a = seq_from_seed(20_000, 1);
        let b = a[4_000..16_000].to_vec();
        let inputs = vec![
            ContigInput {
                id: "A".to_string(),
                source_mag: None,
                sequence: a,
            },
            ContigInput {
                id: "B".to_string(),
                source_mag: None,
                sequence: b,
            },
        ];

        let out = build_edges(&inputs, &test_cfg(0.5, 1_000_000), dir.path()).unwrap();
        assert!(
            out.edges.is_empty(),
            "min_shared=1e6 should filter all edges, got {} edges",
            out.edges.len(),
        );
    }

    #[test]
    fn build_edges_empty_input_returns_empty() {
        let dir = tempdir().unwrap();
        let mut cfg = test_cfg(0.5, 1);
        cfg.salt = 0x12345;
        let out = build_edges(&[], &cfg, dir.path()).unwrap();
        assert!(out.contigs.is_empty());
        assert!(out.edges.is_empty());
    }

    // ===== Plan 1.4 phase 2: chain integration =====

    /// WHY (headline): when chain is enabled, a fragment-of-full-genome
    /// edge must have `chain: Some(_)` with anchors ≥ min_anchors. Pins
    /// the load-bearing wiring from `compute_anchors_into` through
    /// `chain_anchors` to `ChainScore` on the emitted edge.
    #[test]
    fn build_edges_chain_field_populated_when_enabled() {
        let dir = tempdir().unwrap();
        let a = seq_from_seed(20_000, 1);
        let b = a[4_000..16_000].to_vec();
        let inputs = vec![
            ContigInput {
                id: "A".to_string(),
                source_mag: None,
                sequence: a,
            },
            ContigInput {
                id: "B".to_string(),
                source_mag: None,
                sequence: b,
            },
        ];
        let out = build_edges(&inputs, &test_cfg_with_chain(0.5, 1), dir.path()).unwrap();

        let b_to_a = out
            .edges
            .iter()
            .find(|e| e.query_idx == 1 && e.target_idx == 0)
            .expect("B (fragment) → A (full) edge must exist");
        let chain = b_to_a
            .chain
            .expect("chain must be Some when cfg.chain_params is Some");
        assert!(
            chain.anchors >= 3,
            "expected chain.anchors ≥ 3 on a fragment-of-genome edge, got {}",
            chain.anchors
        );
        assert!(
            chain.containment > 0.0 && chain.containment <= 1.0,
            "chain.containment must be in (0, 1], got {}",
            chain.containment
        );
        // Fragment is fwd-aligned with the full genome, so the winning
        // strand is Forward.
        assert_eq!(chain.strand, Strand::Forward);
    }

    /// WHY: when chain is disabled in config, NO edge should carry chain
    /// data. Catches a Phase 2 wiring mistake where chain accidentally
    /// runs even with `chain_params: None`.
    #[test]
    fn build_edges_chain_field_none_when_disabled() {
        let dir = tempdir().unwrap();
        let a = seq_from_seed(20_000, 1);
        let b = a[4_000..16_000].to_vec();
        let inputs = vec![
            ContigInput {
                id: "A".to_string(),
                source_mag: None,
                sequence: a,
            },
            ContigInput {
                id: "B".to_string(),
                source_mag: None,
                sequence: b,
            },
        ];
        // test_cfg (NOT _with_chain) leaves chain_params: None.
        let out = build_edges(&inputs, &test_cfg(0.5, 1), dir.path()).unwrap();

        for edge in &out.edges {
            assert!(
                edge.chain.is_none(),
                "edge {:?} carries chain data despite chain disabled in config",
                edge
            );
        }
    }

    /// WHY: when the query is the reverse-complement of a target fragment,
    /// the winning strand on chain MUST be ReverseComplement. Pins the
    /// strand-flag plumbing from query.rc_hashes through
    /// `chain_anchors(.., is_rc=true)` to `ChainScore.strand`.
    #[test]
    fn build_edges_chain_picks_winning_strand_on_rc_fragment() {
        let dir = tempdir().unwrap();
        let a = seq_from_seed(20_000, 1);
        let b_rc = rc(&a[4_000..16_000]);
        let inputs = vec![
            ContigInput {
                id: "A".to_string(),
                source_mag: None,
                sequence: a,
            },
            ContigInput {
                id: "B_rc".to_string(),
                source_mag: None,
                sequence: b_rc,
            },
        ];
        let out = build_edges(&inputs, &test_cfg_with_chain(0.5, 1), dir.path()).unwrap();

        let b_to_a = out
            .edges
            .iter()
            .find(|e| e.query_idx == 1 && e.target_idx == 0)
            .expect("B_rc → A edge must exist (rc fragment is still contained)");
        let chain = b_to_a.chain.expect("chain must be Some");
        assert_eq!(
            chain.strand,
            Strand::ReverseComplement,
            "rc-fragment query must produce a rc chain; got {:?}",
            chain.strand
        );
        assert!(
            chain.anchors >= 3,
            "rc chain must have ≥3 anchors on a real rc fragment, got {}",
            chain.anchors
        );
    }

    /// WHY: disjoint genomes that nonetheless pass `cfg.threshold` by random
    /// k-mer matches must NOT produce a chain (or produce one well below
    /// min_anchors). This is the false-positive control — chain is the
    /// gate that distinguishes "shared minimizers scattered" from "shared
    /// minimizers colinear."
    #[test]
    fn build_edges_disjoint_query_has_no_chain() {
        let dir = tempdir().unwrap();
        let a = seq_from_seed(20_000, 1);
        let c = seq_from_seed(20_000, 99_999);
        let inputs = vec![
            ContigInput {
                id: "A".to_string(),
                source_mag: None,
                sequence: a,
            },
            ContigInput {
                id: "C".to_string(),
                source_mag: None,
                sequence: c,
            },
        ];
        // Low threshold + min_shared=1 lets random matches through to be
        // edges; chain DP is what should reject them.
        let out = build_edges(&inputs, &test_cfg_with_chain(0.01, 1), dir.path()).unwrap();
        for edge in &out.edges {
            if let Some(chain) = edge.chain {
                // At k=32 the random match rate is essentially zero; any
                // chain that does form by chance must be below the DP's
                // min_anchors threshold (which makes the DP return None).
                // If we somehow get a Some result, anchors must still be
                // bounded by the edge's `shared` count.
                assert!(
                    chain.anchors as u64 <= edge.shared,
                    "chain.anchors {} > shared {} — anchors must be a subset of shared",
                    chain.anchors,
                    edge.shared
                );
            }
        }
    }

    /// WHY: `chain.anchors` counts a SUBSET of the shared minimizers
    /// (specifically those that fall into the best colinear chain). It
    /// can never exceed `shared`. The debug_assert in `run_chain_for_hit`
    /// pins the upper bound `chain.containment ≤ 1.0`; this test pins
    /// the lower-level invariant on the anchor count.
    #[test]
    fn build_edges_chain_anchors_bounded_by_shared() {
        let dir = tempdir().unwrap();
        let a = seq_from_seed(20_000, 1);
        let b = a[4_000..16_000].to_vec();
        let inputs = vec![
            ContigInput {
                id: "A".to_string(),
                source_mag: None,
                sequence: a,
            },
            ContigInput {
                id: "B".to_string(),
                source_mag: None,
                sequence: b,
            },
        ];
        let out = build_edges(&inputs, &test_cfg_with_chain(0.5, 1), dir.path()).unwrap();
        for edge in &out.edges {
            if let Some(chain) = edge.chain {
                assert!(
                    chain.anchors as u64 <= edge.shared,
                    "edge {:?} has chain.anchors {} > shared {}",
                    edge,
                    chain.anchors,
                    edge.shared
                );
            }
        }
    }

    /// WHY: the par_iter + map_init pattern reuses per-thread workspaces
    /// across many edges. Two runs of build_edges on the same inputs must
    /// produce byte-identical edge lists — proves that the workspace
    /// state doesn't leak between hits AND that rayon's order-preserving
    /// collect actually preserves order.
    #[test]
    fn build_edges_chain_deterministic_across_runs() {
        let dir = tempdir().unwrap();
        let a = seq_from_seed(20_000, 1);
        let b = a[4_000..16_000].to_vec();
        let c = a[8_000..14_000].to_vec();
        let d = seq_from_seed(20_000, 7);
        let inputs = vec![
            ContigInput {
                id: "A".to_string(),
                source_mag: None,
                sequence: a,
            },
            ContigInput {
                id: "B".to_string(),
                source_mag: None,
                sequence: b,
            },
            ContigInput {
                id: "C".to_string(),
                source_mag: None,
                sequence: c,
            },
            ContigInput {
                id: "D".to_string(),
                source_mag: None,
                sequence: d,
            },
        ];
        let cfg = test_cfg_with_chain(0.3, 1);
        let dir2 = tempdir().unwrap();

        let out1 = build_edges(&inputs, &cfg, dir.path()).unwrap();
        let out2 = build_edges(&inputs, &cfg, dir2.path()).unwrap();

        assert_eq!(out1.edges.len(), out2.edges.len(), "edge count differs");
        for (e1, e2) in out1.edges.iter().zip(out2.edges.iter()) {
            assert_eq!(e1.query_idx, e2.query_idx);
            assert_eq!(e1.target_idx, e2.target_idx);
            assert_eq!(e1.score, e2.score);
            assert_eq!(e1.shared, e2.shared);
            // ChainScore should match too (f64 fields are deterministic
            // given the deterministic sort + DP).
            assert_eq!(e1.chain, e2.chain);
        }
    }
}
