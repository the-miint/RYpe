//! Build sparse containment edges between contigs via the existing classify
//! pipeline.
//!
//! Each bucket stores the single-strand minimizer set of one contig; each
//! query is presented as `(fwd_mins, rc_mins)` so classify's
//! `max(fwd_score, rc_score)` gives orientation-independent containment.
//!
//! `score` is rype's classify score (containment of query in target).
//! `shared` is approximated as `round(score * |fwd_mins|)` — exact when fwd
//! wins, within ~1% when rc wins (for non-palindromic sequences).

use std::path::Path;

use anyhow::{Context, Result};
use rayon::prelude::*;

use crate::{
    classify_from_extracted_minimizers, create_parquet_inverted_index, extract_dual_strand_into,
    BucketData, MinimizerWorkspace, ShardedInvertedIndex,
};

use super::types::{ClusterEdge, ContigInfo, ContigInput};
use super::ClusterConfig;

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

    type Extracted = (ContigInfo, Vec<u64>, Vec<u64>);
    let extracted: Vec<Extracted> = inputs
        .par_iter()
        .map_init(MinimizerWorkspace::new, |ws, input| {
            let (mut fwd, mut rc) =
                extract_dual_strand_into(&input.sequence, cfg.k, cfg.w, cfg.salt, ws);
            fwd.sort_unstable();
            fwd.dedup();
            rc.sort_unstable();
            rc.dedup();
            (
                ContigInfo {
                    id: input.id.clone(),
                    source_mag: input.source_mag.clone(),
                    length: input.sequence.len() as u64,
                },
                fwd,
                rc,
            )
        })
        .collect();

    let mut contigs: Vec<ContigInfo> = Vec::with_capacity(extracted.len());
    let mut query_mins: Vec<(Vec<u64>, Vec<u64>)> = Vec::with_capacity(extracted.len());
    for (info, fwd, rc) in extracted {
        contigs.push(info);
        query_mins.push((fwd, rc));
    }

    // INVARIANT: `bucket_id == idx + 1` where `idx` is the position in the
    // `inputs` slice. `bucket_id_to_idx` relies on this.
    let buckets: Vec<BucketData> = query_mins
        .iter()
        .zip(inputs.iter())
        .enumerate()
        .filter(|(_, ((fwd, _), _))| !fwd.is_empty())
        .map(|(idx, ((fwd, _), input))| BucketData {
            bucket_id: bucket_id_for(idx),
            bucket_name: input.id.clone(),
            sources: input
                .source_mag
                .clone()
                .map(|m| vec![m])
                .unwrap_or_default(),
            minimizers: fwd.clone(),
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

    let mut edges = Vec::with_capacity(hits.len());
    for hit in hits {
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
        let Some(target_idx) = bucket_id_to_idx(hit.bucket_id) else {
            continue;
        };

        if query_idx == target_idx {
            continue;
        }

        let fwd_total = query_mins[query_idx as usize].0.len() as f64;
        let shared = (hit.score * fwd_total).round() as u64;
        if shared < cfg.min_shared {
            continue;
        }
        edges.push(ClusterEdge {
            query_idx,
            target_idx,
            score: hit.score,
            shared,
            // Plan 1.4 phase 1: chain field present but unpopulated. Phase 2
            // wires the chain DP into build_edges and sets this to Some(_)
            // when cfg.chain_params is enabled.
            chain: None,
        });
    }

    Ok(EdgeBuildOutput { contigs, edges })
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
            // Phase 1: chain disabled by default in this test helper. Phase 2's
            // build_edges tests will use a separate helper that enables chain.
            chain_params: None,
            min_chain_containment: None,
        }
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
}
