//! Build sparse containment edges between contigs via the existing classify
//! pipeline.
//!
//! The flow is:
//! 1. Extract a canonical minimizer set (fwd + rc, sorted, deduplicated) per
//!    contig.
//! 2. Build an on-disk Parquet inverted index where each contig is one bucket.
//! 3. Query each contig's minimizers against the index. Rype's classify score
//!    `shared / |query_mins|` IS the containment of the query in the bucket,
//!    so we use the cluster `threshold` directly as the classify threshold to
//!    avoid materializing useless low-containment edges.
//! 4. Convert each `HitResult` to a `ClusterEdge`, recovering the absolute
//!    shared-minimizer count by rounding `score * |query_mins|`.

use std::path::Path;

use anyhow::{Context, Result};

use crate::{
    classify_from_extracted_minimizers, create_parquet_inverted_index, extract_dual_strand_into,
    BucketData, MinimizerWorkspace, ShardedInvertedIndex,
};

use super::types::{ClusterEdge, ContigInfo, ContigInput};

/// Output of the edge-build step: contig metadata + sparse edges.
///
/// Order of `contigs` matches the input order; edge indices refer to this
/// same ordering.
#[derive(Debug, Clone)]
pub struct EdgeBuildOutput {
    pub contigs: Vec<ContigInfo>,
    pub edges: Vec<ClusterEdge>,
}

/// Build containment edges for a set of contigs.
///
/// `workdir` must exist and be writable. A Parquet index named
/// `cluster_index.ryxdi` is created inside it (and left behind on success
/// for caller-managed cleanup — typically via a `tempfile::TempDir`).
pub fn build_edges(
    inputs: &[ContigInput],
    k: usize,
    w: usize,
    salt: u64,
    threshold: f64,
    workdir: &Path,
) -> Result<EdgeBuildOutput> {
    let mut ws = MinimizerWorkspace::new();

    let mut contigs: Vec<ContigInfo> = Vec::with_capacity(inputs.len());
    let mut canonical: Vec<Vec<u64>> = Vec::with_capacity(inputs.len());
    for input in inputs {
        let mins = extract_canonical(&input.sequence, k, w, salt, &mut ws);
        contigs.push(ContigInfo {
            id: input.id.clone(),
            source_mag: input.source_mag.clone(),
            length: input.sequence.len() as u64,
            mins_count: mins.len() as u64,
        });
        canonical.push(mins);
    }

    if inputs.is_empty() {
        return Ok(EdgeBuildOutput {
            contigs,
            edges: Vec::new(),
        });
    }

    let buckets: Vec<BucketData> = canonical
        .iter()
        .zip(inputs.iter())
        .enumerate()
        .filter(|(_, (mins, _))| !mins.is_empty())
        .map(|(idx, (mins, input))| BucketData {
            bucket_id: bucket_id_for(idx),
            bucket_name: input.id.clone(),
            sources: input
                .source_mag
                .clone()
                .map(|m| vec![m])
                .unwrap_or_default(),
            minimizers: mins.clone(),
        })
        .collect();

    if buckets.is_empty() {
        return Ok(EdgeBuildOutput {
            contigs,
            edges: Vec::new(),
        });
    }

    let index_path = workdir.join("cluster_index.ryxdi");
    create_parquet_inverted_index(&index_path, buckets, k, w, salt, None, None, None)
        .context("creating clustering index")?;
    let index = ShardedInvertedIndex::open(&index_path).context("opening clustering index")?;

    let extracted: Vec<(Vec<u64>, Vec<u64>)> = canonical
        .iter()
        .map(|mins| (mins.clone(), Vec::new()))
        .collect();
    let query_ids: Vec<i64> = (0..inputs.len() as i64).collect();

    let hits = classify_from_extracted_minimizers(&index, &extracted, &query_ids, threshold, None)
        .context("classifying contigs against clustering index")?;

    let mut edges = Vec::with_capacity(hits.len());
    for hit in hits {
        let query_idx = hit.query_id as u32;
        let target_idx = bucket_id_to_idx(hit.bucket_id);
        if query_idx == target_idx {
            continue;
        }
        let q_mins = contigs[query_idx as usize].mins_count;
        let shared = (hit.score * q_mins as f64).round() as u64;
        if shared == 0 {
            continue;
        }
        edges.push(ClusterEdge {
            query_idx,
            target_idx,
            shared,
        });
    }

    Ok(EdgeBuildOutput { contigs, edges })
}

fn extract_canonical(
    seq: &[u8],
    k: usize,
    w: usize,
    salt: u64,
    ws: &mut MinimizerWorkspace,
) -> Vec<u64> {
    let (fwd, rc) = extract_dual_strand_into(seq, k, w, salt, ws);
    let mut combined = fwd;
    combined.extend(rc);
    combined.sort_unstable();
    combined.dedup();
    combined
}

#[inline]
fn bucket_id_for(idx: usize) -> u32 {
    (idx as u32) + 1
}

#[inline]
fn bucket_id_to_idx(bucket_id: u32) -> u32 {
    bucket_id - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn seq_from_seed(len: usize, seed: u64) -> Vec<u8> {
        let bases = [b'A', b'C', b'G', b'T'];
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
        (0..len)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                bases[((s >> 56) & 0b11) as usize]
            })
            .collect()
    }

    #[test]
    fn build_edges_produces_edge_for_fragment_of_full_genome() {
        let dir = tempdir().unwrap();

        // contig A: full pseudo-random genome
        // contig B: middle 60% of A (a fragment)
        // contig C: independent pseudo-random sequence
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

        let out = build_edges(&inputs, 32, 20, 0x5555_5555_5555_5555, 0.5, dir.path()).unwrap();

        // ContigInfo invariants
        assert_eq!(out.contigs.len(), 3);
        assert_eq!(out.contigs[0].id, "A");
        assert_eq!(out.contigs[1].id, "B");
        assert_eq!(out.contigs[2].id, "C");
        assert!(out.contigs[0].mins_count > 0);
        assert!(out.contigs[1].mins_count > 0);
        assert!(out.contigs[2].mins_count > 0);

        // B should be highly contained in A: edge (B -> A) with high containment.
        let b_to_a = out
            .edges
            .iter()
            .find(|e| e.query_idx == 1 && e.target_idx == 0)
            .expect("expected an edge from B (fragment) to A (full genome)");
        let b_mins = out.contigs[1].mins_count;
        let containment = b_to_a.shared as f64 / b_mins as f64;
        assert!(
            containment > 0.85,
            "fragment B should be > 85% contained in A, got {} ({}/{})",
            containment,
            b_to_a.shared,
            b_mins,
        );

        // C should NOT have a high-containment edge to A or B
        for e in &out.edges {
            if e.query_idx == 2 {
                let c_mins = out.contigs[2].mins_count;
                let c_to_other = e.shared as f64 / c_mins as f64;
                assert!(
                    c_to_other < 0.5,
                    "unrelated C should have low containment in others, got {}",
                    c_to_other,
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
        let out = build_edges(&inputs, 32, 20, 0x12345, 0.5, dir.path()).unwrap();
        for e in &out.edges {
            assert_ne!(
                e.query_idx, e.target_idx,
                "self-hit should be excluded from edges"
            );
        }
    }

    #[test]
    fn build_edges_empty_input_returns_empty() {
        let dir = tempdir().unwrap();
        let out = build_edges(&[], 32, 20, 0x12345, 0.5, dir.path()).unwrap();
        assert!(out.contigs.is_empty());
        assert!(out.edges.is_empty());
    }
}
