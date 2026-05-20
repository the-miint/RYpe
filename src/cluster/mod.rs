//! Contig-level clustering for mixed-completeness genome assemblies.
//!
//! Given a set of contigs (possibly from many fragmented MAGs), produce a
//! dereplicated set of representative contigs using minimizer-based
//! containment and length-sorted greedy clustering.
//!
//! Entry point: [`cluster_contigs`].

pub mod containment;
pub mod edges;
pub mod greedy;
pub mod types;

use anyhow::{Context, Result};

pub use types::{ClusterEdge, ClusterResult, ClusterRow, ContigInfo, ContigInput};

#[derive(Debug, Clone)]
pub struct ClusterConfig {
    pub k: usize,
    pub w: usize,
    pub salt: u64,
    pub min_length: u64,
    pub threshold: f64,
    pub min_shared: u64,
}

impl ClusterConfig {
    /// Recommended starting point for strain-level (~99% ANI) clustering of
    /// PacBio-era long-read contigs. Threshold and `min_shared` are explicit
    /// starting points pending empirical calibration on user data.
    pub fn strain_default() -> Self {
        Self {
            k: 64,
            w: 50,
            salt: 0x5555_5555_5555_5555,
            min_length: 10_000,
            threshold: 0.85,
            min_shared: 500,
        }
    }
}

/// Cluster a set of contigs end-to-end.
///
/// Filters inputs to `sequence.len() >= cfg.min_length`, builds a temp
/// Parquet inverted index for sparse containment-edge computation, and
/// runs length-sorted greedy dereplication. The temp index is cleaned up
/// when this function returns.
pub fn cluster_contigs(inputs: Vec<ContigInput>, cfg: &ClusterConfig) -> Result<ClusterResult> {
    let filtered: Vec<ContigInput> = inputs
        .into_iter()
        .filter(|c| (c.sequence.len() as u64) >= cfg.min_length)
        .collect();

    if filtered.is_empty() {
        return Ok(ClusterResult::default());
    }

    let workdir = tempfile::tempdir().context("creating clustering workdir")?;

    let edge_out = edges::build_edges(&filtered, cfg, workdir.path())?;
    drop(filtered);

    let rows = greedy::greedy_dereplicate(
        &edge_out.edges,
        &edge_out.contigs,
        cfg.threshold,
        cfg.min_shared,
    );

    Ok(ClusterResult { rows })
}
