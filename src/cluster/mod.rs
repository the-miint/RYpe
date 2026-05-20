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

/// Parameters for a single clustering run.
///
/// `min_length` filters contigs from the input before any work is done;
/// `threshold` and `min_shared` control absorption in the greedy step (and
/// `threshold` is also used as the classify threshold to prune low-containment
/// edges at query time).
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
/// Steps:
/// 1. Filter inputs to those with `sequence.len() >= cfg.min_length`.
/// 2. Build a temp Parquet inverted index (one bucket per surviving contig),
///    extract per-contig minimizers, and classify each contig as a query to
///    materialize sparse containment edges.
/// 3. Run length-sorted greedy dereplication on the edges.
///
/// The temp index directory is created via [`tempfile::TempDir`] and
/// cleaned up when this function returns.
pub fn cluster_contigs(inputs: &[ContigInput], cfg: &ClusterConfig) -> Result<ClusterResult> {
    let filtered: Vec<ContigInput> = inputs
        .iter()
        .filter(|c| (c.sequence.len() as u64) >= cfg.min_length)
        .cloned()
        .collect();

    if filtered.is_empty() {
        return Ok(ClusterResult::default());
    }

    let workdir = tempfile::tempdir().context("creating clustering workdir")?;

    let edge_out = edges::build_edges(
        &filtered,
        cfg.k,
        cfg.w,
        cfg.salt,
        cfg.threshold,
        cfg.min_shared,
        workdir.path(),
    )?;

    let rows = greedy::greedy_dereplicate(
        &edge_out.edges,
        &edge_out.contigs,
        cfg.threshold,
        cfg.min_shared,
    );

    Ok(ClusterResult { rows })
}
