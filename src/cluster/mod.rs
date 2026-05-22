//! Contig-level clustering for mixed-completeness genome assemblies.
//!
//! Given a set of contigs (possibly from many fragmented MAGs), produce a
//! dereplicated set of representative contigs using minimizer-based
//! containment and length-sorted greedy clustering.
//!
//! Entry point: [`cluster_contigs`].

pub mod chain;
pub mod containment;
pub mod edges;
pub mod greedy;
pub mod types;

use anyhow::{anyhow, Context, Result};

pub use chain::ChainParams;
pub use types::{ChainScore, ClusterEdge, ClusterResult, ClusterRow, ContigInfo, ContigInput};

#[derive(Debug, Clone)]
pub struct ClusterConfig {
    pub k: usize,
    pub w: usize,
    pub salt: u64,
    pub min_length: u64,
    pub threshold: f64,
    pub min_shared: u64,
    /// Chain DP parameters. `None` skips the chain step entirely in
    /// `build_edges`; resulting edges have `chain: None`. `Some(params)`
    /// runs chain on every classify-candidate edge.
    pub chain_params: Option<ChainParams>,
    /// Minimum `chain.containment` required for greedy absorption. `None`
    /// disables the chain gate (chain is informational only). When `Some`:
    /// an edge with `chain: None` is rejected, and an edge with chain
    /// present must clear this threshold in addition to the set-containment
    /// `threshold`.
    ///
    /// Setting `min_chain_containment = Some(_)` while `chain_params = None`
    /// is incoherent — the CLI rejects the combination loudly.
    pub min_chain_containment: Option<f64>,
}

impl ClusterConfig {
    /// Recommended starting point for strain-level (~99% ANI) clustering of
    /// PacBio-era long-read contigs. Threshold and `min_shared` are explicit
    /// starting points pending empirical calibration on user data.
    ///
    /// Chain DP is enabled by default with `ChainParams::starting_for_w(50)`
    /// (uncalibrated research-doc values; Plan 1.6 will revise). The chain
    /// **gate** is `None` by default — chain output is informational only
    /// until calibration produces a blessed threshold.
    pub fn strain_default() -> Self {
        Self {
            k: 64,
            w: 50,
            salt: 0x5555_5555_5555_5555,
            min_length: 10_000,
            threshold: 0.85,
            min_shared: 500,
            chain_params: Some(ChainParams::starting_for_w(50)),
            min_chain_containment: None,
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
    // Reject the incoherent (chain disabled, chain gate enabled) combination
    // here rather than in the CLI alone — a library caller who builds
    // `ClusterConfig` directly would otherwise get silently empty
    // dereplication (every edge fails the gate because `chain` is None).
    if cfg.min_chain_containment.is_some() && cfg.chain_params.is_none() {
        return Err(anyhow!(
            "min_chain_containment requires chain_params to be Some — \
             chain DP must be enabled to gate on chain containment"
        ));
    }

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
        cfg.min_chain_containment,
    );

    Ok(ClusterResult { rows })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: `strain_default()` deliberately enables chain DP (`Some`) and
    /// deliberately leaves the chain GATE disabled (`None`). This pins the
    /// minimum-risk default — chain output is computed and reported, but
    /// absorption is unchanged until Plan 1.6 calibration produces a
    /// blessed `min_chain_containment` value. If a future PR flips either
    /// side of this default, this test fires.
    #[test]
    fn strain_default_enables_chain_disables_gate() {
        let cfg = ClusterConfig::strain_default();
        assert!(
            cfg.chain_params.is_some(),
            "strain_default enables chain (chain_params: Some)"
        );
        assert!(
            cfg.min_chain_containment.is_none(),
            "strain_default leaves the greedy chain gate disabled \
             (min_chain_containment: None) — Plan 1.6 calibrates before flipping"
        );
        // Pin the specific ChainParams shape: starting_for_w(50). Catches
        // a future change that silently replaces it with starting_for_w(<other>).
        let p = cfg.chain_params.as_ref().unwrap();
        let reference = ChainParams::starting_for_w(50);
        assert_eq!(p.anchor_credit, reference.anchor_credit);
        assert_eq!(p.max_gap_length, reference.max_gap_length);
        assert_eq!(p.max_lin_length, reference.max_lin_length);
        assert_eq!(p.band_anchors, reference.band_anchors);
        assert_eq!(p.band_bp, reference.band_bp);
        assert_eq!(p.min_anchors, reference.min_anchors);
    }

    /// WHY: `min_chain_containment = Some(_)` with `chain_params = None` is
    /// the silent-empty-output trap — every edge ends up with `chain: None`
    /// and the gate rejects all of them. A library caller who built
    /// `ClusterConfig` by hand and forgot to enable chain DP would
    /// previously get an empty (or near-empty) result with no signal.
    /// `cluster_contigs` rejects the combination loudly, matching the CLI's
    /// own conflict check.
    #[test]
    fn cluster_contigs_rejects_incoherent_chain_config() {
        let cfg = ClusterConfig {
            k: 32,
            w: 20,
            salt: 0,
            min_length: 0,
            threshold: 0.8,
            min_shared: 1,
            chain_params: None,
            min_chain_containment: Some(0.5),
        };
        let err = cluster_contigs(Vec::new(), &cfg)
            .err()
            .expect("incoherent config must error");
        let msg = format!("{}", err);
        assert!(
            msg.contains("min_chain_containment") && msg.contains("chain_params"),
            "error must name both fields, got: {}",
            msg
        );
    }
}
