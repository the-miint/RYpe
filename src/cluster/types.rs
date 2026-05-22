//! Public data types for the clustering API.

use crate::Strand;

/// Input contig as provided to the clustering pipeline.
///
/// `id` must be unique across the input set; it appears verbatim in output
/// rows as `rep_contig` and `member_contig`.
#[derive(Debug, Clone)]
pub struct ContigInput {
    pub id: String,
    pub source_mag: Option<String>,
    pub sequence: Vec<u8>,
}

/// Positional-chain output for one `ClusterEdge`.
///
/// Populated by `cluster::edges::build_edges` when chain DP runs (i.e. when
/// `ClusterConfig.chain_params.is_some()`); `None` on edges that skipped
/// chain (either chain disabled in config or DP returned `None` because
/// the chain fell below `ChainParams.min_anchors`).
///
/// `containment` is normalized by the **winning strand's** query minimizer
/// count — matching the set-containment `score` field on `ClusterEdge`,
/// which is `max(C(q_fwd, t), C(q_rc, t))`. When the chain winner is on a
/// different strand from the set-containment winner (rare — repetitive
/// regions can do this), the two `[0,1]` numbers describe different
/// strands. That's fine; document it at the consuming site.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChainScore {
    /// Banded-DP score from `chain_anchors` (sum of `anchor_credit − gap`
    /// over chained transitions plus the initial chain-start credit).
    pub score: f64,
    /// Number of anchors in the winning chain (≥ `ChainParams.min_anchors`).
    pub anchors: u32,
    /// `anchors / |q_<strand>_minimizers|` — chain analog of `ClusterEdge.score`.
    pub containment: f64,
    /// Which query strand produced the winning chain.
    pub strand: Strand,
}

/// Metadata about a contig that the greedy step needs.
///
/// Separated from [`ContigInput`] so the greedy algorithm is a pure function
/// over numeric inputs and does not depend on owning sequence bytes.
#[derive(Debug, Clone)]
pub struct ContigInfo {
    pub id: String,
    pub source_mag: Option<String>,
    pub length: u64,
}

/// A sparse containment edge between two contigs.
///
/// Indices reference positions in the contigs slice passed alongside the
/// edges to the greedy step. `score` is the containment of the query in the
/// target (rype's classify score: `max(C(q.fwd, t), C(q.rc, t))`), and
/// `shared` is the absolute count of shared minimizers on the winning strand.
///
/// `chain` carries the positional-chain output (Plan 1.3 chain DP applied to
/// this edge's anchor list). `None` means either:
///   - chain step was disabled in `ClusterConfig`, OR
///   - the chain DP returned `None` (chain too short — fewer than
///     `ChainParams.min_anchors` anchors).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClusterEdge {
    pub query_idx: u32,
    pub target_idx: u32,
    pub score: f64,
    pub shared: u64,
    pub chain: Option<ChainScore>,
}

/// One row of clustering output — either a representative pointing at
/// itself (`rep_contig == member_contig`, `containment == 1.0`) or an
/// absorbed member with its containment in the representative.
///
/// `chain` carries the chain DP output for the edge that drove this row's
/// absorption. `None` means:
///   - the row is a representative (no edge, so chain doesn't apply), OR
///   - chain DP was globally disabled (`ClusterConfig.chain_params` was `None`), OR
///   - chain DP returned `None` for the edge (fewer than `min_anchors` colinear anchors).
///
/// All four downstream surfaces (Rust callers, Arrow record batches, Parquet
/// CLI output, C-API `RypeClusterRow`) map `None` to their respective null
/// representation.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterRow {
    pub rep_contig: String,
    pub member_contig: String,
    pub source_mag: Option<String>,
    pub containment: f64,
    pub chain: Option<ChainScore>,
}

/// Full clustering result — one row per input contig (a complete partition).
#[derive(Debug, Clone, Default)]
pub struct ClusterResult {
    pub rows: Vec<ClusterRow>,
}
