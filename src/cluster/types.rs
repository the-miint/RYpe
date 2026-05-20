//! Public data types for the clustering API.

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
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClusterEdge {
    pub query_idx: u32,
    pub target_idx: u32,
    pub score: f64,
    pub shared: u64,
}

/// One row of clustering output — either a representative pointing at
/// itself (`rep_contig == member_contig`, `containment == 1.0`) or an
/// absorbed member with its containment in the representative.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterRow {
    pub rep_contig: String,
    pub member_contig: String,
    pub source_mag: Option<String>,
    pub containment: f64,
}

/// Full clustering result — one row per input contig (a complete partition).
#[derive(Debug, Clone, Default)]
pub struct ClusterResult {
    pub rows: Vec<ClusterRow>,
}
