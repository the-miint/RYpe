//! Contig-level clustering for mixed-completeness genome assemblies.
//!
//! Given a set of contigs (possibly from many fragmented MAGs), produce a
//! dereplicated set of representative contigs using minimizer-based
//! containment and length-sorted greedy clustering.

pub mod containment;
pub mod edges;
pub mod greedy;
pub mod types;

pub use types::{ClusterEdge, ClusterResult, ClusterRow, ContigInfo, ContigInput};
