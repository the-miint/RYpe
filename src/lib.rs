//! Rype: High-performance genomic sequence classification using minimizer-based k-mer sketching.
//!
//! This library provides efficient classification of DNA sequences against reference indices
//! using RY-space (purine/pyrimidine) encoding and minimizer sketching.
//!
//! # Core Concepts
//!
//! - **RY Encoding**: Reduces the 4-base DNA alphabet to 2 bits (purines → 1, pyrimidines → 0)
//! - **Minimizers**: Representative k-mers selected from sliding windows for efficient sketching
//! - **Inverted Index**: Enables O(Q log U) lookups instead of O(B × Q × log M)
//!
//! # Main Types
//!
//! - [`Index`]: Primary index structure with per-bucket minimizer sets
//! - [`InvertedIndex`]: Minimizer → bucket mappings for fast classification
//! - [`ShardedInvertedIndex`]: Memory-efficient sharded inverted index
//! - [`MinimizerWorkspace`]: Reusable workspace for minimizer extraction
//!
//! # Classification Functions
//!
//! - [`classify_batch`]: Classify against an Index (per-bucket binary search)
//! - [`classify_batch_inverted`]: Classify using InvertedIndex (recommended for small batches)
//! - [`classify_batch_with_query_index`]: Classify using merge-join (recommended for large batches with high overlap)
//! - [`classify_batch_sharded_sequential`]: Classify with sharded index (low memory)
//! - [`aggregate_batch`]: Aggregated paired-end classification

// Internal modules
mod constants;
mod encoding;
mod extraction;
mod index;
mod inverted;
mod sharded;
mod sharded_main;
mod classify;
mod types;
mod workspace;

// Public modules
pub mod c_api;
pub mod config;

// Re-export types
pub use types::{QueryRecord, HitResult, IndexMetadata};

// Re-export workspace
pub use workspace::MinimizerWorkspace;

// Re-export encoding utilities (only public function)
pub use encoding::base_to_bit;

// Re-export extraction functions and types
pub use extraction::{
    extract_into,
    extract_dual_strand_into,
    get_paired_minimizers_into,
    extract_with_positions,
    count_hits,
    Strand,
    MinimizerWithPosition,
};

// Re-export index
pub use index::Index;

// Re-export inverted index
pub use inverted::{InvertedIndex, QueryInvertedIndex};

// Re-export sharded index types
pub use sharded::{ShardInfo, ShardManifest, ShardedInvertedIndex};

// Re-export sharded main index types
pub use sharded_main::{
    MainIndexShardInfo,
    MainIndexManifest,
    MainIndexShard,
    ShardedMainIndex,
    ShardedMainIndexBuilder,
    estimate_bucket_bytes,
    plan_shards,
};

// Re-export classification functions
pub use classify::{
    classify_batch,
    classify_batch_inverted,
    classify_batch_sharded_sequential,
    classify_batch_sharded_main,
    classify_batch_merge_join,
    classify_batch_with_query_index,
    aggregate_batch,
};
