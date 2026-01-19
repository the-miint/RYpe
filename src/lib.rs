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
//! - [`classify_batch_sharded_sequential`]: Classify with sharded inverted index (low memory)
//! - [`classify_batch_sharded_merge_join`]: Classify with merge-join algorithm (high overlap)
//! - [`aggregate_batch`]: Aggregated paired-end classification

// Internal modules
mod classify;
mod constants;
mod encoding;
mod extraction;
mod index;
mod inverted;
mod sharded;
mod sharded_main;
mod types;
mod workspace;

// Public modules
pub mod c_api;
pub mod config;
pub mod memory;

// Arrow FFI integration (optional feature for exporting RecordBatches)
#[cfg(feature = "arrow-ffi")]
pub mod arrow;

// Parquet-based index format
pub mod parquet_index;

// Re-export Parquet types
pub use parquet_index::{
    compute_source_hash, create_parquet_inverted_index, is_parquet_index, BucketData,
    BucketMetadata, InvertedManifest, InvertedShardInfo, ParquetCompression, ParquetManifest,
    ParquetReadOptions, ParquetWriteOptions,
};

// Re-export types
pub use types::{HitResult, IndexMetadata, QueryRecord};

// Re-export workspace
pub use workspace::MinimizerWorkspace;

// Re-export encoding utilities (only public function)
pub use encoding::base_to_bit;

// Re-export extraction functions and types
pub use extraction::{
    count_hits, extract_dual_strand_into, extract_into, extract_with_positions,
    get_paired_minimizers_into, MinimizerWithPosition, Strand,
};

// Re-export index
pub use index::Index;

// Re-export inverted index
pub use inverted::{InvertedIndex, QueryInvertedIndex};

// Re-export sharded index types
pub use sharded::{ShardFormat, ShardInfo, ShardManifest, ShardedInvertedIndex};

// Re-export sharded main index types
pub use sharded_main::{
    estimate_bucket_bytes, plan_shards, MainIndexManifest, MainIndexShard, MainIndexShardInfo,
    ShardedMainIndex, ShardedMainIndexBuilder,
};

// Re-export classification functions
pub use classify::{
    aggregate_batch, classify_batch, classify_batch_merge_join, classify_batch_sharded_main,
    classify_batch_sharded_merge_join, classify_batch_sharded_sequential, ENABLE_TIMING,
};
