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
//! - [`InvertedIndex`]: Minimizer → bucket mappings for fast classification
//! - [`ShardedInvertedIndex`]: Memory-efficient sharded inverted index
//! - [`MinimizerWorkspace`]: Reusable workspace for minimizer extraction
//!
//! # Classification Functions
//!
//! - [`classify_batch_sharded_merge_join`]: Classify with sharded inverted index using merge-join (default)
//! - [`classify_batch_sharded_parallel_rg`]: Classify with parallel row group processing

// Internal modules
mod classify;
mod constants;
mod core;
mod error;
mod indices;
mod types;

// Public modules
pub mod c_api;
pub mod config;
pub mod memory;

// ============================================================================
// Timing utilities (cross-cutting concern)
// ============================================================================

/// Controls whether timing diagnostics are printed to stderr.
///
/// Set to `true` to enable timing output for classification operations.
/// This is useful for debugging and performance analysis.
pub static ENABLE_TIMING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Print timing info to stderr if [`ENABLE_TIMING`] is enabled.
#[inline]
pub fn log_timing(phase: &str, elapsed_ms: u128) {
    if ENABLE_TIMING.load(std::sync::atomic::Ordering::Relaxed) {
        eprintln!("[TIMING] {}: {}ms", phase, elapsed_ms);
    }
}

// Arrow FFI integration (optional feature for exporting RecordBatches)
#[cfg(feature = "arrow-ffi")]
pub mod arrow;

// ============================================================================
// Essential types (commonly used in most workflows)
// ============================================================================

// Core types
pub use error::{FirstErrorCapture, Result as RypeResult, RypeError};
pub use types::{HitResult, IndexMetadata, QueryRecord};

// Primary index types
pub use indices::{InvertedIndex, ShardedInvertedIndex};

// Minimizer extraction
pub use core::{extract_into, get_paired_minimizers_into, MinimizerWorkspace, Strand};

// Classification functions
pub use classify::{
    classify_batch_merge_join, classify_batch_sharded_merge_join,
    classify_batch_sharded_parallel_rg, classify_with_sharded_negative, filter_best_hits,
};

// ============================================================================
// Specialized types (for advanced use cases - consider using qualified paths)
// e.g., `rype::indices::sharded::ShardManifest` or `rype::parquet_index::ParquetWriteOptions`
// ============================================================================

// Core utilities (low-level extraction)
pub use core::{
    base_to_bit, count_hits, extract_dual_strand_into, extract_with_positions,
    MinimizerWithPosition,
};

// Orientation utilities (for bucket building)
pub use core::{
    choose_orientation, choose_orientation_sampled, merge_sorted_into, Orientation,
    ORIENTATION_SAMPLE_SIZE,
};

// Constants
pub use constants::BUCKET_SOURCE_DELIM;

// Sharded index internals
pub use indices::{
    // Inverted index internals
    QueryInvertedIndex,
    ShardInfo,
    ShardManifest,
};

// Parquet index types (also available via `rype::parquet_index::*`)
pub use indices::{
    compute_source_hash, create_parquet_inverted_index, is_parquet_index, BucketData,
    BucketMetadata, InvertedManifest, InvertedShardInfo, ParquetCompression, ParquetManifest,
    ParquetReadOptions, ParquetWriteOptions,
};

// Re-export parquet module for qualified access
pub use indices::parquet as parquet_index;
