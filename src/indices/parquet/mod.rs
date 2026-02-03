//! Parquet-based format implementation for inverted index.
//!
//! This module provides read/write support for the Parquet-based inverted index format,
//! which stores inverted index data in a directory structure:
//!
//! ```text
//! index.parquet.ryxdi/
//! ├── manifest.toml        # Metadata: k, w, salt, format version
//! ├── buckets.parquet      # bucket_id → bucket_name, sources
//! └── inverted/
//!     └── shard.0.parquet  # (minimizer, bucket_id) inverted data
//! ```
//!
//! Note: The main index uses the single-file format (.ryidx) only.
//! Parquet format is only used for the inverted index.

mod buckets;
mod manifest;
pub mod merge;
mod options;
mod streaming;

/// Format version for the Parquet-based index.
/// Increment when making breaking changes to the format.
pub const FORMAT_VERSION: u32 = 1;

/// Magic bytes to identify Parquet-based index directories.
/// Written to manifest.toml for format detection.
pub const FORMAT_MAGIC: &str = "RYPE_PARQUET_V1";

/// Standard file names within the index directory.
pub mod files {
    pub const MANIFEST: &str = "manifest.toml";
    pub const BUCKETS: &str = "buckets.parquet";
    pub const INVERTED_DIR: &str = "inverted";

    /// Generate shard filename for inverted index.
    pub fn inverted_shard(shard_id: u32) -> String {
        format!("shard.{}.parquet", shard_id)
    }
}

// Re-export public types from submodules
pub use buckets::{read_buckets_parquet, write_buckets_parquet};
pub use manifest::{
    create_index_directory, is_parquet_index, BucketData, BucketMetadata, InvertedManifest,
    InvertedShardInfo, ParquetManifest, ParquetShardFormat,
};
pub use options::{hex_u64, ParquetCompression, ParquetReadOptions, ParquetWriteOptions};
pub use streaming::{
    compute_source_hash, create_parquet_inverted_index, ShardAccumulator, MIN_SHARD_BYTES,
};
