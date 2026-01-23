//! Index structures for storing and querying minimizer data.
//!
//! This module contains all index-related types:
//! - `InvertedIndex`: CSR-format inverted index for fast lookups
//! - `ShardedInvertedIndex`: Memory-efficient sharded inverted index
//! - Parquet-based index format support

mod inverted;
pub mod parquet;
pub mod sharded;

// Re-export primary types at the indices module level
pub use inverted::{get_row_group_ranges, load_row_group_pairs, InvertedIndex, QueryInvertedIndex};
pub use parquet::{
    compute_source_hash, create_parquet_inverted_index, is_parquet_index, BucketData,
    BucketMetadata, InvertedManifest, InvertedShardInfo, ParquetCompression, ParquetManifest,
    ParquetReadOptions, ParquetWriteOptions,
};
pub use sharded::{ShardInfo, ShardManifest, ShardedInvertedIndex};
