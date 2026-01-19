//! Index structures for storing and querying minimizer data.
//!
//! This module contains all index-related types:
//! - `Index`: Primary single-file index structure
//! - `InvertedIndex`: CSR-format inverted index for fast lookups
//! - `ShardedInvertedIndex`: Memory-efficient sharded inverted index
//! - `ShardedMainIndex`: Sharded main index for large datasets
//! - Parquet-based index format support

pub mod inverted;
pub mod main;
pub mod parquet;
pub mod sharded;
pub mod sharded_main;

// Re-export primary types at the indices module level
pub use inverted::{InvertedIndex, QueryInvertedIndex};
pub use main::Index;
pub use parquet::{
    compute_source_hash, create_parquet_inverted_index, is_parquet_index, BucketData,
    BucketMetadata, InvertedManifest, InvertedShardInfo, ParquetCompression, ParquetManifest,
    ParquetReadOptions, ParquetWriteOptions,
};
pub use sharded::{ShardFormat, ShardInfo, ShardManifest, ShardedInvertedIndex};
pub use sharded_main::{
    estimate_bucket_bytes, plan_shards, MainIndexManifest, MainIndexShard, MainIndexShardInfo,
    ShardedMainIndex, ShardedMainIndexBuilder,
};
