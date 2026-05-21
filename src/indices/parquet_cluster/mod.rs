//! Positional cluster-index format (`.ryci`).
//!
//! Parallel to `crate::indices::parquet` but carries a per-minimizer
//! `position: u32` column for chainability scoring. Distinct on-disk
//! format with a separate magic and version; the classify hot path is
//! never affected.
//!
//! Directory layout:
//!
//! ```text
//! index.ryci/
//! ├── manifest.toml         # ClusterParquetManifest (TOML)
//! ├── buckets.parquet       # (bucket_id, bucket_name, sources) — schema identical to .ryxdi
//! └── inverted/
//!     └── shard.N.parquet   # (minimizer u64, bucket_id u32, position u32)
//! ```

pub mod manifest;

/// Format version for the `.ryci` cluster-index format. Bumped on breaking change.
pub const FORMAT_VERSION_CLUSTER: u32 = 1;

/// Magic string identifying a `.ryci` cluster index. Stored in `manifest.toml`.
pub const FORMAT_MAGIC_CLUSTER: &str = "RYPE_CLUSTER_V1";

/// File-name constants within a `.ryci` directory.
pub mod files {
    pub const MANIFEST: &str = "manifest.toml";
    pub const BUCKETS: &str = "buckets.parquet";
    pub const INVERTED_DIR: &str = "inverted";

    pub fn inverted_shard(shard_id: u32) -> String {
        format!("shard.{}.parquet", shard_id)
    }
}

pub use manifest::{
    create_cluster_index_directory, is_cluster_parquet_index, ClusterBucketData,
    ClusterInvertedManifest, ClusterInvertedShardInfo, ClusterParquetManifest,
};
