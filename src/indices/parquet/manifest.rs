//! Manifest and metadata structures for Parquet-based indices.
//!
//! This module contains the manifest format and related utility functions
//! for Parquet-based inverted index directories.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

use super::options::hex_u64;
use super::{files, FORMAT_MAGIC, FORMAT_VERSION};

/// Manifest containing index metadata.
///
/// Stored as TOML for human readability and easy inspection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParquetManifest {
    /// Magic string for format identification.
    pub magic: String,

    /// Format version for compatibility checking.
    pub format_version: u32,

    /// K-mer size (16, 32, or 64).
    pub k: usize,

    /// Window size for minimizer selection.
    pub w: usize,

    /// XOR salt applied to k-mer hashes (hex string for TOML).
    #[serde(with = "hex_u64")]
    pub salt: u64,

    /// Hash of source data for change detection (hex string for TOML).
    #[serde(with = "hex_u64")]
    pub source_hash: u64,

    /// Number of buckets in the index.
    pub num_buckets: u32,

    /// Total minimizers in main index.
    pub total_minimizers: u64,

    /// Inverted index shard information (if present).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inverted: Option<InvertedManifest>,
}

/// Shard format identifier stored in manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ParquetShardFormat {
    /// Parquet format (the only format for ParquetManifest)
    #[default]
    Parquet,
}

/// Manifest section for inverted index shards.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvertedManifest {
    /// Shard format - always "parquet" for Parquet indices.
    /// Stored explicitly to avoid file-existence guessing.
    #[serde(default)]
    pub format: ParquetShardFormat,

    /// Number of shards.
    pub num_shards: u32,

    /// Total entries across all shards.
    pub total_entries: u64,

    /// Whether shards have overlapping minimizer ranges.
    pub has_overlapping_shards: bool,

    /// Per-shard metadata.
    pub shards: Vec<InvertedShardInfo>,
}

/// Per-shard metadata for inverted index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvertedShardInfo {
    /// Shard identifier.
    pub shard_id: u32,

    /// Minimum minimizer value in this shard (hex string).
    #[serde(with = "hex_u64")]
    pub min_minimizer: u64,

    /// Maximum minimizer value in this shard (hex string).
    #[serde(with = "hex_u64")]
    pub max_minimizer: u64,

    /// Number of entries in this shard.
    pub num_entries: u64,
}

impl ParquetManifest {
    /// Create a new manifest with the given parameters.
    pub fn new(k: usize, w: usize, salt: u64) -> Self {
        Self {
            magic: FORMAT_MAGIC.to_string(),
            format_version: FORMAT_VERSION,
            k,
            w,
            salt,
            source_hash: 0,
            num_buckets: 0,
            total_minimizers: 0,
            inverted: None,
        }
    }

    /// Save manifest to the index directory.
    pub fn save(&self, index_dir: &Path) -> Result<()> {
        let path = index_dir.join(files::MANIFEST);
        let toml_str =
            toml::to_string_pretty(self).context("Failed to serialize manifest to TOML")?;
        fs::write(&path, toml_str)
            .with_context(|| format!("Failed to write manifest: {}", path.display()))?;
        Ok(())
    }

    /// Load manifest from the index directory.
    pub fn load(index_dir: &Path) -> Result<Self> {
        let path = index_dir.join(files::MANIFEST);
        let toml_str = fs::read_to_string(&path)
            .with_context(|| format!("Failed to read manifest: {}", path.display()))?;
        let manifest: Self = toml::from_str(&toml_str)
            .with_context(|| format!("Failed to parse manifest: {}", path.display()))?;

        // Validate magic and version
        if manifest.magic != FORMAT_MAGIC {
            anyhow::bail!(
                "Invalid manifest magic: expected '{}', got '{}'",
                FORMAT_MAGIC,
                manifest.magic
            );
        }
        if manifest.format_version > FORMAT_VERSION {
            anyhow::bail!(
                "Unsupported format version: {} (max supported: {})",
                manifest.format_version,
                FORMAT_VERSION
            );
        }

        Ok(manifest)
    }
}

/// Bucket metadata for a single bucket.
#[derive(Debug, Clone)]
pub struct BucketMetadata {
    pub bucket_id: u32,
    pub bucket_name: String,
    pub sources: Vec<String>,
    pub minimizer_count: usize,
}

/// Bucket data with minimizers for building inverted index.
#[derive(Debug, Clone)]
pub struct BucketData {
    pub bucket_id: u32,
    pub bucket_name: String,
    pub sources: Vec<String>,
    /// Sorted, deduplicated minimizers for this bucket.
    /// INVARIANT: Must be sorted and contain no duplicates.
    pub minimizers: Vec<u64>,
}

impl BucketData {
    /// Verify that minimizers are sorted and deduplicated.
    /// Returns an error if the invariant is violated.
    pub fn validate(&self) -> Result<()> {
        for i in 1..self.minimizers.len() {
            if self.minimizers[i] <= self.minimizers[i - 1] {
                if self.minimizers[i] == self.minimizers[i - 1] {
                    anyhow::bail!(
                        "Bucket {} has duplicate minimizer at position {}: {:#x}",
                        self.bucket_id,
                        i,
                        self.minimizers[i]
                    );
                } else {
                    anyhow::bail!(
                        "Bucket {} has unsorted minimizers at positions {}-{}: {:#x} > {:#x}",
                        self.bucket_id,
                        i - 1,
                        i,
                        self.minimizers[i - 1],
                        self.minimizers[i]
                    );
                }
            }
        }
        Ok(())
    }
}

/// Check if a path is a Parquet-based index directory.
pub fn is_parquet_index(path: &Path) -> bool {
    if !path.is_dir() {
        return false;
    }
    let manifest_path = path.join(files::MANIFEST);
    if !manifest_path.exists() {
        return false;
    }
    // Quick check: try to read magic from manifest
    if let Ok(content) = fs::read_to_string(&manifest_path) {
        return content.contains(FORMAT_MAGIC);
    }
    false
}

/// Create the directory structure for a new Parquet index.
pub fn create_index_directory(path: &Path) -> Result<()> {
    // Create main directory
    fs::create_dir_all(path)
        .with_context(|| format!("Failed to create index directory: {}", path.display()))?;

    // Create inverted subdirectory
    let inverted_dir = path.join(files::INVERTED_DIR);
    fs::create_dir_all(&inverted_dir).with_context(|| {
        format!(
            "Failed to create inverted directory: {}",
            inverted_dir.display()
        )
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_manifest_round_trip() {
        let tmp = TempDir::new().unwrap();
        let index_dir = tmp.path().join("test.ryidx");
        create_index_directory(&index_dir).unwrap();

        let mut manifest = ParquetManifest::new(64, 50, 12345);
        manifest.num_buckets = 10;
        manifest.total_minimizers = 1_000_000;
        manifest.source_hash = 0xDEADBEEF;

        manifest.save(&index_dir).unwrap();
        let loaded = ParquetManifest::load(&index_dir).unwrap();

        assert_eq!(loaded.magic, FORMAT_MAGIC);
        assert_eq!(loaded.format_version, FORMAT_VERSION);
        assert_eq!(loaded.k, 64);
        assert_eq!(loaded.w, 50);
        assert_eq!(loaded.salt, 12345);
        assert_eq!(loaded.num_buckets, 10);
        assert_eq!(loaded.total_minimizers, 1_000_000);
        assert_eq!(loaded.source_hash, 0xDEADBEEF);
    }

    #[test]
    fn test_is_parquet_index() {
        let tmp = TempDir::new().unwrap();

        // Not a directory
        let file_path = tmp.path().join("not_a_dir.ryidx");
        std::fs::write(&file_path, "test").unwrap();
        assert!(!is_parquet_index(&file_path));

        // Directory without manifest
        let empty_dir = tmp.path().join("empty.ryidx");
        std::fs::create_dir(&empty_dir).unwrap();
        assert!(!is_parquet_index(&empty_dir));

        // Valid Parquet index
        let valid_dir = tmp.path().join("valid.ryidx");
        create_index_directory(&valid_dir).unwrap();
        let manifest = ParquetManifest::new(64, 50, 0);
        manifest.save(&valid_dir).unwrap();
        assert!(is_parquet_index(&valid_dir));
    }
}
