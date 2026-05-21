//! Manifest and bucket-data types for the `.ryci` cluster-index format.
//!
//! Mirrors the structure of `crate::indices::parquet::manifest` but with a
//! distinct magic, a distinct file extension, and a `ClusterBucketData`
//! that carries per-minimizer positions alongside the minimizer values.

use crate::error::{Result, RypeError};
use crate::indices::parquet::hex_u64;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

use super::{files, FORMAT_MAGIC_CLUSTER, FORMAT_VERSION_CLUSTER};

/// Cluster-index manifest. Stored as `manifest.toml` for human readability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterParquetManifest {
    /// Magic string for format identification (`RYPE_CLUSTER_V1`).
    pub magic: String,
    /// Format version for compatibility checking.
    pub format_version: u32,
    /// K-mer size (16, 32, or 64).
    pub k: usize,
    /// Window size for minimizer selection.
    pub w: usize,
    /// XOR salt applied to k-mer hashes (hex string in TOML).
    #[serde(with = "hex_u64")]
    pub salt: u64,
    /// Hash of source data for change detection (hex string in TOML).
    #[serde(with = "hex_u64")]
    pub source_hash: u64,
    /// Number of buckets.
    pub num_buckets: u32,
    /// Total minimizers across all buckets.
    pub total_minimizers: u64,
    /// Inverted-index shard manifest (set once shards are written).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inverted: Option<ClusterInvertedManifest>,
}

/// Top-level inverted-index manifest section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterInvertedManifest {
    /// Number of shard files.
    pub num_shards: u32,
    /// Total `(minimizer, bucket_id, position)` triples across all shards.
    pub total_entries: u64,
    /// True if buckets may share minimizers across shards.
    pub has_overlapping_shards: bool,
    /// Per-shard descriptors.
    pub shards: Vec<ClusterInvertedShardInfo>,
}

/// Per-shard descriptor.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ClusterInvertedShardInfo {
    pub shard_id: u32,
    /// Minimum minimizer value in this shard (hex string).
    #[serde(with = "hex_u64")]
    pub min_minimizer: u64,
    /// Maximum minimizer value in this shard (hex string).
    #[serde(with = "hex_u64")]
    pub max_minimizer: u64,
    /// Triple count in this shard.
    pub num_entries: u64,
}

impl ClusterParquetManifest {
    pub fn new(k: usize, w: usize, salt: u64) -> Self {
        Self {
            magic: FORMAT_MAGIC_CLUSTER.to_string(),
            format_version: FORMAT_VERSION_CLUSTER,
            k,
            w,
            salt,
            source_hash: 0,
            num_buckets: 0,
            total_minimizers: 0,
            inverted: None,
        }
    }

    /// Write to `<index_dir>/manifest.toml`.
    pub fn save(&self, index_dir: &Path) -> Result<()> {
        let path = index_dir.join(files::MANIFEST);
        let toml_str = toml::to_string_pretty(self)
            .map_err(|e| RypeError::encoding(format!("serialize cluster manifest: {}", e)))?;
        fs::write(&path, &toml_str)
            .map_err(|e| RypeError::io(path.clone(), "write cluster manifest", e))?;
        Ok(())
    }

    /// Read from `<index_dir>/manifest.toml`. Rejects wrong magic and newer-than-supported version.
    pub fn load(index_dir: &Path) -> Result<Self> {
        let path = index_dir.join(files::MANIFEST);
        let toml_str = fs::read_to_string(&path)
            .map_err(|e| RypeError::io(path.clone(), "read cluster manifest", e))?;
        let manifest: Self = toml::from_str(&toml_str).map_err(|e| {
            RypeError::format(path.clone(), format!("parse cluster manifest: {}", e))
        })?;

        if manifest.magic != FORMAT_MAGIC_CLUSTER {
            return Err(RypeError::format(
                path,
                format!(
                    "invalid cluster manifest magic: expected '{}', got '{}'",
                    FORMAT_MAGIC_CLUSTER, manifest.magic
                ),
            ));
        }
        if manifest.format_version > FORMAT_VERSION_CLUSTER {
            return Err(RypeError::format(
                path,
                format!(
                    "unsupported cluster format version: {} (max supported: {})",
                    manifest.format_version, FORMAT_VERSION_CLUSTER
                ),
            ));
        }
        // Invariant: every minimizer becomes exactly one (minimizer, bucket_id,
        // position) triple, so the two totals must agree when the inverted
        // section is populated. Catching a mismatch here means downstream code
        // can pick either field without worrying about which is canonical.
        if let Some(inv) = &manifest.inverted {
            if inv.total_entries != manifest.total_minimizers {
                return Err(RypeError::format(
                    path,
                    format!(
                        "cluster manifest invariant violated: total_minimizers={} != inverted.total_entries={}",
                        manifest.total_minimizers, inv.total_entries
                    ),
                ));
            }
        }
        Ok(manifest)
    }
}

/// One bucket of input data for `.ryci` construction.
///
/// `minimizers` and `positions` are parallel arrays of the same length;
/// `minimizers[i]` is the i-th minimizer of this bucket, `positions[i]` is
/// the start position of that k-mer in the source contig.
#[derive(Debug, Clone)]
pub struct ClusterBucketData {
    pub bucket_id: u32,
    pub bucket_name: String,
    pub sources: Vec<String>,
    /// Sorted ascending, no duplicates within bucket. Same invariant as `.ryxdi`'s `BucketData`.
    pub minimizers: Vec<u64>,
    /// Index-parallel with `minimizers`. No ordering constraint (positions correspond 1:1
    /// to minimizers, so ordering follows from the minimizer sort).
    pub positions: Vec<u32>,
}

impl ClusterBucketData {
    /// Verify invariants:
    /// 1. `minimizers.len() == positions.len()` (parallel arrays).
    /// 2. `minimizers` is strictly ascending (sorted + no duplicates within bucket).
    pub fn validate(&self) -> Result<()> {
        if self.minimizers.len() != self.positions.len() {
            return Err(RypeError::validation(format!(
                "bucket {} has minimizer/position length mismatch: {} vs {}",
                self.bucket_id,
                self.minimizers.len(),
                self.positions.len()
            )));
        }
        for i in 1..self.minimizers.len() {
            if self.minimizers[i] == self.minimizers[i - 1] {
                return Err(RypeError::validation(format!(
                    "bucket {} has duplicate minimizer at position {}: {:#x}",
                    self.bucket_id, i, self.minimizers[i]
                )));
            }
            if self.minimizers[i] < self.minimizers[i - 1] {
                return Err(RypeError::validation(format!(
                    "bucket {} has unsorted minimizers at positions {}-{}: {:#x} > {:#x}",
                    self.bucket_id,
                    i - 1,
                    i,
                    self.minimizers[i - 1],
                    self.minimizers[i]
                )));
            }
        }
        Ok(())
    }
}

/// Returns true iff `path` is a `.ryci` cluster-index directory (per its manifest magic).
pub fn is_cluster_parquet_index(path: &Path) -> bool {
    if !path.is_dir() {
        return false;
    }
    let manifest_path = path.join(files::MANIFEST);
    if !manifest_path.exists() {
        return false;
    }
    if let Ok(content) = fs::read_to_string(&manifest_path) {
        return content.contains(FORMAT_MAGIC_CLUSTER);
    }
    false
}

/// Create the directory structure for a new `.ryci` index.
pub fn create_cluster_index_directory(path: &Path) -> Result<()> {
    fs::create_dir_all(path)
        .map_err(|e| RypeError::io(path.to_path_buf(), "create cluster index directory", e))?;
    let inverted_dir = path.join(files::INVERTED_DIR);
    fs::create_dir_all(&inverted_dir)
        .map_err(|e| RypeError::io(inverted_dir.clone(), "create cluster inverted directory", e))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // WHY: round-trip is the load-bearing contract for any on-disk format.
    // If write→read doesn't restore identity, every downstream consumer breaks.
    #[test]
    fn manifest_round_trip_preserves_all_fields() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("test.ryci");
        create_cluster_index_directory(&dir).unwrap();

        let mut manifest = ClusterParquetManifest::new(64, 50, 0x5555_5555_5555_5555);
        manifest.num_buckets = 7;
        manifest.total_minimizers = 12345;
        manifest.source_hash = 0xDEAD_BEEF_CAFE_BABE;
        manifest.inverted = Some(ClusterInvertedManifest {
            num_shards: 2,
            total_entries: 12345,
            has_overlapping_shards: true,
            shards: vec![
                ClusterInvertedShardInfo {
                    shard_id: 0,
                    min_minimizer: 1,
                    max_minimizer: 5000,
                    num_entries: 7000,
                },
                ClusterInvertedShardInfo {
                    shard_id: 1,
                    min_minimizer: 5000,
                    max_minimizer: u64::MAX,
                    num_entries: 5345,
                },
            ],
        });

        manifest.save(&dir).unwrap();
        let loaded = ClusterParquetManifest::load(&dir).unwrap();

        assert_eq!(loaded.magic, FORMAT_MAGIC_CLUSTER);
        assert_eq!(loaded.format_version, FORMAT_VERSION_CLUSTER);
        assert_eq!(loaded.k, 64);
        assert_eq!(loaded.w, 50);
        assert_eq!(loaded.salt, 0x5555_5555_5555_5555);
        assert_eq!(loaded.num_buckets, 7);
        assert_eq!(loaded.total_minimizers, 12345);
        assert_eq!(loaded.source_hash, 0xDEAD_BEEF_CAFE_BABE);
        let inv = loaded.inverted.unwrap();
        assert_eq!(inv.num_shards, 2);
        assert_eq!(inv.shards[1].max_minimizer, u64::MAX);
        assert!(inv.has_overlapping_shards);
    }

    // WHY: opening the wrong format must fail cleanly. If a .ryxdi manifest
    // squeaked through here, downstream `.ryci` consumers would crash on
    // the missing position column with a confusing error.
    #[test]
    fn manifest_load_rejects_wrong_magic() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("wrong.ryci");
        create_cluster_index_directory(&dir).unwrap();

        let bad_toml = r#"
magic = "RYPE_PARQUET_V1"
format_version = 1
k = 64
w = 50
salt = "0x5555555555555555"
source_hash = "0x0"
num_buckets = 0
total_minimizers = 0
"#;
        fs::write(dir.join(files::MANIFEST), bad_toml).unwrap();

        let err = ClusterParquetManifest::load(&dir).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("invalid cluster manifest magic"),
            "expected magic-rejection error, got: {}",
            msg
        );
    }

    // WHY: total_minimizers and inverted.total_entries must agree — every minimizer
    // becomes exactly one triple. A mismatch on disk would mean either the writer
    // disagreed with itself or a manifest was hand-edited; either way the file
    // is unsafe to use without flagging.
    #[test]
    fn manifest_load_rejects_total_mismatch() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("mismatch.ryci");
        create_cluster_index_directory(&dir).unwrap();

        let bad_toml = r#"
magic = "RYPE_CLUSTER_V1"
format_version = 1
k = 64
w = 50
salt = "0x0"
source_hash = "0x0"
num_buckets = 1
total_minimizers = 100

[inverted]
num_shards = 1
total_entries = 99
has_overlapping_shards = false

[[inverted.shards]]
shard_id = 0
min_minimizer = "0x1"
max_minimizer = "0xff"
num_entries = 99
"#;
        fs::write(dir.join(files::MANIFEST), bad_toml).unwrap();

        let err = ClusterParquetManifest::load(&dir).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("invariant violated"),
            "expected invariant-violation error, got: {}",
            msg
        );
    }

    // WHY: when we bump FORMAT_VERSION_CLUSTER, older binaries must refuse
    // newer files rather than silently misinterpret them.
    #[test]
    fn manifest_load_rejects_newer_version() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("newer.ryci");
        create_cluster_index_directory(&dir).unwrap();

        let too_new = r#"
magic = "RYPE_CLUSTER_V1"
format_version = 99
k = 64
w = 50
salt = "0x5555555555555555"
source_hash = "0x0"
num_buckets = 0
total_minimizers = 0
"#;
        fs::write(dir.join(files::MANIFEST), too_new).unwrap();

        let err = ClusterParquetManifest::load(&dir).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("unsupported cluster format version"),
            "expected version-rejection error, got: {}",
            msg
        );
    }

    fn make_bucket(mins: Vec<u64>, poss: Vec<u32>) -> ClusterBucketData {
        ClusterBucketData {
            bucket_id: 1,
            bucket_name: "test".to_string(),
            sources: vec![],
            minimizers: mins,
            positions: poss,
        }
    }

    #[test]
    fn validate_accepts_well_formed_bucket() {
        let b = make_bucket(vec![1, 5, 9, 42], vec![100, 200, 300, 50]);
        assert!(b.validate().is_ok());
    }

    #[test]
    fn validate_accepts_empty_bucket() {
        // Empty buckets are legal: a contig that produces no minimizers.
        let b = make_bucket(vec![], vec![]);
        assert!(b.validate().is_ok());
    }

    // WHY: length mismatch would silently align minimizers to wrong positions
    // — chains would chase phantom coordinates. Catch at validate time.
    #[test]
    fn validate_rejects_length_mismatch() {
        let b = make_bucket(vec![1, 5, 9], vec![100, 200]);
        let err = b.validate().unwrap_err();
        assert!(format!("{}", err).contains("length mismatch"));
    }

    // WHY: shards are sorted-by-minimizer for the merge-join contract;
    // an unsorted bucket breaks that contract.
    #[test]
    fn validate_rejects_unsorted_minimizers() {
        let b = make_bucket(vec![5, 1, 9], vec![100, 200, 300]);
        let err = b.validate().unwrap_err();
        assert!(format!("{}", err).contains("unsorted"));
    }

    // WHY: duplicates within a bucket break the (minimizer, bucket_id) sort
    // assumption that the merge-join relies on.
    #[test]
    fn validate_rejects_duplicate_minimizer_within_bucket() {
        let b = make_bucket(vec![1, 5, 5, 9], vec![100, 200, 250, 300]);
        let err = b.validate().unwrap_err();
        assert!(format!("{}", err).contains("duplicate"));
    }

    #[test]
    fn is_cluster_parquet_index_recognizes_valid_directory() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("valid.ryci");
        create_cluster_index_directory(&dir).unwrap();
        let manifest = ClusterParquetManifest::new(64, 50, 0);
        manifest.save(&dir).unwrap();
        assert!(is_cluster_parquet_index(&dir));
    }

    #[test]
    fn is_cluster_parquet_index_rejects_missing_manifest() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("empty.ryci");
        std::fs::create_dir(&dir).unwrap();
        assert!(!is_cluster_parquet_index(&dir));
    }

    #[test]
    fn is_cluster_parquet_index_rejects_non_directory() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("not_a_dir");
        std::fs::write(&path, "anything").unwrap();
        assert!(!is_cluster_parquet_index(&path));
    }
}
