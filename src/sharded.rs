//! Sharded inverted index structures.
//!
//! For very large indices that exceed available memory, the inverted index can be
//! split into multiple shard files that are loaded on-demand during classification.

use anyhow::{Result, anyhow};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use crate::constants::{MAX_INVERTED_MINIMIZERS, MAX_INVERTED_BUCKET_IDS};
use crate::inverted::InvertedIndex;
use crate::types::IndexMetadata;

/// Information about a single shard in a sharded inverted index.
#[derive(Debug, Clone)]
pub struct ShardInfo {
    /// Shard identifier (0-indexed)
    pub shard_id: u32,
    /// First minimizer value in this shard (inclusive)
    pub min_start: u64,
    /// Last minimizer value in this shard (exclusive), or 0 if this is the last shard
    pub min_end: u64,
    /// Whether this is the last shard (covers all values >= min_start)
    pub is_last_shard: bool,
    /// Number of minimizers in this shard
    pub num_minimizers: usize,
    /// Number of bucket ID entries in this shard
    pub num_bucket_ids: usize,
}

/// Manifest describing a sharded inverted index.
///
/// Format v3:
/// - Magic: "RYXM" (4 bytes)
/// - Version: 3 (u32)
/// - k (u64), w (u64), salt (u64), source_hash (u64)
/// - total_minimizers (u64), total_bucket_ids (u64)
/// - has_overlapping_shards (u8): 1 if shards have overlapping minimizer ranges, 0 otherwise
/// - num_shards (u32)
/// - For each shard: shard_id (u32), min_start (u64), min_end (u64), is_last_shard (u8), num_minimizers (u64), num_bucket_ids (u64)
///
/// # Shard Partitioning Strategies
///
/// **Range-partitioned** (`has_overlapping_shards = false`):
/// - Created by `InvertedIndex::save_sharded()` which splits a single inverted index
/// - Minimizer ranges are sorted and contiguous: `[0, 500), [500, 1000), ...`
/// - Each minimizer appears in exactly one shard
/// - `total_minimizers` equals the unique minimizer count
///
/// **Bucket-partitioned** (`has_overlapping_shards = true`):
/// - Created by inverting each main index shard independently (1:1 correspondence)
/// - Minimizer ranges may overlap (same minimizer can appear in multiple shards)
/// - `min_start`/`min_end` are advisory only (not sorted or contiguous)
/// - `total_minimizers` is the SUM across shards (includes duplicates, NOT unique count)
/// - Classification must iterate through ALL shards for each query
///
/// # Migration Note (v2 → v3)
///
/// Version 2 manifests did not have the `has_overlapping_shards` flag and assumed
/// range-partitioned shards. If you have v2 manifests that were created from sharded
/// main indices (bucket-partitioned), they will fail validation and must be regenerated.
#[derive(Debug, Clone)]
pub struct ShardManifest {
    pub k: usize,
    pub w: usize,
    pub salt: u64,
    pub source_hash: u64,
    /// Total minimizer entries across all shards.
    /// For range-partitioned shards: equals unique minimizer count (no duplicates).
    /// For overlapping shards: includes duplicates (same minimizer counted per shard).
    pub total_minimizers: usize,
    /// Total bucket ID entries across all shards (same duplicate semantics as total_minimizers).
    pub total_bucket_ids: usize,
    /// If true, shards have overlapping minimizer ranges and classification must check all shards.
    /// If false, shards are range-partitioned with sorted, contiguous, non-overlapping ranges.
    pub has_overlapping_shards: bool,
    pub shards: Vec<ShardInfo>,
}

impl ShardManifest {
    /// Maximum allowed shards in a manifest (prevents DoS via huge allocations)
    const MAX_SHARDS: u32 = 10_000;

    /// Save the manifest to a file.
    pub fn save(&self, path: &Path) -> Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);

        writer.write_all(b"RYXM")?;
        writer.write_all(&3u32.to_le_bytes())?;

        writer.write_all(&(self.k as u64).to_le_bytes())?;
        writer.write_all(&(self.w as u64).to_le_bytes())?;
        writer.write_all(&self.salt.to_le_bytes())?;
        writer.write_all(&self.source_hash.to_le_bytes())?;

        writer.write_all(&(self.total_minimizers as u64).to_le_bytes())?;
        writer.write_all(&(self.total_bucket_ids as u64).to_le_bytes())?;

        writer.write_all(&[if self.has_overlapping_shards { 1u8 } else { 0u8 }])?;
        writer.write_all(&(self.shards.len() as u32).to_le_bytes())?;

        for shard in &self.shards {
            writer.write_all(&shard.shard_id.to_le_bytes())?;
            writer.write_all(&shard.min_start.to_le_bytes())?;
            writer.write_all(&shard.min_end.to_le_bytes())?;
            writer.write_all(&[if shard.is_last_shard { 1u8 } else { 0u8 }])?;
            writer.write_all(&(shard.num_minimizers as u64).to_le_bytes())?;
            writer.write_all(&(shard.num_bucket_ids as u64).to_le_bytes())?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Load a manifest from a file.
    pub fn load(path: &Path) -> Result<Self> {
        let mut reader = BufReader::new(File::open(path)?);
        let mut buf1 = [0u8; 1];
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        reader.read_exact(&mut buf4)?;
        if &buf4 != b"RYXM" {
            return Err(anyhow!("Invalid shard manifest format (expected RYXM)"));
        }

        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if !matches!(version, 2 | 3) {
            return Err(anyhow!("Unsupported shard manifest version: {} (expected 2 or 3)", version));
        }

        reader.read_exact(&mut buf8)?;
        let k = u64::from_le_bytes(buf8) as usize;
        if !matches!(k, 16 | 32 | 64) {
            return Err(anyhow!("Invalid K value in manifest: {} (must be 16, 32, or 64)", k));
        }

        reader.read_exact(&mut buf8)?;
        let w = u64::from_le_bytes(buf8) as usize;

        reader.read_exact(&mut buf8)?;
        let salt = u64::from_le_bytes(buf8);

        reader.read_exact(&mut buf8)?;
        let source_hash = u64::from_le_bytes(buf8);

        reader.read_exact(&mut buf8)?;
        let total_minimizers = u64::from_le_bytes(buf8) as usize;
        if total_minimizers > MAX_INVERTED_MINIMIZERS {
            return Err(anyhow!(
                "Invalid total_minimizers in manifest: {} (max {})",
                total_minimizers, MAX_INVERTED_MINIMIZERS
            ));
        }

        reader.read_exact(&mut buf8)?;
        let total_bucket_ids = u64::from_le_bytes(buf8) as usize;
        if total_bucket_ids > MAX_INVERTED_BUCKET_IDS {
            return Err(anyhow!(
                "Invalid total_bucket_ids in manifest: {} (max {})",
                total_bucket_ids, MAX_INVERTED_BUCKET_IDS
            ));
        }

        // Version 3 adds has_overlapping_shards flag
        let has_overlapping_shards = if version >= 3 {
            reader.read_exact(&mut buf1)?;
            buf1[0] != 0
        } else {
            false // v2 manifests are always range-partitioned
        };

        reader.read_exact(&mut buf4)?;
        let num_shards = u32::from_le_bytes(buf4);

        if num_shards > Self::MAX_SHARDS {
            return Err(anyhow!("Too many shards: {} (max {})", num_shards, Self::MAX_SHARDS));
        }

        let mut shards = Vec::with_capacity(num_shards as usize);
        for _ in 0..num_shards {
            reader.read_exact(&mut buf4)?;
            let shard_id = u32::from_le_bytes(buf4);

            reader.read_exact(&mut buf8)?;
            let min_start = u64::from_le_bytes(buf8);

            reader.read_exact(&mut buf8)?;
            let min_end = u64::from_le_bytes(buf8);

            reader.read_exact(&mut buf1)?;
            let is_last_shard = buf1[0] != 0;

            reader.read_exact(&mut buf8)?;
            let num_minimizers = u64::from_le_bytes(buf8) as usize;

            reader.read_exact(&mut buf8)?;
            let num_bucket_ids = u64::from_le_bytes(buf8) as usize;

            shards.push(ShardInfo {
                shard_id,
                min_start,
                min_end,
                is_last_shard,
                num_minimizers,
                num_bucket_ids,
            });
        }

        // Validate shard structure
        if !shards.is_empty() {
            // Check shard IDs are sequential (applies to all shard types)
            for i in 1..shards.len() {
                if shards[i].shard_id != shards[i - 1].shard_id + 1 {
                    return Err(anyhow!(
                        "Invalid manifest: shard IDs not sequential (shard {} followed by {})",
                        shards[i - 1].shard_id, shards[i].shard_id
                    ));
                }
            }

            // Range-partitioned shards have strict ordering and contiguity requirements.
            // Bucket-partitioned shards can have overlapping/unordered ranges.
            if !has_overlapping_shards {
                for i in 1..shards.len() {
                    if shards[i].min_start < shards[i - 1].min_start {
                        return Err(anyhow!(
                            "Invalid manifest: shards not sorted by min_start (shard {} has min_start {} < shard {} min_start {})",
                            shards[i].shard_id, shards[i].min_start, shards[i - 1].shard_id, shards[i - 1].min_start
                        ));
                    }
                }

                for i in 0..shards.len() - 1 {
                    if shards[i].is_last_shard {
                        return Err(anyhow!(
                            "Invalid manifest: shard {} marked as last but not final shard",
                            shards[i].shard_id
                        ));
                    }
                    if shards[i].min_end != shards[i + 1].min_start {
                        return Err(anyhow!(
                            "Invalid manifest: shard ranges not contiguous (shard {} ends at {}, shard {} starts at {})",
                            shards[i].shard_id, shards[i].min_end, shards[i + 1].shard_id, shards[i + 1].min_start
                        ));
                    }
                }

                if let Some(last) = shards.last() {
                    if !last.is_last_shard {
                        return Err(anyhow!(
                            "Invalid manifest: final shard {} not marked as is_last_shard",
                            last.shard_id
                        ));
                    }
                }
            }

            // Total counts must match (applies to all shard types)
            let sum_minimizers: usize = shards.iter().map(|s| s.num_minimizers).sum();
            let sum_bucket_ids: usize = shards.iter().map(|s| s.num_bucket_ids).sum();
            if sum_minimizers != total_minimizers {
                return Err(anyhow!(
                    "Invalid manifest: shard minimizer counts sum to {}, expected {}",
                    sum_minimizers, total_minimizers
                ));
            }
            if sum_bucket_ids != total_bucket_ids {
                return Err(anyhow!(
                    "Invalid manifest: shard bucket_id counts sum to {}, expected {}",
                    sum_bucket_ids, total_bucket_ids
                ));
            }
        }

        Ok(ShardManifest {
            k,
            w,
            salt,
            source_hash,
            total_minimizers,
            total_bucket_ids,
            has_overlapping_shards,
            shards,
        })
    }

    /// Get the path for the manifest file given a base path.
    pub fn manifest_path(base: &Path) -> PathBuf {
        let mut path = base.to_path_buf();
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        path.set_file_name(format!("{}.manifest", name));
        path
    }

    /// Get the path for a shard file given a base path and shard ID.
    pub fn shard_path(base: &Path, shard_id: u32) -> PathBuf {
        let mut path = base.to_path_buf();
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        path.set_file_name(format!("{}.shard.{}", name, shard_id));
        path
    }
}

/// Handle for a sharded inverted index.
///
/// This struct holds a manifest describing the shards. Shards are loaded
/// on-demand during classification via `classify_batch_sharded_sequential`.
#[derive(Debug)]
pub struct ShardedInvertedIndex {
    manifest: ShardManifest,
    base_path: PathBuf,
}

impl ShardedInvertedIndex {
    /// Open a sharded inverted index by loading just the manifest.
    pub fn open(base_path: &Path) -> Result<Self> {
        let manifest_path = ShardManifest::manifest_path(base_path);
        let manifest = ShardManifest::load(&manifest_path)?;

        Ok(ShardedInvertedIndex {
            manifest,
            base_path: base_path.to_path_buf(),
        })
    }

    /// Returns the K value (k-mer size).
    pub fn k(&self) -> usize {
        self.manifest.k
    }

    /// Returns the window size.
    pub fn w(&self) -> usize {
        self.manifest.w
    }

    /// Returns the salt value.
    pub fn salt(&self) -> u64 {
        self.manifest.salt
    }

    /// Returns the source hash for validation.
    pub fn source_hash(&self) -> u64 {
        self.manifest.source_hash
    }

    /// Returns the total number of shards.
    pub fn num_shards(&self) -> usize {
        self.manifest.shards.len()
    }

    /// Returns the total number of minimizers across all shards.
    pub fn total_minimizers(&self) -> usize {
        self.manifest.total_minimizers
    }

    /// Returns the total number of bucket ID entries across all shards.
    pub fn total_bucket_ids(&self) -> usize {
        self.manifest.total_bucket_ids
    }

    /// Returns a reference to the manifest.
    pub fn manifest(&self) -> &ShardManifest {
        &self.manifest
    }

    /// Returns a reference to the base path.
    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    /// Validate against Index metadata.
    pub fn validate_against_metadata(&self, metadata: &IndexMetadata) -> Result<()> {
        if self.manifest.k != metadata.k {
            return Err(anyhow!("K mismatch: sharded index has K={}, metadata has K={}",
                self.manifest.k, metadata.k));
        }
        if self.manifest.w != metadata.w {
            return Err(anyhow!("W mismatch: sharded index has W={}, metadata has W={}",
                self.manifest.w, metadata.w));
        }
        if self.manifest.salt != metadata.salt {
            return Err(anyhow!("Salt mismatch: sharded index has salt={:#x}, metadata has salt={:#x}",
                self.manifest.salt, metadata.salt));
        }

        let expected_hash = InvertedIndex::compute_metadata_hash(metadata);
        if self.manifest.source_hash != expected_hash {
            return Err(anyhow!("Source hash mismatch: sharded index is stale or was built from different source"));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::Index;
    use tempfile::NamedTempFile;

    #[test]
    fn test_shard_manifest_save_load_roundtrip() -> Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();

        let manifest = ShardManifest {
            k: 64,
            w: 50,
            salt: 0xDEADBEEF,
            source_hash: 0x12345678,
            total_minimizers: 1000,
            total_bucket_ids: 5000,
            has_overlapping_shards: false,
            shards: vec![
                ShardInfo {
                    shard_id: 0,
                    min_start: 0,
                    min_end: 500,
                    is_last_shard: false,
                    num_minimizers: 400,
                    num_bucket_ids: 2000,
                },
                ShardInfo {
                    shard_id: 1,
                    min_start: 500,
                    min_end: 0,
                    is_last_shard: true,
                    num_minimizers: 600,
                    num_bucket_ids: 3000,
                },
            ],
        };

        manifest.save(&path)?;
        let loaded = ShardManifest::load(&path)?;

        assert_eq!(loaded.k, 64);
        assert_eq!(loaded.w, 50);
        assert_eq!(loaded.salt, 0xDEADBEEF);
        assert_eq!(loaded.source_hash, 0x12345678);
        assert_eq!(loaded.total_minimizers, 1000);
        assert_eq!(loaded.total_bucket_ids, 5000);
        assert!(!loaded.has_overlapping_shards);
        assert_eq!(loaded.shards.len(), 2);

        assert_eq!(loaded.shards[0].shard_id, 0);
        assert_eq!(loaded.shards[0].min_start, 0);
        assert_eq!(loaded.shards[0].min_end, 500);
        assert!(!loaded.shards[0].is_last_shard);
        assert_eq!(loaded.shards[0].num_minimizers, 400);
        assert_eq!(loaded.shards[0].num_bucket_ids, 2000);

        assert_eq!(loaded.shards[1].shard_id, 1);
        assert_eq!(loaded.shards[1].min_start, 500);
        assert_eq!(loaded.shards[1].min_end, 0);
        assert!(loaded.shards[1].is_last_shard);
        assert_eq!(loaded.shards[1].num_minimizers, 600);
        assert_eq!(loaded.shards[1].num_bucket_ids, 3000);

        Ok(())
    }

    #[test]
    fn test_shard_manifest_overlapping_roundtrip() -> Result<()> {
        // Test round-trip for bucket-partitioned (overlapping) shards
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();

        // Simulate overlapping shards created from main index shards
        // Note: min_start values are NOT sorted, ranges overlap, min_end is 0
        let manifest = ShardManifest {
            k: 32,
            w: 20,
            salt: 0xCAFEBABE,
            source_hash: 0x87654321,
            total_minimizers: 150, // Sum with duplicates, not unique count
            total_bucket_ids: 150,
            has_overlapping_shards: true,
            shards: vec![
                ShardInfo {
                    shard_id: 0,
                    min_start: 100,  // Overlaps with shard 1
                    min_end: 0,
                    is_last_shard: false,
                    num_minimizers: 50,
                    num_bucket_ids: 50,
                },
                ShardInfo {
                    shard_id: 1,
                    min_start: 100,  // Same min_start as shard 0 (overlapping)
                    min_end: 0,
                    is_last_shard: false,
                    num_minimizers: 50,
                    num_bucket_ids: 50,
                },
                ShardInfo {
                    shard_id: 2,
                    min_start: 50,   // Lower than previous shards (not sorted)
                    min_end: 0,
                    is_last_shard: true,
                    num_minimizers: 50,
                    num_bucket_ids: 50,
                },
            ],
        };

        manifest.save(&path)?;
        let loaded = ShardManifest::load(&path)?;

        // Verify all fields round-trip correctly
        assert_eq!(loaded.k, 32);
        assert_eq!(loaded.w, 20);
        assert_eq!(loaded.salt, 0xCAFEBABE);
        assert_eq!(loaded.source_hash, 0x87654321);
        assert_eq!(loaded.total_minimizers, 150);
        assert_eq!(loaded.total_bucket_ids, 150);
        assert!(loaded.has_overlapping_shards); // Critical: flag preserved
        assert_eq!(loaded.shards.len(), 3);

        // Verify shard info preserved (including "invalid" range data)
        assert_eq!(loaded.shards[0].min_start, 100);
        assert_eq!(loaded.shards[1].min_start, 100); // Same as shard 0
        assert_eq!(loaded.shards[2].min_start, 50);  // Lower than previous

        Ok(())
    }

    #[test]
    fn test_shard_manifest_invalid_magic() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path();
        std::fs::write(path, b"NOTM").unwrap();

        let result = ShardManifest::load(path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid shard manifest format"));
    }

    #[test]
    fn test_shard_manifest_invalid_version() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path();

        let mut data = Vec::new();
        data.extend_from_slice(b"RYXM");
        data.extend_from_slice(&99u32.to_le_bytes());
        std::fs::write(path, data).unwrap();

        let result = ShardManifest::load(path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unsupported shard manifest version"));
    }

    #[test]
    fn test_shard_manifest_path_helpers() {
        let base = std::path::Path::new("/tmp/test.ryxdi");

        let manifest_path = ShardManifest::manifest_path(base);
        assert_eq!(manifest_path.to_str().unwrap(), "/tmp/test.ryxdi.manifest");

        let shard_path = ShardManifest::shard_path(base, 3);
        assert_eq!(shard_path.to_str().unwrap(), "/tmp/test.ryxdi.shard.3");
    }

    #[test]
    fn test_sharded_inverted_index_open() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let base_path = dir.path().join("test.ryxdi");

        let mut index = Index::new(64, 50, 0x1234).unwrap();
        index.buckets.insert(1, vec![100, 200, 300, 400]);
        index.bucket_names.insert(1, "A".into());
        let inverted = InvertedIndex::build_from_index(&index);
        inverted.save_sharded(&base_path, 2)?;

        let sharded = ShardedInvertedIndex::open(&base_path)?;

        assert_eq!(sharded.k(), 64);
        assert_eq!(sharded.w(), 50);
        assert_eq!(sharded.salt(), 0x1234);
        assert_eq!(sharded.num_shards(), 2);

        Ok(())
    }
}
