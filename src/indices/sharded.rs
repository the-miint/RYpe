//! Sharded inverted index structures.
//!
//! All inverted indices use the sharded format, even when there's only one shard.
//! This unified approach simplifies the codebase at the cost of having two files
//! (manifest + shard) instead of one for small indices.
//!
//! For small indices, the overhead is minimal: an extra file open and a small manifest
//! read. For large indices, sharding enables memory-efficient classification by loading
//! one shard at a time.

use crate::error::{Result, RypeError};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use super::inverted::InvertedIndex;
use crate::constants::{
    MANIFEST_MAGIC, MANIFEST_VERSION, MAX_INVERTED_BUCKET_IDS, MAX_INVERTED_MINIMIZERS, MAX_SHARDS,
};
use crate::types::IndexMetadata;

/// Format for inverted index shard files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShardFormat {
    /// Legacy RYXS format (custom binary with zstd compression)
    #[default]
    Legacy,
    /// Parquet format (requires parquet feature)
    Parquet,
}

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
/// Format v5:
/// - Magic: "RYXM" (4 bytes)
/// - Version: 5 (u32)
/// - k (u64), w (u64), salt (u64), source_hash (u64)
/// - total_minimizers (u64), total_bucket_ids (u64)
/// - has_overlapping_shards (u8): 1 if shards have overlapping minimizer ranges, 0 otherwise
/// - shard_format (u8): 0 = Legacy, 1 = Parquet (FIX #4: explicit format storage)
/// - num_shards (u32)
/// - For each shard: shard_id (u32), min_start (u64), min_end (u64), is_last_shard (u8), num_minimizers (u64), num_bucket_ids (u64)
/// - Bucket metadata (same as v4)
///
/// # Shard Partitioning
///
/// Shards are created by inverting each main index shard independently (1:1 correspondence).
/// This is called "bucket-partitioned" because each main shard contains complete buckets,
/// so each inverted shard contains only bucket IDs from its corresponding main shard.
///
/// **Key consequence**: Since different buckets can share the same minimizers, the same
/// minimizer value can appear in multiple inverted shards. This is why `has_overlapping_shards`
/// is always `true` for bucket-partitioned shards.
///
/// - For non-sharded main index, creates a 1-shard inverted index
/// - `min_start`/`min_end` are advisory only (not sorted or contiguous across shards)
/// - `total_minimizers` is the SUM across shards (includes duplicates, NOT unique count)
/// - Classification must iterate through ALL shards for each query (no range-based skipping)
#[derive(Debug, Clone)]
pub struct ShardManifest {
    pub k: usize,
    pub w: usize,
    pub salt: u64,
    pub source_hash: u64,
    /// Total minimizer entries across all shards (includes duplicates across shards).
    pub total_minimizers: usize,
    /// Total bucket ID entries across all shards (includes duplicates across shards).
    pub total_bucket_ids: usize,
    /// Always `true` for bucket-partitioned shards (the only type currently supported).
    /// When true, shards have overlapping minimizer ranges and classification must check all shards.
    pub has_overlapping_shards: bool,
    /// FIX #4: Explicit shard format stored in manifest (no file-existence guessing).
    pub shard_format: ShardFormat,
    pub shards: Vec<ShardInfo>,
    /// Bucket names (v4+). Maps bucket_id to human-readable name.
    pub bucket_names: HashMap<u32, String>,
    /// Bucket sources (v4+). Maps bucket_id to list of source sequence names.
    pub bucket_sources: HashMap<u32, Vec<String>>,
    /// Bucket minimizer counts (v4+). Maps bucket_id to number of minimizers.
    pub bucket_minimizer_counts: HashMap<u32, usize>,
}

impl ShardManifest {
    /// Save the manifest to a file (v5 format with shard_format and bucket metadata).
    pub fn save(&self, path: &Path) -> Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);

        writer.write_all(MANIFEST_MAGIC)?;
        writer.write_all(&MANIFEST_VERSION.to_le_bytes())?;

        writer.write_all(&(self.k as u64).to_le_bytes())?;
        writer.write_all(&(self.w as u64).to_le_bytes())?;
        writer.write_all(&self.salt.to_le_bytes())?;
        writer.write_all(&self.source_hash.to_le_bytes())?;

        writer.write_all(&(self.total_minimizers as u64).to_le_bytes())?;
        writer.write_all(&(self.total_bucket_ids as u64).to_le_bytes())?;

        writer.write_all(&[if self.has_overlapping_shards {
            1u8
        } else {
            0u8
        }])?;

        // FIX #4: Write shard format explicitly
        writer.write_all(&[match self.shard_format {
            ShardFormat::Legacy => 0u8,
            ShardFormat::Parquet => 1u8,
        }])?;

        writer.write_all(&(self.shards.len() as u32).to_le_bytes())?;

        for shard in &self.shards {
            writer.write_all(&shard.shard_id.to_le_bytes())?;
            writer.write_all(&shard.min_start.to_le_bytes())?;
            writer.write_all(&shard.min_end.to_le_bytes())?;
            writer.write_all(&[if shard.is_last_shard { 1u8 } else { 0u8 }])?;
            writer.write_all(&(shard.num_minimizers as u64).to_le_bytes())?;
            writer.write_all(&(shard.num_bucket_ids as u64).to_le_bytes())?;
        }

        // V4: Write bucket metadata
        let num_buckets = self.bucket_names.len() as u32;
        writer.write_all(&num_buckets.to_le_bytes())?;

        // Sort bucket IDs for deterministic output
        let mut sorted_bucket_ids: Vec<_> = self.bucket_names.keys().copied().collect();
        sorted_bucket_ids.sort_unstable();

        for bucket_id in &sorted_bucket_ids {
            writer.write_all(&bucket_id.to_le_bytes())?;

            // Write name
            let name = self
                .bucket_names
                .get(bucket_id)
                .map(|s| s.as_str())
                .unwrap_or("");
            let name_bytes = name.as_bytes();
            writer.write_all(&(name_bytes.len() as u64).to_le_bytes())?;
            writer.write_all(name_bytes)?;

            // Write minimizer count
            let minimizer_count = self
                .bucket_minimizer_counts
                .get(bucket_id)
                .copied()
                .unwrap_or(0);
            writer.write_all(&(minimizer_count as u64).to_le_bytes())?;

            // Write sources
            let empty_sources: Vec<String> = Vec::new();
            let sources = self.bucket_sources.get(bucket_id).unwrap_or(&empty_sources);
            writer.write_all(&(sources.len() as u64).to_le_bytes())?;
            for src in sources {
                let s_bytes = src.as_bytes();
                writer.write_all(&(s_bytes.len() as u64).to_le_bytes())?;
                writer.write_all(s_bytes)?;
            }
        }

        writer.flush()?;
        Ok(())
    }

    /// Load a manifest from a file.
    ///
    /// Supports v3 (backward compat), v4 (bucket metadata), and v5 (shard_format field).
    pub fn load(path: &Path) -> Result<Self> {
        use crate::constants::{MAX_NUM_BUCKETS, MAX_STRING_LENGTH};

        let mut reader = BufReader::new(File::open(path)?);
        let mut buf1 = [0u8; 1];
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        reader.read_exact(&mut buf4)?;
        if &buf4 != MANIFEST_MAGIC {
            return Err(RypeError::format(
                path,
                "Invalid shard manifest format (expected RYXM)",
            ));
        }

        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if !matches!(version, 3..=MANIFEST_VERSION) {
            return Err(RypeError::format(
                path,
                format!(
                    "Unsupported inverted index manifest version: {} (expected 3, 4, or 5). \
                     Regenerate it: rype index invert -i <main-index.ryidx>",
                    version
                ),
            ));
        }

        reader.read_exact(&mut buf8)?;
        let k = u64::from_le_bytes(buf8) as usize;
        if !matches!(k, 16 | 32 | 64) {
            return Err(RypeError::format(
                path,
                format!("Invalid K value in manifest: {} (must be 16, 32, or 64)", k),
            ));
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
            return Err(RypeError::overflow(
                "total_minimizers in manifest",
                MAX_INVERTED_MINIMIZERS,
                total_minimizers,
            ));
        }

        reader.read_exact(&mut buf8)?;
        let total_bucket_ids = u64::from_le_bytes(buf8) as usize;
        if total_bucket_ids > MAX_INVERTED_BUCKET_IDS {
            return Err(RypeError::overflow(
                "total_bucket_ids in manifest",
                MAX_INVERTED_BUCKET_IDS,
                total_bucket_ids,
            ));
        }

        reader.read_exact(&mut buf1)?;
        let has_overlapping_shards = buf1[0] != 0;

        // FIX #4: Read shard_format for v5+, default to Legacy for older versions
        let shard_format = if version >= 5 {
            reader.read_exact(&mut buf1)?;
            match buf1[0] {
                0 => ShardFormat::Legacy,
                1 => ShardFormat::Parquet,
                other => {
                    return Err(RypeError::format(
                        path,
                        format!("Invalid shard_format value: {} (expected 0 or 1)", other),
                    ));
                }
            }
        } else {
            ShardFormat::Legacy // Default for v3/v4
        };

        reader.read_exact(&mut buf4)?;
        let num_shards = u32::from_le_bytes(buf4);

        if num_shards > MAX_SHARDS {
            return Err(RypeError::overflow(
                "number of shards",
                MAX_SHARDS as usize,
                num_shards as usize,
            ));
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
            // Check shard IDs are sequential
            for i in 1..shards.len() {
                if shards[i].shard_id != shards[i - 1].shard_id + 1 {
                    return Err(RypeError::format(
                        path,
                        format!(
                            "shard IDs not sequential (shard {} followed by {})",
                            shards[i - 1].shard_id,
                            shards[i].shard_id
                        ),
                    ));
                }
            }

            // Total counts must match
            let sum_minimizers: usize = shards.iter().map(|s| s.num_minimizers).sum();
            let sum_bucket_ids: usize = shards.iter().map(|s| s.num_bucket_ids).sum();
            if sum_minimizers != total_minimizers {
                return Err(RypeError::format(
                    path,
                    format!(
                        "shard minimizer counts sum to {}, expected {}",
                        sum_minimizers, total_minimizers
                    ),
                ));
            }
            if sum_bucket_ids != total_bucket_ids {
                return Err(RypeError::format(
                    path,
                    format!(
                        "shard bucket_id counts sum to {}, expected {}",
                        sum_bucket_ids, total_bucket_ids
                    ),
                ));
            }
        }

        // V4: Read bucket metadata, V3: use empty maps
        let (bucket_names, bucket_sources, bucket_minimizer_counts) = if version >= 4 {
            reader.read_exact(&mut buf4)?;
            let num_buckets = u32::from_le_bytes(buf4);
            if num_buckets > MAX_NUM_BUCKETS {
                return Err(RypeError::overflow(
                    "number of buckets",
                    MAX_NUM_BUCKETS as usize,
                    num_buckets as usize,
                ));
            }

            let mut bucket_names = HashMap::new();
            let mut bucket_sources = HashMap::new();
            let mut bucket_minimizer_counts = HashMap::new();

            for _ in 0..num_buckets {
                reader.read_exact(&mut buf4)?;
                let bucket_id = u32::from_le_bytes(buf4);

                // Read name
                reader.read_exact(&mut buf8)?;
                let name_len = u64::from_le_bytes(buf8) as usize;
                if name_len > MAX_STRING_LENGTH {
                    return Err(RypeError::overflow(
                        "bucket name length",
                        MAX_STRING_LENGTH,
                        name_len,
                    ));
                }
                let mut name_buf = vec![0u8; name_len];
                reader.read_exact(&mut name_buf)?;
                let name = String::from_utf8(name_buf).map_err(|e| {
                    RypeError::format(path, format!("invalid UTF-8 in bucket name: {}", e))
                })?;
                bucket_names.insert(bucket_id, name);

                // Read minimizer count
                reader.read_exact(&mut buf8)?;
                let minimizer_count = u64::from_le_bytes(buf8) as usize;
                bucket_minimizer_counts.insert(bucket_id, minimizer_count);

                // Read sources
                reader.read_exact(&mut buf8)?;
                let src_count = u64::from_le_bytes(buf8) as usize;
                let mut sources = Vec::with_capacity(src_count);
                for _ in 0..src_count {
                    reader.read_exact(&mut buf8)?;
                    let src_len = u64::from_le_bytes(buf8) as usize;
                    if src_len > MAX_STRING_LENGTH {
                        return Err(RypeError::overflow(
                            "source string length",
                            MAX_STRING_LENGTH,
                            src_len,
                        ));
                    }
                    let mut src_buf = vec![0u8; src_len];
                    reader.read_exact(&mut src_buf)?;
                    let src = String::from_utf8(src_buf).map_err(|e| {
                        RypeError::format(path, format!("invalid UTF-8 in source string: {}", e))
                    })?;
                    sources.push(src);
                }
                bucket_sources.insert(bucket_id, sources);
            }

            (bucket_names, bucket_sources, bucket_minimizer_counts)
        } else {
            // V3 backward compat: empty metadata maps
            (HashMap::new(), HashMap::new(), HashMap::new())
        };

        Ok(ShardManifest {
            k,
            w,
            salt,
            source_hash,
            total_minimizers,
            total_bucket_ids,
            has_overlapping_shards,
            shard_format, // FIX #4: Explicit format from manifest
            shards,
            bucket_names,
            bucket_sources,
            bucket_minimizer_counts,
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

    /// Get the path for a Parquet shard file given a base path and shard ID.
    ///
    /// For Parquet-based indices, shards are stored in an `inverted/` subdirectory
    /// with `.parquet` extension.
    pub fn shard_path_parquet(base: &Path, shard_id: u32) -> PathBuf {
        base.join("inverted")
            .join(format!("shard.{}.parquet", shard_id))
    }

    /// Get the shard path for a specific format.
    pub fn shard_path_for_format(base: &Path, shard_id: u32, format: ShardFormat) -> PathBuf {
        match format {
            ShardFormat::Legacy => Self::shard_path(base, shard_id),
            ShardFormat::Parquet => Self::shard_path_parquet(base, shard_id),
        }
    }

    /// Convert to IndexMetadata for compatibility with existing code.
    ///
    /// Returns None if the manifest was loaded from v3 format (no bucket metadata).
    pub fn to_metadata(&self) -> Option<IndexMetadata> {
        if self.bucket_names.is_empty() {
            // V3 manifest - no metadata available
            None
        } else {
            Some(IndexMetadata {
                k: self.k,
                w: self.w,
                salt: self.salt,
                bucket_names: self.bucket_names.clone(),
                bucket_sources: self.bucket_sources.clone(),
                bucket_minimizer_counts: self.bucket_minimizer_counts.clone(),
            })
        }
    }

    /// Check if this manifest has bucket metadata (v4+).
    pub fn has_bucket_metadata(&self) -> bool {
        !self.bucket_names.is_empty()
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
    shard_format: ShardFormat,
}

impl ShardedInvertedIndex {
    /// Open a sharded inverted index by loading just the manifest.
    ///
    /// FIX #4: Uses the shard_format stored in the manifest instead of
    /// file-existence guessing. Falls back to file checking for v3/v4
    /// manifests that don't have explicit format.
    pub fn open(base_path: &Path) -> Result<Self> {
        let manifest_path = ShardManifest::manifest_path(base_path);
        let manifest = ShardManifest::load(&manifest_path)?;

        // FIX #4: Use format from manifest (v5+), fall back to file checking for v3/v4
        let shard_format = if manifest.shard_format != ShardFormat::Legacy {
            // v5+ manifest has explicit format
            manifest.shard_format
        } else if !manifest.shards.is_empty() {
            // v3/v4 manifest - fall back to file-existence checking
            let parquet_path = ShardManifest::shard_path_parquet(base_path, 0);
            let legacy_path = ShardManifest::shard_path(base_path, 0);

            if parquet_path.exists() {
                ShardFormat::Parquet
            } else if legacy_path.exists() {
                ShardFormat::Legacy
            } else {
                return Err(RypeError::io(
                    &legacy_path,
                    "open shard",
                    std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!(
                            "No shard file found at {} or {}",
                            parquet_path.display(),
                            legacy_path.display()
                        ),
                    ),
                ));
            }
        } else {
            ShardFormat::Legacy // Default for empty manifests
        };

        Ok(ShardedInvertedIndex {
            manifest,
            base_path: base_path.to_path_buf(),
            shard_format,
        })
    }

    /// Open with explicit format (useful for testing or when format is known).
    pub fn open_with_format(base_path: &Path, format: ShardFormat) -> Result<Self> {
        let manifest_path = ShardManifest::manifest_path(base_path);
        let manifest = ShardManifest::load(&manifest_path)?;

        Ok(ShardedInvertedIndex {
            manifest,
            base_path: base_path.to_path_buf(),
            shard_format: format,
        })
    }

    /// Open a Parquet-format inverted index directory.
    ///
    /// This is for indices created with `--parquet` flag where the directory
    /// contains manifest.toml instead of .manifest binary file.
    pub fn open_parquet(base_path: &Path) -> Result<Self> {
        use super::parquet::ParquetManifest;

        // Load Parquet manifest
        let parquet_manifest = ParquetManifest::load(base_path)
            .map_err(|e| RypeError::format(base_path, e.to_string()))?;

        // Load bucket metadata from buckets.parquet
        let (bucket_names, bucket_sources) = super::parquet::read_buckets_parquet(base_path)
            .map_err(|e| RypeError::format(base_path, e.to_string()))?;

        // Convert to ShardManifest format
        let inverted = parquet_manifest
            .inverted
            .as_ref()
            .ok_or_else(|| RypeError::format(base_path, "missing inverted section in manifest"))?;

        let shards: Vec<ShardInfo> = inverted
            .shards
            .iter()
            .map(|s| ShardInfo {
                shard_id: s.shard_id,
                min_start: s.min_minimizer,
                min_end: s.max_minimizer,
                is_last_shard: s.shard_id == inverted.num_shards.saturating_sub(1),
                num_minimizers: s.num_entries as usize,
                num_bucket_ids: s.num_entries as usize, // In Parquet, each entry is a (minimizer, bucket_id) pair
            })
            .collect();

        let manifest = ShardManifest {
            k: parquet_manifest.k,
            w: parquet_manifest.w,
            salt: parquet_manifest.salt,
            source_hash: parquet_manifest.source_hash,
            total_minimizers: inverted.total_entries as usize,
            total_bucket_ids: inverted.total_entries as usize,
            has_overlapping_shards: inverted.has_overlapping_shards,
            shard_format: ShardFormat::Parquet, // FIX #4: Explicit format
            shards,
            bucket_names,
            bucket_sources,
            bucket_minimizer_counts: HashMap::new(), // Not stored in Parquet format
        };

        Ok(ShardedInvertedIndex {
            manifest,
            base_path: base_path.to_path_buf(),
            shard_format: ShardFormat::Parquet,
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

    /// Returns the shard format (Legacy or Parquet).
    pub fn shard_format(&self) -> ShardFormat {
        self.shard_format
    }

    /// Get the path for a specific shard, using the detected format.
    pub fn shard_path(&self, shard_id: u32) -> PathBuf {
        ShardManifest::shard_path_for_format(&self.base_path, shard_id, self.shard_format)
    }

    /// Load a specific shard by ID.
    ///
    /// Uses the detected format to load either legacy RYXS or Parquet shards.
    pub fn load_shard(&self, shard_id: u32) -> Result<InvertedIndex> {
        let path = self.shard_path(shard_id);
        match self.shard_format {
            ShardFormat::Legacy => InvertedIndex::load_shard(&path)
                .map_err(|e| RypeError::format(&path, e.to_string())),
            ShardFormat::Parquet => InvertedIndex::load_shard_parquet_with_params(
                &path,
                self.manifest.k,
                self.manifest.w,
                self.manifest.salt,
                self.manifest.source_hash,
            )
            .map_err(|e| RypeError::format(&path, e.to_string())),
        }
    }

    /// Load a shard, filtering to only include data relevant to query minimizers.
    ///
    /// For Parquet shards, this uses merge-scan to identify which row groups contain
    /// query minimizers, then loads only those row groups and filters rows. This can
    /// skip 90%+ of data for sparse queries.
    ///
    /// For legacy shards, falls back to full loading (no row group filtering available).
    ///
    /// # Arguments
    /// * `shard_id` - The shard to load
    /// * `query_minimizers` - Sorted slice of query minimizers to match against
    ///
    /// # Returns
    /// An InvertedIndex containing only minimizers present in the query set.
    ///
    /// # Arguments
    /// * `shard_id` - The shard to load
    /// * `query_minimizers` - Sorted slice of query minimizers
    /// * `options` - Parquet read options (None = default behavior without bloom filters)
    pub fn load_shard_for_query(
        &self,
        shard_id: u32,
        query_minimizers: &[u64],
        options: Option<&super::parquet::ParquetReadOptions>,
    ) -> Result<InvertedIndex> {
        let path = self.shard_path(shard_id);
        match self.shard_format {
            ShardFormat::Legacy => {
                // Legacy format doesn't support row group filtering or bloom filters
                // Load full shard (caller will filter during merge-join)
                InvertedIndex::load_shard(&path)
                    .map_err(|e| RypeError::format(&path, e.to_string()))
            }
            ShardFormat::Parquet => InvertedIndex::load_shard_parquet_for_query(
                &path,
                self.manifest.k,
                self.manifest.w,
                self.manifest.salt,
                self.manifest.source_hash,
                query_minimizers,
                options,
            )
            .map_err(|e| RypeError::format(&path, e.to_string())),
        }
    }

    /// Check if this index uses Parquet format (supports filtered loading).
    pub fn is_parquet(&self) -> bool {
        matches!(self.shard_format, ShardFormat::Parquet)
    }

    /// Validate against Index metadata.
    pub fn validate_against_metadata(&self, metadata: &IndexMetadata) -> Result<()> {
        if self.manifest.k != metadata.k {
            return Err(RypeError::validation(format!(
                "K mismatch: sharded index has K={}, metadata has K={}",
                self.manifest.k, metadata.k
            )));
        }
        if self.manifest.w != metadata.w {
            return Err(RypeError::validation(format!(
                "W mismatch: sharded index has W={}, metadata has W={}",
                self.manifest.w, metadata.w
            )));
        }
        if self.manifest.salt != metadata.salt {
            return Err(RypeError::validation(format!(
                "Salt mismatch: sharded index has salt={:#x}, metadata has salt={:#x}",
                self.manifest.salt, metadata.salt
            )));
        }

        let expected_hash = InvertedIndex::compute_metadata_hash(metadata);
        if self.manifest.source_hash != expected_hash {
            return Err(RypeError::validation(
                "Source hash mismatch: sharded index is stale or was built from different source",
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indices::main::Index;
    use tempfile::NamedTempFile;

    #[test]
    fn test_shard_manifest_save_load_roundtrip() -> anyhow::Result<()> {
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
            shard_format: ShardFormat::Legacy,
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
            bucket_names: HashMap::new(),
            bucket_sources: HashMap::new(),
            bucket_minimizer_counts: HashMap::new(),
        };

        manifest.save(&path)?;
        let loaded = ShardManifest::load(&path)?;

        assert_eq!(loaded.k, 64);
        assert_eq!(loaded.w, 50);
        assert_eq!(loaded.salt, 0xDEADBEEF);
        assert_eq!(loaded.source_hash, 0x12345678);
        assert_eq!(loaded.total_minimizers, 1000);
        assert_eq!(loaded.total_bucket_ids, 5000);
        assert_eq!(loaded.shard_format, ShardFormat::Legacy);
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

        // Empty metadata maps should round-trip
        assert!(loaded.bucket_names.is_empty());
        assert!(loaded.bucket_sources.is_empty());
        assert!(loaded.bucket_minimizer_counts.is_empty());

        Ok(())
    }

    #[test]
    fn test_shard_manifest_v4_roundtrip_with_metadata() -> anyhow::Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();

        let mut bucket_names = HashMap::new();
        bucket_names.insert(1, "BucketA".to_string());
        bucket_names.insert(2, "BucketB".to_string());

        let mut bucket_sources = HashMap::new();
        bucket_sources.insert(
            1,
            vec!["file1.fa||seq1".to_string(), "file1.fa||seq2".to_string()],
        );
        bucket_sources.insert(2, vec!["file2.fa||seqX".to_string()]);

        let mut bucket_minimizer_counts = HashMap::new();
        bucket_minimizer_counts.insert(1, 1000);
        bucket_minimizer_counts.insert(2, 2000);

        let manifest = ShardManifest {
            k: 64,
            w: 50,
            salt: 0xDEADBEEF,
            source_hash: 0x12345678,
            total_minimizers: 3000,
            total_bucket_ids: 3000,
            has_overlapping_shards: true,
            shard_format: ShardFormat::Parquet, // Test Parquet format
            shards: vec![ShardInfo {
                shard_id: 0,
                min_start: 0,
                min_end: 0,
                is_last_shard: true,
                num_minimizers: 3000,
                num_bucket_ids: 3000,
            }],
            bucket_names: bucket_names.clone(),
            bucket_sources: bucket_sources.clone(),
            bucket_minimizer_counts: bucket_minimizer_counts.clone(),
        };

        manifest.save(&path)?;
        let loaded = ShardManifest::load(&path)?;

        assert_eq!(loaded.bucket_names, bucket_names);
        assert_eq!(loaded.bucket_sources, bucket_sources);
        assert_eq!(loaded.bucket_minimizer_counts, bucket_minimizer_counts);
        assert_eq!(loaded.shard_format, ShardFormat::Parquet);

        // Verify to_metadata works
        let metadata = loaded.to_metadata().expect("should have metadata");
        assert_eq!(metadata.k, 64);
        assert_eq!(metadata.w, 50);
        assert_eq!(metadata.salt, 0xDEADBEEF);
        assert_eq!(metadata.bucket_names, bucket_names);
        assert_eq!(metadata.bucket_sources, bucket_sources);
        assert_eq!(metadata.bucket_minimizer_counts, bucket_minimizer_counts);

        assert!(loaded.has_bucket_metadata());

        Ok(())
    }

    #[test]
    fn test_shard_manifest_v3_backward_compat() {
        // Manually construct a v3 manifest file (without bucket metadata)
        let file = NamedTempFile::new().unwrap();
        let path = file.path();

        // Write v3 format data
        let mut data = Vec::new();
        data.extend_from_slice(MANIFEST_MAGIC); // Magic
        data.extend_from_slice(&3u32.to_le_bytes()); // Version 3
        data.extend_from_slice(&64u64.to_le_bytes()); // k
        data.extend_from_slice(&50u64.to_le_bytes()); // w
        data.extend_from_slice(&0xDEADBEEFu64.to_le_bytes()); // salt
        data.extend_from_slice(&0x12345678u64.to_le_bytes()); // source_hash
        data.extend_from_slice(&100u64.to_le_bytes()); // total_minimizers
        data.extend_from_slice(&100u64.to_le_bytes()); // total_bucket_ids
        data.push(1u8); // has_overlapping_shards = true
        data.extend_from_slice(&1u32.to_le_bytes()); // num_shards = 1

        // One shard
        data.extend_from_slice(&0u32.to_le_bytes()); // shard_id
        data.extend_from_slice(&0u64.to_le_bytes()); // min_start
        data.extend_from_slice(&0u64.to_le_bytes()); // min_end
        data.push(1u8); // is_last_shard = true
        data.extend_from_slice(&100u64.to_le_bytes()); // num_minimizers
        data.extend_from_slice(&100u64.to_le_bytes()); // num_bucket_ids

        std::fs::write(path, data).unwrap();

        // Load and verify backward compat
        let loaded = ShardManifest::load(path).unwrap();
        assert_eq!(loaded.k, 64);
        assert_eq!(loaded.w, 50);
        assert_eq!(loaded.shards.len(), 1);

        // V3 manifests have empty metadata maps
        assert!(loaded.bucket_names.is_empty());
        assert!(loaded.bucket_sources.is_empty());
        assert!(loaded.bucket_minimizer_counts.is_empty());
        assert!(!loaded.has_bucket_metadata());
        assert!(loaded.to_metadata().is_none());
    }

    #[test]
    fn test_shard_manifest_overlapping_roundtrip() -> anyhow::Result<()> {
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
            shard_format: ShardFormat::Legacy,
            shards: vec![
                ShardInfo {
                    shard_id: 0,
                    min_start: 100, // Overlaps with shard 1
                    min_end: 0,
                    is_last_shard: false,
                    num_minimizers: 50,
                    num_bucket_ids: 50,
                },
                ShardInfo {
                    shard_id: 1,
                    min_start: 100, // Same min_start as shard 0 (overlapping)
                    min_end: 0,
                    is_last_shard: false,
                    num_minimizers: 50,
                    num_bucket_ids: 50,
                },
                ShardInfo {
                    shard_id: 2,
                    min_start: 50, // Lower than previous shards (not sorted)
                    min_end: 0,
                    is_last_shard: true,
                    num_minimizers: 50,
                    num_bucket_ids: 50,
                },
            ],
            bucket_names: HashMap::new(),
            bucket_sources: HashMap::new(),
            bucket_minimizer_counts: HashMap::new(),
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
        assert_eq!(loaded.shards[2].min_start, 50); // Lower than previous

        Ok(())
    }

    #[test]
    fn test_shard_manifest_invalid_magic() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path();
        std::fs::write(path, b"NOTM").unwrap();

        let result = ShardManifest::load(path);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid shard manifest format"));
    }

    #[test]
    fn test_shard_manifest_invalid_version() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path();

        let mut data = Vec::new();
        data.extend_from_slice(MANIFEST_MAGIC);
        data.extend_from_slice(&99u32.to_le_bytes());
        std::fs::write(path, data).unwrap();

        let result = ShardManifest::load(path);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Unsupported inverted index manifest version"));
    }

    #[test]
    fn test_shard_manifest_v2_rejected_with_helpful_error() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path();

        // Write a v2 manifest (magic + version 2)
        let mut data = Vec::new();
        data.extend_from_slice(MANIFEST_MAGIC);
        data.extend_from_slice(&2u32.to_le_bytes());
        std::fs::write(path, data).unwrap();

        let result = ShardManifest::load(path);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Unsupported inverted index manifest version"),
            "Error should mention version: {}",
            err
        );
        assert!(
            err.contains("rype index invert"),
            "Error should suggest regenerating: {}",
            err
        );
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
    fn test_sharded_inverted_index_open() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let base_path = dir.path().join("test.ryxdi");

        let mut index = Index::new(64, 50, 0x1234).unwrap();
        index.buckets.insert(1, vec![100, 200, 300, 400]);
        index.bucket_names.insert(1, "A".into());
        let inverted = InvertedIndex::build_from_index(&index);

        // Save as single shard
        let shard_path = ShardManifest::shard_path(&base_path, 0);
        let shard_info = inverted.save_shard(&shard_path, 0, 0, inverted.num_minimizers(), true)?;

        let manifest = ShardManifest {
            k: inverted.k,
            w: inverted.w,
            salt: inverted.salt,
            source_hash: 0,
            total_minimizers: inverted.num_minimizers(),
            total_bucket_ids: inverted.num_bucket_entries(),
            has_overlapping_shards: true,
            shard_format: ShardFormat::Legacy,
            shards: vec![shard_info],
            bucket_names: HashMap::new(),
            bucket_sources: HashMap::new(),
            bucket_minimizer_counts: HashMap::new(),
        };
        let manifest_path = ShardManifest::manifest_path(&base_path);
        manifest.save(&manifest_path)?;

        let sharded = ShardedInvertedIndex::open(&base_path)?;

        assert_eq!(sharded.k(), 64);
        assert_eq!(sharded.w(), 50);
        assert_eq!(sharded.salt(), 0x1234);
        assert_eq!(sharded.num_shards(), 1);
        assert_eq!(sharded.shard_format(), ShardFormat::Legacy);

        Ok(())
    }
}
