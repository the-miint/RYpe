//! Sharded inverted index structures.
//!
//! All inverted indices use the Parquet-based sharded format. This format stores
//! indices as directories containing a manifest.toml and Parquet shard files.
//!
//! For small indices, the overhead is minimal: an extra file open and a small manifest
//! read. For large indices, sharding enables memory-efficient classification by loading
//! one shard at a time.

use crate::error::{Result, RypeError};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::inverted::InvertedIndex;
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
/// This is an in-memory representation created from the TOML manifest file
/// stored in the index directory.
///
/// # Shard Partitioning
///
/// **Key consequence**: Since different buckets can share the same minimizers, the same
/// minimizer value can appear in multiple inverted shards. This is why `has_overlapping_shards`
/// is always `true` for bucket-partitioned shards.
///
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
    pub shards: Vec<ShardInfo>,
    /// Bucket names. Maps bucket_id to human-readable name.
    pub bucket_names: HashMap<u32, String>,
    /// Bucket sources. Maps bucket_id to list of source sequence names.
    pub bucket_sources: HashMap<u32, Vec<String>>,
    /// Bucket minimizer counts. Maps bucket_id to number of minimizers.
    pub bucket_minimizer_counts: HashMap<u32, usize>,
}

impl ShardManifest {
    /// Get the path for a Parquet shard file given a base path and shard ID.
    ///
    /// Shards are stored in an `inverted/` subdirectory with `.parquet` extension.
    pub fn shard_path_parquet(base: &Path, shard_id: u32) -> PathBuf {
        base.join("inverted")
            .join(format!("shard.{}.parquet", shard_id))
    }

    /// Convert to IndexMetadata for compatibility with existing code.
    pub fn to_metadata(&self) -> IndexMetadata {
        IndexMetadata {
            k: self.k,
            w: self.w,
            salt: self.salt,
            bucket_names: self.bucket_names.clone(),
            bucket_sources: self.bucket_sources.clone(),
            bucket_minimizer_counts: self.bucket_minimizer_counts.clone(),
        }
    }

    /// Check if this manifest has bucket metadata.
    pub fn has_bucket_metadata(&self) -> bool {
        !self.bucket_names.is_empty()
    }
}

/// Handle for a sharded inverted index.
///
/// This struct holds a manifest describing the shards. Shards are loaded
/// on-demand during classification via `classify_batch_sharded_sequential`.
///
/// Row group metadata (min/max ranges) is preloaded during open() to avoid
/// file I/O during classification hot path.
#[derive(Debug, Clone)]
pub struct ShardedInvertedIndex {
    manifest: ShardManifest,
    base_path: PathBuf,
    /// Cached row group ranges for shards: Vec indexed by shard position in manifest.
    /// Each inner Vec contains RowGroupRangeInfo with rg_idx, min, max, and uncompressed_size.
    rg_ranges_cache: Vec<Vec<super::inverted::RowGroupRangeInfo>>,
}

impl ShardedInvertedIndex {
    /// Open a Parquet-format inverted index directory.
    ///
    /// This loads the manifest.toml and bucket metadata from the index directory.
    pub fn open(base_path: &Path) -> Result<Self> {
        use super::parquet::ParquetManifest;

        // Check for old format and give helpful error
        if base_path.extension().is_some_and(|ext| ext == "ryidx") {
            return Err(RypeError::format(
                base_path,
                "This appears to be an old .ryidx file. Rype now only supports \
                 Parquet indices (.ryxdi directories). Please rebuild your index \
                 using: rype index create -o output.ryxdi -r your_refs.fasta",
            ));
        }

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
            shards,
            bucket_names,
            bucket_sources,
            bucket_minimizer_counts: HashMap::new(), // Not stored in Parquet format
        };

        // Preload row group ranges for Parquet shards
        let rg_ranges_cache = Self::load_rg_ranges_for_shards(base_path, &manifest.shards)?;

        Ok(ShardedInvertedIndex {
            manifest,
            base_path: base_path.to_path_buf(),
            rg_ranges_cache,
        })
    }

    /// Load row group ranges for all shards.
    ///
    /// Returns a Vec indexed by shard position (not shard_id).
    fn load_rg_ranges_for_shards(
        base_path: &Path,
        shards: &[ShardInfo],
    ) -> Result<Vec<Vec<super::inverted::RowGroupRangeInfo>>> {
        use super::inverted::get_row_group_ranges;

        let mut cache = Vec::with_capacity(shards.len());

        for shard_info in shards {
            let shard_path = ShardManifest::shard_path_parquet(base_path, shard_info.shard_id);
            let ranges = get_row_group_ranges(&shard_path)?;
            cache.push(ranges);
        }

        Ok(cache)
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

    /// Get the path for a specific shard.
    pub fn shard_path(&self, shard_id: u32) -> PathBuf {
        ShardManifest::shard_path_parquet(&self.base_path, shard_id)
    }

    /// Get cached row group ranges for a shard by its position in the manifest.
    ///
    /// Returns None if shard_pos is out of bounds.
    /// Returns the preloaded RowGroupRangeInfo entries.
    pub fn rg_ranges(&self, shard_pos: usize) -> Option<&[super::inverted::RowGroupRangeInfo]> {
        self.rg_ranges_cache.get(shard_pos).map(|v| v.as_slice())
    }

    /// Check if row group ranges are cached (should always be true for valid indices).
    pub fn has_rg_cache(&self) -> bool {
        !self.rg_ranges_cache.is_empty()
    }

    /// Calculate total uncompressed size of all row groups across all shards.
    ///
    /// This is used for memory estimation to decide whether to preload data.
    pub fn total_uncompressed_size(&self) -> usize {
        self.rg_ranges_cache
            .iter()
            .flat_map(|rgs| rgs.iter())
            .map(|info| info.uncompressed_size)
            .sum()
    }

    /// Advise the kernel to prefetch Parquet shard files into page cache.
    ///
    /// Uses mmap + madvise(MADV_WILLNEED) to tell the kernel to asynchronously
    /// read the shard files into memory. This is non-blocking - the kernel
    /// handles prefetching in the background while other work continues.
    ///
    /// # Arguments
    /// * `max_bytes` - Maximum bytes to prefetch (prefetches largest shards first up to budget)
    ///
    /// # Returns
    /// Number of bytes advised for prefetching, or 0 if prefetching is not available.
    #[cfg(unix)]
    pub fn advise_prefetch(&self, max_bytes: Option<usize>) -> usize {
        use std::os::unix::io::AsRawFd;

        if self.rg_ranges_cache.is_empty() {
            return 0;
        }

        let budget = max_bytes.unwrap_or(usize::MAX);
        let mut total_advised = 0usize;

        for shard_info in &self.manifest.shards {
            let shard_path =
                ShardManifest::shard_path_parquet(&self.base_path, shard_info.shard_id);

            // Get file size
            let file_size = match std::fs::metadata(&shard_path) {
                Ok(meta) => meta.len() as usize,
                Err(_) => continue,
            };

            // Check budget
            if total_advised + file_size > budget {
                log::debug!(
                    "Prefetch budget reached at {} bytes, skipping remaining shards",
                    total_advised
                );
                break;
            }

            // Open and mmap the file
            let file = match std::fs::File::open(&shard_path) {
                Ok(f) => f,
                Err(_) => continue,
            };

            // Use madvise to advise the kernel to prefetch
            // SAFETY: We're just advising the kernel, not dereferencing the memory
            unsafe {
                let ptr = libc::mmap(
                    std::ptr::null_mut(),
                    file_size,
                    libc::PROT_READ,
                    libc::MAP_PRIVATE,
                    file.as_raw_fd(),
                    0,
                );

                if ptr != libc::MAP_FAILED {
                    // Tell kernel to prefetch this region
                    libc::madvise(ptr, file_size, libc::MADV_WILLNEED);

                    // Immediately unmap - the kernel will still do the prefetch
                    libc::munmap(ptr, file_size);

                    total_advised += file_size;
                }
            }
        }

        if total_advised > 0 {
            log::debug!(
                "Advised kernel to prefetch {} bytes across {} shards",
                total_advised,
                self.manifest.shards.len()
            );
        }

        total_advised
    }

    /// Non-unix stub - prefetching not available.
    #[cfg(not(unix))]
    pub fn advise_prefetch(&self, _max_bytes: Option<usize>) -> usize {
        0
    }

    /// Load a specific shard by ID.
    pub fn load_shard(&self, shard_id: u32) -> Result<InvertedIndex> {
        let path = self.shard_path(shard_id);
        InvertedIndex::load_shard_parquet_with_params(
            &path,
            self.manifest.k,
            self.manifest.w,
            self.manifest.salt,
            self.manifest.source_hash,
        )
        .map_err(|e| RypeError::format(&path, e.to_string()))
    }

    /// Load a shard, filtering to only include data relevant to query minimizers.
    ///
    /// This uses merge-scan to identify which row groups contain query minimizers,
    /// then loads only those row groups and filters rows. This can skip 90%+ of
    /// data for sparse queries.
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
        InvertedIndex::load_shard_parquet_for_query(
            &path,
            self.manifest.k,
            self.manifest.w,
            self.manifest.salt,
            self.manifest.source_hash,
            query_minimizers,
            options,
        )
        .map_err(|e| RypeError::format(&path, e.to_string()))
    }

    /// Check if this index uses Parquet format (always true).
    pub fn is_parquet(&self) -> bool {
        true
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
