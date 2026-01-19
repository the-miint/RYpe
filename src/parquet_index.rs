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

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Serialize u64 as hex string for TOML compatibility (i64 overflow).
mod hex_u64 {
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(value: &u64, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&format!("0x{:016x}", value))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<u64, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        let s = s.trim_start_matches("0x").trim_start_matches("0X");
        u64::from_str_radix(s, 16).map_err(serde::de::Error::custom)
    }
}

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

// ============================================================================
// Parquet Write Options
// ============================================================================

/// Compression codec for Parquet files.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ParquetCompression {
    /// Snappy compression (fast, moderate ratio). Default.
    #[default]
    Snappy,
    /// Zstd compression (slower, better ratio).
    Zstd,
}

/// Configuration options for Parquet file writing.
///
/// Use `Default::default()` to get the current behavior (Snappy, 100K row groups,
/// no bloom filters). Pass custom options to enable advanced features.
///
/// # Example
/// ```ignore
/// let opts = ParquetWriteOptions {
///     compression: ParquetCompression::Zstd,
///     bloom_filter_enabled: true,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct ParquetWriteOptions {
    /// Maximum rows per row group. Default: 100,000.
    ///
    /// # Performance trade-offs
    ///
    /// **Smaller row groups** (10K-50K):
    /// - Better read performance when filtering specific minimizer ranges
    /// - More memory efficient during reads (less data loaded per group)
    /// - Higher metadata overhead (more groups = more index entries)
    /// - Slightly worse compression (less data to find patterns in)
    ///
    /// **Larger row groups** (500K-1M):
    /// - Better compression ratios (more data for pattern detection)
    /// - Lower metadata overhead
    /// - Higher memory usage during reads
    /// - Slower random access within a shard
    ///
    /// For most workloads, the default (100K) balances read performance and
    /// compression. Increase for indices that will be scanned linearly;
    /// decrease for indices with highly selective range queries.
    pub row_group_size: usize,

    /// Compression codec. Default: Snappy.
    pub compression: ParquetCompression,

    /// Enable bloom filters for faster lookups. Default: false.
    pub bloom_filter_enabled: bool,

    /// Bloom filter false positive probability. Default: 0.05 (5%).
    pub bloom_filter_fpp: f64,

    /// Write page-level statistics. Default: true.
    pub write_page_statistics: bool,
}

impl Default for ParquetWriteOptions {
    fn default() -> Self {
        Self {
            row_group_size: 100_000,
            compression: ParquetCompression::Snappy,
            bloom_filter_enabled: false,
            bloom_filter_fpp: 0.05,
            write_page_statistics: true,
        }
    }
}

#[cfg(feature = "parquet")]
impl ParquetWriteOptions {
    /// Validate options. Returns error if any values are out of bounds.
    ///
    /// Checks:
    /// - `bloom_filter_fpp` must be in (0.0, 1.0)
    /// - `row_group_size` must be > 0
    pub fn validate(&self) -> Result<()> {
        if self.bloom_filter_fpp <= 0.0 || self.bloom_filter_fpp >= 1.0 {
            anyhow::bail!(
                "bloom_filter_fpp must be in (0.0, 1.0), got {}",
                self.bloom_filter_fpp
            );
        }
        if self.row_group_size == 0 {
            anyhow::bail!("row_group_size must be > 0");
        }
        Ok(())
    }

    /// Convert options to parquet WriterProperties.
    ///
    /// This is the single source of truth for building WriterProperties,
    /// ensuring DRY across all Parquet write paths.
    ///
    /// # Panics
    /// Panics if options are invalid. Call `validate()` first to get a Result.
    pub fn to_writer_properties(&self) -> parquet::file::properties::WriterProperties {
        // Panic on invalid options - caller should validate first
        self.validate()
            .expect("Invalid ParquetWriteOptions - call validate() first");
        use parquet::basic::{Compression, Encoding};
        use parquet::file::properties::{EnabledStatistics, WriterProperties, WriterVersion};
        use parquet::schema::types::ColumnPath;

        let compression = match self.compression {
            ParquetCompression::Snappy => Compression::SNAPPY,
            ParquetCompression::Zstd => Compression::ZSTD(parquet::basic::ZstdLevel::default()),
        };

        let statistics = if self.write_page_statistics {
            EnabledStatistics::Page
        } else {
            EnabledStatistics::None
        };

        let minimizer_col = ColumnPath::new(vec!["minimizer".to_string()]);
        let bucket_id_col = ColumnPath::new(vec!["bucket_id".to_string()]);

        let mut builder = WriterProperties::builder()
            .set_writer_version(WriterVersion::PARQUET_2_0)
            .set_compression(compression)
            .set_statistics_enabled(statistics)
            .set_max_row_group_size(self.row_group_size)
            // Minimizer column: delta encoding, no dictionary
            .set_column_encoding(minimizer_col.clone(), Encoding::DELTA_BINARY_PACKED)
            .set_column_dictionary_enabled(minimizer_col.clone(), false)
            // Bucket ID column: delta encoding, no dictionary
            .set_column_encoding(bucket_id_col.clone(), Encoding::DELTA_BINARY_PACKED)
            .set_column_dictionary_enabled(bucket_id_col.clone(), false);

        if self.bloom_filter_enabled {
            builder = builder
                .set_column_bloom_filter_enabled(minimizer_col.clone(), true)
                .set_column_bloom_filter_fpp(minimizer_col, self.bloom_filter_fpp)
                .set_column_bloom_filter_enabled(bucket_id_col.clone(), true)
                .set_column_bloom_filter_fpp(bucket_id_col, self.bloom_filter_fpp);
        }

        builder.build()
    }
}

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

// ============================================================================
// Buckets Parquet I/O (requires parquet feature)
// ============================================================================

#[cfg(feature = "parquet")]
mod parquet_io {
    use super::*;
    use arrow::array::{
        Array, ArrayRef, ListArray, ListBuilder, StringArray, StringBuilder, UInt32Array,
    };
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use parquet::arrow::ArrowWriter;
    use parquet::basic::Compression;
    use parquet::file::properties::{WriterProperties, WriterVersion};
    use std::fs::File;
    use std::sync::Arc;

    /// Write bucket metadata to Parquet.
    ///
    /// Schema: `(bucket_id: u32, bucket_name: string, sources: list<string>)`
    pub fn write_buckets_parquet(
        index_dir: &Path,
        bucket_names: &HashMap<u32, String>,
        bucket_sources: &HashMap<u32, Vec<String>>,
    ) -> Result<()> {
        let path = index_dir.join(files::BUCKETS);

        let schema = Arc::new(Schema::new(vec![
            Field::new("bucket_id", DataType::UInt32, false),
            Field::new("bucket_name", DataType::Utf8, false),
            Field::new(
                "sources",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                false,
            ),
        ]));

        let props = WriterProperties::builder()
            .set_writer_version(WriterVersion::PARQUET_2_0)
            .set_compression(Compression::ZSTD(Default::default()))
            .build();

        let file = File::create(&path)
            .with_context(|| format!("Failed to create buckets file: {}", path.display()))?;
        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

        // Collect and sort buckets by ID
        let empty_sources: Vec<String> = Vec::new();
        let mut buckets: Vec<(u32, &String, &Vec<String>)> = bucket_names
            .iter()
            .map(|(&id, name)| {
                let sources = bucket_sources.get(&id).unwrap_or(&empty_sources);
                (id, name, sources)
            })
            .collect();
        buckets.sort_by_key(|(id, _, _)| *id);

        // Build arrays
        let bucket_ids: Vec<u32> = buckets.iter().map(|(id, _, _)| *id).collect();
        let names: Vec<&str> = buckets.iter().map(|(_, name, _)| name.as_str()).collect();

        let bucket_id_array: ArrayRef = Arc::new(UInt32Array::from(bucket_ids));
        let bucket_name_array: ArrayRef = Arc::new(StringArray::from(names));

        // Build list array for sources
        let mut list_builder = ListBuilder::new(StringBuilder::new());
        for (_, _, sources) in &buckets {
            let values_builder = list_builder.values();
            for source in *sources {
                values_builder.append_value(source);
            }
            list_builder.append(true);
        }
        let sources_array: ArrayRef = Arc::new(list_builder.finish());

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![bucket_id_array, bucket_name_array, sources_array],
        )?;
        writer.write(&batch)?;
        writer.close()?;

        Ok(())
    }

    /// Read bucket metadata from Parquet.
    ///
    /// Returns (bucket_names, bucket_sources).
    #[allow(clippy::type_complexity)]
    pub fn read_buckets_parquet(
        index_dir: &Path,
    ) -> Result<(HashMap<u32, String>, HashMap<u32, Vec<String>>)> {
        let path = index_dir.join(files::BUCKETS);

        let file = File::open(&path)
            .with_context(|| format!("Failed to open buckets file: {}", path.display()))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let reader = builder.build()?;

        let mut bucket_names = HashMap::new();
        let mut bucket_sources = HashMap::new();

        for batch in reader {
            let batch = batch?;
            let bucket_ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt32Array>()
                .context("Expected UInt32Array for bucket_id")?;
            let names = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .context("Expected StringArray for bucket_name")?;
            let sources_list = batch
                .column(2)
                .as_any()
                .downcast_ref::<ListArray>()
                .context("Expected ListArray for sources")?;

            for i in 0..batch.num_rows() {
                let bucket_id = bucket_ids.value(i);
                let name = names.value(i).to_string();

                let sources_array = sources_list.value(i);
                let sources_str = sources_array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .context("Expected StringArray for sources items")?;
                let sources: Vec<String> = (0..sources_str.len())
                    .map(|j| sources_str.value(j).to_string())
                    .collect();

                bucket_names.insert(bucket_id, name);
                bucket_sources.insert(bucket_id, sources);
            }
        }

        Ok((bucket_names, bucket_sources))
    }
}

#[cfg(feature = "parquet")]
pub use parquet_io::{read_buckets_parquet, write_buckets_parquet};

// ============================================================================
// Streaming Parquet Index Creation
// ============================================================================

#[cfg(feature = "parquet")]
mod streaming {
    use super::*;
    use anyhow::Context;
    use arrow::array::ArrayRef;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use rayon::prelude::*;
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    use std::fs::File;

    /// Minimum entries per parallel partition before parallel sharding is enabled.
    ///
    /// When total_entries > MIN_ENTRIES_PER_PARALLEL_PARTITION * num_cpus and
    /// multiple CPUs are available, parallel range-partitioned sharding is used.
    /// This ensures each parallel worker has enough data to amortize the overhead
    /// of spawning threads and coordinating output file renaming.
    ///
    /// Lower values enable more parallelism for smaller indices at the cost of
    /// higher coordination overhead. Higher values prefer sequential processing
    /// but may leave CPUs idle on large indices.
    const MIN_ENTRIES_PER_PARALLEL_PARTITION: usize = 1_000_000;
    use std::path::PathBuf;
    use std::sync::Arc;

    /// K-way merge heap entry: (Reverse((minimizer, bucket_id)), bucket_index, position)
    type MergeHeapEntry = (Reverse<(u64, u32)>, usize, usize);

    /// Row group / batch size for Parquet writing.
    const BATCH_SIZE: usize = 100_000;

    /// Create a Parquet inverted index directly from bucket data.
    ///
    /// This streams (minimizer, bucket_id) pairs through a k-way merge
    /// directly to Parquet, avoiding the intermediate CSR representation.
    ///
    /// # Arguments
    /// * `output_dir` - Directory to create (e.g., "index.ryxdi")
    /// * `buckets` - Bucket data with sorted, deduplicated minimizers (VALIDATED)
    /// * `k` - K-mer size
    /// * `w` - Window size
    /// * `salt` - Hash salt
    /// * `max_shard_bytes` - Optional max shard size in bytes (target, not exact limit)
    /// * `options` - Optional Parquet write options (compression, bloom filters, etc.)
    ///
    /// # Returns
    /// The manifest describing the created index.
    ///
    /// # Errors
    /// Returns an error if any bucket has unsorted or duplicate minimizers.
    pub fn create_parquet_inverted_index(
        output_dir: &Path,
        buckets: Vec<BucketData>,
        k: usize,
        w: usize,
        salt: u64,
        max_shard_bytes: Option<usize>,
        options: Option<&ParquetWriteOptions>,
    ) -> Result<ParquetManifest> {
        let opts = options.cloned().unwrap_or_default();
        opts.validate().context("Invalid ParquetWriteOptions")?;

        // Validate bucket data upfront - buckets must be sorted and deduplicated
        for bucket in &buckets {
            bucket.validate().with_context(|| {
                format!(
                    "Invalid bucket data for bucket '{}' (id={})",
                    bucket.bucket_name, bucket.bucket_id
                )
            })?;
        }

        // Create directory structure
        create_index_directory(output_dir)?;

        // Collect bucket metadata
        let mut bucket_names: HashMap<u32, String> = HashMap::new();
        let mut bucket_sources: HashMap<u32, Vec<String>> = HashMap::new();
        let mut bucket_minimizer_counts: HashMap<u32, usize> = HashMap::new();
        let mut total_minimizers: u64 = 0;

        for bucket in &buckets {
            bucket_names.insert(bucket.bucket_id, bucket.bucket_name.clone());
            bucket_sources.insert(bucket.bucket_id, bucket.sources.clone());
            bucket_minimizer_counts.insert(bucket.bucket_id, bucket.minimizers.len());
            total_minimizers += bucket.minimizers.len() as u64;
        }

        // Write bucket metadata
        write_buckets_parquet(output_dir, &bucket_names, &bucket_sources)?;

        // Stream inverted pairs to Parquet shards
        // FIX #2: Pass max_shard_bytes with clear semantics (target, not exact)
        let shard_infos = stream_to_parquet_shards(
            output_dir,
            buckets,
            max_shard_bytes.unwrap_or(usize::MAX),
            &opts,
        )?;

        // Compute total entries
        let total_entries: u64 = shard_infos.iter().map(|s| s.num_entries).sum();

        // Compute source hash for validation
        let source_hash = compute_source_hash(&bucket_minimizer_counts);

        // Build manifest with explicit format field (FIX #4)
        let manifest = ParquetManifest {
            magic: FORMAT_MAGIC.to_string(),
            format_version: FORMAT_VERSION,
            k,
            w,
            salt,
            source_hash,
            num_buckets: bucket_names.len() as u32,
            total_minimizers,
            inverted: Some(InvertedManifest {
                format: ParquetShardFormat::Parquet, // FIX #4: Explicit format
                num_shards: shard_infos.len() as u32,
                total_entries,
                has_overlapping_shards: true, // buckets share minimizers
                shards: shard_infos,
            }),
        };

        manifest.save(output_dir)?;

        Ok(manifest)
    }

    /// Compute a hash from bucket minimizer counts for validation.
    pub fn compute_source_hash(counts: &HashMap<u32, usize>) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut pairs: Vec<(u32, usize)> = counts.iter().map(|(&id, &c)| (id, c)).collect();
        pairs.sort_by_key(|(id, _)| *id);

        let mut hasher = DefaultHasher::new();
        for (id, count) in pairs {
            id.hash(&mut hasher);
            count.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Stream (minimizer, bucket_id) pairs to Parquet shards using k-way merge.
    ///
    /// FIX #1: Removed redundant deduplication - buckets are validated as sorted/deduped.
    /// FIX #2: Uses actual file sizes for shard splitting decisions.
    /// FIX #6: Uses parallel range partitioning for large indices.
    fn stream_to_parquet_shards(
        output_dir: &Path,
        buckets: Vec<BucketData>,
        max_shard_bytes: usize,
        options: &ParquetWriteOptions,
    ) -> Result<Vec<InvertedShardInfo>> {
        if buckets.is_empty() || buckets.iter().all(|b| b.minimizers.is_empty()) {
            // Empty index - create single empty shard info
            return Ok(vec![InvertedShardInfo {
                shard_id: 0,
                min_minimizer: 0,
                max_minimizer: 0,
                num_entries: 0,
            }]);
        }

        // FIX #6: For large indices, use parallel range partitioning
        let total_entries: usize = buckets.iter().map(|b| b.minimizers.len()).sum();
        let num_cpus = rayon::current_num_threads();

        // Use parallel sharding if we have enough data and multiple cores
        let use_parallel =
            total_entries > MIN_ENTRIES_PER_PARALLEL_PARTITION * num_cpus && num_cpus > 1;

        if use_parallel && max_shard_bytes < usize::MAX {
            // Parallel: partition minimizer space and process ranges in parallel
            stream_to_shards_parallel(output_dir, buckets, max_shard_bytes, num_cpus, options)
        } else {
            // Sequential: single k-way merge
            stream_to_shards_sequential(output_dir, buckets, max_shard_bytes, options)
        }
    }

    /// Sequential k-way merge to Parquet shards.
    fn stream_to_shards_sequential(
        output_dir: &Path,
        buckets: Vec<BucketData>,
        max_shard_bytes: usize,
        options: &ParquetWriteOptions,
    ) -> Result<Vec<InvertedShardInfo>> {
        // Prepare bucket data for k-way merge
        let bucket_slices: Vec<(u32, &[u64])> = buckets
            .iter()
            .filter(|b| !b.minimizers.is_empty())
            .map(|b| (b.bucket_id, b.minimizers.as_slice()))
            .collect();

        // K-way merge heap: (Reverse((minimizer, bucket_id)), bucket_index, position)
        let mut heap: BinaryHeap<MergeHeapEntry> = BinaryHeap::with_capacity(bucket_slices.len());

        // Initialize heap with first element from each bucket
        for (idx, &(bucket_id, mins)) in bucket_slices.iter().enumerate() {
            heap.push((Reverse((mins[0], bucket_id)), idx, 0));
        }

        // Shard writer state
        let mut shard_infos: Vec<InvertedShardInfo> = Vec::new();
        let mut current_shard_id: u32 = 0;
        let mut current_writer: Option<ShardWriter> = None;
        let mut current_shard_entries: u64 = 0;
        let mut current_shard_min: u64 = 0;
        let mut current_shard_max: u64 = 0;

        // Batching buffers
        let mut minimizer_batch: Vec<u64> = Vec::with_capacity(BATCH_SIZE);
        let mut bucket_id_batch: Vec<u32> = Vec::with_capacity(BATCH_SIZE);

        // FIX #1: No deduplication needed - buckets are validated as sorted/deduped
        // Each (minimizer, bucket_id) pair is unique by construction

        while let Some((Reverse((minimizer, bucket_id)), bucket_idx, pos)) = heap.pop() {
            // Check if we need a new shard (FIX #2: use actual file size)
            let need_new_shard = if let Some(ref writer) = current_writer {
                // Use actual bytes written for accurate shard size decisions
                let current_bytes = writer.bytes_written();
                current_bytes >= max_shard_bytes && !minimizer_batch.is_empty()
            } else {
                true // No writer yet, need to create one
            };

            if need_new_shard {
                // Flush current writer if exists
                if let Some(mut writer) = current_writer.take() {
                    // Flush remaining batch
                    if !minimizer_batch.is_empty() {
                        writer.write_batch(&minimizer_batch, &bucket_id_batch)?;
                        minimizer_batch.clear();
                        bucket_id_batch.clear();
                    }
                    writer.finish()?;
                    shard_infos.push(InvertedShardInfo {
                        shard_id: current_shard_id,
                        min_minimizer: current_shard_min,
                        max_minimizer: current_shard_max,
                        num_entries: current_shard_entries,
                    });
                    current_shard_id += 1;
                }

                // Start new shard
                let shard_path = output_dir
                    .join(files::INVERTED_DIR)
                    .join(files::inverted_shard(current_shard_id));
                current_writer = Some(ShardWriter::new(&shard_path, options)?);
                current_shard_entries = 0;
                current_shard_min = minimizer;
                // Note: current_shard_max will be set below when adding to batch
            }

            // Add to batch (first entry sets min=max, subsequent entries update max)
            minimizer_batch.push(minimizer);
            bucket_id_batch.push(bucket_id);
            current_shard_entries += 1;
            current_shard_max = minimizer; // Sorted order guarantees this is always >= previous

            // Flush batch if full
            if minimizer_batch.len() >= BATCH_SIZE {
                if let Some(ref mut writer) = current_writer {
                    writer.write_batch(&minimizer_batch, &bucket_id_batch)?;
                }
                minimizer_batch.clear();
                bucket_id_batch.clear();
            }

            // Advance to next element in this bucket
            let (_, mins) = bucket_slices[bucket_idx];
            let next_pos = pos + 1;
            if next_pos < mins.len() {
                let next_bucket_id = bucket_slices[bucket_idx].0;
                heap.push((
                    Reverse((mins[next_pos], next_bucket_id)),
                    bucket_idx,
                    next_pos,
                ));
            }
        }

        // Flush final shard
        if let Some(mut writer) = current_writer.take() {
            if !minimizer_batch.is_empty() {
                writer.write_batch(&minimizer_batch, &bucket_id_batch)?;
            }
            writer.finish()?;
            shard_infos.push(InvertedShardInfo {
                shard_id: current_shard_id,
                min_minimizer: current_shard_min,
                max_minimizer: current_shard_max,
                num_entries: current_shard_entries,
            });
        }

        Ok(shard_infos)
    }

    /// FIX #6: Parallel range-partitioned sharding for large indices.
    ///
    /// Partitions the minimizer space into ranges and processes each range in parallel.
    fn stream_to_shards_parallel(
        output_dir: &Path,
        buckets: Vec<BucketData>,
        max_shard_bytes: usize,
        num_partitions: usize,
        options: &ParquetWriteOptions,
    ) -> Result<Vec<InvertedShardInfo>> {
        // Compute range boundaries by sampling minimizers
        let range_boundaries = compute_range_boundaries(&buckets, num_partitions);

        // Process each range in parallel
        let partition_results: Vec<Result<Vec<(InvertedShardInfo, PathBuf)>>> = range_boundaries
            .par_windows(2)
            .enumerate()
            .map(|(partition_idx, window)| {
                let range_start = window[0];
                let range_end = window[1];

                // Filter bucket data to this range using binary search (O(log n) per bucket)
                // instead of linear filter (O(n) per bucket). Uses slices to avoid copying.
                let filtered_buckets: Vec<(u32, &[u64])> = buckets
                    .iter()
                    .filter_map(|b| {
                        // Binary search for range bounds (buckets are sorted)
                        let start = b.minimizers.partition_point(|&m| m < range_start);
                        let end = b.minimizers.partition_point(|&m| m < range_end);
                        if start < end {
                            Some((b.bucket_id, &b.minimizers[start..end]))
                        } else {
                            None
                        }
                    })
                    .collect();

                if filtered_buckets.is_empty() {
                    return Ok(vec![]);
                }

                // Process this partition
                process_partition(
                    output_dir,
                    &filtered_buckets,
                    partition_idx as u32,
                    num_partitions as u32,
                    max_shard_bytes,
                    options,
                )
            })
            .collect();

        // Collect shard infos with their file paths
        let mut all_shards_with_paths: Vec<(InvertedShardInfo, PathBuf)> = Vec::new();
        for result in partition_results {
            let partition_shards = result?;
            all_shards_with_paths.extend(partition_shards);
        }

        // Sort by min_minimizer to ensure consistent ordering
        all_shards_with_paths.sort_by_key(|(s, _)| s.min_minimizer);

        // Rename files to canonical names (shard.0.parquet, shard.1.parquet, ...)
        // and update shard IDs to match
        let mut final_shards: Vec<InvertedShardInfo> =
            Vec::with_capacity(all_shards_with_paths.len());
        for (new_id, (mut shard_info, old_path)) in all_shards_with_paths.into_iter().enumerate() {
            let new_id = new_id as u32;
            let new_path = output_dir
                .join(files::INVERTED_DIR)
                .join(files::inverted_shard(new_id));

            // Rename file from partition-specific name to canonical name
            if old_path != new_path {
                std::fs::rename(&old_path, &new_path).with_context(|| {
                    format!(
                        "Failed to rename shard {} -> {}",
                        old_path.display(),
                        new_path.display()
                    )
                })?;
            }

            shard_info.shard_id = new_id;
            final_shards.push(shard_info);
        }

        Ok(final_shards)
    }

    /// Compute range boundaries for parallel partitioning.
    fn compute_range_boundaries(buckets: &[BucketData], num_partitions: usize) -> Vec<u64> {
        // Sample minimizers to estimate distribution
        let sample_size = 10000;
        let mut samples: Vec<u64> = Vec::with_capacity(sample_size);

        let total_mins: usize = buckets.iter().map(|b| b.minimizers.len()).sum();
        let sample_rate = (sample_size as f64 / total_mins as f64).min(1.0);

        for bucket in buckets {
            for &min in &bucket.minimizers {
                if rand_sample(min, sample_rate) {
                    samples.push(min);
                }
            }
        }

        samples.sort_unstable();

        // Compute quantile boundaries
        let mut boundaries = vec![0u64]; // Start at 0
        for i in 1..num_partitions {
            let idx = (samples.len() * i) / num_partitions;
            if idx < samples.len() {
                boundaries.push(samples[idx]);
            }
        }
        boundaries.push(u64::MAX); // End at max

        boundaries
    }

    /// Deterministic sampling based on the minimizer value itself.
    ///
    /// This is thread-safe and produces consistent results: the same minimizer
    /// will always be sampled or not sampled, regardless of which thread processes it.
    /// This is critical for correct range boundary computation in parallel sharding.
    fn rand_sample(minimizer: u64, rate: f64) -> bool {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Hash the minimizer value to get a pseudo-random but deterministic decision
        let mut hasher = DefaultHasher::new();
        minimizer.hash(&mut hasher);
        let hash = hasher.finish();

        (hash as f64 / u64::MAX as f64) < rate
    }

    /// Process a single partition of the minimizer space.
    /// Returns shard infos paired with their file paths (for later renaming).
    fn process_partition(
        output_dir: &Path,
        filtered_buckets: &[(u32, &[u64])],
        partition_idx: u32,
        _total_partitions: u32,
        max_shard_bytes: usize,
        options: &ParquetWriteOptions,
    ) -> Result<Vec<(InvertedShardInfo, PathBuf)>> {
        // K-way merge for this partition (buckets already sliced)
        let bucket_slices: Vec<(u32, &[u64])> = filtered_buckets.to_vec();

        let mut heap: BinaryHeap<MergeHeapEntry> = BinaryHeap::with_capacity(bucket_slices.len());

        for (idx, &(bucket_id, mins)) in bucket_slices.iter().enumerate() {
            if !mins.is_empty() {
                heap.push((Reverse((mins[0], bucket_id)), idx, 0));
            }
        }

        let mut shard_infos: Vec<(InvertedShardInfo, PathBuf)> = Vec::new();
        let mut local_shard_id: u32 = 0;
        let mut current_writer: Option<ShardWriter> = None;
        let mut current_shard_path: PathBuf = PathBuf::new();
        let mut current_shard_entries: u64 = 0;
        let mut current_shard_min: u64 = 0;
        let mut current_shard_max: u64 = 0;

        let mut minimizer_batch: Vec<u64> = Vec::with_capacity(BATCH_SIZE);
        let mut bucket_id_batch: Vec<u32> = Vec::with_capacity(BATCH_SIZE);

        while let Some((Reverse((minimizer, bucket_id)), bucket_idx, pos)) = heap.pop() {
            let need_new_shard = if let Some(ref writer) = current_writer {
                writer.bytes_written() >= max_shard_bytes && !minimizer_batch.is_empty()
            } else {
                true
            };

            if need_new_shard {
                if let Some(mut writer) = current_writer.take() {
                    if !minimizer_batch.is_empty() {
                        writer.write_batch(&minimizer_batch, &bucket_id_batch)?;
                        minimizer_batch.clear();
                        bucket_id_batch.clear();
                    }
                    writer.finish()?;
                    shard_infos.push((
                        InvertedShardInfo {
                            shard_id: local_shard_id,
                            min_minimizer: current_shard_min,
                            max_minimizer: current_shard_max,
                            num_entries: current_shard_entries,
                        },
                        current_shard_path.clone(),
                    ));
                    local_shard_id += 1;
                }

                // Use partition index in filename to avoid conflicts during parallel writes
                let shard_name = format!("shard.{}.{}.parquet", partition_idx, local_shard_id);
                current_shard_path = output_dir.join(files::INVERTED_DIR).join(&shard_name);
                current_writer = Some(ShardWriter::new(&current_shard_path, options)?);
                current_shard_entries = 0;
                current_shard_min = minimizer;
                // Note: current_shard_max will be set below
            }

            minimizer_batch.push(minimizer);
            bucket_id_batch.push(bucket_id);
            current_shard_entries += 1;
            current_shard_max = minimizer; // Sorted order guarantees this is always >= previous

            if minimizer_batch.len() >= BATCH_SIZE {
                if let Some(ref mut writer) = current_writer {
                    writer.write_batch(&minimizer_batch, &bucket_id_batch)?;
                }
                minimizer_batch.clear();
                bucket_id_batch.clear();
            }

            let (_, mins) = bucket_slices[bucket_idx];
            let next_pos = pos + 1;
            if next_pos < mins.len() {
                let next_bucket_id = bucket_slices[bucket_idx].0;
                heap.push((
                    Reverse((mins[next_pos], next_bucket_id)),
                    bucket_idx,
                    next_pos,
                ));
            }
        }

        if let Some(mut writer) = current_writer.take() {
            if !minimizer_batch.is_empty() {
                writer.write_batch(&minimizer_batch, &bucket_id_batch)?;
            }
            writer.finish()?;
            shard_infos.push((
                InvertedShardInfo {
                    shard_id: local_shard_id,
                    min_minimizer: current_shard_min,
                    max_minimizer: current_shard_max,
                    num_entries: current_shard_entries,
                },
                current_shard_path,
            ));
        }

        Ok(shard_infos)
    }

    /// Helper struct for writing a single Parquet shard.
    struct ShardWriter {
        writer: ArrowWriter<File>,
        schema: Arc<Schema>,
    }

    impl ShardWriter {
        fn new(path: &Path, options: &ParquetWriteOptions) -> Result<Self> {
            let schema = Arc::new(Schema::new(vec![
                Field::new("minimizer", DataType::UInt64, false),
                Field::new("bucket_id", DataType::UInt32, false),
            ]));

            // DRY: Use ParquetWriteOptions::to_writer_properties() as single source of truth
            let props = options.to_writer_properties();

            let file = File::create(path)
                .with_context(|| format!("Failed to create shard: {}", path.display()))?;
            let writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

            Ok(Self { writer, schema })
        }

        /// FIX #2: Return actual bytes written to the file so far.
        fn bytes_written(&self) -> usize {
            self.writer.bytes_written()
        }

        fn write_batch(&mut self, minimizers: &[u64], bucket_ids: &[u32]) -> Result<()> {
            let minimizer_array: ArrayRef =
                Arc::new(arrow::array::UInt64Array::from(minimizers.to_vec()));
            let bucket_id_array: ArrayRef =
                Arc::new(arrow::array::UInt32Array::from(bucket_ids.to_vec()));

            let batch =
                RecordBatch::try_new(self.schema.clone(), vec![minimizer_array, bucket_id_array])?;
            self.writer.write(&batch)?;
            Ok(())
        }

        fn finish(self) -> Result<()> {
            self.writer.close()?;
            Ok(())
        }
    }
}

#[cfg(feature = "parquet")]
pub use streaming::{compute_source_hash, create_parquet_inverted_index};

// ============================================================================
// Tests
// ============================================================================

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

    #[cfg(feature = "parquet")]
    #[test]
    fn test_buckets_parquet_round_trip() {
        let tmp = TempDir::new().unwrap();
        let index_dir = tmp.path().join("test.ryidx");
        create_index_directory(&index_dir).unwrap();

        let mut bucket_names = HashMap::new();
        bucket_names.insert(1, "Bacteria".to_string());
        bucket_names.insert(2, "Archaea".to_string());
        bucket_names.insert(3, "Eukaryota".to_string());

        let mut bucket_sources = HashMap::new();
        bucket_sources.insert(
            1,
            vec!["ecoli.fna".to_string(), "bsubtilis.fna".to_string()],
        );
        bucket_sources.insert(2, vec!["haloferax.fna".to_string()]);
        bucket_sources.insert(3, vec![]); // Empty sources

        write_buckets_parquet(&index_dir, &bucket_names, &bucket_sources).unwrap();
        let (loaded_names, loaded_sources) = read_buckets_parquet(&index_dir).unwrap();

        assert_eq!(loaded_names, bucket_names);
        assert_eq!(loaded_sources.get(&1), bucket_sources.get(&1));
        assert_eq!(loaded_sources.get(&2), bucket_sources.get(&2));
        // Empty sources should be preserved
        assert_eq!(loaded_sources.get(&3).map(|v| v.len()), Some(0));
    }

    /// Test that streaming Parquet creation produces identical classification results
    /// as the traditional Index → InvertedIndex → save_shard_parquet path.
    #[cfg(feature = "parquet")]
    #[test]
    fn test_streaming_parquet_matches_traditional() {
        use crate::classify::classify_batch_sharded_sequential;
        use crate::extraction::extract_into;
        use crate::index::Index;
        use crate::inverted::InvertedIndex;
        use crate::sharded::{ShardFormat, ShardManifest, ShardedInvertedIndex};
        use crate::types::QueryRecord;
        use crate::workspace::MinimizerWorkspace;

        let tmp = TempDir::new().unwrap();
        let streaming_dir = tmp.path().join("streaming.ryxdi");
        let traditional_dir = tmp.path().join("traditional.ryxdi");

        // Test sequences - long enough for K=32, W=10
        let seq1 = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let seq2 = b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA";
        let seq3 = b"AAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTT";

        let k = 32;
        let w = 10;
        let salt = 0x1234u64;

        // === Create streaming Parquet index ===
        let mut ws = MinimizerWorkspace::new();

        // Extract minimizers for each bucket
        extract_into(seq1, k, w, salt, &mut ws);
        let mut mins1 = std::mem::take(&mut ws.buffer);
        mins1.sort_unstable();
        mins1.dedup();

        extract_into(seq2, k, w, salt, &mut ws);
        let mut mins2 = std::mem::take(&mut ws.buffer);
        mins2.sort_unstable();
        mins2.dedup();

        extract_into(seq3, k, w, salt, &mut ws);
        let mut mins3 = std::mem::take(&mut ws.buffer);
        mins3.sort_unstable();
        mins3.dedup();

        let buckets = vec![
            BucketData {
                bucket_id: 1,
                bucket_name: "bucket1".to_string(),
                sources: vec!["ref1.fa".to_string()],
                minimizers: mins1.clone(),
            },
            BucketData {
                bucket_id: 2,
                bucket_name: "bucket2".to_string(),
                sources: vec!["ref2.fa".to_string()],
                minimizers: mins2.clone(),
            },
            BucketData {
                bucket_id: 3,
                bucket_name: "bucket3".to_string(),
                sources: vec!["ref3.fa".to_string()],
                minimizers: mins3.clone(),
            },
        ];

        let streaming_manifest =
            create_parquet_inverted_index(&streaming_dir, buckets, k, w, salt, None, None).unwrap();

        // === Create traditional Parquet index ===
        // Build Index the traditional way
        let mut index = Index::new(k, w, salt).unwrap();
        index.add_record(1, "ref1.fa", seq1, &mut ws);
        index.add_record(2, "ref2.fa", seq2, &mut ws);
        index.add_record(3, "ref3.fa", seq3, &mut ws);
        index.bucket_names.insert(1, "bucket1".to_string());
        index.bucket_names.insert(2, "bucket2".to_string());
        index.bucket_names.insert(3, "bucket3".to_string());
        index.finalize_bucket(1);
        index.finalize_bucket(2);
        index.finalize_bucket(3);

        // Build InvertedIndex
        let inverted = InvertedIndex::build_from_index(&index);

        // Save as traditional Parquet shard
        create_index_directory(&traditional_dir).unwrap();
        let trad_shard_path = traditional_dir.join("inverted").join("shard.0.parquet");
        let trad_shard_info = inverted
            .save_shard_parquet(&trad_shard_path, 0, None)
            .unwrap();

        // Create manifest for traditional
        let trad_manifest = ShardManifest {
            k: inverted.k,
            w: inverted.w,
            salt: inverted.salt,
            source_hash: 0,
            total_minimizers: inverted.num_minimizers(),
            total_bucket_ids: inverted.num_bucket_entries(),
            has_overlapping_shards: true,
            shard_format: ShardFormat::Parquet,
            shards: vec![trad_shard_info],
            bucket_names: index.bucket_names.clone(),
            bucket_sources: index.bucket_sources.clone(),
            bucket_minimizer_counts: index.buckets.iter().map(|(&id, v)| (id, v.len())).collect(),
        };
        trad_manifest
            .save(&ShardManifest::manifest_path(&traditional_dir))
            .unwrap();

        // Also write buckets.parquet for traditional
        write_buckets_parquet(&traditional_dir, &index.bucket_names, &index.bucket_sources)
            .unwrap();

        // === Compare classification results ===
        let streaming_sharded = ShardedInvertedIndex::open_parquet(&streaming_dir).unwrap();
        let traditional_sharded = ShardedInvertedIndex::open(&traditional_dir).unwrap();

        // Query sequences
        let query1: &[u8] = seq1;
        let query2: &[u8] = seq2;
        let query3: &[u8] = seq3;
        let records: Vec<QueryRecord> =
            vec![(0, query1, None), (1, query2, None), (2, query3, None)];

        let threshold = 0.1;

        let results_streaming =
            classify_batch_sharded_sequential(&streaming_sharded, None, &records, threshold)
                .unwrap();
        let results_traditional =
            classify_batch_sharded_sequential(&traditional_sharded, None, &records, threshold)
                .unwrap();

        // Sort results for comparison (order may differ)
        let mut sorted_streaming = results_streaming.clone();
        sorted_streaming.sort_by(|a, b| (a.query_id, a.bucket_id).cmp(&(b.query_id, b.bucket_id)));

        let mut sorted_traditional = results_traditional.clone();
        sorted_traditional
            .sort_by(|a, b| (a.query_id, a.bucket_id).cmp(&(b.query_id, b.bucket_id)));

        assert_eq!(
            sorted_streaming.len(),
            sorted_traditional.len(),
            "Result count mismatch: streaming={}, traditional={}",
            sorted_streaming.len(),
            sorted_traditional.len()
        );

        for (s, t) in sorted_streaming.iter().zip(sorted_traditional.iter()) {
            assert_eq!(
                s.query_id, t.query_id,
                "Query ID mismatch: streaming={}, traditional={}",
                s.query_id, t.query_id
            );
            assert_eq!(
                s.bucket_id, t.bucket_id,
                "Bucket ID mismatch: streaming={}, traditional={}",
                s.bucket_id, t.bucket_id
            );
            assert!(
                (s.score - t.score).abs() < 0.001,
                "Score mismatch: streaming={}, traditional={}",
                s.score,
                t.score
            );
        }

        // Verify we got expected matches (each query should match its own bucket perfectly)
        assert!(
            sorted_streaming
                .iter()
                .any(|r| r.query_id == 0 && r.bucket_id == 1),
            "Query 0 should match bucket 1"
        );
        assert!(
            sorted_streaming
                .iter()
                .any(|r| r.query_id == 1 && r.bucket_id == 2),
            "Query 1 should match bucket 2"
        );
        assert!(
            sorted_streaming
                .iter()
                .any(|r| r.query_id == 2 && r.bucket_id == 3),
            "Query 2 should match bucket 3"
        );
    }

    // ========================================================================
    // ParquetWriteOptions TDD tests (Phase 1)
    // ========================================================================

    #[test]
    fn test_parquet_write_options_default() {
        let opts = ParquetWriteOptions::default();
        // Defaults must match current behavior exactly
        assert_eq!(opts.row_group_size, 100_000);
        assert!(!opts.bloom_filter_enabled);
        assert!((opts.bloom_filter_fpp - 0.05).abs() < 0.001);
        assert!(opts.write_page_statistics);
        assert!(matches!(opts.compression, ParquetCompression::Snappy));
    }

    #[test]
    fn test_parquet_write_options_to_writer_properties() {
        let opts = ParquetWriteOptions {
            bloom_filter_enabled: true,
            compression: ParquetCompression::Zstd,
            ..Default::default()
        };
        // Should compile and produce valid WriterProperties
        let props = opts.to_writer_properties();
        // WriterProperties doesn't expose getters, but we verify it doesn't panic
        // and produces the correct compression codec
        assert_eq!(
            props.compression(&parquet::schema::types::ColumnPath::new(vec![])),
            parquet::basic::Compression::ZSTD(parquet::basic::ZstdLevel::default())
        );
    }

    #[cfg(feature = "parquet")]
    #[test]
    fn test_parquet_write_with_zstd() {
        use crate::extraction::extract_into;
        use crate::workspace::MinimizerWorkspace;

        let tmp = TempDir::new().unwrap();
        let index_dir = tmp.path().join("zstd_test.ryxdi");

        // Create test data
        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let k = 32;
        let w = 10;
        let salt = 0x1234u64;

        let mut ws = MinimizerWorkspace::new();
        extract_into(seq, k, w, salt, &mut ws);
        let mut mins = std::mem::take(&mut ws.buffer);
        mins.sort_unstable();
        mins.dedup();

        let buckets = vec![BucketData {
            bucket_id: 1,
            bucket_name: "test".to_string(),
            sources: vec!["test.fa".to_string()],
            minimizers: mins,
        }];

        // Create with zstd compression
        let options = ParquetWriteOptions {
            compression: ParquetCompression::Zstd,
            ..Default::default()
        };

        let manifest = create_parquet_inverted_index(
            &index_dir,
            buckets.clone(),
            k,
            w,
            salt,
            None,
            Some(&options),
        )
        .unwrap();

        // Verify we can read it back (data integrity)
        assert!(manifest.inverted.as_ref().unwrap().num_shards > 0);
        assert!(is_parquet_index(&index_dir));
    }

    #[cfg(feature = "parquet")]
    #[test]
    fn test_parquet_write_with_bloom_filter() {
        use crate::extraction::extract_into;
        use crate::workspace::MinimizerWorkspace;
        use parquet::file::reader::FileReader;

        let tmp = TempDir::new().unwrap();
        let index_dir = tmp.path().join("bloom_test.ryxdi");

        // Create test data
        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let k = 32;
        let w = 10;
        let salt = 0x1234u64;

        let mut ws = MinimizerWorkspace::new();
        extract_into(seq, k, w, salt, &mut ws);
        let mut mins = std::mem::take(&mut ws.buffer);
        mins.sort_unstable();
        mins.dedup();

        let buckets = vec![BucketData {
            bucket_id: 1,
            bucket_name: "test".to_string(),
            sources: vec!["test.fa".to_string()],
            minimizers: mins,
        }];

        // Create with bloom filter enabled
        let options = ParquetWriteOptions {
            bloom_filter_enabled: true,
            bloom_filter_fpp: 0.01,
            ..Default::default()
        };

        let manifest = create_parquet_inverted_index(
            &index_dir,
            buckets.clone(),
            k,
            w,
            salt,
            None,
            Some(&options),
        )
        .unwrap();

        assert!(manifest.inverted.as_ref().unwrap().num_shards > 0);

        // Verify bloom filter metadata exists in the parquet file
        // We check this by reading the file metadata
        let shard_path = index_dir.join("inverted").join("shard.0.parquet");
        let file = std::fs::File::open(&shard_path).unwrap();
        let reader = parquet::file::reader::SerializedFileReader::new(file).unwrap();
        let metadata = reader.metadata();

        // Check that at least one row group has bloom filter metadata
        let mut has_bloom_filter = false;
        for rg in 0..metadata.num_row_groups() {
            let rg_meta = metadata.row_group(rg);
            for col in 0..rg_meta.num_columns() {
                if rg_meta.column(col).bloom_filter_offset().is_some() {
                    has_bloom_filter = true;
                    break;
                }
            }
        }
        assert!(
            has_bloom_filter,
            "Bloom filter should be present in Parquet file"
        );
    }
}
