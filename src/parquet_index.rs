//! Parquet-based index format implementation.
//!
//! This module provides read/write support for the Parquet-based index format,
//! which stores index data in a directory structure:
//!
//! ```text
//! index.ryidx/
//! ├── manifest.toml        # Metadata: k, w, salt, format version
//! ├── buckets.parquet      # bucket_id → bucket_name, sources
//! ├── main.parquet         # (bucket_id, minimizer) flat data
//! └── inverted/
//!     └── shard.0.parquet  # (minimizer, bucket_id) inverted data
//! ```

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

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
    pub const MAIN: &str = "main.parquet";
    pub const INVERTED_DIR: &str = "inverted";

    /// Generate shard filename for inverted index.
    pub fn inverted_shard(shard_id: u32) -> String {
        format!("shard.{}.parquet", shard_id)
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

    /// XOR salt applied to k-mer hashes.
    pub salt: u64,

    /// Hash of source data for change detection.
    pub source_hash: u64,

    /// Number of buckets in the index.
    pub num_buckets: u32,

    /// Total minimizers in main index.
    pub total_minimizers: u64,

    /// Inverted index shard information (if present).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inverted: Option<InvertedManifest>,
}

/// Manifest section for inverted index shards.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvertedManifest {
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

    /// Minimum minimizer value in this shard.
    pub min_minimizer: u64,

    /// Maximum minimizer value in this shard.
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
}
