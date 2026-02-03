//! Index loading utilities for classification commands.
//!
//! This module provides shared index loading logic used by both `run_classify`
//! and `run_log_ratio` commands.

use anyhow::{anyhow, Result};
use std::path::Path;

use rype::memory::{detect_available_memory, format_bytes};
use rype::{parquet_index, IndexMetadata, ShardedInvertedIndex};

use super::metadata::load_index_metadata;

/// Options controlling how the index is loaded for classification.
#[derive(Debug, Clone, Default)]
pub struct IndexLoadOptions {
    /// Enable bloom filter row group filtering for faster classification.
    pub use_bloom_filter: bool,
    /// Enable parallel row group processing (affects prefetch behavior).
    pub parallel_rg: bool,
}

/// A fully loaded index ready for classification.
///
/// Contains all the components needed for classification:
/// - Metadata (k, w, salt, bucket names/sources)
/// - Sharded inverted index for memory-efficient classification
/// - Optional Parquet read options (e.g., bloom filter settings)
#[derive(Debug)]
pub struct LoadedIndex {
    /// Index metadata including bucket names and sources.
    pub metadata: IndexMetadata,
    /// Memory-efficient sharded inverted index.
    pub sharded: ShardedInvertedIndex,
    /// Optional read options for Parquet index access.
    pub read_options: Option<parquet_index::ParquetReadOptions>,
}

/// Validate that the given path is a valid Parquet index directory.
///
/// # Errors
/// Returns an error with a helpful message if the index is not found
/// or not in Parquet format.
pub fn validate_parquet_index(path: &Path) -> Result<()> {
    if !rype::is_parquet_index(path) {
        return Err(anyhow!(
            "Index not found or not in Parquet format: {}\n\
             Create an index with: rype index create -o index.ryxdi -r refs.fasta",
            path.display()
        ));
    }
    Ok(())
}

/// Load an index for classification with the given options.
///
/// This function:
/// 1. Validates the index path is a Parquet index
/// 2. Loads the index metadata (k, w, salt, bucket names/sources)
/// 3. Opens the sharded inverted index
/// 4. Builds read options if bloom filter is enabled
/// 5. Optionally advises kernel to prefetch index data (if parallel_rg enabled)
///
/// # Arguments
/// * `path` - Path to the Parquet index directory (.ryxdi)
/// * `options` - Options controlling bloom filter and prefetch behavior
///
/// # Returns
/// A `LoadedIndex` containing metadata, sharded index, and read options.
///
/// # Errors
/// Returns an error if the index cannot be loaded.
pub fn load_index_for_classification(
    path: &Path,
    options: &IndexLoadOptions,
) -> Result<LoadedIndex> {
    // Validate index path
    validate_parquet_index(path)?;

    // Load metadata
    log::info!("Detected Parquet index at {:?}", path);
    let metadata = load_index_metadata(path)?;
    log::info!("Metadata loaded: {} buckets", metadata.bucket_names.len());

    // Open sharded inverted index
    log::info!("Loading Parquet inverted index from {:?}", path);
    let sharded = ShardedInvertedIndex::open(path)?;
    log::info!(
        "Sharded index: {} shards, {} total minimizers",
        sharded.num_shards(),
        sharded.total_minimizers()
    );

    // Build read options
    let read_options = if options.use_bloom_filter {
        log::info!("Bloom filter row group filtering enabled");
        Some(parquet_index::ParquetReadOptions::with_bloom_filter())
    } else {
        None
    };

    // Advise kernel to prefetch if parallel_rg is enabled
    if options.parallel_rg {
        let available = detect_available_memory();
        let prefetch_budget = available.bytes / 2;
        let advised = sharded.advise_prefetch(Some(prefetch_budget));
        if advised > 0 {
            log::info!(
                "Advised kernel to prefetch {} of index data",
                format_bytes(advised)
            );
        }
    }

    Ok(LoadedIndex {
        metadata,
        sharded,
        read_options,
    })
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_validate_parquet_index_nonexistent_path() {
        let path = PathBuf::from("/nonexistent/path/index.ryxdi");
        let result = validate_parquet_index(&path);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not found or not in Parquet format"));
        assert!(err.contains("rype index create"));
    }

    #[test]
    fn test_validate_parquet_index_regular_file() {
        // Use Cargo.toml as a non-index file
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        let result = validate_parquet_index(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_index_load_options_default() {
        let options = IndexLoadOptions::default();
        assert!(!options.use_bloom_filter);
        assert!(!options.parallel_rg);
    }

    #[test]
    fn test_index_load_options_with_values() {
        let options = IndexLoadOptions {
            use_bloom_filter: true,
            parallel_rg: true,
        };
        assert!(options.use_bloom_filter);
        assert!(options.parallel_rg);
    }

    #[test]
    fn test_load_index_nonexistent_path() {
        let path = PathBuf::from("/nonexistent/path/index.ryxdi");
        let options = IndexLoadOptions::default();
        let result = load_index_for_classification(&path, &options);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not found or not in Parquet format"));
    }
}
