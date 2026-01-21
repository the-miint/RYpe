//! Parquet write and read options.
//!
//! This module contains configuration options for Parquet file I/O,
//! including compression settings and bloom filter configuration.

use crate::constants::DEFAULT_ROW_GROUP_SIZE;
use crate::error::{Result, RypeError};

/// Serialize u64 as hex string for TOML compatibility (i64 overflow).
pub mod hex_u64 {
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
            row_group_size: DEFAULT_ROW_GROUP_SIZE,
            compression: ParquetCompression::Snappy,
            bloom_filter_enabled: false,
            bloom_filter_fpp: 0.05,
            write_page_statistics: true,
        }
    }
}

impl ParquetWriteOptions {
    /// Validate options. Returns error if any values are out of bounds.
    ///
    /// Checks:
    /// - `bloom_filter_fpp` must be in (0.0, 1.0)
    /// - `row_group_size` must be > 0
    pub fn validate(&self) -> Result<()> {
        if self.bloom_filter_fpp <= 0.0 || self.bloom_filter_fpp >= 1.0 {
            return Err(RypeError::validation(format!(
                "bloom_filter_fpp must be in (0.0, 1.0), got {}",
                self.bloom_filter_fpp
            )));
        }
        if self.row_group_size == 0 {
            return Err(RypeError::validation(
                "row_group_size must be > 0".to_string(),
            ));
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
            // Set NDV (number of distinct values) to row_group_size.
            // Analysis shows ~99% of minimizers in a row group are unique, so NDV ≈ row_group_size.
            // Without this, arrow-rs defaults to NDV=1,000,000 which creates 10x larger bloom filters.
            let ndv = self.row_group_size as u64;
            builder = builder
                .set_column_bloom_filter_enabled(minimizer_col.clone(), true)
                .set_column_bloom_filter_fpp(minimizer_col.clone(), self.bloom_filter_fpp)
                .set_column_bloom_filter_ndv(minimizer_col, ndv)
                .set_column_bloom_filter_enabled(bucket_id_col.clone(), true)
                .set_column_bloom_filter_fpp(bucket_id_col.clone(), self.bloom_filter_fpp)
                .set_column_bloom_filter_ndv(bucket_id_col, ndv);
        }

        builder.build()
    }
}

/// Configuration options for reading Parquet inverted index files.
///
/// Use `Default::default()` to get backward-compatible behavior (no bloom filter usage).
/// Pass custom options to enable bloom filter row group filtering.
///
/// # Example
/// ```ignore
/// let opts = ParquetReadOptions {
///     use_bloom_filter: true,
/// };
/// ```
#[derive(Debug, Clone, Default)]
pub struct ParquetReadOptions {
    /// Use bloom filters for row group filtering if available. Default: false.
    ///
    /// When true, row groups are rejected if the bloom filter definitively indicates
    /// that none of the query minimizers are present. This can significantly reduce
    /// I/O for sparse queries.
    ///
    /// Falls back gracefully when bloom filters are not present in the file.
    /// Requires the index to have been built with `--parquet-bloom-filter`.
    pub use_bloom_filter: bool,
}

impl ParquetReadOptions {
    /// Create options with bloom filter enabled.
    pub fn with_bloom_filter() -> Self {
        Self {
            use_bloom_filter: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
