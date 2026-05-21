//! Write options for the `.ryci` cluster-index format.
//!
//! Mirrors `crate::indices::parquet::options::ParquetWriteOptions` but applies
//! `DELTA_BINARY_PACKED` to all three columns (minimizer, bucket_id, position).
//! Bloom filters are intentionally absent: per the plan, prior measurements on
//! the classify-side index showed they did not help, and the cluster index has
//! no analogue of classify's random-access lookup pattern that bloom filters
//! exist to accelerate.

use crate::constants::DEFAULT_ROW_GROUP_SIZE;
use crate::error::{Result, RypeError};
use crate::indices::parquet::ParquetCompression;
use parquet::file::properties::WriterProperties;

/// Configuration options for writing `.ryci` cluster-index shards.
///
/// Defaults match the classify-side defaults (Snappy, 100K row groups, no bloom).
#[derive(Debug, Clone)]
pub struct ClusterParquetWriteOptions {
    /// Maximum rows per row group. Default: `DEFAULT_ROW_GROUP_SIZE` (100,000).
    pub row_group_size: usize,
    /// Compression codec. Default: Snappy.
    pub compression: ParquetCompression,
    /// Write page-level statistics. Default: true.
    pub write_page_statistics: bool,
}

impl Default for ClusterParquetWriteOptions {
    fn default() -> Self {
        Self {
            row_group_size: DEFAULT_ROW_GROUP_SIZE,
            compression: ParquetCompression::Snappy,
            write_page_statistics: true,
        }
    }
}

impl ClusterParquetWriteOptions {
    pub fn validate(&self) -> Result<()> {
        if self.row_group_size == 0 {
            return Err(RypeError::validation(
                "row_group_size must be > 0".to_string(),
            ));
        }
        Ok(())
    }

    /// Build the Parquet `WriterProperties` for this options set.
    ///
    /// All three columns (minimizer, bucket_id, position) use
    /// `DELTA_BINARY_PACKED` encoding with dictionary encoding disabled.
    /// Note: only the minimizer column is actually sorted within a row group,
    /// so position's delta encoding is best-effort — accepted in v1 per the plan.
    pub fn to_writer_properties(&self) -> WriterProperties {
        self.validate()
            .expect("Invalid ClusterParquetWriteOptions - call validate() first");
        use parquet::basic::{Compression, Encoding};
        use parquet::file::properties::{EnabledStatistics, WriterVersion};
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
        let position_col = ColumnPath::new(vec!["position".to_string()]);

        WriterProperties::builder()
            .set_writer_version(WriterVersion::PARQUET_2_0)
            .set_compression(compression)
            .set_statistics_enabled(statistics)
            .set_max_row_group_size(self.row_group_size)
            .set_column_encoding(minimizer_col.clone(), Encoding::DELTA_BINARY_PACKED)
            .set_column_dictionary_enabled(minimizer_col, false)
            .set_column_encoding(bucket_id_col.clone(), Encoding::DELTA_BINARY_PACKED)
            .set_column_dictionary_enabled(bucket_id_col, false)
            .set_column_encoding(position_col.clone(), Encoding::DELTA_BINARY_PACKED)
            .set_column_dictionary_enabled(position_col, false)
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_validate() {
        ClusterParquetWriteOptions::default().validate().unwrap();
    }

    #[test]
    fn zero_row_group_size_rejected() {
        let opts = ClusterParquetWriteOptions {
            row_group_size: 0,
            ..Default::default()
        };
        assert!(opts.validate().is_err());
    }

    #[test]
    fn to_writer_properties_produces_zstd_when_requested() {
        let opts = ClusterParquetWriteOptions {
            compression: ParquetCompression::Zstd,
            ..Default::default()
        };
        let props = opts.to_writer_properties();
        assert_eq!(
            props.compression(&parquet::schema::types::ColumnPath::new(vec![])),
            parquet::basic::Compression::ZSTD(parquet::basic::ZstdLevel::default())
        );
    }
}
