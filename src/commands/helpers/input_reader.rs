//! Input reader setup for classification commands.
//!
//! Provides unified configuration and creation of input readers for both
//! Parquet and FASTX (FASTA/FASTQ) input formats.

use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};

use super::fastx_io::PrefetchingIoHandler;
use super::parquet_io::PrefetchingParquetReader;

/// Configuration for creating an input reader.
pub struct InputReaderConfig<'a> {
    /// Path to the R1 (first) read file.
    pub r1_path: &'a Path,
    /// Optional path to the R2 (second) read file for paired-end FASTX input.
    pub r2_path: Option<&'a PathBuf>,
    /// Number of records per batch.
    pub batch_size: usize,
    /// Number of row groups to read in parallel (0 = sequential).
    pub parallel_input_rg: usize,
    /// Whether the input is in Parquet format.
    pub is_parquet: bool,
    /// Optional trim length for sequences. When set, sequences are trimmed at read time
    /// to reduce memory usage for long reads. Records with R1 shorter than this are skipped.
    pub trim_to: Option<usize>,
    /// Optional minimum R1 length filter. Applied before `trim_to`:
    /// reads with R1 shorter than this are skipped entirely (not modified).
    pub minimum_length: Option<usize>,
}

/// Unified input reader for classification.
///
/// Wraps either a Parquet or FASTX reader, allowing the caller to match
/// on the variant and use the appropriate processing logic.
#[allow(clippy::large_enum_variant)]
pub enum ClassificationInput {
    /// Parquet input with zero-copy batch reading.
    Parquet(PrefetchingParquetReader),
    /// FASTX (FASTA/FASTQ) input with background prefetching.
    Fastx(PrefetchingIoHandler),
}

impl ClassificationInput {
    /// Finish reading and clean up resources.
    ///
    /// Returns any errors that occurred during background I/O operations.
    pub fn finish(&mut self) -> Result<()> {
        match self {
            ClassificationInput::Parquet(reader) => reader.finish(),
            ClassificationInput::Fastx(io) => io.finish(),
        }
    }
}

/// Validate input configuration for classification.
///
/// Checks for incompatible input combinations.
///
/// # Errors
///
/// Returns an error if:
/// - Parquet input is combined with a separate R2 file (Parquet paired-end
///   data should use the 'sequence2' column instead).
pub fn validate_input_config(is_parquet: bool, r2_path: Option<&PathBuf>) -> Result<()> {
    if is_parquet && r2_path.is_some() {
        return Err(anyhow!(
            "Parquet input with separate R2 file is not supported. \
             Use a Parquet file with 'sequence2' column for paired-end data."
        ));
    }
    Ok(())
}

/// Create an input reader based on the configuration.
///
/// Creates either a Parquet or FASTX reader depending on the input format.
///
/// `preserve_for_output` controls whether quality scores are captured in FASTX records.
/// This is a separate parameter (not part of `InputReaderConfig`) to make the opt-in
/// explicit — it uses ~25% more memory and should only be enabled for `--output-sequences`.
pub fn create_input_reader(
    config: &InputReaderConfig,
    preserve_for_output: bool,
) -> Result<ClassificationInput> {
    if config.is_parquet {
        let parallel_rg_opt = if config.parallel_input_rg > 0 {
            Some(config.parallel_input_rg)
        } else {
            None
        };
        log::info!(
            "Using prefetching Parquet input reader (batch_size={}, parallel_rg={:?}) for {:?}",
            config.batch_size,
            parallel_rg_opt,
            config.r1_path
        );
        let reader = PrefetchingParquetReader::with_parallel_row_groups(
            config.r1_path,
            config.batch_size,
            parallel_rg_opt,
            config.trim_to,
            config.minimum_length,
        )?;
        Ok(ClassificationInput::Parquet(reader))
    } else {
        log::debug!(
            "Using prefetching FASTX input reader (batch_size={}, preserve_for_output={}) for {:?}",
            config.batch_size,
            preserve_for_output,
            config.r1_path
        );
        let reader = PrefetchingIoHandler::with_options(
            config.r1_path,
            config.r2_path,
            None, // No output - just reading
            config.batch_size,
            config.trim_to,
            config.minimum_length,
            preserve_for_output,
        )?;
        Ok(ClassificationInput::Fastx(reader))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_input_config_parquet_with_r2_fails() {
        let r2 = PathBuf::from("reads_R2.fastq");
        let result = validate_input_config(true, Some(&r2));

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Parquet input with separate R2 file is not supported"));
        assert!(err.contains("sequence2"));
    }

    #[test]
    fn test_validate_input_config_parquet_without_r2_passes() {
        let result = validate_input_config(true, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_input_config_fastx_with_r2_passes() {
        let r2 = PathBuf::from("reads_R2.fastq");
        let result = validate_input_config(false, Some(&r2));
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_input_config_fastx_without_r2_passes() {
        let result = validate_input_config(false, None);
        assert!(result.is_ok());
    }
}
