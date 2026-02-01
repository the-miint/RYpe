//! Output format detection and writing for classification results.

use anyhow::{anyhow, Context, Result};
use arrow::array::{ArrayRef, Float64Builder, RecordBatch, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use flate2::write::GzEncoder;
use flate2::Compression;
use parquet::arrow::ArrowWriter;
use parquet::basic::ZstdLevel;
use parquet::file::properties::WriterProperties;
use parquet::schema::types::ColumnPath;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;

/// Default batch size for Parquet output buffering.
const DEFAULT_PARQUET_OUTPUT_BATCH_SIZE: usize = 10_000;

/// Output format auto-detected from file extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// Plain TSV (default, .tsv, or no extension)
    Tsv,
    /// Gzip-compressed TSV (.tsv.gz or .gz)
    TsvGz,
    /// Apache Parquet (.parquet)
    Parquet,
}

impl OutputFormat {
    /// Detect output format from file path.
    ///
    /// Format detection rules:
    /// - `None` or `"-"` → TSV to stdout
    /// - `.parquet` extension → Parquet
    /// - `.gz` extension (including `.tsv.gz`) → Gzip-compressed TSV
    /// - Everything else → Plain TSV
    ///
    /// Uses proper Path extension parsing to avoid false matches.
    pub fn detect(path: Option<&PathBuf>) -> Self {
        let Some(p) = path else {
            return OutputFormat::Tsv;
        };

        // Check for stdout marker
        if p.as_os_str() == "-" {
            return OutputFormat::Tsv;
        }

        // Check extension using proper path parsing
        let ext = p.extension().and_then(|e| e.to_str());

        match ext {
            Some("parquet") => OutputFormat::Parquet,
            Some("gz") => OutputFormat::TsvGz,
            _ => OutputFormat::Tsv,
        }
    }

    /// Returns true if this format outputs to stdout (when path is None or "-").
    pub fn is_stdout(path: Option<&PathBuf>) -> bool {
        match path {
            None => true,
            Some(p) => p.as_os_str() == "-",
        }
    }
}

/// Writer abstraction supporting multiple output formats.
///
/// Handles TSV, gzip-compressed TSV, and Parquet output formats.
/// For Parquet output, use `write_record()` to avoid TSV serialization overhead.
#[allow(clippy::large_enum_variant)]
pub enum OutputWriter {
    /// Plain TSV output (file or stdout)
    Tsv(BufWriter<Box<dyn Write + Send>>),
    /// Gzip-compressed TSV output. Uses Option to allow taking ownership for proper finalization.
    TsvGz(Option<BufWriter<GzEncoder<File>>>),
    /// Parquet output with batched writing (long format: read_id, bucket_name, score)
    Parquet {
        writer: ArrowWriter<File>,
        read_ids: Vec<String>,
        bucket_names: Vec<String>,
        scores: Vec<f64>,
        batch_size: usize,
    },
    /// Parquet output with wide format (read_id, bucket_1_score, bucket_2_score, ...)
    ParquetWide {
        writer: ArrowWriter<File>,
        /// Sorted bucket names for column names (ordered by bucket_id)
        bucket_col_names: Vec<String>,
        read_ids: Vec<String>,
        /// Flat score array: scores[row * num_buckets + col] = score for row's col-th bucket
        scores: Vec<f64>,
        batch_size: usize,
    },
}

impl OutputWriter {
    /// Create a new output writer based on format and path.
    ///
    /// # Arguments
    /// * `format` - Output format (Tsv, TsvGz, or Parquet)
    /// * `path` - Output path (None or "-" for stdout, only valid for Tsv format)
    /// * `parquet_batch_size` - Batch size for Parquet buffering (ignored for TSV formats)
    pub fn new(
        format: OutputFormat,
        path: Option<&PathBuf>,
        parquet_batch_size: Option<usize>,
    ) -> Result<Self> {
        // Check for stdout
        let is_stdout = OutputFormat::is_stdout(path);

        match format {
            OutputFormat::Tsv => {
                let output: Box<dyn Write + Send> = if is_stdout {
                    Box::new(io::stdout())
                } else {
                    let p = path.unwrap(); // Safe: not stdout means path exists
                    Box::new(
                        File::create(p)
                            .with_context(|| format!("Failed to create output file: {:?}", p))?,
                    )
                };
                Ok(OutputWriter::Tsv(BufWriter::new(output)))
            }
            OutputFormat::TsvGz => {
                if is_stdout {
                    return Err(anyhow!("Gzip output requires a file path, not stdout"));
                }
                let path = path.unwrap(); // Safe: not stdout
                let file = File::create(path)
                    .with_context(|| format!("Failed to create output file: {:?}", path))?;
                let encoder = GzEncoder::new(file, Compression::default());
                Ok(OutputWriter::TsvGz(Some(BufWriter::new(encoder))))
            }
            OutputFormat::Parquet => {
                if is_stdout {
                    return Err(anyhow!("Parquet output requires a file path, not stdout"));
                }
                let path = path.unwrap(); // Safe: not stdout
                let file = File::create(path)
                    .with_context(|| format!("Failed to create output file: {:?}", path))?;

                // Define schema: (read_id: Utf8, bucket_name: Utf8, score: Float64)
                let schema = Arc::new(Schema::new(vec![
                    Field::new("read_id", DataType::Utf8, false),
                    Field::new("bucket_name", DataType::Utf8, false),
                    Field::new("score", DataType::Float64, false),
                ]));

                // Use zstd compression and dictionary encoding for bucket_name (categorical)
                let bucket_name_col = ColumnPath::new(vec!["bucket_name".to_string()]);
                let props = WriterProperties::builder()
                    .set_compression(parquet::basic::Compression::ZSTD(ZstdLevel::default()))
                    .set_column_dictionary_enabled(bucket_name_col, true)
                    .build();

                let writer = ArrowWriter::try_new(file, schema, Some(props))
                    .context("Failed to create Parquet writer")?;

                let batch_size = parquet_batch_size.unwrap_or(DEFAULT_PARQUET_OUTPUT_BATCH_SIZE);

                Ok(OutputWriter::Parquet {
                    writer,
                    read_ids: Vec::with_capacity(batch_size),
                    bucket_names: Vec::with_capacity(batch_size),
                    scores: Vec::with_capacity(batch_size),
                    batch_size,
                })
            }
        }
    }

    /// Create a new wide-format output writer.
    ///
    /// Wide format outputs one row per read with scores for all buckets as columns.
    /// For TSV/TsvGz formats, this is equivalent to `new()` - the caller is responsible
    /// for formatting wide-format data. For Parquet, creates a dynamic schema with
    /// one Float64 column per bucket.
    ///
    /// # Arguments
    /// * `format` - Output format (Tsv, TsvGz, or Parquet)
    /// * `path` - Output path (None or "-" for stdout, only valid for Tsv format)
    /// * `bucket_names` - Map of bucket_id to bucket_name, used to build Parquet schema
    /// * `parquet_batch_size` - Batch size for Parquet buffering (ignored for TSV formats)
    pub fn new_wide(
        format: OutputFormat,
        path: Option<&PathBuf>,
        bucket_names: &HashMap<u32, String>,
        parquet_batch_size: Option<usize>,
    ) -> Result<Self> {
        // For TSV formats, wide output is just pre-formatted data - use normal writers
        match format {
            OutputFormat::Tsv | OutputFormat::TsvGz => {
                // TSV/TsvGz: just pass through - caller formats wide data
                Self::new(format, path, parquet_batch_size)
            }
            OutputFormat::Parquet => {
                let is_stdout = OutputFormat::is_stdout(path);
                if is_stdout {
                    return Err(anyhow!("Parquet output requires a file path, not stdout"));
                }
                let path = path.unwrap(); // Safe: not stdout
                let file = File::create(path)
                    .with_context(|| format!("Failed to create output file: {:?}", path))?;

                // Sort bucket IDs and build dynamic schema
                let mut bucket_ids: Vec<u32> = bucket_names.keys().copied().collect();
                bucket_ids.sort_unstable();

                // Build schema: read_id + one Float64 column per bucket (ordered by bucket_id)
                let mut fields = vec![Field::new("read_id", DataType::Utf8, false)];
                for bucket_id in &bucket_ids {
                    let name = bucket_names
                        .get(bucket_id)
                        .map(|s| s.as_str())
                        .unwrap_or("unknown");
                    fields.push(Field::new(name, DataType::Float64, false));
                }
                let schema = Arc::new(Schema::new(fields));

                // Use zstd compression
                let props = WriterProperties::builder()
                    .set_compression(parquet::basic::Compression::ZSTD(ZstdLevel::default()))
                    .build();

                let writer = ArrowWriter::try_new(file, schema, Some(props))
                    .context("Failed to create Parquet writer")?;

                let batch_size = parquet_batch_size.unwrap_or(DEFAULT_PARQUET_OUTPUT_BATCH_SIZE);
                let num_buckets = bucket_ids.len();

                // Build sorted bucket column names
                let bucket_col_names: Vec<String> = bucket_ids
                    .iter()
                    .map(|id| {
                        bucket_names
                            .get(id)
                            .cloned()
                            .unwrap_or_else(|| format!("bucket_{}", id))
                    })
                    .collect();

                Ok(OutputWriter::ParquetWide {
                    writer,
                    bucket_col_names,
                    read_ids: Vec::with_capacity(batch_size),
                    scores: Vec::with_capacity(batch_size * num_buckets),
                    batch_size,
                })
            }
        }
    }

    /// Write TSV header (no-op for Parquet).
    pub fn write_header(&mut self, header: &[u8]) -> Result<()> {
        match self {
            OutputWriter::Tsv(w) => {
                w.write_all(header)?;
                Ok(())
            }
            OutputWriter::TsvGz(Some(w)) => {
                w.write_all(header)?;
                Ok(())
            }
            OutputWriter::TsvGz(None) => Err(anyhow!("Writer already finished")),
            OutputWriter::Parquet { .. } | OutputWriter::ParquetWide { .. } => {
                // No header needed for Parquet - schema is embedded
                Ok(())
            }
        }
    }

    /// Write a single record directly (avoids TSV serialization for Parquet).
    ///
    /// This is the preferred method for Parquet output as it avoids the
    /// overhead of serializing to TSV and then parsing back.
    ///
    /// Note: This method is for long-format output only. For wide-format Parquet,
    /// use `write_chunk()` with pre-formatted wide data.
    #[allow(dead_code)]
    pub fn write_record(&mut self, read_id: &str, bucket_name: &str, score: f64) -> Result<()> {
        match self {
            OutputWriter::Tsv(w) => {
                writeln!(w, "{}\t{}\t{:.4}", read_id, bucket_name, score)?;
                Ok(())
            }
            OutputWriter::TsvGz(Some(w)) => {
                writeln!(w, "{}\t{}\t{:.4}", read_id, bucket_name, score)?;
                Ok(())
            }
            OutputWriter::TsvGz(None) => Err(anyhow!("Writer already finished")),
            OutputWriter::Parquet {
                writer,
                read_ids,
                bucket_names,
                scores,
                batch_size,
            } => {
                read_ids.push(read_id.to_string());
                bucket_names.push(bucket_name.to_string());
                scores.push(score);

                // Flush if buffer is full
                if read_ids.len() >= *batch_size {
                    Self::flush_parquet_batch(writer, read_ids, bucket_names, scores)?;
                }
                Ok(())
            }
            OutputWriter::ParquetWide { .. } => Err(anyhow!(
                "write_record() is not supported for wide-format Parquet; use write_chunk()"
            )),
        }
    }

    /// Write a chunk of data to the output.
    ///
    /// For TSV and TsvGz formats, writes the raw bytes directly.
    /// For Parquet output, parses the data as TSV lines and buffers records.
    /// Prefer `write_record()` for Parquet output to avoid serialization overhead.
    pub fn write_chunk(&mut self, data: Vec<u8>) -> Result<()> {
        match self {
            OutputWriter::Tsv(w) => {
                w.write_all(&data)?;
                Ok(())
            }
            OutputWriter::TsvGz(Some(w)) => {
                w.write_all(&data)?;
                Ok(())
            }
            OutputWriter::TsvGz(None) => Err(anyhow!("Writer already finished")),
            OutputWriter::Parquet {
                writer,
                read_ids,
                bucket_names,
                scores,
                batch_size,
            } => {
                // Parse TSV lines and buffer for Parquet
                let text = String::from_utf8_lossy(&data);
                for line in text.lines() {
                    if line.is_empty() {
                        continue;
                    }
                    let parts: Vec<&str> = line.split('\t').collect();
                    // Validate exact column count (read_id, bucket_name, score)
                    if parts.len() != 3 {
                        return Err(anyhow!(
                            "Long format line has {} columns, expected 3 (read_id, bucket_name, score). \
                             Line starts with: '{}'",
                            parts.len(),
                            parts.first().unwrap_or(&"<empty>")
                        ));
                    }
                    read_ids.push(parts[0].to_string());
                    bucket_names.push(parts[1].to_string());
                    let score: f64 = parts[2].parse().with_context(|| {
                        format!("Invalid score value '{}' for read '{}'", parts[2], parts[0])
                    })?;
                    scores.push(score);

                    // Flush if buffer is full
                    if read_ids.len() >= *batch_size {
                        Self::flush_parquet_batch(writer, read_ids, bucket_names, scores)?;
                    }
                }
                Ok(())
            }
            OutputWriter::ParquetWide {
                writer,
                bucket_col_names,
                read_ids,
                scores,
                batch_size,
            } => {
                // Parse wide-format TSV: "read_id\tscore1\tscore2\t...\n"
                let text = String::from_utf8_lossy(&data);
                let num_buckets = bucket_col_names.len();
                let expected_cols = 1 + num_buckets;
                for line in text.lines() {
                    if line.is_empty() {
                        continue;
                    }
                    let parts: Vec<&str> = line.split('\t').collect();
                    // Validate exact column count
                    if parts.len() != expected_cols {
                        return Err(anyhow!(
                            "Wide format line has {} columns, expected {} (read_id + {} buckets). \
                             Line starts with: '{}'",
                            parts.len(),
                            expected_cols,
                            num_buckets,
                            parts.first().unwrap_or(&"<empty>")
                        ));
                    }
                    read_ids.push(parts[0].to_string());
                    for (i, part) in parts.iter().skip(1).enumerate() {
                        let score: f64 = part.parse().with_context(|| {
                            format!(
                                "Invalid score value '{}' at column {} for read '{}'",
                                part,
                                i + 1,
                                parts[0]
                            )
                        })?;
                        scores.push(score);
                    }

                    // Flush if buffer is full
                    if read_ids.len() >= *batch_size {
                        Self::flush_parquet_wide_batch(writer, bucket_col_names, read_ids, scores)?;
                    }
                }
                Ok(())
            }
        }
    }

    /// Flush Parquet batch to writer.
    fn flush_parquet_batch(
        writer: &mut ArrowWriter<File>,
        read_ids: &mut Vec<String>,
        bucket_names: &mut Vec<String>,
        scores: &mut Vec<f64>,
    ) -> Result<()> {
        if read_ids.is_empty() {
            return Ok(());
        }

        // Build arrays from the data (don't drain until we're sure the write succeeds)
        let mut read_id_builder = StringBuilder::with_capacity(read_ids.len(), read_ids.len() * 32);
        let mut bucket_name_builder =
            StringBuilder::with_capacity(bucket_names.len(), bucket_names.len() * 64);
        let mut score_builder = Float64Builder::with_capacity(scores.len());

        for read_id in read_ids.iter() {
            read_id_builder.append_value(read_id);
        }
        for bucket_name in bucket_names.iter() {
            bucket_name_builder.append_value(bucket_name);
        }
        for &score in scores.iter() {
            score_builder.append_value(score);
        }

        let batch = RecordBatch::try_from_iter(vec![
            ("read_id", Arc::new(read_id_builder.finish()) as ArrayRef),
            (
                "bucket_name",
                Arc::new(bucket_name_builder.finish()) as ArrayRef,
            ),
            ("score", Arc::new(score_builder.finish()) as ArrayRef),
        ])
        .context("Failed to create RecordBatch")?;

        writer
            .write(&batch)
            .context("Failed to write Parquet batch")?;

        // Only clear after successful write
        read_ids.clear();
        bucket_names.clear();
        scores.clear();

        Ok(())
    }

    /// Flush wide-format Parquet batch to writer.
    ///
    /// Builds a RecordBatch with read_id column + one Float64 column per bucket.
    fn flush_parquet_wide_batch(
        writer: &mut ArrowWriter<File>,
        bucket_col_names: &[String],
        read_ids: &mut Vec<String>,
        scores: &mut Vec<f64>,
    ) -> Result<()> {
        if read_ids.is_empty() {
            return Ok(());
        }

        let num_rows = read_ids.len();
        let num_buckets = bucket_col_names.len();

        // Build read_id array
        let mut read_id_builder = StringBuilder::with_capacity(num_rows, num_rows * 32);
        for read_id in read_ids.iter() {
            read_id_builder.append_value(read_id);
        }

        // Build columns vector with named columns for try_from_iter
        let mut columns: Vec<(&str, ArrayRef)> =
            vec![("read_id", Arc::new(read_id_builder.finish()) as ArrayRef)];

        // Build score columns - one Float64Array per bucket
        // scores is stored as: scores[row * num_buckets + col]
        for (bucket_col, col_name) in bucket_col_names.iter().enumerate() {
            let mut score_builder = Float64Builder::with_capacity(num_rows);
            for row in 0..num_rows {
                let idx = row * num_buckets + bucket_col;
                score_builder.append_value(scores[idx]);
            }
            columns.push((
                col_name.as_str(),
                Arc::new(score_builder.finish()) as ArrayRef,
            ));
        }

        // Build RecordBatch with named columns
        let batch =
            RecordBatch::try_from_iter(columns).context("Failed to create wide RecordBatch")?;

        writer
            .write(&batch)
            .context("Failed to write Parquet wide batch")?;

        // Only clear after successful write
        read_ids.clear();
        scores.clear();

        Ok(())
    }

    /// Finish writing and close the output.
    ///
    /// This MUST be called to properly finalize gzip and Parquet files.
    /// For gzip, this writes the gzip trailer. For Parquet, this writes
    /// any remaining buffered records and the file footer.
    pub fn finish(&mut self) -> Result<()> {
        match self {
            OutputWriter::Tsv(w) => {
                w.flush()?;
                Ok(())
            }
            OutputWriter::TsvGz(opt_writer) => {
                // Take ownership of the writer to properly finish the gzip stream
                let writer = opt_writer
                    .take()
                    .ok_or_else(|| anyhow!("Writer already finished"))?;

                // Flush the BufWriter and get the GzEncoder
                let encoder = writer
                    .into_inner()
                    .map_err(|e| anyhow!("Failed to flush buffer: {}", e))?;

                // Finish the gzip stream - this writes the gzip trailer
                encoder.finish().context("Failed to finish gzip stream")?;

                Ok(())
            }
            OutputWriter::Parquet {
                writer,
                read_ids,
                bucket_names,
                scores,
                ..
            } => {
                // Flush remaining buffered records
                Self::flush_parquet_batch(writer, read_ids, bucket_names, scores)?;
                writer.finish().context("Failed to finish Parquet file")?;
                Ok(())
            }
            OutputWriter::ParquetWide {
                writer,
                bucket_col_names,
                read_ids,
                scores,
                ..
            } => {
                // Flush remaining buffered records
                Self::flush_parquet_wide_batch(writer, bucket_col_names, read_ids, scores)?;
                writer.finish().context("Failed to finish Parquet file")?;
                Ok(())
            }
        }
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{RecordBatchReader as _, StringArray};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::io::Read as _;
    use tempfile::NamedTempFile;

    // -------------------------------------------------------------------------
    // Tests for OutputFormat::detect
    // -------------------------------------------------------------------------

    #[test]
    fn test_output_format_detect_none() {
        assert_eq!(OutputFormat::detect(None), OutputFormat::Tsv);
    }

    #[test]
    fn test_output_format_detect_stdout() {
        let path = PathBuf::from("-");
        assert_eq!(OutputFormat::detect(Some(&path)), OutputFormat::Tsv);
    }

    #[test]
    fn test_output_format_detect_tsv() {
        let path = PathBuf::from("output.tsv");
        assert_eq!(OutputFormat::detect(Some(&path)), OutputFormat::Tsv);
    }

    #[test]
    fn test_output_format_detect_tsv_gz() {
        let path = PathBuf::from("output.tsv.gz");
        assert_eq!(OutputFormat::detect(Some(&path)), OutputFormat::TsvGz);
    }

    #[test]
    fn test_output_format_detect_gz() {
        let path = PathBuf::from("output.gz");
        assert_eq!(OutputFormat::detect(Some(&path)), OutputFormat::TsvGz);
    }

    #[test]
    fn test_output_format_detect_parquet() {
        let path = PathBuf::from("output.parquet");
        assert_eq!(OutputFormat::detect(Some(&path)), OutputFormat::Parquet);
    }

    #[test]
    fn test_output_format_detect_no_extension() {
        let path = PathBuf::from("output");
        assert_eq!(OutputFormat::detect(Some(&path)), OutputFormat::Tsv);
    }

    #[test]
    fn test_output_format_detect_unknown_extension() {
        let path = PathBuf::from("output.csv");
        assert_eq!(OutputFormat::detect(Some(&path)), OutputFormat::Tsv);
    }

    #[test]
    fn test_output_format_is_stdout() {
        assert!(OutputFormat::is_stdout(None));
        assert!(OutputFormat::is_stdout(Some(&PathBuf::from("-"))));
        assert!(!OutputFormat::is_stdout(Some(&PathBuf::from("file.tsv"))));
    }

    // -------------------------------------------------------------------------
    // Tests for OutputWriter
    // -------------------------------------------------------------------------

    #[test]
    fn test_output_writer_tsv() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();

        let mut writer = OutputWriter::new(OutputFormat::Tsv, Some(&path), None).unwrap();
        writer.write_header(b"col1\tcol2\tcol3\n").unwrap();
        writer.write_record("read1", "bucket1", 0.95).unwrap();
        writer.write_record("read2", "bucket2", 0.85).unwrap();
        writer.finish().unwrap();

        let mut content = String::new();
        std::fs::File::open(&path)
            .unwrap()
            .read_to_string(&mut content)
            .unwrap();

        assert!(content.contains("col1\tcol2\tcol3"));
        assert!(content.contains("read1\tbucket1\t0.9500"));
        assert!(content.contains("read2\tbucket2\t0.8500"));
    }

    #[test]
    fn test_output_writer_gzip() {
        let tmp = NamedTempFile::with_suffix(".gz").unwrap();
        let path = tmp.path().to_path_buf();

        let mut writer = OutputWriter::new(OutputFormat::TsvGz, Some(&path), None).unwrap();
        writer.write_header(b"col1\tcol2\tcol3\n").unwrap();
        writer.write_record("read1", "bucket1", 0.95).unwrap();
        writer.finish().unwrap();

        // Verify it's a valid gzip file by decompressing
        let file = std::fs::File::open(&path).unwrap();
        let mut decoder = flate2::read::GzDecoder::new(file);
        let mut content = String::new();
        decoder.read_to_string(&mut content).unwrap();

        assert!(content.contains("col1\tcol2\tcol3"));
        assert!(content.contains("read1\tbucket1\t0.9500"));
    }

    #[test]
    fn test_output_writer_gzip_requires_path() {
        let result = OutputWriter::new(OutputFormat::TsvGz, None, None);
        assert!(result.is_err());
        let err_msg = result.err().unwrap().to_string();
        assert!(err_msg.contains("file path"), "Error was: {}", err_msg);
    }

    #[test]
    fn test_output_writer_parquet_requires_path() {
        let result = OutputWriter::new(OutputFormat::Parquet, None, None);
        assert!(result.is_err());
        let err_msg = result.err().unwrap().to_string();
        assert!(err_msg.contains("file path"), "Error was: {}", err_msg);
    }

    // -------------------------------------------------------------------------
    // Tests for OutputWriter wide format
    // -------------------------------------------------------------------------

    #[test]
    fn test_output_writer_new_wide_creates_parquet_with_dynamic_schema() {
        let tmp = NamedTempFile::with_suffix(".parquet").unwrap();
        let path = tmp.path().to_path_buf();

        let mut bucket_names = HashMap::new();
        bucket_names.insert(1, "Bucket_A".to_string());
        bucket_names.insert(2, "Bucket_B".to_string());
        bucket_names.insert(3, "Bucket_C".to_string());

        let mut writer =
            OutputWriter::new_wide(OutputFormat::Parquet, Some(&path), &bucket_names, None)
                .unwrap();

        // Write some wide-format data
        let wide_data = b"read_1\t0.8500\t0.7500\t0.6500\nread_2\t0.9000\t0.0000\t0.3000\n";
        writer.write_chunk(wide_data.to_vec()).unwrap();
        writer.finish().unwrap();

        // Verify file was created and is valid Parquet
        let file = File::open(&path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();

        // Verify schema has expected columns
        let schema = reader.schema();
        assert_eq!(schema.fields().len(), 4); // read_id + 3 buckets
        assert_eq!(schema.field(0).name(), "read_id");
        assert_eq!(schema.field(1).name(), "Bucket_A");
        assert_eq!(schema.field(2).name(), "Bucket_B");
        assert_eq!(schema.field(3).name(), "Bucket_C");
    }

    #[test]
    fn test_output_writer_new_wide_parquet_schema_has_correct_types() {
        let tmp = NamedTempFile::with_suffix(".parquet").unwrap();
        let path = tmp.path().to_path_buf();

        let mut bucket_names = HashMap::new();
        bucket_names.insert(1, "Score1".to_string());
        bucket_names.insert(2, "Score2".to_string());

        let mut writer =
            OutputWriter::new_wide(OutputFormat::Parquet, Some(&path), &bucket_names, None)
                .unwrap();
        writer.finish().unwrap();

        // Verify types
        let file = File::open(&path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();

        let schema = reader.schema();
        assert_eq!(*schema.field(0).data_type(), DataType::Utf8); // read_id
        assert_eq!(*schema.field(1).data_type(), DataType::Float64); // Score1
        assert_eq!(*schema.field(2).data_type(), DataType::Float64); // Score2
    }

    #[test]
    fn test_output_writer_new_wide_parquet_columns_ordered_by_bucket_id() {
        let tmp = NamedTempFile::with_suffix(".parquet").unwrap();
        let path = tmp.path().to_path_buf();

        let mut bucket_names = HashMap::new();
        // Insert in non-sorted order
        bucket_names.insert(10, "Z_last".to_string());
        bucket_names.insert(1, "A_first".to_string());
        bucket_names.insert(5, "M_middle".to_string());

        let mut writer =
            OutputWriter::new_wide(OutputFormat::Parquet, Some(&path), &bucket_names, None)
                .unwrap();
        writer.finish().unwrap();

        // Verify column order follows bucket_id (1, 5, 10), not alphabetical
        let file = File::open(&path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();

        let schema = reader.schema();
        assert_eq!(schema.field(0).name(), "read_id");
        assert_eq!(schema.field(1).name(), "A_first"); // bucket_id 1
        assert_eq!(schema.field(2).name(), "M_middle"); // bucket_id 5
        assert_eq!(schema.field(3).name(), "Z_last"); // bucket_id 10
    }

    #[test]
    fn test_output_writer_new_wide_parquet_write_chunk_and_read_back() {
        let tmp = NamedTempFile::with_suffix(".parquet").unwrap();
        let path = tmp.path().to_path_buf();

        let mut bucket_names = HashMap::new();
        bucket_names.insert(1, "Bucket_A".to_string());
        bucket_names.insert(2, "Bucket_B".to_string());

        let mut writer =
            OutputWriter::new_wide(OutputFormat::Parquet, Some(&path), &bucket_names, None)
                .unwrap();

        // Write wide-format TSV data (as produced by format_results_wide)
        let wide_data = b"read_1\t0.8500\t0.7500\nread_2\t0.9100\t0.0000\n";
        writer.write_chunk(wide_data.to_vec()).unwrap();
        writer.finish().unwrap();

        // Read back and verify data
        let file = File::open(&path).unwrap();
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();

        let batch = reader.next().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 2);

        // Verify read_ids
        let read_ids = batch
            .column_by_name("read_id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(read_ids.value(0), "read_1");
        assert_eq!(read_ids.value(1), "read_2");

        // Verify scores
        let scores_a = batch
            .column_by_name("Bucket_A")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap();
        assert!((scores_a.value(0) - 0.85).abs() < 0.0001);
        assert!((scores_a.value(1) - 0.91).abs() < 0.0001);

        let scores_b = batch
            .column_by_name("Bucket_B")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap();
        assert!((scores_b.value(0) - 0.75).abs() < 0.0001);
        assert!((scores_b.value(1) - 0.0).abs() < 0.0001);
    }

    #[test]
    fn test_output_writer_new_wide_tsv_passthrough() {
        // For TSV format, new_wide should work the same as new (just pass through bytes)
        let tmp = NamedTempFile::with_suffix(".tsv").unwrap();
        let path = tmp.path().to_path_buf();

        let mut bucket_names = HashMap::new();
        bucket_names.insert(1, "Bucket_A".to_string());
        bucket_names.insert(2, "Bucket_B".to_string());

        let mut writer =
            OutputWriter::new_wide(OutputFormat::Tsv, Some(&path), &bucket_names, None).unwrap();

        writer
            .write_header(b"read_id\tBucket_A\tBucket_B\n")
            .unwrap();
        writer
            .write_chunk(b"read_1\t0.8500\t0.7500\n".to_vec())
            .unwrap();
        writer.finish().unwrap();

        let mut content = String::new();
        std::fs::File::open(&path)
            .unwrap()
            .read_to_string(&mut content)
            .unwrap();

        assert!(content.contains("read_id\tBucket_A\tBucket_B"));
        assert!(content.contains("read_1\t0.8500\t0.7500"));
    }

    #[test]
    fn test_output_writer_new_wide_parquet_requires_path() {
        let bucket_names = HashMap::new();
        let result = OutputWriter::new_wide(OutputFormat::Parquet, None, &bucket_names, None);
        assert!(result.is_err());
    }
}
