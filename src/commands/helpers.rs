//! Helper functions and utilities for the rype CLI.

use anyhow::{anyhow, Context, Result};
use arrow::array::{ArrayRef, Float64Builder, RecordBatch, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use flate2::write::GzEncoder;
use flate2::Compression;
use needletail::{parse_fastx_file, FastxReader};
use parquet::arrow::ArrowWriter;
use parquet::basic::ZstdLevel;
use parquet::file::properties::WriterProperties;
use parquet::schema::types::ColumnPath;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use rype::memory::parse_byte_suffix;
use rype::{FirstErrorCapture, IndexMetadata};

/// Owned record type: (query_id, seq1, optional_seq2)
pub type OwnedRecord = (i64, Vec<u8>, Option<Vec<u8>>);

/// Parse a byte size string from CLI (e.g., "4G", "512M", "auto").
/// Returns 0 for "auto" (signals auto-detection), bytes otherwise.
pub fn parse_max_memory_arg(s: &str) -> Result<usize, String> {
    parse_byte_suffix(s)
        .map(|opt| opt.unwrap_or(0)) // None (auto) -> 0
        .map_err(|e| e.to_string())
}

/// Parse a byte size string from CLI, requiring a concrete value (no "auto").
pub fn parse_shard_size_arg(s: &str) -> Result<usize, String> {
    match parse_byte_suffix(s) {
        Ok(Some(bytes)) => Ok(bytes),
        Ok(None) => Err("'auto' not supported for shard size".to_string()),
        Err(e) => Err(e.to_string()),
    }
}

/// Parse bloom filter false positive probability, validating range (0.0, 1.0).
pub fn parse_bloom_fpp(s: &str) -> Result<f64, String> {
    let fpp: f64 = s
        .parse()
        .map_err(|_| format!("'{}' is not a valid number", s))?;
    if fpp <= 0.0 || fpp >= 1.0 {
        return Err(format!(
            "bloom_filter_fpp must be in (0.0, 1.0), got {}",
            fpp
        ));
    }
    Ok(fpp)
}

/// Validate the --trim-to argument.
///
/// - Must be a positive integer greater than 0
/// - Values smaller than typical k-mer sizes (16) will produce a warning but are allowed
pub fn validate_trim_to(s: &str) -> Result<usize, String> {
    let val: usize = s
        .parse()
        .map_err(|_| format!("'{}' is not a valid positive integer", s))?;
    if val == 0 {
        return Err("trim_to must be greater than 0".to_string());
    }
    // Warn about very small values that won't produce useful results
    // (minimum k-mer size is 16, so anything less won't generate minimizers)
    if val < 16 {
        eprintln!(
            "Warning: --trim-to {} is smaller than the minimum k-mer size (16). \
             This will likely produce no classification results.",
            val
        );
    }
    Ok(val)
}

/// Sanitize bucket names by replacing nonprintable characters with "_"
pub fn sanitize_bucket_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_control() || !c.is_ascii_graphic() && !c.is_whitespace() {
                '_'
            } else {
                c
            }
        })
        .collect()
}

/// Resolve a bucket identifier (either numeric ID or name) to a bucket ID.
///
/// This function allows users to specify buckets by either:
/// - Numeric ID (e.g., "1", "42") - takes precedence if input parses as u32
/// - Bucket name (e.g., "Bacteria", "my_bucket") - case-sensitive exact match
///
/// # Numeric Priority
///
/// If the identifier parses as a valid u32, it is treated as a numeric ID.
/// This means bucket names that are valid numbers (e.g., a bucket named "42")
/// cannot be referenced by name - use the numeric ID instead.
///
/// # Arguments
/// * `identifier` - String that is either a numeric ID or a bucket name
/// * `bucket_names` - Map of bucket ID to bucket name from the index
///
/// # Returns
/// The resolved bucket ID. Note: This function does NOT validate that the
/// bucket ID exists in the index - the caller should verify existence.
///
/// # Errors
/// Returns an error if:
/// - The identifier is empty
/// - The identifier is not a valid number AND not found as a bucket name
/// - Multiple buckets have the same name (ambiguous)
pub fn resolve_bucket_id(identifier: &str, bucket_names: &HashMap<u32, String>) -> Result<u32> {
    // Validate non-empty input
    let identifier = identifier.trim();
    if identifier.is_empty() {
        return Err(anyhow!("Bucket identifier cannot be empty"));
    }

    // First, try to parse as a numeric ID (takes precedence over name lookup)
    if let Ok(id) = identifier.parse::<u32>() {
        return Ok(id);
    }

    // Not a number, search by name (case-sensitive exact match)
    let matches: Vec<u32> = bucket_names
        .iter()
        .filter(|(_, name)| name.as_str() == identifier)
        .map(|(id, _)| *id)
        .collect();

    match matches.len() {
        0 => Err(anyhow!("Bucket '{}' not found in index", identifier)),
        1 => Ok(matches[0]),
        _ => Err(anyhow!(
            "Ambiguous bucket name '{}': matches {} buckets (IDs: {:?}). Use numeric ID instead.",
            identifier,
            matches.len(),
            matches
        )),
    }
}

/// Load metadata from a Parquet inverted index.
///
/// This helper handles Parquet inverted index directories (with manifest.toml).
pub fn load_index_metadata(path: &Path) -> Result<IndexMetadata> {
    // Parquet format (directory with manifest.toml)
    if rype::is_parquet_index(path) {
        let manifest = rype::ParquetManifest::load(path)?;
        let (bucket_names, bucket_sources) = rype::parquet_index::read_buckets_parquet(path)?;
        return Ok(IndexMetadata {
            k: manifest.k,
            w: manifest.w,
            salt: manifest.salt,
            bucket_names,
            bucket_sources,
            bucket_minimizer_counts: HashMap::new(),
        });
    }

    Err(anyhow!(
        "Invalid index format: expected Parquet index directory (.ryxdi/) at {:?}",
        path
    ))
}

/// Type alias for batch data sent through the prefetch channel.
type BatchResult = Result<Option<(Vec<OwnedRecord>, Vec<String>)>>;

/// Default timeout for waiting on prefetch batches (5 minutes).
const DEFAULT_PREFETCH_TIMEOUT: Duration = Duration::from_secs(300);

/// I/O handler with background prefetching for overlapping I/O and computation.
///
/// This handler spawns a background thread that reads and decompresses the next
/// batch while the main thread processes the current batch. This is especially
/// effective for gzipped input where decompression can be overlapped with
/// minimizer extraction and classification.
///
/// # Usage
/// ```ignore
/// let mut io = PrefetchingIoHandler::new(&r1, r2.as_ref(), output, batch_size)?;
/// while let Some((records, headers)) = io.next_batch()? {
///     // Process batch...
/// }
/// io.finish()?;
/// ```
pub struct PrefetchingIoHandler {
    receiver: Receiver<BatchResult>,
    prefetch_thread: Option<JoinHandle<()>>,
    writer: OutputWriter,
    /// Thread-safe error capture for errors that occur when sender.send() fails.
    /// Only captures the first error to avoid race conditions.
    error_capture: Arc<FirstErrorCapture>,
    /// Timeout for waiting on batches
    timeout: Duration,
}

impl PrefetchingIoHandler {
    /// Create a new prefetching I/O handler.
    ///
    /// # Arguments
    /// * `r1_path` - Path to the first read file (FASTQ/FASTA, optionally gzipped)
    /// * `r2_path` - Optional path to the second read file for paired-end
    /// * `out_path` - Optional output path (stdout if None). Format auto-detected from extension:
    ///   - `.tsv` or no extension: Plain TSV
    ///   - `.tsv.gz`: Gzip-compressed TSV
    ///   - `.parquet`: Apache Parquet with zstd compression
    ///   - `-`: stdout (TSV)
    /// * `batch_size` - Number of records per batch
    ///
    /// # Returns
    /// A handler that prefetches batches in a background thread.
    pub fn new(
        r1_path: &Path,
        r2_path: Option<&PathBuf>,
        out_path: Option<PathBuf>,
        batch_size: usize,
    ) -> Result<Self> {
        // Clone paths for the background thread
        let r1_path = r1_path.to_path_buf();
        let r2_path = r2_path.cloned();

        // Thread-safe error capture (only stores first error)
        let error_capture = Arc::new(FirstErrorCapture::new());
        let thread_error = Arc::clone(&error_capture);

        // Use sync_channel with buffer of 2 to allow actual prefetching:
        // - Slot 1: batch currently being processed by main thread
        // - Slot 2: next batch being prefetched by reader thread
        let (sender, receiver): (SyncSender<BatchResult>, Receiver<BatchResult>) =
            mpsc::sync_channel(2);

        // Spawn background thread for reading
        let prefetch_thread = thread::spawn(move || {
            Self::reader_thread(r1_path, r2_path, batch_size, sender, thread_error);
        });

        // Set up output writer with auto-detected format
        let output_format = OutputFormat::detect(out_path.as_ref());
        let writer = OutputWriter::new(output_format, out_path.as_ref(), None)?;

        Ok(Self {
            receiver,
            prefetch_thread: Some(prefetch_thread),
            writer,
            error_capture,
            timeout: DEFAULT_PREFETCH_TIMEOUT,
        })
    }

    /// Extract the base read ID (before any space, tab, or /1 /2 suffix).
    ///
    /// This handles common FASTQ header formats:
    /// - `@READ1 comment` → `READ1`
    /// - `@READ1/1` → `READ1`
    /// - `@READ1/1 comment` → `READ1`
    /// - `@READ1\tcomment` → `READ1`
    ///
    /// Note: Only `/1` and `/2` suffixes at the end of the ID portion (before space/tab)
    /// are stripped. The function assumes ASCII input (standard for FASTQ).
    pub fn base_read_id(id: &[u8]) -> &[u8] {
        // Find first space or tab - everything after is a comment
        let id_end = id
            .iter()
            .position(|&b| b == b' ' || b == b'\t')
            .unwrap_or(id.len());

        let id_portion = &id[..id_end];

        // Check for /1 or /2 suffix at the end of the ID portion
        if id_portion.len() >= 2 {
            let len = id_portion.len();
            if id_portion[len - 2] == b'/'
                && (id_portion[len - 1] == b'1' || id_portion[len - 1] == b'2')
            {
                return &id_portion[..len - 2];
            }
        }

        id_portion
    }

    /// Background thread function that reads batches and sends them through the channel.
    fn reader_thread(
        r1_path: PathBuf,
        r2_path: Option<PathBuf>,
        batch_size: usize,
        sender: SyncSender<BatchResult>,
        error_capture: Arc<FirstErrorCapture>,
    ) {
        // Helper macro to send error and store if send fails
        macro_rules! send_error {
            ($msg:expr) => {{
                let err_msg = $msg;
                if sender.send(Err(anyhow!("{}", &err_msg))).is_err() {
                    error_capture.store_msg(&err_msg);
                }
                return;
            }};
        }

        // Open readers in the background thread
        let mut r1 = match parse_fastx_file(&r1_path) {
            Ok(r) => r,
            Err(e) => {
                send_error!(format!("Failed to open R1 at {}: {}", r1_path.display(), e));
            }
        };

        let mut r2: Option<Box<dyn FastxReader>> = match r2_path {
            Some(ref p) => match parse_fastx_file(p) {
                Ok(r) => Some(r),
                Err(e) => {
                    send_error!(format!("Failed to open R2 at {}: {}", p.display(), e));
                }
            },
            None => None,
        };

        let mut global_record_id: i64 = 0; // For error messages only

        loop {
            let mut records = Vec::with_capacity(batch_size);
            let mut headers = Vec::with_capacity(batch_size);

            for batch_idx in 0..batch_size {
                let s1_rec = match r1.next() {
                    Some(Ok(rec)) => rec,
                    Some(Err(e)) => {
                        send_error!(format!(
                            "Error reading R1 at record {}: {}",
                            global_record_id, e
                        ));
                    }
                    None => break, // End of file
                };

                let s2_vec = if let Some(ref mut r2_reader) = r2 {
                    match r2_reader.next() {
                        Some(Ok(rec)) => {
                            // Validate that read IDs match
                            let r1_base = Self::base_read_id(s1_rec.id());
                            let r2_base = Self::base_read_id(rec.id());
                            if r1_base != r2_base {
                                send_error!(format!(
                                    "R1/R2 read ID mismatch at record {}: R1='{}' R2='{}'",
                                    global_record_id,
                                    String::from_utf8_lossy(r1_base),
                                    String::from_utf8_lossy(r2_base)
                                ));
                            }
                            Some(rec.seq().into_owned())
                        }
                        Some(Err(e)) => {
                            send_error!(format!(
                                "Error reading R2 at record {}: {}",
                                global_record_id, e
                            ));
                        }
                        None => {
                            send_error!(format!(
                                "R1/R2 mismatch: R2 ended early at record {}",
                                global_record_id
                            ));
                        }
                    }
                } else {
                    None
                };

                let base_id = Self::base_read_id(s1_rec.id());
                let header = String::from_utf8_lossy(base_id).to_string();
                // Use batch-local index (0-based within each batch) for query_id.
                // This is intentional - the classification code maps results back
                // to headers using this batch-local index.
                records.push((batch_idx as i64, s1_rec.seq().into_owned(), s2_vec));
                headers.push(header);
                global_record_id += 1;
            }

            if records.is_empty() {
                // End of input - send None to signal completion
                let _ = sender.send(Ok(None));
                return;
            }

            // Send the batch - this will block if the channel is full (backpressure)
            if sender.send(Ok(Some((records, headers)))).is_err() {
                // Receiver was dropped - not an error, just exit cleanly
                return;
            }
        }
    }

    /// Get the next batch of records.
    ///
    /// Returns `Ok(Some((records, headers)))` for each batch,
    /// `Ok(None)` when all records have been read,
    /// or `Err` if an error occurred during reading.
    ///
    /// Uses a timeout (default 5 minutes) to avoid hanging indefinitely
    /// if the reader thread stalls (e.g., NFS mount issues, disk errors).
    pub fn next_batch(&mut self) -> Result<Option<(Vec<OwnedRecord>, Vec<String>)>> {
        match self.receiver.recv_timeout(self.timeout) {
            Ok(result) => result,
            Err(RecvTimeoutError::Timeout) => {
                // Check if there's a stored error from the reader thread
                if let Some(err) = self.error_capture.get() {
                    return Err(anyhow!("Reader thread error: {}", err));
                }
                Err(anyhow!(
                    "Timeout waiting for next batch ({}s) - reader thread may be stalled",
                    self.timeout.as_secs()
                ))
            }
            Err(RecvTimeoutError::Disconnected) => {
                // Channel closed - check error state first
                if let Some(err) = self.error_capture.get() {
                    return Err(anyhow!("Reader thread error: {}", err));
                }
                // Then check if thread panicked
                if let Some(handle) = self.prefetch_thread.take() {
                    match handle.join() {
                        Ok(()) => Err(anyhow!("Prefetch thread exited unexpectedly")),
                        Err(_) => Err(anyhow!("Prefetch thread panicked")),
                    }
                } else {
                    Err(anyhow!("Prefetch channel closed"))
                }
            }
        }
    }

    /// Flush the output and wait for the prefetch thread to complete.
    pub fn finish(&mut self) -> Result<()> {
        self.writer.finish()?;

        // Wait for the prefetch thread to complete
        if let Some(handle) = self.prefetch_thread.take() {
            handle
                .join()
                .map_err(|_| anyhow!("Prefetch thread panicked"))?;
        }

        Ok(())
    }
}

// ============================================================================
// Output Format Detection and Writing
// ============================================================================

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

/// Default batch size for Parquet output buffering.
const DEFAULT_PARQUET_OUTPUT_BATCH_SIZE: usize = 10_000;

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
    /// Parquet output with batched writing
    Parquet {
        writer: ArrowWriter<File>,
        read_ids: Vec<String>,
        bucket_names: Vec<String>,
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

    /// Returns true if this writer outputs to Parquet format.
    #[allow(dead_code)]
    pub fn is_parquet(&self) -> bool {
        matches!(self, OutputWriter::Parquet { .. })
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
            OutputWriter::Parquet { .. } => {
                // No header needed for Parquet - schema is embedded
                Ok(())
            }
        }
    }

    /// Write a single record directly (avoids TSV serialization for Parquet).
    ///
    /// This is the preferred method for Parquet output as it avoids the
    /// overhead of serializing to TSV and then parsing back.
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
                    if parts.len() >= 3 {
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
        }
    }
}

// ============================================================================
// Parquet Input Reading
// ============================================================================

use arrow::array::{Array, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

/// Check if a file path indicates Parquet input.
pub fn is_parquet_input(path: &Path) -> bool {
    path.extension()
        .map(|ext| ext.eq_ignore_ascii_case("parquet"))
        .unwrap_or(false)
}

/// Reader for Parquet input files with read_id, sequence1, and optional sequence2 columns.
#[allow(dead_code)]
pub struct ParquetInputReader {
    reader: parquet::arrow::arrow_reader::ParquetRecordBatchReader,
    is_paired: bool,
    current_batch: Option<RecordBatch>,
    current_idx: usize,
    global_record_id: i64,
}

#[allow(dead_code)]
impl ParquetInputReader {
    /// Check if a data type is a valid string type (Utf8 or LargeUtf8).
    fn is_string_type(dt: &DataType) -> bool {
        matches!(dt, DataType::Utf8 | DataType::LargeUtf8)
    }

    /// Open a Parquet file and validate schema.
    ///
    /// Required columns: read_id (string), sequence1 (string)
    /// Optional column: sequence2 (string) - if first row is non-null, data is paired
    ///
    /// # Errors
    /// Returns an error if:
    /// - The file cannot be opened
    /// - Required columns are missing
    /// - Columns have incorrect types (must be string types)
    pub fn new(path: &Path) -> Result<Self> {
        let file =
            File::open(path).with_context(|| format!("Failed to open Parquet file: {:?}", path))?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .context("Failed to create Parquet reader")?;

        let schema = builder.schema();

        // Validate required columns exist and have correct types
        let read_id_field = schema
            .fields()
            .iter()
            .find(|f| f.name() == "read_id")
            .ok_or_else(|| anyhow!("Parquet input missing required column 'read_id'"))?;

        if !Self::is_string_type(read_id_field.data_type()) {
            return Err(anyhow!(
                "Column 'read_id' must be string type (Utf8 or LargeUtf8), got {:?}",
                read_id_field.data_type()
            ));
        }

        let sequence1_field = schema
            .fields()
            .iter()
            .find(|f| f.name() == "sequence1")
            .ok_or_else(|| anyhow!("Parquet input missing required column 'sequence1'"))?;

        if !Self::is_string_type(sequence1_field.data_type()) {
            return Err(anyhow!(
                "Column 'sequence1' must be string type (Utf8 or LargeUtf8), got {:?}",
                sequence1_field.data_type()
            ));
        }

        // Check optional sequence2 column
        let has_sequence2 =
            if let Some(field) = schema.fields().iter().find(|f| f.name() == "sequence2") {
                if !Self::is_string_type(field.data_type()) {
                    return Err(anyhow!(
                        "Column 'sequence2' must be string type (Utf8 or LargeUtf8), got {:?}",
                        field.data_type()
                    ));
                }
                true
            } else {
                false
            };

        let mut reader = builder.build().context("Failed to build Parquet reader")?;

        // Read first batch to detect if paired-end (sequence2 non-null)
        let first_batch = reader.next();
        let (is_paired, current_batch) = match first_batch {
            Some(Ok(batch)) => {
                let is_paired = if has_sequence2 {
                    // Check if first value in sequence2 is non-null
                    if let Some(col) = batch.column_by_name("sequence2") {
                        col.null_count() < col.len()
                    } else {
                        false
                    }
                } else {
                    false
                };
                (is_paired, Some(batch))
            }
            Some(Err(e)) => return Err(anyhow!("Error reading first Parquet batch: {}", e)),
            None => (false, None), // Empty file
        };

        // Log paired-end detection result
        log::info!(
            "Parquet input '{}': detected as {} data",
            path.display(),
            if is_paired {
                "paired-end"
            } else {
                "single-end"
            }
        );

        Ok(Self {
            reader,
            is_paired,
            current_batch,
            current_idx: 0,
            global_record_id: 0,
        })
    }

    /// Returns whether the input is paired-end.
    #[allow(dead_code)]
    pub fn is_paired(&self) -> bool {
        self.is_paired
    }

    /// Get the next batch of records.
    ///
    /// Returns `Ok(Some((records, headers)))` for each batch,
    /// `Ok(None)` when all records have been read.
    pub fn next_batch(
        &mut self,
        batch_size: usize,
    ) -> Result<Option<(Vec<OwnedRecord>, Vec<String>)>> {
        let mut records = Vec::with_capacity(batch_size);
        let mut headers = Vec::with_capacity(batch_size);

        while records.len() < batch_size {
            // Get next record from current batch
            if let Some(ref batch) = self.current_batch {
                if self.current_idx < batch.num_rows() {
                    // Extract columns
                    let read_id_col = batch
                        .column_by_name("read_id")
                        .ok_or_else(|| anyhow!("Missing read_id column"))?;
                    let sequence1_col = batch
                        .column_by_name("sequence1")
                        .ok_or_else(|| anyhow!("Missing sequence1 column"))?;

                    let read_id_arr = read_id_col
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| anyhow!("read_id column is not a string array"))?;
                    let sequence1_arr = sequence1_col
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| anyhow!("sequence1 column is not a string array"))?;

                    let sequence2_arr = if self.is_paired {
                        batch
                            .column_by_name("sequence2")
                            .and_then(|col| col.as_any().downcast_ref::<StringArray>())
                    } else {
                        None
                    };

                    let idx = self.current_idx;
                    let read_id = read_id_arr.value(idx).to_string();
                    let sequence1 = sequence1_arr.value(idx).as_bytes().to_vec();
                    let sequence2 = sequence2_arr
                        .filter(|arr| !arr.is_null(idx))
                        .map(|arr| arr.value(idx).as_bytes().to_vec());

                    // Use batch-local index for query_id (will be adjusted in output)
                    records.push((records.len() as i64, sequence1, sequence2));
                    headers.push(read_id);
                    self.global_record_id += 1;
                    self.current_idx += 1;
                    continue;
                }
            }

            // Need to load next batch
            match self.reader.next() {
                Some(Ok(batch)) => {
                    self.current_batch = Some(batch);
                    self.current_idx = 0;
                }
                Some(Err(e)) => return Err(anyhow!("Error reading Parquet batch: {}", e)),
                None => {
                    // No more batches
                    break;
                }
            }
        }

        if records.is_empty() {
            Ok(None)
        } else {
            Ok(Some((records, headers)))
        }
    }
}

// ============================================================================
// Prefetching Parquet Reader
// ============================================================================

use arrow::array::LargeStringArray;
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
use rayon::prelude::*;
use rype::QueryRecord;

/// Type alias for Parquet batch data sent through the prefetch channel.
/// Contains the raw RecordBatch for zero-copy access and pre-extracted headers.
type ParquetBatchResult = Result<Option<(RecordBatch, Vec<String>)>>;

/// Default timeout for waiting on Parquet prefetch batches (5 minutes).
const DEFAULT_PARQUET_PREFETCH_TIMEOUT: Duration = Duration::from_secs(300);

/// Prefetching Parquet reader with background I/O for overlapping I/O and computation.
///
/// This reader spawns a background thread that reads RecordBatch objects from Parquet
/// while the main thread processes the current batch. Unlike `ParquetInputReader` which
/// copies sequences record-by-record, this reader enables **zero-copy** sequence access
/// by keeping the RecordBatch alive during classification.
///
/// # Zero-Copy Design
///
/// The background thread reads complete RecordBatch objects and extracts only headers
/// (read_ids). The main thread then uses `batch_to_records_parquet()` to get zero-copy
/// references into the Arrow buffer memory.
///
/// # Usage
///
/// ```ignore
/// let mut reader = PrefetchingParquetReader::new(&path)?;
/// while let Some((batch, headers)) = reader.next_batch()? {
///     // Zero-copy conversion - references point into batch's Arrow buffers
///     let records: Vec<QueryRecord> = batch_to_records_parquet(&batch, headers.len())?;
///     let results = classify_batch(..., &records, ...);
///     // batch must stay alive until classification completes
/// }
/// ```
pub struct PrefetchingParquetReader {
    receiver: Receiver<ParquetBatchResult>,
    prefetch_thread: Option<JoinHandle<()>>,
    error_capture: Arc<FirstErrorCapture>,
    is_paired: bool,
    timeout: Duration,
}

impl PrefetchingParquetReader {
    /// Create a new prefetching Parquet reader.
    ///
    /// # Arguments
    /// * `path` - Path to the Parquet file
    /// * `batch_size` - Number of records per batch (controls Parquet reader batch size)
    ///
    /// # Returns
    /// A reader that prefetches RecordBatch objects in a background thread.
    #[allow(dead_code)]
    pub fn new(path: &Path, batch_size: usize) -> Result<Self> {
        Self::with_parallel_row_groups(path, batch_size, None)
    }

    /// Create a new prefetching Parquet reader with optional parallel row group processing.
    ///
    /// # Arguments
    /// * `path` - Path to the Parquet file
    /// * `batch_size` - Number of records per batch (controls Parquet reader batch size)
    /// * `parallel_row_groups` - Optional number of row groups to process in parallel.
    ///   If `Some(n)`, uses parallel reading with n row groups per chunk.
    ///   If `None`, uses sequential reading (original behavior).
    ///
    /// # Returns
    /// A reader that prefetches RecordBatch objects in a background thread.
    ///
    /// # Parallel Reading
    ///
    /// When `parallel_row_groups` is `Some(n)`:
    /// - Row groups are processed in parallel chunks of size n
    /// - Each parallel task opens its own file handle
    /// - Results are sorted within each chunk to maintain ordering
    /// - Most effective when decompression is CPU-bound (not I/O-bound)
    ///
    /// Recommended values:
    /// - `Some(4)` - Good balance for most SSDs (default when enabled)
    /// - `Some(2)` - More conservative, lower memory usage
    /// - `None` - Sequential reading (original behavior)
    pub fn with_parallel_row_groups(
        path: &Path,
        batch_size: usize,
        parallel_row_groups: Option<usize>,
    ) -> Result<Self> {
        // Clone path for the background thread
        let path = path.to_path_buf();

        // Thread-safe error capture (only stores first error)
        let error_capture = Arc::new(FirstErrorCapture::new());
        let thread_error = Arc::clone(&error_capture);

        // Use sync_channel with buffer of 4 for prefetching:
        // - Larger buffer allows more read-ahead when classification is slower than decompression
        // - Trade-off: more memory usage for buffered RecordBatches
        let (sender, receiver): (SyncSender<ParquetBatchResult>, Receiver<ParquetBatchResult>) =
            mpsc::sync_channel(4);

        // Read parquet schema to validate and get column indices
        let file = File::open(&path)
            .with_context(|| format!("Failed to open Parquet file: {:?}", path))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .context("Failed to create Parquet reader")?;
        let schema = builder.schema().clone();

        // Validate schema and get column indices for projection
        let (col_indices, has_sequence2) = Self::validate_and_get_projection(&schema)?;

        // Spawn background thread - either parallel or sequential based on option
        // Note: Some(0) is treated as disabled (sequential mode) - users who want
        // default parallelism should use Some(DEFAULT_PARALLEL_ROW_GROUPS) explicitly
        let prefetch_thread = if let Some(parallel_rg) = parallel_row_groups.filter(|&n| n > 0) {
            log::info!(
                "Using parallel Parquet row group reading (parallelism={})",
                parallel_rg
            );
            thread::spawn(move || {
                Self::reader_thread_parallel(
                    path,
                    batch_size,
                    col_indices,
                    parallel_rg,
                    sender,
                    thread_error,
                );
            })
        } else {
            thread::spawn(move || {
                Self::reader_thread(path, batch_size, col_indices, sender, thread_error);
            })
        };

        Ok(Self {
            receiver,
            prefetch_thread: Some(prefetch_thread),
            error_capture,
            is_paired: has_sequence2,
            timeout: DEFAULT_PARQUET_PREFETCH_TIMEOUT,
        })
    }

    /// Validate the Parquet schema and return column indices for projection.
    ///
    /// Returns (column_indices, has_sequence2) where column_indices contains
    /// the indices of read_id, sequence1, and optionally sequence2.
    fn validate_and_get_projection(
        schema: &arrow::datatypes::Schema,
    ) -> Result<(Vec<usize>, bool)> {
        fn is_string_type(dt: &DataType) -> bool {
            matches!(dt, DataType::Utf8 | DataType::LargeUtf8)
        }

        // Find read_id column
        let read_id_idx = schema
            .fields()
            .iter()
            .position(|f| f.name() == "read_id")
            .ok_or_else(|| anyhow!("Parquet input missing required column 'read_id'"))?;

        if !is_string_type(schema.field(read_id_idx).data_type()) {
            return Err(anyhow!(
                "Column 'read_id' must be string type (Utf8 or LargeUtf8), got {:?}",
                schema.field(read_id_idx).data_type()
            ));
        }

        // Find sequence1 column
        let sequence1_idx = schema
            .fields()
            .iter()
            .position(|f| f.name() == "sequence1")
            .ok_or_else(|| anyhow!("Parquet input missing required column 'sequence1'"))?;

        if !is_string_type(schema.field(sequence1_idx).data_type()) {
            return Err(anyhow!(
                "Column 'sequence1' must be string type (Utf8 or LargeUtf8), got {:?}",
                schema.field(sequence1_idx).data_type()
            ));
        }

        // Find optional sequence2 column
        let sequence2_idx = schema.fields().iter().position(|f| f.name() == "sequence2");

        if let Some(idx) = sequence2_idx {
            if !is_string_type(schema.field(idx).data_type()) {
                return Err(anyhow!(
                    "Column 'sequence2' must be string type (Utf8 or LargeUtf8), got {:?}",
                    schema.field(idx).data_type()
                ));
            }
        }

        // Build projection mask - only read the columns we need
        let mut col_indices = vec![read_id_idx, sequence1_idx];
        let has_sequence2 = sequence2_idx.is_some();
        if let Some(idx) = sequence2_idx {
            col_indices.push(idx);
        }

        Ok((col_indices, has_sequence2))
    }

    /// Background thread function that reads RecordBatches and sends them through the channel.
    ///
    /// Note: We do NOT use the user's batch_size here because Parquet's byte array decoder
    /// can overflow with very large batches (>200K rows with long sequences). Instead, we
    /// let the Parquet reader use natural row-group-based batching, which is safe.
    /// The main thread can accumulate multiple batches if larger batches are needed.
    fn reader_thread(
        path: PathBuf,
        _batch_size: usize, // Ignored - use natural Parquet batching to avoid overflow
        col_indices: Vec<usize>,
        sender: SyncSender<ParquetBatchResult>,
        error_capture: Arc<FirstErrorCapture>,
    ) {
        // Helper macro to send error and store if send fails
        macro_rules! send_error {
            ($msg:expr) => {{
                let err_msg = $msg;
                if sender.send(Err(anyhow!("{}", &err_msg))).is_err() {
                    error_capture.store_msg(&err_msg);
                }
                return;
            }};
        }

        // Open file and create reader in background thread
        let file = match File::open(&path) {
            Ok(f) => f,
            Err(e) => {
                send_error!(format!("Failed to open Parquet file: {}", e));
            }
        };

        let builder = match ParquetRecordBatchReaderBuilder::try_new(file) {
            Ok(b) => b,
            Err(e) => {
                send_error!(format!("Failed to create Parquet reader: {}", e));
            }
        };

        // Build projection mask from column indices
        let projection =
            parquet::arrow::ProjectionMask::roots(builder.parquet_schema(), col_indices);

        // Use column projection but let Parquet use natural row-group-based batching.
        // Do NOT use .with_batch_size() with large values as it can cause
        // "index overflow decoding byte array" errors with string columns.
        let reader = match builder.with_projection(projection).build() {
            Ok(r) => r,
            Err(e) => {
                send_error!(format!("Failed to build Parquet reader: {}", e));
            }
        };

        for batch_result in reader {
            let batch = match batch_result {
                Ok(b) => b,
                Err(e) => {
                    send_error!(format!("Error reading Parquet batch: {}", e));
                }
            };

            // Extract headers (read_ids) - these are small string copies
            let headers = match Self::extract_headers(&batch) {
                Ok(h) => h,
                Err(e) => {
                    send_error!(format!("Error extracting headers: {}", e));
                }
            };

            // Send the batch and headers
            if sender.send(Ok(Some((batch, headers)))).is_err() {
                // Receiver was dropped - exit cleanly
                return;
            }
        }

        // Send None to signal completion
        let _ = sender.send(Ok(None));
    }

    /// Background thread function that reads row groups in parallel chunks and sends batches.
    ///
    /// This function processes row groups in parallel chunks using rayon, providing better
    /// throughput when decompression is CPU-bound. Results are sorted within each chunk
    /// to maintain ordering.
    ///
    /// # Arguments
    /// * `path` - Path to the Parquet file
    /// * `_batch_size` - Ignored (use natural Parquet batching to avoid overflow)
    /// * `col_indices` - Column indices for projection
    /// * `parallel_rg` - Number of row groups to process in parallel per chunk
    /// * `sender` - Channel sender for batch results
    /// * `error_capture` - Thread-safe error capture for errors during send failures
    fn reader_thread_parallel(
        path: PathBuf,
        _batch_size: usize, // Ignored - use natural Parquet batching to avoid overflow
        col_indices: Vec<usize>,
        parallel_rg: usize,
        sender: SyncSender<ParquetBatchResult>,
        error_capture: Arc<FirstErrorCapture>,
    ) {
        // Helper macro to send error and store if send fails
        macro_rules! send_error {
            ($msg:expr) => {{
                let err_msg = $msg;
                if sender.send(Err(anyhow!("{}", &err_msg))).is_err() {
                    error_capture.store_msg(&err_msg);
                }
                return;
            }};
        }

        // Load metadata to get the number of row groups
        let file = match File::open(&path) {
            Ok(f) => f,
            Err(e) => {
                send_error!(format!("Failed to open Parquet file: {}", e));
            }
        };

        let initial_metadata = match ArrowReaderMetadata::load(&file, ArrowReaderOptions::default())
        {
            Ok(m) => m,
            Err(e) => {
                send_error!(format!("Failed to load Parquet metadata: {}", e));
            }
        };

        let num_row_groups = initial_metadata.metadata().num_row_groups();
        if num_row_groups == 0 {
            let _ = sender.send(Ok(None));
            return;
        }

        // Drop the initial file handle - each parallel task will open its own
        drop(file);

        log::debug!(
            "Parallel Parquet reader: {} row groups, parallelism={}",
            num_row_groups,
            parallel_rg
        );

        // Wrap col_indices in Arc for sharing across threads
        let col_indices = Arc::new(col_indices);

        // Process row groups in chunks of parallel_rg
        for chunk_start in (0..num_row_groups).step_by(parallel_rg) {
            let chunk_end = (chunk_start + parallel_rg).min(num_row_groups);
            let rg_indices: Vec<usize> = (chunk_start..chunk_end).collect();

            // Read row groups in parallel, each task loads its own metadata for robustness
            // Use Result collection pattern for clean error handling - first error fails the chunk
            #[allow(clippy::type_complexity)]
            let chunk_results: Result<
                Vec<(usize, Vec<(RecordBatch, Vec<String>)>)>,
                String,
            > = rg_indices
                .into_par_iter()
                .map(|rg_idx| {
                    // Each parallel task opens its own file handle and loads fresh metadata
                    // This is more robust than sharing metadata across file handles
                    let file = File::open(&path)
                        .map_err(|e| format!("Failed to open file for RG {}: {}", rg_idx, e))?;

                    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
                        .map_err(|e| format!("Failed to create reader for RG {}: {}", rg_idx, e))?;

                    // Build projection mask for this reader
                    let projection = parquet::arrow::ProjectionMask::roots(
                        builder.parquet_schema(),
                        col_indices.iter().copied(),
                    );

                    // Use row group selection and projection, but let Parquet use natural batching.
                    // Do NOT use .with_batch_size() with large values to avoid offset overflow.
                    let reader = builder
                        .with_row_groups(vec![rg_idx])
                        .with_projection(projection)
                        .build()
                        .map_err(|e| format!("Failed to build reader for RG {}: {}", rg_idx, e))?;

                    // Collect all batches from this row group
                    let mut batches = Vec::new();
                    for batch_result in reader {
                        let batch = batch_result.map_err(|e| {
                            format!("Error reading batch from RG {}: {}", rg_idx, e)
                        })?;

                        let headers = Self::extract_headers(&batch).map_err(|e| {
                            format!("Error extracting headers from RG {}: {}", rg_idx, e)
                        })?;

                        batches.push((batch, headers));
                    }

                    Ok((rg_idx, batches))
                })
                .collect();

            // Handle chunk result - either process success or report first error
            let mut sorted_results = match chunk_results {
                Ok(results) => results,
                Err(e) => {
                    error_capture.store_msg(&e);
                    return;
                }
            };

            // Sort by row group index to maintain ordering
            sorted_results.sort_by_key(|(idx, _)| *idx);

            // Send batches in order
            for (_, batches) in sorted_results {
                for (batch, headers) in batches {
                    if sender.send(Ok(Some((batch, headers)))).is_err() {
                        // Receiver dropped - exit cleanly
                        return;
                    }
                }
            }
        }

        // Send None to signal completion
        let _ = sender.send(Ok(None));
    }

    /// Extract headers (read_ids) from a RecordBatch.
    fn extract_headers(batch: &RecordBatch) -> Result<Vec<String>> {
        let col = batch
            .column_by_name("read_id")
            .ok_or_else(|| anyhow!("Missing read_id column"))?;

        // Try StringArray first, then LargeStringArray
        if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
            let mut headers = Vec::with_capacity(batch.num_rows());
            for i in 0..batch.num_rows() {
                headers.push(arr.value(i).to_string());
            }
            return Ok(headers);
        }

        if let Some(arr) = col.as_any().downcast_ref::<LargeStringArray>() {
            let mut headers = Vec::with_capacity(batch.num_rows());
            for i in 0..batch.num_rows() {
                headers.push(arr.value(i).to_string());
            }
            return Ok(headers);
        }

        Err(anyhow!(
            "read_id column is not a string type: {:?}",
            col.data_type()
        ))
    }

    /// Returns whether the input is paired-end.
    #[allow(dead_code)]
    pub fn is_paired(&self) -> bool {
        self.is_paired
    }

    /// Get the next batch of records.
    ///
    /// Returns `Ok(Some((batch, headers)))` for each batch,
    /// `Ok(None)` when all records have been read,
    /// or `Err` if an error occurred during reading.
    ///
    /// The returned RecordBatch contains the raw Arrow data. Use `batch_to_records_parquet()`
    /// for zero-copy conversion to QueryRecord references.
    pub fn next_batch(&mut self) -> Result<Option<(RecordBatch, Vec<String>)>> {
        match self.receiver.recv_timeout(self.timeout) {
            Ok(result) => result,
            Err(RecvTimeoutError::Timeout) => {
                if let Some(err) = self.error_capture.get() {
                    return Err(anyhow!("Reader thread error: {}", err));
                }
                Err(anyhow!(
                    "Timeout waiting for next batch ({}s) - reader thread may be stalled",
                    self.timeout.as_secs()
                ))
            }
            Err(RecvTimeoutError::Disconnected) => {
                if let Some(err) = self.error_capture.get() {
                    return Err(anyhow!("Reader thread error: {}", err));
                }
                if let Some(handle) = self.prefetch_thread.take() {
                    match handle.join() {
                        Ok(()) => Err(anyhow!("Prefetch thread exited unexpectedly")),
                        Err(_) => Err(anyhow!("Prefetch thread panicked")),
                    }
                } else {
                    Err(anyhow!("Prefetch channel closed"))
                }
            }
        }
    }

    /// Finish and wait for the prefetch thread to complete.
    pub fn finish(&mut self) -> Result<()> {
        if let Some(handle) = self.prefetch_thread.take() {
            handle
                .join()
                .map_err(|_| anyhow!("Prefetch thread panicked"))?;
        }
        Ok(())
    }
}

/// Enum for uniform access to String and LargeString sequence columns.
enum SequenceColumnRef<'a> {
    String(&'a StringArray),
    LargeString(&'a LargeStringArray),
}

impl<'a> SequenceColumnRef<'a> {
    #[inline]
    fn value(&self, i: usize) -> &'a [u8] {
        match self {
            SequenceColumnRef::String(arr) => arr.value(i).as_bytes(),
            SequenceColumnRef::LargeString(arr) => arr.value(i).as_bytes(),
        }
    }

    #[inline]
    fn is_null(&self, i: usize) -> bool {
        match self {
            SequenceColumnRef::String(arr) => arr.is_null(i),
            SequenceColumnRef::LargeString(arr) => arr.is_null(i),
        }
    }
}

/// Extract a string column from a RecordBatch as SequenceColumnRef.
fn get_string_column<'a>(batch: &'a RecordBatch, col_name: &str) -> Result<SequenceColumnRef<'a>> {
    let col = batch
        .column_by_name(col_name)
        .ok_or_else(|| anyhow!("Missing {} column", col_name))?;

    if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
        return Ok(SequenceColumnRef::String(arr));
    }

    if let Some(arr) = col.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(SequenceColumnRef::LargeString(arr));
    }

    Err(anyhow!(
        "{} column is not a string type: {:?}",
        col_name,
        col.data_type()
    ))
}

/// Convert a Parquet RecordBatch to QueryRecord references with zero-copy semantics.
///
/// This function creates batch-local indices as query_ids (0, 1, 2, ...) and
/// returns zero-copy references into the Arrow buffer memory for sequences.
///
/// # Arguments
/// * `batch` - RecordBatch from Parquet with 'sequence1' and optional 'sequence2' columns
///
/// # Returns
/// A vector of QueryRecord tuples with batch-local indices as IDs.
///
/// # Zero-Copy Guarantee
/// The returned sequence slices point directly into the Arrow buffers.
/// The batch must remain alive until classification is complete.
#[allow(dead_code)]
pub fn batch_to_records_parquet(batch: &RecordBatch) -> Result<Vec<QueryRecord<'_>>> {
    batch_to_records_parquet_with_offset(batch, 0)
}

/// Convert a Parquet RecordBatch to QueryRecord references with a starting index offset.
///
/// This is used when stacking multiple batches - each batch gets an offset so that
/// query_ids are globally unique across all stacked batches.
///
/// # Arguments
/// * `batch` - RecordBatch from Parquet with 'sequence1' and optional 'sequence2' columns
/// * `id_offset` - Starting index for query_ids in this batch
///
/// # Returns
/// A vector of QueryRecord tuples with offset indices as IDs.
pub fn batch_to_records_parquet_with_offset(
    batch: &RecordBatch,
    id_offset: usize,
) -> Result<Vec<QueryRecord<'_>>> {
    let num_rows = batch.num_rows();
    if num_rows == 0 {
        return Ok(Vec::new());
    }

    // Get sequence1 column
    let seq_col = get_string_column(batch, "sequence1")?;

    // Check if we have sequence2 column
    let pair_col = batch
        .column_by_name("sequence2")
        .map(|_| get_string_column(batch, "sequence2"))
        .transpose()?;

    // Build records with offset indices for stacking support
    let mut records = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        // Safe overflow checking: both usize addition and i64 conversion can fail
        let sum = id_offset
            .checked_add(i)
            .ok_or_else(|| anyhow!("Query ID overflow: offset {} + index {}", id_offset, i))?;
        let query_id =
            i64::try_from(sum).map_err(|_| anyhow!("Query ID {} exceeds i64::MAX", sum))?;
        let seq = seq_col.value(i);
        let pair = pair_col
            .as_ref()
            .and_then(|p| if p.is_null(i) { None } else { Some(p.value(i)) });
        records.push((query_id, seq, pair));
    }

    Ok(records)
}

/// Convert multiple stacked RecordBatches to a combined QueryRecord vector with zero-copy.
///
/// This function processes multiple batches together, assigning globally unique
/// query_ids across all batches. All batches must remain alive while the returned
/// records are in use.
///
/// # Arguments
/// * `batches` - Slice of (RecordBatch, headers) pairs to process together
///
/// # Returns
/// A tuple of:
/// - Combined QueryRecord vector with zero-copy references into all batches
/// - Combined headers vector
///
/// # Zero-Copy Guarantee
/// The returned sequence slices point directly into the Arrow buffers of the
/// respective batches. ALL batches in the input slice must remain alive until
/// classification is complete.
///
/// # Errors
/// - Returns an error if any batch is missing the required 'sequence1' column
/// - Returns an error if query ID calculation overflows (cumulative rows > i64::MAX)
/// - Returns an error if the 'sequence1' column is not a valid string array type
///
/// # Example
/// ```ignore
/// // Stack multiple batches for efficient parallel classification
/// let stacked: Vec<(RecordBatch, Vec<String>)> = collect_batches();
/// let (records, headers) = stacked_batches_to_records(&stacked)?;
///
/// // IMPORTANT: stacked must remain alive while records are in use
/// let results = index.classify_batch(&records, threshold, ...)?;
///
/// // Process results using headers for read IDs
/// for result in results {
///     let read_id = &headers[result.query_id as usize];
///     println!("{}\t{}", read_id, result.score);
/// }
/// // Now safe to drop stacked
/// drop(stacked);
/// ```
pub fn stacked_batches_to_records<'a>(
    batches: &'a [(RecordBatch, Vec<String>)],
) -> Result<(Vec<QueryRecord<'a>>, Vec<&'a str>)> {
    // Calculate total capacity
    let total_rows: usize = batches.iter().map(|(b, _)| b.num_rows()).sum();

    let mut all_records = Vec::with_capacity(total_rows);
    let mut all_headers = Vec::with_capacity(total_rows);
    let mut offset = 0usize;

    for (batch, headers) in batches {
        // Convert this batch with the current offset
        let records = batch_to_records_parquet_with_offset(batch, offset)?;
        all_records.extend(records);

        // Add headers as references (zero-copy for strings too)
        all_headers.extend(headers.iter().map(|s| s.as_str()));

        offset += batch.num_rows();
    }

    Ok((all_records, all_headers))
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read as _;
    use tempfile::NamedTempFile;

    // -------------------------------------------------------------------------
    // Tests for base_read_id
    // -------------------------------------------------------------------------

    #[test]
    fn test_base_read_id_simple() {
        // Simple read ID with no suffix or comment
        assert_eq!(PrefetchingIoHandler::base_read_id(b"READ1"), b"READ1");
    }

    #[test]
    fn test_base_read_id_with_space_comment() {
        // Read ID with space-separated comment
        assert_eq!(
            PrefetchingIoHandler::base_read_id(b"READ1 comment text here"),
            b"READ1"
        );
    }

    #[test]
    fn test_base_read_id_with_tab_comment() {
        // Read ID with tab-separated comment
        assert_eq!(
            PrefetchingIoHandler::base_read_id(b"READ1\tcomment"),
            b"READ1"
        );
    }

    #[test]
    fn test_base_read_id_with_slash_1_suffix() {
        // Read ID with /1 suffix (forward read)
        assert_eq!(PrefetchingIoHandler::base_read_id(b"READ1/1"), b"READ1");
    }

    #[test]
    fn test_base_read_id_with_slash_2_suffix() {
        // Read ID with /2 suffix (reverse read)
        assert_eq!(PrefetchingIoHandler::base_read_id(b"READ1/2"), b"READ1");
    }

    #[test]
    fn test_base_read_id_with_slash_1_and_comment() {
        // Read ID with /1 suffix followed by comment
        // The /1 should be stripped, then the space would stop parsing
        // But since /1 is at position 5, and space is at position 7,
        // we find space first... no wait, we iterate and check both conditions.
        // Actually the logic finds first space OR /1 at end of ID portion.
        // Let's trace: "READ1/1 comment"
        // - First find space at position 7
        // - id_portion = "READ1/1"
        // - Check if ends with /1 or /2: yes, /1
        // - Return "READ1"
        assert_eq!(
            PrefetchingIoHandler::base_read_id(b"READ1/1 comment"),
            b"READ1"
        );
    }

    #[test]
    fn test_base_read_id_illumina_style() {
        // Illumina-style header: @HWUSI-EAS100R:6:73:941:1973/1 length=36
        assert_eq!(
            PrefetchingIoHandler::base_read_id(b"HWUSI-EAS100R:6:73:941:1973/1 length=36"),
            b"HWUSI-EAS100R:6:73:941:1973"
        );
    }

    #[test]
    fn test_base_read_id_casava_style() {
        // CASAVA 1.8+ style: @EAS139:136:FC706VJ:2:2104:15343:197393 1:Y:18:ATCACG
        assert_eq!(
            PrefetchingIoHandler::base_read_id(
                b"EAS139:136:FC706VJ:2:2104:15343:197393 1:Y:18:ATCACG"
            ),
            b"EAS139:136:FC706VJ:2:2104:15343:197393"
        );
    }

    #[test]
    fn test_base_read_id_empty() {
        // Empty input
        assert_eq!(PrefetchingIoHandler::base_read_id(b""), b"");
    }

    #[test]
    fn test_base_read_id_slash_only() {
        // Just a slash (edge case)
        assert_eq!(PrefetchingIoHandler::base_read_id(b"/"), b"/");
    }

    #[test]
    fn test_base_read_id_slash_3() {
        // /3 should NOT be stripped (only /1 and /2 are paired-end markers)
        assert_eq!(PrefetchingIoHandler::base_read_id(b"READ1/3"), b"READ1/3");
    }

    #[test]
    fn test_base_read_id_internal_slash() {
        // Internal slash should not affect anything
        assert_eq!(
            PrefetchingIoHandler::base_read_id(b"path/to/READ1"),
            b"path/to/READ1"
        );
    }

    #[test]
    fn test_base_read_id_multiple_spaces() {
        // Multiple spaces - only first one matters
        assert_eq!(
            PrefetchingIoHandler::base_read_id(b"READ1  extra  spaces"),
            b"READ1"
        );
    }

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

    #[test]
    fn test_output_writer_is_parquet() {
        let tmp = NamedTempFile::with_suffix(".parquet").unwrap();
        let path = tmp.path().to_path_buf();
        let writer = OutputWriter::new(OutputFormat::Parquet, Some(&path), None).unwrap();
        assert!(writer.is_parquet());

        let tmp2 = NamedTempFile::with_suffix(".tsv").unwrap();
        let path2 = tmp2.path().to_path_buf();
        let writer2 = OutputWriter::new(OutputFormat::Tsv, Some(&path2), None).unwrap();
        assert!(!writer2.is_parquet());
    }

    // -------------------------------------------------------------------------
    // Tests for is_parquet_input
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_parquet_input() {
        assert!(is_parquet_input(Path::new("input.parquet")));
        assert!(is_parquet_input(Path::new("input.PARQUET")));
        assert!(is_parquet_input(Path::new("/path/to/input.parquet")));
        assert!(!is_parquet_input(Path::new("input.fastq")));
        assert!(!is_parquet_input(Path::new("input.fasta")));
        assert!(!is_parquet_input(Path::new("input.parquet.gz")));
    }

    // -------------------------------------------------------------------------
    // Tests for resolve_bucket_id
    // -------------------------------------------------------------------------

    #[test]
    fn test_resolve_bucket_id_numeric() {
        let mut bucket_names = HashMap::new();
        bucket_names.insert(1, "Bacteria".to_string());
        bucket_names.insert(2, "Archaea".to_string());
        bucket_names.insert(42, "Eukaryota".to_string());

        // Resolve by numeric ID
        assert_eq!(resolve_bucket_id("1", &bucket_names).unwrap(), 1);
        assert_eq!(resolve_bucket_id("2", &bucket_names).unwrap(), 2);
        assert_eq!(resolve_bucket_id("42", &bucket_names).unwrap(), 42);
    }

    #[test]
    fn test_resolve_bucket_id_by_name() {
        let mut bucket_names = HashMap::new();
        bucket_names.insert(1, "Bacteria".to_string());
        bucket_names.insert(2, "Archaea".to_string());
        bucket_names.insert(42, "Eukaryota".to_string());

        // Resolve by name
        assert_eq!(resolve_bucket_id("Bacteria", &bucket_names).unwrap(), 1);
        assert_eq!(resolve_bucket_id("Archaea", &bucket_names).unwrap(), 2);
        assert_eq!(resolve_bucket_id("Eukaryota", &bucket_names).unwrap(), 42);
    }

    #[test]
    fn test_resolve_bucket_id_name_not_found() {
        let mut bucket_names = HashMap::new();
        bucket_names.insert(1, "Bacteria".to_string());

        // Name not found should error
        let result = resolve_bucket_id("NotFound", &bucket_names);
        assert!(result.is_err());
        let err_msg = result.err().unwrap().to_string();
        assert!(err_msg.contains("not found"), "Error was: {}", err_msg);
    }

    #[test]
    fn test_resolve_bucket_id_ambiguous_name() {
        let mut bucket_names = HashMap::new();
        bucket_names.insert(1, "SameName".to_string());
        bucket_names.insert(2, "SameName".to_string());
        bucket_names.insert(3, "Unique".to_string());

        // Ambiguous name should error
        let result = resolve_bucket_id("SameName", &bucket_names);
        assert!(result.is_err());
        let err_msg = result.err().unwrap().to_string();
        assert!(err_msg.contains("Ambiguous"), "Error was: {}", err_msg);

        // Unique name still works
        assert_eq!(resolve_bucket_id("Unique", &bucket_names).unwrap(), 3);

        // Numeric ID still works for ambiguous names
        assert_eq!(resolve_bucket_id("1", &bucket_names).unwrap(), 1);
        assert_eq!(resolve_bucket_id("2", &bucket_names).unwrap(), 2);
    }

    #[test]
    fn test_resolve_bucket_id_numeric_not_in_index() {
        let mut bucket_names = HashMap::new();
        bucket_names.insert(1, "Bacteria".to_string());

        // Numeric ID that doesn't exist still returns (caller handles the error)
        // This allows the caller to give a better "bucket not found" error
        assert_eq!(resolve_bucket_id("999", &bucket_names).unwrap(), 999);
    }

    #[test]
    fn test_resolve_bucket_id_empty_bucket_names() {
        let bucket_names = HashMap::new();

        // Numeric ID works with empty map
        assert_eq!(resolve_bucket_id("1", &bucket_names).unwrap(), 1);

        // Name lookup fails with empty map
        let result = resolve_bucket_id("Bacteria", &bucket_names);
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_bucket_id_name_with_spaces() {
        let mut bucket_names = HashMap::new();
        bucket_names.insert(1, "My Bucket Name".to_string());

        // Names with spaces work
        assert_eq!(
            resolve_bucket_id("My Bucket Name", &bucket_names).unwrap(),
            1
        );
    }

    #[test]
    fn test_resolve_bucket_id_case_sensitive() {
        let mut bucket_names = HashMap::new();
        bucket_names.insert(1, "Bacteria".to_string());

        // Name matching is case-sensitive
        assert_eq!(resolve_bucket_id("Bacteria", &bucket_names).unwrap(), 1);

        let result = resolve_bucket_id("bacteria", &bucket_names);
        assert!(result.is_err());

        let result = resolve_bucket_id("BACTERIA", &bucket_names);
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_bucket_id_empty_string() {
        let mut bucket_names = HashMap::new();
        bucket_names.insert(1, "Bacteria".to_string());

        // Empty string should error
        let result = resolve_bucket_id("", &bucket_names);
        assert!(result.is_err());
        let err_msg = result.err().unwrap().to_string();
        assert!(
            err_msg.contains("cannot be empty"),
            "Error was: {}",
            err_msg
        );
    }

    #[test]
    fn test_resolve_bucket_id_whitespace_only() {
        let mut bucket_names = HashMap::new();
        bucket_names.insert(1, "Bacteria".to_string());

        // Whitespace-only string should error (trimmed to empty)
        let result = resolve_bucket_id("   ", &bucket_names);
        assert!(result.is_err());
        let err_msg = result.err().unwrap().to_string();
        assert!(
            err_msg.contains("cannot be empty"),
            "Error was: {}",
            err_msg
        );

        let result = resolve_bucket_id("\t\n", &bucket_names);
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_bucket_id_whitespace_trimmed() {
        let mut bucket_names = HashMap::new();
        bucket_names.insert(42, "Bacteria".to_string());

        // Leading/trailing whitespace should be trimmed
        assert_eq!(resolve_bucket_id("  42  ", &bucket_names).unwrap(), 42);
        assert_eq!(resolve_bucket_id("\t42\n", &bucket_names).unwrap(), 42);

        // Name with surrounding whitespace should also work after trim
        assert_eq!(
            resolve_bucket_id("  Bacteria  ", &bucket_names).unwrap(),
            42
        );
    }

    #[test]
    fn test_resolve_bucket_id_leading_zeros() {
        let mut bucket_names = HashMap::new();
        bucket_names.insert(7, "Bacteria".to_string());

        // Leading zeros parse correctly (007 -> 7)
        assert_eq!(resolve_bucket_id("007", &bucket_names).unwrap(), 7);
        assert_eq!(resolve_bucket_id("0007", &bucket_names).unwrap(), 7);
        assert_eq!(resolve_bucket_id("00", &bucket_names).unwrap(), 0);
    }

    #[test]
    fn test_resolve_bucket_id_numeric_name_collision() {
        let mut bucket_names = HashMap::new();
        bucket_names.insert(1, "Bacteria".to_string());
        bucket_names.insert(42, "1".to_string()); // Bucket named "1"

        // Numeric ID takes precedence - "1" resolves to ID 1, not bucket named "1"
        assert_eq!(resolve_bucket_id("1", &bucket_names).unwrap(), 1);

        // To get bucket 42 (named "1"), must use numeric ID
        assert_eq!(resolve_bucket_id("42", &bucket_names).unwrap(), 42);

        // Bucket named "1" is unreachable by name (this is documented behavior)
        // Attempting to resolve "1" gives ID 1, not the bucket named "1"
    }

    #[test]
    fn test_resolve_bucket_id_negative_number() {
        let bucket_names = HashMap::new();

        // Negative numbers don't parse as u32, so treated as names
        let result = resolve_bucket_id("-1", &bucket_names);
        assert!(result.is_err());
        let err_msg = result.err().unwrap().to_string();
        assert!(err_msg.contains("not found"), "Error was: {}", err_msg);
    }

    #[test]
    fn test_resolve_bucket_id_overflow() {
        let bucket_names = HashMap::new();

        // Number too large for u32 is treated as a name
        let result = resolve_bucket_id("999999999999999", &bucket_names);
        assert!(result.is_err());
        let err_msg = result.err().unwrap().to_string();
        assert!(err_msg.contains("not found"), "Error was: {}", err_msg);
    }
}
