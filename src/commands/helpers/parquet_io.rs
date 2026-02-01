//! Parquet input reading with background prefetching.

use anyhow::{anyhow, Context, Result};
use arrow::array::{Array, LargeStringArray, RecordBatch, StringArray};
use arrow::datatypes::DataType;
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder,
};
use rayon::prelude::*;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use rype::{FirstErrorCapture, QueryRecord};

use super::fastx_io::OwnedRecord;

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

// ============================================================================
// Zero-Copy Batch Conversion
// ============================================================================

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
    use std::path::Path;

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
}
