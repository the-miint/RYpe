//! FASTX (FASTA/FASTQ) I/O with background prefetching.

use anyhow::{anyhow, Result};
use needletail::{parse_fastx_file, FastxReader};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use rype::FirstErrorCapture;

use super::output::{OutputFormat, OutputWriter};

/// Owned FASTX record with sequence and quality data.
///
/// Headers are stored separately in the `Vec<String>` returned alongside records
/// from batch readers, avoiding duplicate storage. Use the batch-local `query_id`
/// to index into the headers vector when writing output.
///
/// Quality scores are only captured when `preserve_for_output` is enabled in the reader.
#[derive(Debug, Clone)]
pub struct OwnedFastxRecord {
    pub query_id: i64,
    pub seq1: Vec<u8>,
    pub qual1: Option<Vec<u8>>,
    pub seq2: Option<Vec<u8>>,
    pub qual2: Option<Vec<u8>>,
}

impl OwnedFastxRecord {
    /// Create a new owned FASTX record.
    pub fn new(
        query_id: i64,
        seq1: Vec<u8>,
        qual1: Option<Vec<u8>>,
        seq2: Option<Vec<u8>>,
        qual2: Option<Vec<u8>>,
    ) -> Self {
        Self {
            query_id,
            seq1,
            qual1,
            seq2,
            qual2,
        }
    }

    /// Returns true if this is a FASTQ record (has quality scores).
    #[allow(dead_code)]
    pub fn is_fastq(&self) -> bool {
        self.qual1.is_some()
    }

    /// Returns true if this is a paired-end record.
    #[allow(dead_code)]
    pub fn is_paired(&self) -> bool {
        self.seq2.is_some()
    }

    /// Get references to the sequences.
    #[allow(dead_code)]
    pub fn sequences(&self) -> (&[u8], Option<&[u8]>) {
        (&self.seq1, self.seq2.as_deref())
    }
}

/// Type alias for batch data sent through the prefetch channel.
type BatchResult = Result<Option<(Vec<OwnedFastxRecord>, Vec<String>)>>;

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
    #[allow(dead_code)]
    pub fn new(
        r1_path: &Path,
        r2_path: Option<&PathBuf>,
        out_path: Option<PathBuf>,
        batch_size: usize,
    ) -> Result<Self> {
        Self::with_trim(r1_path, r2_path, out_path, batch_size, None)
    }

    /// Create a new prefetching I/O handler with optional sequence trimming.
    ///
    /// When `trim_to` is specified, sequences are trimmed at read time to reduce
    /// memory usage for long reads. Records with R1 shorter than `trim_to` are skipped.
    ///
    /// # Arguments
    /// * `r1_path` - Path to the first read file (FASTQ/FASTA, optionally gzipped)
    /// * `r2_path` - Optional path to the second read file for paired-end
    /// * `out_path` - Optional output path (stdout if None)
    /// * `batch_size` - Number of records per batch
    /// * `trim_to` - Optional maximum sequence length. Sequences longer than this are
    ///   truncated at read time. Records with R1 shorter than this are skipped.
    ///
    /// # Returns
    /// A handler that prefetches batches in a background thread.
    pub fn with_trim(
        r1_path: &Path,
        r2_path: Option<&PathBuf>,
        out_path: Option<PathBuf>,
        batch_size: usize,
        trim_to: Option<usize>,
    ) -> Result<Self> {
        Self::with_options(r1_path, r2_path, out_path, batch_size, trim_to, false)
    }

    /// Create a new prefetching I/O handler with full options.
    ///
    /// This constructor provides control over all reader options including
    /// quality score preservation for sequence output.
    ///
    /// # Arguments
    /// * `r1_path` - Path to the first read file (FASTQ/FASTA, optionally gzipped)
    /// * `r2_path` - Optional path to the second read file for paired-end
    /// * `out_path` - Optional output path (stdout if None)
    /// * `batch_size` - Number of records per batch
    /// * `trim_to` - Optional maximum sequence length
    /// * `preserve_for_output` - When true, capture quality scores for FASTQ output.
    ///   Only enable when using `--output-sequences` to avoid wasting memory.
    ///
    /// # Returns
    /// A handler that prefetches batches in a background thread.
    pub fn with_options(
        r1_path: &Path,
        r2_path: Option<&PathBuf>,
        out_path: Option<PathBuf>,
        batch_size: usize,
        trim_to: Option<usize>,
        preserve_for_output: bool,
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
            Self::reader_thread(
                r1_path,
                r2_path,
                batch_size,
                trim_to,
                preserve_for_output,
                sender,
                thread_error,
            );
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
    ///
    /// When `trim_to` is specified:
    /// - Records with R1 shorter than `trim_to` are skipped entirely
    /// - R1 sequences are truncated to `trim_to` length
    /// - R2 sequences are truncated to `min(len, trim_to)` (never skip based on R2)
    ///
    /// When `preserve_for_output` is true:
    /// - Quality scores are captured for FASTQ files (needed for `--output-sequences`)
    /// - This uses more memory, so only enable when writing sequences out
    fn reader_thread(
        r1_path: PathBuf,
        r2_path: Option<PathBuf>,
        batch_size: usize,
        trim_to: Option<usize>,
        preserve_for_output: bool,
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

            while records.len() < batch_size {
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

                // Get R1 sequence - check trim_to requirement
                let s1_seq = s1_rec.seq();
                if let Some(trim_len) = trim_to {
                    if s1_seq.len() < trim_len {
                        // R1 too short - skip this record (and its R2 pair if present)
                        if let Some(ref mut r2_reader) = r2 {
                            // Must consume R2 to keep files in sync
                            match r2_reader.next() {
                                Some(Ok(_)) => {}
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
                        }
                        global_record_id += 1;
                        continue; // Skip this record
                    }
                }

                // Helper to trim and copy a byte slice
                let trim_copy = |data: &[u8], trim_to: Option<usize>| -> Vec<u8> {
                    match trim_to {
                        Some(trim_len) => data[..trim_len.min(data.len())].to_vec(),
                        None => data.to_vec(),
                    }
                };

                // Copy R1 sequence with optional trimming
                let s1_vec = trim_copy(&s1_seq, trim_to);

                // Capture R1 quality if preserving for output and this is FASTQ
                let q1_vec = if preserve_for_output {
                    s1_rec.qual().map(|q| trim_copy(q, trim_to))
                } else {
                    None
                };

                // Handle R2 if present
                let (s2_vec, q2_vec) = if let Some(ref mut r2_reader) = r2 {
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
                            // Copy R2 sequence with optional trimming
                            let s2_seq = rec.seq();
                            let s2 = Some(trim_copy(&s2_seq, trim_to));

                            // Capture R2 quality if preserving for output
                            let q2 = if preserve_for_output {
                                rec.qual().map(|q| trim_copy(q, trim_to))
                            } else {
                                None
                            };

                            (s2, q2)
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
                    (None, None)
                };

                let base_id = Self::base_read_id(s1_rec.id());
                let header = String::from_utf8_lossy(base_id).to_string();
                // Use batch-local index (0-based within each batch) for query_id.
                // This is intentional - the classification code maps results back
                // to headers using this batch-local index.
                records.push(OwnedFastxRecord::new(
                    records.len() as i64,
                    s1_vec,
                    q1_vec,
                    s2_vec,
                    q2_vec,
                ));
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
    pub fn next_batch(&mut self) -> Result<Option<(Vec<OwnedFastxRecord>, Vec<String>)>> {
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

    /// Get mutable access to the output writer.
    #[allow(dead_code)]
    pub fn writer(&mut self) -> &mut OutputWriter {
        &mut self.writer
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
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Tests for OwnedFastxRecord
    // -------------------------------------------------------------------------

    #[test]
    fn test_owned_fastx_record_creation_single_end_fasta() {
        let record = OwnedFastxRecord::new(0, b"ACGT".to_vec(), None, None, None);
        assert_eq!(record.query_id, 0);
        assert_eq!(record.seq1, b"ACGT");
        assert!(!record.is_fastq());
        assert!(!record.is_paired());
    }

    #[test]
    fn test_owned_fastx_record_creation_paired_fastq() {
        let record = OwnedFastxRecord::new(
            1,
            b"ACGT".to_vec(),
            Some(b"IIII".to_vec()),
            Some(b"TGCA".to_vec()),
            Some(b"JJJJ".to_vec()),
        );
        assert!(record.is_fastq());
        assert!(record.is_paired());
        assert_eq!(record.qual1.as_ref().unwrap(), b"IIII");
        assert_eq!(record.seq2.as_ref().unwrap(), b"TGCA");
        assert_eq!(record.qual2.as_ref().unwrap(), b"JJJJ");
    }

    #[test]
    fn test_owned_fastx_record_sequences_tuple() {
        let record = OwnedFastxRecord::new(5, b"AA".to_vec(), None, Some(b"TT".to_vec()), None);
        let (s1, s2) = record.sequences();
        assert_eq!(s1, b"AA");
        assert_eq!(s2.unwrap(), b"TT");
    }

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
    // Tests for preserve_for_output (quality score preservation)
    // -------------------------------------------------------------------------

    #[test]
    fn test_reader_preserves_quality_when_requested() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut tmp = NamedTempFile::new().unwrap();
        writeln!(tmp, "@read1\nACGT\n+\nIIII").unwrap();
        tmp.flush().unwrap();

        // preserve_for_output = true
        let mut handler = PrefetchingIoHandler::with_options(
            tmp.path(),
            None,
            None,
            100,
            None,
            true, // preserve_for_output
        )
        .unwrap();

        let (records, _) = handler.next_batch().unwrap().unwrap();
        assert!(records[0].is_fastq());
        assert_eq!(records[0].qual1.as_ref().unwrap(), b"IIII");
    }

    #[test]
    fn test_reader_skips_quality_when_not_requested() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut tmp = NamedTempFile::new().unwrap();
        writeln!(tmp, "@read1\nACGT\n+\nIIII").unwrap();
        tmp.flush().unwrap();

        // preserve_for_output = false (default behavior)
        let mut handler =
            PrefetchingIoHandler::with_trim(tmp.path(), None, None, 100, None).unwrap();

        let (records, _) = handler.next_batch().unwrap().unwrap();
        assert!(!records[0].is_fastq()); // qual1 is None
        assert!(records[0].qual1.is_none());
    }

    #[test]
    fn test_reader_paired_quality_when_requested() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create paired FASTQ files
        let mut tmp_r1 = NamedTempFile::new().unwrap();
        writeln!(tmp_r1, "@read1/1\nACGT\n+\nIIII").unwrap();
        tmp_r1.flush().unwrap();

        let mut tmp_r2 = NamedTempFile::new().unwrap();
        writeln!(tmp_r2, "@read1/2\nTGCA\n+\nJJJJ").unwrap();
        tmp_r2.flush().unwrap();

        let r2_path = tmp_r2.path().to_path_buf();

        // preserve_for_output = true
        let mut handler = PrefetchingIoHandler::with_options(
            tmp_r1.path(),
            Some(&r2_path),
            None,
            100,
            None,
            true, // preserve_for_output
        )
        .unwrap();

        let (records, _) = handler.next_batch().unwrap().unwrap();
        assert!(records[0].is_fastq());
        assert!(records[0].is_paired());
        assert_eq!(records[0].qual1.as_ref().unwrap(), b"IIII");
        assert_eq!(records[0].qual2.as_ref().unwrap(), b"JJJJ");
        assert_eq!(records[0].seq2.as_ref().unwrap(), b"TGCA");
    }

    #[test]
    fn test_reader_fasta_has_no_quality() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut tmp = NamedTempFile::new().unwrap();
        writeln!(tmp, ">read1\nACGT").unwrap();
        tmp.flush().unwrap();

        // Even with preserve_for_output = true, FASTA has no quality
        let mut handler = PrefetchingIoHandler::with_options(
            tmp.path(),
            None,
            None,
            100,
            None,
            true, // preserve_for_output
        )
        .unwrap();

        let (records, _) = handler.next_batch().unwrap().unwrap();
        assert!(!records[0].is_fastq());
        assert!(records[0].qual1.is_none());
    }

    #[test]
    fn test_reader_quality_trimmed_with_sequence() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut tmp = NamedTempFile::new().unwrap();
        // 8bp sequence and quality
        writeln!(tmp, "@read1\nACGTACGT\n+\nIIIIJJJJ").unwrap();
        tmp.flush().unwrap();

        // preserve_for_output = true with trim_to = 4
        let mut handler = PrefetchingIoHandler::with_options(
            tmp.path(),
            None,
            None,
            100,
            Some(4), // trim_to
            true,    // preserve_for_output
        )
        .unwrap();

        let (records, _) = handler.next_batch().unwrap().unwrap();
        // Sequence should be trimmed to 4bp
        assert_eq!(records[0].seq1, b"ACGT");
        // Quality should also be trimmed to 4bp
        assert_eq!(records[0].qual1.as_ref().unwrap(), b"IIII");
    }
}
