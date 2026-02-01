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

/// Owned record type: (query_id, seq1, optional_seq2)
pub type OwnedRecord = (i64, Vec<u8>, Option<Vec<u8>>);

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
}
