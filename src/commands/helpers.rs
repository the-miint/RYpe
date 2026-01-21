//! Helper functions and utilities for the rype CLI.

use anyhow::{anyhow, Result};
use needletail::{parse_fastx_file, FastxReader};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use rype::memory::parse_byte_suffix;
use rype::{Index, IndexMetadata, MainIndexManifest};

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

/// Parse shard format argument ("legacy" or "parquet")
pub fn parse_shard_format(s: &str) -> Result<String, String> {
    match s.to_lowercase().as_str() {
        "legacy" | "ryxs" => Ok("legacy".to_string()),
        "parquet" | "pq" => Ok("parquet".to_string()),
        _ => Err(format!(
            "Unknown format '{}'. Valid options: legacy, parquet",
            s
        )),
    }
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

/// Load metadata from Parquet, sharded, or single-file indices.
///
/// This helper handles:
/// - Parquet inverted index directories (with manifest.toml)
/// - Sharded main indices (with .manifest and .shard.* files)
/// - Single-file indices (.ryidx)
pub fn load_index_metadata(path: &Path) -> Result<IndexMetadata> {
    // Check for Parquet format first (directory with manifest.toml)
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

    // Check for sharded main index
    if MainIndexManifest::is_sharded(path) {
        let manifest = MainIndexManifest::load(&MainIndexManifest::manifest_path(path))?;
        Ok(manifest.to_metadata())
    } else {
        // Single-file index
        Ok(Index::load_metadata(path)?)
    }
}

/// Type alias for batch data sent through the prefetch channel.
type BatchResult = Result<Option<(Vec<OwnedRecord>, Vec<String>)>>;

/// Shared error state for capturing errors when the receiver drops early.
type SharedError = Arc<Mutex<Option<String>>>;

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
    writer: BufWriter<Box<dyn Write>>,
    /// Shared error state for errors that occur when sender.send() fails
    shared_error: SharedError,
    /// Timeout for waiting on batches
    timeout: Duration,
}

impl PrefetchingIoHandler {
    /// Create a new prefetching I/O handler.
    ///
    /// # Arguments
    /// * `r1_path` - Path to the first read file (FASTQ/FASTA, optionally gzipped)
    /// * `r2_path` - Optional path to the second read file for paired-end
    /// * `out_path` - Optional output path (stdout if None)
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

        // Shared error state for capturing errors when send() fails
        let shared_error: SharedError = Arc::new(Mutex::new(None));
        let thread_error = Arc::clone(&shared_error);

        // Use sync_channel with buffer of 2 to allow actual prefetching:
        // - Slot 1: batch currently being processed by main thread
        // - Slot 2: next batch being prefetched by reader thread
        let (sender, receiver): (SyncSender<BatchResult>, Receiver<BatchResult>) =
            mpsc::sync_channel(2);

        // Spawn background thread for reading
        let prefetch_thread = thread::spawn(move || {
            Self::reader_thread(r1_path, r2_path, batch_size, sender, thread_error);
        });

        // Set up output writer
        let output: Box<dyn Write> = if let Some(p) = out_path {
            Box::new(File::create(p).expect("Failed to create output file"))
        } else {
            Box::new(io::stdout())
        };

        Ok(Self {
            receiver,
            prefetch_thread: Some(prefetch_thread),
            writer: BufWriter::new(output),
            shared_error,
            timeout: DEFAULT_PREFETCH_TIMEOUT,
        })
    }

    /// Extract the base read ID (before any space or /1 /2 suffix).
    fn base_read_id(id: &[u8]) -> &[u8] {
        // Find first space or /1 /2 suffix
        for (i, &b) in id.iter().enumerate() {
            if b == b' ' || b == b'\t' {
                return &id[..i];
            }
            // Check for /1 or /2 suffix (common in Illumina paired-end)
            if b == b'/' && i + 1 < id.len() && (id[i + 1] == b'1' || id[i + 1] == b'2') {
                return &id[..i];
            }
        }
        id
    }

    /// Background thread function that reads batches and sends them through the channel.
    fn reader_thread(
        r1_path: PathBuf,
        r2_path: Option<PathBuf>,
        batch_size: usize,
        sender: SyncSender<BatchResult>,
        shared_error: SharedError,
    ) {
        // Helper to store error in shared state when send fails
        let store_error = |err: &str| {
            if let Ok(mut guard) = shared_error.lock() {
                *guard = Some(err.to_string());
            }
        };

        // Open readers in the background thread
        let mut r1 = match parse_fastx_file(&r1_path) {
            Ok(r) => r,
            Err(e) => {
                let err_msg = format!("Failed to open R1 at {}: {}", r1_path.display(), e);
                if sender.send(Err(anyhow!("{}", &err_msg))).is_err() {
                    store_error(&err_msg);
                }
                return;
            }
        };

        let mut r2: Option<Box<dyn FastxReader>> = match r2_path {
            Some(ref p) => match parse_fastx_file(p) {
                Ok(r) => Some(r),
                Err(e) => {
                    let err_msg = format!("Failed to open R2 at {}: {}", p.display(), e);
                    if sender.send(Err(anyhow!("{}", &err_msg))).is_err() {
                        store_error(&err_msg);
                    }
                    return;
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
                        let err_msg =
                            format!("Error reading R1 at record {}: {}", global_record_id, e);
                        if sender.send(Err(anyhow!("{}", &err_msg))).is_err() {
                            store_error(&err_msg);
                        }
                        return;
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
                                let err_msg = format!(
                                    "R1/R2 read ID mismatch at record {}: R1='{}' R2='{}'",
                                    global_record_id,
                                    String::from_utf8_lossy(r1_base),
                                    String::from_utf8_lossy(r2_base)
                                );
                                if sender.send(Err(anyhow!("{}", &err_msg))).is_err() {
                                    store_error(&err_msg);
                                }
                                return;
                            }
                            Some(rec.seq().into_owned())
                        }
                        Some(Err(e)) => {
                            let err_msg =
                                format!("Error reading R2 at record {}: {}", global_record_id, e);
                            if sender.send(Err(anyhow!("{}", &err_msg))).is_err() {
                                store_error(&err_msg);
                            }
                            return;
                        }
                        None => {
                            let err_msg = format!(
                                "R1/R2 mismatch: R2 ended early at record {}",
                                global_record_id
                            );
                            if sender.send(Err(anyhow!("{}", &err_msg))).is_err() {
                                store_error(&err_msg);
                            }
                            return;
                        }
                    }
                } else {
                    None
                };

                let header = String::from_utf8_lossy(s1_rec.id()).to_string();
                // Use batch-local index (0-based within each batch) for query_id
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
                if let Ok(guard) = self.shared_error.lock() {
                    if let Some(ref err) = *guard {
                        return Err(anyhow!("Reader thread error: {}", err));
                    }
                }
                Err(anyhow!(
                    "Timeout waiting for next batch ({}s) - reader thread may be stalled",
                    self.timeout.as_secs()
                ))
            }
            Err(RecvTimeoutError::Disconnected) => {
                // Channel closed - check shared error state first
                if let Ok(guard) = self.shared_error.lock() {
                    if let Some(ref err) = *guard {
                        return Err(anyhow!("Reader thread error: {}", err));
                    }
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

    /// Write data to the output.
    pub fn write(&mut self, data: Vec<u8>) -> Result<()> {
        self.writer.write_all(&data)?;
        Ok(())
    }

    /// Flush the output and wait for the prefetch thread to complete.
    pub fn finish(&mut self) -> Result<()> {
        self.writer.flush()?;

        // Wait for the prefetch thread to complete
        if let Some(handle) = self.prefetch_thread.take() {
            handle
                .join()
                .map_err(|_| anyhow!("Prefetch thread panicked"))?;
        }

        Ok(())
    }
}
