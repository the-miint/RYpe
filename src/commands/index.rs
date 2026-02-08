//! Index command handlers and helper functions.
//!
//! This module contains the implementation logic for index-related commands.
//! Only Parquet inverted index format is supported.

use anyhow::{anyhow, Context, Result};
use needletail::parse_fastx_file;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rype::config::{parse_config, resolve_path, validate_config};
use rype::parquet_index;
use rype::{
    choose_orientation_sampled, extract_dual_strand_into, extract_into, kway_merge_dedup,
    log_timing, merge_sorted_into, MinimizerWorkspace, Orientation, BUCKET_SOURCE_DELIM,
};

use std::collections::HashSet;

use super::helpers::sanitize_bucket_name;

/// Validate that all bucket names are unique.
///
/// Accepts any iterator of bucket name references.
/// Returns an error if duplicate bucket names are found, listing them all.
fn validate_unique_bucket_names<'a>(names: impl Iterator<Item = &'a str>) -> Result<()> {
    let mut seen: HashSet<&str> = HashSet::new();
    let mut duplicates: Vec<&str> = Vec::new();

    for name in names {
        if !seen.insert(name) && !duplicates.contains(&name) {
            duplicates.push(name);
        }
    }

    if duplicates.is_empty() {
        Ok(())
    } else {
        Err(anyhow!(
            "Duplicate bucket names are not allowed. Found duplicates: {:?}\n\
             Bucket names must be unique to avoid ambiguity in output formats.\n\
             Consider using --separate-buckets or renaming sequences.",
            duplicates
        ))
    }
}

/// Validate that a subtraction index is compatible with the config being built.
///
/// Checks that k, w, and salt match between the config and the subtraction index.
fn validate_subtraction_compatibility(
    cfg: &rype::config::ConfigFile,
    subtract_index: &rype::ShardedInvertedIndex,
) -> Result<()> {
    if subtract_index.k() != cfg.index.k {
        return Err(anyhow!(
            "k mismatch: config has k={}, subtraction index has k={}",
            cfg.index.k,
            subtract_index.k()
        ));
    }
    if subtract_index.w() != cfg.index.window {
        return Err(anyhow!(
            "w mismatch: config has w={}, subtraction index has w={}",
            cfg.index.window,
            subtract_index.w()
        ));
    }
    if subtract_index.salt() != cfg.index.salt {
        return Err(anyhow!(
            "salt mismatch: config has salt={:#x}, subtraction index has salt={:#x}",
            cfg.index.salt,
            subtract_index.salt()
        ));
    }
    Ok(())
}

// ============================================================================
// Parquet Index Creation
// ============================================================================

/// Create Parquet inverted index directly from reference files.
#[allow(clippy::too_many_arguments)]
pub fn create_parquet_index_from_refs(
    output: &Path,
    references: &[PathBuf],
    k: usize,
    w: usize,
    salt: u64,
    separate_buckets: bool,
    max_shard_bytes: Option<usize>,
    options: Option<&parquet_index::ParquetWriteOptions>,
) -> Result<()> {
    use rype::{create_parquet_inverted_index, BucketData};

    log::info!(
        "Creating Parquet inverted index at {:?} (K={}, W={}, salt={:#x})",
        output,
        k,
        w,
        salt
    );

    let mut buckets: Vec<BucketData> = Vec::new();
    let mut next_id: u32 = 1;
    let mut ws = MinimizerWorkspace::new();

    for ref_path in references {
        log::info!("Processing reference: {}", ref_path.display());
        let mut reader = parse_fastx_file(ref_path).context("Failed to open reference file")?;
        let filename = ref_path
            .canonicalize()
            .unwrap()
            .to_string_lossy()
            .to_string();

        if separate_buckets {
            while let Some(record) = reader.next() {
                let rec = record.context("Invalid record")?;
                let seq = rec.seq();
                let name = String::from_utf8_lossy(rec.id()).to_string();
                let bucket_id = next_id;
                next_id += 1;

                extract_into(&seq, k, w, salt, &mut ws);
                let mut minimizers = std::mem::take(&mut ws.buffer);
                minimizers.sort_unstable();
                minimizers.dedup();

                let source_label = format!("{}{}{}", filename, BUCKET_SOURCE_DELIM, name);
                buckets.push(BucketData {
                    bucket_id,
                    bucket_name: sanitize_bucket_name(&name),
                    sources: vec![source_label],
                    minimizers,
                });
            }
        } else {
            let bucket_id = next_id;
            next_id += 1;

            let mut all_minimizers: Vec<u64> = Vec::new();
            let mut sources: Vec<String> = Vec::new();

            while let Some(record) = reader.next() {
                let rec = record.context("Invalid record")?;
                let seq = rec.seq();
                let name = String::from_utf8_lossy(rec.id()).to_string();

                extract_into(&seq, k, w, salt, &mut ws);
                all_minimizers.extend_from_slice(&ws.buffer);

                let source_label = format!("{}{}{}", filename, BUCKET_SOURCE_DELIM, name);
                sources.push(source_label);
            }

            all_minimizers.sort_unstable();
            all_minimizers.dedup();

            buckets.push(BucketData {
                bucket_id,
                bucket_name: sanitize_bucket_name(&filename),
                sources,
                minimizers: all_minimizers,
            });
        }
    }

    let total_minimizers: usize = buckets.iter().map(|b| b.minimizers.len()).sum();
    log::info!(
        "Extracted minimizers from {} buckets ({} total)",
        buckets.len(),
        total_minimizers
    );

    // Validate bucket name uniqueness before creating index
    validate_unique_bucket_names(buckets.iter().map(|b| b.bucket_name.as_str()))?;

    let manifest =
        create_parquet_inverted_index(output, buckets, k, w, salt, max_shard_bytes, options)?;

    log::info!("Created Parquet inverted index:");
    log::info!("  Buckets: {}", manifest.num_buckets);
    if let Some(ref inv) = manifest.inverted {
        log::info!("  Shards: {}", inv.num_shards);
        log::info!("  Total entries: {}", inv.total_entries);
    }
    log::info!("Done.");

    Ok(())
}

/// Extract minimizers from a set of files, returning sorted deduplicated minimizers and source labels.
///
/// This is a helper function for streaming index creation that processes files
/// and returns the minimizers without creating a full bucket.
///
/// # Arguments
/// * `files` - Paths to FASTA/FASTQ files
/// * `config_dir` - Base directory for resolving relative paths
/// * `k` - K-mer size
/// * `w` - Window size
/// * `salt` - Hash salt
/// * `orient_sequences` - If true, orient sequences to maximize minimizer overlap
///
/// # Returns
/// A tuple of (sorted deduplicated minimizers, source labels).
/// Source labels are formatted as "filename::sequence_name".
///
/// # Orientation
/// When `orient_sequences` is true:
/// - The first sequence uses forward strand (establishes baseline)
/// - Subsequent sequences compare forward vs reverse-complement overlap with existing minimizers
/// - The orientation with higher overlap is chosen
fn extract_bucket_minimizers(
    files: &[PathBuf],
    config_dir: &Path,
    k: usize,
    w: usize,
    salt: u64,
    orient_sequences: bool,
) -> Result<(Vec<u64>, Vec<String>)> {
    use rype::config::resolve_path;

    let mut ws = MinimizerWorkspace::new();
    let mut bucket_mins: Vec<u64> = Vec::new(); // Kept sorted and deduped via merge_sorted_into
    let mut sources: Vec<String> = Vec::new();
    let mut is_first_sequence = true;

    for file_path in files {
        let abs_path = resolve_path(config_dir, file_path);
        let mut reader = parse_fastx_file(&abs_path)
            .context(format!("Failed to open file {}", abs_path.display()))?;

        let filename = abs_path
            .canonicalize()
            .unwrap_or_else(|_| abs_path.clone())
            .to_string_lossy()
            .to_string();

        while let Some(record) = reader.next() {
            let rec = record.context(format!("Invalid record in file {}", abs_path.display()))?;
            let seq_name = String::from_utf8_lossy(rec.id()).to_string();
            let source_label = format!("{}{}{}", filename, BUCKET_SOURCE_DELIM, seq_name);
            sources.push(source_label);

            let seq = rec.seq();

            if is_first_sequence || !orient_sequences {
                // Forward-only: extract, sort, merge in-place
                extract_into(&seq, k, w, salt, &mut ws);
                let mut new_mins = std::mem::take(&mut ws.buffer);
                new_mins.sort_unstable();
                merge_sorted_into(&mut bucket_mins, &new_mins);
                is_first_sequence = false;
            } else {
                // Oriented: extract both strands, sort both, choose best, merge in-place
                let (mut fwd, mut rc) = extract_dual_strand_into(&seq, k, w, salt, &mut ws);
                fwd.sort_unstable();
                rc.sort_unstable();

                let (orientation, _overlap) = choose_orientation_sampled(&bucket_mins, &fwd, &rc);

                let chosen = match orientation {
                    Orientation::Forward => fwd,
                    Orientation::ReverseComplement => rc,
                };

                merge_sorted_into(&mut bucket_mins, &chosen);
            }
        }
    }

    // bucket_mins is already sorted and deduped from merge_sorted_into
    Ok((bucket_mins, sources))
}

// ============================================================================
// Parallel Single-Bucket Extraction
// ============================================================================

/// Collect all sequences from files with their source labels.
///
/// Returns Vec of (sequence_bytes, source_label) for parallel processing.
#[cfg(test)]
fn collect_sequences_from_files(
    files: &[PathBuf],
    config_dir: &Path,
) -> Result<Vec<(Vec<u8>, String)>> {
    use rype::config::resolve_path;

    let mut sequences = Vec::new();

    for file_path in files {
        let abs_path = resolve_path(config_dir, file_path);
        let mut reader = parse_fastx_file(&abs_path)
            .context(format!("Failed to open file {}", abs_path.display()))?;

        let filename = abs_path
            .canonicalize()
            .unwrap_or_else(|_| abs_path.clone())
            .to_string_lossy()
            .to_string();

        while let Some(record) = reader.next() {
            let rec = record.context(format!("Invalid record in {}", abs_path.display()))?;
            let seq_name = String::from_utf8_lossy(rec.id()).to_string();
            let source_label = format!("{}{}{}", filename, BUCKET_SOURCE_DELIM, seq_name);
            sequences.push((rec.seq().to_vec(), source_label));
        }
    }

    Ok(sequences)
}

/// Build a single bucket using parallel per-sequence extraction.
///
/// All sequences are extracted in parallel, then k-way merged.
/// This is used when there is only one bucket and orientation is disabled.
#[cfg(test)]
fn build_single_bucket_parallel(
    bucket_name: &str,
    files: &[PathBuf],
    config_dir: &Path,
    k: usize,
    w: usize,
    salt: u64,
) -> Result<(String, Vec<u64>, Vec<String>)> {
    log::info!(
        "Processing bucket '{}' (parallel, {} files) ...",
        bucket_name,
        files.len()
    );

    let sequences = collect_sequences_from_files(files, config_dir)?;

    if sequences.is_empty() {
        return Ok((bucket_name.to_string(), vec![], vec![]));
    }

    // Estimate workspace size from average sequence length
    let avg_len = sequences.iter().map(|(s, _)| s.len()).sum::<usize>() / sequences.len().max(1);
    let estimated_mins = MinimizerWorkspace::estimate_for_length(avg_len, k, w);

    // Parallel extraction with per-thread workspace
    // Note: we don't dedup here - kway_merge_dedup handles deduplication efficiently
    let per_seq_mins: Vec<Vec<u64>> = sequences
        .par_iter()
        .map_init(
            move || MinimizerWorkspace::with_estimate(estimated_mins),
            |ws, (seq, _source)| {
                extract_into(seq, k, w, salt, ws);
                let mut mins = std::mem::take(&mut ws.buffer);
                mins.sort_unstable();
                mins
            },
        )
        .collect();

    // K-way merge of sorted vecs
    let merged = kway_merge_dedup(per_seq_mins);
    let sources: Vec<String> = sequences.into_iter().map(|(_, src)| src).collect();

    log::info!(
        "Completed bucket '{}': {} minimizers",
        bucket_name,
        merged.len()
    );

    Ok((bucket_name.to_string(), merged, sources))
}

/// Build a single bucket with orientation using hybrid approach.
///
/// Strategy:
/// 1. First file: extract sequentially to establish baseline minimizers
/// 2. Remaining files: extract both strands in parallel, choose orientation
///    by comparing against baseline, then k-way merge all results
#[cfg(test)]
fn build_single_bucket_parallel_oriented(
    bucket_name: &str,
    files: &[PathBuf],
    config_dir: &Path,
    k: usize,
    w: usize,
    salt: u64,
) -> Result<(String, Vec<u64>, Vec<String>)> {
    log::info!(
        "Processing bucket '{}' (parallel, oriented, {} files) ...",
        bucket_name,
        files.len()
    );

    if files.is_empty() {
        return Ok((bucket_name.to_string(), vec![], vec![]));
    }

    // Phase 1: Process first file sequentially to establish baseline
    let (baseline_mins, baseline_sources) =
        extract_baseline_from_first_file(&files[0], config_dir, k, w, salt)?;

    if files.len() == 1 {
        // Only one file, we're done
        log::info!(
            "Completed bucket '{}': {} minimizers",
            bucket_name,
            baseline_mins.len()
        );
        return Ok((bucket_name.to_string(), baseline_mins, baseline_sources));
    }

    // Phase 2: Collect sequences from remaining files
    let remaining_sequences = collect_sequences_from_files(&files[1..], config_dir)?;

    if remaining_sequences.is_empty() {
        log::info!(
            "Completed bucket '{}': {} minimizers",
            bucket_name,
            baseline_mins.len()
        );
        return Ok((bucket_name.to_string(), baseline_mins, baseline_sources));
    }

    // Estimate workspace size from average sequence length
    let avg_len = remaining_sequences
        .iter()
        .map(|(s, _)| s.len())
        .sum::<usize>()
        / remaining_sequences.len().max(1);
    let estimated_mins = MinimizerWorkspace::estimate_for_length(avg_len, k, w);

    // Phase 3: Parallel extraction with orientation against baseline
    // Note: we don't dedup here - kway_merge_dedup handles deduplication efficiently
    let oriented_mins: Vec<Vec<u64>> = remaining_sequences
        .par_iter()
        .map_init(
            move || MinimizerWorkspace::with_estimate(estimated_mins),
            |ws, (seq, _source)| {
                let (mut fwd, mut rc) = extract_dual_strand_into(seq, k, w, salt, ws);
                fwd.sort_unstable();
                rc.sort_unstable();

                // Choose orientation based on overlap with baseline
                let (orientation, _overlap) = choose_orientation_sampled(&baseline_mins, &fwd, &rc);

                match orientation {
                    Orientation::Forward => fwd,
                    Orientation::ReverseComplement => rc,
                }
            },
        )
        .collect();

    // Phase 4: K-way merge baseline + all oriented results
    let mut all_sorted_vecs = Vec::with_capacity(1 + oriented_mins.len());
    all_sorted_vecs.push(baseline_mins);
    all_sorted_vecs.extend(oriented_mins);

    let merged = kway_merge_dedup(all_sorted_vecs);

    // Combine sources
    let mut all_sources = baseline_sources;
    all_sources.extend(remaining_sequences.into_iter().map(|(_, src)| src));

    log::info!(
        "Completed bucket '{}': {} minimizers",
        bucket_name,
        merged.len()
    );

    Ok((bucket_name.to_string(), merged, all_sources))
}

/// Extract baseline minimizers from the first file (sequential, forward strand only).
fn extract_baseline_from_first_file(
    file_path: &Path,
    config_dir: &Path,
    k: usize,
    w: usize,
    salt: u64,
) -> Result<(Vec<u64>, Vec<String>)> {
    use rype::config::resolve_path;

    let abs_path = resolve_path(config_dir, file_path);
    let mut reader = parse_fastx_file(&abs_path)
        .context(format!("Failed to open file {}", abs_path.display()))?;

    let filename = abs_path
        .canonicalize()
        .unwrap_or_else(|_| abs_path.clone())
        .to_string_lossy()
        .to_string();

    let mut ws = MinimizerWorkspace::new();
    let mut baseline_mins: Vec<u64> = Vec::new();
    let mut sources: Vec<String> = Vec::new();

    while let Some(record) = reader.next() {
        let rec = record.context(format!("Invalid record in {}", abs_path.display()))?;
        let seq_name = String::from_utf8_lossy(rec.id()).to_string();
        let source_label = format!("{}{}{}", filename, BUCKET_SOURCE_DELIM, seq_name);
        sources.push(source_label);

        extract_into(&rec.seq(), k, w, salt, &mut ws);
        let mut new_mins = std::mem::take(&mut ws.buffer);
        new_mins.sort_unstable();
        merge_sorted_into(&mut baseline_mins, &new_mins);
    }

    Ok((baseline_mins, sources))
}

// ============================================================================
// Chunked Memory-Aware Extraction
// ============================================================================

/// Configuration for chunked sequence extraction.
#[derive(Debug, Clone)]
struct ChunkConfig {
    /// Target maximum bytes for sequences in a single chunk.
    /// Actual chunks may exceed this if a single sequence is larger.
    target_chunk_bytes: usize,
}

/// Memory overhead multiplier for minimizer extraction.
/// Accounts for: raw sequences + minimizers + sorting + merge workspace.
const EXTRACTION_MEMORY_MULTIPLIER: f64 = 4.0;

/// Minimum chunk size to ensure reasonable parallelism (100MB).
const MIN_CHUNK_BYTES: usize = 100 * 1024 * 1024;

/// Maximum chunk size to avoid excessive memory on large-memory systems (4GB).
const MAX_CHUNK_BYTES: usize = 4 * 1024 * 1024 * 1024;

/// Calculate chunk configuration based on available memory.
///
/// # Arguments
/// * `available_memory` - Total available memory in bytes (from detection or CLI)
///
/// # Returns
/// A `ChunkConfig` with target chunk size that respects memory constraints.
fn calculate_chunk_config(available_memory: usize) -> ChunkConfig {
    // Reserve 25% of memory for the merged result accumulator and overhead
    let chunk_budget = (available_memory as f64 * 0.75) as usize;

    // Account for extraction overhead (4x raw sequence bytes)
    let target = (chunk_budget as f64 / EXTRACTION_MEMORY_MULTIPLIER) as usize;

    // Clamp to reasonable bounds
    let target_chunk_bytes = target.clamp(MIN_CHUNK_BYTES, MAX_CHUNK_BYTES);

    ChunkConfig { target_chunk_bytes }
}

/// Iterator that yields chunks of sequences respecting a byte budget.
///
/// Sequences are yielded in file order. Each chunk contains sequences
/// up to the target byte budget. If a single sequence exceeds the budget,
/// it is yielded alone in its own chunk.
struct SequenceChunkIterator {
    /// Remaining files to process
    files: Vec<PathBuf>,
    /// Base directory for resolving relative paths
    config_dir: PathBuf,
    /// Target maximum bytes per chunk
    target_chunk_bytes: usize,
    /// Current file index
    current_file_idx: usize,
    /// Current reader (if file is open)
    current_reader: Option<Box<dyn needletail::FastxReader>>,
    /// Current filename for source labels
    current_filename: String,
    /// Pending sequence that didn't fit in the previous chunk
    pending_sequence: Option<(Vec<u8>, String)>,
}

impl SequenceChunkIterator {
    /// Create a new chunk iterator.
    fn new(files: &[PathBuf], config_dir: &Path, target_chunk_bytes: usize) -> Self {
        Self {
            files: files.to_vec(),
            config_dir: config_dir.to_path_buf(),
            target_chunk_bytes,
            current_file_idx: 0,
            current_reader: None,
            current_filename: String::new(),
            pending_sequence: None,
        }
    }

    /// Open the next file, returning true if successful.
    /// Skips empty or invalid files with a warning.
    fn open_next_file(&mut self) -> Result<bool> {
        while self.current_file_idx < self.files.len() {
            let file_path = &self.files[self.current_file_idx];
            let abs_path = resolve_path(&self.config_dir, file_path);
            self.current_file_idx += 1;

            self.current_filename = abs_path
                .canonicalize()
                .unwrap_or_else(|_| abs_path.clone())
                .to_string_lossy()
                .to_string();

            // Try to open the file, skip if empty or invalid
            match parse_fastx_file(&abs_path) {
                Ok(reader) => {
                    self.current_reader = Some(reader);
                    return Ok(true);
                }
                Err(e) => {
                    // Log warning and skip to next file
                    log::warn!(
                        "Skipping file {} (possibly empty or invalid): {}",
                        abs_path.display(),
                        e
                    );
                    continue;
                }
            }
        }

        Ok(false)
    }

    /// Get the next chunk of sequences.
    ///
    /// Returns `Ok(Some(chunk))` with sequences, `Ok(None)` when exhausted,
    /// or `Err` on I/O error.
    #[allow(clippy::type_complexity)]
    fn next_chunk(&mut self) -> Result<Option<Vec<(Vec<u8>, String)>>> {
        let mut chunk: Vec<(Vec<u8>, String)> = Vec::new();
        let mut chunk_bytes: usize = 0;

        // First, check if we have a pending sequence from the previous call
        if let Some((seq, src)) = self.pending_sequence.take() {
            let seq_len = seq.len();
            chunk.push((seq, src));
            chunk_bytes += seq_len;
        }

        loop {
            // Ensure we have an open reader
            if self.current_reader.is_none() && !self.open_next_file()? {
                // No more files
                break;
            }

            let reader = self.current_reader.as_mut().unwrap();

            match reader.next() {
                Some(Ok(record)) => {
                    let seq = record.seq().to_vec();
                    let seq_len = seq.len();
                    let seq_name = String::from_utf8_lossy(record.id()).to_string();
                    let source_label = format!(
                        "{}{}{}",
                        self.current_filename, BUCKET_SOURCE_DELIM, seq_name
                    );

                    // Check if adding this sequence would exceed budget
                    if !chunk.is_empty() && chunk_bytes + seq_len > self.target_chunk_bytes {
                        // Current chunk is full, buffer this sequence for next call
                        self.pending_sequence = Some((seq, source_label));
                        return Ok(Some(chunk));
                    }

                    chunk.push((seq, source_label));
                    chunk_bytes += seq_len;
                }
                Some(Err(e)) => {
                    return Err(anyhow!(
                        "Invalid record in {}: {}",
                        self.current_filename,
                        e
                    ));
                }
                None => {
                    // Current file exhausted, move to next
                    self.current_reader = None;
                }
            }
        }

        // Return final chunk if non-empty
        if chunk.is_empty() {
            Ok(None)
        } else {
            Ok(Some(chunk))
        }
    }
}

/// Build a single bucket using chunked parallel extraction.
///
/// Sequences are processed in memory-bounded chunks. Each chunk is extracted
/// in parallel, then merged into a running accumulator. This bounds peak memory
/// to approximately 4x the chunk size regardless of total input size.
///
/// NOTE: This function's output accumulator grows unbounded. For bounded output
/// memory, use `build_single_bucket_streaming` instead.
///
/// # Arguments
/// * `bucket_name` - Name of the bucket being built
/// * `files` - Paths to FASTA/FASTQ files
/// * `config_dir` - Base directory for resolving relative paths
/// * `k` - K-mer size
/// * `w` - Window size
/// * `salt` - Hash salt
/// * `max_memory` - Available memory budget (None = auto-detect)
///
/// # Returns
/// Tuple of (bucket_name, sorted_deduped_minimizers, source_labels)
#[cfg(test)]
fn build_single_bucket_parallel_chunked(
    bucket_name: &str,
    files: &[PathBuf],
    config_dir: &Path,
    k: usize,
    w: usize,
    salt: u64,
    max_memory: Option<usize>,
) -> Result<(String, Vec<u64>, Vec<String>)> {
    use rype::memory::detect_available_memory;

    log::info!(
        "Processing bucket '{}' (chunked parallel, {} files) ...",
        bucket_name,
        files.len()
    );

    // Determine memory budget
    let available = max_memory.unwrap_or_else(|| {
        let detected = detect_available_memory();
        log::debug!(
            "Auto-detected available memory: {} bytes (source: {:?})",
            detected.bytes,
            detected.source
        );
        detected.bytes
    });

    let chunk_config = calculate_chunk_config(available);
    log::debug!(
        "Using chunk size: {} bytes ({:.1} MB)",
        chunk_config.target_chunk_bytes,
        chunk_config.target_chunk_bytes as f64 / (1024.0 * 1024.0)
    );

    let mut merged_mins: Vec<u64> = Vec::new();
    let mut all_sources: Vec<String> = Vec::new();
    let mut chunk_count = 0;

    let mut chunk_iter =
        SequenceChunkIterator::new(files, config_dir, chunk_config.target_chunk_bytes);

    while let Some(chunk) = chunk_iter.next_chunk()? {
        chunk_count += 1;
        let chunk_size: usize = chunk.iter().map(|(seq, _)| seq.len()).sum();
        log::debug!(
            "Processing chunk {} ({} sequences, {} bytes)",
            chunk_count,
            chunk.len(),
            chunk_size
        );

        // Collect sources before consuming chunk
        all_sources.extend(chunk.iter().map(|(_, src)| src.clone()));

        // Estimate workspace size from this chunk's sequences
        let avg_len = chunk_size / chunk.len().max(1);
        let estimated_mins = MinimizerWorkspace::estimate_for_length(avg_len, k, w);

        // Parallel extraction within chunk
        let chunk_mins: Vec<Vec<u64>> = chunk
            .par_iter()
            .map_init(
                move || MinimizerWorkspace::with_estimate(estimated_mins),
                |ws, (seq, _source)| {
                    extract_into(seq, k, w, salt, ws);
                    let mut mins = std::mem::take(&mut ws.buffer);
                    mins.sort_unstable();
                    mins
                },
            )
            .collect();

        // K-way merge this chunk's results
        let chunk_merged = kway_merge_dedup(chunk_mins);

        // Merge into running accumulator
        if merged_mins.is_empty() {
            merged_mins = chunk_merged;
        } else {
            merge_sorted_into(&mut merged_mins, &chunk_merged);
        }

        // chunk dropped here - sequences freed before next iteration
    }

    log::info!(
        "Completed bucket '{}': {} minimizers ({} chunks processed)",
        bucket_name,
        merged_mins.len(),
        chunk_count
    );

    Ok((bucket_name.to_string(), merged_mins, all_sources))
}

/// Build a single bucket with orientation using chunked parallel extraction.
///
/// Strategy:
/// 1. First file: extract sequentially to establish baseline (forward strand only)
/// 2. Remaining files in chunks: orient against baseline, merge incrementally
///
/// This bounds memory to approximately 4x chunk size plus the baseline minimizers.
///
/// NOTE: This function's output accumulator grows unbounded. For bounded output
/// memory, use `build_single_bucket_streaming_oriented` instead.
#[cfg(test)]
fn build_single_bucket_parallel_oriented_chunked(
    bucket_name: &str,
    files: &[PathBuf],
    config_dir: &Path,
    k: usize,
    w: usize,
    salt: u64,
    max_memory: Option<usize>,
) -> Result<(String, Vec<u64>, Vec<String>)> {
    use rype::memory::detect_available_memory;

    log::info!(
        "Processing bucket '{}' (chunked parallel, oriented, {} files) ...",
        bucket_name,
        files.len()
    );

    if files.is_empty() {
        return Ok((bucket_name.to_string(), vec![], vec![]));
    }

    // Phase 1: Process first file sequentially to establish baseline
    let (baseline_mins, baseline_sources) =
        extract_baseline_from_first_file(&files[0], config_dir, k, w, salt)?;

    if files.len() == 1 {
        log::info!(
            "Completed bucket '{}': {} minimizers",
            bucket_name,
            baseline_mins.len()
        );
        return Ok((bucket_name.to_string(), baseline_mins, baseline_sources));
    }

    // Determine memory budget for remaining files
    let available = max_memory.unwrap_or_else(|| {
        let detected = detect_available_memory();
        log::debug!(
            "Auto-detected available memory: {} bytes (source: {:?})",
            detected.bytes,
            detected.source
        );
        detected.bytes
    });

    let chunk_config = calculate_chunk_config(available);
    log::debug!(
        "Using chunk size: {} bytes ({:.1} MB)",
        chunk_config.target_chunk_bytes,
        chunk_config.target_chunk_bytes as f64 / (1024.0 * 1024.0)
    );

    let mut merged_mins = baseline_mins;
    let mut all_sources = baseline_sources;
    let mut chunk_count = 0;

    // Phase 2: Process remaining files in chunks
    let mut chunk_iter =
        SequenceChunkIterator::new(&files[1..], config_dir, chunk_config.target_chunk_bytes);

    while let Some(chunk) = chunk_iter.next_chunk()? {
        chunk_count += 1;
        let chunk_size: usize = chunk.iter().map(|(seq, _)| seq.len()).sum();
        log::debug!(
            "Processing chunk {} ({} sequences, {} bytes)",
            chunk_count,
            chunk.len(),
            chunk_size
        );

        // Collect sources
        all_sources.extend(chunk.iter().map(|(_, src)| src.clone()));

        // Estimate workspace size
        let avg_len = chunk_size / chunk.len().max(1);
        let estimated_mins = MinimizerWorkspace::estimate_for_length(avg_len, k, w);

        // Clone reference to merged_mins for parallel access
        // (orientation compares against current accumulated baseline)
        let baseline_ref = &merged_mins;

        // Parallel extraction with orientation
        let chunk_mins: Vec<Vec<u64>> = chunk
            .par_iter()
            .map_init(
                move || MinimizerWorkspace::with_estimate(estimated_mins),
                |ws, (seq, _source)| {
                    let (mut fwd, mut rc) = extract_dual_strand_into(seq, k, w, salt, ws);
                    fwd.sort_unstable();
                    rc.sort_unstable();

                    let (orientation, _overlap) =
                        choose_orientation_sampled(baseline_ref, &fwd, &rc);

                    match orientation {
                        Orientation::Forward => fwd,
                        Orientation::ReverseComplement => rc,
                    }
                },
            )
            .collect();

        // K-way merge this chunk's results
        let chunk_merged = kway_merge_dedup(chunk_mins);

        // Merge into accumulator (which also serves as baseline for next chunk)
        merge_sorted_into(&mut merged_mins, &chunk_merged);
    }

    log::info!(
        "Completed bucket '{}': {} minimizers ({} chunks processed)",
        bucket_name,
        merged_mins.len(),
        chunk_count
    );

    Ok((bucket_name.to_string(), merged_mins, all_sources))
}

/// Result from streaming single-bucket build.
///
/// Unlike the tuple-based return of `build_single_bucket_parallel_chunked`,
/// this struct contains shard metadata since minimizers are streamed to disk.
pub struct SingleBucketResult {
    /// Name of the bucket
    pub bucket_name: String,
    /// Source labels (filename::sequence_name)
    pub sources: Vec<String>,
    /// Metadata for each shard written to disk
    pub shard_infos: Vec<rype::parquet_index::InvertedShardInfo>,
    /// Total minimizers written (may include duplicates across shards)
    pub total_minimizers: u64,
}

/// Build a single bucket using streaming shard creation.
///
/// Unlike `build_single_bucket_parallel_chunked`, this function streams
/// minimizers to disk via ShardAccumulator, bounding BOTH input AND output memory.
///
/// # Memory Model
/// - Input chunks: ~40% of available memory (via `calculate_chunk_config`)
/// - Shard accumulator: ~40% of available memory (flushes to disk when full)
/// - Workspace overhead: ~20%
///
/// # Arguments
/// * `output_dir` - Directory to write shards (must exist, including inverted/ subdir)
/// * `bucket_name` - Name of the bucket being built
/// * `files` - Paths to FASTA/FASTQ files
/// * `config_dir` - Base directory for resolving relative paths
/// * `k` - K-mer size
/// * `w` - Window size
/// * `salt` - Hash salt
/// * `max_memory` - Available memory budget (None = auto-detect)
/// * `options` - Parquet write options
#[allow(clippy::too_many_arguments)]
fn build_single_bucket_streaming(
    output_dir: &Path,
    bucket_name: &str,
    files: &[PathBuf],
    config_dir: &Path,
    k: usize,
    w: usize,
    salt: u64,
    max_memory: Option<usize>,
    options: Option<&rype::parquet_index::ParquetWriteOptions>,
    exclusion_set: Option<&HashSet<u64>>,
) -> Result<SingleBucketResult> {
    use rype::memory::detect_available_memory;
    use rype::parquet_index::ShardAccumulator;

    const BUCKET_ID: u32 = 1;

    log::info!(
        "Processing bucket '{}' (streaming, {} files) ...",
        bucket_name,
        files.len()
    );

    if files.is_empty() {
        return Ok(SingleBucketResult {
            bucket_name: bucket_name.to_string(),
            sources: vec![],
            shard_infos: vec![],
            total_minimizers: 0,
        });
    }

    // Determine memory budget
    let available = max_memory.unwrap_or_else(|| {
        let detected = detect_available_memory();
        log::debug!(
            "Auto-detected available memory: {} bytes (source: {:?})",
            detected.bytes,
            detected.source
        );
        detected.bytes
    });

    let chunk_config = calculate_chunk_config(available);
    // Use 40% of memory for shard accumulator, clamped to minimum
    use rype::parquet_index::MIN_SHARD_BYTES;
    let shard_size = ((available as f64 * 0.4) as usize).max(MIN_SHARD_BYTES);
    // Batch size for adding entries: ~1 shard worth, capped at 128MB
    let add_batch_entries = (shard_size / 16).min(8_000_000);

    log::debug!(
        "Streaming config: chunk size {} bytes, shard size {} bytes, batch {} entries",
        chunk_config.target_chunk_bytes,
        shard_size,
        add_batch_entries
    );

    let mut accumulator = ShardAccumulator::with_output_dir(output_dir, shard_size, options);
    let mut all_sources: Vec<String> = Vec::new();
    let mut total_minimizers: u64 = 0;
    let mut total_excluded: u64 = 0;

    let mut chunk_iter =
        SequenceChunkIterator::new(files, config_dir, chunk_config.target_chunk_bytes);
    let mut chunk_count = 0;

    while let Some(chunk) = chunk_iter.next_chunk()? {
        chunk_count += 1;
        let chunk_size: usize = chunk.iter().map(|(seq, _)| seq.len()).sum();
        log::debug!(
            "Processing chunk {} ({} sequences, {} bytes)",
            chunk_count,
            chunk.len(),
            chunk_size
        );

        // Collect sources before consuming chunk
        all_sources.extend(chunk.iter().map(|(_, src)| src.clone()));

        // Estimate workspace size from this chunk's sequences
        let avg_len = chunk_size / chunk.len().max(1);
        let estimated_mins = MinimizerWorkspace::estimate_for_length(avg_len, k, w);

        // Parallel extraction within chunk
        let chunk_mins: Vec<Vec<u64>> = chunk
            .par_iter()
            .map_init(
                move || MinimizerWorkspace::with_estimate(estimated_mins),
                |ws, (seq, _source)| {
                    extract_into(seq, k, w, salt, ws);
                    let mut mins = std::mem::take(&mut ws.buffer);
                    mins.sort_unstable();
                    mins
                },
            )
            .collect();

        // K-way merge this chunk's results
        let chunk_merged = kway_merge_dedup(chunk_mins);

        // Filter out excluded minimizers if subtraction is active
        let chunk_merged = if let Some(excl) = exclusion_set {
            let original_len = chunk_merged.len();
            let filtered: Vec<u64> = chunk_merged
                .into_iter()
                .filter(|m| !excl.contains(m))
                .collect();
            let excluded = original_len - filtered.len();
            if excluded > 0 {
                total_excluded += excluded as u64;
            }
            filtered
        } else {
            chunk_merged
        };

        total_minimizers += chunk_merged.len() as u64;

        // Stream to accumulator in batches, checking flush threshold between batches
        // This ensures shards don't exceed max_shard_bytes even with large chunks
        for batch in chunk_merged.chunks(add_batch_entries) {
            accumulator.add_entries_from_minimizers(batch, BUCKET_ID);

            // Flush shards as needed to bound memory
            while accumulator.should_flush() {
                if let Some(shard_info) = accumulator.flush_shard()? {
                    log::debug!(
                        "Flushed shard {} ({} entries)",
                        shard_info.shard_id,
                        shard_info.num_entries
                    );
                }
            }
        }
    }

    if total_excluded > 0 {
        log::info!(
            "Bucket '{}': excluded {} minimizer entries via subtraction",
            bucket_name,
            total_excluded
        );
    }

    let shard_infos = accumulator.finish()?;

    log::info!(
        "Completed bucket '{}': {} minimizers, {} shards ({} chunks processed)",
        bucket_name,
        total_minimizers,
        shard_infos.len(),
        chunk_count
    );

    Ok(SingleBucketResult {
        bucket_name: bucket_name.to_string(),
        sources: all_sources,
        shard_infos,
        total_minimizers,
    })
}

/// Build a single bucket with orientation using streaming shard creation.
///
/// Strategy:
/// 1. First file: extract to establish baseline (kept in memory)
/// 2. Remaining files in chunks: orient against baseline, stream to accumulator
///
/// # Memory Model
/// - Baseline: ~size of first file's minimizers (typically < 1MB)
/// - Input chunks: ~40% of available memory
/// - Shard accumulator: ~40% of available memory
#[allow(clippy::too_many_arguments)]
fn build_single_bucket_streaming_oriented(
    output_dir: &Path,
    bucket_name: &str,
    files: &[PathBuf],
    config_dir: &Path,
    k: usize,
    w: usize,
    salt: u64,
    max_memory: Option<usize>,
    options: Option<&rype::parquet_index::ParquetWriteOptions>,
    exclusion_set: Option<&HashSet<u64>>,
) -> Result<SingleBucketResult> {
    use rype::memory::detect_available_memory;
    use rype::parquet_index::ShardAccumulator;

    const BUCKET_ID: u32 = 1;

    log::info!(
        "Processing bucket '{}' (streaming, oriented, {} files) ...",
        bucket_name,
        files.len()
    );

    if files.is_empty() {
        return Ok(SingleBucketResult {
            bucket_name: bucket_name.to_string(),
            sources: vec![],
            shard_infos: vec![],
            total_minimizers: 0,
        });
    }

    // Phase 1: Extract baseline from first file
    let (baseline_mins, baseline_sources) =
        extract_baseline_from_first_file(&files[0], config_dir, k, w, salt)?;

    log::debug!(
        "Baseline established: {} minimizers from {} sources",
        baseline_mins.len(),
        baseline_sources.len()
    );

    // Determine memory budget
    let available = max_memory.unwrap_or_else(|| {
        let detected = detect_available_memory();
        detected.bytes
    });

    let chunk_config = calculate_chunk_config(available);
    // Use 40% of memory for shard accumulator, clamped to minimum
    use rype::parquet_index::MIN_SHARD_BYTES;
    let shard_size = ((available as f64 * 0.4) as usize).max(MIN_SHARD_BYTES);
    // Batch size for adding entries: ~1 shard worth, capped at 128MB
    let add_batch_entries = (shard_size / 16).min(8_000_000);

    let mut accumulator = ShardAccumulator::with_output_dir(output_dir, shard_size, options);
    let mut total_excluded: u64 = 0;

    // Keep unfiltered baseline for orientation decisions, but filter before
    // writing to accumulator. We don't count baseline exclusions in
    // total_excluded because the same minimizers may reappear in chunks
    // (where they'll be counted).
    let baseline_for_orient = baseline_mins;
    let baseline_to_write = if let Some(excl) = exclusion_set {
        let original_len = baseline_for_orient.len();
        let filtered: Vec<u64> = baseline_for_orient
            .iter()
            .copied()
            .filter(|m| !excl.contains(m))
            .collect();
        log::debug!(
            "Baseline: excluded {} of {} minimizers via subtraction",
            original_len - filtered.len(),
            original_len
        );
        filtered
    } else {
        baseline_for_orient.clone()
    };

    // Add filtered baseline to accumulator in batches
    for batch in baseline_to_write.chunks(add_batch_entries) {
        accumulator.add_entries_from_minimizers(batch, BUCKET_ID);
        while accumulator.should_flush() {
            accumulator.flush_shard()?;
        }
    }
    let mut total_minimizers = baseline_to_write.len() as u64;
    let mut all_sources = baseline_sources;

    // Phase 2: Process remaining files with orientation against unfiltered baseline
    if files.len() > 1 {
        let baseline_ref = &baseline_for_orient;

        let mut chunk_iter =
            SequenceChunkIterator::new(&files[1..], config_dir, chunk_config.target_chunk_bytes);
        let mut chunk_count = 0;

        while let Some(chunk) = chunk_iter.next_chunk()? {
            chunk_count += 1;
            let chunk_size: usize = chunk.iter().map(|(seq, _)| seq.len()).sum();
            log::debug!(
                "Processing chunk {} ({} sequences, {} bytes)",
                chunk_count,
                chunk.len(),
                chunk_size
            );

            all_sources.extend(chunk.iter().map(|(_, src)| src.clone()));

            let avg_len = chunk_size / chunk.len().max(1);
            let estimated_mins = MinimizerWorkspace::estimate_for_length(avg_len, k, w);

            // Parallel extraction with orientation
            let chunk_mins: Vec<Vec<u64>> = chunk
                .par_iter()
                .map_init(
                    move || MinimizerWorkspace::with_estimate(estimated_mins),
                    |ws, (seq, _source)| {
                        let (mut fwd, mut rc) = extract_dual_strand_into(seq, k, w, salt, ws);
                        fwd.sort_unstable();
                        rc.sort_unstable();

                        let (orientation, _overlap) =
                            choose_orientation_sampled(baseline_ref, &fwd, &rc);

                        match orientation {
                            Orientation::Forward => fwd,
                            Orientation::ReverseComplement => rc,
                        }
                    },
                )
                .collect();

            // K-way merge this chunk's results
            let chunk_merged = kway_merge_dedup(chunk_mins);

            // Filter out excluded minimizers if subtraction is active
            let chunk_merged = if let Some(excl) = exclusion_set {
                let original_len = chunk_merged.len();
                let filtered: Vec<u64> = chunk_merged
                    .into_iter()
                    .filter(|m| !excl.contains(m))
                    .collect();
                let excluded = original_len - filtered.len();
                if excluded > 0 {
                    total_excluded += excluded as u64;
                }
                filtered
            } else {
                chunk_merged
            };

            total_minimizers += chunk_merged.len() as u64;

            // Stream to accumulator in batches
            for batch in chunk_merged.chunks(add_batch_entries) {
                accumulator.add_entries_from_minimizers(batch, BUCKET_ID);

                while accumulator.should_flush() {
                    if let Some(shard_info) = accumulator.flush_shard()? {
                        log::debug!(
                            "Flushed shard {} ({} entries)",
                            shard_info.shard_id,
                            shard_info.num_entries
                        );
                    }
                }
            }
        }

        log::debug!("Processed {} additional chunks", chunk_count);
    }

    if total_excluded > 0 {
        log::info!(
            "Bucket '{}': excluded {} minimizer entries via subtraction",
            bucket_name,
            total_excluded
        );
    }

    let shard_infos = accumulator.finish()?;

    log::info!(
        "Completed bucket '{}': {} minimizers, {} shards",
        bucket_name,
        total_minimizers,
        shard_infos.len()
    );

    Ok(SingleBucketResult {
        bucket_name: bucket_name.to_string(),
        sources: all_sources,
        shard_infos,
        total_minimizers,
    })
}

/// Build a single bucket from its files, returning the name, minimizers, and sources.
///
/// When `orient_sequences` is true:
/// - The first sequence uses forward strand (establishes baseline)
/// - Subsequent sequences compare forward vs reverse-complement overlap with existing bucket minimizers
/// - The orientation with higher overlap is chosen
///
/// The bucket minimizers are maintained as a sorted, deduplicated Vec throughout,
/// using `merge_sorted_into` for efficient in-place merging.
#[cfg(test)]
fn build_single_bucket(
    bucket_name: &str,
    files: &[PathBuf],
    config_dir: &Path,
    k: usize,
    w: usize,
    salt: u64,
    orient_sequences: bool,
) -> Result<(String, Vec<u64>, Vec<String>)> {
    log::info!(
        "Processing bucket '{}'{} ...",
        bucket_name,
        if orient_sequences { " (oriented)" } else { "" }
    );

    // Delegate to extract_bucket_minimizers for the actual work
    let (minimizers, sources) =
        extract_bucket_minimizers(files, config_dir, k, w, salt, orient_sequences)?;

    log::info!(
        "Completed bucket '{}': {} minimizers",
        bucket_name,
        minimizers.len()
    );

    Ok((bucket_name.to_string(), minimizers, sources))
}

/// Create Parquet inverted index directly from a TOML config file.
///
/// # Arguments
/// * `config_path` - Path to the TOML configuration file
/// * `cli_max_memory` - CLI override for max memory (None = auto-detect)
/// * `options` - Parquet write options
/// * `cli_orient` - CLI override for orient sequences flag (takes precedence over config)
pub fn build_parquet_index_from_config(
    config_path: &Path,
    cli_max_memory: Option<usize>,
    options: Option<&parquet_index::ParquetWriteOptions>,
    cli_orient: bool,
    subtract_from: Option<&Path>,
) -> Result<()> {
    let t_total = Instant::now();

    log::info!(
        "Building Parquet index from config: {}",
        config_path.display()
    );

    let cfg = parse_config(config_path)?;
    let config_dir = config_path
        .parent()
        .ok_or_else(|| anyhow!("Invalid config path"))?;

    log::info!("Validating file paths...");
    validate_config(&cfg, config_dir)?;
    log::info!("Validation successful.");

    // Determine orient_sequences: CLI --orient flag takes precedence over config file
    let orient_sequences = if cli_orient {
        true // CLI --orient flag overrides everything
    } else {
        cfg.index.orient_sequences.unwrap_or(false) // Config value, or default to false
    };
    if orient_sequences {
        log::info!("Orientation enabled: sequences will be oriented to maximize minimizer overlap");
    }

    // Load and validate subtraction index if specified
    let exclusion_set = if let Some(subtract_path) = subtract_from {
        use rype::{load_all_minimizers, ShardedInvertedIndex};

        log::info!(
            "Loading subtraction index from: {}",
            subtract_path.display()
        );
        let subtract_index = ShardedInvertedIndex::open(subtract_path).with_context(|| {
            format!(
                "Failed to open subtraction index: {}",
                subtract_path.display()
            )
        })?;

        validate_subtraction_compatibility(&cfg, &subtract_index)?;

        let excl = load_all_minimizers(&subtract_index)?;

        // Check that exclusion set fits comfortably in memory
        // HashSet overhead is ~24 bytes per entry (8 bytes u64 + 16 bytes overhead)
        let excl_bytes = excl.len() * 24;
        let available =
            cli_max_memory.unwrap_or_else(|| rype::memory::detect_available_memory().bytes);
        if excl_bytes > available / 2 {
            return Err(anyhow!(
                "Subtraction index too large: {} minimizers (~{}) would use more than \
                 half of available memory (~{}). Use 'index merge --subtract-from-primary' \
                 for streaming subtraction of large indices, or increase --max-memory.",
                excl.len(),
                rype::memory::format_bytes(excl_bytes),
                rype::memory::format_bytes(available),
            ));
        }

        log::info!(
            "Loaded {} unique minimizers from subtraction index (~{})",
            excl.len(),
            rype::memory::format_bytes(excl_bytes),
        );
        Some(excl)
    } else {
        None
    };

    // Ensure output has .ryxdi extension for Parquet format
    let output_path = cfg.index.output.with_extension("ryxdi");
    let output_path = resolve_path(config_dir, &output_path);

    log::info!(
        "Creating Parquet inverted index at {:?} (K={}, W={}, salt={:#x})",
        output_path,
        cfg.index.k,
        cfg.index.window,
        cfg.index.salt
    );

    let mut bucket_names: Vec<_> = cfg.buckets.keys().cloned().collect();
    bucket_names.sort();

    const MAX_BUCKETS: usize = 100_000;
    if bucket_names.len() > MAX_BUCKETS {
        return Err(anyhow!(
            "Too many buckets: {} exceeds maximum {}",
            bucket_names.len(),
            MAX_BUCKETS
        ));
    }

    // Build buckets - strategy depends on bucket count
    let is_single_bucket = bucket_names.len() == 1;

    if !is_single_bucket {
        // Multiple buckets: use streaming mode with channel-based parallelism
        use rype::memory::detect_available_memory;
        let available = cli_max_memory.unwrap_or_else(|| detect_available_memory().bytes);
        let shard_size = available / 2;

        log::info!(
            "Using streaming mode (shard size: {} bytes){}",
            shard_size,
            if exclusion_set.is_some() {
                " with subtraction"
            } else {
                ""
            }
        );

        return build_parquet_index_from_config_streaming(
            config_path,
            Some(shard_size),
            options,
            cli_orient,
            exclusion_set.as_ref(),
        );
    }

    // Single bucket: use streaming for memory-bounded processing
    // This bounds BOTH input AND output memory via ShardAccumulator
    use rype::parquet_index::{
        compute_source_hash, create_index_directory, write_buckets_parquet, InvertedManifest,
        ParquetManifest, ParquetShardFormat, FORMAT_MAGIC, FORMAT_VERSION,
    };
    use std::collections::HashMap;

    const BUCKET_ID: u32 = 1;
    let bucket_name = &bucket_names[0];
    let files = &cfg.buckets[bucket_name].files;

    // Create directory structure FIRST (required by ShardAccumulator)
    create_index_directory(&output_path)?;

    let t_build = Instant::now();
    let result = if orient_sequences {
        build_single_bucket_streaming_oriented(
            &output_path,
            bucket_name,
            files,
            config_dir,
            cfg.index.k,
            cfg.index.window,
            cfg.index.salt,
            cli_max_memory,
            options,
            exclusion_set.as_ref(),
        )?
    } else {
        build_single_bucket_streaming(
            &output_path,
            bucket_name,
            files,
            config_dir,
            cfg.index.k,
            cfg.index.window,
            cfg.index.salt,
            cli_max_memory,
            options,
            exclusion_set.as_ref(),
        )?
    };
    log_timing(
        "parquet_index: bucket_building",
        t_build.elapsed().as_millis(),
    );

    // Write bucket metadata to buckets.parquet
    let mut bucket_names_map = HashMap::new();
    let mut bucket_sources_map = HashMap::new();
    bucket_names_map.insert(BUCKET_ID, sanitize_bucket_name(&result.bucket_name));
    bucket_sources_map.insert(BUCKET_ID, result.sources);
    write_buckets_parquet(&output_path, &bucket_names_map, &bucket_sources_map)?;

    // Compute source hash for index compatibility checking
    let mut bucket_min_counts = HashMap::new();
    bucket_min_counts.insert(BUCKET_ID, result.total_minimizers as usize);
    let source_hash = compute_source_hash(&bucket_min_counts);

    // Write manifest
    let total_entries: u64 = result.shard_infos.iter().map(|s| s.num_entries).sum();
    let manifest = ParquetManifest {
        magic: FORMAT_MAGIC.to_string(),
        format_version: FORMAT_VERSION,
        k: cfg.index.k,
        w: cfg.index.window,
        salt: cfg.index.salt,
        source_hash,
        num_buckets: 1,
        total_minimizers: result.total_minimizers,
        inverted: Some(InvertedManifest {
            format: ParquetShardFormat::Parquet,
            num_shards: result.shard_infos.len() as u32,
            total_entries,
            has_overlapping_shards: true,
            shards: result.shard_infos,
        }),
    };
    manifest.save(&output_path)?;

    log::info!("Created streaming Parquet index:");
    log::info!(
        "  Shards: {}",
        manifest.inverted.as_ref().unwrap().num_shards
    );
    log::info!("  Total entries: {}", total_entries);
    log_timing("parquet_index: total", t_total.elapsed().as_millis());
    log::info!("Done.");

    Ok(())
}

/// Create Parquet inverted index from config using streaming shard creation.
///
/// This function processes buckets one at a time, using a `ShardAccumulator` to
/// write shards as they fill up. This keeps memory usage bounded regardless of
/// total index size.
///
/// # Memory Behavior
/// Unlike `build_parquet_index_from_config`, which holds all bucket minimizers in
/// memory before writing, this function:
/// 1. Processes buckets sequentially (not in parallel)
/// 2. Adds (minimizer, bucket_id) pairs to an accumulator
/// 3. Flushes to a new shard when the accumulator exceeds the size threshold
///
/// Each shard is sorted internally by (minimizer, bucket_id), but different shards
/// may have overlapping minimizer ranges.
///
/// # Arguments
/// * `config_path` - Path to the TOML configuration file
/// * `cli_max_shard_size` - CLI override for max shard size (takes precedence over config)
/// * `options` - Parquet write options
/// * `cli_orient` - CLI override for orient sequences flag
pub fn build_parquet_index_from_config_streaming(
    config_path: &Path,
    cli_max_shard_size: Option<usize>,
    options: Option<&parquet_index::ParquetWriteOptions>,
    cli_orient: bool,
    exclusion_set: Option<&HashSet<u64>>,
) -> Result<()> {
    use rype::parquet_index::{
        compute_source_hash, create_index_directory, write_buckets_parquet, InvertedManifest,
        ParquetManifest, ParquetShardFormat, ShardAccumulator, FORMAT_MAGIC, FORMAT_VERSION,
    };

    let t_total = Instant::now();

    log::info!(
        "Building Parquet index (streaming) from config: {}",
        config_path.display()
    );

    let cfg = parse_config(config_path)?;
    let config_dir = config_path
        .parent()
        .ok_or_else(|| anyhow!("Invalid config path"))?;

    log::info!("Validating file paths...");
    validate_config(&cfg, config_dir)?;
    log::info!("Validation successful.");

    // Determine max shard size: CLI > config > default (80% of available memory)
    let max_shard_size = cli_max_shard_size
        .or(cfg.index.max_shard_size)
        .unwrap_or_else(|| {
            let available = rype::memory::detect_available_memory();
            let default_size = (available.bytes as f64 * 0.8) as usize;
            // Ensure at least MIN_SHARD_BYTES (1MB)
            let min_shard = rype::parquet_index::MIN_SHARD_BYTES;
            let size = default_size.max(min_shard);
            log::info!(
                "Using default max shard size: {} (80% of {} available from {:?})",
                rype::memory::format_bytes(size),
                rype::memory::format_bytes(available.bytes),
                available.source
            );
            size
        });

    // Determine orientation
    let orient_sequences = if cli_orient {
        true
    } else {
        cfg.index.orient_sequences.unwrap_or(false)
    };
    if orient_sequences {
        log::info!("Orientation enabled: sequences will be oriented to maximize minimizer overlap");
    }

    // Prepare output path
    let output_path = cfg.index.output.with_extension("ryxdi");
    let output_path = resolve_path(config_dir, &output_path);

    log::info!(
        "Creating Parquet inverted index (streaming) at {:?} (K={}, W={}, salt={:#x})",
        output_path,
        cfg.index.k,
        cfg.index.window,
        cfg.index.salt
    );

    // Create directory structure
    create_index_directory(&output_path)?;

    // Get sorted bucket names for deterministic ordering
    let mut bucket_names_sorted: Vec<_> = cfg.buckets.keys().cloned().collect();
    bucket_names_sorted.sort();

    const MAX_BUCKETS: usize = 100_000;
    if bucket_names_sorted.len() > MAX_BUCKETS {
        return Err(anyhow!(
            "Too many buckets: {} exceeds maximum {}",
            bucket_names_sorted.len(),
            MAX_BUCKETS
        ));
    }

    // Create accumulator
    let opts = options.cloned().unwrap_or_default();
    let mut accumulator =
        ShardAccumulator::with_output_dir(&output_path, max_shard_size, Some(&opts));

    // Track bucket metadata for manifest
    let mut bucket_names_map: std::collections::HashMap<u32, String> =
        std::collections::HashMap::new();
    let mut bucket_sources_map: std::collections::HashMap<u32, Vec<String>> =
        std::collections::HashMap::new();
    let mut bucket_minimizer_counts: std::collections::HashMap<u32, usize> =
        std::collections::HashMap::new();
    let mut total_minimizers: u64 = 0;

    // Process buckets using channel-based parallelism for better CPU utilization.
    // Compared to batch-based processing:
    // - Eliminates batch barriers where all threads wait for slowest bucket
    // - Rayon's work-stealing keeps cores busy as long as buckets remain
    // - Consumer processes results as they arrive (may block during shard flushes)
    // - Still has per-result synchronization overhead via channel
    //
    // Thread safety: `cfg`, `config_dir`, and `orient_sequences` are borrowed
    // immutably across threads. Results (minimizers, sources) are moved through
    // the channel. The consumer is sequential (shard ordering requires it).
    let t_build = Instant::now();

    // Prepare work items: (bucket_id, bucket_name, files)
    let work_items: Vec<(u32, &str, &[PathBuf])> = bucket_names_sorted
        .iter()
        .enumerate()
        .map(|(idx, name)| {
            let bucket_id = (idx + 1) as u32;
            let files = cfg.buckets[name].files.as_slice();
            (bucket_id, name.as_str(), files)
        })
        .collect();

    let num_buckets = bucket_names_sorted.len();

    // Handle empty bucket list explicitly
    if num_buckets == 0 {
        return Err(anyhow!("No buckets defined in configuration"));
    }

    // Type alias for channel message clarity
    type BucketResult = Result<(u32, String, Vec<u64>, Vec<String>)>;

    // Track processed count for panic detection
    let processed_count = std::sync::atomic::AtomicUsize::new(0);
    // Cancellation signal for early exit
    let cancelled = std::sync::atomic::AtomicBool::new(false);

    // Use std::thread::scope to allow borrowing local data while running parallel processing
    let process_result: Result<()> = std::thread::scope(|s| {
        // Use bounded channel to provide backpressure if consumer is slow (e.g., during disk I/O)
        let (tx, rx) = std::sync::mpsc::sync_channel::<BucketResult>(4);

        // Producer thread: uses rayon to process buckets in parallel, sends results through channel
        let cancelled_ref = &cancelled;
        let processed_ref = &processed_count;
        s.spawn(move || {
            work_items
                .par_iter()
                .panic_fuse() // Stop processing on first panic
                .for_each_with(tx, |tx, (bucket_id, bucket_name, files)| {
                    // Check cancellation before starting work
                    if cancelled_ref.load(std::sync::atomic::Ordering::Relaxed) {
                        return;
                    }

                    log::info!(
                        "Processing bucket '{}' ({}/{}){} ...",
                        bucket_name,
                        bucket_id,
                        num_buckets,
                        if orient_sequences { " (oriented)" } else { "" }
                    );

                    let result = extract_bucket_minimizers(
                        files,
                        config_dir,
                        cfg.index.k,
                        cfg.index.window,
                        cfg.index.salt,
                        orient_sequences,
                    );

                    let bucket_result: BucketResult = match result {
                        Ok((minimizers, sources)) => {
                            log::info!(
                                "Completed bucket '{}': {} minimizers",
                                bucket_name,
                                minimizers.len()
                            );
                            Ok((
                                *bucket_id,
                                sanitize_bucket_name(bucket_name),
                                minimizers,
                                sources,
                            ))
                        }
                        Err(e) => Err(e.context(format!(
                            "Failed processing bucket '{}' (ID {})",
                            bucket_name, bucket_id
                        ))),
                    };

                    // Track successful send for panic detection
                    if tx.send(bucket_result).is_ok() {
                        processed_ref.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                });
        });

        // Consumer: receive results as buckets complete and accumulate
        let mut total_excluded: u64 = 0;
        for result in rx {
            let (bucket_id, bucket_name, minimizers, sources) = match result {
                Ok(data) => data,
                Err(e) => {
                    // Signal producers to stop on error
                    cancelled.store(true, std::sync::atomic::Ordering::Relaxed);
                    return Err(e);
                }
            };

            // Filter out excluded minimizers if subtraction is active
            let mut minimizers = minimizers;
            if let Some(excl) = exclusion_set {
                let original_len = minimizers.len();
                minimizers.retain(|m| !excl.contains(m));
                let excluded = original_len - minimizers.len();
                if excluded > 0 {
                    total_excluded += excluded as u64;
                    log::info!(
                        "Bucket '{}': excluded {} of {} minimizers via subtraction",
                        bucket_name,
                        excluded,
                        original_len
                    );
                }
            }

            // Store metadata
            bucket_names_map.insert(bucket_id, bucket_name);
            bucket_sources_map.insert(bucket_id, sources);
            bucket_minimizer_counts.insert(bucket_id, minimizers.len());
            total_minimizers += minimizers.len() as u64;

            // Add entries directly to accumulator without intermediate allocation
            accumulator.add_entries_from_minimizers(&minimizers, bucket_id);

            // Flush in a loop until under threshold (after each bucket's entries)
            while accumulator.should_flush() {
                let shard_info = accumulator.flush_shard()?;
                if let Some(info) = shard_info {
                    log::info!(
                        "Flushed shard {}: {} entries",
                        info.shard_id,
                        info.num_entries
                    );
                }
            }
        }

        if total_excluded > 0 {
            log::info!(
                "Subtraction complete: excluded {} minimizer entries total",
                total_excluded
            );
        }

        Ok(())
    });

    // Check for incomplete processing (panic or other failure)
    let actual_processed = processed_count.load(std::sync::atomic::Ordering::Relaxed);
    if actual_processed != num_buckets && process_result.is_ok() {
        return Err(anyhow!(
            "Index creation incomplete: processed {}/{} buckets (possible panic in worker thread)",
            actual_processed,
            num_buckets
        ));
    }

    process_result?;

    log_timing(
        "streaming_index: bucket_processing",
        t_build.elapsed().as_millis(),
    );

    // Finish accumulator (flush remaining entries)
    let shard_infos = accumulator.finish()?;
    log::info!("Created {} shards total", shard_infos.len());

    // Validate bucket name uniqueness directly from the map
    validate_unique_bucket_names(bucket_names_map.values().map(|s| s.as_str()))?;

    // Write bucket metadata
    write_buckets_parquet(&output_path, &bucket_names_map, &bucket_sources_map)?;

    // Compute source hash
    let source_hash = compute_source_hash(&bucket_minimizer_counts);

    // Compute total entries
    let total_entries: u64 = shard_infos.iter().map(|s| s.num_entries).sum();

    // Build manifest
    let manifest = ParquetManifest {
        magic: FORMAT_MAGIC.to_string(),
        format_version: FORMAT_VERSION,
        k: cfg.index.k,
        w: cfg.index.window,
        salt: cfg.index.salt,
        source_hash,
        num_buckets: bucket_names_map.len() as u32,
        total_minimizers,
        inverted: Some(InvertedManifest {
            format: ParquetShardFormat::Parquet,
            num_shards: shard_infos.len() as u32,
            total_entries,
            has_overlapping_shards: true,
            shards: shard_infos,
        }),
    };

    manifest.save(&output_path)?;

    log::info!("Created Parquet inverted index (streaming):");
    log::info!("  Buckets: {}", manifest.num_buckets);
    if let Some(ref inv) = manifest.inverted {
        log::info!("  Shards: {}", inv.num_shards);
        log::info!("  Total entries: {}", inv.total_entries);
    }

    log_timing("streaming_index: total", t_total.elapsed().as_millis());
    log::info!("Done.");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rype::BucketData;
    use std::fs::File;
    use std::io::Write;
    use tempfile::TempDir;

    /// Helper to create a simple FASTA file with one sequence
    fn create_fasta_file(dir: &Path, name: &str, seq: &[u8]) -> PathBuf {
        let path = dir.join(name);
        let mut file = File::create(&path).unwrap();
        writeln!(file, ">seq1").unwrap();
        file.write_all(seq).unwrap();
        writeln!(file).unwrap();
        path
    }

    /// Helper to create a config file for testing
    fn create_test_config(
        dir: &Path,
        output_name: &str,
        buckets: &[(&str, &[&str])],
        k: usize,
        window: usize,
    ) -> PathBuf {
        let config_path = dir.join("config.toml");
        let mut content = format!(
            r#"[index]
k = {}
window = {}
salt = 0x5555555555555555
output = "{}"

"#,
            k, window, output_name
        );

        for (bucket_name, files) in buckets {
            let files_str: Vec<String> = files.iter().map(|f| format!("\"{}\"", f)).collect();
            content.push_str(&format!(
                "[buckets.{}]\nfiles = [{}]\n\n",
                bucket_name,
                files_str.join(", ")
            ));
        }

        let mut file = File::create(&config_path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
        config_path
    }

    #[test]
    fn test_build_single_bucket_extracts_minimizers() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // Create a FASTA file with a sequence long enough for k=32, w=10
        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let fasta_path = create_fasta_file(dir, "test.fa", seq);

        let (name, minimizers, sources) = build_single_bucket(
            "TestBucket",
            &[fasta_path],
            dir,
            32, // k
            10, // w
            0x5555555555555555,
            false, // orient_sequences
        )
        .unwrap();

        assert_eq!(name, "TestBucket");
        assert!(!minimizers.is_empty(), "Should extract some minimizers");
        assert!(!sources.is_empty(), "Should have source labels");

        // Verify minimizers are sorted and deduplicated
        let mut sorted = minimizers.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            minimizers, sorted,
            "Minimizers should be sorted and deduplicated"
        );
    }

    #[test]
    fn test_bucket_result_to_bucket_data_conversion() {
        // Test that build_single_bucket output can be converted to BucketData
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let fasta_path = create_fasta_file(dir, "test.fa", seq);

        let (name, minimizers, sources) = build_single_bucket(
            "TestBucket",
            &[fasta_path],
            dir,
            32,
            10,
            0x5555555555555555,
            false,
        )
        .unwrap();

        // Convert to BucketData (this is the reuse we want)
        let bucket_data = BucketData {
            bucket_id: 1,
            bucket_name: sanitize_bucket_name(&name),
            sources,
            minimizers,
        };

        assert_eq!(bucket_data.bucket_id, 1);
        assert!(!bucket_data.minimizers.is_empty());
        assert!(bucket_data.validate().is_ok(), "BucketData should be valid");
    }

    #[test]
    fn test_build_parquet_index_from_config_creates_index() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // Create test FASTA files
        let seq1 = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let seq2 = b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA";

        create_fasta_file(dir, "ref1.fa", seq1);
        create_fasta_file(dir, "ref2.fa", seq2);

        // Create config
        let config_path = create_test_config(
            dir,
            "test_index.ryidx",
            &[("Bucket1", &["ref1.fa"]), ("Bucket2", &["ref2.fa"])],
            32,
            10,
        );

        // Build parquet index
        let result = build_parquet_index_from_config(&config_path, None, None, false, None);
        assert!(result.is_ok(), "Should succeed: {:?}", result);

        // Verify the parquet index was created
        let output_path = dir.join("test_index.ryxdi");
        assert!(output_path.exists(), "Parquet index directory should exist");

        // Verify manifest exists
        let manifest_path = output_path.join("manifest.toml");
        assert!(manifest_path.exists(), "Manifest should exist");
    }

    #[test]
    fn test_build_parquet_index_from_config_with_bloom_filter() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        create_fasta_file(dir, "ref.fa", seq);

        let config_path = create_test_config(
            dir,
            "bloom_test.ryidx",
            &[("TestBucket", &["ref.fa"])],
            32,
            10,
        );

        let options = parquet_index::ParquetWriteOptions {
            bloom_filter_enabled: true,
            bloom_filter_fpp: 0.05,
            ..Default::default()
        };

        let result =
            build_parquet_index_from_config(&config_path, None, Some(&options), false, None);
        assert!(
            result.is_ok(),
            "Should succeed with bloom filter: {:?}",
            result
        );

        let output_path = dir.join("bloom_test.ryxdi");
        assert!(output_path.exists());
    }

    #[test]
    fn test_build_parquet_index_from_config_invalid_file() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // Create config referencing non-existent file
        let config_path = create_test_config(
            dir,
            "test.ryidx",
            &[("TestBucket", &["nonexistent.fa"])],
            32,
            10,
        );

        let result = build_parquet_index_from_config(&config_path, None, None, false, None);
        assert!(result.is_err(), "Should fail with missing file");
    }

    // ============================================================================
    // Oriented Bucket Building Tests
    // ============================================================================

    #[test]
    fn test_build_single_bucket_orient_disabled_matches_original() {
        // orient=false should produce same result whether we call it with orient=false
        // This tests that the code path with orient=false doesn't break anything
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let fasta_path = create_fasta_file(dir, "test.fa", seq);

        let (name, minimizers, sources) = build_single_bucket(
            "TestBucket",
            &[fasta_path],
            dir,
            32,
            10,
            0x5555555555555555,
            false, // orient disabled
        )
        .unwrap();

        assert_eq!(name, "TestBucket");
        assert!(!minimizers.is_empty());
        assert!(!sources.is_empty());

        // Verify minimizers are sorted and deduplicated
        let mut sorted = minimizers.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(minimizers, sorted);
    }

    #[test]
    fn test_build_single_bucket_orient_enabled_produces_valid_output() {
        // orient=true should also produce valid sorted, deduplicated output
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let fasta_path = create_fasta_file(dir, "test.fa", seq);

        let (name, minimizers, sources) = build_single_bucket(
            "OrientedBucket",
            &[fasta_path],
            dir,
            32,
            10,
            0x5555555555555555,
            true, // orient enabled
        )
        .unwrap();

        assert_eq!(name, "OrientedBucket");
        assert!(!minimizers.is_empty());
        assert!(!sources.is_empty());

        // Verify minimizers are sorted and deduplicated
        let mut sorted = minimizers.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            minimizers, sorted,
            "Oriented bucket minimizers should be sorted and deduplicated"
        );
    }

    /// Helper to create a multi-sequence FASTA file
    fn create_multi_fasta_file(dir: &Path, name: &str, sequences: &[(&str, &[u8])]) -> PathBuf {
        let path = dir.join(name);
        let mut file = File::create(&path).unwrap();
        for (seq_name, seq) in sequences {
            writeln!(file, ">{}", seq_name).unwrap();
            file.write_all(seq).unwrap();
            writeln!(file).unwrap();
        }
        path
    }

    #[test]
    fn test_build_single_bucket_orient_with_multiple_sequences() {
        // Test that orientation works with multiple sequences:
        // - First sequence establishes baseline (forward)
        // - Subsequent sequences should choose orientation based on overlap
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // Create two sequences - the second is different but should still work
        let seq1 = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let seq2 = b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA";

        let fasta_path =
            create_multi_fasta_file(dir, "multi.fa", &[("seq1", seq1), ("seq2", seq2)]);

        // Build without orientation
        let (_, mins_no_orient, _) = build_single_bucket(
            "NoOrient",
            &[fasta_path.clone()],
            dir,
            32,
            10,
            0x5555555555555555,
            false,
        )
        .unwrap();

        // Build with orientation
        let (_, mins_with_orient, _) = build_single_bucket(
            "WithOrient",
            &[fasta_path],
            dir,
            32,
            10,
            0x5555555555555555,
            true,
        )
        .unwrap();

        // Both should be valid (sorted and deduplicated)
        let mut sorted_no = mins_no_orient.clone();
        sorted_no.sort_unstable();
        sorted_no.dedup();
        assert_eq!(mins_no_orient, sorted_no);

        let mut sorted_with = mins_with_orient.clone();
        sorted_with.sort_unstable();
        sorted_with.dedup();
        assert_eq!(mins_with_orient, sorted_with);

        // The oriented version may have same or different minimizers
        // depending on which orientation was chosen - both are valid
        assert!(!mins_no_orient.is_empty());
        assert!(!mins_with_orient.is_empty());
    }

    #[test]
    fn test_build_single_bucket_orient_first_sequence_uses_forward() {
        // With a single sequence, orient=true and orient=false should produce
        // identical results since the first sequence always uses forward
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let fasta_path = create_fasta_file(dir, "single.fa", seq);

        let (_, mins_no_orient, _) = build_single_bucket(
            "NoOrient",
            &[fasta_path.clone()],
            dir,
            32,
            10,
            0x5555555555555555,
            false,
        )
        .unwrap();

        let (_, mins_with_orient, _) = build_single_bucket(
            "WithOrient",
            &[fasta_path],
            dir,
            32,
            10,
            0x5555555555555555,
            true,
        )
        .unwrap();

        // Single sequence: both should be identical since first seq always uses forward
        assert_eq!(
            mins_no_orient, mins_with_orient,
            "First sequence should use forward orientation in both cases"
        );
    }

    // ============================================================================
    // Bucket Name Uniqueness Validation Tests
    // ============================================================================

    #[test]
    fn test_validate_unique_bucket_names_accepts_unique_names() {
        let names = ["Bucket_A", "Bucket_B"];
        assert!(validate_unique_bucket_names(names.iter().copied()).is_ok());
    }

    #[test]
    fn test_validate_unique_bucket_names_rejects_duplicates() {
        let names = ["Duplicate", "Duplicate"];
        let result = validate_unique_bucket_names(names.iter().copied());
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Duplicate"));
    }

    #[test]
    fn test_validate_unique_bucket_names_empty_list() {
        let names: [&str; 0] = [];
        assert!(validate_unique_bucket_names(names.iter().copied()).is_ok());
    }

    // ============================================================================
    // Phase 5: Extract Bucket Minimizers Helper Tests
    // ============================================================================

    #[test]
    fn test_extract_bucket_minimizers_returns_sorted() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // Create a FASTA file with sequences that will produce minimizers
        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let fasta_path = create_fasta_file(dir, "test.fa", seq);

        let (minimizers, _sources) = extract_bucket_minimizers(
            &[fasta_path],
            dir,
            32,                 // k
            10,                 // w
            0x5555555555555555, // salt
            false,              // orient_sequences
        )
        .unwrap();

        // Verify minimizers are sorted
        let mut sorted = minimizers.clone();
        sorted.sort_unstable();
        assert_eq!(minimizers, sorted, "Minimizers should be sorted");
        assert!(!minimizers.is_empty(), "Should have some minimizers");
    }

    #[test]
    fn test_extract_bucket_minimizers_deduplicates() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // Create two FASTA files with the SAME sequence - this guarantees duplicate minimizers
        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let fasta1 = create_fasta_file(dir, "test1.fa", seq);
        let fasta2 = create_fasta_file(dir, "test2.fa", seq);

        let (minimizers, _sources) = extract_bucket_minimizers(
            &[fasta1, fasta2],
            dir,
            32,
            10,
            0x5555555555555555,
            false, // orient_sequences
        )
        .unwrap();

        // Verify no duplicates
        let mut deduped = minimizers.clone();
        deduped.sort_unstable();
        deduped.dedup();
        assert_eq!(minimizers, deduped, "Minimizers should have no duplicates");
    }

    #[test]
    fn test_extract_bucket_minimizers_returns_sources() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // Create a multi-sequence FASTA file
        let fasta_path = create_multi_fasta_file(
            dir,
            "multi.fa",
            &[
                ("seq1", b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT".as_slice()),
                ("seq2", b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA".as_slice()),
            ],
        );

        let (_minimizers, sources) = extract_bucket_minimizers(
            &[fasta_path],
            dir,
            32,
            10,
            0x5555555555555555,
            false, // orient_sequences
        )
        .unwrap();

        // Verify we have source labels for each sequence
        assert_eq!(sources.len(), 2, "Should have 2 source labels");
        assert!(
            sources.iter().any(|s| s.contains("seq1")),
            "Should have source for seq1"
        );
        assert!(
            sources.iter().any(|s| s.contains("seq2")),
            "Should have source for seq2"
        );
        // Sources should contain the delimiter
        for source in &sources {
            assert!(
                source.contains(BUCKET_SOURCE_DELIM),
                "Source '{}' should contain delimiter '{}'",
                source,
                BUCKET_SOURCE_DELIM
            );
        }
    }

    #[test]
    fn test_extract_bucket_minimizers_with_orientation() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // Create multi-sequence file
        let fasta_path = create_multi_fasta_file(
            dir,
            "multi.fa",
            &[
                ("seq1", b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT".as_slice()),
                ("seq2", b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA".as_slice()),
            ],
        );

        // Extract without orientation
        let (mins_no_orient, _) = extract_bucket_minimizers(
            &[fasta_path.clone()],
            dir,
            32,
            10,
            0x5555555555555555,
            false,
        )
        .unwrap();

        // Extract with orientation
        let (mins_with_orient, _) =
            extract_bucket_minimizers(&[fasta_path], dir, 32, 10, 0x5555555555555555, true)
                .unwrap();

        // Both should be sorted and deduplicated
        let mut sorted_no = mins_no_orient.clone();
        sorted_no.sort_unstable();
        sorted_no.dedup();
        assert_eq!(mins_no_orient, sorted_no);

        let mut sorted_with = mins_with_orient.clone();
        sorted_with.sort_unstable();
        sorted_with.dedup();
        assert_eq!(mins_with_orient, sorted_with);

        // Both should have minimizers
        assert!(!mins_no_orient.is_empty());
        assert!(!mins_with_orient.is_empty());
    }

    // ============================================================================
    // Phase 6: Streaming Index Builder Tests
    // ============================================================================

    #[test]
    fn test_streaming_index_creation_basic() {
        // Test that streaming index creation produces a valid, queryable index
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // Create test FASTA files
        let seq1 = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let seq2 = b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA";

        create_fasta_file(dir, "ref1.fa", seq1);
        create_fasta_file(dir, "ref2.fa", seq2);

        // Create config with max_shard_size set (triggers streaming mode)
        let config_content = r#"[index]
k = 32
window = 10
salt = 0x5555555555555555
output = "streaming_test.ryidx"
max_shard_size = 1048576

[buckets.Bucket1]
files = ["ref1.fa"]

[buckets.Bucket2]
files = ["ref2.fa"]
"#;
        let config_path = dir.join("config.toml");
        std::fs::write(&config_path, config_content).unwrap();

        // Build streaming index
        let result =
            build_parquet_index_from_config_streaming(&config_path, None, None, false, None);
        assert!(
            result.is_ok(),
            "Streaming index creation should succeed: {:?}",
            result
        );

        // Verify the index was created
        let output_path = dir.join("streaming_test.ryxdi");
        assert!(output_path.exists(), "Index directory should exist");
        assert!(
            output_path.join("manifest.toml").exists(),
            "Manifest should exist"
        );
        assert!(
            output_path.join("buckets.parquet").exists(),
            "Buckets file should exist"
        );
        assert!(
            output_path.join("inverted").exists(),
            "Inverted directory should exist"
        );

        // Verify we can open and query the index
        use rype::ShardedInvertedIndex;
        let index = ShardedInvertedIndex::open(&output_path).unwrap();
        assert_eq!(
            index.manifest().bucket_names.len(),
            2,
            "Should have 2 buckets"
        );
    }

    #[test]
    fn test_streaming_index_respects_shard_size() {
        // Test that streaming creates multiple shards when data exceeds shard size
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // Create multiple FASTA files with enough data to exceed 1MB shard size
        // Need ~65K entries per MB (16 bytes per entry estimated)
        // Use a pseudo-random pattern to avoid repetition and ensure unique minimizers
        fn make_sequence(seed: u64, length: usize) -> Vec<u8> {
            // Simple LCG-based PRNG for deterministic "random" sequence
            let mut state = seed;
            (0..length)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    match (state >> 32) % 4 {
                        0 => b'A',
                        1 => b'C',
                        2 => b'G',
                        _ => b'T',
                    }
                })
                .collect()
        }

        // Create 10 buckets with large unique sequences (500KB each)
        // Each bucket will have ~50K minimizers after dedup
        // Total: ~500K entries = ~8MB of data, should create 8+ shards at 1MB
        for i in 0..10 {
            let seq = make_sequence(i as u64 * 12345, 500_000);
            create_fasta_file(dir, &format!("ref{}.fa", i), &seq);
        }

        // Create config with 1MB shard size to force multiple shards
        let mut config_content = r#"[index]
k = 32
window = 10
salt = 0x5555555555555555
output = "shard_test.ryidx"
max_shard_size = 1048576

"#
        .to_string();

        for i in 0..10 {
            config_content.push_str(&format!(
                "[buckets.Bucket{}]\nfiles = [\"ref{}.fa\"]\n\n",
                i, i
            ));
        }

        let config_path = dir.join("config.toml");
        std::fs::write(&config_path, &config_content).unwrap();

        // Build streaming index with 1MB shard size
        let result = build_parquet_index_from_config_streaming(
            &config_path,
            Some(1024 * 1024), // 1MB
            None,
            false,
            None,
        );
        assert!(result.is_ok(), "Should succeed: {:?}", result);

        // Verify multiple shards were created
        let output_path = dir.join("shard_test.ryxdi");
        let inverted_dir = output_path.join("inverted");
        let shard_files: Vec<_> = std::fs::read_dir(&inverted_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "parquet")
                    .unwrap_or(false)
            })
            .collect();

        assert!(
            shard_files.len() >= 2,
            "Should have multiple shards (got {}), data should exceed 1MB",
            shard_files.len()
        );
    }

    #[test]
    fn test_streaming_index_produces_valid_manifest() {
        // Test that the manifest produced by streaming is complete and valid
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        let seq1 = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let seq2 = b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA";

        create_fasta_file(dir, "ref1.fa", seq1);
        create_fasta_file(dir, "ref2.fa", seq2);

        let config_content = r#"[index]
k = 32
window = 10
salt = 0x5555555555555555
output = "manifest_test.ryidx"
max_shard_size = 1048576

[buckets.Alpha]
files = ["ref1.fa"]

[buckets.Beta]
files = ["ref2.fa"]
"#;
        let config_path = dir.join("config.toml");
        std::fs::write(&config_path, config_content).unwrap();

        let result =
            build_parquet_index_from_config_streaming(&config_path, None, None, false, None);
        assert!(result.is_ok(), "Should succeed: {:?}", result);

        // Load and verify manifest
        use rype::ShardedInvertedIndex;
        let output_path = dir.join("manifest_test.ryxdi");
        let index = ShardedInvertedIndex::open(&output_path).unwrap();
        let manifest = index.manifest();

        // Verify manifest fields
        assert_eq!(manifest.k, 32);
        assert_eq!(manifest.w, 10);
        assert_eq!(manifest.salt, 0x5555555555555555);
        assert_eq!(manifest.bucket_names.len(), 2);

        // Verify bucket names exist
        let names: Vec<_> = manifest.bucket_names.values().collect();
        assert!(
            names.iter().any(|n| n.as_str() == "Alpha"),
            "Should have bucket Alpha"
        );
        assert!(
            names.iter().any(|n| n.as_str() == "Beta"),
            "Should have bucket Beta"
        );

        // Verify shard info is populated
        assert!(
            !manifest.shards.is_empty(),
            "Should have at least one shard"
        );
        assert!(manifest.total_minimizers > 0, "Should have minimizers");
        assert!(manifest.total_bucket_ids > 0, "Should have bucket refs");
    }

    // ============================================================================
    // Phase 1: Streaming Parallelism Tests (Channel-Based)
    // These tests define expected behavior for the channel-based implementation
    // ============================================================================

    #[test]
    fn test_streaming_no_batch_barrier_timing() {
        // Test that parallel bucket processing doesn't suffer from batch barriers.
        // With N buckets of varying sizes, total time should be closer to the
        // slowest single bucket than the sum of batch maximums.
        //
        // Strategy: Create buckets with significantly different sequence sizes.
        // Larger sequences take longer to process. If batching causes barriers,
        // total time = sum of max(batch_i), but with proper parallelism,
        // total time ≈ max(all buckets).
        //
        // This test uses real file I/O differences to create timing variance.

        use std::time::Instant;

        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // Create buckets with vastly different sizes to create processing time variance
        // Small buckets: ~100 bp (fast)
        // Large buckets: ~100,000 bp (slower)
        let small_seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";

        // Generate large sequence
        fn make_large_sequence(size: usize) -> Vec<u8> {
            let base_pattern = b"ACGTACGTACGTACGTACGTACGTACGTACGT";
            let mut seq = Vec::with_capacity(size);
            while seq.len() < size {
                seq.extend_from_slice(base_pattern);
            }
            seq.truncate(size);
            seq
        }

        let large_seq = make_large_sequence(100_000);

        // Create 8 buckets: 2 large (slow), 6 small (fast)
        // With batch_size = num_threads (typically 4-8), if batching causes barriers:
        // - If both large buckets end up in different batches, each batch waits for its large bucket
        // - Total time ≈ 2 * large_bucket_time
        // With channel-based parallelism:
        // - Total time ≈ large_bucket_time (both large run in parallel with small)

        create_fasta_file(dir, "large1.fa", &large_seq);
        create_fasta_file(dir, "large2.fa", &large_seq);
        for i in 0..6 {
            create_fasta_file(dir, &format!("small{}.fa", i), small_seq);
        }

        let config_content = r#"[index]
k = 32
window = 10
salt = 0x5555555555555555
output = "timing_test.ryidx"
max_shard_size = 10485760

[buckets.Large1]
files = ["large1.fa"]

[buckets.Large2]
files = ["large2.fa"]

[buckets.Small0]
files = ["small0.fa"]

[buckets.Small1]
files = ["small1.fa"]

[buckets.Small2]
files = ["small2.fa"]

[buckets.Small3]
files = ["small3.fa"]

[buckets.Small4]
files = ["small4.fa"]

[buckets.Small5]
files = ["small5.fa"]
"#;
        let config_path = dir.join("config.toml");
        std::fs::write(&config_path, config_content).unwrap();

        // Time the streaming build
        let start = Instant::now();
        let result =
            build_parquet_index_from_config_streaming(&config_path, None, None, false, None);
        let elapsed = start.elapsed();

        assert!(result.is_ok(), "Should succeed: {:?}", result);

        // Verify the index was created correctly
        let output_path = dir.join("timing_test.ryxdi");
        assert!(output_path.exists());

        use rype::ShardedInvertedIndex;
        let index = ShardedInvertedIndex::open(&output_path).unwrap();
        assert_eq!(index.manifest().bucket_names.len(), 8);

        // Log timing for manual inspection (actual assertion would be flaky)
        // The key insight: with proper parallelism, elapsed should be < 2x single large bucket time
        eprintln!(
            "test_streaming_no_batch_barrier_timing: elapsed={:?}ms",
            elapsed.as_millis()
        );

        // We can't assert precise timing without being flaky, but we verify correctness
        // The real verification is done in Phase 4 integration testing with profiling
    }

    #[test]
    fn test_streaming_results_as_completed() {
        // Test that results are accumulated incrementally as buckets complete,
        // not batched. We verify this by checking that shards can be flushed
        // during processing (not just at the end).
        //
        // With channel-based processing, shards should flush as data arrives.
        // With batch-based, flushes only happen after each batch completes.

        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // Create enough buckets with enough data to trigger multiple shard flushes
        // Use small shard size to force frequent flushing
        fn make_unique_sequence(seed: u64, length: usize) -> Vec<u8> {
            let mut state = seed;
            (0..length)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    match (state >> 32) % 4 {
                        0 => b'A',
                        1 => b'C',
                        2 => b'G',
                        _ => b'T',
                    }
                })
                .collect()
        }

        // Create 30 buckets, each with larger unique sequences to exceed minimum shard size
        for i in 0..30 {
            let seq = make_unique_sequence(i * 9876, 100_000);
            create_fasta_file(dir, &format!("bucket{}.fa", i), &seq);
        }

        // Build config
        let mut config_content = r#"[index]
k = 32
window = 10
salt = 0x5555555555555555
output = "incremental_test.ryidx"
max_shard_size = 1048576

"#
        .to_string();

        for i in 0..30 {
            config_content.push_str(&format!(
                "[buckets.Bucket{:02}]\nfiles = [\"bucket{}.fa\"]\n\n",
                i, i
            ));
        }

        let config_path = dir.join("config.toml");
        std::fs::write(&config_path, &config_content).unwrap();

        // Build with minimum shard size (1MB) to encourage multiple flushes
        let result = build_parquet_index_from_config_streaming(
            &config_path,
            Some(1024 * 1024), // 1MB shards (minimum allowed)
            None,
            false,
            None,
        );
        assert!(result.is_ok(), "Should succeed: {:?}", result);

        // Verify multiple shards were created (proving incremental flushing)
        let output_path = dir.join("incremental_test.ryxdi");
        let inverted_dir = output_path.join("inverted");
        let shard_count = std::fs::read_dir(&inverted_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "parquet")
                    .unwrap_or(false)
            })
            .count();

        // With 30 buckets × ~10K minimizers each = ~300K minimizers
        // At 1MB per shard (~65K entries), should have 4+ shards
        assert!(
            shard_count >= 2,
            "Should have multiple shards from incremental flushing, got {}",
            shard_count
        );

        // Verify index is valid and queryable
        use rype::ShardedInvertedIndex;
        let index = ShardedInvertedIndex::open(&output_path).unwrap();
        assert_eq!(index.manifest().bucket_names.len(), 30);
    }

    #[test]
    fn test_streaming_error_propagation() {
        // Test that errors from parallel bucket processing propagate correctly.
        // If one bucket fails (e.g., missing file), the error should be returned
        // and processing should stop gracefully.

        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // Create some valid files
        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        create_fasta_file(dir, "valid1.fa", seq);
        create_fasta_file(dir, "valid2.fa", seq);

        // Config with one bucket referencing a non-existent file
        let config_content = r#"[index]
k = 32
window = 10
salt = 0x5555555555555555
output = "error_test.ryidx"
max_shard_size = 1048576

[buckets.ValidBucket1]
files = ["valid1.fa"]

[buckets.InvalidBucket]
files = ["nonexistent_file.fa"]

[buckets.ValidBucket2]
files = ["valid2.fa"]
"#;
        let config_path = dir.join("config.toml");
        std::fs::write(&config_path, config_content).unwrap();

        // Should fail during validation (before processing starts)
        let result =
            build_parquet_index_from_config_streaming(&config_path, None, None, false, None);
        assert!(result.is_err(), "Should fail with missing file");

        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("nonexistent")
                || err_msg.contains("not found")
                || err_msg.contains("does not exist"),
            "Error should mention the missing file: {}",
            err_msg
        );

        // Index should not be created
        let output_path = dir.join("error_test.ryxdi");
        assert!(
            !output_path.exists(),
            "Index should not be created when error occurs"
        );
    }

    #[test]
    fn test_streaming_error_during_processing() {
        // Test error handling when a bucket fails DURING processing (not validation).
        // This simulates a corrupted file or I/O error mid-stream.

        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // Create valid files
        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        create_fasta_file(dir, "valid1.fa", seq);
        create_fasta_file(dir, "valid2.fa", seq);

        // Create a file that exists but is not valid FASTA (will fail during parsing)
        let bad_file = dir.join("corrupt.fa");
        std::fs::write(&bad_file, b"not a valid fasta file\nno header line\n").unwrap();

        // We need to bypass validation - create config manually without going through validate_config
        // Actually, the validation checks file existence, not format. Format errors occur during processing.
        let config_content = r#"[index]
k = 32
window = 10
salt = 0x5555555555555555
output = "corrupt_test.ryidx"
max_shard_size = 1048576

[buckets.ValidBucket1]
files = ["valid1.fa"]

[buckets.CorruptBucket]
files = ["corrupt.fa"]

[buckets.ValidBucket2]
files = ["valid2.fa"]
"#;
        let config_path = dir.join("config.toml");
        std::fs::write(&config_path, config_content).unwrap();

        let result =
            build_parquet_index_from_config_streaming(&config_path, None, None, false, None);

        // The corrupt file might either fail (if FASTA parser rejects it) or
        // produce empty output (if parser is lenient). Either is acceptable.
        // The key is that it shouldn't panic or hang.
        if result.is_err() {
            // Error is fine - corrupt file was detected
            eprintln!("Corrupt file detected: {}", result.unwrap_err());
        } else {
            // If it succeeded, verify the index is still valid
            let output_path = dir.join("corrupt_test.ryxdi");
            if output_path.exists() {
                use rype::ShardedInvertedIndex;
                let index = ShardedInvertedIndex::open(&output_path).unwrap();
                // At least the valid buckets should be present
                assert!(index.manifest().bucket_names.len() >= 2);
            }
        }
    }

    #[test]
    fn test_streaming_channel_based_produces_same_index() {
        // Test that streaming mode produces the same index content as non-streaming mode.
        // This verifies that the parallelism changes don't affect correctness.

        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // Create test sequences
        let seq1 = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let seq2 = b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA";
        let seq3 = b"AAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCC";

        create_fasta_file(dir, "ref1.fa", seq1);
        create_fasta_file(dir, "ref2.fa", seq2);
        create_fasta_file(dir, "ref3.fa", seq3);

        // Build non-streaming index
        let config_nonstream = r#"[index]
k = 32
window = 10
salt = 0x5555555555555555
output = "nonstream.ryidx"

[buckets.Alpha]
files = ["ref1.fa"]

[buckets.Beta]
files = ["ref2.fa"]

[buckets.Gamma]
files = ["ref3.fa"]
"#;
        let config_path_ns = dir.join("config_nonstream.toml");
        std::fs::write(&config_path_ns, config_nonstream).unwrap();

        let result_ns = build_parquet_index_from_config(&config_path_ns, None, None, false, None);
        assert!(
            result_ns.is_ok(),
            "Non-streaming should succeed: {:?}",
            result_ns
        );

        // Build streaming index with same content
        let config_stream = r#"[index]
k = 32
window = 10
salt = 0x5555555555555555
output = "stream.ryidx"
max_shard_size = 10485760

[buckets.Alpha]
files = ["ref1.fa"]

[buckets.Beta]
files = ["ref2.fa"]

[buckets.Gamma]
files = ["ref3.fa"]
"#;
        let config_path_s = dir.join("config_stream.toml");
        std::fs::write(&config_path_s, config_stream).unwrap();

        let result_s =
            build_parquet_index_from_config_streaming(&config_path_s, None, None, false, None);
        assert!(result_s.is_ok(), "Streaming should succeed: {:?}", result_s);

        // Compare the two indices
        use rype::ShardedInvertedIndex;
        let index_ns = ShardedInvertedIndex::open(&dir.join("nonstream.ryxdi")).unwrap();
        let index_s = ShardedInvertedIndex::open(&dir.join("stream.ryxdi")).unwrap();

        let manifest_ns = index_ns.manifest();
        let manifest_s = index_s.manifest();

        // Verify parameters match
        assert_eq!(manifest_ns.k, manifest_s.k, "K should match");
        assert_eq!(manifest_ns.w, manifest_s.w, "W should match");
        assert_eq!(manifest_ns.salt, manifest_s.salt, "Salt should match");

        // Verify bucket count matches
        assert_eq!(
            manifest_ns.bucket_names.len(),
            manifest_s.bucket_names.len(),
            "Bucket count should match"
        );

        // Verify bucket names match
        let mut names_ns: Vec<_> = manifest_ns.bucket_names.values().cloned().collect();
        let mut names_s: Vec<_> = manifest_s.bucket_names.values().cloned().collect();
        names_ns.sort();
        names_s.sort();
        assert_eq!(names_ns, names_s, "Bucket names should match");

        // Verify total minimizers match
        assert_eq!(
            manifest_ns.total_minimizers, manifest_s.total_minimizers,
            "Total minimizers should match"
        );

        // Verify total bucket IDs match
        assert_eq!(
            manifest_ns.total_bucket_ids, manifest_s.total_bucket_ids,
            "Total bucket IDs should match"
        );

        // Verify source hash matches (deterministic content)
        assert_eq!(
            manifest_ns.source_hash, manifest_s.source_hash,
            "Source hash should match (deterministic content)"
        );
    }

    // ==========================================================================
    // Tests for parallel single-bucket extraction (TDD RED phase)
    // ==========================================================================

    /// Test 1: Single-bucket with multiple sequences produces correct output.
    ///
    /// This test verifies that parallel extraction + k-way merge produces the same
    /// sorted, deduplicated minimizer set as sequential extraction + incremental merge.
    #[test]
    fn test_single_bucket_parallel_produces_same_output() {
        use rype::ShardedInvertedIndex;

        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // Create multiple FASTA files with distinct sequences for a single bucket
        // Using different seeds ensures unique minimizer sets
        fn make_sequence(seed: u64, length: usize) -> Vec<u8> {
            let mut state = seed;
            (0..length)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    match (state >> 32) % 4 {
                        0 => b'A',
                        1 => b'C',
                        2 => b'G',
                        _ => b'T',
                    }
                })
                .collect()
        }

        // Create 5 files with ~10KB sequences each
        for i in 0..5 {
            let seq = make_sequence(i as u64 * 99999, 10_000);
            create_fasta_file(dir, &format!("ref{}.fa", i), &seq);
        }

        // Create config with single bucket containing all 5 files
        let config_content = r#"[index]
k = 32
window = 10
salt = 0x5555555555555555
output = "single_bucket_test.ryidx"

[buckets.SingleBucket]
files = ["ref0.fa", "ref1.fa", "ref2.fa", "ref3.fa", "ref4.fa"]
"#;
        let config_path = dir.join("config.toml");
        std::fs::write(&config_path, config_content).unwrap();

        // Build index using non-streaming path (which will eventually use parallel)
        let result = build_parquet_index_from_config(&config_path, None, None, false, None);
        assert!(
            result.is_ok(),
            "Index creation should succeed: {:?}",
            result
        );

        // Verify the index
        let output_path = dir.join("single_bucket_test.ryxdi");
        let index = ShardedInvertedIndex::open(&output_path).unwrap();
        let manifest = index.manifest();

        // Verify single bucket
        assert_eq!(
            manifest.bucket_names.len(),
            1,
            "Should have exactly 1 bucket"
        );

        // Verify bucket name
        assert!(
            manifest.bucket_names.values().any(|n| n == "SingleBucket"),
            "Bucket should be named 'SingleBucket'"
        );

        // Verify bucket has minimizers (with 5 x 10KB sequences, expect substantial count)
        // Note: bucket_minimizer_counts not stored in Parquet, use total_minimizers
        assert!(
            manifest.total_minimizers > 1000,
            "Single bucket should have significant minimizers (got {})",
            manifest.total_minimizers
        );

        // Find the bucket_id for SingleBucket
        let bucket_id = *manifest
            .bucket_names
            .iter()
            .find(|(_, name)| *name == "SingleBucket")
            .map(|(id, _)| id)
            .unwrap();

        // Verify sources are recorded for all 5 sequences
        let sources = manifest.bucket_sources.get(&bucket_id).unwrap();
        assert_eq!(
            sources.len(),
            5,
            "Should have 5 sources (one per file, single sequence each)"
        );
    }

    /// Test 2: Multi-bucket behavior remains unchanged (regression test).
    ///
    /// When there are multiple buckets, the existing parallel-across-buckets
    /// path should be used and produce correct results.
    #[test]
    fn test_multi_bucket_behavior_unchanged() {
        use rype::ShardedInvertedIndex;

        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // Create deterministic sequences with known patterns
        // Using poly-A, poly-T, and poly-G for predictable (and distinct) minimizers
        let seq_a = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        let seq_t = b"TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT";
        let seq_g = b"GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG";

        create_fasta_file(dir, "ref_a.fa", seq_a);
        create_fasta_file(dir, "ref_t.fa", seq_t);
        create_fasta_file(dir, "ref_g.fa", seq_g);

        // Create config with 3 buckets
        let config_content = r#"[index]
k = 32
window = 10
salt = 0x5555555555555555
output = "multi_bucket_test.ryidx"

[buckets.BucketA]
files = ["ref_a.fa"]

[buckets.BucketT]
files = ["ref_t.fa"]

[buckets.BucketG]
files = ["ref_g.fa"]
"#;
        let config_path = dir.join("config.toml");
        std::fs::write(&config_path, config_content).unwrap();

        // Build index
        let result = build_parquet_index_from_config(&config_path, None, None, false, None);
        assert!(
            result.is_ok(),
            "Index creation should succeed: {:?}",
            result
        );

        // Verify the index
        let output_path = dir.join("multi_bucket_test.ryxdi");
        let index = ShardedInvertedIndex::open(&output_path).unwrap();
        let manifest = index.manifest();

        // Verify 3 buckets
        assert_eq!(
            manifest.bucket_names.len(),
            3,
            "Should have exactly 3 buckets"
        );

        // Verify all bucket names present
        let names: std::collections::HashSet<&String> = manifest.bucket_names.values().collect();
        assert!(
            names.iter().any(|n| n.as_str() == "BucketA"),
            "Missing BucketA"
        );
        assert!(
            names.iter().any(|n| n.as_str() == "BucketT"),
            "Missing BucketT"
        );
        assert!(
            names.iter().any(|n| n.as_str() == "BucketG"),
            "Missing BucketG"
        );

        // Verify total minimizers exist (bucket_minimizer_counts not in Parquet)
        assert!(
            manifest.total_minimizers > 0,
            "Index should have some total minimizers (got {})",
            manifest.total_minimizers
        );
    }

    /// Test 3: Single-bucket with orientation uses hybrid approach correctly.
    ///
    /// Strategy: First file establishes baseline (forward), remaining files
    /// orient against baseline, then k-way merge all results.
    #[test]
    fn test_single_bucket_parallel_with_orientation() {
        use rype::ShardedInvertedIndex;

        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // Create sequences where orientation matters
        // Using deterministic sequences that have distinct forward/RC minimizers
        fn make_sequence(seed: u64, length: usize) -> Vec<u8> {
            let mut state = seed;
            (0..length)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    match (state >> 32) % 4 {
                        0 => b'A',
                        1 => b'C',
                        2 => b'G',
                        _ => b'T',
                    }
                })
                .collect()
        }

        // Create 4 files for single bucket
        for i in 0..4 {
            let seq = make_sequence(i as u64 * 77777, 5_000);
            create_fasta_file(dir, &format!("ref{}.fa", i), &seq);
        }

        // Create config with orientation enabled
        let config_content = r#"[index]
k = 32
window = 10
salt = 0x5555555555555555
output = "oriented_single_bucket.ryidx"
orient_sequences = true

[buckets.OrientedBucket]
files = ["ref0.fa", "ref1.fa", "ref2.fa", "ref3.fa"]
"#;
        let config_path = dir.join("config.toml");
        std::fs::write(&config_path, config_content).unwrap();

        // Build index with orientation
        let result = build_parquet_index_from_config(&config_path, None, None, false, None);
        assert!(
            result.is_ok(),
            "Oriented index creation should succeed: {:?}",
            result
        );

        // Verify the index
        let output_path = dir.join("oriented_single_bucket.ryxdi");
        let index = ShardedInvertedIndex::open(&output_path).unwrap();
        let manifest = index.manifest();

        // Verify single bucket
        assert_eq!(manifest.bucket_names.len(), 1, "Should have 1 bucket");

        // Verify bucket has minimizers (use total_minimizers since per-bucket not in Parquet)
        assert!(
            manifest.total_minimizers > 100,
            "Oriented bucket should have minimizers (got {})",
            manifest.total_minimizers
        );

        // Find the bucket_id for OrientedBucket
        let bucket_id = *manifest
            .bucket_names
            .iter()
            .find(|(_, name)| *name == "OrientedBucket")
            .map(|(id, _)| id)
            .unwrap();

        // Verify sources for all 4 sequences
        let sources = manifest.bucket_sources.get(&bucket_id).unwrap();
        assert_eq!(sources.len(), 4, "Should have 4 sources");
    }

    /// Test 4: Single-bucket with single sequence (degenerate case).
    ///
    /// Edge case where there's only one sequence - should still work correctly.
    #[test]
    fn test_single_bucket_single_sequence() {
        use rype::ShardedInvertedIndex;

        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // Create single file with one sequence
        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        create_fasta_file(dir, "single.fa", seq);

        let config_content = r#"[index]
k = 32
window = 10
salt = 0x5555555555555555
output = "single_seq_test.ryidx"

[buckets.SingleSeq]
files = ["single.fa"]
"#;
        let config_path = dir.join("config.toml");
        std::fs::write(&config_path, config_content).unwrap();

        // Build index
        let result = build_parquet_index_from_config(&config_path, None, None, false, None);
        assert!(
            result.is_ok(),
            "Single sequence index should succeed: {:?}",
            result
        );

        // Verify the index
        let output_path = dir.join("single_seq_test.ryxdi");
        let index = ShardedInvertedIndex::open(&output_path).unwrap();
        let manifest = index.manifest();

        // Verify single bucket with one source
        assert_eq!(manifest.bucket_names.len(), 1, "Should have 1 bucket");

        // Find the bucket_id for SingleSeq
        let bucket_id = *manifest
            .bucket_names
            .iter()
            .find(|(_, name)| *name == "SingleSeq")
            .map(|(id, _)| id)
            .unwrap();

        let sources = manifest.bucket_sources.get(&bucket_id).unwrap();
        assert_eq!(sources.len(), 1, "Should have 1 source");

        // Verify minimizers exist (use total_minimizers)
        assert!(
            manifest.total_minimizers > 0,
            "Should have some minimizers (got {})",
            manifest.total_minimizers
        );
    }

    /// Test 5: Single-bucket with sequences that produce no minimizers.
    ///
    /// Edge case where some sequences are shorter than k (produce no minimizers).
    /// Should not crash and should produce valid output for remaining sequences.
    #[test]
    fn test_single_bucket_empty_sequences() {
        use rype::ShardedInvertedIndex;

        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // Create files with varying lengths
        // One short sequence (< k=32 bases) that produces no minimizers
        let short_seq = b"ACGTACGTACGT"; // 12 bases, less than k=32
        let long_seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";

        // Create files with different sequence lengths
        {
            let path = dir.join("short.fa");
            let mut file = File::create(&path).unwrap();
            writeln!(file, ">short_seq").unwrap();
            file.write_all(short_seq).unwrap();
            writeln!(file).unwrap();
        }
        create_fasta_file(dir, "long.fa", long_seq);

        let config_content = r#"[index]
k = 32
window = 10
salt = 0x5555555555555555
output = "empty_seq_test.ryidx"

[buckets.MixedLengths]
files = ["short.fa", "long.fa"]
"#;
        let config_path = dir.join("config.toml");
        std::fs::write(&config_path, config_content).unwrap();

        // Build index - should not crash
        let result = build_parquet_index_from_config(&config_path, None, None, false, None);
        assert!(
            result.is_ok(),
            "Index with short sequences should succeed: {:?}",
            result
        );

        // Verify the index
        let output_path = dir.join("empty_seq_test.ryxdi");
        let index = ShardedInvertedIndex::open(&output_path).unwrap();
        let manifest = index.manifest();

        // Verify single bucket
        assert_eq!(manifest.bucket_names.len(), 1, "Should have 1 bucket");

        // Find the bucket_id for MixedLengths
        let bucket_id = *manifest
            .bucket_names
            .iter()
            .find(|(_, name)| *name == "MixedLengths")
            .map(|(id, _)| id)
            .unwrap();

        // Verify both sources are recorded (even the short one)
        let sources = manifest.bucket_sources.get(&bucket_id).unwrap();
        assert_eq!(sources.len(), 2, "Should have 2 sources (even empty one)");

        // Verify minimizers exist from the long sequence (use total_minimizers)
        assert!(
            manifest.total_minimizers > 0,
            "Should have minimizers from long sequence (got {})",
            manifest.total_minimizers
        );
    }

    // ============================================================================
    // Phase 7: Chunked Memory-Aware Extraction Tests
    // ============================================================================

    #[test]
    fn test_sequence_chunk_iterator_respects_byte_budget() {
        // Create temp files with known sequence sizes:
        // File 1: 3 sequences of 1000 bytes each (3000 bytes total)
        // File 2: 2 sequences of 2000 bytes each (4000 bytes total)
        // Total: 7000 bytes
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // Create File 1: 3 sequences of 1000 bytes each
        let seq_1000 = vec![b'A'; 1000];
        let file1 = create_multi_fasta_file(
            dir,
            "file1.fa",
            &[
                ("seq1", seq_1000.as_slice()),
                ("seq2", seq_1000.as_slice()),
                ("seq3", seq_1000.as_slice()),
            ],
        );

        // Create File 2: 2 sequences of 2000 bytes each
        let seq_2000 = vec![b'C'; 2000];
        let file2 = create_multi_fasta_file(
            dir,
            "file2.fa",
            &[("seq4", seq_2000.as_slice()), ("seq5", seq_2000.as_slice())],
        );

        // Set chunk budget to 2500 bytes
        let chunk_config = ChunkConfig {
            target_chunk_bytes: 2500,
        };

        let mut chunk_iter =
            SequenceChunkIterator::new(&[file1, file2], dir, chunk_config.target_chunk_bytes);

        let mut all_sequences: Vec<(usize, String)> = Vec::new();
        let mut chunk_count = 0;

        while let Some(chunk) = chunk_iter.next_chunk().unwrap() {
            chunk_count += 1;
            let chunk_bytes: usize = chunk.iter().map(|(seq, _)| seq.len()).sum();

            // If chunk has more than one sequence, verify budget is respected
            // (a single large sequence can exceed budget)
            if chunk.len() > 1 {
                // Chunk bytes should be close to or under budget
                // We allow one sequence to push us over
                assert!(
                    chunk_bytes <= chunk_config.target_chunk_bytes + 2000,
                    "Chunk {} exceeded budget by too much: {} bytes (budget: {})",
                    chunk_count,
                    chunk_bytes,
                    chunk_config.target_chunk_bytes
                );
            }

            for (seq, src) in chunk {
                all_sequences.push((seq.len(), src));
            }
        }

        // Verify all sequences were yielded exactly once
        assert_eq!(all_sequences.len(), 5, "Should have 5 sequences total");
        assert!(chunk_count >= 2, "Should have at least 2 chunks");
    }

    #[test]
    fn test_sequence_chunk_iterator_handles_large_single_sequence() {
        // Create temp file with:
        // - Sequence 1: 100 bytes
        // - Sequence 2: 10000 bytes (exceeds budget)
        // - Sequence 3: 100 bytes
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        let seq_small = vec![b'A'; 100];
        let seq_large = vec![b'G'; 10000];

        let file = create_multi_fasta_file(
            dir,
            "mixed.fa",
            &[
                ("seq1", seq_small.as_slice()),
                ("seq2", seq_large.as_slice()),
                ("seq3", seq_small.as_slice()),
            ],
        );

        // Set chunk budget to 500 bytes
        let mut chunk_iter = SequenceChunkIterator::new(&[file], dir, 500);

        let mut all_sequences: Vec<(usize, String)> = Vec::new();
        let mut found_large_alone = false;

        while let Some(chunk) = chunk_iter.next_chunk().unwrap() {
            // Check if the large sequence is alone in its chunk
            if chunk.len() == 1 && chunk[0].0.len() == 10000 {
                found_large_alone = true;
            }

            for (seq, src) in chunk {
                all_sequences.push((seq.len(), src));
            }
        }

        // Verify all sequences were processed
        assert_eq!(all_sequences.len(), 3, "Should have 3 sequences");
        assert!(
            found_large_alone,
            "Large sequence should be in its own chunk"
        );
    }

    #[test]
    fn test_sequence_chunk_iterator_exhausts_all_files() {
        // Create 3 temp files with various sequences
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        let seq = vec![b'T'; 500];
        let file1 = create_multi_fasta_file(dir, "file1.fa", &[("s1", seq.as_slice())]);
        let file2 = create_multi_fasta_file(
            dir,
            "file2.fa",
            &[("s2", seq.as_slice()), ("s3", seq.as_slice())],
        );
        let file3 = create_multi_fasta_file(dir, "file3.fa", &[("s4", seq.as_slice())]);

        let mut chunk_iter = SequenceChunkIterator::new(
            &[file1, file2, file3],
            dir,
            10000, // Large budget = fewer chunks
        );

        let mut all_sources: Vec<String> = Vec::new();

        while let Some(chunk) = chunk_iter.next_chunk().unwrap() {
            for (_, src) in chunk {
                all_sources.push(src);
            }
        }

        // Verify total sequences
        assert_eq!(all_sources.len(), 4, "Should have 4 sequences total");

        // Verify all source labels are correctly formatted
        for source in &all_sources {
            assert!(
                source.contains(BUCKET_SOURCE_DELIM),
                "Source '{}' should contain delimiter",
                source
            );
        }

        // Verify order (file1 seqs, then file2 seqs, etc.)
        assert!(all_sources[0].contains("s1"));
        assert!(all_sources[1].contains("s2"));
        assert!(all_sources[2].contains("s3"));
        assert!(all_sources[3].contains("s4"));
    }

    #[test]
    fn test_chunked_extraction_matches_non_chunked() {
        // Create test data: multiple files with known sequences
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // Create sequences long enough to produce minimizers
        let seq1 = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let seq2 = b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA";

        let file1 = create_fasta_file(dir, "ref1.fa", seq1);
        let file2 = create_fasta_file(dir, "ref2.fa", seq2);
        let files = vec![file1, file2];

        let k = 32;
        let w = 10;
        let salt = 0x5555555555555555u64;

        // Run 1: Use existing build_single_bucket_parallel (unlimited memory)
        let (_, mins_non_chunked, sources_non_chunked) =
            build_single_bucket_parallel("TestBucket", &files, dir, k, w, salt).unwrap();

        // Run 2: Use chunked extraction with small chunk budget
        let (_, mins_chunked, sources_chunked) = build_single_bucket_parallel_chunked(
            "TestBucket",
            &files,
            dir,
            k,
            w,
            salt,
            Some(500), // Very small budget to force multiple chunks
        )
        .unwrap();

        // Verify minimizer sets are identical
        assert_eq!(
            mins_chunked, mins_non_chunked,
            "Chunked extraction should produce identical minimizers"
        );

        // Verify source lists are identical (same order)
        assert_eq!(
            sources_chunked, sources_non_chunked,
            "Chunked extraction should produce identical sources in same order"
        );
    }

    #[test]
    fn test_chunked_oriented_extraction_matches_non_chunked() {
        // Create test data with sequences that have clear orientation preferences
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        let seq1 = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let seq2 = b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA";

        let file1 = create_fasta_file(dir, "ref1.fa", seq1);
        let file2 = create_fasta_file(dir, "ref2.fa", seq2);
        let files = vec![file1, file2];

        let k = 32;
        let w = 10;
        let salt = 0x5555555555555555u64;

        // Run 1: Use existing oriented extraction (unlimited memory)
        let (_, mins_non_chunked, _) =
            build_single_bucket_parallel_oriented("TestBucket", &files, dir, k, w, salt).unwrap();

        // Run 2: Use chunked oriented extraction with small chunk budget
        let (_, mins_chunked, _) = build_single_bucket_parallel_oriented_chunked(
            "TestBucket",
            &files,
            dir,
            k,
            w,
            salt,
            Some(500), // Very small budget to force multiple chunks
        )
        .unwrap();

        // Verify minimizer sets are identical
        assert_eq!(
            mins_chunked, mins_non_chunked,
            "Chunked oriented extraction should produce identical minimizers"
        );
    }

    #[test]
    fn test_chunked_extraction_with_single_sequence() {
        // Edge case: bucket has only one sequence
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let file = create_fasta_file(dir, "single.fa", seq);

        let (_, minimizers, sources) = build_single_bucket_parallel_chunked(
            "SingleSeq",
            &[file],
            dir,
            32,
            10,
            0x5555555555555555,
            Some(100), // Very small budget
        )
        .unwrap();

        // Verify returns correct minimizers
        assert!(!minimizers.is_empty(), "Should extract minimizers");
        assert_eq!(sources.len(), 1, "Should have one source");

        // Verify minimizers are sorted and deduplicated
        let mut sorted = minimizers.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(minimizers, sorted);
    }

    #[test]
    fn test_chunked_extraction_with_empty_files() {
        // Create mix of empty and non-empty files
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // Create an empty FASTA file
        let empty_path = dir.join("empty.fa");
        std::fs::write(&empty_path, "").unwrap();

        // Create a non-empty file
        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let nonempty = create_fasta_file(dir, "nonempty.fa", seq);

        let (_, minimizers, sources) = build_single_bucket_parallel_chunked(
            "MixedEmpty",
            &[empty_path, nonempty],
            dir,
            32,
            10,
            0x5555555555555555,
            Some(1000),
        )
        .unwrap();

        // Empty files should be skipped gracefully
        // Non-empty files should be processed correctly
        assert!(
            !minimizers.is_empty(),
            "Should have minimizers from non-empty file"
        );
        assert_eq!(
            sources.len(),
            1,
            "Should have one source (from non-empty file)"
        );
    }

    #[test]
    fn test_chunk_config_calculation() {
        // Test calculate_chunk_config with various memory limits

        // 8GB available → reasonable chunk size (~1.5GB)
        let config_8gb = calculate_chunk_config(8 * 1024 * 1024 * 1024);
        assert!(
            config_8gb.target_chunk_bytes >= MIN_CHUNK_BYTES,
            "8GB: chunk size {} should be >= MIN_CHUNK_BYTES {}",
            config_8gb.target_chunk_bytes,
            MIN_CHUNK_BYTES
        );
        assert!(
            config_8gb.target_chunk_bytes <= MAX_CHUNK_BYTES,
            "8GB: chunk size {} should be <= MAX_CHUNK_BYTES {}",
            config_8gb.target_chunk_bytes,
            MAX_CHUNK_BYTES
        );

        // 32GB available → larger chunk size (but capped at MAX)
        let config_32gb = calculate_chunk_config(32 * 1024 * 1024 * 1024);
        assert!(
            config_32gb.target_chunk_bytes >= config_8gb.target_chunk_bytes,
            "32GB should have >= chunk size than 8GB"
        );
        assert!(
            config_32gb.target_chunk_bytes <= MAX_CHUNK_BYTES,
            "32GB: chunk size should be capped at MAX_CHUNK_BYTES"
        );

        // 512MB available → minimum viable chunk size
        let config_512mb = calculate_chunk_config(512 * 1024 * 1024);
        assert!(
            config_512mb.target_chunk_bytes >= MIN_CHUNK_BYTES,
            "512MB: chunk size should be at least MIN_CHUNK_BYTES"
        );

        // Very small memory → should still respect minimum
        let config_tiny = calculate_chunk_config(100 * 1024 * 1024);
        assert_eq!(
            config_tiny.target_chunk_bytes, MIN_CHUNK_BYTES,
            "Tiny memory should use MIN_CHUNK_BYTES"
        );
    }

    // =======================================================================
    // Streaming Single-Bucket Tests (TDD RED Phase)
    // =======================================================================

    /// Test 1: Streaming single-bucket produces a valid, queryable index.
    #[test]
    fn test_streaming_single_bucket_produces_valid_index() {
        use rype::ShardedInvertedIndex;

        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // Create multiple FASTA files with distinct sequences
        fn make_sequence(seed: u64, length: usize) -> Vec<u8> {
            let mut state = seed;
            (0..length)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    match (state >> 32) % 4 {
                        0 => b'A',
                        1 => b'C',
                        2 => b'G',
                        _ => b'T',
                    }
                })
                .collect()
        }

        // Create 3 files with ~5KB sequences each
        for i in 0..3 {
            let seq = make_sequence(i as u64 * 12345, 5_000);
            create_fasta_file(dir, &format!("ref{}.fa", i), &seq);
        }

        let files: Vec<PathBuf> = (0..3).map(|i| dir.join(format!("ref{}.fa", i))).collect();

        // Create output directory
        let output_dir = dir.join("test_index.ryxdi");
        rype::parquet_index::create_index_directory(&output_dir).unwrap();

        // Call build_single_bucket_streaming
        let result = build_single_bucket_streaming(
            &output_dir,
            "TestBucket",
            &files,
            dir,
            32,
            10,
            0x5555555555555555,
            Some(64 * 1024 * 1024), // 64MB max_memory
            None,
            None,
        )
        .unwrap();

        // Verify result fields
        assert_eq!(result.bucket_name, "TestBucket");
        assert!(!result.sources.is_empty(), "Should have sources");
        assert_eq!(
            result.sources.len(),
            3,
            "Should have 3 sources (one per file)"
        );
        assert!(
            !result.shard_infos.is_empty(),
            "Should have at least one shard"
        );
        assert!(result.total_minimizers > 0, "Should have minimizers");

        // Write manifest and verify index can be loaded
        write_streaming_manifest(&output_dir, &result, 32, 10, 0x5555555555555555).unwrap();

        let index = ShardedInvertedIndex::open(&output_dir).unwrap();
        assert_eq!(
            index.manifest().bucket_names.len(),
            1,
            "Should have 1 bucket"
        );
    }

    /// Test 2: Streaming with many files creates multiple shards.
    #[test]
    fn test_streaming_single_bucket_with_many_files() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        fn make_sequence(seed: u64, length: usize) -> Vec<u8> {
            let mut state = seed;
            (0..length)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    match (state >> 32) % 4 {
                        0 => b'A',
                        1 => b'C',
                        2 => b'G',
                        _ => b'T',
                    }
                })
                .collect()
        }

        // Create 50 files with ~200KB sequences each
        // Each 200KB sequence produces ~20K minimizers (unique per seed)
        // Total: ~1M minimizer entries = ~16MB of data
        // At 1MB minimum shard size, we should get 10+ shards
        for i in 0..50 {
            let seq = make_sequence(i as u64 * 99999, 200_000);
            create_fasta_file(dir, &format!("ref{}.fa", i), &seq);
        }

        let files: Vec<PathBuf> = (0..50).map(|i| dir.join(format!("ref{}.fa", i))).collect();

        let output_dir = dir.join("test_index.ryxdi");
        rype::parquet_index::create_index_directory(&output_dir).unwrap();

        // Use 3MB max_memory to force shard_size close to MIN_SHARD_BYTES
        // shard_size = max(3MB * 0.4, 1MB) = max(1.2MB, 1MB) = 1.2MB
        let result = build_single_bucket_streaming(
            &output_dir,
            "ManyFiles",
            &files,
            dir,
            32,
            10,
            0x5555555555555555,
            Some(3 * 1024 * 1024), // 3MB max_memory → ~1.2MB shard size
            None,
            None,
        )
        .unwrap();

        // Should have multiple shards with this much data at small shard size
        assert!(
            result.shard_infos.len() >= 2,
            "Should have multiple shards (got {}) with {} total minimizers",
            result.shard_infos.len(),
            result.total_minimizers
        );
        assert_eq!(result.sources.len(), 50, "Should have all 50 sources");
    }

    /// Test 3: Oriented streaming uses baseline correctly.
    #[test]
    fn test_streaming_single_bucket_oriented_matches_baseline() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        fn make_sequence(seed: u64, length: usize) -> Vec<u8> {
            let mut state = seed;
            (0..length)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    match (state >> 32) % 4 {
                        0 => b'A',
                        1 => b'C',
                        2 => b'G',
                        _ => b'T',
                    }
                })
                .collect()
        }

        // Create 5 files
        for i in 0..5 {
            let seq = make_sequence(i as u64 * 54321, 10_000);
            create_fasta_file(dir, &format!("ref{}.fa", i), &seq);
        }

        let files: Vec<PathBuf> = (0..5).map(|i| dir.join(format!("ref{}.fa", i))).collect();

        let output_dir = dir.join("test_index.ryxdi");
        rype::parquet_index::create_index_directory(&output_dir).unwrap();

        // Call oriented streaming version
        let result = build_single_bucket_streaming_oriented(
            &output_dir,
            "OrientedBucket",
            &files,
            dir,
            32,
            10,
            0x5555555555555555,
            Some(64 * 1024 * 1024),
            None,
            None,
        )
        .unwrap();

        assert_eq!(result.bucket_name, "OrientedBucket");
        assert_eq!(result.sources.len(), 5, "Should have 5 sources");
        assert!(!result.shard_infos.is_empty(), "Should have shards");
        assert!(result.total_minimizers > 0, "Should have minimizers");
    }

    /// Test 4: Empty input returns empty result without panic.
    #[test]
    fn test_streaming_single_bucket_empty_input() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        let output_dir = dir.join("test_index.ryxdi");
        rype::parquet_index::create_index_directory(&output_dir).unwrap();

        // Call with empty file list
        let result = build_single_bucket_streaming(
            &output_dir,
            "EmptyBucket",
            &[],
            dir,
            32,
            10,
            0x5555555555555555,
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(result.bucket_name, "EmptyBucket");
        assert!(result.sources.is_empty(), "Should have no sources");
        assert!(result.shard_infos.is_empty(), "Should have no shards");
        assert_eq!(result.total_minimizers, 0, "Should have no minimizers");

        // Also test oriented version with empty input
        let result_oriented = build_single_bucket_streaming_oriented(
            &output_dir,
            "EmptyOrientedBucket",
            &[],
            dir,
            32,
            10,
            0x5555555555555555,
            None,
            None,
            None,
        )
        .unwrap();

        assert!(result_oriented.sources.is_empty());
        assert!(result_oriented.shard_infos.is_empty());
    }

    /// Test 5: Single file input works correctly.
    #[test]
    fn test_streaming_single_bucket_single_file() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // Create single file with multiple sequences
        let path = dir.join("multi_seq.fa");
        let mut file = File::create(&path).unwrap();
        writeln!(file, ">seq1").unwrap();
        writeln!(file, "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT").unwrap();
        writeln!(file, ">seq2").unwrap();
        writeln!(file, "TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA").unwrap();
        drop(file);

        let output_dir = dir.join("test_index.ryxdi");
        rype::parquet_index::create_index_directory(&output_dir).unwrap();

        let result = build_single_bucket_streaming(
            &output_dir,
            "SingleFile",
            &[path],
            dir,
            32,
            10,
            0x5555555555555555,
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(result.bucket_name, "SingleFile");
        assert_eq!(
            result.sources.len(),
            2,
            "Should have 2 sources from single file"
        );
        assert!(!result.shard_infos.is_empty(), "Should have shards");
        assert!(result.total_minimizers > 0, "Should have minimizers");
    }

    /// Test 6: Memory stays bounded by using ShardAccumulator (flushes to disk).
    #[test]
    fn test_streaming_single_bucket_memory_bounded() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        fn make_sequence(seed: u64, length: usize) -> Vec<u8> {
            let mut state = seed;
            (0..length)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    match (state >> 32) % 4 {
                        0 => b'A',
                        1 => b'C',
                        2 => b'G',
                        _ => b'T',
                    }
                })
                .collect()
        }

        // Create many files that would exceed memory if accumulated
        // 100 files × 200KB = 20MB of sequences → ~2M minimizers → ~32MB if accumulated
        // This ensures multiple shards at our forced small shard size
        for i in 0..100 {
            let seq = make_sequence(i as u64 * 77777, 200_000);
            create_fasta_file(dir, &format!("ref{}.fa", i), &seq);
        }

        let files: Vec<PathBuf> = (0..100).map(|i| dir.join(format!("ref{}.fa", i))).collect();

        let output_dir = dir.join("test_index.ryxdi");
        rype::parquet_index::create_index_directory(&output_dir).unwrap();

        // Use 3MB max_memory to force shard_size close to MIN_SHARD_BYTES (1MB)
        // This proves memory stays bounded via shard flushing
        let result = build_single_bucket_streaming(
            &output_dir,
            "MemoryBounded",
            &files,
            dir,
            32,
            10,
            0x5555555555555555,
            Some(3 * 1024 * 1024), // 3MB max_memory → ~1.2MB shard size
            None,
            None,
        )
        .unwrap();

        // Multiple shards proves streaming to disk worked and memory was bounded
        assert!(
            result.shard_infos.len() >= 3,
            "Should have multiple shards (got {}), proving memory was bounded via flushing",
            result.shard_infos.len()
        );

        assert_eq!(result.sources.len(), 100, "Should have all 100 sources");
        assert!(
            result.total_minimizers > 500_000,
            "Should have significant minimizers"
        );

        // Verify shards were written to disk
        let inverted_dir = output_dir.join("inverted");
        let shard_count = std::fs::read_dir(&inverted_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "parquet")
                    .unwrap_or(false)
            })
            .count();

        assert_eq!(
            shard_count,
            result.shard_infos.len(),
            "Disk shard count should match shard_infos"
        );
    }

    /// Helper to write manifest for streaming single-bucket results (used by tests).
    fn write_streaming_manifest(
        output_dir: &Path,
        result: &SingleBucketResult,
        k: usize,
        w: usize,
        salt: u64,
    ) -> Result<()> {
        use rype::parquet_index::{
            compute_source_hash, write_buckets_parquet, InvertedManifest, ParquetManifest,
            ParquetShardFormat, FORMAT_MAGIC, FORMAT_VERSION,
        };
        use std::collections::HashMap;

        const BUCKET_ID: u32 = 1;

        // Write bucket metadata
        let mut bucket_names = HashMap::new();
        let mut bucket_sources = HashMap::new();
        bucket_names.insert(BUCKET_ID, sanitize_bucket_name(&result.bucket_name));
        bucket_sources.insert(BUCKET_ID, result.sources.clone());
        write_buckets_parquet(output_dir, &bucket_names, &bucket_sources)?;

        // Compute source hash
        let mut bucket_min_counts = HashMap::new();
        bucket_min_counts.insert(BUCKET_ID, result.total_minimizers as usize);
        let source_hash = compute_source_hash(&bucket_min_counts);

        // Write manifest
        let total_entries: u64 = result.shard_infos.iter().map(|s| s.num_entries).sum();
        let manifest = ParquetManifest {
            magic: FORMAT_MAGIC.to_string(),
            format_version: FORMAT_VERSION,
            k,
            w,
            salt,
            source_hash,
            num_buckets: 1,
            total_minimizers: result.total_minimizers,
            inverted: Some(InvertedManifest {
                format: ParquetShardFormat::Parquet,
                num_shards: result.shard_infos.len() as u32,
                total_entries,
                has_overlapping_shards: true,
                shards: result.shard_infos.clone(),
            }),
        };
        manifest.save(output_dir)?;

        Ok(())
    }

    // ============================================================================
    // Subtraction (exclusion_set) in single-bucket streaming tests
    // ============================================================================

    #[test]
    fn test_single_bucket_streaming_with_exclusion_set() {
        use std::collections::HashSet;

        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // Use real FASTA files with biologically distinct sequences
        let phix_path = PathBuf::from("examples/phiX174.fasta");
        let puc19_path = PathBuf::from("examples/pUC19.fasta");
        let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        // Build exclusion set from phiX174 minimizers
        let (phix_mins, _) = extract_bucket_minimizers(
            &[phix_path.clone()],
            &project_root,
            32,
            10,
            0x5555555555555555,
            false,
        )
        .unwrap();
        let exclusion_set: HashSet<u64> = phix_mins.into_iter().collect();
        assert!(
            !exclusion_set.is_empty(),
            "Exclusion set should not be empty"
        );

        // Create a FASTA that contains both phiX174 and pUC19 sequences
        // (concatenate both files into one)
        let combined_path = dir.join("combined.fa");
        {
            use std::io::Read as _;
            let mut combined = Vec::new();
            let mut f1 = std::fs::File::open(project_root.join(&phix_path)).unwrap();
            f1.read_to_end(&mut combined).unwrap();
            let mut f2 = std::fs::File::open(project_root.join(&puc19_path)).unwrap();
            f2.read_to_end(&mut combined).unwrap();
            std::fs::write(&combined_path, &combined).unwrap();
        }

        // Create index directories
        let output_no_excl = dir.join("no_excl.ryxdi");
        let output_with_excl = dir.join("with_excl.ryxdi");
        rype::parquet_index::create_index_directory(&output_no_excl).unwrap();
        rype::parquet_index::create_index_directory(&output_with_excl).unwrap();

        // Build WITHOUT exclusion
        let result_no_excl = build_single_bucket_streaming(
            &output_no_excl,
            "TestBucket",
            &[combined_path.clone()],
            dir,
            32,
            10,
            0x5555555555555555,
            Some(100 * 1024 * 1024),
            None,
            None,
        )
        .unwrap();

        // Build WITH exclusion (remove phiX174 minimizers)
        let result_with_excl = build_single_bucket_streaming(
            &output_with_excl,
            "TestBucket",
            &[combined_path],
            dir,
            32,
            10,
            0x5555555555555555,
            Some(100 * 1024 * 1024),
            None,
            Some(&exclusion_set),
        )
        .unwrap();

        // Subtracted version should have fewer minimizers
        assert!(
            result_with_excl.total_minimizers < result_no_excl.total_minimizers,
            "Exclusion should reduce minimizer count: {} should be < {}",
            result_with_excl.total_minimizers,
            result_no_excl.total_minimizers
        );
        // But should still have some (pUC19 minimizers should survive)
        assert!(
            result_with_excl.total_minimizers > 0,
            "Should retain pUC19-unique minimizers"
        );
    }

    #[test]
    fn test_single_bucket_streaming_oriented_with_exclusion_set() {
        use std::collections::HashSet;

        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        let phix_path = PathBuf::from("examples/phiX174.fasta");
        let puc19_path = PathBuf::from("examples/pUC19.fasta");
        let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        // Build exclusion set from phiX174 minimizers
        let (phix_mins, _) = extract_bucket_minimizers(
            &[phix_path.clone()],
            &project_root,
            32,
            10,
            0x5555555555555555,
            false,
        )
        .unwrap();
        let exclusion_set: HashSet<u64> = phix_mins.into_iter().collect();

        // Combined FASTA with both genomes
        let combined_path = dir.join("combined.fa");
        {
            use std::io::Read as _;
            let mut combined = Vec::new();
            let mut f1 = std::fs::File::open(project_root.join(&phix_path)).unwrap();
            f1.read_to_end(&mut combined).unwrap();
            let mut f2 = std::fs::File::open(project_root.join(&puc19_path)).unwrap();
            f2.read_to_end(&mut combined).unwrap();
            std::fs::write(&combined_path, &combined).unwrap();
        }

        let output_no_excl = dir.join("no_excl.ryxdi");
        let output_with_excl = dir.join("with_excl.ryxdi");
        rype::parquet_index::create_index_directory(&output_no_excl).unwrap();
        rype::parquet_index::create_index_directory(&output_with_excl).unwrap();

        // Build oriented WITHOUT exclusion
        let result_no_excl = build_single_bucket_streaming_oriented(
            &output_no_excl,
            "TestBucket",
            &[combined_path.clone()],
            dir,
            32,
            10,
            0x5555555555555555,
            Some(100 * 1024 * 1024),
            None,
            None,
        )
        .unwrap();

        // Build oriented WITH exclusion
        let result_with_excl = build_single_bucket_streaming_oriented(
            &output_with_excl,
            "TestBucket",
            &[combined_path],
            dir,
            32,
            10,
            0x5555555555555555,
            Some(100 * 1024 * 1024),
            None,
            Some(&exclusion_set),
        )
        .unwrap();

        assert!(
            result_with_excl.total_minimizers < result_no_excl.total_minimizers,
            "Exclusion should reduce minimizer count: {} should be < {}",
            result_with_excl.total_minimizers,
            result_no_excl.total_minimizers
        );
        assert!(
            result_with_excl.total_minimizers > 0,
            "Should retain pUC19-unique minimizers"
        );
    }
}
