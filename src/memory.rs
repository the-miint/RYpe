//! Memory utilities for adaptive batch sizing.
//!
//! Provides:
//! - Byte suffix parsing (e.g., "4G", "512M", "1.5G")
//! - Platform-specific memory detection
//! - Batch size calculation based on memory constraints

use crate::error::{Result, RypeError};

/// Parse a byte size string with optional suffix.
///
/// Supports:
/// - Integer values: "1024" -> 1024 bytes
/// - Decimal values with suffix: "1.5G" -> 1.5 * 1024^3 bytes
/// - Suffixes (case-insensitive): B, K, KB, M, MB, G, GB, T, TB
/// - "auto" returns None (signals auto-detection)
///
/// # Examples
/// ```
/// use rype::memory::parse_byte_suffix;
/// assert_eq!(parse_byte_suffix("4G").unwrap(), Some(4 * 1024 * 1024 * 1024));
/// assert_eq!(parse_byte_suffix("512M").unwrap(), Some(512 * 1024 * 1024));
/// assert_eq!(parse_byte_suffix("auto").unwrap(), None);
/// ```
pub fn parse_byte_suffix(s: &str) -> Result<Option<usize>> {
    let s = s.trim();

    if s.eq_ignore_ascii_case("auto") {
        return Ok(None);
    }

    // Find where the numeric part ends
    let numeric_end = s
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(s.len());

    if numeric_end == 0 {
        return Err(RypeError::validation(format!(
            "Invalid byte size: '{}' (no numeric value)",
            s
        )));
    }

    let numeric_part = &s[..numeric_end];
    let suffix_part = s[numeric_end..].trim();

    let value: f64 = numeric_part
        .parse()
        .map_err(|_| RypeError::validation(format!("Invalid numeric value: '{}'", numeric_part)))?;

    if value < 0.0 {
        return Err(RypeError::validation(format!(
            "Byte size cannot be negative: {}",
            value
        )));
    }

    let multiplier: u64 = match suffix_part.to_ascii_uppercase().as_str() {
        "" | "B" => 1,
        "K" | "KB" => 1024,
        "M" | "MB" => 1024 * 1024,
        "G" | "GB" => 1024 * 1024 * 1024,
        "T" | "TB" => 1024 * 1024 * 1024 * 1024,
        _ => {
            return Err(RypeError::validation(format!(
                "Unknown byte suffix: '{}' (use B, K, M, G, or T)",
                suffix_part
            )))
        }
    };

    let result = value * multiplier as f64;
    if !result.is_finite() || result < 0.0 || result > usize::MAX as f64 {
        return Err(RypeError::validation(format!(
            "Byte size overflow: '{}' exceeds maximum representable value",
            s
        )));
    }
    let bytes = result.round() as usize;
    Ok(Some(bytes))
}

/// Source of available memory information.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemorySource {
    /// cgroups v2 memory.max
    CgroupsV2,
    /// cgroups v1 memory.limit_in_bytes
    CgroupsV1,
    /// SLURM job info (scontrol show job)
    Slurm,
    /// /proc/meminfo MemAvailable
    ProcMeminfo,
    /// macOS sysctl hw.memsize
    #[allow(dead_code)]
    MacOsSysctl,
    /// Fallback default (8GB)
    Fallback,
}

/// Result of available memory detection.
#[derive(Debug, Clone)]
pub struct AvailableMemory {
    pub bytes: usize,
    pub source: MemorySource,
}

/// Default fallback memory (8GB).
pub const FALLBACK_MEMORY_BYTES: usize = 8 * 1024 * 1024 * 1024;

/// Detect available system memory.
///
/// On Linux, tries (in order):
/// 1. cgroups v2: /sys/fs/cgroup/memory.max
/// 2. cgroups v1: /sys/fs/cgroup/memory/memory.limit_in_bytes
/// 3. SLURM: scontrol show job $SLURM_JOB_ID
/// 4. /proc/meminfo MemAvailable field
///
/// On macOS: sysctl hw.memsize
///
/// Falls back to 8GB if detection fails.
pub fn detect_available_memory() -> AvailableMemory {
    #[cfg(target_os = "linux")]
    {
        // Try cgroups v2 first
        if let Some(bytes) = read_cgroups_v2_limit() {
            return AvailableMemory {
                bytes,
                source: MemorySource::CgroupsV2,
            };
        }

        // Try cgroups v1
        if let Some(bytes) = read_cgroups_v1_limit() {
            return AvailableMemory {
                bytes,
                source: MemorySource::CgroupsV1,
            };
        }

        // Try SLURM job info
        if let Some(bytes) = read_slurm_job_memory() {
            return AvailableMemory {
                bytes,
                source: MemorySource::Slurm,
            };
        }

        // Try /proc/meminfo
        if let Some(bytes) = read_proc_meminfo_available() {
            return AvailableMemory {
                bytes,
                source: MemorySource::ProcMeminfo,
            };
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Some(bytes) = read_macos_memsize() {
            return AvailableMemory {
                bytes,
                source: MemorySource::MacOsSysctl,
            };
        }
    }

    // Fallback
    AvailableMemory {
        bytes: FALLBACK_MEMORY_BYTES,
        source: MemorySource::Fallback,
    }
}

#[cfg(target_os = "linux")]
fn read_cgroups_v2_limit() -> Option<usize> {
    // Find the process's cgroup path from /proc/self/cgroup
    // v2 format: "0::<path>"
    let cgroup_content = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let mut cgroup_path = None;

    for line in cgroup_content.lines() {
        let parts: Vec<&str> = line.splitn(3, ':').collect();
        if parts.len() == 3 && parts[0] == "0" && parts[1].is_empty() {
            let path = parts[2];
            if !path.is_empty() && path != "/" {
                cgroup_path = Some(path.to_string());
            }
            break;
        }
    }

    let path = cgroup_path?;
    let memory_max_path = format!("/sys/fs/cgroup{}/memory.max", path);
    let content = std::fs::read_to_string(&memory_max_path).ok()?;
    let trimmed = content.trim();

    // "max" means no limit
    if trimmed == "max" {
        return None;
    }

    trimmed.parse().ok()
}

#[cfg(target_os = "linux")]
fn read_cgroups_v1_limit() -> Option<usize> {
    // Find the process's memory cgroup path from /proc/self/cgroup
    let cgroup_content = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let mut memory_path = None;

    for line in cgroup_content.lines() {
        // v1 format: "<id>:<controllers>:<path>"
        // e.g., "6:memory:/slurm/uid_1156392/job_3532212/step_0/task_0"
        let parts: Vec<&str> = line.splitn(3, ':').collect();
        if parts.len() == 3 && parts[1] == "memory" {
            memory_path = Some(parts[2].to_string());
            break;
        }
    }

    let path = memory_path?;
    let limit_path = format!("/sys/fs/cgroup/memory{}/memory.limit_in_bytes", path);
    let content = std::fs::read_to_string(&limit_path).ok()?;
    let value: usize = content.trim().parse().ok()?;

    // Very large values mean "no limit" (usually 2^63 - page_size)
    // Use 1TB as a reasonable threshold
    const ONE_TB: usize = 1024 * 1024 * 1024 * 1024;
    if value > ONE_TB {
        return None;
    }

    Some(value)
}

/// Read memory limit from SLURM job info via scontrol.
/// Parses MinMemoryNode from `scontrol show job $SLURM_JOB_ID`.
#[cfg(target_os = "linux")]
fn read_slurm_job_memory() -> Option<usize> {
    use std::process::Command;

    let job_id = std::env::var("SLURM_JOB_ID").ok()?;

    let output = Command::new("scontrol")
        .args(["show", "job", &job_id])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Look for MinMemoryNode=32G or similar
    for line in stdout.lines() {
        for field in line.split_whitespace() {
            if field.starts_with("MinMemoryNode=") {
                let value = field.strip_prefix("MinMemoryNode=")?;
                return parse_slurm_memory(value);
            }
        }
    }

    None
}

/// Parse SLURM memory string (e.g., "32G", "4096M", "1T").
#[cfg(target_os = "linux")]
fn parse_slurm_memory(s: &str) -> Option<usize> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }

    // Find where numeric part ends
    let numeric_end = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    if numeric_end == 0 {
        return None;
    }

    let numeric: usize = s[..numeric_end].parse().ok()?;
    let suffix = &s[numeric_end..];

    let multiplier: usize = match suffix.to_ascii_uppercase().as_str() {
        "" | "M" => 1024 * 1024, // Default is MB
        "G" => 1024 * 1024 * 1024,
        "T" => 1024 * 1024 * 1024 * 1024,
        "K" => 1024,
        _ => return None,
    };

    numeric.checked_mul(multiplier)
}

#[cfg(target_os = "linux")]
fn read_proc_meminfo_available() -> Option<usize> {
    let content = std::fs::read_to_string("/proc/meminfo").ok()?;

    for line in content.lines() {
        if line.starts_with("MemAvailable:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let kb: usize = parts[1].parse().ok()?;
                return Some(kb * 1024); // Convert KB to bytes
            }
        }
    }

    None
}

#[cfg(target_os = "macos")]
fn read_macos_memsize() -> Option<usize> {
    use std::process::Command;

    let output = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        stdout.trim().parse().ok()
    } else {
        None
    }
}

/// Profile of read lengths for memory estimation.
#[derive(Debug, Clone)]
pub struct ReadMemoryProfile {
    /// Average length of a single read
    pub avg_read_length: usize,
    /// Average total length for a query (single or paired)
    pub avg_query_length: usize,
    /// Estimated minimizers per query
    pub minimizers_per_query: usize,
}

impl ReadMemoryProfile {
    /// Create a profile from sampling results.
    ///
    /// # Arguments
    /// * `avg_read_length` - Average length of individual reads
    /// * `is_paired` - Whether reads are paired-end
    /// * `k` - K-mer size (for estimating minimizers)
    /// * `w` - Window size (for estimating minimizers)
    pub fn new(avg_read_length: usize, is_paired: bool, k: usize, w: usize) -> Self {
        let avg_query_length = if is_paired {
            avg_read_length * 2
        } else {
            avg_read_length
        };

        // Estimate minimizers: roughly (length - k + 1) / w for each strand
        // Multiply by 2 for both strands, but many are duplicates
        let minimizers_per_query = if avg_query_length > k {
            ((avg_query_length - k + 1) / w).max(1) * 2
        } else {
            0
        };

        ReadMemoryProfile {
            avg_read_length,
            avg_query_length,
            minimizers_per_query,
        }
    }

    /// Create a default profile for when sampling isn't possible.
    pub fn default_profile(is_paired: bool, k: usize, w: usize) -> Self {
        Self::new(5000, is_paired, k, w)
    }

    /// Sample read lengths from input files to create an accurate profile.
    ///
    /// Reads the first `sample_size` records from each file to estimate average
    /// read length. For paired-end data, samples R1 and R2 separately and combines
    /// their average lengths for the query length calculation.
    ///
    /// # Arguments
    /// * `r1_path` - Path to R1 FASTQ/FASTA file (or Parquet file if `is_parquet` is true)
    /// * `r2_path` - Optional path to R2 FASTQ/FASTA file (for paired-end, ignored for Parquet)
    /// * `sample_size` - Number of records to sample from each file
    /// * `k` - K-mer size
    /// * `w` - Window size
    /// * `is_parquet` - Whether the input is Parquet format (uses sequence1/sequence2 columns)
    /// * `trim_to` - Optional maximum read length (for `--trim-to` option)
    ///
    /// # Returns
    /// A `ReadMemoryProfile` based on sampled lengths, or None if sampling fails.
    pub fn from_files(
        r1_path: &std::path::Path,
        r2_path: Option<&std::path::Path>,
        sample_size: usize,
        k: usize,
        w: usize,
        is_parquet: bool,
        trim_to: Option<usize>,
    ) -> Option<Self> {
        if is_parquet {
            // For Parquet input, sequence1 and sequence2 are in the same file
            let (total_length, count, is_paired) = sample_parquet_lengths(r1_path, sample_size)?;

            // For Parquet paired-end, total_length includes both seq1 and seq2
            // avg_query_length = total_length / count (already includes both sequences)
            // avg_read_length = total_length / (count * 2) for paired, total_length / count for single
            let avg_query_length = total_length / count;
            let avg_read_length = if is_paired {
                total_length / (count * 2)
            } else {
                avg_query_length
            };

            // Apply trim_to limit if specified
            let (avg_read_length, avg_query_length) =
                apply_trim_to_limit(avg_read_length, avg_query_length, is_paired, trim_to);

            // Estimate minimizers
            let minimizers_per_query = if avg_query_length > k {
                ((avg_query_length - k + 1) / w).max(1) * 2
            } else {
                0
            };

            return Some(ReadMemoryProfile {
                avg_read_length,
                avg_query_length,
                minimizers_per_query,
            });
        }

        // FASTX input: sample R1
        let (r1_total, r1_count) = sample_fastx_lengths(r1_path, sample_size)?;
        if r1_count == 0 {
            return None;
        }
        let avg_r1_length = r1_total / r1_count;

        // Sample R2 if provided
        let (avg_query_length, avg_read_length, is_paired) = if let Some(r2) = r2_path {
            let (r2_total, r2_count) = sample_fastx_lengths(r2, sample_size)?;
            if r2_count == 0 {
                return None;
            }
            let avg_r2_length = r2_total / r2_count;
            // For paired-end: avg_query_length = R1 avg + R2 avg
            // avg_read_length = average of individual read lengths
            let avg_read = (r1_total + r2_total) / (r1_count + r2_count);
            (avg_r1_length + avg_r2_length, avg_read, true)
        } else {
            (avg_r1_length, avg_r1_length, false)
        };

        // Apply trim_to limit if specified
        let (avg_read_length, avg_query_length) =
            apply_trim_to_limit(avg_read_length, avg_query_length, is_paired, trim_to);

        // Estimate minimizers: roughly (length - k + 1) / w for each strand
        // Multiply by 2 for both strands, but many are duplicates
        let minimizers_per_query = if avg_query_length > k {
            ((avg_query_length - k + 1) / w).max(1) * 2
        } else {
            0
        };

        Some(ReadMemoryProfile {
            avg_read_length,
            avg_query_length,
            minimizers_per_query,
        })
    }

    /// Estimate Arrow buffer bytes per row based on read lengths.
    ///
    /// Arrow string columns use:
    /// - Offset buffer: (batch_size + 1) * 4 bytes total for i32 offsets
    /// - Data buffer: actual string bytes
    /// - Validity buffer: (batch_size + 7) / 8 bytes (bit-packed)
    ///
    /// Note: This is an approximation. The actual per-row overhead from offsets
    /// and validity is amortized across the batch. We include a builder overhead
    /// factor to account for memory during RecordBatch construction.
    ///
    /// Adds overhead for read_id column (~32-80 bytes for Illumina IDs).
    pub fn estimate_arrow_bytes_per_row(&self, is_paired: bool) -> usize {
        // Number of string columns: read_id + seq1 (+ seq2 if paired)
        let num_string_cols = if is_paired { 3 } else { 2 };

        // Per-row data: sequence bytes + read_id (estimate 50 bytes for Illumina headers)
        let read_id_bytes = 50;
        let data_bytes = read_id_bytes + self.avg_query_length;

        // Amortized offset overhead: ~4 bytes per string column per row
        // (actual is (batch_size + 1) * 4 / batch_size ≈ 4 bytes per row for large batches)
        let offset_overhead = 4 * num_string_cols;

        // Validity bitmap: ~1 bit per column per row, rounded up
        let validity_overhead = (num_string_cols + 7) / 8;

        // Arrow ArrayData struct overhead per column (~40 bytes per array)
        // Amortized per row for typical batch sizes (10K rows): negligible
        // But we add a small per-row overhead to account for it
        let array_overhead = 1;

        data_bytes + offset_overhead + validity_overhead + array_overhead
    }

    /// Estimate memory per row for FASTX OwnedRecord format.
    ///
    /// `OwnedRecord = (i64, Vec<u8>, Option<Vec<u8>>)`:
    /// - 8 bytes for i64 read ID
    /// - 24 bytes `Vec<u8>` overhead for seq1 + actual sequence bytes
    /// - 24 bytes `Option<Vec<u8>>` overhead for seq2 (if paired) + actual sequence bytes
    pub fn estimate_owned_record_bytes(&self, is_paired: bool) -> usize {
        // i64 read ID
        let id_bytes = 8;
        // Vec<u8> overhead (ptr + len + capacity on 64-bit)
        let vec_overhead = 24;
        // seq1: Vec overhead + actual sequence bytes
        let seq1_bytes = vec_overhead + self.avg_read_length;
        // seq2: Option<Vec> overhead + sequence bytes if paired
        let seq2_bytes = if is_paired {
            vec_overhead + self.avg_read_length
        } else {
            0 // None variant is zero-size for memory layout
        };

        id_bytes + seq1_bytes + seq2_bytes
    }
}

/// Input format for memory estimation.
///
/// Different input formats have different memory characteristics:
/// - FASTX: Uses `OwnedRecord (i64, Vec<u8>, Option<Vec<u8>>)`
/// - Parquet: Uses Arrow RecordBatch with columnar string arrays
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputFormat {
    /// FASTX input (FASTQ/FASTA files)
    Fastx { is_paired: bool },
    /// Parquet input with Arrow RecordBatches
    Parquet { is_paired: bool },
}

impl InputFormat {
    /// Get the number of prefetch buffer slots for this format.
    pub fn prefetch_slots(&self) -> usize {
        match self {
            InputFormat::Fastx { .. } => FASTX_PREFETCH_BUFFER_SLOTS,
            InputFormat::Parquet { .. } => PARQUET_PREFETCH_BUFFER_SLOTS,
        }
    }

    /// Estimate bytes per row in the prefetch buffer for this format.
    pub fn estimate_buffer_bytes_per_row(&self, profile: &ReadMemoryProfile) -> usize {
        match self {
            InputFormat::Fastx { is_paired } => profile.estimate_owned_record_bytes(*is_paired),
            InputFormat::Parquet { is_paired } => profile.estimate_arrow_bytes_per_row(*is_paired),
        }
    }

    /// Whether this format uses paired-end reads.
    pub fn is_paired(&self) -> bool {
        match self {
            InputFormat::Fastx { is_paired } | InputFormat::Parquet { is_paired } => *is_paired,
        }
    }
}

/// Apply trim_to limit to read lengths.
///
/// When `--trim-to N` is specified (with N > 0), reads are trimmed to N nucleotides
/// before classification. This function caps the estimated read lengths accordingly
/// for accurate memory estimation.
///
/// Note: `trim_to = Some(0)` is treated as no trimming (same as `None`).
///
/// # Arguments
/// * `avg_read_length` - Average length of individual reads (before trimming)
/// * `avg_query_length` - Average total query length (before trimming)
/// * `is_paired` - Whether reads are paired-end
/// * `trim_to` - Optional trim limit (0 is treated as no limit)
///
/// # Returns
/// A tuple of (capped_avg_read_length, capped_avg_query_length)
fn apply_trim_to_limit(
    avg_read_length: usize,
    avg_query_length: usize,
    is_paired: bool,
    trim_to: Option<usize>,
) -> (usize, usize) {
    match trim_to {
        Some(limit) if limit > 0 => {
            // Cap individual read length at the trim limit
            let capped_read_length = avg_read_length.min(limit);
            // For paired-end, query length is sum of both reads (each capped)
            // For single-end, query length equals read length
            let capped_query_length = if is_paired {
                capped_read_length * 2
            } else {
                capped_read_length
            };
            (capped_read_length, capped_query_length)
        }
        // None or Some(0) - no trimming
        _ => (avg_read_length, avg_query_length),
    }
}

/// Helper function to sample read lengths from a FASTX file.
/// Returns (total_length, count) or None if the file cannot be read.
fn sample_fastx_lengths(path: &std::path::Path, sample_size: usize) -> Option<(usize, usize)> {
    use needletail::parse_fastx_file;

    let mut total_length: usize = 0;
    let mut count: usize = 0;

    let mut reader = parse_fastx_file(path).ok()?;
    while let Some(record) = reader.next() {
        let record = record.ok()?;
        total_length += record.seq().len();
        count += 1;
        if count >= sample_size {
            break;
        }
    }

    Some((total_length, count))
}

/// Helper function to sample read lengths from a Parquet file.
/// Returns (total_length, count, is_paired) or None if the file cannot be read.
///
/// Reads the first `sample_size` rows and extracts lengths from `sequence1` and
/// optionally `sequence2` columns.
fn sample_parquet_lengths(
    path: &std::path::Path,
    sample_size: usize,
) -> Option<(usize, usize, bool)> {
    use arrow::array::{Array, LargeStringArray, StringArray};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::fs::File;

    // Open Parquet file
    let file = File::open(path).ok()?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).ok()?;

    let schema = builder.schema();

    // Find sequence1 column index (required)
    let seq1_idx = schema
        .fields()
        .iter()
        .position(|f| f.name() == "sequence1")?;

    // Find sequence2 column index (optional)
    let seq2_idx = schema.fields().iter().position(|f| f.name() == "sequence2");

    // Build projection mask to only read sequence columns
    let mut col_indices = vec![seq1_idx];
    if let Some(idx) = seq2_idx {
        col_indices.push(idx);
    }
    let projection =
        parquet::arrow::ProjectionMask::roots(builder.parquet_schema(), col_indices.clone());

    // Build reader with projection
    let reader = builder.with_projection(projection).build().ok()?;

    let mut total_length: usize = 0;
    let mut count: usize = 0;
    let mut is_paired = false;
    let mut checked_paired = false;

    for batch_result in reader {
        let batch = batch_result.ok()?;

        // Get sequence1 column
        let seq1_col = batch.column_by_name("sequence1")?;

        // Get sequence2 column if present
        let seq2_col = batch.column_by_name("sequence2");

        // Check if paired on first batch with data
        if !checked_paired {
            if let Some(col) = &seq2_col {
                // Paired if sequence2 has any non-null values
                is_paired = col.null_count() < col.len();
            }
            checked_paired = true;
        }

        // Helper to get string length from either StringArray or LargeStringArray
        fn get_string_len(col: &dyn Array, idx: usize) -> Option<usize> {
            if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
                if !arr.is_null(idx) {
                    return Some(arr.value(idx).len());
                }
            } else if let Some(arr) = col.as_any().downcast_ref::<LargeStringArray>() {
                if !arr.is_null(idx) {
                    return Some(arr.value(idx).len());
                }
            }
            None
        }

        for i in 0..batch.num_rows() {
            if count >= sample_size {
                break;
            }

            // Add sequence1 length
            if let Some(len) = get_string_len(seq1_col.as_ref(), i) {
                total_length += len;
                count += 1;

                // Add sequence2 length if paired
                if is_paired {
                    if let Some(col) = &seq2_col {
                        if let Some(len2) = get_string_len(col.as_ref(), i) {
                            total_length += len2;
                        }
                    }
                }
            }
        }

        if count >= sample_size {
            break;
        }
    }

    if count == 0 {
        return None;
    }

    Some((total_length, count, is_paired))
}

/// Number of batches buffered in FASTX prefetch channel.
/// See helpers.rs PrefetchingIoHandler which uses sync_channel(2).
pub const FASTX_PREFETCH_BUFFER_SLOTS: usize = 2;

/// Number of RecordBatches buffered in Parquet prefetch channel.
/// See helpers.rs PrefetchingParquetReader which uses sync_channel(4).
pub const PARQUET_PREFETCH_BUFFER_SLOTS: usize = 4;

/// Default number of batches buffered in prefetch channel.
/// Preserved for backwards compatibility - prefer using format-specific constants.
pub const DEFAULT_PREFETCH_BUFFER_SLOTS: usize = PARQUET_PREFETCH_BUFFER_SLOTS;

/// Configuration for batch memory calculation.
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Maximum memory to use (from --max-memory or auto-detected)
    pub max_memory: usize,
    /// Number of threads available
    pub num_threads: usize,
    /// Memory used by loaded index structures
    pub index_memory: usize,
    /// Memory reservation for loading shards (largest shard size)
    pub shard_reservation: usize,
    /// Profile of read lengths
    pub read_profile: ReadMemoryProfile,
    /// Number of buckets in the index
    pub num_buckets: usize,
    /// Input format (determines prefetch buffer size and per-row memory)
    pub input_format: InputFormat,
}

impl MemoryConfig {
    /// Create a new MemoryConfig with validation.
    ///
    /// Returns an error if configuration values are invalid.
    pub fn new(
        max_memory: usize,
        num_threads: usize,
        index_memory: usize,
        shard_reservation: usize,
        read_profile: ReadMemoryProfile,
        num_buckets: usize,
        input_format: InputFormat,
    ) -> Result<Self> {
        // Validate configuration
        if max_memory == 0 {
            return Err(RypeError::validation("max_memory must be > 0"));
        }
        if num_threads == 0 {
            return Err(RypeError::validation("num_threads must be > 0"));
        }
        if num_buckets == 0 {
            return Err(RypeError::validation("num_buckets must be > 0"));
        }

        Ok(Self {
            max_memory,
            num_threads,
            index_memory,
            shard_reservation,
            read_profile,
            num_buckets,
            input_format,
        })
    }

    /// Get the number of prefetch buffer slots for the configured input format.
    pub fn prefetch_buffer_slots(&self) -> usize {
        self.input_format.prefetch_slots()
    }

    /// Get the estimated bytes per row in prefetch buffers.
    pub fn buffer_bytes_per_row(&self) -> usize {
        self.input_format
            .estimate_buffer_bytes_per_row(&self.read_profile)
    }
}

/// Result of batch configuration calculation.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Number of records per batch
    pub batch_size: usize,
    /// Number of batches to process in parallel
    pub batch_count: usize,
    /// Estimated memory per batch
    pub per_batch_memory: usize,
    /// Estimated peak memory usage
    pub peak_memory: usize,
}

/// Minimum batch size (processing fewer reads than this is inefficient).
pub const MIN_BATCH_SIZE: usize = 1000;

/// Maximum batch size.
/// Set high enough to allow efficient memory utilization for short reads.
/// The memory estimation will still constrain batch sizes based on available memory.
pub const MAX_BATCH_SIZE: usize = 5_000_000;

/// Safety margin: max(256MB, 10% of max_memory)
const SAFETY_MARGIN_PERCENT: f64 = 0.10;
const SAFETY_MARGIN_MIN_BYTES: usize = 256 * 1024 * 1024;

/// Fudge factor for memory estimation (accounts for HashMap overhead, HitResults, etc.)
///
/// This factor accounts for:
/// - HashMap internal overhead (load factor, bucket array)
/// - HitResult vectors that grow during classification
/// - Temporary allocations during minimizer extraction
/// - Rayon's per-thread workspace overhead
const MEMORY_FUDGE_FACTOR: f64 = 1.3;

/// Builder overhead factor for Arrow RecordBatch construction.
///
/// Arrow builders often pre-allocate with growth strategies (1.5x-2x).
/// This factor accounts for peak memory during RecordBatch construction
/// in the prefetch thread before the batch is handed to the main thread.
const ARROW_BUILDER_OVERHEAD: f64 = 1.5;

/// Estimate memory usage for a single batch.
///
/// Memory components:
/// - Input records: batch_size * (72 + avg_query_length) for OwnedRecord
/// - Minimizers: batch_size * minimizers_per_query * 16 bytes (`Vec<u64>` for fwd + rc)
/// - QueryInvertedIndex CSR: batch_size * minimizers_per_query * 12 bytes
/// - Accumulators: batch_size * estimated_buckets_per_read * 24 bytes (HashMap overhead)
///
/// Returns None if arithmetic overflow occurs.
pub fn estimate_batch_memory(
    batch_size: usize,
    profile: &ReadMemoryProfile,
    num_buckets: usize,
) -> Option<usize> {
    // OwnedRecord: (i64, Vec<u8>, Option<Vec<u8>>) ≈ 72 bytes + sequence data
    let record_overhead: usize = 72;
    let input_records =
        batch_size.checked_mul(record_overhead.checked_add(profile.avg_query_length)?)?;

    // Minimizer vectors: Vec<u64> for forward and reverse-complement
    let minimizer_vecs = batch_size
        .checked_mul(profile.minimizers_per_query)?
        .checked_mul(16)?;

    // QueryInvertedIndex CSR structure
    // minimizers: Vec<u64>, offsets: Vec<u32>, read_ids: Vec<u32>
    let query_index = batch_size
        .checked_mul(profile.minimizers_per_query)?
        .checked_mul(12)?;

    // Per-read accumulators: HashMap<u32, (u32, u32)>
    // Estimate ~4 buckets per read on average
    let estimated_buckets_per_read = 4.min(num_buckets);
    let accumulators = batch_size
        .checked_mul(estimated_buckets_per_read)?
        .checked_mul(24)?; // HashMap entry overhead

    // Sum components with overflow checking
    let base_estimate = input_records
        .checked_add(minimizer_vecs)?
        .checked_add(query_index)?
        .checked_add(accumulators)?;

    // Apply fudge factor (safe since we're multiplying by a small factor)
    let result = (base_estimate as f64 * MEMORY_FUDGE_FACTOR).round() as usize;
    Some(result)
}

/// Calculate I/O buffer memory overhead.
///
/// The prefetch channel can hold `prefetch_buffer_slots` batches, each containing
/// up to `batch_size` rows. This memory is shared across all parallel batches.
///
/// For Parquet input, includes builder overhead to account for memory during
/// RecordBatch construction.
///
/// Returns None if arithmetic overflow occurs.
fn estimate_io_buffer_memory(batch_size: usize, config: &MemoryConfig) -> Option<usize> {
    let prefetch_slots = config.prefetch_buffer_slots();
    let bytes_per_row = config.buffer_bytes_per_row();

    let base_memory = prefetch_slots
        .checked_mul(batch_size)?
        .checked_mul(bytes_per_row)?;

    // Apply builder overhead for Parquet (Arrow builders allocate extra capacity)
    let result = match config.input_format {
        InputFormat::Parquet { .. } => {
            (base_memory as f64 * ARROW_BUILDER_OVERHEAD).round() as usize
        }
        InputFormat::Fastx { .. } => base_memory,
    };

    Some(result)
}

/// Calculate total memory for a batch configuration including I/O buffers.
///
/// Returns None if arithmetic overflow occurs.
fn estimate_total_batch_memory(
    batch_size: usize,
    batch_count: usize,
    config: &MemoryConfig,
) -> Option<usize> {
    let per_batch = estimate_batch_memory(batch_size, &config.read_profile, config.num_buckets)?;
    let io_buffers = estimate_io_buffer_memory(batch_size, config)?;
    // I/O buffers are shared, per-batch memory scales with batch_count
    per_batch.checked_mul(batch_count)?.checked_add(io_buffers)
}

/// Calculate optimal batch configuration based on memory constraints.
///
/// Algorithm:
/// 1. Calculate available memory after subtracting index, shard reservation, and safety margin
/// 2. Start with batch_count = num_threads
/// 3. Binary search for maximum batch_size that fits in (available / batch_count)
/// 4. Validate that the result actually fits within budget
/// 5. If too tight, reduce batch_count and retry
/// 6. Enforce minimum batch_size of 1000
///
/// Note: When using Rayon for parallel batch processing, all batch_count batches may be
/// in flight simultaneously (worst case). The memory calculation accounts for this by
/// dividing available memory by batch_count, ensuring peak_memory = batch_count × per_batch_memory
/// stays within budget.
///
/// I/O buffer memory (prefetch channel) is also accounted for. This is shared across all
/// parallel batches and scales with batch_size. The prefetch buffer size depends on input
/// format: FASTX uses 2 slots, Parquet uses 4 slots.
pub fn calculate_batch_config(config: &MemoryConfig) -> BatchConfig {
    // Calculate safety margin
    let safety_margin = (config.max_memory as f64 * SAFETY_MARGIN_PERCENT).round() as usize;
    let safety_margin = safety_margin.max(SAFETY_MARGIN_MIN_BYTES);

    // Base reserved memory (not dependent on batch_size)
    let base_reserved = config
        .index_memory
        .saturating_add(config.shard_reservation)
        .saturating_add(safety_margin);
    let available = config.max_memory.saturating_sub(base_reserved);

    // Helper to create minimum config
    let make_min_config = || {
        let per_batch_memory =
            estimate_batch_memory(MIN_BATCH_SIZE, &config.read_profile, config.num_buckets)
                .unwrap_or(usize::MAX);
        let io_buffer_memory =
            estimate_io_buffer_memory(MIN_BATCH_SIZE, config).unwrap_or(usize::MAX);
        BatchConfig {
            batch_size: MIN_BATCH_SIZE,
            batch_count: 1,
            per_batch_memory,
            peak_memory: base_reserved
                .saturating_add(per_batch_memory)
                .saturating_add(io_buffer_memory),
        }
    };

    // If we have very little memory, use minimum config
    let min_total = estimate_total_batch_memory(MIN_BATCH_SIZE, 1, config);
    if min_total.map_or(true, |m| available < m) {
        return make_min_config();
    }

    // Try decreasing batch counts
    for batch_count in (1..=config.num_threads).rev() {
        // Binary search for batch_size that fits within available memory
        let batch_size = binary_search_batch_size_with_io(available, batch_count, config);

        if batch_size >= MIN_BATCH_SIZE {
            // Validate that the result actually fits within budget
            // This handles edge cases where binary search returns MIN_BATCH_SIZE
            // but that still exceeds the budget with the given batch_count
            let total = estimate_total_batch_memory(batch_size, batch_count, config);
            if total.is_some_and(|t| t <= available) {
                let per_batch_memory =
                    estimate_batch_memory(batch_size, &config.read_profile, config.num_buckets)
                        .unwrap_or(usize::MAX);
                let io_buffer_memory =
                    estimate_io_buffer_memory(batch_size, config).unwrap_or(usize::MAX);
                let peak_memory = base_reserved
                    .saturating_add(per_batch_memory.saturating_mul(batch_count))
                    .saturating_add(io_buffer_memory);

                return BatchConfig {
                    batch_size,
                    batch_count,
                    per_batch_memory,
                    peak_memory,
                };
            }
            // If validation failed, continue to try smaller batch_count
        }
    }

    // Fallback to minimum
    make_min_config()
}

/// Binary search for maximum batch size that fits in memory budget including I/O buffers.
///
/// This algorithm solves the circular dependency where I/O buffer memory depends on
/// batch_size, which in turn depends on available memory minus I/O buffers. By including
/// I/O overhead in the target function, we find the maximum batch_size where:
///   (per_batch_memory × batch_count) + io_buffer_memory <= memory_budget
///
/// Returns MIN_BATCH_SIZE if even the minimum doesn't fit (caller should validate).
fn binary_search_batch_size_with_io(
    memory_budget: usize,
    batch_count: usize,
    config: &MemoryConfig,
) -> usize {
    let mut low = MIN_BATCH_SIZE;
    let mut high = MAX_BATCH_SIZE;
    let mut best = MIN_BATCH_SIZE;

    while low <= high {
        let mid = low + (high - low) / 2;
        let total_memory = estimate_total_batch_memory(mid, batch_count, config);

        // If overflow occurred, the value is too large
        let fits = total_memory.is_some_and(|m| m <= memory_budget);

        if fits {
            best = mid;
            low = mid + 1;
        } else {
            // Use saturating_sub to handle underflow when mid = 0
            // (shouldn't happen since low starts at MIN_BATCH_SIZE, but be safe)
            if mid == 0 {
                break;
            }
            high = mid - 1;
        }
    }

    best
}
/// Format bytes as human-readable string.
pub fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;
    const TB: usize = GB * 1024;

    if bytes >= TB {
        format!("{:.2} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // === Byte suffix parsing tests ===

    #[test]
    fn test_parse_byte_suffix_gigabytes() {
        assert_eq!(
            parse_byte_suffix("4G").unwrap(),
            Some(4 * 1024 * 1024 * 1024)
        );
        assert_eq!(
            parse_byte_suffix("4GB").unwrap(),
            Some(4 * 1024 * 1024 * 1024)
        );
        assert_eq!(
            parse_byte_suffix("4g").unwrap(),
            Some(4 * 1024 * 1024 * 1024)
        );
        assert_eq!(
            parse_byte_suffix("4gb").unwrap(),
            Some(4 * 1024 * 1024 * 1024)
        );
    }

    #[test]
    fn test_parse_byte_suffix_megabytes() {
        assert_eq!(parse_byte_suffix("512M").unwrap(), Some(512 * 1024 * 1024));
        assert_eq!(parse_byte_suffix("512MB").unwrap(), Some(512 * 1024 * 1024));
        assert_eq!(parse_byte_suffix("512m").unwrap(), Some(512 * 1024 * 1024));
    }

    #[test]
    fn test_parse_byte_suffix_kilobytes() {
        assert_eq!(parse_byte_suffix("1024K").unwrap(), Some(1024 * 1024));
        assert_eq!(parse_byte_suffix("1024KB").unwrap(), Some(1024 * 1024));
    }

    #[test]
    fn test_parse_byte_suffix_bytes() {
        assert_eq!(parse_byte_suffix("1024").unwrap(), Some(1024));
        assert_eq!(parse_byte_suffix("1024B").unwrap(), Some(1024));
    }

    #[test]
    fn test_parse_byte_suffix_terabytes() {
        assert_eq!(
            parse_byte_suffix("1T").unwrap(),
            Some(1024 * 1024 * 1024 * 1024)
        );
        assert_eq!(
            parse_byte_suffix("1TB").unwrap(),
            Some(1024 * 1024 * 1024 * 1024)
        );
    }

    #[test]
    fn test_parse_byte_suffix_decimal() {
        assert_eq!(
            parse_byte_suffix("1.5G").unwrap(),
            Some((1.5 * 1024.0 * 1024.0 * 1024.0) as usize)
        );
        assert_eq!(
            parse_byte_suffix("2.5M").unwrap(),
            Some((2.5 * 1024.0 * 1024.0) as usize)
        );
    }

    #[test]
    fn test_parse_byte_suffix_auto() {
        assert_eq!(parse_byte_suffix("auto").unwrap(), None);
        assert_eq!(parse_byte_suffix("AUTO").unwrap(), None);
        assert_eq!(parse_byte_suffix("Auto").unwrap(), None);
    }

    #[test]
    fn test_parse_byte_suffix_whitespace() {
        assert_eq!(
            parse_byte_suffix("  4G  ").unwrap(),
            Some(4 * 1024 * 1024 * 1024)
        );
        assert_eq!(
            parse_byte_suffix("4 G").unwrap(),
            Some(4 * 1024 * 1024 * 1024)
        );
    }

    #[test]
    fn test_parse_byte_suffix_invalid() {
        assert!(parse_byte_suffix("").is_err());
        assert!(parse_byte_suffix("G").is_err());
        assert!(parse_byte_suffix("abc").is_err());
        assert!(parse_byte_suffix("4X").is_err());
        assert!(parse_byte_suffix("-4G").is_err());
    }

    #[test]
    fn test_parse_byte_suffix_overflow() {
        // Values that would overflow usize when multiplied
        assert!(parse_byte_suffix("99999999999G").is_err());
        assert!(parse_byte_suffix("99999999999T").is_err());
        // Infinity from f64 overflow
        assert!(parse_byte_suffix("1e400G").is_err());
    }

    // === Memory detection tests ===

    #[test]
    fn test_detect_available_memory_returns_nonzero() {
        let result = detect_available_memory();
        assert!(result.bytes > 0);
    }

    #[test]
    fn test_fallback_memory_is_8gb() {
        assert_eq!(FALLBACK_MEMORY_BYTES, 8 * 1024 * 1024 * 1024);
    }

    // === Batch memory estimation tests ===

    #[test]
    fn test_estimate_batch_memory_scales_linearly() {
        let profile = ReadMemoryProfile::new(150, false, 64, 50);

        let mem_1k = estimate_batch_memory(1000, &profile, 100).unwrap();
        let mem_2k = estimate_batch_memory(2000, &profile, 100).unwrap();

        // Should roughly double (within 10% tolerance for fixed overheads)
        let ratio = mem_2k as f64 / mem_1k as f64;
        assert!(
            ratio > 1.8 && ratio < 2.2,
            "Expected ~2x scaling, got {}",
            ratio
        );
    }

    #[test]
    fn test_estimate_batch_memory_increases_with_read_length() {
        let profile_short = ReadMemoryProfile::new(150, false, 64, 50);
        let profile_long = ReadMemoryProfile::new(10000, false, 64, 50);

        let mem_short = estimate_batch_memory(10000, &profile_short, 100).unwrap();
        let mem_long = estimate_batch_memory(10000, &profile_long, 100).unwrap();

        assert!(mem_long > mem_short);
    }

    #[test]
    fn test_estimate_batch_memory_overflow_protection() {
        let profile = ReadMemoryProfile::new(150, false, 64, 50);

        // Very large batch size that would overflow
        let result = estimate_batch_memory(usize::MAX, &profile, 100);
        assert!(result.is_none(), "Should return None on overflow");

        // Also test with large minimizers_per_query
        let large_profile = ReadMemoryProfile {
            avg_read_length: 150,
            avg_query_length: 150,
            minimizers_per_query: usize::MAX / 2,
        };
        let result = estimate_batch_memory(1000000, &large_profile, 100);
        assert!(result.is_none(), "Should return None on overflow");
    }

    // === Batch calculation tests ===

    #[test]
    fn test_calculate_batch_config_respects_limit() {
        let profile = ReadMemoryProfile::new(150, false, 64, 50);
        let config = MemoryConfig {
            max_memory: 1024 * 1024 * 1024, // 1GB
            num_threads: 4,
            index_memory: 100 * 1024 * 1024,     // 100MB
            shard_reservation: 50 * 1024 * 1024, // 50MB
            read_profile: profile,
            num_buckets: 100,
            input_format: InputFormat::Fastx { is_paired: false },
        };

        let batch_config = calculate_batch_config(&config);

        // Peak memory should not exceed max_memory
        assert!(
            batch_config.peak_memory <= config.max_memory,
            "Peak memory {} exceeds max {}",
            batch_config.peak_memory,
            config.max_memory
        );
    }

    #[test]
    fn test_calculate_batch_config_accounts_for_index() {
        let profile = ReadMemoryProfile::new(150, false, 64, 50);

        // Same total memory, different index sizes
        let config_small_index = MemoryConfig {
            max_memory: 1024 * 1024 * 1024,
            num_threads: 4,
            index_memory: 100 * 1024 * 1024,
            shard_reservation: 0,
            read_profile: profile.clone(),
            num_buckets: 100,
            input_format: InputFormat::Fastx { is_paired: false },
        };

        let config_large_index = MemoryConfig {
            max_memory: 1024 * 1024 * 1024,
            num_threads: 4,
            index_memory: 500 * 1024 * 1024,
            shard_reservation: 0,
            read_profile: profile,
            num_buckets: 100,
            input_format: InputFormat::Fastx { is_paired: false },
        };

        let batch_small = calculate_batch_config(&config_small_index);
        let batch_large = calculate_batch_config(&config_large_index);

        // Larger index should result in smaller batches
        assert!(
            batch_small.batch_size >= batch_large.batch_size,
            "Small index batch {} should be >= large index batch {}",
            batch_small.batch_size,
            batch_large.batch_size
        );
    }

    #[test]
    fn test_calculate_batch_config_minimum_values() {
        let profile = ReadMemoryProfile::new(150, false, 64, 50);

        // Very constrained memory
        let config = MemoryConfig {
            max_memory: 50 * 1024 * 1024, // Only 50MB
            num_threads: 4,
            index_memory: 10 * 1024 * 1024,
            shard_reservation: 5 * 1024 * 1024,
            read_profile: profile,
            num_buckets: 100,
            input_format: InputFormat::Fastx { is_paired: false },
        };

        let batch_config = calculate_batch_config(&config);

        // Should still get at least minimum batch size
        assert!(batch_config.batch_size >= MIN_BATCH_SIZE);
        assert!(batch_config.batch_count >= 1);
    }

    #[test]
    fn test_calculate_batch_config_uses_threads() {
        let profile = ReadMemoryProfile::new(150, false, 64, 50);

        // Plenty of memory
        let config = MemoryConfig {
            max_memory: 16 * 1024 * 1024 * 1024, // 16GB
            num_threads: 8,
            index_memory: 100 * 1024 * 1024,
            shard_reservation: 0,
            read_profile: profile,
            num_buckets: 100,
            input_format: InputFormat::Fastx { is_paired: false },
        };

        let batch_config = calculate_batch_config(&config);

        // With plenty of memory, should use multiple batches
        assert!(batch_config.batch_count >= 1);
    }

    // === Read memory profile tests ===

    #[test]
    fn test_read_memory_profile_paired() {
        let profile_single = ReadMemoryProfile::new(150, false, 64, 50);
        let profile_paired = ReadMemoryProfile::new(150, true, 64, 50);

        assert_eq!(profile_single.avg_query_length, 150);
        assert_eq!(profile_paired.avg_query_length, 300);
    }

    #[test]
    fn test_read_memory_profile_minimizers() {
        let profile = ReadMemoryProfile::new(1000, false, 64, 50);

        // Should estimate some minimizers for a 1000bp read
        assert!(profile.minimizers_per_query > 0);
    }

    // === Format bytes tests ===

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
        assert_eq!(format_bytes(1024 * 1024 * 1024 * 1024), "1.00 TB");
    }

    // === File sampling tests ===

    #[test]
    fn test_read_memory_profile_from_files() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create a temp FASTQ file with known read lengths
        let mut file = NamedTempFile::new().unwrap();
        // Write 3 reads of 100bp each
        for i in 0..3 {
            writeln!(file, "@read{}", i).unwrap();
            writeln!(file, "{}", "A".repeat(100)).unwrap();
            writeln!(file, "+").unwrap();
            writeln!(file, "{}", "I".repeat(100)).unwrap();
        }
        file.flush().unwrap();

        let profile = ReadMemoryProfile::from_files(
            file.path(),
            None,
            10,    // sample size
            64,    // k
            50,    // w
            false, // is_parquet
            None,  // trim_to
        );

        assert!(profile.is_some());
        let profile = profile.unwrap();
        assert_eq!(profile.avg_read_length, 100);
        assert_eq!(profile.avg_query_length, 100); // single-end
    }

    #[test]
    fn test_read_memory_profile_from_files_paired() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create R1 with 100bp reads
        let mut r1 = NamedTempFile::new().unwrap();
        for i in 0..3 {
            writeln!(r1, "@read{}", i).unwrap();
            writeln!(r1, "{}", "A".repeat(100)).unwrap();
            writeln!(r1, "+").unwrap();
            writeln!(r1, "{}", "I".repeat(100)).unwrap();
        }
        r1.flush().unwrap();

        // Create R2 with 150bp reads
        let mut r2 = NamedTempFile::new().unwrap();
        for i in 0..3 {
            writeln!(r2, "@read{}", i).unwrap();
            writeln!(r2, "{}", "T".repeat(150)).unwrap();
            writeln!(r2, "+").unwrap();
            writeln!(r2, "{}", "I".repeat(150)).unwrap();
        }
        r2.flush().unwrap();

        let profile =
            ReadMemoryProfile::from_files(r1.path(), Some(r2.path()), 10, 64, 50, false, None);

        assert!(profile.is_some());
        let profile = profile.unwrap();
        // avg_read_length should be (100*3 + 150*3) / 6 = 125
        assert_eq!(profile.avg_read_length, 125);
        // avg_query_length for paired = 125 * 2 = 250
        assert_eq!(profile.avg_query_length, 250);
    }

    #[test]
    fn test_read_memory_profile_from_files_nonexistent() {
        let profile = ReadMemoryProfile::from_files(
            std::path::Path::new("/nonexistent/file.fq"),
            None,
            10,
            64,
            50,
            false,
            None,
        );
        assert!(profile.is_none());
    }

    // === Parquet sampling tests ===

    #[test]
    fn test_read_memory_profile_from_parquet_single_end() {
        use arrow::array::{ArrayRef, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;
        use tempfile::NamedTempFile;

        // Create a temp Parquet file with known sequence lengths
        let file = NamedTempFile::new().unwrap();

        let schema = Arc::new(Schema::new(vec![
            Field::new("read_id", DataType::Utf8, false),
            Field::new("sequence1", DataType::Utf8, false),
        ]));

        // 3 reads of 100bp each
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["read0", "read1", "read2"])) as ArrayRef,
                Arc::new(StringArray::from(vec![
                    "A".repeat(100),
                    "A".repeat(100),
                    "A".repeat(100),
                ])) as ArrayRef,
            ],
        )
        .unwrap();

        let writer_file = std::fs::File::create(file.path()).unwrap();
        let mut writer = ArrowWriter::try_new(writer_file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let profile = ReadMemoryProfile::from_files(
            file.path(),
            None,
            10,   // sample size
            64,   // k
            50,   // w
            true, // is_parquet
            None, // trim_to
        );

        assert!(profile.is_some());
        let profile = profile.unwrap();
        assert_eq!(profile.avg_read_length, 100);
        assert_eq!(profile.avg_query_length, 100); // single-end
    }

    #[test]
    fn test_read_memory_profile_from_parquet_paired_end() {
        use arrow::array::{ArrayRef, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;
        use tempfile::NamedTempFile;

        // Create a temp Parquet file with paired sequences
        let file = NamedTempFile::new().unwrap();

        let schema = Arc::new(Schema::new(vec![
            Field::new("read_id", DataType::Utf8, false),
            Field::new("sequence1", DataType::Utf8, false),
            Field::new("sequence2", DataType::Utf8, true), // nullable for paired-end
        ]));

        // 3 reads with 100bp seq1 and 150bp seq2
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["read0", "read1", "read2"])) as ArrayRef,
                Arc::new(StringArray::from(vec![
                    "A".repeat(100),
                    "A".repeat(100),
                    "A".repeat(100),
                ])) as ArrayRef,
                Arc::new(StringArray::from(vec![
                    "T".repeat(150),
                    "T".repeat(150),
                    "T".repeat(150),
                ])) as ArrayRef,
            ],
        )
        .unwrap();

        let writer_file = std::fs::File::create(file.path()).unwrap();
        let mut writer = ArrowWriter::try_new(writer_file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let profile = ReadMemoryProfile::from_files(
            file.path(),
            None,
            10,   // sample size
            64,   // k
            50,   // w
            true, // is_parquet
            None, // trim_to
        );

        assert!(profile.is_some());
        let profile = profile.unwrap();
        // total_length = 3 * (100 + 150) = 750
        // count = 3
        // avg_query_length = 750 / 3 = 250
        // avg_read_length = 750 / 6 = 125
        assert_eq!(profile.avg_read_length, 125);
        assert_eq!(profile.avg_query_length, 250);
    }

    #[test]
    fn test_read_memory_profile_from_parquet_nonexistent() {
        let profile = ReadMemoryProfile::from_files(
            std::path::Path::new("/nonexistent/file.parquet"),
            None,
            10,
            64,
            50,
            true, // is_parquet
            None, // trim_to
        );
        assert!(profile.is_none());
    }

    #[test]
    fn test_sample_parquet_lengths_with_large_utf8() {
        use arrow::array::{ArrayRef, LargeStringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;
        use tempfile::NamedTempFile;

        // Create a temp Parquet file with LargeUtf8 columns
        let file = NamedTempFile::new().unwrap();

        let schema = Arc::new(Schema::new(vec![
            Field::new("read_id", DataType::LargeUtf8, false),
            Field::new("sequence1", DataType::LargeUtf8, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(LargeStringArray::from(vec!["read0", "read1"])) as ArrayRef,
                Arc::new(LargeStringArray::from(vec![
                    "A".repeat(200),
                    "A".repeat(200),
                ])) as ArrayRef,
            ],
        )
        .unwrap();

        let writer_file = std::fs::File::create(file.path()).unwrap();
        let mut writer = ArrowWriter::try_new(writer_file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let profile = ReadMemoryProfile::from_files(
            file.path(),
            None,
            10,   // sample size
            64,   // k
            50,   // w
            true, // is_parquet
            None, // trim_to
        );

        assert!(profile.is_some());
        let profile = profile.unwrap();
        assert_eq!(profile.avg_read_length, 200);
        assert_eq!(profile.avg_query_length, 200);
    }

    // === trim_to tests ===

    #[test]
    fn test_apply_trim_to_limit_single_end() {
        // Single-end: both read and query lengths should be capped
        let (read_len, query_len) = apply_trim_to_limit(1000, 1000, false, Some(100));
        assert_eq!(read_len, 100);
        assert_eq!(query_len, 100);

        // No trim: lengths unchanged
        let (read_len, query_len) = apply_trim_to_limit(1000, 1000, false, None);
        assert_eq!(read_len, 1000);
        assert_eq!(query_len, 1000);

        // Trim larger than read: no change
        let (read_len, query_len) = apply_trim_to_limit(100, 100, false, Some(1000));
        assert_eq!(read_len, 100);
        assert_eq!(query_len, 100);

        // trim_to=0 treated as no trimming
        let (read_len, query_len) = apply_trim_to_limit(1000, 1000, false, Some(0));
        assert_eq!(read_len, 1000);
        assert_eq!(query_len, 1000);
    }

    #[test]
    fn test_apply_trim_to_limit_paired_end() {
        // Paired-end: read length capped, query = 2 * capped_read
        let (read_len, query_len) = apply_trim_to_limit(1000, 2000, true, Some(100));
        assert_eq!(read_len, 100);
        assert_eq!(query_len, 200); // 2 * 100

        // No trim: lengths unchanged
        let (read_len, query_len) = apply_trim_to_limit(1000, 2000, true, None);
        assert_eq!(read_len, 1000);
        assert_eq!(query_len, 2000);

        // trim_to=0 treated as no trimming
        let (read_len, query_len) = apply_trim_to_limit(1000, 2000, true, Some(0));
        assert_eq!(read_len, 1000);
        assert_eq!(query_len, 2000);
    }

    #[test]
    fn test_read_memory_profile_from_fastx_with_trim_to() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create a temp FASTQ file with 1000bp reads
        let mut file = NamedTempFile::new().unwrap();
        for i in 0..3 {
            writeln!(file, "@read{}", i).unwrap();
            writeln!(file, "{}", "A".repeat(1000)).unwrap();
            writeln!(file, "+").unwrap();
            writeln!(file, "{}", "I".repeat(1000)).unwrap();
        }
        file.flush().unwrap();

        // Without trim_to
        let profile =
            ReadMemoryProfile::from_files(file.path(), None, 10, 64, 50, false, None).unwrap();
        assert_eq!(profile.avg_read_length, 1000);
        assert_eq!(profile.avg_query_length, 1000);

        // With trim_to=100
        let profile =
            ReadMemoryProfile::from_files(file.path(), None, 10, 64, 50, false, Some(100)).unwrap();
        assert_eq!(profile.avg_read_length, 100);
        assert_eq!(profile.avg_query_length, 100);
    }

    #[test]
    fn test_read_memory_profile_from_parquet_with_trim_to() {
        use arrow::array::{ArrayRef, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;
        use tempfile::NamedTempFile;

        // Create a temp Parquet file with 1000bp sequences
        let file = NamedTempFile::new().unwrap();

        let schema = Arc::new(Schema::new(vec![
            Field::new("read_id", DataType::Utf8, false),
            Field::new("sequence1", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["read0", "read1", "read2"])) as ArrayRef,
                Arc::new(StringArray::from(vec![
                    "A".repeat(1000),
                    "A".repeat(1000),
                    "A".repeat(1000),
                ])) as ArrayRef,
            ],
        )
        .unwrap();

        let writer_file = std::fs::File::create(file.path()).unwrap();
        let mut writer = ArrowWriter::try_new(writer_file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        // Without trim_to
        let profile =
            ReadMemoryProfile::from_files(file.path(), None, 10, 64, 50, true, None).unwrap();
        assert_eq!(profile.avg_read_length, 1000);
        assert_eq!(profile.avg_query_length, 1000);

        // With trim_to=100
        let profile =
            ReadMemoryProfile::from_files(file.path(), None, 10, 64, 50, true, Some(100)).unwrap();
        assert_eq!(profile.avg_read_length, 100);
        assert_eq!(profile.avg_query_length, 100);
    }

    #[test]
    fn test_read_memory_profile_paired_fastx_with_trim_to() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create R1 with 1000bp reads
        let mut r1 = NamedTempFile::new().unwrap();
        for i in 0..3 {
            writeln!(r1, "@read{}", i).unwrap();
            writeln!(r1, "{}", "A".repeat(1000)).unwrap();
            writeln!(r1, "+").unwrap();
            writeln!(r1, "{}", "I".repeat(1000)).unwrap();
        }
        r1.flush().unwrap();

        // Create R2 with 1000bp reads
        let mut r2 = NamedTempFile::new().unwrap();
        for i in 0..3 {
            writeln!(r2, "@read{}", i).unwrap();
            writeln!(r2, "{}", "T".repeat(1000)).unwrap();
            writeln!(r2, "+").unwrap();
            writeln!(r2, "{}", "I".repeat(1000)).unwrap();
        }
        r2.flush().unwrap();

        // Without trim_to
        let profile =
            ReadMemoryProfile::from_files(r1.path(), Some(r2.path()), 10, 64, 50, false, None)
                .unwrap();
        assert_eq!(profile.avg_read_length, 1000);
        assert_eq!(profile.avg_query_length, 2000);

        // With trim_to=100: each read capped at 100, query = 200
        let profile =
            ReadMemoryProfile::from_files(r1.path(), Some(r2.path()), 10, 64, 50, false, Some(100))
                .unwrap();
        assert_eq!(profile.avg_read_length, 100);
        assert_eq!(profile.avg_query_length, 200);
    }

    #[test]
    fn test_read_memory_profile_paired_parquet_with_trim_to() {
        use arrow::array::{ArrayRef, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;
        use tempfile::NamedTempFile;

        // Create a temp Parquet file with paired 1000bp sequences
        let file = NamedTempFile::new().unwrap();

        let schema = Arc::new(Schema::new(vec![
            Field::new("read_id", DataType::Utf8, false),
            Field::new("sequence1", DataType::Utf8, false),
            Field::new("sequence2", DataType::Utf8, true),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["read0", "read1", "read2"])) as ArrayRef,
                Arc::new(StringArray::from(vec![
                    "A".repeat(1000),
                    "A".repeat(1000),
                    "A".repeat(1000),
                ])) as ArrayRef,
                Arc::new(StringArray::from(vec![
                    "T".repeat(1000),
                    "T".repeat(1000),
                    "T".repeat(1000),
                ])) as ArrayRef,
            ],
        )
        .unwrap();

        let writer_file = std::fs::File::create(file.path()).unwrap();
        let mut writer = ArrowWriter::try_new(writer_file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        // Without trim_to
        let profile =
            ReadMemoryProfile::from_files(file.path(), None, 10, 64, 50, true, None).unwrap();
        // avg_query_length = (1000+1000)*3 / 3 = 2000
        // avg_read_length = 6000 / 6 = 1000
        assert_eq!(profile.avg_read_length, 1000);
        assert_eq!(profile.avg_query_length, 2000);

        // With trim_to=100: each read capped at 100, query = 200
        let profile =
            ReadMemoryProfile::from_files(file.path(), None, 10, 64, 50, true, Some(100)).unwrap();
        assert_eq!(profile.avg_read_length, 100);
        assert_eq!(profile.avg_query_length, 200);
    }

    // ==========================================================================
    // Phase 1 TDD Tests: I/O Buffer Memory Accounting
    // These tests verify that memory estimation accounts for prefetch buffers
    // and format-specific overhead (Arrow vs OwnedRecord).
    // ==========================================================================

    #[test]
    fn test_batch_config_accounts_for_prefetch_buffer() {
        let profile = ReadMemoryProfile::new(150, false, 64, 50);
        let config = MemoryConfig {
            max_memory: 1024 * 1024 * 1024, // 1GB
            num_threads: 4,
            index_memory: 100 * 1024 * 1024,
            shard_reservation: 0,
            read_profile: profile.clone(),
            num_buckets: 100,
            input_format: InputFormat::Parquet { is_paired: false },
        };

        let batch_config = calculate_batch_config(&config);

        // Peak memory should include prefetch buffer overhead
        // Prefetch overhead = slots × batch_size × bytes_per_row
        let prefetch_overhead = config.prefetch_buffer_slots()
            * batch_config.batch_size
            * config.buffer_bytes_per_row();

        // Verify prefetch overhead is factored into peak memory
        assert!(
            batch_config.peak_memory >= prefetch_overhead,
            "Peak memory {} should include prefetch overhead {}",
            batch_config.peak_memory,
            prefetch_overhead
        );
    }

    #[test]
    fn test_fastx_vs_parquet_uses_different_prefetch_slots() {
        let profile = ReadMemoryProfile::new(150, false, 64, 50);

        // FASTX uses 2 slots
        let config_fastx = MemoryConfig {
            max_memory: 1024 * 1024 * 1024,
            num_threads: 4,
            index_memory: 100 * 1024 * 1024,
            shard_reservation: 0,
            read_profile: profile.clone(),
            num_buckets: 100,
            input_format: InputFormat::Fastx { is_paired: false },
        };

        // Parquet uses 4 slots
        let config_parquet = MemoryConfig {
            max_memory: 1024 * 1024 * 1024,
            num_threads: 4,
            index_memory: 100 * 1024 * 1024,
            shard_reservation: 0,
            read_profile: profile,
            num_buckets: 100,
            input_format: InputFormat::Parquet { is_paired: false },
        };

        assert_eq!(
            config_fastx.prefetch_buffer_slots(),
            FASTX_PREFETCH_BUFFER_SLOTS
        );
        assert_eq!(
            config_parquet.prefetch_buffer_slots(),
            PARQUET_PREFETCH_BUFFER_SLOTS
        );
        assert_eq!(config_fastx.prefetch_buffer_slots(), 2);
        assert_eq!(config_parquet.prefetch_buffer_slots(), 4);

        // FASTX should allow larger batch sizes due to smaller prefetch buffer
        let batch_fastx = calculate_batch_config(&config_fastx);
        let batch_parquet = calculate_batch_config(&config_parquet);

        assert!(
            batch_fastx.batch_size >= batch_parquet.batch_size,
            "FASTX batch {} should be >= Parquet batch {} (fewer prefetch slots)",
            batch_fastx.batch_size,
            batch_parquet.batch_size
        );
    }

    #[test]
    fn test_owned_record_vs_arrow_bytes_estimation() {
        let profile = ReadMemoryProfile::new(150, false, 64, 50);

        // OwnedRecord estimation for FASTX
        let owned_bytes = profile.estimate_owned_record_bytes(false);
        // Arrow estimation for Parquet
        let arrow_bytes = profile.estimate_arrow_bytes_per_row(false);

        // Both should be reasonable for 150bp reads
        assert!(
            owned_bytes > 100 && owned_bytes < 500,
            "OwnedRecord bytes {} should be reasonable",
            owned_bytes
        );
        assert!(
            arrow_bytes > 100 && arrow_bytes < 500,
            "Arrow bytes {} should be reasonable",
            arrow_bytes
        );

        // OwnedRecord for paired-end should be larger
        let owned_paired = profile.estimate_owned_record_bytes(true);
        assert!(
            owned_paired > owned_bytes,
            "Paired OwnedRecord {} should be > single {}",
            owned_paired,
            owned_bytes
        );
    }

    #[test]
    fn test_read_length_affects_buffer_bytes() {
        let profile_short = ReadMemoryProfile::new(150, false, 64, 50);
        let profile_long = ReadMemoryProfile::new(10000, false, 64, 50);

        // Longer reads should have larger buffer bytes for both formats
        let short_owned = profile_short.estimate_owned_record_bytes(false);
        let long_owned = profile_long.estimate_owned_record_bytes(false);
        assert!(long_owned > short_owned);

        let short_arrow = profile_short.estimate_arrow_bytes_per_row(false);
        let long_arrow = profile_long.estimate_arrow_bytes_per_row(false);
        assert!(long_arrow > short_arrow);
    }

    #[test]
    fn test_total_memory_with_io_buffers_within_budget() {
        let profile = ReadMemoryProfile::new(5000, true, 64, 50); // paired long reads
        let config = MemoryConfig {
            max_memory: 8 * 1024 * 1024 * 1024, // 8GB
            num_threads: 8,
            index_memory: 500 * 1024 * 1024,
            shard_reservation: 100 * 1024 * 1024,
            read_profile: profile,
            num_buckets: 1000,
            input_format: InputFormat::Parquet { is_paired: true },
        };

        let batch_config = calculate_batch_config(&config);

        // Total peak memory must not exceed max_memory
        assert!(
            batch_config.peak_memory <= config.max_memory,
            "Peak {} exceeds max {}",
            batch_config.peak_memory,
            config.max_memory
        );
    }

    #[test]
    fn test_estimate_arrow_bytes_per_row() {
        // Test the helper method that estimates Arrow buffer overhead
        let profile_short = ReadMemoryProfile::new(150, false, 64, 50);
        let profile_long = ReadMemoryProfile::new(10000, false, 64, 50);

        let bytes_short = profile_short.estimate_arrow_bytes_per_row(false);
        let bytes_long = profile_long.estimate_arrow_bytes_per_row(false);

        // Should include fixed overhead (offsets, validity, read_id)
        // plus variable sequence data
        assert!(
            bytes_short > 40, // at least offset + validity + read_id overhead
            "Short read arrow bytes {} should be > 40 (fixed overhead)",
            bytes_short
        );

        // Longer reads should have more Arrow buffer overhead
        assert!(
            bytes_long > bytes_short,
            "Long read arrow bytes {} should be > short read bytes {}",
            bytes_long,
            bytes_short
        );

        // The difference should roughly correspond to sequence length difference
        let expected_diff = 10000 - 150;
        let actual_diff = bytes_long - bytes_short;
        assert!(
            actual_diff >= expected_diff - 100 && actual_diff <= expected_diff + 100,
            "Arrow bytes difference {} should be close to sequence length difference {}",
            actual_diff,
            expected_diff
        );
    }

    #[test]
    fn test_binary_search_validates_result() {
        // Test that binary search returns valid results even in edge cases
        let profile = ReadMemoryProfile::new(150, false, 64, 50);
        let config = MemoryConfig {
            max_memory: 500 * 1024 * 1024, // 500MB - moderately constrained
            num_threads: 4,
            index_memory: 100 * 1024 * 1024,
            shard_reservation: 50 * 1024 * 1024,
            read_profile: profile,
            num_buckets: 100,
            input_format: InputFormat::Fastx { is_paired: false },
        };

        let batch_config = calculate_batch_config(&config);

        // Verify the result is valid
        let total =
            estimate_total_batch_memory(batch_config.batch_size, batch_config.batch_count, &config);

        // Calculate available memory (same as in calculate_batch_config)
        let safety_margin = (config.max_memory as f64 * 0.10).round() as usize;
        let safety_margin = safety_margin.max(256 * 1024 * 1024);
        let base_reserved = config.index_memory + config.shard_reservation + safety_margin;
        let available = config.max_memory.saturating_sub(base_reserved);

        assert!(
            total.is_some_and(|t| t <= available),
            "Binary search result should fit in available memory"
        );
    }

    #[test]
    fn test_memory_config_validation() {
        let profile = ReadMemoryProfile::new(150, false, 64, 50);

        // Valid config should succeed
        let valid = MemoryConfig::new(
            1024 * 1024 * 1024,
            4,
            100 * 1024 * 1024,
            0,
            profile.clone(),
            100,
            InputFormat::Fastx { is_paired: false },
        );
        assert!(valid.is_ok());

        // Invalid: max_memory = 0
        let invalid = MemoryConfig::new(
            0,
            4,
            100 * 1024 * 1024,
            0,
            profile.clone(),
            100,
            InputFormat::Fastx { is_paired: false },
        );
        assert!(invalid.is_err());

        // Invalid: num_threads = 0
        let invalid = MemoryConfig::new(
            1024 * 1024 * 1024,
            0,
            100 * 1024 * 1024,
            0,
            profile.clone(),
            100,
            InputFormat::Fastx { is_paired: false },
        );
        assert!(invalid.is_err());

        // Invalid: num_buckets = 0
        let invalid = MemoryConfig::new(
            1024 * 1024 * 1024,
            4,
            100 * 1024 * 1024,
            0,
            profile,
            0,
            InputFormat::Fastx { is_paired: false },
        );
        assert!(invalid.is_err());
    }
}
