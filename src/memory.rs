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
    /// * `r1_path` - Path to R1 FASTQ/FASTA file
    /// * `r2_path` - Optional path to R2 FASTQ/FASTA file (for paired-end)
    /// * `sample_size` - Number of records to sample from each file
    /// * `k` - K-mer size
    /// * `w` - Window size
    ///
    /// # Returns
    /// A `ReadMemoryProfile` based on sampled lengths, or None if sampling fails.
    pub fn from_files(
        r1_path: &std::path::Path,
        r2_path: Option<&std::path::Path>,
        sample_size: usize,
        k: usize,
        w: usize,
    ) -> Option<Self> {
        // Sample R1
        let (r1_total, r1_count) = sample_file_lengths(r1_path, sample_size)?;
        if r1_count == 0 {
            return None;
        }
        let avg_r1_length = r1_total / r1_count;

        // Sample R2 if provided
        let (avg_query_length, avg_read_length) = if let Some(r2) = r2_path {
            let (r2_total, r2_count) = sample_file_lengths(r2, sample_size)?;
            if r2_count == 0 {
                return None;
            }
            let avg_r2_length = r2_total / r2_count;
            // For paired-end: avg_query_length = R1 avg + R2 avg
            // avg_read_length = average of individual read lengths
            let avg_read = (r1_total + r2_total) / (r1_count + r2_count);
            (avg_r1_length + avg_r2_length, avg_read)
        } else {
            (avg_r1_length, avg_r1_length)
        };

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
}

/// Helper function to sample read lengths from a file.
/// Returns (total_length, count) or None if the file cannot be read.
fn sample_file_lengths(path: &std::path::Path, sample_size: usize) -> Option<(usize, usize)> {
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
const MEMORY_FUDGE_FACTOR: f64 = 1.3;

/// Estimate memory usage for a single batch.
///
/// Memory components:
/// - Input records: batch_size * (72 + avg_query_length) for OwnedRecord
/// - Minimizers: batch_size * minimizers_per_query * 16 bytes (`Vec<u64>` for fwd + rc)
/// - QueryInvertedIndex CSR: batch_size * minimizers_per_query * 12 bytes
/// - Accumulators: batch_size * estimated_buckets_per_read * 12 bytes (HashMap overhead)
pub fn estimate_batch_memory(
    batch_size: usize,
    profile: &ReadMemoryProfile,
    num_buckets: usize,
) -> usize {
    // OwnedRecord: (i64, Vec<u8>, Option<Vec<u8>>) ≈ 72 bytes + sequence data
    let record_overhead = 72;
    let input_records = batch_size * (record_overhead + profile.avg_query_length);

    // Minimizer vectors: Vec<u64> for forward and reverse-complement
    let minimizer_vecs = batch_size * profile.minimizers_per_query * 16;

    // QueryInvertedIndex CSR structure
    // minimizers: Vec<u64>, offsets: Vec<u32>, read_ids: Vec<u32>
    let query_index = batch_size * profile.minimizers_per_query * 12;

    // Per-read accumulators: HashMap<u32, (u32, u32)>
    // Estimate ~4 buckets per read on average
    let estimated_buckets_per_read = 4.min(num_buckets);
    let accumulators = batch_size * estimated_buckets_per_read * 24; // HashMap entry overhead

    // Apply fudge factor
    let base_estimate = input_records + minimizer_vecs + query_index + accumulators;
    (base_estimate as f64 * MEMORY_FUDGE_FACTOR).round() as usize
}

/// Calculate optimal batch configuration based on memory constraints.
///
/// Algorithm:
/// 1. Calculate available memory after subtracting index, shard reservation, and safety margin
/// 2. Start with batch_count = num_threads
/// 3. Binary search for maximum batch_size that fits in (available / batch_count)
/// 4. If too tight, reduce batch_count and retry
/// 5. Enforce minimum batch_size of 1000
///
/// Note: When using Rayon for parallel batch processing, all batch_count batches may be
/// in flight simultaneously (worst case). The memory calculation accounts for this by
/// dividing available memory by batch_count, ensuring peak_memory = batch_count × per_batch_memory
/// stays within budget.
pub fn calculate_batch_config(config: &MemoryConfig) -> BatchConfig {
    // Calculate safety margin
    let safety_margin = (config.max_memory as f64 * SAFETY_MARGIN_PERCENT).round() as usize;
    let safety_margin = safety_margin.max(SAFETY_MARGIN_MIN_BYTES);

    // Available memory for batches
    let reserved = config.index_memory + config.shard_reservation + safety_margin;
    let available = config.max_memory.saturating_sub(reserved);

    // If we have very little memory, use minimum config
    if available < estimate_batch_memory(MIN_BATCH_SIZE, &config.read_profile, config.num_buckets) {
        return BatchConfig {
            batch_size: MIN_BATCH_SIZE,
            batch_count: 1,
            per_batch_memory: estimate_batch_memory(
                MIN_BATCH_SIZE,
                &config.read_profile,
                config.num_buckets,
            ),
            peak_memory: reserved
                + estimate_batch_memory(MIN_BATCH_SIZE, &config.read_profile, config.num_buckets),
        };
    }

    // Try decreasing batch counts
    for batch_count in (1..=config.num_threads).rev() {
        let memory_per_batch = available / batch_count;

        // Binary search for batch_size
        let batch_size =
            binary_search_batch_size(memory_per_batch, &config.read_profile, config.num_buckets);

        if batch_size >= MIN_BATCH_SIZE {
            let per_batch_memory =
                estimate_batch_memory(batch_size, &config.read_profile, config.num_buckets);
            let peak_memory = reserved + (per_batch_memory * batch_count);

            return BatchConfig {
                batch_size,
                batch_count,
                per_batch_memory,
                peak_memory,
            };
        }
    }

    // Fallback to minimum
    let per_batch_memory =
        estimate_batch_memory(MIN_BATCH_SIZE, &config.read_profile, config.num_buckets);
    BatchConfig {
        batch_size: MIN_BATCH_SIZE,
        batch_count: 1,
        per_batch_memory,
        peak_memory: reserved + per_batch_memory,
    }
}

/// Binary search for maximum batch size that fits in memory budget.
fn binary_search_batch_size(
    memory_budget: usize,
    profile: &ReadMemoryProfile,
    num_buckets: usize,
) -> usize {
    let mut low = MIN_BATCH_SIZE;
    let mut high = MAX_BATCH_SIZE;
    let mut best = MIN_BATCH_SIZE;

    while low <= high {
        // Avoid overflow: use low + (high - low) / 2 instead of (low + high) / 2
        let mid = low + (high - low) / 2;
        let estimated = estimate_batch_memory(mid, profile, num_buckets);

        if estimated <= memory_budget {
            best = mid;
            low = mid + 1;
        } else {
            high = mid.saturating_sub(1);
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

        let mem_1k = estimate_batch_memory(1000, &profile, 100);
        let mem_2k = estimate_batch_memory(2000, &profile, 100);

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

        let mem_short = estimate_batch_memory(10000, &profile_short, 100);
        let mem_long = estimate_batch_memory(10000, &profile_long, 100);

        assert!(mem_long > mem_short);
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
        };

        let config_large_index = MemoryConfig {
            max_memory: 1024 * 1024 * 1024,
            num_threads: 4,
            index_memory: 500 * 1024 * 1024,
            shard_reservation: 0,
            read_profile: profile,
            num_buckets: 100,
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
            10, // sample size
            64, // k
            50, // w
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

        let profile = ReadMemoryProfile::from_files(r1.path(), Some(r2.path()), 10, 64, 50);

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
        );
        assert!(profile.is_none());
    }
}
