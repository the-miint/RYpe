//! Constants used throughout the rype library for safety limits and performance tuning.
//!
//! Centralizing these constants ensures consistency across the codebase and makes
//! it easy to adjust values when needed.

// ============================================================================
// Safety Limits for Parquet Inverted Index
// ============================================================================

/// Maximum minimizers in inverted index (1 trillion).
pub(crate) const MAX_INVERTED_MINIMIZERS: usize = 1_000_000_000_000;

/// Maximum total bucket ID entries in inverted index (4 billion).
pub(crate) const MAX_INVERTED_BUCKET_IDS: usize = 4_000_000_000;

// ============================================================================
// Batch Processing
// ============================================================================

/// Default batch size for Parquet writes (rows per batch).
pub(crate) const PARQUET_BATCH_SIZE: usize = 100_000;

/// Default row group size for Parquet files.
pub(crate) const DEFAULT_ROW_GROUP_SIZE: usize = 100_000;

// ============================================================================
// Workspace Defaults
// ============================================================================

/// Default capacity for minimizer deques (typical window size range).
pub(crate) const DEFAULT_DEQUE_CAPACITY: usize = 128;

/// Size of the fixed ring buffer for minimizer extraction (covers w up to ~200).
pub(crate) const RING_BUFFER_SIZE: usize = 256;

/// Estimated minimizers per sequence (conservative).
pub(crate) const ESTIMATED_MINIMIZERS_PER_SEQUENCE: usize = 32;

// ============================================================================
// Classification Tuning
// ============================================================================

/// Threshold for switching to HashSet-based lookup in filtered loading.
/// Above this many query minimizers, use HashSet instead of binary search.
pub(crate) const QUERY_HASHSET_THRESHOLD: usize = 1000;

/// Threshold for using HashSet vs binary search for bounded query filtering
/// during row group loading. When the bounded query slice exceeds this size,
/// build a local HashSet for O(1) lookups instead of O(log n) binary search.
pub(crate) const BOUNDED_QUERY_HASHSET_THRESHOLD: usize = 100;

/// Threshold for switching to galloping search in merge-join.
/// When one index is more than GALLOP_THRESHOLD times larger, use galloping.
pub(crate) const GALLOP_THRESHOLD: usize = 16;

/// Estimated buckets per read for HashMap pre-allocation.
pub(crate) const ESTIMATED_BUCKETS_PER_READ: usize = 4;

// ============================================================================
// Delimiters
// ============================================================================

/// Delimiter between filename and sequence name in bucket sources.
/// Format: `path/to/file.fa::sequence_name`
pub const BUCKET_SOURCE_DELIM: &str = "::";

// ============================================================================
// C API Limits
// ============================================================================

/// Maximum sequence length for C API (2GB on 64-bit systems).
pub(crate) const MAX_SEQUENCE_LENGTH: usize = 2_000_000_000;

// ============================================================================
// QueryInvertedIndex Bit-Packing
// ============================================================================

/// Mask to extract read index from packed value (lower 31 bits).
pub(crate) const READ_INDEX_MASK: u32 = 0x7FFFFFFF;

/// Flag bit indicating reverse-complement strand (bit 31).
pub(crate) const RC_FLAG_BIT: u32 = 0x80000000;

/// Maximum number of reads supported by bit-packing (2^31 - 1).
pub(crate) const MAX_READS: usize = 0x7FFFFFFF;

// ============================================================================
// Parallel Processing
// ============================================================================

/// Minimum entries per parallel partition before parallel sharding is enabled.
///
/// When total_entries > MIN_ENTRIES_PER_PARALLEL_PARTITION * num_cpus and
/// multiple CPUs are available, parallel range-partitioned sharding is used.
/// This ensures each parallel worker has enough data to amortize the overhead
/// of spawning threads and coordinating output file renaming.
pub(crate) const MIN_ENTRIES_PER_PARALLEL_PARTITION: usize = 1_000_000;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gallop_threshold_sane() {
        assert!(
            GALLOP_THRESHOLD > 1,
            "GALLOP_THRESHOLD must be > 1 for the algorithm to work"
        );
    }

    #[test]
    fn test_bit_packing_constants_consistent() {
        // RC_FLAG_BIT should be the high bit
        assert_eq!(RC_FLAG_BIT, 0x80000000);
        // READ_INDEX_MASK should be all lower 31 bits
        assert_eq!(READ_INDEX_MASK, 0x7FFFFFFF);
        // They should be complementary
        assert_eq!(RC_FLAG_BIT | READ_INDEX_MASK, u32::MAX);
        // MAX_READS should match READ_INDEX_MASK
        assert_eq!(MAX_READS, READ_INDEX_MASK as usize);
    }

    #[test]
    fn test_parquet_batch_sizes_reasonable() {
        assert!(PARQUET_BATCH_SIZE > 0);
        assert!(DEFAULT_ROW_GROUP_SIZE > 0);
        // Batch size shouldn't be larger than row group size for efficiency
        assert!(PARQUET_BATCH_SIZE <= DEFAULT_ROW_GROUP_SIZE * 10);
    }
}
