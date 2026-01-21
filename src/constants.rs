//! Constants used throughout the rype library for safety limits, performance tuning,
//! and binary format definitions.
//!
//! Centralizing these constants ensures consistency across the codebase and makes
//! it easy to adjust values when needed.

// ============================================================================
// I/O Buffer Sizes
// ============================================================================

/// Buffer size for writing binary index files (8MB).
pub(crate) const WRITE_BUF_SIZE: usize = 8 * 1024 * 1024;

/// Buffer size for reading binary index files (8MB).
pub(crate) const READ_BUF_SIZE: usize = 8 * 1024 * 1024;

// ============================================================================
// Binary Format Magic Bytes
// ============================================================================

/// Magic bytes for single-file main index files (.ryidx).
pub(crate) const SINGLE_FILE_INDEX_MAGIC: &[u8; 4] = b"RYP5";

/// Magic bytes for inverted index shard files (.ryxdi.shard.*).
pub(crate) const SHARD_MAGIC: &[u8; 4] = b"RYXS";

/// Magic bytes for inverted index manifest files (.ryxdi.manifest).
pub(crate) const MANIFEST_MAGIC: &[u8; 4] = b"RYXM";

/// Magic bytes for sharded main index manifest files (.ryidx.manifest).
pub(crate) const MAIN_MANIFEST_MAGIC: &[u8; 4] = b"RYPM";

/// Magic bytes for sharded main index shard files (.ryidx.shard.*).
pub(crate) const MAIN_SHARD_MAGIC: &[u8; 4] = b"RYPS";

// ============================================================================
// Binary Format Versions
// ============================================================================

/// Current version for single-file main index files.
pub(crate) const SINGLE_FILE_INDEX_VERSION: u32 = 5;

/// Current version for inverted index shard files.
pub(crate) const SHARD_VERSION: u32 = 1;

/// Current version for inverted index manifest files.
/// Supports versions 3-5 for backwards compatibility.
pub(crate) const MANIFEST_VERSION: u32 = 5;

/// Current version for sharded main index manifest files.
pub(crate) const MAIN_MANIFEST_VERSION: u32 = 2;

/// Current version for sharded main index shard files.
pub(crate) const MAIN_SHARD_VERSION: u32 = 2;

// ============================================================================
// Safety Limits for Loading Files
// ============================================================================

/// Maximum minimizers per bucket (~8GB at 8 bytes each).
pub(crate) const MAX_BUCKET_SIZE: usize = 1_000_000_000;

/// Maximum length for name/source strings (10KB).
pub(crate) const MAX_STRING_LENGTH: usize = 10_000;

/// Maximum number of buckets per index.
pub(crate) const MAX_NUM_BUCKETS: u32 = 100_000;

/// Maximum minimizers in inverted index (1 trillion).
pub(crate) const MAX_INVERTED_MINIMIZERS: usize = 1_000_000_000_000;

/// Maximum total bucket ID entries in inverted index (4 billion).
pub(crate) const MAX_INVERTED_BUCKET_IDS: usize = 4_000_000_000;

// ============================================================================
// Sharded Index Limits
// ============================================================================

/// Maximum number of shards for inverted index.
pub(crate) const MAX_SHARDS: u32 = 10_000;

/// Maximum number of shards for main index.
pub(crate) const MAX_MAIN_SHARDS: u32 = 10_000;

/// Maximum total bytes for string table in sharded main index (100MB).
pub(crate) const MAX_STRING_TABLE_BYTES: usize = 100_000_000;

/// Maximum entries in string table for sharded main index (10 million).
pub(crate) const MAX_STRING_TABLE_ENTRIES: u32 = 10_000_000;

/// Maximum sources per bucket in sharded main index (100 million).
pub(crate) const MAX_SOURCES_PER_BUCKET: usize = 100_000_000;

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
/// Re-exported publicly via `Index::BUCKET_SOURCE_DELIM`.
pub(crate) const BUCKET_SOURCE_DELIM: &str = "::";

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
// Memory Estimation
// ============================================================================

/// Bytes per minimizer for compressed output estimation (delta+varint+zstd).
pub(crate) const BYTES_PER_MINIMIZER_COMPRESSED: usize = 4;

/// Bytes per minimizer for in-memory estimation (u64 = 8 bytes).
pub(crate) const BYTES_PER_MINIMIZER_MEMORY: usize = 8;

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
    fn test_max_bucket_size_no_overflow() {
        // 8 bytes per minimizer, should not overflow usize
        assert!(
            MAX_BUCKET_SIZE <= usize::MAX / BYTES_PER_MINIMIZER_MEMORY,
            "MAX_BUCKET_SIZE would overflow when multiplied by 8"
        );
    }

    #[test]
    fn test_gallop_threshold_sane() {
        assert!(
            GALLOP_THRESHOLD > 1,
            "GALLOP_THRESHOLD must be > 1 for the algorithm to work"
        );
    }

    #[test]
    fn test_buffer_sizes_are_power_of_two() {
        assert!(
            WRITE_BUF_SIZE.is_power_of_two(),
            "WRITE_BUF_SIZE should be power of 2"
        );
        assert!(
            READ_BUF_SIZE.is_power_of_two(),
            "READ_BUF_SIZE should be power of 2"
        );
    }

    #[test]
    fn test_magic_bytes_are_4_bytes() {
        assert_eq!(SINGLE_FILE_INDEX_MAGIC.len(), 4);
        assert_eq!(SHARD_MAGIC.len(), 4);
        assert_eq!(MANIFEST_MAGIC.len(), 4);
        assert_eq!(MAIN_MANIFEST_MAGIC.len(), 4);
        assert_eq!(MAIN_SHARD_MAGIC.len(), 4);
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
