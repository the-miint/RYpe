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

/// Estimated buckets per read for HashMap pre-allocation.
pub(crate) const ESTIMATED_BUCKETS_PER_READ: usize = 4;

/// Maximum bucket count for using dense (array-based) accumulators.
/// Indices with more buckets fall back to sparse (HashMap-based) accumulators.
/// At 256 buckets, a dense accumulator uses 2 KB per read (256 × 8 bytes).
pub(crate) const DENSE_ACCUMULATOR_MAX_BUCKETS: usize = 256;

/// Size ratio threshold for switching from merge-join to galloping search.
/// When one index is more than GALLOP_THRESHOLD times larger, galloping is used.
pub(crate) const GALLOP_THRESHOLD: usize = 16;

/// Maximum bucket count for using COO merge-join in the shard loop.
/// Indices with more buckets use CSR merge-join, which iterates only unique
/// minimizers and does compact bucket-slice lookups — much faster when the
/// reference COO would be N× larger than the unique minimizer array.
///
/// Set to 10 based on the Phase 2 regression analysis: at 160 buckets, COO
/// merge-join was 8× slower than CSR (COO pairs are ~160× larger than unique
/// minimizers). At 10 buckets, COO overhead is ≤10× — comparable to the
/// CSR conversion cost it avoids. Conservative to avoid regressions; could
/// potentially be raised to ~20 with further benchmarking.
pub(crate) const COO_MERGE_JOIN_MAX_BUCKETS: usize = 10;

/// Minimum reference shard size (COO pairs) for parallel within-shard merge-join.
/// Shards smaller than this use single-threaded merge-join to avoid parallelism
/// overhead (thread spawning, chunk splitting, sparse hit merging).
///
/// At 10K pairs with 8 threads, each chunk gets ~1.25K pairs — enough work to
/// amortize rayon overhead (~10μs per task) against merge-join cost (~1μs per pair).
pub(crate) const MIN_PARALLEL_SHARD_SIZE: usize = 10_000;

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
