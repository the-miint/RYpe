//! Constants used throughout the rype library for safety limits and performance tuning.

// Maximum sizes for safety checks when loading files
pub(crate) const MAX_BUCKET_SIZE: usize = 1_000_000_000; // 1B minimizers (~8GB)
pub(crate) const MAX_STRING_LENGTH: usize = 10_000; // 10KB for names/sources
pub(crate) const MAX_NUM_BUCKETS: u32 = 100_000; // Reasonable upper limit

// Maximum sizes for inverted index
pub(crate) const MAX_INVERTED_MINIMIZERS: usize = usize::MAX; // Allow system memory to be the limit
pub(crate) const MAX_INVERTED_BUCKET_IDS: usize = 4_000_000_000; // 4B total bucket ID entries

// Default capacities for workspace (document the reasoning)
pub(crate) const DEFAULT_DEQUE_CAPACITY: usize = 128; // Typical window size range
pub(crate) const ESTIMATED_MINIMIZERS_PER_SEQUENCE: usize = 32; // Conservative estimate
