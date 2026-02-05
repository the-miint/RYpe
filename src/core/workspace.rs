//! Reusable workspace for minimizer extraction.
//!
//! The workspace pattern avoids repeated allocations in hot loops by providing
//! pre-allocated buffers that can be reused across multiple extraction calls.

use super::ring_buffer::RingBuffer;
use crate::constants::{
    DEFAULT_DEQUE_CAPACITY, ESTIMATED_MINIMIZERS_PER_SEQUENCE, RING_BUFFER_SIZE,
};

/// Workspace for minimizer extraction algorithms.
///
/// Contains pre-allocated deques for the sliding window algorithm and
/// a buffer for collecting output minimizers. Reusing a workspace across
/// multiple extraction calls avoids repeated heap allocations.
///
/// # Usage
/// ```
/// use rype::MinimizerWorkspace;
///
/// let mut ws = MinimizerWorkspace::new();
/// // Pass &mut ws to extraction functions
/// // Results will be in ws.buffer after extraction
/// ```
pub struct MinimizerWorkspace {
    /// Monotonic deque for forward strand k-mer hashes (array-backed for cache locality)
    pub(crate) q_fwd: RingBuffer<(usize, u64), RING_BUFFER_SIZE>,
    /// Monotonic deque for reverse complement k-mer hashes (array-backed for cache locality)
    pub(crate) q_rc: RingBuffer<(usize, u64), RING_BUFFER_SIZE>,
    /// Output buffer for extracted minimizers
    pub buffer: Vec<u64>,
    /// Estimated minimizers per sequence for pre-allocation.
    /// Use `with_estimate()` to set based on actual read length profiles.
    pub estimated_minimizers: usize,
}

impl MinimizerWorkspace {
    /// Create a new workspace with default capacity.
    ///
    /// The ring buffers are stack-allocated with fixed size for cache locality.
    /// Uses `ESTIMATED_MINIMIZERS_PER_SEQUENCE` (32) for output buffer pre-allocation.
    pub fn new() -> Self {
        Self {
            q_fwd: RingBuffer::new(),
            q_rc: RingBuffer::new(),
            buffer: Vec::with_capacity(DEFAULT_DEQUE_CAPACITY),
            estimated_minimizers: ESTIMATED_MINIMIZERS_PER_SEQUENCE,
        }
    }

    /// Create a new workspace with a custom minimizer estimate.
    ///
    /// Use this when you have profiled read lengths (e.g., via `ReadMemoryProfile::from_files()`)
    /// to reduce reallocations for non-standard read lengths.
    ///
    /// # Arguments
    /// * `estimated_minimizers` - Expected minimizers per sequence (from `ReadMemoryProfile::minimizers_per_query`)
    pub fn with_estimate(estimated_minimizers: usize) -> Self {
        Self {
            q_fwd: RingBuffer::new(),
            q_rc: RingBuffer::new(),
            buffer: Vec::with_capacity(DEFAULT_DEQUE_CAPACITY),
            estimated_minimizers: estimated_minimizers.max(ESTIMATED_MINIMIZERS_PER_SEQUENCE),
        }
    }

    /// Estimate the number of minimizers for a sequence of given length.
    ///
    /// Uses the formula: `((seq_len - k) / w + 1) * 2` for both strands.
    /// Falls back to `ESTIMATED_MINIMIZERS_PER_SEQUENCE` for short sequences.
    ///
    /// # Arguments
    /// * `seq_len` - Length of the sequence in bases
    /// * `k` - K-mer size
    /// * `w` - Window size for minimizer selection
    ///
    /// # Returns
    /// Estimated number of minimizers, at least `ESTIMATED_MINIMIZERS_PER_SEQUENCE`.
    ///
    /// # Examples
    /// ```
    /// use rype::MinimizerWorkspace;
    ///
    /// // 200bp sequence with k=32, w=10
    /// let estimate = MinimizerWorkspace::estimate_for_length(200, 32, 10);
    /// // ((200 - 32) / 10 + 1) * 2 = 36
    /// assert!(estimate >= 32);
    /// ```
    pub fn estimate_for_length(seq_len: usize, k: usize, w: usize) -> usize {
        if seq_len <= k {
            ESTIMATED_MINIMIZERS_PER_SEQUENCE
        } else {
            (((seq_len - k) / w + 1) * 2).max(ESTIMATED_MINIMIZERS_PER_SEQUENCE)
        }
    }
}

impl Default for MinimizerWorkspace {
    fn default() -> Self {
        Self::new()
    }
}
