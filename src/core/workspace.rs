//! Reusable workspace for minimizer extraction.
//!
//! The workspace pattern avoids repeated allocations in hot loops by providing
//! pre-allocated buffers that can be reused across multiple extraction calls.

use std::collections::VecDeque;

use crate::constants::{DEFAULT_DEQUE_CAPACITY, ESTIMATED_MINIMIZERS_PER_SEQUENCE};

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
    /// Monotonic deque for forward strand k-mer hashes
    pub(crate) q_fwd: VecDeque<(usize, u64)>,
    /// Monotonic deque for reverse complement k-mer hashes
    pub(crate) q_rc: VecDeque<(usize, u64)>,
    /// Output buffer for extracted minimizers
    pub buffer: Vec<u64>,
    /// Estimated minimizers per sequence for pre-allocation.
    /// Use `with_estimate()` to set based on actual read length profiles.
    pub estimated_minimizers: usize,
}

impl MinimizerWorkspace {
    /// Create a new workspace with default capacity.
    ///
    /// The default capacity is sized for typical window sizes (up to 128).
    /// Uses `ESTIMATED_MINIMIZERS_PER_SEQUENCE` (32) for pre-allocation.
    pub fn new() -> Self {
        Self {
            q_fwd: VecDeque::with_capacity(DEFAULT_DEQUE_CAPACITY),
            q_rc: VecDeque::with_capacity(DEFAULT_DEQUE_CAPACITY),
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
            q_fwd: VecDeque::with_capacity(DEFAULT_DEQUE_CAPACITY),
            q_rc: VecDeque::with_capacity(DEFAULT_DEQUE_CAPACITY),
            buffer: Vec::with_capacity(DEFAULT_DEQUE_CAPACITY),
            estimated_minimizers: estimated_minimizers.max(ESTIMATED_MINIMIZERS_PER_SEQUENCE),
        }
    }
}

impl Default for MinimizerWorkspace {
    fn default() -> Self {
        Self::new()
    }
}
