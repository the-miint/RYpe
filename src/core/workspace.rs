//! Reusable workspace for minimizer extraction.
//!
//! The workspace pattern avoids repeated allocations in hot loops by providing
//! pre-allocated buffers that can be reused across multiple extraction calls.

use std::collections::VecDeque;

use crate::constants::DEFAULT_DEQUE_CAPACITY;

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
}

impl MinimizerWorkspace {
    /// Create a new workspace with default capacity.
    ///
    /// The default capacity is sized for typical window sizes (up to 128).
    pub fn new() -> Self {
        Self {
            q_fwd: VecDeque::with_capacity(DEFAULT_DEQUE_CAPACITY),
            q_rc: VecDeque::with_capacity(DEFAULT_DEQUE_CAPACITY),
            buffer: Vec::with_capacity(DEFAULT_DEQUE_CAPACITY),
        }
    }
}

impl Default for MinimizerWorkspace {
    fn default() -> Self {
        Self::new()
    }
}
