//! Core algorithms for minimizer-based sequence processing.
//!
//! This module contains the fundamental algorithms used throughout rype:
//! - RY-space encoding and varint utilities
//! - Minimizer extraction algorithms
//! - Reusable workspace for avoiding allocations in hot loops
//! - Array-backed ring buffer for cache-efficient deque operations
//! - Merge utilities for sorted vectors

pub mod encoding;
pub mod extraction;
pub mod merge;
pub mod orientation;
pub mod ring_buffer;
pub mod workspace;

// Re-export commonly used items at the core module level
pub use encoding::base_to_bit;
pub use extraction::{
    count_hits, extract_dual_strand_into, extract_into, extract_with_positions,
    get_paired_minimizers_into, MinimizerWithPosition, Strand,
};
pub use workspace::MinimizerWorkspace;

// Re-export merge utilities
pub use merge::{kway_merge_dedup, merge_sorted_into};

// Re-export orientation items
pub use orientation::{
    choose_orientation, choose_orientation_sampled, Orientation, ORIENTATION_FIRST_N,
};

// Internal-only exports for crate use
pub(crate) use orientation::gallop_for_each;
