//! Core algorithms for minimizer-based sequence processing.
//!
//! This module contains the fundamental algorithms used throughout rype:
//! - RY-space encoding and varint utilities
//! - Minimizer extraction algorithms
//! - Reusable workspace for avoiding allocations in hot loops
//! - Array-backed ring buffer for cache-efficient deque operations

pub mod encoding;
pub mod extraction;
pub mod ring_buffer;
pub mod workspace;

// Re-export commonly used items at the core module level
pub use encoding::base_to_bit;
pub use extraction::{
    count_hits, extract_dual_strand_into, extract_into, extract_with_positions,
    get_paired_minimizers_into, MinimizerWithPosition, Strand,
};
pub use workspace::MinimizerWorkspace;
