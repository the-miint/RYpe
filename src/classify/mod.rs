//! Classification functions for matching query sequences against indexed references.
//!
//! Provides multiple classification strategies:
//! - `classify_batch_sharded_sequential`: Classification using a ShardedInvertedIndex
//! - `classify_batch_sharded_merge_join`: Classification using merge-join algorithm
//! - `classify_batch_sharded_parallel_rg`: Parallel row group processing

mod common;
mod merge_join;
mod scoring;
mod sharded;

// Re-export public API
pub use merge_join::classify_batch_merge_join;
pub use sharded::{
    classify_batch_sharded_merge_join, classify_batch_sharded_parallel_rg,
    classify_batch_sharded_sequential,
};
