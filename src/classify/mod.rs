//! Classification functions for matching query sequences against indexed references.
//!
//! Provides multiple classification strategies:
//! - `classify_batch`: Direct classification against an Index
//! - `classify_batch_sharded_sequential`: Classification using a ShardedInvertedIndex
//! - `classify_batch_sharded_merge_join`: Classification using merge-join algorithm
//! - `aggregate_batch`: Aggregated classification for paired-end reads

mod batch;
mod common;
mod merge_join;
mod scoring;
mod sharded;

// Re-export public API
pub use batch::{aggregate_batch, classify_batch};
pub use merge_join::classify_batch_merge_join;
pub use sharded::{
    classify_batch_sharded_main, classify_batch_sharded_merge_join,
    classify_batch_sharded_sequential,
};
