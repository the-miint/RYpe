//! Command-line interface definitions and helpers for the rype CLI.

pub mod args;
pub mod classify;
pub mod cluster;
pub mod helpers;
pub mod index;
pub mod inspect;

#[allow(unused_imports)]
pub use args::{ClassifyCommands, Cli, ClusterArgs, Commands, IndexCommands, InspectCommands};
pub use classify::{
    run_aggregate, run_classify, run_log_ratio, ClassifyAggregateArgs, ClassifyLogRatioArgs,
    ClassifyRunArgs, CommonClassifyArgs,
};
pub use cluster::run_cluster;
pub use helpers::{load_index_metadata, resolve_bucket_id};
pub use index::{build_parquet_index_from_config, create_parquet_index_from_refs};
pub use inspect::inspect_matches;
