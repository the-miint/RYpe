//! Command-line interface definitions and helpers for the rype CLI.

pub mod args;
pub mod classify;
pub mod helpers;
pub mod index;
pub mod inspect;

pub use args::{ClassifyCommands, Cli, Commands, IndexCommands, InspectCommands};
pub use classify::{run_aggregate, run_classify, ClassifyAggregateArgs, ClassifyRunArgs};
pub use helpers::{load_index_metadata, resolve_bucket_id};
pub use index::{
    build_parquet_index_from_config, build_parquet_index_from_config_streaming,
    create_parquet_index_from_refs,
};
pub use inspect::inspect_matches;
