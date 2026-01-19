//! Command-line interface definitions and helpers for the rype CLI.

pub mod args;
pub mod helpers;
pub mod index;
pub mod inspect;

pub use args::{ClassifyCommands, Cli, Commands, IndexCommands, InspectCommands};
pub use helpers::{load_index_metadata, sanitize_bucket_name, IoHandler};
pub use index::{
    add_reference_file_to_index, bucket_add_from_config, build_index_from_config,
    create_parquet_index_from_refs,
};
pub use inspect::inspect_matches;
