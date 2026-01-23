//! Command-line interface definitions and helpers for the rype CLI.

pub mod args;
pub mod helpers;
pub mod index;
pub mod inspect;

pub use args::{ClassifyCommands, Cli, Commands, IndexCommands, InspectCommands};
pub use helpers::{
    is_parquet_input, load_index_metadata, stacked_batches_to_records, OutputFormat, OutputWriter,
    PrefetchingIoHandler, PrefetchingParquetReader,
};
pub use index::{build_parquet_index_from_config, create_parquet_index_from_refs};
pub use inspect::inspect_matches;
