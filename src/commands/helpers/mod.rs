//! Helper functions and utilities for the rype CLI.
//!
//! This module is organized into focused submodules:
//! - `arg_parsing` - CLI argument parsing utilities
//! - `batch_config` - Batch size computation for classification
//! - `index_loading` - Index loading for classification commands
//! - `input_reader` - Input reader setup for classification commands
//! - `metadata` - Bucket name and index metadata utilities
//! - `fastx_io` - FASTA/FASTQ I/O with background prefetching
//! - `output` - Output format detection and writing
//! - `parquet_io` - Parquet input reading with prefetching
//! - `log_ratio` - Log-ratio computation for two-bucket classification

mod arg_parsing;
mod batch_config;
pub(crate) mod deferred_denom;
mod fastx_io;
mod formatting;
mod index_loading;
mod input_reader;
mod log_ratio;
mod metadata;
mod output;
mod parquet_io;
mod passing_tracker;
pub(crate) mod seq_writer;

// Re-export everything for backward compatibility.
// Some items are marked #[allow(dead_code)] in their modules as they provide
// complete API surface for future use; we re-export them here for API stability.
pub use arg_parsing::{
    parse_bloom_fpp, parse_max_memory_arg, parse_shard_size_arg, validate_minimum_length,
    validate_trim_to,
};
pub use batch_config::{compute_effective_batch_size, BatchSizeConfig};
#[allow(unused_imports)]
pub use deferred_denom::{DeferredDenomBuffer, DeferredMeta};
#[allow(unused_imports)]
pub use fastx_io::OwnedFastxRecord;
#[allow(unused_imports)]
pub use fastx_io::PrefetchingIoHandler;
pub use formatting::format_classification_results;
#[allow(unused_imports)]
pub use index_loading::{
    load_index_for_classification, validate_parquet_index, IndexLoadOptions, LoadedIndex,
};
pub use input_reader::{
    create_input_reader, validate_input_config, ClassificationInput, InputReaderConfig,
};
#[allow(unused_imports)]
pub use log_ratio::{
    compute_log_ratio, format_log_ratio_bucket_name, format_log_ratio_output, FastPath,
    LogRatioResult,
};
pub use metadata::{load_index_metadata, resolve_bucket_id, sanitize_bucket_name};
pub use output::{OutputFormat, OutputWriter};
#[allow(unused_imports)]
pub use parquet_io::{
    accumulate_owned_batches, batch_to_owned_records_trimmed, batch_to_records_parquet,
    batch_to_records_parquet_with_offset, is_parquet_input, stacked_batches_to_records,
    ParquetBatch, ParquetInputReader, PrefetchingParquetReader, TrimmedBatchResult,
};
pub use passing_tracker::PassingReadTracker;
#[allow(unused_imports)]
pub use seq_writer::{SeqFormat, SequenceWriter};
