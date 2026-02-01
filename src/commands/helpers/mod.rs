//! Helper functions and utilities for the rype CLI.
//!
//! This module is organized into focused submodules:
//! - `arg_parsing` - CLI argument parsing utilities
//! - `metadata` - Bucket name and index metadata utilities
//! - `fastx_io` - FASTA/FASTQ I/O with background prefetching
//! - `output` - Output format detection and writing
//! - `parquet_io` - Parquet input reading with prefetching

mod arg_parsing;
mod fastx_io;
mod metadata;
mod output;
mod parquet_io;

// Re-export everything for backward compatibility.
// Some items are marked #[allow(dead_code)] in their modules as they provide
// complete API surface for future use; we re-export them here for API stability.
pub use arg_parsing::{
    parse_bloom_fpp, parse_max_memory_arg, parse_shard_size_arg, validate_trim_to,
};
#[allow(unused_imports)]
pub use fastx_io::OwnedRecord;
pub use fastx_io::PrefetchingIoHandler;
pub use metadata::{load_index_metadata, resolve_bucket_id, sanitize_bucket_name};
pub use output::{OutputFormat, OutputWriter};
#[allow(unused_imports)]
pub use parquet_io::{
    batch_to_records_parquet, batch_to_records_parquet_with_offset, is_parquet_input,
    stacked_batches_to_records, ParquetInputReader, PrefetchingParquetReader,
};
