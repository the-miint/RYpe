//! Inspect command handlers for debugging minimizer matches.
//!
//! Note: This command is not currently available with Parquet indices.

use anyhow::{anyhow, Result};
use std::path::Path;

/// Main inspect matches function
///
/// Note: This debugging command is not currently supported with Parquet indices.
pub fn inspect_matches(
    _index_path: &Path,
    _queries_path: &Path,
    _ids_file: &Path,
    _bucket_filter: &[u32],
) -> Result<()> {
    Err(anyhow!(
        "inspect matches command is not currently supported with Parquet indices.\n\
         This debugging feature requires the legacy index format."
    ))
}
