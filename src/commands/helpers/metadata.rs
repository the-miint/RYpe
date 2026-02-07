//! Metadata utilities for bucket names and index loading.

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::path::Path;

use rype::IndexMetadata;

/// Sanitize bucket names by replacing nonprintable characters with "_"
pub fn sanitize_bucket_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_control() || !c.is_ascii_graphic() && !c.is_whitespace() {
                '_'
            } else {
                c
            }
        })
        .collect()
}

/// Resolve a bucket identifier (either numeric ID or name) to a bucket ID.
///
/// This function allows users to specify buckets by either:
/// - Numeric ID (e.g., "1", "42") - takes precedence if input parses as u32
/// - Bucket name (e.g., "Bacteria", "my_bucket") - case-sensitive exact match
///
/// # Numeric Priority
///
/// If the identifier parses as a valid u32, it is treated as a numeric ID.
/// This means bucket names that are valid numbers (e.g., a bucket named "42")
/// cannot be referenced by name - use the numeric ID instead.
///
/// # Arguments
/// * `identifier` - String that is either a numeric ID or a bucket name
/// * `bucket_names` - Map of bucket ID to bucket name from the index
///
/// # Returns
/// The resolved bucket ID. Note: This function does NOT validate that the
/// bucket ID exists in the index - the caller should verify existence.
///
/// # Errors
/// Returns an error if:
/// - The identifier is empty
/// - The identifier is not a valid number AND not found as a bucket name
/// - Multiple buckets have the same name (ambiguous)
pub fn resolve_bucket_id(identifier: &str, bucket_names: &HashMap<u32, String>) -> Result<u32> {
    // Validate non-empty input
    let identifier = identifier.trim();
    if identifier.is_empty() {
        return Err(anyhow!("Bucket identifier cannot be empty"));
    }

    // First, try to parse as a numeric ID (takes precedence over name lookup)
    if let Ok(id) = identifier.parse::<u32>() {
        return Ok(id);
    }

    // Not a number, search by name (case-sensitive exact match)
    let matches: Vec<u32> = bucket_names
        .iter()
        .filter(|(_, name)| name.as_str() == identifier)
        .map(|(id, _)| *id)
        .collect();

    match matches.len() {
        0 => Err(anyhow!("Bucket '{}' not found in index", identifier)),
        1 => Ok(matches[0]),
        _ => Err(anyhow!(
            "Ambiguous bucket name '{}': matches {} buckets (IDs: {:?}). Use numeric ID instead.",
            identifier,
            matches.len(),
            matches
        )),
    }
}

/// Load metadata from a Parquet inverted index.
///
/// This helper handles Parquet inverted index directories (with manifest.toml).
pub fn load_index_metadata(path: &Path) -> Result<IndexMetadata> {
    // Parquet format (directory with manifest.toml)
    if rype::is_parquet_index(path) {
        let manifest = rype::ParquetManifest::load(path)?;
        let (bucket_names, bucket_sources) = rype::parquet_index::read_buckets_parquet(path)?;
        let largest_shard_entries = manifest
            .inverted
            .as_ref()
            .and_then(|inv| inv.shards.iter().map(|s| s.num_entries).max())
            .unwrap_or(0);
        return Ok(IndexMetadata {
            k: manifest.k,
            w: manifest.w,
            salt: manifest.salt,
            bucket_names,
            bucket_sources,
            bucket_minimizer_counts: HashMap::new(),
            largest_shard_entries,
        });
    }

    Err(anyhow!(
        "Invalid index format: expected Parquet index directory (.ryxdi/) at {:?}",
        path
    ))
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Tests for resolve_bucket_id
    // -------------------------------------------------------------------------

    #[test]
    fn test_resolve_bucket_id_numeric() {
        let mut bucket_names = HashMap::new();
        bucket_names.insert(1, "Bacteria".to_string());
        bucket_names.insert(2, "Archaea".to_string());
        bucket_names.insert(42, "Eukaryota".to_string());

        // Resolve by numeric ID
        assert_eq!(resolve_bucket_id("1", &bucket_names).unwrap(), 1);
        assert_eq!(resolve_bucket_id("2", &bucket_names).unwrap(), 2);
        assert_eq!(resolve_bucket_id("42", &bucket_names).unwrap(), 42);
    }

    #[test]
    fn test_resolve_bucket_id_by_name() {
        let mut bucket_names = HashMap::new();
        bucket_names.insert(1, "Bacteria".to_string());
        bucket_names.insert(2, "Archaea".to_string());
        bucket_names.insert(42, "Eukaryota".to_string());

        // Resolve by name
        assert_eq!(resolve_bucket_id("Bacteria", &bucket_names).unwrap(), 1);
        assert_eq!(resolve_bucket_id("Archaea", &bucket_names).unwrap(), 2);
        assert_eq!(resolve_bucket_id("Eukaryota", &bucket_names).unwrap(), 42);
    }

    #[test]
    fn test_resolve_bucket_id_name_not_found() {
        let mut bucket_names = HashMap::new();
        bucket_names.insert(1, "Bacteria".to_string());

        // Name not found should error
        let result = resolve_bucket_id("NotFound", &bucket_names);
        assert!(result.is_err());
        let err_msg = result.err().unwrap().to_string();
        assert!(err_msg.contains("not found"), "Error was: {}", err_msg);
    }

    #[test]
    fn test_resolve_bucket_id_ambiguous_name() {
        let mut bucket_names = HashMap::new();
        bucket_names.insert(1, "SameName".to_string());
        bucket_names.insert(2, "SameName".to_string());
        bucket_names.insert(3, "Unique".to_string());

        // Ambiguous name should error
        let result = resolve_bucket_id("SameName", &bucket_names);
        assert!(result.is_err());
        let err_msg = result.err().unwrap().to_string();
        assert!(err_msg.contains("Ambiguous"), "Error was: {}", err_msg);

        // Unique name still works
        assert_eq!(resolve_bucket_id("Unique", &bucket_names).unwrap(), 3);

        // Numeric ID still works for ambiguous names
        assert_eq!(resolve_bucket_id("1", &bucket_names).unwrap(), 1);
        assert_eq!(resolve_bucket_id("2", &bucket_names).unwrap(), 2);
    }

    #[test]
    fn test_resolve_bucket_id_numeric_not_in_index() {
        let mut bucket_names = HashMap::new();
        bucket_names.insert(1, "Bacteria".to_string());

        // Numeric ID that doesn't exist still returns (caller handles the error)
        // This allows the caller to give a better "bucket not found" error
        assert_eq!(resolve_bucket_id("999", &bucket_names).unwrap(), 999);
    }

    #[test]
    fn test_resolve_bucket_id_empty_bucket_names() {
        let bucket_names = HashMap::new();

        // Numeric ID works with empty map
        assert_eq!(resolve_bucket_id("1", &bucket_names).unwrap(), 1);

        // Name lookup fails with empty map
        let result = resolve_bucket_id("Bacteria", &bucket_names);
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_bucket_id_name_with_spaces() {
        let mut bucket_names = HashMap::new();
        bucket_names.insert(1, "My Bucket Name".to_string());

        // Names with spaces work
        assert_eq!(
            resolve_bucket_id("My Bucket Name", &bucket_names).unwrap(),
            1
        );
    }

    #[test]
    fn test_resolve_bucket_id_case_sensitive() {
        let mut bucket_names = HashMap::new();
        bucket_names.insert(1, "Bacteria".to_string());

        // Name matching is case-sensitive
        assert_eq!(resolve_bucket_id("Bacteria", &bucket_names).unwrap(), 1);

        let result = resolve_bucket_id("bacteria", &bucket_names);
        assert!(result.is_err());

        let result = resolve_bucket_id("BACTERIA", &bucket_names);
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_bucket_id_empty_string() {
        let mut bucket_names = HashMap::new();
        bucket_names.insert(1, "Bacteria".to_string());

        // Empty string should error
        let result = resolve_bucket_id("", &bucket_names);
        assert!(result.is_err());
        let err_msg = result.err().unwrap().to_string();
        assert!(
            err_msg.contains("cannot be empty"),
            "Error was: {}",
            err_msg
        );
    }

    #[test]
    fn test_resolve_bucket_id_whitespace_only() {
        let mut bucket_names = HashMap::new();
        bucket_names.insert(1, "Bacteria".to_string());

        // Whitespace-only string should error (trimmed to empty)
        let result = resolve_bucket_id("   ", &bucket_names);
        assert!(result.is_err());
        let err_msg = result.err().unwrap().to_string();
        assert!(
            err_msg.contains("cannot be empty"),
            "Error was: {}",
            err_msg
        );

        let result = resolve_bucket_id("\t\n", &bucket_names);
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_bucket_id_whitespace_trimmed() {
        let mut bucket_names = HashMap::new();
        bucket_names.insert(42, "Bacteria".to_string());

        // Leading/trailing whitespace should be trimmed
        assert_eq!(resolve_bucket_id("  42  ", &bucket_names).unwrap(), 42);
        assert_eq!(resolve_bucket_id("\t42\n", &bucket_names).unwrap(), 42);

        // Name with surrounding whitespace should also work after trim
        assert_eq!(
            resolve_bucket_id("  Bacteria  ", &bucket_names).unwrap(),
            42
        );
    }

    #[test]
    fn test_resolve_bucket_id_leading_zeros() {
        let mut bucket_names = HashMap::new();
        bucket_names.insert(7, "Bacteria".to_string());

        // Leading zeros parse correctly (007 -> 7)
        assert_eq!(resolve_bucket_id("007", &bucket_names).unwrap(), 7);
        assert_eq!(resolve_bucket_id("0007", &bucket_names).unwrap(), 7);
        assert_eq!(resolve_bucket_id("00", &bucket_names).unwrap(), 0);
    }

    #[test]
    fn test_resolve_bucket_id_numeric_name_collision() {
        let mut bucket_names = HashMap::new();
        bucket_names.insert(1, "Bacteria".to_string());
        bucket_names.insert(42, "1".to_string()); // Bucket named "1"

        // Numeric ID takes precedence - "1" resolves to ID 1, not bucket named "1"
        assert_eq!(resolve_bucket_id("1", &bucket_names).unwrap(), 1);

        // To get bucket 42 (named "1"), must use numeric ID
        assert_eq!(resolve_bucket_id("42", &bucket_names).unwrap(), 42);

        // Bucket named "1" is unreachable by name (this is documented behavior)
        // Attempting to resolve "1" gives ID 1, not the bucket named "1"
    }

    #[test]
    fn test_resolve_bucket_id_negative_number() {
        let bucket_names = HashMap::new();

        // Negative numbers don't parse as u32, so treated as names
        let result = resolve_bucket_id("-1", &bucket_names);
        assert!(result.is_err());
        let err_msg = result.err().unwrap().to_string();
        assert!(err_msg.contains("not found"), "Error was: {}", err_msg);
    }

    #[test]
    fn test_resolve_bucket_id_overflow() {
        let bucket_names = HashMap::new();

        // Number too large for u32 is treated as a name
        let result = resolve_bucket_id("999999999999999", &bucket_names);
        assert!(result.is_err());
        let err_msg = result.err().unwrap().to_string();
        assert!(err_msg.contains("not found"), "Error was: {}", err_msg);
    }

    // =========================================================================
    // Regression: load_index_metadata must populate largest_shard_entries
    // =========================================================================

    /// Helper to create a minimal Parquet index with known shard entries.
    fn create_test_index_for_metadata() -> tempfile::TempDir {
        use std::fs;

        let dir = tempfile::TempDir::new().unwrap();
        let index_path = dir.path().join("test.ryxdi");
        fs::create_dir(&index_path).unwrap();

        // Manifest with 2 shards: 1000 and 3000 entries
        let manifest = r#"magic = "RYPE_PARQUET_V1"
format_version = 1
k = 64
w = 50
salt = "0x5555555555555555"
source_hash = "0xDEADBEEF"
num_buckets = 2
total_minimizers = 4000

[inverted]
num_shards = 2
total_entries = 4000
has_overlapping_shards = false

[[inverted.shards]]
shard_id = 0
min_minimizer = "0x0000000000000001"
max_minimizer = "0x0000000000000064"
num_entries = 1000

[[inverted.shards]]
shard_id = 1
min_minimizer = "0x0000000000000065"
max_minimizer = "0x00000000000000C8"
num_entries = 3000
"#;
        fs::write(index_path.join("manifest.toml"), manifest).unwrap();

        // Create minimal buckets.parquet
        use arrow::array::{
            ArrayRef, LargeListBuilder, LargeStringArray, LargeStringBuilder, UInt32Array,
        };
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("bucket_id", DataType::UInt32, false),
            Field::new("bucket_name", DataType::LargeUtf8, false),
            Field::new(
                "sources",
                DataType::LargeList(Arc::new(Field::new("item", DataType::LargeUtf8, true))),
                false,
            ),
        ]));

        let mut list_builder = LargeListBuilder::new(LargeStringBuilder::new());
        list_builder.values().append_value("source0");
        list_builder.append(true);
        list_builder.values().append_value("source1");
        list_builder.append(true);
        let sources_array: ArrayRef = Arc::new(list_builder.finish());

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![0, 1])) as ArrayRef,
                Arc::new(LargeStringArray::from(vec!["bucket0", "bucket1"])) as ArrayRef,
                sources_array,
            ],
        )
        .unwrap();

        let file = fs::File::create(index_path.join("buckets.parquet")).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        // Create inverted directory (not strictly needed for metadata loading
        // but keeps the index structure valid)
        fs::create_dir(index_path.join("inverted")).unwrap();

        dir
    }

    /// load_index_metadata must read shard info from manifest and populate
    /// largest_shard_entries with the max num_entries across shards.
    #[test]
    fn test_load_index_metadata_populates_shard_entries() {
        let dir = create_test_index_for_metadata();
        let index_path = dir.path().join("test.ryxdi");

        let metadata = load_index_metadata(&index_path).unwrap();

        // The manifest has 2 shards: 1000 and 3000 entries.
        // largest_shard_entries should be 3000.
        assert_eq!(
            metadata.largest_shard_entries, 3000,
            "Should pick the max shard num_entries; got {}",
            metadata.largest_shard_entries
        );
    }

    /// When the manifest has no inverted section, largest_shard_entries
    /// should default to 0.
    #[test]
    fn test_load_index_metadata_no_inverted_returns_zero() {
        use std::fs;

        let dir = tempfile::TempDir::new().unwrap();
        let index_path = dir.path().join("noinv.ryxdi");
        fs::create_dir(&index_path).unwrap();

        // Manifest without inverted section
        let manifest = r#"magic = "RYPE_PARQUET_V1"
format_version = 1
k = 64
w = 50
salt = "0x5555555555555555"
source_hash = "0xDEADBEEF"
num_buckets = 1
total_minimizers = 0
"#;
        fs::write(index_path.join("manifest.toml"), manifest).unwrap();

        // Still need buckets.parquet for the load to succeed
        use arrow::array::{
            ArrayRef, LargeListBuilder, LargeStringArray, LargeStringBuilder, UInt32Array,
        };
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("bucket_id", DataType::UInt32, false),
            Field::new("bucket_name", DataType::LargeUtf8, false),
            Field::new(
                "sources",
                DataType::LargeList(Arc::new(Field::new("item", DataType::LargeUtf8, true))),
                false,
            ),
        ]));

        let mut list_builder = LargeListBuilder::new(LargeStringBuilder::new());
        list_builder.values().append_value("src");
        list_builder.append(true);
        let sources_array: ArrayRef = Arc::new(list_builder.finish());

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![0])) as ArrayRef,
                Arc::new(LargeStringArray::from(vec!["b0"])) as ArrayRef,
                sources_array,
            ],
        )
        .unwrap();

        let file = fs::File::create(index_path.join("buckets.parquet")).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let metadata = load_index_metadata(&index_path).unwrap();
        assert_eq!(
            metadata.largest_shard_entries, 0,
            "No inverted section should give largest_shard_entries=0, got {}",
            metadata.largest_shard_entries
        );
    }
}
