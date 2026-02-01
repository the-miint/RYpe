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
        return Ok(IndexMetadata {
            k: manifest.k,
            w: manifest.w,
            salt: manifest.salt,
            bucket_names,
            bucket_sources,
            bucket_minimizer_counts: HashMap::new(),
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
}
