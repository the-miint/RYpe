//! Integration tests for the C API with Parquet indices.
//!
//! These tests verify that the C API works correctly with Parquet inverted indices.

use anyhow::Result;
use rype::c_api::{
    rype_bucket_name, rype_classify, rype_classify_with_negative, rype_get_last_error,
    rype_index_free, rype_index_is_sharded, rype_index_k, rype_index_load, rype_index_num_buckets,
    rype_index_num_shards, rype_index_salt, rype_index_w, rype_negative_set_create,
    rype_negative_set_free, rype_results_free, RypeQuery,
};
use rype::{
    extract_into, BucketData, IndexMetadata, InvertedIndex, MinimizerWorkspace, ParquetWriteOptions,
};
use std::collections::HashMap;
use std::ffi::CString;
use std::ptr;
use tempfile::tempdir;

/// Helper to generate a DNA sequence with a deterministic pattern.
fn generate_sequence(len: usize, seed: u8) -> Vec<u8> {
    let bases = [b'A', b'C', b'G', b'T'];
    (0..len).map(|i| bases[(i + seed as usize) % 4]).collect()
}

/// Helper to create a Parquet index with known content.
fn create_test_parquet_index(
    dir: &std::path::Path,
    k: usize,
    w: usize,
    salt: u64,
) -> Result<std::path::PathBuf> {
    let index_path = dir.join("test.ryxdi");

    let mut ws = MinimizerWorkspace::new();

    // Generate sequences and extract minimizers
    let seq1 = generate_sequence(200, 0);
    let seq2 = generate_sequence(200, 1);

    extract_into(&seq1, k, w, salt, &mut ws);
    let mut mins1: Vec<u64> = ws.buffer.drain(..).collect();
    mins1.sort();
    mins1.dedup();

    extract_into(&seq2, k, w, salt, &mut ws);
    let mut mins2: Vec<u64> = ws.buffer.drain(..).collect();
    mins2.sort();
    mins2.dedup();

    // Build bucket data
    let buckets = vec![
        BucketData {
            bucket_id: 1,
            bucket_name: "BucketA".to_string(),
            sources: vec!["src1::seq1".to_string()],
            minimizers: mins1,
        },
        BucketData {
            bucket_id: 2,
            bucket_name: "BucketB".to_string(),
            sources: vec!["src2::seq1".to_string()],
            minimizers: mins2,
        },
    ];

    let options = ParquetWriteOptions::default();
    rype::create_parquet_inverted_index(&index_path, buckets, k, w, salt, None, Some(&options))?;

    Ok(index_path)
}

/// Helper to build queries for classification.
fn make_query(id: i64, seq: &[u8]) -> (RypeQuery, Vec<u8>) {
    let seq_owned = seq.to_vec();
    let query = RypeQuery {
        id,
        seq: seq_owned.as_ptr() as *const i8,
        seq_len: seq_owned.len(),
        pair_seq: ptr::null(),
        pair_len: 0,
    };
    (query, seq_owned)
}

// =============================================================================
// Parquet Index Loading Tests
// =============================================================================

#[test]
fn test_unified_load_parquet_index() -> Result<()> {
    let dir = tempdir()?;
    let index_path = create_test_parquet_index(dir.path(), 32, 10, 0x12345)?;

    // Load via C API
    let path_cstr = CString::new(index_path.to_str().unwrap())?;
    let loaded = rype_index_load(path_cstr.as_ptr());

    assert!(!loaded.is_null(), "Should load Parquet index");

    // Verify accessors
    assert_eq!(rype_index_k(loaded), 32);
    assert_eq!(rype_index_w(loaded), 10);
    assert_eq!(rype_index_salt(loaded), 0x12345);
    assert_eq!(rype_index_num_buckets(loaded), 2);
    assert_eq!(rype_index_is_sharded(loaded), 1); // Parquet is always "sharded"
    assert!(rype_index_num_shards(loaded) >= 1);

    // Verify bucket names
    let name_ptr = rype_bucket_name(loaded, 1);
    assert!(!name_ptr.is_null());
    let name = unsafe { std::ffi::CStr::from_ptr(name_ptr) };
    assert_eq!(name.to_str().unwrap(), "BucketA");

    let name_ptr_2 = rype_bucket_name(loaded, 2);
    assert!(!name_ptr_2.is_null());
    let name_2 = unsafe { std::ffi::CStr::from_ptr(name_ptr_2) };
    assert_eq!(name_2.to_str().unwrap(), "BucketB");

    rype_index_free(loaded);
    Ok(())
}

#[test]
fn test_unified_classify_parquet_index() -> Result<()> {
    let dir = tempdir()?;
    let index_path = create_test_parquet_index(dir.path(), 32, 10, 0x12345)?;

    let path_cstr = CString::new(index_path.to_str().unwrap())?;
    let loaded = rype_index_load(path_cstr.as_ptr());
    assert!(!loaded.is_null());

    // Create a query that matches bucket 1
    let query_seq = generate_sequence(200, 0);
    let (query, _seq_holder) = make_query(1, &query_seq);

    let results = rype_classify(loaded, &query, 1, 0.1);

    assert!(!results.is_null(), "Classification should succeed");

    let results_ref = unsafe { &*results };
    assert!(results_ref.len > 0, "Should have at least one hit");

    // Check that the hit is for the expected bucket
    let hits = unsafe { std::slice::from_raw_parts(results_ref.data, results_ref.len) };
    let bucket_1_hit = hits.iter().any(|h| h.bucket_id == 1);
    assert!(bucket_1_hit, "Should have a hit for bucket 1");

    rype_results_free(results);
    rype_index_free(loaded);
    Ok(())
}

#[test]
fn test_unified_classify_multiple_queries() -> Result<()> {
    let dir = tempdir()?;
    let index_path = create_test_parquet_index(dir.path(), 32, 10, 0x12345)?;

    let path_cstr = CString::new(index_path.to_str().unwrap())?;
    let loaded = rype_index_load(path_cstr.as_ptr());
    assert!(!loaded.is_null());

    // Create multiple queries
    let seq1 = generate_sequence(200, 0);
    let seq2 = generate_sequence(200, 1);
    let seq3 = generate_sequence(200, 2); // Different from both

    let (query1, _h1) = make_query(1, &seq1);
    let (query2, _h2) = make_query(2, &seq2);
    let (query3, _h3) = make_query(3, &seq3);

    let queries = [query1, query2, query3];

    let results = rype_classify(loaded, queries.as_ptr(), 3, 0.1);

    assert!(!results.is_null(), "Classification should succeed");

    let results_ref = unsafe { &*results };
    // At least queries 1 and 2 should have hits (they match the buckets)
    assert!(
        results_ref.len >= 2,
        "Should have hits for matching queries"
    );

    rype_results_free(results);
    rype_index_free(loaded);
    Ok(())
}

// =============================================================================
// Negative Set Tests
// =============================================================================

#[test]
fn test_unified_negative_set_not_supported() -> Result<()> {
    let dir = tempdir()?;
    let index_path = create_test_parquet_index(dir.path(), 32, 10, 0x12345)?;

    let path_cstr = CString::new(index_path.to_str().unwrap())?;
    let loaded = rype_index_load(path_cstr.as_ptr());
    assert!(!loaded.is_null());

    // Creating negative set from sharded index is not supported
    let neg_set = rype_negative_set_create(loaded);
    assert!(
        neg_set.is_null(),
        "Negative set creation should return NULL for sharded indices"
    );

    // Should have an error message
    let err = rype_get_last_error();
    assert!(!err.is_null(), "Should set error message");

    rype_index_free(loaded);
    Ok(())
}

#[test]
fn test_unified_classify_with_null_negative_set() -> Result<()> {
    let dir = tempdir()?;
    let index_path = create_test_parquet_index(dir.path(), 32, 10, 0x12345)?;

    let path_cstr = CString::new(index_path.to_str().unwrap())?;
    let loaded = rype_index_load(path_cstr.as_ptr());
    assert!(!loaded.is_null());

    let query_seq = generate_sequence(200, 0);
    let (query, _seq_holder) = make_query(1, &query_seq);

    // Classify with NULL negative set (should work)
    let results = rype_classify_with_negative(loaded, ptr::null(), &query, 1, 0.1);

    assert!(
        !results.is_null(),
        "Classification should succeed with NULL negative set"
    );

    let results_ref = unsafe { &*results };
    assert!(results_ref.len > 0, "Should have hits");

    rype_results_free(results);
    rype_index_free(loaded);
    Ok(())
}

// =============================================================================
// Error Handling Tests
// =============================================================================

#[test]
fn test_unified_load_null_path() {
    let result = rype_index_load(ptr::null());
    assert!(result.is_null(), "Should return NULL for null path");

    let err = rype_get_last_error();
    assert!(!err.is_null(), "Should set error message");
}

#[test]
fn test_unified_load_nonexistent_path() {
    let path = CString::new("/nonexistent/path/index.ryxdi").unwrap();
    let result = rype_index_load(path.as_ptr());
    assert!(result.is_null(), "Should return NULL for nonexistent path");

    let err = rype_get_last_error();
    assert!(!err.is_null(), "Should set error message");
}

#[test]
fn test_unified_classify_null_index() {
    let seq = b"ACGTACGTACGT";
    let query = RypeQuery {
        id: 1,
        seq: seq.as_ptr() as *const i8,
        seq_len: seq.len(),
        pair_seq: ptr::null(),
        pair_len: 0,
    };

    let result = rype_classify(ptr::null(), &query, 1, 0.1);
    assert!(result.is_null(), "Should return NULL for null index");
}

#[test]
fn test_unified_classify_invalid_threshold() -> Result<()> {
    let dir = tempdir()?;
    let index_path = create_test_parquet_index(dir.path(), 32, 10, 0x12345)?;

    let path_cstr = CString::new(index_path.to_str().unwrap())?;
    let loaded = rype_index_load(path_cstr.as_ptr());
    assert!(!loaded.is_null());

    let seq = b"ACGTACGTACGT";
    let query = RypeQuery {
        id: 1,
        seq: seq.as_ptr() as *const i8,
        seq_len: seq.len(),
        pair_seq: ptr::null(),
        pair_len: 0,
    };

    // Test invalid threshold > 1.0
    let result = rype_classify(loaded, &query, 1, 1.5);
    assert!(
        result.is_null(),
        "Should return NULL for invalid threshold > 1.0"
    );

    // Test invalid threshold < 0.0
    let result = rype_classify(loaded, &query, 1, -0.5);
    assert!(
        result.is_null(),
        "Should return NULL for invalid threshold < 0.0"
    );

    rype_index_free(loaded);
    Ok(())
}

// =============================================================================
// Index Metadata Tests
// =============================================================================

#[test]
fn test_unified_index_accessors_null_safety() {
    // All accessors should return 0 for NULL index
    assert_eq!(rype_index_k(ptr::null()), 0);
    assert_eq!(rype_index_w(ptr::null()), 0);
    assert_eq!(rype_index_salt(ptr::null()), 0);
    assert_eq!(rype_index_num_buckets(ptr::null()), 0);
    assert_eq!(rype_index_is_sharded(ptr::null()), 0);
    assert_eq!(rype_index_num_shards(ptr::null()), 0);

    // bucket_name should return NULL for NULL index
    assert!(rype_bucket_name(ptr::null(), 1).is_null());
}

#[test]
fn test_unified_bucket_name_invalid_id() -> Result<()> {
    let dir = tempdir()?;
    let index_path = create_test_parquet_index(dir.path(), 32, 10, 0x12345)?;

    let path_cstr = CString::new(index_path.to_str().unwrap())?;
    let loaded = rype_index_load(path_cstr.as_ptr());
    assert!(!loaded.is_null());

    // Non-existent bucket ID should return NULL
    let name_ptr = rype_bucket_name(loaded, 999);
    assert!(
        name_ptr.is_null(),
        "Should return NULL for non-existent bucket"
    );

    rype_index_free(loaded);
    Ok(())
}

// =============================================================================
// Memory Safety Tests
// =============================================================================

#[test]
fn test_unified_free_null_safe() {
    // Free functions should be safe to call with NULL
    rype_index_free(ptr::null_mut());
    rype_results_free(ptr::null_mut());
    rype_negative_set_free(ptr::null_mut());
    // If we get here without crashing, the test passes
}

#[test]
fn test_unified_results_empty_query() -> Result<()> {
    let dir = tempdir()?;
    let index_path = create_test_parquet_index(dir.path(), 32, 10, 0x12345)?;

    let path_cstr = CString::new(index_path.to_str().unwrap())?;
    let loaded = rype_index_load(path_cstr.as_ptr());
    assert!(!loaded.is_null());

    // Query with sequence that won't match (too short)
    let seq = b"ACGT"; // 4 bases, less than k=32
    let query = RypeQuery {
        id: 1,
        seq: seq.as_ptr() as *const i8,
        seq_len: seq.len(),
        pair_seq: ptr::null(),
        pair_len: 0,
    };

    let results = rype_classify(loaded, &query, 1, 0.1);

    assert!(
        !results.is_null(),
        "Classification should succeed even with short query"
    );

    let results_ref = unsafe { &*results };
    // Short query should have 0 hits (no minimizers extracted)
    assert_eq!(results_ref.len, 0, "Short query should have no hits");

    rype_results_free(results);
    rype_index_free(loaded);
    Ok(())
}

// =============================================================================
// Direct InvertedIndex Tests (for coverage)
// =============================================================================

#[test]
fn test_inverted_index_build_from_bucket_map() {
    let mut bucket_map: HashMap<u32, Vec<u64>> = HashMap::new();
    bucket_map.insert(1, vec![100, 200, 300]);
    bucket_map.insert(2, vec![200, 300, 400]);

    let mut bucket_names = HashMap::new();
    bucket_names.insert(1, "BucketA".to_string());
    bucket_names.insert(2, "BucketB".to_string());

    let metadata = IndexMetadata {
        k: 32,
        w: 10,
        salt: 0x12345,
        bucket_names,
        bucket_sources: HashMap::new(),
        bucket_minimizer_counts: bucket_map.iter().map(|(&id, v)| (id, v.len())).collect(),
    };

    let inverted = InvertedIndex::build_from_bucket_map(32, 10, 0x12345, &bucket_map, &metadata);

    assert_eq!(inverted.k, 32);
    assert_eq!(inverted.w, 10);
    assert_eq!(inverted.num_minimizers(), 4); // 100, 200, 300, 400
    assert!(inverted.num_bucket_entries() > 0);

    // Test query
    let hits = inverted.get_bucket_hits(&[200, 300]);
    assert_eq!(hits.get(&1), Some(&2)); // Both minimizers in bucket 1
    assert_eq!(hits.get(&2), Some(&2)); // Both minimizers in bucket 2
}

// =============================================================================
// Paired-End Classification Tests
// =============================================================================

/// Helper to build paired-end queries for classification.
fn make_paired_query(id: i64, seq1: &[u8], seq2: &[u8]) -> (RypeQuery, Vec<u8>, Vec<u8>) {
    let seq1_owned = seq1.to_vec();
    let seq2_owned = seq2.to_vec();
    let query = RypeQuery {
        id,
        seq: seq1_owned.as_ptr() as *const i8,
        seq_len: seq1_owned.len(),
        pair_seq: seq2_owned.as_ptr() as *const i8,
        pair_len: seq2_owned.len(),
    };
    (query, seq1_owned, seq2_owned)
}

#[test]
fn test_unified_classify_paired_end_basic() -> Result<()> {
    let dir = tempdir()?;
    let index_path = create_test_parquet_index(dir.path(), 32, 10, 0x12345)?;

    let path_cstr = CString::new(index_path.to_str().unwrap())?;
    let loaded = rype_index_load(path_cstr.as_ptr());
    assert!(!loaded.is_null());

    // Create a paired-end query: read1 matches bucket 1, read2 matches bucket 2
    let seq1 = generate_sequence(200, 0); // Matches bucket 1
    let seq2 = generate_sequence(200, 1); // Matches bucket 2
    let (query, _h1, _h2) = make_paired_query(1, &seq1, &seq2);

    let results = rype_classify(loaded, &query, 1, 0.1);

    assert!(
        !results.is_null(),
        "Paired-end classification should succeed"
    );

    let results_ref = unsafe { &*results };
    assert!(results_ref.len >= 2, "Should have hits for both buckets");

    // Check that we have hits for both bucket 1 and bucket 2
    let hits = unsafe { std::slice::from_raw_parts(results_ref.data, results_ref.len) };
    let bucket_1_hit = hits.iter().any(|h| h.bucket_id == 1);
    let bucket_2_hit = hits.iter().any(|h| h.bucket_id == 2);

    assert!(bucket_1_hit, "Should have hit for bucket 1 (from read1)");
    assert!(bucket_2_hit, "Should have hit for bucket 2 (from read2)");

    rype_results_free(results);
    rype_index_free(loaded);
    Ok(())
}

#[test]
fn test_unified_classify_paired_end_same_bucket() -> Result<()> {
    let dir = tempdir()?;
    let index_path = create_test_parquet_index(dir.path(), 32, 10, 0x12345)?;

    let path_cstr = CString::new(index_path.to_str().unwrap())?;
    let loaded = rype_index_load(path_cstr.as_ptr());
    assert!(!loaded.is_null());

    // Both read1 and read2 match bucket 1
    let seq1 = generate_sequence(200, 0);
    let seq2 = generate_sequence(200, 0); // Same as seq1
    let (query, _h1, _h2) = make_paired_query(1, &seq1, &seq2);

    let results = rype_classify(loaded, &query, 1, 0.1);

    assert!(
        !results.is_null(),
        "Paired-end classification should succeed"
    );

    let results_ref = unsafe { &*results };
    assert!(results_ref.len >= 1, "Should have at least one hit");

    // Check bucket 1 hit with high score (both reads match)
    let hits = unsafe { std::slice::from_raw_parts(results_ref.data, results_ref.len) };
    let bucket_1_hit = hits.iter().find(|h| h.bucket_id == 1);
    assert!(bucket_1_hit.is_some(), "Should have hit for bucket 1");
    assert!(
        bucket_1_hit.unwrap().score > 0.9,
        "Score should be high when both reads match same bucket"
    );

    rype_results_free(results);
    rype_index_free(loaded);
    Ok(())
}

#[test]
fn test_unified_classify_paired_end_multiple_queries() -> Result<()> {
    let dir = tempdir()?;
    let index_path = create_test_parquet_index(dir.path(), 32, 10, 0x12345)?;

    let path_cstr = CString::new(index_path.to_str().unwrap())?;
    let loaded = rype_index_load(path_cstr.as_ptr());
    assert!(!loaded.is_null());

    // Multiple paired-end queries
    let seq1a = generate_sequence(200, 0);
    let seq1b = generate_sequence(200, 1);
    let seq2a = generate_sequence(200, 1);
    let seq2b = generate_sequence(200, 0);

    let (query1, _h1a, _h1b) = make_paired_query(1, &seq1a, &seq1b);
    let (query2, _h2a, _h2b) = make_paired_query(2, &seq2a, &seq2b);

    let queries = [query1, query2];

    let results = rype_classify(loaded, queries.as_ptr(), 2, 0.1);

    assert!(
        !results.is_null(),
        "Multiple paired-end classification should succeed"
    );

    let results_ref = unsafe { &*results };
    let hits = unsafe { std::slice::from_raw_parts(results_ref.data, results_ref.len) };

    // Query 1 should have hits for bucket 1 (read1) and bucket 2 (read2)
    let q1_b1 = hits.iter().any(|h| h.query_id == 1 && h.bucket_id == 1);
    let q1_b2 = hits.iter().any(|h| h.query_id == 1 && h.bucket_id == 2);
    assert!(q1_b1, "Query 1 should hit bucket 1");
    assert!(q1_b2, "Query 1 should hit bucket 2");

    // Query 2 should have hits for bucket 2 (read1) and bucket 1 (read2)
    let q2_b1 = hits.iter().any(|h| h.query_id == 2 && h.bucket_id == 1);
    let q2_b2 = hits.iter().any(|h| h.query_id == 2 && h.bucket_id == 2);
    assert!(q2_b1, "Query 2 should hit bucket 1");
    assert!(q2_b2, "Query 2 should hit bucket 2");

    rype_results_free(results);
    rype_index_free(loaded);
    Ok(())
}

#[test]
fn test_unified_classify_paired_end_short_read2() -> Result<()> {
    let dir = tempdir()?;
    let index_path = create_test_parquet_index(dir.path(), 32, 10, 0x12345)?;

    let path_cstr = CString::new(index_path.to_str().unwrap())?;
    let loaded = rype_index_load(path_cstr.as_ptr());
    assert!(!loaded.is_null());

    // Read1 is valid, read2 is too short (< k)
    let seq1 = generate_sequence(200, 0);
    let seq2 = b"ACGTACGT"; // 8 bases, less than k=32
    let (query, _h1, _h2) = make_paired_query(1, &seq1, seq2);

    let results = rype_classify(loaded, &query, 1, 0.1);

    assert!(
        !results.is_null(),
        "Classification should succeed even with short read2"
    );

    // Should still have results from read1
    let results_ref = unsafe { &*results };
    assert!(results_ref.len > 0, "Should have hits from read1");

    rype_results_free(results);
    rype_index_free(loaded);
    Ok(())
}

#[test]
fn test_unified_classify_paired_end_short_read1() -> Result<()> {
    let dir = tempdir()?;
    let index_path = create_test_parquet_index(dir.path(), 32, 10, 0x12345)?;

    let path_cstr = CString::new(index_path.to_str().unwrap())?;
    let loaded = rype_index_load(path_cstr.as_ptr());
    assert!(!loaded.is_null());

    // Read1 is too short, read2 is valid
    let seq1 = b"ACGTACGT"; // 8 bases, less than k=32
    let seq2 = generate_sequence(200, 1);
    let (query, _h1, _h2) = make_paired_query(1, seq1, &seq2);

    let results = rype_classify(loaded, &query, 1, 0.1);

    assert!(
        !results.is_null(),
        "Classification should succeed even with short read1"
    );

    // Should still have results from read2
    let results_ref = unsafe { &*results };
    assert!(results_ref.len > 0, "Should have hits from read2");

    rype_results_free(results);
    rype_index_free(loaded);
    Ok(())
}

#[test]
fn test_unified_classify_paired_end_both_short() -> Result<()> {
    let dir = tempdir()?;
    let index_path = create_test_parquet_index(dir.path(), 32, 10, 0x12345)?;

    let path_cstr = CString::new(index_path.to_str().unwrap())?;
    let loaded = rype_index_load(path_cstr.as_ptr());
    assert!(!loaded.is_null());

    // Both reads too short
    let seq1 = b"ACGT";
    let seq2 = b"TGCA";
    let (query, _h1, _h2) = make_paired_query(1, seq1, seq2);

    let results = rype_classify(loaded, &query, 1, 0.1);

    assert!(
        !results.is_null(),
        "Classification should succeed even with both reads short"
    );

    // Should have no results (no minimizers extracted)
    let results_ref = unsafe { &*results };
    assert_eq!(
        results_ref.len, 0,
        "Both reads too short should produce no hits"
    );

    rype_results_free(results);
    rype_index_free(loaded);
    Ok(())
}

#[test]
fn test_unified_classify_paired_vs_single_end() -> Result<()> {
    let dir = tempdir()?;
    let index_path = create_test_parquet_index(dir.path(), 32, 10, 0x12345)?;

    let path_cstr = CString::new(index_path.to_str().unwrap())?;
    let loaded = rype_index_load(path_cstr.as_ptr());
    assert!(!loaded.is_null());

    let seq1 = generate_sequence(200, 0);
    let seq2 = generate_sequence(200, 1);

    // Single-end query with seq1
    let (single_query, _sh) = make_query(1, &seq1);
    let single_results = rype_classify(loaded, &single_query, 1, 0.1);
    assert!(!single_results.is_null());
    let single_ref = unsafe { &*single_results };

    // Paired-end query with seq1 + seq2
    let (paired_query, _ph1, _ph2) = make_paired_query(2, &seq1, &seq2);
    let paired_results = rype_classify(loaded, &paired_query, 1, 0.1);
    assert!(!paired_results.is_null());
    let paired_ref = unsafe { &*paired_results };

    // Paired-end should have more or equal hits (adds read2's matches)
    assert!(
        paired_ref.len >= single_ref.len,
        "Paired-end should have at least as many hits as single-end"
    );

    rype_results_free(single_results);
    rype_results_free(paired_results);
    rype_index_free(loaded);
    Ok(())
}
