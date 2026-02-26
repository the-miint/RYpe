//! Integration tests for the C API with Parquet indices.
//!
//! These tests verify that the C API works correctly with Parquet inverted indices.

#![allow(deprecated)] // rype_index_is_sharded is deprecated but tested for backward compat

use anyhow::Result;
use rype::c_api::{
    rype_bucket_file_stats, rype_bucket_file_stats_free, rype_bucket_name,
    rype_calculate_batch_config, rype_classify, rype_classify_best_hit, rype_classify_log_ratio,
    rype_classify_with_negative, rype_get_last_error, rype_index_free, rype_index_is_sharded,
    rype_index_k, rype_index_load, rype_index_num_buckets, rype_index_num_shards, rype_index_salt,
    rype_index_w, rype_log_ratio_results_free, rype_negative_set_create, rype_negative_set_free,
    rype_negative_set_size, rype_recommend_batch_size, rype_results_free,
    rype_validate_log_ratio_indices, RypeQuery,
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
    rype::create_parquet_inverted_index(
        &index_path,
        buckets,
        k,
        w,
        salt,
        None,
        Some(&options),
        None,
    )?;

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
fn test_unified_negative_set_creation() -> Result<()> {
    let dir = tempdir()?;
    let index_path = create_test_parquet_index(dir.path(), 32, 10, 0x12345)?;

    let path_cstr = CString::new(index_path.to_str().unwrap())?;
    let loaded = rype_index_load(path_cstr.as_ptr());
    assert!(!loaded.is_null());

    // Creating negative set from sharded index now works (memory-efficient sharded filtering)
    let neg_set = rype_negative_set_create(loaded);
    assert!(
        !neg_set.is_null(),
        "Negative set creation should succeed for sharded indices"
    );

    // Should NOT have an error message
    let err = rype_get_last_error();
    assert!(err.is_null(), "Should not set error message on success");

    // Verify we can get the size (returns total minimizer count from manifest)
    let size = rype_negative_set_size(neg_set);
    assert!(size > 0, "Negative set should have non-zero size");

    rype_negative_set_free(neg_set);
    rype_index_free(loaded);
    Ok(())
}

#[test]
fn test_unified_negative_set_null_index() -> Result<()> {
    // Creating negative set from NULL index should fail
    let neg_set = rype_negative_set_create(ptr::null());
    assert!(
        neg_set.is_null(),
        "Negative set creation should return NULL for NULL index"
    );

    // Should have an error message
    let err = rype_get_last_error();
    assert!(!err.is_null(), "Should set error message");

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

#[test]
fn test_unified_classify_with_negative_filtering() -> Result<()> {
    // Create two separate indices: positive and negative
    let pos_dir = tempdir()?;
    let neg_dir = tempdir()?;

    // Create positive index
    let pos_index_path = create_test_parquet_index(pos_dir.path(), 32, 10, 0x12345)?;

    // Create a negative index from the same seed (so minimizers overlap)
    let neg_index_path = create_test_parquet_index(neg_dir.path(), 32, 10, 0x12345)?;

    // Load both indices
    let pos_path_cstr = CString::new(pos_index_path.to_str().unwrap())?;
    let neg_path_cstr = CString::new(neg_index_path.to_str().unwrap())?;

    let pos_loaded = rype_index_load(pos_path_cstr.as_ptr());
    let neg_loaded = rype_index_load(neg_path_cstr.as_ptr());
    assert!(!pos_loaded.is_null());
    assert!(!neg_loaded.is_null());

    // Create negative set from negative index
    let neg_set = rype_negative_set_create(neg_loaded);
    assert!(!neg_set.is_null(), "Negative set creation should succeed");

    // Create query sequence with same seed as indices (so it matches)
    let query_seq = generate_sequence(200, 0);
    let (query, _seq_holder) = make_query(1, &query_seq);

    // Classify without negative filtering
    let results_no_neg = rype_classify(pos_loaded, &query, 1, 0.0);
    assert!(!results_no_neg.is_null());
    let results_no_neg_ref = unsafe { &*results_no_neg };
    let hits_without_filtering = results_no_neg_ref.len;

    // Classify with negative filtering
    let results_with_neg = rype_classify_with_negative(pos_loaded, neg_set, &query, 1, 0.0);
    assert!(!results_with_neg.is_null());
    let results_with_neg_ref = unsafe { &*results_with_neg };
    let hits_with_filtering = results_with_neg_ref.len;

    // With threshold 0.0, negative filtering should reduce scores (since negative
    // index has overlapping minimizers). The exact behavior depends on how many
    // minimizers are filtered. With identical indices, all minimizers should be
    // filtered out, resulting in 0 hits or very low scores.
    //
    // Note: hits_with_filtering could be equal if there are no query minimizers
    // that hit both indices, or lower if there is overlap.
    assert!(
        hits_with_filtering <= hits_without_filtering,
        "Negative filtering should not increase hits"
    );

    rype_results_free(results_no_neg);
    rype_results_free(results_with_neg);
    rype_negative_set_free(neg_set);
    rype_index_free(neg_loaded);
    rype_index_free(pos_loaded);
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
        largest_shard_entries: 0,
        bucket_file_stats: None,
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

// =============================================================================
// Best Hit Filtering Tests
// =============================================================================

/// Helper to create a Parquet index where buckets share some minimizers.
/// This ensures a query can match both buckets.
fn create_overlapping_bucket_index(
    dir: &std::path::Path,
    k: usize,
    w: usize,
    salt: u64,
) -> Result<std::path::PathBuf> {
    let index_path = dir.join("overlap.ryxdi");

    let mut ws = MinimizerWorkspace::new();

    // Generate a base sequence and two variants
    let base_seq = generate_sequence(300, 0);

    // Bucket 1: minimizers from first 200 bases
    extract_into(&base_seq[..200], k, w, salt, &mut ws);
    let mut mins1: Vec<u64> = ws.buffer.drain(..).collect();
    mins1.sort();
    mins1.dedup();

    // Bucket 2: minimizers from last 200 bases (overlaps with bucket 1 in middle)
    extract_into(&base_seq[100..], k, w, salt, &mut ws);
    let mut mins2: Vec<u64> = ws.buffer.drain(..).collect();
    mins2.sort();
    mins2.dedup();

    // Build bucket data
    let buckets = vec![
        BucketData {
            bucket_id: 1,
            bucket_name: "Bucket1".to_string(),
            sources: vec!["src1".to_string()],
            minimizers: mins1,
        },
        BucketData {
            bucket_id: 2,
            bucket_name: "Bucket2".to_string(),
            sources: vec!["src2".to_string()],
            minimizers: mins2,
        },
    ];

    let options = ParquetWriteOptions::default();
    rype::create_parquet_inverted_index(
        &index_path,
        buckets,
        k,
        w,
        salt,
        None,
        Some(&options),
        None,
    )?;

    Ok(index_path)
}

#[test]
fn test_rype_classify_best_hit_filters_results() -> Result<()> {
    let dir = tempdir()?;
    let index_path = create_overlapping_bucket_index(dir.path(), 32, 10, 0x12345)?;

    let path_cstr = CString::new(index_path.to_str().unwrap())?;
    let loaded = rype_index_load(path_cstr.as_ptr());
    assert!(!loaded.is_null());

    // Create a query that overlaps with the middle portion (should match both buckets)
    let query_seq = generate_sequence(300, 0);
    let (query, _seq_holder) = make_query(1, &query_seq[100..200]); // Middle 100 bases

    // Test rype_classify (should return hits for both buckets)
    let results_all = rype_classify(loaded, &query, 1, 0.0); // threshold 0 to get all matches
    assert!(!results_all.is_null(), "Classification should succeed");

    let results_all_ref = unsafe { &*results_all };
    let hits_all = unsafe { std::slice::from_raw_parts(results_all_ref.data, results_all_ref.len) };

    // Count unique bucket IDs in regular results
    let mut bucket_ids_all: Vec<u32> = hits_all.iter().map(|h| h.bucket_id).collect();
    bucket_ids_all.sort();
    bucket_ids_all.dedup();

    // Test rype_classify_best_hit (should return at most one hit per query)
    let results_best = rype_classify_best_hit(loaded, &query, 1, 0.0);
    assert!(
        !results_best.is_null(),
        "Best-hit classification should succeed"
    );

    let results_best_ref = unsafe { &*results_best };

    // With best_hit, we should have at most one result per query
    assert!(
        results_best_ref.len <= 1,
        "Best-hit should return at most one result per query. Got {} results.",
        results_best_ref.len
    );

    // If we have a result, it should be the one with the highest score
    if results_best_ref.len == 1 && results_all_ref.len > 1 {
        let best_hit = unsafe { &*results_best_ref.data };

        // Find the max score from all results
        let max_score = hits_all.iter().map(|h| h.score).fold(0.0_f64, f64::max);

        assert!(
            (best_hit.score - max_score).abs() < 1e-10,
            "Best hit score {} should equal max score {}",
            best_hit.score,
            max_score
        );
    }

    println!(
        "Results: all={} hits ({} unique buckets), best_hit={} hits",
        results_all_ref.len,
        bucket_ids_all.len(),
        results_best_ref.len
    );

    rype_results_free(results_all);
    rype_results_free(results_best);
    rype_index_free(loaded);
    Ok(())
}

// =============================================================================
// Log-Ratio Classification Tests
// =============================================================================

/// Helper to create a single-bucket Parquet index for log-ratio testing.
fn create_single_bucket_index(
    dir: &std::path::Path,
    name: &str,
    k: usize,
    w: usize,
    salt: u64,
    seq: &[u8],
) -> Result<std::path::PathBuf> {
    let index_path = dir.join(format!("{}.ryxdi", name));

    let mut ws = MinimizerWorkspace::new();
    extract_into(seq, k, w, salt, &mut ws);
    let mut mins: Vec<u64> = ws.buffer.drain(..).collect();
    mins.sort();
    mins.dedup();

    let buckets = vec![BucketData {
        bucket_id: 0,
        bucket_name: name.to_string(),
        sources: vec![format!("test::{}", name)],
        minimizers: mins,
    }];

    let options = ParquetWriteOptions::default();
    rype::create_parquet_inverted_index(
        &index_path,
        buckets,
        k,
        w,
        salt,
        None,
        Some(&options),
        None,
    )?;

    Ok(index_path)
}

#[test]
fn test_log_ratio_c_api_end_to_end() -> Result<()> {
    let dir = tempdir()?;
    let k = 32;
    let w = 10;
    let salt = 0x5555555555555555_u64;

    // Create two single-bucket indices with very different sequences
    // Use long, dissimilar sequences to ensure distinct minimizer sets
    let seq_num: Vec<u8> = (0..1000)
        .map(|i| b"ACGTACGTACGTACGTAACCGGTTAACCGGTTACGTACGTACGT"[i % 44])
        .collect();
    let seq_denom: Vec<u8> = (0..1000)
        .map(|i| b"TGCATGCATGCATGCATTGGCCAATTGGCCAATGCATGCATGCA"[i % 44])
        .collect();

    let num_path = create_single_bucket_index(dir.path(), "numerator", k, w, salt, &seq_num)?;
    let denom_path = create_single_bucket_index(dir.path(), "denominator", k, w, salt, &seq_denom)?;

    // Load both via C API
    let num_cstr = CString::new(num_path.to_str().unwrap())?;
    let denom_cstr = CString::new(denom_path.to_str().unwrap())?;

    let num_idx = rype_index_load(num_cstr.as_ptr());
    let denom_idx = rype_index_load(denom_cstr.as_ptr());
    assert!(!num_idx.is_null(), "numerator index should load");
    assert!(!denom_idx.is_null(), "denominator index should load");

    // Validate indices are compatible
    let validate_result = rype_validate_log_ratio_indices(num_idx, denom_idx);
    assert_eq!(
        validate_result, 0,
        "Indices should be compatible for log-ratio"
    );

    // Build queries: one that matches numerator, one that matches denominator
    let (query_num, _hold_num) = make_query(0, &seq_num);
    let (query_denom, _hold_denom) = make_query(1, &seq_denom);
    let queries = [query_num, query_denom];

    // Classify with threshold disabled
    let results = rype_classify_log_ratio(num_idx, denom_idx, queries.as_ptr(), 2, -1.0);
    assert!(
        !results.is_null(),
        "Log-ratio classification should succeed"
    );

    let results_ref = unsafe { &*results };
    assert_eq!(results_ref.len, 2, "Should have one result per query");

    let hits = unsafe { std::slice::from_raw_parts(results_ref.data, results_ref.len) };

    // Query 0 (matches numerator) should have positive log-ratio
    let q0 = hits.iter().find(|h| h.query_id == 0).unwrap();
    assert!(
        q0.log_ratio > 0.0 || q0.log_ratio.is_infinite() && q0.log_ratio.is_sign_positive(),
        "Query matching numerator should have positive log-ratio, got {}",
        q0.log_ratio
    );
    assert_eq!(q0.fast_path, 0, "No fast-path with threshold disabled");

    // Query 1 (matches denominator) should have negative log-ratio
    let q1 = hits.iter().find(|h| h.query_id == 1).unwrap();
    assert!(
        q1.log_ratio < 0.0 || q1.log_ratio.is_infinite() && q1.log_ratio.is_sign_negative(),
        "Query matching denominator should have negative log-ratio, got {}",
        q1.log_ratio
    );

    rype_log_ratio_results_free(results);

    // Classify with fast-path threshold enabled
    let results2 = rype_classify_log_ratio(num_idx, denom_idx, queries.as_ptr(), 2, 0.5);
    assert!(
        !results2.is_null(),
        "Log-ratio with threshold should succeed"
    );

    let results2_ref = unsafe { &*results2 };
    let hits2 = unsafe { std::slice::from_raw_parts(results2_ref.data, results2_ref.len) };

    // Query 0 should likely be fast-path (high numerator score)
    let q0_fp = hits2.iter().find(|h| h.query_id == 0).unwrap();
    // If the numerator score >= 0.5, this should be fast-path (1), log_ratio = +inf
    if q0_fp.fast_path == 1 {
        assert!(
            q0_fp.log_ratio.is_infinite() && q0_fp.log_ratio.is_sign_positive(),
            "Fast-path should produce +inf"
        );
    }

    rype_log_ratio_results_free(results2);
    rype_index_free(num_idx);
    rype_index_free(denom_idx);
    Ok(())
}

/// Test that log-ratio classification works with non-sequential query IDs.
/// This catches the bug where partition_by_numerator_score assumes dense 0..N IDs.
#[test]
fn test_log_ratio_c_api_nonsequential_query_ids() -> Result<()> {
    let dir = tempdir()?;
    let k = 32;
    let w = 10;
    let salt = 0x5555555555555555_u64;

    let seq_num: Vec<u8> = (0..1000)
        .map(|i| b"ACGTACGTACGTACGTAACCGGTTAACCGGTTACGTACGTACGT"[i % 44])
        .collect();
    let seq_denom: Vec<u8> = (0..1000)
        .map(|i| b"TGCATGCATGCATGCATTGGCCAATTGGCCAATGCATGCATGCA"[i % 44])
        .collect();

    let num_path = create_single_bucket_index(dir.path(), "numerator", k, w, salt, &seq_num)?;
    let denom_path = create_single_bucket_index(dir.path(), "denominator", k, w, salt, &seq_denom)?;

    let num_cstr = CString::new(num_path.to_str().unwrap())?;
    let denom_cstr = CString::new(denom_path.to_str().unwrap())?;

    let num_idx = rype_index_load(num_cstr.as_ptr());
    let denom_idx = rype_index_load(denom_cstr.as_ptr());
    assert!(!num_idx.is_null());
    assert!(!denom_idx.is_null());

    // Use non-sequential query IDs (100, 999) — would panic without the fix
    let (query_num, _hold_num) = make_query(100, &seq_num);
    let (query_denom, _hold_denom) = make_query(999, &seq_denom);
    let queries = [query_num, query_denom];

    let results = rype_classify_log_ratio(num_idx, denom_idx, queries.as_ptr(), 2, -1.0);
    assert!(
        !results.is_null(),
        "Log-ratio with non-sequential IDs should succeed"
    );

    let results_ref = unsafe { &*results };
    assert_eq!(results_ref.len, 2);

    let hits = unsafe { std::slice::from_raw_parts(results_ref.data, results_ref.len) };

    // Verify original query IDs are preserved in results
    let q100 = hits.iter().find(|h| h.query_id == 100).unwrap();
    assert!(
        q100.log_ratio > 0.0 || q100.log_ratio.is_infinite() && q100.log_ratio.is_sign_positive(),
        "Query 100 (matches numerator) should have positive log-ratio, got {}",
        q100.log_ratio
    );

    let q999 = hits.iter().find(|h| h.query_id == 999).unwrap();
    assert!(
        q999.log_ratio < 0.0 || q999.log_ratio.is_infinite() && q999.log_ratio.is_sign_negative(),
        "Query 999 (matches denominator) should have negative log-ratio, got {}",
        q999.log_ratio
    );

    rype_log_ratio_results_free(results);
    rype_index_free(num_idx);
    rype_index_free(denom_idx);
    Ok(())
}

#[test]
fn test_log_ratio_c_api_validation_incompatible_indices() -> Result<()> {
    let dir = tempdir()?;
    let seq = generate_sequence(500, 0);

    // Create indices with different k values
    let idx1_path = create_single_bucket_index(dir.path(), "idx1", 32, 10, 0x5555, &seq)?;
    let idx2_path = create_single_bucket_index(dir.path(), "idx2", 64, 10, 0x5555, &seq)?;

    let cstr1 = CString::new(idx1_path.to_str().unwrap())?;
    let cstr2 = CString::new(idx2_path.to_str().unwrap())?;

    let idx1 = rype_index_load(cstr1.as_ptr());
    let idx2 = rype_index_load(cstr2.as_ptr());
    assert!(!idx1.is_null());
    assert!(!idx2.is_null());

    // Validation should fail — different k
    let result = rype_validate_log_ratio_indices(idx1, idx2);
    assert_eq!(result, -1, "Should fail for incompatible indices");

    let err = unsafe {
        std::ffi::CStr::from_ptr(rype_get_last_error())
            .to_string_lossy()
            .into_owned()
    };
    assert!(err.contains("different k values"), "Error: {}", err);

    rype_index_free(idx1);
    rype_index_free(idx2);
    Ok(())
}

#[test]
fn test_log_ratio_c_api_validation_multi_bucket_rejected() -> Result<()> {
    let dir = tempdir()?;
    let seq = generate_sequence(500, 0);

    // Create a multi-bucket index (the standard test helper creates 2 buckets)
    let multi_path = create_test_parquet_index(dir.path(), 32, 10, 0x5555)?;
    let single_path = create_single_bucket_index(dir.path(), "single", 32, 10, 0x5555, &seq)?;

    let cstr_multi = CString::new(multi_path.to_str().unwrap())?;
    let cstr_single = CString::new(single_path.to_str().unwrap())?;

    let multi_idx = rype_index_load(cstr_multi.as_ptr());
    let single_idx = rype_index_load(cstr_single.as_ptr());
    assert!(!multi_idx.is_null());
    assert!(!single_idx.is_null());

    // Multi-bucket as numerator should fail
    let result = rype_validate_log_ratio_indices(multi_idx, single_idx);
    assert_eq!(result, -1, "Multi-bucket numerator should fail");

    let err = unsafe {
        std::ffi::CStr::from_ptr(rype_get_last_error())
            .to_string_lossy()
            .into_owned()
    };
    assert!(err.contains("exactly 1 bucket"), "Error: {}", err);

    rype_index_free(multi_idx);
    rype_index_free(single_idx);
    Ok(())
}

/// Test rype_bucket_file_stats returns stats when the index has them.
#[test]
fn test_rype_bucket_file_stats() -> Result<()> {
    let dir = tempdir()?;
    let index_path = dir.path().join("stats.ryxdi");

    // Create an index with file stats
    let k = 32;
    let w = 10;
    let salt = 0x5555555555555555u64;
    let seq1 = generate_sequence(200, 0);
    let seq2 = generate_sequence(300, 1);

    let mut ws = MinimizerWorkspace::new();
    extract_into(&seq1, k, w, salt, &mut ws);
    let mins1: Vec<u64> = ws.buffer.drain(..).collect();
    extract_into(&seq2, k, w, salt, &mut ws);
    let mins2: Vec<u64> = ws.buffer.drain(..).collect();

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

    // Create index WITH file stats
    let mut file_stats = HashMap::new();
    file_stats.insert(
        1u32,
        rype::BucketFileStats {
            mean: 1000.0,
            median: 900.0,
            stdev: 100.0,
            min: 800.0,
            max: 1200.0,
        },
    );
    file_stats.insert(
        2u32,
        rype::BucketFileStats {
            mean: 2000.0,
            median: 1800.0,
            stdev: 200.0,
            min: 1600.0,
            max: 2400.0,
        },
    );

    let options = ParquetWriteOptions::default();
    rype::create_parquet_inverted_index(
        &index_path,
        buckets,
        k,
        w,
        salt,
        None,
        Some(&options),
        Some(&file_stats),
    )?;

    // Load via C API
    let path_cstr = CString::new(index_path.to_str().unwrap()).unwrap();
    let idx = rype_index_load(path_cstr.as_ptr());
    assert!(!idx.is_null(), "Index should load");

    // Get file stats
    let stats_ptr = rype_bucket_file_stats(idx);
    assert!(
        !stats_ptr.is_null(),
        "Should return non-null stats for index with file stats"
    );

    unsafe {
        let stats_array = &*stats_ptr;
        assert_eq!(stats_array.count, 2, "Should have 2 bucket stats");

        let entries = std::slice::from_raw_parts(stats_array.stats, stats_array.count);

        // Entries are sorted by bucket_id
        assert_eq!(entries[0].bucket_id, 1);
        assert!((entries[0].mean - 1000.0).abs() < 1e-6);
        assert!((entries[0].median - 900.0).abs() < 1e-6);
        assert!((entries[0].stdev - 100.0).abs() < 1e-6);
        assert!((entries[0].min - 800.0).abs() < 1e-6);
        assert!((entries[0].max - 1200.0).abs() < 1e-6);

        assert_eq!(entries[1].bucket_id, 2);
        assert!((entries[1].mean - 2000.0).abs() < 1e-6);
        assert!((entries[1].median - 1800.0).abs() < 1e-6);
        assert!((entries[1].stdev - 200.0).abs() < 1e-6);
        assert!((entries[1].min - 1600.0).abs() < 1e-6);
        assert!((entries[1].max - 2400.0).abs() < 1e-6);
    }

    rype_bucket_file_stats_free(stats_ptr);
    rype_index_free(idx);
    Ok(())
}

/// Test rype_bucket_file_stats returns NULL for index without stats.
#[test]
fn test_rype_bucket_file_stats_returns_null_without_stats() -> Result<()> {
    let dir = tempdir()?;
    let index_path = create_test_parquet_index(dir.path(), 32, 10, 0x5555555555555555)?;

    let path_cstr = CString::new(index_path.to_str().unwrap()).unwrap();
    let idx = rype_index_load(path_cstr.as_ptr());
    assert!(!idx.is_null());

    // Index created without file stats should return NULL
    let stats_ptr = rype_bucket_file_stats(idx);
    assert!(
        stats_ptr.is_null(),
        "Should return NULL for index without file stats"
    );

    // Safe to call free with NULL
    rype_bucket_file_stats_free(stats_ptr);

    rype_index_free(idx);
    Ok(())
}

// =============================================================================
// Batch Size Recommendation Tests
// =============================================================================

#[test]
fn test_recommend_batch_size_basic() -> Result<()> {
    let dir = tempdir()?;
    let index_path = create_test_parquet_index(dir.path(), 32, 10, 0x12345)?;

    let path_cstr = CString::new(index_path.to_str().unwrap())?;
    let idx = rype_index_load(path_cstr.as_ptr());
    assert!(!idx.is_null());

    // Basic single-end recommendation with 150bp reads
    let batch_se = rype_recommend_batch_size(idx, 150, 0, 4 * 1024 * 1024 * 1024);
    assert!(
        batch_se >= 1000,
        "batch_size should be >= MIN_BATCH_SIZE (1000), got {}",
        batch_se
    );

    // Auto-detect memory (max_memory=0) should return a reasonable value
    let batch_auto = rype_recommend_batch_size(idx, 150, 0, 0);
    assert!(
        batch_auto >= 1000,
        "auto-detect batch should be >= MIN_BATCH_SIZE, got {}",
        batch_auto
    );

    rype_index_free(idx);
    Ok(())
}

/// Use a tight memory budget so paired vs single-end and large vs small memory
/// differences are actually observable (not both capped at MAX_BATCH_SIZE).
#[test]
fn test_recommend_batch_size_memory_constraints() -> Result<()> {
    let dir = tempdir()?;
    let index_path = create_test_parquet_index(dir.path(), 32, 10, 0x12345)?;

    let path_cstr = CString::new(index_path.to_str().unwrap())?;
    let idx = rype_index_load(path_cstr.as_ptr());
    assert!(!idx.is_null());

    // Use a tight memory budget (512MB) so the memory constraint actually bites
    let tight_budget = 512 * 1024 * 1024;

    let batch_se = rype_recommend_batch_size(idx, 150, 0, tight_budget);
    assert!(
        batch_se >= 1000,
        "single-end batch should be >= MIN_BATCH_SIZE, got {}",
        batch_se
    );

    // Paired-end should be <= single-end (paired uses more memory per row)
    let batch_pe = rype_recommend_batch_size(idx, 150, 1, tight_budget);
    assert!(
        batch_pe >= 1000,
        "paired batch_size should be >= MIN_BATCH_SIZE, got {}",
        batch_pe
    );
    assert!(
        batch_pe <= batch_se,
        "paired batch_size ({}) should be <= single-end ({})",
        batch_pe,
        batch_se
    );

    // Smaller memory should give <= batch size
    let smaller_budget = 300 * 1024 * 1024;
    let batch_small = rype_recommend_batch_size(idx, 150, 0, smaller_budget);
    assert!(
        batch_small <= batch_se,
        "smaller memory batch ({}) should be <= larger memory batch ({})",
        batch_small,
        batch_se
    );

    rype_index_free(idx);
    Ok(())
}

/// When avg_read_length < k, minimizers_per_query is 0. Should still return
/// a valid batch size (MIN_BATCH_SIZE at minimum), not panic.
#[test]
fn test_recommend_batch_size_read_shorter_than_k() -> Result<()> {
    let dir = tempdir()?;
    // k=32, so reads of length 10 produce zero minimizers
    let index_path = create_test_parquet_index(dir.path(), 32, 10, 0x12345)?;

    let path_cstr = CString::new(index_path.to_str().unwrap())?;
    let idx = rype_index_load(path_cstr.as_ptr());
    assert!(!idx.is_null());

    let batch = rype_recommend_batch_size(idx, 10, 0, 4 * 1024 * 1024 * 1024);
    assert!(
        batch >= 1000,
        "degenerate read length should still return >= MIN_BATCH_SIZE, got {}",
        batch
    );

    rype_index_free(idx);
    Ok(())
}

#[test]
fn test_recommend_batch_size_error_cases() {
    // Null index should return 0
    let result = rype_recommend_batch_size(ptr::null(), 150, 0, 4 * 1024 * 1024 * 1024);
    assert_eq!(result, 0, "null index should return 0");

    let err = rype_get_last_error();
    assert!(!err.is_null(), "error should be set for null index");

    // Zero read length should return 0
    let dir = tempdir().unwrap();
    let index_path = create_test_parquet_index(dir.path(), 32, 10, 0x12345).unwrap();
    let path_cstr = CString::new(index_path.to_str().unwrap()).unwrap();
    let idx = rype_index_load(path_cstr.as_ptr());
    assert!(!idx.is_null());

    let result = rype_recommend_batch_size(idx, 0, 0, 4 * 1024 * 1024 * 1024);
    assert_eq!(result, 0, "zero avg_read_length should return 0");

    let err = rype_get_last_error();
    assert!(!err.is_null(), "error should be set for zero read length");

    rype_index_free(idx);
}

/// Verify that a zero-bucket index cannot even be loaded (caught at load time).
/// The pre-flight `num_buckets == 0` check in `rype_recommend_batch_size` exists
/// as defense-in-depth but cannot be triggered through normal API usage because
/// `rype_index_load` rejects zero-bucket indices first.
#[test]
fn test_recommend_batch_size_zero_bucket_index_rejected_at_load() -> Result<()> {
    let dir = tempdir()?;
    let index_path = dir.path().join("empty.ryxdi");

    // Build an index with no buckets
    let buckets: Vec<BucketData> = vec![];
    let options = ParquetWriteOptions::default();
    rype::create_parquet_inverted_index(
        &index_path,
        buckets,
        32,
        10,
        0x12345,
        None,
        Some(&options),
        None,
    )?;

    let path_cstr = CString::new(index_path.to_str().unwrap())?;
    let idx = rype_index_load(path_cstr.as_ptr());
    assert!(idx.is_null(), "zero-bucket index should fail to load");

    let err = rype_get_last_error();
    assert!(
        !err.is_null(),
        "error should be set for zero-bucket index load"
    );

    Ok(())
}

// =============================================================================
// Batch Config Calculation Tests
// =============================================================================

#[test]
fn test_calculate_batch_config_basic() -> Result<()> {
    let dir = tempdir()?;
    let index_path = create_test_parquet_index(dir.path(), 32, 10, 0x12345)?;

    let path_cstr = CString::new(index_path.to_str().unwrap())?;
    let idx = rype_index_load(path_cstr.as_ptr());
    assert!(!idx.is_null());

    let config = rype_calculate_batch_config(idx, 150, 0, 4 * 1024 * 1024 * 1024);
    assert!(
        config.batch_size >= 1000,
        "batch_size should be >= MIN_BATCH_SIZE (1000), got {}",
        config.batch_size
    );
    assert!(
        config.batch_count >= 1,
        "batch_count should be >= 1, got {}",
        config.batch_count
    );
    assert!(
        config.per_batch_memory > 0,
        "per_batch_memory should be > 0, got {}",
        config.per_batch_memory
    );
    assert!(
        config.peak_memory > 0,
        "peak_memory should be > 0, got {}",
        config.peak_memory
    );
    assert!(
        config.peak_memory >= config.per_batch_memory,
        "peak_memory ({}) should be >= per_batch_memory ({})",
        config.peak_memory,
        config.per_batch_memory
    );

    // batch_size should agree with rype_recommend_batch_size
    let batch_size_only = rype_recommend_batch_size(idx, 150, 0, 4 * 1024 * 1024 * 1024);
    assert_eq!(
        config.batch_size, batch_size_only,
        "calculate_batch_config and recommend_batch_size should agree"
    );

    rype_index_free(idx);
    Ok(())
}

#[test]
fn test_calculate_batch_config_error_cases() {
    // Null index
    let config = rype_calculate_batch_config(ptr::null(), 150, 0, 4 * 1024 * 1024 * 1024);
    assert_eq!(
        config.batch_size, 0,
        "null index should return batch_size 0"
    );
    assert_eq!(config.batch_count, 0);
    assert_eq!(config.per_batch_memory, 0);
    assert_eq!(config.peak_memory, 0);

    let err = rype_get_last_error();
    assert!(!err.is_null(), "error should be set for null index");

    // Zero read length
    let dir = tempdir().unwrap();
    let index_path = create_test_parquet_index(dir.path(), 32, 10, 0x12345).unwrap();
    let path_cstr = CString::new(index_path.to_str().unwrap()).unwrap();
    let idx = rype_index_load(path_cstr.as_ptr());
    assert!(!idx.is_null());

    let config = rype_calculate_batch_config(idx, 0, 0, 4 * 1024 * 1024 * 1024);
    assert_eq!(
        config.batch_size, 0,
        "zero avg_read_length should return batch_size 0"
    );

    rype_index_free(idx);
}

/// Paired-end should use more memory per row, yielding a smaller batch_size
/// under a tight budget. Also tests auto-detect memory (max_memory=0).
#[test]
fn test_calculate_batch_config_paired_and_memory() -> Result<()> {
    let dir = tempdir()?;
    let index_path = create_test_parquet_index(dir.path(), 32, 10, 0x12345)?;

    let path_cstr = CString::new(index_path.to_str().unwrap())?;
    let idx = rype_index_load(path_cstr.as_ptr());
    assert!(!idx.is_null());

    let tight_budget = 512 * 1024 * 1024; // 512 MB

    let cfg_se = rype_calculate_batch_config(idx, 150, 0, tight_budget);
    assert!(cfg_se.batch_size >= 1000);

    // Paired-end should be <= single-end batch_size under same budget
    let cfg_pe = rype_calculate_batch_config(idx, 150, 1, tight_budget);
    assert!(cfg_pe.batch_size >= 1000);
    assert!(
        cfg_pe.batch_size <= cfg_se.batch_size,
        "paired batch_size ({}) should be <= single-end ({})",
        cfg_pe.batch_size,
        cfg_se.batch_size
    );

    // Smaller budget should give <= batch_size
    let smaller_budget = 300 * 1024 * 1024;
    let cfg_small = rype_calculate_batch_config(idx, 150, 0, smaller_budget);
    assert!(
        cfg_small.batch_size <= cfg_se.batch_size,
        "smaller memory batch ({}) should be <= larger memory batch ({})",
        cfg_small.batch_size,
        cfg_se.batch_size
    );

    // Auto-detect memory (max_memory=0) should return a valid config
    let cfg_auto = rype_calculate_batch_config(idx, 150, 0, 0);
    assert!(
        cfg_auto.batch_size >= 1000,
        "auto-detect should return >= MIN_BATCH_SIZE, got {}",
        cfg_auto.batch_size
    );
    assert!(cfg_auto.peak_memory > 0);

    rype_index_free(idx);
    Ok(())
}
