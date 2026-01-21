//! Integration tests for the unified C API.
//!
//! These tests verify that the unified `RypeIndex` type works transparently
//! with all index formats: single-file main, sharded main, and sharded inverted indices.

use anyhow::Result;
use rype::c_api::{
    rype_bucket_name, rype_classify, rype_classify_with_negative, rype_get_last_error,
    rype_index_free, rype_index_is_sharded, rype_index_k, rype_index_load, rype_index_num_buckets,
    rype_index_num_shards, rype_index_salt, rype_index_w, rype_negative_set_create,
    rype_negative_set_free, rype_negative_set_size, rype_results_free, RypeIndex, RypeQuery,
};
use rype::{
    Index, InvertedIndex, MinimizerWorkspace, ShardFormat, ShardManifest, ShardedMainIndexBuilder,
};
use std::ffi::CString;
use std::ptr;
use tempfile::tempdir;

/// Helper to generate a DNA sequence with a deterministic pattern.
fn generate_sequence(len: usize, seed: u8) -> Vec<u8> {
    let bases = [b'A', b'C', b'G', b'T'];
    (0..len).map(|i| bases[(i + seed as usize) % 4]).collect()
}

/// Helper to create a test index with known content.
fn create_test_index(k: usize, w: usize) -> Index {
    let mut index = Index::new(k, w, 0x12345).unwrap();
    let mut ws = MinimizerWorkspace::new();

    // Add two buckets with different sequences
    let seq1 = generate_sequence(200, 0);
    let seq2 = generate_sequence(200, 1);

    index.add_record(1, "bucket1::seq1", &seq1, &mut ws);
    index.finalize_bucket(1);
    index.bucket_names.insert(1, "BucketA".to_string());

    index.add_record(2, "bucket2::seq1", &seq2, &mut ws);
    index.finalize_bucket(2);
    index.bucket_names.insert(2, "BucketB".to_string());

    index
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
// Single-file Main Index Tests
// =============================================================================

#[test]
fn test_unified_load_single_main_index() -> Result<()> {
    let dir = tempdir()?;
    let index_path = dir.path().join("test.ryidx");

    // Create and save a single-file index
    let index = create_test_index(64, 50);
    index.save(&index_path)?;

    // Load via unified API
    let path_cstr = CString::new(index_path.to_str().unwrap())?;
    let loaded = rype_index_load(path_cstr.as_ptr());

    assert!(!loaded.is_null(), "Should load single-file main index");

    // Verify accessors
    assert_eq!(rype_index_k(loaded), 64);
    assert_eq!(rype_index_w(loaded), 50);
    assert_eq!(rype_index_salt(loaded), 0x12345);
    assert_eq!(rype_index_num_buckets(loaded), 2);
    assert_eq!(rype_index_is_sharded(loaded), 0); // Not sharded
    assert_eq!(rype_index_num_shards(loaded), 1);

    // Verify bucket names
    let name_ptr = rype_bucket_name(loaded, 1);
    assert!(!name_ptr.is_null());
    let name = unsafe { std::ffi::CStr::from_ptr(name_ptr) };
    assert_eq!(name.to_str().unwrap(), "BucketA");

    rype_index_free(loaded);
    Ok(())
}

#[test]
fn test_unified_classify_single_main_index() -> Result<()> {
    let dir = tempdir()?;
    let index_path = dir.path().join("test.ryidx");

    let index = create_test_index(64, 50);
    index.save(&index_path)?;

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

    rype_results_free(results);
    rype_index_free(loaded);
    Ok(())
}

// =============================================================================
// Sharded Main Index Tests
// =============================================================================

#[test]
fn test_unified_load_sharded_main_index() -> Result<()> {
    use rype::extract_into;

    let dir = tempdir()?;
    let index_path = dir.path().join("test.ryidx");

    // Create a sharded main index
    // max_shard_bytes = 1000 to force small shards for testing
    let mut builder = ShardedMainIndexBuilder::new(64, 50, 0xABCDE, &index_path, 1000)?;
    let mut ws = MinimizerWorkspace::new();

    let seq1 = generate_sequence(200, 0);
    let seq2 = generate_sequence(200, 1);

    // Extract minimizers for each bucket
    extract_into(&seq1, 64, 50, 0xABCDE, &mut ws);
    let mut mins1: Vec<u64> = ws.buffer.drain(..).collect();
    mins1.sort();
    mins1.dedup();

    extract_into(&seq2, 64, 50, 0xABCDE, &mut ws);
    let mut mins2: Vec<u64> = ws.buffer.drain(..).collect();
    mins2.sort();
    mins2.dedup();

    builder.add_bucket(1, "BucketA", vec!["src1".to_string()], mins1)?;
    builder.add_bucket(2, "BucketB", vec!["src2".to_string()], mins2)?;
    builder.finish()?;

    // Load via unified API
    let path_cstr = CString::new(index_path.to_str().unwrap())?;
    let loaded = rype_index_load(path_cstr.as_ptr());

    assert!(!loaded.is_null(), "Should load sharded main index");

    // Verify accessors
    assert_eq!(rype_index_k(loaded), 64);
    assert_eq!(rype_index_w(loaded), 50);
    assert_eq!(rype_index_salt(loaded), 0xABCDE);
    assert_eq!(rype_index_num_buckets(loaded), 2);
    assert_eq!(rype_index_is_sharded(loaded), 1); // Is sharded
    assert!(rype_index_num_shards(loaded) >= 1);

    rype_index_free(loaded);
    Ok(())
}

#[test]
fn test_unified_classify_sharded_main_index() -> Result<()> {
    use rype::extract_into;

    let dir = tempdir()?;
    let index_path = dir.path().join("test.ryidx");

    let mut builder = ShardedMainIndexBuilder::new(64, 50, 0x12345, &index_path, 1000)?;
    let mut ws = MinimizerWorkspace::new();

    let seq1 = generate_sequence(200, 0);

    // Extract minimizers for the bucket
    extract_into(&seq1, 64, 50, 0x12345, &mut ws);
    let mut mins1: Vec<u64> = ws.buffer.drain(..).collect();
    mins1.sort();
    mins1.dedup();

    builder.add_bucket(1, "BucketA", vec!["src1".to_string()], mins1)?;
    builder.finish()?;

    let path_cstr = CString::new(index_path.to_str().unwrap())?;
    let loaded = rype_index_load(path_cstr.as_ptr());
    assert!(!loaded.is_null());

    // Query with matching sequence
    let query_seq = generate_sequence(200, 0);
    let (query, _seq_holder) = make_query(1, &query_seq);

    let results = rype_classify(loaded, &query, 1, 0.1);

    assert!(!results.is_null(), "Classification should succeed");

    let results_ref = unsafe { &*results };
    assert!(
        results_ref.len > 0,
        "Should have hits for matching sequence"
    );

    rype_results_free(results);
    rype_index_free(loaded);
    Ok(())
}

// =============================================================================
// Sharded Inverted Index Tests
// =============================================================================

#[test]
fn test_unified_load_sharded_inverted_index() -> Result<()> {
    let dir = tempdir()?;
    let main_path = dir.path().join("test.ryidx");
    let inverted_path = dir.path().join("test.ryxdi");

    // Create main index, build inverted, save as sharded
    let index = create_test_index(64, 50);
    index.save(&main_path)?;

    let inverted = InvertedIndex::build_from_index(&index);

    // Save as single shard
    let shard_path = ShardManifest::shard_path(&inverted_path, 0);
    let shard_info = inverted.save_shard(&shard_path, 0, 0, inverted.num_minimizers(), true)?;

    let manifest = ShardManifest {
        k: inverted.k,
        w: inverted.w,
        salt: inverted.salt,
        source_hash: 0,
        total_minimizers: inverted.num_minimizers(),
        total_bucket_ids: inverted.num_bucket_entries(),
        has_overlapping_shards: true,
        shard_format: ShardFormat::Legacy,
        shards: vec![shard_info],
        bucket_names: std::collections::HashMap::new(),
        bucket_sources: std::collections::HashMap::new(),
        bucket_minimizer_counts: std::collections::HashMap::new(),
    };
    let manifest_path = ShardManifest::manifest_path(&inverted_path);
    manifest.save(&manifest_path)?;

    // Load sharded inverted index via unified API
    let path_cstr = CString::new(inverted_path.to_str().unwrap())?;
    let loaded = rype_index_load(path_cstr.as_ptr());

    assert!(!loaded.is_null(), "Should load sharded inverted index");

    // Verify accessors
    assert_eq!(rype_index_k(loaded), 64);
    assert_eq!(rype_index_w(loaded), 50);
    assert_eq!(rype_index_is_sharded(loaded), 1); // Is sharded
    assert!(rype_index_num_shards(loaded) >= 1);

    rype_index_free(loaded);
    Ok(())
}

#[test]
fn test_unified_classify_sharded_inverted_index() -> Result<()> {
    let dir = tempdir()?;
    let main_path = dir.path().join("test.ryidx");
    let inverted_path = dir.path().join("test.ryxdi");

    let index = create_test_index(64, 50);
    index.save(&main_path)?;

    let inverted = InvertedIndex::build_from_index(&index);

    // Save as single shard
    let shard_path = ShardManifest::shard_path(&inverted_path, 0);
    let shard_info = inverted.save_shard(&shard_path, 0, 0, inverted.num_minimizers(), true)?;

    let manifest = ShardManifest {
        k: inverted.k,
        w: inverted.w,
        salt: inverted.salt,
        source_hash: 0,
        total_minimizers: inverted.num_minimizers(),
        total_bucket_ids: inverted.num_bucket_entries(),
        has_overlapping_shards: true,
        shard_format: ShardFormat::Legacy,
        shards: vec![shard_info],
        bucket_names: std::collections::HashMap::new(),
        bucket_sources: std::collections::HashMap::new(),
        bucket_minimizer_counts: std::collections::HashMap::new(),
    };
    let manifest_path = ShardManifest::manifest_path(&inverted_path);
    manifest.save(&manifest_path)?;

    let path_cstr = CString::new(inverted_path.to_str().unwrap())?;
    let loaded = rype_index_load(path_cstr.as_ptr());
    assert!(!loaded.is_null());

    let query_seq = generate_sequence(200, 0);
    let (query, _seq_holder) = make_query(1, &query_seq);

    let results = rype_classify(loaded, &query, 1, 0.1);

    assert!(!results.is_null(), "Classification should succeed");

    let results_ref = unsafe { &*results };
    assert!(results_ref.len > 0, "Should have hits");

    rype_results_free(results);
    rype_index_free(loaded);
    Ok(())
}

// =============================================================================
// Negative Set Tests (with unified index)
// =============================================================================

#[test]
fn test_unified_negative_set_creation() -> Result<()> {
    let dir = tempdir()?;
    let index_path = dir.path().join("test.ryidx");

    let index = create_test_index(64, 50);
    index.save(&index_path)?;

    let path_cstr = CString::new(index_path.to_str().unwrap())?;
    let loaded = rype_index_load(path_cstr.as_ptr());
    assert!(!loaded.is_null());

    // Create negative set from unified index
    let neg_set = rype_negative_set_create(loaded);
    assert!(!neg_set.is_null(), "Should create negative set");

    let size = rype_negative_set_size(neg_set);
    assert!(size > 0, "Negative set should have minimizers");

    rype_negative_set_free(neg_set);
    rype_index_free(loaded);
    Ok(())
}

#[test]
fn test_unified_classify_with_negative_filtering() -> Result<()> {
    let dir = tempdir()?;
    let pos_path = dir.path().join("positive.ryidx");
    let neg_path = dir.path().join("negative.ryidx");

    // Create positive index
    let mut pos_index = Index::new(64, 50, 0x12345).unwrap();
    let mut ws = MinimizerWorkspace::new();
    let seq1 = generate_sequence(200, 0);
    pos_index.add_record(1, "pos::seq1", &seq1, &mut ws);
    pos_index.finalize_bucket(1);
    pos_index.bucket_names.insert(1, "Positive".to_string());
    pos_index.save(&pos_path)?;

    // Create negative index with same sequence (to filter it out)
    let mut neg_index = Index::new(64, 50, 0x12345).unwrap();
    neg_index.add_record(1, "neg::seq1", &seq1, &mut ws);
    neg_index.finalize_bucket(1);
    neg_index.bucket_names.insert(1, "Negative".to_string());
    neg_index.save(&neg_path)?;

    // Load both via unified API
    let pos_cstr = CString::new(pos_path.to_str().unwrap())?;
    let neg_cstr = CString::new(neg_path.to_str().unwrap())?;

    let pos_loaded = rype_index_load(pos_cstr.as_ptr());
    let neg_loaded = rype_index_load(neg_cstr.as_ptr());

    assert!(!pos_loaded.is_null());
    assert!(!neg_loaded.is_null());

    // Create negative set
    let neg_set = rype_negative_set_create(neg_loaded);
    assert!(!neg_set.is_null());

    // Query with the sequence that's in both indices
    let (query, _seq_holder) = make_query(1, &seq1);

    // Classify without negative filtering - should have hits
    let results_no_neg = rype_classify(pos_loaded, &query, 1, 0.1);
    assert!(!results_no_neg.is_null());
    let no_neg_len = unsafe { (*results_no_neg).len };

    // Classify with negative filtering - should have fewer/no hits
    let results_with_neg = rype_classify_with_negative(pos_loaded, neg_set, &query, 1, 0.5);
    assert!(!results_with_neg.is_null());
    let with_neg_len = unsafe { (*results_with_neg).len };

    // With full negative filtering at high threshold, should filter out matches
    assert!(
        with_neg_len <= no_neg_len,
        "Negative filtering should reduce or eliminate hits"
    );

    rype_results_free(results_no_neg);
    rype_results_free(results_with_neg);
    rype_negative_set_free(neg_set);
    rype_index_free(pos_loaded);
    rype_index_free(neg_loaded);
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
    let path = CString::new("/nonexistent/path/index.ryidx").unwrap();
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

// =============================================================================
// Cross-format Consistency Tests
// =============================================================================

#[test]
fn test_unified_results_consistent_across_formats() -> Result<()> {
    let dir = tempdir()?;

    // Create base index
    let index = create_test_index(64, 50);

    // Save in different formats
    let single_main_path = dir.path().join("single.ryidx");
    let sharded_inv_path = dir.path().join("sharded.ryxdi");

    index.save(&single_main_path)?;

    let inverted = InvertedIndex::build_from_index(&index);

    // Save inverted index as single shard
    let shard_path = ShardManifest::shard_path(&sharded_inv_path, 0);
    let shard_info = inverted.save_shard(&shard_path, 0, 0, inverted.num_minimizers(), true)?;

    let manifest = ShardManifest {
        k: inverted.k,
        w: inverted.w,
        salt: inverted.salt,
        source_hash: 0,
        total_minimizers: inverted.num_minimizers(),
        total_bucket_ids: inverted.num_bucket_entries(),
        has_overlapping_shards: true,
        shard_format: ShardFormat::Legacy,
        shards: vec![shard_info],
        bucket_names: std::collections::HashMap::new(),
        bucket_sources: std::collections::HashMap::new(),
        bucket_minimizer_counts: std::collections::HashMap::new(),
    };
    let manifest_path = ShardManifest::manifest_path(&sharded_inv_path);
    manifest.save(&manifest_path)?;

    // Load both via unified API
    let paths = [
        CString::new(single_main_path.to_str().unwrap())?,
        CString::new(sharded_inv_path.to_str().unwrap())?,
    ];

    let indices: Vec<*mut RypeIndex> = paths.iter().map(|p| rype_index_load(p.as_ptr())).collect();

    for idx in &indices {
        assert!(!idx.is_null());
    }

    // All should have same k, w, salt
    for idx in &indices {
        assert_eq!(rype_index_k(*idx), 64);
        assert_eq!(rype_index_w(*idx), 50);
        assert_eq!(rype_index_salt(*idx), 0x12345);
    }

    // Classify same query against all - results should be equivalent
    let query_seq = generate_sequence(200, 0);
    let (query, _seq_holder) = make_query(1, &query_seq);

    let mut hit_counts = Vec::new();
    for idx in &indices {
        let results = rype_classify(*idx, &query, 1, 0.1);
        assert!(!results.is_null());
        hit_counts.push(unsafe { (*results).len });
        rype_results_free(results);
    }

    // All formats should produce the same number of hits
    assert!(
        hit_counts.iter().all(|&c| c == hit_counts[0]),
        "All index formats should produce consistent results: {:?}",
        hit_counts
    );

    for idx in indices {
        rype_index_free(idx);
    }
    Ok(())
}
