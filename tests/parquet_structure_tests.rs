//! Tests verifying the Parquet inverted index data structure.
//!
//! These tests go beyond classification correctness to verify:
//! - Parquet schema is correct
//! - Data is sorted by (minimizer, bucket_id)
//! - No duplicate (minimizer, bucket_id) pairs
//! - Manifest counts match actual data

use anyhow::Result;
use arrow::array::{Array, UInt32Array, UInt64Array};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::HashSet;
use std::fs::File;
use std::path::Path;
use tempfile::tempdir;

use rype::parquet_index::{
    compute_source_hash, create_parquet_inverted_index, is_parquet_index, BucketData,
    ParquetManifest,
};

/// Create test bucket data with known minimizer values.
fn create_test_buckets() -> Vec<BucketData> {
    vec![
        BucketData {
            bucket_id: 0,
            bucket_name: "bacteria".to_string(),
            sources: vec!["ecoli.fna".to_string(), "bsubtilis.fna".to_string()],
            minimizers: vec![100, 200, 300, 500, 700], // sorted, unique
        },
        BucketData {
            bucket_id: 1,
            bucket_name: "archaea".to_string(),
            sources: vec!["methanobacterium.fna".to_string()],
            minimizers: vec![150, 200, 400, 600], // sorted, unique; 200 shared with bucket 0
        },
        BucketData {
            bucket_id: 2,
            bucket_name: "fungi".to_string(),
            sources: vec!["yeast.fna".to_string()],
            minimizers: vec![50, 300, 800, 900], // sorted, unique; 300 shared with bucket 0
        },
    ]
}

/// Read all (minimizer, bucket_id) pairs from a Parquet shard file.
fn read_shard_pairs(path: &Path) -> Result<Vec<(u64, u32)>> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut pairs = Vec::new();
    for batch_result in reader {
        let batch = batch_result?;

        let minimizers = batch
            .column_by_name("minimizer")
            .expect("minimizer column missing")
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("minimizer should be UInt64");

        let bucket_ids = batch
            .column_by_name("bucket_id")
            .expect("bucket_id column missing")
            .as_any()
            .downcast_ref::<UInt32Array>()
            .expect("bucket_id should be UInt32");

        for i in 0..batch.num_rows() {
            pairs.push((minimizers.value(i), bucket_ids.value(i)));
        }
    }

    Ok(pairs)
}

#[test]
fn test_parquet_schema_is_correct() -> Result<()> {
    let dir = tempdir()?;
    let index_path = dir.path().join("test.ryxdi");

    let buckets = create_test_buckets();
    create_parquet_inverted_index(&index_path, buckets, 64, 50, 12345, None, None)?;

    // Read the shard and verify schema
    let shard_path = index_path.join("inverted").join("shard.0.parquet");
    let file = File::open(&shard_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema();

    assert_eq!(schema.fields().len(), 2, "Should have exactly 2 columns");

    let minimizer_field = schema.field_with_name("minimizer")?;
    assert_eq!(
        minimizer_field.data_type(),
        &arrow::datatypes::DataType::UInt64,
        "minimizer should be UInt64"
    );
    assert!(
        !minimizer_field.is_nullable(),
        "minimizer should not be nullable"
    );

    let bucket_id_field = schema.field_with_name("bucket_id")?;
    assert_eq!(
        bucket_id_field.data_type(),
        &arrow::datatypes::DataType::UInt32,
        "bucket_id should be UInt32"
    );
    assert!(
        !bucket_id_field.is_nullable(),
        "bucket_id should not be nullable"
    );

    Ok(())
}

#[test]
fn test_parquet_data_is_sorted() -> Result<()> {
    let dir = tempdir()?;
    let index_path = dir.path().join("test.ryxdi");

    let buckets = create_test_buckets();
    create_parquet_inverted_index(&index_path, buckets, 64, 50, 12345, None, None)?;

    // Read all pairs from the shard
    let shard_path = index_path.join("inverted").join("shard.0.parquet");
    let pairs = read_shard_pairs(&shard_path)?;

    // Verify sorted by (minimizer, bucket_id)
    for i in 1..pairs.len() {
        let prev = pairs[i - 1];
        let curr = pairs[i];
        assert!(
            (curr.0, curr.1) >= (prev.0, prev.1),
            "Data not sorted at index {}: {:?} should come after {:?}",
            i,
            curr,
            prev
        );
    }

    Ok(())
}

#[test]
fn test_parquet_no_duplicate_pairs() -> Result<()> {
    let dir = tempdir()?;
    let index_path = dir.path().join("test.ryxdi");

    let buckets = create_test_buckets();
    create_parquet_inverted_index(&index_path, buckets, 64, 50, 12345, None, None)?;

    // Read all pairs from the shard
    let shard_path = index_path.join("inverted").join("shard.0.parquet");
    let pairs = read_shard_pairs(&shard_path)?;

    // Check for duplicates using a HashSet
    let mut seen: HashSet<(u64, u32)> = HashSet::with_capacity(pairs.len());
    for pair in &pairs {
        assert!(
            seen.insert(*pair),
            "Duplicate (minimizer, bucket_id) pair found: {:?}",
            pair
        );
    }

    Ok(())
}

#[test]
fn test_parquet_manifest_counts_match_data() -> Result<()> {
    let dir = tempdir()?;
    let index_path = dir.path().join("test.ryxdi");

    let buckets = create_test_buckets();
    let expected_total_minimizers: u64 = buckets.iter().map(|b| b.minimizers.len() as u64).sum();
    // Total entries = sum of all minimizers (each becomes one row)
    let expected_total_entries = expected_total_minimizers;

    let manifest = create_parquet_inverted_index(&index_path, buckets, 64, 50, 12345, None, None)?;

    // Verify manifest totals
    assert_eq!(
        manifest.total_minimizers, expected_total_minimizers,
        "total_minimizers mismatch"
    );

    let inverted = manifest.inverted.expect("Should have inverted manifest");
    assert_eq!(
        inverted.total_entries, expected_total_entries,
        "total_entries mismatch"
    );

    // Verify by reading actual data
    let shard_path = index_path.join("inverted").join("shard.0.parquet");
    let pairs = read_shard_pairs(&shard_path)?;
    assert_eq!(
        pairs.len() as u64,
        expected_total_entries,
        "Actual row count doesn't match manifest"
    );

    Ok(())
}

#[test]
fn test_parquet_shared_minimizers_have_multiple_bucket_ids() -> Result<()> {
    let dir = tempdir()?;
    let index_path = dir.path().join("test.ryxdi");

    let buckets = create_test_buckets();
    create_parquet_inverted_index(&index_path, buckets, 64, 50, 12345, None, None)?;

    // Read all pairs from the shard
    let shard_path = index_path.join("inverted").join("shard.0.parquet");
    let pairs = read_shard_pairs(&shard_path)?;

    // Check that minimizer 200 has bucket_ids 0 and 1
    let bucket_ids_for_200: Vec<u32> = pairs
        .iter()
        .filter(|(m, _)| *m == 200)
        .map(|(_, b)| *b)
        .collect();
    assert_eq!(
        bucket_ids_for_200.len(),
        2,
        "Minimizer 200 should have 2 bucket entries"
    );
    assert!(
        bucket_ids_for_200.contains(&0),
        "Minimizer 200 should map to bucket 0"
    );
    assert!(
        bucket_ids_for_200.contains(&1),
        "Minimizer 200 should map to bucket 1"
    );

    // Check that minimizer 300 has bucket_ids 0 and 2
    let bucket_ids_for_300: Vec<u32> = pairs
        .iter()
        .filter(|(m, _)| *m == 300)
        .map(|(_, b)| *b)
        .collect();
    assert_eq!(
        bucket_ids_for_300.len(),
        2,
        "Minimizer 300 should have 2 bucket entries"
    );
    assert!(
        bucket_ids_for_300.contains(&0),
        "Minimizer 300 should map to bucket 0"
    );
    assert!(
        bucket_ids_for_300.contains(&2),
        "Minimizer 300 should map to bucket 2"
    );

    Ok(())
}

#[test]
fn test_parquet_manifest_shard_ranges() -> Result<()> {
    let dir = tempdir()?;
    let index_path = dir.path().join("test.ryxdi");

    let buckets = create_test_buckets();
    create_parquet_inverted_index(&index_path, buckets, 64, 50, 12345, None, None)?;

    // Load manifest and verify shard ranges
    let manifest = ParquetManifest::load(&index_path)?;
    let inverted = manifest.inverted.expect("Should have inverted manifest");

    // With small data, should be single shard
    assert_eq!(inverted.num_shards, 1, "Should have 1 shard for small data");
    assert_eq!(inverted.shards.len(), 1, "Shard list should have 1 entry");

    let shard_info = &inverted.shards[0];
    assert_eq!(shard_info.shard_id, 0);
    assert_eq!(shard_info.min_minimizer, 50, "Min minimizer should be 50");
    assert_eq!(shard_info.max_minimizer, 900, "Max minimizer should be 900");

    Ok(())
}

#[test]
fn test_parquet_directory_structure() -> Result<()> {
    let dir = tempdir()?;
    let index_path = dir.path().join("test.ryxdi");

    let buckets = create_test_buckets();
    create_parquet_inverted_index(&index_path, buckets, 64, 50, 12345, None, None)?;

    // Verify directory structure
    assert!(index_path.is_dir(), "Index should be a directory");
    assert!(
        index_path.join("manifest.toml").is_file(),
        "manifest.toml should exist"
    );
    assert!(
        index_path.join("buckets.parquet").is_file(),
        "buckets.parquet should exist"
    );
    assert!(
        index_path.join("inverted").is_dir(),
        "inverted/ directory should exist"
    );
    assert!(
        index_path
            .join("inverted")
            .join("shard.0.parquet")
            .is_file(),
        "shard.0.parquet should exist"
    );

    Ok(())
}

#[test]
fn test_is_parquet_index_detection() -> Result<()> {
    let dir = tempdir()?;
    let index_path = dir.path().join("test.ryxdi");

    let buckets = create_test_buckets();
    create_parquet_inverted_index(&index_path, buckets, 64, 50, 12345, None, None)?;

    // Should be detected as a Parquet index
    assert!(is_parquet_index(&index_path), "Should detect Parquet index");

    // Non-existent path should return false
    assert!(
        !is_parquet_index(&dir.path().join("nonexistent")),
        "Non-existent path should not be Parquet index"
    );

    Ok(())
}

#[test]
fn test_parquet_empty_buckets_handled() -> Result<()> {
    let dir = tempdir()?;
    let index_path = dir.path().join("test.ryxdi");

    // All empty buckets
    let buckets = vec![
        BucketData {
            bucket_id: 0,
            bucket_name: "empty1".to_string(),
            sources: vec![],
            minimizers: vec![],
        },
        BucketData {
            bucket_id: 1,
            bucket_name: "empty2".to_string(),
            sources: vec![],
            minimizers: vec![],
        },
    ];

    let manifest = create_parquet_inverted_index(&index_path, buckets, 64, 50, 12345, None, None)?;

    assert_eq!(manifest.total_minimizers, 0);
    let inverted = manifest.inverted.expect("Should have inverted manifest");
    assert_eq!(inverted.total_entries, 0);

    Ok(())
}

#[test]
fn test_parquet_bucket_metadata_preserved() -> Result<()> {
    let dir = tempdir()?;
    let index_path = dir.path().join("test.ryxdi");

    let buckets = create_test_buckets();
    create_parquet_inverted_index(&index_path, buckets, 64, 50, 12345, None, None)?;

    // Load manifest and verify bucket count
    let manifest = ParquetManifest::load(&index_path)?;
    assert_eq!(manifest.num_buckets, 3, "Should have 3 buckets in manifest");

    // Verify parameters
    assert_eq!(manifest.k, 64);
    assert_eq!(manifest.w, 50);
    assert_eq!(manifest.salt, 12345);

    // Verify source_hash is non-zero (computed from bucket minimizer counts)
    assert_ne!(manifest.source_hash, 0, "source_hash should be computed");

    Ok(())
}

#[test]
fn test_parquet_validation_rejects_unsorted_buckets() {
    let dir = tempdir().unwrap();
    let index_path = dir.path().join("test.ryxdi");

    // Create bucket with unsorted minimizers
    let buckets = vec![BucketData {
        bucket_id: 0,
        bucket_name: "bad".to_string(),
        sources: vec![],
        minimizers: vec![300, 100, 200], // NOT sorted!
    }];

    let result = create_parquet_inverted_index(&index_path, buckets, 64, 50, 12345, None, None);
    assert!(result.is_err(), "Should reject unsorted bucket data");
    // The error chain includes context - check the full debug output
    let err = format!("{:?}", result.unwrap_err());
    assert!(
        err.contains("unsorted"),
        "Error should mention sorting issue: {}",
        err
    );
}

#[test]
fn test_parquet_validation_rejects_duplicate_minimizers() {
    let dir = tempdir().unwrap();
    let index_path = dir.path().join("test.ryxdi");

    // Create bucket with duplicate minimizers
    let buckets = vec![BucketData {
        bucket_id: 0,
        bucket_name: "bad".to_string(),
        sources: vec![],
        minimizers: vec![100, 200, 200, 300], // Duplicate 200!
    }];

    let result = create_parquet_inverted_index(&index_path, buckets, 64, 50, 12345, None, None);
    assert!(
        result.is_err(),
        "Should reject bucket with duplicate minimizers"
    );
    // The error chain includes context - check the full debug output
    let err = format!("{:?}", result.unwrap_err());
    assert!(
        err.contains("duplicate"),
        "Error should mention duplicate issue: {}",
        err
    );
}

#[test]
fn test_source_hash_validation() -> Result<()> {
    use std::collections::HashMap;

    let dir = tempdir()?;
    let index_path = dir.path().join("test.ryxdi");

    let buckets = create_test_buckets();

    // Compute expected source_hash from bucket minimizer counts
    let mut bucket_minimizer_counts: HashMap<u32, usize> = HashMap::new();
    for bucket in &buckets {
        bucket_minimizer_counts.insert(bucket.bucket_id, bucket.minimizers.len());
    }
    let expected_hash = compute_source_hash(&bucket_minimizer_counts);

    // Create index
    let manifest = create_parquet_inverted_index(&index_path, buckets, 64, 50, 12345, None, None)?;

    // Verify the manifest's source_hash matches our expected value
    assert_eq!(
        manifest.source_hash, expected_hash,
        "Manifest source_hash should match computed value"
    );

    // Reload manifest and verify hash is preserved
    let reloaded = ParquetManifest::load(&index_path)?;
    assert_eq!(
        reloaded.source_hash, expected_hash,
        "Reloaded manifest source_hash should match original"
    );

    Ok(())
}

#[test]
fn test_source_hash_detects_different_counts() {
    use std::collections::HashMap;

    // Two different bucket configurations should have different source_hashes
    let mut counts1: HashMap<u32, usize> = HashMap::new();
    counts1.insert(0, 100);
    counts1.insert(1, 200);

    let mut counts2: HashMap<u32, usize> = HashMap::new();
    counts2.insert(0, 100);
    counts2.insert(1, 201); // Different count

    let hash1 = compute_source_hash(&counts1);
    let hash2 = compute_source_hash(&counts2);

    assert_ne!(
        hash1, hash2,
        "Different bucket counts should produce different hashes"
    );
}

#[test]
fn test_source_hash_stable_across_order() {
    use std::collections::HashMap;

    // Hash should be the same regardless of insertion order
    let mut counts1: HashMap<u32, usize> = HashMap::new();
    counts1.insert(0, 100);
    counts1.insert(1, 200);
    counts1.insert(2, 300);

    let mut counts2: HashMap<u32, usize> = HashMap::new();
    counts2.insert(2, 300);
    counts2.insert(0, 100);
    counts2.insert(1, 200);

    let hash1 = compute_source_hash(&counts1);
    let hash2 = compute_source_hash(&counts2);

    assert_eq!(
        hash1, hash2,
        "Hash should be stable regardless of insertion order"
    );
}

// ============================================================================
// Tests for critical bug fixes
// ============================================================================

/// Test that sampling is deterministic: same minimizers produce same output.
/// This tests fix #1 where rand_sample() was using thread ID instead of
/// minimizer value for the random decision.
#[test]
fn test_sampling_is_deterministic() -> Result<()> {
    let dir1 = tempdir()?;
    let dir2 = tempdir()?;
    let index_path1 = dir1.path().join("test1.ryxdi");
    let index_path2 = dir2.path().join("test2.ryxdi");

    // Create buckets with many minimizers (enough to trigger sampling)
    let minimizers: Vec<u64> = (0..10_000u64).collect();
    let buckets = vec![BucketData {
        bucket_id: 0,
        bucket_name: "test".to_string(),
        sources: vec!["test.fna".to_string()],
        minimizers: minimizers.clone(),
    }];

    // Create two indices with same data
    create_parquet_inverted_index(&index_path1, buckets.clone(), 64, 50, 12345, None, None)?;
    create_parquet_inverted_index(&index_path2, buckets, 64, 50, 12345, None, None)?;

    // Read pairs from both shards
    let shard_path1 = index_path1.join("inverted").join("shard.0.parquet");
    let shard_path2 = index_path2.join("inverted").join("shard.0.parquet");

    let pairs1 = read_shard_pairs(&shard_path1)?;
    let pairs2 = read_shard_pairs(&shard_path2)?;

    // Verify they're identical (deterministic sampling)
    assert_eq!(
        pairs1.len(),
        pairs2.len(),
        "Same input should produce same number of entries"
    );
    assert_eq!(
        pairs1, pairs2,
        "Same input should produce identical output (deterministic sampling)"
    );

    Ok(())
}

/// Test that parallel shard creation produces correctly named files.
/// This tests fix #2 where parallel partitions could have file naming collisions.
#[test]
fn test_parallel_shard_files_correctly_named() -> Result<()> {
    use rype::parquet_index::ParquetWriteOptions;

    let dir = tempdir()?;
    let index_path = dir.path().join("test.ryxdi");

    // Create large dataset to trigger parallel processing
    // Each bucket has minimizers spanning different ranges
    let mut buckets = Vec::new();
    for bucket_id in 0..10u32 {
        let base = bucket_id as u64 * 10_000;
        let minimizers: Vec<u64> = (base..base + 5_000).collect();
        buckets.push(BucketData {
            bucket_id,
            bucket_name: format!("bucket_{}", bucket_id),
            sources: vec![format!("ref_{}.fna", bucket_id)],
            minimizers,
        });
    }

    // Use small shard size to create multiple shards
    let options = ParquetWriteOptions {
        row_group_size: 1000,
        ..Default::default()
    };
    create_parquet_inverted_index(&index_path, buckets, 64, 50, 12345, None, Some(&options))?;

    // Load manifest to get shard count
    let manifest = ParquetManifest::load(&index_path)?;
    let inverted = manifest.inverted.expect("Should have inverted manifest");

    // Verify each shard file exists with correct naming
    let inverted_dir = index_path.join("inverted");
    for shard_id in 0..inverted.num_shards {
        let shard_path = inverted_dir.join(format!("shard.{}.parquet", shard_id));
        assert!(
            shard_path.exists(),
            "Shard file {} should exist",
            shard_path.display()
        );
    }

    // Verify shard ranges are in ascending order
    for i in 1..inverted.shards.len() {
        assert!(
            inverted.shards[i].min_minimizer >= inverted.shards[i - 1].max_minimizer,
            "Shard {} should start after shard {} ends",
            i,
            i - 1
        );
    }

    // Verify no leftover partition files (p0_*, p1_*, etc.)
    let entries: Vec<_> = std::fs::read_dir(&inverted_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    for entry in &entries {
        assert!(
            entry.starts_with("shard."),
            "Unexpected file in inverted dir: {}",
            entry
        );
    }

    Ok(())
}

/// Test that parallel processing produces correctly ordered data.
/// Verifies the final output is globally sorted by minimizer.
#[test]
fn test_parallel_output_globally_sorted() -> Result<()> {
    use rype::parquet_index::ParquetWriteOptions;

    let dir = tempdir()?;
    let index_path = dir.path().join("test.ryxdi");

    // Create overlapping minimizer ranges to stress parallel merging
    let buckets = vec![
        BucketData {
            bucket_id: 0,
            bucket_name: "a".to_string(),
            sources: vec![],
            minimizers: (0..5000u64).collect(),
        },
        BucketData {
            bucket_id: 1,
            bucket_name: "b".to_string(),
            sources: vec![],
            minimizers: (2500..7500u64).collect(), // Overlaps with bucket 0
        },
        BucketData {
            bucket_id: 2,
            bucket_name: "c".to_string(),
            sources: vec![],
            minimizers: (5000..10000u64).collect(),
        },
    ];

    let options = ParquetWriteOptions {
        row_group_size: 1000,
        ..Default::default()
    };
    create_parquet_inverted_index(&index_path, buckets, 64, 50, 12345, None, Some(&options))?;

    // Read all pairs from all shards
    let manifest = ParquetManifest::load(&index_path)?;
    let inverted = manifest.inverted.expect("Should have inverted manifest");
    let inverted_dir = index_path.join("inverted");

    let mut all_pairs = Vec::new();
    for shard_id in 0..inverted.num_shards {
        let shard_path = inverted_dir.join(format!("shard.{}.parquet", shard_id));
        let pairs = read_shard_pairs(&shard_path)?;
        all_pairs.extend(pairs);
    }

    // Verify global sort order: (minimizer, bucket_id)
    for i in 1..all_pairs.len() {
        let (m1, b1) = all_pairs[i - 1];
        let (m2, b2) = all_pairs[i];
        assert!(
            (m2, b2) >= (m1, b1),
            "Pairs should be sorted: ({}, {}) should come after ({}, {})",
            m2,
            b2,
            m1,
            b1
        );
    }

    Ok(())
}

// ============================================================================
// Tests for ParquetWriteOptions validation
// ============================================================================

#[test]
fn test_fpp_validation_rejects_zero() {
    use rype::parquet_index::ParquetWriteOptions;

    let options = ParquetWriteOptions {
        bloom_filter_fpp: 0.0,
        ..Default::default()
    };
    let result = options.validate();
    assert!(result.is_err(), "FPP of 0.0 should be rejected");
    let err = result.unwrap_err().to_string();
    assert!(err.contains("bloom_filter_fpp"), "Error should mention fpp");
}

#[test]
fn test_fpp_validation_rejects_one() {
    use rype::parquet_index::ParquetWriteOptions;

    let options = ParquetWriteOptions {
        bloom_filter_fpp: 1.0,
        ..Default::default()
    };
    let result = options.validate();
    assert!(result.is_err(), "FPP of 1.0 should be rejected");
}

#[test]
fn test_fpp_validation_rejects_negative() {
    use rype::parquet_index::ParquetWriteOptions;

    let options = ParquetWriteOptions {
        bloom_filter_fpp: -0.5,
        ..Default::default()
    };
    let result = options.validate();
    assert!(result.is_err(), "Negative FPP should be rejected");
}

#[test]
fn test_fpp_validation_accepts_valid() {
    use rype::parquet_index::ParquetWriteOptions;

    let options = ParquetWriteOptions {
        bloom_filter_fpp: 0.05,
        ..Default::default()
    };
    let result = options.validate();
    assert!(result.is_ok(), "FPP of 0.05 should be valid");
}

#[test]
fn test_row_group_validation_rejects_zero() {
    use rype::parquet_index::ParquetWriteOptions;

    let options = ParquetWriteOptions {
        row_group_size: 0,
        ..Default::default()
    };
    let result = options.validate();
    assert!(result.is_err(), "row_group_size of 0 should be rejected");
}
