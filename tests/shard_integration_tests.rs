//! Integration tests for sharded Parquet index classification.
//!
//! These tests verify that different classification methods
//! produce consistent results with Parquet indices.

use std::collections::HashMap;

use anyhow::Result;
use tempfile::tempdir;

use rype::{
    classify_batch_sharded_merge_join, extract_into, BucketData, HitResult, IndexMetadata,
    InvertedIndex, MinimizerWorkspace, ParquetWriteOptions, QueryRecord, ShardedInvertedIndex,
};

/// Test helper to sort results for comparison
fn sort_results(mut results: Vec<HitResult>) -> Vec<HitResult> {
    results.sort_by(|a, b| (a.query_id, a.bucket_id).cmp(&(b.query_id, b.bucket_id)));
    results
}

/// Test helper to compare two result sets
fn compare_results(name_a: &str, results_a: &[HitResult], name_b: &str, results_b: &[HitResult]) {
    let a = sort_results(results_a.to_vec());
    let b = sort_results(results_b.to_vec());

    if a.len() != b.len() {
        eprintln!("\n=== RESULT COUNT MISMATCH ===");
        eprintln!("{}: {} results", name_a, a.len());
        eprintln!("{}: {} results", name_b, b.len());

        // Show what's different
        for r in &a {
            if !b
                .iter()
                .any(|br| br.query_id == r.query_id && br.bucket_id == r.bucket_id)
            {
                eprintln!(
                    "  In {} only: query={}, bucket={}, score={:.4}",
                    name_a, r.query_id, r.bucket_id, r.score
                );
            }
        }
        for r in &b {
            if !a
                .iter()
                .any(|ar| ar.query_id == r.query_id && ar.bucket_id == r.bucket_id)
            {
                eprintln!(
                    "  In {} only: query={}, bucket={}, score={:.4}",
                    name_b, r.query_id, r.bucket_id, r.score
                );
            }
        }
    }

    assert_eq!(
        a.len(),
        b.len(),
        "{} has {} results, {} has {} results",
        name_a,
        a.len(),
        name_b,
        b.len()
    );

    for (ra, rb) in a.iter().zip(b.iter()) {
        assert_eq!(ra.query_id, rb.query_id, "Query ID mismatch");
        assert_eq!(ra.bucket_id, rb.bucket_id, "Bucket ID mismatch");
        let diff = (ra.score - rb.score).abs();
        assert!(
            diff < 0.001,
            "Score mismatch for query {} bucket {}: {} ({}) vs {} ({}) diff={}",
            ra.query_id,
            ra.bucket_id,
            ra.score,
            name_a,
            rb.score,
            name_b,
            diff
        );
    }
}

/// Create test sequences that produce distinct minimizer sets
fn create_test_sequences() -> Vec<Vec<u8>> {
    // Create sequences with different patterns that will hash to different minimizers
    vec![
        // Sequence 0: ACGT repeat
        (0..200)
            .map(|i| match i % 4 {
                0 => b'A',
                1 => b'C',
                2 => b'G',
                _ => b'T',
            })
            .collect(),
        // Sequence 1: AT repeat
        (0..200)
            .map(|i| if i % 2 == 0 { b'A' } else { b'T' })
            .collect(),
        // Sequence 2: GC repeat
        (0..200)
            .map(|i| if i % 2 == 0 { b'G' } else { b'C' })
            .collect(),
        // Sequence 3: Different pattern
        (0..200)
            .map(|i| match i % 6 {
                0 => b'A',
                1 => b'A',
                2 => b'C',
                3 => b'G',
                4 => b'T',
                _ => b'T',
            })
            .collect(),
        // Sequence 4: Yet another pattern
        (0..200)
            .map(|i| match i % 3 {
                0 => b'A',
                1 => b'C',
                _ => b'T',
            })
            .collect(),
    ]
}

/// Create bucket data from test sequences
fn create_bucket_data(seqs: &[Vec<u8>], k: usize, w: usize, salt: u64) -> Vec<BucketData> {
    let mut ws = MinimizerWorkspace::new();
    let mut buckets = Vec::new();

    for (i, seq) in seqs.iter().enumerate() {
        extract_into(seq, k, w, salt, &mut ws);
        let mut mins: Vec<u64> = ws.buffer.drain(..).collect();
        mins.sort();
        mins.dedup();

        buckets.push(BucketData {
            bucket_id: (i + 1) as u32,
            bucket_name: format!("Bucket{}", i + 1),
            sources: vec![format!("seq{}", i)],
            minimizers: mins,
        });
    }

    buckets
}

/// Create query records from sequences
fn create_query_records(seqs: &[Vec<u8>]) -> Vec<QueryRecord<'_>> {
    seqs.iter()
        .enumerate()
        .map(|(i, seq)| (i as i64, seq.as_slice(), None))
        .collect()
}

/// Build an InvertedIndex directly from bucket data for testing
fn build_inverted_from_buckets(
    buckets: &[BucketData],
    k: usize,
    w: usize,
    salt: u64,
) -> InvertedIndex {
    let mut bucket_map: HashMap<u32, Vec<u64>> = HashMap::new();
    let mut bucket_names: HashMap<u32, String> = HashMap::new();
    let mut bucket_sources: HashMap<u32, Vec<String>> = HashMap::new();
    let mut bucket_minimizer_counts: HashMap<u32, usize> = HashMap::new();

    for bucket in buckets {
        bucket_map.insert(bucket.bucket_id, bucket.minimizers.clone());
        bucket_names.insert(bucket.bucket_id, bucket.bucket_name.clone());
        bucket_sources.insert(bucket.bucket_id, bucket.sources.clone());
        bucket_minimizer_counts.insert(bucket.bucket_id, bucket.minimizers.len());
    }

    let metadata = IndexMetadata {
        k,
        w,
        salt,
        bucket_names,
        bucket_sources,
        bucket_minimizer_counts,
        largest_shard_entries: 0,
        bucket_file_stats: None,
    };

    InvertedIndex::build_from_bucket_map(k, w, salt, &bucket_map, &metadata)
}

#[test]
fn test_parquet_index_classification() -> Result<()> {
    eprintln!("\n\n========== PARQUET INDEX CLASSIFICATION ==========");

    let dir = tempdir()?;
    let index_path = dir.path().join("test.ryxdi");

    let seqs = create_test_sequences();
    let k = 32;
    let w = 10;
    let salt = 0x1234u64;

    let buckets = create_bucket_data(&seqs, k, w, salt);
    let records = create_query_records(&seqs);

    eprintln!("\nCreated {} buckets:", buckets.len());
    for b in &buckets {
        eprintln!(
            "  Bucket {}: {} minimizers",
            b.bucket_id,
            b.minimizers.len()
        );
    }

    // Create Parquet index
    let options = ParquetWriteOptions::default();
    rype::create_parquet_inverted_index(
        &index_path,
        buckets.clone(),
        k,
        w,
        salt,
        None,
        Some(&options),
        None,
    )?;

    // Load and classify
    let sharded = ShardedInvertedIndex::open(&index_path)?;
    eprintln!(
        "\nLoaded Parquet index: {} shards",
        sharded.manifest().shards.len()
    );

    let threshold = 0.1;

    let results = classify_batch_sharded_merge_join(&sharded, None, &records, threshold, None)?;

    eprintln!("\n=== Results ===");
    eprintln!("Merge-join: {} results", results.len());

    // Verify results - each query should strongly match its corresponding bucket (self-match)
    // Query i comes from sequence i, which was used to build bucket (i+1)
    assert!(!results.is_empty(), "Should have classification results");

    // Verify self-matches exist with high scores
    for i in 0..seqs.len() {
        let bucket_id = (i + 1) as u32;
        let self_match = results
            .iter()
            .find(|r| r.query_id == i as i64 && r.bucket_id == bucket_id);
        assert!(
            self_match.is_some(),
            "Expected self-match for query {} in bucket {}",
            i,
            bucket_id
        );
        assert!(
            self_match.unwrap().score > 0.9,
            "Self-match score should be high (>0.9), got {:.4} for query {} bucket {}",
            self_match.unwrap().score,
            i,
            bucket_id
        );
    }

    // Print detailed results
    eprintln!("\nDetailed results:");
    for r in &results {
        eprintln!(
            "  Query {} -> Bucket {}: {:.4}",
            r.query_id, r.bucket_id, r.score
        );
    }

    Ok(())
}

#[test]
fn test_parquet_index_matches_inverted_index() -> Result<()> {
    eprintln!("\n\n========== PARQUET VS DIRECT INVERTED INDEX ==========");

    let dir = tempdir()?;
    let index_path = dir.path().join("test.ryxdi");

    let seqs = create_test_sequences();
    let k = 32;
    let w = 10;
    let salt = 0x1234u64;

    let buckets = create_bucket_data(&seqs, k, w, salt);
    let records = create_query_records(&seqs);

    // Build direct inverted index for comparison
    let inverted_direct = build_inverted_from_buckets(&buckets, k, w, salt);
    eprintln!(
        "\nDirect InvertedIndex: {} minimizers, {} bucket entries",
        inverted_direct.num_minimizers(),
        inverted_direct.num_bucket_entries()
    );

    // Create and load Parquet index
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
    let sharded = ShardedInvertedIndex::open(&index_path)?;

    let threshold = 0.1;

    // Test direct inverted index query
    let mut direct_results = Vec::new();
    for (i, seq) in seqs.iter().enumerate() {
        let mut ws = MinimizerWorkspace::new();
        extract_into(seq, k, w, salt, &mut ws);
        let mut mins: Vec<u64> = ws.buffer.drain(..).collect();
        mins.sort();
        mins.dedup();

        let hits = inverted_direct.get_bucket_hits(&mins);
        let total_mins = mins.len();

        for (bucket_id, hit_count) in hits {
            let score = hit_count as f64 / total_mins as f64;
            if score >= threshold {
                direct_results.push(HitResult {
                    query_id: i as i64,
                    bucket_id,
                    score,
                });
            }
        }
    }

    // Test Parquet index
    let sharded_results =
        classify_batch_sharded_merge_join(&sharded, None, &records, threshold, None)?;

    eprintln!("\n=== Results ===");
    eprintln!("Direct: {} results", direct_results.len());
    eprintln!("Parquet: {} results", sharded_results.len());

    compare_results("Direct", &direct_results, "Parquet", &sharded_results);

    Ok(())
}

#[test]
fn test_parquet_index_with_multiple_shards() -> Result<()> {
    eprintln!("\n\n========== PARQUET INDEX WITH MULTIPLE SHARDS ==========");

    let dir = tempdir()?;
    let index_path = dir.path().join("test.ryxdi");

    // Create more sequences to potentially trigger multiple shards
    let seqs: Vec<Vec<u8>> = (0..10)
        .map(|i| {
            let pattern = match i % 5 {
                0 => vec![b'A', b'C', b'G', b'T'],
                1 => vec![b'T', b'A', b'T', b'A'],
                2 => vec![b'G', b'C', b'G', b'C'],
                3 => vec![b'A', b'A', b'C', b'C'],
                _ => vec![b'G', b'G', b'T', b'T'],
            };
            (0..300).map(|j| pattern[j % pattern.len()]).collect()
        })
        .collect();

    let k = 32;
    let w = 20;
    let salt = 0x5555555555555555u64;

    let buckets = create_bucket_data(&seqs, k, w, salt);
    let records = create_query_records(&seqs);

    eprintln!("\nCreated {} buckets:", buckets.len());
    let mut total_mins = 0;
    for b in &buckets {
        eprintln!(
            "  Bucket {}: {} minimizers",
            b.bucket_id,
            b.minimizers.len()
        );
        total_mins += b.minimizers.len();
    }
    eprintln!("Total minimizers across all buckets: {}", total_mins);

    // Create Parquet index with small row group to potentially create multiple
    let options = ParquetWriteOptions {
        row_group_size: 1000, // Small row groups
        ..Default::default()
    };
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

    // Load and verify
    let sharded = ShardedInvertedIndex::open(&index_path)?;
    let manifest = sharded.manifest();

    eprintln!("\nLoaded Parquet index: {} shards", manifest.shards.len());
    eprintln!("Total minimizers: {}", manifest.total_minimizers);
    eprintln!("Total bucket entries: {}", manifest.total_bucket_ids);
    for shard in &manifest.shards {
        eprintln!(
            "  Shard {}: {} minimizers, {} bucket entries",
            shard.shard_id, shard.num_minimizers, shard.num_bucket_ids
        );
    }

    let threshold = 0.1;

    let results = classify_batch_sharded_merge_join(&sharded, None, &records, threshold, None)?;

    eprintln!("\n=== Classification Results ===");
    eprintln!("Merge-join: {} results", results.len());

    // Verify we have results - each query should match its corresponding bucket
    assert!(!results.is_empty(), "Should have classification results");

    // Verify self-matches exist for at least the first few queries
    // (Different sequences may have some minimizer overlap, but self-match should be strong)
    for i in 0..5 {
        let bucket_id = (i + 1) as u32;
        let self_match = results
            .iter()
            .find(|r| r.query_id == i as i64 && r.bucket_id == bucket_id);
        assert!(
            self_match.is_some(),
            "Expected self-match for query {} in bucket {}",
            i,
            bucket_id
        );
        assert!(
            self_match.unwrap().score > 0.9,
            "Self-match score should be high (>0.9), got {:.4} for query {} bucket {}",
            self_match.unwrap().score,
            i,
            bucket_id
        );
    }

    Ok(())
}

#[test]
fn test_parquet_index_diagnostics() -> Result<()> {
    eprintln!("\n\n========== PARQUET INDEX DIAGNOSTICS ==========");

    let dir = tempdir()?;
    let index_path = dir.path().join("test.ryxdi");

    let seqs = create_test_sequences();
    let k = 32;
    let w = 10;
    let salt = 0x1234u64;

    let buckets = create_bucket_data(&seqs, k, w, salt);

    // Build direct inverted for comparison
    let inverted_direct = build_inverted_from_buckets(&buckets, k, w, salt);

    // Create Parquet index
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

    // Load and inspect
    let sharded = ShardedInvertedIndex::open(&index_path)?;
    let manifest = sharded.manifest();

    eprintln!("\n=== Index Parameters ===");
    eprintln!("K: {}", manifest.k);
    eprintln!("W: {}", manifest.w);
    eprintln!("Salt: {:#x}", manifest.salt);

    eprintln!("\n=== Direct InvertedIndex ===");
    eprintln!("Unique minimizers: {}", inverted_direct.num_minimizers());
    eprintln!("Bucket entries: {}", inverted_direct.num_bucket_entries());

    eprintln!("\n=== Parquet Index ===");
    eprintln!("Shards: {}", manifest.shards.len());
    eprintln!("Total minimizers: {}", manifest.total_minimizers);
    eprintln!("Total bucket entries: {}", manifest.total_bucket_ids);
    eprintln!("Buckets: {}", manifest.bucket_names.len());

    // Verify bucket names were stored
    for (id, name) in &manifest.bucket_names {
        eprintln!("  Bucket {}: {}", id, name);
    }

    // Verify bucket entry counts match
    // total_bucket_ids is the count of (minimizer, bucket_id) pairs
    assert_eq!(
        manifest.total_bucket_ids,
        inverted_direct.num_bucket_entries(),
        "Bucket entry counts should match"
    );

    Ok(())
}

#[test]
fn test_empty_query_results() -> Result<()> {
    eprintln!("\n\n========== EMPTY QUERY RESULTS ==========");

    let dir = tempdir()?;
    let index_path = dir.path().join("test.ryxdi");

    let seqs = create_test_sequences();
    let k = 32;
    let w = 10;
    let salt = 0x1234u64;

    let buckets = create_bucket_data(&seqs, k, w, salt);

    // Create Parquet index
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

    let sharded = ShardedInvertedIndex::open(&index_path)?;

    // Query with sequence that won't match (too short)
    let short_seq = vec![b'A'; 10]; // Much shorter than k
    let records: Vec<QueryRecord<'_>> = vec![(0i64, short_seq.as_slice(), None)];

    let threshold = 0.1;

    let results = classify_batch_sharded_merge_join(&sharded, None, &records, threshold, None)?;

    eprintln!("Results for short query: {}", results.len());
    assert!(results.is_empty(), "Short query should have no results");

    Ok(())
}

#[test]
fn test_high_threshold_filters_results() -> Result<()> {
    eprintln!("\n\n========== HIGH THRESHOLD FILTERING ==========");

    let dir = tempdir()?;
    let index_path = dir.path().join("test.ryxdi");

    let seqs = create_test_sequences();
    let k = 32;
    let w = 10;
    let salt = 0x1234u64;

    let buckets = create_bucket_data(&seqs, k, w, salt);
    let records = create_query_records(&seqs);

    // Create Parquet index
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

    let sharded = ShardedInvertedIndex::open(&index_path)?;

    // Compare results at different thresholds
    let results_low = classify_batch_sharded_merge_join(&sharded, None, &records, 0.01, None)?;
    let results_mid = classify_batch_sharded_merge_join(&sharded, None, &records, 0.5, None)?;
    let results_high = classify_batch_sharded_merge_join(&sharded, None, &records, 0.99, None)?;

    eprintln!("Results at threshold 0.01: {}", results_low.len());
    eprintln!("Results at threshold 0.50: {}", results_mid.len());
    eprintln!("Results at threshold 0.99: {}", results_high.len());

    // Higher threshold should have fewer or equal results
    assert!(
        results_mid.len() <= results_low.len(),
        "Higher threshold should filter more results"
    );
    assert!(
        results_high.len() <= results_mid.len(),
        "Higher threshold should filter more results"
    );

    Ok(())
}
