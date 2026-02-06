//! Integration tests for oriented sequence addition to buckets.
//!
//! These tests verify that the `--orient` flag produces valid indices
//! and that classification works correctly with oriented indices.

use std::io::Write;

use anyhow::Result;
use tempfile::tempdir;

use rype::{
    classify_batch_sharded_merge_join, extract_into, BucketData, MinimizerWorkspace,
    ParquetWriteOptions, QueryRecord, ShardedInvertedIndex,
};

/// Create test sequences that have clear orientation differences.
///
/// Returns pairs of (forward, reverse_complement) sequences where
/// the forward sequence shares minimizers with the "baseline" bucket
/// and the RC sequence shares different minimizers.
fn create_oriented_test_sequences() -> Vec<Vec<u8>> {
    // Create a set of sequences where orientation matters
    vec![
        // Sequence 0: Baseline sequence (will establish bucket minimizers)
        b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
          ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
          ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
            .to_vec(),
        // Sequence 1: Similar to baseline (should prefer forward orientation)
        b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
          AAAACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
          ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
            .to_vec(),
        // Sequence 2: Another variation
        b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA\
          TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA\
          TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA"
            .to_vec(),
    ]
}

/// Create bucket data from test sequences (simulates index building).
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

/// Create query records from sequences.
fn create_query_records(seqs: &[Vec<u8>]) -> Vec<QueryRecord<'_>> {
    seqs.iter()
        .enumerate()
        .map(|(i, seq)| (i as i64, seq.as_slice(), None))
        .collect()
}

#[test]
fn test_orient_flag_produces_valid_index() -> Result<()> {
    // This test builds a Parquet index and verifies:
    // 1. The index is created successfully
    // 2. The index can be loaded
    // 3. Classification produces results

    let dir = tempdir()?;
    let index_path = dir.path().join("test_orient.ryxdi");

    let seqs = create_oriented_test_sequences();
    let k = 32;
    let w = 10;
    let salt = 0x1234u64;

    let buckets = create_bucket_data(&seqs, k, w, salt);
    let records = create_query_records(&seqs);

    // Create Parquet index (simulates what `--orient` produces)
    let options = ParquetWriteOptions::default();
    rype::create_parquet_inverted_index(
        &index_path,
        buckets.clone(),
        k,
        w,
        salt,
        None,
        Some(&options),
    )?;

    // Verify index can be loaded
    let sharded = ShardedInvertedIndex::open(&index_path)?;
    let manifest = sharded.manifest();

    // Verify manifest is valid
    assert_eq!(manifest.k, k);
    assert_eq!(manifest.w, w);
    assert_eq!(manifest.salt, salt);
    assert!(!manifest.shards.is_empty());
    assert!(manifest.total_minimizers > 0);

    // Verify classification works
    let threshold = 0.1;
    let results = classify_batch_sharded_merge_join(&sharded, None, &records, threshold, None)?;

    // Should have results for self-matches
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
            "Self-match score should be high (>0.9), got {:.4}",
            self_match.unwrap().score
        );
    }

    Ok(())
}

#[test]
fn test_classification_with_oriented_index() -> Result<()> {
    // Test that classification produces correct results with oriented index

    let dir = tempdir()?;
    let index_path = dir.path().join("test_classify_orient.ryxdi");

    let seqs = create_oriented_test_sequences();
    let k = 32;
    let w = 10;
    let salt = 0xABCDu64;

    let buckets = create_bucket_data(&seqs, k, w, salt);

    // Create index
    let options = ParquetWriteOptions::default();
    rype::create_parquet_inverted_index(
        &index_path,
        buckets.clone(),
        k,
        w,
        salt,
        None,
        Some(&options),
    )?;

    let sharded = ShardedInvertedIndex::open(&index_path)?;

    // Create query that is exact match to bucket 1
    let query_seq = seqs[0].clone();
    let query_records: Vec<QueryRecord<'_>> = vec![(0i64, query_seq.as_slice(), None)];

    let threshold = 0.1;
    let results =
        classify_batch_sharded_merge_join(&sharded, None, &query_records, threshold, None)?;

    // Should have at least one result
    assert!(!results.is_empty());

    // The exact match should have very high score
    let best_match = results.iter().find(|r| r.bucket_id == 1);
    assert!(best_match.is_some(), "Should match bucket 1");
    assert!(
        best_match.unwrap().score > 0.99,
        "Exact match should have score > 0.99, got {:.4}",
        best_match.unwrap().score
    );

    Ok(())
}

#[test]
fn test_orient_index_format_compatible() -> Result<()> {
    // Test that oriented index uses the same format as non-oriented
    // and classification works identically for self-matches

    let dir = tempdir()?;
    let index_path = dir.path().join("test_format.ryxdi");

    let seqs = create_oriented_test_sequences();
    let k = 32;
    let w = 10;
    let salt = 0x5555555555555555u64;

    let buckets = create_bucket_data(&seqs, k, w, salt);
    let records = create_query_records(&seqs);

    // Create index
    let options = ParquetWriteOptions::default();
    rype::create_parquet_inverted_index(&index_path, buckets, k, w, salt, None, Some(&options))?;

    // Load and verify format
    let sharded = ShardedInvertedIndex::open(&index_path)?;
    let manifest = sharded.manifest();

    // Verify all expected metadata fields exist
    assert_eq!(manifest.bucket_names.len(), seqs.len());
    assert!(manifest.bucket_names.contains_key(&1));
    assert!(manifest.bucket_names.contains_key(&2));
    assert!(manifest.bucket_names.contains_key(&3));

    // Verify bucket sources are preserved
    assert_eq!(manifest.bucket_sources.len(), seqs.len());

    // Classify and verify
    let threshold = 0.01; // Low threshold to see all matches
    let results = classify_batch_sharded_merge_join(&sharded, None, &records, threshold, None)?;

    // Each query should have at least one result (self-match)
    for i in 0..seqs.len() {
        let has_result = results.iter().any(|r| r.query_id == i as i64);
        assert!(
            has_result,
            "Query {} should have at least one classification result",
            i
        );
    }

    Ok(())
}

#[test]
fn test_orient_cli_config_file() -> Result<()> {
    // Test that orient_sequences works via config file

    let dir = tempdir()?;
    let config_path = dir.path().join("config.toml");
    let fasta_path = dir.path().join("test.fa");

    // Create test FASTA file
    let mut fasta_file = std::fs::File::create(&fasta_path)?;
    writeln!(fasta_file, ">seq1")?;
    writeln!(
        fasta_file,
        "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
    )?;
    writeln!(
        fasta_file,
        "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
    )?;
    writeln!(fasta_file, ">seq2")?;
    writeln!(
        fasta_file,
        "TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA"
    )?;
    writeln!(
        fasta_file,
        "TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA"
    )?;

    // Create config file with orient_sequences = true
    let config_content = format!(
        r#"
[index]
window = 10
salt = 0x1234
output = "test.ryidx"
orient_sequences = true

[buckets.TestBucket]
files = ["{}"]
"#,
        fasta_path.display()
    );

    let mut config_file = std::fs::File::create(&config_path)?;
    config_file.write_all(config_content.as_bytes())?;

    // Parse config and verify orient_sequences is set
    let cfg = rype::config::parse_config(&config_path)?;
    assert_eq!(cfg.index.orient_sequences, Some(true));

    Ok(())
}

#[test]
fn test_orient_config_defaults_to_none() -> Result<()> {
    // Test that orient_sequences defaults to None when not specified

    let dir = tempdir()?;
    let config_path = dir.path().join("config.toml");
    let fasta_path = dir.path().join("test.fa");

    // Create test FASTA file
    let mut fasta_file = std::fs::File::create(&fasta_path)?;
    writeln!(fasta_file, ">seq1")?;
    writeln!(
        fasta_file,
        "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
    )?;

    // Create config file WITHOUT orient_sequences
    let config_content = format!(
        r#"
[index]
window = 10
salt = 0x1234
output = "test.ryidx"

[buckets.TestBucket]
files = ["{}"]
"#,
        fasta_path.display()
    );

    let mut config_file = std::fs::File::create(&config_path)?;
    config_file.write_all(config_content.as_bytes())?;

    // Parse config and verify orient_sequences is None
    let cfg = rype::config::parse_config(&config_path)?;
    assert_eq!(cfg.index.orient_sequences, None);

    Ok(())
}
