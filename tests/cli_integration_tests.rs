use anyhow::Result;
use rype::QueryRecord;
use rype::{
    classify_batch, extract_with_positions, get_paired_minimizers_into, Index, MinimizerWorkspace,
    Strand,
};
use std::collections::HashSet;
use std::fs;
use tempfile::tempdir;

/// Test that index creation with multiple records creates single bucket (Issue #1)
#[test]
fn test_index_multi_record_single_bucket() -> Result<()> {
    let dir = tempdir()?;
    let ref_file = dir.path().join("reference.fa");

    // Create FASTA with multiple records
    fs::write(
        &ref_file,
        ">seq1\n\
         AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n\
         >seq2\n\
         TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT\n\
         >seq3\n\
         GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG\n",
    )?;

    let index_file = dir.path().join("test.ryidx");

    // Simulate: cargo run -- index -o test.ryidx -r reference.fa -w 50
    let mut index = Index::new(64, 50, 0x5555555555555555).unwrap();
    let mut ws = rype::MinimizerWorkspace::new();
    let filename = "reference.fa";

    // Add all records to bucket 1 (default behavior without --separate-buckets)
    use needletail::parse_fastx_file;
    let mut reader = parse_fastx_file(&ref_file)?;

    index.bucket_names.insert(1, filename.to_string());

    while let Some(record) = reader.next() {
        let rec = record?;
        let seq = rec.seq();
        let name = String::from_utf8_lossy(rec.id()).to_string();
        let source_label = format!("{}::{}", filename, name);
        index.add_record(1, &source_label, &seq, &mut ws);
    }

    index.finalize_bucket(1);
    index.save(&index_file)?;

    // Verify: Should have exactly 1 bucket
    let loaded = Index::load(&index_file)?;
    assert_eq!(loaded.buckets.len(), 1, "Should have exactly 1 bucket");
    assert!(loaded.buckets.contains_key(&1), "Bucket ID should be 1");

    // Verify: All 3 sequences should be in the bucket sources
    let sources = &loaded.bucket_sources[&1];
    assert_eq!(sources.len(), 3, "Should have 3 source sequences");
    assert!(sources[0].contains("seq1"));
    assert!(sources[1].contains("seq2"));
    assert!(sources[2].contains("seq3"));

    Ok(())
}

/// Test that index-bucket-add with multiple records creates single bucket (Issue #1 & #2)
#[test]
fn test_index_bucket_add_multi_record_single_bucket() -> Result<()> {
    let dir = tempdir()?;

    // Create initial index with one bucket
    let index_file = dir.path().join("test.ryidx");
    let mut index = Index::new(64, 50, 0x5555555555555555).unwrap();
    let mut ws = rype::MinimizerWorkspace::new();

    index.bucket_names.insert(1, "initial.fa".to_string());
    let seq = vec![b'A'; 80];
    index.add_record(1, "initial.fa::seq_init", &seq, &mut ws);
    index.finalize_bucket(1);
    index.save(&index_file)?;

    // Create new reference file with multiple records
    let new_ref = dir.path().join("new_reference.fa");
    fs::write(
        &new_ref,
        ">seq1\n\
         AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n\
         >seq2\n\
         CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n",
    )?;

    // Simulate: cargo run -- index-bucket-add -i test.ryidx -r new_reference.fa
    let mut idx = Index::load(&index_file)?;
    let next_id = idx.next_id()?;
    let filename = "new_reference.fa";

    // This is the fixed behavior: all records go into ONE bucket
    idx.bucket_names.insert(next_id, filename.to_string());

    use needletail::parse_fastx_file;
    let mut reader = parse_fastx_file(&new_ref)?;

    while let Some(record) = reader.next() {
        let rec = record?;
        let seq = rec.seq();
        let name = String::from_utf8_lossy(rec.id()).to_string();
        let source_label = format!("{}::{}", filename, name);
        idx.add_record(next_id, &source_label, &seq, &mut ws);
    }

    idx.finalize_bucket(next_id);
    idx.save(&index_file)?;

    // Verify
    let loaded = Index::load(&index_file)?;
    assert_eq!(loaded.buckets.len(), 2, "Should have exactly 2 buckets");

    // Verify the new bucket (ID=2) has both records
    let sources = &loaded.bucket_sources[&next_id];
    assert_eq!(
        sources.len(),
        2,
        "New bucket should have 2 source sequences"
    );
    assert!(sources[0].contains("seq1"));
    assert!(sources[1].contains("seq2"));

    Ok(())
}

/// Test that bucket naming is consistent (Issue #2)
#[test]
fn test_bucket_naming_consistency() -> Result<()> {
    let dir = tempdir()?;
    let ref_file = dir.path().join("myfile.fasta");

    fs::write(
        &ref_file,
        ">record1\n\
         AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n\
         >record2\n\
         TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT\n",
    )?;

    let index_file = dir.path().join("test.ryidx");

    // Create index using 'index' command behavior
    let mut index = Index::new(64, 50, 0x5555555555555555).unwrap();
    let mut ws = rype::MinimizerWorkspace::new();
    let filename = "myfile.fasta";

    index.bucket_names.insert(1, filename.to_string());

    use needletail::parse_fastx_file;
    let mut reader = parse_fastx_file(&ref_file)?;

    while let Some(record) = reader.next() {
        let rec = record?;
        let seq = rec.seq();
        let name = String::from_utf8_lossy(rec.id()).to_string();
        let source_label = format!("{}::{}", filename, name);
        index.add_record(1, &source_label, &seq, &mut ws);
    }

    index.finalize_bucket(1);
    index.save(&index_file)?;

    // Verify: Bucket name should be filename, not record name
    let loaded = Index::load(&index_file)?;
    assert_eq!(
        loaded.bucket_names[&1], filename,
        "Bucket name should be filename, not record name"
    );

    // But sources should include record names
    assert!(loaded.bucket_sources[&1][0].contains("record1"));
    assert!(loaded.bucket_sources[&1][1].contains("record2"));

    Ok(())
}

/// Test that merge_buckets correctly updates minimizer count (Issue #3)
#[test]
fn test_merge_buckets_minimizer_count() -> Result<()> {
    let dir = tempdir()?;
    let index_file = dir.path().join("test.ryidx");

    // Create index with two buckets
    let mut index = Index::new(64, 50, 0).unwrap();
    let mut ws = rype::MinimizerWorkspace::new();

    // Bucket 1: Poly-A sequence
    let seq1 = vec![b'A'; 80];
    index.bucket_names.insert(1, "bucket1".to_string());
    index.add_record(1, "source1", &seq1, &mut ws);
    index.finalize_bucket(1);

    // Bucket 2: Poly-T sequence
    let seq2 = vec![b'T'; 80];
    index.bucket_names.insert(2, "bucket2".to_string());
    index.add_record(2, "source2", &seq2, &mut ws);
    index.finalize_bucket(2);

    let count_before = index.buckets[&2].len();
    index.save(&index_file)?;

    // Simulate: cargo run -- index-bucket-merge -i test.ryidx --src 1 --dest 2
    let mut idx = Index::load(&index_file)?;
    idx.merge_buckets(1, 2)?;
    idx.save(&index_file)?;

    // Verify: Minimizer count should be updated
    let loaded = Index::load(&index_file)?;
    assert_eq!(loaded.buckets.len(), 1, "Should have 1 bucket after merge");
    assert!(
        !loaded.buckets.contains_key(&1),
        "Source bucket should be removed"
    );
    assert!(loaded.buckets.contains_key(&2), "Dest bucket should exist");

    let count_after = loaded.buckets[&2].len();
    assert!(
        count_after >= count_before,
        "Minimizer count should be >= original count after merge"
    );

    // Verify sources were merged
    assert_eq!(loaded.bucket_sources[&2].len(), 2, "Should have 2 sources");

    Ok(())
}

/// Test extract_with_positions returns correct positions and strands
#[test]
fn test_extract_with_positions_correctness() -> Result<()> {
    let mut ws = MinimizerWorkspace::new();

    // Create a simple sequence
    let seq = b"AAAATTTTGGGGCCCCAAAATTTTGGGGCCCC"; // 32 bases

    let results = extract_with_positions(seq, 16, 4, 0, &mut ws);

    // Should have some minimizers
    assert!(!results.is_empty(), "Should extract minimizers");

    // All positions should be valid
    for m in &results {
        assert!(
            m.position + 16 <= seq.len(),
            "Position {} invalid for seq len {}",
            m.position,
            seq.len()
        );

        // Strand should be either Forward or ReverseComplement
        match m.strand {
            Strand::Forward | Strand::ReverseComplement => {}
        }
    }

    // Should have both forward and reverse complement minimizers
    let has_fwd = results.iter().any(|m| m.strand == Strand::Forward);
    let has_rc = results
        .iter()
        .any(|m| m.strand == Strand::ReverseComplement);

    assert!(has_fwd, "Should have forward strand minimizers");
    assert!(has_rc, "Should have reverse complement minimizers");

    Ok(())
}

/// Test that minimizers from a query match minimizers in an index
#[test]
fn test_minimizer_matching_with_positions() -> Result<()> {
    let dir = tempdir()?;

    // Create reference sequence (long enough for minimizers)
    let ref_seq = b"AAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCC";
    let ref_file = dir.path().join("reference.fa");
    fs::write(
        &ref_file,
        format!(">ref1\n{}\n", String::from_utf8_lossy(ref_seq)),
    )?;

    // Create index
    let index_file = dir.path().join("test.ryidx");
    let mut index = Index::new(16, 4, 0)?;
    let mut ws = MinimizerWorkspace::new();

    use needletail::parse_fastx_file;
    let abs_path = ref_file.canonicalize()?;
    let filename = abs_path.to_string_lossy().to_string();

    index.bucket_names.insert(1, "reference".to_string());

    let mut reader = parse_fastx_file(&ref_file)?;
    while let Some(record) = reader.next() {
        let rec = record?;
        let seq = rec.seq();
        let name = String::from_utf8_lossy(rec.id()).to_string();
        let source_label = format!("{}::{}", filename, name);
        index.add_record(1, &source_label, &seq, &mut ws);
    }

    index.finalize_bucket(1);
    index.save(&index_file)?;

    // Now extract minimizers from a query that shares sequence with reference
    let query_seq = b"AAAATTTTGGGGCCCCAAAATTTTGGGGCCCC"; // substring of reference
    let query_mins = extract_with_positions(query_seq, 16, 4, 0, &mut ws);

    // Load index and check for matches
    let loaded = Index::load(&index_file)?;
    let bucket = &loaded.buckets[&1];

    // At least some query minimizers should match the bucket
    let matches: Vec<_> = query_mins
        .iter()
        .filter(|m| bucket.binary_search(&m.hash).is_ok())
        .collect();

    assert!(
        !matches.is_empty(),
        "Query should have minimizers matching the reference"
    );

    // All matched positions should be valid
    for m in &matches {
        assert!(m.position + 16 <= query_seq.len());
    }

    Ok(())
}

// ==================== Negative Index Tests ====================

/// Test that negative index filtering removes matching minimizers from queries before scoring.
///
/// This test creates:
/// 1. A positive index with a "target" sequence
/// 2. A negative index with a "contaminant" sequence that shares some minimizers with the query
/// 3. A query that matches both target and contaminant
///
/// Without negative filtering: query scores against positive index with all minimizers
/// With negative filtering: minimizers matching negative index are removed first,
///                         resulting in a different (typically lower) score
#[test]
fn test_negative_index_filters_query_minimizers() -> Result<()> {
    let dir = tempdir()?;

    // Parameters for both indices (must match)
    let k = 32;
    let w = 10;
    let salt = 0x1234567890abcdef_u64;

    // Create sequences:
    // - Target: ACGT repeated pattern (will be in positive index)
    // - Contaminant: TGCA repeated pattern (will be in negative index)
    // - Query: mixture that shares minimizers with both

    // Target sequence (120 bases)
    let target_seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";

    // Contaminant sequence (120 bases) - different pattern
    let contaminant_seq = b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA";

    // Query that is a concatenation of parts from both (will share minimizers with both)
    // First half from target, second half from contaminant
    let mut query_seq = Vec::with_capacity(120);
    query_seq.extend_from_slice(&target_seq[0..60]);
    query_seq.extend_from_slice(&contaminant_seq[0..60]);

    // Create positive index
    let positive_index_file = dir.path().join("positive.ryidx");
    let mut positive_index = Index::new(k, w, salt)?;
    let mut ws = MinimizerWorkspace::new();

    positive_index.bucket_names.insert(1, "target".to_string());
    positive_index.add_record(1, "target::seq1", target_seq, &mut ws);
    positive_index.finalize_bucket(1);
    positive_index.save(&positive_index_file)?;

    // Create negative index with same parameters
    let negative_index_file = dir.path().join("negative.ryidx");
    let mut negative_index = Index::new(k, w, salt)?;

    negative_index
        .bucket_names
        .insert(1, "contaminant".to_string());
    negative_index.add_record(1, "contaminant::seq1", contaminant_seq, &mut ws);
    negative_index.finalize_bucket(1);
    negative_index.save(&negative_index_file)?;

    // Load both indices
    let pos_idx = Index::load(&positive_index_file)?;
    let neg_idx = Index::load(&negative_index_file)?;

    // Verify parameters match (this is a requirement)
    assert_eq!(
        pos_idx.k, neg_idx.k,
        "K must match between positive and negative index"
    );
    assert_eq!(
        pos_idx.w, neg_idx.w,
        "W must match between positive and negative index"
    );
    assert_eq!(
        pos_idx.salt, neg_idx.salt,
        "Salt must match between positive and negative index"
    );

    // Extract minimizers from query
    let (fwd_mins, rc_mins) = get_paired_minimizers_into(&query_seq, None, k, w, salt, &mut ws);

    // Collect all negative minimizers into a set for filtering
    let negative_minimizers: HashSet<u64> = neg_idx
        .buckets
        .values()
        .flat_map(|v| v.iter().copied())
        .collect();

    // Count how many query minimizers match the negative index
    let fwd_negative_matches = fwd_mins
        .iter()
        .filter(|m| negative_minimizers.contains(m))
        .count();
    let rc_negative_matches = rc_mins
        .iter()
        .filter(|m| negative_minimizers.contains(m))
        .count();

    // We need at least some negative matches for this test to be meaningful
    assert!(
        fwd_negative_matches > 0 || rc_negative_matches > 0,
        "Test requires query to have minimizers matching negative index. \
         fwd_negative_matches={}, rc_negative_matches={}",
        fwd_negative_matches,
        rc_negative_matches
    );

    // Classify WITHOUT negative index (current behavior)
    let records_without_neg: Vec<QueryRecord> = vec![(0, &query_seq[..], None)];
    let results_without_neg = classify_batch(&pos_idx, None, &records_without_neg, 0.0);

    // There should be some hit without negative filtering
    assert!(
        !results_without_neg.is_empty(),
        "Query should have hits against positive index without negative filtering"
    );
    let score_without_neg = results_without_neg[0].score;

    // Now test WITH negative index filtering
    // Filter out minimizers that match the negative index
    let filtered_fwd: Vec<u64> = fwd_mins
        .iter()
        .copied()
        .filter(|m| !negative_minimizers.contains(m))
        .collect();
    let filtered_rc: Vec<u64> = rc_mins
        .iter()
        .copied()
        .filter(|m| !negative_minimizers.contains(m))
        .collect();

    // Manually compute the score with filtered minimizers
    // (This is what classify_batch_with_negative should do internally)
    let pos_bucket = &pos_idx.buckets[&1];

    let fwd_hits = filtered_fwd
        .iter()
        .filter(|m| pos_bucket.binary_search(m).is_ok())
        .count();
    let rc_hits = filtered_rc
        .iter()
        .filter(|m| pos_bucket.binary_search(m).is_ok())
        .count();

    let fwd_score = if !filtered_fwd.is_empty() {
        fwd_hits as f64 / filtered_fwd.len() as f64
    } else {
        0.0
    };
    let rc_score = if !filtered_rc.is_empty() {
        rc_hits as f64 / filtered_rc.len() as f64
    } else {
        0.0
    };
    let expected_score_with_neg = fwd_score.max(rc_score);

    // The score with negative filtering should be different (typically lower or equal)
    // because we removed some minimizers
    println!("Score without negative filtering: {}", score_without_neg);
    println!(
        "Expected score with negative filtering: {}",
        expected_score_with_neg
    );
    println!(
        "Query fwd minimizers: {} -> {} after filtering",
        fwd_mins.len(),
        filtered_fwd.len()
    );
    println!(
        "Query rc minimizers: {} -> {} after filtering",
        rc_mins.len(),
        filtered_rc.len()
    );

    // Classify WITH negative index filtering
    let results_with_neg = classify_batch(
        &pos_idx,
        Some(&negative_minimizers),
        &records_without_neg,
        0.0,
    );

    // The score with negative filtering should match our manual calculation
    // Note: if expected_score_with_neg is 0 (no hits after filtering), results may be empty
    let score_with_neg = if results_with_neg.is_empty() {
        0.0
    } else {
        results_with_neg[0].score
    };

    // Verify the score matches our manual calculation
    assert!(
        (score_with_neg - expected_score_with_neg).abs() < 0.001,
        "Score with negative filtering ({}) should match expected ({})",
        score_with_neg,
        expected_score_with_neg
    );

    // The key assertion: negative filtering should reduce or eliminate the score
    // (unless no negative minimizers matched, which we verified earlier isn't the case)
    assert!(
        score_with_neg <= score_without_neg,
        "Score with negative filtering ({}) should be <= score without ({})",
        score_with_neg,
        score_without_neg
    );

    // Since we verified negative matches exist, score should actually be different
    // (either lower, or same if negative minimizers didn't affect positive matches)
    println!(
        "Test passed: negative filtering reduced score from {} to {}",
        score_without_neg, score_with_neg
    );

    Ok(())
}

/// Test that classification without negative index produces identical results to current behavior
#[test]
fn test_negative_index_none_matches_current_behavior() -> Result<()> {
    let dir = tempdir()?;

    let k = 32;
    let w = 10;
    let salt = 0x1234567890abcdef_u64;

    // Create a simple index
    let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";

    let index_file = dir.path().join("test.ryidx");
    let mut index = Index::new(k, w, salt)?;
    let mut ws = MinimizerWorkspace::new();

    index.bucket_names.insert(1, "test".to_string());
    index.add_record(1, "test::seq1", seq, &mut ws);
    index.finalize_bucket(1);
    index.save(&index_file)?;

    let idx = Index::load(&index_file)?;

    // Query sequence
    let query = seq;
    let records: Vec<QueryRecord> = vec![(0, &query[..], None)];

    // Current behavior
    let results_current = classify_batch(&idx, None, &records, 0.0);

    // With negative=None should produce identical results - this is the same call
    // but we test both paths explicitly for clarity
    let results_with_none = classify_batch(
        &idx, None, // No negative index - should behave identically
        &records, 0.0,
    );

    assert_eq!(results_current.len(), results_with_none.len());

    for (curr, with_none) in results_current.iter().zip(results_with_none.iter()) {
        assert_eq!(curr.query_id, with_none.query_id);
        assert_eq!(curr.bucket_id, with_none.bucket_id);
        assert!((curr.score - with_none.score).abs() < 0.001);
    }

    Ok(())
}

/// Test that negative index parameter validation works correctly
/// Note: Parameter validation is done at the CLI level (in main.rs) when loading the negative index.
/// This test verifies that using a negative minimizer set produces expected results.
#[test]
fn test_negative_index_parameter_validation() -> Result<()> {
    let dir = tempdir()?;

    // Create positive index
    let positive_file = dir.path().join("positive.ryidx");
    let mut positive = Index::new(32, 10, 0x1234)?;
    let mut ws = MinimizerWorkspace::new();
    let seq = vec![b'A'; 80];
    positive.bucket_names.insert(1, "pos".to_string());
    positive.add_record(1, "pos::seq1", &seq, &mut ws);
    positive.finalize_bucket(1);
    positive.save(&positive_file)?;

    let pos_idx = Index::load(&positive_file)?;

    // Build a negative minimizer set from the positive index minimizers
    // (in practice this would be built from a separate negative index file)
    let neg_mins: HashSet<u64> = pos_idx.buckets[&1].iter().copied().collect();

    let query = vec![b'A'; 80];
    let records: Vec<QueryRecord> = vec![(0, &query[..], None)];

    // Without negative filtering, should get a hit
    let results_without_neg = classify_batch(&pos_idx, None, &records, 0.0);
    assert!(
        !results_without_neg.is_empty(),
        "Should have hits without negative filtering"
    );

    // With negative filtering (all minimizers filtered), should get no hits or 0 score
    let results_with_neg = classify_batch(&pos_idx, Some(&neg_mins), &records, 0.0);

    // Since we're filtering with the same minimizers that are in the positive index,
    // the score should be reduced (or results empty if all query minimizers were filtered)
    if results_with_neg.is_empty() {
        println!("All query minimizers filtered out by negative index");
    } else {
        assert!(
            results_with_neg[0].score <= results_without_neg[0].score,
            "Score with negative filtering should be <= score without"
        );
    }

    Ok(())
}
