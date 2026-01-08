use rype::Index;
use std::fs;
use tempfile::tempdir;
use anyhow::Result;

/// Test that index creation with multiple records creates single bucket (Issue #1)
#[test]
fn test_index_multi_record_single_bucket() -> Result<()> {
    let dir = tempdir()?;
    let ref_file = dir.path().join("reference.fa");

    // Create FASTA with multiple records
    fs::write(&ref_file,
        ">seq1\n\
         AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n\
         >seq2\n\
         TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT\n\
         >seq3\n\
         GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG\n"
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
    fs::write(&new_ref,
        ">seq1\n\
         AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n\
         >seq2\n\
         CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n"
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
    assert_eq!(sources.len(), 2, "New bucket should have 2 source sequences");
    assert!(sources[0].contains("seq1"));
    assert!(sources[1].contains("seq2"));

    Ok(())
}

/// Test that bucket naming is consistent (Issue #2)
#[test]
fn test_bucket_naming_consistency() -> Result<()> {
    let dir = tempdir()?;
    let ref_file = dir.path().join("myfile.fasta");

    fs::write(&ref_file,
        ">record1\n\
         AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n\
         >record2\n\
         TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT\n"
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
    assert_eq!(loaded.bucket_names[&1], filename,
               "Bucket name should be filename, not record name");

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
    assert!(!loaded.buckets.contains_key(&1), "Source bucket should be removed");
    assert!(loaded.buckets.contains_key(&2), "Dest bucket should exist");

    let count_after = loaded.buckets[&2].len();
    assert!(count_after >= count_before,
            "Minimizer count should be >= original count after merge");

    // Verify sources were merged
    assert_eq!(loaded.bucket_sources[&2].len(), 2, "Should have 2 sources");

    Ok(())
}
