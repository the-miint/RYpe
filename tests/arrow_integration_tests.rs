//! Integration tests for Apache Arrow support.
//!
//! These tests verify end-to-end functionality of the Arrow integration,
//! including consistency with the regular API and streaming behavior.

#![cfg(feature = "arrow")]

use anyhow::Result;
use arrow::array::{Array, BinaryArray, Float64Array, Int64Array, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;
use tempfile::tempdir;

use rype::arrow::{
    batch_to_records, classify_arrow_batch, classify_arrow_batch_sharded, hits_to_record_batch,
    result_schema, validate_input_schema, IndexStreamClassifier, COL_ID, COL_PAIR_SEQUENCE,
    COL_SEQUENCE,
};
use rype::{
    classify_batch, Index, InvertedIndex, MinimizerWorkspace, ShardManifest, ShardedInvertedIndex,
};

/// Generate a DNA sequence of given length with a deterministic pattern.
fn generate_sequence(len: usize, seed: u8) -> Vec<u8> {
    let bases = [b'A', b'C', b'G', b'T'];
    (0..len).map(|i| bases[(i + seed as usize) % 4]).collect()
}

/// Create a test batch with the expected schema.
fn make_test_batch(ids: &[i64], seqs: &[&[u8]]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new(COL_ID, DataType::Int64, false),
        Field::new(COL_SEQUENCE, DataType::Binary, false),
    ]));

    let id_array = Int64Array::from(ids.to_vec());
    let seq_array = BinaryArray::from_iter_values(seqs.iter().copied());

    RecordBatch::try_new(schema, vec![Arc::new(id_array), Arc::new(seq_array)]).unwrap()
}

/// Create a test batch with paired-end sequences.
fn make_test_batch_paired(ids: &[i64], seqs: &[&[u8]], pairs: &[Option<&[u8]>]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new(COL_ID, DataType::Int64, false),
        Field::new(COL_SEQUENCE, DataType::Binary, false),
        Field::new(COL_PAIR_SEQUENCE, DataType::Binary, true),
    ]));

    let id_array = Int64Array::from(ids.to_vec());
    let seq_array = BinaryArray::from_iter_values(seqs.iter().copied());
    let pair_array = BinaryArray::from_iter(pairs.iter().copied());

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(id_array),
            Arc::new(seq_array),
            Arc::new(pair_array),
        ],
    )
    .unwrap()
}

/// Create a test index with multiple buckets.
fn create_test_index() -> Index {
    let mut index = Index::new(16, 5, 0x12345).unwrap();
    let mut ws = MinimizerWorkspace::new();

    // Bucket 1: Pattern starting with seed 0
    let ref_seq1 = generate_sequence(100, 0);
    index.add_record(1, "ref1", &ref_seq1, &mut ws);
    index.finalize_bucket(1);
    index.bucket_names.insert(1, "bucket1".into());

    // Bucket 2: Pattern starting with seed 2
    let ref_seq2 = generate_sequence(100, 2);
    index.add_record(2, "ref2", &ref_seq2, &mut ws);
    index.finalize_bucket(2);
    index.bucket_names.insert(2, "bucket2".into());

    index
}

#[test]
fn test_arrow_roundtrip_classification() -> Result<()> {
    let index = create_test_index();

    // Query that matches bucket 1
    let query_seq = generate_sequence(100, 0);
    let batch = make_test_batch(&[101], &[&query_seq]);

    let result = classify_arrow_batch(&index, None, &batch, 0.0)?;

    assert!(result.num_rows() > 0, "Should have classification results");

    // Verify schema
    assert_eq!(result.schema(), result_schema());

    // Verify we can read the data
    let query_ids = result
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(query_ids.value(0), 101);

    Ok(())
}

#[test]
fn test_arrow_vs_regular_api_consistency() -> Result<()> {
    let index = create_test_index();
    let threshold = 0.1;

    // Create test queries
    let query1 = generate_sequence(100, 0); // Matches bucket 1
    let query2 = generate_sequence(100, 2); // Matches bucket 2
    let query3 = generate_sequence(100, 1); // Different pattern

    // Regular API
    let records: Vec<rype::QueryRecord> = vec![
        (1, query1.as_slice(), None),
        (2, query2.as_slice(), None),
        (3, query3.as_slice(), None),
    ];
    let regular_hits = classify_batch(&index, None, &records, threshold);

    // Arrow API
    let batch = make_test_batch(&[1, 2, 3], &[&query1, &query2, &query3]);
    let arrow_result = classify_arrow_batch(&index, None, &batch, threshold)?;

    // Results should be consistent
    assert_eq!(
        regular_hits.len(),
        arrow_result.num_rows(),
        "Arrow and regular API should produce same number of hits"
    );

    // Verify each regular hit is present in arrow results
    let arrow_query_ids = arrow_result
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let arrow_bucket_ids = arrow_result
        .column(1)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .unwrap();
    let arrow_scores = arrow_result
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();

    for regular_hit in &regular_hits {
        let found = (0..arrow_result.num_rows()).any(|i| {
            arrow_query_ids.value(i) == regular_hit.query_id
                && arrow_bucket_ids.value(i) == regular_hit.bucket_id
                && (arrow_scores.value(i) - regular_hit.score).abs() < 1e-10
        });
        assert!(
            found,
            "Regular hit {:?} not found in Arrow results",
            regular_hit
        );
    }

    Ok(())
}

#[test]
fn test_arrow_with_paired_end() -> Result<()> {
    let index = create_test_index();

    let seq1 = generate_sequence(80, 0);
    let pair1 = generate_sequence(80, 0);
    let seq2 = generate_sequence(80, 2);

    let batch = make_test_batch_paired(&[1, 2], &[&seq1, &seq2], &[Some(pair1.as_slice()), None]);

    let result = classify_arrow_batch(&index, None, &batch, 0.0)?;

    // Should have results for both queries
    assert!(result.num_rows() >= 2);

    Ok(())
}

#[test]
fn test_arrow_large_batch() -> Result<()> {
    let index = create_test_index();

    // Create a large batch
    let num_queries = 1000;
    let ids: Vec<i64> = (0..num_queries).collect();
    let sequences: Vec<Vec<u8>> = (0..num_queries)
        .map(|i| generate_sequence(100, (i % 4) as u8))
        .collect();
    let seq_refs: Vec<&[u8]> = sequences.iter().map(|s| s.as_slice()).collect();

    let batch = make_test_batch(&ids, &seq_refs);

    let result = classify_arrow_batch(&index, None, &batch, 0.0)?;

    // Should handle large batches without error
    assert!(result.num_rows() > 0);

    Ok(())
}

#[test]
fn test_arrow_streaming_multiple_batches() -> Result<()> {
    let index = create_test_index();
    let classifier = IndexStreamClassifier::new(&index, None, 0.0);

    // Create multiple batches
    let batch1 = make_test_batch(
        &[1, 2],
        &[&generate_sequence(100, 0), &generate_sequence(100, 1)],
    );
    let batch2 = make_test_batch(
        &[3, 4],
        &[&generate_sequence(100, 2), &generate_sequence(100, 3)],
    );
    let batch3 = make_test_batch(&[5], &[&generate_sequence(100, 0)]);

    let input_batches: Vec<Result<RecordBatch, arrow::error::ArrowError>> =
        vec![Ok(batch1), Ok(batch2), Ok(batch3)];

    let results: Vec<_> = classifier
        .classify_iter(input_batches.into_iter())
        .collect();

    assert_eq!(results.len(), 3, "Should have one result per input batch");
    for result in results {
        assert!(result.is_ok(), "Each batch should classify successfully");
    }

    Ok(())
}

#[test]
fn test_arrow_threshold_filtering() -> Result<()> {
    let index = create_test_index();

    let query_seq = generate_sequence(100, 0);
    let batch = make_test_batch(&[1], &[&query_seq]);

    // With very high threshold
    let high_result = classify_arrow_batch(&index, None, &batch, 1.0)?;

    // With zero threshold
    let low_result = classify_arrow_batch(&index, None, &batch, 0.0)?;

    // High threshold should filter more results
    assert!(
        low_result.num_rows() >= high_result.num_rows(),
        "Lower threshold should have more or equal results"
    );

    Ok(())
}

#[test]
fn test_arrow_empty_batch() -> Result<()> {
    let index = create_test_index();

    let schema = Arc::new(Schema::new(vec![
        Field::new(COL_ID, DataType::Int64, false),
        Field::new(COL_SEQUENCE, DataType::Binary, false),
    ]));
    let empty_batch = RecordBatch::new_empty(schema);

    let result = classify_arrow_batch(&index, None, &empty_batch, 0.1)?;

    assert_eq!(result.num_rows(), 0);
    assert_eq!(result.schema(), result_schema());

    Ok(())
}

#[test]
fn test_arrow_with_inverted_index() -> Result<()> {
    let dir = tempdir()?;
    let index = create_test_index();

    // Save main index
    let index_path = dir.path().join("test.ryidx");
    index.save(&index_path)?;

    // Build and save inverted index as sharded format
    let inverted_path = dir.path().join("test.ryxdi");
    let inverted = InvertedIndex::build_from_index(&index);

    // Save as single shard
    let shard_path = ShardManifest::shard_path(&inverted_path, 0);
    let shard_info = inverted.save_shard(&shard_path, 0, 0, inverted.num_minimizers(), true)?;

    // Create and save manifest
    let manifest = ShardManifest {
        k: inverted.k,
        w: inverted.w,
        salt: inverted.salt,
        source_hash: 0,
        total_minimizers: inverted.num_minimizers(),
        total_bucket_ids: inverted.num_bucket_entries(),
        has_overlapping_shards: true,
        shards: vec![shard_info],
    };
    let manifest_path = ShardManifest::manifest_path(&inverted_path);
    manifest.save(&manifest_path)?;

    // Open sharded inverted index
    let sharded = ShardedInvertedIndex::open(&inverted_path)?;

    // Test classification
    let query_seq = generate_sequence(100, 0);
    let batch = make_test_batch(&[1], &[&query_seq]);

    let result = classify_arrow_batch_sharded(&sharded, None, &batch, 0.0, false)?;

    assert!(result.num_rows() > 0);

    Ok(())
}

#[test]
fn test_arrow_schema_validation() -> Result<()> {
    // Valid schema
    let valid_schema = Schema::new(vec![
        Field::new(COL_ID, DataType::Int64, false),
        Field::new(COL_SEQUENCE, DataType::Binary, false),
    ]);
    assert!(validate_input_schema(&valid_schema).is_ok());

    // Invalid: wrong ID type
    let invalid_schema = Schema::new(vec![
        Field::new(COL_ID, DataType::Utf8, false),
        Field::new(COL_SEQUENCE, DataType::Binary, false),
    ]);
    assert!(validate_input_schema(&invalid_schema).is_err());

    // Invalid: missing sequence
    let missing_seq = Schema::new(vec![Field::new(COL_ID, DataType::Int64, false)]);
    assert!(validate_input_schema(&missing_seq).is_err());

    Ok(())
}

#[test]
fn test_arrow_zero_copy_verification() -> Result<()> {
    let seq_data = b"ACGTACGTACGTACGTACGTACGTACGTACGT";
    let batch = make_test_batch(&[1], &[seq_data]);

    let records = batch_to_records(&batch)?;

    // Get pointer to sequence in record
    let record_ptr = records[0].1.as_ptr();

    // Get pointer to sequence in Arrow array
    let seq_col = batch.column(1);
    let binary_arr = seq_col.as_any().downcast_ref::<BinaryArray>().unwrap();
    let arrow_ptr = binary_arr.value(0).as_ptr();

    // Verify zero-copy: pointers should be identical
    assert_eq!(
        record_ptr, arrow_ptr,
        "batch_to_records should provide zero-copy access to Arrow buffers"
    );

    Ok(())
}

#[test]
fn test_arrow_hits_to_batch_roundtrip() -> Result<()> {
    let hits = vec![
        rype::HitResult {
            query_id: 1,
            bucket_id: 10,
            score: 0.95,
        },
        rype::HitResult {
            query_id: 2,
            bucket_id: 20,
            score: 0.85,
        },
        rype::HitResult {
            query_id: 3,
            bucket_id: 10,
            score: 0.75,
        },
    ];

    let batch = hits_to_record_batch(hits.clone())?;

    assert_eq!(batch.num_rows(), 3);
    assert_eq!(batch.num_columns(), 3);

    // Verify data
    let query_ids = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let bucket_ids = batch
        .column(1)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .unwrap();
    let scores = batch
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();

    for (i, hit) in hits.iter().enumerate() {
        assert_eq!(query_ids.value(i), hit.query_id);
        assert_eq!(bucket_ids.value(i), hit.bucket_id);
        assert!((scores.value(i) - hit.score).abs() < 1e-10);
    }

    Ok(())
}
