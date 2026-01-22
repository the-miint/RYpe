//! # Apache Arrow Integration
//!
//! Efficient data exchange with Arrow-compatible systems like DuckDB, Polars, and PyArrow.
//!
//! This module provides functions to classify genomic sequences using Arrow's
//! columnar format, enabling efficient integration with data processing pipelines.
//!
//! ## Input Schema Requirements
//!
//! Input RecordBatches must have the following columns:
//!
//! | Column | Type | Nullable | Description |
//! |--------|------|----------|-------------|
//! | `id` | Int64 | No | Query identifier |
//! | `sequence` | Binary or LargeBinary | No | DNA sequence bytes |
//! | `pair_sequence` | Binary or LargeBinary | Yes | Optional paired-end sequence |
//!
//! ## Output Schema
//!
//! Result RecordBatches have the following columns:
//!
//! | Column | Type | Nullable | Description |
//! |--------|------|----------|-------------|
//! | `query_id` | Int64 | No | Matching query ID |
//! | `bucket_id` | UInt32 | No | Matched bucket/reference ID |
//! | `score` | Float64 | No | Classification score (0.0-1.0) |
//!
//! ## Memory Semantics
//!
//! **Input (Zero-Copy):** The `batch_to_records()` function returns slices pointing
//! directly into Arrow buffers. No sequence data is copied. The returned records are
//! valid only while the source RecordBatch is alive.
//!
//! **Output (Copied):** The `hits_to_record_batch()` function copies classification
//! results from the internal `HitResult` struct (Array-of-Structs layout) into Arrow
//! columnar arrays (Struct-of-Arrays layout). This copy is unavoidable due to the
//! layout difference and is efficient for the small result struct (24 bytes per hit).
//!
//! ## Thread Safety
//!
//! All public types in this module are `Send + Sync`:
//!
//! - `ArrowClassifyError`: Safe to send across threads
//! - `IndexStreamClassifier<'a>`: Safe to send if the referenced `Index` is `Sync`
//!
//! Classification functions use internal rayon parallelism. The streaming classifiers
//! are designed for single-threaded use per stream instance, but multiple streams can
//! run concurrently in separate threads.
//!
//! ## Limitations
//!
//! - Maximum sequence length: 2GB (i32::MAX bytes, enforced at input validation)
//! - Empty sequences produce score 0 (no minimizers extracted)
//! - Null sequences in required columns cause errors
//! - Result order may differ from input order due to parallel processing
//!
//! ## Example
//!
//! ```ignore
//! use rype::arrow::{classify_arrow_batch, result_schema};
//! use rype::Index;
//!
//! // Load index
//! let index = Index::load("reference.ryidx")?;
//!
//! // Create or receive Arrow batch from DuckDB, Parquet, etc.
//! let sequences: RecordBatch = /* ... */;
//!
//! // Classify
//! let results = classify_arrow_batch(&index, None, &sequences, 0.1)?;
//!
//! // Results can be passed to DuckDB, written to Parquet, etc.
//! assert_eq!(results.schema(), result_schema());
//! ```

mod error;
mod input;
mod output;
mod schema;
mod stream;

// Re-export public API
pub use error::ArrowClassifyError;
pub use input::{batch_to_records, batch_to_records_with_columns};
pub use output::{empty_result_batch, hits_to_record_batch};
pub use schema::{
    result_schema, validate_input_schema, COL_BUCKET_ID, COL_ID, COL_PAIR_SEQUENCE, COL_QUERY_ID,
    COL_SCORE, COL_SEQUENCE,
};
pub use stream::IndexStreamClassifier;

use std::collections::HashSet;

use arrow::record_batch::RecordBatch;

use crate::{
    classify_batch, classify_batch_sharded_main, classify_batch_sharded_merge_join,
    classify_batch_sharded_sequential, Index, ShardedInvertedIndex, ShardedMainIndex,
};

/// Classify sequences from an Arrow RecordBatch using an Index.
///
/// This is a convenience function that combines input conversion, classification,
/// and output conversion in a single call.
///
/// # Arguments
///
/// * `index` - The index to classify against
/// * `negative_mins` - Optional set of minimizers to exclude from queries
/// * `batch` - Input RecordBatch with sequences (see module docs for schema)
/// * `threshold` - Minimum score threshold for reporting hits (0.0-1.0)
///
/// # Returns
///
/// A RecordBatch containing classification results (query_id, bucket_id, score).
///
/// # Errors
///
/// Returns an error if:
/// - Schema validation fails
/// - Required columns contain null values
/// - Type conversion fails
///
/// # Example
///
/// ```ignore
/// let results = classify_arrow_batch(&index, None, &sequences, 0.1)?;
/// println!("Found {} hits", results.num_rows());
/// ```
pub fn classify_arrow_batch(
    index: &Index,
    negative_mins: Option<&HashSet<u64>>,
    batch: &RecordBatch,
    threshold: f64,
) -> Result<RecordBatch, ArrowClassifyError> {
    let records = batch_to_records(batch)?;
    let hits = classify_batch(index, negative_mins, &records, threshold);
    hits_to_record_batch(hits)
}

/// Classify sequences from an Arrow RecordBatch using a ShardedInvertedIndex.
///
/// Loads shards sequentially from disk, enabling classification when the full
/// inverted index exceeds available memory.
///
/// # Memory Complexity
///
/// O(batch_size × minimizers_per_read) + O(single_shard_size)
///
/// # Arguments
///
/// * `sharded` - The sharded inverted index (manifest + shard paths)
/// * `negative_mins` - Optional set of minimizers to exclude from queries
/// * `batch` - Input RecordBatch with sequences (see module docs for schema)
/// * `threshold` - Minimum score threshold for reporting hits (0.0-1.0)
/// * `use_merge_join` - If true, uses merge-join strategy (more efficient with high
///   minimizer overlap); if false, uses sequential lookup strategy
///
/// # Returns
///
/// A RecordBatch containing classification results (query_id, bucket_id, score).
///
/// # Errors
///
/// Returns an error if schema validation fails, shards cannot be loaded, or
/// type conversion fails.
///
/// # Example
///
/// ```ignore
/// let sharded = ShardedInvertedIndex::load("index.ryxdi.manifest")?;
/// let results = classify_arrow_batch_sharded(&sharded, None, &sequences, 0.1, true)?;
/// ```
pub fn classify_arrow_batch_sharded(
    sharded: &ShardedInvertedIndex,
    negative_mins: Option<&HashSet<u64>>,
    batch: &RecordBatch,
    threshold: f64,
    use_merge_join: bool,
) -> Result<RecordBatch, ArrowClassifyError> {
    let records = batch_to_records(batch)?;
    let hits = if use_merge_join {
        classify_batch_sharded_merge_join(sharded, negative_mins, &records, threshold, None)
    } else {
        classify_batch_sharded_sequential(sharded, negative_mins, &records, threshold, None)
    }
    .map_err(|e| ArrowClassifyError::Classification(e.to_string()))?;
    hits_to_record_batch(hits)
}

/// Classify sequences from an Arrow RecordBatch using a ShardedMainIndex.
///
/// For very large main indices that don't fit in memory, loads index shards
/// sequentially during classification.
///
/// # Memory Complexity
///
/// O(batch_size × minimizers_per_read) + O(single_shard_buckets)
///
/// # Arguments
///
/// * `sharded` - The sharded main index
/// * `negative_mins` - Optional set of minimizers to exclude from queries
/// * `batch` - Input RecordBatch with sequences (see module docs for schema)
/// * `threshold` - Minimum score threshold for reporting hits (0.0-1.0)
///
/// # Returns
///
/// A RecordBatch containing classification results (query_id, bucket_id, score).
///
/// # Errors
///
/// Returns an error if schema validation fails, shards cannot be loaded, or
/// type conversion fails.
pub fn classify_arrow_batch_sharded_main(
    sharded: &ShardedMainIndex,
    negative_mins: Option<&HashSet<u64>>,
    batch: &RecordBatch,
    threshold: f64,
) -> Result<RecordBatch, ArrowClassifyError> {
    let records = batch_to_records(batch)?;
    let hits = classify_batch_sharded_main(sharded, negative_mins, &records, threshold)
        .map_err(|e| ArrowClassifyError::Classification(e.to_string()))?;
    hits_to_record_batch(hits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MinimizerWorkspace;
    use arrow::array::{BinaryArray, Float64Array, Int64Array, UInt32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    /// Helper to generate a DNA sequence.
    fn generate_sequence(len: usize, seed: u8) -> Vec<u8> {
        let bases = [b'A', b'C', b'G', b'T'];
        (0..len).map(|i| bases[(i + seed as usize) % 4]).collect()
    }

    /// Helper to create a test batch.
    fn make_test_batch(ids: &[i64], seqs: &[&[u8]]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new(COL_ID, DataType::Int64, false),
            Field::new(COL_SEQUENCE, DataType::Binary, false),
        ]));

        let id_array = Int64Array::from(ids.to_vec());
        let seq_array = BinaryArray::from_iter_values(seqs.iter().copied());

        RecordBatch::try_new(schema, vec![Arc::new(id_array), Arc::new(seq_array)]).unwrap()
    }

    /// Create a simple test index.
    fn create_test_index() -> Index {
        let mut index = Index::new(16, 5, 0x12345).unwrap();
        let mut ws = MinimizerWorkspace::new();

        let ref_seq = generate_sequence(100, 0);
        index.add_record(1, "ref1", &ref_seq, &mut ws);
        index.finalize_bucket(1);
        index.bucket_names.insert(1, "test_bucket".into());

        index
    }

    #[test]
    fn test_classify_arrow_batch_basic() {
        let index = create_test_index();

        // Query that matches reference
        let query_seq = generate_sequence(100, 0);
        let batch = make_test_batch(&[101], &[&query_seq]);

        let result = classify_arrow_batch(&index, None, &batch, 0.0).unwrap();

        assert!(result.num_rows() > 0, "Should have classification results");
        assert_eq!(result.num_columns(), 3);

        // Verify schema
        assert_eq!(result.schema().field(0).name(), COL_QUERY_ID);
        assert_eq!(result.schema().field(1).name(), COL_BUCKET_ID);
        assert_eq!(result.schema().field(2).name(), COL_SCORE);
    }

    #[test]
    fn test_classify_arrow_batch_no_matches() {
        let index = create_test_index();

        // Query with completely different sequence
        let query_seq = vec![b'N'; 100]; // All N's produce no minimizers
        let batch = make_test_batch(&[101], &[&query_seq]);

        let result = classify_arrow_batch(&index, None, &batch, 0.5).unwrap();

        // May have 0 rows if nothing matches threshold
        assert_eq!(result.num_columns(), 3);
    }

    #[test]
    fn test_classify_arrow_batch_multiple_queries() {
        let index = create_test_index();

        let query1 = generate_sequence(100, 0); // Matches ref
        let query2 = generate_sequence(100, 1); // Different pattern
        let batch = make_test_batch(&[1, 2], &[&query1, &query2]);

        let result = classify_arrow_batch(&index, None, &batch, 0.0).unwrap();

        // Should have results
        assert!(result.num_rows() >= 1);

        // Verify we can read the data
        let query_ids = result
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let bucket_ids = result
            .column(1)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let scores = result
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        for i in 0..result.num_rows() {
            assert!(query_ids.value(i) == 1 || query_ids.value(i) == 2);
            assert_eq!(bucket_ids.value(i), 1); // Only bucket in index
            assert!(scores.value(i) >= 0.0 && scores.value(i) <= 1.0);
        }
    }

    #[test]
    fn test_classify_arrow_batch_empty() {
        let index = create_test_index();

        let schema = Arc::new(Schema::new(vec![
            Field::new(COL_ID, DataType::Int64, false),
            Field::new(COL_SEQUENCE, DataType::Binary, false),
        ]));
        let empty_batch = RecordBatch::new_empty(schema);

        let result = classify_arrow_batch(&index, None, &empty_batch, 0.1).unwrap();
        assert_eq!(result.num_rows(), 0);
    }

    #[test]
    fn test_classify_arrow_batch_invalid_schema() {
        let index = create_test_index();

        // Wrong ID type
        let schema = Arc::new(Schema::new(vec![
            Field::new(COL_ID, DataType::Utf8, false),
            Field::new(COL_SEQUENCE, DataType::Binary, false),
        ]));
        let batch = RecordBatch::new_empty(schema);

        let result = classify_arrow_batch(&index, None, &batch, 0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_classify_arrow_matches_regular_api() {
        let index = create_test_index();

        // Create test data
        let query1 = generate_sequence(100, 0);
        let query2 = generate_sequence(100, 2);
        let threshold = 0.1;

        // Regular API
        let records: Vec<crate::QueryRecord> =
            vec![(1, query1.as_slice(), None), (2, query2.as_slice(), None)];
        let regular_hits = classify_batch(&index, None, &records, threshold);

        // Arrow API
        let batch = make_test_batch(&[1, 2], &[&query1, &query2]);
        let arrow_result = classify_arrow_batch(&index, None, &batch, threshold).unwrap();

        // Results should be consistent
        assert_eq!(
            regular_hits.len(),
            arrow_result.num_rows(),
            "Arrow and regular API should produce same number of hits"
        );

        // Verify hit content matches (may be in different order)
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
            // Find matching hit in arrow results
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
    }
}
