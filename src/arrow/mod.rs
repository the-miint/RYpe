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
//! - `ShardedStreamClassifier<'a>`: Safe to send if the referenced index is `Sync`
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
//! use rype::arrow::{classify_arrow_batch_sharded, result_schema};
//! use rype::ShardedInvertedIndex;
//!
//! // Load index
//! let index = ShardedInvertedIndex::open("reference.ryxdi")?;
//!
//! // Create or receive Arrow batch from DuckDB, Parquet, etc.
//! let sequences: RecordBatch = /* ... */;
//!
//! // Classify sequences
//! let results = classify_arrow_batch_sharded(&index, None, &sequences, 0.1)?;
//!
//! // Results can be passed to DuckDB, written to Parquet, etc.
//! assert_eq!(results.schema(), result_schema());
//! ```

mod error;
pub mod extraction;
mod input;
mod output;
mod schema;
mod stream;

// Re-export public API
pub use error::ArrowClassifyError;
pub use extraction::{
    extract_minimizer_set_batch, extract_strand_minimizers_batch, minimizer_set_schema,
    strand_minimizers_schema,
};
pub use input::{batch_to_records, batch_to_records_with_columns};
pub use output::{empty_result_batch, hits_to_record_batch};
pub use schema::{
    log_ratio_result_schema, result_schema, validate_input_schema, COL_BUCKET_ID, COL_FAST_PATH,
    COL_ID, COL_LOG_RATIO, COL_PAIR_SEQUENCE, COL_QUERY_ID, COL_SCORE, COL_SEQUENCE,
};
pub use stream::ShardedStreamClassifier;

use std::collections::HashSet;

use arrow::record_batch::RecordBatch;

use crate::{classify_batch_sharded_merge_join, ShardedInvertedIndex};

/// Classify sequences from an Arrow RecordBatch using a ShardedInvertedIndex.
///
/// Loads shards sequentially from disk, enabling classification when the full
/// inverted index exceeds available memory. Uses merge-join algorithm for
/// efficient classification.
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
/// let sharded = ShardedInvertedIndex::open("index.ryxdi")?;
/// let results = classify_arrow_batch_sharded(&sharded, None, &sequences, 0.1)?;
/// ```
pub fn classify_arrow_batch_sharded(
    sharded: &ShardedInvertedIndex,
    negative_mins: Option<&HashSet<u64>>,
    batch: &RecordBatch,
    threshold: f64,
) -> Result<RecordBatch, ArrowClassifyError> {
    classify_arrow_batch_sharded_internal(sharded, negative_mins, batch, threshold, false)
}

/// Classify sequences in an Arrow RecordBatch and return only the best hit per query.
///
/// Same as `classify_arrow_batch_sharded` but filters results to keep only the
/// highest-scoring bucket for each query. If multiple buckets tie for the best
/// score, one is chosen arbitrarily.
///
/// # Arguments
///
/// * `sharded` - The sharded inverted index to classify against
/// * `negative_mins` - Optional set of minimizers to exclude (for contamination filtering)
/// * `batch` - Arrow RecordBatch with columns: `read_id` (Int64), `sequence` (Binary/LargeBinary)
/// * `threshold` - Minimum score threshold (0.0-1.0) for reporting matches
///
/// # Returns
///
/// RecordBatch with at most one row per query_id, containing the highest-scoring hit.
pub fn classify_arrow_batch_sharded_best_hit(
    sharded: &ShardedInvertedIndex,
    negative_mins: Option<&HashSet<u64>>,
    batch: &RecordBatch,
    threshold: f64,
) -> Result<RecordBatch, ArrowClassifyError> {
    classify_arrow_batch_sharded_internal(sharded, negative_mins, batch, threshold, true)
}

/// Internal implementation that supports both regular and best-hit modes.
fn classify_arrow_batch_sharded_internal(
    sharded: &ShardedInvertedIndex,
    negative_mins: Option<&HashSet<u64>>,
    batch: &RecordBatch,
    threshold: f64,
    best_hit_only: bool,
) -> Result<RecordBatch, ArrowClassifyError> {
    let records = batch_to_records(batch)?;
    let hits = classify_batch_sharded_merge_join(sharded, negative_mins, &records, threshold, None)
        .map_err(|e| ArrowClassifyError::Classification(e.to_string()))?;

    let hits = if best_hit_only {
        crate::classify::filter_best_hits(hits)
    } else {
        hits
    };

    hits_to_record_batch(hits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        create_parquet_inverted_index, extract_into, BucketData, MinimizerWorkspace,
        ParquetWriteOptions,
    };
    use arrow::array::{BinaryArray, Float64Array, Int64Array, UInt32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;
    use tempfile::tempdir;

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

    /// Create a test Parquet index
    fn create_test_parquet_index() -> (tempfile::TempDir, ShardedInvertedIndex) {
        let dir = tempdir().unwrap();
        let index_path = dir.path().join("test.ryxdi");

        let mut ws = MinimizerWorkspace::new();
        let ref_seq = generate_sequence(100, 0);
        extract_into(&ref_seq, 16, 5, 0x12345, &mut ws);
        let mut mins: Vec<u64> = ws.buffer.drain(..).collect();
        mins.sort();
        mins.dedup();

        let buckets = vec![BucketData {
            bucket_id: 1,
            bucket_name: "test_bucket".to_string(),
            sources: vec!["ref1".to_string()],
            minimizers: mins,
        }];

        let options = ParquetWriteOptions::default();
        create_parquet_inverted_index(
            &index_path,
            buckets,
            16,
            5,
            0x12345,
            None,
            Some(&options),
            None,
        )
        .unwrap();

        let index = ShardedInvertedIndex::open(&index_path).unwrap();
        (dir, index)
    }

    #[test]
    fn test_classify_arrow_batch_sharded_basic() {
        let (_dir, index) = create_test_parquet_index();

        // Query that matches reference
        let query_seq = generate_sequence(100, 0);
        let batch = make_test_batch(&[101], &[&query_seq]);

        let result = classify_arrow_batch_sharded(&index, None, &batch, 0.0).unwrap();

        assert!(result.num_rows() > 0, "Should have classification results");
        assert_eq!(result.num_columns(), 3);

        // Verify schema
        assert_eq!(result.schema().field(0).name(), COL_QUERY_ID);
        assert_eq!(result.schema().field(1).name(), COL_BUCKET_ID);
        assert_eq!(result.schema().field(2).name(), COL_SCORE);
    }

    #[test]
    fn test_classify_arrow_batch_sharded_no_matches() {
        let (_dir, index) = create_test_parquet_index();

        // Query with completely different sequence
        let query_seq = vec![b'N'; 100]; // All N's produce no minimizers
        let batch = make_test_batch(&[101], &[&query_seq]);

        let result = classify_arrow_batch_sharded(&index, None, &batch, 0.5).unwrap();

        // May have 0 rows if nothing matches threshold
        assert_eq!(result.num_columns(), 3);
    }

    #[test]
    fn test_classify_arrow_batch_sharded_multiple_queries() {
        let (_dir, index) = create_test_parquet_index();

        let query1 = generate_sequence(100, 0); // Matches ref
        let query2 = generate_sequence(100, 1); // Different pattern
        let batch = make_test_batch(&[1, 2], &[&query1, &query2]);

        let result = classify_arrow_batch_sharded(&index, None, &batch, 0.0).unwrap();

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
    fn test_classify_arrow_batch_sharded_empty() {
        let (_dir, index) = create_test_parquet_index();

        let schema = Arc::new(Schema::new(vec![
            Field::new(COL_ID, DataType::Int64, false),
            Field::new(COL_SEQUENCE, DataType::Binary, false),
        ]));
        let empty_batch = RecordBatch::new_empty(schema);

        let result = classify_arrow_batch_sharded(&index, None, &empty_batch, 0.1).unwrap();
        assert_eq!(result.num_rows(), 0);
    }

    #[test]
    fn test_classify_arrow_batch_sharded_invalid_schema() {
        let (_dir, index) = create_test_parquet_index();

        // Wrong ID type
        let schema = Arc::new(Schema::new(vec![
            Field::new(COL_ID, DataType::Utf8, false),
            Field::new(COL_SEQUENCE, DataType::Binary, false),
        ]));
        let batch = RecordBatch::new_empty(schema);

        let result = classify_arrow_batch_sharded(&index, None, &batch, 0.1);
        assert!(result.is_err());
    }
}
