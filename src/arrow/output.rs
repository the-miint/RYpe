//! Conversion from HitResult to Arrow RecordBatch.
//!
//! This module provides functions to convert classification results
//! to Arrow columnar format for efficient downstream processing.
//!
//! # Memory Semantics
//!
//! The `hits_to_record_batch()` function **copies** data from the input
//! `Vec<HitResult>` into Arrow columnar arrays. This is intentional and
//! unavoidable because:
//!
//! 1. `HitResult` is stored as Array-of-Structs (AoS): each struct contains
//!    `query_id`, `bucket_id`, and `score` contiguously.
//! 2. Arrow RecordBatch requires Struct-of-Arrays (SoA): separate contiguous
//!    arrays for each column.
//!
//! The copy overhead is minimal because `HitResult` is only 24 bytes
//! (8 + 4 + padding + 8 bytes), and modern CPUs handle sequential memory
//! copies very efficiently.

use arrow::array::{Float64Array, Int64Array, UInt32Array};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

use super::error::ArrowClassifyError;
use super::schema::result_schema;
use crate::HitResult;

/// Convert a vector of HitResults to an Arrow RecordBatch.
///
/// # Output Schema
///
/// - `query_id`: Int64 (non-nullable)
/// - `bucket_id`: UInt32 (non-nullable)
/// - `score`: Float64 (non-nullable)
///
/// # Arguments
///
/// * `hits` - Vector of classification results
///
/// # Returns
///
/// A RecordBatch containing the hit data in columnar format.
///
/// # Thread Safety
///
/// This function is thread-safe and can be called from multiple threads.
pub fn hits_to_record_batch(hits: Vec<HitResult>) -> Result<RecordBatch, ArrowClassifyError> {
    let schema = result_schema();

    if hits.is_empty() {
        return Ok(RecordBatch::new_empty(schema));
    }

    let capacity = hits.len();

    // Pre-allocate arrays
    let mut query_ids = Vec::with_capacity(capacity);
    let mut bucket_ids = Vec::with_capacity(capacity);
    let mut scores = Vec::with_capacity(capacity);

    for hit in hits {
        query_ids.push(hit.query_id);
        bucket_ids.push(hit.bucket_id);
        scores.push(hit.score);
    }

    let query_id_array = Int64Array::from(query_ids);
    let bucket_id_array = UInt32Array::from(bucket_ids);
    let score_array = Float64Array::from(scores);

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(query_id_array),
            Arc::new(bucket_id_array),
            Arc::new(score_array),
        ],
    )
    .map_err(ArrowClassifyError::from)
}

/// Create an empty result batch with the correct schema.
///
/// Useful when no hits exceed the threshold.
pub fn empty_result_batch() -> RecordBatch {
    RecordBatch::new_empty(result_schema())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Array;
    use arrow::datatypes::DataType;

    use super::super::schema::{COL_BUCKET_ID, COL_QUERY_ID, COL_SCORE};

    #[test]
    fn test_hits_to_batch_basic() {
        let hits = vec![
            HitResult {
                query_id: 1,
                bucket_id: 10,
                score: 0.95,
            },
            HitResult {
                query_id: 2,
                bucket_id: 20,
                score: 0.85,
            },
        ];

        let batch = hits_to_record_batch(hits).unwrap();

        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 3);

        // Check query_id column
        let query_ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(query_ids.value(0), 1);
        assert_eq!(query_ids.value(1), 2);

        // Check bucket_id column
        let bucket_ids = batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        assert_eq!(bucket_ids.value(0), 10);
        assert_eq!(bucket_ids.value(1), 20);

        // Check score column
        let scores = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert!((scores.value(0) - 0.95).abs() < 1e-10);
        assert!((scores.value(1) - 0.85).abs() < 1e-10);
    }

    #[test]
    fn test_hits_to_batch_empty() {
        let hits: Vec<HitResult> = vec![];
        let batch = hits_to_record_batch(hits).unwrap();

        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.num_columns(), 3);

        // Schema should still be correct
        assert_eq!(batch.schema().field(0).name(), COL_QUERY_ID);
        assert_eq!(batch.schema().field(1).name(), COL_BUCKET_ID);
        assert_eq!(batch.schema().field(2).name(), COL_SCORE);
    }

    #[test]
    fn test_hits_to_batch_large() {
        // Test with many results
        let hits: Vec<HitResult> = (0..10000)
            .map(|i| HitResult {
                query_id: i as i64,
                bucket_id: (i % 100) as u32,
                score: (i as f64) / 10000.0,
            })
            .collect();

        let batch = hits_to_record_batch(hits).unwrap();

        assert_eq!(batch.num_rows(), 10000);

        // Spot check some values
        let query_ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(query_ids.value(0), 0);
        assert_eq!(query_ids.value(9999), 9999);
    }

    #[test]
    fn test_hits_to_batch_schema_correct() {
        let hits = vec![HitResult {
            query_id: 1,
            bucket_id: 1,
            score: 0.5,
        }];

        let batch = hits_to_record_batch(hits).unwrap();
        let schema = batch.schema();

        // Verify column names
        assert_eq!(schema.field(0).name(), COL_QUERY_ID);
        assert_eq!(schema.field(1).name(), COL_BUCKET_ID);
        assert_eq!(schema.field(2).name(), COL_SCORE);

        // Verify data types
        assert_eq!(schema.field(0).data_type(), &DataType::Int64);
        assert_eq!(schema.field(1).data_type(), &DataType::UInt32);
        assert_eq!(schema.field(2).data_type(), &DataType::Float64);

        // Verify nullability
        assert!(!schema.field(0).is_nullable());
        assert!(!schema.field(1).is_nullable());
        assert!(!schema.field(2).is_nullable());
    }

    #[test]
    fn test_empty_result_batch() {
        let batch = empty_result_batch();

        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.num_columns(), 3);
        assert_eq!(batch.schema().field(0).name(), COL_QUERY_ID);
    }

    #[test]
    fn test_hits_to_batch_extreme_values() {
        let hits = vec![
            HitResult {
                query_id: i64::MIN,
                bucket_id: 0,
                score: 0.0,
            },
            HitResult {
                query_id: i64::MAX,
                bucket_id: u32::MAX,
                score: 1.0,
            },
        ];

        let batch = hits_to_record_batch(hits).unwrap();

        let query_ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(query_ids.value(0), i64::MIN);
        assert_eq!(query_ids.value(1), i64::MAX);

        let bucket_ids = batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        assert_eq!(bucket_ids.value(0), 0);
        assert_eq!(bucket_ids.value(1), u32::MAX);
    }
}
