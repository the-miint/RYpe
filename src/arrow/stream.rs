//! Streaming classification with Arrow.
//!
//! This module provides streaming classification that processes input batches
//! incrementally and yields result batches as they are ready.
//!
//! # Streaming Benefits
//!
//! - **Memory bounded**: Only one input batch and one output batch in memory at a time
//! - **Low latency**: Results available as soon as each batch is processed
//! - **Backpressure**: Consumer controls pace by pulling from iterator
//!
//! # Thread Safety
//!
//! The streaming classifier is single-threaded per stream instance.
//! Classification within each batch uses internal rayon parallelism.

use std::collections::HashSet;

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;

use super::error::ArrowClassifyError;
use super::input::batch_to_records;
use super::output::hits_to_record_batch;
use super::schema::result_schema;
use crate::{classify_batch, Index};

/// Streaming classifier for Index-based classification.
///
/// Processes input batches one at a time and yields result batches.
///
/// # Example
///
/// ```ignore
/// let classifier = IndexStreamClassifier::new(&index, None, 0.1);
/// for result_batch in classifier.classify_iter(input_batches) {
///     let batch = result_batch?;
///     // Process results...
/// }
/// ```
pub struct IndexStreamClassifier<'a> {
    index: &'a Index,
    negative_mins: Option<&'a HashSet<u64>>,
    threshold: f64,
    output_schema: SchemaRef,
}

impl<'a> IndexStreamClassifier<'a> {
    /// Create a new streaming classifier.
    ///
    /// # Arguments
    ///
    /// * `index` - The index to classify against
    /// * `negative_mins` - Optional set of minimizers to exclude
    /// * `threshold` - Minimum score threshold for reporting hits
    pub fn new(index: &'a Index, negative_mins: Option<&'a HashSet<u64>>, threshold: f64) -> Self {
        Self {
            index,
            negative_mins,
            threshold,
            output_schema: result_schema(),
        }
    }

    /// Get the output schema for result batches.
    pub fn output_schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }

    /// Classify a single batch and return results.
    pub fn classify_batch(&self, batch: &RecordBatch) -> Result<RecordBatch, ArrowClassifyError> {
        let records = batch_to_records(batch)?;
        let hits = classify_batch(self.index, self.negative_mins, &records, self.threshold);
        hits_to_record_batch(hits)
    }

    /// Create an iterator that classifies batches from the input iterator.
    ///
    /// # Arguments
    ///
    /// * `input` - Iterator yielding input RecordBatches
    ///
    /// # Returns
    ///
    /// An iterator yielding result RecordBatches
    pub fn classify_iter<I>(
        &self,
        input: I,
    ) -> impl Iterator<Item = Result<RecordBatch, ArrowClassifyError>> + '_
    where
        I: Iterator<Item = Result<RecordBatch, arrow::error::ArrowError>> + 'a,
    {
        input.map(move |batch_result| {
            let batch = batch_result.map_err(ArrowClassifyError::from)?;
            self.classify_batch(&batch)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MinimizerWorkspace;
    use arrow::array::{BinaryArray, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    use super::super::schema::{COL_ID, COL_SEQUENCE};

    /// Helper to generate a DNA sequence of given length.
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

    /// Create a simple test index with one bucket.
    fn create_test_index() -> Index {
        let mut index = Index::new(16, 5, 0x12345).unwrap();
        let mut ws = MinimizerWorkspace::new();

        // Add a reference sequence
        let ref_seq = generate_sequence(100, 0);
        index.add_record(1, "ref1", &ref_seq, &mut ws);
        index.finalize_bucket(1);
        index.bucket_names.insert(1, "test_bucket".into());

        index
    }

    #[test]
    fn test_stream_classifier_single_batch() {
        let index = create_test_index();
        let classifier = IndexStreamClassifier::new(&index, None, 0.0);

        // Create a query that matches the reference
        let query_seq = generate_sequence(100, 0);
        let batch = make_test_batch(&[1], &[&query_seq]);

        let result = classifier.classify_batch(&batch).unwrap();

        // Should have at least one hit (query matches reference)
        assert!(result.num_rows() > 0, "Should have classification results");
    }

    #[test]
    fn test_stream_classifier_multiple_batches() {
        let index = create_test_index();
        let classifier = IndexStreamClassifier::new(&index, None, 0.0);

        // Create multiple batches
        let query_seq1 = generate_sequence(100, 0);
        let query_seq2 = generate_sequence(100, 1);
        let batch1 = make_test_batch(&[1], &[&query_seq1]);
        let batch2 = make_test_batch(&[2], &[&query_seq2]);

        let input_batches: Vec<Result<RecordBatch, arrow::error::ArrowError>> =
            vec![Ok(batch1), Ok(batch2)];

        let results: Vec<_> = classifier
            .classify_iter(input_batches.into_iter())
            .collect();

        assert_eq!(results.len(), 2);
        assert!(results[0].is_ok());
        assert!(results[1].is_ok());
    }

    #[test]
    fn test_stream_classifier_empty_input() {
        let index = create_test_index();
        let classifier = IndexStreamClassifier::new(&index, None, 0.1);

        let input_batches: Vec<Result<RecordBatch, arrow::error::ArrowError>> = vec![];

        let results: Vec<_> = classifier
            .classify_iter(input_batches.into_iter())
            .collect();

        assert!(results.is_empty());
    }

    #[test]
    fn test_stream_classifier_empty_batch() {
        let index = create_test_index();
        let classifier = IndexStreamClassifier::new(&index, None, 0.1);

        let schema = Arc::new(Schema::new(vec![
            Field::new(COL_ID, DataType::Int64, false),
            Field::new(COL_SEQUENCE, DataType::Binary, false),
        ]));
        let empty_batch = RecordBatch::new_empty(schema);

        let result = classifier.classify_batch(&empty_batch).unwrap();
        assert_eq!(result.num_rows(), 0);
    }

    #[test]
    fn test_stream_classifier_output_schema() {
        let index = create_test_index();
        let classifier = IndexStreamClassifier::new(&index, None, 0.1);

        let schema = classifier.output_schema();

        assert_eq!(schema.fields().len(), 3);
        assert_eq!(schema.field(0).name(), "query_id");
        assert_eq!(schema.field(1).name(), "bucket_id");
        assert_eq!(schema.field(2).name(), "score");
    }

    #[test]
    fn test_stream_classifier_threshold_filtering() {
        let index = create_test_index();

        // With very high threshold, should get no results
        let classifier_high = IndexStreamClassifier::new(&index, None, 1.1);
        let query_seq = generate_sequence(100, 0);
        let batch = make_test_batch(&[1], &[&query_seq]);

        let result_high = classifier_high.classify_batch(&batch).unwrap();
        assert_eq!(
            result_high.num_rows(),
            0,
            "High threshold should filter all"
        );

        // With zero threshold, should get results
        let classifier_low = IndexStreamClassifier::new(&index, None, 0.0);
        let result_low = classifier_low.classify_batch(&batch).unwrap();
        assert!(result_low.num_rows() > 0, "Zero threshold should pass some");
    }

    #[test]
    fn test_stream_classifier_error_propagation() {
        let index = create_test_index();
        let classifier = IndexStreamClassifier::new(&index, None, 0.1);

        // Create an error in the input stream
        let error = arrow::error::ArrowError::InvalidArgumentError("test error".into());
        let input_batches: Vec<Result<RecordBatch, arrow::error::ArrowError>> = vec![Err(error)];

        let results: Vec<_> = classifier
            .classify_iter(input_batches.into_iter())
            .collect();

        assert_eq!(results.len(), 1);
        assert!(results[0].is_err());
    }
}
