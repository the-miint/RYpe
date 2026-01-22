//! Zero-copy conversion from Arrow RecordBatch to QueryRecord.
//!
//! This module provides functions to convert Arrow columnar data to the
//! `QueryRecord` format expected by classification functions, with zero-copy
//! semantics where possible.
//!
//! # Zero-Copy Guarantee
//!
//! The `batch_to_records()` function returns `QueryRecord` tuples containing
//! references (`&[u8]`) that point directly into the Arrow buffer memory.
//! No data is copied during conversion. The returned records are only valid
//! while the source `RecordBatch` is alive.
//!
//! # Example
//!
//! ```ignore
//! let batch: RecordBatch = /* from DuckDB, Parquet, etc. */;
//! let records = batch_to_records(&batch)?;
//! // records[i].1 points directly into batch's memory
//! let results = classify_batch(&index, None, &records, threshold);
//! // batch must remain alive until classification is complete
//! ```

use arrow::array::{
    Array, BinaryArray, Int64Array, LargeBinaryArray, LargeStringArray, StringArray,
};
use arrow::record_batch::RecordBatch;

use super::error::{ArrowClassifyError, MAX_SEQUENCE_LENGTH};
use super::schema::{validate_input_schema, COL_ID, COL_PAIR_SEQUENCE, COL_SEQUENCE};
use crate::QueryRecord;

/// Trait for uniform access to Binary and LargeBinary arrays.
///
/// This trait enables monomorphization: when the concrete type is known at compile time,
/// the methods are inlined directly without virtual dispatch overhead.
trait BinaryArrayAccess<'a>: Sized {
    /// Get the byte slice at index i.
    fn value_at(&self, i: usize) -> &'a [u8];

    /// Check if the value at index i is null.
    fn is_null_at(&self, i: usize) -> bool;
}

impl<'a> BinaryArrayAccess<'a> for &'a BinaryArray {
    #[inline]
    fn value_at(&self, i: usize) -> &'a [u8] {
        self.value(i)
    }

    #[inline]
    fn is_null_at(&self, i: usize) -> bool {
        self.is_null(i)
    }
}

impl<'a> BinaryArrayAccess<'a> for &'a LargeBinaryArray {
    #[inline]
    fn value_at(&self, i: usize) -> &'a [u8] {
        self.value(i)
    }

    #[inline]
    fn is_null_at(&self, i: usize) -> bool {
        self.is_null(i)
    }
}

impl<'a> BinaryArrayAccess<'a> for &'a StringArray {
    #[inline]
    fn value_at(&self, i: usize) -> &'a [u8] {
        self.value(i).as_bytes()
    }

    #[inline]
    fn is_null_at(&self, i: usize) -> bool {
        self.is_null(i)
    }
}

impl<'a> BinaryArrayAccess<'a> for &'a LargeStringArray {
    #[inline]
    fn value_at(&self, i: usize) -> &'a [u8] {
        self.value(i).as_bytes()
    }

    #[inline]
    fn is_null_at(&self, i: usize) -> bool {
        self.is_null(i)
    }
}

/// Abstraction over Binary, LargeBinary, String, and LargeString arrays for uniform access.
///
/// Uses an enum internally but with `#[inline]` hints to allow the compiler
/// to optimize away the match when processing uniform batches.
enum BinaryColumnRef<'a> {
    Binary(&'a BinaryArray),
    LargeBinary(&'a LargeBinaryArray),
    String(&'a StringArray),
    LargeString(&'a LargeStringArray),
}

impl<'a> BinaryColumnRef<'a> {
    /// Get the byte slice at index i.
    #[inline]
    fn value(&self, i: usize) -> &'a [u8] {
        match self {
            BinaryColumnRef::Binary(arr) => arr.value_at(i),
            BinaryColumnRef::LargeBinary(arr) => arr.value_at(i),
            BinaryColumnRef::String(arr) => arr.value_at(i),
            BinaryColumnRef::LargeString(arr) => arr.value_at(i),
        }
    }

    /// Check if the value at index i is null.
    #[inline]
    fn is_null(&self, i: usize) -> bool {
        match self {
            BinaryColumnRef::Binary(arr) => arr.is_null_at(i),
            BinaryColumnRef::LargeBinary(arr) => arr.is_null_at(i),
            BinaryColumnRef::String(arr) => arr.is_null_at(i),
            BinaryColumnRef::LargeString(arr) => arr.is_null_at(i),
        }
    }
}

/// Extract a binary or string column from a RecordBatch by index.
///
/// Supports Binary, LargeBinary, Utf8 (String), and LargeUtf8 (LargeString) arrays.
/// String arrays are converted to byte slices via `.as_bytes()` (valid for ASCII DNA sequences).
fn get_binary_column(
    batch: &RecordBatch,
    col_idx: usize,
) -> Result<BinaryColumnRef<'_>, ArrowClassifyError> {
    let column = batch.column(col_idx);

    if let Some(arr) = column.as_any().downcast_ref::<BinaryArray>() {
        return Ok(BinaryColumnRef::Binary(arr));
    }

    if let Some(arr) = column.as_any().downcast_ref::<LargeBinaryArray>() {
        return Ok(BinaryColumnRef::LargeBinary(arr));
    }

    if let Some(arr) = column.as_any().downcast_ref::<StringArray>() {
        return Ok(BinaryColumnRef::String(arr));
    }

    if let Some(arr) = column.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(BinaryColumnRef::LargeString(arr));
    }

    let schema = batch.schema();
    let field = schema.field(col_idx);
    Err(ArrowClassifyError::TypeError {
        column: field.name().clone(),
        expected: "Binary, LargeBinary, Utf8, or LargeUtf8".into(),
        actual: format!("{:?}", column.data_type()),
    })
}

/// Convert an Arrow RecordBatch to a vector of QueryRecords with zero-copy semantics.
///
/// # Zero-Copy Guarantee
///
/// The returned `QueryRecord` references point directly into the Arrow buffers.
/// No sequence data is copied. The lifetime of the returned records is tied to
/// the input `RecordBatch`.
///
/// # Arguments
///
/// * `batch` - A RecordBatch with the expected input schema (see `validate_input_schema`)
///
/// # Returns
///
/// A vector of `QueryRecord` tuples: `(id, sequence, optional_pair)`
///
/// # Errors
///
/// Returns an error if:
/// - Schema validation fails
/// - A required column has an unexpected null value
/// - Type conversion fails
pub fn batch_to_records(batch: &RecordBatch) -> Result<Vec<QueryRecord<'_>>, ArrowClassifyError> {
    // Validate schema first
    validate_input_schema(batch.schema().as_ref())?;
    // Delegate to the parameterized version with standard column names
    batch_to_records_with_columns(batch, COL_ID, COL_SEQUENCE, Some(COL_PAIR_SEQUENCE))
}

/// Convert an Arrow RecordBatch to a vector of QueryRecords with custom column names.
///
/// This is the parameterized version that allows custom column names for different
/// input schemas (e.g., Parquet files with different column naming conventions).
///
/// # Zero-Copy Guarantee
///
/// The returned `QueryRecord` references point directly into the Arrow buffers.
/// No sequence data is copied. The lifetime of the returned records is tied to
/// the input `RecordBatch`.
///
/// # Arguments
///
/// * `batch` - A RecordBatch containing the required columns
/// * `id_col` - Name of the ID column (must be Int64)
/// * `seq_col` - Name of the primary sequence column (must be Binary or LargeBinary)
/// * `pair_col` - Optional name of the paired sequence column (must be Binary or LargeBinary if present)
///
/// # Returns
///
/// A vector of `QueryRecord` tuples: `(id, sequence, optional_pair)`
///
/// # Errors
///
/// Returns an error if:
/// - A required column is not found
/// - A column has an unexpected type
/// - A required column has an unexpected null value
pub fn batch_to_records_with_columns<'a>(
    batch: &'a RecordBatch,
    id_col: &str,
    seq_col: &str,
    pair_col: Option<&str>,
) -> Result<Vec<QueryRecord<'a>>, ArrowClassifyError> {
    let num_rows = batch.num_rows();
    if num_rows == 0 {
        return Ok(Vec::new());
    }

    // Get column indices
    let id_idx = batch
        .schema()
        .index_of(id_col)
        .map_err(|_| ArrowClassifyError::ColumnNotFound(id_col.into()))?;
    let seq_idx = batch
        .schema()
        .index_of(seq_col)
        .map_err(|_| ArrowClassifyError::ColumnNotFound(seq_col.into()))?;
    let pair_idx = pair_col.and_then(|col| batch.schema().index_of(col).ok());

    // Get typed column references
    let ids = batch
        .column(id_idx)
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| ArrowClassifyError::TypeError {
            column: id_col.into(),
            expected: "Int64".into(),
            actual: format!("{:?}", batch.column(id_idx).data_type()),
        })?;

    let seqs = get_binary_column(batch, seq_idx).map_err(|e| match e {
        ArrowClassifyError::TypeError {
            expected, actual, ..
        } => ArrowClassifyError::TypeError {
            column: seq_col.into(),
            expected,
            actual,
        },
        other => other,
    })?;

    let pairs = pair_idx
        .map(|idx| {
            get_binary_column(batch, idx).map_err(|e| match e {
                ArrowClassifyError::TypeError {
                    expected, actual, ..
                } => ArrowClassifyError::TypeError {
                    column: pair_col.unwrap_or("pair").into(),
                    expected,
                    actual,
                },
                other => other,
            })
        })
        .transpose()?;

    // Build records with zero-copy references
    let mut records = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        // Check for null ID
        if ids.is_null(i) {
            return Err(ArrowClassifyError::NullError {
                column: id_col.into(),
                row: i,
            });
        }
        let id = ids.value(i);

        // Check for null sequence
        if seqs.is_null(i) {
            return Err(ArrowClassifyError::NullError {
                column: seq_col.into(),
                row: i,
            });
        }
        let seq: &[u8] = seqs.value(i);

        // Validate sequence length (2GB limit, matching C API)
        if seq.len() > MAX_SEQUENCE_LENGTH {
            return Err(ArrowClassifyError::SequenceTooLong {
                row: i,
                length: seq.len(),
                max_length: MAX_SEQUENCE_LENGTH,
            });
        }

        // Handle optional pair sequence (null is allowed)
        let pair: Option<&[u8]> = match pairs.as_ref() {
            Some(p) if !p.is_null(i) => {
                let pair_seq = p.value(i);
                // Validate paired sequence length too
                if pair_seq.len() > MAX_SEQUENCE_LENGTH {
                    return Err(ArrowClassifyError::SequenceTooLong {
                        row: i,
                        length: pair_seq.len(),
                        max_length: MAX_SEQUENCE_LENGTH,
                    });
                }
                Some(pair_seq)
            }
            _ => None,
        };

        records.push((id, seq, pair));
    }

    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{BinaryArray, Int64Array, LargeBinaryArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    /// Helper to create a test batch with Binary arrays.
    fn make_test_batch(ids: &[i64], seqs: &[&[u8]]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new(COL_ID, DataType::Int64, false),
            Field::new(COL_SEQUENCE, DataType::Binary, false),
        ]));

        let id_array = Int64Array::from(ids.to_vec());
        let seq_array = BinaryArray::from_iter_values(seqs.iter().copied());

        RecordBatch::try_new(schema, vec![Arc::new(id_array), Arc::new(seq_array)]).unwrap()
    }

    /// Helper to create a test batch with paired sequences.
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

    #[test]
    fn test_batch_to_records_single_end() {
        let batch = make_test_batch(&[1, 2, 3], &[b"ACGT", b"TGCA", b"GGCC"]);
        let records = batch_to_records(&batch).unwrap();

        assert_eq!(records.len(), 3);
        assert_eq!(records[0].0, 1);
        assert_eq!(records[0].1, b"ACGT");
        assert!(records[0].2.is_none());

        assert_eq!(records[1].0, 2);
        assert_eq!(records[1].1, b"TGCA");

        assert_eq!(records[2].0, 3);
        assert_eq!(records[2].1, b"GGCC");
    }

    #[test]
    fn test_batch_to_records_paired_end() {
        let batch = make_test_batch_paired(
            &[1, 2],
            &[b"ACGT", b"TGCA"],
            &[Some(b"AAAA" as &[u8]), Some(b"TTTT")],
        );
        let records = batch_to_records(&batch).unwrap();

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].0, 1);
        assert_eq!(records[0].1, b"ACGT");
        assert_eq!(records[0].2, Some(b"AAAA" as &[u8]));

        assert_eq!(records[1].0, 2);
        assert_eq!(records[1].1, b"TGCA");
        assert_eq!(records[1].2, Some(b"TTTT" as &[u8]));
    }

    #[test]
    fn test_batch_to_records_null_pair_handling() {
        let batch = make_test_batch_paired(
            &[1, 2, 3],
            &[b"ACGT", b"TGCA", b"GGCC"],
            &[Some(b"AAAA" as &[u8]), None, Some(b"CCCC")],
        );
        let records = batch_to_records(&batch).unwrap();

        assert_eq!(records.len(), 3);
        assert_eq!(records[0].2, Some(b"AAAA" as &[u8]));
        assert!(records[1].2.is_none()); // Null pair is None
        assert_eq!(records[2].2, Some(b"CCCC" as &[u8]));
    }

    #[test]
    fn test_batch_to_records_large_binary_array() {
        let schema = Arc::new(Schema::new(vec![
            Field::new(COL_ID, DataType::Int64, false),
            Field::new(COL_SEQUENCE, DataType::LargeBinary, false),
        ]));

        let id_array = Int64Array::from(vec![1, 2]);
        let seq_array = LargeBinaryArray::from_iter_values([b"ACGT" as &[u8], b"TGCA"]);

        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(id_array), Arc::new(seq_array)]).unwrap();

        let records = batch_to_records(&batch).unwrap();

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].1, b"ACGT");
        assert_eq!(records[1].1, b"TGCA");
    }

    #[test]
    fn test_batch_to_records_empty_batch() {
        let schema = Arc::new(Schema::new(vec![
            Field::new(COL_ID, DataType::Int64, false),
            Field::new(COL_SEQUENCE, DataType::Binary, false),
        ]));

        let batch = RecordBatch::new_empty(schema);
        let records = batch_to_records(&batch).unwrap();

        assert!(records.is_empty());
    }

    #[test]
    fn test_batch_to_records_zero_copy_verification() {
        // Create batch with known data
        let seq1 = b"ACGTACGTACGT";
        let seq2 = b"TGCATGCATGCA";
        let batch = make_test_batch(&[1, 2], &[seq1 as &[u8], seq2]);

        let records = batch_to_records(&batch).unwrap();

        // Get pointers to the sequence data
        let record_ptr_1 = records[0].1.as_ptr();
        let record_ptr_2 = records[1].1.as_ptr();

        // Get pointer to the underlying Arrow buffer
        let seq_col = batch.column(1);
        let binary_arr = seq_col.as_any().downcast_ref::<BinaryArray>().unwrap();
        let arrow_ptr_1 = binary_arr.value(0).as_ptr();
        let arrow_ptr_2 = binary_arr.value(1).as_ptr();

        // Verify zero-copy: pointers should be identical
        assert_eq!(
            record_ptr_1, arrow_ptr_1,
            "Record should point directly into Arrow buffer"
        );
        assert_eq!(
            record_ptr_2, arrow_ptr_2,
            "Record should point directly into Arrow buffer"
        );
    }

    #[test]
    fn test_batch_to_records_null_id_error() {
        // Create a batch with nullable ID column containing a null
        let schema = Arc::new(Schema::new(vec![
            Field::new(COL_ID, DataType::Int64, true), // nullable
            Field::new(COL_SEQUENCE, DataType::Binary, false),
        ]));

        let id_array = Int64Array::from(vec![Some(1), None, Some(3)]);
        let seq_array = BinaryArray::from_iter_values([b"ACGT" as &[u8], b"TGCA", b"GGCC"]);

        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(id_array), Arc::new(seq_array)]).unwrap();

        let result = batch_to_records(&batch);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ArrowClassifyError::NullError { column, row } if column == COL_ID && row == 1
        ));
    }

    #[test]
    fn test_batch_to_records_null_sequence_error() {
        // Create a batch with nullable sequence column containing a null
        let schema = Arc::new(Schema::new(vec![
            Field::new(COL_ID, DataType::Int64, false),
            Field::new(COL_SEQUENCE, DataType::Binary, true), // nullable
        ]));

        let id_array = Int64Array::from(vec![1, 2, 3]);
        let seq_array = BinaryArray::from_iter([
            Some(b"ACGT" as &[u8]),
            Some(b"TGCA"),
            None, // null sequence
        ]);

        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(id_array), Arc::new(seq_array)]).unwrap();

        let result = batch_to_records(&batch);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ArrowClassifyError::NullError { column, row } if column == COL_SEQUENCE && row == 2
        ));
    }

    // -------------------------------------------------------------------------
    // Tests for batch_to_records_with_columns
    // -------------------------------------------------------------------------

    #[test]
    fn test_batch_to_records_with_columns_custom_names() {
        // Create batch with Parquet-style column names (read_id, sequence1, sequence2)
        let schema = Arc::new(Schema::new(vec![
            Field::new("read_id", DataType::Int64, false),
            Field::new("sequence1", DataType::Binary, false),
            Field::new("sequence2", DataType::Binary, true),
        ]));

        let id_array = Int64Array::from(vec![1, 2]);
        let seq_array = BinaryArray::from_iter_values([b"ACGT" as &[u8], b"TGCA"]);
        let pair_array = BinaryArray::from_iter([Some(b"AAAA" as &[u8]), Some(b"TTTT")]);

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(id_array),
                Arc::new(seq_array),
                Arc::new(pair_array),
            ],
        )
        .unwrap();

        // Test custom column mapping
        let records =
            batch_to_records_with_columns(&batch, "read_id", "sequence1", Some("sequence2"))
                .unwrap();

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].0, 1);
        assert_eq!(records[0].1, b"ACGT");
        assert_eq!(records[0].2, Some(b"AAAA" as &[u8]));

        assert_eq!(records[1].0, 2);
        assert_eq!(records[1].1, b"TGCA");
        assert_eq!(records[1].2, Some(b"TTTT" as &[u8]));
    }

    #[test]
    fn test_batch_to_records_with_columns_single_end() {
        // Test without pair column
        let schema = Arc::new(Schema::new(vec![
            Field::new("my_id", DataType::Int64, false),
            Field::new("my_seq", DataType::Binary, false),
        ]));

        let id_array = Int64Array::from(vec![1, 2, 3]);
        let seq_array = BinaryArray::from_iter_values([b"ACGT" as &[u8], b"TGCA", b"GGCC"]);

        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(id_array), Arc::new(seq_array)]).unwrap();

        let records = batch_to_records_with_columns(&batch, "my_id", "my_seq", None).unwrap();

        assert_eq!(records.len(), 3);
        assert_eq!(records[0].0, 1);
        assert_eq!(records[0].1, b"ACGT");
        assert!(records[0].2.is_none());

        assert_eq!(records[2].0, 3);
        assert_eq!(records[2].1, b"GGCC");
    }

    #[test]
    fn test_batch_to_records_with_columns_zero_copy() {
        // Verify zero-copy: pointers should point into Arrow buffer
        let schema = Arc::new(Schema::new(vec![
            Field::new("id_col", DataType::Int64, false),
            Field::new("seq_col", DataType::Binary, false),
        ]));

        let id_array = Int64Array::from(vec![1, 2]);
        let seq_array = BinaryArray::from_iter_values([b"ACGTACGTACGT" as &[u8], b"TGCATGCATGCA"]);

        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(id_array), Arc::new(seq_array)]).unwrap();

        let records = batch_to_records_with_columns(&batch, "id_col", "seq_col", None).unwrap();

        // Get pointer to the sequence data in records
        let record_ptr = records[0].1.as_ptr();

        // Get pointer to the underlying Arrow buffer
        let seq_col = batch.column(1);
        let binary_arr = seq_col.as_any().downcast_ref::<BinaryArray>().unwrap();
        let arrow_ptr = binary_arr.value(0).as_ptr();

        // Verify zero-copy: pointers should be identical
        assert_eq!(
            record_ptr, arrow_ptr,
            "Record should point directly into Arrow buffer"
        );
    }

    #[test]
    fn test_batch_to_records_with_columns_missing_id() {
        // Use a non-empty batch so we don't early-return before column lookup
        let schema = Arc::new(Schema::new(vec![
            Field::new("wrong_id", DataType::Int64, false),
            Field::new("sequence", DataType::Binary, false),
        ]));

        let id_array = Int64Array::from(vec![1]);
        let seq_array = BinaryArray::from_iter_values([b"ACGT" as &[u8]]);
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(id_array), Arc::new(seq_array)]).unwrap();

        let result = batch_to_records_with_columns(&batch, "id", "sequence", None);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ArrowClassifyError::ColumnNotFound(col) if col == "id"
        ));
    }

    #[test]
    fn test_batch_to_records_with_columns_missing_sequence() {
        // Use a non-empty batch so we don't early-return before column lookup
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("wrong_seq", DataType::Binary, false),
        ]));

        let id_array = Int64Array::from(vec![1]);
        let seq_array = BinaryArray::from_iter_values([b"ACGT" as &[u8]]);
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(id_array), Arc::new(seq_array)]).unwrap();

        let result = batch_to_records_with_columns(&batch, "id", "sequence", None);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ArrowClassifyError::ColumnNotFound(col) if col == "sequence"
        ));
    }

    #[test]
    fn test_batch_to_records_with_columns_wrong_id_type() {
        // Use a non-empty batch so we don't early-return before type check
        use arrow::array::StringArray;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false), // Wrong type
            Field::new("sequence", DataType::Binary, false),
        ]));

        let id_array = StringArray::from(vec!["1"]);
        let seq_array = BinaryArray::from_iter_values([b"ACGT" as &[u8]]);
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(id_array), Arc::new(seq_array)]).unwrap();

        let result = batch_to_records_with_columns(&batch, "id", "sequence", None);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ArrowClassifyError::TypeError { column, .. } if column == "id"
        ));
    }

    #[test]
    fn test_batch_to_records_with_columns_large_binary() {
        // Test with LargeBinary arrays
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("seq", DataType::LargeBinary, false),
            Field::new("pair", DataType::LargeBinary, true),
        ]));

        let id_array = Int64Array::from(vec![1, 2]);
        let seq_array = LargeBinaryArray::from_iter_values([b"ACGT" as &[u8], b"TGCA"]);
        let pair_array = LargeBinaryArray::from_iter([Some(b"AAAA" as &[u8]), None]); // Second is null

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(id_array),
                Arc::new(seq_array),
                Arc::new(pair_array),
            ],
        )
        .unwrap();

        let records = batch_to_records_with_columns(&batch, "id", "seq", Some("pair")).unwrap();

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].1, b"ACGT");
        assert_eq!(records[0].2, Some(b"AAAA" as &[u8]));
        assert_eq!(records[1].1, b"TGCA");
        assert!(records[1].2.is_none()); // Null pair is None
    }

    #[test]
    fn test_batch_to_records_with_columns_string_arrays() {
        // Test with Utf8 (String) arrays - used by Parquet input
        use arrow::array::StringArray;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("seq", DataType::Utf8, false),
            Field::new("pair", DataType::Utf8, true),
        ]));

        let id_array = Int64Array::from(vec![1, 2]);
        let seq_array = StringArray::from(vec!["ACGT", "TGCA"]);
        let pair_array = StringArray::from(vec![Some("AAAA"), None]);

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(id_array),
                Arc::new(seq_array),
                Arc::new(pair_array),
            ],
        )
        .unwrap();

        let records = batch_to_records_with_columns(&batch, "id", "seq", Some("pair")).unwrap();

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].0, 1);
        assert_eq!(records[0].1, b"ACGT");
        assert_eq!(records[0].2, Some(b"AAAA" as &[u8]));
        assert_eq!(records[1].0, 2);
        assert_eq!(records[1].1, b"TGCA");
        assert!(records[1].2.is_none()); // Null pair is None
    }

    #[test]
    fn test_batch_to_records_with_columns_string_zero_copy() {
        // Verify zero-copy for String arrays
        use arrow::array::StringArray;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("seq", DataType::Utf8, false),
        ]));

        let id_array = Int64Array::from(vec![1]);
        let seq_array = StringArray::from(vec!["ACGTACGTACGT"]);

        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(id_array), Arc::new(seq_array)]).unwrap();

        let records = batch_to_records_with_columns(&batch, "id", "seq", None).unwrap();

        // Get pointer to the sequence data in records
        let record_ptr = records[0].1.as_ptr();

        // Get pointer to the underlying Arrow buffer
        let seq_col = batch.column(1);
        let string_arr = seq_col.as_any().downcast_ref::<StringArray>().unwrap();
        let arrow_ptr = string_arr.value(0).as_bytes().as_ptr();

        // Verify zero-copy: pointers should be identical
        assert_eq!(
            record_ptr, arrow_ptr,
            "Record should point directly into Arrow buffer"
        );
    }

    #[test]
    fn test_batch_to_records_with_columns_large_string_arrays() {
        // Test with LargeUtf8 (LargeString) arrays
        use arrow::array::LargeStringArray;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("seq", DataType::LargeUtf8, false),
        ]));

        let id_array = Int64Array::from(vec![1, 2]);
        let seq_array = LargeStringArray::from(vec!["ACGT", "TGCA"]);

        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(id_array), Arc::new(seq_array)]).unwrap();

        let records = batch_to_records_with_columns(&batch, "id", "seq", None).unwrap();

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].1, b"ACGT");
        assert_eq!(records[1].1, b"TGCA");
    }

    #[test]
    fn test_batch_to_records_invalid_schema() {
        // Schema with wrong ID type
        let schema = Arc::new(Schema::new(vec![
            Field::new(COL_ID, DataType::Utf8, false),
            Field::new(COL_SEQUENCE, DataType::Binary, false),
        ]));

        let batch = RecordBatch::new_empty(schema);
        let result = batch_to_records(&batch);
        assert!(result.is_err());
    }
}
