//! Arrow-based minimizer extraction.
//!
//! This module provides batch extraction of minimizer hashes and positions
//! from Arrow RecordBatches, producing output RecordBatches with `List<UInt64>`
//! columns for variable-length minimizer arrays.
//!
//! # Output Schemas
//!
//! ## Minimizer Set
//! | Column    | Type           | Description                          |
//! |-----------|----------------|--------------------------------------|
//! | id        | Int64          | Query identifier from input          |
//! | fwd_set   | List\<UInt64\> | Sorted, deduplicated forward hashes  |
//! | rc_set    | List\<UInt64\> | Sorted, deduplicated RC hashes       |
//!
//! ## Strand Minimizers
//! | Column         | Type           | Description                     |
//! |----------------|----------------|---------------------------------|
//! | id             | Int64          | Query identifier from input     |
//! | fwd_hashes     | List\<UInt64\> | Forward strand hash values      |
//! | fwd_positions  | List\<UInt64\> | Forward strand positions        |
//! | rc_hashes      | List\<UInt64\> | RC strand hash values           |
//! | rc_positions   | List\<UInt64\> | RC strand positions             |

use arrow::array::{ArrayRef, Int64Array, ListBuilder, UInt64Builder};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

use super::error::ArrowClassifyError;
use super::input::batch_to_records_with_columns;
use super::schema::COL_ID;
use crate::MinimizerWorkspace;

// Column names for extraction output
const COL_FWD_SET: &str = "fwd_set";
const COL_RC_SET: &str = "rc_set";
const COL_FWD_HASHES: &str = "fwd_hashes";
const COL_FWD_POSITIONS: &str = "fwd_positions";
const COL_RC_HASHES: &str = "rc_hashes";
const COL_RC_POSITIONS: &str = "rc_positions";

/// Returns the output schema for minimizer set extraction.
///
/// Schema: `id` (Int64), `fwd_set` (List\<UInt64\>), `rc_set` (List\<UInt64\>)
pub fn minimizer_set_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(COL_ID, DataType::Int64, false),
        Field::new(
            COL_FWD_SET,
            DataType::List(Arc::new(Field::new("item", DataType::UInt64, true))),
            false,
        ),
        Field::new(
            COL_RC_SET,
            DataType::List(Arc::new(Field::new("item", DataType::UInt64, true))),
            false,
        ),
    ]))
}

/// Returns the output schema for strand minimizers extraction.
///
/// Schema: `id` (Int64), `fwd_hashes` (List\<UInt64\>), `fwd_positions` (List\<UInt64\>),
/// `rc_hashes` (List\<UInt64\>), `rc_positions` (List\<UInt64\>)
pub fn strand_minimizers_schema() -> SchemaRef {
    let list_field = Arc::new(Field::new("item", DataType::UInt64, true));
    Arc::new(Schema::new(vec![
        Field::new(COL_ID, DataType::Int64, false),
        Field::new(COL_FWD_HASHES, DataType::List(list_field.clone()), false),
        Field::new(COL_FWD_POSITIONS, DataType::List(list_field.clone()), false),
        Field::new(COL_RC_HASHES, DataType::List(list_field.clone()), false),
        Field::new(COL_RC_POSITIONS, DataType::List(list_field), false),
    ]))
}

/// Append a `&[u64]` slice as a single list entry to a `ListBuilder<UInt64Builder>`.
fn append_u64_list(builder: &mut ListBuilder<UInt64Builder>, values: &[u64]) {
    let values_builder = builder.values();
    for &v in values {
        values_builder.append_value(v);
    }
    builder.append(true);
}

/// Extract sorted, deduplicated minimizer sets from an Arrow RecordBatch.
///
/// For each row, extracts forward and reverse complement minimizer hash sets
/// using `extract_minimizer_set()`.
///
/// # Input Schema
/// Must contain `id` (Int64) and `sequence` (Binary/LargeBinary) columns.
///
/// # Output Schema
/// `id` (Int64), `fwd_set` (List\<UInt64\>), `rc_set` (List\<UInt64\>)
pub fn extract_minimizer_set_batch(
    batch: &RecordBatch,
    k: usize,
    w: usize,
    salt: u64,
) -> Result<RecordBatch, ArrowClassifyError> {
    let schema = minimizer_set_schema();

    // Extract records from input batch using standard column names
    let records = batch_to_records_with_columns(batch, COL_ID, "sequence", None)?;

    if records.is_empty() {
        return Ok(RecordBatch::new_empty(schema));
    }

    let num_rows = records.len();
    let mut ids = Vec::with_capacity(num_rows);
    let mut fwd_builder = ListBuilder::new(UInt64Builder::new());
    let mut rc_builder = ListBuilder::new(UInt64Builder::new());

    let mut ws = MinimizerWorkspace::new();

    for (id, seq, _pair) in &records {
        ids.push(*id);
        let (fwd, rc) = crate::extract_minimizer_set(seq, k, w, salt, &mut ws);
        append_u64_list(&mut fwd_builder, &fwd);
        append_u64_list(&mut rc_builder, &rc);
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ids)) as ArrayRef,
            Arc::new(fwd_builder.finish()) as ArrayRef,
            Arc::new(rc_builder.finish()) as ArrayRef,
        ],
    )
    .map_err(|e| ArrowClassifyError::Classification(format!("Failed to build RecordBatch: {}", e)))
}

/// Extract ordered minimizers with positions from an Arrow RecordBatch.
///
/// For each row, extracts forward and reverse complement minimizer hashes
/// and their positions using `extract_strand_minimizers()`.
///
/// # Input Schema
/// Must contain `id` (Int64) and `sequence` (Binary/LargeBinary) columns.
///
/// # Output Schema
/// `id` (Int64), `fwd_hashes` (List\<UInt64\>), `fwd_positions` (List\<UInt64\>),
/// `rc_hashes` (List\<UInt64\>), `rc_positions` (List\<UInt64\>)
pub fn extract_strand_minimizers_batch(
    batch: &RecordBatch,
    k: usize,
    w: usize,
    salt: u64,
) -> Result<RecordBatch, ArrowClassifyError> {
    let schema = strand_minimizers_schema();

    let records = batch_to_records_with_columns(batch, COL_ID, "sequence", None)?;

    if records.is_empty() {
        return Ok(RecordBatch::new_empty(schema));
    }

    let num_rows = records.len();
    let mut ids = Vec::with_capacity(num_rows);
    let mut fwd_hashes_builder = ListBuilder::new(UInt64Builder::new());
    let mut fwd_positions_builder = ListBuilder::new(UInt64Builder::new());
    let mut rc_hashes_builder = ListBuilder::new(UInt64Builder::new());
    let mut rc_positions_builder = ListBuilder::new(UInt64Builder::new());

    let mut ws = MinimizerWorkspace::new();

    for (id, seq, _pair) in &records {
        ids.push(*id);
        let (fwd, rc) = crate::extract_strand_minimizers(seq, k, w, salt, &mut ws);

        append_u64_list(&mut fwd_hashes_builder, &fwd.hashes);
        // Convert positions from usize to u64
        let fwd_pos_u64: Vec<u64> = fwd.positions.into_iter().map(|p| p as u64).collect();
        append_u64_list(&mut fwd_positions_builder, &fwd_pos_u64);

        append_u64_list(&mut rc_hashes_builder, &rc.hashes);
        let rc_pos_u64: Vec<u64> = rc.positions.into_iter().map(|p| p as u64).collect();
        append_u64_list(&mut rc_positions_builder, &rc_pos_u64);
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ids)) as ArrayRef,
            Arc::new(fwd_hashes_builder.finish()) as ArrayRef,
            Arc::new(fwd_positions_builder.finish()) as ArrayRef,
            Arc::new(rc_hashes_builder.finish()) as ArrayRef,
            Arc::new(rc_positions_builder.finish()) as ArrayRef,
        ],
    )
    .map_err(|e| ArrowClassifyError::Classification(format!("Failed to build RecordBatch: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{BinaryArray, Int64Array, ListArray, UInt64Array};
    use arrow::datatypes::{DataType, Field, Schema};

    /// Helper to create a test batch with sequences.
    fn make_test_batch(ids: &[i64], seqs: &[&[u8]]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new(COL_ID, DataType::Int64, false),
            Field::new("sequence", DataType::Binary, false),
        ]));

        let id_array = Int64Array::from(ids.to_vec());
        let seq_array = BinaryArray::from_iter_values(seqs.iter().copied());

        RecordBatch::try_new(schema, vec![Arc::new(id_array), Arc::new(seq_array)]).unwrap()
    }

    // =========================================================================
    // Minimizer Set Tests
    // =========================================================================

    #[test]
    fn test_minimizer_set_output_schema() {
        let schema = minimizer_set_schema();
        assert_eq!(schema.fields().len(), 3);

        let id_field = schema.field_with_name(COL_ID).unwrap();
        assert_eq!(id_field.data_type(), &DataType::Int64);

        let fwd_field = schema.field_with_name(COL_FWD_SET).unwrap();
        assert!(matches!(fwd_field.data_type(), DataType::List(_)));

        let rc_field = schema.field_with_name(COL_RC_SET).unwrap();
        assert!(matches!(rc_field.data_type(), DataType::List(_)));
    }

    #[test]
    fn test_extract_minimizer_set_batch_basic() {
        let seq1 = b"AAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCC";
        let seq2 = b"AAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCC";
        let batch = make_test_batch(&[1, 2], &[seq1 as &[u8], seq2]);

        let result = extract_minimizer_set_batch(&batch, 16, 5, 0).unwrap();

        assert_eq!(result.num_rows(), 2);
        assert_eq!(result.num_columns(), 3);

        // Check that list columns have non-zero lengths
        let fwd_list = result
            .column(1)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let rc_list = result
            .column(2)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();

        // First row should have minimizers (67 bp, k=16, w=5)
        assert!(
            fwd_list.value(0).len() > 0,
            "Row 0 fwd set should be non-empty"
        );
        assert!(
            rc_list.value(0).len() > 0,
            "Row 0 rc set should be non-empty"
        );

        // Second row too (46 bp > k+w-1=20)
        assert!(
            fwd_list.value(1).len() > 0,
            "Row 1 fwd set should be non-empty"
        );
    }

    #[test]
    fn test_extract_minimizer_set_batch_empty() {
        let schema = Arc::new(Schema::new(vec![
            Field::new(COL_ID, DataType::Int64, false),
            Field::new("sequence", DataType::Binary, false),
        ]));
        let empty_batch = RecordBatch::new_empty(schema);

        let result = extract_minimizer_set_batch(&empty_batch, 16, 5, 0).unwrap();
        assert_eq!(result.num_rows(), 0);
    }

    #[test]
    fn test_extract_minimizer_set_batch_short_seq() {
        // Sequence shorter than k=16 → empty lists
        let batch = make_test_batch(&[1], &[b"ACGT"]);
        let result = extract_minimizer_set_batch(&batch, 16, 5, 0).unwrap();

        assert_eq!(result.num_rows(), 1);

        let fwd_list = result
            .column(1)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        assert_eq!(
            fwd_list.value(0).len(),
            0,
            "Short seq should produce empty fwd set"
        );

        let rc_list = result
            .column(2)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        assert_eq!(
            rc_list.value(0).len(),
            0,
            "Short seq should produce empty rc set"
        );
    }

    #[test]
    fn test_extract_minimizer_set_batch_sorted() {
        let seq = b"AAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCC";
        let batch = make_test_batch(&[1], &[seq as &[u8]]);

        let result = extract_minimizer_set_batch(&batch, 16, 5, 0).unwrap();

        let fwd_list = result
            .column(1)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let fwd_value_arr = fwd_list.value(0);
        let fwd_values = fwd_value_arr
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();

        // Verify sorted
        for i in 1..fwd_values.len() {
            assert!(
                fwd_values.value(i - 1) < fwd_values.value(i),
                "Forward set not strictly sorted at index {}",
                i
            );
        }
    }

    // =========================================================================
    // Strand Minimizers Tests
    // =========================================================================

    #[test]
    fn test_strand_minimizers_output_schema() {
        let schema = strand_minimizers_schema();
        assert_eq!(schema.fields().len(), 5);

        assert_eq!(
            schema.field_with_name(COL_ID).unwrap().data_type(),
            &DataType::Int64
        );
        assert!(matches!(
            schema.field_with_name(COL_FWD_HASHES).unwrap().data_type(),
            DataType::List(_)
        ));
        assert!(matches!(
            schema
                .field_with_name(COL_FWD_POSITIONS)
                .unwrap()
                .data_type(),
            DataType::List(_)
        ));
        assert!(matches!(
            schema.field_with_name(COL_RC_HASHES).unwrap().data_type(),
            DataType::List(_)
        ));
        assert!(matches!(
            schema
                .field_with_name(COL_RC_POSITIONS)
                .unwrap()
                .data_type(),
            DataType::List(_)
        ));
    }

    #[test]
    fn test_extract_strand_minimizers_batch_basic() {
        let seq1 = b"AAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCC";
        let seq2 = b"AAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCC";
        let batch = make_test_batch(&[1, 2], &[seq1 as &[u8], seq2]);

        let result = extract_strand_minimizers_batch(&batch, 16, 5, 0).unwrap();

        assert_eq!(result.num_rows(), 2);
        assert_eq!(result.num_columns(), 5);

        // Check fwd_hashes and fwd_positions have same length per row
        let fwd_hashes = result
            .column(1)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let fwd_positions = result
            .column(2)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();

        for row in 0..2 {
            assert_eq!(
                fwd_hashes.value(row).len(),
                fwd_positions.value(row).len(),
                "Row {} fwd_hashes.len != fwd_positions.len",
                row
            );
        }

        // Check non-empty
        assert!(
            fwd_hashes.value(0).len() > 0,
            "Row 0 fwd should be non-empty"
        );
    }

    #[test]
    fn test_extract_strand_minimizers_batch_positions_inbounds() {
        let seq = b"AAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCC";
        let seq_len = seq.len();
        let k: usize = 16;
        let batch = make_test_batch(&[1], &[seq as &[u8]]);

        let result = extract_strand_minimizers_batch(&batch, k, 5, 0).unwrap();

        // Check forward positions
        let fwd_positions = result
            .column(2)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let fwd_pos_arr = fwd_positions.value(0);
        let fwd_pos_values = fwd_pos_arr.as_any().downcast_ref::<UInt64Array>().unwrap();

        for i in 0..fwd_pos_values.len() {
            let pos = fwd_pos_values.value(i) as usize;
            assert!(
                pos + k <= seq_len,
                "Forward position {} out of bounds (pos + k = {}, seq_len = {})",
                pos,
                pos + k,
                seq_len
            );
        }

        // Check RC positions
        let rc_positions = result
            .column(4)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let rc_pos_arr = rc_positions.value(0);
        let rc_pos_values = rc_pos_arr.as_any().downcast_ref::<UInt64Array>().unwrap();

        for i in 0..rc_pos_values.len() {
            let pos = rc_pos_values.value(i) as usize;
            assert!(pos + k <= seq_len, "RC position {} out of bounds", pos);
        }
    }

    #[test]
    fn test_extract_strand_minimizers_batch_empty() {
        let schema = Arc::new(Schema::new(vec![
            Field::new(COL_ID, DataType::Int64, false),
            Field::new("sequence", DataType::Binary, false),
        ]));
        let empty_batch = RecordBatch::new_empty(schema);

        let result = extract_strand_minimizers_batch(&empty_batch, 16, 5, 0).unwrap();
        assert_eq!(result.num_rows(), 0);
    }
}
