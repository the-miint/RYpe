//! Arrow integration for clustering.
//!
//! # Output Schema (cluster result rows)
//!
//! | Column          | Arrow Type | Nullable | Description |
//! |-----------------|------------|----------|-------------|
//! | `rep_contig`    | Utf8       | No       | Representative contig id |
//! | `member_contig` | Utf8       | No       | Member contig id (equals rep_contig for representatives themselves) |
//! | `source_mag`    | Utf8       | Yes      | MAG/assembly id the member came from |
//! | `containment`   | Float64    | No       | Containment of member in representative (1.0 for representatives) |
//!
//! # Input Schema (contigs to be clustered)
//!
//! | Column       | Arrow Type | Nullable | Description |
//! |--------------|------------|----------|-------------|
//! | `contig_id`  | Utf8/LargeUtf8 | No   | Unique contig id |
//! | `source_mag` | Utf8/LargeUtf8 | Yes  | MAG/assembly id the contig came from |
//! | `sequence`   | Binary/LargeBinary/BinaryView/Utf8/LargeUtf8/Utf8View | No | DNA bytes |

use arrow::array::{Float64Array, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

use super::error::ArrowClassifyError;
use crate::cluster::ClusterResult;

/// Column name for representative contig id in cluster output.
pub const COL_REP_CONTIG: &str = "rep_contig";
/// Column name for member contig id in cluster output.
pub const COL_MEMBER_CONTIG: &str = "member_contig";
/// Column name for source MAG id (member's MAG) in cluster output. Nullable.
pub const COL_SOURCE_MAG: &str = "source_mag";
/// Column name for containment score in cluster output.
pub const COL_CONTAINMENT: &str = "containment";

/// Column name for contig id in cluster input batches.
pub const COL_CONTIG_ID: &str = "contig_id";
/// Column name for sequence bytes in cluster input batches.
pub const COL_CLUSTER_SEQUENCE: &str = "sequence";

/// Returns the schema for cluster result batches.
pub fn cluster_result_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(COL_REP_CONTIG, DataType::Utf8, false),
        Field::new(COL_MEMBER_CONTIG, DataType::Utf8, false),
        Field::new(COL_SOURCE_MAG, DataType::Utf8, true),
        Field::new(COL_CONTAINMENT, DataType::Float64, false),
    ]))
}

/// Convert a [`ClusterResult`] into an Arrow `RecordBatch` with
/// [`cluster_result_schema`].
///
/// The conversion copies all strings into Arrow buffers (unavoidable —
/// `ClusterRow` owns `String`s). For typical dereplication runs (one row per
/// input contig) this is bounded by the input size, not the all-vs-all space.
pub fn cluster_result_to_record_batch(
    result: &ClusterResult,
) -> Result<RecordBatch, ArrowClassifyError> {
    let schema = cluster_result_schema();

    if result.rows.is_empty() {
        return Ok(RecordBatch::new_empty(schema));
    }

    let n = result.rows.len();
    let mut rep = StringBuilder::with_capacity(n, n * 16);
    let mut member = StringBuilder::with_capacity(n, n * 16);
    let mut mag = StringBuilder::with_capacity(n, n * 16);
    let mut containment = Vec::with_capacity(n);

    for row in &result.rows {
        rep.append_value(&row.rep_contig);
        member.append_value(&row.member_contig);
        match &row.source_mag {
            Some(m) => mag.append_value(m),
            None => mag.append_null(),
        }
        containment.push(row.containment);
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(rep.finish()) as Arc<dyn arrow::array::Array>,
            Arc::new(member.finish()),
            Arc::new(mag.finish()),
            Arc::new(Float64Array::from(containment)),
        ],
    )
    .map_err(ArrowClassifyError::from)
}

/// Helper so callers can construct an empty result batch with the correct
/// schema without going through a `ClusterResult`.
pub fn empty_cluster_result_batch() -> RecordBatch {
    RecordBatch::new_empty(cluster_result_schema())
}

/// Check if a DataType is a valid string type for contig ids / source MAGs.
fn is_string_type(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
    )
}

/// Check if a DataType is a valid sequence type (mirrors classify input).
fn is_sequence_type(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Binary
            | DataType::LargeBinary
            | DataType::BinaryView
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Utf8View
    )
}

/// Validate that a schema matches the expected cluster-input schema.
///
/// # Requirements
/// - `contig_id`: Utf8 / LargeUtf8 / Utf8View, non-nullable
/// - `source_mag`: Utf8 / LargeUtf8 / Utf8View, nullable (optional column)
/// - `sequence`: any binary or string type, non-nullable
pub fn validate_cluster_input_schema(schema: &Schema) -> Result<(), ArrowClassifyError> {
    match schema.column_with_name(COL_CONTIG_ID) {
        Some((_, field)) => {
            if !is_string_type(field.data_type()) {
                return Err(ArrowClassifyError::TypeError {
                    column: COL_CONTIG_ID.into(),
                    expected: "Utf8, LargeUtf8, or Utf8View".into(),
                    actual: format!("{:?}", field.data_type()),
                });
            }
        }
        None => return Err(ArrowClassifyError::ColumnNotFound(COL_CONTIG_ID.into())),
    }

    if let Some((_, field)) = schema.column_with_name(COL_SOURCE_MAG) {
        if !is_string_type(field.data_type()) {
            return Err(ArrowClassifyError::TypeError {
                column: COL_SOURCE_MAG.into(),
                expected: "Utf8, LargeUtf8, or Utf8View".into(),
                actual: format!("{:?}", field.data_type()),
            });
        }
    }

    match schema.column_with_name(COL_CLUSTER_SEQUENCE) {
        Some((_, field)) => {
            if !is_sequence_type(field.data_type()) {
                return Err(ArrowClassifyError::TypeError {
                    column: COL_CLUSTER_SEQUENCE.into(),
                    expected: "Binary, LargeBinary, BinaryView, Utf8, LargeUtf8, or Utf8View"
                        .into(),
                    actual: format!("{:?}", field.data_type()),
                });
            }
        }
        None => {
            return Err(ArrowClassifyError::ColumnNotFound(
                COL_CLUSTER_SEQUENCE.into(),
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_schema_has_four_columns_in_documented_order() {
        let schema = cluster_result_schema();
        assert_eq!(schema.fields().len(), 4);
        assert_eq!(schema.field(0).name(), COL_REP_CONTIG);
        assert_eq!(schema.field(1).name(), COL_MEMBER_CONTIG);
        assert_eq!(schema.field(2).name(), COL_SOURCE_MAG);
        assert_eq!(schema.field(3).name(), COL_CONTAINMENT);
    }

    #[test]
    fn output_schema_types_and_nullability_match_docs() {
        let schema = cluster_result_schema();
        assert_eq!(schema.field(0).data_type(), &DataType::Utf8);
        assert!(!schema.field(0).is_nullable());
        assert_eq!(schema.field(1).data_type(), &DataType::Utf8);
        assert!(!schema.field(1).is_nullable());
        assert_eq!(schema.field(2).data_type(), &DataType::Utf8);
        assert!(schema.field(2).is_nullable(), "source_mag must be nullable");
        assert_eq!(schema.field(3).data_type(), &DataType::Float64);
        assert!(!schema.field(3).is_nullable());
    }

    #[test]
    fn input_schema_valid_minimum_columns() {
        let schema = Schema::new(vec![
            Field::new(COL_CONTIG_ID, DataType::Utf8, false),
            Field::new(COL_CLUSTER_SEQUENCE, DataType::Binary, false),
        ]);
        assert!(validate_cluster_input_schema(&schema).is_ok());
    }

    #[test]
    fn input_schema_valid_with_source_mag_and_large_types() {
        let schema = Schema::new(vec![
            Field::new(COL_CONTIG_ID, DataType::LargeUtf8, false),
            Field::new(COL_SOURCE_MAG, DataType::Utf8, true),
            Field::new(COL_CLUSTER_SEQUENCE, DataType::LargeBinary, false),
        ]);
        assert!(validate_cluster_input_schema(&schema).is_ok());
    }

    #[test]
    fn input_schema_missing_contig_id_rejected() {
        let schema = Schema::new(vec![Field::new(
            COL_CLUSTER_SEQUENCE,
            DataType::Binary,
            false,
        )]);
        assert!(matches!(
            validate_cluster_input_schema(&schema).unwrap_err(),
            ArrowClassifyError::ColumnNotFound(col) if col == COL_CONTIG_ID
        ));
    }

    #[test]
    fn input_schema_missing_sequence_rejected() {
        let schema = Schema::new(vec![Field::new(COL_CONTIG_ID, DataType::Utf8, false)]);
        assert!(matches!(
            validate_cluster_input_schema(&schema).unwrap_err(),
            ArrowClassifyError::ColumnNotFound(col) if col == COL_CLUSTER_SEQUENCE
        ));
    }

    #[test]
    fn input_schema_wrong_contig_id_type_rejected() {
        let schema = Schema::new(vec![
            Field::new(COL_CONTIG_ID, DataType::Int64, false),
            Field::new(COL_CLUSTER_SEQUENCE, DataType::Binary, false),
        ]);
        assert!(matches!(
            validate_cluster_input_schema(&schema).unwrap_err(),
            ArrowClassifyError::TypeError { column, .. } if column == COL_CONTIG_ID
        ));
    }

    #[test]
    fn input_schema_wrong_source_mag_type_rejected() {
        let schema = Schema::new(vec![
            Field::new(COL_CONTIG_ID, DataType::Utf8, false),
            Field::new(COL_SOURCE_MAG, DataType::Int64, true),
            Field::new(COL_CLUSTER_SEQUENCE, DataType::Binary, false),
        ]);
        assert!(matches!(
            validate_cluster_input_schema(&schema).unwrap_err(),
            ArrowClassifyError::TypeError { column, .. } if column == COL_SOURCE_MAG
        ));
    }

    #[test]
    fn input_schema_extra_columns_allowed() {
        let schema = Schema::new(vec![
            Field::new(COL_CONTIG_ID, DataType::Utf8, false),
            Field::new(COL_CLUSTER_SEQUENCE, DataType::Binary, false),
            Field::new("ignored_extra", DataType::Int64, true),
        ]);
        assert!(validate_cluster_input_schema(&schema).is_ok());
    }

    use crate::cluster::ClusterRow;
    use arrow::array::Array;

    fn row(rep: &str, member: &str, mag: Option<&str>, c: f64) -> ClusterRow {
        ClusterRow {
            rep_contig: rep.to_string(),
            member_contig: member.to_string(),
            source_mag: mag.map(|s| s.to_string()),
            containment: c,
        }
    }

    #[test]
    fn round_trip_three_row_result() {
        let result = ClusterResult {
            rows: vec![
                row("A", "A", Some("mag1"), 1.0),
                row("A", "B", Some("mag2"), 0.93),
                row("C", "C", None, 1.0),
            ],
        };

        let batch = cluster_result_to_record_batch(&result).unwrap();

        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.schema(), cluster_result_schema());

        let rep = batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap();
        let member = batch
            .column(1)
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap();
        let mag = batch
            .column(2)
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap();
        let containment = batch
            .column(3)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap();

        assert_eq!(rep.value(0), "A");
        assert_eq!(member.value(0), "A");
        assert_eq!(mag.value(0), "mag1");
        assert!(!mag.is_null(0));
        assert_eq!(containment.value(0), 1.0);

        assert_eq!(rep.value(1), "A");
        assert_eq!(member.value(1), "B");
        assert_eq!(mag.value(1), "mag2");
        assert!((containment.value(1) - 0.93).abs() < 1e-12);

        assert_eq!(rep.value(2), "C");
        assert_eq!(member.value(2), "C");
        assert!(mag.is_null(2), "source_mag should be NULL when None");
        assert_eq!(containment.value(2), 1.0);
    }

    #[test]
    fn empty_result_produces_zero_row_batch_with_correct_schema() {
        let batch = cluster_result_to_record_batch(&ClusterResult::default()).unwrap();
        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.schema(), cluster_result_schema());
    }

    #[test]
    fn empty_helper_matches_conversion_of_empty_result() {
        let from_helper = empty_cluster_result_batch();
        let from_conversion = cluster_result_to_record_batch(&ClusterResult::default()).unwrap();
        assert_eq!(from_helper.schema(), from_conversion.schema());
        assert_eq!(from_helper.num_rows(), from_conversion.num_rows());
    }
}
