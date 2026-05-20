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

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use std::sync::Arc;

use super::error::ArrowClassifyError;

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
}
