//! Schema definitions and validation for Arrow-based classification.
//!
//! # Input Schema
//!
//! | Column | Arrow Type | Nullable | Description |
//! |--------|-----------|----------|-------------|
//! | `id` | Int64 | No | Query identifier |
//! | `sequence` | Binary or LargeBinary | No | DNA sequence bytes |
//! | `pair_sequence` | Binary or LargeBinary | Yes | Optional paired-end sequence |
//!
//! # Output Schema
//!
//! | Column | Arrow Type | Nullable | Description |
//! |--------|-----------|----------|-------------|
//! | `query_id` | Int64 | No | Matching query ID |
//! | `bucket_id` | UInt32 | No | Matched bucket/reference ID |
//! | `score` | Float64 | No | Classification score (0.0-1.0) |

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use std::sync::Arc;

use super::error::ArrowClassifyError;

/// Column name for query ID in input batches.
pub const COL_ID: &str = "id";
/// Column name for primary sequence in input batches.
pub const COL_SEQUENCE: &str = "sequence";
/// Column name for paired-end sequence in input batches (optional).
pub const COL_PAIR_SEQUENCE: &str = "pair_sequence";

/// Column name for query ID in output batches.
pub const COL_QUERY_ID: &str = "query_id";
/// Column name for bucket ID in output batches.
pub const COL_BUCKET_ID: &str = "bucket_id";
/// Column name for score in output batches.
pub const COL_SCORE: &str = "score";

/// Returns the schema for classification result batches.
///
/// Schema:
/// - `query_id`: Int64 (non-nullable)
/// - `bucket_id`: UInt32 (non-nullable)
/// - `score`: Float64 (non-nullable)
pub fn result_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(COL_QUERY_ID, DataType::Int64, false),
        Field::new(COL_BUCKET_ID, DataType::UInt32, false),
        Field::new(COL_SCORE, DataType::Float64, false),
    ]))
}

/// Check if a DataType is a valid binary type (Binary or LargeBinary).
fn is_binary_type(dt: &DataType) -> bool {
    matches!(dt, DataType::Binary | DataType::LargeBinary)
}

/// Validate that a schema matches the expected input schema for classification.
///
/// # Requirements
///
/// - Must have column `id` with type Int64
/// - Must have column `sequence` with type Binary or LargeBinary
/// - May optionally have column `pair_sequence` with type Binary or LargeBinary (nullable)
///
/// # Errors
///
/// Returns `ArrowClassifyError` if the schema does not match requirements.
pub fn validate_input_schema(schema: &Schema) -> Result<(), ArrowClassifyError> {
    // Check for 'id' column
    match schema.column_with_name(COL_ID) {
        Some((_, field)) => {
            if field.data_type() != &DataType::Int64 {
                return Err(ArrowClassifyError::TypeError {
                    column: COL_ID.into(),
                    expected: "Int64".into(),
                    actual: format!("{:?}", field.data_type()),
                });
            }
        }
        None => {
            return Err(ArrowClassifyError::ColumnNotFound(COL_ID.into()));
        }
    }

    // Check for 'sequence' column
    match schema.column_with_name(COL_SEQUENCE) {
        Some((_, field)) => {
            if !is_binary_type(field.data_type()) {
                return Err(ArrowClassifyError::TypeError {
                    column: COL_SEQUENCE.into(),
                    expected: "Binary or LargeBinary".into(),
                    actual: format!("{:?}", field.data_type()),
                });
            }
        }
        None => {
            return Err(ArrowClassifyError::ColumnNotFound(COL_SEQUENCE.into()));
        }
    }

    // Check optional 'pair_sequence' column if present
    if let Some((_, field)) = schema.column_with_name(COL_PAIR_SEQUENCE) {
        if !is_binary_type(field.data_type()) {
            return Err(ArrowClassifyError::TypeError {
                column: COL_PAIR_SEQUENCE.into(),
                expected: "Binary or LargeBinary".into(),
                actual: format!("{:?}", field.data_type()),
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field, Schema};

    fn make_valid_schema() -> Schema {
        Schema::new(vec![
            Field::new(COL_ID, DataType::Int64, false),
            Field::new(COL_SEQUENCE, DataType::Binary, false),
        ])
    }

    fn make_valid_schema_with_pair() -> Schema {
        Schema::new(vec![
            Field::new(COL_ID, DataType::Int64, false),
            Field::new(COL_SEQUENCE, DataType::Binary, false),
            Field::new(COL_PAIR_SEQUENCE, DataType::Binary, true),
        ])
    }

    #[test]
    fn test_input_schema_valid() {
        let schema = make_valid_schema();
        assert!(validate_input_schema(&schema).is_ok());
    }

    #[test]
    fn test_input_schema_valid_with_large_binary() {
        let schema = Schema::new(vec![
            Field::new(COL_ID, DataType::Int64, false),
            Field::new(COL_SEQUENCE, DataType::LargeBinary, false),
        ]);
        assert!(validate_input_schema(&schema).is_ok());
    }

    #[test]
    fn test_input_schema_missing_id_column() {
        let schema = Schema::new(vec![Field::new(COL_SEQUENCE, DataType::Binary, false)]);
        let result = validate_input_schema(&schema);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ArrowClassifyError::ColumnNotFound(col) if col == COL_ID
        ));
    }

    #[test]
    fn test_input_schema_wrong_id_type() {
        let schema = Schema::new(vec![
            Field::new(COL_ID, DataType::Utf8, false), // Wrong type
            Field::new(COL_SEQUENCE, DataType::Binary, false),
        ]);
        let result = validate_input_schema(&schema);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ArrowClassifyError::TypeError { column, .. } if column == COL_ID
        ));
    }

    #[test]
    fn test_input_schema_missing_sequence_column() {
        let schema = Schema::new(vec![Field::new(COL_ID, DataType::Int64, false)]);
        let result = validate_input_schema(&schema);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ArrowClassifyError::ColumnNotFound(col) if col == COL_SEQUENCE
        ));
    }

    #[test]
    fn test_input_schema_wrong_sequence_type() {
        let schema = Schema::new(vec![
            Field::new(COL_ID, DataType::Int64, false),
            Field::new(COL_SEQUENCE, DataType::Utf8, false), // Wrong type
        ]);
        let result = validate_input_schema(&schema);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ArrowClassifyError::TypeError { column, .. } if column == COL_SEQUENCE
        ));
    }

    #[test]
    fn test_input_schema_with_optional_pair() {
        let schema = make_valid_schema_with_pair();
        assert!(validate_input_schema(&schema).is_ok());
    }

    #[test]
    fn test_input_schema_with_optional_pair_large_binary() {
        let schema = Schema::new(vec![
            Field::new(COL_ID, DataType::Int64, false),
            Field::new(COL_SEQUENCE, DataType::Binary, false),
            Field::new(COL_PAIR_SEQUENCE, DataType::LargeBinary, true),
        ]);
        assert!(validate_input_schema(&schema).is_ok());
    }

    #[test]
    fn test_input_schema_wrong_pair_type() {
        let schema = Schema::new(vec![
            Field::new(COL_ID, DataType::Int64, false),
            Field::new(COL_SEQUENCE, DataType::Binary, false),
            Field::new(COL_PAIR_SEQUENCE, DataType::Utf8, true), // Wrong type
        ]);
        let result = validate_input_schema(&schema);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ArrowClassifyError::TypeError { column, .. } if column == COL_PAIR_SEQUENCE
        ));
    }

    #[test]
    fn test_input_schema_extra_columns_allowed() {
        // Extra columns should be ignored
        let schema = Schema::new(vec![
            Field::new(COL_ID, DataType::Int64, false),
            Field::new(COL_SEQUENCE, DataType::Binary, false),
            Field::new("extra_column", DataType::Utf8, true),
            Field::new("another_extra", DataType::Float64, true),
        ]);
        assert!(validate_input_schema(&schema).is_ok());
    }

    #[test]
    fn test_output_schema_structure() {
        let schema = result_schema();

        assert_eq!(schema.fields().len(), 3);

        let query_id_field = schema.field_with_name(COL_QUERY_ID).unwrap();
        assert_eq!(query_id_field.data_type(), &DataType::Int64);
        assert!(!query_id_field.is_nullable());

        let bucket_id_field = schema.field_with_name(COL_BUCKET_ID).unwrap();
        assert_eq!(bucket_id_field.data_type(), &DataType::UInt32);
        assert!(!bucket_id_field.is_nullable());

        let score_field = schema.field_with_name(COL_SCORE).unwrap();
        assert_eq!(score_field.data_type(), &DataType::Float64);
        assert!(!score_field.is_nullable());
    }
}
