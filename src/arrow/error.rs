//! Arrow-specific error types for the rype library.

use std::fmt;

/// Maximum sequence length in bytes (2GB, matching C API limit).
pub const MAX_SEQUENCE_LENGTH: usize = 2_000_000_000;

/// Errors that can occur during Arrow-based classification.
#[derive(Debug)]
pub enum ArrowClassifyError {
    /// Schema validation failed
    SchemaError(String),
    /// Type mismatch in column
    TypeError {
        column: String,
        expected: String,
        actual: String,
    },
    /// Null value in non-nullable context
    NullError { column: String, row: usize },
    /// Column not found
    ColumnNotFound(String),
    /// Sequence exceeds maximum allowed length
    SequenceTooLong {
        row: usize,
        length: usize,
        max_length: usize,
    },
    /// Underlying Arrow error
    Arrow(arrow::error::ArrowError),
    /// Classification error from core library
    Classification(String),
}

impl fmt::Display for ArrowClassifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArrowClassifyError::SchemaError(msg) => write!(f, "Schema error: {}", msg),
            ArrowClassifyError::TypeError {
                column,
                expected,
                actual,
            } => write!(
                f,
                "Type error in column '{}': expected {}, got {}",
                column, expected, actual
            ),
            ArrowClassifyError::NullError { column, row } => {
                write!(f, "Unexpected null in column '{}' at row {}", column, row)
            }
            ArrowClassifyError::ColumnNotFound(name) => {
                write!(f, "Required column '{}' not found", name)
            }
            ArrowClassifyError::SequenceTooLong {
                row,
                length,
                max_length,
            } => {
                write!(
                    f,
                    "Sequence at row {} is {} bytes, exceeds maximum of {} bytes",
                    row, length, max_length
                )
            }
            ArrowClassifyError::Arrow(e) => write!(f, "Arrow error: {}", e),
            ArrowClassifyError::Classification(msg) => write!(f, "Classification error: {}", msg),
        }
    }
}

impl std::error::Error for ArrowClassifyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ArrowClassifyError::Arrow(e) => Some(e),
            _ => None,
        }
    }
}

impl From<arrow::error::ArrowError> for ArrowClassifyError {
    fn from(err: arrow::error::ArrowError) -> Self {
        ArrowClassifyError::Arrow(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn test_error_display_schema_error() {
        let err = ArrowClassifyError::SchemaError("missing required column".into());
        assert_eq!(err.to_string(), "Schema error: missing required column");
    }

    #[test]
    fn test_error_display_type_error() {
        let err = ArrowClassifyError::TypeError {
            column: "id".into(),
            expected: "Int64".into(),
            actual: "Utf8".into(),
        };
        assert_eq!(
            err.to_string(),
            "Type error in column 'id': expected Int64, got Utf8"
        );
    }

    #[test]
    fn test_error_display_null_error() {
        let err = ArrowClassifyError::NullError {
            column: "sequence".into(),
            row: 42,
        };
        assert_eq!(
            err.to_string(),
            "Unexpected null in column 'sequence' at row 42"
        );
    }

    #[test]
    fn test_error_display_column_not_found() {
        let err = ArrowClassifyError::ColumnNotFound("pair_sequence".into());
        assert_eq!(
            err.to_string(),
            "Required column 'pair_sequence' not found"
        );
    }

    #[test]
    fn test_error_display_classification() {
        let err = ArrowClassifyError::Classification("index not loaded".into());
        assert_eq!(err.to_string(), "Classification error: index not loaded");
    }

    #[test]
    fn test_error_display_sequence_too_long() {
        let err = ArrowClassifyError::SequenceTooLong {
            row: 5,
            length: 3_000_000_000,
            max_length: MAX_SEQUENCE_LENGTH,
        };
        assert_eq!(
            err.to_string(),
            "Sequence at row 5 is 3000000000 bytes, exceeds maximum of 2000000000 bytes"
        );
    }

    #[test]
    fn test_error_from_arrow_error() {
        let arrow_err = arrow::error::ArrowError::InvalidArgumentError("test error".into());
        let err: ArrowClassifyError = arrow_err.into();
        assert!(matches!(err, ArrowClassifyError::Arrow(_)));
        assert!(err.to_string().contains("test error"));
    }

    #[test]
    fn test_error_source() {
        let arrow_err = arrow::error::ArrowError::InvalidArgumentError("source test".into());
        let err: ArrowClassifyError = arrow_err.into();
        assert!(err.source().is_some());

        let schema_err = ArrowClassifyError::SchemaError("no source".into());
        assert!(schema_err.source().is_none());
    }
}
