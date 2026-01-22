//! Unified error type for the rype library.
//!
//! This module provides a structured error type that captures all failure modes
//! in the library with appropriate context. Library code uses `RypeError` while
//! CLI code continues using `anyhow::Result` for convenience.
//!
//! # Error Categories
//!
//! - **Io**: File system operations (open, read, write)
//! - **Format**: Invalid file format (magic bytes, version mismatch)
//! - **Validation**: Invalid parameters or data (k-mer size, bucket counts)
//! - **Parquet**: Parquet-specific errors (schema, encoding)
//! - **Encoding**: Varint or other encoding errors
//! - **Overflow**: Numeric overflow or size limit exceeded

use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

/// Unified error type for the rype library.
#[derive(Debug)]
pub enum RypeError {
    /// I/O error with path context.
    Io {
        path: PathBuf,
        operation: &'static str,
        source: std::io::Error,
    },

    /// Invalid file format (magic bytes, version, structure).
    Format { path: PathBuf, detail: String },

    /// Validation error (invalid parameters, data invariants).
    Validation(String),

    /// Parquet-specific error.
    Parquet {
        context: String,
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Encoding/decoding error (varint, delta encoding).
    Encoding(String),

    /// Numeric overflow or size limit exceeded.
    Overflow {
        context: String,
        limit: usize,
        actual: usize,
    },
}

impl fmt::Display for RypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RypeError::Io {
                path,
                operation,
                source,
            } => {
                write!(
                    f,
                    "I/O error during {} on '{}': {}",
                    operation,
                    path.display(),
                    source
                )
            }
            RypeError::Format { path, detail } => {
                write!(f, "Invalid format in '{}': {}", path.display(), detail)
            }
            RypeError::Validation(msg) => write!(f, "Validation error: {}", msg),
            RypeError::Parquet { context, source } => {
                if let Some(src) = source {
                    write!(f, "Parquet error ({}): {}", context, src)
                } else {
                    write!(f, "Parquet error: {}", context)
                }
            }
            RypeError::Encoding(msg) => write!(f, "Encoding error: {}", msg),
            RypeError::Overflow {
                context,
                limit,
                actual,
            } => {
                write!(
                    f,
                    "Overflow in {}: limit is {}, got {}",
                    context, limit, actual
                )
            }
        }
    }
}

impl std::error::Error for RypeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RypeError::Io { source, .. } => Some(source),
            RypeError::Parquet {
                source: Some(s), ..
            } => Some(s.as_ref()),
            _ => None,
        }
    }
}

// ============================================================================
// Conversion traits
// ============================================================================

impl From<std::io::Error> for RypeError {
    fn from(err: std::io::Error) -> Self {
        RypeError::Io {
            path: PathBuf::new(),
            operation: "unknown",
            source: err,
        }
    }
}

impl From<parquet::errors::ParquetError> for RypeError {
    fn from(err: parquet::errors::ParquetError) -> Self {
        RypeError::Parquet {
            context: "parquet operation".to_string(),
            source: Some(Box::new(err)),
        }
    }
}

impl From<arrow::error::ArrowError> for RypeError {
    fn from(err: arrow::error::ArrowError) -> Self {
        RypeError::Parquet {
            context: "arrow operation".to_string(),
            source: Some(Box::new(err)),
        }
    }
}

/// Convenience type alias for Results using RypeError.
pub type Result<T> = std::result::Result<T, RypeError>;

// ============================================================================
// Helper constructors
// ============================================================================

impl RypeError {
    /// Create an I/O error with path context.
    pub fn io(path: impl Into<PathBuf>, operation: &'static str, source: std::io::Error) -> Self {
        RypeError::Io {
            path: path.into(),
            operation,
            source,
        }
    }

    /// Create a format error.
    pub fn format(path: impl Into<PathBuf>, detail: impl Into<String>) -> Self {
        RypeError::Format {
            path: path.into(),
            detail: detail.into(),
        }
    }

    /// Create a validation error.
    pub fn validation(msg: impl Into<String>) -> Self {
        RypeError::Validation(msg.into())
    }

    /// Create a Parquet error without source.
    pub fn parquet(context: impl Into<String>) -> Self {
        RypeError::Parquet {
            context: context.into(),
            source: None,
        }
    }

    /// Create an encoding error.
    pub fn encoding(msg: impl Into<String>) -> Self {
        RypeError::Encoding(msg.into())
    }

    /// Create an overflow error.
    pub fn overflow(context: impl Into<String>, limit: usize, actual: usize) -> Self {
        RypeError::Overflow {
            context: context.into(),
            limit,
            actual,
        }
    }
}

// ============================================================================
// Thread-safe error capture
// ============================================================================

/// Thread-safe error capture that stores only the first error.
///
/// This is useful for background threads that need to communicate errors
/// back to the main thread when the channel may have been dropped.
/// Uses atomic operations to ensure only the first error is stored.
pub struct FirstErrorCapture {
    has_error: AtomicBool,
    error: Mutex<Option<RypeError>>,
}

impl FirstErrorCapture {
    /// Create a new, empty error capture.
    pub fn new() -> Self {
        Self {
            has_error: AtomicBool::new(false),
            error: Mutex::new(None),
        }
    }

    /// Store an error, but only if no error has been stored yet.
    /// Returns true if this error was stored, false if an error already existed.
    pub fn store(&self, err: RypeError) -> bool {
        // Use compare_exchange to atomically check and set the flag
        if self
            .has_error
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            // We won the race - store the error
            if let Ok(mut guard) = self.error.lock() {
                *guard = Some(err);
            }
            true
        } else {
            false
        }
    }

    /// Store a string error (convenience method that wraps in RypeError::Validation).
    pub fn store_msg(&self, msg: impl Into<String>) -> bool {
        self.store(RypeError::Validation(msg.into()))
    }

    /// Retrieve the stored error, if any.
    pub fn get(&self) -> Option<RypeError> {
        if self.has_error.load(Ordering::SeqCst) {
            self.error.lock().ok().and_then(|mut g| g.take())
        } else {
            None
        }
    }

    /// Check if an error has been stored.
    pub fn has_error(&self) -> bool {
        self.has_error.load(Ordering::SeqCst)
    }
}

impl Default for FirstErrorCapture {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_io_error_display() {
        let err = RypeError::io(
            "/path/to/file.ryidx",
            "read",
            std::io::Error::new(std::io::ErrorKind::NotFound, "file not found"),
        );
        let msg = err.to_string();
        assert!(msg.contains("/path/to/file.ryidx"));
        assert!(msg.contains("read"));
        assert!(msg.contains("file not found"));
    }

    #[test]
    fn test_format_error_display() {
        let err = RypeError::format("/path/to/file.ryidx", "invalid magic bytes");
        let msg = err.to_string();
        assert!(msg.contains("/path/to/file.ryidx"));
        assert!(msg.contains("invalid magic bytes"));
    }

    #[test]
    fn test_validation_error_display() {
        let err = RypeError::validation("k must be 16, 32, or 64");
        assert!(err.to_string().contains("k must be 16, 32, or 64"));
    }

    #[test]
    fn test_overflow_error_display() {
        let err = RypeError::overflow("bucket count", 100_000, 150_000);
        let msg = err.to_string();
        assert!(msg.contains("bucket count"));
        assert!(msg.contains("100000"));
        assert!(msg.contains("150000"));
    }

    #[test]
    fn test_error_source_chain() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "not found");
        let err = RypeError::io("/path", "open", io_err);

        // Should have a source
        assert!(std::error::Error::source(&err).is_some());
    }

    #[test]
    fn test_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
        let err: RypeError = io_err.into();

        match err {
            RypeError::Io { operation, .. } => assert_eq!(operation, "unknown"),
            _ => panic!("Expected Io variant"),
        }
    }

    // -------------------------------------------------------------------------
    // FirstErrorCapture tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_first_error_capture_stores_first() {
        let capture = FirstErrorCapture::new();

        assert!(capture.store(RypeError::validation("first error")));
        assert!(!capture.store(RypeError::validation("second error"))); // Should return false

        let err = capture.get().expect("Should have error");
        assert!(err.to_string().contains("first error"));
    }

    #[test]
    fn test_first_error_capture_store_msg() {
        let capture = FirstErrorCapture::new();

        assert!(capture.store_msg("first error"));
        assert!(!capture.store_msg("second error"));

        let err = capture.get().expect("Should have error");
        assert!(err.to_string().contains("first error"));
    }

    #[test]
    fn test_first_error_capture_empty() {
        let capture = FirstErrorCapture::new();
        assert!(capture.get().is_none());
        assert!(!capture.has_error());
    }

    #[test]
    fn test_first_error_capture_has_error() {
        let capture = FirstErrorCapture::new();
        assert!(!capture.has_error());

        capture.store_msg("test error");
        assert!(capture.has_error());
    }

    #[test]
    fn test_first_error_capture_default() {
        let capture = FirstErrorCapture::default();
        assert!(!capture.has_error());
    }
}
