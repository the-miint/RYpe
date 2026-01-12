use crate::{Index, InvertedIndex, QueryRecord, classify_batch, classify_batch_inverted};
use std::ffi::{CStr, CString};
use std::slice;
use std::path::Path;
use std::cell::RefCell;
use std::collections::HashMap;
use libc::{c_char, size_t, c_double};

#[cfg(feature = "arrow")]
use crate::arrow::{
    classify_arrow_batch, classify_arrow_batch_inverted,
    classify_arrow_batch_sharded,
};
#[cfg(feature = "arrow")]
use crate::ShardedInvertedIndex;
#[cfg(feature = "arrow")]
use arrow::ffi::FFI_ArrowSchema;
#[cfg(feature = "arrow")]
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
#[cfg(feature = "arrow")]
use arrow::record_batch::RecordBatch;

// --- Error Reporting ---

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = RefCell::new(None);
}

fn set_last_error(err: String) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = CString::new(err).ok();
    });
}

fn clear_last_error() {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = None;
    });
}

// Maximum sequence length: 2GB (fits in isize on 64-bit, u32::MAX on 32-bit)
const MAX_SEQUENCE_LENGTH: usize = if cfg!(target_pointer_width = "64") {
    2_000_000_000 // 2GB
} else {
    i32::MAX as usize // ~2.1GB
};

/// Validates a RypeQuery for safety invariants
/// Returns Err with error message if invalid
fn validate_query(q: &RypeQuery) -> Result<(), &'static str> {
    // Check seq is non-null
    if q.seq.is_null() {
        return Err("query seq pointer is NULL");
    }

    // Check seq_len is reasonable
    if q.seq_len == 0 {
        return Err("query seq_len is zero");
    }
    if q.seq_len > MAX_SEQUENCE_LENGTH {
        return Err("query seq_len exceeds maximum (2GB)");
    }

    // Check paired-end consistency
    if q.pair_seq.is_null() {
        if q.pair_len != 0 {
            return Err("pair_seq is NULL but pair_len is non-zero");
        }
    } else {
        if q.pair_len == 0 {
            return Err("pair_seq is non-NULL but pair_len is zero");
        }
        if q.pair_len > MAX_SEQUENCE_LENGTH {
            return Err("query pair_len exceeds maximum (2GB)");
        }
    }

    Ok(())
}

// --- C-Compatible Structs ---

#[repr(C)]
pub struct RypeQuery {
    pub id: i64,
    pub seq: *const c_char,
    pub seq_len: size_t,
    pub pair_seq: *const c_char,
    pub pair_len: size_t,
}

#[repr(C)]
pub struct RypeHit {
    pub query_id: i64,
    pub bucket_id: u32,
    pub score: c_double,
}

#[repr(C)]
pub struct RypeResultArray {
    pub data: *mut RypeHit,
    pub len: size_t,
    pub capacity: size_t,
}

// --- API Functions ---

#[no_mangle]
pub extern "C" fn rype_index_load(path: *const c_char) -> *mut Index {
    if path.is_null() {
        set_last_error("path is NULL".to_string());
        return std::ptr::null_mut();
    }

    let c_str = unsafe { CStr::from_ptr(path) };
    let r_str = match c_str.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid UTF-8 in path: {}", e));
            return std::ptr::null_mut();
        }
    };

    match Index::load(Path::new(r_str)) {
        Ok(idx) => {
            clear_last_error();
            Box::into_raw(Box::new(idx))
        }
        Err(e) => {
            set_last_error(format!("Failed to load index: {}", e));
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn rype_index_free(ptr: *mut Index) {
    if !ptr.is_null() {
        unsafe { let _ = Box::from_raw(ptr); }
    }
}

// --- Index Metadata Accessors ---

/// Returns the k-mer size of the index, or 0 if index is NULL.
#[no_mangle]
pub extern "C" fn rype_index_k(index_ptr: *const Index) -> size_t {
    if index_ptr.is_null() {
        return 0;
    }
    let index = unsafe { &*index_ptr };
    index.k
}

/// Returns the window size of the index, or 0 if index is NULL.
#[no_mangle]
pub extern "C" fn rype_index_w(index_ptr: *const Index) -> size_t {
    if index_ptr.is_null() {
        return 0;
    }
    let index = unsafe { &*index_ptr };
    index.w
}

/// Returns the salt of the index, or 0 if index is NULL.
#[no_mangle]
pub extern "C" fn rype_index_salt(index_ptr: *const Index) -> u64 {
    if index_ptr.is_null() {
        return 0;
    }
    let index = unsafe { &*index_ptr };
    index.salt
}

/// Returns the number of buckets in the index, or 0 if index is NULL.
#[no_mangle]
pub extern "C" fn rype_index_num_buckets(index_ptr: *const Index) -> u32 {
    if index_ptr.is_null() {
        return 0;
    }
    let index = unsafe { &*index_ptr };
    index.buckets.len() as u32
}

// --- Bucket Name Lookup ---

// Thread-local storage for bucket name CStrings to maintain lifetime
thread_local! {
    static BUCKET_NAME_CACHE: RefCell<HashMap<u32, CString>> = RefCell::new(HashMap::new());
}

/// Returns the name of a bucket by ID, or NULL if not found.
/// The returned string is owned by the library and must NOT be freed by the caller.
/// The string remains valid until the next call to rype_bucket_name on this thread
/// or until the index is freed.
#[no_mangle]
pub extern "C" fn rype_bucket_name(index_ptr: *const Index, bucket_id: u32) -> *const c_char {
    if index_ptr.is_null() {
        return std::ptr::null();
    }

    let index = unsafe { &*index_ptr };

    match index.bucket_names.get(&bucket_id) {
        Some(name) => {
            BUCKET_NAME_CACHE.with(|cache| {
                let mut cache = cache.borrow_mut();
                // Insert/update the CString in our cache
                let cstring = match CString::new(name.as_str()) {
                    Ok(s) => s,
                    Err(_) => return std::ptr::null(),
                };
                cache.insert(bucket_id, cstring);
                // Return pointer to the cached CString
                cache.get(&bucket_id).map(|s| s.as_ptr()).unwrap_or(std::ptr::null())
            })
        }
        None => std::ptr::null(),
    }
}

// --- Negative Index API ---

use std::collections::HashSet;

/// Opaque handle to a pre-built negative minimizer set for filtering.
/// This allows efficient reuse when classifying multiple batches.
pub struct RypeNegativeSet {
    minimizers: HashSet<u64>,
}

/// Creates a negative minimizer set from an index.
/// All minimizers from all buckets in the index will be used for filtering.
/// Returns NULL on error; call rype_get_last_error() for details.
///
/// The negative index must have the same k, w, and salt as the positive index
/// used for classification. This is NOT validated here but will affect results.
#[no_mangle]
pub extern "C" fn rype_negative_set_create(negative_index_ptr: *const Index) -> *mut RypeNegativeSet {
    if negative_index_ptr.is_null() {
        set_last_error("negative_index is NULL".to_string());
        return std::ptr::null_mut();
    }

    let neg_index = unsafe { &*negative_index_ptr };

    let minimizers: HashSet<u64> = neg_index.buckets.values()
        .flat_map(|v| v.iter().copied())
        .collect();

    clear_last_error();
    Box::into_raw(Box::new(RypeNegativeSet { minimizers }))
}

/// Frees a negative set. NULL is safe to pass.
#[no_mangle]
pub extern "C" fn rype_negative_set_free(ptr: *mut RypeNegativeSet) {
    if !ptr.is_null() {
        unsafe { let _ = Box::from_raw(ptr); }
    }
}

/// Returns the number of unique minimizers in the negative set.
#[no_mangle]
pub extern "C" fn rype_negative_set_size(ptr: *const RypeNegativeSet) -> size_t {
    if ptr.is_null() {
        return 0;
    }
    let neg_set = unsafe { &*ptr };
    neg_set.minimizers.len()
}

// --- Inverted Index API ---

/// Loads an inverted index from disk.
/// Returns NULL on error; call rype_get_last_error() for details.
#[no_mangle]
pub extern "C" fn rype_inverted_load(path: *const c_char) -> *mut InvertedIndex {
    if path.is_null() {
        set_last_error("path is NULL".to_string());
        return std::ptr::null_mut();
    }

    let c_str = unsafe { CStr::from_ptr(path) };
    let r_str = match c_str.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid UTF-8 in path: {}", e));
            return std::ptr::null_mut();
        }
    };

    match InvertedIndex::load(Path::new(r_str)) {
        Ok(inv) => {
            clear_last_error();
            Box::into_raw(Box::new(inv))
        }
        Err(e) => {
            set_last_error(format!("Failed to load inverted index: {}", e));
            std::ptr::null_mut()
        }
    }
}

/// Frees an inverted index. NULL is safe to pass.
#[no_mangle]
pub extern "C" fn rype_inverted_free(ptr: *mut InvertedIndex) {
    if !ptr.is_null() {
        unsafe { let _ = Box::from_raw(ptr); }
    }
}

/// Classifies a batch of queries using the inverted index for faster lookups.
/// Returns NULL on error; call rype_get_last_error() for details.
///
/// Parameters:
/// - inverted: The inverted index for fast minimizer → bucket lookups
/// - negative_set: Optional negative set for filtering (NULL to disable)
/// - queries: Array of query sequences
/// - num_queries: Number of queries
/// - threshold: Classification threshold (0.0-1.0)
#[no_mangle]
pub extern "C" fn rype_classify_inverted(
    inverted_ptr: *const InvertedIndex,
    negative_set_ptr: *const RypeNegativeSet,
    queries_ptr: *const RypeQuery,
    num_queries: size_t,
    threshold: c_double
) -> *mut RypeResultArray {
    if inverted_ptr.is_null() {
        set_last_error("inverted index is NULL".to_string());
        return std::ptr::null_mut();
    }
    if queries_ptr.is_null() || num_queries == 0 {
        set_last_error("Invalid arguments: queries or num_queries is invalid".to_string());
        return std::ptr::null_mut();
    }

    // Validate threshold
    if !threshold.is_finite() {
        set_last_error("Invalid threshold: must be finite".to_string());
        return std::ptr::null_mut();
    }
    if threshold < 0.0 || threshold > 1.0 {
        set_last_error(format!("Invalid threshold: {} (expected 0.0-1.0)", threshold));
        return std::ptr::null_mut();
    }

    // Validate num_queries is reasonable
    if num_queries > isize::MAX as size_t {
        set_last_error("num_queries exceeds maximum".to_string());
        return std::ptr::null_mut();
    }

    let result = std::panic::catch_unwind(|| {
        let inverted = unsafe { &*inverted_ptr };
        let c_queries = unsafe { slice::from_raw_parts(queries_ptr, num_queries) };

        // Get negative set if provided
        let neg_mins: Option<&HashSet<u64>> = if negative_set_ptr.is_null() {
            None
        } else {
            let neg_set = unsafe { &*negative_set_ptr };
            Some(&neg_set.minimizers)
        };

        let rust_queries: Vec<QueryRecord> = c_queries.iter().enumerate().map(|(idx, q)| {
            if let Err(msg) = validate_query(q) {
                panic!("Query {} validation failed: {}", idx, msg);
            }

            let s1 = unsafe { slice::from_raw_parts(q.seq as *const u8, q.seq_len) };
            let s2 = if !q.pair_seq.is_null() {
                Some(unsafe { slice::from_raw_parts(q.pair_seq as *const u8, q.pair_len) })
            } else {
                None
            };
            (q.id, s1, s2)
        }).collect();

        let hits = classify_batch_inverted(inverted, neg_mins, &rust_queries, threshold);

        let mut c_hits: Vec<RypeHit> = hits.into_iter().map(|h| RypeHit {
            query_id: h.query_id,
            bucket_id: h.bucket_id,
            score: h.score,
        }).collect();

        c_hits.shrink_to_fit();
        let len = c_hits.len();
        let capacity = c_hits.capacity();
        let data = c_hits.as_mut_ptr();
        std::mem::forget(c_hits);

        let result_array = Box::new(RypeResultArray { data, len, capacity });
        Box::into_raw(result_array)
    });

    match result {
        Ok(ptr) => {
            clear_last_error();
            ptr
        }
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown error during classification".to_string()
            };
            set_last_error(msg);
            std::ptr::null_mut()
        }
    }
}

// --- Primary Index Classification ---

/// Classifies queries using the primary index without negative filtering.
/// For negative filtering support, use rype_classify_with_negative().
#[no_mangle]
pub extern "C" fn rype_classify(
    index_ptr: *const Index,
    queries_ptr: *const RypeQuery,
    num_queries: size_t,
    threshold: c_double
) -> *mut RypeResultArray {
    // Delegate to the full version with NULL negative set
    rype_classify_with_negative(index_ptr, std::ptr::null(), queries_ptr, num_queries, threshold)
}

/// Classifies queries using the primary index with optional negative filtering.
///
/// Parameters:
/// - index: Primary index for classification
/// - negative_set: Optional negative set for filtering (NULL to disable)
/// - queries: Array of query sequences
/// - num_queries: Number of queries
/// - threshold: Classification threshold (0.0-1.0)
///
/// Negative filtering removes minimizers that appear in the negative set from
/// queries before scoring, reducing false positives from contaminating sequences.
#[no_mangle]
pub extern "C" fn rype_classify_with_negative(
    index_ptr: *const Index,
    negative_set_ptr: *const RypeNegativeSet,
    queries_ptr: *const RypeQuery,
    num_queries: size_t,
    threshold: c_double
) -> *mut RypeResultArray {
    if index_ptr.is_null() {
        set_last_error("index is NULL".to_string());
        return std::ptr::null_mut();
    }
    if queries_ptr.is_null() || num_queries == 0 {
        set_last_error("Invalid arguments: queries or num_queries is invalid".to_string());
        return std::ptr::null_mut();
    }

    // Validate threshold
    if !threshold.is_finite() {
        set_last_error("Invalid threshold: must be finite".to_string());
        return std::ptr::null_mut();
    }
    if threshold < 0.0 || threshold > 1.0 {
        set_last_error(format!("Invalid threshold: {} (expected 0.0-1.0)", threshold));
        return std::ptr::null_mut();
    }

    // Validate num_queries is reasonable (prevent integer overflow)
    if num_queries > isize::MAX as size_t {
        set_last_error("num_queries exceeds maximum".to_string());
        return std::ptr::null_mut();
    }

    let result = std::panic::catch_unwind(|| {
        let index = unsafe { &*index_ptr };
        let c_queries = unsafe { slice::from_raw_parts(queries_ptr, num_queries) };

        // Get negative set if provided
        let neg_mins: Option<&HashSet<u64>> = if negative_set_ptr.is_null() {
            None
        } else {
            let neg_set = unsafe { &*negative_set_ptr };
            Some(&neg_set.minimizers)
        };

        let rust_queries: Vec<QueryRecord> = c_queries.iter().enumerate().map(|(idx, q)| {
            // Validate this query
            if let Err(msg) = validate_query(q) {
                panic!("Query {} validation failed: {}", idx, msg);
            }

            let s1 = unsafe { slice::from_raw_parts(q.seq as *const u8, q.seq_len) };
            let s2 = if !q.pair_seq.is_null() {
                Some(unsafe { slice::from_raw_parts(q.pair_seq as *const u8, q.pair_len) })
            } else {
                None
            };
            (q.id, s1, s2)
        }).collect();

        let hits = classify_batch(index, neg_mins, &rust_queries, threshold);

        let mut c_hits: Vec<RypeHit> = hits.into_iter().map(|h| RypeHit {
            query_id: h.query_id,
            bucket_id: h.bucket_id,
            score: h.score,
        }).collect();

        c_hits.shrink_to_fit();
        let len = c_hits.len();
        let capacity = c_hits.capacity();
        let data = c_hits.as_mut_ptr();
        std::mem::forget(c_hits);

        let result_array = Box::new(RypeResultArray { data, len, capacity });
        Box::into_raw(result_array)
    });

    match result {
        Ok(ptr) => {
            clear_last_error();
            ptr
        }
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown error during classification".to_string()
            };
            set_last_error(msg);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn rype_results_free(ptr: *mut RypeResultArray) {
    if !ptr.is_null() {
        unsafe {
            let array = Box::from_raw(ptr);

            // Debug-mode check: verify data pointer hasn't been freed already
            #[cfg(debug_assertions)]
            {
                // Check for null data pointer (might indicate double-free)
                if array.data.is_null() && array.len > 0 {
                    eprintln!("WARNING: Potential double-free detected in rype_results_free");
                }
            }

            let _ = Vec::from_raw_parts(array.data, array.len, array.capacity);
        }
    }
}

#[no_mangle]
pub extern "C" fn rype_get_last_error() -> *const c_char {
    LAST_ERROR.with(|e| {
        e.borrow()
            .as_ref()
            .map(|s| s.as_ptr())
            .unwrap_or(std::ptr::null())
    })
}

// --- ABI Layout Tests ---

#[cfg(test)]
mod layout_tests {
    use super::*;

    #[test]
    fn test_rype_query_layout() {
        use std::mem::{size_of, align_of};

        // Expected sizes (on 64-bit systems)
        assert_eq!(size_of::<RypeQuery>(), 40);
        assert_eq!(align_of::<RypeQuery>(), 8);

        // Verify field offsets match C struct layout
        assert_eq!(std::mem::offset_of!(RypeQuery, id), 0);
        assert_eq!(std::mem::offset_of!(RypeQuery, seq), 8);
        assert_eq!(std::mem::offset_of!(RypeQuery, seq_len), 16);
        assert_eq!(std::mem::offset_of!(RypeQuery, pair_seq), 24);
        assert_eq!(std::mem::offset_of!(RypeQuery, pair_len), 32);
    }

    #[test]
    fn test_rype_hit_layout() {
        use std::mem::{size_of, align_of};
        // i64 (8) + u32 (4) + padding (4) + f64 (8) = 24 bytes
        assert_eq!(size_of::<RypeHit>(), 24);
        assert_eq!(align_of::<RypeHit>(), 8);
    }

    #[test]
    fn test_rype_result_array_layout() {
        use std::mem::{size_of, align_of};
        assert_eq!(size_of::<RypeResultArray>(), 24);
        assert_eq!(align_of::<RypeResultArray>(), 8);
    }
}

// =============================================================================
// Arrow C FFI API
// =============================================================================
//
// These functions use the Arrow C Data Interface to exchange RecordBatches
// with other languages (Python/PyArrow, R, C++, etc.) via the FFI_ArrowArrayStream.
//
// The Arrow C Data Interface is documented at:
// https://arrow.apache.org/docs/format/CDataInterface.html
//
// Thread Safety:
// - All Arrow FFI functions are thread-safe for classification
// - Multiple threads can share the same Index/InvertedIndex
// - Each thread must use its own FFI_ArrowArrayStream for input/output

#[cfg(feature = "arrow")]
mod arrow_ffi {
    use super::*;
    use arrow::datatypes::SchemaRef;
    use arrow::record_batch::RecordBatchReader;

    // -------------------------------------------------------------------------
    // Helper: Validate threshold
    // -------------------------------------------------------------------------

    fn validate_threshold(threshold: c_double) -> Result<(), String> {
        if !threshold.is_finite() || threshold < 0.0 || threshold > 1.0 {
            Err(format!(
                "Invalid threshold: {} (expected finite value 0.0-1.0)",
                threshold
            ))
        } else {
            Ok(())
        }
    }

    // -------------------------------------------------------------------------
    // Send-safe pointer wrapper for FFI
    // -------------------------------------------------------------------------
    //
    // Raw pointers aren't Send in Rust. For FFI, we need to wrap them and
    // assert Send safety. The caller guarantees the pointers remain valid
    // for the lifetime of the stream.

    /// Wrapper for raw pointer that asserts Send safety.
    /// SAFETY: Caller must guarantee the pointer remains valid and is safe to access
    /// from any thread (i.e., the underlying data is immutable or synchronized).
    struct SendPtr<T>(*const T);

    // SAFETY: We only use this for immutable Index/InvertedIndex/ShardedInvertedIndex
    // which are read-only during classification. The caller guarantees validity.
    unsafe impl<T> Send for SendPtr<T> {}
    unsafe impl<T> Sync for SendPtr<T> {}

    impl<T> SendPtr<T> {
        unsafe fn new(ptr: *const T) -> Self {
            SendPtr(ptr)
        }

        unsafe fn get(&self) -> &T {
            &*self.0
        }
    }

    // -------------------------------------------------------------------------
    // Streaming Classification Reader
    // -------------------------------------------------------------------------
    //
    // This struct implements true streaming: it reads one batch at a time from
    // the input, classifies it, and yields the result. Memory usage is O(1)
    // with respect to the number of batches.

    /// Streaming classifier that wraps an input stream and classifies on-demand.
    struct StreamingClassifier<F> {
        input_reader: ArrowArrayStreamReader,
        output_schema: SchemaRef,
        classify_fn: F,
    }

    impl<F> StreamingClassifier<F>
    where
        F: Fn(&RecordBatch) -> Result<RecordBatch, arrow::error::ArrowError>,
    {
        fn new(
            input_reader: ArrowArrayStreamReader,
            output_schema: SchemaRef,
            classify_fn: F,
        ) -> Self {
            Self {
                input_reader,
                output_schema,
                classify_fn,
            }
        }
    }

    impl<F> Iterator for StreamingClassifier<F>
    where
        F: Fn(&RecordBatch) -> Result<RecordBatch, arrow::error::ArrowError>,
    {
        type Item = Result<RecordBatch, arrow::error::ArrowError>;

        fn next(&mut self) -> Option<Self::Item> {
            match self.input_reader.next() {
                Some(Ok(batch)) => Some((self.classify_fn)(&batch)),
                Some(Err(e)) => Some(Err(e)),
                None => None,
            }
        }
    }

    // SAFETY: StreamingClassifier is Send if F is Send, which we require
    // in create_streaming_output. The ArrowArrayStreamReader is also Send.
    unsafe impl<F: Send> Send for StreamingClassifier<F> {}

    impl<F> RecordBatchReader for StreamingClassifier<F>
    where
        F: Fn(&RecordBatch) -> Result<RecordBatch, arrow::error::ArrowError> + Send,
    {
        fn schema(&self) -> SchemaRef {
            self.output_schema.clone()
        }
    }

    // -------------------------------------------------------------------------
    // Helper: Create streaming output
    // -------------------------------------------------------------------------

    /// Creates a streaming FFI output stream from an input stream and classifier.
    unsafe fn create_streaming_output<F>(
        input_stream: *mut FFI_ArrowArrayStream,
        out_stream: *mut FFI_ArrowArrayStream,
        classify_fn: F,
    ) -> Result<(), String>
    where
        F: Fn(&RecordBatch) -> Result<RecordBatch, arrow::error::ArrowError> + Send + 'static,
    {
        let input_reader = ArrowArrayStreamReader::from_raw(input_stream)
            .map_err(|e| format!("Failed to create stream reader: {}", e))?;

        let output_schema = crate::arrow::result_schema();
        let streaming = StreamingClassifier::new(input_reader, output_schema, classify_fn);

        let export_stream = FFI_ArrowArrayStream::new(Box::new(streaming));
        std::ptr::write(out_stream, export_stream);

        Ok(())
    }

    // -------------------------------------------------------------------------
    // Public API: Classification functions (TRUE STREAMING)
    // -------------------------------------------------------------------------

    /// Classifies sequences from an Arrow stream using an Index.
    ///
    /// TRUE STREAMING: Processes one batch at a time. Memory usage is O(batch_size),
    /// not O(total_data). Results are available as soon as each batch is processed.
    ///
    /// # Input Schema
    /// - `id`: Int64 - Query identifier
    /// - `sequence`: Binary/LargeBinary - DNA sequence
    /// - `pair_sequence`: Binary/LargeBinary (optional) - Paired-end sequence
    ///
    /// # Output Schema
    /// - `query_id`: Int64 - Matching query ID
    /// - `bucket_id`: UInt32 - Matched bucket/reference ID
    /// - `score`: Float64 - Classification score (0.0-1.0)
    ///
    /// # Safety
    /// - index_ptr must remain valid until the output stream is fully consumed
    /// - negative_set_ptr (if non-null) must remain valid until output is consumed
    ///
    /// # Returns
    /// 0 on success, -1 on error. Call rype_get_last_error() for details.
    #[no_mangle]
    pub unsafe extern "C" fn rype_classify_arrow_batch(
        index_ptr: *const Index,
        negative_set_ptr: *const RypeNegativeSet,
        input_stream: *mut FFI_ArrowArrayStream,
        threshold: c_double,
        out_stream: *mut FFI_ArrowArrayStream,
    ) -> i32 {
        // Validate parameters
        if index_ptr.is_null() {
            set_last_error("index is NULL".to_string());
            return -1;
        }
        if input_stream.is_null() {
            set_last_error("input_stream is NULL".to_string());
            return -1;
        }
        if out_stream.is_null() {
            set_last_error("out_stream is NULL".to_string());
            return -1;
        }
        if let Err(e) = validate_threshold(threshold) {
            set_last_error(e);
            return -1;
        }

        // Wrap pointers in Send-safe wrappers
        // SAFETY: Caller guarantees pointers remain valid until stream is consumed
        let index_send = SendPtr::new(index_ptr);
        let neg_set_send = if negative_set_ptr.is_null() {
            None
        } else {
            Some(SendPtr::new(negative_set_ptr))
        };

        let classify_fn = move |batch: &RecordBatch| {
            let index = unsafe { index_send.get() };
            let neg_mins: Option<&HashSet<u64>> = neg_set_send
                .as_ref()
                .map(|p| unsafe { &p.get().minimizers });

            classify_arrow_batch(index, neg_mins, batch, threshold)
                .map_err(|e| arrow::error::ArrowError::ExternalError(Box::new(e)))
        };

        match create_streaming_output(input_stream, out_stream, classify_fn) {
            Ok(()) => { clear_last_error(); 0 }
            Err(e) => { set_last_error(e); -1 }
        }
    }

    /// Classifies sequences using an InvertedIndex (recommended for large indices).
    ///
    /// TRUE STREAMING: Processes one batch at a time.
    ///
    /// # Safety
    /// - inverted_ptr must remain valid until the output stream is fully consumed
    #[no_mangle]
    pub unsafe extern "C" fn rype_classify_arrow_batch_inverted(
        inverted_ptr: *const InvertedIndex,
        negative_set_ptr: *const RypeNegativeSet,
        input_stream: *mut FFI_ArrowArrayStream,
        threshold: c_double,
        out_stream: *mut FFI_ArrowArrayStream,
    ) -> i32 {
        if inverted_ptr.is_null() {
            set_last_error("inverted index is NULL".to_string());
            return -1;
        }
        if input_stream.is_null() {
            set_last_error("input_stream is NULL".to_string());
            return -1;
        }
        if out_stream.is_null() {
            set_last_error("out_stream is NULL".to_string());
            return -1;
        }
        if let Err(e) = validate_threshold(threshold) {
            set_last_error(e);
            return -1;
        }

        // Wrap pointers in Send-safe wrappers
        // SAFETY: Caller guarantees pointers remain valid until stream is consumed
        let inverted_send = SendPtr::new(inverted_ptr);
        let neg_set_send = if negative_set_ptr.is_null() {
            None
        } else {
            Some(SendPtr::new(negative_set_ptr))
        };

        let classify_fn = move |batch: &RecordBatch| {
            let inverted = unsafe { inverted_send.get() };
            let neg_mins: Option<&HashSet<u64>> = neg_set_send
                .as_ref()
                .map(|p| unsafe { &p.get().minimizers });

            classify_arrow_batch_inverted(inverted, neg_mins, batch, threshold)
                .map_err(|e| arrow::error::ArrowError::ExternalError(Box::new(e)))
        };

        match create_streaming_output(input_stream, out_stream, classify_fn) {
            Ok(()) => { clear_last_error(); 0 }
            Err(e) => { set_last_error(e); -1 }
        }
    }

    /// Loads a sharded inverted index from a manifest file.
    ///
    /// Sharded indices are used when the full inverted index doesn't fit in memory.
    /// The manifest file describes the shard layout; actual shards are loaded on demand.
    ///
    /// # Returns
    ///
    /// Pointer to ShardedInvertedIndex on success, NULL on error.
    /// Call rype_get_last_error() for details.
    #[no_mangle]
    pub unsafe extern "C" fn rype_sharded_load(path: *const c_char) -> *mut ShardedInvertedIndex {
        if path.is_null() {
            set_last_error("path is NULL".to_string());
            return std::ptr::null_mut();
        }

        let c_str = CStr::from_ptr(path);
        let r_str = match c_str.to_str() {
            Ok(s) => s,
            Err(e) => {
                set_last_error(format!("Invalid UTF-8 in path: {}", e));
                return std::ptr::null_mut();
            }
        };

        match ShardedInvertedIndex::open(Path::new(r_str)) {
            Ok(sharded) => {
                clear_last_error();
                Box::into_raw(Box::new(sharded))
            }
            Err(e) => {
                set_last_error(format!("Failed to load sharded index: {}", e));
                std::ptr::null_mut()
            }
        }
    }

    /// Frees a sharded inverted index. NULL is safe to pass.
    #[no_mangle]
    pub unsafe extern "C" fn rype_sharded_free(ptr: *mut ShardedInvertedIndex) {
        if !ptr.is_null() {
            let _ = Box::from_raw(ptr);
        }
    }

    /// Classifies sequences using a ShardedInvertedIndex (for large indices).
    ///
    /// TRUE STREAMING: Processes one batch at a time.
    ///
    /// # Safety
    /// - sharded_ptr must remain valid until the output stream is fully consumed
    #[no_mangle]
    pub unsafe extern "C" fn rype_classify_arrow_batch_sharded(
        sharded_ptr: *const ShardedInvertedIndex,
        negative_set_ptr: *const RypeNegativeSet,
        input_stream: *mut FFI_ArrowArrayStream,
        threshold: c_double,
        use_merge_join: i32,
        out_stream: *mut FFI_ArrowArrayStream,
    ) -> i32 {
        if sharded_ptr.is_null() {
            set_last_error("sharded index is NULL".to_string());
            return -1;
        }
        if input_stream.is_null() {
            set_last_error("input_stream is NULL".to_string());
            return -1;
        }
        if out_stream.is_null() {
            set_last_error("out_stream is NULL".to_string());
            return -1;
        }
        if let Err(e) = validate_threshold(threshold) {
            set_last_error(e);
            return -1;
        }

        let merge_join = use_merge_join != 0;

        // Wrap pointers in Send-safe wrappers
        // SAFETY: Caller guarantees pointers remain valid until stream is consumed
        let sharded_send = SendPtr::new(sharded_ptr);
        let neg_set_send = if negative_set_ptr.is_null() {
            None
        } else {
            Some(SendPtr::new(negative_set_ptr))
        };

        let classify_fn = move |batch: &RecordBatch| {
            let sharded = unsafe { sharded_send.get() };
            let neg_mins: Option<&HashSet<u64>> = neg_set_send
                .as_ref()
                .map(|p| unsafe { &p.get().minimizers });

            classify_arrow_batch_sharded(sharded, neg_mins, batch, threshold, merge_join)
                .map_err(|e| arrow::error::ArrowError::ExternalError(Box::new(e)))
        };

        match create_streaming_output(input_stream, out_stream, classify_fn) {
            Ok(()) => { clear_last_error(); 0 }
            Err(e) => { set_last_error(e); -1 }
        }
    }

    /// Returns the schema for classification result batches as an FFI_ArrowSchema.
    ///
    /// This allows callers to pre-allocate memory or validate expected output format.
    /// The caller takes ownership and must release the schema.
    ///
    /// # Returns
    ///
    /// 0 on success, -1 on error.
    #[no_mangle]
    pub unsafe extern "C" fn rype_arrow_result_schema(out_schema: *mut FFI_ArrowSchema) -> i32 {
        if out_schema.is_null() {
            set_last_error("out_schema is NULL".to_string());
            return -1;
        }

        let schema = crate::arrow::result_schema();

        match FFI_ArrowSchema::try_from(schema.as_ref()) {
            Ok(ffi_schema) => {
                std::ptr::write(out_schema, ffi_schema);
                clear_last_error();
                0
            }
            Err(e) => {
                set_last_error(format!("Failed to export schema: {}", e));
                -1
            }
        }
    }
}

// Re-export Arrow FFI functions at module level when feature is enabled
#[cfg(feature = "arrow")]
pub use arrow_ffi::*;

#[cfg(test)]
mod c_api_tests {
    use super::*;

    #[test]
    fn test_validate_query_null_seq() {
        let query = RypeQuery {
            id: 1,
            seq: std::ptr::null(),
            seq_len: 100,
            pair_seq: std::ptr::null(),
            pair_len: 0,
        };

        let result = validate_query(&query);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "query seq pointer is NULL");
    }

    #[test]
    fn test_validate_query_zero_len() {
        let seq = CString::new("ACGT").unwrap();
        let query = RypeQuery {
            id: 1,
            seq: seq.as_ptr(),
            seq_len: 0,
            pair_seq: std::ptr::null(),
            pair_len: 0,
        };

        let result = validate_query(&query);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "query seq_len is zero");
    }

    #[test]
    fn test_validate_query_oversized() {
        let seq = CString::new("ACGT").unwrap();
        let query = RypeQuery {
            id: 1,
            seq: seq.as_ptr(),
            seq_len: MAX_SEQUENCE_LENGTH + 1,
            pair_seq: std::ptr::null(),
            pair_len: 0,
        };

        let result = validate_query(&query);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("exceeds maximum"));
    }

    #[test]
    fn test_validate_query_mismatched_pair() {
        let seq = CString::new("ACGT").unwrap();
        let query = RypeQuery {
            id: 1,
            seq: seq.as_ptr(),
            seq_len: 4,
            pair_seq: std::ptr::null(),
            pair_len: 100,  // Non-zero with null pointer!
        };

        let result = validate_query(&query);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("NULL but pair_len is non-zero"));
    }

    #[test]
    fn test_rype_index_load_null_path() {
        let result = rype_index_load(std::ptr::null());
        assert!(result.is_null());

        let err = unsafe {
            let err_ptr = rype_get_last_error();
            assert!(!err_ptr.is_null());
            CStr::from_ptr(err_ptr).to_string_lossy().into_owned()
        };
        assert!(err.contains("NULL"));
    }
}

// Arrow FFI tests
#[cfg(all(test, feature = "arrow"))]
mod arrow_ffi_tests {
    use super::*;
    use crate::{Index, MinimizerWorkspace};
    use arrow::array::{BinaryArray, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::ffi_stream::FFI_ArrowArrayStream;
    use arrow::record_batch::{RecordBatch, RecordBatchIterator};
    use std::sync::Arc;

    /// Generate a DNA sequence with a deterministic pattern.
    fn generate_sequence(len: usize, seed: u8) -> Vec<u8> {
        let bases = [b'A', b'C', b'G', b'T'];
        (0..len)
            .map(|i| bases[(i + seed as usize) % 4])
            .collect()
    }

    /// Create a test batch with the expected schema.
    fn make_test_batch(ids: &[i64], seqs: &[&[u8]]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("sequence", DataType::Binary, false),
        ]));

        let id_array = Int64Array::from(ids.to_vec());
        let seq_array = BinaryArray::from_iter_values(seqs.iter().copied());

        RecordBatch::try_new(schema, vec![Arc::new(id_array), Arc::new(seq_array)]).unwrap()
    }

    /// Create a test index.
    fn create_test_index() -> Index {
        let mut index = Index::new(16, 5, 0x12345).unwrap();
        let mut ws = MinimizerWorkspace::new();

        let ref_seq = generate_sequence(100, 0);
        index.add_record(1, "ref1", &ref_seq, &mut ws);
        index.finalize_bucket(1);
        index.bucket_names.insert(1, "test_bucket".into());

        index
    }

    /// Convert a RecordBatch to an FFI stream.
    fn batch_to_ffi_stream(batch: RecordBatch) -> FFI_ArrowArrayStream {
        let schema = batch.schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        FFI_ArrowArrayStream::new(Box::new(reader))
    }

    /// Convert multiple RecordBatches to an FFI stream.
    fn batches_to_ffi_stream(batches: Vec<RecordBatch>) -> FFI_ArrowArrayStream {
        let schema = batches[0].schema();
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
        FFI_ArrowArrayStream::new(Box::new(reader))
    }

    /// Read all batches from an FFI stream.
    unsafe fn read_all_from_ffi_stream(
        stream: &mut FFI_ArrowArrayStream,
    ) -> Vec<RecordBatch> {
        let reader = ArrowArrayStreamReader::from_raw(stream as *mut _).unwrap();
        reader.collect::<Result<Vec<_>, _>>().unwrap()
    }

    #[test]
    fn test_arrow_ffi_classify_basic() {
        let index = create_test_index();
        let index_ptr = &index as *const Index;

        let query_seq = generate_sequence(100, 0);
        let batch = make_test_batch(&[101], &[&query_seq]);

        let mut input_stream = batch_to_ffi_stream(batch);
        let mut output_stream: FFI_ArrowArrayStream = unsafe { std::mem::zeroed() };

        let result = unsafe {
            rype_classify_arrow_batch(
                index_ptr,
                std::ptr::null(),
                &mut input_stream,
                0.0,
                &mut output_stream,
            )
        };

        assert_eq!(result, 0, "Classification should succeed");

        // Read results
        let result_batches = unsafe { read_all_from_ffi_stream(&mut output_stream) };
        assert_eq!(result_batches.len(), 1, "Should have one result batch");

        let result_batch = &result_batches[0];
        assert!(result_batch.num_rows() > 0, "Should have classification hits");

        // Verify schema
        assert_eq!(result_batch.num_columns(), 3);
        assert_eq!(result_batch.schema().field(0).name(), "query_id");
        assert_eq!(result_batch.schema().field(1).name(), "bucket_id");
        assert_eq!(result_batch.schema().field(2).name(), "score");

        // Release output stream
        unsafe {
            if let Some(release) = output_stream.release {
                release(&mut output_stream);
            }
        }
    }

    #[test]
    fn test_arrow_ffi_classify_multiple_batches() {
        let index = create_test_index();
        let index_ptr = &index as *const Index;

        // Create multiple input batches
        let batch1 = make_test_batch(&[1, 2], &[&generate_sequence(100, 0), &generate_sequence(100, 1)]);
        let batch2 = make_test_batch(&[3, 4], &[&generate_sequence(100, 0), &generate_sequence(100, 2)]);
        let batch3 = make_test_batch(&[5], &[&generate_sequence(100, 0)]);

        let mut input_stream = batches_to_ffi_stream(vec![batch1, batch2, batch3]);
        let mut output_stream: FFI_ArrowArrayStream = unsafe { std::mem::zeroed() };

        let result = unsafe {
            rype_classify_arrow_batch(
                index_ptr,
                std::ptr::null(),
                &mut input_stream,
                0.0,
                &mut output_stream,
            )
        };

        assert_eq!(result, 0, "Classification should succeed");

        // Read results - should have results from ALL input batches
        let result_batches = unsafe { read_all_from_ffi_stream(&mut output_stream) };

        // Count total result rows
        let total_rows: usize = result_batches.iter().map(|b| b.num_rows()).sum();
        assert!(total_rows > 0, "Should have results from all batches");

        // Verify we have results for query IDs from all batches (1-5)
        let mut seen_query_ids = std::collections::HashSet::new();
        for batch in &result_batches {
            let query_ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            for i in 0..batch.num_rows() {
                seen_query_ids.insert(query_ids.value(i));
            }
        }

        // All queries with seed 0 should match (IDs 1, 3, 5)
        assert!(seen_query_ids.contains(&1), "Should have result for query 1");
        assert!(seen_query_ids.contains(&3), "Should have result for query 3");
        assert!(seen_query_ids.contains(&5), "Should have result for query 5");

        // Release output stream
        unsafe {
            if let Some(release) = output_stream.release {
                release(&mut output_stream);
            }
        }
    }

    #[test]
    fn test_arrow_ffi_null_index() {
        let batch = make_test_batch(&[1], &[b"ACGT"]);
        let mut input_stream = batch_to_ffi_stream(batch);
        let mut output_stream: FFI_ArrowArrayStream = unsafe { std::mem::zeroed() };

        let result = unsafe {
            rype_classify_arrow_batch(
                std::ptr::null(),
                std::ptr::null(),
                &mut input_stream,
                0.1,
                &mut output_stream,
            )
        };

        assert_eq!(result, -1, "Should fail with null index");

        let err = unsafe {
            let err_ptr = rype_get_last_error();
            CStr::from_ptr(err_ptr).to_string_lossy().into_owned()
        };
        assert!(err.contains("NULL"), "Error should mention NULL");
    }

    #[test]
    fn test_arrow_ffi_null_input_stream() {
        let index = create_test_index();
        let index_ptr = &index as *const Index;
        let mut output_stream: FFI_ArrowArrayStream = unsafe { std::mem::zeroed() };

        let result = unsafe {
            rype_classify_arrow_batch(
                index_ptr,
                std::ptr::null(),
                std::ptr::null_mut(),
                0.1,
                &mut output_stream,
            )
        };

        assert_eq!(result, -1, "Should fail with null input stream");
    }

    #[test]
    fn test_arrow_ffi_null_output_stream() {
        let index = create_test_index();
        let index_ptr = &index as *const Index;
        let batch = make_test_batch(&[1], &[b"ACGT"]);
        let mut input_stream = batch_to_ffi_stream(batch);

        let result = unsafe {
            rype_classify_arrow_batch(
                index_ptr,
                std::ptr::null(),
                &mut input_stream,
                0.1,
                std::ptr::null_mut(),
            )
        };

        assert_eq!(result, -1, "Should fail with null output stream");
    }

    #[test]
    fn test_arrow_ffi_invalid_threshold() {
        let index = create_test_index();
        let index_ptr = &index as *const Index;
        let batch = make_test_batch(&[1], &[&generate_sequence(100, 0)]);
        let mut output_stream: FFI_ArrowArrayStream = unsafe { std::mem::zeroed() };

        // Test negative threshold
        let mut input_stream = batch_to_ffi_stream(batch.clone());
        let result = unsafe {
            rype_classify_arrow_batch(
                index_ptr,
                std::ptr::null(),
                &mut input_stream,
                -0.1,
                &mut output_stream,
            )
        };
        assert_eq!(result, -1, "Should fail with negative threshold");

        // Test threshold > 1
        let mut input_stream = batch_to_ffi_stream(batch.clone());
        let result = unsafe {
            rype_classify_arrow_batch(
                index_ptr,
                std::ptr::null(),
                &mut input_stream,
                1.5,
                &mut output_stream,
            )
        };
        assert_eq!(result, -1, "Should fail with threshold > 1");

        // Test NaN threshold
        let mut input_stream = batch_to_ffi_stream(batch);
        let result = unsafe {
            rype_classify_arrow_batch(
                index_ptr,
                std::ptr::null(),
                &mut input_stream,
                f64::NAN,
                &mut output_stream,
            )
        };
        assert_eq!(result, -1, "Should fail with NaN threshold");
    }

    #[test]
    fn test_arrow_ffi_empty_stream() {
        let index = create_test_index();
        let index_ptr = &index as *const Index;

        // Create an empty stream
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("sequence", DataType::Binary, false),
        ]));
        let reader = RecordBatchIterator::new(std::iter::empty(), schema);
        let mut input_stream = FFI_ArrowArrayStream::new(Box::new(reader));
        let mut output_stream: FFI_ArrowArrayStream = unsafe { std::mem::zeroed() };

        let result = unsafe {
            rype_classify_arrow_batch(
                index_ptr,
                std::ptr::null(),
                &mut input_stream,
                0.1,
                &mut output_stream,
            )
        };

        assert_eq!(result, 0, "Empty stream should succeed");

        // Result should be empty
        let result_batches = unsafe { read_all_from_ffi_stream(&mut output_stream) };
        let total_rows: usize = result_batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 0, "Empty input should produce empty output");

        unsafe {
            if let Some(release) = output_stream.release {
                release(&mut output_stream);
            }
        }
    }

    #[test]
    fn test_arrow_ffi_result_schema() {
        let mut schema: FFI_ArrowSchema = unsafe { std::mem::zeroed() };

        let result = unsafe { rype_arrow_result_schema(&mut schema) };
        assert_eq!(result, 0, "Should succeed");

        // Import and verify schema - this also takes ownership and releases
        let imported_schema = Schema::try_from(&schema).expect("Should be valid schema");

        assert_eq!(imported_schema.fields().len(), 3);
        assert_eq!(imported_schema.field(0).name(), "query_id");
        assert_eq!(imported_schema.field(0).data_type(), &DataType::Int64);
        assert_eq!(imported_schema.field(1).name(), "bucket_id");
        assert_eq!(imported_schema.field(1).data_type(), &DataType::UInt32);
        assert_eq!(imported_schema.field(2).name(), "score");
        assert_eq!(imported_schema.field(2).data_type(), &DataType::Float64);
        // Schema is released when imported_schema drops
    }

    #[test]
    fn test_arrow_ffi_result_schema_null() {
        let result = unsafe { rype_arrow_result_schema(std::ptr::null_mut()) };
        assert_eq!(result, -1, "Should fail with null pointer");
    }
}

