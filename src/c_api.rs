use crate::{Index, InvertedIndex, QueryRecord, classify_batch, classify_batch_inverted};
use std::ffi::{CStr, CString};
use std::slice;
use std::path::Path;
use std::cell::RefCell;
use std::collections::HashMap;
use libc::{c_char, size_t, c_double};

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

