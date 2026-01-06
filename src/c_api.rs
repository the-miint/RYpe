use crate::{Index, QueryRecord, classify_batch}; // Removed HitResult
use std::ffi::{CStr, CString};
use std::slice;
use std::path::Path;
use std::cell::RefCell;
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

#[no_mangle]
pub extern "C" fn rype_classify(
    index_ptr: *const Index,
    queries_ptr: *const RypeQuery,
    num_queries: size_t,
    threshold: c_double
) -> *mut RypeResultArray {
    if index_ptr.is_null() || queries_ptr.is_null() || num_queries == 0 {
        set_last_error("Invalid arguments: index, queries, or num_queries is invalid".to_string());
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

        let hits = classify_batch(index, &rust_queries, threshold);

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

