use crate::{Index, QueryRecord, classify_batch}; // Removed HitResult
use std::ffi::CStr;
use std::slice;
use std::path::Path;
use libc::{c_char, size_t, c_double};

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
    if path.is_null() { return std::ptr::null_mut(); }
    let c_str = unsafe { CStr::from_ptr(path) };
    let r_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    match Index::load(Path::new(r_str)) {
        Ok(idx) => Box::into_raw(Box::new(idx)),
        Err(_) => std::ptr::null_mut(),
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
        return std::ptr::null_mut();
    }

    let index = unsafe { &*index_ptr };
    let c_queries = unsafe { slice::from_raw_parts(queries_ptr, num_queries) };

    let rust_queries: Vec<QueryRecord> = c_queries.iter().map(|q| {
        let s1 = unsafe { slice::from_raw_parts(q.seq as *const u8, q.seq_len) };
        let s2 = if !q.pair_seq.is_null() && q.pair_len > 0 {
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
}

#[no_mangle]
pub extern "C" fn rype_results_free(ptr: *mut RypeResultArray) {
    if !ptr.is_null() {
        unsafe {
            let array = Box::from_raw(ptr);
            let _ = Vec::from_raw_parts(array.data, array.len, array.capacity);
        }
    }
}

