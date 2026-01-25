//! C API for rype - FFI bindings for external language integration.
//!
//! This module provides C-compatible bindings for the rype library, enabling
//! integration with C, Python (via ctypes/cffi), and other languages.
//!
//! # Index Format
//!
//! All indices are Parquet-based sharded inverted indices stored as directories:
//!
//! ```text
//! index.ryxdi/
//! ├── manifest.toml           # TOML metadata (k, w, salt, bucket info)
//! ├── buckets.parquet         # Bucket metadata (id, name, sources)
//! └── inverted/
//!     ├── shard.0.parquet     # (minimizer, bucket_id) pairs
//!     └── ...                 # Additional shards for large indices
//! ```
//!
//! # Safety
//!
//! All functions that take raw pointers perform null checks and validation internally.
//! These `extern "C"` functions cannot be marked `unsafe` in Rust since they are
//! designed to be called from C code.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::classify_batch_sharded_sequential;
use crate::constants::MAX_SEQUENCE_LENGTH;
use crate::memory::{detect_available_memory, parse_byte_suffix};
use crate::{QueryRecord, ShardedInvertedIndex};
use libc::{c_char, c_double, size_t};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString};
use std::path::Path;
use std::slice;

#[cfg(feature = "arrow-ffi")]
use crate::arrow::{classify_arrow_batch_sharded, classify_arrow_batch_sharded_best_hit};
#[cfg(feature = "arrow-ffi")]
use arrow::ffi::FFI_ArrowSchema;
#[cfg(feature = "arrow-ffi")]
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
#[cfg(feature = "arrow-ffi")]
use arrow::record_batch::RecordBatch;

// --- Unified Index Type ---

/// Unified index type that wraps a ShardedInvertedIndex.
///
/// The only supported format is the Parquet inverted index (.ryxdi directory).
pub struct RypeIndex(ShardedInvertedIndex);

impl RypeIndex {
    /// Returns the k-mer size.
    pub fn k(&self) -> usize {
        self.0.k()
    }

    /// Returns the window size.
    pub fn w(&self) -> usize {
        self.0.w()
    }

    /// Returns the salt.
    pub fn salt(&self) -> u64 {
        self.0.salt()
    }

    /// Returns the number of buckets.
    pub fn num_buckets(&self) -> usize {
        self.0.manifest().bucket_names.len()
    }

    /// Returns the bucket name for a given ID, if available.
    pub fn bucket_name(&self, bucket_id: u32) -> Option<&str> {
        self.0
            .manifest()
            .bucket_names
            .get(&bucket_id)
            .map(|s| s.as_str())
    }

    /// Returns the number of shards.
    pub fn num_shards(&self) -> usize {
        self.0.num_shards()
    }

    /// Estimate the memory footprint of all shards when loaded.
    ///
    /// Returns a **lower bound estimate** based on raw data size:
    /// - minimizer column: num_entries × 8 bytes (u64)
    /// - bucket_id column: num_entries × 4 bytes (u32)
    ///
    /// Actual memory usage will be higher due to Arrow array overhead,
    /// Parquet decompression buffers, and temporary allocations during
    /// classification. Plan for 1.5-2x this estimate in practice.
    pub fn estimate_memory_bytes(&self) -> usize {
        let manifest = self.0.manifest();
        manifest
            .shards
            .iter()
            .map(|s| s.num_bucket_ids * 12) // 8 bytes minimizer + 4 bytes bucket_id per entry
            .sum()
    }

    /// Estimate the memory needed to load the largest single shard.
    ///
    /// Returns the estimated size of the largest shard in bytes.
    /// Use this for memory planning when classifying with limited RAM.
    pub fn largest_shard_bytes(&self) -> usize {
        let manifest = self.0.manifest();
        manifest
            .shards
            .iter()
            .map(|s| s.num_bucket_ids * 12) // 8 bytes minimizer + 4 bytes bucket_id per entry
            .max()
            .unwrap_or(0)
    }
}

// --- Error Reporting ---

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error(err: String) {
    LAST_ERROR.with(|e| {
        // Sanitize null bytes to prevent silent error suppression
        let sanitized = err.replace('\0', "\\0");
        *e.borrow_mut() =
            Some(CString::new(sanitized).expect("sanitized string should not contain null bytes"));
    });
}

fn clear_last_error() {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = None;
    });
}

/// Validates that a pointer is non-null and properly aligned for type T.
/// Returns true if valid, false otherwise.
///
/// # Limitations
///
/// This does NOT verify the pointer points to valid memory or a live object.
/// Dereferencing an aligned pointer to freed memory causes undefined behavior.
/// This is an inherent limitation of C FFI - we cannot fully validate C pointers.
#[inline]
fn is_valid_ptr<T>(ptr: *const T) -> bool {
    !ptr.is_null() && (ptr as usize) % std::mem::align_of::<T>() == 0
}

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

/// Loads an index from disk.
///
/// Supported format:
/// - Parquet inverted index directory (.ryxdi with manifest.toml)
///
/// Returns NULL on error; call rype_get_last_error() for details.
#[no_mangle]
pub extern "C" fn rype_index_load(path: *const c_char) -> *mut RypeIndex {
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

    let path = Path::new(r_str);

    // Open as Parquet inverted index
    match ShardedInvertedIndex::open(path) {
        Ok(idx) => {
            clear_last_error();
            Box::into_raw(Box::new(RypeIndex(idx)))
        }
        Err(e) => {
            set_last_error(format!("Failed to load index: {}", e));
            std::ptr::null_mut()
        }
    }
}

/// Frees an index. NULL is safe to pass.
#[no_mangle]
pub extern "C" fn rype_index_free(ptr: *mut RypeIndex) {
    if !ptr.is_null() {
        // Clear bucket name cache entries for this index to prevent memory leak
        let index_addr = ptr as usize;
        BUCKET_NAME_CACHE.with(|cache| {
            cache
                .borrow_mut()
                .retain(|(addr, _), _| *addr != index_addr);
        });

        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

// --- Index Metadata Accessors ---
//
// Safety: All accessor functions validate that pointers are non-null and
// properly aligned before dereferencing. Misaligned pointers return 0/-1.

/// Returns the k-mer size of the index, or 0 if index is NULL or misaligned.
#[no_mangle]
pub extern "C" fn rype_index_k(index_ptr: *const RypeIndex) -> size_t {
    if !is_valid_ptr(index_ptr) {
        return 0;
    }
    let index = unsafe { &*index_ptr };
    index.k()
}

/// Returns the window size of the index, or 0 if index is NULL or misaligned.
#[no_mangle]
pub extern "C" fn rype_index_w(index_ptr: *const RypeIndex) -> size_t {
    if !is_valid_ptr(index_ptr) {
        return 0;
    }
    let index = unsafe { &*index_ptr };
    index.w()
}

/// Returns the salt of the index, or 0 if index is NULL or misaligned.
#[no_mangle]
pub extern "C" fn rype_index_salt(index_ptr: *const RypeIndex) -> u64 {
    if !is_valid_ptr(index_ptr) {
        return 0;
    }
    let index = unsafe { &*index_ptr };
    index.salt()
}

/// Returns the number of buckets in the index.
///
/// Returns 0 if index is NULL or misaligned.
#[no_mangle]
pub extern "C" fn rype_index_num_buckets(index_ptr: *const RypeIndex) -> i32 {
    if !is_valid_ptr(index_ptr) {
        return 0;
    }
    let index = unsafe { &*index_ptr };
    index.num_buckets() as i32
}

/// Returns whether the index is sharded (always true for Parquet indices).
#[no_mangle]
pub extern "C" fn rype_index_is_sharded(index_ptr: *const RypeIndex) -> i32 {
    if !is_valid_ptr(index_ptr) {
        return 0;
    }
    1 // Always true for Parquet format
}

/// Returns the number of shards in the index.
#[no_mangle]
pub extern "C" fn rype_index_num_shards(index_ptr: *const RypeIndex) -> u32 {
    if !is_valid_ptr(index_ptr) {
        return 0;
    }
    let index = unsafe { &*index_ptr };
    index.num_shards() as u32
}

// --- Memory Management ---

/// Returns the estimated memory footprint of the loaded index in bytes.
///
/// Returns only the manifest memory (shards load on-demand).
///
/// Returns 0 if index_ptr is invalid.
#[no_mangle]
pub extern "C" fn rype_index_memory_bytes(index_ptr: *const RypeIndex) -> size_t {
    if !is_valid_ptr(index_ptr) {
        return 0;
    }
    let index = unsafe { &*index_ptr };
    index.estimate_memory_bytes()
}

/// Returns the estimated size of the largest shard in bytes.
///
/// Use this for memory planning when classifying against sharded indices.
#[no_mangle]
pub extern "C" fn rype_index_largest_shard_bytes(index_ptr: *const RypeIndex) -> size_t {
    if !is_valid_ptr(index_ptr) {
        return 0;
    }
    let index = unsafe { &*index_ptr };
    index.largest_shard_bytes()
}

/// Detect available system memory.
///
/// On Linux: tries cgroups v2, cgroups v1, then /proc/meminfo
/// On macOS: uses sysctl hw.memsize
/// Fallback: returns 8GB
///
/// Returns the detected memory in bytes.
#[no_mangle]
pub extern "C" fn rype_detect_available_memory() -> size_t {
    detect_available_memory().bytes
}

/// Parse a byte size string (e.g., "4G", "512M", "1024K").
///
/// Supports suffixes: B, K, KB, M, MB, G, GB, T, TB (case-insensitive)
/// Also supports decimal values: "1.5G"
///
/// Returns the size in bytes, or 0 for "auto" or parse errors.
/// To distinguish between "auto" and errors, call rype_get_last_error() after:
/// - If returns 0 and rype_get_last_error() is NULL: input was "auto"
/// - If returns 0 and rype_get_last_error() is non-NULL: parse error
#[no_mangle]
pub extern "C" fn rype_parse_byte_suffix(str_ptr: *const c_char) -> size_t {
    if str_ptr.is_null() {
        set_last_error("str is NULL".to_string());
        return 0;
    }

    let c_str = unsafe { CStr::from_ptr(str_ptr) };
    let s = match c_str.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid UTF-8 in string: {}", e));
            return 0;
        }
    };

    match parse_byte_suffix(s) {
        Ok(Some(bytes)) => {
            clear_last_error();
            bytes
        }
        Ok(None) => {
            // "auto" returns 0 with no error set
            clear_last_error();
            0
        }
        Err(e) => {
            set_last_error(format!("{}", e));
            0
        }
    }
}

// --- Bucket Name Lookup ---

// Thread-local storage for bucket name CStrings to maintain lifetime.
// Key includes both index pointer and bucket_id to avoid collisions when
// multiple indices have the same bucket IDs.
thread_local! {
    static BUCKET_NAME_CACHE: RefCell<HashMap<(usize, u32), CString>> = RefCell::new(HashMap::new());
}

/// Returns the name of a bucket by ID, or NULL if not found.
///
/// The returned string is owned by the library and must NOT be freed by the caller.
/// The string remains valid until the index is freed or this thread exits.
///
/// # Safety
/// - index_ptr must be a valid pointer obtained from rype_index_load()
/// - The returned pointer is only valid while index_ptr remains valid
#[no_mangle]
pub extern "C" fn rype_bucket_name(index_ptr: *const RypeIndex, bucket_id: u32) -> *const c_char {
    if !is_valid_ptr(index_ptr) {
        return std::ptr::null();
    }

    let index = unsafe { &*index_ptr };

    match index.bucket_name(bucket_id) {
        Some(name) => {
            BUCKET_NAME_CACHE.with(|cache| {
                let mut cache = cache.borrow_mut();
                // Key includes index pointer to avoid collisions between indices
                let key = (index_ptr as usize, bucket_id);
                // Insert/update the CString in our cache
                let cstring = match CString::new(name) {
                    Ok(s) => s,
                    Err(_) => return std::ptr::null(),
                };
                cache.insert(key, cstring);
                // Return pointer to the cached CString
                cache
                    .get(&key)
                    .map(|s| s.as_ptr())
                    .unwrap_or(std::ptr::null())
            })
        }
        None => std::ptr::null(),
    }
}

// --- Negative Index API ---

/// Opaque handle to a pre-built negative minimizer set for filtering.
/// This allows efficient reuse when classifying multiple batches.
pub struct RypeNegativeSet {
    minimizers: HashSet<u64>,
}

/// Creates a negative minimizer set from an index.
///
/// Returns NULL because negative set creation requires loading all minimizers
/// from all shards into memory, which is not currently supported.
///
/// For negative filtering, create the minimizer set directly from the original
/// FASTA reference sequences using the CLI or Rust API.
#[no_mangle]
pub extern "C" fn rype_negative_set_create(
    _negative_index_ptr: *const RypeIndex,
) -> *mut RypeNegativeSet {
    set_last_error("Negative set creation is not supported for Parquet indices. Create minimizers directly from FASTA reference sequences instead.".to_string());
    std::ptr::null_mut()
}

/// Frees a negative set. NULL is safe to pass.
#[no_mangle]
pub extern "C" fn rype_negative_set_free(ptr: *mut RypeNegativeSet) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
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

// --- Primary Index Classification ---

/// Classifies queries using any index type without negative filtering.
/// For negative filtering support, use rype_classify_with_negative().
#[no_mangle]
pub extern "C" fn rype_classify(
    index_ptr: *const RypeIndex,
    queries_ptr: *const RypeQuery,
    num_queries: size_t,
    threshold: c_double,
) -> *mut RypeResultArray {
    // Delegate to the full version with NULL negative set
    rype_classify_with_negative(
        index_ptr,
        std::ptr::null(),
        queries_ptr,
        num_queries,
        threshold,
    )
}

/// Classifies queries using a Parquet inverted index with optional negative filtering.
///
/// Parameters:
/// - index: Index loaded via rype_index_load
/// - negative_set: Optional negative set for filtering (NULL to disable)
/// - queries: Array of query sequences
/// - num_queries: Number of queries
/// - threshold: Classification threshold (0.0-1.0)
///
/// Negative filtering removes minimizers that appear in the negative set from
/// queries before scoring, reducing false positives from contaminating sequences.
///
/// # Safety
/// - index_ptr must be a valid pointer obtained from rype_index_load()
/// - queries_ptr must point to a valid array of num_queries RypeQuery structs
/// - All sequence pointers in queries must remain valid for the duration of this call
/// - negative_set_ptr (if non-null) must be obtained from rype_negative_set_create()
#[no_mangle]
pub extern "C" fn rype_classify_with_negative(
    index_ptr: *const RypeIndex,
    negative_set_ptr: *const RypeNegativeSet,
    queries_ptr: *const RypeQuery,
    num_queries: size_t,
    threshold: c_double,
) -> *mut RypeResultArray {
    classify_internal(
        index_ptr,
        negative_set_ptr,
        queries_ptr,
        num_queries,
        threshold,
        false,
    )
}

/// Internal classification helper shared by all classify functions.
///
/// # Arguments
/// * `best_hit_only` - If true, filter results to keep only the best hit per query
fn classify_internal(
    index_ptr: *const RypeIndex,
    negative_set_ptr: *const RypeNegativeSet,
    queries_ptr: *const RypeQuery,
    num_queries: size_t,
    threshold: c_double,
    best_hit_only: bool,
) -> *mut RypeResultArray {
    // Validate pointers with alignment checks
    if !is_valid_ptr(index_ptr) {
        set_last_error("index is NULL or misaligned".to_string());
        return std::ptr::null_mut();
    }
    if queries_ptr.is_null() || num_queries == 0 {
        set_last_error("Invalid arguments: queries is NULL or num_queries is zero".to_string());
        return std::ptr::null_mut();
    }
    if (queries_ptr as usize) % std::mem::align_of::<RypeQuery>() != 0 {
        set_last_error("queries pointer is misaligned".to_string());
        return std::ptr::null_mut();
    }
    if !negative_set_ptr.is_null() && !is_valid_ptr(negative_set_ptr) {
        set_last_error("negative_set pointer is misaligned".to_string());
        return std::ptr::null_mut();
    }

    // Validate threshold
    if !threshold.is_finite() {
        set_last_error("Invalid threshold: must be finite".to_string());
        return std::ptr::null_mut();
    }
    if !(0.0..=1.0).contains(&threshold) {
        set_last_error(format!(
            "Invalid threshold: {} (expected 0.0-1.0)",
            threshold
        ));
        return std::ptr::null_mut();
    }

    // Validate num_queries is reasonable (prevent integer overflow)
    if num_queries > isize::MAX as size_t {
        set_last_error("num_queries exceeds maximum".to_string());
        return std::ptr::null_mut();
    }

    // Validate all queries BEFORE entering catch_unwind for better error messages
    let c_queries = unsafe { slice::from_raw_parts(queries_ptr, num_queries) };
    for (idx, q) in c_queries.iter().enumerate() {
        if let Err(msg) = validate_query(q) {
            set_last_error(format!("Query {} validation failed: {}", idx, msg));
            return std::ptr::null_mut();
        }
    }

    // Now proceed with classification inside catch_unwind to handle any panics
    // from the underlying classification functions
    let result = std::panic::catch_unwind(|| {
        let index = unsafe { &*index_ptr };

        // Get negative set if provided
        let neg_mins: Option<&HashSet<u64>> = if negative_set_ptr.is_null() {
            None
        } else {
            let neg_set = unsafe { &*negative_set_ptr };
            Some(&neg_set.minimizers)
        };

        // Build query records (validation already done above)
        let rust_queries: Vec<QueryRecord> = c_queries
            .iter()
            .map(|q| {
                let s1 = unsafe { slice::from_raw_parts(q.seq as *const u8, q.seq_len) };
                let s2 = if !q.pair_seq.is_null() {
                    Some(unsafe { slice::from_raw_parts(q.pair_seq as *const u8, q.pair_len) })
                } else {
                    None
                };
                (q.id, s1, s2)
            })
            .collect();

        // Classify using sharded inverted index
        let hits =
            classify_batch_sharded_sequential(&index.0, neg_mins, &rust_queries, threshold, None)
                .map_err(|e| format!("Classification failed: {}", e))?;

        // Apply best-hit filter if requested
        let hits = if best_hit_only {
            crate::classify::filter_best_hits(hits)
        } else {
            hits
        };

        Ok(hits)
    });

    // Handle the result
    match result {
        Ok(Ok(hits)) => {
            // Success - convert hits to C array
            let mut c_hits: Vec<RypeHit> = hits
                .into_iter()
                .map(|h| RypeHit {
                    query_id: h.query_id,
                    bucket_id: h.bucket_id,
                    score: h.score,
                })
                .collect();

            let len = c_hits.len();
            let capacity = c_hits.capacity();
            let data = if c_hits.is_empty() {
                std::ptr::null_mut()
            } else {
                let ptr = c_hits.as_mut_ptr();
                std::mem::forget(c_hits);
                ptr
            };

            let result_array = Box::new(RypeResultArray {
                data,
                len,
                capacity,
            });
            clear_last_error();
            Box::into_raw(result_array)
        }
        Ok(Err(e)) => {
            // Classification function returned an error
            set_last_error(e);
            std::ptr::null_mut()
        }
        Err(panic_err) => {
            // Panic occurred - extract message
            let msg = if let Some(s) = panic_err.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = panic_err.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic during classification".to_string()
            };
            set_last_error(msg);
            std::ptr::null_mut()
        }
    }
}

/// Classifies queries and returns only the best hit per query.
///
/// Same as rype_classify() but filters results to keep only the highest-scoring
/// bucket for each query. If multiple buckets tie for the best score, one is
/// chosen arbitrarily.
///
/// # Safety
/// - index_ptr must be a valid pointer obtained from rype_index_load()
/// - queries_ptr must point to a valid array of num_queries RypeQuery structs
/// - All sequence pointers in queries must remain valid for the duration of this call
#[no_mangle]
pub extern "C" fn rype_classify_best_hit(
    index_ptr: *const RypeIndex,
    queries_ptr: *const RypeQuery,
    num_queries: size_t,
    threshold: c_double,
) -> *mut RypeResultArray {
    rype_classify_best_hit_with_negative(
        index_ptr,
        std::ptr::null(),
        queries_ptr,
        num_queries,
        threshold,
    )
}

/// Classifies queries with negative filtering and returns only the best hit per query.
///
/// Same as rype_classify_with_negative() but filters results to keep only the
/// highest-scoring bucket for each query. If multiple buckets tie for the best
/// score, one is chosen arbitrarily.
///
/// # Safety
/// - index_ptr must be a valid pointer obtained from rype_index_load()
/// - queries_ptr must point to a valid array of num_queries RypeQuery structs
/// - All sequence pointers in queries must remain valid for the duration of this call
/// - negative_set_ptr (if non-null) must be obtained from rype_negative_set_create()
#[no_mangle]
pub extern "C" fn rype_classify_best_hit_with_negative(
    index_ptr: *const RypeIndex,
    negative_set_ptr: *const RypeNegativeSet,
    queries_ptr: *const RypeQuery,
    num_queries: size_t,
    threshold: c_double,
) -> *mut RypeResultArray {
    classify_internal(
        index_ptr,
        negative_set_ptr,
        queries_ptr,
        num_queries,
        threshold,
        true,
    )
}

#[no_mangle]
pub extern "C" fn rype_results_free(ptr: *mut RypeResultArray) {
    if !ptr.is_null() {
        unsafe {
            let array = Box::from_raw(ptr);

            // Safety check: verify data pointer is valid before reconstructing Vec
            // A null data pointer with non-zero len/capacity indicates corruption or double-free
            if array.data.is_null() {
                if array.len > 0 || array.capacity > 0 {
                    // This is undefined behavior territory - abort to prevent memory corruption
                    eprintln!("FATAL: Corrupted RypeResultArray detected in rype_results_free (null data with len={}, capacity={}). Possible double-free.", array.len, array.capacity);
                    std::process::abort();
                }
                // Empty array with null data is valid (no allocation to free)
                return;
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
        use std::mem::{align_of, size_of};

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
        use std::mem::{align_of, size_of};
        // i64 (8) + u32 (4) + padding (4) + f64 (8) = 24 bytes
        assert_eq!(size_of::<RypeHit>(), 24);
        assert_eq!(align_of::<RypeHit>(), 8);
    }

    #[test]
    fn test_rype_result_array_layout() {
        use std::mem::{align_of, size_of};
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
// - Multiple threads can share the same RypeIndex
// - Each thread must use its own FFI_ArrowArrayStream for input/output

#[cfg(feature = "arrow-ffi")]
mod arrow_ffi {
    use super::*;
    use arrow::datatypes::SchemaRef;
    use arrow::record_batch::RecordBatchReader;

    // -------------------------------------------------------------------------
    // Helper: Validate threshold
    // -------------------------------------------------------------------------

    fn validate_threshold(threshold: c_double) -> Result<(), String> {
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            Err(format!(
                "Invalid threshold: {} (expected finite value 0.0-1.0)",
                threshold
            ))
        } else {
            Ok(())
        }
    }

    // -------------------------------------------------------------------------
    // Send-safe pointer wrappers for FFI
    // -------------------------------------------------------------------------
    //
    // Raw pointers aren't Send in Rust. For FFI, we need to wrap them and
    // assert Send safety. The caller guarantees the pointers remain valid
    // for the lifetime of the stream.
    //
    // We use type-specific wrappers rather than a generic SendPtr<T> to:
    // 1. Limit Send impl to only the types we've verified are safe
    // 2. Document the safety invariants for each specific type
    // 3. Prevent accidental misuse with other pointer types

    /// Send-safe wrapper for RypeIndex pointer.
    /// SAFETY: RypeIndex is immutable during classification (read-only access).
    /// Caller must guarantee the pointer remains valid until the stream is consumed.
    struct SendRypeIndexPtr(*const RypeIndex);

    // SAFETY: RypeIndex is read-only during classification. The ShardedInvertedIndex
    // is safe for concurrent reads.
    unsafe impl Send for SendRypeIndexPtr {}
    unsafe impl Sync for SendRypeIndexPtr {}

    impl SendRypeIndexPtr {
        /// Create a new send-safe wrapper.
        /// SAFETY: Caller must ensure ptr is valid and remains valid for the wrapper's lifetime.
        unsafe fn new(ptr: *const RypeIndex) -> Self {
            SendRypeIndexPtr(ptr)
        }

        /// Get a reference to the underlying RypeIndex.
        /// SAFETY: Caller must ensure the pointer is still valid.
        unsafe fn get(&self) -> &RypeIndex {
            &*self.0
        }
    }

    /// Send-safe wrapper for RypeNegativeSet pointer.
    /// SAFETY: RypeNegativeSet contains a HashSet<u64> which is read-only during classification.
    /// Caller must guarantee the pointer remains valid until the stream is consumed.
    struct SendRypeNegativeSetPtr(*const RypeNegativeSet);

    // SAFETY: RypeNegativeSet.minimizers is a HashSet<u64> that is only read during classification.
    // HashSet is safe for concurrent reads.
    unsafe impl Send for SendRypeNegativeSetPtr {}
    unsafe impl Sync for SendRypeNegativeSetPtr {}

    impl SendRypeNegativeSetPtr {
        /// Create a new send-safe wrapper.
        /// SAFETY: Caller must ensure ptr is valid and remains valid for the wrapper's lifetime.
        unsafe fn new(ptr: *const RypeNegativeSet) -> Self {
            SendRypeNegativeSetPtr(ptr)
        }

        /// Get a reference to the underlying RypeNegativeSet.
        /// SAFETY: Caller must ensure the pointer is still valid.
        unsafe fn get(&self) -> &RypeNegativeSet {
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
    // Public API: Unified Classification (TRUE STREAMING)
    // -------------------------------------------------------------------------

    /// Classifies sequences from an Arrow stream using a Parquet inverted index.
    ///
    /// TRUE STREAMING: Processes one batch at a time. Memory usage is O(batch_size),
    /// not O(total_data). Results are available as soon as each batch is processed.
    ///
    /// # Arguments
    /// - `index_ptr`: Pointer to RypeIndex (from rype_index_load)
    /// - `negative_set_ptr`: Optional pointer to RypeNegativeSet (can be NULL)
    /// - `input_stream`: Arrow input stream with sequences
    /// - `threshold`: Minimum score threshold (0.0-1.0)
    /// - `use_merge_join`: 1 = use merge-join strategy (more efficient when minimizers
    ///   overlap across queries), 0 = use sequential strategy.
    /// - `out_stream`: Output stream pointer to receive results
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
    pub unsafe extern "C" fn rype_classify_arrow(
        index_ptr: *const RypeIndex,
        negative_set_ptr: *const RypeNegativeSet,
        input_stream: *mut FFI_ArrowArrayStream,
        threshold: c_double,
        use_merge_join: i32,
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

        // Convert use_merge_join to bool (C convention: 0 = false, non-zero = true)
        let merge_join = use_merge_join != 0;

        // Wrap pointers in type-specific Send-safe wrappers
        // SAFETY: Caller guarantees pointers remain valid until stream is consumed
        let index_send = SendRypeIndexPtr::new(index_ptr);
        let neg_set_send = if negative_set_ptr.is_null() {
            None
        } else {
            Some(SendRypeNegativeSetPtr::new(negative_set_ptr))
        };

        let classify_fn = move |batch: &RecordBatch| {
            let index = unsafe { index_send.get() };
            let neg_mins: Option<&HashSet<u64>> = neg_set_send
                .as_ref()
                .map(|p| unsafe { &p.get().minimizers });

            let result =
                classify_arrow_batch_sharded(&index.0, neg_mins, batch, threshold, merge_join);
            result.map_err(|e| arrow::error::ArrowError::ExternalError(Box::new(e)))
        };

        match create_streaming_output(input_stream, out_stream, classify_fn) {
            Ok(()) => {
                clear_last_error();
                0
            }
            Err(e) => {
                set_last_error(e);
                -1
            }
        }
    }

    /// Classify sequences from an Arrow stream and return only the best hit per query.
    ///
    /// Same as `rype_classify_arrow` but filters results to keep only the highest-scoring
    /// bucket for each query. If multiple buckets tie for the best score, one is chosen
    /// arbitrarily.
    ///
    /// ## Input Schema
    ///
    /// Same requirements as `rype_classify_arrow`:
    /// - `read_id`: Int64 - Query identifier
    /// - `sequence`: Binary or LargeBinary - DNA sequence
    /// - Optional `sequence2`: Binary or LargeBinary - Paired-end read
    ///
    /// ## Output Schema
    ///
    /// Same as `rype_classify_arrow` but with at most one row per query_id:
    /// - `query_id`: Int64 - Query ID from input
    /// - `bucket_id`: UInt32 - Matched bucket/reference ID (highest scoring)
    /// - `score`: Float64 - Classification score (0.0-1.0)
    ///
    /// # Safety
    /// - index_ptr must remain valid until the output stream is fully consumed
    /// - negative_set_ptr (if non-null) must remain valid until output is consumed
    ///
    /// # Returns
    /// 0 on success, -1 on error. Call rype_get_last_error() for details.
    #[no_mangle]
    pub unsafe extern "C" fn rype_classify_arrow_best_hit(
        index_ptr: *const RypeIndex,
        negative_set_ptr: *const RypeNegativeSet,
        input_stream: *mut FFI_ArrowArrayStream,
        threshold: c_double,
        use_merge_join: i32,
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

        let merge_join = use_merge_join != 0;

        // Wrap pointers in Send-safe wrappers
        let index_send = SendRypeIndexPtr::new(index_ptr);
        let neg_set_send = if negative_set_ptr.is_null() {
            None
        } else {
            Some(SendRypeNegativeSetPtr::new(negative_set_ptr))
        };

        let classify_fn = move |batch: &RecordBatch| {
            let index = unsafe { index_send.get() };
            let neg_mins: Option<&HashSet<u64>> = neg_set_send
                .as_ref()
                .map(|p| unsafe { &p.get().minimizers });

            let result = classify_arrow_batch_sharded_best_hit(
                &index.0, neg_mins, batch, threshold, merge_join,
            );
            result.map_err(|e| arrow::error::ArrowError::ExternalError(Box::new(e)))
        };

        match create_streaming_output(input_stream, out_stream, classify_fn) {
            Ok(()) => {
                clear_last_error();
                0
            }
            Err(e) => {
                set_last_error(e);
                -1
            }
        }
    }

    /// Returns the schema for classification result batches as an FFI_ArrowSchema.
    ///
    /// This allows callers to pre-allocate memory or validate expected output format.
    /// The caller takes ownership and must release the schema.
    ///
    /// # Safety
    ///
    /// - `out_schema` must be a valid, non-null pointer to writable memory for an `FFI_ArrowSchema`.
    /// - The caller is responsible for releasing the schema according to the Arrow C Data Interface.
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
#[cfg(feature = "arrow-ffi")]
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
            pair_len: 100, // Non-zero with null pointer!
        };

        let result = validate_query(&query);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("NULL but pair_len is non-zero"));
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

    #[test]
    fn test_rype_classify_best_hit_null_index() {
        let seq = CString::new("ACGTACGTACGT").unwrap();
        let query = RypeQuery {
            id: 1,
            seq: seq.as_ptr(),
            seq_len: 12,
            pair_seq: std::ptr::null(),
            pair_len: 0,
        };

        let result = rype_classify_best_hit(std::ptr::null(), &query, 1, 0.1);
        assert!(result.is_null());

        let err = unsafe {
            let err_ptr = rype_get_last_error();
            assert!(!err_ptr.is_null());
            CStr::from_ptr(err_ptr).to_string_lossy().into_owned()
        };
        assert!(err.contains("NULL") || err.contains("misaligned"));
    }

    #[test]
    fn test_rype_classify_best_hit_with_negative_null_index() {
        let seq = CString::new("ACGTACGTACGT").unwrap();
        let query = RypeQuery {
            id: 1,
            seq: seq.as_ptr(),
            seq_len: 12,
            pair_seq: std::ptr::null(),
            pair_len: 0,
        };

        let result = rype_classify_best_hit_with_negative(
            std::ptr::null(),
            std::ptr::null(),
            &query,
            1,
            0.1,
        );
        assert!(result.is_null());

        let err = unsafe {
            let err_ptr = rype_get_last_error();
            assert!(!err_ptr.is_null());
            CStr::from_ptr(err_ptr).to_string_lossy().into_owned()
        };
        assert!(err.contains("NULL") || err.contains("misaligned"));
    }

    #[test]
    fn test_rype_classify_best_hit_invalid_threshold() {
        let seq = CString::new("ACGTACGTACGT").unwrap();
        let query = RypeQuery {
            id: 1,
            seq: seq.as_ptr(),
            seq_len: 12,
            pair_seq: std::ptr::null(),
            pair_len: 0,
        };

        // Create a fake "index" pointer - we'll catch the error before dereferencing
        let fake_index: usize = 0x1000; // Non-null aligned pointer
        let result = rype_classify_best_hit(
            fake_index as *const RypeIndex,
            &query,
            1,
            1.5, // Invalid threshold > 1.0
        );
        assert!(result.is_null());

        let err = unsafe {
            let err_ptr = rype_get_last_error();
            assert!(!err_ptr.is_null());
            CStr::from_ptr(err_ptr).to_string_lossy().into_owned()
        };
        assert!(err.contains("threshold") || err.contains("Invalid"));
    }
}
