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

use crate::classify_with_sharded_negative;
use crate::constants::MAX_SEQUENCE_LENGTH;
use crate::memory::{
    calculate_batch_config, detect_available_memory, estimate_shard_reservation, parse_byte_suffix,
    InputFormat, MemoryConfig, ReadMemoryProfile,
};
use crate::{QueryRecord, ShardedInvertedIndex};
use libc::{c_char, c_double, c_int, size_t};
use std::cell::RefCell;
use std::collections::HashMap;
#[cfg(feature = "arrow-ffi")]
use std::collections::HashSet;
use std::ffi::{CStr, CString};
use std::path::Path;
use std::slice;
use std::sync::Arc;

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
/// Uses Arc internally to allow cheap sharing with RypeNegativeSet.
pub struct RypeIndex(Arc<ShardedInvertedIndex>);

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

    /// Bytes per shard entry: u64 minimizer (8) + u32 bucket_id (4).
    const BYTES_PER_SHARD_ENTRY: usize = 12;

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
            .map(|s| s.num_bucket_ids * Self::BYTES_PER_SHARD_ENTRY)
            .sum()
    }

    /// Estimate the memory needed to load the largest single shard.
    ///
    /// Returns the estimated size of the largest shard in bytes.
    /// Use this for memory planning when classifying with limited RAM.
    pub fn largest_shard_bytes(&self) -> usize {
        self.largest_shard_entries() as usize * Self::BYTES_PER_SHARD_ENTRY
    }

    /// Returns the entry count for the largest shard.
    ///
    /// This feeds directly into `estimate_shard_reservation()` for memory planning.
    fn largest_shard_entries(&self) -> u64 {
        self.0
            .manifest()
            .shards
            .iter()
            .map(|s| s.num_bucket_ids as u64)
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

/// Debug sanity check: pointer is non-null and has correct alignment for T.
///
/// This catches accidental null or obviously-garbled pointers (e.g. truncated
/// casts). It CANNOT detect use-after-free, wild pointers, or other real
/// memory errors — those are inherent C FFI limitations.
#[inline]
fn is_nonnull_aligned<T>(ptr: *const T) -> bool {
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

// --- Batch Configuration Struct ---

/// Full batch configuration returned by `rype_calculate_batch_config`.
///
/// On error, all fields are 0. Check `batch_size == 0` to detect errors,
/// then call `rype_get_last_error()` for details.
#[repr(C)]
pub struct RypeBatchConfig {
    /// Number of records per batch (>= 1000 on success, 0 on error).
    pub batch_size: size_t,
    /// Number of parallel batches. Reserved for forward-compatibility; always 1.
    /// Do not write application logic that depends on this being > 1.
    pub batch_count: size_t,
    /// Estimated memory per batch in bytes.
    pub per_batch_memory: size_t,
    /// Estimated peak memory usage in bytes.
    pub peak_memory: size_t,
}

// --- Log-Ratio C-Compatible Structs ---

/// A single log-ratio classification result for one query.
///
/// `fast_path`: 0 = computed exactly (None), 1 = numerator exceeded skip threshold (NumHigh).
#[repr(C)]
pub struct RypeLogRatioHit {
    pub query_id: i64,
    pub log_ratio: c_double,
    pub fast_path: i32,
}

/// Array of log-ratio results returned by `rype_classify_log_ratio`.
///
/// Free with `rype_log_ratio_results_free`. Do NOT call twice on the same pointer.
#[repr(C)]
pub struct RypeLogRatioResultArray {
    pub data: *mut RypeLogRatioHit,
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
            Box::into_raw(Box::new(RypeIndex(Arc::new(idx))))
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
    if !is_nonnull_aligned(index_ptr) {
        return 0;
    }
    let index = unsafe { &*index_ptr };
    index.k()
}

/// Returns the window size of the index, or 0 if index is NULL or misaligned.
#[no_mangle]
pub extern "C" fn rype_index_w(index_ptr: *const RypeIndex) -> size_t {
    if !is_nonnull_aligned(index_ptr) {
        return 0;
    }
    let index = unsafe { &*index_ptr };
    index.w()
}

/// Returns the salt of the index, or 0 if index is NULL or misaligned.
#[no_mangle]
pub extern "C" fn rype_index_salt(index_ptr: *const RypeIndex) -> u64 {
    if !is_nonnull_aligned(index_ptr) {
        return 0;
    }
    let index = unsafe { &*index_ptr };
    index.salt()
}

/// Returns the number of buckets in the index.
///
/// Returns 0 if index is NULL or misaligned.
#[no_mangle]
pub extern "C" fn rype_index_num_buckets(index_ptr: *const RypeIndex) -> u32 {
    if !is_nonnull_aligned(index_ptr) {
        return 0;
    }
    let index = unsafe { &*index_ptr };
    index.num_buckets() as u32
}

/// Returns whether the index is sharded (always true for Parquet indices).
///
/// # Deprecated
/// This function always returns 1, since all indices are now Parquet-based and
/// sharded. It is retained for backward compatibility with existing C callers.
#[deprecated(note = "Always returns 1. All indices are Parquet-based and sharded.")]
#[no_mangle]
pub extern "C" fn rype_index_is_sharded(index_ptr: *const RypeIndex) -> i32 {
    if !is_nonnull_aligned(index_ptr) {
        return 0;
    }
    1 // Always true for Parquet format
}

/// Returns the number of shards in the index.
#[no_mangle]
pub extern "C" fn rype_index_num_shards(index_ptr: *const RypeIndex) -> u32 {
    if !is_nonnull_aligned(index_ptr) {
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
    if !is_nonnull_aligned(index_ptr) {
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
    if !is_nonnull_aligned(index_ptr) {
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

// --- Batch Size Recommendation ---

/// Recommend an optimal batch size for Arrow streaming classification.
///
/// Convenience wrapper around `rype_calculate_batch_config` that returns only
/// the `batch_size` field. See that function for full documentation.
///
/// # Returns
/// The recommended number of rows per RecordBatch (always >= 1000 on success),
/// or 0 on error. Call `rype_get_last_error()` for details when 0 is returned.
#[no_mangle]
pub extern "C" fn rype_recommend_batch_size(
    index_ptr: *const RypeIndex,
    avg_read_length: size_t,
    is_paired: c_int,
    max_memory: size_t,
) -> size_t {
    rype_calculate_batch_config(index_ptr, avg_read_length, is_paired, max_memory).batch_size
}

/// Calculate the full batch configuration for Arrow streaming classification.
///
/// Returns the same information that the CLI uses internally to size batches,
/// including per-batch memory estimates and peak memory. This is a superset of
/// `rype_recommend_batch_size` which returns only `batch_size`.
///
/// # Parameters
///
/// * `index_ptr` - A loaded index (from `rype_index_load`)
/// * `avg_read_length` - Average nucleotide length of individual reads
/// * `is_paired` - Any non-zero value means paired-end, 0 means single-end
/// * `max_memory` - Maximum memory budget in bytes, or 0 to auto-detect
///
/// # Returns
/// A `RypeBatchConfig` struct. On error, all fields are 0.
/// Call `rype_get_last_error()` for details when `batch_size == 0`.
///
/// # Thread Safety
/// Thread-safe (read-only access to index).
#[no_mangle]
pub extern "C" fn rype_calculate_batch_config(
    index_ptr: *const RypeIndex,
    avg_read_length: size_t,
    is_paired: c_int,
    max_memory: size_t,
) -> RypeBatchConfig {
    let error_result = RypeBatchConfig {
        batch_size: 0,
        batch_count: 0,
        per_batch_memory: 0,
        peak_memory: 0,
    };

    if !is_nonnull_aligned(index_ptr) {
        set_last_error("index_ptr is NULL or misaligned".to_string());
        return error_result;
    }

    if avg_read_length == 0 {
        set_last_error("avg_read_length must be > 0".to_string());
        return error_result;
    }

    let index = unsafe { &*index_ptr };

    let k = index.k();
    let w = index.w();
    let num_buckets = index.num_buckets();

    if num_buckets == 0 {
        set_last_error("index has no buckets".to_string());
        return error_result;
    }

    let is_paired_bool = is_paired != 0;

    let memory_budget = if max_memory == 0 {
        detect_available_memory().bytes
    } else {
        max_memory
    };

    let num_threads = rayon::current_num_threads();
    let read_profile = ReadMemoryProfile::new(avg_read_length, is_paired_bool, k, w);
    let shard_reservation = estimate_shard_reservation(index.largest_shard_entries(), num_threads);

    // Build memory config:
    // - index_memory: 0 (already loaded, shards on-demand)
    // - InputFormat::Parquet (Arrow RecordBatches match the untrimmed Parquet columnar path)
    // - is_log_ratio: false (standard Arrow classify)
    let config = match MemoryConfig::new(
        memory_budget,
        num_threads,
        0, // index_memory: shards load on-demand, manifest is tiny
        shard_reservation,
        read_profile,
        num_buckets,
        InputFormat::Parquet {
            is_paired: is_paired_bool,
            trimmed_in_reader: false,
        },
        false, // is_log_ratio
    ) {
        Ok(c) => c,
        Err(e) => {
            set_last_error(format!("Failed to build memory config: {}", e));
            return error_result;
        }
    };

    let batch_config = calculate_batch_config(&config);
    clear_last_error();
    RypeBatchConfig {
        batch_size: batch_config.batch_size,
        batch_count: batch_config.batch_count,
        per_batch_memory: batch_config.per_batch_memory,
        peak_memory: batch_config.peak_memory,
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
    if !is_nonnull_aligned(index_ptr) {
        return std::ptr::null();
    }

    let index = unsafe { &*index_ptr };

    match index.bucket_name(bucket_id) {
        Some(name) => {
            BUCKET_NAME_CACHE.with(|cache| {
                let mut cache = cache.borrow_mut();
                let key = (index_ptr as usize, bucket_id);
                let entry = cache
                    .entry(key)
                    .or_insert_with(|| CString::new("").unwrap());
                // Update the cached CString (name may change if index is reloaded at same address)
                match CString::new(name) {
                    Ok(s) => {
                        *entry = s;
                        entry.as_ptr()
                    }
                    Err(_) => std::ptr::null(),
                }
            })
        }
        None => std::ptr::null(),
    }
}

// --- Bucket File Stats API ---

/// Per-bucket file statistics, exposed to C callers.
#[repr(C)]
pub struct RypeBucketFileStats {
    /// Bucket ID this stats entry belongs to.
    pub bucket_id: u32,
    /// Mean of per-file total sequence lengths.
    pub mean: c_double,
    /// Median of per-file total sequence lengths.
    pub median: c_double,
    /// Population standard deviation of per-file total sequence lengths.
    pub stdev: c_double,
    /// Minimum per-file total sequence length.
    pub min: c_double,
    /// Maximum per-file total sequence length.
    pub max: c_double,
}

/// Array of per-bucket file statistics returned by `rype_bucket_file_stats()`.
#[repr(C)]
pub struct RypeBucketFileStatsArray {
    /// Pointer to array of stats entries (owned by this struct).
    pub stats: *mut RypeBucketFileStats,
    /// Number of entries in the array.
    pub count: size_t,
}

/// Returns per-bucket file statistics for all buckets that have them.
///
/// Returns NULL if the index has no file statistics (e.g., old format indices
/// or merged indices). The caller takes ownership and must call
/// `rype_bucket_file_stats_free()` when done.
///
/// # Safety
/// - index_ptr must be a valid pointer obtained from rype_index_load()
#[no_mangle]
pub extern "C" fn rype_bucket_file_stats(
    index_ptr: *const RypeIndex,
) -> *mut RypeBucketFileStatsArray {
    if !is_nonnull_aligned(index_ptr) {
        set_last_error("index_ptr is NULL or misaligned".to_string());
        return std::ptr::null_mut();
    }

    let index = unsafe { &*index_ptr };
    let manifest = index.0.manifest();

    match &manifest.bucket_file_stats {
        Some(stats_map) if !stats_map.is_empty() => {
            let mut entries: Vec<RypeBucketFileStats> = stats_map
                .iter()
                .map(|(&bucket_id, stats)| RypeBucketFileStats {
                    bucket_id,
                    mean: stats.mean,
                    median: stats.median,
                    stdev: stats.stdev,
                    min: stats.min,
                    max: stats.max,
                })
                .collect();

            // Sort by bucket_id for deterministic ordering
            entries.sort_by_key(|e| e.bucket_id);

            let count = entries.len();
            let stats_ptr = entries.as_mut_ptr();
            std::mem::forget(entries);

            let result = Box::new(RypeBucketFileStatsArray {
                stats: stats_ptr,
                count,
            });
            Box::into_raw(result)
        }
        _ => std::ptr::null_mut(),
    }
}

/// Frees a `RypeBucketFileStatsArray` previously returned by `rype_bucket_file_stats()`.
///
/// Safe to call with NULL.
#[no_mangle]
pub extern "C" fn rype_bucket_file_stats_free(ptr: *mut RypeBucketFileStatsArray) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let array = Box::from_raw(ptr);
        if !array.stats.is_null() && array.count > 0 {
            let _ = Vec::from_raw_parts(array.stats, array.count, array.count);
        }
    }
}

// --- Negative Index API ---

/// Opaque handle to a negative index for memory-efficient filtering.
///
/// This holds an `Arc<ShardedInvertedIndex>` that is queried shard-by-shard during
/// classification, avoiding the need to load all minimizers into memory at once.
/// Using Arc allows cheap sharing with the source RypeIndex without cloning
/// the entire index metadata (manifest, path, rg_ranges_cache).
pub struct RypeNegativeSet {
    index: Arc<ShardedInvertedIndex>,
}

/// Creates a negative minimizer set from an index.
///
/// Returns a `RypeNegativeSet` that uses memory-efficient sharded filtering.
/// Memory usage during classification is O(single_shard), not O(entire_index).
///
/// The caller takes ownership and must call `rype_negative_set_free()` when done.
/// The original `negative_index_ptr` remains valid and can continue to be used.
///
/// Note: This clones an Arc, not the underlying data. The RypeNegativeSet shares
/// the same underlying ShardedInvertedIndex with the source RypeIndex via Arc.
/// Both handles must remain valid during classification, but either can be freed
/// independently once classification is complete.
#[no_mangle]
pub extern "C" fn rype_negative_set_create(
    negative_index_ptr: *const RypeIndex,
) -> *mut RypeNegativeSet {
    if !is_nonnull_aligned(negative_index_ptr) {
        set_last_error("negative_index_ptr is NULL or misaligned".to_string());
        return std::ptr::null_mut();
    }

    let index = unsafe { &*negative_index_ptr };
    // Clone the Arc - this is a cheap reference count increment, not a deep copy.
    // The underlying ShardedInvertedIndex (manifest, path, rg_ranges_cache) is shared.
    let neg_set = RypeNegativeSet {
        index: Arc::clone(&index.0),
    };
    clear_last_error();
    Box::into_raw(Box::new(neg_set))
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
///
/// This returns the total minimizer count from the index manifest.
/// Note that shards may contain duplicate entries across shards, so this
/// is an upper bound rather than an exact count.
#[no_mangle]
pub extern "C" fn rype_negative_set_size(ptr: *const RypeNegativeSet) -> size_t {
    if !is_nonnull_aligned(ptr) {
        return 0;
    }
    let neg_set = unsafe { &*ptr };
    neg_set.index.manifest().total_minimizers
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
    if !is_nonnull_aligned(index_ptr) {
        set_last_error("index is NULL or misaligned".to_string());
        return std::ptr::null_mut();
    }
    if queries_ptr.is_null() || num_queries == 0 {
        set_last_error("Invalid arguments: queries is NULL or num_queries is zero".to_string());
        return std::ptr::null_mut();
    }
    if !is_nonnull_aligned(queries_ptr) {
        set_last_error("queries pointer is misaligned".to_string());
        return std::ptr::null_mut();
    }
    if !negative_set_ptr.is_null() && !is_nonnull_aligned(negative_set_ptr) {
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

        // Get negative index if provided (for memory-efficient sharded filtering)
        let neg_index: Option<&ShardedInvertedIndex> = if negative_set_ptr.is_null() {
            None
        } else {
            let neg_set = unsafe { &*negative_set_ptr };
            Some(&neg_set.index)
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

        // Classify using sharded inverted index with memory-efficient negative filtering
        let hits =
            classify_with_sharded_negative(&index.0, neg_index, &rust_queries, threshold, None)
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

/// Common layout shared by `RypeResultArray` and `RypeLogRatioResultArray`.
/// Used only inside `free_result_array_inner` to avoid function-pointer indirection.
#[repr(C)]
struct ResultArrayRepr<T> {
    data: *mut T,
    len: size_t,
    capacity: size_t,
}

/// Free a result array (outer Box + inner Vec).
///
/// # Safety
/// - `ptr` must be a valid pointer previously obtained from Box::into_raw, or null.
/// - The struct at `ptr` must have the layout of `ResultArrayRepr<T>`.
unsafe fn free_result_array_inner<T>(ptr: *mut ResultArrayRepr<T>) {
    if !ptr.is_null() {
        let array = Box::from_raw(ptr);
        if !array.data.is_null() {
            let _ = Vec::from_raw_parts(array.data, array.len, array.capacity);
        }
        // Empty result (data=null, len=0, capacity=0): Box drop handles the outer struct.
    }
}

#[no_mangle]
pub extern "C" fn rype_results_free(ptr: *mut RypeResultArray) {
    // SAFETY: RypeResultArray has identical layout to ResultArrayRepr<RypeHit>.
    unsafe { free_result_array_inner(ptr as *mut ResultArrayRepr<RypeHit>) }
}

// --- Log-Ratio API Functions ---

/// Validate two indices are compatible for log-ratio classification.
///
/// Each index must have exactly 1 bucket, and both must share the same k, w, and salt.
/// Returns 0 on success, -1 on error (call `rype_get_last_error()` for details).
///
/// Thread Safety: Thread-safe (read-only access).
#[no_mangle]
pub extern "C" fn rype_validate_log_ratio_indices(
    numerator: *const RypeIndex,
    denominator: *const RypeIndex,
) -> c_int {
    if !is_nonnull_aligned(numerator) {
        set_last_error("numerator is NULL or misaligned".to_string());
        return -1;
    }
    if !is_nonnull_aligned(denominator) {
        set_last_error("denominator is NULL or misaligned".to_string());
        return -1;
    }

    let num = unsafe { &*numerator };
    let denom = unsafe { &*denominator };

    if let Err(e) = crate::validate_log_ratio_indices(&num.0, &denom.0) {
        set_last_error(format!("{}", e));
        return -1;
    }

    clear_last_error();
    0
}

/// Classify a batch using log-ratio (numerator vs denominator).
///
/// `numerator_skip_threshold` semantics:
///   - `<= 0.0` → disabled (all reads classified against both indices)
///   - `(0.0, 1.0]` → enabled; reads scoring >= threshold get +inf fast-path
///   - `> 1.0`, NaN, inf → error (returns NULL)
///
/// Thread Safety: Thread-safe. Multiple threads can classify concurrently
/// with the same RypeIndex pointers.
#[no_mangle]
pub extern "C" fn rype_classify_log_ratio(
    numerator: *const RypeIndex,
    denominator: *const RypeIndex,
    queries: *const RypeQuery,
    num_queries: size_t,
    numerator_skip_threshold: c_double,
) -> *mut RypeLogRatioResultArray {
    let result = std::panic::catch_unwind(|| {
        // Validate pointers
        if !is_nonnull_aligned(numerator) {
            set_last_error("numerator is NULL or misaligned".to_string());
            return std::ptr::null_mut();
        }
        if !is_nonnull_aligned(denominator) {
            set_last_error("denominator is NULL or misaligned".to_string());
            return std::ptr::null_mut();
        }
        if queries.is_null() || num_queries == 0 {
            set_last_error("Invalid arguments: queries is NULL or num_queries is zero".to_string());
            return std::ptr::null_mut();
        }
        if !is_nonnull_aligned(queries) {
            set_last_error("queries pointer is misaligned".to_string());
            return std::ptr::null_mut();
        }

        // Validate threshold
        let skip_threshold = if numerator_skip_threshold.is_nan()
            || numerator_skip_threshold.is_infinite()
            || numerator_skip_threshold > 1.0
        {
            set_last_error(format!(
                "Invalid numerator_skip_threshold: {}. Must be <= 1.0 and finite, or <= 0.0 to disable.",
                numerator_skip_threshold
            ));
            return std::ptr::null_mut();
        } else if numerator_skip_threshold <= 0.0 {
            None
        } else {
            Some(numerator_skip_threshold)
        };

        let num = unsafe { &*numerator };
        let denom = unsafe { &*denominator };

        // Build QueryRecords from C queries
        let query_slice = unsafe { std::slice::from_raw_parts(queries, num_queries) };
        let mut records: Vec<QueryRecord> = Vec::with_capacity(num_queries);
        for (idx, q) in query_slice.iter().enumerate() {
            if let Err(msg) = validate_query(q) {
                set_last_error(format!("Query {} validation failed: {}", idx, msg));
                return std::ptr::null_mut();
            }
            let seq = unsafe { std::slice::from_raw_parts(q.seq as *const u8, q.seq_len) };
            let pair_seq = if !q.pair_seq.is_null() && q.pair_len > 0 {
                Some(unsafe { std::slice::from_raw_parts(q.pair_seq as *const u8, q.pair_len) })
            } else {
                None
            };
            records.push((q.id, seq, pair_seq));
        }

        // Run log-ratio classification (validates indices internally)
        let lr_results =
            match crate::classify_log_ratio_batch(&num.0, &denom.0, &records, skip_threshold) {
                Ok(r) => r,
                Err(e) => {
                    set_last_error(format!("Log-ratio classification failed: {}", e));
                    return std::ptr::null_mut();
                }
            };

        // Convert LogRatioResult → RypeLogRatioHit
        let mut results: Vec<RypeLogRatioHit> = lr_results
            .into_iter()
            .map(|lr| RypeLogRatioHit {
                query_id: lr.query_id,
                log_ratio: lr.log_ratio,
                fast_path: match lr.fast_path {
                    crate::FastPath::NumHigh => 1,
                    crate::FastPath::None => 0,
                },
            })
            .collect();

        let len = results.len();
        let capacity = results.capacity();
        let data = if results.is_empty() {
            std::ptr::null_mut()
        } else {
            let ptr = results.as_mut_ptr();
            std::mem::forget(results);
            ptr
        };

        clear_last_error();
        Box::into_raw(Box::new(RypeLogRatioResultArray {
            data,
            len,
            capacity,
        }))
    });

    match result {
        Ok(ptr) => ptr,
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Unknown panic in rype_classify_log_ratio".to_string()
            };
            set_last_error(msg);
            std::ptr::null_mut()
        }
    }
}

/// Free log-ratio results. NULL is safe to pass.
/// Do NOT call twice on the same pointer (undefined behavior).
#[no_mangle]
pub extern "C" fn rype_log_ratio_results_free(ptr: *mut RypeLogRatioResultArray) {
    // SAFETY: RypeLogRatioResultArray has identical layout to ResultArrayRepr<RypeLogRatioHit>.
    unsafe { free_result_array_inner(ptr as *mut ResultArrayRepr<RypeLogRatioHit>) }
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

// =============================================================================
// Minimizer Extraction API
// =============================================================================
//
// These functions expose minimizer extraction to C callers without requiring
// an index. They are pure computation functions that extract minimizer hashes
// and/or positions from raw DNA sequences.
//
// Thread Safety:
// - All extraction functions are thread-safe (no shared mutable state)
// - Each call creates a fresh MinimizerWorkspace internally

// --- Extraction C-Compatible Structs ---

/// Array of u64 values. Used as a building block for extraction results.
///
/// Free the containing result struct (NOT this directly).
#[repr(C)]
pub struct RypeU64Array {
    pub data: *mut u64,
    pub len: size_t,
}

/// Result of `rype_extract_minimizer_set`: sorted, deduplicated hash sets per strand.
///
/// Free with `rype_minimizer_set_result_free()`. Do NOT call twice.
#[repr(C)]
pub struct RypeMinimizerSetResult {
    pub forward: RypeU64Array,
    pub reverse_complement: RypeU64Array,
}

/// Minimizer hashes and positions for a single strand (SoA layout).
///
/// `hashes[i]` corresponds to `positions[i]`. Both arrays have length `len`.
#[repr(C)]
pub struct RypeStrandResult {
    pub hashes: *mut u64,
    pub positions: *mut u64,
    pub len: size_t,
}

/// Result of `rype_extract_strand_minimizers`: hashes + positions per strand.
///
/// Free with `rype_strand_minimizers_result_free()`. Do NOT call twice.
#[repr(C)]
pub struct RypeStrandMinimizersResult {
    pub forward: RypeStrandResult,
    pub reverse_complement: RypeStrandResult,
}

// --- Extraction Helpers ---

/// Validate extraction parameters. Returns Ok(()) or Err(error message).
fn validate_extraction_params(
    seq: *const u8,
    seq_len: size_t,
    k: size_t,
    w: size_t,
) -> Result<(), String> {
    if seq.is_null() {
        return Err("seq is NULL".to_string());
    }
    if seq_len == 0 {
        return Err("seq_len is zero".to_string());
    }
    if !matches!(k, 16 | 32 | 64) {
        return Err(format!("Invalid k: {} (must be 16, 32, or 64)", k));
    }
    if w == 0 {
        return Err("w is zero".to_string());
    }
    Ok(())
}

/// Convert a Vec<u64> into a RypeU64Array using the boxed-slice pattern.
///
/// The Vec is converted to a boxed slice (capacity == len) so that C code
/// only needs (data, len) to reconstruct and free.
fn vec_to_rype_u64_array(v: Vec<u64>) -> RypeU64Array {
    if v.is_empty() {
        return RypeU64Array {
            data: std::ptr::null_mut(),
            len: 0,
        };
    }
    let boxed = v.into_boxed_slice();
    let len = boxed.len();
    let data = Box::into_raw(boxed) as *mut u64;
    RypeU64Array { data, len }
}

/// Free a RypeU64Array's backing memory. Does nothing if data is null.
///
/// # Safety
/// Must only be called once per array, and only on arrays created by vec_to_rype_u64_array.
unsafe fn free_rype_u64_array(arr: &RypeU64Array) {
    if !arr.data.is_null() && arr.len > 0 {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(arr.data, arr.len));
    }
}

/// Convert a StrandMinimizers into a RypeStrandResult, casting usize positions to u64.
fn strand_to_c_result(sm: crate::StrandMinimizers) -> RypeStrandResult {
    let len = sm.hashes.len();
    debug_assert_eq!(len, sm.positions.len());

    if len == 0 {
        return RypeStrandResult {
            hashes: std::ptr::null_mut(),
            positions: std::ptr::null_mut(),
            len: 0,
        };
    }

    // Convert positions from usize to u64
    let positions_u64: Vec<u64> = sm.positions.into_iter().map(|p| p as u64).collect();

    let hashes_boxed = sm.hashes.into_boxed_slice();
    let positions_boxed = positions_u64.into_boxed_slice();

    let hashes = Box::into_raw(hashes_boxed) as *mut u64;
    let positions = Box::into_raw(positions_boxed) as *mut u64;

    RypeStrandResult {
        hashes,
        positions,
        len,
    }
}

/// Free a RypeStrandResult's backing memory.
///
/// # Safety
/// Must only be called once per result, and only on results created by strand_to_c_result.
unsafe fn free_strand_result(sr: &RypeStrandResult) {
    if !sr.hashes.is_null() && sr.len > 0 {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(sr.hashes, sr.len));
    }
    if !sr.positions.is_null() && sr.len > 0 {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(sr.positions, sr.len));
    }
}

// --- Extraction API Functions ---

/// Extract sorted, deduplicated minimizer hash sets per strand.
///
/// Returns a heap-allocated `RypeMinimizerSetResult` containing two sorted,
/// deduplicated arrays of minimizer hashes (forward and reverse complement).
///
/// # Parameters
/// - `seq`: Pointer to DNA sequence bytes (A/C/G/T, case-insensitive)
/// - `seq_len`: Length of sequence in bytes
/// - `k`: K-mer size (must be 16, 32, or 64)
/// - `w`: Window size for minimizer selection (must be > 0)
/// - `salt`: XOR salt applied to k-mer hashes
///
/// # Returns
/// Non-NULL pointer on success, NULL on error.
/// Call `rype_get_last_error()` for error details.
///
/// # Memory
/// Caller must free with `rype_minimizer_set_result_free()`.
///
/// # Thread Safety
/// Thread-safe. No shared state.
#[no_mangle]
pub extern "C" fn rype_extract_minimizer_set(
    seq: *const u8,
    seq_len: size_t,
    k: size_t,
    w: size_t,
    salt: u64,
) -> *mut RypeMinimizerSetResult {
    if let Err(e) = validate_extraction_params(seq, seq_len, k, w) {
        set_last_error(e);
        return std::ptr::null_mut();
    }

    let result = std::panic::catch_unwind(|| {
        let seq_slice = unsafe { slice::from_raw_parts(seq, seq_len) };
        let mut ws = crate::MinimizerWorkspace::new();

        let (fwd, rc) = crate::extract_minimizer_set(seq_slice, k, w, salt, &mut ws);

        Box::new(RypeMinimizerSetResult {
            forward: vec_to_rype_u64_array(fwd),
            reverse_complement: vec_to_rype_u64_array(rc),
        })
    });

    match result {
        Ok(boxed) => {
            clear_last_error();
            Box::into_raw(boxed)
        }
        Err(panic_err) => {
            let msg = if let Some(s) = panic_err.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = panic_err.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic in rype_extract_minimizer_set".to_string()
            };
            set_last_error(msg);
            std::ptr::null_mut()
        }
    }
}

/// Free a minimizer set result. NULL is safe to pass.
///
/// Do NOT call twice on the same pointer.
#[no_mangle]
pub extern "C" fn rype_minimizer_set_result_free(ptr: *mut RypeMinimizerSetResult) {
    if !ptr.is_null() {
        unsafe {
            let result = Box::from_raw(ptr);
            free_rype_u64_array(&result.forward);
            free_rype_u64_array(&result.reverse_complement);
        }
    }
}

/// Extract ordered minimizers with positions per strand (SoA layout).
///
/// Returns a heap-allocated `RypeStrandMinimizersResult` containing hashes
/// and 0-based positions for both forward and reverse complement strands.
/// Positions are non-decreasing within each strand.
///
/// # Parameters
/// - `seq`: Pointer to DNA sequence bytes
/// - `seq_len`: Length of sequence in bytes
/// - `k`: K-mer size (must be 16, 32, or 64)
/// - `w`: Window size for minimizer selection (must be > 0)
/// - `salt`: XOR salt applied to k-mer hashes
///
/// # Returns
/// Non-NULL pointer on success, NULL on error.
///
/// # Memory
/// Caller must free with `rype_strand_minimizers_result_free()`.
///
/// # Thread Safety
/// Thread-safe. No shared state.
#[no_mangle]
pub extern "C" fn rype_extract_strand_minimizers(
    seq: *const u8,
    seq_len: size_t,
    k: size_t,
    w: size_t,
    salt: u64,
) -> *mut RypeStrandMinimizersResult {
    if let Err(e) = validate_extraction_params(seq, seq_len, k, w) {
        set_last_error(e);
        return std::ptr::null_mut();
    }

    let result = std::panic::catch_unwind(|| {
        let seq_slice = unsafe { slice::from_raw_parts(seq, seq_len) };
        let mut ws = crate::MinimizerWorkspace::new();

        let (fwd, rc) = crate::extract_strand_minimizers(seq_slice, k, w, salt, &mut ws);

        Box::new(RypeStrandMinimizersResult {
            forward: strand_to_c_result(fwd),
            reverse_complement: strand_to_c_result(rc),
        })
    });

    match result {
        Ok(boxed) => {
            clear_last_error();
            Box::into_raw(boxed)
        }
        Err(panic_err) => {
            let msg = if let Some(s) = panic_err.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = panic_err.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic in rype_extract_strand_minimizers".to_string()
            };
            set_last_error(msg);
            std::ptr::null_mut()
        }
    }
}

/// Free a strand minimizers result. NULL is safe to pass.
///
/// Do NOT call twice on the same pointer.
#[no_mangle]
pub extern "C" fn rype_strand_minimizers_result_free(ptr: *mut RypeStrandMinimizersResult) {
    if !ptr.is_null() {
        unsafe {
            let result = Box::from_raw(ptr);
            free_strand_result(&result.forward);
            free_strand_result(&result.reverse_complement);
        }
    }
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

    #[test]
    fn test_rype_log_ratio_hit_layout() {
        use std::mem::{align_of, size_of};
        // i64 (8) + f64 (8) + i32 (4) + padding (4) = 24 bytes
        assert_eq!(size_of::<RypeLogRatioHit>(), 24);
        assert_eq!(align_of::<RypeLogRatioHit>(), 8);

        assert_eq!(std::mem::offset_of!(RypeLogRatioHit, query_id), 0);
        assert_eq!(std::mem::offset_of!(RypeLogRatioHit, log_ratio), 8);
        assert_eq!(std::mem::offset_of!(RypeLogRatioHit, fast_path), 16);
    }

    #[test]
    fn test_rype_log_ratio_result_array_layout() {
        use std::mem::{align_of, size_of};
        assert_eq!(size_of::<RypeLogRatioResultArray>(), 24);
        assert_eq!(align_of::<RypeLogRatioResultArray>(), 8);
    }

    // --- Extraction struct layout tests ---

    #[test]
    fn test_rype_u64_array_layout() {
        use std::mem::{align_of, size_of};
        assert_eq!(size_of::<RypeU64Array>(), 16);
        assert_eq!(align_of::<RypeU64Array>(), 8);
        assert_eq!(std::mem::offset_of!(RypeU64Array, data), 0);
        assert_eq!(std::mem::offset_of!(RypeU64Array, len), 8);
    }

    #[test]
    fn test_rype_minimizer_set_result_layout() {
        use std::mem::{align_of, size_of};
        assert_eq!(size_of::<RypeMinimizerSetResult>(), 32);
        assert_eq!(align_of::<RypeMinimizerSetResult>(), 8);
        assert_eq!(std::mem::offset_of!(RypeMinimizerSetResult, forward), 0);
        assert_eq!(
            std::mem::offset_of!(RypeMinimizerSetResult, reverse_complement),
            16
        );
    }

    #[test]
    fn test_rype_strand_result_layout() {
        use std::mem::{align_of, size_of};
        assert_eq!(size_of::<RypeStrandResult>(), 24);
        assert_eq!(align_of::<RypeStrandResult>(), 8);
        assert_eq!(std::mem::offset_of!(RypeStrandResult, hashes), 0);
        assert_eq!(std::mem::offset_of!(RypeStrandResult, positions), 8);
        assert_eq!(std::mem::offset_of!(RypeStrandResult, len), 16);
    }

    #[test]
    fn test_rype_strand_minimizers_result_layout() {
        use std::mem::{align_of, size_of};
        assert_eq!(size_of::<RypeStrandMinimizersResult>(), 48);
        assert_eq!(align_of::<RypeStrandMinimizersResult>(), 8);
        assert_eq!(std::mem::offset_of!(RypeStrandMinimizersResult, forward), 0);
        assert_eq!(
            std::mem::offset_of!(RypeStrandMinimizersResult, reverse_complement),
            24
        );
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
    // Helper: Sharded Negative Filtering for Arrow Batches
    // -------------------------------------------------------------------------

    /// Collect negative minimizers from a sharded index for the given batch.
    ///
    /// Returns a HashSet of minimizers that should be filtered out during
    /// classification. This processes the negative index shard-by-shard to
    /// avoid loading all negative minimizers into memory at once.
    fn collect_negative_mins_for_batch(
        negative_index: &ShardedInvertedIndex,
        batch: &RecordBatch,
        positive_index: &ShardedInvertedIndex,
    ) -> Result<HashSet<u64>, arrow::error::ArrowError> {
        use crate::arrow::batch_to_records;
        use crate::classify::collect_negative_minimizers_sharded;

        // Convert batch to records
        let records = batch_to_records(batch)
            .map_err(|e| arrow::error::ArrowError::ExternalError(Box::new(e)))?;

        if records.is_empty() {
            return Ok(HashSet::new());
        }

        let manifest = positive_index.manifest();

        // Extract minimizers using the shared batch extraction function
        let extracted =
            crate::extract_batch_minimizers(manifest.k, manifest.w, manifest.salt, None, &records);

        // Build sorted unique minimizers for querying negative index
        let mut all_minimizers: Vec<u64> = extracted
            .iter()
            .flat_map(|(fwd, rc)| fwd.iter().chain(rc.iter()).copied())
            .collect();
        all_minimizers.sort_unstable();
        all_minimizers.dedup();

        // Collect hitting minimizers from negative index (memory-efficient)
        collect_negative_minimizers_sharded(negative_index, &all_minimizers, None).map_err(|e| {
            arrow::error::ArrowError::ComputeError(format!(
                "Failed to collect negative minimizers: {}",
                e
            ))
        })
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
    /// SAFETY: RypeIndex wraps Arc<ShardedInvertedIndex> which is immutable during
    /// classification (read-only access). Arc itself is Send+Sync.
    /// Caller must guarantee the pointer remains valid until the stream is consumed.
    struct SendRypeIndexPtr(*const RypeIndex);

    // SAFETY: RypeIndex contains Arc<ShardedInvertedIndex>. Arc is Send+Sync when T is,
    // and ShardedInvertedIndex is safe for concurrent reads (no interior mutability).
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
    /// SAFETY: RypeNegativeSet contains Arc<ShardedInvertedIndex> which is read-only
    /// during classification. Arc itself is Send+Sync.
    /// Caller must guarantee the pointer remains valid until the stream is consumed.
    struct SendRypeNegativeSetPtr(*const RypeNegativeSet);

    // SAFETY: RypeNegativeSet.index is Arc<ShardedInvertedIndex>. Arc is Send+Sync when T is,
    // and ShardedInvertedIndex is safe for concurrent reads (no interior mutability).
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

    // SAFETY: StreamingClassifier<F> is Send when F: Send because:
    // 1. ArrowArrayStreamReader is Send in arrow v57+ (wraps FFI_ArrowArrayStream
    //    which transfers ownership across threads via the C Data Interface).
    // 2. SchemaRef (Arc<Schema>) is Send+Sync.
    // 3. The closure F captures only Send-safe wrapped pointers (SendRypeIndexPtr,
    //    SendRypeNegativeSetPtr) which wrap Arc<ShardedInvertedIndex> — itself
    //    Send+Sync due to immutable read-only access during classification.
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
        output_schema: SchemaRef,
        classify_fn: F,
    ) -> Result<(), String>
    where
        F: Fn(&RecordBatch) -> Result<RecordBatch, arrow::error::ArrowError> + Send + 'static,
    {
        let input_reader = ArrowArrayStreamReader::from_raw(input_stream)
            .map_err(|e| format!("Failed to create stream reader: {}", e))?;

        let streaming = StreamingClassifier::new(input_reader, output_schema, classify_fn);

        let export_stream = FFI_ArrowArrayStream::new(Box::new(streaming));
        std::ptr::write(out_stream, export_stream);

        Ok(())
    }

    // -------------------------------------------------------------------------
    // Internal helper: Shared Arrow classification logic
    // -------------------------------------------------------------------------

    /// Internal helper for Arrow FFI classification, shared by both
    /// `rype_classify_arrow` and `rype_classify_arrow_best_hit`.
    ///
    /// Validates parameters, sets up Send-safe pointer wrappers, builds the
    /// classify closure (with optional negative filtering and best-hit filtering),
    /// and creates the streaming output.
    ///
    /// # Safety
    /// - index_ptr must remain valid until the output stream is fully consumed
    /// - negative_set_ptr (if non-null) must remain valid until output is consumed
    unsafe fn classify_arrow_internal(
        index_ptr: *const RypeIndex,
        negative_set_ptr: *const RypeNegativeSet,
        input_stream: *mut FFI_ArrowArrayStream,
        threshold: c_double,
        out_stream: *mut FFI_ArrowArrayStream,
        best_hit: bool,
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

        // Wrap pointers in type-specific Send-safe wrappers
        // SAFETY: Caller guarantees pointers remain valid until stream is consumed
        let index_send = SendRypeIndexPtr::new(index_ptr);
        let neg_set_send = if negative_set_ptr.is_null() {
            None
        } else {
            Some(SendRypeNegativeSetPtr::new(negative_set_ptr))
        };

        // Validate negative index compatibility upfront (not per-batch)
        if let Some(ref neg_send) = neg_set_send {
            let pos_manifest = unsafe { &*index_ptr }.0.manifest();
            let neg_manifest = unsafe { &neg_send.get().index }.manifest();
            if pos_manifest.k != neg_manifest.k
                || pos_manifest.w != neg_manifest.w
                || pos_manifest.salt != neg_manifest.salt
            {
                set_last_error(format!(
                    "Negative index parameters (k={}, w={}, salt=0x{:x}) do not match \
                     positive index (k={}, w={}, salt=0x{:x})",
                    neg_manifest.k,
                    neg_manifest.w,
                    neg_manifest.salt,
                    pos_manifest.k,
                    pos_manifest.w,
                    pos_manifest.salt,
                ));
                return -1;
            }
        }

        let classify_fn = move |batch: &RecordBatch| {
            let index = unsafe { index_send.get() };

            // Handle sharded negative filtering: collect hitting minimizers per batch
            let neg_mins_owned: Option<HashSet<u64>> = if let Some(ref neg_send) = neg_set_send {
                let neg_index = unsafe { &neg_send.get().index };
                Some(collect_negative_mins_for_batch(neg_index, batch, &index.0)?)
            } else {
                None
            };

            let result = if best_hit {
                classify_arrow_batch_sharded_best_hit(
                    &index.0,
                    neg_mins_owned.as_ref(),
                    batch,
                    threshold,
                )
            } else {
                classify_arrow_batch_sharded(&index.0, neg_mins_owned.as_ref(), batch, threshold)
            };
            result.map_err(|e| arrow::error::ArrowError::ExternalError(Box::new(e)))
        };

        match create_streaming_output(
            input_stream,
            out_stream,
            crate::arrow::result_schema(),
            classify_fn,
        ) {
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
        out_stream: *mut FFI_ArrowArrayStream,
    ) -> i32 {
        classify_arrow_internal(
            index_ptr,
            negative_set_ptr,
            input_stream,
            threshold,
            out_stream,
            false,
        )
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
        out_stream: *mut FFI_ArrowArrayStream,
    ) -> i32 {
        classify_arrow_internal(
            index_ptr,
            negative_set_ptr,
            input_stream,
            threshold,
            out_stream,
            true,
        )
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

    /// Get the output schema for Arrow log-ratio results.
    ///
    /// Schema: query_id (Int64), log_ratio (Float64), fast_path (Int32)
    ///
    /// Caller must release the schema.
    ///
    /// # Safety
    ///
    /// `out_schema` must be a valid, non-null pointer to an uninitialized `FFI_ArrowSchema`.
    ///
    /// # Returns
    ///
    /// 0 on success, -1 on error.
    #[no_mangle]
    pub unsafe extern "C" fn rype_arrow_log_ratio_result_schema(
        out_schema: *mut FFI_ArrowSchema,
    ) -> i32 {
        if out_schema.is_null() {
            set_last_error("out_schema is NULL".to_string());
            return -1;
        }

        let schema = crate::arrow::log_ratio_result_schema();

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

    // -------------------------------------------------------------------------
    // Arrow Minimizer Extraction
    // -------------------------------------------------------------------------

    /// Get the output schema for Arrow minimizer set extraction.
    ///
    /// Schema: id (Int64), fwd_set (List\<UInt64\>), rc_set (List\<UInt64\>)
    ///
    /// # Safety
    /// `out_schema` must be a valid, non-null pointer to an uninitialized `FFI_ArrowSchema`.
    ///
    /// # Returns
    /// 0 on success, -1 on error.
    #[no_mangle]
    pub unsafe extern "C" fn rype_arrow_minimizer_set_schema(
        out_schema: *mut FFI_ArrowSchema,
    ) -> i32 {
        if out_schema.is_null() {
            set_last_error("out_schema is NULL".to_string());
            return -1;
        }

        let schema = crate::arrow::minimizer_set_schema();

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

    /// Get the output schema for Arrow strand minimizers extraction.
    ///
    /// Schema: id (Int64), fwd_hashes (List\<UInt64\>), fwd_positions (List\<UInt64\>),
    /// rc_hashes (List\<UInt64\>), rc_positions (List\<UInt64\>)
    ///
    /// # Safety
    /// `out_schema` must be a valid, non-null pointer to an uninitialized `FFI_ArrowSchema`.
    ///
    /// # Returns
    /// 0 on success, -1 on error.
    #[no_mangle]
    pub unsafe extern "C" fn rype_arrow_strand_minimizers_schema(
        out_schema: *mut FFI_ArrowSchema,
    ) -> i32 {
        if out_schema.is_null() {
            set_last_error("out_schema is NULL".to_string());
            return -1;
        }

        let schema = crate::arrow::strand_minimizers_schema();

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

    /// Extract minimizer sets from an Arrow stream.
    ///
    /// TRUE STREAMING: Processes one batch at a time.
    ///
    /// # Arguments
    /// - `input_stream`: Arrow input stream with `id` (Int64) and `sequence` (Binary) columns
    /// - `k`: K-mer size (must be 16, 32, or 64)
    /// - `w`: Window size (must be > 0)
    /// - `salt`: XOR salt for k-mer hashing
    /// - `out_stream`: Output stream pointer to receive results
    ///
    /// # Output Schema
    /// `id` (Int64), `fwd_set` (List\<UInt64\>), `rc_set` (List\<UInt64\>)
    ///
    /// # Safety
    /// - input_stream is consumed (ownership transferred)
    /// - Caller owns out_stream and must call out_stream->release() when done
    ///
    /// # Returns
    /// 0 on success, -1 on error.
    #[no_mangle]
    pub unsafe extern "C" fn rype_extract_minimizer_set_arrow(
        input_stream: *mut FFI_ArrowArrayStream,
        k: size_t,
        w: size_t,
        salt: u64,
        out_stream: *mut FFI_ArrowArrayStream,
    ) -> i32 {
        if input_stream.is_null() {
            set_last_error("input_stream is NULL".to_string());
            return -1;
        }
        if out_stream.is_null() {
            set_last_error("out_stream is NULL".to_string());
            return -1;
        }
        if !matches!(k, 16 | 32 | 64) {
            set_last_error(format!("Invalid k: {} (must be 16, 32, or 64)", k));
            return -1;
        }
        if w == 0 {
            set_last_error("w is zero".to_string());
            return -1;
        }

        let extract_fn = move |batch: &RecordBatch| {
            crate::arrow::extract_minimizer_set_batch(batch, k, w, salt)
                .map_err(|e| arrow::error::ArrowError::ExternalError(Box::new(e)))
        };

        match create_streaming_output(
            input_stream,
            out_stream,
            crate::arrow::minimizer_set_schema(),
            extract_fn,
        ) {
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

    /// Extract strand minimizers (hashes + positions) from an Arrow stream.
    ///
    /// TRUE STREAMING: Processes one batch at a time.
    ///
    /// # Arguments
    /// - `input_stream`: Arrow input stream with `id` (Int64) and `sequence` (Binary) columns
    /// - `k`: K-mer size (must be 16, 32, or 64)
    /// - `w`: Window size (must be > 0)
    /// - `salt`: XOR salt for k-mer hashing
    /// - `out_stream`: Output stream pointer to receive results
    ///
    /// # Output Schema
    /// `id` (Int64), `fwd_hashes` (List\<UInt64\>), `fwd_positions` (List\<UInt64\>),
    /// `rc_hashes` (List\<UInt64\>), `rc_positions` (List\<UInt64\>)
    ///
    /// # Safety
    /// - input_stream is consumed (ownership transferred)
    /// - Caller owns out_stream and must call out_stream->release() when done
    ///
    /// # Returns
    /// 0 on success, -1 on error.
    #[no_mangle]
    pub unsafe extern "C" fn rype_extract_strand_minimizers_arrow(
        input_stream: *mut FFI_ArrowArrayStream,
        k: size_t,
        w: size_t,
        salt: u64,
        out_stream: *mut FFI_ArrowArrayStream,
    ) -> i32 {
        if input_stream.is_null() {
            set_last_error("input_stream is NULL".to_string());
            return -1;
        }
        if out_stream.is_null() {
            set_last_error("out_stream is NULL".to_string());
            return -1;
        }
        if !matches!(k, 16 | 32 | 64) {
            set_last_error(format!("Invalid k: {} (must be 16, 32, or 64)", k));
            return -1;
        }
        if w == 0 {
            set_last_error("w is zero".to_string());
            return -1;
        }

        let extract_fn = move |batch: &RecordBatch| {
            crate::arrow::extract_strand_minimizers_batch(batch, k, w, salt)
                .map_err(|e| arrow::error::ArrowError::ExternalError(Box::new(e)))
        };

        match create_streaming_output(
            input_stream,
            out_stream,
            crate::arrow::strand_minimizers_schema(),
            extract_fn,
        ) {
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

    /// Classify sequences from an Arrow stream using log-ratio (numerator vs denominator).
    ///
    /// TRUE STREAMING: Processes one batch at a time. Memory usage is O(batch_size).
    ///
    /// `numerator_skip_threshold` semantics:
    ///   - `<= 0.0` → disabled (all reads classified against both indices)
    ///   - `(0.0, 1.0]` → enabled; reads scoring >= threshold get +inf fast-path
    ///   - `> 1.0`, NaN, inf → error (returns -1)
    ///
    /// # Safety
    ///
    /// - numerator and denominator must remain valid until the output stream is fully consumed
    ///
    /// # Returns
    ///
    /// 0 on success, -1 on error. Call rype_get_last_error() for details.
    #[no_mangle]
    pub unsafe extern "C" fn rype_classify_arrow_log_ratio(
        numerator: *const RypeIndex,
        denominator: *const RypeIndex,
        input_stream: *mut FFI_ArrowArrayStream,
        numerator_skip_threshold: c_double,
        out_stream: *mut FFI_ArrowArrayStream,
    ) -> i32 {
        // Validate pointers
        if !is_nonnull_aligned(numerator) {
            set_last_error("numerator is NULL or misaligned".to_string());
            return -1;
        }
        if !is_nonnull_aligned(denominator) {
            set_last_error("denominator is NULL or misaligned".to_string());
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

        // Validate threshold
        let skip_threshold = if numerator_skip_threshold.is_nan()
            || numerator_skip_threshold.is_infinite()
            || numerator_skip_threshold > 1.0
        {
            set_last_error(format!(
                "Invalid numerator_skip_threshold: {}. Must be <= 1.0 and finite, or <= 0.0 to disable.",
                numerator_skip_threshold
            ));
            return -1;
        } else if numerator_skip_threshold <= 0.0 {
            None
        } else {
            Some(numerator_skip_threshold)
        };

        // Validate indices eagerly (fail fast before creating the stream)
        let num = &*numerator;
        let denom = &*denominator;
        if let Err(e) = crate::validate_log_ratio_indices(&num.0, &denom.0) {
            set_last_error(format!("{}", e));
            return -1;
        }

        // Wrap pointers in Send-safe wrappers
        let num_send = SendRypeIndexPtr::new(numerator);
        let denom_send = SendRypeIndexPtr::new(denominator);

        let classify_fn = move |batch: &RecordBatch| {
            use crate::arrow::batch_to_records;

            let num_idx = unsafe { num_send.get() };
            let denom_idx = unsafe { denom_send.get() };

            let records = batch_to_records(batch)
                .map_err(|e| arrow::error::ArrowError::ExternalError(Box::new(e)))?;

            // classify_log_ratio_batch validates indices internally on each call,
            // but we already validated eagerly above for fast failure
            let lr_results =
                crate::classify_log_ratio_batch(&num_idx.0, &denom_idx.0, &records, skip_threshold)
                    .map_err(|e| arrow::error::ArrowError::ExternalError(e.into()))?;

            // Convert Vec<LogRatioResult> → Arrow RecordBatch
            use arrow::array::{Float64Array, Int32Array, Int64Array};

            let schema = crate::arrow::log_ratio_result_schema();
            if lr_results.is_empty() {
                return Ok(RecordBatch::new_empty(schema));
            }

            let mut query_ids = Vec::with_capacity(lr_results.len());
            let mut log_ratios = Vec::with_capacity(lr_results.len());
            let mut fast_paths = Vec::with_capacity(lr_results.len());

            for lr in &lr_results {
                query_ids.push(lr.query_id);
                log_ratios.push(lr.log_ratio);
                fast_paths.push(match lr.fast_path {
                    crate::FastPath::NumHigh => 1i32,
                    crate::FastPath::None => 0i32,
                });
            }

            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Int64Array::from(query_ids)),
                    Arc::new(Float64Array::from(log_ratios)),
                    Arc::new(Int32Array::from(fast_paths)),
                ],
            )
            .map_err(|e| arrow::error::ArrowError::ExternalError(Box::new(e)))
        };

        match create_streaming_output(
            input_stream,
            out_stream,
            crate::arrow::log_ratio_result_schema(),
            classify_fn,
        ) {
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

    // --- Log-ratio validation tests (Cycle 2) ---

    #[test]
    fn test_rype_validate_log_ratio_null_numerator() {
        let result = rype_validate_log_ratio_indices(std::ptr::null(), 0x1000 as *const RypeIndex);
        assert_eq!(result, -1);

        let err = unsafe {
            let err_ptr = rype_get_last_error();
            assert!(!err_ptr.is_null());
            CStr::from_ptr(err_ptr).to_string_lossy().into_owned()
        };
        assert!(err.contains("NULL") || err.contains("misaligned"));
    }

    #[test]
    fn test_rype_validate_log_ratio_null_denominator() {
        let result = rype_validate_log_ratio_indices(0x1000 as *const RypeIndex, std::ptr::null());
        assert_eq!(result, -1);

        let err = unsafe {
            let err_ptr = rype_get_last_error();
            assert!(!err_ptr.is_null());
            CStr::from_ptr(err_ptr).to_string_lossy().into_owned()
        };
        assert!(err.contains("NULL") || err.contains("misaligned"));
    }

    // --- Log-ratio free tests (Cycle 3) ---

    #[test]
    fn test_rype_log_ratio_results_free_null() {
        // Should not crash
        rype_log_ratio_results_free(std::ptr::null_mut());
    }

    // --- Log-ratio classify error path tests (Cycle 4) ---

    #[test]
    fn test_rype_classify_log_ratio_null_numerator() {
        let seq = CString::new("ACGTACGTACGT").unwrap();
        let query = RypeQuery {
            id: 0,
            seq: seq.as_ptr(),
            seq_len: 12,
            pair_seq: std::ptr::null(),
            pair_len: 0,
        };

        let result = rype_classify_log_ratio(
            std::ptr::null(),
            0x1000 as *const RypeIndex,
            &query,
            1,
            -1.0,
        );
        assert!(result.is_null());

        let err = unsafe {
            CStr::from_ptr(rype_get_last_error())
                .to_string_lossy()
                .into_owned()
        };
        assert!(err.contains("NULL") || err.contains("misaligned"));
    }

    #[test]
    fn test_rype_classify_log_ratio_null_queries() {
        let result = rype_classify_log_ratio(
            0x1000 as *const RypeIndex,
            0x2000 as *const RypeIndex,
            std::ptr::null(),
            1,
            -1.0,
        );
        assert!(result.is_null());

        let err = unsafe {
            CStr::from_ptr(rype_get_last_error())
                .to_string_lossy()
                .into_owned()
        };
        assert!(err.contains("NULL") || err.contains("zero"));
    }

    #[test]
    fn test_rype_classify_log_ratio_invalid_threshold() {
        let seq = CString::new("ACGTACGTACGT").unwrap();
        let query = RypeQuery {
            id: 0,
            seq: seq.as_ptr(),
            seq_len: 12,
            pair_seq: std::ptr::null(),
            pair_len: 0,
        };

        // threshold > 1.0 — threshold check happens after pointer validation
        // but before any dereference, so fake aligned pointers are safe here
        let result = rype_classify_log_ratio(
            0x1000 as *const RypeIndex,
            0x2000 as *const RypeIndex,
            &query,
            1,
            1.5,
        );
        assert!(result.is_null());

        let err = unsafe {
            CStr::from_ptr(rype_get_last_error())
                .to_string_lossy()
                .into_owned()
        };
        assert!(err.contains("threshold") || err.contains("Invalid"));

        // threshold NaN — also fails before touching index pointers
        let result = rype_classify_log_ratio(
            0x1000 as *const RypeIndex,
            0x2000 as *const RypeIndex,
            &query,
            1,
            f64::NAN,
        );
        assert!(result.is_null());
    }

    #[test]
    fn test_rype_classify_log_ratio_disabled_threshold() {
        // threshold -1.0 should NOT error at the threshold validation stage
        // Use NULL denominator so it fails at the NULL pointer check, not dereference
        let seq = CString::new("ACGTACGTACGT").unwrap();
        let query = RypeQuery {
            id: 0,
            seq: seq.as_ptr(),
            seq_len: 12,
            pair_seq: std::ptr::null(),
            pair_len: 0,
        };

        let result = rype_classify_log_ratio(
            0x1000 as *const RypeIndex,
            std::ptr::null(), // NULL denom → fails at NULL check, not threshold
            &query,
            1,
            -1.0,
        );
        assert!(result.is_null());

        let err = unsafe {
            CStr::from_ptr(rype_get_last_error())
                .to_string_lossy()
                .into_owned()
        };
        // Error should be about NULL denominator, NOT about threshold
        assert!(
            !err.contains("threshold"),
            "Disabled threshold (-1.0) should not cause a threshold error, got: {}",
            err
        );
        assert!(err.contains("NULL") || err.contains("misaligned"));
    }

    #[test]
    fn test_rype_classify_log_ratio_zero_threshold() {
        // threshold 0.0 should disable fast-path (not error)
        // Use NULL numerator so it fails at the NULL pointer check
        let seq = CString::new("ACGTACGTACGT").unwrap();
        let query = RypeQuery {
            id: 0,
            seq: seq.as_ptr(),
            seq_len: 12,
            pair_seq: std::ptr::null(),
            pair_len: 0,
        };

        let result = rype_classify_log_ratio(
            std::ptr::null(), // NULL num → fails at NULL check
            0x2000 as *const RypeIndex,
            &query,
            1,
            0.0,
        );
        assert!(result.is_null());

        let err = unsafe {
            CStr::from_ptr(rype_get_last_error())
                .to_string_lossy()
                .into_owned()
        };
        assert!(
            !err.contains("threshold"),
            "Zero threshold should not cause a threshold error, got: {}",
            err
        );
    }

    // =========================================================================
    // Minimizer Extraction Tests
    // =========================================================================

    // --- extract_minimizer_set tests ---

    #[test]
    fn test_rype_extract_minimizer_set_null_seq() {
        let result = rype_extract_minimizer_set(std::ptr::null(), 100, 16, 5, 0);
        assert!(result.is_null());
    }

    #[test]
    fn test_rype_extract_minimizer_set_zero_len() {
        let seq = b"ACGT";
        let result = rype_extract_minimizer_set(seq.as_ptr(), 0, 16, 5, 0);
        assert!(result.is_null());
    }

    #[test]
    fn test_rype_extract_minimizer_set_invalid_k() {
        let seq = b"ACGTACGTACGTACGTACGT";
        // k=0
        let result = rype_extract_minimizer_set(seq.as_ptr(), seq.len(), 0, 5, 0);
        assert!(result.is_null());
        // k=100
        let result = rype_extract_minimizer_set(seq.as_ptr(), seq.len(), 100, 5, 0);
        assert!(result.is_null());
    }

    #[test]
    fn test_rype_extract_minimizer_set_short_seq() {
        // 10 bytes, k=16: too short to produce minimizers
        let seq = b"ACGTACGTAC";
        let result = rype_extract_minimizer_set(seq.as_ptr(), seq.len(), 16, 5, 0);
        assert!(!result.is_null());
        unsafe {
            let r = &*result;
            assert_eq!(r.forward.len, 0);
            assert!(r.forward.data.is_null());
            assert_eq!(r.reverse_complement.len, 0);
            assert!(r.reverse_complement.data.is_null());
            rype_minimizer_set_result_free(result);
        }
    }

    #[test]
    fn test_rype_extract_minimizer_set_basic() {
        // 67-byte sequence with mixed bases
        let seq = b"AAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCC";
        let result = rype_extract_minimizer_set(seq.as_ptr(), seq.len(), 16, 5, 0);
        assert!(!result.is_null());
        unsafe {
            let r = &*result;
            // Should have non-empty results
            assert!(r.forward.len > 0, "Forward set should be non-empty");
            assert!(r.reverse_complement.len > 0, "RC set should be non-empty");

            // Forward should be sorted
            let fwd = slice::from_raw_parts(r.forward.data, r.forward.len);
            for w in fwd.windows(2) {
                assert!(
                    w[0] < w[1],
                    "Forward not strictly sorted: {} >= {}",
                    w[0],
                    w[1]
                );
            }

            // RC should be sorted
            let rc = slice::from_raw_parts(r.reverse_complement.data, r.reverse_complement.len);
            for w in rc.windows(2) {
                assert!(w[0] < w[1], "RC not strictly sorted: {} >= {}", w[0], w[1]);
            }

            rype_minimizer_set_result_free(result);
        }
    }

    #[test]
    fn test_rype_minimizer_set_result_free_null() {
        // Should not crash
        rype_minimizer_set_result_free(std::ptr::null_mut());
    }

    // --- extract_strand_minimizers tests ---

    #[test]
    fn test_rype_extract_strand_minimizers_null_seq() {
        let result = rype_extract_strand_minimizers(std::ptr::null(), 100, 16, 5, 0);
        assert!(result.is_null());
    }

    #[test]
    fn test_rype_extract_strand_minimizers_invalid_k() {
        let seq = b"ACGTACGTACGTACGTACGT";
        let result = rype_extract_strand_minimizers(seq.as_ptr(), seq.len(), 0, 5, 0);
        assert!(result.is_null());
    }

    #[test]
    fn test_rype_extract_strand_minimizers_short_seq() {
        // 10 bytes, k=16: too short
        let seq = b"ACGTACGTAC";
        let result = rype_extract_strand_minimizers(seq.as_ptr(), seq.len(), 16, 5, 0);
        assert!(!result.is_null());
        unsafe {
            let r = &*result;
            assert_eq!(r.forward.len, 0);
            assert!(r.forward.hashes.is_null());
            assert!(r.forward.positions.is_null());
            assert_eq!(r.reverse_complement.len, 0);
            rype_strand_minimizers_result_free(result);
        }
    }

    #[test]
    fn test_rype_extract_strand_minimizers_basic() {
        let seq = b"AAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCC";
        let result = rype_extract_strand_minimizers(seq.as_ptr(), seq.len(), 16, 5, 0);
        assert!(!result.is_null());
        unsafe {
            let r = &*result;

            // Both strands should be non-empty
            assert!(r.forward.len > 0, "Forward should be non-empty");
            assert!(r.reverse_complement.len > 0, "RC should be non-empty");

            // Check forward strand
            let fwd_hashes = slice::from_raw_parts(r.forward.hashes, r.forward.len);
            let fwd_positions = slice::from_raw_parts(r.forward.positions, r.forward.len);

            // Positions should be non-decreasing
            for w in fwd_positions.windows(2) {
                assert!(
                    w[0] <= w[1],
                    "Forward positions not non-decreasing: {} > {}",
                    w[0],
                    w[1]
                );
            }

            // Positions should be in bounds: pos + k <= seq_len
            for &p in fwd_positions {
                assert!(
                    (p as usize) + 16 <= seq.len(),
                    "Forward position {} out of bounds",
                    p
                );
            }

            // Hashes should be non-zero (sanity check)
            assert!(fwd_hashes.iter().any(|&h| h != 0), "All hashes are zero");

            // Check RC strand positions
            let rc_positions =
                slice::from_raw_parts(r.reverse_complement.positions, r.reverse_complement.len);
            for w in rc_positions.windows(2) {
                assert!(
                    w[0] <= w[1],
                    "RC positions not non-decreasing: {} > {}",
                    w[0],
                    w[1]
                );
            }
            for &p in rc_positions {
                assert!(
                    (p as usize) + 16 <= seq.len(),
                    "RC position {} out of bounds",
                    p
                );
            }

            rype_strand_minimizers_result_free(result);
        }
    }

    #[test]
    fn test_rype_strand_minimizers_result_free_null() {
        // Should not crash
        rype_strand_minimizers_result_free(std::ptr::null_mut());
    }
}
