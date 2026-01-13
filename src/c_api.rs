use crate::{
    Index, InvertedIndex, QueryRecord,
    classify_batch, classify_batch_inverted,
    classify_batch_sharded_sequential, classify_batch_sharded_main,
    ShardedMainIndex, ShardedInvertedIndex,
    MainIndexManifest, ShardManifest,
};
use crate::memory::{parse_byte_suffix, detect_available_memory};
use std::ffi::{CStr, CString};
use std::slice;
use std::path::Path;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use libc::{c_char, size_t, c_double};

#[cfg(feature = "arrow")]
use crate::arrow::{
    classify_arrow_batch, classify_arrow_batch_inverted,
    classify_arrow_batch_sharded, classify_arrow_batch_sharded_main,
};
#[cfg(feature = "arrow")]
use arrow::ffi::FFI_ArrowSchema;
#[cfg(feature = "arrow")]
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
#[cfg(feature = "arrow")]
use arrow::record_batch::RecordBatch;

// --- Unified Index Type ---

/// Unified index type that abstracts over all index formats.
///
/// Callers don't need to know which format is being used - the API
/// automatically handles:
/// - Single-file main indices (.ryidx)
/// - Sharded main indices (.ryidx.manifest + .ryidx.shard.*)
/// - Single-file inverted indices (.ryxdi)
/// - Sharded inverted indices (.ryxdi.manifest + .ryxdi.shard.*)
pub enum RypeIndex {
    /// Single-file main index
    Single(Index),
    /// Sharded main index (manifest + shard files)
    ShardedMain(ShardedMainIndex),
    /// Single-file inverted index
    Inverted(InvertedIndex),
    /// Sharded inverted index (manifest + shard files)
    ShardedInverted(ShardedInvertedIndex),
}

impl RypeIndex {
    /// Returns the k-mer size.
    pub fn k(&self) -> usize {
        match self {
            RypeIndex::Single(idx) => idx.k,
            RypeIndex::ShardedMain(idx) => idx.k(),
            RypeIndex::Inverted(idx) => idx.k,
            RypeIndex::ShardedInverted(idx) => idx.k(),
        }
    }

    /// Returns the window size.
    pub fn w(&self) -> usize {
        match self {
            RypeIndex::Single(idx) => idx.w,
            RypeIndex::ShardedMain(idx) => idx.w(),
            RypeIndex::Inverted(idx) => idx.w,
            RypeIndex::ShardedInverted(idx) => idx.w(),
        }
    }

    /// Returns the salt.
    pub fn salt(&self) -> u64 {
        match self {
            RypeIndex::Single(idx) => idx.salt,
            RypeIndex::ShardedMain(idx) => idx.salt(),
            RypeIndex::Inverted(idx) => idx.salt,
            RypeIndex::ShardedInverted(idx) => idx.salt(),
        }
    }

    /// Returns the number of buckets, if available.
    ///
    /// Returns `None` for inverted indices as they don't store bucket metadata.
    /// Use the original main index to get bucket information.
    pub fn num_buckets(&self) -> Option<usize> {
        match self {
            RypeIndex::Single(idx) => Some(idx.buckets.len()),
            RypeIndex::ShardedMain(idx) => Some(idx.manifest().bucket_names.len()),
            RypeIndex::Inverted(_) => None, // Inverted indices don't store bucket names
            RypeIndex::ShardedInverted(_) => None,
        }
    }

    /// Returns the bucket name for a given ID, if available.
    /// Note: Inverted indices don't store bucket names - use the original main index.
    pub fn bucket_name(&self, bucket_id: u32) -> Option<&str> {
        match self {
            RypeIndex::Single(idx) => idx.bucket_names.get(&bucket_id).map(|s| s.as_str()),
            RypeIndex::ShardedMain(idx) => idx.manifest().bucket_names.get(&bucket_id).map(|s| s.as_str()),
            RypeIndex::Inverted(_) => None,
            RypeIndex::ShardedInverted(_) => None,
        }
    }

    /// Returns whether this is a sharded index.
    pub fn is_sharded(&self) -> bool {
        matches!(self, RypeIndex::ShardedMain(_) | RypeIndex::ShardedInverted(_))
    }

    /// Returns the number of shards (1 for single-file indices).
    pub fn num_shards(&self) -> usize {
        match self {
            RypeIndex::Single(_) => 1,
            RypeIndex::ShardedMain(idx) => idx.num_shards(),
            RypeIndex::Inverted(_) => 1,
            RypeIndex::ShardedInverted(idx) => idx.num_shards(),
        }
    }

    /// Returns whether this is an inverted index.
    pub fn is_inverted(&self) -> bool {
        matches!(self, RypeIndex::Inverted(_) | RypeIndex::ShardedInverted(_))
    }

    /// Estimate the memory footprint of the currently loaded index structures.
    ///
    /// For single-file indices, this estimates the memory used by all loaded data.
    /// For sharded indices, this only counts the manifest (shards are loaded on-demand).
    pub fn estimate_memory_bytes(&self) -> usize {
        match self {
            RypeIndex::Single(idx) => {
                // HashMap overhead: ~56 bytes per entry + Vec<u64> data
                let bucket_mem: usize = idx.buckets.iter()
                    .map(|(_, v)| 56 + v.len() * 8 + v.capacity() * 8)
                    .sum();
                // bucket_names: ~56 bytes per entry + String data
                let names_mem: usize = idx.bucket_names.iter()
                    .map(|(_, s)| 56 + s.len())
                    .sum();
                // bucket_sources: ~56 bytes per entry + Vec + Strings
                let sources_mem: usize = idx.bucket_sources.iter()
                    .map(|(_, v)| 56 + v.iter().map(|s| 24 + s.len()).sum::<usize>())
                    .sum();
                bucket_mem + names_mem + sources_mem
            }
            RypeIndex::ShardedMain(idx) => {
                // Only manifest is loaded; estimate based on bucket count
                let manifest = idx.manifest();
                let num_buckets = manifest.bucket_names.len();
                // Rough estimate: 200 bytes per bucket for names/maps
                num_buckets * 200
            }
            RypeIndex::Inverted(idx) => {
                // CSR format: minimizers Vec<u64> + offsets Vec<u32> + bucket_ids Vec<u32>
                idx.minimizers.len() * 8 +
                idx.offsets.len() * 4 +
                idx.bucket_ids.len() * 4
            }
            RypeIndex::ShardedInverted(idx) => {
                // Only manifest is loaded
                let manifest = idx.manifest();
                // Rough estimate based on shard count
                manifest.shards.len() * 100 + 1000
            }
        }
    }

    /// Estimate the memory needed to load a single shard.
    ///
    /// For single-file indices, returns 0 (no shards to load).
    /// For sharded indices, returns the estimated size of the largest shard.
    pub fn largest_shard_bytes(&self) -> usize {
        match self {
            RypeIndex::Single(_) | RypeIndex::Inverted(_) => 0,
            RypeIndex::ShardedMain(idx) => {
                // Estimate: 8 bytes per minimizer + HashMap overhead
                let manifest = idx.manifest();
                manifest.shards.iter()
                    .map(|s| s.num_minimizers * 8 + s.bucket_ids.len() * 100)
                    .max()
                    .unwrap_or(0)
            }
            RypeIndex::ShardedInverted(idx) => {
                // CSR format: minimizers (8 bytes) + offsets (4 bytes) + bucket_ids (4 bytes)
                let manifest = idx.manifest();
                manifest.shards.iter()
                    .map(|s| s.num_minimizers * 8 + s.num_minimizers * 4 + s.num_bucket_ids * 4)
                    .max()
                    .unwrap_or(0)
            }
        }
    }

    /// Collects all minimizers from all buckets.
    ///
    /// This only works for single-file indices. Sharded indices would require
    /// loading all shards into memory, which could exhaust RAM for large indices.
    ///
    /// For sharded indices, returns an error. If you need this functionality,
    /// process shards one at a time using the underlying shard loading APIs.
    pub fn collect_all_minimizers(&self) -> Result<HashSet<u64>, String> {
        match self {
            RypeIndex::Single(idx) => {
                Ok(idx.buckets.values()
                    .flat_map(|v| v.iter().copied())
                    .collect())
            }
            RypeIndex::ShardedMain(_) => {
                Err("Cannot collect minimizers from sharded main index: would require loading all shards into memory. Use single-file index for negative set creation.".to_string())
            }
            RypeIndex::Inverted(idx) => {
                Ok(idx.minimizers.iter().copied().collect())
            }
            RypeIndex::ShardedInverted(_) => {
                Err("Cannot collect minimizers from sharded inverted index: would require loading all shards into memory. Use single-file index for negative set creation.".to_string())
            }
        }
    }
}

// --- Error Reporting ---

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = RefCell::new(None);
}

fn set_last_error(err: String) {
    LAST_ERROR.with(|e| {
        // Sanitize null bytes to prevent silent error suppression
        let sanitized = err.replace('\0', "\\0");
        *e.borrow_mut() = Some(
            CString::new(sanitized).expect("sanitized string should not contain null bytes")
        );
    });
}

fn clear_last_error() {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = None;
    });
}

/// Validates that a pointer is non-null and properly aligned for type T.
/// Returns true if valid, false otherwise.
#[inline]
fn is_valid_ptr<T>(ptr: *const T) -> bool {
    !ptr.is_null() && (ptr as usize) % std::mem::align_of::<T>() == 0
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

/// Loads an index from disk, automatically detecting the format.
///
/// Supported formats:
/// - Single-file main index (.ryidx)
/// - Sharded main index (.ryidx with .manifest + .shard.* files)
/// - Single-file inverted index (.ryxdi)
/// - Sharded inverted index (.ryxdi with .manifest + .shard.* files)
///
/// The format is detected by:
/// 1. File extension (.ryidx = main, .ryxdi = inverted)
/// 2. Presence of .manifest file (indicates sharded)
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
    let path_str = r_str.to_lowercase();

    // Detect format based on extension and manifest presence
    let is_inverted = path_str.ends_with(".ryxdi") || path_str.contains(".ryxdi.");

    if is_inverted {
        // Inverted index: check for sharded
        let manifest_path = ShardManifest::manifest_path(path);
        if manifest_path.exists() {
            // Sharded inverted index
            match ShardedInvertedIndex::open(path) {
                Ok(idx) => {
                    clear_last_error();
                    Box::into_raw(Box::new(RypeIndex::ShardedInverted(idx)))
                }
                Err(e) => {
                    set_last_error(format!("Failed to load sharded inverted index: {}", e));
                    std::ptr::null_mut()
                }
            }
        } else {
            // Single-file inverted index
            match InvertedIndex::load(path) {
                Ok(idx) => {
                    clear_last_error();
                    Box::into_raw(Box::new(RypeIndex::Inverted(idx)))
                }
                Err(e) => {
                    set_last_error(format!("Failed to load inverted index: {}", e));
                    std::ptr::null_mut()
                }
            }
        }
    } else {
        // Main index: check for sharded
        if MainIndexManifest::is_sharded(path) {
            // Sharded main index
            match ShardedMainIndex::open(path) {
                Ok(idx) => {
                    clear_last_error();
                    Box::into_raw(Box::new(RypeIndex::ShardedMain(idx)))
                }
                Err(e) => {
                    set_last_error(format!("Failed to load sharded main index: {}", e));
                    std::ptr::null_mut()
                }
            }
        } else {
            // Single-file main index
            match Index::load(path) {
                Ok(idx) => {
                    clear_last_error();
                    Box::into_raw(Box::new(RypeIndex::Single(idx)))
                }
                Err(e) => {
                    set_last_error(format!("Failed to load index: {}", e));
                    std::ptr::null_mut()
                }
            }
        }
    }
}

/// Frees an index. NULL is safe to pass.
#[no_mangle]
pub extern "C" fn rype_index_free(ptr: *mut RypeIndex) {
    if !ptr.is_null() {
        unsafe { let _ = Box::from_raw(ptr); }
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
/// Returns:
/// - The bucket count for main indices (single or sharded)
/// - -1 (as i32, cast to u32) for inverted indices (bucket metadata not available)
/// - 0 if index is NULL or misaligned
///
/// Check with rype_index_is_inverted() to distinguish between "no buckets" and
/// "bucket count unavailable".
#[no_mangle]
pub extern "C" fn rype_index_num_buckets(index_ptr: *const RypeIndex) -> i32 {
    if !is_valid_ptr(index_ptr) {
        return 0;
    }
    let index = unsafe { &*index_ptr };
    match index.num_buckets() {
        Some(n) => n as i32,
        None => -1, // Inverted indices don't have bucket metadata
    }
}

/// Returns whether the index is an inverted index.
#[no_mangle]
pub extern "C" fn rype_index_is_inverted(index_ptr: *const RypeIndex) -> i32 {
    if !is_valid_ptr(index_ptr) {
        return 0;
    }
    let index = unsafe { &*index_ptr };
    if index.is_inverted() { 1 } else { 0 }
}

/// Returns whether the index is sharded.
#[no_mangle]
pub extern "C" fn rype_index_is_sharded(index_ptr: *const RypeIndex) -> i32 {
    if !is_valid_ptr(index_ptr) {
        return 0;
    }
    let index = unsafe { &*index_ptr };
    if index.is_sharded() { 1 } else { 0 }
}

/// Returns the number of shards in the index (1 for single-file indices).
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
/// For single-file indices, returns total memory used by the loaded data.
/// For sharded indices, returns only the manifest memory (shards load on-demand).
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
/// For single-file indices, returns 0 (no shards).
/// For sharded indices, returns the size needed to load the largest shard.
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
/// Returns NULL for inverted indices (they don't store bucket names).
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
                cache.get(&key).map(|s| s.as_ptr()).unwrap_or(std::ptr::null())
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
/// All minimizers from all buckets in the index will be used for filtering.
/// Returns NULL on error; call rype_get_last_error() for details.
///
/// The negative index must have the same k, w, and salt as the positive index
/// used for classification. This is NOT validated here but will affect results.
///
/// WARNING: For sharded indices, this loads all shards to collect minimizers,
/// which may use significant memory.
#[no_mangle]
pub extern "C" fn rype_negative_set_create(negative_index_ptr: *const RypeIndex) -> *mut RypeNegativeSet {
    if negative_index_ptr.is_null() {
        set_last_error("negative_index is NULL".to_string());
        return std::ptr::null_mut();
    }

    let neg_index = unsafe { &*negative_index_ptr };

    match neg_index.collect_all_minimizers() {
        Ok(minimizers) => {
            clear_last_error();
            Box::into_raw(Box::new(RypeNegativeSet { minimizers }))
        }
        Err(e) => {
            set_last_error(format!("Failed to collect minimizers: {}", e));
            std::ptr::null_mut()
        }
    }
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


// --- Primary Index Classification ---

/// Classifies queries using any index type without negative filtering.
/// For negative filtering support, use rype_classify_with_negative().
///
/// This function automatically dispatches to the correct classification
/// algorithm based on the index type (single main, sharded main, single
/// inverted, or sharded inverted).
#[no_mangle]
pub extern "C" fn rype_classify(
    index_ptr: *const RypeIndex,
    queries_ptr: *const RypeQuery,
    num_queries: size_t,
    threshold: c_double
) -> *mut RypeResultArray {
    // Delegate to the full version with NULL negative set
    rype_classify_with_negative(index_ptr, std::ptr::null(), queries_ptr, num_queries, threshold)
}

/// Classifies queries using any index type with optional negative filtering.
///
/// This function automatically dispatches to the correct classification
/// algorithm based on the index type:
/// - Single main index: uses classify_batch
/// - Sharded main index: uses classify_batch_sharded_main
/// - Single inverted index: uses classify_batch_inverted
/// - Sharded inverted index: uses classify_batch_sharded_sequential
///
/// Parameters:
/// - index: Any index type (loaded via rype_index_load)
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
    threshold: c_double
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
    if threshold < 0.0 || threshold > 1.0 {
        set_last_error(format!("Invalid threshold: {} (expected 0.0-1.0)", threshold));
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
        let rust_queries: Vec<QueryRecord> = c_queries.iter().map(|q| {
            let s1 = unsafe { slice::from_raw_parts(q.seq as *const u8, q.seq_len) };
            let s2 = if !q.pair_seq.is_null() {
                Some(unsafe { slice::from_raw_parts(q.pair_seq as *const u8, q.pair_len) })
            } else {
                None
            };
            (q.id, s1, s2)
        }).collect();

        // Dispatch to appropriate classification function based on index type
        // Return Result to handle errors uniformly
        let hits_result: Result<Vec<crate::HitResult>, String> = match index {
            RypeIndex::Single(idx) => {
                Ok(classify_batch(idx, neg_mins, &rust_queries, threshold))
            }
            RypeIndex::ShardedMain(idx) => {
                classify_batch_sharded_main(idx, neg_mins, &rust_queries, threshold)
                    .map_err(|e| format!("Sharded main classification failed: {}", e))
            }
            RypeIndex::Inverted(idx) => {
                Ok(classify_batch_inverted(idx, neg_mins, &rust_queries, threshold))
            }
            RypeIndex::ShardedInverted(idx) => {
                classify_batch_sharded_sequential(idx, neg_mins, &rust_queries, threshold)
                    .map_err(|e| format!("Sharded inverted classification failed: {}", e))
            }
        };

        hits_result
    });

    // Handle the result
    match result {
        Ok(Ok(hits)) => {
            // Success - convert hits to C array
            let mut c_hits: Vec<RypeHit> = hits.into_iter().map(|h| RypeHit {
                query_id: h.query_id,
                bucket_id: h.bucket_id,
                score: h.score,
            }).collect();

            let len = c_hits.len();
            let capacity = c_hits.capacity();
            let data = if c_hits.is_empty() {
                std::ptr::null_mut()
            } else {
                let ptr = c_hits.as_mut_ptr();
                std::mem::forget(c_hits);
                ptr
            };

            let result_array = Box::new(RypeResultArray { data, len, capacity });
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
// - Multiple threads can share the same RypeIndex
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
    /// SAFETY: RypeIndex variants are immutable during classification (read-only access).
    /// Caller must guarantee the pointer remains valid until the stream is consumed.
    struct SendRypeIndexPtr(*const RypeIndex);

    // SAFETY: RypeIndex is read-only during classification. All variants
    // (Single, ShardedMain, Inverted, ShardedInverted) are safe for concurrent reads.
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

    /// Classifies sequences from an Arrow stream using any index type.
    ///
    /// This function automatically dispatches to the correct classification
    /// algorithm based on the index type (single main, sharded main, single
    /// inverted, or sharded inverted).
    ///
    /// TRUE STREAMING: Processes one batch at a time. Memory usage is O(batch_size),
    /// not O(total_data). Results are available as soon as each batch is processed.
    ///
    /// # Arguments
    /// - `index_ptr`: Pointer to RypeIndex (from rype_index_load)
    /// - `negative_set_ptr`: Optional pointer to RypeNegativeSet (can be NULL)
    /// - `input_stream`: Arrow input stream with sequences
    /// - `threshold`: Minimum score threshold (0.0-1.0)
    /// - `use_merge_join`: For sharded inverted indices only: 1 = use merge-join
    ///   strategy (more efficient when minimizers overlap across queries),
    ///   0 = use sequential strategy. Ignored for other index types.
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

            // Dispatch to appropriate classification function based on index type
            let result = match index {
                RypeIndex::Single(idx) => {
                    classify_arrow_batch(idx, neg_mins, batch, threshold)
                }
                RypeIndex::ShardedMain(idx) => {
                    classify_arrow_batch_sharded_main(idx, neg_mins, batch, threshold)
                }
                RypeIndex::Inverted(idx) => {
                    classify_arrow_batch_inverted(idx, neg_mins, batch, threshold)
                }
                RypeIndex::ShardedInverted(idx) => {
                    classify_arrow_batch_sharded(idx, neg_mins, batch, threshold, merge_join)
                }
            };
            result.map_err(|e| arrow::error::ArrowError::ExternalError(Box::new(e)))
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

    #[test]
    fn test_rype_index_estimate_memory_single() {
        use crate::{Index, MinimizerWorkspace};

        let mut index = Index::new(64, 50, 0x1234).unwrap();
        let mut ws = MinimizerWorkspace::new();

        // Add some data
        let seq = vec![b'A'; 200];
        index.add_record(1, "test::seq1", &seq, &mut ws);
        index.finalize_bucket(1);
        index.bucket_names.insert(1, "TestBucket".into());

        let rype_index = RypeIndex::Single(index);

        let mem = rype_index.estimate_memory_bytes();
        // Should be non-zero (we have data)
        assert!(mem > 0, "Memory estimate should be non-zero");

        // Shard bytes should be 0 for single index
        assert_eq!(rype_index.largest_shard_bytes(), 0);
    }

    #[test]
    fn test_rype_index_estimate_memory_inverted() {
        use crate::{Index, InvertedIndex, MinimizerWorkspace};

        let mut index = Index::new(64, 50, 0x1234).unwrap();
        let mut ws = MinimizerWorkspace::new();

        let seq = vec![b'A'; 200];
        index.add_record(1, "test::seq1", &seq, &mut ws);
        index.finalize_bucket(1);

        let inverted = InvertedIndex::build_from_index(&index);
        let rype_index = RypeIndex::Inverted(inverted);

        let mem = rype_index.estimate_memory_bytes();
        assert!(mem > 0, "Memory estimate should be non-zero");

        // Shard bytes should be 0 for single inverted index
        assert_eq!(rype_index.largest_shard_bytes(), 0);
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

    /// Create a test index wrapped in RypeIndex.
    fn create_test_rype_index() -> RypeIndex {
        let mut index = Index::new(16, 5, 0x12345).unwrap();
        let mut ws = MinimizerWorkspace::new();

        let ref_seq = generate_sequence(100, 0);
        index.add_record(1, "ref1", &ref_seq, &mut ws);
        index.finalize_bucket(1);
        index.bucket_names.insert(1, "test_bucket".into());

        RypeIndex::Single(index)
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
        let rype_index = create_test_rype_index();
        let index_ptr = &rype_index as *const RypeIndex;

        let query_seq = generate_sequence(100, 0);
        let batch = make_test_batch(&[101], &[&query_seq]);

        let mut input_stream = batch_to_ffi_stream(batch);
        let mut output_stream: FFI_ArrowArrayStream = unsafe { std::mem::zeroed() };

        let result = unsafe {
            rype_classify_arrow(
                index_ptr,
                std::ptr::null(),
                &mut input_stream,
                0.0,
                0, // use_merge_join = false
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
        let rype_index = create_test_rype_index();
        let index_ptr = &rype_index as *const RypeIndex;

        // Create multiple input batches
        let batch1 = make_test_batch(&[1, 2], &[&generate_sequence(100, 0), &generate_sequence(100, 1)]);
        let batch2 = make_test_batch(&[3, 4], &[&generate_sequence(100, 0), &generate_sequence(100, 2)]);
        let batch3 = make_test_batch(&[5], &[&generate_sequence(100, 0)]);

        let mut input_stream = batches_to_ffi_stream(vec![batch1, batch2, batch3]);
        let mut output_stream: FFI_ArrowArrayStream = unsafe { std::mem::zeroed() };

        let result = unsafe {
            rype_classify_arrow(
                index_ptr,
                std::ptr::null(),
                &mut input_stream,
                0.0,
                0, // use_merge_join = false
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
            rype_classify_arrow(
                std::ptr::null(),
                std::ptr::null(),
                &mut input_stream,
                0.1,
                0, // use_merge_join = false
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
        let rype_index = create_test_rype_index();
        let index_ptr = &rype_index as *const RypeIndex;
        let mut output_stream: FFI_ArrowArrayStream = unsafe { std::mem::zeroed() };

        let result = unsafe {
            rype_classify_arrow(
                index_ptr,
                std::ptr::null(),
                std::ptr::null_mut(),
                0.1,
                0, // use_merge_join = false
                &mut output_stream,
            )
        };

        assert_eq!(result, -1, "Should fail with null input stream");
    }

    #[test]
    fn test_arrow_ffi_null_output_stream() {
        let rype_index = create_test_rype_index();
        let index_ptr = &rype_index as *const RypeIndex;
        let batch = make_test_batch(&[1], &[b"ACGT"]);
        let mut input_stream = batch_to_ffi_stream(batch);

        let result = unsafe {
            rype_classify_arrow(
                index_ptr,
                std::ptr::null(),
                &mut input_stream,
                0.1,
                0, // use_merge_join = false
                std::ptr::null_mut(),
            )
        };

        assert_eq!(result, -1, "Should fail with null output stream");
    }

    #[test]
    fn test_arrow_ffi_invalid_threshold() {
        let rype_index = create_test_rype_index();
        let index_ptr = &rype_index as *const RypeIndex;
        let batch = make_test_batch(&[1], &[&generate_sequence(100, 0)]);
        let mut output_stream: FFI_ArrowArrayStream = unsafe { std::mem::zeroed() };

        // Test negative threshold
        let mut input_stream = batch_to_ffi_stream(batch.clone());
        let result = unsafe {
            rype_classify_arrow(
                index_ptr,
                std::ptr::null(),
                &mut input_stream,
                -0.1,
                0, // use_merge_join = false
                &mut output_stream,
            )
        };
        assert_eq!(result, -1, "Should fail with negative threshold");

        // Test threshold > 1
        let mut input_stream = batch_to_ffi_stream(batch.clone());
        let result = unsafe {
            rype_classify_arrow(
                index_ptr,
                std::ptr::null(),
                &mut input_stream,
                1.5,
                0, // use_merge_join = false
                &mut output_stream,
            )
        };
        assert_eq!(result, -1, "Should fail with threshold > 1");

        // Test NaN threshold
        let mut input_stream = batch_to_ffi_stream(batch);
        let result = unsafe {
            rype_classify_arrow(
                index_ptr,
                std::ptr::null(),
                &mut input_stream,
                f64::NAN,
                0, // use_merge_join = false
                &mut output_stream,
            )
        };
        assert_eq!(result, -1, "Should fail with NaN threshold");
    }

    #[test]
    fn test_arrow_ffi_empty_stream() {
        let rype_index = create_test_rype_index();
        let index_ptr = &rype_index as *const RypeIndex;

        // Create an empty stream
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("sequence", DataType::Binary, false),
        ]));
        let reader = RecordBatchIterator::new(std::iter::empty(), schema);
        let mut input_stream = FFI_ArrowArrayStream::new(Box::new(reader));
        let mut output_stream: FFI_ArrowArrayStream = unsafe { std::mem::zeroed() };

        let result = unsafe {
            rype_classify_arrow(
                index_ptr,
                std::ptr::null(),
                &mut input_stream,
                0.1,
                0, // use_merge_join = false
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

