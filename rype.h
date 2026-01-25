#ifndef RYPE_H
#define RYPE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Rype: RY-encoded K-mer Partitioning Engine
 *
 * Version: 2.0.0
 *
 * A high-performance genomic sequence classification library using
 * minimizer-based k-mer sketching in RY (purine/pyrimidine) space.
 *
 * ## Quick Start
 *
 *     RypeIndex* idx = rype_index_load("index.ryidx");
 *     if (!idx) {
 *         fprintf(stderr, "Error: %s\n", rype_get_last_error());
 *         return 1;
 *     }
 *
 *     RypeQuery queries[1] = {{
 *         .id = 0,
 *         .seq = "ACGTACGTACGT...",
 *         .seq_len = 100,
 *         .pair_seq = NULL,
 *         .pair_len = 0
 *     }};
 *
 *     RypeResultArray* results = rype_classify(idx, queries, 1, 0.1);
 *     if (results) {
 *         for (size_t i = 0; i < results->len; i++) {
 *             const char* name = rype_bucket_name(idx, results->data[i].bucket_id);
 *             printf("Query %ld -> %s (score: %.4f)\n",
 *                    results->data[i].query_id,
 *                    name ? name : "unknown",
 *                    results->data[i].score);
 *         }
 *         rype_results_free(results);
 *     }
 *
 *     rype_index_free(idx);
 *
 * ## Unified Index API
 *
 * This library uses a unified index type that automatically handles:
 * - Single-file main indices (.ryidx)
 * - Sharded main indices (.ryidx.manifest + .ryidx.shard.*)
 * - Single-file inverted indices (.ryxdi)
 * - Sharded inverted indices (.ryxdi.manifest + .ryxdi.shard.*)
 *
 * The format is auto-detected by rype_index_load() based on file extension
 * and presence of manifest files. Classification functions automatically
 * dispatch to the appropriate algorithm.
 *
 * ## Thread Safety
 *
 * - Index loading/freeing: NOT thread-safe (use external synchronization)
 * - Index metadata queries: Thread-safe (read-only)
 * - Classification: Thread-safe (multiple threads can classify with same RypeIndex)
 * - Results: NOT thread-safe (each thread needs its own result array)
 * - Error reporting: Thread-safe (thread-local errors)
 *
 * ## Memory Management
 *
 * - All pointers returned by rype_*_load() must be freed with corresponding _free function
 * - Do NOT free RypeIndex while any rype_classify() calls are in progress
 * - Do NOT free RypeResultArray twice (undefined behavior)
 * - Do NOT free RypeResultArray from multiple threads simultaneously
 * - Strings returned by rype_bucket_name() are owned by the RypeIndex (do NOT free)
 * - Strings returned by rype_get_last_error() are valid until next API call on same thread
 *
 * ## Score Semantics
 *
 * The score returned in RypeHit represents the fraction of query minimizers
 * that match the bucket:
 *
 *     score = matching_minimizers / total_query_minimizers
 *
 * - Range: 0.0 to 1.0
 * - Higher scores indicate stronger matches
 * - Threshold of 0.1 means >= 10% of query minimizers must match
 * - Typical values: 0.05-0.2 for metagenomic classification
 */

// ============================================================================
// OPAQUE TYPES
// ============================================================================

/**
 * Opaque pointer to a unified index
 *
 * RypeIndex abstracts over all supported index formats:
 * - Single-file main indices (.ryidx)
 * - Sharded main indices (.ryidx.manifest + .ryidx.shard.*)
 * - Single-file inverted indices (.ryxdi)
 * - Sharded inverted indices (.ryxdi.manifest + .ryxdi.shard.*)
 *
 * The format is auto-detected when loading. Use rype_index_is_inverted()
 * and rype_index_is_sharded() to query the format if needed.
 *
 * Create with rype_index_load(), free with rype_index_free().
 */
typedef struct RypeIndex RypeIndex;

/**
 * Opaque pointer to a negative minimizer set
 *
 * Contains a set of minimizers to exclude from query scoring.
 * Used to filter out contaminating sequences (e.g., host DNA, adapters).
 *
 * Workflow:
 * 1. Load a negative index from contaminating sequences (same k/w/salt)
 * 2. Create a negative set: rype_negative_set_create(neg_index)
 * 3. Pass to classification: rype_classify_with_negative(..., neg_set, ...)
 * 4. Free when done: rype_negative_set_free(neg_set)
 *
 * The negative index can be freed after creating the set.
 *
 * Note: Creating negative sets from sharded indices is not supported
 * (would require loading all shards into memory). Use single-file indices.
 */
typedef struct RypeNegativeSet RypeNegativeSet;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/**
 * Query structure for sequence classification
 *
 * ## Field Requirements
 *
 * @field id       User-defined query identifier, returned unchanged in results.
 *                 Can be any int64_t value.
 *
 * @field seq      Pointer to primary sequence bytes.
 *                 MUST be non-NULL and point to at least seq_len bytes.
 *                 Accepts: A, C, G, T (case-insensitive). N and other IUPAC
 *                 codes reset k-mer extraction (reduce sensitivity, won't crash).
 *
 * @field seq_len  Length of seq in bytes.
 *                 MUST be > 0 and <= 2,000,000,000 (2GB).
 *                 Sequences shorter than k (index k-mer size) produce no minimizers.
 *
 * @field pair_seq Pointer to paired-end sequence bytes, or NULL for single-end.
 *                 If non-NULL, MUST point to at least pair_len bytes.
 *                 Same base requirements as seq.
 *
 * @field pair_len Length of pair_seq in bytes.
 *                 MUST be 0 if pair_seq is NULL.
 *                 MUST be > 0 and <= 2GB if pair_seq is non-NULL.
 *
 * ## Memory Lifetime
 *
 * All pointers must remain valid for the duration of rype_classify() call.
 * Sequences are NOT copied internally.
 *
 * ## Example
 *
 *     // Single-end query
 *     RypeQuery single = {
 *         .id = 42,
 *         .seq = "ACGTACGTACGT",
 *         .seq_len = 12,
 *         .pair_seq = NULL,
 *         .pair_len = 0
 *     };
 *
 *     // Paired-end query
 *     RypeQuery paired = {
 *         .id = 43,
 *         .seq = read1_data,
 *         .seq_len = read1_len,
 *         .pair_seq = read2_data,
 *         .pair_len = read2_len
 *     };
 */
typedef struct {
    int64_t id;           ///< User-defined query identifier
    const char* seq;      ///< Primary sequence (ACGT, case-insensitive)
    size_t seq_len;       ///< Length of seq in bytes
    const char* pair_seq; ///< Optional paired-end sequence (NULL if single-end)
    size_t pair_len;      ///< Length of pair_seq (0 if single-end)
} RypeQuery;

/**
 * Classification result for a single query
 *
 * @field query_id  The id value from the corresponding RypeQuery
 * @field bucket_id Numeric ID of the matched bucket/reference
 * @field score     Classification score: fraction of query minimizers matching
 *                  this bucket. Range [0.0, 1.0]. Higher is better.
 *
 * ## Score Interpretation
 *
 * - 1.0 = All query minimizers found in bucket (perfect match)
 * - 0.5 = Half of query minimizers found
 * - 0.1 = 10% of query minimizers found (default threshold)
 *
 * Multiple hits per query are possible if multiple buckets exceed threshold.
 * Use rype_bucket_name() to convert bucket_id to human-readable name.
 */
typedef struct {
    int64_t query_id;    ///< Query ID from RypeQuery
    uint32_t bucket_id;  ///< Matched bucket/reference ID
    double score;        ///< Classification score (0.0 - 1.0)
} RypeHit;

/**
 * Array of classification results
 *
 * Contains zero or more hits for each query that exceeded the threshold.
 * Queries with no matches above threshold are not included.
 *
 * @field data     Pointer to array of RypeHit structs
 * @field len      Number of valid hits in data array
 * @field capacity Internal capacity (ignore this field)
 *
 * ## Memory
 *
 * Owned by caller after rype_classify() returns.
 * MUST be freed with rype_results_free() exactly once.
 * Do NOT free individual RypeHit elements.
 */
typedef struct {
    RypeHit* data;       ///< Array of hits
    size_t len;          ///< Number of hits
    size_t capacity;     ///< Capacity (internal use)
} RypeResultArray;

// ============================================================================
// INDEX API
// ============================================================================

/**
 * Load an index from disk, automatically detecting the format
 *
 * Supported formats (auto-detected by extension and manifest presence):
 * - Single-file main index (.ryidx)
 * - Sharded main index (.ryidx with .manifest + .shard.* files)
 * - Single-file inverted index (.ryxdi)
 * - Sharded inverted index (.ryxdi with .manifest + .shard.* files)
 *
 * @param path  Null-terminated UTF-8 file path to index file
 * @return      Non-NULL RypeIndex pointer on success, NULL on failure
 *
 * ## Errors (returns NULL)
 *
 * - path is NULL
 * - File not found or cannot be opened
 * - File is corrupted or has wrong format/magic
 * - Unsupported index version
 * - Out of memory
 * - File contains invalid data (triggers DoS protection limits)
 *
 * ## Thread Safety
 *
 * NOT thread-safe. Use external synchronization if loading from multiple threads.
 *
 * ## Memory
 *
 * Returned RypeIndex must be freed with rype_index_free() when no longer needed.
 *
 * ## Error Details
 *
 * Call rype_get_last_error() for detailed error message.
 *
 * ## Example
 *
 *     // Load any index type - format is auto-detected
 *     RypeIndex* idx = rype_index_load("myindex.ryidx");      // main index
 *     RypeIndex* inv = rype_index_load("myindex.ryxdi");      // inverted index
 *     RypeIndex* shd = rype_index_load("myindex.ryidx");      // sharded (if .manifest exists)
 */
RypeIndex* rype_index_load(const char* path);

/**
 * Free an index
 *
 * @param index  RypeIndex pointer from rype_index_load(), or NULL (no-op)
 *
 * ## Thread Safety
 *
 * NOT thread-safe. Do NOT call while any thread is using the RypeIndex
 * in rype_classify() - this causes use-after-free.
 */
void rype_index_free(RypeIndex* index);

/**
 * Get the k-mer size of an index
 *
 * @param index  Non-NULL RypeIndex pointer
 * @return       K-mer size (16, 32, or 64), or 0 if index is NULL
 *
 * ## Thread Safety
 *
 * Thread-safe (read-only access).
 */
size_t rype_index_k(const RypeIndex* index);

/**
 * Get the window size of an index
 *
 * @param index  Non-NULL RypeIndex pointer
 * @return       Window size for minimizer selection, or 0 if index is NULL
 *
 * ## Thread Safety
 *
 * Thread-safe (read-only access).
 */
size_t rype_index_w(const RypeIndex* index);

/**
 * Get the salt value of an index
 *
 * @param index  Non-NULL RypeIndex pointer
 * @return       Salt XOR'd with k-mer hashes, or 0 if index is NULL
 *
 * ## Note
 *
 * Indices with different salts are incompatible for comparison.
 *
 * ## Thread Safety
 *
 * Thread-safe (read-only access).
 */
uint64_t rype_index_salt(const RypeIndex* index);

/**
 * Get the number of buckets in an index
 *
 * @param index  Non-NULL RypeIndex pointer
 * @return       Number of buckets for main indices, -1 for inverted indices
 *               (which don't store bucket metadata), 0 if index is NULL
 *
 * ## Note
 *
 * Inverted indices don't store bucket names/counts. Use the original main
 * index to get bucket information, or check with rype_index_is_inverted().
 *
 * ## Thread Safety
 *
 * Thread-safe (read-only access).
 */
int32_t rype_index_num_buckets(const RypeIndex* index);

/**
 * Check if an index is an inverted index
 *
 * @param index  Non-NULL RypeIndex pointer
 * @return       1 if inverted index, 0 if main index or NULL
 *
 * ## Thread Safety
 *
 * Thread-safe (read-only access).
 */
int rype_index_is_inverted(const RypeIndex* index);

/**
 * Check if an index is sharded
 *
 * @param index  Non-NULL RypeIndex pointer
 * @return       1 if sharded, 0 if single-file or NULL
 *
 * ## Thread Safety
 *
 * Thread-safe (read-only access).
 */
int rype_index_is_sharded(const RypeIndex* index);

/**
 * Get the number of shards in an index
 *
 * @param index  Non-NULL RypeIndex pointer
 * @return       Number of shards (1 for single-file indices), 0 if NULL
 *
 * ## Thread Safety
 *
 * Thread-safe (read-only access).
 */
uint32_t rype_index_num_shards(const RypeIndex* index);

// ============================================================================
// MEMORY UTILITIES
// ============================================================================

/**
 * Get the estimated memory footprint of the loaded index in bytes
 *
 * For single-file indices, returns total memory used by the loaded data.
 * For sharded indices, returns only the manifest memory (shards load on-demand).
 *
 * @param index  Non-NULL RypeIndex pointer
 * @return       Estimated memory in bytes, 0 if NULL
 *
 * ## Thread Safety
 *
 * Thread-safe (read-only access).
 */
size_t rype_index_memory_bytes(const RypeIndex* index);

/**
 * Get the estimated size of the largest shard in bytes
 *
 * For single-file indices, returns 0 (no shards).
 * For sharded indices, returns the size needed to load the largest shard.
 *
 * Use this for memory planning when classifying against sharded indices.
 *
 * @param index  Non-NULL RypeIndex pointer
 * @return       Largest shard size in bytes, 0 if NULL or not sharded
 *
 * ## Thread Safety
 *
 * Thread-safe (read-only access).
 */
size_t rype_index_largest_shard_bytes(const RypeIndex* index);

/**
 * Detect available system memory
 *
 * Detection order:
 * - Linux: cgroups v2, cgroups v1, /proc/meminfo
 * - macOS: sysctl hw.memsize
 * - Fallback: 8GB
 *
 * @return  Available memory in bytes
 *
 * ## Thread Safety
 *
 * Thread-safe.
 */
size_t rype_detect_available_memory(void);

/**
 * Parse a byte size string (e.g., "4G", "512M", "1024K")
 *
 * Supported suffixes (case-insensitive): B, K, KB, M, MB, G, GB, T, TB
 * Decimal values are supported: "1.5G"
 *
 * @param str  Null-terminated string to parse
 * @return     Size in bytes, or 0 on parse error or NULL input
 *
 * ## Notes
 *
 * - "auto" returns 0 (use rype_detect_available_memory() instead)
 * - Returns 0 for invalid input - no error message set
 *
 * ## Thread Safety
 *
 * Thread-safe.
 */
size_t rype_parse_byte_suffix(const char* str);

/**
 * Get the name of a bucket by ID
 *
 * @param index      Non-NULL RypeIndex pointer
 * @param bucket_id  Bucket ID from RypeHit.bucket_id
 * @return           Bucket name string, or NULL (see below)
 *
 * ## NULL Return Cases
 *
 * This function returns NULL in the following situations:
 *
 * 1. **index is NULL**: Caller error - always check index pointer validity
 * 2. **Inverted index**: Inverted indices (.ryxdi) don't store bucket names.
 *    Use the original main index (.ryidx) to look up bucket names.
 *    Check with rype_index_is_inverted() if uncertain.
 * 3. **Invalid bucket_id**: The bucket_id doesn't exist in the index.
 *    This is unexpected if bucket_id came from a valid RypeHit result.
 *    May indicate index corruption or version mismatch.
 *
 * ## Recommended Usage Pattern
 *
 *     const char* name = rype_bucket_name(idx, hit->bucket_id);
 *     if (name) {
 *         printf("Matched: %s\n", name);
 *     } else if (rype_index_is_inverted(idx)) {
 *         // Expected for inverted indices - use main index for names
 *         printf("Matched bucket ID: %u\n", hit->bucket_id);
 *     } else {
 *         // Unexpected - bucket_id should exist in main index
 *         fprintf(stderr, "Warning: unknown bucket ID %u\n", hit->bucket_id);
 *     }
 *
 * ## Memory
 *
 * Returned string is owned by the RypeIndex. Do NOT free it.
 * String is valid until RypeIndex is freed.
 *
 * ## Thread Safety
 *
 * Thread-safe (read-only access).
 */
const char* rype_bucket_name(const RypeIndex* index, uint32_t bucket_id);

// ============================================================================
// NEGATIVE FILTERING API
// ============================================================================

/**
 * Create a negative minimizer set from an index
 *
 * @param negative_index  RypeIndex containing sequences to filter out
 * @return                Non-NULL RypeNegativeSet on success, NULL on failure
 *
 * ## Use Case
 *
 * Negative filtering improves specificity by removing minimizers that match
 * contaminating sequences before scoring. Common uses:
 *
 * - Filter host DNA when classifying metagenomic samples
 * - Remove adapter/primer sequences
 * - Exclude known false-positive sources
 *
 * ## Requirements
 *
 * - The negative index MUST have the same k, w, and salt as the positive index
 *   used for classification. Mismatched parameters will produce incorrect results.
 * - Sharded indices are NOT supported (would require loading all shards).
 *   Use single-file indices for negative set creation.
 *
 * ## Memory
 *
 * - The negative index can be freed after creating the set
 * - The returned set must be freed with rype_negative_set_free()
 *
 * ## Thread Safety
 *
 * NOT thread-safe for creation. Safe to share for concurrent classification.
 *
 * ## Example
 *
 *     RypeIndex* host_idx = rype_index_load("human_genome.ryidx");
 *     RypeNegativeSet* host_filter = rype_negative_set_create(host_idx);
 *     rype_index_free(host_idx);  // Safe - set owns its own copy
 *
 *     // Use host_filter for all subsequent classifications
 *     RypeResultArray* results = rype_classify_with_negative(
 *         target_idx, host_filter, queries, num_queries, 0.1);
 *
 *     rype_negative_set_free(host_filter);
 */
RypeNegativeSet* rype_negative_set_create(const RypeIndex* negative_index);

/**
 * Free a negative set
 *
 * @param neg_set  RypeNegativeSet pointer, or NULL (no-op)
 *
 * ## Thread Safety
 *
 * NOT thread-safe. Do not free while classification is in progress.
 */
void rype_negative_set_free(RypeNegativeSet* neg_set);

/**
 * Get the number of unique minimizers in a negative set
 *
 * @param neg_set  RypeNegativeSet pointer
 * @return         Number of minimizers, or 0 if neg_set is NULL
 *
 * Useful for logging/debugging to verify the negative set was built correctly.
 */
size_t rype_negative_set_size(const RypeNegativeSet* neg_set);

// ============================================================================
// CLASSIFICATION API
// ============================================================================

/**
 * Classify a batch of sequences against an index
 *
 * Equivalent to rype_classify_with_negative(index, NULL, queries, num_queries, threshold).
 * Use rype_classify_with_negative() for negative filtering support.
 *
 * This function automatically dispatches to the correct classification
 * algorithm based on the index type (single main, sharded main, single
 * inverted, or sharded inverted).
 *
 * @param index        Non-NULL RypeIndex pointer from rype_index_load()
 * @param queries      Array of RypeQuery structs
 * @param num_queries  Number of queries (must be > 0 and < INTPTR_MAX)
 * @param threshold    Classification threshold (0.0-1.0, must be finite)
 * @return             Non-NULL RypeResultArray on success, NULL on failure
 *
 * See rype_classify_with_negative() for full documentation.
 */
RypeResultArray* rype_classify(
    const RypeIndex* index,
    const RypeQuery* queries,
    size_t num_queries,
    double threshold
);

/**
 * Classify a batch of sequences with optional negative filtering
 *
 * This function automatically dispatches to the correct classification
 * algorithm based on the index type:
 * - Single main index: bucket-by-bucket scoring
 * - Sharded main index: loads shards on-demand
 * - Single inverted index: minimizer-to-bucket lookup
 * - Sharded inverted index: sequential shard processing
 *
 * @param index        Non-NULL RypeIndex pointer from rype_index_load()
 * @param negative_set Optional RypeNegativeSet for filtering (NULL to disable)
 * @param queries      Array of RypeQuery structs
 * @param num_queries  Number of queries (must be > 0 and < INTPTR_MAX)
 * @param threshold    Classification threshold (0.0-1.0, must be finite)
 * @return             Non-NULL RypeResultArray on success, NULL on failure
 *
 * ## Threshold
 *
 * The threshold is the minimum fraction of query minimizers that must match
 * a bucket for it to be reported. Typical values:
 *
 * - 0.05 - High sensitivity, more false positives
 * - 0.10 - Balanced (default)
 * - 0.20 - High specificity, may miss distant matches
 *
 * ## Negative Filtering
 *
 * When negative_set is provided, minimizers matching the negative set are
 * excluded from the query before scoring. This improves specificity when
 * contaminating sequences (e.g., host DNA) share minimizers with targets.
 *
 * The score is still computed as:
 *     score = matching_minimizers / (total_query_minimizers - negative_matches)
 *
 * ## Errors (returns NULL)
 *
 * - index is NULL
 * - queries is NULL or num_queries is 0
 * - threshold is NaN, infinity, < 0.0, or > 1.0
 * - Any query violates RypeQuery requirements
 * - Out of memory
 *
 * ## Thread Safety
 *
 * Thread-safe. Multiple threads can classify concurrently with same RypeIndex
 * and negative_set.
 *
 * ## Memory
 *
 * Returned RypeResultArray MUST be freed with rype_results_free().
 *
 * ## Performance
 *
 * Uses parallel processing internally. For best throughput, batch many
 * queries together rather than calling one at a time.
 *
 * ## Example
 *
 *     // Without negative filtering
 *     RypeResultArray* results = rype_classify_with_negative(
 *         idx, NULL, queries, 1000, 0.1);
 *
 *     // With negative filtering
 *     RypeNegativeSet* host = rype_negative_set_create(host_idx);
 *     RypeResultArray* filtered = rype_classify_with_negative(
 *         idx, host, queries, 1000, 0.1);
 *
 *     for (size_t i = 0; i < filtered->len; i++) {
 *         RypeHit* hit = &filtered->data[i];
 *         const char* name = rype_bucket_name(idx, hit->bucket_id);
 *         printf("Query %ld -> %s (score: %.3f)\n",
 *                hit->query_id, name ? name : "unknown", hit->score);
 *     }
 *
 *     rype_results_free(filtered);
 *     rype_negative_set_free(host);
 */
RypeResultArray* rype_classify_with_negative(
    const RypeIndex* index,
    const RypeNegativeSet* negative_set,
    const RypeQuery* queries,
    size_t num_queries,
    double threshold
);

/**
 * Classify a batch of sequences and return only the best hit per query
 *
 * Same as rype_classify() but filters results to keep only the highest-scoring
 * bucket for each query. If multiple buckets tie for the best score, one is
 * chosen arbitrarily.
 *
 * @param index        Non-NULL RypeIndex pointer from rype_index_load()
 * @param queries      Array of RypeQuery structs
 * @param num_queries  Number of queries (must be > 0 and < INTPTR_MAX)
 * @param threshold    Classification threshold (0.0-1.0, must be finite)
 * @return             Non-NULL RypeResultArray on success, NULL on failure
 *
 * The result array contains at most one hit per query_id.
 */
RypeResultArray* rype_classify_best_hit(
    const RypeIndex* index,
    const RypeQuery* queries,
    size_t num_queries,
    double threshold
);

/**
 * Classify a batch of sequences with negative filtering, returning only the best hit per query
 *
 * Same as rype_classify_with_negative() but filters results to keep only the
 * highest-scoring bucket for each query. If multiple buckets tie for the best
 * score, one is chosen arbitrarily.
 *
 * @param index         Non-NULL RypeIndex pointer from rype_index_load()
 * @param negative_set  Optional RypeNegativeSet (NULL to disable filtering)
 * @param queries       Array of RypeQuery structs
 * @param num_queries   Number of queries (must be > 0 and < INTPTR_MAX)
 * @param threshold     Classification threshold (0.0-1.0, must be finite)
 * @return              Non-NULL RypeResultArray on success, NULL on failure
 *
 * The result array contains at most one hit per query_id.
 */
RypeResultArray* rype_classify_best_hit_with_negative(
    const RypeIndex* index,
    const RypeNegativeSet* negative_set,
    const RypeQuery* queries,
    size_t num_queries,
    double threshold
);

/**
 * Free a result array
 *
 * @param results  RypeResultArray pointer from rype_classify(), or NULL (no-op)
 *
 * ## Thread Safety
 *
 * NOT thread-safe. Each result array should be freed by one thread.
 *
 * ## Warning
 *
 * - Do NOT call twice on the same pointer (undefined behavior)
 * - Do NOT access results->data after calling this function
 */
void rype_results_free(RypeResultArray* results);

// ============================================================================
// ERROR HANDLING
// ============================================================================

/**
 * Get the last error message from a failed API call
 *
 * @return  Null-terminated error string, or NULL if no error
 *
 * ## Lifetime
 *
 * - Error messages are thread-local (each thread has its own)
 * - Error is cleared on successful API calls
 * - Returned pointer is valid until next API call on same thread
 * - Do NOT free the returned pointer
 *
 * ## Thread Safety
 *
 * Thread-safe (returns thread-local error).
 *
 * ## Example
 *
 *     RypeIndex* idx = rype_index_load("missing.ryidx");
 *     if (!idx) {
 *         const char* err = rype_get_last_error();
 *         fprintf(stderr, "Error: %s\n", err ? err : "unknown error");
 *         return 1;
 *     }
 */
const char* rype_get_last_error(void);

// ============================================================================
// ARROW C DATA INTERFACE API (Optional Feature)
// ============================================================================
//
// These functions are only available when rype is built with --features arrow.
// They use the Arrow C Data Interface for zero-copy data exchange with
// Arrow-compatible systems (Python/PyArrow, R/arrow, DuckDB, Polars, etc.).
//
// Reference: https://arrow.apache.org/docs/format/CDataInterface.html
//
// ## Building with Arrow support
//
//     cargo build --release --features arrow
//
// ## Linking
//
// When linking against rype with Arrow support, you must define RYPE_ARROW
// before including this header to enable the Arrow API declarations:
//
//     #define RYPE_ARROW
//     #include "rype.h"
//

#ifdef RYPE_ARROW

// ----------------------------------------------------------------------------
// Arrow C Data Interface Structures
// ----------------------------------------------------------------------------
// These structures match the Arrow C Data Interface specification exactly.
// See: https://arrow.apache.org/docs/format/CDataInterface.html

#ifndef ARROW_C_DATA_INTERFACE
#define ARROW_C_DATA_INTERFACE

#define ARROW_FLAG_DICTIONARY_ORDERED 1
#define ARROW_FLAG_NULLABLE 2
#define ARROW_FLAG_MAP_KEYS_SORTED 4

/**
 * Arrow C Data Interface schema structure
 *
 * Describes the type and metadata of an Arrow array.
 * Part of the Arrow C Data Interface standard.
 */
struct ArrowSchema {
    const char* format;
    const char* name;
    const char* metadata;
    int64_t flags;
    int64_t n_children;
    struct ArrowSchema** children;
    struct ArrowSchema* dictionary;
    void (*release)(struct ArrowSchema*);
    void* private_data;
};

/**
 * Arrow C Data Interface array structure
 *
 * Contains the actual data buffers for an Arrow array.
 * Part of the Arrow C Data Interface standard.
 */
struct ArrowArray {
    int64_t length;
    int64_t null_count;
    int64_t offset;
    int64_t n_buffers;
    int64_t n_children;
    const void** buffers;
    struct ArrowArray** children;
    struct ArrowArray* dictionary;
    void (*release)(struct ArrowArray*);
    void* private_data;
};

/**
 * Arrow C Stream Interface structure
 *
 * A streaming interface for Arrow record batches.
 * Part of the Arrow C Stream Interface standard.
 * See: https://arrow.apache.org/docs/format/CStreamInterface.html
 */
struct ArrowArrayStream {
    int (*get_schema)(struct ArrowArrayStream*, struct ArrowSchema* out);
    int (*get_next)(struct ArrowArrayStream*, struct ArrowArray* out);
    const char* (*get_last_error)(struct ArrowArrayStream*);
    void (*release)(struct ArrowArrayStream*);
    void* private_data;
};

#endif // ARROW_C_DATA_INTERFACE

// ----------------------------------------------------------------------------
// Arrow Classification Functions
// ----------------------------------------------------------------------------

/**
 * Classify sequences from an Arrow stream using any index type
 *
 * This function automatically dispatches to the correct classification
 * algorithm based on the index type (single main, sharded main, single
 * inverted, or sharded inverted).
 *
 * TRUE STREAMING: Processes one batch at a time. Memory usage is O(batch_size),
 * not O(total_data). Results are available as soon as each batch is processed.
 *
 * @param index          Non-NULL RypeIndex pointer from rype_index_load()
 * @param negative_set   Optional RypeNegativeSet for filtering (NULL to disable)
 * @param input_stream   Input ArrowArrayStream containing sequence batches
 * @param threshold      Classification threshold (0.0-1.0)
 * @param use_merge_join For sharded inverted indices: non-zero for merge-join
 *                       strategy, 0 for sequential. Ignored for other index types.
 * @param out_stream     Output ArrowArrayStream for results (caller-allocated)
 * @return               0 on success, -1 on error
 *
 * ## Input Schema
 *
 * The input stream must produce RecordBatches with these columns:
 *
 * | Column         | Type                    | Nullable | Description              |
 * |----------------|-------------------------|----------|--------------------------|
 * | id             | Int64                   | No       | Query identifier         |
 * | sequence       | Binary or LargeBinary   | No       | DNA sequence bytes       |
 * | pair_sequence  | Binary or LargeBinary   | Yes      | Paired-end sequence      |
 *
 * ## Output Schema
 *
 * The output stream produces RecordBatches with these columns:
 *
 * | Column    | Type    | Description                         |
 * |-----------|---------|-------------------------------------|
 * | query_id  | Int64   | Matching query ID from input        |
 * | bucket_id | UInt32  | Matched bucket/reference ID         |
 * | score     | Float64 | Classification score (0.0-1.0)      |
 *
 * ## Memory Management and Ownership
 *
 * **Input Stream Ownership (CRITICAL):**
 * - This function TAKES OWNERSHIP of input_stream via the Arrow C Data Interface
 *   release callback mechanism
 * - After this function returns, input_stream->release has been called internally
 * - The caller MUST NOT call input_stream->release() again (causes double-free)
 * - The caller MUST NOT wrap input_stream in another structure that also manages
 *   its lifetime (e.g., a C++ RAII wrapper that calls release in destructor)
 * - If using PyArrow or similar: the stream is consumed; do not reuse it
 *
 * **Output Stream Ownership:**
 * - out_stream is initialized by this function with a new stream
 * - Caller owns out_stream and MUST call out_stream->release(out_stream) when done
 * - Do NOT release out_stream if this function returns -1 (error case)
 *
 * **Schema Lifetime (Arrow C Data Interface requirement):**
 * - When consuming out_stream, the schema obtained via get_schema() must be
 *   kept alive (not released) while iterating batches via get_next()
 * - Release order: finish all get_next() calls, then release schema, then release stream
 *
 * ## Thread Safety
 *
 * Thread-safe for concurrent classification with the same RypeIndex.
 * Each thread must use its own input/output streams.
 *
 * ## Example
 *
 *     struct ArrowArrayStream input_stream;
 *     struct ArrowArrayStream output_stream;
 *
 *     // ... initialize input_stream from PyArrow, DuckDB, etc. ...
 *
 *     int result = rype_classify_arrow(
 *         idx, NULL, &input_stream, 0.1, 0, &output_stream);
 *
 *     if (result == 0) {
 *         // ... consume output_stream ...
 *         output_stream.release(&output_stream);
 *     } else {
 *         fprintf(stderr, "Error: %s\n", rype_get_last_error());
 *     }
 */
int rype_classify_arrow(
    const RypeIndex* index,
    const RypeNegativeSet* negative_set,
    struct ArrowArrayStream* input_stream,
    double threshold,
    int use_merge_join,
    struct ArrowArrayStream* out_stream
);

/**
 * Get the output schema for Arrow classification results
 *
 * Returns the schema that rype_classify_arrow() produces.
 * Useful for pre-allocating memory or validating expected output format.
 *
 * @param out_schema  Pointer to caller-allocated ArrowSchema to initialize
 * @return            0 on success, -1 on error
 *
 * ## Output Schema
 *
 * - query_id: Int64 (non-nullable)
 * - bucket_id: UInt32 (non-nullable)
 * - score: Float64 (non-nullable)
 *
 * ## Memory
 *
 * Caller must call out_schema->release(out_schema) when done.
 *
 * ## Example
 *
 *     struct ArrowSchema schema;
 *     if (rype_arrow_result_schema(&schema) == 0) {
 *         // Use schema...
 *         schema.release(&schema);
 *     }
 */
int rype_arrow_result_schema(struct ArrowSchema* out_schema);

#endif // RYPE_ARROW

#ifdef __cplusplus
}
#endif

#endif // RYPE_H
