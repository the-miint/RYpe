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
 *     RypeIndex* idx = rype_index_load("index.ryxdi");
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
 * ## Index Format
 *
 * This library uses Parquet-based sharded inverted indices stored as directories:
 *
 *     index.ryxdi/
 *     ├── manifest.toml           # TOML metadata (k, w, salt, bucket info)
 *     ├── buckets.parquet         # Bucket metadata (id, name, sources)
 *     └── inverted/
 *         ├── shard.0.parquet     # (minimizer, bucket_id) pairs
 *         ├── shard.1.parquet     # Additional shards for large indices
 *         └── ...
 *
 * All indices are sharded inverted indices. The format is auto-detected by
 * rype_index_load() when opening a .ryxdi directory.
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
 * Opaque pointer to a Parquet inverted index
 *
 * RypeIndex represents a sharded Parquet inverted index stored as a directory:
 *
 *     index.ryxdi/
 *     ├── manifest.toml       # Index metadata
 *     ├── buckets.parquet     # Bucket names and sources
 *     └── inverted/           # Parquet shards with (minimizer, bucket_id) pairs
 *
 * Create with rype_index_load(), free with rype_index_free().
 */
typedef struct RypeIndex RypeIndex;

/**
 * Opaque pointer to a negative minimizer set
 *
 * Contains a shared reference (via Arc) to an index used to filter query
 * minimizers during classification. Used to filter out contaminating sequences
 * (e.g., host DNA, adapters).
 *
 * Memory-efficient: Uses sharded filtering that loads one shard at a time,
 * rather than loading all minimizers into memory at once. Creating a negative
 * set from an index is a cheap operation (increments a reference count) rather
 * than copying data.
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
 * Load an index from disk
 *
 * Supported format:
 * - Parquet inverted index directory (.ryxdi with manifest.toml)
 *
 * @param path  Null-terminated UTF-8 path to index directory
 * @return      Non-NULL RypeIndex pointer on success, NULL on failure
 *
 * ## Errors (returns NULL)
 *
 * - path is NULL
 * - Directory not found or cannot be opened
 * - Missing or corrupted manifest.toml
 * - Unsupported format version
 * - Out of memory
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
 *     RypeIndex* idx = rype_index_load("bacteria.ryxdi");
 *     if (!idx) {
 *         fprintf(stderr, "Error: %s\n", rype_get_last_error());
 *         return 1;
 *     }
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
 * @return       Number of buckets, or 0 if index is NULL
 *
 * ## Thread Safety
 *
 * Thread-safe (read-only access).
 */
int32_t rype_index_num_buckets(const RypeIndex* index);

/**
 * Check if an index is sharded
 *
 * @param index  Non-NULL RypeIndex pointer
 * @return       Always 1 (all Parquet indices are sharded), or 0 if NULL
 *
 * ## Note
 *
 * All Parquet indices are sharded, even if they contain only one shard.
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
 * @return       Number of shards (>= 1), or 0 if NULL
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
 * Returns a **lower bound estimate** of memory needed to load all shards.
 * Actual usage will be 1.5-2x higher due to Arrow array overhead and
 * temporary allocations. Shards are loaded on-demand during classification.
 *
 * @param index  Non-NULL RypeIndex pointer
 * @return       Estimated memory in bytes (lower bound), 0 if NULL
 *
 * ## Thread Safety
 *
 * Thread-safe (read-only access).
 */
size_t rype_index_memory_bytes(const RypeIndex* index);

/**
 * Get the estimated size of the largest shard in bytes
 *
 * Returns the estimated memory needed to load the largest single shard.
 * Use this for memory planning when classifying against sharded indices.
 *
 * @param index  Non-NULL RypeIndex pointer
 * @return       Largest shard size in bytes, or 0 if NULL
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
 * @return           Bucket name string, or NULL if not found
 *
 * ## NULL Return Cases
 *
 * - index is NULL
 * - bucket_id doesn't exist in the index
 *
 * **WARNING**: ALWAYS check return value before use. Passing NULL to
 * printf("%s") causes undefined behavior. Use pattern:
 *
 *     const char* name = rype_bucket_name(idx, hit->bucket_id);
 *     printf("%s\n", name ? name : "unknown");
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
 * @return                RypeNegativeSet pointer on success, NULL on failure
 *
 * The negative set uses memory-efficient sharded filtering. During
 * classification, negative shards are processed one at a time, so memory
 * usage is O(single_shard) rather than O(entire_index).
 *
 * ## Example
 *
 *     RypeIndex* neg_idx = rype_index_load("contaminants.ryxdi");
 *     RypeNegativeSet* neg_set = rype_negative_set_create(neg_idx);
 *     // neg_idx can be freed after creating neg_set if desired
 *     rype_index_free(neg_idx);
 *
 *     // Use neg_set for classification
 *     RypeResultArray* results = rype_classify_with_negative(
 *         main_idx, neg_set, queries, num_queries, threshold
 *     );
 *
 *     rype_negative_set_free(neg_set);
 *
 * ## Thread Safety
 *
 * NOT thread-safe. Use external synchronization when creating from multiple threads.
 *
 * ## Memory
 *
 * The returned RypeNegativeSet must be freed with rype_negative_set_free().
 * The RypeNegativeSet shares the underlying index data with the source RypeIndex
 * via reference counting (Arc). This means:
 * - Creating a negative set is cheap (no data copying)
 * - The original negative_index can be freed after creating the negative set
 * - The negative set remains valid and usable after freeing the source index
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
 * Get the total minimizer count in a negative set
 *
 * @param neg_set  RypeNegativeSet pointer
 * @return         Total minimizer count from the index manifest, or 0 if NULL
 *
 * Note: This returns the total entry count from the index manifest.
 * Since shards may contain duplicate entries, this is an upper bound
 * rather than an exact count of unique minimizers.
 */
size_t rype_negative_set_size(const RypeNegativeSet* neg_set);

// ============================================================================
// CLASSIFICATION API
// ============================================================================

/**
 * Classify a batch of sequences against an index
 *
 * Equivalent to rype_classify_with_negative(index, NULL, queries, num_queries, threshold).
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
 * Processes queries against a Parquet sharded inverted index, loading
 * shards sequentially to minimize memory usage.
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
 * Thread-safe. Multiple threads can classify concurrently with same RypeIndex.
 *
 * ## Memory
 *
 * Returned RypeResultArray MUST be freed with rype_results_free().
 *
 * ## Performance
 *
 * Uses parallel processing internally. For best throughput, batch many
 * queries together rather than calling one at a time.
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
 * Classify with negative filtering, returning only the best hit per query
 *
 * Same as rype_classify_with_negative() but filters results to keep only the
 * highest-scoring bucket for each query.
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
// LOG-RATIO CLASSIFICATION API
// ============================================================================
//
// Log-ratio classification computes log10(numerator_score / denominator_score)
// for each read against two single-bucket indices. This is useful for
// differential abundance analysis.
//
// ## Score Semantics
//
// - Positive log-ratio: read matches numerator more strongly
// - Negative log-ratio: read matches denominator more strongly
// - +inf: numerator > 0, denominator = 0 (or fast-path: numerator exceeded skip threshold)
// - -inf: numerator = 0, denominator > 0
// - NaN: both numerator and denominator = 0 (no evidence)
//
// ## Fast-Path
//
// When numerator_skip_threshold is set (0.0 < threshold <= 1.0), reads whose
// numerator score exceeds the threshold are assigned +inf without classifying
// against the denominator, saving computation.

/**
 * Log-ratio classification result for a single query
 *
 * @field query_id   The id value from the corresponding RypeQuery
 * @field log_ratio  log10(numerator_score / denominator_score).
 *                   Can be +inf, -inf, or NaN (see score semantics above).
 * @field fast_path  0 = computed exactly (both indices classified),
 *                   1 = numerator exceeded skip threshold (+inf fast-path)
 */
typedef struct {
    int64_t query_id;    ///< Query ID from RypeQuery
    double log_ratio;    ///< log10(num/denom), can be +inf, -inf, NaN
    int32_t fast_path;   ///< 0 = None, 1 = NumHigh
} RypeLogRatioHit;

/**
 * Array of log-ratio classification results
 *
 * Contains one result per input query (unlike RypeResultArray which only
 * includes queries exceeding the threshold).
 *
 * @field data     Pointer to array of RypeLogRatioHit structs
 * @field len      Number of results in data array
 * @field capacity Internal capacity (ignore this field)
 *
 * ## Memory
 *
 * Owned by caller after rype_classify_log_ratio() returns.
 * MUST be freed with rype_log_ratio_results_free() exactly once.
 */
typedef struct {
    RypeLogRatioHit* data;  ///< Array of log-ratio results
    size_t len;             ///< Number of results
    size_t capacity;        ///< Capacity (internal use)
} RypeLogRatioResultArray;

/**
 * Validate two indices are compatible for log-ratio classification
 *
 * Checks that both indices have exactly 1 bucket and that their k, w, and
 * salt parameters match.
 *
 * @param numerator    Non-NULL RypeIndex pointer (must be single-bucket)
 * @param denominator  Non-NULL RypeIndex pointer (must be single-bucket)
 * @return             0 on success, -1 on error
 *
 * ## Thread Safety
 *
 * Thread-safe (read-only access).
 */
int rype_validate_log_ratio_indices(
    const RypeIndex* numerator,
    const RypeIndex* denominator
);

/**
 * Classify a batch using log-ratio (numerator vs denominator)
 *
 * @param numerator                Non-NULL RypeIndex pointer (single-bucket)
 * @param denominator              Non-NULL RypeIndex pointer (single-bucket)
 * @param queries                  Array of RypeQuery structs
 * @param num_queries              Number of queries (must be > 0)
 * @param numerator_skip_threshold Fast-path threshold:
 *                                 <= 0.0: disabled (classify all against both)
 *                                 (0.0, 1.0]: enabled (fast-path for high-scoring reads)
 *                                 > 1.0, NaN, inf: error (returns NULL)
 * @return                         Non-NULL RypeLogRatioResultArray on success, NULL on failure
 *
 * ## Paired-End Support
 *
 * Paired-end reads are supported via RypeQuery pair_seq/pair_len fields.
 * Minimizers from both ends are combined before scoring.
 *
 * ## Thread Safety
 *
 * Thread-safe. Multiple threads can classify concurrently with the same
 * RypeIndex pointers.
 *
 * ## Memory
 *
 * Returned RypeLogRatioResultArray MUST be freed with rype_log_ratio_results_free().
 */
RypeLogRatioResultArray* rype_classify_log_ratio(
    const RypeIndex* numerator,
    const RypeIndex* denominator,
    const RypeQuery* queries,
    size_t num_queries,
    double numerator_skip_threshold
);

/**
 * Free a log-ratio result array
 *
 * @param results  RypeLogRatioResultArray pointer, or NULL (no-op)
 *
 * ## Warning
 *
 * - Do NOT call twice on the same pointer (undefined behavior)
 * - Do NOT access results->data after calling this function
 */
void rype_log_ratio_results_free(RypeLogRatioResultArray* results);

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
 */
const char* rype_get_last_error(void);

// ============================================================================
// ARROW C DATA INTERFACE API (Optional Feature)
// ============================================================================
//
// These functions are only available when rype is built with --features arrow-ffi.
// They use the Arrow C Data Interface for zero-copy data exchange with
// Arrow-compatible systems (Python/PyArrow, R/arrow, DuckDB, Polars, etc.).
//
// Reference: https://arrow.apache.org/docs/format/CDataInterface.html
//
// ## Building with Arrow support
//
//     cargo build --release --features arrow-ffi
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
 * Classify sequences from an Arrow stream
 *
 * TRUE STREAMING: Processes one batch at a time. Memory usage is O(batch_size),
 * not O(total_data). Results are available as soon as each batch is processed.
 *
 * @param index          Non-NULL RypeIndex pointer from rype_index_load()
 * @param negative_set   Optional RypeNegativeSet for filtering (NULL to disable)
 * @param input_stream   Input ArrowArrayStream containing sequence batches
 * @param threshold      Classification threshold (0.0-1.0)
 * @param use_merge_join Non-zero for merge-join strategy, 0 for sequential
 * @param out_stream     Output ArrowArrayStream for results (caller-allocated)
 * @return               0 on success, -1 on error
 *
 * ## Input Schema
 *
 * | Column         | Type                    | Nullable | Description              |
 * |----------------|-------------------------|----------|--------------------------|
 * | id             | Int64                   | No       | Query identifier         |
 * | sequence       | Binary or LargeBinary   | No       | DNA sequence bytes       |
 * | pair_sequence  | Binary or LargeBinary   | Yes      | Paired-end sequence      |
 *
 * ## Output Schema
 *
 * | Column    | Type    | Description                         |
 * |-----------|---------|-------------------------------------|
 * | query_id  | Int64   | Matching query ID from input        |
 * | bucket_id | UInt32  | Matched bucket/reference ID         |
 * | score     | Float64 | Classification score (0.0-1.0)      |
 *
 * ## Memory Management
 *
 * - This function TAKES OWNERSHIP of input_stream
 * - Caller owns out_stream and MUST call out_stream->release() when done
 * - Do NOT release out_stream if this function returns -1
 *
 * ## Thread Safety
 *
 * Thread-safe for concurrent classification with the same RypeIndex.
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
 * Classify from Arrow stream, returning only the best hit per query
 *
 * Same as rype_classify_arrow() but filters results to keep only the
 * highest-scoring bucket for each query.
 */
int rype_classify_arrow_best_hit(
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
 * @param out_schema  Pointer to caller-allocated ArrowSchema to initialize
 * @return            0 on success, -1 on error
 *
 * Caller must call out_schema->release(out_schema) when done.
 */
int rype_arrow_result_schema(struct ArrowSchema* out_schema);

// ----------------------------------------------------------------------------
// Arrow Log-Ratio Classification Functions
// ----------------------------------------------------------------------------

/**
 * Get the output schema for Arrow log-ratio results
 *
 * Schema: query_id (Int64), log_ratio (Float64), fast_path (Int32)
 *
 * @param out_schema  Pointer to caller-allocated ArrowSchema to initialize
 * @return            0 on success, -1 on error
 *
 * Caller must call out_schema->release(out_schema) when done.
 */
int rype_arrow_log_ratio_result_schema(struct ArrowSchema* out_schema);

/**
 * Classify sequences from an Arrow stream using log-ratio
 *
 * TRUE STREAMING: Processes one batch at a time. Memory usage is O(batch_size).
 *
 * @param numerator                Non-NULL RypeIndex pointer (single-bucket)
 * @param denominator              Non-NULL RypeIndex pointer (single-bucket)
 * @param input_stream             Input ArrowArrayStream containing sequence batches
 * @param numerator_skip_threshold Fast-path threshold (see rype_classify_log_ratio)
 * @param out_stream               Output ArrowArrayStream for results (caller-allocated)
 * @return                         0 on success, -1 on error
 *
 * ## Output Schema
 *
 * | Column    | Type    | Description                                     |
 * |-----------|---------|-------------------------------------------------|
 * | query_id  | Int64   | Query ID from input                             |
 * | log_ratio | Float64 | log10(num/denom), can be +inf, -inf, NaN        |
 * | fast_path | Int32   | 0 = computed exactly, 1 = numerator fast-path   |
 *
 * ## Memory Management
 *
 * - This function TAKES OWNERSHIP of input_stream
 * - Caller owns out_stream and MUST call out_stream->release() when done
 * - Do NOT release out_stream if this function returns -1
 *
 * ## Thread Safety
 *
 * Thread-safe for concurrent classification with the same RypeIndex pointers.
 */
int rype_classify_arrow_log_ratio(
    const RypeIndex* numerator,
    const RypeIndex* denominator,
    struct ArrowArrayStream* input_stream,
    double numerator_skip_threshold,
    struct ArrowArrayStream* out_stream
);

#endif // RYPE_ARROW

#ifdef __cplusplus
}
#endif

#endif // RYPE_H
