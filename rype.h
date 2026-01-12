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
 * Version: 1.2.0
 *
 * A high-performance genomic sequence classification library using
 * minimizer-based k-mer sketching in RY (purine/pyrimidine) space.
 *
 * ## Quick Start
 *
 *     Index* idx = rype_index_load("index.ryidx");
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
 * ## Thread Safety
 *
 * - Index loading/freeing: NOT thread-safe (use external synchronization)
 * - Index metadata queries: Thread-safe (read-only)
 * - Classification: Thread-safe (multiple threads can classify with same Index)
 * - Results: NOT thread-safe (each thread needs its own result array)
 * - Error reporting: Thread-safe (thread-local errors)
 *
 * ## Memory Management
 *
 * - All pointers returned by rype_*_load() must be freed with corresponding _free function
 * - Do NOT free Index while any rype_classify() calls are in progress
 * - Do NOT free RypeResultArray twice (undefined behavior)
 * - Do NOT free RypeResultArray from multiple threads simultaneously
 * - Strings returned by rype_bucket_name() are owned by the Index (do NOT free)
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
 * Opaque pointer to a primary index
 *
 * Contains bucket ID → sorted minimizer vectors, plus metadata.
 * Create with rype_index_load(), free with rype_index_free().
 */
typedef struct Index Index;

/**
 * Opaque pointer to an inverted index
 *
 * Contains minimizer → bucket ID mappings for fast O(Q log U) classification.
 * Create with rype_inverted_load(), free with rype_inverted_free().
 *
 * Performance: Use inverted index when classifying many reads against
 * a large index. Falls back gracefully if not available.
 */
typedef struct InvertedIndex InvertedIndex;

/**
 * Opaque pointer to a negative minimizer set
 *
 * Contains a set of minimizers to exclude from query scoring.
 * Used to filter out contaminating sequences (e.g., host DNA, adapters).
 *
 * Workflow:
 * 1. Build a negative index from contaminating sequences (same k/w/salt)
 * 2. Create a negative set: rype_negative_set_create(neg_index)
 * 3. Pass to classification: rype_classify_with_negative(..., neg_set, ...)
 * 4. Free when done: rype_negative_set_free(neg_set)
 *
 * The negative index can be freed after creating the set.
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
// PRIMARY INDEX API
// ============================================================================

/**
 * Load a primary index from disk
 *
 * @param path  Null-terminated UTF-8 file path to .ryidx index file
 * @return      Non-NULL Index pointer on success, NULL on failure
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
 * Returned Index must be freed with rype_index_free() when no longer needed.
 *
 * ## Error Details
 *
 * Call rype_get_last_error() for detailed error message.
 */
Index* rype_index_load(const char* path);

/**
 * Free a primary index
 *
 * @param index  Index pointer from rype_index_load(), or NULL (no-op)
 *
 * ## Thread Safety
 *
 * NOT thread-safe. Do NOT call while any thread is using the Index
 * in rype_classify() - this causes use-after-free.
 */
void rype_index_free(Index* index);

/**
 * Get the k-mer size of an index
 *
 * @param index  Non-NULL Index pointer
 * @return       K-mer size (16, 32, or 64), or 0 if index is NULL
 *
 * ## Thread Safety
 *
 * Thread-safe (read-only access).
 */
size_t rype_index_k(const Index* index);

/**
 * Get the window size of an index
 *
 * @param index  Non-NULL Index pointer
 * @return       Window size for minimizer selection, or 0 if index is NULL
 *
 * ## Thread Safety
 *
 * Thread-safe (read-only access).
 */
size_t rype_index_w(const Index* index);

/**
 * Get the salt value of an index
 *
 * @param index  Non-NULL Index pointer
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
uint64_t rype_index_salt(const Index* index);

/**
 * Get the number of buckets in an index
 *
 * @param index  Non-NULL Index pointer
 * @return       Number of buckets, or 0 if index is NULL
 *
 * ## Thread Safety
 *
 * Thread-safe (read-only access).
 */
uint32_t rype_index_num_buckets(const Index* index);

/**
 * Get the name of a bucket by ID
 *
 * @param index      Non-NULL Index pointer
 * @param bucket_id  Bucket ID from RypeHit.bucket_id
 * @return           Bucket name string, or NULL if not found
 *
 * ## Memory
 *
 * Returned string is owned by the Index. Do NOT free it.
 * String is valid until Index is freed.
 *
 * ## Thread Safety
 *
 * Thread-safe (read-only access).
 *
 * ## Example
 *
 *     for (size_t i = 0; i < results->len; i++) {
 *         const char* name = rype_bucket_name(idx, results->data[i].bucket_id);
 *         printf("Matched: %s\n", name ? name : "unknown");
 *     }
 */
const char* rype_bucket_name(const Index* index, uint32_t bucket_id);

// ============================================================================
// NEGATIVE FILTERING API
// ============================================================================

/**
 * Create a negative minimizer set from an index
 *
 * @param negative_index  Index containing sequences to filter out
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
 * The negative index MUST have the same k, w, and salt as the positive index
 * used for classification. Mismatched parameters will produce incorrect results.
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
 *     Index* host_idx = rype_index_load("human_genome.ryidx");
 *     RypeNegativeSet* host_filter = rype_negative_set_create(host_idx);
 *     rype_index_free(host_idx);  // Safe - set owns its own copy
 *
 *     // Use host_filter for all subsequent classifications
 *     RypeResultArray* results = rype_classify_with_negative(
 *         target_idx, host_filter, queries, num_queries, 0.1);
 *
 *     rype_negative_set_free(host_filter);
 */
RypeNegativeSet* rype_negative_set_create(const Index* negative_index);

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
// INVERTED INDEX API
// ============================================================================

/**
 * Load an inverted index from disk
 *
 * Inverted indices provide faster classification for large indices by
 * mapping minimizers → buckets instead of buckets → minimizers.
 *
 * @param path  Null-terminated UTF-8 file path to .ryxdi inverted index file
 * @return      Non-NULL InvertedIndex pointer on success, NULL on failure
 *
 * ## Creating Inverted Indices
 *
 * Use the CLI to create inverted indices from primary indices:
 *
 *     rype index invert -i primary.ryidx
 *
 * This creates primary.ryxdi in the same directory.
 *
 * ## Errors (returns NULL)
 *
 * - path is NULL
 * - File not found
 * - File corrupted or wrong format
 * - Sharded inverted index (not supported via C API yet)
 *
 * ## Thread Safety
 *
 * NOT thread-safe for loading. Safe for concurrent classification.
 *
 * ## Memory
 *
 * Returned InvertedIndex must be freed with rype_inverted_free().
 */
InvertedIndex* rype_inverted_load(const char* path);

/**
 * Free an inverted index
 *
 * @param index  InvertedIndex pointer from rype_inverted_load(), or NULL (no-op)
 *
 * ## Thread Safety
 *
 * NOT thread-safe. Do NOT call while any thread is classifying.
 */
void rype_inverted_free(InvertedIndex* index);

/**
 * Classify sequences using an inverted index
 *
 * This is typically 2-10x faster than rype_classify() for large indices.
 *
 * @param inverted     InvertedIndex for fast lookups
 * @param negative_set Optional RypeNegativeSet for filtering (NULL to disable)
 * @param queries      Array of RypeQuery structs
 * @param num_queries  Number of queries
 * @param threshold    Classification threshold (0.0-1.0)
 * @return             Non-NULL RypeResultArray on success, NULL on failure
 *
 * ## Requirements
 *
 * - inverted must be non-NULL
 * - The inverted index must have been built from the same primary index
 *   (validated by internal hash check)
 *
 * ## Thread Safety
 *
 * Thread-safe for concurrent classification with same inverted/negative_set pair.
 *
 * ## Memory
 *
 * Returned RypeResultArray MUST be freed with rype_results_free().
 *
 * ## Note
 *
 * Use rype_bucket_name() with the primary Index to convert bucket IDs to names.
 */
RypeResultArray* rype_classify_inverted(
    const InvertedIndex* inverted,
    const RypeNegativeSet* negative_set,
    const RypeQuery* queries,
    size_t num_queries,
    double threshold
);

// ============================================================================
// CLASSIFICATION API
// ============================================================================

/**
 * Classify a batch of sequences against a primary index
 *
 * Equivalent to rype_classify_with_negative(index, NULL, queries, num_queries, threshold).
 * Use rype_classify_with_negative() for negative filtering support.
 *
 * @param index        Non-NULL Index pointer from rype_index_load()
 * @param queries      Array of RypeQuery structs
 * @param num_queries  Number of queries (must be > 0 and < INTPTR_MAX)
 * @param threshold    Classification threshold (0.0-1.0, must be finite)
 * @return             Non-NULL RypeResultArray on success, NULL on failure
 *
 * See rype_classify_with_negative() for full documentation.
 */
RypeResultArray* rype_classify(
    const Index* index,
    const RypeQuery* queries,
    size_t num_queries,
    double threshold
);

/**
 * Classify a batch of sequences with optional negative filtering
 *
 * @param index        Non-NULL Index pointer from rype_index_load()
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
 * Thread-safe. Multiple threads can classify concurrently with same Index
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
    const Index* index,
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
 *     Index* idx = rype_index_load("missing.ryidx");
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
// Sharded Inverted Index (for large indices that don't fit in memory)
// ----------------------------------------------------------------------------

/**
 * Opaque pointer to a sharded inverted index
 *
 * Sharded indices split the inverted index across multiple files, allowing
 * classification when the full index exceeds available memory. Shards are
 * loaded on-demand during classification.
 *
 * Create with rype_sharded_load(), free with rype_sharded_free().
 */
typedef struct ShardedInvertedIndex ShardedInvertedIndex;

/**
 * Load a sharded inverted index from a manifest file
 *
 * @param path  Null-terminated UTF-8 path to the .ryxdi.manifest file
 * @return      Non-NULL ShardedInvertedIndex pointer on success, NULL on failure
 *
 * ## Creating Sharded Indices
 *
 * Use the CLI with --shards to create sharded inverted indices:
 *
 *     rype index invert -i primary.ryidx --shards 4
 *
 * This creates:
 *   - primary.ryxdi.manifest   (manifest file - pass this to rype_sharded_load)
 *   - primary.ryxdi.shard.0    (shard 0)
 *   - primary.ryxdi.shard.1    (shard 1)
 *   - ...
 *
 * ## Memory Usage
 *
 * Only the manifest is loaded immediately. Individual shards are loaded
 * on-demand during classification, keeping memory usage proportional to
 * a single shard rather than the full index.
 *
 * ## Errors (returns NULL)
 *
 * - path is NULL
 * - Manifest file not found
 * - Invalid manifest format
 * - Shard files missing or corrupted
 *
 * ## Thread Safety
 *
 * NOT thread-safe for loading. Safe for concurrent classification.
 */
ShardedInvertedIndex* rype_sharded_load(const char* path);

/**
 * Free a sharded inverted index
 *
 * @param sharded  ShardedInvertedIndex pointer, or NULL (no-op)
 *
 * ## Thread Safety
 *
 * NOT thread-safe. Do NOT call while any thread is classifying.
 */
void rype_sharded_free(ShardedInvertedIndex* sharded);

// ----------------------------------------------------------------------------
// Arrow Classification Functions
// ----------------------------------------------------------------------------

/**
 * Classify sequences from an Arrow RecordBatch using a primary Index
 *
 * Uses the Arrow C Data Interface for zero-copy data exchange.
 *
 * @param index          Non-NULL Index pointer from rype_index_load()
 * @param negative_set   Optional RypeNegativeSet for filtering (NULL to disable)
 * @param input_stream   Input ArrowArrayStream containing sequence batches
 * @param threshold      Classification threshold (0.0-1.0)
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
 * | Column    | Type    | Description                     |
 * |-----------|---------|-------------------------------------|
 * | query_id  | Int64   | Matching query ID from input        |
 * | bucket_id | UInt32  | Matched bucket/reference ID         |
 * | score     | Float64 | Classification score (0.0-1.0)      |
 *
 * ## Memory Management
 *
 * - input_stream: Consumed by this function (stream reader takes ownership)
 * - out_stream: Function writes a new stream; caller must release when done
 *
 * ## Thread Safety
 *
 * Thread-safe for concurrent classification with the same Index.
 * Each thread must use its own input/output streams.
 *
 * ## Example
 *
 *     struct ArrowArrayStream input_stream;
 *     struct ArrowArrayStream output_stream;
 *
 *     // ... initialize input_stream from PyArrow, DuckDB, etc. ...
 *
 *     int result = rype_classify_arrow_batch(
 *         idx, NULL, &input_stream, 0.1, &output_stream);
 *
 *     if (result == 0) {
 *         // ... consume output_stream ...
 *         output_stream.release(&output_stream);
 *     } else {
 *         fprintf(stderr, "Error: %s\n", rype_get_last_error());
 *     }
 */
int rype_classify_arrow_batch(
    const Index* index,
    const RypeNegativeSet* negative_set,
    struct ArrowArrayStream* input_stream,
    double threshold,
    struct ArrowArrayStream* out_stream
);

/**
 * Classify sequences from an Arrow RecordBatch using an InvertedIndex
 *
 * This is the recommended API for large indices. Uses the inverted index
 * for O(Q log U) complexity instead of O(B × Q × log M).
 *
 * @param inverted       Non-NULL InvertedIndex pointer from rype_inverted_load()
 * @param negative_set   Optional RypeNegativeSet for filtering (NULL to disable)
 * @param input_stream   Input ArrowArrayStream containing sequence batches
 * @param threshold      Classification threshold (0.0-1.0)
 * @param out_stream     Output ArrowArrayStream for results (caller-allocated)
 * @return               0 on success, -1 on error
 *
 * See rype_classify_arrow_batch() for input/output schema and usage details.
 */
int rype_classify_arrow_batch_inverted(
    const InvertedIndex* inverted,
    const RypeNegativeSet* negative_set,
    struct ArrowArrayStream* input_stream,
    double threshold,
    struct ArrowArrayStream* out_stream
);

/**
 * Classify sequences from an Arrow RecordBatch using a ShardedInvertedIndex
 *
 * Use this when the full inverted index exceeds available memory. Shards are
 * loaded on-demand from disk during classification.
 *
 * @param sharded        Non-NULL ShardedInvertedIndex from rype_sharded_load()
 * @param negative_set   Optional RypeNegativeSet for filtering (NULL to disable)
 * @param input_stream   Input ArrowArrayStream containing sequence batches
 * @param threshold      Classification threshold (0.0-1.0)
 * @param use_merge_join Non-zero for merge-join strategy, 0 for sequential lookup
 * @param out_stream     Output ArrowArrayStream for results (caller-allocated)
 * @return               0 on success, -1 on error
 *
 * ## Algorithm Selection
 *
 * - use_merge_join=0: Sequential lookup - loads each shard and queries it
 * - use_merge_join=1: Merge-join - more efficient when queries have high
 *   minimizer overlap with the index
 *
 * ## Memory Complexity
 *
 * O(batch_size × minimizers_per_read) + O(single_shard_size)
 *
 * See rype_classify_arrow_batch() for input/output schema and usage details.
 */
int rype_classify_arrow_batch_sharded(
    const ShardedInvertedIndex* sharded,
    const RypeNegativeSet* negative_set,
    struct ArrowArrayStream* input_stream,
    double threshold,
    int use_merge_join,
    struct ArrowArrayStream* out_stream
);

/**
 * Get the output schema for Arrow classification results
 *
 * Returns the schema that all Arrow classification functions produce.
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
