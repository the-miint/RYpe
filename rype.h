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
 * A high-performance genomic sequence classification library using
 * minimizer-based k-mer sketching in RY (purine/pyrimidine) space.
 *
 * Thread Safety:
 * - Index loading/freeing: NOT thread-safe (use external synchronization)
 * - Classification: Thread-safe (multiple threads can classify with same Index)
 * - Results: NOT thread-safe (each thread needs its own result array)
 *
 * Memory Management:
 * - All pointers returned by rype_* functions must be freed with corresponding _free function
 * - Do NOT free Index while any rype_classify() calls are in progress
 * - Do NOT free RypeResultArray twice (undefined behavior)
 * - Do NOT free RypeResultArray from multiple threads simultaneously
 */

// Opaque pointer to the Rust Index object
typedef struct Index Index;

/**
 * Query structure for sequence classification
 *
 * Requirements:
 * - seq: MUST be non-NULL and point to at least seq_len bytes
 * - seq_len: MUST be > 0 and <= 2GB (2,000,000,000 bytes)
 * - pair_seq: Either NULL (single-end) or non-NULL (paired-end)
 * - pair_len: MUST be 0 if pair_seq is NULL, > 0 and <= 2GB otherwise
 *
 * Memory Lifetime:
 * - All pointers must remain valid for the duration of rype_classify() call
 * - Sequences are NOT copied, so caller must ensure they remain valid
 */
typedef struct {
    int64_t id;           // User-defined query identifier
    const char* seq;      // Primary sequence (ACGT, case-insensitive)
    size_t seq_len;       // Length of seq in bytes
    const char* pair_seq; // Optional paired-end sequence (NULL if single-end)
    size_t pair_len;      // Length of pair_seq (0 if single-end)
} RypeQuery;

/**
 * Classification result for a single query
 */
typedef struct {
    int64_t query_id;    // Query ID from RypeQuery
    uint32_t bucket_id;  // Matched bucket/reference ID
    double score;        // Classification score (0.0 - 1.0)
} RypeHit;

/**
 * Array of classification results
 *
 * Memory: Owned by caller after rype_classify() returns
 * MUST be freed with rype_results_free() exactly once
 */
typedef struct {
    RypeHit* data;       // Array of hits
    size_t len;          // Number of hits
    size_t capacity;     // Capacity of data array
} RypeResultArray;

// --- API Functions ---

/**
 * Loads an index from disk
 *
 * @param path: Null-terminated UTF-8 file path to .ryp index file
 * @return: Non-NULL Index pointer on success, NULL on failure
 *
 * Errors (returns NULL):
 * - path is NULL
 * - File not found or cannot be opened
 * - File is corrupted or has wrong format
 * - Unsupported index version
 * - Out of memory
 * - File contains invalid data (triggers DoS protection limits)
 *
 * Thread Safety: NOT thread-safe
 * Memory: Returned Index must be freed with rype_index_free()
 *
 * Error Details: Call rype_get_last_error() for detailed error message
 */
Index* rype_index_load(const char* path);

/**
 * Frees an index
 *
 * @param index: Index pointer from rype_index_load() (NULL is safe)
 *
 * WARNING: Do NOT call this while any thread is using the Index in rype_classify()
 * Doing so will cause use-after-free and undefined behavior
 *
 * Thread Safety: NOT thread-safe (caller must ensure exclusive access)
 */
void rype_index_free(Index* index);

/**
 * Classifies a batch of sequences against the index
 *
 * @param index: Non-NULL Index pointer from rype_index_load()
 * @param queries: Array of RypeQuery structs (see requirements above)
 * @param num_queries: Number of queries (must be > 0 and < INTPTR_MAX)
 * @param threshold: Classification threshold, typically 0.0-1.0 (must be finite)
 * @return: Non-NULL RypeResultArray on success, NULL on failure
 *
 * Requirements:
 * - index must be non-NULL and valid
 * - queries must be non-NULL and all queries must satisfy RypeQuery requirements
 * - num_queries must be > 0
 * - threshold must be finite (not NaN or infinity)
 * - All query memory must remain valid during the call
 *
 * Errors (returns NULL):
 * - Any parameter is NULL
 * - num_queries is 0
 * - threshold is NaN or infinity
 * - Any query violates RypeQuery requirements
 * - Out of memory
 *
 * Thread Safety: Thread-safe (multiple threads can call with same Index)
 * Memory: Returned RypeResultArray MUST be freed with rype_results_free()
 *
 * Performance: Uses parallel processing internally
 *
 * Error Details: Call rype_get_last_error() for detailed error message
 */
RypeResultArray* rype_classify(
    const Index* index,
    const RypeQuery* queries,
    size_t num_queries,
    double threshold
);

/**
 * Frees a result array
 *
 * @param results: RypeResultArray pointer from rype_classify() (NULL is safe)
 *
 * WARNING: Do NOT call this twice on the same pointer (undefined behavior)
 * WARNING: Do NOT access results->data after calling this function
 *
 * Thread Safety: NOT thread-safe (one thread must free each result array)
 */
void rype_results_free(RypeResultArray* results);

/**
 * Gets the last error message from a failed API call
 *
 * @return: Null-terminated error string, or NULL if no error
 *
 * Notes:
 * - Error messages are thread-local (each thread has its own error)
 * - Error is cleared on successful API calls
 * - Pointer is valid until next API call on same thread
 * - Do NOT free the returned pointer
 *
 * Thread Safety: Thread-safe (returns thread-local error)
 *
 * Example:
 *   Index* idx = rype_index_load("missing.ryp");
 *   if (!idx) {
 *       const char* err = rype_get_last_error();
 *       fprintf(stderr, "Error: %s\n", err ? err : "unknown");
 *   }
 */
const char* rype_get_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // RYPE_H
