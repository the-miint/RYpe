#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Matches the Rust #[repr(C)] struct layout
typedef struct {
    int64_t id;
    const char* seq;
    size_t seq_len;
    const char* pair_seq; // Nullable
    size_t pair_len;      // 0 if null
} RypeQuery;

typedef struct {
    int64_t query_id;
    uint32_t bucket_id;
    double score;
} RypeHit;

typedef struct {
    RypeHit* data;
    size_t len;
    size_t capacity;
} RypeResultArray;

// Declare Rust functions
extern void* rype_index_load(const char* path);
extern void rype_index_free(void* index);
extern RypeResultArray* rype_classify(const void* index, const RypeQuery* queries, size_t num_queries, double threshold);
extern void rype_results_free(RypeResultArray* results);

int main() {
    // 1. Load the Index
    printf("Loading index...\n");
    void* index = rype_index_load("my_index.ryp");
    if (!index) {
        fprintf(stderr, "Failed to load index. Did you run 'rype index'?\n");
        return 1;
    }

    // 2. Prepare Query
    // MUST BE >= K (64). Here we use 80 chars of "AT".
    // This matches the reference "AT"*100 perfectly.
    const char* seq_str = "ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT";
    
    RypeQuery q;
    q.id = 101;
    q.seq = seq_str;
    q.seq_len = strlen(seq_str); // 80
    q.pair_seq = NULL;
    q.pair_len = 0;

    printf("Querying sequence (len=%lu): %.10s...\n", q.seq_len, q.seq);

    // 3. Run Classification
    RypeResultArray* results = rype_classify(index, &q, 1, 0.1); // Threshold 0.1

    // 4. Print Results
    if (results) {
        printf("Found %lu hits:\n", results->len);
        for (size_t i = 0; i < results->len; i++) {
            RypeHit h = results->data[i];
            printf("  - Query %ld matched Bucket %u with score %.4f\n", h.query_id, h.bucket_id, h.score);
        }
        rype_results_free(results);
    } else {
        printf("No results returned (Function failed).\n");
    }

    // 5. Cleanup
    rype_index_free(index);
    return 0;
}

