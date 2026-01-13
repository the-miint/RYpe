/**
 * Rype Basic C API Example
 *
 * This example demonstrates the complete lifecycle of using the Rype C API:
 * 1. Loading an index
 * 2. Querying index metadata
 * 3. Classifying sequences
 * 4. Interpreting results
 * 5. Proper cleanup
 *
 * Build:
 *     gcc -o basic_example basic_example.c -L../target/release -lrype -Wl,-rpath,../target/release
 *
 * Or with debug build:
 *     gcc -o basic_example basic_example.c -L../target/debug -lrype -Wl,-rpath,../target/debug
 *
 * Run:
 *     ./basic_example path/to/index.ryidx
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../rype.h"

// Example DNA sequence (150bp, typical Illumina read length)
static const char* EXAMPLE_SEQUENCE =
    "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
    "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
    "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";

// Paired-end mate sequence
static const char* EXAMPLE_PAIR_SEQUENCE =
    "TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA"
    "TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA"
    "TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA";

/**
 * Print index metadata
 */
static void print_index_info(const RypeIndex* idx) {
    printf("Index Information:\n");
    printf("  K-mer size:    %zu\n", rype_index_k(idx));
    printf("  Window size:   %zu\n", rype_index_w(idx));
    printf("  Salt:          0x%016lx\n", (unsigned long)rype_index_salt(idx));
    printf("  Is inverted:   %s\n", rype_index_is_inverted(idx) ? "yes" : "no");
    printf("  Is sharded:    %s\n", rype_index_is_sharded(idx) ? "yes" : "no");
    printf("  Num shards:    %u\n", rype_index_num_shards(idx));

    int32_t num_buckets = rype_index_num_buckets(idx);
    if (num_buckets >= 0) {
        printf("  Num buckets:   %d\n", num_buckets);
    } else {
        printf("  Num buckets:   N/A (inverted index)\n");
    }
    printf("\n");
}

/**
 * Print classification results with bucket name lookup
 */
static void print_results(const RypeIndex* idx, const RypeResultArray* results) {
    if (results->len == 0) {
        printf("No hits above threshold.\n");
        return;
    }

    printf("Classification Results (%zu hits):\n", results->len);
    printf("  %-10s %-10s %-20s %s\n", "Query ID", "Bucket ID", "Bucket Name", "Score");
    printf("  %-10s %-10s %-20s %s\n", "--------", "---------", "-----------", "-----");

    for (size_t i = 0; i < results->len; i++) {
        const RypeHit* hit = &results->data[i];

        // Look up bucket name - handle NULL cases properly
        const char* name = rype_bucket_name(idx, hit->bucket_id);
        const char* display_name;

        if (name) {
            display_name = name;
        } else if (rype_index_is_inverted(idx)) {
            // Expected for inverted indices - they don't store names
            display_name = "(use main idx)";
        } else {
            // Unexpected - bucket should exist in main index
            display_name = "(unknown)";
        }

        printf("  %-10ld %-10u %-20s %.4f\n",
               (long)hit->query_id,
               hit->bucket_id,
               display_name,
               hit->score);
    }
    printf("\n");
}

/**
 * Demonstrate single-end classification
 */
static int classify_single_end(const RypeIndex* idx) {
    printf("=== Single-End Classification ===\n");

    RypeQuery query = {
        .id = 1,
        .seq = EXAMPLE_SEQUENCE,
        .seq_len = strlen(EXAMPLE_SEQUENCE),
        .pair_seq = NULL,
        .pair_len = 0
    };

    printf("Query: id=%ld, length=%zu bp\n", (long)query.id, query.seq_len);

    // Classify with threshold 0.05 (5% of minimizers must match)
    double threshold = 0.05;
    printf("Threshold: %.2f\n\n", threshold);

    RypeResultArray* results = rype_classify(idx, &query, 1, threshold);
    if (!results) {
        fprintf(stderr, "Classification failed: %s\n", rype_get_last_error());
        return -1;
    }

    print_results(idx, results);
    rype_results_free(results);
    return 0;
}

/**
 * Demonstrate paired-end classification
 */
static int classify_paired_end(const RypeIndex* idx) {
    printf("=== Paired-End Classification ===\n");

    RypeQuery query = {
        .id = 2,
        .seq = EXAMPLE_SEQUENCE,
        .seq_len = strlen(EXAMPLE_SEQUENCE),
        .pair_seq = EXAMPLE_PAIR_SEQUENCE,
        .pair_len = strlen(EXAMPLE_PAIR_SEQUENCE)
    };

    printf("Query: id=%ld, R1=%zu bp, R2=%zu bp\n",
           (long)query.id, query.seq_len, query.pair_len);

    double threshold = 0.05;
    printf("Threshold: %.2f\n\n", threshold);

    RypeResultArray* results = rype_classify(idx, &query, 1, threshold);
    if (!results) {
        fprintf(stderr, "Classification failed: %s\n", rype_get_last_error());
        return -1;
    }

    print_results(idx, results);
    rype_results_free(results);
    return 0;
}

/**
 * Demonstrate batch classification (multiple queries at once)
 */
static int classify_batch(const RypeIndex* idx) {
    printf("=== Batch Classification ===\n");

    // Create multiple queries
    RypeQuery queries[3] = {
        {
            .id = 100,
            .seq = EXAMPLE_SEQUENCE,
            .seq_len = strlen(EXAMPLE_SEQUENCE),
            .pair_seq = NULL,
            .pair_len = 0
        },
        {
            .id = 101,
            .seq = EXAMPLE_PAIR_SEQUENCE,
            .seq_len = strlen(EXAMPLE_PAIR_SEQUENCE),
            .pair_seq = NULL,
            .pair_len = 0
        },
        {
            .id = 102,
            .seq = EXAMPLE_SEQUENCE,
            .seq_len = strlen(EXAMPLE_SEQUENCE),
            .pair_seq = EXAMPLE_PAIR_SEQUENCE,
            .pair_len = strlen(EXAMPLE_PAIR_SEQUENCE)
        }
    };

    printf("Batch: 3 queries (IDs: 100, 101, 102)\n");

    double threshold = 0.05;
    printf("Threshold: %.2f\n\n", threshold);

    RypeResultArray* results = rype_classify(idx, queries, 3, threshold);
    if (!results) {
        fprintf(stderr, "Batch classification failed: %s\n", rype_get_last_error());
        return -1;
    }

    print_results(idx, results);
    rype_results_free(results);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <index_path>\n", argv[0]);
        fprintf(stderr, "\nExample:\n");
        fprintf(stderr, "  %s ../my_index.ryidx\n", argv[0]);
        return 1;
    }

    const char* index_path = argv[1];

    // Step 1: Load the index
    printf("Loading index from: %s\n\n", index_path);

    RypeIndex* idx = rype_index_load(index_path);
    if (!idx) {
        const char* err = rype_get_last_error();
        fprintf(stderr, "Failed to load index: %s\n", err ? err : "unknown error");
        return 1;
    }

    // Step 2: Print index metadata
    print_index_info(idx);

    // Step 3: Run classification examples
    int status = 0;

    if (classify_single_end(idx) != 0) {
        status = 1;
        goto cleanup;
    }

    if (classify_paired_end(idx) != 0) {
        status = 1;
        goto cleanup;
    }

    if (classify_batch(idx) != 0) {
        status = 1;
        goto cleanup;
    }

cleanup:
    // Step 4: Free the index
    // IMPORTANT: Do NOT free while classification is in progress (use-after-free)
    rype_index_free(idx);

    printf("Done.\n");
    return status;
}
