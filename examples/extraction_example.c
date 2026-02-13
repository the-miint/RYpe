/**
 * Rype Minimizer Extraction C API Example
 *
 * This example demonstrates the minimizer extraction API, which extracts
 * minimizer hashes and positions from raw DNA sequences without requiring
 * an index.
 *
 * Build:
 *     gcc -o extraction_example extraction_example.c -L../target/release -lrype -Wl,-rpath,../target/release
 *
 * Or with debug build:
 *     gcc -o extraction_example extraction_example.c -L../target/debug -lrype -Wl,-rpath,../target/debug
 *
 * Run:
 *     ./extraction_example
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../rype.h"

/* 67-byte test sequence */
static const char* TEST_SEQ =
    "AAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCC";

int main(void) {
    const uint8_t* seq = (const uint8_t*)TEST_SEQ;
    size_t seq_len = strlen(TEST_SEQ);
    size_t k = 16;
    size_t w = 5;
    uint64_t salt = 0;

    /* ===== Test 1: extract_minimizer_set ===== */
    {
        RypeMinimizerSetResult* result = rype_extract_minimizer_set(seq, seq_len, k, w, salt);
        if (!result) {
            fprintf(stderr, "extract_minimizer_set failed: %s\n",
                    rype_get_last_error());
            return 1;
        }

        printf("MINIMIZER_SET_OK=1\n");
        printf("FWD_SET_LEN=%zu\n", result->forward.len);
        printf("RC_SET_LEN=%zu\n", result->reverse_complement.len);

        /* Check forward is sorted */
        int fwd_sorted = 1;
        for (size_t i = 1; i < result->forward.len; i++) {
            if (result->forward.data[i - 1] >= result->forward.data[i]) {
                fwd_sorted = 0;
                break;
            }
        }
        printf("FWD_SET_SORTED=%d\n", fwd_sorted);

        /* Check RC is sorted */
        int rc_sorted = 1;
        for (size_t i = 1; i < result->reverse_complement.len; i++) {
            if (result->reverse_complement.data[i - 1] >= result->reverse_complement.data[i]) {
                rc_sorted = 0;
                break;
            }
        }
        printf("RC_SET_SORTED=%d\n", rc_sorted);

        rype_minimizer_set_result_free(result);
    }

    /* ===== Test 2: extract_strand_minimizers ===== */
    {
        RypeStrandMinimizersResult* result = rype_extract_strand_minimizers(seq, seq_len, k, w, salt);
        if (!result) {
            fprintf(stderr, "extract_strand_minimizers failed: %s\n",
                    rype_get_last_error());
            return 1;
        }

        printf("STRAND_MINIMIZERS_OK=1\n");
        printf("FWD_STRAND_LEN=%zu\n", result->forward.len);
        printf("RC_STRAND_LEN=%zu\n", result->reverse_complement.len);

        /* Check forward positions are non-decreasing */
        int fwd_ordered = 1;
        for (size_t i = 1; i < result->forward.len; i++) {
            if (result->forward.positions[i - 1] > result->forward.positions[i]) {
                fwd_ordered = 0;
                break;
            }
        }
        printf("FWD_POSITIONS_ORDERED=%d\n", fwd_ordered);

        /* Check forward positions are in bounds */
        int fwd_inbounds = 1;
        for (size_t i = 0; i < result->forward.len; i++) {
            if (result->forward.positions[i] + k > seq_len) {
                fwd_inbounds = 0;
                break;
            }
        }
        printf("FWD_POSITIONS_INBOUNDS=%d\n", fwd_inbounds);

        /* Check RC positions are non-decreasing */
        int rc_ordered = 1;
        for (size_t i = 1; i < result->reverse_complement.len; i++) {
            if (result->reverse_complement.positions[i - 1] > result->reverse_complement.positions[i]) {
                rc_ordered = 0;
                break;
            }
        }
        printf("RC_POSITIONS_ORDERED=%d\n", rc_ordered);

        /* Check RC positions are in bounds */
        int rc_inbounds = 1;
        for (size_t i = 0; i < result->reverse_complement.len; i++) {
            if (result->reverse_complement.positions[i] + k > seq_len) {
                rc_inbounds = 0;
                break;
            }
        }
        printf("RC_POSITIONS_INBOUNDS=%d\n", rc_inbounds);

        rype_strand_minimizers_result_free(result);
    }

    return 0;
}
