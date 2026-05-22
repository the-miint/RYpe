/**
 * Rype Cluster C API Example
 *
 * Demonstrates contig-level dereplication via the C API:
 *   1. Build a small set of contigs in memory
 *   2. Call rype_cluster() with strain-level-ish defaults
 *   3. Walk the result rows, look up rep / member / source_mag
 *   4. Free the result
 *
 * The example uses three contigs:
 *   - A: full pseudo-random genome (20kb)
 *   - B: middle fragment of A (8kb)
 *   - C: independent pseudo-random sequence (20kb)
 *
 * Expected output: A becomes a representative and absorbs B; C is its own
 * representative.
 *
 * Build:
 *     gcc -o cluster_example cluster_example.c -L../target/release -lrype -Wl,-rpath,../target/release
 *
 * Run:
 *     ./cluster_example
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../rype.h"

/* Deterministic LCG-based pseudo-random DNA generator. */
static void seed_seq(unsigned char* buf, size_t len, unsigned long long seed) {
    static const char bases[4] = {'A', 'C', 'G', 'T'};
    unsigned long long s = seed * 0x9E3779B97F4A7C15ULL;
    for (size_t i = 0; i < len; ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        buf[i] = (unsigned char)bases[(s >> 56) & 3ULL];
    }
}

static int run_cluster_example(void) {
    const size_t n = 20000;
    unsigned char* a = (unsigned char*)malloc(n);
    unsigned char* c = (unsigned char*)malloc(n);
    if (!a || !c) {
        fprintf(stderr, "out of memory\n");
        free(a);
        free(c);
        return 1;
    }
    seed_seq(a, n, 1);
    seed_seq(c, n, 2);

    /* B is the middle 8kb of A. Borrow it from A's buffer with offset. */
    const unsigned char* b = a + 2000;
    const size_t b_len = 8000;

    RypeClusterInput inputs[3] = {
        {
            .id = "A",
            .source_mag = "mag1",
            .sequence = (const char*)a,
            .sequence_len = n,
        },
        {
            .id = "B",
            .source_mag = "mag2",
            .sequence = (const char*)b,
            .sequence_len = b_len,
        },
        {
            .id = "C",
            .source_mag = NULL, /* C has no source MAG */
            .sequence = (const char*)c,
            .sequence_len = n,
        },
    };

    RypeClusterConfig cfg = {
        .k = 32,
        .w = 20,
        .salt = 0x5555555555555555ULL,
        .min_length = 1000,
        .threshold = 0.80,
        .min_shared = 50,
    };

    /* Pass NULL for the chain config to keep chain disabled (Plan 1.4
     * sidecar — new callers populate a RypeChainConfig to enable
     * positional chain confirmation). */
    RypeClusterRowArray* result = rype_cluster(inputs, 3, &cfg, NULL);
    if (!result) {
        const char* err = rype_get_last_error();
        fprintf(stderr, "rype_cluster failed: %s\n", err ? err : "unknown");
        free(a);
        free(c);
        return 1;
    }

    printf("Cluster result (%zu rows):\n", result->len);
    printf("  %-12s %-12s %-12s %s\n", "rep_contig", "member", "source_mag", "containment");
    printf("  %-12s %-12s %-12s %s\n", "----------", "------", "----------", "-----------");
    for (size_t i = 0; i < result->len; ++i) {
        const RypeClusterRow* r = &result->data[i];
        printf("  %-12s %-12s %-12s %.4f\n",
               r->rep_contig,
               r->member_contig,
               r->source_mag ? r->source_mag : "(none)",
               r->containment);
    }

    rype_cluster_results_free(result);
    free(a);
    free(c);
    return 0;
}

int main(void) {
    printf("=== Rype Cluster Example ===\n\n");
    int rc = run_cluster_example();
    if (rc == 0) {
        printf("\nDone.\n");
    }
    return rc;
}
