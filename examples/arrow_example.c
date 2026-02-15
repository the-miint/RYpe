/**
 * Rype Arrow C Data Interface Example
 *
 * This example demonstrates using the Arrow streaming API for high-throughput
 * classification. The Arrow C Data Interface enables zero-copy data exchange
 * with Arrow-compatible systems (PyArrow, DuckDB, Polars, etc.).
 *
 * Key concepts demonstrated:
 * 1. Creating Arrow streams from C data
 * 2. Understanding ownership semantics (CRITICAL for avoiding double-free)
 * 3. Consuming output streams correctly (schema lifetime)
 * 4. Proper cleanup
 *
 * Build (requires Arrow FFI feature):
 *     cargo build --release --lib --features arrow-ffi
 *     gcc -o arrow_example arrow_example.c \
 *         -L../target/release -lrype -Wl,-rpath,../target/release
 *
 * Run:
 *     ./arrow_example path/to/index.ryxdi
 *
 * NOTE: This is a simplified example showing the API flow. In practice, you
 * would typically receive Arrow streams from PyArrow, DuckDB, or similar
 * rather than constructing them manually in C.
 */

#ifndef RYPE_ARROW
#define RYPE_ARROW
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../rype.h"

// ============================================================================
// MINIMAL ARROW STREAM IMPLEMENTATION (for demonstration)
// ============================================================================
//
// In real usage, you would receive ArrowArrayStream from:
// - PyArrow: pyarrow.RecordBatchReader._export_to_c()
// - DuckDB: result.fetch_arrow_stream()
// - Polars: df.to_arrow()._export_to_c()
// - etc.
//
// This example constructs a minimal stream to show the API flow.

// Example sequences
static const char* SEQ1 = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
static const char* SEQ2 = "TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA";

// Private data for our example stream
typedef struct {
    int64_t* ids;
    const char** sequences;
    size_t* seq_lengths;
    size_t num_rows;
    size_t batch_index;  // 0 = not yet returned, 1 = returned
    int schema_exported;
} ExampleStreamData;

// Schema release callback
static void release_schema(struct ArrowSchema* schema) {
    if (schema->release == NULL) return;

    // Free children
    if (schema->children) {
        for (int64_t i = 0; i < schema->n_children; i++) {
            if (schema->children[i] && schema->children[i]->release) {
                schema->children[i]->release(schema->children[i]);
            }
            free(schema->children[i]);
        }
        free(schema->children);
    }

    // Free format strings (we allocated them)
    free((void*)schema->format);
    free((void*)schema->name);

    schema->release = NULL;
}

// Array release callback
static void release_array(struct ArrowArray* array) {
    if (array->release == NULL) return;

    // Free buffers
    if (array->buffers) {
        for (int64_t i = 0; i < array->n_buffers; i++) {
            free((void*)array->buffers[i]);
        }
        free(array->buffers);
    }

    // Free children
    if (array->children) {
        for (int64_t i = 0; i < array->n_children; i++) {
            if (array->children[i] && array->children[i]->release) {
                array->children[i]->release(array->children[i]);
            }
            free(array->children[i]);
        }
        free(array->children);
    }

    array->release = NULL;
}

// Create a child schema
static struct ArrowSchema* make_child_schema(const char* name, const char* format) {
    struct ArrowSchema* child = calloc(1, sizeof(struct ArrowSchema));
    child->format = strdup(format);
    child->name = strdup(name);
    child->flags = 0;
    child->n_children = 0;
    child->children = NULL;
    child->dictionary = NULL;
    child->release = release_schema;
    child->private_data = NULL;
    return child;
}

// get_schema implementation
static int example_get_schema(struct ArrowArrayStream* stream, struct ArrowSchema* out) {
    ExampleStreamData* data = (ExampleStreamData*)stream->private_data;

    // Schema: struct with fields {id: int64, sequence: binary}
    out->format = strdup("+s");  // struct
    out->name = strdup("");
    out->metadata = NULL;
    out->flags = 0;
    out->n_children = 2;
    out->children = malloc(2 * sizeof(struct ArrowSchema*));
    out->children[0] = make_child_schema("id", "l");        // int64
    out->children[1] = make_child_schema("sequence", "z");  // binary
    out->dictionary = NULL;
    out->release = release_schema;
    out->private_data = NULL;

    data->schema_exported = 1;
    return 0;
}

// Create int64 array
static struct ArrowArray* make_int64_array(int64_t* values, size_t n) {
    struct ArrowArray* arr = calloc(1, sizeof(struct ArrowArray));
    arr->length = n;
    arr->null_count = 0;
    arr->offset = 0;
    arr->n_buffers = 2;
    arr->buffers = malloc(2 * sizeof(void*));
    arr->buffers[0] = NULL;  // validity bitmap (none, all valid)

    int64_t* data_buf = malloc(n * sizeof(int64_t));
    memcpy(data_buf, values, n * sizeof(int64_t));
    arr->buffers[1] = data_buf;

    arr->n_children = 0;
    arr->children = NULL;
    arr->dictionary = NULL;
    arr->release = release_array;
    arr->private_data = NULL;
    return arr;
}

// Create binary array
static struct ArrowArray* make_binary_array(const char** strings, size_t* lengths, size_t n) {
    struct ArrowArray* arr = calloc(1, sizeof(struct ArrowArray));
    arr->length = n;
    arr->null_count = 0;
    arr->offset = 0;
    arr->n_buffers = 3;
    arr->buffers = malloc(3 * sizeof(void*));
    arr->buffers[0] = NULL;  // validity bitmap

    // Offsets buffer
    int32_t* offsets = malloc((n + 1) * sizeof(int32_t));
    offsets[0] = 0;
    size_t total_len = 0;
    for (size_t i = 0; i < n; i++) {
        total_len += lengths[i];
        offsets[i + 1] = (int32_t)total_len;
    }
    arr->buffers[1] = offsets;

    // Data buffer
    char* data_buf = malloc(total_len);
    size_t pos = 0;
    for (size_t i = 0; i < n; i++) {
        memcpy(data_buf + pos, strings[i], lengths[i]);
        pos += lengths[i];
    }
    arr->buffers[2] = data_buf;

    arr->n_children = 0;
    arr->children = NULL;
    arr->dictionary = NULL;
    arr->release = release_array;
    arr->private_data = NULL;
    return arr;
}

// get_next implementation
static int example_get_next(struct ArrowArrayStream* stream, struct ArrowArray* out) {
    ExampleStreamData* data = (ExampleStreamData*)stream->private_data;

    if (data->batch_index > 0) {
        // No more batches - signal end of stream
        memset(out, 0, sizeof(struct ArrowArray));
        out->release = NULL;
        return 0;
    }

    // Create the batch (struct array with children)
    out->length = data->num_rows;
    out->null_count = 0;
    out->offset = 0;
    out->n_buffers = 1;
    out->buffers = malloc(sizeof(void*));
    out->buffers[0] = NULL;  // struct has no validity for now

    out->n_children = 2;
    out->children = malloc(2 * sizeof(struct ArrowArray*));
    out->children[0] = make_int64_array(data->ids, data->num_rows);
    out->children[1] = make_binary_array(data->sequences, data->seq_lengths, data->num_rows);

    out->dictionary = NULL;
    out->release = release_array;
    out->private_data = NULL;

    data->batch_index = 1;
    return 0;
}

// get_last_error implementation
static const char* example_get_last_error(struct ArrowArrayStream* stream) {
    (void)stream;
    return NULL;
}

// Stream release callback
static void release_stream(struct ArrowArrayStream* stream) {
    if (stream->release == NULL) return;

    ExampleStreamData* data = (ExampleStreamData*)stream->private_data;
    if (data) {
        free(data->ids);
        free(data->seq_lengths);
        free((void*)data->sequences);
        free(data);
    }

    stream->release = NULL;
}

// Create an example input stream
static void create_example_stream(struct ArrowArrayStream* stream) {
    ExampleStreamData* data = calloc(1, sizeof(ExampleStreamData));
    data->num_rows = 2;
    data->batch_index = 0;
    data->schema_exported = 0;

    data->ids = malloc(2 * sizeof(int64_t));
    data->ids[0] = 1;
    data->ids[1] = 2;

    data->sequences = malloc(2 * sizeof(char*));
    data->sequences[0] = SEQ1;
    data->sequences[1] = SEQ2;

    data->seq_lengths = malloc(2 * sizeof(size_t));
    data->seq_lengths[0] = strlen(SEQ1);
    data->seq_lengths[1] = strlen(SEQ2);

    stream->get_schema = example_get_schema;
    stream->get_next = example_get_next;
    stream->get_last_error = example_get_last_error;
    stream->release = release_stream;
    stream->private_data = data;
}

// ============================================================================
// MAIN EXAMPLE
// ============================================================================

/**
 * Consume an output stream following proper lifetime rules
 */
static int consume_output_stream(struct ArrowArrayStream* output_stream) {
    printf("\n=== Consuming Output Stream ===\n");

    // Step 1: Get schema FIRST and keep it alive during iteration
    struct ArrowSchema schema;
    memset(&schema, 0, sizeof(schema));

    int err = output_stream->get_schema(output_stream, &schema);
    if (err != 0) {
        fprintf(stderr, "Failed to get schema: %s\n",
                output_stream->get_last_error(output_stream));
        return -1;
    }

    printf("Output schema: %ld fields\n", (long)schema.n_children);
    for (int64_t i = 0; i < schema.n_children; i++) {
        printf("  - %s: %s\n", schema.children[i]->name, schema.children[i]->format);
    }

    // Step 2: Iterate batches (schema MUST stay alive during this)
    int batch_count = 0;
    int total_rows = 0;

    while (1) {
        struct ArrowArray batch;
        memset(&batch, 0, sizeof(batch));

        err = output_stream->get_next(output_stream, &batch);
        if (err != 0) {
            fprintf(stderr, "Error getting next batch: %s\n",
                    output_stream->get_last_error(output_stream));
            schema.release(&schema);
            return -1;
        }

        // Check for end of stream (release is NULL)
        if (batch.release == NULL) {
            printf("End of stream.\n");
            break;
        }

        batch_count++;
        total_rows += batch.length;
        printf("Batch %d: %ld rows\n", batch_count, (long)batch.length);

        // Access the data (query_id, bucket_id, score)
        if (batch.n_children >= 3 && batch.length > 0) {
            // Children: query_id (int64), bucket_id (uint32), score (float64)
            struct ArrowArray* query_ids = batch.children[0];
            struct ArrowArray* bucket_ids = batch.children[1];
            struct ArrowArray* scores = batch.children[2];

            int64_t* qid_data = (int64_t*)query_ids->buffers[1];
            uint32_t* bid_data = (uint32_t*)bucket_ids->buffers[1];
            double* score_data = (double*)scores->buffers[1];

            for (int64_t i = 0; i < batch.length; i++) {
                printf("  Hit: query_id=%ld, bucket_id=%u, score=%.4f\n",
                       (long)qid_data[i], bid_data[i], score_data[i]);
            }
        }

        // Release this batch
        batch.release(&batch);
    }

    printf("Total: %d batches, %d rows\n", batch_count, total_rows);

    // Step 3: Release schema AFTER all batches consumed
    // CRITICAL: This order prevents use-after-free
    schema.release(&schema);

    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <index_path>\n", argv[0]);
        fprintf(stderr, "\nThis example requires building rype with Arrow support:\n");
        fprintf(stderr, "  cargo build --release --features arrow\n");
        return 1;
    }

    const char* index_path = argv[1];

    // Step 1: Load index
    printf("Loading index from: %s\n", index_path);

    RypeIndex* idx = rype_index_load(index_path);
    if (!idx) {
        fprintf(stderr, "Failed to load index: %s\n", rype_get_last_error());
        return 1;
    }

    printf("Index loaded: k=%zu, w=%zu, shards=%u\n",
           rype_index_k(idx), rype_index_w(idx),
           rype_index_num_shards(idx));

    // Step 2: Create input stream
    // In practice, you'd receive this from PyArrow, DuckDB, etc.
    printf("\nCreating input stream with 2 sequences...\n");

    struct ArrowArrayStream input_stream;
    create_example_stream(&input_stream);

    // Step 3: Prepare output stream (caller-allocated, function initializes)
    struct ArrowArrayStream output_stream;
    memset(&output_stream, 0, sizeof(output_stream));

    // Step 4: Call rype_classify_arrow
    //
    // OWNERSHIP SEMANTICS (CRITICAL):
    // - input_stream is CONSUMED by this function
    // - After return, input_stream->release has been called internally
    // - DO NOT call input_stream.release(&input_stream) again!
    //
    printf("\nClassifying (threshold=0.05)...\n");

    int result = rype_classify_arrow(
        idx,
        NULL,           // no negative filtering
        &input_stream,  // CONSUMED - do not release after!
        0.05,           // threshold
        &output_stream
    );

    // DO NOT DO THIS - causes double-free:
    // input_stream.release(&input_stream);  // WRONG!

    if (result != 0) {
        fprintf(stderr, "Classification failed: %s\n", rype_get_last_error());
        rype_index_free(idx);
        return 1;
    }

    // Step 5: Consume output stream
    int consume_result = consume_output_stream(&output_stream);

    // Step 6: Release output stream (we own it)
    // Only release if function succeeded (result == 0)
    if (output_stream.release) {
        output_stream.release(&output_stream);
    }

    // Step 7: Cleanup index
    rype_index_free(idx);

    printf("\nDone.\n");
    return consume_result;
}
