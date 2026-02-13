/**
 * Rype Arrow Minimizer Extraction Example
 *
 * Demonstrates the Arrow-based minimizer extraction API using Arrow GLib.
 * This example:
 *   1. Builds an Arrow RecordBatch with id + sequence columns
 *   2. Exports it as an ArrowArrayStream (C Data Interface)
 *   3. Calls rype_extract_minimizer_set_arrow() for set extraction
 *   4. Calls rype_extract_strand_minimizers_arrow() for strand extraction
 *   5. Imports the output streams and prints the results
 *
 * Prerequisites:
 *   - libarrow-glib-dev (Arrow GLib development headers and libraries)
 *   - librype (built from this repository)
 *
 * Tested with:
 *   - Arrow GLib 23.0.0
 *   - GCC 14.2.0 (Debian 14.2.0-19)
 *   - Debian GNU/Linux 13 (trixie), x86_64
 *   - February 2026
 *
 * This example is NOT part of the regular test suite. It requires Arrow GLib
 * which is not a build dependency of this project.
 *
 * Build librype with Arrow FFI support:
 *   cargo build --release --lib --features arrow-ffi
 *
 * Build this example (from the examples/ directory):
 *   gcc -o arrow_extraction_example arrow_extraction_example.c \
 *       $(pkg-config --cflags arrow-glib) \
 *       -L../target/release -lrype -Wl,-rpath,../target/release \
 *       $(pkg-config --libs arrow-glib)
 *
 * Run:
 *   ./arrow_extraction_example
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <arrow-glib/arrow-glib.h>

/* We only need the Arrow FFI declarations from rype.h */
#define RYPE_ARROW
#include "../rype.h"

/* Test sequences */
static const char *SEQUENCES[] = {
    "AAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCC",
    "AAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCC",
};
static const int NUM_SEQUENCES = 2;

/**
 * Build an input RecordBatch with id (Int64) + sequence (Binary) columns.
 * Returns NULL on error.
 */
static GArrowRecordBatch *
build_input_batch(GError **error)
{
    /* Build id column */
    GArrowInt64ArrayBuilder *id_builder = garrow_int64_array_builder_new();
    for (int i = 0; i < NUM_SEQUENCES; i++) {
        if (!garrow_int64_array_builder_append_value(id_builder, i + 1, error)) {
            g_object_unref(id_builder);
            return NULL;
        }
    }
    GArrowArray *id_array =
        garrow_array_builder_finish(GARROW_ARRAY_BUILDER(id_builder), error);
    g_object_unref(id_builder);
    if (!id_array)
        return NULL;

    /* Build sequence column */
    GArrowBinaryArrayBuilder *seq_builder = garrow_binary_array_builder_new();
    for (int i = 0; i < NUM_SEQUENCES; i++) {
        gint32 len = (gint32)strlen(SEQUENCES[i]);
        gboolean ok = garrow_binary_array_builder_append_value(
            seq_builder, (const guint8 *)SEQUENCES[i], len, error);
        if (!ok) {
            g_object_unref(seq_builder);
            g_object_unref(id_array);
            return NULL;
        }
    }
    GArrowArray *seq_array =
        garrow_array_builder_finish(GARROW_ARRAY_BUILDER(seq_builder), error);
    g_object_unref(seq_builder);
    if (!seq_array) {
        g_object_unref(id_array);
        return NULL;
    }

    /* Build schema: id (Int64), sequence (Binary) */
    GArrowField *id_field =
        garrow_field_new("id", GARROW_DATA_TYPE(garrow_int64_data_type_new()));
    GArrowField *seq_field =
        garrow_field_new("sequence", GARROW_DATA_TYPE(garrow_binary_data_type_new()));
    GList *fields = NULL;
    fields = g_list_append(fields, id_field);
    fields = g_list_append(fields, seq_field);
    GArrowSchema *schema = garrow_schema_new(fields);
    g_list_free(fields);
    g_object_unref(id_field);
    g_object_unref(seq_field);

    /* Build RecordBatch */
    GList *columns = NULL;
    columns = g_list_append(columns, id_array);
    columns = g_list_append(columns, seq_array);
    GArrowRecordBatch *batch =
        garrow_record_batch_new(schema, NUM_SEQUENCES, columns, error);
    g_list_free(columns);
    g_object_unref(id_array);
    g_object_unref(seq_array);
    g_object_unref(schema);

    return batch;
}

/**
 * Export a RecordBatch as an ArrowArrayStream via Arrow GLib.
 *
 * Wraps the batch in a RecordBatchReader, then exports it.
 * The returned pointer is heap-allocated and must be freed by the caller
 * (after the stream is consumed/released).
 */
static struct ArrowArrayStream *
export_batch_as_stream(GArrowRecordBatch *batch, GError **error)
{
    GArrowSchema *schema =
        garrow_record_batch_get_schema(batch);

    GList *batches = NULL;
    batches = g_list_append(batches, batch);

    GArrowRecordBatchReader *reader =
        garrow_record_batch_reader_new(batches, schema, error);
    g_list_free(batches);
    g_object_unref(schema);
    if (!reader)
        return NULL;

    gpointer c_stream = garrow_record_batch_reader_export(reader, error);
    g_object_unref(reader);

    return (struct ArrowArrayStream *)c_stream;
}

/**
 * Print the contents of a UInt64 list array column for all rows.
 */
static void
print_uint64_list_column(GArrowRecordBatch *batch, guint col_idx, const char *label)
{
    GArrowArray *array = garrow_record_batch_get_column_data(batch, col_idx);
    GArrowListArray *list_array = GARROW_LIST_ARRAY(array);
    gint64 n_rows = garrow_array_get_length(array);

    for (gint64 row = 0; row < n_rows; row++) {
        GArrowArray *values = garrow_list_array_get_value(list_array, row);
        GArrowUInt64Array *u64_values = GARROW_UINT64_ARRAY(values);
        gint64 len = garrow_array_get_length(values);

        g_print("  row %ld %s (%ld values):", (long)row, label, (long)len);
        gint64 print_count = len < 5 ? len : 5;
        for (gint64 i = 0; i < print_count; i++) {
            g_print(" 0x%016lx", (unsigned long)garrow_uint64_array_get_value(u64_values, i));
        }
        if (len > print_count) {
            g_print(" ...");
        }
        g_print("\n");
        g_object_unref(values);
    }
    g_object_unref(array);
}

/**
 * Read all batches from an output ArrowArrayStream and print results.
 *
 * Takes ownership of out_stream (imports it into Arrow GLib).
 */
static int
read_and_print_results(struct ArrowArrayStream *out_stream, const char *title)
{
    GError *error = NULL;

    GArrowRecordBatchReader *reader =
        garrow_record_batch_reader_import(out_stream, &error);
    if (!reader) {
        g_printerr("Failed to import output stream: %s\n", error->message);
        g_error_free(error);
        return -1;
    }

    /* Print schema */
    GArrowSchema *schema = garrow_record_batch_reader_get_schema(reader);
    gchar *schema_str = garrow_schema_to_string(schema);
    g_print("\n=== %s ===\n", title);
    g_print("Schema:\n%s\n", schema_str);
    g_free(schema_str);
    g_object_unref(schema);

    /* Read all batches */
    int total_rows = 0;
    while (TRUE) {
        GArrowRecordBatch *batch =
            garrow_record_batch_reader_read_next(reader, &error);
        if (error) {
            g_printerr("Failed to read batch: %s\n", error->message);
            g_error_free(error);
            g_object_unref(reader);
            return -1;
        }
        if (!batch)
            break;

        gint64 n_rows = garrow_record_batch_get_n_rows(batch);
        guint n_cols = garrow_record_batch_get_n_columns(batch);
        g_print("\nBatch: %ld rows, %u columns\n", (long)n_rows, n_cols);

        /* Print each List<UInt64> column (skip column 0 which is the id) */
        for (guint col = 1; col < n_cols; col++) {
            const gchar *name = garrow_record_batch_get_column_name(batch, col);
            print_uint64_list_column(batch, col, name);
        }

        total_rows += (int)n_rows;
        g_object_unref(batch);
    }

    g_print("Total rows: %d\n", total_rows);
    g_object_unref(reader);
    return 0;
}

int
main(void)
{
    GError *error = NULL;

    g_print("Rype Arrow Extraction Example\n");
    g_print("Sequences: %d\n", NUM_SEQUENCES);
    for (int i = 0; i < NUM_SEQUENCES; i++) {
        g_print("  [%d] len=%zu: %.40s%s\n", i + 1, strlen(SEQUENCES[i]),
                SEQUENCES[i], strlen(SEQUENCES[i]) > 40 ? "..." : "");
    }

    size_t k = 16;
    size_t w = 5;
    uint64_t salt = 0;
    g_print("Parameters: k=%zu, w=%zu, salt=0x%016lx\n", k, w, (unsigned long)salt);

    /* ===== Minimizer Set Extraction ===== */
    {
        GArrowRecordBatch *input_batch = build_input_batch(&error);
        if (!input_batch) {
            g_printerr("Failed to build input batch: %s\n", error->message);
            g_error_free(error);
            return EXIT_FAILURE;
        }

        struct ArrowArrayStream *input_stream =
            export_batch_as_stream(input_batch, &error);
        g_object_unref(input_batch);
        if (!input_stream) {
            g_printerr("Failed to export input stream: %s\n", error->message);
            g_error_free(error);
            return EXIT_FAILURE;
        }

        /* Allocate output stream on the stack */
        struct ArrowArrayStream out_stream;
        memset(&out_stream, 0, sizeof(out_stream));

        /* rype takes ownership of input_stream */
        int rc = rype_extract_minimizer_set_arrow(input_stream, k, w, salt, &out_stream);
        if (rc != 0) {
            g_printerr("rype_extract_minimizer_set_arrow failed: %s\n",
                        rype_get_last_error());
            return EXIT_FAILURE;
        }

        if (read_and_print_results(&out_stream, "Minimizer Set Extraction") != 0) {
            return EXIT_FAILURE;
        }
    }

    /* ===== Strand Minimizers Extraction ===== */
    {
        GArrowRecordBatch *input_batch = build_input_batch(&error);
        if (!input_batch) {
            g_printerr("Failed to build input batch: %s\n", error->message);
            g_error_free(error);
            return EXIT_FAILURE;
        }

        struct ArrowArrayStream *input_stream =
            export_batch_as_stream(input_batch, &error);
        g_object_unref(input_batch);
        if (!input_stream) {
            g_printerr("Failed to export input stream: %s\n", error->message);
            g_error_free(error);
            return EXIT_FAILURE;
        }

        struct ArrowArrayStream out_stream;
        memset(&out_stream, 0, sizeof(out_stream));

        int rc = rype_extract_strand_minimizers_arrow(input_stream, k, w, salt, &out_stream);
        if (rc != 0) {
            g_printerr("rype_extract_strand_minimizers_arrow failed: %s\n",
                        rype_get_last_error());
            return EXIT_FAILURE;
        }

        if (read_and_print_results(&out_stream, "Strand Minimizers Extraction") != 0) {
            return EXIT_FAILURE;
        }
    }

    g_print("\nDone.\n");
    return EXIT_SUCCESS;
}
