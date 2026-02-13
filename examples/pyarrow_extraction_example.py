#!/usr/bin/env python3
"""
Rype PyArrow Minimizer Extraction Example

Demonstrates calling rype's Arrow-based minimizer extraction functions
directly from Python using PyArrow's C Data Interface and ctypes.

This example:
  1. Loads librype.so via ctypes
  2. Creates a PyArrow table with id + sequence columns
  3. Calls rype_extract_minimizer_set_arrow() for sorted/deduped hash sets
  4. Calls rype_extract_strand_minimizers_arrow() for ordered hashes + positions
  5. Prints the results as PyArrow tables

Prerequisites:
  - pyarrow (tested with 23.0.0)
  - librype built with Arrow FFI support:
        cargo build --release --lib --features arrow-ffi

Usage:
  conda activate rype-pyarrow   # or any env with pyarrow
  python examples/pyarrow_extraction_example.py

Tested with:
  - PyArrow 23.0.0, Python 3.13.9
  - Debian GNU/Linux 13 (trixie), x86_64
  - February 2026

NOTE: This example is NOT part of the regular test suite or build
dependencies. PyArrow is not required to build or test rype. This file
exists to demonstrate and verify the Arrow C Data Interface interop.
"""

import ctypes
import os
import sys

import pyarrow as pa


def load_librype():
    """Load librype.so from target/release, built with --features arrow-ffi."""
    # Look relative to this script's location (examples/ -> target/release/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(script_dir, "..", "target", "release", "librype.so")
    lib_path = os.path.normpath(lib_path)

    if not os.path.exists(lib_path):
        print(f"ERROR: {lib_path} not found.", file=sys.stderr)
        print(
            "Build with: cargo build --release --lib --features arrow-ffi",
            file=sys.stderr,
        )
        sys.exit(1)

    lib = ctypes.CDLL(lib_path)

    # int rype_extract_minimizer_set_arrow(
    #     ArrowArrayStream* input, size_t k, size_t w, uint64_t salt,
    #     ArrowArrayStream* output)
    lib.rype_extract_minimizer_set_arrow.argtypes = [
        ctypes.c_void_p,  # input_stream ptr
        ctypes.c_size_t,  # k
        ctypes.c_size_t,  # w
        ctypes.c_uint64,  # salt
        ctypes.c_void_p,  # out_stream ptr
    ]
    lib.rype_extract_minimizer_set_arrow.restype = ctypes.c_int

    # int rype_extract_strand_minimizers_arrow(
    #     ArrowArrayStream* input, size_t k, size_t w, uint64_t salt,
    #     ArrowArrayStream* output)
    lib.rype_extract_strand_minimizers_arrow.argtypes = [
        ctypes.c_void_p,  # input_stream ptr
        ctypes.c_size_t,  # k
        ctypes.c_size_t,  # w
        ctypes.c_uint64,  # salt
        ctypes.c_void_p,  # out_stream ptr
    ]
    lib.rype_extract_strand_minimizers_arrow.restype = ctypes.c_int

    # const char* rype_get_last_error(void)
    lib.rype_get_last_error.argtypes = []
    lib.rype_get_last_error.restype = ctypes.c_char_p

    return lib


# ArrowArrayStream is 5 pointers: get_schema, get_next, get_last_error, release, private_data
ARROW_ARRAY_STREAM_SIZE = 5 * ctypes.sizeof(ctypes.c_void_p)


def alloc_arrow_stream():
    """Allocate a zeroed ArrowArrayStream buffer, return (buffer, pointer)."""
    buf = ctypes.create_string_buffer(ARROW_ARRAY_STREAM_SIZE)
    return buf, ctypes.cast(buf, ctypes.c_void_p).value


def call_extraction(lib, func, reader, k, w, salt):
    """
    Call a rype Arrow extraction function.

    Args:
        lib: ctypes CDLL handle to librype
        func: the ctypes function to call (e.g. lib.rype_extract_minimizer_set_arrow)
        reader: pyarrow.RecordBatchReader with input data
        k: k-mer size
        w: window size
        salt: XOR salt

    Returns:
        pyarrow.RecordBatchReader with extraction results
    """
    in_buf, in_ptr = alloc_arrow_stream()
    out_buf, out_ptr = alloc_arrow_stream()

    # Export PyArrow reader into the input ArrowArrayStream
    reader._export_to_c(in_ptr)

    # Call rype — it consumes in_ptr and populates out_ptr
    rc = func(in_ptr, k, w, salt, out_ptr)
    if rc != 0:
        err = lib.rype_get_last_error()
        msg = err.decode("utf-8") if err else "unknown error"
        raise RuntimeError(f"rype extraction failed: {msg}")

    # Import the output ArrowArrayStream back into PyArrow
    return pa.RecordBatchReader._import_from_c(out_ptr)


def make_input_reader(ids, sequences):
    """Create a PyArrow RecordBatchReader with id (int64) + sequence (binary) columns."""
    schema = pa.schema([("id", pa.int64()), ("sequence", pa.binary())])
    batch = pa.record_batch(
        [pa.array(ids, type=pa.int64()), pa.array(sequences, type=pa.binary())],
        schema=schema,
    )
    return pa.RecordBatchReader.from_batches(schema, [batch])


def main():
    lib = load_librype()

    # Test sequences — long enough for k=16, w=5 (need >= k+w-1 = 20 bases)
    sequences = [
        b"AAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCC",
        b"AAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCC",
        b"ACGT",  # too short for k=16 — should produce empty lists
    ]
    ids = [1, 2, 3]

    k, w, salt = 16, 5, 0

    print(f"PyArrow version: {pa.__version__}")
    print(f"Parameters: k={k}, w={w}, salt=0x{salt:016x}")
    print(f"Sequences: {len(sequences)}")
    for i, seq in enumerate(sequences):
        display = seq[:40].decode() + ("..." if len(seq) > 40 else "")
        print(f"  [{ids[i]}] len={len(seq)}: {display}")

    # --- Minimizer Set Extraction ---
    print("\n=== Minimizer Set Extraction ===")
    reader = make_input_reader(ids, sequences)
    result = call_extraction(lib, lib.rype_extract_minimizer_set_arrow, reader, k, w, salt)
    table = result.read_all()
    print(f"Schema: {table.schema}")
    print(f"Rows: {table.num_rows}")
    for i in range(table.num_rows):
        row_id = table.column("id")[i].as_py()
        fwd = table.column("fwd_set")[i].as_py()
        rc = table.column("rc_set")[i].as_py()
        print(f"  id={row_id}: fwd_set({len(fwd)} hashes), rc_set({len(rc)} hashes)")
        if fwd:
            print(f"    fwd[0:3] = {['0x%016x' % h for h in fwd[:3]]}")
        if rc:
            print(f"    rc[0:3]  = {['0x%016x' % h for h in rc[:3]]}")

    # --- Strand Minimizers Extraction ---
    print("\n=== Strand Minimizers Extraction ===")
    reader = make_input_reader(ids, sequences)
    result = call_extraction(lib, lib.rype_extract_strand_minimizers_arrow, reader, k, w, salt)
    table = result.read_all()
    print(f"Schema: {table.schema}")
    print(f"Rows: {table.num_rows}")
    for i in range(table.num_rows):
        row_id = table.column("id")[i].as_py()
        fwd_h = table.column("fwd_hashes")[i].as_py()
        fwd_p = table.column("fwd_positions")[i].as_py()
        rc_h = table.column("rc_hashes")[i].as_py()
        rc_p = table.column("rc_positions")[i].as_py()
        print(f"  id={row_id}: fwd({len(fwd_h)} minimizers), rc({len(rc_h)} minimizers)")
        if fwd_h:
            print(f"    fwd[0]: hash=0x{fwd_h[0]:016x}, pos={fwd_p[0]}")
        if rc_h:
            print(f"    rc[0]:  hash=0x{rc_h[0]:016x}, pos={rc_p[0]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
