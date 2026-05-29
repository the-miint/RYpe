#!/usr/bin/env python3
"""
Rype PyArrow Index Building Example

Demonstrates building a .ryxdi index from streaming Arrow data via rype's
`rype_index_build_from_arrow` FFI entry point, using PyArrow's C Data Interface
and ctypes. This mirrors a DuckDB/DuckLake pipeline that stores genome sequences
as fixed-size chunks (here ~64 bytes for the demo) keyed by feature_idx, plus a
small feature -> bucket_name mapping.

This example:
  1. Loads librype.so via ctypes
  2. Builds a chunk RecordBatchReader (feature_idx, chunk_index, chunk_data)
  3. Builds a mapping RecordBatchReader (feature_idx, bucket_name)
  4. Exports both to ArrowArrayStreams and calls rype_index_build_from_arrow()
  5. Reports success; the .ryxdi directory can then be used with `rype classify`

Prerequisites:
  - pyarrow
  - librype built with Arrow FFI support:
        cargo build --release --lib --features arrow-ffi

Usage:
  python examples/pyarrow_build_example.py [output.ryxdi]

NOTE: This example is NOT part of the regular test suite or build dependencies.
PyArrow is not required to build or test rype. It exists to demonstrate and
verify the Arrow C Data Interface interop for index building.
"""

import ctypes
import os
import sys

import pyarrow as pa


def load_librype():
    """Load librype.so from target/release, built with --features arrow-ffi."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.normpath(
        os.path.join(script_dir, "..", "target", "release", "librype.so")
    )
    if not os.path.exists(lib_path):
        print(f"ERROR: {lib_path} not found.", file=sys.stderr)
        print(
            "Build with: cargo build --release --lib --features arrow-ffi",
            file=sys.stderr,
        )
        sys.exit(1)

    lib = ctypes.CDLL(lib_path)

    # int rype_index_build_from_arrow(
    #     const char* output_path,
    #     ArrowArrayStream* chunk_stream,
    #     ArrowArrayStream* mapping_stream,
    #     size_t k, size_t w, uint64_t salt, int orient, size_t max_memory_bytes)
    lib.rype_index_build_from_arrow.argtypes = [
        ctypes.c_char_p,  # output_path
        ctypes.c_void_p,  # chunk_stream ptr
        ctypes.c_void_p,  # mapping_stream ptr
        ctypes.c_size_t,  # k
        ctypes.c_size_t,  # w
        ctypes.c_uint64,  # salt
        ctypes.c_int,     # orient
        ctypes.c_size_t,  # max_memory_bytes
    ]
    lib.rype_index_build_from_arrow.restype = ctypes.c_int

    lib.rype_get_last_error.argtypes = []
    lib.rype_get_last_error.restype = ctypes.c_char_p

    return lib


# ArrowArrayStream is 5 pointers: get_schema, get_next, get_last_error, release, private_data
ARROW_ARRAY_STREAM_SIZE = 5 * ctypes.sizeof(ctypes.c_void_p)


def alloc_arrow_stream():
    """Allocate a zeroed ArrowArrayStream buffer, return (buffer, pointer)."""
    buf = ctypes.create_string_buffer(ARROW_ARRAY_STREAM_SIZE)
    return buf, ctypes.cast(buf, ctypes.c_void_p).value


def chunk_reader(genomes, chunk_size=64):
    """
    Build a RecordBatchReader of (feature_idx, chunk_index, chunk_data).

    `genomes` maps feature_idx (int) -> sequence bytes. Each sequence is split
    into `chunk_size`-byte blocks emitted in ascending, 0-based chunk_index order.
    A feature's chunks are kept contiguous, as rype requires.
    """
    schema = pa.schema(
        [
            ("feature_idx", pa.int64()),
            ("chunk_index", pa.int32()),
            ("chunk_data", pa.binary()),
        ]
    )
    feats, idxs, datas = [], [], []
    for fid, seq in genomes.items():
        for i in range(0, len(seq), chunk_size):
            feats.append(fid)
            idxs.append(i // chunk_size)
            datas.append(seq[i : i + chunk_size])
    batch = pa.record_batch(
        [
            pa.array(feats, type=pa.int64()),
            pa.array(idxs, type=pa.int32()),
            pa.array(datas, type=pa.binary()),
        ],
        schema=schema,
    )
    return pa.RecordBatchReader.from_batches(schema, [batch])


def mapping_reader(feature_to_bucket):
    """Build a RecordBatchReader of (feature_idx, bucket_name)."""
    schema = pa.schema([("feature_idx", pa.int64()), ("bucket_name", pa.utf8())])
    batch = pa.record_batch(
        [
            pa.array(list(feature_to_bucket.keys()), type=pa.int64()),
            pa.array(list(feature_to_bucket.values()), type=pa.utf8()),
        ],
        schema=schema,
    )
    return pa.RecordBatchReader.from_batches(schema, [batch])


def build_index(lib, out_path, genomes, feature_to_bucket, k=32, w=50, salt=0x5555555555555555, orient=False):
    chunk_buf, chunk_ptr = alloc_arrow_stream()
    map_buf, map_ptr = alloc_arrow_stream()

    # Export both readers into ArrowArrayStreams; rype takes ownership of each.
    chunk_reader(genomes)._export_to_c(chunk_ptr)
    mapping_reader(feature_to_bucket)._export_to_c(map_ptr)

    rc = lib.rype_index_build_from_arrow(
        out_path.encode("utf-8"),
        chunk_ptr,
        map_ptr,
        k,
        w,
        salt,
        1 if orient else 0,
        0,  # max_memory_bytes = auto
    )
    if rc != 0:
        err = lib.rype_get_last_error()
        msg = err.decode("utf-8") if err else "unknown error"
        raise RuntimeError(f"rype index build failed: {msg}")


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "scratch/pyarrow_built.ryxdi"
    lib = load_librype()

    # Two toy "genomes" (feature_idx -> sequence), and a feature -> bucket mapping.
    genomes = {
        1: b"ACGTACGTTGCAACGTGGTTACACGGTACACACGTTTGACACGTGGTACTTACAGGTAACGTACGTACGTTTGCA",
        2: b"TTGCAACGTGGTTACAGGTACGTACGTACACACGTTTGACACGTGGTACTTACAGGTAACGTACGTACGTGGTTA",
    }
    feature_to_bucket = {1: "genome_A", 2: "genome_B"}

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    build_index(lib, out_path, genomes, feature_to_bucket, k=32, w=10)
    print(f"Built index at {out_path}")
    print("Inspect it with:  rype index stats -i", out_path)


if __name__ == "__main__":
    main()
