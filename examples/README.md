# Rype API Examples

This directory contains examples demonstrating the Rype library API from C and Python.

## Examples

### basic_example.c

Demonstrates the core C API:
- Loading Parquet inverted indices (.ryxdi)
- Querying index metadata
- Single-end and paired-end classification
- Batch classification
- Proper error handling and cleanup

### arrow_example.c

Demonstrates the Arrow C Data Interface API:
- Creating Arrow streams (simplified example)
- Understanding ownership semantics (avoiding double-free)
- Consuming output streams with correct schema lifetime
- Proper cleanup order

### extraction_example.c

Demonstrates the C struct-based minimizer extraction API:
- `rype_extract_minimizer_set()` — sorted, deduplicated hash sets per strand
- `rype_extract_strand_minimizers()` — ordered hashes + positions per strand

### arrow_extraction_example.c

Demonstrates the Arrow C Data Interface extraction API using Arrow GLib:
- `rype_extract_minimizer_set_arrow()` — streaming set extraction
- `rype_extract_strand_minimizers_arrow()` — streaming strand extraction
- Requires `libarrow-glib-dev`

### ctypes_extraction_example.py

Demonstrates calling the single-sequence extraction C API from Python via
ctypes. No special dependencies beyond the standard library — uses
`librype.so` directly.

- `rype_extract_minimizer_set()` — sorted, deduplicated hash sets per strand
- `rype_extract_strand_minimizers()` — ordered hashes + positions per strand
- Includes invariant checks (sorted, positions in bounds, SoA lengths)
- Only requires `cargo build --release` (no `--features arrow-ffi`)

### pyarrow_extraction_example.py

Demonstrates calling the Arrow streaming extraction API from Python via
PyArrow's C Data Interface and ctypes. No Rust changes or native Python
extension needed — uses the same `librype.so` shared library.

- `rype_extract_minimizer_set_arrow()` — streaming batch set extraction
- `rype_extract_strand_minimizers_arrow()` — streaming batch strand extraction
- Requires `pyarrow` (tested with 23.0.0, Python 3.13)
- Requires `cargo build --release --lib --features arrow-ffi`

**These Python examples are NOT part of the regular test suite or build
dependencies.** They have been manually verified but are not run by `cargo test`.

## Building

### Prerequisites

```bash
# Build rype library (release mode)
cd ..
cargo build --release

# For Arrow FFI support (required for Arrow and PyArrow examples)
cargo build --release --lib --features arrow-ffi
```

### Compile C Examples

```bash
# Basic example
gcc -o basic_example basic_example.c \
    -L../target/release -lrype \
    -Wl,-rpath,../target/release

# Extraction example (struct-based, no Arrow dependency)
gcc -o extraction_example extraction_example.c \
    -L../target/release -lrype \
    -Wl,-rpath,../target/release

# Arrow example (requires --features arrow-ffi)
gcc -o arrow_example arrow_example.c \
    -L../target/release -lrype \
    -Wl,-rpath,../target/release

# Arrow extraction example (requires Arrow GLib + --features arrow-ffi)
gcc -o arrow_extraction_example arrow_extraction_example.c \
    $(pkg-config --cflags arrow-glib) \
    -L../target/release -lrype -Wl,-rpath,../target/release \
    $(pkg-config --libs arrow-glib)
```

### Run Python Examples

No compilation needed — uses ctypes to load `librype.so` directly.

```bash
# Single-sequence extraction (no extra dependencies)
python examples/ctypes_extraction_example.py

# PyArrow streaming extraction (requires pyarrow)
# One-time setup: create a conda environment with pyarrow
conda create -n rype-pyarrow python=3.13 pyarrow -y
conda run -n rype-pyarrow python examples/pyarrow_extraction_example.py
```

### Debug Build

```bash
# Use debug library for better error messages during development
gcc -o basic_example basic_example.c \
    -L../target/debug -lrype \
    -Wl,-rpath,../target/debug
```

## Running

```bash
# Create a test index first
cd ..
cargo run --release -- index create -o test.ryxdi -r some_reference.fasta -k 64 -w 50

# Run examples
cd examples
./basic_example ../test.ryxdi
./arrow_example ../test.ryxdi
```

## Key API Concepts

### Memory Management

1. **Index ownership**: `rype_index_load()` returns an owned pointer. Free with `rype_index_free()`.
2. **Results ownership**: `rype_classify()` returns an owned array. Free with `rype_results_free()`.
3. **String ownership**: `rype_bucket_name()` returns a borrowed pointer owned by the index. Do NOT free.

### Thread Safety

- Index loading/freeing: NOT thread-safe
- Classification: Thread-safe (multiple threads can use same index)
- Results: NOT thread-safe (each thread needs its own)

### Arrow Stream Ownership (CRITICAL)

When using `rype_classify_arrow()`:

1. **Input stream is CONSUMED**: The function takes ownership via the Arrow release callback. Do NOT release the input stream after calling.

2. **Output stream is OWNED by caller**: You must release it when done (unless the function returned an error).

3. **Schema lifetime**: Keep the schema alive while iterating batches. Release order: batches → schema → stream.

See `arrow_example.c` for detailed comments on these semantics.
