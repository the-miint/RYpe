# Rype C API Examples

This directory contains example C programs demonstrating the Rype library API.

## Examples

### basic_example.c

Demonstrates the core C API:
- Loading indices (main, inverted, sharded)
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

## Building

### Prerequisites

```bash
# Build rype library (release mode)
cd ..
cargo build --release

# For Arrow support
cargo build --release --features arrow
```

### Compile Examples

```bash
# Basic example
gcc -o basic_example basic_example.c \
    -L../target/release -lrype \
    -Wl,-rpath,../target/release

# Arrow example (requires --features arrow)
gcc -DRYPE_ARROW -o arrow_example arrow_example.c \
    -L../target/release -lrype \
    -Wl,-rpath,../target/release
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
cargo run --release -- index create -o test.ryidx -r some_reference.fasta -k 64 -w 50

# Run examples
cd examples
./basic_example ../test.ryidx
./arrow_example ../test.ryidx
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
