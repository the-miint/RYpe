# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Rype** is a high-performance genomic sequence classification library using minimizer-based k-mer sketching in RY (purine/pyrimidine) space. It's written in Rust and provides both a Rust library, CLI tool, and C API for FFI integration.

## Build and Development Commands

### Building
```bash
# Build the project
cargo build --release

# Build for development with debug symbols
cargo build
```

### Testing
```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_name
```

### C API Development
```bash
# Build C example
gcc example.c -L target/debug -lrype -o c_example

# Set library path and run
LD_LIBRARY_PATH=target/debug ./c_example
```

### CLI Usage
```bash
# Create an index from reference sequences
cargo run --release -- index create -o output.ryidx -r ref1.fasta -r ref2.fasta -k 64 -w 50

# Show index statistics
cargo run --release -- index stats -i index.ryidx

# Add sequences to an existing index as a new bucket
cargo run --release -- index bucket-add -i index.ryidx -r new_ref.fasta

# Merge two buckets within an index
cargo run --release -- index bucket-merge -i index.ryidx --src 2 --dest 1

# Merge multiple indices into one
cargo run --release -- index merge -o merged.ryidx -i idx1.ryidx -i idx2.ryidx

# Build index from a TOML configuration file
cargo run --release -- index from-config -c config.toml

# Classify sequences (single-end)
cargo run --release -- classify run -i index.ryidx -1 reads.fastq -t 0.1

# Classify sequences (paired-end)
cargo run --release -- classify run -i index.ryidx -1 reads_R1.fastq -2 reads_R2.fastq -t 0.1

# Aggregate classification (for higher sensitivity)
cargo run --release -- classify aggregate -i index.ryidx -1 reads.fastq -t 0.05
```

## Architecture Overview

### RY Encoding (Core Concept)

The library uses a reduced 2-bit alphabet that collapses purines and pyrimidines:
- **Purines** (A/G) → 1
- **Pyrimidines** (T/C) → 0
- **Other bases** (N, ambiguous) → invalid (resets k-mer extraction)

This enables purine/pyrimidine-aware matching where AG-purine mutations don't break matches, and allows 64bp k-mers to fit in a single u64.

### Minimizer Sketching Algorithm

The library reduces sequence representation using minimizers:
1. Sliding window of size `w` over k-mers
2. Select minimum hash value within each window as representative
3. Deduplicate consecutive identical minimizers

Implementation uses monotonic deque for O(n) time complexity (see `extract_into()` in src/lib.rs).

### Key Data Structures

**Index** (src/lib.rs:216-389):
- `buckets: HashMap<u32, Vec<u64>>` - Maps bucket ID to sorted, deduplicated minimizers
- `bucket_names: HashMap<u32, String>` - Human-readable bucket names
- `bucket_sources: HashMap<u32, Vec<String>>` - Source sequences for each bucket
- `w: usize` - Window size for minimizer selection
- `salt: u64` - XOR salt applied to k-mer hashes

**MinimizerWorkspace** (src/lib.rs:54-68):
- Reusable workspace to avoid allocations in hot loops
- Contains deques for forward/reverse-complement k-mer tracking
- `buffer: Vec<u64>` - Output minimizers

**HitResult** (src/lib.rs:40-45):
- Classification result: query_id, bucket_id, score

### Core Algorithms

**Minimizer Extraction** (src/lib.rs:78-169):
- `extract_into()` - Single-strand minimizer extraction
- `extract_dual_strand_into()` - Forward + reverse-complement extraction
- `get_paired_minimizers_into()` - Paired-end read handling

**Classification** (src/lib.rs:394-489):
- `classify_batch()` - Parallel batch classification with inverted index
  - Builds maps of minimizer → query indices
  - For each bucket, binary searches minimizers and accumulates hits
  - Scores per-read: max(forward_score, reverse_score)
- `aggregate_batch()` - Aggregates paired-end reads into single score

**Index Building** (src/lib.rs:236-254):
- `add_record()` - Add sequence to bucket (accumulates minimizers)
- `finalize_bucket()` - Sort and deduplicate minimizers

**Serialization** (src/lib.rs:282-389):
- Custom binary format: "RYP3" magic + version 2
- Stores K, w, salt, then buckets with names/sources/minimizers
- Safe deserialization with MAX_BUCKET_SIZE, MAX_STRING_LENGTH, MAX_NUM_BUCKETS checks

### C API (src/c_api.rs)

FFI layer exposing core functionality to C:

**Thread Safety**:
- Index loading/freeing: NOT thread-safe
- Classification (`rype_classify`): Thread-safe (multiple threads can use same Index)
- Results: NOT thread-safe (each thread needs own RypeResultArray)

**Key Functions**:
- `rype_index_load(path)` - Load index from disk
- `rype_classify(index, queries, num_queries, threshold)` - Batch classify
- `rype_results_free(results)` - Free result array
- `rype_get_last_error()` - Get thread-local error message

**Safety**:
- Input validation for all C pointers and sizes
- Thread-local error reporting
- MAX_SEQUENCE_LENGTH limit (2GB)
- Panic catching in `rype_classify`

### CLI (src/main.rs)

Nested subcommands using clap:

**`rype index`** - Index operations:
- `create` - Build index from FASTA/FASTQ
- `stats` - Show index statistics
- `bucket-source-detail` - Show source details for a specific bucket
- `bucket-add` - Add sequences to existing index as new bucket
- `bucket-merge` - Merge two buckets within an index
- `merge` - Merge multiple indices into one
- `from-config` - Build index from TOML configuration file

**`rype classify`** - Classification operations:
- `run` - Per-read classification
- `batch` - Batch classify (alias for run)
- `aggregate` - Aggregated classification for paired-end (alias: `agg`)

## Important Constants

- `K ∈ {16, 32, 64}` - K-mer size (configurable per-index, always uses u64 representation)
- `MAX_BUCKET_SIZE = 1_000_000_000` - DoS protection for index loading
- `MAX_STRING_LENGTH = 10_000` - Max name/source string length
- `MAX_NUM_BUCKETS = 100_000` - Max buckets per index
- `MAX_SEQUENCE_LENGTH = 2_000_000_000` - Max sequence size for C API

## Critical Implementation Details

### K-mer Encoding
The `base_to_bit()` function (src/lib.rs:49-52) uses unsafe lookup table for performance. Invalid bases return `u64::MAX` which triggers window reset.

### Canonical K-mers
K-mers and their reverse complements are treated as equivalent. Reverse complement calculated via bitwise NOT: `!kmer` in RY-space.

### Parallel Processing
Uses `rayon` for data parallelism:
- `classify_batch()` parallelizes minimizer extraction AND bucket scoring
- `map_init()` pattern provides per-thread workspace to avoid allocations

### Index File Format (version 4)
```
HEADER (uncompressed):
Magic: "RYP4" (4 bytes)
Version: u32 (4 bytes) = 4
K: u64 (8 bytes) - must be 16, 32, or 64
W: u64 (8 bytes)
Salt: u64 (8 bytes)
Num Buckets: u32 (4 bytes)

METADATA (uncompressed, for each bucket in sorted ID order):
  - Minimizer count: u64
  - Bucket ID: u32
  - Name length: u64, then UTF-8 bytes
  - Source count: u64
    - For each source: length (u64), UTF-8 bytes

MINIMIZERS (zstd compressed stream):
  - All minimizers for all buckets, in bucket order (u64 little-endian each)
```

This format keeps metadata uncompressed for fast `load_metadata()` while compressing the bulk of the data (minimizers).

### Error Handling
- Rust API: Uses `anyhow::Result<T>` for all fallible operations
- C API: Returns NULL on error, call `rype_get_last_error()` for details
- Safe loading: Validates version, enforces size limits, uses safe deserialization

## Memory Management Notes

### Rust Side
- Workspace reuse pattern minimizes allocations (pass `&mut MinimizerWorkspace`)
- Buckets store sorted, deduplicated minimizers (Vec<u64>)
- Classification builds temporary inverted indices per batch

### C API Side
- Index ownership transferred via `Box::into_raw()` / `Box::from_raw()`
- Results allocated in Rust, freed by caller with `rype_results_free()`
- Never free Index while classification is in progress (use-after-free)
- Never double-free RypeResultArray (undefined behavior)

## Testing Strategy

Existing tests in src/lib.rs cover:
- Minimizer extraction correctness
- Index save/load round-trips
- Error paths (invalid format, oversized allocations, overflow)
- C API validation logic

When adding features:
1. Add unit tests for core logic
2. Add error path tests
3. Test with C example if touching FFI
4. Consider edge cases: empty sequences, N-bases, very long sequences

## Performance Considerations

- **Hot path**: `extract_into()`, `classify_batch()` - avoid allocations
- **Deque capacity**: Pre-sized to avoid reallocation during sliding window
- **Binary search**: Buckets must be sorted for `binary_search()` correctness
- **Parallelism**: Batch processing amortizes thread pool overhead
- **Inverted index**: Reduces per-bucket work from O(queries × minimizers) to O(unique_minimizers)

## Common Pitfalls

1. **Modifying buckets after finalization**: Must call `finalize_bucket()` after `add_record()` to ensure sorted/deduplicated minimizers
2. **K-mer size**: K must be 16, 32, or 64. K is set at index creation and stored in the index file. All merged indices must have matching K values.
3. **C API thread safety**: Don't share RypeResultArray across threads
4. **Index compatibility**: Version mismatch between save/load will fail
5. **Short sequences**: Sequences < K bases produce no minimizers
