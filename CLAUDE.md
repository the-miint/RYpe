# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Rype** is a high-performance genomic sequence classification library using minimizer-based k-mer sketching in RY (purine/pyrimidine) space. It's written in Rust and provides both a Rust library, CLI tool, and C API for FFI integration.

All indices are stored in Parquet format (`.ryxdi` directories).

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
cargo run --release -- index create -o index.ryxdi -r ref1.fasta -r ref2.fasta -k 64 -w 50

# Create index with one bucket per sequence
cargo run --release -- index create -o genes.ryxdi -r genes.fasta --separate-buckets

# Show index statistics
cargo run --release -- index stats -i index.ryxdi

# Show source details for a bucket
cargo run --release -- index bucket-source-detail -i index.ryxdi -b 1

# Build index from a TOML configuration file
cargo run --release -- index from-config -c config.toml

# Merge two indices into one
cargo run --release -- index merge --index-primary idx1.ryxdi --index-secondary idx2.ryxdi -o merged.ryxdi

# Merge with subtraction (remove secondary minimizers that exist in primary)
# Useful for creating non-host indices
cargo run --release -- index merge --index-primary host.ryxdi --index-secondary sample.ryxdi \
    -o non_host.ryxdi --subtract-from-primary

# Classify sequences (single-end)
cargo run --release -- classify run -i index.ryxdi -1 reads.fastq -t 0.1

# Classify sequences (paired-end)
cargo run --release -- classify run -i index.ryxdi -1 reads_R1.fastq -2 reads_R2.fastq -t 0.1

# Classify with negative filtering (host depletion)
cargo run --release -- classify run -i target.ryxdi -N host.ryxdi -1 reads.fastq -t 0.1

# Aggregate classification (for higher sensitivity)
cargo run --release -- classify aggregate -i index.ryxdi -1 reads.fastq -t 0.05

# Best-hit-only classification
cargo run --release -- classify run -i index.ryxdi -1 reads.fastq --best-hit

# Classify with sequence trimming (use first N bases only)
# Useful when read starts are more reliable than ends
cargo run --release -- classify run -i index.ryxdi -1 reads.fastq -t 0.1 --trim-to 100
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

Implementation uses monotonic deque for O(n) time complexity (see `extract_into()` in src/core.rs).

### Key Data Structures

**InvertedIndex** (src/indices/):
- Minimizer → bucket ID mappings for fast classification
- Loaded from Parquet shards on-demand

**ShardedInvertedIndex** (src/indices/):
- Memory-efficient sharded inverted index
- Holds manifest; shards loaded on-demand during classification

**MinimizerWorkspace** (src/core.rs):
- Reusable workspace to avoid allocations in hot loops
- Contains deques for forward/reverse-complement k-mer tracking
- `buffer: Vec<u64>` - Output minimizers

**HitResult** (src/types.rs):
- Classification result: query_id, bucket_id, score

**BucketData** (src/indices/parquet/):
- Used during index creation: bucket_id, bucket_name, sources, minimizers

### Constants Module (src/constants.rs)

Centralized constants for consistency and maintainability:

**Safety Limits:**
- `MAX_INVERTED_MINIMIZERS` - Maximum minimizers in inverted index (1 trillion)
- `MAX_INVERTED_BUCKET_IDS` - Maximum bucket ID entries (4 billion)
- `MAX_SEQUENCE_LENGTH` - Max sequence size for C API (2GB)
- `MAX_READS` - Bit-packing limit (2^31 - 1)

**Performance Tuning:**
- `GALLOP_THRESHOLD` = 16 - Merge-join vs galloping switch point
- `QUERY_HASHSET_THRESHOLD` = 1000 - Linear vs HashSet lookup
- `PARQUET_BATCH_SIZE`, `DEFAULT_ROW_GROUP_SIZE` - Parquet I/O sizing

**Delimiters:**
- `BUCKET_SOURCE_DELIM` = "::" - Separates filename from sequence name in bucket sources

### Core Algorithms

**Minimizer Extraction** (src/core.rs):
- `extract_into()` - Single-strand minimizer extraction
- `extract_dual_strand_into()` - Forward + reverse-complement extraction
- `get_paired_minimizers_into()` - Paired-end read handling

**Classification** (src/classify.rs):
- `classify_batch_sharded_merge_join()` - Default classification using merge-join
- `classify_batch_sharded_parallel_rg()` - Classification with parallel row group processing
- `classify_with_sharded_negative()` - Classification with negative filtering

**Index Building** (src/indices/parquet/):
- `create_parquet_inverted_index()` - Create Parquet index from BucketData

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
- `create` - Build Parquet index from FASTA/FASTQ
- `stats` - Show index statistics
- `bucket-source-detail` - Show source details for a specific bucket
- `bucket-add` - Add sequences to existing index as new bucket (development pending)
- `from-config` - Build index from TOML configuration file
- `bucket-add-config` - Add files using TOML config (development pending)
- `merge` - Merge two indices into one (with optional subtraction)
- `summarize` - Show detailed minimizer statistics

**`rype classify`** - Classification operations:
- `run` - Per-read classification
- `aggregate` - Aggregated classification for paired-end (alias: `agg`)

**`rype inspect`** - Debugging operations:
- `matches` - Show matching minimizers between queries and buckets (not supported with Parquet)

## Important Constants

- `K ∈ {16, 32, 64}` - K-mer size (configurable per-index, always uses u64 representation)
- `MAX_SEQUENCE_LENGTH = 2_000_000_000` - Max sequence size for C API

## Critical Implementation Details

### K-mer Encoding
The `base_to_bit()` function uses unsafe lookup table for performance. Invalid bases return `u64::MAX` which triggers window reset.

### Canonical K-mers
K-mers and their reverse complements are treated as equivalent. Reverse complement calculated via bitwise NOT: `!kmer` in RY-space.

### Parallel Processing
Uses `rayon` for data parallelism:
- Classification parallelizes minimizer extraction AND bucket scoring
- `map_init()` pattern provides per-thread workspace to avoid allocations

### Parquet Index Format

All indices use Parquet format stored as `.ryxdi` directories:

```
index.ryxdi/
├── manifest.toml           # TOML metadata (k, w, salt, bucket info)
├── buckets.parquet         # (bucket_id, bucket_name, sources)
└── inverted/
    ├── shard.0.parquet     # (minimizer: u64, bucket_id: u32) sorted pairs
    └── ...                 # Additional shards for large indices
```

**Manifest Format (TOML)**:
```toml
magic = "RYPE_PARQUET_V1"
format_version = 1
k = 64
w = 50
salt = "0x5555555555555555"  # Hex string for large values
source_hash = "0xDEADBEEF"
num_buckets = 10
total_minimizers = 1000000

[inverted]
num_shards = 2
total_entries = 5000000
has_overlapping_shards = true  # Buckets may share minimizers across shards
```

**Benefits**:
- Parquet provides efficient columnar storage with DELTA_BINARY_PACKED encoding
- Streaming k-way merge enables building large indices with bounded memory
- Human-readable TOML manifest for easy inspection
- Shards loaded on-demand during classification

**Memory Benefits**:
- Manifest loads instantly (no minimizer data)
- Classification loads one shard at a time via `classify_batch_sharded_merge_join`
- Memory usage: O(batch_size × minimizers_per_read) + O(single_shard_size)
- Enables classification when total index exceeds available RAM

### Error Handling
- Rust API: Uses `anyhow::Result<T>` for all fallible operations
- C API: Returns NULL on error, call `rype_get_last_error()` for details
- Safe loading: Validates format, enforces size limits

## Memory Management Notes

### Rust Side
- Workspace reuse pattern minimizes allocations (pass `&mut MinimizerWorkspace`)
- Shards loaded on-demand, not all at once

### C API Side
- Index ownership transferred via `Box::into_raw()` / `Box::from_raw()`
- Results allocated in Rust, freed by caller with `rype_results_free()`
- Never free Index while classification is in progress (use-after-free)
- Never double-free RypeResultArray (undefined behavior)

## Testing Strategy

Existing tests cover:
- Minimizer extraction correctness
- Index creation and loading
- Classification accuracy
- C API validation logic

When adding features:
1. Add unit tests for core logic
2. Add error path tests
3. Test with C example if touching FFI
4. Consider edge cases: empty sequences, N-bases, very long sequences

## Performance Considerations

- **Hot path**: `extract_into()`, classification functions - avoid allocations
- **Deque capacity**: Pre-sized to avoid reallocation during sliding window
- **Parallelism**: Batch processing amortizes thread pool overhead
- **Inverted index**: Reduces per-bucket work from O(queries × minimizers) to O(unique_minimizers)
- **Row group filtering**: Bloom filters can reduce I/O by rejecting row groups early

## Common Pitfalls

1. **K-mer size**: K must be 16, 32, or 64. K is set at index creation and stored in the index.
2. **C API thread safety**: Don't share RypeResultArray across threads
3. **Index compatibility**: Indices with different k, w, or salt cannot be used together for negative filtering
4. **Short sequences**: Sequences < K bases produce no minimizers

## Development Environment Notes

- **Temporary files**: Do NOT use `/tmp` - it has insufficient space on this system. Use `scratch/` directory within the project for temporary files and test data. This directory is gitignored.
