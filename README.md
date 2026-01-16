# Rype

[![CI](https://github.com/YOUR_USERNAME/rype/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/rype/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/rype.svg)](https://crates.io/crates/rype)
[![Documentation](https://docs.rs/rype/badge.svg)](https://docs.rs/rype)
[![License](https://img.shields.io/crates/l/rype.svg)](LICENSE)

High-performance genomic sequence classification using minimizer-based k-mer sketching in RY (purine/pyrimidine) space.

## Overview

Rype is a Rust library and CLI tool for fast sequence classification. It uses a reduced 2-bit alphabet that collapses purines (A/G) and pyrimidines (T/C), enabling:

- **Mutation tolerance**: Purine-purine and pyrimidine-pyrimidine mutations don't break k-mer matches
- **Compact representation**: 64bp k-mers fit in a single `u64`
- **High performance**: Minimizer sketching with O(n) extraction using monotonic deques

## Installation

### From source

```bash
cargo install --path .
```

### Building

```bash
# Release build
cargo build --release

# Development build
cargo build
```

## Quick Start

### Create an index

```bash
# Create index from reference sequences
rype index create -o index.ryidx -r ref1.fasta -r ref2.fasta -k 64 -w 50

# Or build from a TOML configuration file
rype index from-config -c config.toml
```

### Classify sequences

```bash
# Single-end reads
rype classify run -i index.ryidx -1 reads.fastq -t 0.1

# Paired-end reads
rype classify run -i index.ryidx -1 reads_R1.fastq -2 reads_R2.fastq -t 0.1

# Aggregated classification (higher sensitivity)
rype classify aggregate -i index.ryidx -1 reads.fastq -t 0.05
```

### Index management

```bash
# Show index statistics
rype index stats -i index.ryidx

# Add sequences as a new bucket
rype index bucket-add -i index.ryidx -r new_ref.fasta

# Merge buckets within an index
rype index bucket-merge -i index.ryidx --src 2 --dest 1

# Merge multiple indices
rype index merge -o merged.ryidx -i idx1.ryidx -i idx2.ryidx

# Create sharded index for large datasets
rype index shard -i large.ryidx -o sharded.ryidx --max-shard-size 1G

# Create inverted index for memory-efficient classification
rype index invert -i index.ryidx
```

## CLI Reference

### `rype index`

| Subcommand | Description |
|------------|-------------|
| `create` | Build index from FASTA/FASTQ files |
| `stats` | Show index statistics |
| `bucket-source-detail` | Show source details for a specific bucket |
| `bucket-add` | Add sequences to existing index as new bucket |
| `bucket-merge` | Merge two buckets within an index |
| `merge` | Merge multiple indices into one |
| `from-config` | Build index from TOML configuration file |
| `shard` | Convert single-file index to sharded format |
| `invert` | Create inverted index for memory-efficient classification |

### `rype classify`

| Subcommand | Description |
|------------|-------------|
| `run` | Per-read classification |
| `batch` | Batch classify (alias for run) |
| `aggregate` | Aggregated classification for paired-end reads |

## Configuration File

Build complex indices using a TOML configuration file:

```toml
[index]
output = "output.ryidx"
k = 64
w = 50

[[buckets]]
name = "species_a"
sources = ["ref_a1.fasta", "ref_a2.fasta"]

[[buckets]]
name = "species_b"
sources = ["ref_b.fasta"]
```

## Library Usage

```rust
use rype::{Index, MinimizerWorkspace};

// Create an index
let mut index = Index::new(64, 50, 0); // k=64, w=50, salt=0

// Add sequences to buckets
let mut workspace = MinimizerWorkspace::new();
index.add_record(1, b"ACGTACGT...", &mut workspace);
index.finalize_bucket(1);

// Save and load
index.save("index.ryidx")?;
let loaded = Index::load("index.ryidx")?;

// Classify sequences
let results = loaded.classify_batch(&queries, threshold, num_threads);
```

## C API

Rype provides a C API for FFI integration:

```c
#include "rype.h"

// Load index
RypeIndex* index = rype_index_load("index.ryidx");

// Classify sequences
RypeResultArray* results = rype_classify(index, queries, num_queries, threshold);

// Process results
for (size_t i = 0; i < results->len; i++) {
    printf("Query %u -> Bucket %u (score: %f)\n",
           results->data[i].query_id,
           results->data[i].bucket_id,
           results->data[i].score);
}

// Cleanup
rype_results_free(results);
rype_index_free(index);
```

Build with:
```bash
gcc example.c -L target/release -lrype -o example
LD_LIBRARY_PATH=target/release ./example
```

### Thread Safety

- Index loading/freeing: NOT thread-safe
- Classification (`rype_classify`): Thread-safe (multiple threads can share the same index)
- Results: NOT thread-safe (each thread needs its own `RypeResultArray`)

## Algorithm Details

### RY Encoding

The library uses a reduced 2-bit alphabet:

| Base | Category | Encoding |
|------|----------|----------|
| A, G | Purine | 1 |
| T, C | Pyrimidine | 0 |
| N, etc. | Invalid | Resets k-mer |

### Minimizer Extraction

1. Slide a window of size `w` over k-mers
2. Select the minimum hash value within each window
3. Deduplicate consecutive identical minimizers

The implementation uses a monotonic deque for O(n) time complexity.

### Supported K-mer Sizes

- K = 16, 32, or 64 (stored as `u64`)

## Index Formats

### Main Index (`.ryidx`)

Binary format with zstd-compressed, delta-encoded minimizers. Achieves ~65% compression vs raw storage.

### Sharded Index

For large datasets that exceed available RAM:
```
index.ryidx.manifest     # Manifest file
index.ryidx.shard.0      # Shard 0
index.ryidx.shard.1      # Shard 1
...
```

### Inverted Index (`.ryxdi`)

Memory-efficient format for classification:
```
index.ryxdi.manifest     # Manifest file
index.ryxdi.shard.0      # Shard 0 (1:1 with main shards)
...
```

## Performance Tips

- Use `--release` builds for production
- Batch processing amortizes thread pool overhead
- Sharded indices enable classification when total index exceeds RAM
- Inverted indices reduce per-bucket search from O(queries × minimizers) to O(unique_minimizers)

## Testing

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_name
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
