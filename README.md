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
rype index create -o index.ryxdi -r ref1.fasta -r ref2.fasta -k 64 -w 50

# Or build from a TOML configuration file
rype index from-config -c config.toml
```

### Classify sequences

```bash
# Single-end reads
rype classify run -i index.ryxdi -1 reads.fastq -t 0.1

# Paired-end reads
rype classify run -i index.ryxdi -1 reads_R1.fastq -2 reads_R2.fastq -t 0.1
```

### Index management

```bash
# Show index statistics
rype index stats -i index.ryxdi
```

## CLI Reference

### `rype index`

| Subcommand | Description |
|------------|-------------|
| `create` | Build index from FASTA/FASTQ files |
| `stats` | Show index statistics |
| `bucket-source-detail` | Show source details for a specific bucket |
| `from-config` | Build index from TOML configuration file |

### `rype classify`

| Subcommand | Description |
|------------|-------------|
| `run` | Per-read classification |
| `batch` | Batch classify (alias for run) |

## Configuration File

Build complex indices using a TOML configuration file:

```toml
[index]
output = "output.ryxdi"
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
use needletail::parse_fastx_file;
use rype::{
    classify_batch_sharded_merge_join, extract_into, BucketData, MinimizerWorkspace,
    ParquetWriteOptions, ShardedInvertedIndex,
};

fn main() -> anyhow::Result<()> {
    let mut workspace = MinimizerWorkspace::new();
    let k = 32;
    let w = 10;
    let salt = 0u64;

    // Read phiX174 genome and extract minimizers
    let mut phix_mins = Vec::new();
    let mut reader = parse_fastx_file("examples/phiX174.fasta")?;
    while let Some(record) = reader.next() {
        let record = record?;
        extract_into(&record.seq(), k, w, salt, &mut workspace);
        phix_mins.extend(workspace.buffer.drain(..));
    }
    phix_mins.sort();
    phix_mins.dedup();

    // Read pUC19 plasmid and extract minimizers
    let mut puc19_mins = Vec::new();
    let mut reader = parse_fastx_file("examples/pUC19.fasta")?;
    while let Some(record) = reader.next() {
        let record = record?;
        extract_into(&record.seq(), k, w, salt, &mut workspace);
        puc19_mins.extend(workspace.buffer.drain(..));
    }
    puc19_mins.sort();
    puc19_mins.dedup();

    // Build bucket data
    let buckets = vec![
        BucketData {
            bucket_id: 1,
            bucket_name: "phiX174".to_string(),
            sources: vec!["phiX174.fasta".to_string()],
            minimizers: phix_mins,
        },
        BucketData {
            bucket_id: 2,
            bucket_name: "pUC19".to_string(),
            sources: vec!["pUC19.fasta".to_string()],
            minimizers: puc19_mins,
        },
    ];

    // Create index in temporary directory
    let temp_dir = tempfile::tempdir()?;
    let index_path = temp_dir.path().join("example.ryxdi");
    let options = ParquetWriteOptions::default();
    rype::create_parquet_inverted_index(&index_path, buckets, k, w, salt, None, Some(&options))?;

    // Load and classify
    let index = ShardedInvertedIndex::open(&index_path)?;

    // Create query from first 100bp of phiX174 - should match bucket 1
    let query_seq = b"GAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCTGATGAGTCGAAAAATTATCTTGATAAAGCAGGAATTACTACTGCTTGTTT";
    let queries = vec![(1_i64, query_seq.as_slice(), None)];

    let results = classify_batch_sharded_merge_join(&index, None, &queries, 0.1, None, None)?;
    for hit in &results {
        let name = index.manifest().bucket_names.get(&hit.bucket_id).unwrap();
        println!("Query {} -> {} (score: {:.2})", hit.query_id, name, hit.score);
    }

    Ok(())
}
```

## C API

Rype provides a C API for FFI integration:

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rype.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <index.ryxdi>\n", argv[0]);
        return 1;
    }

    // Load index
    RypeIndex* index = rype_index_load(argv[1]);
    if (!index) {
        fprintf(stderr, "Failed to load index: %s\n", rype_get_last_error());
        return 1;
    }

    // Create a query (70bp from phiX174)
    const char* seq = "GAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCTGATGAGTCGAAAAATTATCTT";
    RypeQuery queries[1];
    queries[0].seq = (const uint8_t*)seq;
    queries[0].seq_len = strlen(seq);
    queries[0].pair_seq = NULL;
    queries[0].pair_len = 0;

    // Classify sequences
    RypeResultArray* results = rype_classify(index, queries, 1, 0.1);
    if (!results) {
        fprintf(stderr, "Classification failed: %s\n", rype_get_last_error());
        rype_index_free(index);
        return 1;
    }

    // Process results
    printf("Found %zu hits:\n", results->len);
    for (size_t i = 0; i < results->len; i++) {
        const char* name = rype_bucket_name(index, results->data[i].bucket_id);
        printf("  Query %ld -> %s (bucket %u, score: %.2f)\n",
               results->data[i].query_id,
               name ? name : "unknown",
               results->data[i].bucket_id,
               results->data[i].score);
    }

    // Cleanup
    rype_results_free(results);
    rype_index_free(index);
    return 0;
}
```

Build and run:
```bash
gcc example.c -L target/release -lrype -o example
LD_LIBRARY_PATH=target/release ./example index.ryxdi
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

## Index Format

Rype uses a Parquet-based inverted index format stored as a directory:

```text
index.ryxdi/
├── manifest.toml         # Index metadata (k, w, salt, bucket info)
├── buckets.parquet       # Bucket metadata (id, name, sources)
└── inverted/
    ├── shard.0.parquet   # Inverted index shard (minimizer -> bucket_id pairs)
    └── ...               # Additional shards for large indices
```

This format provides:
- Efficient columnar storage with compression
- Memory-efficient streaming classification
- Support for indices larger than available RAM

## Performance Tips

- Use `--release` builds for production
- Batch processing amortizes thread pool overhead
- Parquet format enables memory-efficient classification for large indices

## Testing

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_name
```

## Development Setup

```bash
# Enable pre-commit hooks (runs fmt, clippy, tests, doc checks)
git config core.hooksPath .githooks

# Manually run checks
cargo fmt --check
cargo clippy --all-features -- -D warnings
cargo test --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
```

The test suite includes automated verification that README examples compile and run correctly (`readme_example_test.rs` and `cli_integration_tests.rs::test_readme_bash_examples`).

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
