# Rype Architecture

Design rationale for cross-cutting decisions in Rype. For data structure layouts and field listings, read the code: `src/core.rs`, `src/classify.rs`, `src/indices/`. For constants and their values, see `src/constants.rs`.

## Why RY (purine/pyrimidine) encoding

A reduced 2-bit alphabet collapses purines and pyrimidines:

- Purines (A/G) → 1
- Pyrimidines (T/C) → 0
- Other bases (N, ambiguous) → invalid; resets k-mer extraction

This buys two properties:

1. **Mutation tolerance.** A↔G and T↔C substitutions don't break matches, which is desirable for noisy long reads and cross-strain comparisons.
2. **Density.** 64bp k-mers fit in a single `u64`, enabling fast hashing and compact storage. Reverse complement is `!kmer` in RY space.

The tradeoff is reduced specificity per k-mer. We compensate with longer k (default 64) and minimizer sketching.

## Why minimizers

A sliding window of size `w` over k-mers selects the minimum hash per window as the representative, with consecutive duplicates collapsed. Implemented with a monotonic deque for O(n) extraction.

This reduces the index size from ~|sequence| to ~|sequence|/w entries while preserving the property that homologous regions share minimizers with high probability. Typical `w` is 50–200.

## Why Parquet for indices

Indices are stored as `.ryxdi` directories:

- `manifest.toml` — k, w, salt, bucket metadata (human-readable)
- `buckets.parquet` — `(bucket_id, bucket_name, sources)`
- `inverted/shard.N.parquet` — sorted `(minimizer: u64, bucket_id: u32)` pairs

The shard layout enables the three properties we need:

- **Bounded memory during build** via streaming k-way merge of pre-sorted shards.
- **Bounded memory during classify** by loading one shard at a time (see `classify_batch_sharded_merge_join`).
- **Inspectability.** Manifests are human-readable; shards open in any Parquet tool.

DELTA_BINARY_PACKED encoding gives strong compression on sorted minimizer columns. Per-row-group bloom filters can reject I/O early when a batch's minimizers don't appear in a shard.

## Build → classify lifecycle

1. **Build** (`rype index create` or `from-config`): FASTA → minimizer extraction → sorted Parquet shards → manifest.
2. **Classify** (`rype classify run`): manifest loads instantly; for each batch of reads, extract query minimizers in parallel (rayon), then merge-join against each shard on disk.
3. **Negative filtering** (`-N` flag): a second index whose minimizers subtract from per-bucket scores. Indices must share `k`, `w`, and `salt` to be combinable.

The C API wraps steps 2–3 for FFI consumers.
