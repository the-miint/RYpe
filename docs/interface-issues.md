# Interface Issues: Memory-Efficient Negative Filtering

This document tracks known issues with the memory-efficient negative filtering implementation that should be addressed in a future refactoring pass.

**Created:** 2025-01-25
**Related:** `docs/PLAN-memory-efficient-negative-filtering.md`

---

## Issue 1: Double Minimizer Extraction

**Severity:** Performance
**Affects:** CLI (`--negative-index`), C API (`rype_classify_with_negative`, Arrow FFI)

### Description

When negative filtering is enabled, minimizers are extracted **twice** per batch:

1. First extraction in `classify_with_sharded_negative()` to build the `all_minimizers` set for querying the negative index
2. Second extraction inside `classify_batch_sharded_sequential()` during the actual classification

This doubles the CPU cost of the hot path (hashing, deque operations, RY encoding, canonical k-mer computation).

### Code Locations

**`src/classify/sharded.rs`:**
- Lines 524-534: First extraction in `classify_with_sharded_negative()`
- Lines 75-86: Second extraction in `classify_batch_sharded_sequential()`

**`src/c_api.rs`:**
- Lines 904-936: `collect_negative_mins_for_batch()` extracts minimizers for negative filtering
- The subsequent `classify_arrow_batch_sharded()` call extracts again internally

### Current Workaround

The code includes a comment acknowledging this (lines 548-549):
```rust
// Note: This will re-extract minimizers, but the memory savings from sharded
// negative filtering (24GB -> 1-2GB) far outweigh the extraction overhead.
```

This comment is misleading - it's not a designed tradeoff but a shortcut that avoided refactoring.

### Recommended Fix

Create a `classify_batch_sharded_sequential_preextracted()` function that accepts pre-extracted minimizers:

```rust
type ExtractedMins = (i64, Vec<u64>, Vec<u64>);  // (query_id, fwd, rc)

fn classify_batch_sharded_sequential_preextracted(
    sharded: &ShardedInvertedIndex,
    extracted: &[ExtractedMins],
    threshold: f64,
    read_options: Option<&ParquetReadOptions>,
) -> Result<Vec<HitResult>>
```

Then modify `classify_with_sharded_negative()` to:
1. Extract minimizers once
2. Build `all_minimizers` from the extracted data
3. Collect negative hits
4. Filter the extracted minimizers
5. Call the pre-extracted classification function

---

## Issue 3: Inconsistent Negative Filtering Support Across Classification Strategies

**Severity:** Feature Gap
**Affects:** CLI, C API

### Description

Negative filtering only works with sequential shard processing. Users cannot combine negative filtering with the faster `--parallel-rg` or `--merge-join` classification strategies.

### Code Locations

**CLI (`src/commands/classify.rs:267-272`):**
```rust
if args.parallel_rg || args.merge_join {
    return Err(anyhow!(
        "Negative index filtering is only supported with sequential mode.\n\
         Remove --parallel-rg or --merge-join flags when using --negative-index."
    ));
}
```

**C API (`src/c_api.rs`):**
- `classify_internal()` always uses `classify_with_sharded_negative()` which internally calls `classify_batch_sharded_sequential()`
- No option to use merge-join or parallel-rg with negative filtering

### User Impact

- Users must choose between fast classification (parallel-rg/merge-join) OR negative filtering
- For large datasets, this forces suboptimal performance when contaminant filtering is needed
- The limitation is architectural, not fundamental - all strategies could support negative filtering

### Recommended Fix

Integrate negative filtering into all classification strategies:

1. `classify_batch_sharded_merge_join()` - already accepts `negative_mins: Option<&HashSet<u64>>`, just needs sharded variant
2. `classify_batch_sharded_parallel_rg()` - already accepts `negative_mins: Option<&HashSet<u64>>`, just needs sharded variant

Create wrapper functions or modify existing ones to:
1. Accept `Option<&ShardedInvertedIndex>` for the negative index
2. Collect negative minimizers once per batch (using `collect_negative_minimizers_sharded`)
3. Pass the resulting `HashSet` to the underlying classification function

**Alternative approach:** Create a unified `classify_with_options()` function that accepts a strategy enum and handles negative filtering uniformly.

---

## Summary

| Issue | Type | CLI | C API | Priority |
|-------|------|-----|-------|----------|
| Double extraction | Performance | Yes | Yes | Medium |
| Inconsistent strategy support | Feature gap | Yes | Yes | Low |

Both issues stem from the same root cause: `classify_with_sharded_negative()` was implemented as a wrapper around `classify_batch_sharded_sequential()` rather than refactoring the classification pipeline to support negative filtering at a lower level.

A comprehensive fix would involve:
1. Separating minimizer extraction from classification
2. Creating a common "prepare batch" phase that handles negative filtering
3. Allowing all classification strategies to consume pre-prepared batches
