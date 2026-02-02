# Technical Debt Remediation Plan

## Overview
This plan addresses technical debt in the Rype codebase through pure refactoring. **No changes will affect classification results or behavior.** Each phase is designed to be independently testable via `cargo test`.

---

## Phase Status Summary

| Phase | Description | Status | Notes |
|-------|-------------|--------|-------|
| 1 | Split helpers.rs into modules | DONE | Reduced 2621-line file to 5 focused modules |
| 2 | Consolidate duplicate validation | DONE | Unified iterator-based API |
| 3 | Remove dead code and stale comments | DONE | Removed unused methods and stale TODO |
| 4 | Document deferred items | DONE | Created TECH_DEBT.md with all deferred items |

---

## Phase 1: Split helpers.rs into Modules

**Status:** DONE

**Goal:** Split `src/commands/helpers.rs` (2,621 lines) into focused modules.

### Target Module Structure
```
src/commands/
├── mod.rs              (existing, add module declarations)
├── helpers/
│   ├── mod.rs          (re-exports for backward compatibility)
│   ├── arg_parsing.rs  (~80 lines)
│   ├── metadata.rs     (~90 lines)
│   ├── output.rs       (~575 lines)
│   ├── fastx_io.rs     (~282 lines)
│   └── parquet_io.rs   (~910 lines)
├── classify.rs         (existing)
└── index.rs            (existing)
```

### Module Contents

**arg_parsing.rs** (lines 28-80):
- `parse_max_memory_arg()` - Parse byte sizes with "auto" support
- `parse_shard_size_arg()` - Parse byte sizes
- `parse_bloom_fpp()` - Validate false positive probability
- `validate_trim_to()` - Validate sequence trimming

**metadata.rs** (lines 83-173):
- `sanitize_bucket_name()` - Replace non-printable chars
- `resolve_bucket_id()` - Resolve bucket by ID or name
- `load_index_metadata()` - Load k, w, salt from manifest

**output.rs** (lines 458-1032):
- `OutputFormat` enum + impl
- `OutputWriter` enum + impl

**fastx_io.rs** (lines 176-456):
- `OwnedRecord` type alias
- `BatchResult` type alias
- `PrefetchingIoHandler` struct + impl

**parquet_io.rs** (lines 1034-1944):
- `is_parquet_input()`
- `ParquetInputReader` struct + impl
- `PrefetchingParquetReader` struct + impl
- `SequenceColumnRef` enum
- `batch_to_records_parquet*()` functions
- `stacked_batches_to_records()`

### Dependencies
```
arg_parsing.rs   → (standalone)
metadata.rs      → rype crate types
output.rs        → Arrow, Parquet, flate2
fastx_io.rs      → output.rs, needletail, rype::FirstErrorCapture
parquet_io.rs    → Arrow, Parquet, rayon, rype types
```

### Files to Modify
- `src/commands/helpers.rs` → delete after extraction
- `src/commands/mod.rs` → add `mod helpers;`
- Create: `src/commands/helpers/mod.rs`
- Create: `src/commands/helpers/arg_parsing.rs`
- Create: `src/commands/helpers/metadata.rs`
- Create: `src/commands/helpers/output.rs`
- Create: `src/commands/helpers/fastx_io.rs`
- Create: `src/commands/helpers/parquet_io.rs`

### Verification
```bash
cargo test
cargo clippy --all-features -- -D warnings
```

### Phase 1 TODOs
- [x] Create helpers/ directory structure
- [x] Extract arg_parsing.rs
- [x] Extract metadata.rs
- [x] Extract output.rs
- [x] Extract fastx_io.rs
- [x] Extract parquet_io.rs
- [x] Create mod.rs with re-exports
- [x] Delete old helpers.rs
- [x] Verify all tests pass (347 tests: 269 lib + 78 binary)
- [x] Verify clippy clean

---

## Phase 2: Consolidate Duplicate Validation

**Status:** DONE

**Goal:** Merge duplicate bucket name validation functions in `src/commands/index.rs`.

### Implementation
Replaced two duplicate functions with a single generic function:
```rust
fn validate_unique_bucket_names<'a>(names: impl Iterator<Item = &'a str>) -> Result<()>
```

**Call sites updated:**
- `validate_unique_bucket_names(buckets.iter().map(|b| b.bucket_name.as_str()))?;`
- `validate_unique_bucket_names(bucket_names_map.values().map(|s| s.as_str()))?;`

**Tests simplified:** Tests now pass string slices directly instead of constructing `BucketData` structs.

### Phase 2 TODOs
- [x] Create unified validation function
- [x] Update callers to use iterator pattern
- [x] Remove old functions
- [x] Verify tests pass (78 binary + 269 library)
- [x] Verify clippy clean

---

## Phase 3: Remove Dead Code and Stale Comments

**Status:** DONE

**Goal:** Remove genuinely unused code and delete the bloom filter TODO comment.

### Items to Remove

**Delete TODO comment:**
- `src/indices/inverted/query_loading.rs:270` - bloom filter parallelization TODO

**Remove unused methods (with tests confirming no usage):**
- `src/commands/helpers.rs` (or new location after Phase 1):
  - `OutputWriter::is_parquet()` (line 703)
- `src/core/ring_buffer.rs`:
  - `RingBuffer::is_empty()` (line 55)
  - `RingBuffer::len()` (line 62)

**Move or remove from binary:**
- `src/bin/orient_scale_test.rs:160` - `performance_test()` function

### Items to KEEP (document why)
These have `#[allow(dead_code)]` but are intentional:
- `ParquetInputReader` family - complete infrastructure for future Parquet input support
- `ClassifyAggregateArgs` - for pending aggregate command
- `reverse_complement()` in encoding.rs - used in tests
- `MacOsSysctl` in memory.rs - cross-platform support
- `batch_to_records_parquet()` - convenience wrapper
- `PrefetchingParquetReader::new()` - simpler API

### Files to Modify
- `src/indices/inverted/query_loading.rs` (delete TODO)
- `src/commands/helpers/output.rs` or `src/commands/helpers.rs` (remove is_parquet)
- `src/core/ring_buffer.rs` (remove is_empty, len)
- `src/bin/orient_scale_test.rs` (remove or move performance_test)

### Verification
```bash
cargo test
cargo clippy --all-features -- -D warnings
```

### Phase 3 TODOs
- [x] Delete bloom filter TODO comment
- [x] Remove OutputWriter::is_parquet()
- [x] Remove RingBuffer::is_empty() and ::len()
- [x] Handle orient_scale_test.rs performance_test() - N/A, file doesn't exist
- [x] Verify tests pass (347 tests: 269 lib + 78 binary)
- [x] Verify clippy clean

---

## Phase 4: Document Deferred Items

**Status:** DONE

**Goal:** Create `TECH_DEBT.md` documenting items not addressed in this plan.

### Items to Document

**Error Handling (Low Priority - Acceptable Current State):**
- `src/core/encoding.rs:59` - panic in `reverse_complement()`
  - Only called from `#[allow(dead_code)]` test helper
  - Function documented with `# Panics` section
  - Not in hot path
- `src/indices/inverted/mod.rs:133,161` - expect on CSR overflow
  - Would require 4 billion bucket IDs to trigger
  - Acceptable for current use case
- `src/c_api.rs:136` - expect on sanitized CString
  - String already sanitized, null bytes replaced
  - Documented panic is acceptable
- `src/indices/parquet/options.rs:132` - expect in to_writer_properties
  - Documented `# Panics` section, caller should validate first

**Large File Candidates (Future Refactoring):**
- `src/memory.rs` (1,566 lines) - Could split into estimation/detection/config
- `src/indices/inverted/query_loading.rs` (1,504 lines) - Could separate tests

**API Cleanup (Future):**
- 56 functions with `_parquet` suffix could be renamed since only Parquet format exists
- Would be a breaking change for any external callers

**Inconsistent Logging (Low Priority):**
- Mix of `eprintln!()`, `log::` macros, and custom `log_timing()`
- Would require audit of all logging call sites

**Test Coverage Gaps (Future):**
- C API thread safety - no concurrent stress tests
- Integration tests - no end-to-end workflow tests in tests/ directory

### Files to Create
- `TECH_DEBT.md` in project root

### Phase 4 TODOs
- [x] Create TECH_DEBT.md with all deferred items
- [x] Include rationale for deferral
- [x] Include suggested remediation for each item

---

## Verification Strategy

After each phase:
1. `cargo build --release` - Ensure compilation
2. `cargo test` - All tests pass
3. `cargo clippy --all-features -- -D warnings` - No new warnings
4. `cargo doc --no-deps` - Documentation builds

Final verification:
- Create a test index and classify reads to confirm identical output
- Compare before/after behavior with example files

---

## Files Modified Summary

| Phase | Files Modified | Files Created | Files Deleted |
|-------|---------------|---------------|---------------|
| 1 | src/commands/mod.rs | 6 new files in helpers/ | helpers.rs |
| 2 | src/commands/index.rs | - | - |
| 3 | query_loading.rs, ring_buffer.rs, output.rs, orient_scale_test.rs | - | - |
| 4 | - | TECH_DEBT.md | - |

---

## Rollback Strategy

Each phase is atomic. If issues arise:
1. `git checkout .` to revert all changes
2. Re-run `cargo test` to confirm clean state
3. Investigate issue before retrying

---

## Notes

- All changes are pure refactoring - no behavioral changes
- Re-exports in `helpers/mod.rs` maintain backward compatibility for any internal callers
- **All existing tests will be preserved** - tests are moved with their corresponding code to the new module locations. Tests will only be removed if they are genuinely not useful (e.g., testing removed dead code)
