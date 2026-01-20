# Rype Refactoring Plan

## Overview
Address technical debt: fix code duplication, break apart monolithic files, make parquet non-optional, centralize constants, and improve error handling consistency.

## Target Directory Structure
```
src/
├── core/                    # Fundamental algorithms
│   ├── mod.rs
│   ├── encoding.rs          # from src/encoding.rs
│   ├── extraction.rs        # from src/extraction.rs
│   └── workspace.rs         # from src/workspace.rs
├── indices/                 # Index structures
│   ├── mod.rs
│   ├── main.rs              # from src/index.rs
│   ├── inverted/
│   │   ├── mod.rs
│   │   ├── core.rs          # InvertedIndex struct, build_from_*, get_bucket_hits
│   │   ├── io.rs            # save_shard, load_shard_*
│   │   └── query.rs         # QueryInvertedIndex
│   ├── sharded.rs           # from src/sharded.rs
│   ├── sharded_main.rs      # from src/sharded_main.rs
│   └── parquet/
│       ├── mod.rs
│       ├── manifest.rs      # ParquetManifest, BucketMetadata, BucketData
│       ├── options.rs       # ParquetWriteOptions, ParquetReadOptions
│       ├── io.rs            # write_buckets_parquet, read_buckets_parquet
│       └── streaming.rs     # create_parquet_inverted_index, stream_to_*
├── classify/                # Classification strategies
│   ├── mod.rs
│   ├── batch.rs             # classify_batch, aggregate_batch
│   ├── sharded.rs           # classify_batch_sharded_sequential, _merge_join, _main
│   ├── merge_join.rs        # classify_batch_merge_join, merge_join, gallop_join
│   └── scoring.rs           # compute_score, filter_negative_mins, accumulate_*
├── commands/                # CLI command handlers
│   ├── mod.rs
│   ├── args.rs              # Cli, Commands, IndexCommands, ClassifyCommands enums
│   ├── index.rs             # Index subcommand handlers
│   ├── classify.rs          # Classify subcommand handlers
│   └── inspect.rs           # Inspect subcommand handlers
├── arrow/                   # Arrow integration (remains optional)
├── c_api.rs
├── config.rs
├── constants.rs             # Expanded with all magic numbers
├── error.rs                 # NEW: unified RypeError type
├── memory.rs
├── types.rs
├── lib.rs
└── main.rs                  # Thin entry point (~50 lines)
```

---

## Phase 1: Make Parquet Non-Optional ✅ COMPLETE

**Commit:** 3522b07

### 1.1 Update Cargo.toml
**File:** `Cargo.toml`

Changes:
- Move `parquet` and `bytes` from optional to required dependencies
- Keep `arrow` optional
- Remove `required-features = ["parquet"]` from examples
- Update features section

```toml
[dependencies]
parquet = { version = "57", default-features = false, features = ["zstd", "snap", "arrow"] }
bytes = "1.5"
arrow = { version = "57", default-features = false, features = ["ffi"], optional = true }

[features]
default = []
arrow = ["dep:arrow"]
```

### 1.2 Remove Feature Gates from Library
**Files to modify:**
- `src/lib.rs:47-57` - Remove `#[cfg(feature = "parquet")]` from module and re-exports
- `src/parquet_index.rs` - Remove all `#[cfg(feature = "parquet")]` attributes
- `src/inverted.rs:467-486, 666-684, 700-815, 829-1100` - Remove parquet feature gates
- `src/sharded.rs:552-601, 706-752` - Remove parquet feature gates and non-parquet fallbacks
- `src/main.rs:18-19, 740+` - Remove parquet feature gates
- `src/c_api.rs` - Keep Arrow optional but remove parquet gates

### 1.3 Fix Classification Code Duplication
**File:** `src/classify.rs`

Remove duplicated functions:
- Delete non-parquet `classify_batch_sharded_sequential` (lines 267-358)
- Delete non-parquet `classify_batch_sharded_merge_join` (lines 478-559)
- Remove `#[cfg(feature = "parquet")]` from remaining versions
- Keep `read_options: Option<&ParquetReadOptions>` parameter (callers pass `None`)

**Verification:**
```bash
cargo test
cargo build --release
cargo build --release --features arrow
```

---

## Phase 2: Centralize Magic Numbers ✅ COMPLETE

**Commit:** 3522b07

**File:** `src/constants.rs`

**Actual changes made:**
- Renamed `INDEX_MAGIC` → `SINGLE_FILE_INDEX_MAGIC` for clarity
- Renamed `INDEX_VERSION` → `SINGLE_FILE_INDEX_VERSION` for clarity
- Fixed duplicate `SOURCE_DELIM` definition (now `BUCKET_SOURCE_DELIM` only)
- Changed `BUCKET_SOURCE_DELIM` visibility from `pub` to `pub(crate)`
- Moved `BYTES_PER_MINIMIZER_*` from `sharded_main.rs` to `constants.rs`
- Moved `MIN_ENTRIES_PER_PARALLEL_PARTITION` from `parquet_index.rs` to `constants.rs`
- Added tests for constant invariants (overflow protection, sane values)
- Documented constants module in CLAUDE.md

**File:** `src/constants.rs` (original plan)

Add constants from scattered locations:

```rust
// I/O buffer sizes
pub(crate) const WRITE_BUF_SIZE: usize = 8 * 1024 * 1024;  // 8MB
pub(crate) const READ_BUF_SIZE: usize = 8 * 1024 * 1024;   // 8MB

// Binary format magic bytes
pub(crate) const INDEX_MAGIC: &[u8; 4] = b"RYP5";
pub(crate) const INDEX_VERSION: u32 = 5;
pub(crate) const SHARD_MAGIC: &[u8; 4] = b"RYXS";
pub(crate) const SHARD_VERSION: u32 = 1;
pub(crate) const MANIFEST_MAGIC: &[u8; 4] = b"RYXM";
pub(crate) const MANIFEST_VERSION: u32 = 3;
pub(crate) const MAIN_MANIFEST_MAGIC: &[u8; 4] = b"RYPM";
pub(crate) const MAIN_SHARD_MAGIC: &[u8; 4] = b"RYPS";

// Parquet format
pub(crate) const PARQUET_FORMAT_MAGIC: &str = "RYPE_PARQUET_V1";
pub(crate) const PARQUET_FORMAT_VERSION: u32 = 1;
pub(crate) const PARQUET_MANIFEST_FILE: &str = "manifest.toml";
pub(crate) const PARQUET_BUCKETS_FILE: &str = "buckets.parquet";
pub(crate) const PARQUET_INVERTED_DIR: &str = "inverted";

// Sharded index limits
pub(crate) const MAX_SHARDS: u32 = 10_000;
pub(crate) const MAX_MAIN_SHARDS: u32 = 10_000;
pub(crate) const MAX_STRING_TABLE_BYTES: usize = 100_000_000;
pub(crate) const MAX_STRING_TABLE_ENTRIES: u32 = 10_000_000;
pub(crate) const MAX_SOURCES_PER_BUCKET: usize = 100_000_000;

// Size estimates
pub(crate) const BYTES_PER_MINIMIZER_COMPRESSED: usize = 4;
pub(crate) const BYTES_PER_MINIMIZER_MEMORY: usize = 8;

// Batch processing
pub(crate) const PARQUET_BATCH_SIZE: usize = 100_000;
pub(crate) const ROW_GROUP_SIZE: usize = 100_000;
pub(crate) const MIN_ENTRIES_PER_PARALLEL_PARTITION: usize = 1_000_000;

// Classification tuning
pub(crate) const QUERY_HASHSET_THRESHOLD: usize = 1000;
pub(crate) const GALLOP_THRESHOLD: usize = 16;
pub(crate) const ESTIMATED_BUCKETS_PER_READ: usize = 4;

// Delimiters
pub(crate) const SOURCE_DELIM: &str = "::";

// C API limits
pub(crate) const MAX_SEQUENCE_LENGTH: usize = 2_000_000_000;

// QueryInvertedIndex bit-packing
pub(crate) const READ_INDEX_MASK: u32 = 0x7FFFFFFF;
pub(crate) const RC_FLAG_BIT: u32 = 0x80000000;
pub(crate) const MAX_READS: usize = 0x7FFFFFFF;
```

Update all files to use constants from this module.

**Verification:**
```bash
cargo test
cargo clippy -- -D warnings
```

---

## Phase 3: Create Unified Error Type ✅ COMPLETE

**Commit:** e885d76

**New file:** `src/error.rs`

**Actual implementation:**
- Created `RypeError` enum with variants: `Io`, `Format`, `Validation`, `Parquet`, `Encoding`, `Overflow`
- Implemented `Display`, `Error`, and `From` traits
- Added helper constructors for ergonomic error creation
- Added conversions from `std::io::Error`, `parquet::errors::ParquetError`, `arrow::error::ArrowError`
- Re-exported as `RypeError` and `RypeResult` from `lib.rs`
- Added 6 unit tests for error display and source chain
- Library code can use `RypeError`, CLI continues using `anyhow::Result`

**New file:** `src/error.rs` (original plan)

```rust
use std::path::PathBuf;

#[derive(Debug)]
pub enum RypeError {
    Io { path: PathBuf, source: std::io::Error },
    Format { path: PathBuf, detail: String },
    Validation(String),
    Parquet(String),
    Encoding(String),
}

impl std::fmt::Display for RypeError { ... }
impl std::error::Error for RypeError { ... }
impl From<std::io::Error> for RypeError { ... }

pub type Result<T> = std::result::Result<T, RypeError>;
```

Usage:
- Library code (`indices/`, `classify/`, `core/`) uses `RypeError`
- CLI code (`commands/`, `main.rs`) continues using `anyhow::Result`
- C API converts `RypeError` to thread-local string

**Verification:**
```bash
cargo test
cargo doc --no-deps
```

---

## Phase 4: Break Apart Monolithic Files ✅ COMPLETE

**Commit:** 4f354c9

### Actual Implementation

**4.1 core/ Directory** ✅
- Moved `encoding.rs`, `extraction.rs`, `workspace.rs` to `src/core/`
- Created `src/core/mod.rs` with re-exports

**4.2 indices/ Directory** ✅
- Moved files intact (did not split inverted.rs or parquet_index.rs into sub-modules as originally planned):
  - `src/index.rs` → `src/indices/main.rs`
  - `src/inverted.rs` → `src/indices/inverted.rs`
  - `src/sharded.rs` → `src/indices/sharded.rs`
  - `src/sharded_main.rs` → `src/indices/sharded_main.rs`
  - `src/parquet_index.rs` → `src/indices/parquet.rs`
- Created `src/indices/mod.rs` with re-exports

**4.3 classify/ Directory** ✅
- Split `src/classify.rs` into submodules:
  - `classify/batch.rs`: classify_batch, aggregate_batch
  - `classify/sharded.rs`: classify_batch_sharded_sequential, _merge_join, _main
  - `classify/merge_join.rs`: classify_batch_merge_join, merge_join, gallop_join
  - `classify/scoring.rs`: compute_score
  - `classify/common.rs`: filter_negative_mins (ENABLE_TIMING/log_timing moved to lib.rs)
- Tests moved with their functions

**4.4 commands/ Directory** ✅
- Split `src/main.rs` into:
  - `commands/args.rs`: CLI argument definitions (~548 lines)
  - `commands/helpers.rs`: IoHandler, parsing utilities (~178 lines)
  - `commands/index.rs`: Index command handlers (~1195 lines)
  - `commands/inspect.rs`: Inspect command handlers (~292 lines)
- main.rs reduced from ~3944 to ~1501 lines

**4.5 lib.rs Updates** ✅
- Reorganized re-exports into "Essential" vs "Specialized" sections
- Moved ENABLE_TIMING and log_timing to lib.rs as cross-cutting utilities
- Maintained backward-compatible public API

**Code Review Improvements:**
- Moved `accumulate_match` to `InvertedIndex::accumulate_hits_for_match()` method
- Changed helper visibility from `pub(crate)` to `pub(super)` for module-private functions
- Standardized imports: `super::` for siblings, `crate::` for distant modules

**Final Structure:**
```
src/
├── core/
│   ├── mod.rs
│   ├── encoding.rs
│   ├── extraction.rs
│   └── workspace.rs
├── indices/
│   ├── mod.rs
│   ├── main.rs
│   ├── inverted.rs
│   ├── sharded.rs
│   ├── sharded_main.rs
│   └── parquet.rs
├── classify/
│   ├── mod.rs
│   ├── batch.rs
│   ├── common.rs
│   ├── merge_join.rs
│   ├── scoring.rs
│   └── sharded.rs
├── commands/
│   ├── mod.rs
│   ├── args.rs
│   ├── helpers.rs
│   ├── index.rs
│   └── inspect.rs
├── lib.rs
└── main.rs
```

All 201+ tests pass.

---

## Implementation Order

```
Phase 1.1 (Cargo.toml)
    ↓
Phase 1.2 (remove feature gates) + Phase 1.3 (fix duplication)
    ↓
Phase 2 (constants.rs)
    ↓
Phase 3 (error.rs)
    ↓
Phase 4.1 (core/) → 4.2 (indices/) → 4.3 (classify/) → 4.4 (commands/) → 4.5 (lib.rs)
```

---

## Verification Plan

After each phase:
```bash
cargo test
cargo build --release
```

Final verification:
```bash
cargo test --all-features
cargo clippy --all-features -- -D warnings
cargo doc --no-deps

# Run examples
cargo run --example parquet_poc
cargo run --example bench_e2e_classification

# CLI smoke test
cargo run --release -- index stats -i test_data/index.ryidx
```

---

## Risk Mitigation

1. **API Compatibility**: Use `pub use` re-exports to maintain existing public API
2. **Incremental Changes**: Complete one phase before starting next; commit after each
3. **Test Coverage**: Run full test suite after each file move
4. **Git History**: Keep commits atomic and well-described for easy rollback
5. **Preserve All Tests**: All existing tests will be preserved and moved with their associated code (inline `#[cfg(test)]` modules stay with their source)

## Critical Files
- `src/classify.rs` - Duplicate removal target
- `src/inverted.rs` - Largest file (3465 lines), most complex split
- `src/lib.rs` - Central module declarations
- `src/constants.rs` - Single source of truth for magic numbers
- `Cargo.toml` - Dependency changes

---

## Completion Summary ✅

**All 4 phases completed successfully.**

| Phase | Description | Commit |
|-------|-------------|--------|
| 1 | Make Parquet Non-Optional | 3522b07 |
| 2 | Centralize Magic Numbers | 3522b07 |
| 3 | Create Unified Error Type | e885d76 |
| 4 | Break Apart Monolithic Files | 4f354c9 |

**Key Metrics:**
- main.rs: 3944 → 1501 lines (-62%)
- classify.rs: Split into 6 focused modules
- All 201+ tests passing
- Backward-compatible public API maintained
- Pre-commit hooks (fmt + clippy) passing

**Deferred Work:**
- Further splitting of `indices/inverted.rs` (1750+ lines) and `indices/parquet.rs` (1400+ lines) into sub-modules was deemed unnecessary for now - the files are well-organized internally
- Additional error type migration from `anyhow::Result` to `RypeResult` in library code could be done incrementally
