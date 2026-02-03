# Refactoring Plan: Deduplicate `run_classify` and `run_log_ratio`

## Overview

Extract ~295 lines of duplicated code from `src/commands/classify.rs` into focused helper modules. The `run_classify` (lines 55-522) and `run_log_ratio` (lines 592-1027) functions share nearly identical infrastructure code.

**WORKFLOW**: Each phase requires explicit user approval before proceeding to the next.

## Phase Summary

| Phase | Focus | New File | LOC Saved | Status |
|-------|-------|----------|-----------|--------|
| 1 | Internal log_ratio fixes | - | ~40 | ✅ COMPLETE |
| 2 | Memory/batch sizing | `batch_config.rs` | ~80 | ✅ COMPLETE |
| 3 | Index loading | `index_loading.rs` | ~45 | ✅ COMPLETE |
| 4 | Input reader setup | `input_reader.rs` | ~30 | ✅ COMPLETE |
| 5 | Output formatting | `formatting.rs` | ~40 | ✅ COMPLETE |
| 6 | Args consolidation | modify `classify.rs` | ~20 | ✅ COMPLETE |

**Total: ~255 duplicated LOC eliminated**

---

## Phase 1: Fix Internal Duplication in `run_log_ratio`

**Status**: ✅ COMPLETE

Fix issues within `run_log_ratio` before extracting shared code.

### TODO
- [x] Add `format_log_ratio_output<S: AsRef<str>>()` generic function to `log_ratio.rs`
- [x] Add `filter_log_ratios_by_threshold()` function to `log_ratio.rs`
- [x] Replace `format_log_ratio_results` and `format_log_ratio_results_owned` closures in `classify.rs`
- [x] Replace duplicated threshold filtering blocks (lines 899-920 and 970-989) in `classify.rs`
- [x] Add unit tests for new functions
- [x] Run `cargo test log_ratio` - all tests pass (21 tests)
- [x] Run `cargo test test_cli_log_ratio` - all integration tests pass (7 tests)

### Changes

**1.1 Deduplicate format helpers with generics**

Current (lines 823-850): Two closures with identical bodies.

New function in `log_ratio.rs`:
```rust
pub fn format_log_ratio_output<S: AsRef<str>>(
    log_ratios: &[LogRatioResult],
    headers: &[S],
    ratio_bucket_name: &str,
) -> Vec<u8>
```

**1.2 Extract threshold filtering**

Current (lines 899-920 and 970-989): Same 20-line block copied twice.

New function in `log_ratio.rs`:
```rust
pub fn filter_log_ratios_by_threshold(
    log_ratios: &mut Vec<LogRatioResult>,
    original_results: &[HitResult],
    num_id: u32,
    denom_id: u32,
    threshold: f64,
)
```

### Files Modified
- `src/commands/helpers/log_ratio.rs` - Add 2 functions
- `src/commands/classify.rs` - Replace closures and duplicated filtering

---

## Phase 2: Extract Batch Size Computation

**Status**: ✅ COMPLETE

### TODO
- [x] Create `src/commands/helpers/batch_config.rs` with `BatchSizeConfig` struct
- [x] Implement `compute_effective_batch_size()` function
- [x] Add module export to `src/commands/helpers/mod.rs`
- [x] Replace batch sizing logic in `run_classify` (lines 93-175)
- [x] Replace batch sizing logic in `run_log_ratio` (lines 655-721)
- [x] Add unit tests for `compute_effective_batch_size()`
- [x] Run `cargo test` - all tests pass

### New File: `src/commands/helpers/batch_config.rs`

```rust
pub struct BatchSizeConfig<'a> {
    pub batch_size_override: Option<usize>,
    pub max_memory: usize,
    pub r1_path: &'a Path,
    pub r2_path: Option<&'a Path>,
    pub is_parquet_input: bool,
    pub index_path: &'a Path,
}

pub fn compute_effective_batch_size(config: &BatchSizeConfig) -> Result<usize>
```

### Files Modified
- `src/commands/helpers/batch_config.rs` - **Create** (~100 lines)
- `src/commands/helpers/mod.rs` - Add module export
- `src/commands/classify.rs` - Replace lines 93-175 and 655-721

---

## Phase 3: Extract Index Loading

**Status**: ✅ COMPLETE

### TODO
- [x] Create `src/commands/helpers/index_loading.rs`
- [x] Implement `LoadedIndex` and `IndexLoadOptions` structs
- [x] Implement `validate_parquet_index()` function
- [x] Implement `load_index_for_classification()` function
- [x] Add module export to `src/commands/helpers/mod.rs`
- [x] Replace index loading in `run_classify`
- [x] Replace index loading in `run_log_ratio`
- [x] Add unit tests for validation and error paths
- [x] Run `cargo test` - all tests pass (269 unit + 23 CLI + 7 log-ratio)

### New File: `src/commands/helpers/index_loading.rs`

```rust
pub struct LoadedIndex {
    pub metadata: IndexMetadata,
    pub sharded: ShardedInvertedIndex,
    pub read_options: Option<ParquetReadOptions>,
}

pub struct IndexLoadOptions {
    pub use_bloom_filter: bool,
    pub parallel_rg: bool,
}

pub fn validate_parquet_index(path: &Path) -> Result<()>
pub fn load_index_for_classification(path: &Path, options: &IndexLoadOptions) -> Result<LoadedIndex>
```

### Files Modified
- `src/commands/helpers/index_loading.rs` - **Create** (~90 lines)
- `src/commands/helpers/mod.rs` - Add module export
- `src/commands/classify.rs` - Replace index loading sections

---

## Phase 4: Extract Input Reader Setup

**Status**: ✅ COMPLETE

### TODO
- [x] Create `src/commands/helpers/input_reader.rs`
- [x] Implement `InputReaderConfig` struct
- [x] Implement `ClassificationInput` enum
- [x] Implement `validate_input_config()` function
- [x] Implement `create_input_reader()` function
- [x] Add module export to `src/commands/helpers/mod.rs`
- [x] Replace input setup in `run_classify`
- [x] Replace input setup in `run_log_ratio`
- [x] Add unit tests for validation error cases (4 tests)
- [x] Run `cargo test` - all tests pass (121 unit + 23 CLI + 7 log-ratio)

### New File: `src/commands/helpers/input_reader.rs`

```rust
pub struct InputReaderConfig<'a> {
    pub r1_path: &'a Path,
    pub r2_path: Option<&'a PathBuf>,
    pub batch_size: usize,
    pub parallel_input_rg: usize,
    pub is_parquet: bool,
}

pub enum ClassificationInput {
    Parquet(PrefetchingParquetReader),
    Fastx(PrefetchingIoHandler),
}

impl ClassificationInput {
    pub fn finish(&mut self) -> Result<()>
}

pub fn validate_input_config(is_parquet: bool, r2_path: Option<&PathBuf>) -> Result<()>
pub fn create_input_reader(config: &InputReaderConfig) -> Result<ClassificationInput>
```

### Changes
- Extracted duplicated input validation (Parquet + R2 incompatibility check)
- Extracted duplicated reader creation code (Parquet vs FASTX)
- Updated main loops to use `match` on `ClassificationInput` enum
- Unified `finish()` handling via enum method

### Files Modified
- `src/commands/helpers/input_reader.rs` - **Created** (~130 lines)
- `src/commands/helpers/mod.rs` - Add module export and re-exports
- `src/commands/classify.rs` - Replace input setup in both `run_classify` and `run_log_ratio`

---

## Phase 5: Extract Standard Formatting

**Status**: ✅ COMPLETE

### TODO
- [x] Create `src/commands/helpers/formatting.rs`
- [x] Implement `format_classification_results<S: AsRef<str>>()` function
- [x] Add module export to `src/commands/helpers/mod.rs`
- [x] Replace `format_results` and `format_results_ref` closures in `run_classify`
- [x] Add unit tests for formatting (5 tests)
- [x] Run `cargo test` - all tests pass (269 unit + 23 CLI + integration tests)

### New File: `src/commands/helpers/formatting.rs`

```rust
pub fn format_classification_results<S: AsRef<str>>(
    results: &[HitResult],
    headers: &[S],
    bucket_names: &HashMap<u32, String>,
) -> Vec<u8>
```

### Changes
- Created generic function that handles both `&[&str]` and `&[String]` header types
- Removed two nearly-identical closures from `run_classify` (~28 lines)
- Removed unused `std::io::Write` import from `classify.rs`

### Files Modified
- `src/commands/helpers/formatting.rs` - **Created** (~110 lines with tests)
- `src/commands/helpers/mod.rs` - Add module export
- `src/commands/classify.rs` - Replace formatting closures, remove unused import

---

## Phase 6: Consolidate Argument Structs

**Status**: ✅ COMPLETE

### TODO
- [x] Create `CommonClassifyArgs` struct in `classify.rs`
- [x] Refactor `ClassifyRunArgs` to use `common: CommonClassifyArgs`
- [x] Refactor `ClassifyLogRatioArgs` to use `common: CommonClassifyArgs`
- [x] Update `run_classify` to use `args.common.*`
- [x] Update `run_log_ratio` to use `args.common.*`
- [x] Update dispatch in `main.rs` to build common struct
- [x] Run `cargo test` - all tests pass (23 CLI integration + full suite)

### New Struct in `classify.rs`

```rust
pub struct CommonClassifyArgs {
    pub index: PathBuf,
    pub r1: PathBuf,
    pub r2: Option<PathBuf>,
    pub threshold: f64,
    pub max_memory: usize,
    pub batch_size: Option<usize>,
    pub output: Option<PathBuf>,
    pub parallel_rg: bool,
    pub use_bloom_filter: bool,
    pub parallel_input_rg: usize,
    pub trim_to: Option<usize>,
}

pub struct ClassifyRunArgs {
    pub common: CommonClassifyArgs,
    pub negative_index: Option<PathBuf>,
    pub best_hit: bool,
    pub wide: bool,
}

pub struct ClassifyLogRatioArgs {
    pub common: CommonClassifyArgs,
    pub swap_buckets: bool,
}
```

### Changes
- Created `CommonClassifyArgs` struct with 11 shared fields
- Refactored `ClassifyRunArgs` to embed `common: CommonClassifyArgs` + 3 unique fields
- Refactored `ClassifyLogRatioArgs` to embed `common: CommonClassifyArgs` + 1 unique field
- Updated all `args.field` references to `args.common.field` in both `run_classify` and `run_log_ratio`
- Updated `main.rs` dispatch to construct nested structs

### Files Modified
- `src/commands/classify.rs` - Add `CommonClassifyArgs`, restructure args structs, update field access
- `src/commands/mod.rs` - Export `CommonClassifyArgs`
- `src/main.rs` - Import `CommonClassifyArgs`, update dispatch to build nested structs

---

## Files Summary

| File | Action |
|------|--------|
| `src/commands/helpers/log_ratio.rs` | Modify (Phase 1) |
| `src/commands/helpers/batch_config.rs` | **Create** (Phase 2) |
| `src/commands/helpers/index_loading.rs` | **Create** (Phase 3) |
| `src/commands/helpers/input_reader.rs` | **Create** (Phase 4) |
| `src/commands/helpers/formatting.rs` | **Create** (Phase 5) |
| `src/commands/helpers/mod.rs` | Modify (Phases 2-5) |
| `src/commands/classify.rs` | Modify (all phases) |
| `src/main.rs` | Modify (Phase 6) |

---

## Verification (run after each phase)

```bash
# Unit tests for affected modules
cargo test log_ratio
cargo test batch_config
cargo test index_loading
cargo test input_reader
cargo test formatting

# Full test suite
cargo test

# Integration tests specifically
cargo test test_cli_log_ratio
cargo test test_cli_classify

# Manual verification
cargo run --release -- classify log-ratio -i test.ryxdi -1 reads.fastq
cargo run --release -- classify run -i test.ryxdi -1 reads.fastq -t 0.1
```

---

## Out of Scope (Deferred)

**Unified batch processing loop** - The core classification loops have fundamentally different strategies. Merging would require trait objects or complex generics with low ROI.
