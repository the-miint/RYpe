# Implementation Plan: `rype classify log-ratio` Mode

## Progress

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Core Ratio Computation | ✅ COMPLETE | 12 unit tests passing |
| Phase 2: CLI Arguments | ✅ COMPLETE | LogRatio variant added to ClassifyCommands |
| Phase 3: Handler Function | ✅ COMPLETE | ClassifyLogRatioArgs, validate_two_bucket_index, run_log_ratio, 5 unit tests |
| Phase 4: Wire Up Main | ✅ COMPLETE | LogRatio dispatch added to main.rs |
| Phase 5: Integration Tests | ✅ COMPLETE | 7 integration tests passing |
| Phase 6: Documentation | ⏳ NOT STARTED | |

---

## Summary

Add a new `rype classify log-ratio` subcommand that computes log10(score_A / score_B) between exactly two buckets for each read.

## Requirements

- Only works with indices having exactly 2 buckets
- Computes log10(numerator_score / denominator_score) for each read
- Default: lower bucket_id is numerator; `--swap-buckets` reverses this
- Edge cases: num=0 → 0.0; denom=0 & num>0 → +infinity; both=0 → 0.0
- Output formats: TSV, TSV.gz, Parquet (no wide format)
- bucket_name in output: `log10([BucketA] / [BucketB])`
- Honors the same performance options as `classify run`: `--parallel-rg`, `--use-bloom-filter`, `--batch-size`, `--max-memory`, `--parallel-input-rg`

## Out of Scope

- **C API**: Support for log-ratio mode in the C API (`src/c_api.rs`) will be implemented at a later time. This initial implementation is CLI-only.

---

## Phase 1: Core Ratio Computation (Unit Tests First) ✅ COMPLETE

### Files
- **Create**: `src/commands/helpers/log_ratio.rs` ✅
- **Modify**: `src/commands/helpers/mod.rs` (add module export) ✅

### Red: Write failing tests

```rust
// src/commands/helpers/log_ratio.rs
#[cfg(test)]
mod tests {
    #[test] fn test_compute_log_ratio_both_positive() { ... }
    #[test] fn test_compute_log_ratio_equal_scores() { ... }
    #[test] fn test_compute_log_ratio_numerator_zero() { ... }
    #[test] fn test_compute_log_ratio_denominator_zero() { ... }
    #[test] fn test_compute_log_ratio_both_zero() { ... }
    #[test] fn test_format_bucket_name() { ... }
    #[test] fn test_compute_from_hits_both_present() { ... }
    #[test] fn test_compute_from_hits_only_numerator() { ... }
    #[test] fn test_compute_from_hits_multiple_queries() { ... }
}
```

### Green: Implement functions

```rust
pub fn compute_log_ratio(numerator: f64, denominator: f64) -> f64
pub fn format_log_ratio_bucket_name(num_name: &str, denom_name: &str) -> String
pub fn compute_log_ratio_from_hits(results: &[HitResult], num_id: u32, denom_id: u32) -> Vec<LogRatioResult>
```

---

## Phase 2: CLI Arguments ✅ COMPLETE

### Files
- **Modify**: `src/commands/args.rs` (line ~270, ClassifyCommands enum) ✅

**Note**: Build currently fails with "non-exhaustive patterns" error because Phase 4 (wire up main.rs) has not been completed yet. This is expected.

### Red: Write failing tests

```rust
// tests/cli_integration_tests.rs
#[test] fn test_cli_log_ratio_subcommand_recognized() { ... }
#[test] fn test_cli_log_ratio_requires_two_buckets() { ... }
#[test] fn test_cli_log_ratio_swap_buckets_flag() { ... }
```

### Green: Add enum variant

```rust
// In ClassifyCommands enum
LogRatio {
    #[arg(short, long)] index: PathBuf,
    #[arg(short = '1', long)] r1: PathBuf,
    #[arg(short = '2', long)] r2: Option<PathBuf>,
    #[arg(short, long, default_value_t = 0.0)] threshold: f64,
    #[arg(long, default_value = "auto")] max_memory: usize,
    #[arg(short, long)] batch_size: Option<usize>,
    #[arg(short, long)] output: Option<PathBuf>,
    #[arg(long)] parallel_rg: bool,
    #[arg(long)] use_bloom_filter: bool,
    #[arg(long, default_value_t = 0)] parallel_input_rg: usize,
    #[arg(long)] timing: bool,
    #[arg(long)] trim_to: Option<usize>,
    #[arg(long)] swap_buckets: bool,
}
```

---

## Phase 3: Handler Function ✅ COMPLETE

### Files
- **Modify**: `src/commands/classify.rs` ✅
- **Modify**: `src/commands/mod.rs` (export new handler) ✅

### Tests implemented

```rust
// src/commands/classify.rs - in tests module
#[test] fn test_validate_two_bucket_index_passes() { ... }
#[test] fn test_validate_two_bucket_index_fails_one_bucket() { ... }
#[test] fn test_validate_two_bucket_index_fails_three_buckets() { ... }
#[test] fn test_validate_two_bucket_index_orders_by_id() { ... }
#[test] fn test_validate_two_bucket_index_fails_empty() { ... }
```

### Implementation

```rust
pub struct ClassifyLogRatioArgs { ... }

fn validate_two_bucket_index(bucket_names: &HashMap<u32, String>)
    -> Result<(u32, u32, String, String)>

pub fn run_log_ratio(args: ClassifyLogRatioArgs) -> Result<()>
```

Handler flow (implemented):
1. Load index manifest, validate num_buckets == 2
2. Load bucket_names, determine numerator/denominator based on bucket_id order
3. If `--swap-buckets`, swap numerator and denominator
4. Set up output writer (reuse OutputWriter)
5. For each batch:
   - Classify with threshold=0.0 to get all scores
   - Compute log ratios via `compute_log_ratio_from_hits()`
   - Filter by user threshold (if either original score >= threshold)
   - Format and write output
6. Finish writer

**Note**: Build currently fails with "non-exhaustive patterns" error because Phase 4 (wire up main.rs) has not been completed yet. This is expected.

---

## Phase 4: Wire Up Main

### Files
- **Modify**: `src/main.rs` (line ~242, Classify match arm)

### Green: Add dispatch

```rust
ClassifyCommands::LogRatio { index, r1, ... } => {
    run_log_ratio(ClassifyLogRatioArgs { ... })?;
}
```

---

## Phase 5: Integration Tests ✅ COMPLETE

### Files
- **Modify**: `tests/cli_integration_tests.rs` ✅

### Tests implemented

```rust
#[test] fn test_cli_log_ratio_end_to_end() { ... }           // Basic end-to-end test
#[test] fn test_cli_log_ratio_swap_buckets_negates() { ... } // Verify --swap-buckets negates output
#[test] fn test_cli_log_ratio_parquet_output() { ... }       // Parquet output format
#[test] fn test_cli_log_ratio_threshold_filters() { ... }    // Threshold filtering
#[test] fn test_cli_log_ratio_infinity_output() { ... }      // Infinity when one bucket has score 0
#[test] fn test_cli_log_ratio_fails_with_one_bucket() { ... }    // Error with 1-bucket index
#[test] fn test_cli_log_ratio_fails_with_three_buckets() { ... } // Error with 3-bucket index
```

---

## Phase 6: Documentation

### Files
- **Modify**: `CLAUDE.md` - Add to CLI usage section

---

## File Summary

| File | Action |
|------|--------|
| `src/commands/helpers/log_ratio.rs` | Create |
| `src/commands/helpers/mod.rs` | Modify (add export) |
| `src/commands/args.rs` | Modify (add LogRatio variant) |
| `src/commands/classify.rs` | Modify (add handler + args struct) |
| `src/commands/mod.rs` | Modify (export handler) |
| `src/main.rs` | Modify (add dispatch) |
| `tests/cli_integration_tests.rs` | Modify (add tests) |
| `CLAUDE.md` | Modify (add docs) |

---

## Verification

1. `cargo test log_ratio` - Unit tests for ratio computation
2. `cargo test test_cli_log_ratio` - CLI integration tests
3. `cargo test` - Full test suite
4. Manual test with 2-bucket index:
   ```bash
   # Create 2-bucket index
   cargo run --release -- index create -o test.ryxdi -r ref1.fasta -r ref2.fasta -k 32 -w 10

   # Run log-ratio
   cargo run --release -- classify log-ratio -i test.ryxdi -1 reads.fastq

   # Verify swap-buckets negates output
   cargo run --release -- classify log-ratio -i test.ryxdi -1 reads.fastq --swap-buckets
   ```
