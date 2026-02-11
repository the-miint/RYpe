# Plan: Add `--minimum-length` filter and fix Parquet trim OOM bug

## Context

Large `--trim-to` values cause OOM on Parquet input because the memory estimator budgets I/O buffers using the **trimmed** read length, but the Parquet prefetch channel (`sync_channel(4)`) holds **full-length** Arrow `RecordBatch` objects. This causes the computed batch_size to be inflated far beyond what memory can handle.

The fix: move trimming/filtering into the Parquet reader thread so the prefetch channel only ever holds trimmed owned records. Additionally, add a new `--minimum-length` filter (applied before `--trim-to`) that skips reads shorter than N bases, supported for both FASTX and Parquet input.

## Key Files

| File | Role |
|------|------|
| `src/commands/helpers/arg_parsing.rs` | Validation for `--minimum-length` |
| `src/commands/args.rs` | CLI arg definition |
| `src/main.rs` | Thread args to internal structs |
| `src/commands/classify.rs` | Main classify loops, internal arg structs |
| `src/commands/helpers/fastx_io.rs` | FASTX reader thread — add minimum_length |
| `src/commands/helpers/parquet_io.rs` | `ParquetBatch` enum, channel type change, trim in reader thread, `batch_to_owned_records_trimmed` gains minimum_length |
| `src/commands/helpers/input_reader.rs` | `InputReaderConfig` + `create_input_reader` — pass trim_to/minimum_length to Parquet reader |
| `src/commands/helpers/batch_config.rs` | Wire minimum_length, set `trimmed_in_reader` flag |
| `src/memory.rs` | `InputFormat::Parquet` gains `trimmed_in_reader` flag for correct buffer estimate |
| `tests/cli_integration_tests.rs` | Integration tests |

## Scope

- **In scope:** `classify run` and `classify log-ratio` subcommands.
- **Out of scope:** `classify aggregate` — it does not have `--trim-to` today and does not need `--minimum-length`.

## Processing Order

For both FASTX and Parquet:
1. Check `minimum_length` on **original** R1 length → skip if too short
2. Check `trim_to` on **original** R1 length → skip if too short
3. Truncate R1 (and R2) to `trim_to` length

**Edge case:** `--minimum-length 100 --trim-to 50` — minimum_length is the binding constraint (all reads passing it are ≥ 100 > 50, so the trim_to skip is redundant). Both checks are applied; the stricter one wins. This must be an explicit test case.

## Compatibility Notes

- **`--minimum-length` + `--output-sequences` is allowed.** Unlike `--trim-to`, minimum_length only filters (skips) reads — surviving reads are complete and unmodified. No change needed to `validate_seq_output`.
- **`--minimum-length` alone (no `--trim-to`) triggers the Owned path for Parquet**, losing zero-copy. This is an acceptable trade-off for correctness. Future optimization: use `arrow::compute::filter_record_batch` to stay in Arrow land when only filtering (no trimming).

## Cycle Status

| Cycle | Description | Status |
|-------|-------------|--------|
| 1 | Arg validation (`arg_parsing.rs`) | done |
| 2 | CLI arg definition (`args.rs`, `main.rs`, `classify.rs`) | done |
| 3 | FASTX reader minimum_length (`fastx_io.rs`) | done |
| 4 | `InputReaderConfig` + `create_input_reader` (`input_reader.rs`) | done |
| 5 | `batch_to_owned_records_trimmed` gains minimum_length (`parquet_io.rs`) | done |
| 6 | `ParquetBatch` enum (`parquet_io.rs`) | done |
| 7 | Parquet reader threads trim at parse (`parquet_io.rs`) | done |
| 8 | Wire Parquet reader creation (`input_reader.rs`, `classify.rs`) | done |
| 9 | Classify main loops + accumulation helper (`classify.rs`) | done |
| 10 | Memory estimation fix (`memory.rs`) | done |
| 11 | Wire batch sizing (`batch_config.rs`) | done |
| 12 | Integration tests (`cli_integration_tests.rs`) | done |

## TDD Cycles

### Cycle 1: Arg validation (`arg_parsing.rs`)

**Status:** done

**Red:** `test_validate_minimum_length_*` — call `validate_minimum_length("100")` → Ok(100), `"0"` → Err, `"abc"` → Err. Fails: function doesn't exist.

**Green:** Add `validate_minimum_length()` with same logic as `validate_trim_to`.

**Refactor:** Extract shared `validate_positive_length(s, flag_name)` helper to deduplicate both validators.

**Done when:** `cargo test validate_minimum_length` passes; `cargo test validate_trim_to` still passes (no regression from refactor).

---

### Cycle 2: CLI arg definition (`args.rs`, `main.rs`, `classify.rs`)

**Status:** done

**Red:** Integration test `test_cli_minimum_length_argument_parsing` — run binary with `--minimum-length 50`. Fails: clap doesn't know the flag.

**Green:**
- Add `minimum_length: Option<usize>` to `ClassifyCommands::Run` and `ClassifyCommands::LogRatio` in `args.rs`
- Add `minimum_length` to `CommonClassifyArgs` and `ClassifyLogRatioArgs` in `classify.rs`
- Thread through `main.rs` match arms

**Done when:** `cargo test test_cli_minimum_length_argument_parsing` passes; `cargo test` has no compilation errors.

---

### Cycle 3: FASTX reader minimum_length (`fastx_io.rs`)

**Status:** done

**Red:** Unit tests:
- `test_fastx_reader_minimum_length_filters_short_reads` — 3 reads (30bp, 80bp, 50bp), min_length=50 → 2 records
- `test_fastx_reader_minimum_length_before_trim_to` — min_length=50, trim_to=70 → 40bp skipped, 100bp trimmed to 70, 60bp trimmed to 60
- `test_fastx_reader_minimum_length_paired_end` — R1=40bp pair skipped, R1=60bp pair kept
- `test_fastx_reader_minimum_length_gt_trim_to` — min_length=100, trim_to=50 → reads ≥ 100bp kept and trimmed to 50; reads < 100bp skipped

**Green:** Add `minimum_length: Option<usize>` to `with_options()` and `reader_thread()`. Add check BEFORE existing `trim_to` check:
```rust
if let Some(min_len) = minimum_length {
    if s1_seq.len() < min_len {
        // skip (consume R2 if paired)
        continue;
    }
}
```

**Refactor:** Extract shared "skip and consume R2" closure to deduplicate with the trim_to skip block.

**Note on constructor chain:** `PrefetchingIoHandler` has `new()` → `with_trim()` → `with_options()`. Adding `minimum_length` to `with_options` is sufficient. `with_trim` passes `None` for `minimum_length`.

**Done when:** All 4 unit tests pass; existing `trim_to` FASTX tests still pass.

---

### Cycle 4: `InputReaderConfig` + `create_input_reader` (`input_reader.rs`)

**Status:** done

**Green:** Add `minimum_length: Option<usize>` to `InputReaderConfig`. Pass to FASTX handler. Update all construction sites in `classify.rs`.

**Done when:** `cargo test` compiles and all existing tests pass.

---

### Cycle 5: `batch_to_owned_records_trimmed` gains minimum_length (`parquet_io.rs`)

**Status:** done

**Red:** Unit tests:
- `test_batch_to_owned_records_with_minimum_length` — 30bp, 80bp, 50bp; min_length=50 → 2 records
- `test_batch_to_owned_records_minimum_length_before_trim` — min_length=50, trim_to=70 → correct filtering and trimming
- `test_batch_to_owned_records_minimum_length_with_paired` — pair skipped when R1 < min_length
- `test_batch_to_owned_records_minimum_length_gt_trim_to` — min_length=100, trim_to=50 → only reads ≥ 100bp kept, trimmed to 50

**Green:** Add `minimum_length: Option<usize>` parameter. Add check before the `trim_to` check in the per-row loop.

**Refactor:** Update `read_parquet_batch_trimmed` signature and all callers to pass `minimum_length` (can be `None` at existing call sites for now).

**Done when:** All 4 new unit tests pass; all existing `batch_to_owned_records_trimmed` tests still pass.

---

### Cycle 6: `ParquetBatch` enum (`parquet_io.rs`)

**Status:** done

**Red:** `test_parquet_batch_enum_*` — construct `ParquetBatch::Arrow` and `ParquetBatch::Owned`, verify helpers.

**Green:** Define enum:
```rust
pub enum ParquetBatch {
    Arrow(RecordBatch, Vec<String>),
    Owned(Vec<OwnedFastxRecord>, Vec<String>),
}
```
Change channel type alias. Update `next_batch()` return type.

**Done when:** Enum tests pass; `cargo test` compiles. Existing callers of `next_batch()` may need temporary updates to destructure the new type.

---

### Cycle 7: Parquet reader threads trim at parse (`parquet_io.rs`)

**Status:** done

**Red:** Tests:
- `test_prefetching_parquet_reader_trims_in_reader_thread` — create Parquet with 30bp, 100bp, 60bp; reader with trim_to=50, min_length=40. `next_batch()` returns `Owned` variant with filtered/trimmed records.
- `test_prefetching_parquet_reader_parallel_trims_in_reader_thread` — same but with `parallel_row_groups=Some(2)`. Verify the parallel path also produces `Owned` variant with correct filtering/trimming.
- `test_prefetching_parquet_reader_no_filter_returns_arrow` — reader with trim_to=None, min_length=None returns `Arrow` variant (zero-copy path preserved).

**Green:**
- Add `trim_to` and `minimum_length` to `PrefetchingParquetReader::with_parallel_row_groups()` (and `new()`)
- In `reader_thread()`: if trim/filter active, call `batch_to_owned_records_trimmed` and send `ParquetBatch::Owned`; otherwise send `ParquetBatch::Arrow`
- In `reader_thread_parallel()`: same logic. The rayon closure return type changes from `Vec<(RecordBatch, Vec<String>)>` to `Vec<ParquetBatch>`. After sorting by row group index, send each `ParquetBatch` through the channel.
- Each `batch_to_owned_records_trimmed` call in the reader thread uses `id_offset=0` (batch-local IDs). ID remapping happens during accumulation (Cycle 9).

**Done when:** All 3 tests pass; `cargo test` compiles.

---

### Cycle 8: Wire Parquet reader creation (`input_reader.rs`, `classify.rs`)

**Status:** done

**Green:**
- Pass `config.trim_to` and `config.minimum_length` to `PrefetchingParquetReader` in `create_input_reader()`
- Remove the conditional `trim_to: if input_is_parquet { None } else { ... }` in `classify.rs` — now both paths pass through the reader config

**Done when:** `cargo test` compiles and all existing tests pass.

---

### Cycle 9: Classify main loops + accumulation helper (`classify.rs`)

**Status:** done

This is the largest cycle. It replaces `read_parquet_batch_trimmed` with a new accumulation pattern that handles the `ParquetBatch` enum.

**Red:** Integration test `test_classify_run_with_parquet_trim` — Parquet input with mixed lengths + `--trim-to`. Verify correct output.

**Green:**

**Step 9a: `accumulate_owned_batches` helper** (in `parquet_io.rs`)

Replace `read_parquet_batch_trimmed` with:

```rust
pub fn accumulate_owned_batches(
    reader: &mut PrefetchingParquetReader,
    target_batch_size: usize,
) -> Result<TrimmedBatchResult> {
    let mut records: Vec<OwnedFastxRecord> = Vec::new();
    let mut headers: Vec<String> = Vec::new();
    let mut reached_end = false;
    let mut rg_count = 0usize;

    while records.len() < target_batch_size {
        match reader.next_batch()? {
            Some(ParquetBatch::Owned(mut batch_records, batch_headers)) => {
                rg_count += 1;
                // Remap query_ids to be globally sequential across accumulated batches
                let offset = records.len() as i64;
                for rec in &mut batch_records {
                    rec.query_id += offset;
                }
                records.extend(batch_records);
                headers.extend(batch_headers);
            }
            Some(ParquetBatch::Arrow(..)) => {
                unreachable!("Expected Owned variant when trim/filter is active");
            }
            None => {
                reached_end = true;
                break;
            }
        }
    }

    Ok(TrimmedBatchResult { records, headers, rg_count, reached_end })
}
```

**Critical detail: query ID remapping.** Each `Owned` batch arrives from the reader thread with 0-based IDs. The accumulation loop adds `records.len()` as offset before extending, producing globally-sequential IDs (0, 1, 2, ...) across all accumulated row groups. Without this, every batch has overlapping IDs and classification results get scrambled.

**Step 9b: Update `run_classify()` and `run_log_ratio()`**

- Use `args.common.trim_to.is_some() || args.common.minimum_length.is_some()` to decide between Owned vs Arrow path upfront (same condition the reader uses to decide which variant to produce)
- Owned path → call `accumulate_owned_batches`
- Arrow path → zero-copy stacking (unchanged)

**Refactor:** Remove `read_parquet_batch_trimmed()` — fully replaced by `accumulate_owned_batches`.

**Done when:** Integration test passes; both `run_classify` and `run_log_ratio` Parquet paths work with trim_to; existing integration tests for `--trim-to` still pass.

---

### Cycle 10: Memory estimation fix (`memory.rs`)

**Status:** done

**Red:** Tests:
- `test_parquet_trimmed_uses_owned_format_for_io_buffer_estimate` — `InputFormat::Parquet { trimmed_in_reader: true }` uses `estimate_owned_record_bytes`, not `estimate_arrow_bytes_per_row`.
- `test_parquet_untrimmed_uses_arrow_estimate` — `InputFormat::Parquet { trimmed_in_reader: false }` uses `estimate_arrow_bytes_per_row` (regression guard).
- `test_parquet_trimmed_prefetch_slots_still_four` — `Parquet { trimmed_in_reader: true }` still returns 4 prefetch slots (not 2 like FASTX).
- `test_parquet_trimmed_no_arrow_builder_overhead` — verify `ARROW_BUILDER_OVERHEAD` is NOT applied when `trimmed_in_reader` is true.

**Green:** Add `trimmed_in_reader: bool` to `InputFormat::Parquet`. Update `estimate_buffer_bytes_per_row`:
```rust
Parquet { is_paired, trimmed_in_reader: true } => profile.estimate_owned_record_bytes(*is_paired),
Parquet { is_paired, trimmed_in_reader: false } => profile.estimate_arrow_bytes_per_row(*is_paired),
```
In `estimate_io_buffer_memory`, skip `ARROW_BUILDER_OVERHEAD` when `trimmed_in_reader` is true.

**Refactor:** Update all `InputFormat::Parquet` construction sites and match arms across the codebase.

**Done when:** All 4 memory tests pass; all existing memory tests pass; `cargo test` clean.

---

### Cycle 11: Wire batch sizing (`batch_config.rs`)

**Status:** done

**Green:**
- Add `minimum_length: Option<usize>` to `BatchSizeConfig`
- Set `trimmed_in_reader: config.trim_to.is_some() || config.minimum_length.is_some()`
- Update all call sites in `classify.rs`

**Done when:** `cargo test` compiles and passes; batch sizing tests pass.

---

### Cycle 12: Integration tests (`cli_integration_tests.rs`)

**Status:** done

Tests:
- `test_cli_minimum_length_skips_short_reads_fastx` — FASTQ with mixed lengths, `--minimum-length 50`
- `test_cli_minimum_length_before_trim_to_fastx` — `--minimum-length 50 --trim-to 70`
- `test_cli_minimum_length_gt_trim_to_fastx` — `--minimum-length 100 --trim-to 50` (minimum_length is binding)
- `test_cli_minimum_length_with_parquet_input` — Parquet input + `--minimum-length 50`
- `test_cli_minimum_length_log_ratio` — log-ratio with `--minimum-length 50`
- `test_cli_minimum_length_with_output_sequences` — `--minimum-length 50 --output-sequences out.fq.gz` succeeds (not rejected like `--trim-to` would be)

**Done when:** All 6 integration tests pass; full `cargo test` is clean.

---

## Verification

```bash
# Full test suite (run at end of each cycle)
cargo test

# Targeted test commands per cycle
cargo test validate_minimum_length                          # Cycle 1
cargo test test_cli_minimum_length_argument_parsing         # Cycle 2
cargo test test_fastx_reader_minimum_length                 # Cycle 3
cargo test test_batch_to_owned_records_minimum_length       # Cycle 5
cargo test test_parquet_batch_enum                          # Cycle 6
cargo test test_prefetching_parquet_reader                  # Cycle 7
cargo test test_classify_run_with_parquet_trim              # Cycle 9
cargo test test_parquet_trimmed                             # Cycle 10
cargo test --test cli_integration_tests minimum_length      # Cycle 12
```

## Future Optimizations (Out of Scope)

- **Arrow-native filtering for `--minimum-length` without `--trim-to`:** Use `arrow::compute::filter_record_batch` with a boolean mask to filter short reads while staying in Arrow land, preserving zero-copy benefits. Currently we convert to `OwnedFastxRecord` which loses zero-copy.
- **`classify aggregate` support:** Add `--minimum-length` (and possibly `--trim-to`) to the aggregate command if needed.
