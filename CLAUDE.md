# CLAUDE.md

Project-specific instructions for Claude Code. For design rationale (RY encoding, minimizers, Parquet layout, lifecycle), see `docs/architecture.md`. For data structures, constants, and algorithms, read the code directly — it's the source of truth.

## Project Overview

**Rype** is a Rust library + CLI + C API for genomic sequence classification using minimizer-based k-mer sketching in RY (purine/pyrimidine) space. Indices are stored as `.ryxdi` Parquet directories.

## Rules

These rules apply to every task in this project unless explicitly overridden.

### Rule 1 — Think Before Coding
Bias caution over speed on non-trivial work. State assumptions explicitly; ask rather than guess. Push back when a simpler approach exists. Stop when confused.

### Rule 2 — Simplicity First
Minimum code that solves the problem. Nothing speculative. No abstractions for single-use code.

### Rule 3 — Surgical Changes
Touch only what you must. Don't improve adjacent code. Match existing style. Don't refactor what isn't broken.

### Rule 4 — Goal-Driven Execution
Define success criteria up front. Loop until verified. For correctness changes, "verified" means `cargo test` passes (and any new test you added fails without the fix). For performance changes, "verified" means `/usr/bin/time -v` measurements — both wall time and peak RSS — recorded against a baseline.

### Rule 5 — Surface conflicts, don't average them
If two patterns contradict, pick one (more recent / more tested). Explain why. Flag the other for cleanup.

### Rule 6 — Read before you write
Before adding code, read its exports, immediate callers, and the shared utilities it will touch. If you can't explain why existing code is structured a certain way, ask before changing it.

### Rule 7 — Tests verify intent, not just behavior
Tests must encode WHY behavior matters, not just WHAT it does. A test that can't fail when business logic changes is wrong.

### Rule 8 — Checkpoint at task boundaries
After each completed task in a TaskCreate list — or any change that crosses module boundaries — summarize what was done, what's verified, and what's left. Don't continue from a state you can't describe back.

### Rule 9 — Match the codebase's conventions, even if you disagree
Conformance > taste inside the codebase. If you think a convention is harmful, surface it. Don't fork silently.

### Rule 10 — Fail loud
"Completed" is wrong if anything was skipped silently. Don't silently skip tests you caused to be skipped (commented out, `#[ignore]`'d, missed) — existing `require-env` skips for optional deps/oracles are legitimate and expected. Default to surfacing uncertainty, not hiding it.

## Build and Test

```bash
cargo build --release        # production binary at target/release/rype
cargo build                  # dev build with debug symbols
cargo test                   # all tests
cargo test -- --nocapture    # show output
```

`rype --help` (and each subcommand's `--help`) is authoritative for CLI usage. C and Python integration examples live in `examples/`.

## Operational Rules (project-specific)

**Use `target/release/rype` directly, not `cargo run`.** `cargo run --release --bin rype --` can inject empty-string arguments. Always invoke the compiled binary.

**Temporary files go in `scratch/`, not `/tmp`.** `/tmp` has insufficient space on this system. The `scratch/` directory is gitignored.

**Never run performance tests in parallel.** Benchmarks are I/O-bound (shard loading dominates). Concurrent tests produce misleading timings due to disk contention. Run perf tests strictly sequentially.

**Always measure both time and peak RSS.** Use `/usr/bin/time -v` alongside `--timing` for per-phase breakdowns. Record both in plan docs.

```bash
# Log-ratio with minimum-length filter (the OOM-prone scenario):
/usr/bin/time -v target/release/rype classify log-ratio \
  -n perf-assessment/config/numerator-w200.ryxdi \
  -d perf-assessment/config/denominator-w200.ryxdi \
  -1 perf-assessment/query-files/long_read.parquet \
  --minimum-length 100 --max-memory 4G --timing \
  -o scratch/log-ratio-test.tsv
```

## Key Constraints (gotchas not obvious from code)

- **K-mer size**: K must be 16, 32, or 64. K is set at index creation and stored in the manifest.
- **Index compatibility**: Indices with different `k`, `w`, or `salt` cannot be combined (negative filtering, log-ratio, merge).
- **C API thread safety**:
  - Index loading/freeing — NOT thread-safe.
  - `rype_classify` — thread-safe (multiple threads can share one Index).
  - `RypeResultArray` — NOT thread-safe; each thread needs its own.
  - Never free an Index while classification is in progress (use-after-free). Never double-free a result array.
- **Log-ratio mode requires single-bucket indices** (exactly 1 bucket per index). The multi-bucket `n100-w200.ryxdi` cannot be used.
- **Short sequences**: sequences < K bases produce no minimizers.

## Local-Only Performance Test Data

`perf-assessment/` and `perf-data/` contain real-world benchmark data, not checked into git:

- **Genomes**: `perf-data/wol2-genomes/` — ~16,000 compressed FASTA genomes (WoL2 database)
- **Query files** (symlinks in `perf-assessment/query-files/`):
  - `short_read_R1.fastq.gz` / `short_read_R2.fastq.gz` — paired-end short reads
  - `long_read.fastq.gz` — long reads (~2.2GB)
  - Parquet-converted versions also present (`*.parquet`)
- **Pre-built indices**:
  - `perf-assessment/parquet-index/n100-w200.ryxdi/` — 160-bucket index (k=64, w=200, 8 shards, ~486M minimizers)
  - `perf-assessment/config/numerator-w200.ryxdi/` — single-bucket (8000 genomes, buckets 1–80, 267M minimizers)
  - `perf-assessment/config/denominator-w200.ryxdi/` — single-bucket (7952 genomes, buckets 81–160, 216M minimizers)
- **Index configs**: `perf-assessment/config/*.toml`

Tests using this data should be `#[ignore]`.

### Building single-bucket indices

Each config has `[index]` and one `[buckets.<name>]` section with a `files` array:

```toml
[index]
window = 200
salt = 6148914691236517205
output = "numerator-w200.ryxdi"

[buckets.numerator]
files = [ "../../perf-data/wol2-genomes/G001873845.fasta.gz", ... ]
```

Build with `target/release/rype index from-config -c <config.toml>`. Output path is relative to the config file location. Verify with `rype index stats -i <index.ryxdi>` — must show exactly 1 bucket.
