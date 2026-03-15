# WASM Support (wasm32-unknown-emscripten)

Rype's library crate compiles for `wasm32-unknown-emscripten`, enabling use in
WASM-based environments like DuckDB-WASM extensions.

## Prerequisites

1. **Emscripten SDK**: Install from <https://emscripten.org/docs/getting_started/downloads.html>
2. **Rust target**: `rustup target add wasm32-unknown-emscripten`

## Build

```bash
source /path/to/emsdk/emsdk_env.sh
cargo build --target wasm32-unknown-emscripten --features arrow-ffi --lib --release
```

Static library output: `target/wasm32-unknown-emscripten/release/librype.a`

## What Works

- **Classification**: All classification algorithms (merge-join, parallel row group, negative filtering)
- **Minimizer extraction**: Core k-mer and minimizer algorithms
- **Parquet index loading**: Read indices from Emscripten virtual filesystem
- **Arrow FFI**: C API works with 32-bit pointers (`size_t` = 32-bit on wasm32)
- **Batch size recommendation**: `rype_recommend_batch_size()` / `rype_calculate_batch_config()`

## Threading

Rayon degrades to single-threaded execution on wasm32. The classification code
has `num_threads <= 1` fallback paths (see `src/classify/merge_join.rs`) that
avoid thread pool overhead. Unless Emscripten pthreads are enabled, all
operations run single-threaded.

## Memory Detection

Platform-specific memory detection (cgroups, /proc/meminfo, sysctl) is not
available on wasm32. The library falls back to a capped default. Callers should
pass an explicit memory budget via `rype_recommend_batch_size()` or
`--max-memory` rather than relying on auto-detection.

## Filesystem

Standard POSIX I/O works via Emscripten's virtual filesystem (MEMFS, WORKERFS,
NODEFS). The `mmap`/`madvise` prefetch optimization is automatically disabled
on wasm32 — `advise_prefetch()` returns 0.

## Known Limitations

- **No CLI binary**: Only the library crate compiles. The binary (main.rs) depends on terminal/process features unavailable in WASM.
- **Index creation**: Impractical due to wasm32's 4 GB memory limit. Build indices on x86_64 and load them in WASM.
- **Single-threaded**: Unless Emscripten pthreads are explicitly enabled.
- **Memory constants**: `FALLBACK_MEMORY_BYTES` (8 GB) and `MAX_CHUNK_BYTES` (4 GB) are stored as `u64` and capped to `usize::MAX` at runtime on wasm32.

## Verification

```bash
# Library compilation
cargo check --target wasm32-unknown-emscripten --features arrow-ffi --lib

# Test compilation (verifies constants and type safety)
cargo check --target wasm32-unknown-emscripten --features arrow-ffi --tests
```

Running `cargo test --target wasm32-unknown-emscripten` requires a full
Emscripten toolchain with node.js and may have linker issues with some
dependencies. Compilation checks are sufficient to prevent regressions (the
original blockers were compile-time constant overflows).
