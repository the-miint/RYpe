# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-11-18

### Added
- `rype classify filter`: stream reads through classification and write the
  reads (or pairs) that do not match the index, or with `--invert` those that
  do; for host depletion with no intermediate files (#31)
- C API to build an index from streaming Arrow data (#13)
- Cross-shard minimizer dedup filter for single-bucket Arrow builds (#19)
- wasm32 support for index loading, classification and index building (#30)

### Changed
- Faster gzip FASTX input: zlib-rs backend and per-file inflate threads (#29)
- Single index pass per `classify` invocation, plus shard-scan and merge-join
  speedups (#23)
- Parallel accumulator flush sort when building shards (#18)
- Arrow input batch size decoupled from classification batch size (#22)
- arrow and parquet updated from 57 to 59 (#20); needletail from 0.6 to 0.7 (#8)

### Fixed
- `classify run` and `classify filter` stay within `--max-memory` (#31, #32)
- Bounded memory and no quadratic scaling in multi-bucket index builds (#24)
- Bounded memory in single-bucket streaming consolidation (#11)
- Arrow builds consolidate shards so single-bucket indices stay small (#17)
- Minimizer dedup for single-bucket indices (#10)

## [1.0.0-rc.1] - 2026-02-26

### Added
- Initial public release
- RY-space minimizer-based sequence classification
- Support for k-mer sizes 16, 32, and 64
- Single-end and paired-end read classification
- Sharded index support for large datasets
- Inverted index for memory-efficient classification
- C API for FFI integration
- CLI tool for index management and classification
