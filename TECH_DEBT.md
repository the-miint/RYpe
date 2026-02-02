# Technical Debt

This document tracks known technical debt items that were intentionally deferred. Each item includes rationale for deferral and suggested remediation approach.

---

## Error Handling

### 1. Panic in `reverse_complement()` (Low Priority)

**Location:** `src/core/encoding.rs:59`

**Current State:** Function panics on invalid k value.

**Rationale for Deferral:**
- Only called from `#[allow(dead_code)]` test helper
- Function is documented with a `# Panics` section
- Not in any hot path or production code

**Suggested Remediation:**
- Convert to `Result<u64, EncodingError>` if function becomes public API
- For now, panic is acceptable since callers are test code with known-valid inputs

---

### 2. CSR Overflow Expects (Low Priority)

**Location:** `src/indices/inverted/mod.rs:133,161`

**Current State:** Uses `expect()` when converting bucket counts to u32.

**Rationale for Deferral:**
- Would require 4 billion bucket IDs to trigger overflow
- Current use cases have at most thousands of buckets
- Adding Result propagation would complicate hot path with no practical benefit

**Suggested Remediation:**
- If bucket counts approach u32::MAX becomes realistic, convert to checked arithmetic with proper error propagation
- Consider using usize internally if 64-bit bucket IDs become necessary

---

### 3. CString Expect After Sanitization (Low Priority)

**Location:** `src/c_api.rs:136`

**Current State:** Uses `expect()` when creating CString from sanitized string.

**Rationale for Deferral:**
- String is already sanitized with null bytes replaced
- Panic is logically unreachable after sanitization
- Documented panic is acceptable for C API internal code

**Suggested Remediation:**
- Could use `unwrap_unchecked()` with safety comment if micro-optimization needed
- Current state is fine - clear documentation makes intent obvious

---

### 4. Writer Properties Expect (Low Priority)

**Location:** `src/indices/parquet/options.rs:132`

**Current State:** Uses `expect()` in `to_writer_properties()`.

**Rationale for Deferral:**
- Function has documented `# Panics` section
- Callers are expected to validate options before calling
- Used during index creation, not hot path

**Suggested Remediation:**
- Could convert to `TryInto` pattern if validation becomes complex
- Current state with documented panic is acceptable

---

## Large File Candidates

### 1. memory.rs (1,566 lines)

**Location:** `src/memory.rs`

**Current State:** Single file handling memory estimation, detection, and configuration.

**Rationale for Deferral:**
- File is internally cohesive around memory management
- No immediate maintenance pain
- Split would be cosmetic without functional benefit

**Suggested Remediation:**
When file grows or maintenance becomes difficult, consider splitting into:
- `memory/estimation.rs` - Memory requirement calculations
- `memory/detection.rs` - System memory detection (platform-specific)
- `memory/config.rs` - Memory configuration and limits

---

### 2. query_loading.rs (1,504 lines)

**Location:** `src/indices/inverted/query_loading.rs`

**Current State:** Contains query loading logic and associated tests.

**Rationale for Deferral:**
- Tests are co-located with implementation (Rust convention)
- Logic is cohesive around query loading
- No immediate maintenance issues

**Suggested Remediation:**
If file grows further, consider:
- Moving tests to `query_loading/tests.rs` module
- Separating streaming vs batch loading into submodules

---

## API Cleanup

### 1. Functions with `_parquet` Suffix (Low Priority)

**Current State:** 56 functions have `_parquet` suffix despite Parquet being the only format.

**Rationale for Deferral:**
- Would be a breaking change for any external callers
- Suffix provides clarity about data format
- Renaming provides no functional benefit

**Suggested Remediation:**
- If API stability is declared, consider removing suffix in next major version
- Document that Parquet is the only supported format
- Could introduce type aliases without suffix that delegate to `_parquet` versions

---

## Logging Inconsistencies

### 1. Mixed Logging Approaches (Low Priority)

**Current State:** Codebase uses mix of:
- `eprintln!()` for immediate user feedback
- `log::` macros for structured logging
- Custom `log_timing()` for performance metrics

**Rationale for Deferral:**
- Current approach works correctly
- Would require auditing all logging call sites
- Different approaches serve different purposes (user feedback vs debugging)

**Suggested Remediation:**
- Establish logging guidelines document
- Consider `tracing` crate for unified structured logging
- Migrate incrementally when touching affected code

---

## Test Coverage Gaps

### 1. C API Thread Safety Tests

**Current State:** No concurrent stress tests for C API.

**Rationale for Deferral:**
- Thread safety is documented in API
- Manual testing has verified concurrent classification works
- Stress tests are complex to write and maintain

**Suggested Remediation:**
- Add `tests/c_api_concurrent.rs` with multiple threads calling `rype_classify`
- Use `loom` crate for systematic concurrency testing
- Add CI job with thread sanitizer enabled

---

### 2. End-to-End Integration Tests

**Current State:** No integration tests in `tests/` directory for full workflows.

**Rationale for Deferral:**
- Unit tests cover individual components
- CLI is tested manually
- Integration tests require test data fixtures

**Suggested Remediation:**
- Create `tests/integration/` directory
- Add workflow tests: index creation -> classification -> output verification
- Include small test FASTA/FASTQ files in `tests/fixtures/`
- Consider property-based testing with `proptest` for edge cases

---

## Maintenance Notes

- Items marked "Low Priority" have no known user impact
- Revisit this document when:
  - Adding new contributors (onboarding context)
  - Planning major refactoring
  - Preparing for public API stability guarantees
- Remove items from this document as they are addressed
