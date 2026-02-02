# Plan: Python Script for Rype Index Configuration Generation

## Overview

Create a Python script (`scripts/generate_config.py`) to generate and manage TOML configuration files for the `rype index from-config` command.

---

## Phase 1: Core Infrastructure
**Status**: COMPLETE

### Goals
- Set up script skeleton with argparse and subcommand structure
- Implement shared helper functions
- Implement constants

### TODOs
- [x] Create `scripts/generate_config.py` with shebang and imports
- [x] Define constants: `SEQUENCE_EXTENSIONS`, `DEFAULT_K`, `DEFAULT_WINDOW`, `DEFAULT_SALT`, `DEFAULT_ORIENT`
- [x] Implement `get_sequence_files(directory, extensions)` - non-recursive scan
- [x] Implement `derive_bucket_name(filepath)` - strip `.gz` then sequence extension
- [x] Implement `validate_no_whitespace(name, context)` - error on whitespace
- [x] Implement `write_config(index_settings, buckets, output_path)` - manual TOML generation
- [x] Set up argparse with parent parser for common args (`-k`, `-w`, `--salt`, `--orient`)
- [x] Set up subparsers structure

### Completion Notes
Implemented all Phase 1 infrastructure in `scripts/generate_config.py`:
- Constants defined at module level
- Helper functions tested with temporary files and directories
- CLI uses parent parser pattern for shared arguments across subcommands
- Subcommand stubs ready for Phase 2-4 implementation
- TOML output manually formatted (no external dependencies)

---

## Phase 2: `per-file` Subcommand
**Status**: COMPLETE

### Goals
- Implement per-file mode: each sequence file → its own bucket

### Arguments
- `directory` (positional): Directory containing sequence files
- `-o, --output` (required): Output TOML config path
- `--index-name` (required): Output index path
- Common args: `-k`, `-w`, `--salt`, `--orient`, `--extensions`

### TODOs
- [x] Add `per-file` subparser with arguments
- [x] Implement `cmd_per_file(args)`:
  - Scan directory for sequence files
  - Derive bucket name from each file's basename
  - Validate no whitespace in bucket names
  - Build config dict and write TOML
- [x] Test with sample directory

### Completion Notes
Implemented `cmd_per_file` in `scripts/generate_config.py`:
- Scans directory for sequence files using `get_sequence_files()`
- Derives bucket names using `derive_bucket_name()` (strips .gz then extension)
- Validates no whitespace in bucket names
- Detects and reports duplicate bucket names (e.g., sample1.fa + sample1.fasta)
- Writes TOML config with one bucket per file
- Tested: basic operation, custom -k/-w/--no-orient, empty directory error, duplicate names error

---

## Phase 3: `unified` Subcommand
**Status**: COMPLETE

### Goals
- Implement unified mode: all sequence files → single bucket

### Arguments
- `directory` (positional): Directory containing sequence files
- `-o, --output` (required): Output TOML config path
- `--index-name` (required): Output index path
- `--bucket-name`: Bucket name (default: "unified-bucket")
- Common args: `-k`, `-w`, `--salt`, `--orient`, `--extensions`

### TODOs
- [x] Add `unified` subparser with arguments
- [x] Implement `cmd_unified(args)`:
  - Scan directory for sequence files
  - Validate bucket name has no whitespace
  - Build config with single bucket containing all files
  - Write TOML
- [x] Test with sample directory

### Completion Notes
Implemented `cmd_unified` in `scripts/generate_config.py`:
- Scans directory for sequence files using `get_sequence_files()`
- Validates bucket name has no whitespace before processing
- Creates single bucket containing all found files
- Tested: basic operation, custom bucket name, custom -k/-w/--no-orient, whitespace bucket name error, empty directory error

---

## Phase 4: `grouped` Subcommand
**Status**: COMPLETE

### Goals
- Implement grouped mode: TSV-based sorting and even distribution into N buckets

### Arguments
- `tsv` (positional): TSV file with columns: `file_path`, `group_label`
- `-n, --num-buckets` (required): Number of output buckets
- `-o, --output` (required): Output TOML config path
- `--index-name` (required): Output index path
- Common args: `-k`, `-w`, `--salt`, `--orient`

### TSV Format (with header)
```
file_path	group_label
path/to/file1.fa	GroupA
path/to/file2.fa	GroupB
```

### Distribution Algorithm
1. Read TSV with `csv.DictReader`
2. Validate all file paths exist; error with list of missing files
3. Sort all files by `group_label` (keeps similar labels adjacent)
4. Split sorted list into N **contiguous chunks**:
   - `base, remainder = divmod(total_files, N)`
   - First `remainder` buckets get `base + 1` files
   - Remaining buckets get `base` files
   - Example: 295 files, 3 buckets → 99, 98, 98 files
5. Bucket names: `bucket_0`, `bucket_1`, ..., `bucket_N-1`

### TODOs
- [x] Add `grouped` subparser with arguments
- [x] Implement `cmd_grouped(args)`:
  - Read TSV using csv.DictReader
  - Validate all paths exist
  - Sort by group_label
  - Split into N contiguous chunks
  - Write TOML with bucket_0, bucket_1, etc.
- [x] Test with sample TSV

### Completion Notes
Implemented `cmd_grouped` in `scripts/generate_config.py`:
- Reads TSV with csv.DictReader, validates required columns (file_path, group_label)
- Validates all file paths exist; reports up to 10 missing files with count
- Sorts files by group_label to keep similar labels adjacent
- Distributes files evenly: first `remainder` buckets get `base+1` files
- Error handling: empty TSV, missing columns, missing files, buckets > files
- Tested: basic operation (10 files → 3 buckets = 4,3,3), custom -k/-w/--no-orient, error cases

---

## Phase 5: Integration Testing & Polish
**Status**: COMPLETE

### Goals
- End-to-end testing
- Error handling polish
- Help text clarity

### TODOs
- [x] Test `per-file` with real data directory
- [x] Test `unified` with real data directory
- [x] Test `grouped` with real TSV
- [x] Verify generated configs parse correctly with `rype index from-config`
- [x] Ensure helpful error messages for common failures
- [x] Add docstring to script

### Completion Notes
Integration testing completed successfully:

**Bug Fixed**: Salt value was incorrectly quoted as string (`"0x5555555555555555"`) instead of unquoted hex literal. Fixed in `write_config()`.

**Tests Performed**:
- `per-file` on `examples/` directory: Created 2 buckets (pUC19, phiX174) ✓
- `unified` on `examples/` directory: Created 1 bucket with 2 files ✓
- `grouped` on `scratch/grouped_test/groups.tsv` with -n 3: Created 3 buckets (4,3,3 files) ✓
- All generated configs parsed successfully with `rype index from-config` ✓

**Error Messages Verified**:
- Nonexistent directory: "Directory does not exist: /nonexistent"
- No sequence files: "No sequence files found in scratch/"
- Whitespace in bucket name: "Bucket name contains whitespace: 'bad name'"
- Nonexistent TSV: "TSV file not found: /nonexistent.tsv"
- Missing TSV columns: "TSV missing required columns: {'group_label', 'file_path'}"
- Missing files in TSV: "Missing 2 file(s):\n  /nonexistent/file1.fa..."
- More buckets than files: "Number of buckets (20) exceeds number of files (10)"

**Help Text**: Clear and comprehensive for all subcommands.

---

## Reference: TOML Config Format

```toml
[index]
k = 64
window = 200
salt = 0x5555555555555555
output = "index.ryxdi"
orient_sequences = true

[buckets.BucketName]
files = ["ref1.fa", "ref2.fa"]
```

## Reference: Default Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| k | 64 | K-mer size |
| w (window) | 200 | User-specified default |
| salt | 0x5555555555555555 | Standard salt |
| orient | true | Orient sequences |

## Reference: Supported Extensions

`.fa`, `.fasta`, `.fna`, `.fq`, `.fastq` (and `.gz` variants)

## Files to Create

- `scripts/generate_config.py` (~200-250 lines)
