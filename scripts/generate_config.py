#!/usr/bin/env python3
"""
Generate TOML configuration files for rype index from-config command.

Supports three modes:
  - per-file: Each sequence file becomes its own bucket
  - unified: All sequence files go into a single bucket
  - grouped: TSV-based grouping with even distribution into N buckets
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# =============================================================================
# Constants
# =============================================================================

SEQUENCE_EXTENSIONS: Set[str] = {
    ".fa", ".fasta", ".fna", ".fq", ".fastq",
    ".fa.gz", ".fasta.gz", ".fna.gz", ".fq.gz", ".fastq.gz",
}

DEFAULT_K: int = 64
DEFAULT_WINDOW: int = 200
DEFAULT_SALT: str = "0x5555555555555555"
DEFAULT_ORIENT: bool = True


# =============================================================================
# Helper Functions
# =============================================================================

def get_sequence_files(directory: str, extensions: Set[str]) -> List[str]:
    """
    Non-recursively scan directory for sequence files matching given extensions.

    Args:
        directory: Path to directory to scan
        extensions: Set of valid extensions (including .gz variants)

    Returns:
        Sorted list of absolute paths to matching files
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise ValueError(f"Directory does not exist: {directory}")

    matching_files = []
    for entry in dir_path.iterdir():
        if not entry.is_file():
            continue
        name = entry.name.lower()
        # Check for double extensions first (e.g., .fa.gz)
        for ext in extensions:
            if name.endswith(ext):
                matching_files.append(str(entry.resolve()))
                break

    return sorted(matching_files)


def derive_bucket_name(filepath: str) -> str:
    """
    Derive bucket name from filepath by stripping .gz (if present) then sequence extension.

    Args:
        filepath: Path to sequence file

    Returns:
        Bucket name (basename without sequence extension)

    Example:
        "path/to/sample.fa.gz" -> "sample"
        "path/to/sample.fasta" -> "sample"
    """
    name = Path(filepath).name

    # Strip .gz first if present
    if name.lower().endswith(".gz"):
        name = name[:-3]

    # Strip sequence extension
    base_extensions = {".fa", ".fasta", ".fna", ".fq", ".fastq"}
    lower_name = name.lower()
    for ext in sorted(base_extensions, key=len, reverse=True):
        if lower_name.endswith(ext):
            name = name[:-len(ext)]
            break

    return name


def validate_no_whitespace(name: str, context: str) -> None:
    """
    Validate that name contains no whitespace characters.

    Args:
        name: String to validate
        context: Description of what the name represents (for error message)

    Raises:
        ValueError: If name contains whitespace
    """
    if any(c.isspace() for c in name):
        raise ValueError(f"{context} contains whitespace: '{name}'")


def write_config(
    index_settings: Dict,
    buckets: List[Tuple[str, List[str]]],
    output_path: str
) -> None:
    """
    Write TOML configuration file manually (no external dependencies).

    Args:
        index_settings: Dict with keys: k, window, salt, output, orient_sequences
        buckets: List of (bucket_name, [file_paths]) tuples
        output_path: Path to write TOML file

    TOML format:
        [index]
        k = 64
        window = 200
        salt = 0x5555555555555555
        output = "index.ryxdi"
        orient_sequences = true

        [buckets.BucketName]
        files = ["file1.fa", "file2.fa"]
    """
    lines = []

    # [index] section
    lines.append("[index]")
    lines.append(f"k = {index_settings['k']}")
    lines.append(f"window = {index_settings['window']}")
    lines.append(f"salt = {index_settings['salt']}")
    lines.append(f"output = \"{index_settings['output']}\"")
    lines.append(f"orient_sequences = {'true' if index_settings['orient_sequences'] else 'false'}")
    lines.append("")

    # [buckets.*] sections
    for bucket_name, files in buckets:
        lines.append(f"[buckets.{bucket_name}]")
        # Format files array - one file per line for readability
        if len(files) == 1:
            lines.append(f'files = ["{files[0]}"]')
        else:
            lines.append("files = [")
            for f in files:
                lines.append(f'    "{f}",')
            lines.append("]")
        lines.append("")

    # Write to file
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    print(f"Wrote config to: {output_path}")


# =============================================================================
# Subcommand Placeholders (to be implemented in later phases)
# =============================================================================

def cmd_per_file(args: argparse.Namespace) -> int:
    """Phase 2: Each sequence file becomes its own bucket."""
    # Parse extensions
    if args.extensions:
        extensions = {ext.strip() for ext in args.extensions.split(",")}
    else:
        extensions = SEQUENCE_EXTENSIONS

    # Scan directory for sequence files
    files = get_sequence_files(args.directory, extensions)
    if not files:
        raise ValueError(f"No sequence files found in {args.directory}")

    # Build buckets: one per file
    buckets = []
    for filepath in files:
        bucket_name = derive_bucket_name(filepath)
        validate_no_whitespace(bucket_name, f"Bucket name derived from '{filepath}'")
        buckets.append((bucket_name, [filepath]))

    # Check for duplicate bucket names
    bucket_names = [b[0] for b in buckets]
    if len(bucket_names) != len(set(bucket_names)):
        seen = set()
        duplicates = []
        for name in bucket_names:
            if name in seen:
                duplicates.append(name)
            seen.add(name)
        raise ValueError(f"Duplicate bucket names detected: {sorted(set(duplicates))}")

    # Build index settings
    index_settings = {
        "k": args.k,
        "window": args.window,
        "salt": args.salt,
        "output": args.index_name,
        "orient_sequences": args.orient,
    }

    # Write config
    write_config(index_settings, buckets, args.output)
    print(f"Created {len(buckets)} buckets (one per file)")
    return 0


def read_files_from_tsv(tsv_path: str) -> List[str]:
    """
    Read file paths from a TSV file with columns: file_path, group_label.

    Args:
        tsv_path: Path to TSV file

    Returns:
        List of file paths (group_label is ignored)

    Raises:
        FileNotFoundError: If TSV file doesn't exist
        ValueError: If TSV is malformed or files are missing
    """
    path = Path(tsv_path)
    if not path.is_file():
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    files: List[str] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        # Validate header
        if reader.fieldnames is None:
            raise ValueError("TSV file is empty or has no header")
        required_cols = {"file_path", "group_label"}
        if not required_cols.issubset(set(reader.fieldnames)):
            missing = required_cols - set(reader.fieldnames)
            raise ValueError(f"TSV missing required columns: {missing}")

        for row in reader:
            file_path = row["file_path"].strip()
            files.append(file_path)

    if not files:
        raise ValueError("TSV file contains no data rows")

    # Validate all file paths exist
    missing_files = [f for f in files if not Path(f).is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing {len(missing_files)} file(s):\n" +
            "\n".join(f"  {f}" for f in missing_files[:10]) +
            (f"\n  ... and {len(missing_files) - 10} more" if len(missing_files) > 10 else "")
        )

    return files


def cmd_unified(args: argparse.Namespace) -> int:
    """Phase 3: All sequence files go into a single bucket."""
    # Validate bucket name
    validate_no_whitespace(args.bucket_name, "Bucket name")

    input_path = Path(args.input)

    if input_path.is_dir():
        # Directory mode: scan for sequence files
        if args.extensions:
            extensions = {ext.strip() for ext in args.extensions.split(",")}
        else:
            extensions = SEQUENCE_EXTENSIONS

        files = get_sequence_files(str(input_path), extensions)
        if not files:
            raise ValueError(f"No sequence files found in {args.input}")
    elif input_path.is_file():
        # TSV mode: read file paths from TSV
        files = read_files_from_tsv(str(input_path))
    else:
        raise FileNotFoundError(f"Input path does not exist: {args.input}")

    # Build single bucket containing all files
    buckets = [(args.bucket_name, files)]

    # Build index settings
    index_settings = {
        "k": args.k,
        "window": args.window,
        "salt": args.salt,
        "output": args.index_name,
        "orient_sequences": args.orient,
    }

    # Write config
    write_config(index_settings, buckets, args.output)
    print(f"Created 1 bucket '{args.bucket_name}' with {len(files)} files")
    return 0


def cmd_grouped(args: argparse.Namespace) -> int:
    """Phase 4: TSV-based grouping with even distribution into N buckets."""
    # Validate num_buckets
    if args.num_buckets < 1:
        raise ValueError("Number of buckets must be at least 1")

    # Read TSV file
    tsv_path = Path(args.tsv)
    if not tsv_path.is_file():
        raise FileNotFoundError(f"TSV file not found: {args.tsv}")

    files_with_labels: List[Tuple[str, str]] = []
    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        # Validate header
        if reader.fieldnames is None:
            raise ValueError("TSV file is empty or has no header")
        required_cols = {"file_path", "group_label"}
        if not required_cols.issubset(set(reader.fieldnames)):
            missing = required_cols - set(reader.fieldnames)
            raise ValueError(f"TSV missing required columns: {missing}")

        for row in reader:
            file_path = row["file_path"].strip()
            group_label = row["group_label"].strip()
            files_with_labels.append((file_path, group_label))

    if not files_with_labels:
        raise ValueError("TSV file contains no data rows")

    # Validate all file paths exist
    missing_files = []
    for file_path, _ in files_with_labels:
        if not Path(file_path).is_file():
            missing_files.append(file_path)

    if missing_files:
        raise FileNotFoundError(
            f"Missing {len(missing_files)} file(s):\n" +
            "\n".join(f"  {f}" for f in missing_files[:10]) +
            (f"\n  ... and {len(missing_files) - 10} more" if len(missing_files) > 10 else "")
        )

    # Sort by group_label (keeps similar labels adjacent)
    files_with_labels.sort(key=lambda x: x[1])

    # Split into N contiguous chunks
    total_files = len(files_with_labels)
    n = args.num_buckets

    if n > total_files:
        raise ValueError(
            f"Number of buckets ({n}) exceeds number of files ({total_files})"
        )

    base, remainder = divmod(total_files, n)
    # First `remainder` buckets get `base + 1` files
    # Remaining buckets get `base` files

    buckets: List[Tuple[str, List[str]]] = []
    idx = 0
    for bucket_num in range(n):
        size = base + 1 if bucket_num < remainder else base
        bucket_files = [files_with_labels[idx + i][0] for i in range(size)]
        bucket_name = f"bucket_{bucket_num}"
        buckets.append((bucket_name, bucket_files))
        idx += size

    # Build index settings
    index_settings = {
        "k": args.k,
        "window": args.window,
        "salt": args.salt,
        "output": args.index_name,
        "orient_sequences": args.orient,
    }

    # Write config
    write_config(index_settings, buckets, args.output)

    # Print summary
    sizes = [len(b[1]) for b in buckets]
    print(f"Created {n} buckets from {total_files} files")
    print(f"Bucket sizes: min={min(sizes)}, max={max(sizes)}")
    return 0


# =============================================================================
# CLI Setup
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="generate_config",
        description="Generate TOML configuration files for rype index from-config",
    )

    # Parent parser for common arguments shared across subcommands
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "-k",
        type=int,
        default=DEFAULT_K,
        help=f"K-mer size (default: {DEFAULT_K})",
    )
    common_parser.add_argument(
        "-w", "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help=f"Window size (default: {DEFAULT_WINDOW})",
    )
    common_parser.add_argument(
        "--salt",
        type=str,
        default=DEFAULT_SALT,
        help=f"Hash salt as hex string (default: {DEFAULT_SALT})",
    )
    common_parser.add_argument(
        "--orient",
        dest="orient",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ORIENT,
        help=f"Orient sequences (default: {'yes' if DEFAULT_ORIENT else 'no'})",
    )

    # Subparsers
    subparsers = parser.add_subparsers(
        title="subcommands",
        dest="subcommand",
        required=True,
    )

    # per-file subcommand (Phase 2)
    per_file_parser = subparsers.add_parser(
        "per-file",
        parents=[common_parser],
        help="Each sequence file becomes its own bucket",
    )
    per_file_parser.add_argument(
        "directory",
        help="Directory containing sequence files",
    )
    per_file_parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output TOML config path",
    )
    per_file_parser.add_argument(
        "--index-name",
        required=True,
        help="Output index path (e.g., index.ryxdi)",
    )
    per_file_parser.add_argument(
        "--extensions",
        type=str,
        default=None,
        help="Comma-separated list of extensions to include (default: all sequence extensions)",
    )
    per_file_parser.set_defaults(func=cmd_per_file)

    # unified subcommand (Phase 3)
    unified_parser = subparsers.add_parser(
        "unified",
        parents=[common_parser],
        help="All sequence files go into a single bucket",
    )
    unified_parser.add_argument(
        "input",
        help="Directory containing sequence files, or TSV file with columns: file_path, group_label",
    )
    unified_parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output TOML config path",
    )
    unified_parser.add_argument(
        "--index-name",
        required=True,
        help="Output index path (e.g., index.ryxdi)",
    )
    unified_parser.add_argument(
        "--bucket-name",
        default="unified-bucket",
        help="Bucket name (default: unified-bucket)",
    )
    unified_parser.add_argument(
        "--extensions",
        type=str,
        default=None,
        help="Comma-separated list of extensions to include; only used in directory mode (default: all sequence extensions)",
    )
    unified_parser.set_defaults(func=cmd_unified)

    # grouped subcommand (Phase 4)
    grouped_parser = subparsers.add_parser(
        "grouped",
        parents=[common_parser],
        help="TSV-based grouping with even distribution into N buckets",
    )
    grouped_parser.add_argument(
        "tsv",
        help="TSV file with columns: file_path, group_label",
    )
    grouped_parser.add_argument(
        "-n", "--num-buckets",
        type=int,
        required=True,
        help="Number of output buckets",
    )
    grouped_parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output TOML config path",
    )
    grouped_parser.add_argument(
        "--index-name",
        required=True,
        help="Output index path (e.g., index.ryxdi)",
    )
    grouped_parser.set_defaults(func=cmd_grouped)

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    try:
        return args.func(args)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
