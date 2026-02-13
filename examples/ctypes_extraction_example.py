#!/usr/bin/env python3
"""
Rype Python Single-Sequence Minimizer Extraction Example

Demonstrates calling rype's per-sequence minimizer extraction functions
from Python using ctypes. These functions operate on one sequence at a
time and do not require PyArrow.

This example:
  1. Loads librype.so via ctypes
  2. Calls rype_extract_minimizer_set() — sorted, deduplicated hash sets
  3. Calls rype_extract_strand_minimizers() — ordered hashes + positions
  4. Verifies invariants (sorted, positions in bounds, SoA lengths match)
  5. Properly frees all returned memory

C API functions exercised:
  - rype_extract_minimizer_set(seq, seq_len, k, w, salt)
  - rype_minimizer_set_result_free(result)
  - rype_extract_strand_minimizers(seq, seq_len, k, w, salt)
  - rype_strand_minimizers_result_free(result)
  - rype_get_last_error()

Prerequisites:
  - librype built (no special features required):
        cargo build --release

Usage:
  python examples/pyarrow_single_sequence_example.py

Tested with:
  - Python 3.13.9
  - Debian GNU/Linux 13 (trixie), x86_64
  - February 2026

NOTE: This example is NOT part of the regular test suite or build
dependencies. It exists to demonstrate and verify the ctypes interop
for the single-sequence extraction C API.
"""

import ctypes
import os
import sys


# ---------------------------------------------------------------------------
# ctypes struct definitions matching rype.h #[repr(C)] structs
# ---------------------------------------------------------------------------

class RypeU64Array(ctypes.Structure):
    """Mirrors the C RypeU64Array: { uint64_t* data; size_t len; }"""
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_uint64)),
        ("len", ctypes.c_size_t),
    ]

    def to_list(self):
        """Copy the array contents into a Python list."""
        return [self.data[i] for i in range(self.len)]


class RypeMinimizerSetResult(ctypes.Structure):
    """Mirrors the C RypeMinimizerSetResult: { RypeU64Array forward; RypeU64Array reverse_complement; }"""
    _fields_ = [
        ("forward", RypeU64Array),
        ("reverse_complement", RypeU64Array),
    ]


class RypeStrandResult(ctypes.Structure):
    """Mirrors the C RypeStrandResult: { uint64_t* hashes; uint64_t* positions; size_t len; }"""
    _fields_ = [
        ("hashes", ctypes.POINTER(ctypes.c_uint64)),
        ("positions", ctypes.POINTER(ctypes.c_uint64)),
        ("len", ctypes.c_size_t),
    ]

    def hashes_list(self):
        return [self.hashes[i] for i in range(self.len)]

    def positions_list(self):
        return [self.positions[i] for i in range(self.len)]


class RypeStrandMinimizersResult(ctypes.Structure):
    """Mirrors the C RypeStrandMinimizersResult: { RypeStrandResult forward; RypeStrandResult reverse_complement; }"""
    _fields_ = [
        ("forward", RypeStrandResult),
        ("reverse_complement", RypeStrandResult),
    ]


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

def load_librype():
    """Load librype.so from target/release."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.normpath(os.path.join(script_dir, "..", "target", "release", "librype.so"))

    if not os.path.exists(lib_path):
        print(f"ERROR: {lib_path} not found.", file=sys.stderr)
        print("Build with: cargo build --release", file=sys.stderr)
        sys.exit(1)

    lib = ctypes.CDLL(lib_path)

    # RypeMinimizerSetResult* rype_extract_minimizer_set(
    #     const uint8_t* seq, size_t seq_len, size_t k, size_t w, uint64_t salt)
    lib.rype_extract_minimizer_set.argtypes = [
        ctypes.c_char_p,   # seq
        ctypes.c_size_t,   # seq_len
        ctypes.c_size_t,   # k
        ctypes.c_size_t,   # w
        ctypes.c_uint64,   # salt
    ]
    lib.rype_extract_minimizer_set.restype = ctypes.POINTER(RypeMinimizerSetResult)

    # void rype_minimizer_set_result_free(RypeMinimizerSetResult* result)
    lib.rype_minimizer_set_result_free.argtypes = [ctypes.POINTER(RypeMinimizerSetResult)]
    lib.rype_minimizer_set_result_free.restype = None

    # RypeStrandMinimizersResult* rype_extract_strand_minimizers(
    #     const uint8_t* seq, size_t seq_len, size_t k, size_t w, uint64_t salt)
    lib.rype_extract_strand_minimizers.argtypes = [
        ctypes.c_char_p,   # seq
        ctypes.c_size_t,   # seq_len
        ctypes.c_size_t,   # k
        ctypes.c_size_t,   # w
        ctypes.c_uint64,   # salt
    ]
    lib.rype_extract_strand_minimizers.restype = ctypes.POINTER(RypeStrandMinimizersResult)

    # void rype_strand_minimizers_result_free(RypeStrandMinimizersResult* result)
    lib.rype_strand_minimizers_result_free.argtypes = [ctypes.POINTER(RypeStrandMinimizersResult)]
    lib.rype_strand_minimizers_result_free.restype = None

    # const char* rype_get_last_error(void)
    lib.rype_get_last_error.argtypes = []
    lib.rype_get_last_error.restype = ctypes.c_char_p

    return lib


# ---------------------------------------------------------------------------
# Extraction wrappers
# ---------------------------------------------------------------------------

def extract_minimizer_set(lib, seq, k, w, salt):
    """
    Extract sorted, deduplicated minimizer hash sets per strand.

    Returns (forward_hashes, rc_hashes) as Python lists of ints.
    """
    result_ptr = lib.rype_extract_minimizer_set(seq, len(seq), k, w, salt)
    if not result_ptr:
        err = lib.rype_get_last_error()
        raise RuntimeError(f"rype_extract_minimizer_set failed: {err.decode() if err else 'unknown'}")

    try:
        fwd = result_ptr.contents.forward.to_list()
        rc = result_ptr.contents.reverse_complement.to_list()
        return fwd, rc
    finally:
        lib.rype_minimizer_set_result_free(result_ptr)


def extract_strand_minimizers(lib, seq, k, w, salt):
    """
    Extract ordered minimizer hashes and positions per strand.

    Returns ((fwd_hashes, fwd_positions), (rc_hashes, rc_positions))
    as tuples of Python lists.
    """
    result_ptr = lib.rype_extract_strand_minimizers(seq, len(seq), k, w, salt)
    if not result_ptr:
        err = lib.rype_get_last_error()
        raise RuntimeError(f"rype_extract_strand_minimizers failed: {err.decode() if err else 'unknown'}")

    try:
        fwd_h = result_ptr.contents.forward.hashes_list()
        fwd_p = result_ptr.contents.forward.positions_list()
        rc_h = result_ptr.contents.reverse_complement.hashes_list()
        rc_p = result_ptr.contents.reverse_complement.positions_list()
        return (fwd_h, fwd_p), (rc_h, rc_p)
    finally:
        lib.rype_strand_minimizers_result_free(result_ptr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    lib = load_librype()

    test_sequences = [
        (b"AAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCC", "mixed 70bp"),
        (b"AAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCC", "alternating 48bp"),
        (b"ACGT", "too short for k=16"),
        (b"AAAAACCCCCAAAAACCCCCNAAAACCCCCAAAAACCCCCAAAAACCCCC", "N in middle"),
    ]

    k, w, salt = 16, 5, 0
    errors = 0

    print(f"Parameters: k={k}, w={w}, salt=0x{salt:016x}")
    print(f"Test sequences: {len(test_sequences)}")

    # --- Minimizer Set Extraction ---
    print("\n=== rype_extract_minimizer_set ===")
    for seq, label in test_sequences:
        fwd, rc = extract_minimizer_set(lib, seq, k, w, salt)
        display = seq[:40].decode() + ("..." if len(seq) > 40 else "")
        print(f"\n  [{label}] len={len(seq)}: {display}")
        print(f"    fwd: {len(fwd)} hashes, rc: {len(rc)} hashes")

        if fwd:
            print(f"    fwd[0:3] = {['0x%016x' % h for h in fwd[:3]]}")
        if rc:
            print(f"    rc[0:3]  = {['0x%016x' % h for h in rc[:3]]}")

        # Verify: sorted and deduplicated
        for i in range(1, len(fwd)):
            if fwd[i] <= fwd[i - 1]:
                print(f"    ERROR: fwd not strictly sorted at index {i}")
                errors += 1
                break
        for i in range(1, len(rc)):
            if rc[i] <= rc[i - 1]:
                print(f"    ERROR: rc not strictly sorted at index {i}")
                errors += 1
                break

        # Short sequence should produce empty results
        if len(seq) < k:
            if fwd or rc:
                print(f"    ERROR: short sequence produced non-empty results")
                errors += 1
            else:
                print(f"    OK: empty results for short sequence")

    # --- Strand Minimizers Extraction ---
    print("\n=== rype_extract_strand_minimizers ===")
    for seq, label in test_sequences:
        (fwd_h, fwd_p), (rc_h, rc_p) = extract_strand_minimizers(lib, seq, k, w, salt)
        display = seq[:40].decode() + ("..." if len(seq) > 40 else "")
        print(f"\n  [{label}] len={len(seq)}: {display}")
        print(f"    fwd: {len(fwd_h)} minimizers, rc: {len(rc_h)} minimizers")

        if fwd_h:
            print(f"    fwd[0]: hash=0x{fwd_h[0]:016x}, pos={fwd_p[0]}")
        if rc_h:
            print(f"    rc[0]:  hash=0x{rc_h[0]:016x}, pos={rc_p[0]}")

        # Verify: SoA lengths match
        if len(fwd_h) != len(fwd_p):
            print(f"    ERROR: fwd hashes/positions length mismatch: {len(fwd_h)} vs {len(fwd_p)}")
            errors += 1
        if len(rc_h) != len(rc_p):
            print(f"    ERROR: rc hashes/positions length mismatch: {len(rc_h)} vs {len(rc_p)}")
            errors += 1

        # Verify: positions non-decreasing
        for i in range(1, len(fwd_p)):
            if fwd_p[i] < fwd_p[i - 1]:
                print(f"    ERROR: fwd positions not non-decreasing at index {i}")
                errors += 1
                break
        for i in range(1, len(rc_p)):
            if rc_p[i] < rc_p[i - 1]:
                print(f"    ERROR: rc positions not non-decreasing at index {i}")
                errors += 1
                break

        # Verify: positions in bounds (pos + k <= seq_len)
        for i, p in enumerate(fwd_p):
            if p + k > len(seq):
                print(f"    ERROR: fwd position {p} out of bounds (pos+k={p+k} > len={len(seq)})")
                errors += 1
                break
        for i, p in enumerate(rc_p):
            if p + k > len(seq):
                print(f"    ERROR: rc position {p} out of bounds (pos+k={p+k} > len={len(seq)})")
                errors += 1
                break

        # Verify: N at position 20 — no minimizer should span it
        if b"N" in seq:
            n_pos = seq.index(b"N")
            for p in fwd_p:
                if p < n_pos and p + k > n_pos:
                    print(f"    ERROR: fwd position {p} spans N at {n_pos}")
                    errors += 1
                    break
            for p in rc_p:
                if p < n_pos and p + k > n_pos:
                    print(f"    ERROR: rc position {p} spans N at {n_pos}")
                    errors += 1
                    break

        # Short sequence should produce empty results
        if len(seq) < k:
            if fwd_h or rc_h:
                print(f"    ERROR: short sequence produced non-empty results")
                errors += 1
            else:
                print(f"    OK: empty results for short sequence")

    # --- Error handling test ---
    print("\n=== Error handling ===")
    # Invalid k value should return NULL
    result_ptr = lib.rype_extract_minimizer_set(b"ACGTACGT", 8, 17, 5, 0)
    if not result_ptr:
        err = lib.rype_get_last_error()
        print(f"  Invalid k=17 correctly rejected: {err.decode()}")
    else:
        print(f"  ERROR: invalid k=17 was not rejected")
        lib.rype_minimizer_set_result_free(result_ptr)
        errors += 1

    # w=0 should return NULL
    result_ptr = lib.rype_extract_strand_minimizers(b"ACGTACGT", 8, 16, 0, 0)
    if not result_ptr:
        err = lib.rype_get_last_error()
        print(f"  Invalid w=0 correctly rejected: {err.decode()}")
    else:
        print(f"  ERROR: invalid w=0 was not rejected")
        lib.rype_strand_minimizers_result_free(result_ptr)
        errors += 1

    # --- Summary ---
    print(f"\n{'='*40}")
    if errors == 0:
        print("All checks passed.")
    else:
        print(f"FAILED: {errors} error(s) detected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
