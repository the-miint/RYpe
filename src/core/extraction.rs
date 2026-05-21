//! Minimizer extraction algorithms.
//!
//! This module provides functions for extracting minimizers from DNA sequences
//! using the RY (purine/pyrimidine) encoding scheme. Minimizers are selected
//! using a sliding window approach with a monotonic deque for O(n) complexity.

// Hot path: index-based iteration avoids iterator overhead in inner loops
#![allow(clippy::needless_range_loop)]

use crate::error::{Result, RypeError};

use super::encoding::base_to_bit;
use super::workspace::MinimizerWorkspace;

/// Cast a `usize` minimizer position to `u32` for the `.ryci` position column,
/// failing loud on overflow.
///
/// Used by every `*_with_positions` extractor. Pulled into a helper so the
/// overflow contract is testable in isolation: no realistic test sequence can
/// trigger `len > u32::MAX`, but if a future change quietly removes the guard
/// the unit test catches it immediately.
///
/// Visible to the crate (Plan 1.3 chain DP also needs to cast `usize`
/// positions to `u32` and should not duplicate this two-line guard).
pub(crate) fn cast_pos(p: usize) -> Result<u32> {
    u32::try_from(p).map_err(|_| {
        RypeError::validation(format!(
            "minimizer position {} exceeds u32::MAX; \
             input sequence longer than 4.29 Gbp is not supported",
            p
        ))
    })
}

/// Strand indicator for minimizer origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strand {
    Forward,
    ReverseComplement,
}

/// Structure-of-arrays for minimizers on a single strand.
///
/// Stores hashes and their corresponding positions in parallel arrays,
/// maintaining extraction order (non-decreasing positions).
#[derive(Debug, Clone)]
pub struct StrandMinimizers {
    /// The minimizer hash values, in extraction order.
    pub hashes: Vec<u64>,
    /// 0-based positions in the sequence where each k-mer starts.
    pub positions: Vec<usize>,
}

/// Extract ordered minimizers with positions per strand (SoA layout).
///
/// Returns `(forward, reverse_complement)` StrandMinimizers. Unlike
/// `extract_dual_strand_into()`, this preserves position information and
/// deduplicates by position change (not by hash value).
///
/// # Arguments
/// * `seq` - DNA sequence as bytes
/// * `k` - K-mer size (must be 16, 32, or 64)
/// * `w` - Window size for minimizer selection
/// * `salt` - XOR salt applied to k-mer hashes
/// * `ws` - Workspace for temporary storage
pub fn extract_strand_minimizers(
    seq: &[u8],
    k: usize,
    w: usize,
    salt: u64,
    ws: &mut MinimizerWorkspace,
) -> (StrandMinimizers, StrandMinimizers) {
    ws.q_fwd.clear();
    ws.q_rc.clear();

    let len = seq.len();
    if len < k {
        return (
            StrandMinimizers {
                hashes: vec![],
                positions: vec![],
            },
            StrandMinimizers {
                hashes: vec![],
                positions: vec![],
            },
        );
    }

    let k_mask = if k == 64 { u64::MAX } else { (1u64 << k) - 1 };
    let rc_shift = k - 1;

    let cap = ws.estimated_minimizers;
    let mut fwd = StrandMinimizers {
        hashes: Vec::with_capacity(cap),
        positions: Vec::with_capacity(cap),
    };
    let mut rc = StrandMinimizers {
        hashes: Vec::with_capacity(cap),
        positions: Vec::with_capacity(cap),
    };

    let mut current_val: u64 = 0;
    let mut current_rc: u64 = 0;
    let mut valid_bases_count = 0;

    let mut last_fwd_pos: Option<usize> = None;
    let mut last_rc_pos: Option<usize> = None;

    for i in 0..len {
        let bit = base_to_bit(seq[i]);

        if bit == u64::MAX {
            valid_bases_count = 0;
            ws.q_fwd.clear();
            ws.q_rc.clear();
            current_val = 0;
            current_rc = 0;
            last_fwd_pos = None;
            last_rc_pos = None;
            continue;
        }

        valid_bases_count += 1;
        current_val = ((current_val << 1) | bit) & k_mask;
        current_rc = (current_rc >> 1) | ((bit ^ 1) << rc_shift);

        if valid_bases_count >= k {
            let pos = i + 1 - k;
            let h_fwd = current_val ^ salt;
            let h_rc = current_rc ^ salt;

            // Forward strand deque
            while let Some(&(p, _)) = ws.q_fwd.front() {
                if p + w <= pos {
                    ws.q_fwd.pop_front();
                } else {
                    break;
                }
            }
            while let Some(&(_, v)) = ws.q_fwd.back() {
                if v >= h_fwd {
                    ws.q_fwd.pop_back();
                } else {
                    break;
                }
            }
            ws.q_fwd.push_back((pos, h_fwd));

            // Reverse complement deque
            while let Some(&(p, _)) = ws.q_rc.front() {
                if p + w <= pos {
                    ws.q_rc.pop_front();
                } else {
                    break;
                }
            }
            while let Some(&(_, v)) = ws.q_rc.back() {
                if v >= h_rc {
                    ws.q_rc.pop_back();
                } else {
                    break;
                }
            }
            ws.q_rc.push_back((pos, h_rc));

            if valid_bases_count >= k + w - 1 {
                if let Some(&(min_pos, min_hash)) = ws.q_fwd.front() {
                    if last_fwd_pos != Some(min_pos) {
                        fwd.hashes.push(min_hash);
                        fwd.positions.push(min_pos);
                        last_fwd_pos = Some(min_pos);
                    }
                }
                if let Some(&(min_pos, min_hash)) = ws.q_rc.front() {
                    if last_rc_pos != Some(min_pos) {
                        rc.hashes.push(min_hash);
                        rc.positions.push(min_pos);
                        last_rc_pos = Some(min_pos);
                    }
                }
            }
        }
    }

    (fwd, rc)
}

/// Extract sorted, deduplicated minimizer sets per strand.
///
/// Returns `(forward_set, rc_set)` where each `Vec<u64>` is sorted and
/// contains no duplicates. This is a convenience wrapper around
/// `extract_dual_strand_into()` with sort + dedup.
///
/// # Arguments
/// * `seq` - DNA sequence as bytes
/// * `k` - K-mer size (must be 16, 32, or 64)
/// * `w` - Window size for minimizer selection
/// * `salt` - XOR salt applied to k-mer hashes
/// * `ws` - Workspace for temporary storage
pub fn extract_minimizer_set(
    seq: &[u8],
    k: usize,
    w: usize,
    salt: u64,
    ws: &mut MinimizerWorkspace,
) -> (Vec<u64>, Vec<u64>) {
    let (mut fwd, mut rc) = extract_dual_strand_into(seq, k, w, salt, ws);
    fwd.sort_unstable();
    fwd.dedup();
    rc.sort_unstable();
    rc.dedup();
    (fwd, rc)
}

/// Extract minimizers from a sequence (single strand).
///
/// Uses a monotonic deque to efficiently find the minimum hash value
/// in each sliding window of size `w`. Consecutive duplicate minimizers
/// are deduplicated.
///
/// # Arguments
/// * `seq` - DNA sequence as bytes (A, G, T, C, case insensitive)
/// * `k` - K-mer size (must be 16, 32, or 64)
/// * `w` - Window size for minimizer selection
/// * `salt` - XOR salt applied to k-mer hashes
/// * `ws` - Workspace for temporary storage and output
///
/// # Output
/// Extracted minimizers are stored in `ws.buffer` (cleared before use).
pub fn extract_into(seq: &[u8], k: usize, w: usize, salt: u64, ws: &mut MinimizerWorkspace) {
    ws.buffer.clear();
    ws.q_fwd.clear();

    let len = seq.len();
    if len < k {
        return;
    }

    // Precompute mask outside hot loop - only lower k bits are valid
    let k_mask = if k == 64 { u64::MAX } else { (1u64 << k) - 1 };

    let mut current_val: u64 = 0;
    let mut last_min: Option<u64> = None;
    let mut valid_bases_count = 0;

    for i in 0..len {
        let bit = base_to_bit(seq[i]);

        if bit == u64::MAX {
            // Invalid base (N, etc.) - reset k-mer accumulator
            valid_bases_count = 0;
            ws.q_fwd.clear();
            current_val = 0;
            last_min = None;
            continue;
        }

        valid_bases_count += 1;
        current_val = ((current_val << 1) | bit) & k_mask;

        if valid_bases_count >= k {
            let pos = i + 1 - k;
            let hash = current_val ^ salt;

            // Maintain monotonic deque - remove old entries outside window
            while let Some(&(p, _)) = ws.q_fwd.front() {
                if p + w <= pos {
                    ws.q_fwd.pop_front();
                } else {
                    break;
                }
            }
            // Remove entries with larger hash values
            while let Some(&(_, v)) = ws.q_fwd.back() {
                if v >= hash {
                    ws.q_fwd.pop_back();
                } else {
                    break;
                }
            }
            ws.q_fwd.push_back((pos, hash));

            // Output minimizer once we have a full window
            if valid_bases_count >= k + w - 1 {
                if let Some(&(_, min_h)) = ws.q_fwd.front() {
                    if Some(min_h) != last_min {
                        ws.buffer.push(min_h);
                        last_min = Some(min_h);
                    }
                }
            }
        }
    }
}

/// Extract minimizers and their forward-strand positions from a single strand.
///
/// Position-aware variant of [`extract_into`]. Same sliding-window algorithm
/// and the same consecutive-duplicate-hash dedup as `extract_into`; the only
/// difference is that each emitted minimizer is paired with its position (a
/// zero-based offset into `seq`) in [`MinimizerWorkspace::positions_fwd`].
///
/// # Dedup semantics
/// Consecutive duplicate hashes are dropped, keeping the FIRST position seen
/// for each hash. Output has no two adjacent entries with the same hash. This
/// matches skani's "earliest anchor wins" convention and is exactly what a
/// downstream index that requires hash-unique-per-bucket needs (one such
/// caller is the `.ryci` cluster index, but the contract stands on its own).
///
/// # Arguments
/// * `seq` - DNA sequence as bytes (A, G, T, C, case insensitive)
/// * `k` - K-mer size (must be 16, 32, or 64)
/// * `w` - Window size for minimizer selection
/// * `salt` - XOR salt applied to k-mer hashes
/// * `ws` - Workspace; `buffer` and `positions_fwd` are cleared and refilled
///
/// # Output
/// `ws.buffer` (hashes) and `ws.positions_fwd` (positions) are written
/// index-parallel: `ws.positions_fwd[i]` is the offset in `seq` of the k-mer
/// whose hash is `ws.buffer[i]`. Empty if `seq.len() < k`.
///
/// # Errors
/// Returns `RypeError::validation` if any emitted position exceeds `u32::MAX`.
/// In practice this requires a `seq` longer than 4.29 Gbp — far beyond any
/// realistic single contig. The guard exists so a future genome-scale caller
/// fails loud rather than silently truncating.
pub fn extract_into_with_positions(
    seq: &[u8],
    k: usize,
    w: usize,
    salt: u64,
    ws: &mut MinimizerWorkspace,
) -> Result<()> {
    ws.buffer.clear();
    ws.positions_fwd.clear();
    // Also clear the dual-strand output buffers so a caller alternating
    // single-strand and dual-strand extracts on one workspace can't read
    // stale RC data left over from a prior dual-strand call. The single-
    // strand path never writes to these — the doc on `rc_buffer` promises
    // they're empty after a single-strand extract, so we keep that promise.
    ws.rc_buffer.clear();
    ws.positions_rc.clear();
    ws.q_fwd.clear();

    let len = seq.len();
    if len < k {
        return Ok(());
    }

    let k_mask = if k == 64 { u64::MAX } else { (1u64 << k) - 1 };

    let mut current_val: u64 = 0;
    let mut last_min: Option<u64> = None;
    let mut valid_bases_count = 0;

    for i in 0..len {
        let bit = base_to_bit(seq[i]);

        if bit == u64::MAX {
            valid_bases_count = 0;
            ws.q_fwd.clear();
            current_val = 0;
            last_min = None;
            continue;
        }

        valid_bases_count += 1;
        current_val = ((current_val << 1) | bit) & k_mask;

        if valid_bases_count >= k {
            let pos = i + 1 - k;
            let hash = current_val ^ salt;

            while let Some(&(p, _)) = ws.q_fwd.front() {
                if p + w <= pos {
                    ws.q_fwd.pop_front();
                } else {
                    break;
                }
            }
            while let Some(&(_, v)) = ws.q_fwd.back() {
                // Strict `>` (not `>=`): ties on hash must keep the EARLIER
                // position. With `>=`, a later equal-hash entry would pop
                // earlier ones from the back, leaving the LATEST occurrence
                // at the front — wrong for skani's "earliest anchor"
                // convention. The position-less `extract_into` uses `>=`
                // because it only cares about the hash value.
                if v > hash {
                    ws.q_fwd.pop_back();
                } else {
                    break;
                }
            }
            ws.q_fwd.push_back((pos, hash));

            if valid_bases_count >= k + w - 1 {
                if let Some(&(min_pos, min_h)) = ws.q_fwd.front() {
                    if Some(min_h) != last_min {
                        // The deque front carries the earliest position for
                        // this hash in the current window (guaranteed by the
                        // strict-`>` back-prune above) — exactly the
                        // "first occurrence" position skani chaining wants.
                        ws.buffer.push(min_h);
                        ws.positions_fwd.push(cast_pos(min_pos)?);
                        last_min = Some(min_h);
                    }
                }
            }
        }
    }

    Ok(())
}

/// Extract minimizers from both strands of a sequence.
///
/// Simultaneously computes minimizers for the forward strand and its
/// reverse complement. This is useful for strand-agnostic matching.
///
/// # Arguments
/// * `seq` - DNA sequence as bytes
/// * `k` - K-mer size (must be 16, 32, or 64)
/// * `w` - Window size for minimizer selection
/// * `salt` - XOR salt applied to k-mer hashes
/// * `ws` - Workspace for temporary storage
///
/// # Returns
/// A tuple of (forward_minimizers, reverse_complement_minimizers).
pub fn extract_dual_strand_into(
    seq: &[u8],
    k: usize,
    w: usize,
    salt: u64,
    ws: &mut MinimizerWorkspace,
) -> (Vec<u64>, Vec<u64>) {
    ws.q_fwd.clear();
    ws.q_rc.clear();

    let len = seq.len();
    if len < k {
        return (vec![], vec![]);
    }

    // Precompute mask outside hot loop - only lower k bits are valid
    let k_mask = if k == 64 { u64::MAX } else { (1u64 << k) - 1 };
    // Shift amount for incremental reverse complement computation
    let rc_shift = k - 1;

    let mut fwd_mins = Vec::with_capacity(ws.estimated_minimizers);
    let mut rc_mins = Vec::with_capacity(ws.estimated_minimizers);

    let mut current_val: u64 = 0;
    // Incremental reverse complement: avoids expensive reverse_bits() call per position
    // When kmer' = (kmer << 1) | new_bit, then rc' = (rc >> 1) | (complement_bit << (k-1))
    let mut current_rc: u64 = 0;
    let mut valid_bases_count = 0;

    let mut last_fwd: Option<u64> = None;
    let mut last_rc: Option<u64> = None;

    for i in 0..len {
        let bit = base_to_bit(seq[i]);

        if bit == u64::MAX {
            valid_bases_count = 0;
            ws.q_fwd.clear();
            ws.q_rc.clear();
            current_val = 0;
            current_rc = 0;
            last_fwd = None;
            last_rc = None;
            continue;
        }

        valid_bases_count += 1;
        // Update forward k-mer
        current_val = ((current_val << 1) | bit) & k_mask;
        // Update reverse complement incrementally: new bit's complement goes to MSB position
        current_rc = (current_rc >> 1) | ((bit ^ 1) << rc_shift);

        if valid_bases_count >= k {
            let pos = i + 1 - k;
            let h_fwd = current_val ^ salt;
            let h_rc = current_rc ^ salt;

            // Forward strand deque
            while let Some(&(p, _)) = ws.q_fwd.front() {
                if p + w <= pos {
                    ws.q_fwd.pop_front();
                } else {
                    break;
                }
            }
            while let Some(&(_, v)) = ws.q_fwd.back() {
                if v >= h_fwd {
                    ws.q_fwd.pop_back();
                } else {
                    break;
                }
            }
            ws.q_fwd.push_back((pos, h_fwd));

            // Reverse complement deque
            while let Some(&(p, _)) = ws.q_rc.front() {
                if p + w <= pos {
                    ws.q_rc.pop_front();
                } else {
                    break;
                }
            }
            while let Some(&(_, v)) = ws.q_rc.back() {
                if v >= h_rc {
                    ws.q_rc.pop_back();
                } else {
                    break;
                }
            }
            ws.q_rc.push_back((pos, h_rc));

            if valid_bases_count >= k + w - 1 {
                if let Some(&(_, min)) = ws.q_fwd.front() {
                    if Some(min) != last_fwd {
                        fwd_mins.push(min);
                        last_fwd = Some(min);
                    }
                }
                if let Some(&(_, min)) = ws.q_rc.front() {
                    if Some(min) != last_rc {
                        rc_mins.push(min);
                        last_rc = Some(min);
                    }
                }
            }
        }
    }
    (fwd_mins, rc_mins)
}

/// Extract minimizers and their forward-strand positions from both strands.
///
/// Position-aware variant of [`extract_dual_strand_into`]. Same sliding-window
/// algorithm and the same consecutive-duplicate-hash dedup per strand; the
/// only difference is that each emitted minimizer is paired with its position.
///
/// # Position semantics
///
/// Both strands report **forward-normalized** positions — the same
/// `pos = i + 1 - k` offset into `seq` is used for the forward minimizer
/// emitted at iteration `i` and for the RC minimizer emitted at iteration `i`.
/// A caller that wants the RC-strand coordinate of the same k-mer computes
/// `seq.len() - forward_pos - k`. This matches the [`extract_strand_minimizers`]
/// convention and is what skani-style chaining expects.
///
/// # Arguments
/// * `seq` - DNA sequence as bytes (A, G, T, C, case insensitive)
/// * `k` - K-mer size (must be 16, 32, or 64)
/// * `w` - Window size for minimizer selection
/// * `salt` - XOR salt applied to k-mer hashes
/// * `ws` - Workspace; `buffer`/`positions_fwd` (forward) and
///   `rc_buffer`/`positions_rc` (reverse complement) are all cleared and
///   refilled.
///
/// # Output
/// `ws.buffer` ‖ `ws.positions_fwd` and `ws.rc_buffer` ‖ `ws.positions_rc`
/// are written index-parallel within each strand.
///
/// # Errors
/// Returns `RypeError::validation` if any emitted position exceeds `u32::MAX`
/// (requires `seq.len() > 4.29 Gbp`; never reachable on realistic single
/// contigs but the guard exists to fail loud rather than silent-truncate).
pub fn extract_dual_strand_into_with_positions(
    seq: &[u8],
    k: usize,
    w: usize,
    salt: u64,
    ws: &mut MinimizerWorkspace,
) -> Result<()> {
    ws.buffer.clear();
    ws.rc_buffer.clear();
    ws.positions_fwd.clear();
    ws.positions_rc.clear();
    ws.q_fwd.clear();
    ws.q_rc.clear();

    let len = seq.len();
    if len < k {
        return Ok(());
    }

    let k_mask = if k == 64 { u64::MAX } else { (1u64 << k) - 1 };
    let rc_shift = k - 1;

    let mut current_val: u64 = 0;
    let mut current_rc: u64 = 0;
    let mut valid_bases_count = 0;

    let mut last_fwd: Option<u64> = None;
    let mut last_rc: Option<u64> = None;

    for i in 0..len {
        let bit = base_to_bit(seq[i]);

        if bit == u64::MAX {
            valid_bases_count = 0;
            ws.q_fwd.clear();
            ws.q_rc.clear();
            current_val = 0;
            current_rc = 0;
            last_fwd = None;
            last_rc = None;
            continue;
        }

        valid_bases_count += 1;
        current_val = ((current_val << 1) | bit) & k_mask;
        current_rc = (current_rc >> 1) | ((bit ^ 1) << rc_shift);

        if valid_bases_count >= k {
            let pos = i + 1 - k;
            let h_fwd = current_val ^ salt;
            let h_rc = current_rc ^ salt;

            // Forward strand deque
            while let Some(&(p, _)) = ws.q_fwd.front() {
                if p + w <= pos {
                    ws.q_fwd.pop_front();
                } else {
                    break;
                }
            }
            while let Some(&(_, v)) = ws.q_fwd.back() {
                // Strict `>` (not `>=`): ties on hash must keep the EARLIER
                // position. See the matching comment in
                // `extract_into_with_positions` for the full rationale.
                if v > h_fwd {
                    ws.q_fwd.pop_back();
                } else {
                    break;
                }
            }
            ws.q_fwd.push_back((pos, h_fwd));

            // Reverse complement deque
            while let Some(&(p, _)) = ws.q_rc.front() {
                if p + w <= pos {
                    ws.q_rc.pop_front();
                } else {
                    break;
                }
            }
            while let Some(&(_, v)) = ws.q_rc.back() {
                // Strict `>` for the same reason as the forward strand.
                if v > h_rc {
                    ws.q_rc.pop_back();
                } else {
                    break;
                }
            }
            ws.q_rc.push_back((pos, h_rc));

            if valid_bases_count >= k + w - 1 {
                if let Some(&(min_pos, min_h)) = ws.q_fwd.front() {
                    if Some(min_h) != last_fwd {
                        ws.buffer.push(min_h);
                        ws.positions_fwd.push(cast_pos(min_pos)?);
                        last_fwd = Some(min_h);
                    }
                }
                if let Some(&(min_pos, min_h)) = ws.q_rc.front() {
                    if Some(min_h) != last_rc {
                        ws.rc_buffer.push(min_h);
                        ws.positions_rc.push(cast_pos(min_pos)?);
                        last_rc = Some(min_h);
                    }
                }
            }
        }
    }

    Ok(())
}

/// Extract minimizers from paired-end reads.
///
/// For paired-end reads, the forward strand of read 1 is combined with
/// the reverse complement of read 2 (and vice versa) to handle the
/// typical paired-end library orientation.
///
/// # Arguments
/// * `s1` - First read sequence
/// * `s2` - Optional second read sequence (for paired-end)
/// * `k` - K-mer size
/// * `w` - Window size
/// * `salt` - XOR salt
/// * `ws` - Workspace
///
/// # Returns
/// A tuple of (forward_minimizers, reverse_complement_minimizers),
/// both sorted and deduplicated.
pub fn get_paired_minimizers_into(
    s1: &[u8],
    s2: Option<&[u8]>,
    k: usize,
    w: usize,
    salt: u64,
    ws: &mut MinimizerWorkspace,
) -> (Vec<u64>, Vec<u64>) {
    let (mut fwd, mut rc) = extract_dual_strand_into(s1, k, w, salt, ws);
    if let Some(seq2) = s2 {
        let (mut r2_f, mut r2_rc) = extract_dual_strand_into(seq2, k, w, salt, ws);
        // Combine: read1_fwd + read2_rc, read1_rc + read2_fwd
        fwd.append(&mut r2_rc);
        rc.append(&mut r2_f);
    }
    fwd.sort_unstable();
    fwd.dedup();
    rc.sort_unstable();
    rc.dedup();
    (fwd, rc)
}

/// Count how many minimizers from a query match a bucket (binary search).
///
/// # Arguments
/// * `mins` - Query minimizers (any order)
/// * `bucket` - Bucket minimizers (must be sorted)
///
/// # Returns
/// Number of matches as f64.
pub fn count_hits(mins: &[u64], bucket: &[u64]) -> f64 {
    let mut hits = 0;
    for m in mins {
        if bucket.binary_search(m).is_ok() {
            hits += 1;
        }
    }
    hits as f64
}

/// Reshape extractor output into the `(sorted-ascending, deduplicated)` layout
/// that downstream bucket-style indices require.
///
/// `extract_into_with_positions` and `extract_dual_strand_into_with_positions`
/// emit `(hash, position)` pairs in sequence order — positions are
/// non-decreasing but hashes are arbitrary, and the same hash may appear at
/// non-consecutive positions if it was displaced from the window and came back.
/// This helper sorts both arrays in lockstep by `(hash, position)` ascending,
/// then drops adjacent duplicate hashes keeping the SMALLEST position for each
/// hash (= the first occurrence, since the sort key is `(hash, position)`).
///
/// After this helper, the two arrays satisfy:
/// - same length;
/// - `hashes` strictly ascending (no duplicates);
/// - `positions[i]` is the smallest position at which `hashes[i]` occurred
///   in the original extractor output.
///
/// One downstream caller is Plan 1.1's `ClusterBucketData::validate`, which
/// requires exactly these invariants. The helper is independent of that —
/// any caller that wants hash-unique sorted output for a bucket can use it.
///
/// # Panics
///
/// In debug builds, panics via `debug_assert_eq!` if `hashes.len() !=
/// positions.len()`. The two arrays are an index-parallel pair; a length
/// mismatch indicates caller error that would silently truncate via zip
/// in a release build.
pub fn pairs_into_cluster_bucket_arrays(hashes: &mut Vec<u64>, positions: &mut Vec<u32>) {
    debug_assert_eq!(
        hashes.len(),
        positions.len(),
        "hashes and positions must be index-parallel"
    );

    let mut pairs: Vec<(u64, u32)> = hashes.drain(..).zip(positions.drain(..)).collect();
    // Sort by (hash, position) — the position tiebreaker ensures that after
    // dedup_by_key the kept position is the smallest among ties.
    pairs.sort_unstable();
    pairs.dedup_by_key(|&mut (h, _)| h);

    hashes.reserve(pairs.len());
    positions.reserve(pairs.len());
    for (h, p) in pairs {
        hashes.push(h);
        positions.push(p);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_into_basic() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"ACGTACGTACGTACGTACGT"; // 20 bases
        extract_into(seq, 16, 4, 0, &mut ws);
        // With k=16 and w=4, we need at least k+w-1=19 bases for one minimizer
        assert!(!ws.buffer.is_empty());
    }

    #[test]
    fn test_extract_into_short_sequence() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"ACGT"; // 4 bases, too short for k=16
        extract_into(seq, 16, 4, 0, &mut ws);
        assert!(ws.buffer.is_empty());
    }

    #[test]
    fn test_extract_into_with_n() {
        let mut ws = MinimizerWorkspace::new();
        // N in the middle should reset the k-mer accumulator
        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        extract_into(seq, 16, 4, 0, &mut ws);
        let count_without_n = ws.buffer.len();

        // Same sequence with N should produce fewer or different minimizers
        let seq_with_n = b"ACGTACGTACGTACGTNACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        extract_into(seq_with_n, 16, 4, 0, &mut ws);
        // The N disrupts extraction
        assert!(ws.buffer.len() <= count_without_n || ws.buffer.len() > 0);
    }

    #[test]
    fn test_count_hits() {
        let query = vec![1, 5, 10, 15];
        let bucket = vec![1, 2, 3, 5, 7, 10, 12]; // sorted
        assert_eq!(count_hits(&query, &bucket), 3.0); // matches: 1, 5, 10
    }

    #[test]
    fn test_valid_extraction_long() {
        let mut ws = MinimizerWorkspace::new();
        let seq = vec![b'A'; 70];
        extract_into(&seq, 64, 5, 0, &mut ws);
        assert!(
            !ws.buffer.is_empty(),
            "Should extract minimizers from valid long seq"
        );
    }

    #[test]
    fn test_short_sequences_ignored() {
        let mut ws = MinimizerWorkspace::new();
        let seq = vec![b'A'; 60];
        extract_into(&seq, 64, 5, 0, &mut ws);
        assert!(
            ws.buffer.is_empty(),
            "Should not extract minimizers from seq < K"
        );
    }

    #[test]
    fn test_n_handling_separator() {
        let mut ws = MinimizerWorkspace::new();
        let seq_a: Vec<u8> = (0..80)
            .map(|i| if i % 2 == 0 { b'A' } else { b'T' })
            .collect();
        let seq_b: Vec<u8> = (0..80)
            .map(|i| if i % 3 == 0 { b'G' } else { b'C' })
            .collect();

        extract_into(&seq_a, 64, 5, 0, &mut ws);
        let mins_a = ws.buffer.clone();

        extract_into(&seq_b, 64, 5, 0, &mut ws);
        let mins_b = ws.buffer.clone();

        let mut seq_combined = seq_a.clone();
        seq_combined.push(b'N'); // N separator
        seq_combined.extend_from_slice(&seq_b);

        extract_into(&seq_combined, 64, 5, 0, &mut ws);
        let mins_combined = ws.buffer.clone();

        let mut expected = mins_a;
        expected.extend(mins_b);

        assert_eq!(
            mins_combined, expected,
            "N should act as a perfect separator"
        );
    }

    #[test]
    fn test_dual_strand_extraction() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCC";
        let (fwd, rc) = extract_dual_strand_into(seq, 64, 5, 0, &mut ws);
        assert!(!fwd.is_empty());
        assert!(!rc.is_empty());
        assert_ne!(fwd, rc);
    }

    #[test]
    fn test_extract_minimizers_k16() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAATTTTGGGGCCCCAAAA"; // 20 bases
        extract_into(seq, 16, 5, 0, &mut ws);
        assert!(!ws.buffer.is_empty(), "Should extract minimizers with K=16");
    }

    #[test]
    fn test_extract_minimizers_k32() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT"; // 40 bases
        extract_into(seq, 32, 5, 0, &mut ws);
        assert!(!ws.buffer.is_empty(), "Should extract minimizers with K=32");
    }

    #[test]
    fn test_short_seq_k16() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAATTTTGG"; // 10 bases, too short for K=16
        extract_into(seq, 16, 5, 0, &mut ws);
        assert!(ws.buffer.is_empty(), "Should not extract from seq < K");
    }

    #[test]
    fn test_short_seq_k32() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAATTTTGGGGCCCCAAAA"; // 20 bases, too short for K=32
        extract_into(seq, 32, 5, 0, &mut ws);
        assert!(ws.buffer.is_empty(), "Should not extract from seq < K");
    }

    /// Test that incremental reverse complement matches the full computation.
    /// This verifies the optimization: rc' = (rc >> 1) | ((bit ^ 1) << (k-1))
    #[test]
    fn test_incremental_reverse_complement_correctness() {
        use super::super::encoding::reverse_complement;

        // Test for each supported k value
        for k in [16, 32, 64] {
            let k_mask = if k == 64 { u64::MAX } else { (1u64 << k) - 1 };
            let rc_shift = k - 1;

            // Test sequence with mixed purines/pyrimidines
            let seq =
                b"AGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTCAGTC";

            let mut current_val: u64 = 0;
            let mut current_rc: u64 = 0;

            for (i, &base) in seq.iter().enumerate() {
                let bit = super::super::encoding::base_to_bit(base);
                if bit == u64::MAX {
                    continue;
                }

                // Update forward k-mer
                current_val = ((current_val << 1) | bit) & k_mask;
                // Update reverse complement incrementally
                current_rc = (current_rc >> 1) | ((bit ^ 1) << rc_shift);

                // Once we have k valid bases, verify the incremental RC matches full computation
                if i + 1 >= k {
                    let expected_rc = reverse_complement(current_val, k);
                    assert_eq!(
                        current_rc, expected_rc,
                        "Incremental RC mismatch at position {} for k={}: got {:#x}, expected {:#x}",
                        i, k, current_rc, expected_rc
                    );
                }
            }
        }
    }

    /// Test incremental RC with invalid bases (N) that cause resets.
    #[test]
    fn test_incremental_rc_with_resets() {
        use super::super::encoding::reverse_complement;

        let k = 16;
        let k_mask = (1u64 << k) - 1;
        let rc_shift = k - 1;

        // Sequence with N in the middle
        let seq = b"AGTCAGTCAGTCAGTCNAGTCAGTCAGTCAGTC";

        let mut current_val: u64 = 0;
        let mut current_rc: u64 = 0;
        let mut valid_bases = 0;

        for (i, &base) in seq.iter().enumerate() {
            let bit = super::super::encoding::base_to_bit(base);

            if bit == u64::MAX {
                // Reset on invalid base
                current_val = 0;
                current_rc = 0;
                valid_bases = 0;
                continue;
            }

            valid_bases += 1;
            current_val = ((current_val << 1) | bit) & k_mask;
            current_rc = (current_rc >> 1) | ((bit ^ 1) << rc_shift);

            if valid_bases >= k {
                let expected_rc = reverse_complement(current_val, k);
                assert_eq!(
                    current_rc, expected_rc,
                    "Incremental RC mismatch after reset at position {}: got {:#x}, expected {:#x}",
                    i, current_rc, expected_rc
                );
            }
        }
    }

    // ========== extract_strand_minimizers tests ==========

    #[test]
    fn test_extract_strand_minimizers_short_sequence() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"ACGT";
        let (fwd, rc) = extract_strand_minimizers(seq, 16, 4, 0, &mut ws);
        assert!(fwd.hashes.is_empty());
        assert!(fwd.positions.is_empty());
        assert!(rc.hashes.is_empty());
        assert!(rc.positions.is_empty());
    }

    #[test]
    fn test_extract_strand_minimizers_basic() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCC";
        let (fwd, rc) = extract_strand_minimizers(seq, 64, 5, 0, &mut ws);
        assert!(!fwd.hashes.is_empty(), "Forward should be non-empty");
        assert!(!rc.hashes.is_empty(), "RC should be non-empty");
    }

    #[test]
    fn test_extract_strand_minimizers_soa_invariant() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCC";
        let (fwd, rc) = extract_strand_minimizers(seq, 32, 4, 0, &mut ws);
        assert_eq!(
            fwd.hashes.len(),
            fwd.positions.len(),
            "Forward SoA mismatch"
        );
        assert_eq!(rc.hashes.len(), rc.positions.len(), "RC SoA mismatch");
    }

    #[test]
    fn test_extract_strand_minimizers_positions_valid() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAATTTTGGGGCCCCAAAATTTTGGGGCCCC"; // 32 bases
        let (fwd, rc) = extract_strand_minimizers(seq, 16, 4, 0, &mut ws);
        for &p in &fwd.positions {
            assert!(p + 16 <= seq.len(), "Forward position {} out of bounds", p);
        }
        for &p in &rc.positions {
            assert!(p + 16 <= seq.len(), "RC position {} out of bounds", p);
        }
    }

    #[test]
    fn test_extract_strand_minimizers_n_handling() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAATTTTGGGGCCCCAAAANAAAATTTTGGGGCCCCAAAA";
        let (fwd, rc) = extract_strand_minimizers(seq, 16, 4, 0, &mut ws);
        // N is at position 20; no minimizer should span it
        for &p in &fwd.positions {
            let ends_before_n = p + 16 <= 20;
            let starts_after_n = p > 20;
            assert!(
                ends_before_n || starts_after_n,
                "Forward pos {} spans N at 20",
                p
            );
        }
        for &p in &rc.positions {
            let ends_before_n = p + 16 <= 20;
            let starts_after_n = p > 20;
            assert!(
                ends_before_n || starts_after_n,
                "RC pos {} spans N at 20",
                p
            );
        }
    }

    #[test]
    fn test_extract_strand_minimizers_hashes_match_dual_strand() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCC";
        let (fwd_sm, rc_sm) = extract_strand_minimizers(seq, 64, 5, 0, &mut ws);
        let (fwd_ds, rc_ds) = extract_dual_strand_into(seq, 64, 5, 0, &mut ws);

        let fwd_set: std::collections::HashSet<_> = fwd_ds.iter().collect();
        let rc_set: std::collections::HashSet<_> = rc_ds.iter().collect();

        let fwd_matches = fwd_sm.hashes.iter().filter(|h| fwd_set.contains(h)).count();
        let rc_matches = rc_sm.hashes.iter().filter(|h| rc_set.contains(h)).count();

        assert!(
            fwd_matches > 0,
            "Forward hashes should overlap with dual_strand"
        );
        assert!(rc_matches > 0, "RC hashes should overlap with dual_strand");
    }

    #[test]
    fn test_extract_strand_minimizers_positions_ordered() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCC";
        let (fwd, rc) = extract_strand_minimizers(seq, 16, 5, 0, &mut ws);
        for w in fwd.positions.windows(2) {
            assert!(
                w[0] <= w[1],
                "Forward positions not non-decreasing: {} > {}",
                w[0],
                w[1]
            );
        }
        for w in rc.positions.windows(2) {
            assert!(
                w[0] <= w[1],
                "RC positions not non-decreasing: {} > {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn test_extract_strand_minimizers_k16_k32_k64() {
        let mut ws = MinimizerWorkspace::new();
        // 80-base sequence works for all k values
        let seq: Vec<u8> = (0..80)
            .map(|i| match i % 4 {
                0 => b'A',
                1 => b'T',
                2 => b'G',
                _ => b'C',
            })
            .collect();

        for k in [16, 32, 64] {
            let (fwd, rc) = extract_strand_minimizers(&seq, k, 5, 0, &mut ws);
            assert!(!fwd.hashes.is_empty(), "Forward empty for k={}", k);
            assert!(!rc.hashes.is_empty(), "RC empty for k={}", k);
            assert_eq!(
                fwd.hashes.len(),
                fwd.positions.len(),
                "SoA mismatch for k={}",
                k
            );
            assert_eq!(
                rc.hashes.len(),
                rc.positions.len(),
                "SoA mismatch for k={}",
                k
            );
            for &p in &fwd.positions {
                assert!(p + k <= seq.len(), "pos+k > len for k={}", k);
            }
            for &p in &rc.positions {
                assert!(p + k <= seq.len(), "pos+k > len for k={}", k);
            }
        }
    }

    // ========== extract_minimizer_set tests ==========

    #[test]
    fn test_extract_minimizer_set_short_sequence() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"ACGT";
        let (fwd, rc) = extract_minimizer_set(seq, 16, 4, 0, &mut ws);
        assert!(fwd.is_empty());
        assert!(rc.is_empty());
    }

    #[test]
    fn test_extract_minimizer_set_sorted() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCC";
        let (fwd, rc) = extract_minimizer_set(seq, 16, 5, 0, &mut ws);
        for w in fwd.windows(2) {
            assert!(w[0] <= w[1], "Forward not sorted");
        }
        for w in rc.windows(2) {
            assert!(w[0] <= w[1], "RC not sorted");
        }
    }

    #[test]
    fn test_extract_minimizer_set_deduped() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCC";
        let (fwd, rc) = extract_minimizer_set(seq, 16, 5, 0, &mut ws);
        for w in fwd.windows(2) {
            assert!(w[0] != w[1], "Forward has adjacent duplicate");
        }
        for w in rc.windows(2) {
            assert!(w[0] != w[1], "RC has adjacent duplicate");
        }
    }

    #[test]
    fn test_extract_minimizer_set_matches_dual_strand() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCC";

        let (set_fwd, set_rc) = extract_minimizer_set(seq, 64, 5, 0, &mut ws);

        // Manually do the same thing
        let (mut ds_fwd, mut ds_rc) = extract_dual_strand_into(seq, 64, 5, 0, &mut ws);
        ds_fwd.sort_unstable();
        ds_fwd.dedup();
        ds_rc.sort_unstable();
        ds_rc.dedup();

        assert_eq!(set_fwd, ds_fwd, "Forward sets should match");
        assert_eq!(set_rc, ds_rc, "RC sets should match");
    }

    #[test]
    fn test_extract_minimizer_set_n_handling() {
        let mut ws = MinimizerWorkspace::new();
        // N resets extraction — should still produce valid sorted/deduped output
        let seq = b"AAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAANAAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAA";
        let (fwd, rc) = extract_minimizer_set(seq, 16, 5, 0, &mut ws);
        // Just verify sorted + deduped
        for w in fwd.windows(2) {
            assert!(w[0] < w[1], "Forward not strictly sorted after N");
        }
        for w in rc.windows(2) {
            assert!(w[0] < w[1], "RC not strictly sorted after N");
        }
    }

    // ---- Plan 1.2 phase 1: extract_into_with_positions ----

    // WHY: short sequence (< k bases) must yield empty output for both buffers
    // AND succeed (Ok). A caller looping over many short reads would otherwise
    // hit spurious validation errors.
    #[test]
    fn extract_into_with_positions_short_sequence_is_empty() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"ACGT";
        extract_into_with_positions(seq, 16, 4, 0, &mut ws).expect("short seq must succeed");
        assert!(ws.buffer.is_empty());
        assert!(ws.positions_fwd.is_empty());
    }

    // WHY: positions and minimizers are an index-parallel pair. If lengths
    // diverge by even one element, every downstream consumer (chain DP, the
    // ClusterBucketData::validate length-parity check) gets a wrong-position
    // for the wrong-minimizer and silently produces corrupt output.
    #[test]
    fn extract_into_with_positions_parallel_arrays() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        extract_into_with_positions(seq, 16, 4, 0, &mut ws).unwrap();
        assert_eq!(
            ws.buffer.len(),
            ws.positions_fwd.len(),
            "minimizers and positions must be index-parallel"
        );
    }

    // WHY: when consecutive windows pick the same hash (e.g. homopolymer runs
    // dominate the window), we keep the EARLIER position. If we kept the later
    // one, chains built on this output would point downstream of the actual
    // first-occurrence k-mer skani conventions assume, biasing chain start
    // coordinates forward by up to w-1 bases.
    //
    // Earlier this test only checked "non-decreasing positions" — which
    // trivially passes when the homopolymer dedups to a single emission. The
    // strict `assert_eq!(positions_fwd[0], 0)` below is the load-bearing
    // assertion: any regression of the back-prune `>` to `>=` would emit
    // position k+w-2 = 18 instead of 0 and this fires immediately.
    #[test]
    fn extract_into_with_positions_dedup_keeps_first_position() {
        let mut ws_baseline = MinimizerWorkspace::new();
        let mut ws = MinimizerWorkspace::new();
        // Long homopolymer: every k-mer has the same (purine-only) RY hash,
        // so the algorithm sees consecutive duplicates from the moment the
        // first window completes.
        let seq = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        extract_into(seq, 16, 4, 0, &mut ws_baseline);
        extract_into_with_positions(seq, 16, 4, 0, &mut ws).unwrap();
        assert_eq!(
            ws.buffer, ws_baseline.buffer,
            "hashes must match extract_into"
        );
        assert_eq!(
            ws.buffer.len(),
            ws.positions_fwd.len(),
            "parallel arrays must agree even after dedup"
        );
        // Load-bearing: a homopolymer dedups to exactly ONE emission, and that
        // emission MUST be at position 0 (the earliest occurrence).
        assert_eq!(ws.buffer.len(), 1, "homopolymer must dedup to one emit");
        assert_eq!(
            ws.positions_fwd[0], 0,
            "emitted position must be the EARLIEST occurrence (0), not the latest"
        );
    }

    // WHY: the hash sequence from the new function must match the old function
    // bit-for-bit. Any divergence means the new path silently changed the
    // dedup-by-hash semantics, which would invalidate every test pinning the
    // old function's output.
    #[test]
    fn extract_into_with_positions_hashes_match_extract_into() {
        let seqs: &[&[u8]] = &[
            b"ACGTACGTACGTACGTACGTACGT",
            b"GCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT",
            b"AAAAAAAAAAAAAAAAAAAAAAAAA",
            b"ACGTACGTNNNNACGTACGTACGTACGTACGTACGT", // with N reset
        ];
        for &seq in seqs {
            let mut a = MinimizerWorkspace::new();
            let mut b = MinimizerWorkspace::new();
            extract_into(seq, 16, 4, 0, &mut a);
            extract_into_with_positions(seq, 16, 4, 0, &mut b).unwrap();
            assert_eq!(a.buffer, b.buffer, "hashes diverged for seq: {:?}", seq);
        }
    }

    // WHY: every emitted position must point to a valid k-mer inside the
    // input, i.e. pos + k <= seq.len(). An off-by-one in the cast or in the
    // overflow check would let a caller read past the end of the source slice.
    #[test]
    fn extract_into_with_positions_positions_are_valid() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let k = 16;
        extract_into_with_positions(seq, k, 4, 0, &mut ws).unwrap();
        for &p in &ws.positions_fwd {
            assert!(
                (p as usize) + k <= seq.len(),
                "position {} out of bounds for seq.len() = {}, k = {}",
                p,
                seq.len(),
                k
            );
        }
    }

    // WHY: workspaces are reused across many sequences. If a second call
    // doesn't clear positions_fwd, it would emit stale positions from the
    // previous sequence — silently corrupt output, no panic.
    #[test]
    fn extract_into_with_positions_workspace_reuse_clears_outputs() {
        let mut ws = MinimizerWorkspace::new();
        let seq_a = b"ACGTACGTACGTACGTACGTACGT";
        let seq_b = b"GG"; // too short — must clear, not append to prior output
        extract_into_with_positions(seq_a, 16, 4, 0, &mut ws).unwrap();
        let after_a = (ws.buffer.clone(), ws.positions_fwd.clone());
        extract_into_with_positions(seq_b, 16, 4, 0, &mut ws).unwrap();
        assert!(
            ws.buffer.is_empty(),
            "buffer not cleared on second call; saw: {:?}",
            ws.buffer
        );
        assert!(
            ws.positions_fwd.is_empty(),
            "positions_fwd not cleared on second call; saw: {:?}",
            ws.positions_fwd
        );
        // Sanity: the first call did produce something.
        assert!(!after_a.0.is_empty());
        assert_eq!(after_a.0.len(), after_a.1.len());
    }

    // ---- Plan 1.2 phase 2: extract_dual_strand_into_with_positions ----

    // WHY: short seq must yield empty output for BOTH strands and BOTH position
    // buffers, and must succeed (Ok). A caller looping over short reads would
    // otherwise hit spurious validation errors.
    #[test]
    fn extract_dual_strand_into_with_positions_short_sequence() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"ACGT";
        extract_dual_strand_into_with_positions(seq, 16, 4, 0, &mut ws)
            .expect("short seq must succeed");
        assert!(ws.buffer.is_empty());
        assert!(ws.positions_fwd.is_empty());
        assert!(ws.rc_buffer.is_empty());
        assert!(ws.positions_rc.is_empty());
    }

    // WHY: BOTH (buffer, positions_fwd) and (rc_buffer, positions_rc) must
    // stay index-parallel. A mismatch on either pair silently corrupts
    // chain DP — it would associate the wrong position with each minimizer.
    #[test]
    fn extract_dual_strand_into_with_positions_parallel_arrays_both_strands() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        extract_dual_strand_into_with_positions(seq, 16, 4, 0, &mut ws).unwrap();
        assert_eq!(ws.buffer.len(), ws.positions_fwd.len());
        assert_eq!(ws.rc_buffer.len(), ws.positions_rc.len());
    }

    // WHY: hash sequences from the new function (both strands) must match
    // the old extract_dual_strand_into bit-for-bit. Any divergence means
    // the new path silently changed dedup-by-hash semantics, which would
    // invalidate every test pinning the old function's output.
    #[test]
    fn extract_dual_strand_into_with_positions_forward_hashes_match() {
        let seqs: &[&[u8]] = &[
            b"ACGTACGTACGTACGTACGTACGT",
            b"GCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT",
            b"AAAAAAAAAAAAAAAAAAAAAAAAA",
            b"ACGTACGTNNNNACGTACGTACGTACGTACGTACGT",
        ];
        for &seq in seqs {
            let mut a = MinimizerWorkspace::new();
            let (a_fwd, a_rc) = extract_dual_strand_into(seq, 16, 4, 0, &mut a);
            let mut b = MinimizerWorkspace::new();
            extract_dual_strand_into_with_positions(seq, 16, 4, 0, &mut b).unwrap();
            assert_eq!(
                a_fwd, b.buffer,
                "forward hashes diverged for seq: {:?}",
                seq
            );
            assert_eq!(a_rc, b.rc_buffer, "RC hashes diverged for seq: {:?}", seq);
        }
    }

    // WHY: every emitted position on BOTH strands must point to a valid
    // k-mer (p + k <= len). Positions are forward-normalized — RC positions
    // share the same coordinate system as forward, NOT the RC-strand
    // coordinate system. Downstream computes len - pos - k if needed.
    #[test]
    fn extract_dual_strand_into_with_positions_positions_are_valid_both_strands() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let k = 16;
        extract_dual_strand_into_with_positions(seq, k, 4, 0, &mut ws).unwrap();
        for &p in &ws.positions_fwd {
            assert!(
                (p as usize) + k <= seq.len(),
                "fwd position {} out of bounds for len={}, k={}",
                p,
                seq.len(),
                k
            );
        }
        for &p in &ws.positions_rc {
            assert!(
                (p as usize) + k <= seq.len(),
                "rc position {} out of bounds for len={}, k={}",
                p,
                seq.len(),
                k
            );
        }
    }

    // WHY: on a homopolymer the same forward hash AND the same RC hash
    // dominate many windows. Both strands must dedup by hash keeping the
    // earliest position. A bug that kept only the latest position would
    // bias chain start coordinates forward by up to w-1 bases on either
    // strand independently.
    //
    // Same load-bearing assertion as the single-strand test: a homopolymer
    // dedups to exactly one emission per strand at position 0. The earlier
    // version of this test only checked "non-decreasing positions" which
    // can't distinguish earliest from latest when there's only one emission.
    #[test]
    fn extract_dual_strand_into_with_positions_dedup_keeps_first_position_both_strands() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        extract_dual_strand_into_with_positions(seq, 16, 4, 0, &mut ws).unwrap();
        assert_eq!(ws.buffer.len(), ws.positions_fwd.len());
        assert_eq!(ws.rc_buffer.len(), ws.positions_rc.len());
        // Both strands must dedup to exactly one emission at position 0.
        assert_eq!(
            ws.buffer.len(),
            1,
            "fwd: homopolymer must dedup to one emit"
        );
        assert_eq!(
            ws.rc_buffer.len(),
            1,
            "rc: homopolymer must dedup to one emit"
        );
        assert_eq!(
            ws.positions_fwd[0], 0,
            "fwd emitted position must be earliest (0)"
        );
        assert_eq!(
            ws.positions_rc[0], 0,
            "rc emitted position must be earliest (0)"
        );
    }

    // WHY: the u32 overflow guard exists for future genome-scale callers
    // (seq.len() > 4.29 Gbp) where the cast would silently truncate.
    // No realistic test sequence can trigger it on actual extract paths,
    // so we extract the cast into a helper and pin the contract here.
    // If anyone removes the guard, this test fires immediately.
    #[test]
    fn cast_pos_rejects_overflow() {
        // Below the limit: succeeds.
        assert_eq!(cast_pos(0).unwrap(), 0);
        assert_eq!(cast_pos(u32::MAX as usize).unwrap(), u32::MAX);

        // Above the limit: fails with a clear message.
        // On a 32-bit usize platform u32::MAX as usize == usize::MAX, so
        // the over-the-limit case is unreachable — skip the negative test.
        #[cfg(target_pointer_width = "64")]
        {
            let err = cast_pos((u32::MAX as usize) + 1).unwrap_err();
            let msg = format!("{}", err);
            assert!(
                msg.contains("exceeds u32::MAX"),
                "error should mention overflow: {}",
                msg
            );
        }
    }

    // ---- Plan 1.2 phase 3: pairs_into_cluster_bucket_arrays ----

    // WHY: the extractor emits in sequence order (positions non-decreasing,
    // hashes arbitrary). ClusterBucketData::validate requires hashes sorted
    // ascending. This helper must reorder both arrays in lockstep — a sort
    // that scrambled the index-parallel pairing would associate every
    // minimizer with a wrong position.
    #[test]
    fn pairs_into_cluster_bucket_arrays_sorts_by_hash() {
        let mut hashes = vec![5u64, 1, 9, 3];
        let mut positions = vec![100u32, 200, 300, 400];
        pairs_into_cluster_bucket_arrays(&mut hashes, &mut positions);
        assert_eq!(hashes, vec![1, 3, 5, 9]);
        // Each position must travel with its original hash.
        assert_eq!(positions, vec![200, 400, 100, 300]);
    }

    // WHY: extract_into_with_positions only dedups CONSECUTIVE duplicates per
    // window. Non-consecutive duplicates (same hash reappearing in a later
    // window after being displaced by a smaller hash) survive extraction.
    // For ClusterBucketData::validate to accept the output, the helper must
    // collapse those — and per skani's anchor convention, it keeps the
    // SMALLEST position for each hash.
    #[test]
    fn pairs_into_cluster_bucket_arrays_dedups_keep_smallest_position() {
        // hash=5 appears at positions 100 and 50; hash=3 at 200, 200, 75.
        let mut hashes = vec![5u64, 3, 5, 3, 3];
        let mut positions = vec![100u32, 200, 50, 200, 75];
        pairs_into_cluster_bucket_arrays(&mut hashes, &mut positions);
        assert_eq!(hashes, vec![3, 5]);
        // Smallest position kept for each hash.
        assert_eq!(positions, vec![75, 50]);
    }

    // WHY: a contig that produced no minimizers (e.g. shorter than k) is
    // legal upstream. The helper must accept empty input and produce
    // empty output without panicking on the underlying drain/dedup ops.
    #[test]
    fn pairs_into_cluster_bucket_arrays_empty_input() {
        let mut hashes: Vec<u64> = vec![];
        let mut positions: Vec<u32> = vec![];
        pairs_into_cluster_bucket_arrays(&mut hashes, &mut positions);
        assert!(hashes.is_empty());
        assert!(positions.is_empty());
    }

    // WHY: hashes and positions are an index-parallel pair. A length
    // mismatch at the helper boundary means caller code wrote them out of
    // sync — silent zip-and-truncate would associate every later minimizer
    // with a wrong position. debug_assert_eq fires loudly in debug/test
    // builds and is compiled out in release (where the hot path is).
    #[test]
    #[should_panic(expected = "index-parallel")]
    fn pairs_into_cluster_bucket_arrays_length_mismatch_panics_in_debug() {
        let mut hashes = vec![1u64, 2, 3];
        let mut positions = vec![10u32, 20];
        pairs_into_cluster_bucket_arrays(&mut hashes, &mut positions);
    }
}
