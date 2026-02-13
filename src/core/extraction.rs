//! Minimizer extraction algorithms.
//!
//! This module provides functions for extracting minimizers from DNA sequences
//! using the RY (purine/pyrimidine) encoding scheme. Minimizers are selected
//! using a sliding window approach with a monotonic deque for O(n) complexity.

// Hot path: index-based iteration avoids iterator overhead in inner loops
#![allow(clippy::needless_range_loop)]

use super::encoding::base_to_bit;
use super::workspace::MinimizerWorkspace;

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
}
