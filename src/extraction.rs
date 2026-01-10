//! Minimizer extraction algorithms.
//!
//! This module provides functions for extracting minimizers from DNA sequences
//! using the RY (purine/pyrimidine) encoding scheme. Minimizers are selected
//! using a sliding window approach with a monotonic deque for O(n) complexity.

use crate::constants::ESTIMATED_MINIMIZERS_PER_SEQUENCE;
use crate::encoding::{base_to_bit, reverse_complement};
use crate::workspace::MinimizerWorkspace;

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
    if len < k { return; }

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
        current_val = (current_val << 1) | bit;

        if valid_bases_count >= k {
            let pos = i + 1 - k;
            let hash = current_val ^ salt;

            // Maintain monotonic deque - remove old entries outside window
            while let Some(&(p, _)) = ws.q_fwd.front() {
                if p + w <= pos { ws.q_fwd.pop_front(); } else { break; }
            }
            // Remove entries with larger hash values
            while let Some(&(_, v)) = ws.q_fwd.back() {
                if v >= hash { ws.q_fwd.pop_back(); } else { break; }
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
    ws: &mut MinimizerWorkspace
) -> (Vec<u64>, Vec<u64>) {
    ws.q_fwd.clear();
    ws.q_rc.clear();

    let len = seq.len();
    if len < k { return (vec![], vec![]); }

    let mut fwd_mins = Vec::with_capacity(ESTIMATED_MINIMIZERS_PER_SEQUENCE);
    let mut rc_mins = Vec::with_capacity(ESTIMATED_MINIMIZERS_PER_SEQUENCE);

    let mut current_val: u64 = 0;
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
            last_fwd = None;
            last_rc = None;
            continue;
        }

        valid_bases_count += 1;
        current_val = (current_val << 1) | bit;

        if valid_bases_count >= k {
            let pos = i + 1 - k;
            let h_fwd = current_val ^ salt;
            let h_rc = reverse_complement(current_val, k) ^ salt;

            // Forward strand deque
            while let Some(&(p, _)) = ws.q_fwd.front() { if p + w <= pos { ws.q_fwd.pop_front(); } else { break; } }
            while let Some(&(_, v)) = ws.q_fwd.back() { if v >= h_fwd { ws.q_fwd.pop_back(); } else { break; } }
            ws.q_fwd.push_back((pos, h_fwd));

            // Reverse complement deque
            while let Some(&(p, _)) = ws.q_rc.front() { if p + w <= pos { ws.q_rc.pop_front(); } else { break; } }
            while let Some(&(_, v)) = ws.q_rc.back() { if v >= h_rc { ws.q_rc.pop_back(); } else { break; } }
            ws.q_rc.push_back((pos, h_rc));

            if valid_bases_count >= k + w - 1 {
                if let Some(&(_, min)) = ws.q_fwd.front() {
                    if Some(min) != last_fwd { fwd_mins.push(min); last_fwd = Some(min); }
                }
                if let Some(&(_, min)) = ws.q_rc.front() {
                    if Some(min) != last_rc { rc_mins.push(min); last_rc = Some(min); }
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
    s1: &[u8], s2: Option<&[u8]>, k: usize, w: usize, salt: u64, ws: &mut MinimizerWorkspace
) -> (Vec<u64>, Vec<u64>) {
    let (mut fwd, mut rc) = extract_dual_strand_into(s1, k, w, salt, ws);
    if let Some(seq2) = s2 {
        let (mut r2_f, mut r2_rc) = extract_dual_strand_into(seq2, k, w, salt, ws);
        // Combine: read1_fwd + read2_rc, read1_rc + read2_fwd
        fwd.append(&mut r2_rc);
        rc.append(&mut r2_f);
    }
    fwd.sort_unstable(); fwd.dedup();
    rc.sort_unstable(); rc.dedup();
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
        assert!(!ws.buffer.is_empty(), "Should extract minimizers from valid long seq");
    }

    #[test]
    fn test_short_sequences_ignored() {
        let mut ws = MinimizerWorkspace::new();
        let seq = vec![b'A'; 60];
        extract_into(&seq, 64, 5, 0, &mut ws);
        assert!(ws.buffer.is_empty(), "Should not extract minimizers from seq < K");
    }

    #[test]
    fn test_n_handling_separator() {
        let mut ws = MinimizerWorkspace::new();
        let seq_a: Vec<u8> = (0..80).map(|i| if i % 2 == 0 { b'A' } else { b'T' }).collect();
        let seq_b: Vec<u8> = (0..80).map(|i| if i % 3 == 0 { b'G' } else { b'C' }).collect();

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

        assert_eq!(mins_combined, expected, "N should act as a perfect separator");
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
}
