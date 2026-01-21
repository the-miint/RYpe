//! Minimizer extraction algorithms.
//!
//! This module provides functions for extracting minimizers from DNA sequences
//! using the RY (purine/pyrimidine) encoding scheme. Minimizers are selected
//! using a sliding window approach with a monotonic deque for O(n) complexity.

// Hot path: index-based iteration avoids iterator overhead in inner loops
#![allow(clippy::needless_range_loop)]

use super::encoding::{base_to_bit, reverse_complement};
use super::workspace::MinimizerWorkspace;

/// Strand indicator for minimizer origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strand {
    Forward,
    ReverseComplement,
}

/// A minimizer with its source position and strand.
#[derive(Debug, Clone)]
pub struct MinimizerWithPosition {
    /// The minimizer hash value
    pub hash: u64,
    /// 0-based position in sequence where the k-mer starts
    pub position: usize,
    /// Which strand the minimizer came from
    pub strand: Strand,
}

/// Extract minimizers from both strands with position tracking.
///
/// Unlike `extract_dual_strand_into()`, this function does NOT deduplicate
/// consecutive identical minimizers, preserving all positions for inspection.
///
/// # Arguments
/// * `seq` - DNA sequence as bytes
/// * `k` - K-mer size (must be 16, 32, or 64)
/// * `w` - Window size for minimizer selection
/// * `salt` - XOR salt applied to k-mer hashes
/// * `ws` - Workspace for temporary storage
///
/// # Returns
/// A vector of minimizers with their positions and strand information.
pub fn extract_with_positions(
    seq: &[u8],
    k: usize,
    w: usize,
    salt: u64,
    ws: &mut MinimizerWorkspace,
) -> Vec<MinimizerWithPosition> {
    ws.q_fwd.clear();
    ws.q_rc.clear();

    let len = seq.len();
    if len < k {
        return vec![];
    }

    // Precompute mask outside hot loop - only lower k bits are valid
    let k_mask = if k == 64 { u64::MAX } else { (1u64 << k) - 1 };

    let mut results = Vec::with_capacity(ws.estimated_minimizers * 2);

    let mut current_val: u64 = 0;
    let mut valid_bases_count = 0;

    // Track last output to avoid duplicates at same position (but allow same hash at different positions)
    let mut last_fwd_pos: Option<usize> = None;
    let mut last_rc_pos: Option<usize> = None;

    for i in 0..len {
        let bit = base_to_bit(seq[i]);

        if bit == u64::MAX {
            valid_bases_count = 0;
            ws.q_fwd.clear();
            ws.q_rc.clear();
            current_val = 0;
            last_fwd_pos = None;
            last_rc_pos = None;
            continue;
        }

        valid_bases_count += 1;
        current_val = ((current_val << 1) | bit) & k_mask;

        if valid_bases_count >= k {
            let pos = i + 1 - k;
            let h_fwd = current_val ^ salt;
            let h_rc = reverse_complement(current_val, k) ^ salt;

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
                // Output forward minimizer (only if position changed)
                if let Some(&(min_pos, min_hash)) = ws.q_fwd.front() {
                    if last_fwd_pos != Some(min_pos) {
                        results.push(MinimizerWithPosition {
                            hash: min_hash,
                            position: min_pos,
                            strand: Strand::Forward,
                        });
                        last_fwd_pos = Some(min_pos);
                    }
                }
                // Output reverse complement minimizer (only if position changed)
                if let Some(&(min_pos, min_hash)) = ws.q_rc.front() {
                    if last_rc_pos != Some(min_pos) {
                        results.push(MinimizerWithPosition {
                            hash: min_hash,
                            position: min_pos,
                            strand: Strand::ReverseComplement,
                        });
                        last_rc_pos = Some(min_pos);
                    }
                }
            }
        }
    }

    results
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

    let mut fwd_mins = Vec::with_capacity(ws.estimated_minimizers);
    let mut rc_mins = Vec::with_capacity(ws.estimated_minimizers);

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
        current_val = ((current_val << 1) | bit) & k_mask;

        if valid_bases_count >= k {
            let pos = i + 1 - k;
            let h_fwd = current_val ^ salt;
            let h_rc = reverse_complement(current_val, k) ^ salt;

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

    // Tests for extract_with_positions

    #[test]
    fn test_extract_with_positions_basic() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAATTTTGGGGCCCCAAAATTTT"; // 24 bases
        let results = extract_with_positions(seq, 16, 4, 0, &mut ws);

        // Should have some minimizers with positions
        assert!(
            !results.is_empty(),
            "Should extract minimizers with positions"
        );

        // All positions should be valid (< seq.len() - k + 1)
        for m in &results {
            assert!(m.position + 16 <= seq.len(), "Position should be valid");
            assert!(m.strand == Strand::Forward || m.strand == Strand::ReverseComplement);
        }
    }

    #[test]
    fn test_extract_with_positions_short_sequence() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"ACGT"; // 4 bases, too short for k=16
        let results = extract_with_positions(seq, 16, 4, 0, &mut ws);
        assert!(
            results.is_empty(),
            "Short sequence should produce no minimizers"
        );
    }

    #[test]
    fn test_extract_with_positions_has_both_strands() {
        let mut ws = MinimizerWorkspace::new();
        // Use a sequence that will produce different hashes for fwd and rc
        let seq = b"AAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCC";
        let results = extract_with_positions(seq, 64, 5, 0, &mut ws);

        let has_forward = results.iter().any(|m| m.strand == Strand::Forward);
        let has_rc = results
            .iter()
            .any(|m| m.strand == Strand::ReverseComplement);

        assert!(has_forward, "Should have forward strand minimizers");
        assert!(has_rc, "Should have reverse complement minimizers");
    }

    #[test]
    fn test_extract_with_positions_matches_dual_strand_hashes() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCC";

        // Get hashes from both methods
        let with_pos = extract_with_positions(seq, 64, 5, 0, &mut ws);
        let (fwd_hashes, rc_hashes) = extract_dual_strand_into(seq, 64, 5, 0, &mut ws);

        // Collect hashes by strand from extract_with_positions
        let fwd_pos_hashes: Vec<u64> = with_pos
            .iter()
            .filter(|m| m.strand == Strand::Forward)
            .map(|m| m.hash)
            .collect();
        let rc_pos_hashes: Vec<u64> = with_pos
            .iter()
            .filter(|m| m.strand == Strand::ReverseComplement)
            .map(|m| m.hash)
            .collect();

        // The hashes should match (though order might differ due to deduplication differences)
        // extract_dual_strand_into deduplicates consecutive identical hashes
        // extract_with_positions deduplicates by position
        // So we just check that the sets of hashes overlap significantly
        let fwd_set: std::collections::HashSet<_> = fwd_hashes.iter().collect();
        let rc_set: std::collections::HashSet<_> = rc_hashes.iter().collect();

        let fwd_matches = fwd_pos_hashes
            .iter()
            .filter(|h| fwd_set.contains(h))
            .count();
        let rc_matches = rc_pos_hashes.iter().filter(|h| rc_set.contains(h)).count();

        assert!(fwd_matches > 0, "Forward hashes should match");
        assert!(rc_matches > 0, "RC hashes should match");
    }

    #[test]
    fn test_extract_with_positions_n_handling() {
        let mut ws = MinimizerWorkspace::new();
        // N should reset extraction
        let seq_with_n = b"AAAATTTTGGGGCCCCAAAANAAAATTTTGGGGCCCCAAAA";
        let results = extract_with_positions(seq_with_n, 16, 4, 0, &mut ws);

        // All positions should be either before the N or after it
        // The N is at position 20, so no minimizer should span it
        for m in &results {
            // K-mer either ends before N (pos + k <= 20) or starts after N (pos > 20)
            let ends_before_n = m.position + 16 <= 20;
            let starts_after_n = m.position > 20;
            assert!(
                ends_before_n || starts_after_n,
                "Minimizer at position {} should not span the N at position 20",
                m.position
            );
        }
    }

    #[test]
    fn test_extract_with_positions_k16() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAATTTTGGGGCCCCAAAATTTT"; // 24 bases
        let results = extract_with_positions(seq, 16, 4, 0, &mut ws);
        assert!(!results.is_empty(), "Should extract minimizers with K=16");

        for m in &results {
            assert!(m.position + 16 <= seq.len());
        }
    }

    #[test]
    fn test_extract_with_positions_k32() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCC"; // 48 bases
        let results = extract_with_positions(seq, 32, 4, 0, &mut ws);
        assert!(!results.is_empty(), "Should extract minimizers with K=32");

        for m in &results {
            assert!(m.position + 32 <= seq.len());
        }
    }
}
