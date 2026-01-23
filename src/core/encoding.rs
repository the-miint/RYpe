//! RY-space encoding utilities.
//!
//! This module provides:
//! - Base-to-bit conversion (purines → 1, pyrimidines → 0)
//! - Reverse complement computation for k-mers

/// Lookup table for base → bit conversion in RY space.
/// - Purines (A/G) → 1
/// - Pyrimidines (T/C) → 0
/// - Other bases → u64::MAX (invalid)
pub(crate) const BASE_TO_BIT_LUT: [u64; 256] = {
    let mut lut = [u64::MAX; 256];
    lut[b'A' as usize] = 1;
    lut[b'a' as usize] = 1;
    lut[b'G' as usize] = 1;
    lut[b'g' as usize] = 1;
    lut[b'T' as usize] = 0;
    lut[b't' as usize] = 0;
    lut[b'C' as usize] = 0;
    lut[b'c' as usize] = 0;
    lut
};

/// Convert a nucleotide base to its RY-space bit representation.
///
/// # Safety
/// Uses unchecked array access for performance. The lookup table covers
/// all 256 possible byte values, so this is safe for any input.
#[inline(always)]
pub fn base_to_bit(byte: u8) -> u64 {
    unsafe { *BASE_TO_BIT_LUT.get_unchecked(byte as usize) }
}

/// Compute reverse complement of a k-mer in RY space.
///
/// In RY space, complement is simply bitwise NOT. The reverse complement
/// is computed by complementing and then reversing the bit order.
///
/// Note: The hot path uses incremental reverse complement computation instead
/// of this function. This is retained for verification tests and edge cases.
///
/// # Arguments
/// * `kmer` - The k-mer value (rightmost K bits are significant)
/// * `k` - K-mer size (must be 16, 32, or 64)
///
/// # Returns
/// The reversed and complemented bits in the rightmost K positions.
///
/// # Panics
/// Panics if k is not 16, 32, or 64.
#[inline]
#[allow(dead_code)] // Used in tests to verify incremental RC computation
pub(crate) fn reverse_complement(kmer: u64, k: usize) -> u64 {
    let complement = !kmer;
    match k {
        16 => complement.reverse_bits() >> 48,
        32 => complement.reverse_bits() >> 32,
        64 => complement.reverse_bits(),
        _ => panic!("Unsupported K value: {}", k),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_to_bit() {
        assert_eq!(base_to_bit(b'A'), 1);
        assert_eq!(base_to_bit(b'a'), 1);
        assert_eq!(base_to_bit(b'G'), 1);
        assert_eq!(base_to_bit(b'g'), 1);
        assert_eq!(base_to_bit(b'T'), 0);
        assert_eq!(base_to_bit(b't'), 0);
        assert_eq!(base_to_bit(b'C'), 0);
        assert_eq!(base_to_bit(b'c'), 0);
        assert_eq!(base_to_bit(b'N'), u64::MAX);
    }

    #[test]
    fn test_reverse_complement() {
        // In RY space with K=16:
        // A sequence of all purines (1s) should complement to all 0s
        let all_ones_16 = 0xFFFF_u64; // 16 bits of 1
        let rc = reverse_complement(all_ones_16, 16);
        assert_eq!(rc, 0); // complement of all 1s reversed is all 0s
    }
}
