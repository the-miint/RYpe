//! RY-space encoding and varint utilities.
//!
//! This module provides:
//! - Base-to-bit conversion (purines → 1, pyrimidines → 0)
//! - Reverse complement computation for k-mers
//! - LEB128 varint encoding/decoding for compact storage

/// Lookup table for base → bit conversion in RY space.
/// - Purines (A/G) → 1
/// - Pyrimidines (T/C) → 0
/// - Other bases → u64::MAX (invalid)
pub(crate) const BASE_TO_BIT_LUT: [u64; 256] = {
    let mut lut = [u64::MAX; 256];
    lut[b'A' as usize] = 1; lut[b'a' as usize] = 1;
    lut[b'G' as usize] = 1; lut[b'g' as usize] = 1;
    lut[b'T' as usize] = 0; lut[b't' as usize] = 0;
    lut[b'C' as usize] = 0; lut[b'c' as usize] = 0;
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
pub(crate) fn reverse_complement(kmer: u64, k: usize) -> u64 {
    let complement = !kmer;
    match k {
        16 => complement.reverse_bits() >> 48,
        32 => complement.reverse_bits() >> 32,
        64 => complement.reverse_bits(),
        _ => panic!("Unsupported K value: {}", k),
    }
}

// --- VARINT ENCODING (LEB128) ---

/// Encode a u64 as a variable-length integer (LEB128 format).
///
/// Smaller values use fewer bytes (1 byte for 0-127, 2 bytes for 128-16383, etc.).
/// This is useful for delta encoding where most deltas are small.
///
/// # Arguments
/// * `value` - The value to encode
/// * `buf` - Output buffer (must be at least 10 bytes for u64)
///
/// # Returns
/// The number of bytes written to buf.
#[inline]
pub(crate) fn encode_varint(mut value: u64, buf: &mut [u8]) -> usize {
    let mut i = 0;
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            buf[i] = byte;
            return i + 1;
        } else {
            buf[i] = byte | 0x80;
            i += 1;
        }
    }
}

/// Decode a variable-length integer from a byte slice.
///
/// # Arguments
/// * `buf` - Input buffer containing the varint
///
/// # Returns
/// A tuple of (decoded_value, bytes_consumed). If the buffer is truncated
/// or the varint is malformed (>10 bytes), returns (partial_value, bytes_read).
/// The caller should validate that the entire varint was consumed.
#[inline]
pub(crate) fn decode_varint(buf: &[u8]) -> (u64, usize) {
    let mut value: u64 = 0;
    let mut shift = 0;
    let mut i = 0;
    loop {
        // Bounds check - prevent buffer overrun
        if i >= buf.len() {
            // Truncated varint - return what we have
            return (value, i);
        }
        let byte = buf[i];
        value |= ((byte & 0x7F) as u64) << shift;
        i += 1;
        if byte & 0x80 == 0 {
            return (value, i);
        }
        shift += 7;
        if shift >= 64 {
            // Malformed varint (>10 bytes) - return what we have
            return (value, i);
        }
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

    #[test]
    fn test_varint_roundtrip() {
        let mut buf = [0u8; 10];

        // Test small values
        for val in [0u64, 1, 127, 128, 255, 256, 16383, 16384] {
            let len = encode_varint(val, &mut buf);
            let (decoded, consumed) = decode_varint(&buf[..len]);
            assert_eq!(decoded, val);
            assert_eq!(consumed, len);
        }

        // Test large value
        let large = u64::MAX;
        let len = encode_varint(large, &mut buf);
        let (decoded, consumed) = decode_varint(&buf[..len]);
        assert_eq!(decoded, large);
        assert_eq!(consumed, len);
    }
}
