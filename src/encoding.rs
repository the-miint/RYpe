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

/// Maximum number of bytes needed to encode a u64 as LEB128 varint.
pub(crate) const MAX_VARINT_BYTES: usize = 10;

/// Error type for varint decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VarIntError {
    /// Buffer ended before varint was complete (continuation bit was set on last byte).
    /// Contains the number of bytes that were available.
    Truncated(usize),
    /// Varint exceeds maximum size (>10 bytes for u64).
    /// Contains the number of bytes consumed before overflow.
    Overflow(usize),
}

impl std::fmt::Display for VarIntError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VarIntError::Truncated(bytes) => {
                write!(f, "Truncated varint: buffer ended after {} bytes with continuation bit set", bytes)
            }
            VarIntError::Overflow(bytes) => {
                write!(f, "Malformed varint: exceeded 10 bytes at {} bytes consumed", bytes)
            }
        }
    }
}

impl std::error::Error for VarIntError {}

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
/// * `Ok((value, bytes_consumed))` - Successfully decoded varint
/// * `Err(VarIntError::Truncated(n))` - Buffer ended with continuation bit set after n bytes
/// * `Err(VarIntError::Overflow(n))` - Varint exceeded 10 bytes
///
/// # Example
/// ```ignore
/// let buf = [0x80, 0x80, 0x01]; // encodes 16384
/// let (val, consumed) = decode_varint(&buf)?;
/// assert_eq!(val, 16384);
/// assert_eq!(consumed, 3);
/// ```
#[inline]
pub(crate) fn decode_varint(buf: &[u8]) -> Result<(u64, usize), VarIntError> {
    let mut value: u64 = 0;
    let mut shift = 0;
    let mut i = 0;
    loop {
        // Bounds check - prevent buffer overrun
        if i >= buf.len() {
            // Buffer exhausted with continuation bit still set on previous byte
            return Err(VarIntError::Truncated(i));
        }
        let byte = buf[i];
        value |= ((byte & 0x7F) as u64) << shift;
        i += 1;
        if byte & 0x80 == 0 {
            // Complete varint - continuation bit clear
            return Ok((value, i));
        }
        shift += 7;
        if shift >= 64 {
            // Malformed varint (>10 bytes)
            return Err(VarIntError::Overflow(i));
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
            let (decoded, consumed) = decode_varint(&buf[..len]).expect("decode failed");
            assert_eq!(decoded, val);
            assert_eq!(consumed, len);
        }

        // Test large value
        let large = u64::MAX;
        let len = encode_varint(large, &mut buf);
        let (decoded, consumed) = decode_varint(&buf[..len]).expect("decode failed");
        assert_eq!(decoded, large);
        assert_eq!(consumed, len);
    }

    #[test]
    fn test_varint_truncation_detection() {
        let mut buf = [0u8; 10];

        // Encode a 3-byte varint (16384 = [0x80, 0x80, 0x01])
        let val = 16384u64;
        let len = encode_varint(val, &mut buf);
        assert_eq!(len, 3);
        assert_eq!(&buf[..3], &[0x80, 0x80, 0x01]);

        // Full buffer decodes correctly
        let result = decode_varint(&buf[..3]);
        assert_eq!(result, Ok((16384, 3)));

        // Truncated to 2 bytes should return Truncated error
        let result = decode_varint(&buf[..2]);
        assert_eq!(result, Err(VarIntError::Truncated(2)));

        // Truncated to 1 byte should return Truncated error
        let result = decode_varint(&buf[..1]);
        assert_eq!(result, Err(VarIntError::Truncated(1)));

        // Empty buffer should return Truncated error
        let result = decode_varint(&[]);
        assert_eq!(result, Err(VarIntError::Truncated(0)));
    }

    #[test]
    fn test_varint_large_values() {
        let mut buf = [0u8; 10];

        // Test values that require different numbers of bytes
        let test_cases = [
            (0u64, 1),
            (127u64, 1),
            (128u64, 2),
            (16383u64, 2),
            (16384u64, 3),
            (2097151u64, 3),
            (2097152u64, 4),
            (u64::MAX >> 1, 9),
            (u64::MAX, 10),
        ];

        for (val, expected_len) in test_cases {
            let len = encode_varint(val, &mut buf);
            assert_eq!(len, expected_len, "Encoded length mismatch for {}", val);

            let (decoded, consumed) = decode_varint(&buf[..len]).expect("decode failed");
            assert_eq!(decoded, val, "Value mismatch for {}", val);
            assert_eq!(consumed, len, "Consumed mismatch for {}", val);
        }
    }

    #[test]
    fn test_varint_boundary_values() {
        let mut buf = [0u8; 10];

        // Values at byte boundaries (where encoding length changes)
        let boundary_values = [
            127u64,           // max 1-byte
            128u64,           // min 2-byte
            16383u64,         // max 2-byte
            16384u64,         // min 3-byte
            2097151u64,       // max 3-byte
            2097152u64,       // min 4-byte
            268435455u64,     // max 4-byte
            268435456u64,     // min 5-byte
        ];

        for val in boundary_values {
            let len = encode_varint(val, &mut buf);
            let (decoded, _) = decode_varint(&buf[..len]).expect("decode failed");
            assert_eq!(decoded, val, "Boundary value {} failed roundtrip", val);

            // Test truncation at each byte
            for truncate_at in 1..len {
                let result = decode_varint(&buf[..truncate_at]);
                assert!(
                    result.is_err(),
                    "Truncation at byte {} of {} should fail for value {}",
                    truncate_at, len, val
                );
                assert_eq!(
                    result,
                    Err(VarIntError::Truncated(truncate_at)),
                    "Wrong error type for truncation"
                );
            }
        }
    }
}
