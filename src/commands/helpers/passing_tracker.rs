//! Compact bitset tracker for reads that pass the log-ratio filter.
//!
//! Instead of carrying sequence data through the classification pipeline,
//! this tracker records which reads pass the filter using 1 bit per read.
//! After all classification is complete, the input is re-walked to write
//! only the passing sequences.

/// Bitset tracker for passing reads.
///
/// Uses a growable `Vec<u64>` where each bit represents one read.
/// Memory usage: 1 bit per read (e.g., 100M reads = ~12.5MB).
pub struct PassingReadTracker {
    bits: Vec<u64>,
    count: usize,
}

impl PassingReadTracker {
    /// Create a new tracker with initial capacity for `num_reads` reads.
    pub fn with_capacity(num_reads: usize) -> Self {
        let words = (num_reads + 63) / 64;
        Self {
            bits: vec![0u64; words],
            count: 0,
        }
    }

    /// Mark a read at `global_index` as passing.
    ///
    /// Not thread-safe: caller must ensure exclusive access (single-threaded batch loop).
    pub fn mark(&mut self, global_index: usize) {
        let word = global_index / 64;
        let bit = global_index % 64;
        if word >= self.bits.len() {
            self.bits.resize(word + 1, 0);
        }
        let mask = 1u64 << bit;
        if self.bits[word] & mask == 0 {
            self.count += 1;
        }
        self.bits[word] |= mask;
    }

    /// Check if a read at `global_index` is marked as passing.
    pub fn is_passing(&self, global_index: usize) -> bool {
        let word = global_index / 64;
        let bit = global_index % 64;
        if word >= self.bits.len() {
            return false;
        }
        self.bits[word] & (1u64 << bit) != 0
    }

    /// Number of reads marked as passing.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Whether any reads are marked as passing.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_tracker() {
        let tracker = PassingReadTracker::with_capacity(100);
        assert!(tracker.is_empty());
        assert_eq!(tracker.count(), 0);
        assert!(!tracker.is_passing(0));
        assert!(!tracker.is_passing(99));
    }

    #[test]
    fn test_mark_and_check() {
        let mut tracker = PassingReadTracker::with_capacity(100);
        tracker.mark(5);
        tracker.mark(42);
        tracker.mark(99);

        assert!(tracker.is_passing(5));
        assert!(tracker.is_passing(42));
        assert!(tracker.is_passing(99));
        assert!(!tracker.is_passing(0));
        assert!(!tracker.is_passing(6));
        assert!(!tracker.is_passing(41));
        assert_eq!(tracker.count(), 3);
    }

    #[test]
    fn test_double_mark_no_double_count() {
        let mut tracker = PassingReadTracker::with_capacity(100);
        tracker.mark(10);
        tracker.mark(10);
        assert_eq!(tracker.count(), 1);
    }

    #[test]
    fn test_bit_boundaries() {
        let mut tracker = PassingReadTracker::with_capacity(200);
        // Test at u64 word boundaries
        tracker.mark(63);
        tracker.mark(64);
        tracker.mark(127);
        tracker.mark(128);

        assert!(tracker.is_passing(63));
        assert!(tracker.is_passing(64));
        assert!(tracker.is_passing(127));
        assert!(tracker.is_passing(128));
        assert!(!tracker.is_passing(62));
        assert!(!tracker.is_passing(65));
        assert_eq!(tracker.count(), 4);
    }

    #[test]
    fn test_growth_beyond_capacity() {
        let mut tracker = PassingReadTracker::with_capacity(10);
        // Mark beyond initial capacity
        tracker.mark(1000);
        assert!(tracker.is_passing(1000));
        assert!(!tracker.is_passing(999));
        assert_eq!(tracker.count(), 1);
    }

    #[test]
    fn test_is_passing_beyond_capacity() {
        let tracker = PassingReadTracker::with_capacity(10);
        // Checking beyond capacity should return false, not panic
        assert!(!tracker.is_passing(10000));
    }
}
