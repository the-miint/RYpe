//! Deferred denominator classification buffer for log-ratio mode.
//!
//! Accumulates reads that need denominator classification across batches,
//! flushing only when the buffer reaches a threshold. This avoids paying
//! the full shard I/O cost for every batch when most reads fast-path.

/// A read deferred for later denominator classification.
///
/// Stores the header, numerator score, and pre-extracted minimizers.
/// The minimizers were extracted during numerator classification and are
/// cached here to avoid redundant re-extraction against the denominator.
pub struct DeferredRead {
    pub header: String,
    pub num_score: f64,
    /// Cached (forward, reverse-complement) minimizers from numerator extraction.
    pub minimizers: (Vec<u64>, Vec<u64>),
    /// Global read index (position in the input file) for bitset tracking.
    pub global_index: usize,
}

impl DeferredRead {
    /// Approximate heap bytes owned by this deferred read.
    ///
    /// Counts String/Vec capacity (not len) plus Vec struct overhead (24 bytes each).
    pub fn approx_heap_bytes(&self) -> usize {
        self.header.capacity()
            + self.minimizers.0.capacity() * 8
            + self.minimizers.1.capacity() * 8
            + 24 * 3 // Vec overhead for header String + 2 minimizer Vecs
    }
}

/// Buffer that accumulates deferred-denom reads and flushes at a threshold.
///
/// Fast-path results are emitted immediately per batch; only reads needing
/// the denominator are buffered here. When the buffer reaches `threshold`
/// reads, the caller should drain and classify them in one pass.
pub struct DeferredDenomBuffer {
    reads: Vec<DeferredRead>,
    threshold: usize,
    approx_bytes: usize,
}

impl DeferredDenomBuffer {
    /// Create a new buffer that triggers flush at `threshold` accumulated reads.
    pub fn new(threshold: usize) -> Self {
        Self {
            reads: Vec::new(),
            threshold,
            approx_bytes: 0,
        }
    }

    /// Number of reads currently buffered.
    pub fn len(&self) -> usize {
        self.reads.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.reads.is_empty()
    }

    /// Approximate heap bytes used by buffered reads.
    pub fn approx_bytes(&self) -> usize {
        self.approx_bytes
    }

    /// Add a deferred read to the buffer.
    pub fn push(&mut self, read: DeferredRead) {
        self.approx_bytes += read.approx_heap_bytes();
        self.reads.push(read);
    }

    /// Returns true when the buffer has reached or exceeded the flush threshold.
    pub fn should_flush(&self) -> bool {
        self.reads.len() >= self.threshold
    }

    /// Drain all buffered reads, resetting the buffer to empty.
    pub fn drain(&mut self) -> Vec<DeferredRead> {
        self.approx_bytes = 0;
        std::mem::take(&mut self.reads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_deferred(header: &str, num_score: f64) -> DeferredRead {
        DeferredRead {
            header: header.to_string(),
            num_score,
            minimizers: (vec![100, 200], vec![300, 400]),
            global_index: 0,
        }
    }

    #[test]
    fn test_new_buffer_is_empty() {
        let buf = DeferredDenomBuffer::new(10);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        assert!(!buf.should_flush());
    }

    #[test]
    fn test_push_increments_len() {
        let mut buf = DeferredDenomBuffer::new(10);
        buf.push(make_deferred("read_0", 0.5));
        assert_eq!(buf.len(), 1);
        assert!(!buf.is_empty());

        buf.push(make_deferred("read_1", 0.3));
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn test_should_flush_at_threshold() {
        let mut buf = DeferredDenomBuffer::new(3);
        buf.push(make_deferred("r0", 0.1));
        buf.push(make_deferred("r1", 0.2));
        assert!(!buf.should_flush());

        buf.push(make_deferred("r2", 0.3));
        assert!(buf.should_flush());
    }

    #[test]
    fn test_should_flush_above_threshold() {
        let mut buf = DeferredDenomBuffer::new(2);
        buf.push(make_deferred("r0", 0.1));
        buf.push(make_deferred("r1", 0.2));
        buf.push(make_deferred("r2", 0.3));
        assert!(buf.should_flush());
    }

    #[test]
    fn test_drain_returns_all_and_resets() {
        let mut buf = DeferredDenomBuffer::new(10);
        buf.push(make_deferred("r0", 0.1));
        buf.push(make_deferred("r1", 0.2));
        buf.push(make_deferred("r2", 0.3));

        let drained = buf.drain();
        assert_eq!(drained.len(), 3);
        assert_eq!(drained[0].header, "r0");
        assert_eq!(drained[1].header, "r1");
        assert_eq!(drained[2].header, "r2");

        // Buffer is now empty
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        assert!(!buf.should_flush());
    }

    #[test]
    fn test_drain_on_empty_returns_empty_vec() {
        let mut buf = DeferredDenomBuffer::new(10);
        let drained = buf.drain();
        assert!(drained.is_empty());
    }

    #[test]
    fn test_threshold_of_one() {
        let mut buf = DeferredDenomBuffer::new(1);
        assert!(!buf.should_flush());

        buf.push(make_deferred("r0", 0.5));
        assert!(buf.should_flush());

        let drained = buf.drain();
        assert_eq!(drained.len(), 1);
        assert!(!buf.should_flush());
    }

    #[test]
    fn test_push_after_drain() {
        let mut buf = DeferredDenomBuffer::new(5);
        buf.push(make_deferred("r0", 0.1));
        buf.push(make_deferred("r1", 0.2));
        let _ = buf.drain();

        buf.push(make_deferred("r2", 0.3));
        assert_eq!(buf.len(), 1);
        assert_eq!(buf.drain()[0].header, "r2");
    }

    #[test]
    fn test_deferred_read_preserves_num_score() {
        let mut buf = DeferredDenomBuffer::new(10);
        buf.push(make_deferred("r0", 0.42));
        buf.push(make_deferred("r1", 0.99));

        let drained = buf.drain();
        assert!((drained[0].num_score - 0.42).abs() < 1e-10);
        assert!((drained[1].num_score - 0.99).abs() < 1e-10);
    }

    #[test]
    fn test_approx_bytes_tracking() {
        let mut buf = DeferredDenomBuffer::new(10);
        assert_eq!(buf.approx_bytes(), 0);

        // "r0": cap=2 header + 2*8 fwd + 2*8 rc + 72 overhead = 106
        buf.push(make_deferred("r0", 0.1));
        let per_entry = buf.approx_bytes();
        assert!(per_entry > 0);

        buf.push(make_deferred("r1", 0.2));
        assert_eq!(buf.approx_bytes(), per_entry * 2);

        buf.drain();
        assert_eq!(buf.approx_bytes(), 0);
    }

    #[test]
    fn test_approx_heap_bytes_varied_minimizers() {
        let dr = DeferredRead {
            header: "hdr".to_string(),
            num_score: 0.5,
            minimizers: (vec![1, 2, 3], vec![4, 5]),
            global_index: 42,
        };
        // "hdr" cap=3 + fwd=3*8=24 + rc=2*8=16 + 72 overhead = 115
        let bytes = dr.approx_heap_bytes();
        // Must include header + both minimizer vecs + overhead
        assert!(bytes >= 3 + 24 + 16 + 72, "got {}", bytes);
    }

    #[test]
    fn test_deferred_read_preserves_minimizers() {
        let mut buf = DeferredDenomBuffer::new(10);
        buf.push(DeferredRead {
            header: "read1".to_string(),
            num_score: 0.75,
            minimizers: (vec![10, 20, 30], vec![40, 50]),
            global_index: 99,
        });

        let drained = buf.drain();
        let dr = &drained[0];
        assert_eq!(dr.minimizers.0, vec![10, 20, 30]);
        assert_eq!(dr.minimizers.1, vec![40, 50]);
        assert_eq!(dr.global_index, 99);
    }
}
