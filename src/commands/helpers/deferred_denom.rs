//! Deferred denominator classification buffer for log-ratio mode.
//!
//! Accumulates reads that need denominator classification across batches,
//! flushing only when the buffer reaches a threshold. This avoids paying
//! the full shard I/O cost for every batch when most reads fast-path.

use super::fastx_io::OwnedFastxRecord;

/// A read deferred for later denominator classification.
///
/// Stores the header, numerator score, and owned sequence data needed to
/// re-classify the read against the denominator index at flush time.
pub struct DeferredRead {
    pub header: String,
    pub num_score: f64,
    pub record: OwnedFastxRecord,
}

impl DeferredRead {
    /// Approximate heap bytes owned by this deferred read.
    pub fn approx_heap_bytes(&self) -> usize {
        self.header.len()
            + self.record.seq1.len()
            + self.record.qual1.as_ref().map_or(0, |v| v.len())
            + self.record.seq2.as_ref().map_or(0, |v| v.len())
            + self.record.qual2.as_ref().map_or(0, |v| v.len())
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
            record: OwnedFastxRecord::new(0, b"ACGT".to_vec(), None, None, None),
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

        // "r0" = 2 bytes header, "ACGT" = 4 bytes seq1, no qual/seq2
        buf.push(make_deferred("r0", 0.1));
        assert_eq!(buf.approx_bytes(), 2 + 4);

        buf.push(make_deferred("r1", 0.2));
        assert_eq!(buf.approx_bytes(), (2 + 4) * 2);

        buf.drain();
        assert_eq!(buf.approx_bytes(), 0);
    }

    #[test]
    fn test_approx_heap_bytes_with_qual_and_paired() {
        let record = OwnedFastxRecord::new(
            0,
            b"GATTACA".to_vec(),
            Some(b"IIIIIII".to_vec()),
            Some(b"TTTT".to_vec()),
            Some(b"JJJJ".to_vec()),
        );
        let dr = DeferredRead {
            header: "hdr".to_string(),
            num_score: 0.5,
            record,
        };
        // "hdr"=3 + "GATTACA"=7 + "IIIIIII"=7 + "TTTT"=4 + "JJJJ"=4 = 25
        assert_eq!(dr.approx_heap_bytes(), 25);
    }

    #[test]
    fn test_deferred_read_preserves_record() {
        let record = OwnedFastxRecord::new(
            5,
            b"GATTACA".to_vec(),
            Some(b"IIIIIII".to_vec()),
            Some(b"TTTT".to_vec()),
            Some(b"JJJJ".to_vec()),
        );
        let mut buf = DeferredDenomBuffer::new(10);
        buf.push(DeferredRead {
            header: "paired_read".to_string(),
            num_score: 0.75,
            record,
        });

        let drained = buf.drain();
        let rec = &drained[0].record;
        assert_eq!(rec.seq1, b"GATTACA");
        assert_eq!(rec.qual1.as_deref(), Some(b"IIIIIII".as_slice()));
        assert_eq!(rec.seq2.as_deref(), Some(b"TTTT".as_slice()));
        assert_eq!(rec.qual2.as_deref(), Some(b"JJJJ".as_slice()));
    }
}
