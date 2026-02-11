//! Deferred denominator classification buffer for log-ratio mode.
//!
//! Accumulates reads that need denominator classification across batches,
//! flushing only when the buffer reaches a threshold. This avoids paying
//! the full shard I/O cost for every batch when most reads fast-path.
//!
//! Uses flat COO (coordinate) format: minimizers are stored as
//! `(minimizer, packed_read_id)` entries in a single Vec, with per-read
//! metadata stored separately. On drain, entries are sorted by minimizer
//! for direct use with `QueryInvertedIndex::from_sorted_coo()`.

use rype::QueryInvertedIndex;

/// Per-read metadata for deferred denominator classification.
pub struct DeferredMeta {
    pub header: String,
    pub num_score: f64,
    /// Global read index (position in the input file) for bitset tracking.
    pub global_index: usize,
    pub fwd_count: u32,
    pub rc_count: u32,
}

/// Buffer that accumulates deferred-denom reads as flat COO entries.
///
/// Fast-path results are emitted immediately per batch; only reads needing
/// the denominator are buffered here. When the buffer reaches `threshold`
/// reads, the caller should drain and classify them in one pass.
///
/// On `drain()`, entries are sorted by minimizer and returned alongside
/// metadata for direct construction of a `QueryInvertedIndex` via
/// `from_sorted_coo()`.
pub struct DeferredDenomBuffer {
    entries: Vec<(u64, u32)>,
    metadata: Vec<DeferredMeta>,
    threshold: usize,
}

impl DeferredDenomBuffer {
    /// Create a new buffer that triggers flush at `threshold` accumulated reads.
    pub fn new(threshold: usize) -> Self {
        Self {
            entries: Vec::new(),
            metadata: Vec::new(),
            threshold,
        }
    }

    /// Number of reads currently buffered.
    pub fn len(&self) -> usize {
        self.metadata.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.metadata.is_empty()
    }

    /// Approximate heap bytes used by buffered data.
    pub fn approx_bytes(&self) -> usize {
        let entry_bytes = self.entries.capacity() * std::mem::size_of::<(u64, u32)>();
        let meta_struct_bytes = self.metadata.capacity() * std::mem::size_of::<DeferredMeta>();
        let header_bytes: usize = self.metadata.iter().map(|m| m.header.capacity()).sum();
        entry_bytes + meta_struct_bytes + header_bytes
    }

    /// Add a deferred read to the buffer, flattening minimizers into COO entries.
    ///
    /// The minimizers are consumed (moved) and packed into flat `(minimizer, packed_read_id)`
    /// entries. The per-read metadata (header, score, counts) is stored separately.
    pub fn push(
        &mut self,
        header: String,
        num_score: f64,
        global_index: usize,
        fwd_mins: Vec<u64>,
        rc_mins: Vec<u64>,
    ) {
        let read_idx = self.metadata.len() as u32;
        let fwd_count = fwd_mins.len() as u32;
        let rc_count = rc_mins.len() as u32;

        for m in fwd_mins {
            self.entries
                .push((m, QueryInvertedIndex::pack_read_id(read_idx, false)));
        }
        for m in rc_mins {
            self.entries
                .push((m, QueryInvertedIndex::pack_read_id(read_idx, true)));
        }

        self.metadata.push(DeferredMeta {
            header,
            num_score,
            global_index,
            fwd_count,
            rc_count,
        });
    }

    /// Returns true when the buffer has reached or exceeded the flush threshold.
    pub fn should_flush(&self) -> bool {
        self.metadata.len() >= self.threshold
    }

    /// Drain all buffered data, sorting entries by minimizer.
    ///
    /// Preserves allocated capacity for the next fill cycle to avoid reallocation.
    ///
    /// Returns `(sorted_entries, metadata)` ready for
    /// `QueryInvertedIndex::from_sorted_coo()`.
    pub fn drain(&mut self) -> (Vec<(u64, u32)>, Vec<DeferredMeta>) {
        let entry_cap = self.entries.capacity();
        let meta_cap = self.metadata.capacity();
        let mut entries = std::mem::replace(&mut self.entries, Vec::with_capacity(entry_cap));
        let metadata = std::mem::replace(&mut self.metadata, Vec::with_capacity(meta_cap));
        entries.sort_unstable_by_key(|&(m, _)| m);
        (entries, metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_simple(buf: &mut DeferredDenomBuffer, header: &str, num_score: f64) {
        buf.push(
            header.to_string(),
            num_score,
            0,
            vec![100, 200],
            vec![300, 400],
        );
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
        push_simple(&mut buf, "read_0", 0.5);
        assert_eq!(buf.len(), 1);
        assert!(!buf.is_empty());

        push_simple(&mut buf, "read_1", 0.3);
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn test_should_flush_at_threshold() {
        let mut buf = DeferredDenomBuffer::new(3);
        push_simple(&mut buf, "r0", 0.1);
        push_simple(&mut buf, "r1", 0.2);
        assert!(!buf.should_flush());

        push_simple(&mut buf, "r2", 0.3);
        assert!(buf.should_flush());
    }

    #[test]
    fn test_should_flush_above_threshold() {
        let mut buf = DeferredDenomBuffer::new(2);
        push_simple(&mut buf, "r0", 0.1);
        push_simple(&mut buf, "r1", 0.2);
        push_simple(&mut buf, "r2", 0.3);
        assert!(buf.should_flush());
    }

    #[test]
    fn test_drain_returns_sorted_entries_and_metadata() {
        let mut buf = DeferredDenomBuffer::new(10);
        push_simple(&mut buf, "r0", 0.1);
        push_simple(&mut buf, "r1", 0.2);
        push_simple(&mut buf, "r2", 0.3);

        let (entries, metadata) = buf.drain();
        assert_eq!(metadata.len(), 3);
        assert_eq!(metadata[0].header, "r0");
        assert_eq!(metadata[1].header, "r1");
        assert_eq!(metadata[2].header, "r2");

        // Entries should be sorted by minimizer
        assert!(entries.windows(2).all(|w| w[0].0 <= w[1].0));

        // Buffer is now empty
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        assert!(!buf.should_flush());
    }

    #[test]
    fn test_drain_on_empty_returns_empty() {
        let mut buf = DeferredDenomBuffer::new(10);
        let (entries, metadata) = buf.drain();
        assert!(entries.is_empty());
        assert!(metadata.is_empty());
    }

    #[test]
    fn test_threshold_of_one() {
        let mut buf = DeferredDenomBuffer::new(1);
        assert!(!buf.should_flush());

        push_simple(&mut buf, "r0", 0.5);
        assert!(buf.should_flush());

        let (_, metadata) = buf.drain();
        assert_eq!(metadata.len(), 1);
        assert!(!buf.should_flush());
    }

    #[test]
    fn test_push_after_drain() {
        let mut buf = DeferredDenomBuffer::new(5);
        push_simple(&mut buf, "r0", 0.1);
        push_simple(&mut buf, "r1", 0.2);
        let _ = buf.drain();

        push_simple(&mut buf, "r2", 0.3);
        assert_eq!(buf.len(), 1);
        let (_, metadata) = buf.drain();
        assert_eq!(metadata[0].header, "r2");
    }

    #[test]
    fn test_metadata_preserves_num_score() {
        let mut buf = DeferredDenomBuffer::new(10);
        push_simple(&mut buf, "r0", 0.42);
        push_simple(&mut buf, "r1", 0.99);

        let (_, metadata) = buf.drain();
        assert!((metadata[0].num_score - 0.42).abs() < 1e-10);
        assert!((metadata[1].num_score - 0.99).abs() < 1e-10);
    }

    #[test]
    fn test_approx_bytes_tracking() {
        let mut buf = DeferredDenomBuffer::new(10);
        assert_eq!(buf.approx_bytes(), 0);

        push_simple(&mut buf, "r0", 0.1);
        let after_one = buf.approx_bytes();
        assert!(after_one > 0);

        push_simple(&mut buf, "r1", 0.2);
        let after_two = buf.approx_bytes();
        assert!(after_two > after_one);

        buf.drain();
        // After drain, capacity is preserved but no metadata items remain,
        // so header_bytes is 0 but entry/meta capacity bytes remain.
        assert!(buf.is_empty());
    }

    // === Flat COO-specific tests ===

    #[test]
    fn test_flat_coo_push_flattens_minimizers() {
        let mut buf = DeferredDenomBuffer::new(10);
        buf.push("r0".to_string(), 0.5, 0, vec![100, 200], vec![150]);

        // Should have 3 entries: 2 fwd + 1 rc
        let (entries, metadata) = buf.drain();
        assert_eq!(entries.len(), 3);
        assert_eq!(metadata[0].fwd_count, 2);
        assert_eq!(metadata[0].rc_count, 1);

        // Verify packed read_ids: all should be read_idx=0
        for &(_, packed) in &entries {
            let (read_idx, _) = QueryInvertedIndex::unpack_read_id(packed);
            assert_eq!(read_idx, 0);
        }

        // Verify we have both fwd and rc entries
        let fwd_count = entries
            .iter()
            .filter(|&&(_, p)| !QueryInvertedIndex::unpack_read_id(p).1)
            .count();
        let rc_count = entries
            .iter()
            .filter(|&&(_, p)| QueryInvertedIndex::unpack_read_id(p).1)
            .count();
        assert_eq!(fwd_count, 2);
        assert_eq!(rc_count, 1);
    }

    #[test]
    fn test_flat_coo_read_indices_increment() {
        let mut buf = DeferredDenomBuffer::new(10);
        buf.push("r0".to_string(), 0.5, 0, vec![100], vec![200]);
        buf.push("r1".to_string(), 0.3, 1, vec![300], vec![400]);

        let (entries, _) = buf.drain();
        assert_eq!(entries.len(), 4);

        // Collect read indices from all entries
        let read_indices: Vec<u32> = entries
            .iter()
            .map(|&(_, p)| QueryInvertedIndex::unpack_read_id(p).0)
            .collect();
        // First read's entries should have read_idx=0, second's should have read_idx=1
        assert!(read_indices.contains(&0));
        assert!(read_indices.contains(&1));
    }

    #[test]
    fn test_flat_coo_drain_returns_sorted_entries() {
        let mut buf = DeferredDenomBuffer::new(10);
        // Push reads with unsorted minimizers across reads
        buf.push("r0".to_string(), 0.5, 0, vec![500, 300], vec![700]);
        buf.push("r1".to_string(), 0.3, 1, vec![100], vec![200]);

        let (entries, _) = buf.drain();
        // Entries must be sorted by minimizer after drain
        assert!(
            entries.windows(2).all(|w| w[0].0 <= w[1].0),
            "entries not sorted: {:?}",
            entries.iter().map(|(m, _)| m).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_flat_coo_memory_tracking() {
        let mut buf = DeferredDenomBuffer::new(10);
        buf.push(
            "header".to_string(),
            0.5,
            0,
            vec![100, 200, 300],
            vec![400, 500],
        );

        let bytes = buf.approx_bytes();
        let entry_size = std::mem::size_of::<(u64, u32)>();
        // At minimum, should account for 5 entries
        assert!(
            bytes >= 5 * entry_size,
            "approx_bytes {} too small for 5 entries at {} bytes each",
            bytes,
            entry_size
        );
    }

    #[test]
    fn test_flat_coo_preserves_global_index() {
        let mut buf = DeferredDenomBuffer::new(10);
        buf.push("r0".to_string(), 0.75, 99, vec![10, 20, 30], vec![40, 50]);

        let (_, metadata) = buf.drain();
        assert_eq!(metadata[0].global_index, 99);
        assert_eq!(metadata[0].fwd_count, 3);
        assert_eq!(metadata[0].rc_count, 2);
    }
}
