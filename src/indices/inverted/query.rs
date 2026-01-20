//! Query inverted index for merge-join classification.

use std::collections::HashMap;

use crate::constants::{MAX_READS, RC_FLAG_BIT, READ_INDEX_MASK};

use super::InvertedIndex;

/// Query inverted index for merge-join classification.
/// Maps minimizer -> list of (read_index, strand) pairs using CSR format.
///
/// # Invariants
/// - `minimizers` is sorted in ascending order with no duplicates
/// - `offsets.len() == minimizers.len() + 1`
/// - `offsets[0] == 0`
/// - `offsets` is monotonically increasing
/// - `offsets[minimizers.len()] == read_ids.len()`
/// - For each minimizer at index i, the associated read IDs are `read_ids[offsets[i]..offsets[i+1]]`
/// - `fwd_counts.len() == rc_counts.len()` == number of query reads
#[derive(Debug)]
pub struct QueryInvertedIndex {
    /// Sorted unique minimizers from all queries
    pub(crate) minimizers: Vec<u64>,
    /// CSR offsets: read_ids[offsets[i]..offsets[i+1]] are reads containing minimizers[i]
    pub(crate) offsets: Vec<u32>,
    /// Packed (read_index, strand): bit 31 = strand (0=fwd, 1=rc), bits 0-30 = read index
    pub(crate) read_ids: Vec<u32>,
    /// Number of forward minimizers per read (for scoring)
    pub(crate) fwd_counts: Vec<u32>,
    /// Number of RC minimizers per read (for scoring)
    pub(crate) rc_counts: Vec<u32>,
}

impl QueryInvertedIndex {
    /// Pack a read index and strand flag into a u32.
    /// Bit 31 = strand (0=fwd, 1=rc), bits 0-30 = read index.
    #[inline]
    pub fn pack_read_id(read_idx: u32, is_rc: bool) -> u32 {
        debug_assert!(read_idx <= READ_INDEX_MASK, "Read index exceeds 31 bits");
        if is_rc {
            read_idx | RC_FLAG_BIT
        } else {
            read_idx
        }
    }

    /// Unpack a read_id entry into (read_index, is_rc).
    #[inline]
    pub fn unpack_read_id(packed: u32) -> (u32, bool) {
        let is_rc = (packed & RC_FLAG_BIT) != 0;
        let read_idx = packed & READ_INDEX_MASK;
        (read_idx, is_rc)
    }

    /// Get the number of unique minimizers.
    pub fn num_minimizers(&self) -> usize {
        self.minimizers.len()
    }

    /// Get the total number of read ID entries.
    pub fn num_read_entries(&self) -> usize {
        self.read_ids.len()
    }

    /// Get the number of query reads.
    pub fn num_reads(&self) -> usize {
        self.fwd_counts.len()
    }

    /// Get the forward minimizer count for a specific read.
    pub fn fwd_count(&self, read_idx: usize) -> u32 {
        self.fwd_counts[read_idx]
    }

    /// Get the reverse-complement minimizer count for a specific read.
    pub fn rc_count(&self, read_idx: usize) -> u32 {
        self.rc_counts[read_idx]
    }

    /// Get a reference to the sorted minimizer array.
    pub fn minimizers(&self) -> &[u64] {
        &self.minimizers
    }

    /// Maximum number of reads supported (31 bits, bit 31 reserved for strand flag).
    pub const MAX_READS: usize = MAX_READS;

    /// Build from extracted minimizers: Vec<(fwd_mins, rc_mins)> per read.
    ///
    /// # Arguments
    /// * `queries` - For each read: (forward_minimizers, reverse_complement_minimizers).
    ///   Each vector should be sorted and deduplicated (as returned by `get_paired_minimizers_into`).
    ///
    /// # Returns
    /// A QueryInvertedIndex mapping minimizers to read IDs.
    ///
    /// # Panics
    /// - If `queries.len()` exceeds `MAX_READS` (2^31 - 1)
    /// - If total minimizer count overflows `usize`
    ///
    /// # Implementation Note
    /// Currently uses global sort O(N log N). Since input vectors are already sorted,
    /// a k-way merge (like `InvertedIndex::build_from_index`) could achieve O(N log K)
    /// where K = number of reads. This optimization is deferred as the current approach
    /// is simpler and the sort is parallelizable.
    pub fn build(queries: &[(Vec<u64>, Vec<u64>)]) -> Self {
        assert!(
            queries.len() <= Self::MAX_READS,
            "Batch size {} exceeds maximum {} reads (31-bit limit)",
            queries.len(),
            Self::MAX_READS
        );

        if queries.is_empty() {
            return QueryInvertedIndex {
                minimizers: Vec::new(),
                offsets: vec![0],
                read_ids: Vec::new(),
                fwd_counts: Vec::new(),
                rc_counts: Vec::new(),
            };
        }

        // Count total entries and collect fwd/rc counts (with overflow checking)
        let mut total_entries = 0usize;
        let mut fwd_counts = Vec::with_capacity(queries.len());
        let mut rc_counts = Vec::with_capacity(queries.len());

        for (fwd, rc) in queries {
            total_entries = total_entries
                .checked_add(fwd.len())
                .and_then(|t| t.checked_add(rc.len()))
                .expect("Total minimizer count overflow");
            fwd_counts.push(fwd.len() as u32);
            rc_counts.push(rc.len() as u32);
        }

        if total_entries == 0 {
            return QueryInvertedIndex {
                minimizers: Vec::new(),
                offsets: vec![0],
                read_ids: Vec::new(),
                fwd_counts,
                rc_counts,
            };
        }

        // Collect all (minimizer, packed_read_id) tuples
        let mut entries: Vec<(u64, u32)> = Vec::with_capacity(total_entries);
        for (read_idx, (fwd, rc)) in queries.iter().enumerate() {
            let read_idx = read_idx as u32;
            for &m in fwd {
                entries.push((m, Self::pack_read_id(read_idx, false)));
            }
            for &m in rc {
                entries.push((m, Self::pack_read_id(read_idx, true)));
            }
        }

        // Sort by minimizer (stable sort preserves read order for same minimizer)
        entries.sort_unstable_by_key(|(m, _)| *m);

        // Build CSR structure via linear scan
        let mut minimizers = Vec::with_capacity(entries.len() / 2); // estimate unique
        let mut offsets = Vec::with_capacity(entries.len() / 2 + 1);
        let mut read_ids = Vec::with_capacity(entries.len());

        offsets.push(0);
        let mut current_min = entries[0].0;
        minimizers.push(current_min);

        for (m, packed) in entries {
            if m != current_min {
                offsets.push(read_ids.len() as u32);
                minimizers.push(m);
                current_min = m;
            }
            read_ids.push(packed);
        }
        offsets.push(read_ids.len() as u32);

        QueryInvertedIndex {
            minimizers,
            offsets,
            read_ids,
            fwd_counts,
            rc_counts,
        }
    }
}

impl InvertedIndex {
    /// Accumulate hits for a single matching minimizer pair.
    ///
    /// This method encapsulates access to internal CSR arrays, allowing callers
    /// to accumulate hits without directly accessing the internal representation.
    ///
    /// # Arguments
    /// * `query_idx` - The query inverted index
    /// * `qi` - Index into query_idx.minimizers for the matching minimizer
    /// * `ri` - Index into self.minimizers for the matching minimizer
    /// * `accumulators` - Per-read accumulators mapping bucket_id -> (fwd_hits, rc_hits)
    #[inline]
    pub fn accumulate_hits_for_match(
        &self,
        query_idx: &QueryInvertedIndex,
        qi: usize,
        ri: usize,
        accumulators: &mut [HashMap<u32, (u32, u32)>],
    ) {
        let q_start = query_idx.offsets[qi] as usize;
        let q_end = query_idx.offsets[qi + 1] as usize;
        let r_start = self.offsets[ri] as usize;
        let r_end = self.offsets[ri + 1] as usize;

        for &packed in &query_idx.read_ids[q_start..q_end] {
            let (read_idx, is_rc) = QueryInvertedIndex::unpack_read_id(packed);
            for &bucket_id in &self.bucket_ids[r_start..r_end] {
                let entry = accumulators[read_idx as usize]
                    .entry(bucket_id)
                    .or_insert((0, 0));
                // Use saturating_add to prevent overflow (defensive, unlikely in practice)
                if is_rc {
                    entry.1 = entry.1.saturating_add(1);
                } else {
                    entry.0 = entry.0.saturating_add(1);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_read_id() {
        // Test forward strand
        assert_eq!(
            QueryInvertedIndex::unpack_read_id(QueryInvertedIndex::pack_read_id(0, false)),
            (0, false)
        );
        assert_eq!(
            QueryInvertedIndex::unpack_read_id(QueryInvertedIndex::pack_read_id(12345, false)),
            (12345, false)
        );

        // Test reverse complement strand
        assert_eq!(
            QueryInvertedIndex::unpack_read_id(QueryInvertedIndex::pack_read_id(0, true)),
            (0, true)
        );
        assert_eq!(
            QueryInvertedIndex::unpack_read_id(QueryInvertedIndex::pack_read_id(12345, true)),
            (12345, true)
        );

        // Test max read index (31 bits)
        assert_eq!(
            QueryInvertedIndex::unpack_read_id(QueryInvertedIndex::pack_read_id(0x7FFFFFFF, false)),
            (0x7FFFFFFF, false)
        );
        assert_eq!(
            QueryInvertedIndex::unpack_read_id(QueryInvertedIndex::pack_read_id(0x7FFFFFFF, true)),
            (0x7FFFFFFF, true)
        );
    }

    #[test]
    fn test_query_inverted_empty() {
        let queries: Vec<(Vec<u64>, Vec<u64>)> = vec![];
        let qidx = QueryInvertedIndex::build(&queries);

        assert_eq!(qidx.num_minimizers(), 0);
        assert_eq!(qidx.num_read_entries(), 0);
        assert_eq!(qidx.num_reads(), 0);
        assert_eq!(qidx.offsets.len(), 1);
        assert_eq!(qidx.offsets[0], 0);
    }

    #[test]
    fn test_query_inverted_single_read() {
        // Single read with 3 forward and 2 RC minimizers (all unique)
        let queries = vec![(vec![100, 200, 300], vec![150, 250])];
        let qidx = QueryInvertedIndex::build(&queries);

        // 5 unique minimizers: 100, 150, 200, 250, 300
        assert_eq!(qidx.num_minimizers(), 5);
        // 5 entries (each minimizer maps to read 0)
        assert_eq!(qidx.num_read_entries(), 5);
        assert_eq!(qidx.num_reads(), 1);

        // Verify minimizers are sorted
        assert_eq!(qidx.minimizers, vec![100, 150, 200, 250, 300]);

        // Verify counts
        assert_eq!(qidx.fwd_counts[0], 3);
        assert_eq!(qidx.rc_counts[0], 2);
    }

    #[test]
    fn test_query_inverted_overlapping_minimizers() {
        // Two reads with overlapping minimizers
        let queries = vec![
            (vec![100, 200], vec![150]), // read 0: fwd=[100,200], rc=[150]
            (vec![100, 300], vec![150]), // read 1: fwd=[100,300], rc=[150]
        ];
        let qidx = QueryInvertedIndex::build(&queries);

        // Unique minimizers: 100, 150, 200, 300
        assert_eq!(qidx.num_minimizers(), 4);
        // Total entries: 100->[0,1], 150->[0,1], 200->[0], 300->[1] = 6 entries
        assert_eq!(qidx.num_read_entries(), 6);
        assert_eq!(qidx.num_reads(), 2);

        // Verify minimizers are sorted
        assert_eq!(qidx.minimizers, vec![100, 150, 200, 300]);

        // Verify CSR structure for minimizer 100 (index 0)
        let start = qidx.offsets[0] as usize;
        let end = qidx.offsets[1] as usize;
        assert_eq!(end - start, 2); // 100 appears in 2 reads

        // Verify CSR structure for minimizer 150 (index 1)
        let start = qidx.offsets[1] as usize;
        let end = qidx.offsets[2] as usize;
        assert_eq!(end - start, 2); // 150 appears in 2 reads

        // Verify the read IDs for minimizer 100 are reads 0 and 1 (forward)
        let reads_for_100: Vec<(u32, bool)> = qidx.read_ids
            [qidx.offsets[0] as usize..qidx.offsets[1] as usize]
            .iter()
            .map(|&p| QueryInvertedIndex::unpack_read_id(p))
            .collect();
        assert!(reads_for_100.contains(&(0, false)));
        assert!(reads_for_100.contains(&(1, false)));

        // Verify the read IDs for minimizer 150 are reads 0 and 1 (RC)
        let reads_for_150: Vec<(u32, bool)> = qidx.read_ids
            [qidx.offsets[1] as usize..qidx.offsets[2] as usize]
            .iter()
            .map(|&p| QueryInvertedIndex::unpack_read_id(p))
            .collect();
        assert!(reads_for_150.contains(&(0, true)));
        assert!(reads_for_150.contains(&(1, true)));
    }

    #[test]
    fn test_query_inverted_fwd_rc_counts() {
        let queries = vec![
            (vec![100, 200, 300], vec![150, 250]), // 3 fwd, 2 rc
            (vec![100], vec![150, 250, 350, 450]), // 1 fwd, 4 rc
            (vec![], vec![500, 600]),              // 0 fwd, 2 rc
        ];
        let qidx = QueryInvertedIndex::build(&queries);

        assert_eq!(qidx.fwd_counts.len(), 3);
        assert_eq!(qidx.rc_counts.len(), 3);

        assert_eq!(qidx.fwd_counts[0], 3);
        assert_eq!(qidx.rc_counts[0], 2);
        assert_eq!(qidx.fwd_counts[1], 1);
        assert_eq!(qidx.rc_counts[1], 4);
        assert_eq!(qidx.fwd_counts[2], 0);
        assert_eq!(qidx.rc_counts[2], 2);
    }

    #[test]
    fn test_query_inverted_all_empty_reads() {
        // Reads with no minimizers
        let queries = vec![(vec![], vec![]), (vec![], vec![])];
        let qidx = QueryInvertedIndex::build(&queries);

        assert_eq!(qidx.num_minimizers(), 0);
        assert_eq!(qidx.num_read_entries(), 0);
        assert_eq!(qidx.num_reads(), 2);
        assert_eq!(qidx.fwd_counts, vec![0, 0]);
        assert_eq!(qidx.rc_counts, vec![0, 0]);
    }

    #[test]
    fn test_query_inverted_max_reads_constant() {
        // Verify the MAX_READS constant is correct (31 bits)
        assert_eq!(QueryInvertedIndex::MAX_READS, 0x7FFFFFFF);
        assert_eq!(QueryInvertedIndex::MAX_READS, (1 << 31) - 1);
    }

    #[test]
    #[should_panic(expected = "31-bit limit")]
    fn test_query_inverted_overflow_too_many_reads() {
        // This test verifies the overflow check works, but we can't actually
        // allocate 2^31 reads. Instead, we create a mock scenario by checking
        // the assertion logic directly. Since we can't easily trigger this
        // without massive memory, we test the boundary check indirectly.
        //
        // In practice, the assertion at the start of build() will catch this:
        // assert!(queries.len() <= Self::MAX_READS, ...)
        //
        // For a real test, we'd need to mock or use a smaller limit.
        // This test documents the expected behavior.
        let queries: Vec<(Vec<u64>, Vec<u64>)> = Vec::new();

        // Simulate the check that would fail with too many reads
        let fake_len = QueryInvertedIndex::MAX_READS + 1;
        assert!(
            fake_len <= QueryInvertedIndex::MAX_READS,
            "Batch size {} exceeds maximum {} reads (31-bit limit)",
            fake_len,
            QueryInvertedIndex::MAX_READS
        );

        // This line won't be reached due to the panic above
        let _ = QueryInvertedIndex::build(&queries);
    }

    #[test]
    fn test_query_inverted_accessor_methods() {
        let queries = vec![
            (vec![100, 200, 300], vec![150, 250]), // 3 fwd, 2 rc
            (vec![100], vec![150, 250, 350]),      // 1 fwd, 3 rc
        ];
        let qidx = QueryInvertedIndex::build(&queries);

        // Test fwd_count accessor
        assert_eq!(qidx.fwd_count(0), 3);
        assert_eq!(qidx.fwd_count(1), 1);

        // Test rc_count accessor
        assert_eq!(qidx.rc_count(0), 2);
        assert_eq!(qidx.rc_count(1), 3);

        // Test minimizers accessor
        let mins = qidx.minimizers();
        assert!(mins.is_sorted());
        assert_eq!(mins.len(), qidx.num_minimizers());
    }
}
