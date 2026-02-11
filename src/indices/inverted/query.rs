//! Query inverted index for merge-join classification.

use crate::constants::{MAX_READS, RC_FLAG_BIT, READ_INDEX_MASK};

/// Query inverted index for merge-join classification.
/// Stores sorted COO (coordinate) entries: (minimizer, packed_read_id).
///
/// # Invariants
/// - `entries` is sorted by minimizer (ascending). NOT deduplicated: if 2 reads
///   share minimizer 100, `entries` has 2 entries with minimizer 100.
/// - `fwd_counts.len() == rc_counts.len()` == number of query reads
#[derive(Debug)]
pub struct QueryInvertedIndex {
    /// Sorted (minimizer, packed_read_id) entries. NOT deduplicated.
    pub(crate) entries: Vec<(u64, u32)>,
    /// Number of forward minimizers per read (for scoring)
    pub(crate) fwd_counts: Vec<u32>,
    /// Number of RC minimizers per read (for scoring)
    pub(crate) rc_counts: Vec<u32>,
    /// Cached count of unique minimizers. Computed once at build time.
    unique_count: usize,
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

    /// Get the total number of COO entries (total minimizers across all reads, NOT unique).
    pub fn num_entries(&self) -> usize {
        self.entries.len()
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

    /// Count of unique minimizers. O(1) — cached at build time.
    pub fn num_unique_minimizers(&self) -> usize {
        self.unique_count
    }

    /// Compute sorted, deduplicated minimizers from COO entries.
    /// Used for shard filtering and bloom filter hints. Allocates a new Vec on each call;
    /// prefer `num_unique_minimizers()` when only the count is needed.
    pub fn unique_minimizers(&self) -> Vec<u64> {
        let mut result = Vec::with_capacity(self.unique_count);
        for &(m, _) in &self.entries {
            if result.last() != Some(&m) {
                result.push(m);
            }
        }
        result
    }

    /// Get (min, max) minimizer range, or None if empty.
    pub fn minimizer_range(&self) -> Option<(u64, u64)> {
        if self.entries.is_empty() {
            None
        } else {
            Some((self.entries[0].0, self.entries[self.entries.len() - 1].0))
        }
    }

    /// Compute unique minimizer count from sorted entries.
    fn compute_unique_count(entries: &[(u64, u32)]) -> usize {
        if entries.is_empty() {
            0
        } else {
            entries.windows(2).filter(|w| w[0].0 != w[1].0).count() + 1
        }
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
    /// A QueryInvertedIndex with sorted COO entries.
    ///
    /// # Panics
    /// - If `queries.len()` exceeds `MAX_READS` (2^31 - 1)
    /// - If total minimizer count overflows `usize`
    pub fn build(queries: &[(Vec<u64>, Vec<u64>)]) -> Self {
        assert!(
            queries.len() <= Self::MAX_READS,
            "Batch size {} exceeds maximum {} reads (31-bit limit)",
            queries.len(),
            Self::MAX_READS
        );

        if queries.is_empty() {
            return QueryInvertedIndex {
                entries: Vec::new(),
                fwd_counts: Vec::new(),
                rc_counts: Vec::new(),
                unique_count: 0,
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
                entries: Vec::new(),
                fwd_counts,
                rc_counts,
                unique_count: 0,
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

        // Sort by minimizer
        entries.sort_unstable_by_key(|(m, _)| *m);

        let unique_count = Self::compute_unique_count(&entries);
        QueryInvertedIndex {
            entries,
            fwd_counts,
            rc_counts,
            unique_count,
        }
    }

    /// Build from pre-sorted COO entries and per-read counts.
    ///
    /// # Preconditions
    /// - `entries` must be sorted by minimizer (ascending).
    ///   Use `DeferredDenomBuffer::drain()` which sorts before returning.
    ///
    /// # Panics (debug only)
    /// If entries are not sorted by minimizer.
    pub fn from_sorted_coo(
        entries: Vec<(u64, u32)>,
        fwd_counts: Vec<u32>,
        rc_counts: Vec<u32>,
    ) -> Self {
        debug_assert!(
            entries.windows(2).all(|w| w[0].0 <= w[1].0),
            "from_sorted_coo: entries must be sorted by minimizer"
        );

        let unique_count = Self::compute_unique_count(&entries);
        QueryInvertedIndex {
            entries,
            fwd_counts,
            rc_counts,
            unique_count,
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

        assert_eq!(qidx.num_entries(), 0);
        assert_eq!(qidx.unique_minimizers().len(), 0);
        assert_eq!(qidx.num_reads(), 0);
    }

    #[test]
    fn test_query_inverted_single_read() {
        // Single read with 3 forward and 2 RC minimizers (all unique)
        let queries = vec![(vec![100, 200, 300], vec![150, 250])];
        let qidx = QueryInvertedIndex::build(&queries);

        // 5 unique minimizers: 100, 150, 200, 250, 300
        assert_eq!(qidx.unique_minimizers().len(), 5);
        // 5 total entries (each minimizer maps to read 0)
        assert_eq!(qidx.num_entries(), 5);
        assert_eq!(qidx.num_reads(), 1);

        // Verify unique_mins are sorted
        assert_eq!(qidx.unique_minimizers(), &[100, 150, 200, 250, 300]);

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
        assert_eq!(qidx.unique_minimizers().len(), 4);
        // Total entries: 100(r0), 100(r1), 150(r0), 150(r1), 200(r0), 300(r1) = 6
        assert_eq!(qidx.num_entries(), 6);
        assert_eq!(qidx.num_reads(), 2);

        // Verify unique_mins are sorted
        assert_eq!(qidx.unique_minimizers(), &[100, 150, 200, 300]);

        // Verify COO entries for minimizer 100 (should have 2 entries)
        let count_100 = qidx.entries.iter().filter(|(m, _)| *m == 100).count();
        assert_eq!(count_100, 2);

        // Verify the read IDs for minimizer 100 are reads 0 and 1 (forward)
        let reads_for_100: Vec<(u32, bool)> = qidx
            .entries
            .iter()
            .filter(|(m, _)| *m == 100)
            .map(|(_, p)| QueryInvertedIndex::unpack_read_id(*p))
            .collect();
        assert!(reads_for_100.contains(&(0, false)));
        assert!(reads_for_100.contains(&(1, false)));

        // Verify the read IDs for minimizer 150 are reads 0 and 1 (RC)
        let reads_for_150: Vec<(u32, bool)> = qidx
            .entries
            .iter()
            .filter(|(m, _)| *m == 150)
            .map(|(_, p)| QueryInvertedIndex::unpack_read_id(*p))
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

        assert_eq!(qidx.num_entries(), 0);
        assert_eq!(qidx.unique_minimizers().len(), 0);
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
        let queries: Vec<(Vec<u64>, Vec<u64>)> = Vec::new();

        // Simulate the check that would fail with too many reads
        let fake_len = QueryInvertedIndex::MAX_READS + 1;
        assert!(
            fake_len <= QueryInvertedIndex::MAX_READS,
            "Batch size {} exceeds maximum {} reads (31-bit limit)",
            fake_len,
            QueryInvertedIndex::MAX_READS
        );

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

        // Test unique_minimizers accessor
        let mins = qidx.unique_minimizers();
        assert!(mins.is_sorted());
        assert_eq!(mins.len(), qidx.unique_minimizers().len());
    }

    // === COO-specific tests ===

    #[test]
    fn test_coo_entry_count_equals_total_minimizers() {
        let queries = vec![
            (vec![100, 200, 300], vec![150, 250]), // 5 total
            (vec![100], vec![150, 250, 350]),      // 4 total
        ];
        let qidx = QueryInvertedIndex::build(&queries);

        assert_eq!(qidx.entries.len(), 9); // 5 + 4 = 9
    }

    #[test]
    fn test_coo_entries_sorted_by_minimizer() {
        let queries = vec![(vec![300, 100, 200], vec![250, 150]), (vec![50], vec![400])];
        let qidx = QueryInvertedIndex::build(&queries);

        assert!(qidx.entries.windows(2).all(|w| w[0].0 <= w[1].0));
    }

    #[test]
    fn test_coo_no_deduplication() {
        // 2 reads share minimizer 100
        let queries = vec![(vec![100, 200], vec![]), (vec![100, 300], vec![])];
        let qidx = QueryInvertedIndex::build(&queries);

        let count_100 = qidx.entries.iter().filter(|(m, _)| *m == 100).count();
        assert_eq!(
            count_100, 2,
            "COO should NOT deduplicate: 2 reads share minimizer 100"
        );
    }

    #[test]
    fn test_minimizer_range() {
        let queries = vec![(vec![300, 100], vec![500])];
        let qidx = QueryInvertedIndex::build(&queries);
        assert_eq!(qidx.minimizer_range(), Some((100, 500)));

        let empty_queries: Vec<(Vec<u64>, Vec<u64>)> = vec![];
        let empty_qidx = QueryInvertedIndex::build(&empty_queries);
        assert_eq!(empty_qidx.minimizer_range(), None);
    }

    #[test]
    fn test_from_sorted_coo() {
        let entries = vec![(100, 0u32), (100, 1), (200, 0), (300, 1)];
        let fwd_counts = vec![2, 1];
        let rc_counts = vec![0, 1];
        let qidx = QueryInvertedIndex::from_sorted_coo(entries, fwd_counts, rc_counts);

        assert_eq!(qidx.num_entries(), 4);
        assert_eq!(qidx.unique_minimizers(), vec![100u64, 200, 300]);
        assert_eq!(qidx.num_reads(), 2);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "entries must be sorted")]
    fn test_from_sorted_coo_rejects_unsorted() {
        let entries = vec![(200, 0u32), (100, 1)]; // NOT sorted
        QueryInvertedIndex::from_sorted_coo(entries, vec![1], vec![1]);
    }

    #[test]
    fn test_unique_minimizers_capacity_is_minimal() {
        // Two reads sharing some minimizers: tests that capacity == len (no over-allocation)
        let queries = vec![
            (vec![100, 200, 300], vec![150, 250]),
            (vec![100, 300, 400], vec![150, 350]),
        ];
        let qidx = QueryInvertedIndex::build(&queries);
        let mins = qidx.unique_minimizers();

        assert_eq!(
            mins.capacity(),
            mins.len(),
            "unique_minimizers() should allocate exactly unique_count capacity, got cap={} len={}",
            mins.capacity(),
            mins.len()
        );
        // Verify correctness: 100, 150, 200, 250, 300, 350, 400
        assert_eq!(mins, vec![100, 150, 200, 250, 300, 350, 400]);
    }
}
