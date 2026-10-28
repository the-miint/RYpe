//! Cross-batch query accumulation for single-pass classification.
//!
//! `run_log_ratio`'s `DeferredDenomBuffer` (`src/commands/helpers/deferred_denom.rs`)
//! established the pattern this module generalizes: accumulate reads as flat COO
//! `(minimizer, packed_read_id)` entries across many input batches, then drain into
//! one sorted `QueryInvertedIndex` for a single index scan. The reason to flatten
//! immediately rather than hold `Vec<(Vec<u64>, Vec<u64>)>` per read (as
//! `QueryInvertedIndex::build` does) is residency: holding both the raw per-read
//! minimizer vectors AND the flattened, sorted entries at once roughly doubles peak
//! memory for no benefit. `QueryAccumulator::push` drops each read's vectors the
//! moment they are flattened, so only one representation is ever resident.
//!
//! Generic over `M`, the per-read metadata payload, so both callers can share this
//! type: the CLI carries `String` headers (for `format_classification_results`,
//! which indexes output rows by position), and the Arrow/C-API path carries `i64`
//! query ids.

use super::query::QueryInvertedIndex;

/// Approximate heap-allocation footprint of a `QueryAccumulator` per-read
/// metadata payload, so `QueryAccumulator::approx_bytes()` can account for
/// it directly instead of leaving it to each caller (as
/// `DeferredDenomBuffer::approx_bytes` does for its `header: String` field —
/// the pattern this generalizes). Implemented for the payload types
/// `QueryAccumulator` is actually instantiated with: the CLI's owned
/// `String` headers (real heap cost, and often the *largest* per-read term —
/// e.g. short reads against a wide-window index can have more header bytes
/// than minimizer entry bytes) and the Arrow/C-API path's `i64` query ids
/// (no heap allocation).
pub trait MetaHeapBytes {
    fn heap_bytes(&self) -> usize;
}

impl MetaHeapBytes for String {
    fn heap_bytes(&self) -> usize {
        self.capacity()
    }
}

impl MetaHeapBytes for i64 {
    fn heap_bytes(&self) -> usize {
        0
    }
}

// Borrowed &str isn't a real payload type (nothing in the codebase
// accumulates borrowed metadata across flushes, since drain()'d metadata
// must outlive its originating input batch) but is used as a lightweight
// stand-in in this module's own unit tests below.
impl MetaHeapBytes for &str {
    fn heap_bytes(&self) -> usize {
        0
    }
}

/// Accumulates extracted minimizers across many input batches as flat COO entries,
/// flushing when a caller-supplied byte budget (not a read count) is reached.
///
/// # Invariants
/// - `entries` is unsorted while accumulating; `drain()` sorts it.
/// - `fwd_counts.len() == rc_counts.len() == meta.len()` == reads pushed since the
///   last `drain()`.
pub struct QueryAccumulator<M> {
    entries: Vec<(u64, u32)>,
    fwd_counts: Vec<u32>,
    rc_counts: Vec<u32>,
    meta: Vec<M>,
    /// Running total of `meta`'s heap allocations (see `MetaHeapBytes`),
    /// updated incrementally in `push` rather than rescanned on every
    /// `approx_bytes()` call — this is checked after every input batch via
    /// `should_flush()`, so an O(n) rescan would make accumulation overall
    /// O(n^2) in the number of batches.
    meta_heap_bytes: usize,
    byte_budget: usize,
    max_reads: Option<usize>,
    accumulator_bytes_per_read: usize,
}

impl<M: MetaHeapBytes> QueryAccumulator<M> {
    /// Create an accumulator that signals `should_flush()` once accumulated data
    /// reaches approximately `byte_budget` bytes.
    pub fn with_budget(byte_budget: usize) -> Self {
        Self {
            entries: Vec::new(),
            fwd_counts: Vec::new(),
            rc_counts: Vec::new(),
            meta: Vec::new(),
            meta_heap_bytes: 0,
            byte_budget,
            max_reads: None,
            accumulator_bytes_per_read: 0,
        }
    }

    /// Additionally flush once `max_reads` reads have been pushed, regardless of
    /// byte budget. Used to preserve exact read-count semantics when a caller
    /// (e.g. `--batch-size`) asks for a specific number of reads per pass rather
    /// than a memory-driven one.
    pub fn with_max_reads(mut self, max_reads: Option<usize>) -> Self {
        self.max_reads = max_reads;
        self
    }

    /// Reserve `bytes_per_read` of the byte budget per accumulated read for the
    /// `HitAccumulator` a classification pass will allocate over this group —
    /// see `memory::estimate_accumulator_bytes_per_read`. Entries and counts are
    /// bounded by minimizer count, but the accumulator scales with *read* count
    /// (up to ~1.3 KB/read on a 160-bucket dense index), so a pass with few but
    /// long reads can be accumulator-bound rather than entry-bound. Without this,
    /// `should_flush()` only sees entry/count bytes and a large pass can exceed
    /// its budget entirely in accumulator allocation once classification starts.
    /// Default 0 (no accounting) — the byte budget is entry/count bytes only.
    pub fn with_accumulator_cost_per_read(mut self, bytes_per_read: usize) -> Self {
        self.accumulator_bytes_per_read = bytes_per_read;
        self
    }

    /// Number of reads currently buffered.
    pub fn len(&self) -> usize {
        self.meta.len()
    }

    /// Whether the accumulator is empty.
    pub fn is_empty(&self) -> bool {
        self.meta.is_empty()
    }

    /// Approximate heap bytes used by buffered COO entries, counts, and
    /// per-read metadata (see `MetaHeapBytes`) — e.g. for `String` headers,
    /// which are often the *largest* per-read term (short reads against a
    /// wide-window index can have more header bytes than minimizer entry
    /// bytes), not a rounding error to leave out.
    ///
    /// Measures `entries` by `len()`, everything else by `capacity()`.
    /// `entries` is the pass's dominant buffer, and unwritten pages of a large
    /// allocation are never resident; counting its doubling slack made the
    /// per-read `should_flush()` fire at as little as ~60% of the budget.
    /// The per-read buffers (`fwd_counts`, `rc_counts`, `meta`) stay counted
    /// by capacity on purpose: their slack is headroom for per-pass costs the
    /// budget doesn't model yet (e.g. `classify run`'s results and formatted
    /// output), and counting them by length put a 10M-pair 8G run over its cap.
    ///
    /// `meta` counts its inline `size_of::<M>()` per slot plus each payload's
    /// heap bytes. The inline part is 24 bytes for a `String` header but well
    /// over 100 for a payload that carries a whole read (`classify filter`).
    pub fn approx_bytes(&self) -> usize {
        self.entries.len() * std::mem::size_of::<(u64, u32)>()
            + self.fwd_counts.capacity() * std::mem::size_of::<u32>()
            + self.rc_counts.capacity() * std::mem::size_of::<u32>()
            + self.meta.capacity() * std::mem::size_of::<M>()
            + self.meta_heap_bytes
    }

    /// `approx_bytes()` plus what classifying the buffered reads will
    /// allocate on top of them: the classification-pass accumulator (see
    /// `with_accumulator_cost_per_read`) and the pass's unique-minimizer list
    /// (`QueryInvertedIndex::unique_minimizers`, 8 bytes per unique minimizer,
    /// so at most 8 per entry). This, not `approx_bytes()`, is what
    /// `should_flush()` compares against the byte budget.
    pub fn projected_pass_bytes(&self) -> usize {
        self.approx_bytes()
            .saturating_add(
                self.accumulator_bytes_per_read
                    .saturating_mul(self.meta.len()),
            )
            .saturating_add(
                self.entries
                    .len()
                    .saturating_mul(std::mem::size_of::<u64>()),
            )
    }

    /// Add a read to the accumulator, flattening its minimizers into flat COO
    /// entries and dropping the per-read `Vec`s immediately.
    ///
    /// # Panics
    /// If this would push the accumulator past `QueryInvertedIndex::MAX_READS`
    /// (2^31 - 1, the packed read-id limit).
    pub fn push(&mut self, meta: M, fwd_mins: Vec<u64>, rc_mins: Vec<u64>) {
        assert!(
            self.meta.len() < QueryInvertedIndex::MAX_READS,
            "QueryAccumulator: read count exceeds MAX_READS ({})",
            QueryInvertedIndex::MAX_READS
        );

        let read_idx = self.meta.len() as u32;
        let fwd_count = fwd_mins.len() as u32;
        let rc_count = rc_mins.len() as u32;

        self.entries.reserve(fwd_mins.len() + rc_mins.len());
        for m in fwd_mins {
            self.entries
                .push((m, QueryInvertedIndex::pack_read_id(read_idx, false)));
        }
        for m in rc_mins {
            self.entries
                .push((m, QueryInvertedIndex::pack_read_id(read_idx, true)));
        }

        self.fwd_counts.push(fwd_count);
        self.rc_counts.push(rc_count);
        self.meta_heap_bytes += meta.heap_bytes();
        self.meta.push(meta);
    }

    /// Push a whole already-extracted batch (`Vec<(fwd, rc)>`, as produced by
    /// `extract_batch_minimizers`), pairing each read with metadata from `metas`
    /// in order.
    ///
    /// # Panics
    /// If `metas` and `extracted` have different lengths.
    pub fn extend_extracted(
        &mut self,
        metas: impl IntoIterator<Item = M>,
        extracted: Vec<(Vec<u64>, Vec<u64>)>,
    ) {
        let mut metas = metas.into_iter();
        for (fwd, rc) in extracted {
            let meta = metas
                .next()
                .expect("QueryAccumulator::extend_extracted: metas shorter than extracted");
            self.push(meta, fwd, rc);
        }
        assert!(
            metas.next().is_none(),
            "QueryAccumulator::extend_extracted: metas longer than extracted"
        );
    }

    /// Returns true once accumulated data has reached the byte budget, the
    /// optional `max_reads` cap, or the read count is one push away from
    /// `MAX_READS`.
    ///
    /// Never true on an empty accumulator, regardless of budget: a pass
    /// containing zero reads is never useful work, it's a full index scan
    /// for nothing, and a byte budget of 0 (which callers must otherwise
    /// guard against — see `classify_byte_budget`'s floor) would otherwise
    /// make this true immediately after every drain.
    pub fn should_flush(&self) -> bool {
        !self.meta.is_empty()
            && (self.projected_pass_bytes() >= self.byte_budget
                || self.max_reads.is_some_and(|mr| self.meta.len() >= mr)
                || self.meta.len() >= QueryInvertedIndex::MAX_READS)
    }

    /// Drain all buffered data into a sorted `QueryInvertedIndex` and its
    /// per-read metadata, in push order.
    ///
    /// Does *not* preserve allocated capacity for the next fill cycle: the
    /// drained `entries` are moved into the returned `QueryInvertedIndex`,
    /// which stays alive for the whole classification pass, so re-reserving
    /// the same capacity here would leave two full-size buffers resident at
    /// once. The next fill cycle re-grows from empty instead.
    ///
    /// Sorts in parallel via rayon — at accumulator scale (hundreds of millions
    /// of entries) a single-threaded sort is a real cost, unlike
    /// `QueryInvertedIndex::build`'s batch-sized sort.
    pub fn drain(&mut self) -> (QueryInvertedIndex, Vec<M>) {
        use rayon::slice::ParallelSliceMut;

        let mut entries = std::mem::take(&mut self.entries);
        let fwd_counts = std::mem::take(&mut self.fwd_counts);
        let rc_counts = std::mem::take(&mut self.rc_counts);
        let meta = std::mem::take(&mut self.meta);
        self.meta_heap_bytes = 0;

        entries.par_sort_unstable_by_key(|&(m, _)| m);

        (
            QueryInvertedIndex::from_sorted_coo(entries, fwd_counts, rc_counts),
            meta,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_is_empty() {
        let acc: QueryAccumulator<&str> = QueryAccumulator::with_budget(1024);
        assert!(acc.is_empty());
        assert_eq!(acc.len(), 0);
        assert!(!acc.should_flush());
    }

    #[test]
    fn test_push_increments_len() {
        let mut acc = QueryAccumulator::with_budget(1_000_000);
        acc.push("r0", vec![100, 200], vec![300]);
        assert_eq!(acc.len(), 1);
        acc.push("r1", vec![], vec![]);
        assert_eq!(acc.len(), 2);
    }

    #[test]
    fn test_should_flush_on_byte_budget() {
        // Small enough that a single 3-entry push exceeds it.
        let mut acc = QueryAccumulator::with_budget(1);
        assert!(!acc.should_flush());
        acc.push("r0", vec![100, 200], vec![300]);
        assert!(acc.should_flush());
    }

    #[test]
    fn test_drain_sorted_and_matches_build() {
        // Same input via QueryAccumulator (incremental) and QueryInvertedIndex::build
        // (one-shot) must produce byte-identical indices.
        let queries: Vec<(Vec<u64>, Vec<u64>)> = vec![
            (vec![300, 100, 200], vec![250, 150]),
            (vec![100, 400], vec![150]),
            (vec![], vec![]),
            (vec![500], vec![]),
        ];

        let expected = QueryInvertedIndex::build(&queries);

        let mut acc = QueryAccumulator::with_budget(usize::MAX);
        for (i, (fwd, rc)) in queries.into_iter().enumerate() {
            acc.push(format!("read_{i}"), fwd, rc);
        }
        let (actual, metas) = acc.drain();

        assert_eq!(actual.num_entries(), expected.num_entries());
        assert_eq!(actual.num_reads(), expected.num_reads());
        assert_eq!(actual.unique_minimizers(), expected.unique_minimizers());
        for i in 0..expected.num_reads() {
            assert_eq!(actual.fwd_count(i), expected.fwd_count(i));
            assert_eq!(actual.rc_count(i), expected.rc_count(i));
        }
        assert_eq!(metas, vec!["read_0", "read_1", "read_2", "read_3"]);
    }

    #[test]
    fn test_drain_resets_and_is_reusable() {
        let mut acc = QueryAccumulator::with_budget(usize::MAX);
        acc.push("r0", vec![100, 200], vec![300]);
        acc.push("r1", vec![400], vec![500, 600]);

        let (idx, metas) = acc.drain();
        assert_eq!(idx.num_reads(), 2);
        assert_eq!(metas.len(), 2);

        // Accumulator is empty and reusable after drain.
        assert!(acc.is_empty());
        assert_eq!(acc.len(), 0);

        acc.push("r2", vec![700], vec![]);
        assert_eq!(acc.len(), 1);
        let (idx2, metas2) = acc.drain();
        assert_eq!(idx2.num_reads(), 1);
        assert_eq!(metas2, vec!["r2"]);
    }

    /// Regression for the flush-degeneration bug: `approx_bytes()` must
    /// reflect the accumulator's actual (post-drain: empty) contents, not
    /// the capacity it happened to hold right before the drain. Before the
    /// fix, `drain()` re-reserved each buffer's pre-drain capacity and
    /// `approx_bytes()` measured `capacity()`, so `should_flush()` came back
    /// true immediately after a drain with zero reads buffered — every pass
    /// after the first would collapse to exactly one input batch, silently
    /// reintroducing the one-scan-per-batch behavior this type exists to
    /// eliminate (issue #21).
    #[test]
    fn test_should_flush_false_immediately_after_drain() {
        // Budget sized so a handful of pushes trips it, matching how a real
        // caller's byte_budget relates to per-batch push volume.
        let mut acc = QueryAccumulator::with_budget(200);
        for i in 0..20 {
            acc.push(format!("r{i}"), vec![i as u64, i as u64 + 1], vec![]);
            if acc.should_flush() {
                break;
            }
        }
        assert!(
            acc.should_flush(),
            "test setup: should have tripped the budget"
        );

        acc.drain();

        assert_eq!(
            acc.approx_bytes(),
            0,
            "drained accumulator must report zero bytes"
        );
        assert!(
            !acc.should_flush(),
            "should_flush() must be false immediately after drain with nothing re-pushed"
        );
    }

    /// A byte budget of 0 (e.g. a caller-computed `classify_byte_budget` that
    /// collapsed to 0 before a floor was applied — see the re-review's
    /// Finding A) must not make `should_flush()` true on an accumulator that
    /// has nothing pushed to it: that's Finding 1's failure mode again, one
    /// index scan per input batch, just reached via a zero budget instead of
    /// a stale-capacity read. Before the fix, `0 >= 0` was true regardless of
    /// whether anything had been pushed.
    #[test]
    fn test_should_flush_false_on_empty_accumulator_even_at_zero_budget() {
        let acc = QueryAccumulator::<String>::with_budget(0);
        assert!(
            !acc.should_flush(),
            "an empty accumulator must never report should_flush(), even at a zero budget"
        );
    }

    /// Regression: a payload's inline size must count, not just its heap
    /// bytes. `classify filter` buffers whole reads as metadata, and the
    /// per-read struct alone is over 100 bytes; leaving it out let a pass of
    /// short reads run well past its byte budget.
    #[test]
    fn test_meta_inline_size_counted_in_approx_bytes() {
        struct Inline {
            _payload: [u64; 16],
        }
        impl MetaHeapBytes for Inline {
            fn heap_bytes(&self) -> usize {
                0
            }
        }
        let mut acc = QueryAccumulator::with_budget(usize::MAX);
        acc.push(Inline { _payload: [0; 16] }, vec![], vec![]);
        assert!(
            acc.approx_bytes() >= std::mem::size_of::<Inline>(),
            "approx_bytes() ({}) must include the payload's inline size ({})",
            acc.approx_bytes(),
            std::mem::size_of::<Inline>()
        );
    }

    #[test]
    fn test_extend_extracted_matches_sequential_push() {
        let extracted: Vec<(Vec<u64>, Vec<u64>)> =
            vec![(vec![1, 2], vec![3]), (vec![4], vec![5, 6])];
        let metas = vec!["a", "b"];

        let mut via_extend = QueryAccumulator::with_budget(usize::MAX);
        via_extend.extend_extracted(metas.clone(), extracted.clone());

        let mut via_push = QueryAccumulator::with_budget(usize::MAX);
        for (m, (fwd, rc)) in metas.into_iter().zip(extracted) {
            via_push.push(m, fwd, rc);
        }

        let (idx_a, meta_a) = via_extend.drain();
        let (idx_b, meta_b) = via_push.drain();
        assert_eq!(idx_a.num_entries(), idx_b.num_entries());
        assert_eq!(idx_a.unique_minimizers(), idx_b.unique_minimizers());
        assert_eq!(meta_a, meta_b);
    }

    #[test]
    #[should_panic(expected = "metas shorter than extracted")]
    fn test_extend_extracted_rejects_short_metas() {
        let mut acc = QueryAccumulator::with_budget(usize::MAX);
        acc.extend_extracted(vec!["only_one"], vec![(vec![1], vec![]), (vec![2], vec![])]);
    }

    /// Classifying a pass builds its unique-minimizer list (8 bytes per
    /// unique minimizer) alongside the drained entries, so a pass that fits
    /// the budget by entries alone overshoots by up to half its entry bytes
    /// (measured ~0.6 GB on a 1.4 GB long-read pass against a human index).
    /// The accumulator cost per read, by contrast, is opt-in and defaults to
    /// zero.
    #[test]
    fn test_projected_pass_bytes_counts_unique_minimizer_list() {
        let mut acc = QueryAccumulator::with_budget(usize::MAX);
        acc.push("r0", vec![100, 200], vec![300]);
        assert_eq!(
            acc.projected_pass_bytes(),
            acc.approx_bytes() + 3 * std::mem::size_of::<u64>()
        );
    }

    /// Regression: `String` header heap bytes must count toward
    /// `approx_bytes()`, not be silently excluded. For short reads against
    /// a wide-window index, header bytes can exceed entry bytes — a budget
    /// that ignores them lets `should_flush()` admit a pass well past
    /// `--max-memory` in exactly that regime.
    #[test]
    fn test_string_header_heap_bytes_counted_in_approx_bytes() {
        let mut acc: QueryAccumulator<String> = QueryAccumulator::with_budget(usize::MAX);
        // Two minimizers => 2 * 16 = 32 bytes of entries; a header far
        // larger than that should dominate approx_bytes() if counted.
        let long_header = "x".repeat(1000);
        acc.push(long_header.clone(), vec![1, 2], vec![]);

        assert!(
            acc.approx_bytes() >= long_header.capacity(),
            "approx_bytes() ({}) should be at least the header's heap size ({})",
            acc.approx_bytes(),
            long_header.capacity()
        );

        acc.drain();
        assert_eq!(
            acc.approx_bytes(),
            0,
            "meta_heap_bytes must reset to 0 on drain, same as entries/counts"
        );
    }

    /// A pass should fill its budget with entries actually written. Unwritten
    /// `entries` capacity is never resident, and counting it made the
    /// per-read flush fire right after each doubling, at ~60% of the budget
    /// (one extra index scan per run on 4G short-read runs).
    #[test]
    fn test_pass_fills_budget_with_written_entries() {
        let budget = 1_000_000;
        let mut acc: QueryAccumulator<&str> = QueryAccumulator::with_budget(budget);
        let mut next = 0u64;
        while !acc.should_flush() {
            acc.push("r", (next..next + 100).collect(), vec![]);
            next += 100;
        }
        // Entry bytes plus the unique-minimizer reservation, as written.
        let written =
            acc.entries.len() * (std::mem::size_of::<(u64, u32)>() + std::mem::size_of::<u64>());
        assert!(
            written >= budget * 9 / 10,
            "flushed with only {written} of {budget} budget bytes written \
             (entries len {}, capacity {})",
            acc.entries.len(),
            acc.entries.capacity()
        );
    }

    #[test]
    fn test_should_flush_on_accumulator_cost_even_with_few_entries() {
        // A pass with many reads but few minimizers each (e.g. very short
        // reads against a many-bucket index) can be accumulator-bound rather
        // than entry-bound. should_flush() must catch that even though
        // approx_bytes() alone would stay well under budget.
        let bytes_per_read = 1_300; // ~160-bucket dense accumulator per read
        let budget = 10_000usize;
        let mut acc: QueryAccumulator<&str> =
            QueryAccumulator::with_budget(budget).with_accumulator_cost_per_read(bytes_per_read);

        for _ in 0..7 {
            acc.push("r", vec![1], vec![]);
        }
        // 7 reads * 1300 = 9100 < 10000: not yet.
        assert!(acc.approx_bytes() < budget, "entries alone are tiny");
        assert!(!acc.should_flush());

        acc.push("r", vec![1], vec![]);
        // 8 reads * 1300 = 10400 >= 10000: flush, even though entries are
        // still tiny on their own.
        assert!(acc.should_flush());
    }

    #[test]
    fn test_should_flush_on_max_reads_even_under_byte_budget() {
        let mut acc = QueryAccumulator::with_budget(usize::MAX).with_max_reads(Some(2));
        acc.push("r0", vec![1], vec![]);
        assert!(!acc.should_flush());
        acc.push("r1", vec![2], vec![]);
        assert!(acc.should_flush());
    }

    #[test]
    fn test_zero_minimizer_reads_preserved() {
        // A read shorter than k produces no minimizers; it must still occupy a
        // read slot so downstream indexing by position stays correct.
        let mut acc = QueryAccumulator::with_budget(usize::MAX);
        acc.push("short", vec![], vec![]);
        acc.push("normal", vec![10], vec![20]);

        let (idx, metas) = acc.drain();
        assert_eq!(idx.num_reads(), 2);
        assert_eq!(idx.fwd_count(0), 0);
        assert_eq!(idx.rc_count(0), 0);
        assert_eq!(idx.fwd_count(1), 1);
        assert_eq!(idx.rc_count(1), 1);
        assert_eq!(metas, vec!["short", "normal"]);
    }
}
