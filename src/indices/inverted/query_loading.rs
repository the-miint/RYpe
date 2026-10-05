//! Query-aware Parquet loading with bloom filter support.

use crate::error::{Result, RypeError};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use super::InvertedIndex;
use crate::constants::DEFAULT_ROW_GROUP_SIZE;
use crate::log_timing;

impl InvertedIndex {
    /// Check if ANY query minimizer might be in this row group's bloom filter.
    ///
    /// Returns `true` if:
    /// - The bloom filter is not present (conservative fallback)
    /// - Any query minimizer might be present according to the bloom filter
    ///
    /// Returns `false` only if the bloom filter definitively says NONE of the
    /// query minimizers are present.
    ///
    /// # Arguments
    /// * `bloom_filter` - Bloom filter for the minimizer column, if present
    /// * `query_minimizers` - Slice of query minimizers to check
    ///
    /// # Type Safety
    /// Parquet's `Sbbf::check` accepts any type implementing `AsBytes`, which includes
    /// both `u64` and `i64`. Since `AsBytes` uses the raw memory representation (via
    /// `std::mem::transmute`-style casting), both types produce identical byte sequences
    /// for the same bit pattern. We use `u64` directly since that's our native type.
    ///
    /// This correctly handles all u64 values including those >= 2^63, which would be
    /// negative if interpreted as i64. The bloom filter hashes bytes, not numeric values.
    ///
    /// # Note
    /// Bloom filters only work with individual value checks, not range queries.
    /// This is used by load_shard_parquet_for_query() which has individual minimizers.
    pub(super) fn bloom_filter_may_contain_any(
        bloom_filter: Option<&parquet::bloom_filter::Sbbf>,
        query_minimizers: &[u64],
    ) -> bool {
        // Empty query means nothing to find - return false regardless of bloom filter
        if query_minimizers.is_empty() {
            return false;
        }

        let Some(bf) = bloom_filter else {
            return true; // No bloom filter = conservatively include
        };

        // Check all query minimizers against the bloom filter.
        // u64 implements AsBytes, which converts to raw bytes - same representation
        // regardless of sign interpretation. Works correctly for all u64 values.
        query_minimizers.iter().any(|&m| bf.check(&m))
    }

    /// Find which row groups contain at least one query minimizer.
    ///
    /// Uses binary search per row group: O(R × log Q) where R = row groups, Q = query minimizers.
    /// This is correct even if row groups overlap or are unsorted.
    ///
    /// # Arguments
    /// * `query_minimizers` - Sorted slice of query minimizers (caller must ensure sorted)
    /// * `row_group_stats` - Array of (min, max) tuples indexed by row group index.
    ///   Parquet min/max statistics are inclusive ranges `[min, max]`.
    ///
    /// # Returns
    /// Vector of row group indices that contain at least one query minimizer.
    ///
    /// # Note on Double Range Check
    /// This function checks whether ANY query minimizer overlaps the row group range.
    /// When bloom filters are enabled, the caller performs a SECOND range check to find
    /// WHICH specific minimizers overlap, for scoped bloom filter checking. Both checks
    /// serve different purposes and both are necessary.
    pub fn find_matching_row_groups(
        query_minimizers: &[u64],
        row_group_stats: &[(u64, u64)],
    ) -> Vec<usize> {
        if query_minimizers.is_empty() || row_group_stats.is_empty() {
            return Vec::new();
        }

        // For each row group, binary search to check if any query minimizer falls within its range
        let mut matching = Vec::new();

        for (rg_idx, &(rg_min, rg_max)) in row_group_stats.iter().enumerate() {
            // Find first query minimizer >= rg_min
            let start = query_minimizers.partition_point(|&m| m < rg_min);

            // If that minimizer exists and is <= rg_max, this row group matches
            // (Parquet stats are inclusive, so we use <= for the max bound)
            if start < query_minimizers.len() && query_minimizers[start] <= rg_max {
                matching.push(rg_idx);
            }
        }

        matching
    }

    /// Load a Parquet shard, reading only row groups that contain query minimizers.
    ///
    /// Uses binary search to determine which row groups to load, then filters rows
    /// within those groups. This can skip 90%+ of data for sparse queries.
    ///
    /// # Arguments
    /// * `path` - Path to the Parquet shard file
    /// * `k`, `w`, `salt`, `source_hash` - Index parameters from manifest
    /// * `query_minimizers` - Sorted slice of query minimizers to match against
    /// * `options` - Read options (None = default behavior without bloom filters)
    ///
    /// # Panics (debug mode)
    /// Panics if `query_minimizers` is not sorted.
    ///
    /// # Returns
    /// A partial InvertedIndex containing only data from matching row groups.
    pub fn load_shard_parquet_for_query(
        path: &Path,
        k: usize,
        w: usize,
        salt: u64,
        source_hash: u64,
        query_minimizers: &[u64],
        options: Option<&super::super::parquet::ParquetReadOptions>,
    ) -> Result<Self> {
        let all_pairs = load_filtered_coo_pairs(path, k, query_minimizers, options)?;

        if all_pairs.is_empty() {
            return Ok(InvertedIndex {
                k,
                w,
                salt,
                source_hash,
                minimizers: Vec::new(),
                offsets: vec![0],
                bucket_ids: Vec::new(),
            });
        }

        // Build CSR structure from sorted COO pairs
        let mut minimizers: Vec<u64> = Vec::with_capacity(all_pairs.len() / 2);
        let mut offsets: Vec<u32> = Vec::with_capacity(all_pairs.len() / 2 + 1);
        let mut bucket_ids_out: Vec<u32> = Vec::with_capacity(all_pairs.len());

        offsets.push(0);
        let mut current_min = all_pairs[0].0;
        minimizers.push(current_min);

        for &(m, b) in &all_pairs {
            if m != current_min {
                offsets.push(bucket_ids_out.len() as u32);
                minimizers.push(m);
                current_min = m;
            }
            bucket_ids_out.push(b);
        }

        offsets.push(bucket_ids_out.len() as u32);

        Ok(InvertedIndex {
            k,
            w,
            salt,
            source_hash,
            minimizers,
            offsets,
            bucket_ids: bucket_ids_out,
        })
    }

    /// Load a Parquet shard as sorted COO pairs, filtering to query minimizers.
    ///
    /// Same filtering logic as `load_shard_parquet_for_query` (row group stats,
    /// bloom filters, row-level filtering) but returns sorted `(minimizer, bucket_id)`
    /// pairs directly instead of converting to CSR format. This eliminates CSR
    /// conversion overhead and reduces peak memory (no concurrent COO + CSR).
    ///
    /// # Arguments
    /// * `path` - Path to the Parquet shard file
    /// * `k` - K-mer size (for validation only)
    /// * `query_minimizers` - Sorted slice of query minimizers to match against
    /// * `options` - Read options (None = default behavior without bloom filters)
    ///
    /// # Returns
    /// Sorted `(minimizer, bucket_id)` pairs for minimizers in the query set.
    pub fn load_shard_coo_for_query(
        path: &Path,
        k: usize,
        query_minimizers: &[u64],
        options: Option<&super::super::parquet::ParquetReadOptions>,
    ) -> Result<Vec<(u64, u32)>> {
        load_filtered_coo_pairs(path, k, query_minimizers, options)
    }
}

/// Load filtered COO pairs from a Parquet shard.
///
/// Core implementation shared by `load_shard_parquet_for_query` (CSR output) and
/// `load_shard_coo_for_query` (COO output). Handles validation, row group selection
/// (stats + bloom filter), parallel row group loading, row-level filtering, and sorting.
///
/// # Returns
/// Sorted `(minimizer, bucket_id)` pairs. Empty Vec if no matches.
fn load_filtered_coo_pairs(
    path: &Path,
    k: usize,
    query_minimizers: &[u64],
    options: Option<&super::super::parquet::ParquetReadOptions>,
) -> Result<Vec<(u64, u32)>> {
    use arrow::array::{Array, UInt32Array, UInt64Array};
    use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ParquetRecordBatchReaderBuilder};
    use parquet::file::properties::ReaderProperties;
    use parquet::file::reader::FileReader;
    use parquet::file::serialized_reader::{ReadOptionsBuilder, SerializedFileReader};
    use parquet::file::statistics::Statistics;
    use rayon::prelude::*;
    use std::fs::File;

    let use_bloom_filter = options.map(|o| o.use_bloom_filter).unwrap_or(false);

    // Validate query_minimizers is sorted - required for binary search correctness
    // in find_matching_row_groups() and row-level filtering.
    if !query_minimizers.is_empty() && !query_minimizers.windows(2).all(|w| w[0] <= w[1]) {
        return Err(RypeError::validation(
            "query_minimizers must be sorted in ascending order",
        ));
    }

    // Validate k value
    if !matches!(k, 16 | 32 | 64) {
        return Err(RypeError::validation(format!(
            "Invalid K value for Parquet shard: {} (must be 16, 32, or 64)",
            k
        )));
    }

    if query_minimizers.is_empty() {
        return Ok(Vec::new());
    }

    // Read Parquet footer (metadata) only - not the full file data.
    // SerializedFileReader::new(File) reads just the footer (~few KB) to get metadata.
    // The file handle is explicitly scoped so it's closed before parallel reads begin.
    //
    // When bloom filter is enabled, we keep the reader alive longer to access bloom
    // filters for each row group after statistics filtering.
    let matching_row_groups: Vec<(usize, u64, u64)> = {
        let file = File::open(path).map_err(|e| RypeError::io(path, "open Parquet shard", e))?;

        // Use new_with_options when bloom filter reading is requested
        let parquet_reader = if use_bloom_filter {
            SerializedFileReader::new_with_options(
                file,
                ReadOptionsBuilder::new()
                    .with_reader_properties(
                        ReaderProperties::builder()
                            .set_read_bloom_filter(true)
                            .build(),
                    )
                    .build(),
            )?
        } else {
            SerializedFileReader::new(file)?
        };

        let metadata = parquet_reader.metadata();
        let num_row_groups = metadata.num_row_groups();

        if num_row_groups == 0 {
            return Ok(Vec::new());
        }

        // Build row group stats from Parquet metadata.
        // Statistics (min/max) enable filtering row groups before reading data.
        // Array is indexed by row group index for O(1) lookup.
        let mut rg_stats: Vec<(u64, u64)> = Vec::with_capacity(num_row_groups);
        let mut stats_missing_count = 0;

        for rg_idx in 0..num_row_groups {
            let rg_meta = metadata.row_group(rg_idx);
            let col_meta = rg_meta.column(0); // minimizer column

            let (rg_min, rg_max) = if let Some(Statistics::Int64(s)) = col_meta.statistics() {
                let min = s.min_opt().map(|v| *v as u64).unwrap_or(0);
                let max = s.max_opt().map(|v| *v as u64).unwrap_or(u64::MAX);
                (min, max)
            } else {
                // No stats available - assume full range (disables filtering for this row group)
                stats_missing_count += 1;
                (0, u64::MAX)
            };

            rg_stats.push((rg_min, rg_max));
        }

        // Warn if statistics are missing - this disables the row group filtering optimization
        if stats_missing_count == num_row_groups {
            eprintln!(
                "Warning: Parquet file {} has no row group statistics; \
                 row group filtering disabled (all {} row groups will be read)",
                path.display(),
                num_row_groups
            );
        } else if stats_missing_count > 0 {
            eprintln!(
                "Warning: Parquet file {} has {} of {} row groups without statistics",
                path.display(),
                stats_missing_count,
                num_row_groups
            );
        }

        // Phase 1: Filter by min/max statistics
        let stats_filtered = InvertedIndex::find_matching_row_groups(query_minimizers, &rg_stats);

        if stats_filtered.is_empty() {
            return Ok(Vec::new());
        }

        // Phase 2: Filter by bloom filter (if enabled)
        // This happens while we still have the parquet_reader alive.
        //
        // Why we do a SECOND range check here (after find_matching_row_groups):
        // - Phase 1 checks: Does ANY query minimizer overlap this row group? (gate)
        // - Phase 2 checks: WHICH query minimizers overlap? (scope bloom filter checks)
        // Both are necessary - the first is O(1) per row group, the second gives us
        // the exact slice to check against the bloom filter, reducing checks from
        // O(Q * R) to O(sum of overlapping minimizers per row group).
        //
        // Parquet min/max statistics are inclusive ranges [min, max], so we use
        // partition_point with <= for the max bound.
        //
        if use_bloom_filter {
            let _t_bloom = std::time::Instant::now();
            let mut bloom_filtered = Vec::with_capacity(stats_filtered.len());
            let mut bloom_rejected_count = 0usize;
            let mut bloom_filters_found = 0usize;

            for &rg_idx in &stats_filtered {
                // Get the min/max for this row group (indexed directly by rg_idx)
                let (rg_min, rg_max) = rg_stats[rg_idx];

                // Binary search to find query minimizers within [rg_min, rg_max]
                let start = query_minimizers.partition_point(|&m| m < rg_min);
                let end = query_minimizers.partition_point(|&m| m <= rg_max);
                let relevant_mins = &query_minimizers[start..end];

                let rg_reader = parquet_reader.get_row_group(rg_idx)?;
                let bf = rg_reader.get_column_bloom_filter(0); // minimizer column

                if bf.is_some() {
                    bloom_filters_found += 1;
                }

                // Check only the relevant minimizers against bloom filter
                if InvertedIndex::bloom_filter_may_contain_any(bf, relevant_mins) {
                    bloom_filtered.push(rg_idx);
                } else {
                    bloom_rejected_count += 1;
                }
            }

            // Warn if bloom filter was requested but file has no bloom filters
            if bloom_filters_found == 0 && !stats_filtered.is_empty() {
                eprintln!(
                    "Warning: --use-bloom-filter specified but {} has no bloom filters. \
                     Rebuild index with --bloom-filter to enable bloom filtering.",
                    path.display()
                );
            }

            // Log filtering statistics when verbose mode is enabled
            if std::env::var("RYPE_VERBOSE").is_ok() {
                let stats_rejected = num_row_groups - stats_filtered.len();
                eprintln!(
                    "[{}] {} RGs: {} passed stats ({} rejected), {} passed bloom ({} rejected, {} had BF)",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    num_row_groups,
                    stats_filtered.len(),
                    stats_rejected,
                    bloom_filtered.len(),
                    bloom_rejected_count,
                    bloom_filters_found
                );
            }

            bloom_filtered
                .into_iter()
                .map(|rg_idx| {
                    let (rg_min, rg_max) = rg_stats[rg_idx];
                    (rg_idx, rg_min, rg_max)
                })
                .collect::<Vec<(usize, u64, u64)>>()
        } else {
            stats_filtered
                .into_iter()
                .map(|rg_idx| {
                    let (rg_min, rg_max) = rg_stats[rg_idx];
                    (rg_idx, rg_min, rg_max)
                })
                .collect::<Vec<(usize, u64, u64)>>()
        }
    }; // parquet_reader and file handle dropped here

    if matching_row_groups.is_empty() {
        return Ok(Vec::new());
    }

    // Parse the footer once and reuse it in every per-row-group reader below via
    // `new_with_metadata`. Without this, each of the (potentially thousands of)
    // parallel per-row-group tasks below would re-open the file AND re-parse and
    // re-deserialize the full footer (which itself lists every row group's
    // statistics) just to build a reader for a single row group.
    let arrow_metadata = {
        let file = File::open(path).map_err(|e| RypeError::io(path, "open Parquet shard", e))?;
        ArrowReaderMetadata::load(&file, Default::default())?
    };

    // Parallel read of matching row groups, batched into `num_threads` large
    // chunks rather than one rayon task per row group.
    //
    // A large shard can have on the order of 10,000 matching row groups, and
    // one-task-per-row-group was measured (via --timing on the real 22GB
    // perf index) to leave most of the theoretical 12x parallel speedup on
    // the table — decode+filter work summed to ~90s across tasks but the
    // parallel map's wall time was ~65s (~1.4x effective speedup, not 12x)
    // — the tasks are too small and too numerous for rayon to schedule
    // efficiently, and per-task allocation/file-open overhead dominates.
    // The single-threaded concatenation afterward cost another ~39s across
    // ~9,852 small `extend()` calls even after pre-sizing the destination.
    //
    // Batching into ~num_threads chunks fixes both: each task now opens one
    // file and reads many row groups through one `ParquetRecordBatchReader`
    // (`with_row_groups` accepts multiple indices and streams their batches
    // in order), and the final concatenation is ~num_threads `extend()`
    // calls instead of thousands.
    //
    // Correctness: row groups within one chunk are contiguous, non-overlapping,
    // ascending-sorted ranges of one globally sorted stream (established at
    // write time), so the two-pointer merge can run continuously across an
    // entire chunk's batch stream using one combined [chunk_min, chunk_max]
    // bound — no per-row-group bounding is needed within a chunk.
    let path = path.to_path_buf(); // Clone path for parallel closure

    // Oversubscribe chunks relative to thread count. Exactly num_threads
    // chunks (tried first) removed rayon's ability to load-balance: with one
    // chunk per thread and no spare work, an unevenly-sized chunk (row
    // groups don't all match the same number of query minimizers) leaves
    // other threads idle while the straggler finishes alone — measured on
    // the real 22GB index, shard_load_total got *worse* (146.6s -> 173.5s)
    // with exactly num_threads chunks despite far fewer file opens.
    // Oversubscribing gives the work-stealing scheduler enough granularity
    // to rebalance stragglers, while still cutting file-open/concat count
    // by roughly this factor vs one task per row group (was ~9,852/shard).
    const CHUNK_OVERSUBSCRIPTION_FACTOR: usize = 8;
    let num_threads = rayon::current_num_threads();
    let num_chunks = (num_threads * CHUNK_OVERSUBSCRIPTION_FACTOR)
        .min(matching_row_groups.len())
        .max(1);
    // matching_row_groups.len().div_ceil(num_chunks) would be simpler but
    // div_ceil on usize was stabilized in Rust 1.73, above this crate's
    // declared MSRV (1.70) — see the same workaround in c_api.rs.
    let chunk_size = (matching_row_groups.len() + num_chunks - 1) / num_chunks;
    let chunks: Vec<&[(usize, u64, u64)]> = matching_row_groups.chunks(chunk_size).collect();

    // Estimate pairs for pre-allocation: one row group's worth at a
    // conservative 10% selectivity, *not* scaled by how many row groups a
    // chunk covers. Real selectivity is often 1-5% (see the module-level
    // SHARD_SELECTIVITY_ESTIMATE discussion in memory.rs), and a chunk can
    // cover 100+ row groups at high oversubscription — scaling this
    // estimate by chunk.len() would over-reserve by that same factor for
    // every chunk's result Vec, and since chunk_results below buffers every
    // chunk's completed Vec (with whatever capacity it was given) until the
    // final concat loop runs, all of them are resident at once by the time
    // this function returns. Reserving for one row group and letting `pairs`
    // grow via ordinary amortized doubling for the rest of the chunk keeps
    // the initial (and typically dominant, at low real selectivity)
    // allocation bounded independent of chunk size.
    let estimated_pairs_per_rg = DEFAULT_ROW_GROUP_SIZE / 10;

    // Instrumentation: split shard_load_total into "decode" (opening the file,
    // Snappy/DELTA_BINARY_PACKED decompression, Arrow batch materialization —
    // all inside the `parquet`/`arrow` crates, not our code) vs "filter" (our
    // two-pointer sorted-intersection loop below). Only meaningful under
    // --timing; the atomics are Relaxed fetch_adds, cheap enough to leave on
    // the hot path unconditionally rather than branch on ENABLE_TIMING per chunk.
    let decode_ns = AtomicU64::new(0);
    let filter_ns = AtomicU64::new(0);

    let t_parallel_phase = Instant::now();
    let chunk_results: Vec<Result<Vec<(u64, u32)>>> = chunks
        .par_iter()
        .map(|chunk| {
            // Combined bound spanning the whole chunk — row groups within a
            // chunk are contiguous and sorted, so one [min, max] pair covers
            // the entire chunk's minimizer range.
            let chunk_min = chunk.iter().map(|&(_, min, _)| min).min().unwrap();
            let chunk_max = chunk.iter().map(|&(_, _, max)| max).max().unwrap();
            let q_start = query_minimizers.partition_point(|&m| m < chunk_min);
            let q_end = query_minimizers.partition_point(|&m| m <= chunk_max);
            let bounded = &query_minimizers[q_start..q_end];

            if bounded.is_empty() {
                return Ok(Vec::new());
            }

            let rg_indices: Vec<usize> = chunk.iter().map(|&(rg_idx, _, _)| rg_idx).collect();

            let file = File::open(&path)?;
            let builder =
                ParquetRecordBatchReaderBuilder::new_with_metadata(file, arrow_metadata.clone());
            let mut reader = builder.with_row_groups(rg_indices).build()?;

            let mut pairs = Vec::with_capacity(estimated_pairs_per_rg);

            // Two-pointer sorted intersection: `bounded` (query minimizers, unique,
            // ascending) against this chunk's minimizer column, streamed across
            // every row group in the chunk (ascending; may contain runs of
            // duplicate minimizers with different bucket_ids). `qi` is carried
            // across every batch in the chunk, spanning row-group boundaries,
            // since the whole chunk is one continuous sorted range.
            let mut qi = 0usize;

            // Guards the invariant the two-pointer merge below depends on:
            // each row group's minimizer column is ascending, AND row groups
            // within a chunk concatenate into one ascending stream (both
            // established at write time — `ShardAccumulator::flush_shard`
            // sorts before writing, `stream_to_parquet_shards` k-way-merges).
            // This is why a boundary-only check (comparing only each chunk's
            // first/last pair, as the check below this closure does) is
            // insufficient: the two-pointer only ever pushes on `qv == rv`
            // with `qi` advancing monotonically, so its *output* is sorted
            // by construction regardless of whether the input was — an
            // interior violation doesn't produce unsorted output for that
            // check to catch, it silently drops the matches that fell
            // behind `qi`.
            //
            // Checking only *batch* edges (previous batch's last value vs.
            // this batch's first) is ALSO insufficient, despite looking like
            // an O(num_batches) version of the same idea: `with_row_groups`
            // does not guarantee a batch never spans two row groups —
            // verified directly (see the regression test below): row groups
            // smaller than the reader's internal batch size get coalesced
            // into one shared Arrow batch, so a violation between two row
            // groups can land *inside* a single batch's own `mins` array,
            // not at its edge. So this checks each batch's own values for
            // monotonicity too, not just against the previous batch.
            //
            // This is O(total rows read), not O(num_batches) — the same
            // order as the single-threaded `all_pairs.windows(2)` scan this
            // function replaced overall, but done here per-chunk, inline
            // with decode, inside the already-parallel `par_iter` this
            // closure runs under (the same rows the two-pointer below
            // already touches), rather than as a second single-threaded
            // pass over the whole concatenated result.
            let mut prev_batch_last_min: Option<u64> = None;

            loop {
                if qi >= bounded.len() {
                    break;
                }
                // Timed around `reader.next()` itself (not `for batch in reader`,
                // whose desugared `.next()` call runs *before* the loop body —
                // timing inside the body there measures almost nothing, since the
                // real decompression work already happened by the time the body
                // starts). This is the only place actual decode work occurs.
                let t_decode = Instant::now();
                let next = reader.next();
                decode_ns.fetch_add(t_decode.elapsed().as_nanos() as u64, Ordering::Relaxed);
                let Some(batch) = next else { break };
                let batch = batch?;

                let min_col = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| {
                        RypeError::format(&path, "Expected UInt64Array for minimizer column")
                    })?;

                let bid_col = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .ok_or_else(|| {
                        RypeError::format(&path, "Expected UInt32Array for bucket_id column")
                    })?;

                let mins: &[u64] = min_col.values();
                let bids: &[u32] = bid_col.values();

                if let (Some(&first_min), Some(prev_last)) = (mins.first(), prev_batch_last_min) {
                    if prev_last > first_min {
                        return Err(RypeError::format(
                            &path,
                            format!(
                                "non-monotonic minimizer column within a row-group chunk: \
                                 batch starting at {first_min} follows a batch ending at \
                                 {prev_last} (row groups {:?})",
                                chunk.iter().map(|&(idx, _, _)| idx).collect::<Vec<_>>(),
                            ),
                        ));
                    }
                }
                if let Some(pos) = mins.windows(2).position(|w| w[0] > w[1]) {
                    return Err(RypeError::format(
                        &path,
                        format!(
                            "non-monotonic minimizer column within a row-group chunk: \
                             {} follows {} within one batch (row groups {:?})",
                            mins[pos + 1],
                            mins[pos],
                            chunk.iter().map(|&(idx, _, _)| idx).collect::<Vec<_>>(),
                        ),
                    ));
                }
                if let Some(&last_min) = mins.last() {
                    prev_batch_last_min = Some(last_min);
                }

                let t_filter = Instant::now();
                let mut ri = 0usize;
                while qi < bounded.len() && ri < mins.len() {
                    let qv = bounded[qi];
                    let rv = mins[ri];
                    if qv < rv {
                        qi += 1;
                    } else if qv > rv {
                        ri += 1;
                    } else {
                        pairs.push((rv, bids[ri]));
                        ri += 1;
                    }
                }
                filter_ns.fetch_add(t_filter.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }

            Ok(pairs)
        })
        .collect();

    log_timing(
        "load_filtered_coo_pairs: parallel_phase_wall (map+collect, wall time)",
        t_parallel_phase.elapsed().as_millis(),
    );
    log_timing(
        "load_filtered_coo_pairs: decode (open+decompress+decode, sum across chunk tasks)",
        (decode_ns.load(Ordering::Relaxed) / 1_000_000) as u128,
    );
    log_timing(
        "load_filtered_coo_pairs: filter (two-pointer intersection, sum across chunk tasks)",
        (filter_ns.load(Ordering::Relaxed) / 1_000_000) as u128,
    );

    // Concatenate results from all chunks (now ~num_threads of them, not
    // thousands), adding context for failures. Pre-size from the
    // already-known per-task lengths to avoid repeated reallocate-and-copy.
    let t_concat = Instant::now();
    let total_len: usize = chunk_results
        .iter()
        .map(|r| r.as_ref().map(|v| v.len()).unwrap_or(0))
        .sum();
    let mut all_pairs: Vec<(u64, u32)> = Vec::with_capacity(total_len);

    // Each chunk's own `pairs` come out ascending by construction — not
    // because its input was valid, but because the two-pointer above only
    // pushes on `qv == rv` with `qi` advancing monotonically across the
    // whole chunk, so the output can never regress regardless of what the
    // underlying row groups actually contained. That's why this boundary
    // check exists *alongside* the per-batch check inside the chunk closure
    // above, not instead of it: an interior (within-chunk) ordering
    // violation doesn't produce unsorted `pairs` for a boundary check to
    // catch here — it silently drops the matches that `qi` had already
    // advanced past, which the per-batch check catches at the point of
    // decode instead. This check covers only the boundary *between* chunks,
    // which the per-batch check (scoped to one chunk's own reader) cannot
    // see. Checking just those boundaries is O(num_chunks), not
    // O(total_len), so it's cheap enough to run unconditionally in release
    // builds — unlike the full `all_pairs.windows(2)` scan this replaces,
    // which is exactly the per-entry cost the batching in this function
    // exists to avoid.
    //
    // Both checks guard a real invariant, not a hypothetical one: it holds
    // for every shard this repo's writers produce
    // (`ShardAccumulator::flush_shard` sorts before writing,
    // `stream_to_parquet_shards` k-way-merges), but an externally produced
    // or future `.ryxdi` that violates it would otherwise corrupt every
    // downstream binary_search/partition_point/gallop_for_each silently,
    // with no error and no way to detect it from the output — so violations
    // are a hard error, not a debug-only assert.
    let mut prev_chunk_last_min: Option<u64> = None;
    for (chunk_idx, result) in chunk_results.into_iter().enumerate() {
        let pairs = result.map_err(|e| {
            let first_rg = chunks[chunk_idx].first().map(|&(idx, _, _)| idx);
            let last_rg = chunks[chunk_idx].last().map(|&(idx, _, _)| idx);
            RypeError::io(
                path.clone(),
                "read row group chunk",
                std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("row groups {:?}..={:?}: {}", first_rg, last_rg, e),
                ),
            )
        })?;

        if let Some((first_min, _)) = pairs.first().copied() {
            if let Some(prev_last) = prev_chunk_last_min {
                if prev_last > first_min {
                    let first_rg = chunks[chunk_idx].first().map(|&(idx, _, _)| idx);
                    return Err(RypeError::format(
                        &path,
                        format!(
                            "non-monotonic minimizer stream at row group chunk boundary \
                             (starting row group {:?}): previous chunk ended at {}, this \
                             chunk starts at {} — shard file is not globally sorted by \
                             minimizer, which every downstream binary search assumes",
                            first_rg, prev_last, first_min
                        ),
                    ));
                }
            }
            prev_chunk_last_min = pairs.last().map(|&(m, _)| m);
        }

        all_pairs.extend(pairs);
    }
    log_timing(
        "load_filtered_coo_pairs: concat (single-threaded extend into all_pairs)",
        t_concat.elapsed().as_millis(),
    );

    // Row groups are visited in ascending rg_idx order (matching_row_groups is
    // built by iterating 0..num_row_groups in order and rayon's par_iter().collect()
    // preserves input order), and within a single shard file, row groups are
    // non-overlapping contiguous slices of one globally sorted stream (the writer
    // emits one continuous sorted (minimizer, bucket_id) stream chunked into row
    // groups — has_overlapping_shards refers to overlap ACROSS shard files, not
    // within one). So the concatenation above is already fully sorted — verified,
    // not just asserted, at each chunk boundary in the loop above.

    Ok(all_pairs)
}

/// Load a single row group and return filtered (minimizer, bucket_id) pairs.
///
/// Pairs are sorted by minimizer within the row group (enforced at write time).
///
/// # Query Minimizer Filtering
///
/// Before filtering pairs, we narrow the query minimizers to only those
/// that could possibly match this row group:
///
/// 1. Read the RG's min/max minimizer values from Parquet column statistics
/// 2. Binary search the sorted `query_minimizers` to find the range:
///    - `start = query_minimizers.partition_point(|m| m < rg_min)`
///    - `end = query_minimizers.partition_point(|m| m <= rg_max)`
/// 3. Use `query_minimizers[start..end]` for filtering pairs
///
/// This reduces merge-join work when query minimizers span a wider range
/// than any individual row group.
///
/// # Arguments
/// * `path` - Path to the Parquet file
/// * `rg_idx` - Row group index within the file
/// * `query_minimizers` - Sorted, deduplicated query minimizers
///
/// # Returns
/// Sorted (minimizer, bucket_id) pairs where minimizer is in the bounded query range.
///
/// # Errors
/// Returns an error if the row group lacks column statistics (required for range-bounded filtering).
pub fn load_row_group_pairs(
    path: &std::path::Path,
    rg_idx: usize,
    query_minimizers: &[u64],
) -> Result<Vec<(u64, u32)>> {
    use arrow::array::{Array, UInt32Array, UInt64Array};
    use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ParquetRecordBatchReaderBuilder};
    use parquet::file::statistics::Statistics;
    use std::fs::File;

    if query_minimizers.is_empty() {
        return Ok(Vec::new());
    }

    // Open the file once and parse the footer once, reusing the same
    // ArrowReaderMetadata both for statistics (below) and for building the
    // Arrow reader (further down). Previously this opened the file and parsed
    // the footer twice per call, which is expensive when a shard has many row
    // groups (the footer lists every row group's statistics).
    let file = File::open(path).map_err(|e| RypeError::io(path, "open Parquet file", e))?;
    let arrow_metadata = ArrowReaderMetadata::load(&file, Default::default())?;
    let metadata = arrow_metadata.metadata();

    if rg_idx >= metadata.num_row_groups() {
        return Err(RypeError::validation(format!(
            "Row group index {} out of range (file has {} row groups)",
            rg_idx,
            metadata.num_row_groups()
        )));
    }

    // Get RG min/max from Parquet statistics
    let rg_meta = metadata.row_group(rg_idx);
    let col_meta = rg_meta.column(0); // minimizer column

    let (rg_min, rg_max, rg_num_rows) = match col_meta.statistics() {
        Some(Statistics::Int64(s)) => {
            let min = s.min_opt().ok_or_else(|| {
                RypeError::format(
                    path,
                    format!(
                        "Row group {} has statistics but min value is missing",
                        rg_idx
                    ),
                )
            })?;
            let max = s.max_opt().ok_or_else(|| {
                RypeError::format(
                    path,
                    format!(
                        "Row group {} has statistics but max value is missing",
                        rg_idx
                    ),
                )
            })?;
            (*min as u64, *max as u64, rg_meta.num_rows() as usize)
        }
        _ => {
            return Err(RypeError::format(
                path,
                format!(
                    "Row group {} lacks column statistics; parallel RG processing requires \
                     statistics for range-bounded filtering. Rebuild index with statistics enabled.",
                    rg_idx
                ),
            ));
        }
    };

    // Binary search to find bounded query minimizers
    let q_start = query_minimizers.partition_point(|&m| m < rg_min);
    let q_end = query_minimizers.partition_point(|&m| m <= rg_max);

    // No overlap between query minimizers and this row group's range
    if q_start >= q_end {
        return Ok(Vec::new());
    }

    let bounded_queries = &query_minimizers[q_start..q_end];

    // Build the Arrow reader from the already-parsed metadata (no second footer parse).
    let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(file, arrow_metadata);
    let reader = builder.with_row_groups(vec![rg_idx]).build()?;

    // Estimate capacity based on row group size
    let estimated_pairs = rg_num_rows / 10; // Assume ~10% match rate
    let mut pairs = Vec::with_capacity(estimated_pairs);

    // Two-pointer sorted intersection against `bounded_queries` (sorted, unique).
    // The row group's minimizer column is sorted ascending (may contain runs of
    // duplicate minimizers with different bucket_ids), and `qi` is carried across
    // Arrow batches since the column is sorted continuously within the row group.
    let mut qi = 0usize;

    for batch in reader {
        if qi >= bounded_queries.len() {
            break;
        }
        let batch = batch?;

        let min_col = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| RypeError::format(path, "Expected UInt64Array for minimizer column"))?;

        let bid_col = batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| RypeError::format(path, "Expected UInt32Array for bucket_id column"))?;

        let mins: &[u64] = min_col.values();
        let bids: &[u32] = bid_col.values();

        let mut ri = 0usize;
        while qi < bounded_queries.len() && ri < mins.len() {
            let qv = bounded_queries[qi];
            let rv = mins[ri];
            if qv < rv {
                qi += 1;
            } else if qv > rv {
                ri += 1;
            } else {
                pairs.push((rv, bids[ri]));
                ri += 1;
            }
        }
    }

    // Pairs should already be sorted by minimizer (enforced at write time)
    // But verify in debug mode
    debug_assert!(
        pairs.windows(2).all(|w| w[0].0 <= w[1].0),
        "Row group pairs should be sorted by minimizer"
    );

    Ok(pairs)
}

/// Row group metadata for filtering and memory estimation.
#[derive(Debug, Clone, Copy)]
pub struct RowGroupRangeInfo {
    /// Row group index within the file
    pub rg_idx: usize,
    /// Minimum minimizer value in this row group
    pub min: u64,
    /// Maximum minimizer value in this row group
    pub max: u64,
    /// Total uncompressed size in bytes (for memory estimation)
    pub uncompressed_size: usize,
}

/// Get the min/max statistics and sizes for each row group in a Parquet file.
///
/// Returns a vector of `RowGroupRangeInfo` with ranges and uncompressed sizes.
/// Only row groups with valid statistics are included.
///
/// # Errors
/// Returns an error if the file cannot be opened or lacks statistics.
pub fn get_row_group_ranges(path: &std::path::Path) -> Result<Vec<RowGroupRangeInfo>> {
    use parquet::file::reader::FileReader;
    use parquet::file::serialized_reader::SerializedFileReader;
    use parquet::file::statistics::Statistics;
    use std::fs::File;

    let file = File::open(path).map_err(|e| RypeError::io(path, "open Parquet file", e))?;
    let parquet_reader = SerializedFileReader::new(file)?;
    let metadata = parquet_reader.metadata();

    let mut ranges = Vec::with_capacity(metadata.num_row_groups());

    for rg_idx in 0..metadata.num_row_groups() {
        let rg_meta = metadata.row_group(rg_idx);
        let col_meta = rg_meta.column(0); // minimizer column

        match col_meta.statistics() {
            Some(Statistics::Int64(s)) => {
                if let (Some(&min), Some(&max)) = (s.min_opt(), s.max_opt()) {
                    ranges.push(RowGroupRangeInfo {
                        rg_idx,
                        min: min as u64,
                        max: max as u64,
                        uncompressed_size: rg_meta.total_byte_size() as usize,
                    });
                }
            }
            _ => {
                return Err(RypeError::format(
                    path,
                    format!(
                        "Row group {} lacks column statistics; parallel RG processing requires \
                         statistics for range-bounded filtering.",
                        rg_idx
                    ),
                ));
            }
        }
    }

    Ok(ranges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indices::parquet::{ParquetReadOptions, ParquetWriteOptions};
    use crate::types::IndexMetadata;
    use anyhow::Result;
    use std::collections::HashMap;
    use tempfile::TempDir;

    /// Helper to build an InvertedIndex for testing from bucket minimizers.
    fn build_test_inverted_index(
        k: usize,
        w: usize,
        salt: u64,
        buckets: Vec<(u32, &str, Vec<u64>)>,
    ) -> InvertedIndex {
        let mut bucket_map: HashMap<u32, Vec<u64>> = HashMap::new();
        let mut bucket_names: HashMap<u32, String> = HashMap::new();
        let mut bucket_minimizer_counts: HashMap<u32, usize> = HashMap::new();

        for (id, name, mins) in buckets {
            bucket_minimizer_counts.insert(id, mins.len());
            bucket_map.insert(id, mins);
            bucket_names.insert(id, name.to_string());
        }

        let metadata = IndexMetadata {
            k,
            w,
            salt,
            bucket_names,
            bucket_sources: HashMap::new(),
            bucket_minimizer_counts,
            largest_shard_entries: 0,
            bucket_file_stats: None,
        };

        InvertedIndex::build_from_bucket_map(k, w, salt, &bucket_map, &metadata)
    }

    #[test]
    fn test_load_parquet_for_query_basic() -> Result<()> {
        let tmp = TempDir::new()?;
        let path = tmp.path().join("shard.parquet");

        // Create an inverted index with multiple minimizers across buckets
        let inverted = build_test_inverted_index(
            64,
            50,
            0xABCD,
            vec![
                (1, "A", vec![100, 200, 300, 400, 500]),
                (2, "B", vec![200, 300, 600, 700]),
                (3, "C", vec![500, 800, 900]),
            ],
        );

        // Save as Parquet
        inverted.save_shard_parquet(&path, 0, None)?;

        // Query for specific minimizers - should only return matching entries
        let query_minimizers = vec![200, 300, 500]; // sorted
        let loaded = InvertedIndex::load_shard_parquet_for_query(
            &path,
            inverted.k,
            inverted.w,
            inverted.salt,
            inverted.source_hash,
            &query_minimizers,
            None, // No bloom filter options for tests
        )?;

        // Verify we only got the queried minimizers
        assert_eq!(loaded.minimizers().len(), 3);
        assert!(loaded.minimizers().contains(&200));
        assert!(loaded.minimizers().contains(&300));
        assert!(loaded.minimizers().contains(&500));

        // Verify bucket hits are correct
        let hits = loaded.get_bucket_hits(&[200, 300, 500]);
        assert_eq!(hits.get(&1), Some(&3)); // 200, 300, 500
        assert_eq!(hits.get(&2), Some(&2)); // 200, 300
        assert_eq!(hits.get(&3), Some(&1)); // 500

        Ok(())
    }

    #[test]
    fn test_load_parquet_for_query_empty_query() -> Result<()> {
        let tmp = TempDir::new()?;
        let path = tmp.path().join("shard.parquet");

        // Create an inverted index
        let inverted = build_test_inverted_index(64, 50, 0, vec![(1, "A", vec![100, 200, 300])]);
        inverted.save_shard_parquet(&path, 0, None)?;

        // Empty query should return empty result without reading row groups
        let loaded = InvertedIndex::load_shard_parquet_for_query(
            &path,
            inverted.k,
            inverted.w,
            inverted.salt,
            inverted.source_hash,
            &[],  // empty query
            None, // No bloom filter options
        )?;

        assert_eq!(loaded.minimizers().len(), 0);
        assert_eq!(loaded.bucket_ids().len(), 0);

        Ok(())
    }

    #[test]
    fn test_load_parquet_for_query_no_matches() -> Result<()> {
        let tmp = TempDir::new()?;
        let path = tmp.path().join("shard.parquet");

        // Create an inverted index with minimizers in a specific range
        let inverted = build_test_inverted_index(64, 50, 0, vec![(1, "A", vec![100, 200, 300])]);
        inverted.save_shard_parquet(&path, 0, None)?;

        // Query for minimizers not in the file - should return empty
        let query_minimizers = vec![1000, 2000, 3000]; // sorted, but not in file
        let loaded = InvertedIndex::load_shard_parquet_for_query(
            &path,
            inverted.k,
            inverted.w,
            inverted.salt,
            inverted.source_hash,
            &query_minimizers,
            None, // No bloom filter options for tests
        )?;

        assert_eq!(loaded.minimizers().len(), 0);
        assert_eq!(loaded.bucket_ids().len(), 0);

        Ok(())
    }

    #[test]
    fn test_load_parquet_for_query_multiple_row_groups() -> Result<()> {
        let tmp = TempDir::new()?;
        let path = tmp.path().join("large.parquet");

        // Create a large inverted index that will span multiple row groups (>100k pairs)
        // Each bucket has many minimizers to create enough pairs
        // Create 10 buckets, each with 15000 minimizers (150k pairs total = 2+ row groups)
        let buckets: Vec<(u32, &str, Vec<u64>)> = (0..10u32)
            .map(|bucket_id| {
                let base = bucket_id as u64 * 100_000;
                let minimizers: Vec<u64> = (0..15000).map(|i| base + i as u64).collect();
                let name: &str = match bucket_id {
                    0 => "bucket_0",
                    1 => "bucket_1",
                    2 => "bucket_2",
                    3 => "bucket_3",
                    4 => "bucket_4",
                    5 => "bucket_5",
                    6 => "bucket_6",
                    7 => "bucket_7",
                    8 => "bucket_8",
                    _ => "bucket_9",
                };
                (bucket_id, name, minimizers)
            })
            .collect();

        let inverted = build_test_inverted_index(64, 50, 0x1234, buckets);
        inverted.save_shard_parquet(&path, 0, None)?;

        // Query for minimizers from different row groups
        // First row group: bucket 0 minimizers (0-99999)
        // Later row groups: bucket 5 minimizers (500000-514999)
        let query_minimizers = vec![100, 200, 500_100, 500_200]; // sorted
        let loaded = InvertedIndex::load_shard_parquet_for_query(
            &path,
            inverted.k,
            inverted.w,
            inverted.salt,
            inverted.source_hash,
            &query_minimizers,
            None, // No bloom filter options for tests
        )?;

        // Should find all queried minimizers
        assert!(loaded.minimizers().contains(&100));
        assert!(loaded.minimizers().contains(&200));
        assert!(loaded.minimizers().contains(&500_100));
        assert!(loaded.minimizers().contains(&500_200));

        // Bucket 0 should have hits for 100, 200
        // Bucket 5 should have hits for 500100, 500200
        let hits = loaded.get_bucket_hits(&query_minimizers);
        assert_eq!(hits.get(&0), Some(&2));
        assert_eq!(hits.get(&5), Some(&2));

        Ok(())
    }

    #[test]
    fn test_load_parquet_for_query_unsorted_input() -> Result<()> {
        let tmp = TempDir::new()?;
        let path = tmp.path().join("shard.parquet");

        // Create a simple Parquet file
        let inverted = build_test_inverted_index(64, 50, 0, vec![(1, "A", vec![100, 200, 300])]);
        inverted.save_shard_parquet(&path, 0, None)?;

        // Unsorted query should fail with clear error
        let unsorted_query = vec![300, 100, 200]; // NOT sorted
        let result = InvertedIndex::load_shard_parquet_for_query(
            &path,
            inverted.k,
            inverted.w,
            inverted.salt,
            inverted.source_hash,
            &unsorted_query,
            None,
        );

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("sorted"),
            "Error message should mention sorting: {}",
            err_msg
        );

        Ok(())
    }

    /// Regression: `load_filtered_coo_pairs` must reject a shard file whose
    /// row groups are individually sorted but not globally sorted as a
    /// whole (a non-monotonic concatenation), rather than silently
    /// returning wrong data.
    ///
    /// The concatenation step assumes row groups form one continuous sorted
    /// stream (true for every shard this repo's writers produce), and used
    /// to guard that with `all_pairs.windows(2).all(...)` — a no-op in
    /// release builds. This writes a shard file directly (bypassing the
    /// normal sorted writer) with two row groups that are each internally
    /// sorted but decreasing relative to each other, which is exactly the
    /// class of corruption the guard exists to catch, and asserts it
    /// surfaces as an error rather than silently wrong results.
    #[test]
    fn test_load_filtered_coo_pairs_rejects_non_monotonic_row_groups() -> Result<()> {
        use arrow::array::{ArrayRef, UInt32Array, UInt64Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use parquet::file::properties::WriterProperties;
        use std::sync::Arc;

        let tmp = TempDir::new()?;
        let path = tmp.path().join("adversarial.parquet");

        // Row group 0: minimizers 100_000..100_010 (internally sorted).
        // Row group 1: minimizers 0..10 (internally sorted).
        // Concatenated (RG0 then RG1, ascending rg_idx order): decreasing at
        // the boundary — exactly the violation this test targets.
        let rg0_mins: Vec<u64> = (100_000u64..100_010).collect();
        let rg1_mins: Vec<u64> = (0u64..10).collect();
        let all_mins: Vec<u64> = rg0_mins.iter().chain(rg1_mins.iter()).copied().collect();
        let all_bucket_ids: Vec<u32> = vec![1u32; all_mins.len()];

        let schema = Arc::new(Schema::new(vec![
            Field::new("minimizer", DataType::UInt64, false),
            Field::new("bucket_id", DataType::UInt32, false),
        ]));
        // One row group per 10 rows, so the two halves land in separate row
        // groups without needing multiple write_batch calls.
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(10))
            .build();
        let file = std::fs::File::create(&path)?;
        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;
        let min_array: ArrayRef = Arc::new(UInt64Array::from(all_mins));
        let bid_array: ArrayRef = Arc::new(UInt32Array::from(all_bucket_ids));
        let batch = RecordBatch::try_new(schema, vec![min_array, bid_array])?;
        writer.write(&batch)?;
        writer.close()?;

        // Query spans both row groups so both get selected for reading.
        let query_minimizers = vec![5u64, 100_005u64];
        let result = load_filtered_coo_pairs(&path, 64, &query_minimizers, None);

        assert!(
            result.is_err(),
            "non-monotonic row-group concatenation must be rejected, not silently accepted"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("non-monotonic") || err_msg.contains("sorted"),
            "error should describe the sortedness violation: {}",
            err_msg
        );

        Ok(())
    }

    /// Reproduces the re-review's Finding C: the chunk-boundary check above
    /// can only see a violation *between* chunks, because the two-pointer
    /// merge's output is sorted by construction (it only pushes on
    /// `qv == rv` with `qi` advancing monotonically) regardless of whether
    /// its input was — an ordering violation *interior* to a chunk (between
    /// two row groups that land in the same chunk) doesn't produce unsorted
    /// output for the boundary check to catch; it silently drops the
    /// matches `qi` had already advanced past.
    ///
    /// The prior version of this test (2 row groups) passed only because
    /// `num_chunks = min(num_threads * 8, matching_row_groups.len())` made
    /// `chunk_size == 1` at that size, so the violation always landed
    /// exactly on a chunk boundary — the case that check *does* cover, not
    /// the interior case it doesn't. This test sizes the row-group count
    /// from the live thread count so the first chunk always contains at
    /// least 3 row groups (`chunk_size >= 3`) regardless of machine core
    /// count, and swaps the first two row groups' values so the violation
    /// is interior to that chunk.
    #[test]
    fn test_load_filtered_coo_pairs_rejects_non_monotonic_row_groups_interior_to_a_chunk(
    ) -> Result<()> {
        use arrow::array::{ArrayRef, UInt32Array, UInt64Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use parquet::file::properties::WriterProperties;
        use std::sync::Arc;

        let tmp = TempDir::new()?;
        let path = tmp.path().join("adversarial_interior.parquet");

        // `num_chunks = min(num_threads * 8, num_row_groups)`. Sizing
        // `num_row_groups` to exactly `num_threads * 8 * 3` makes
        // `chunk_size == 3` regardless of the live thread count, so the
        // first chunk always spans row groups 0, 1, and 2.
        let num_threads = rayon::current_num_threads().max(1);
        let num_row_groups = num_threads * 8 * 3;

        // Row group i holds one minimizer, i * 1000 — globally ascending by
        // row-group index — except row groups 0 and 1 are swapped (1000,
        // then 0), which is interior to the first chunk (row groups 0..3)
        // rather than at a chunk boundary.
        let mut mins: Vec<u64> = (0..num_row_groups as u64).map(|i| i * 1000).collect();
        mins.swap(0, 1);
        let bucket_ids: Vec<u32> = vec![1u32; mins.len()];

        let schema = Arc::new(Schema::new(vec![
            Field::new("minimizer", DataType::UInt64, false),
            Field::new("bucket_id", DataType::UInt32, false),
        ]));
        // One row per row group, so each element of `mins` becomes its own
        // row group in file order.
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(1))
            .build();
        let file = std::fs::File::create(&path)?;
        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;
        let min_array: ArrayRef = Arc::new(UInt64Array::from(mins.clone()));
        let bid_array: ArrayRef = Arc::new(UInt32Array::from(bucket_ids));
        let batch = RecordBatch::try_new(schema, vec![min_array, bid_array])?;
        writer.write(&batch)?;
        writer.close()?;

        // Query every real value so every row group's [min, max] (a single
        // value each) overlaps the query set and gets selected as matching.
        let mut query_minimizers = mins.clone();
        query_minimizers.sort_unstable();
        query_minimizers.dedup();

        let result = load_filtered_coo_pairs(&path, 64, &query_minimizers, None);

        assert!(
            result.is_err(),
            "a non-monotonic pair interior to a chunk must be rejected, not silently \
             dropped (num_row_groups={num_row_groups}, num_threads={num_threads})"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("non-monotonic") || err_msg.contains("sorted"),
            "error should describe the sortedness violation: {}",
            err_msg
        );

        Ok(())
    }

    #[test]
    fn test_load_parquet_for_query_large_query_set() -> Result<()> {
        let tmp = TempDir::new()?;
        let path = tmp.path().join("shard.parquet");

        // Create Parquet file with many minimizers
        let minimizers: Vec<u64> = (0..5000).map(|i| i as u64 * 10).collect();
        let inverted = build_test_inverted_index(64, 50, 0xBEEF, vec![(1, "big", minimizers)]);
        inverted.save_shard_parquet(&path, 0, None)?;

        // Query with >1000 minimizers to exercise HashSet code path
        let query_minimizers: Vec<u64> = (0..2000).map(|i| i as u64 * 10).collect();
        let loaded = InvertedIndex::load_shard_parquet_for_query(
            &path,
            inverted.k,
            inverted.w,
            inverted.salt,
            inverted.source_hash,
            &query_minimizers,
            None, // No bloom filter options for tests
        )?;

        // Should find all 2000 queried minimizers
        assert_eq!(loaded.minimizers().len(), 2000);

        // Verify bucket hits
        let hits = loaded.get_bucket_hits(&query_minimizers);
        assert_eq!(hits.get(&1), Some(&2000));

        Ok(())
    }

    #[test]
    fn test_load_parquet_for_query_boundary_conditions() -> Result<()> {
        let tmp = TempDir::new()?;
        let path = tmp.path().join("shard.parquet");

        // Create Parquet file with specific minimizers
        let inverted = build_test_inverted_index(
            64,
            50,
            0,
            vec![
                (1, "A", vec![100, 500, 1000]), // min=100, max=1000
            ],
        );
        inverted.save_shard_parquet(&path, 0, None)?;

        // Test query at exact boundaries
        let query_minimizers = vec![100, 1000]; // exactly min and max
        let loaded = InvertedIndex::load_shard_parquet_for_query(
            &path,
            inverted.k,
            inverted.w,
            inverted.salt,
            inverted.source_hash,
            &query_minimizers,
            None, // No bloom filter options for tests
        )?;

        // Should find both boundary minimizers
        assert_eq!(loaded.minimizers().len(), 2);
        assert!(loaded.minimizers().contains(&100));
        assert!(loaded.minimizers().contains(&1000));

        // Test query just outside boundaries
        let query_outside = vec![99, 1001]; // just outside min and max
        let loaded_outside = InvertedIndex::load_shard_parquet_for_query(
            &path,
            inverted.k,
            inverted.w,
            inverted.salt,
            inverted.source_hash,
            &query_outside,
            None,
        )?;

        // Should find nothing (99 < min, 1001 > max)
        assert_eq!(loaded_outside.minimizers().len(), 0);

        Ok(())
    }

    // ========================================================================
    // Bloom Filter Read Tests
    // ========================================================================

    /// Test that bloom filters correctly handle high-value minimizers (>= 2^63).
    ///
    /// This verifies that `bloom_filter_may_contain_any()` correctly handles u64 values
    /// that would be negative if interpreted as i64. The bloom filter uses raw bytes
    /// via the `AsBytes` trait, so the bit pattern must be preserved.
    #[test]
    fn test_bloom_filter_high_value_minimizers() -> Result<()> {
        let tmp = TempDir::new()?;
        let path = tmp.path().join("bloom_high_values.parquet");

        // Test values that span the u64 range, especially values >= 2^63
        // which become negative when interpreted as i64
        let high_value_minimizers: Vec<u64> = vec![
            0x7FFF_FFFF_FFFF_FFFF, // i64::MAX (largest positive i64)
            0x8000_0000_0000_0000, // 2^63 (i64::MIN when cast)
            0x8000_0000_0000_0001, // 2^63 + 1
            0xFFFF_FFFF_FFFF_0000, // Near u64::MAX
            0xFFFF_FFFF_FFFF_FFFE, // u64::MAX - 1
            0xFFFF_FFFF_FFFF_FFFF, // u64::MAX (becomes -1 as i64)
        ];

        let inverted = build_test_inverted_index(
            64,
            50,
            0xDEADBEEF,
            vec![(1, "high_values", high_value_minimizers.clone())],
        );

        // Save with bloom filter enabled
        let write_opts = ParquetWriteOptions {
            bloom_filter_enabled: true,
            bloom_filter_fpp: 0.01,
            row_group_size: 10, // Small row group to ensure bloom filter is created
            ..Default::default()
        };
        inverted.save_shard_parquet(&path, 0, Some(&write_opts))?;

        // Read with bloom filter enabled
        let read_opts = ParquetReadOptions::with_bloom_filter();

        // Query with ALL high-value minimizers - should find all
        let loaded = InvertedIndex::load_shard_parquet_for_query(
            &path,
            inverted.k,
            inverted.w,
            inverted.salt,
            inverted.source_hash,
            &high_value_minimizers,
            Some(&read_opts),
        )?;

        // Verify all high-value minimizers were found
        for &m in &high_value_minimizers {
            let found = loaded.minimizers().contains(&m);
            assert!(
                found,
                "Bloom filter failed to find high-value minimizer 0x{:016X} (as i64: {})",
                m, m as i64
            );
        }

        Ok(())
    }

    /// Test that bloom filter correctly rejects minimizers that are NOT in the file.
    #[test]
    fn test_bloom_filter_rejects_absent_minimizers() -> Result<()> {
        let tmp = TempDir::new()?;
        let path = tmp.path().join("bloom_reject.parquet");

        // Create file with specific minimizers in a tight range
        let present_minimizers: Vec<u64> = vec![1000, 1001, 1002, 1003, 1004];

        let inverted =
            build_test_inverted_index(64, 50, 0, vec![(1, "test", present_minimizers.clone())]);

        // Save with bloom filter enabled and low FPP for accurate rejection
        let write_opts = ParquetWriteOptions {
            bloom_filter_enabled: true,
            bloom_filter_fpp: 0.001, // Very low FPP
            row_group_size: 100,
            ..Default::default()
        };
        inverted.save_shard_parquet(&path, 0, Some(&write_opts))?;

        let read_opts = ParquetReadOptions::with_bloom_filter();

        // Query with values definitely NOT in the file (outside the min/max range)
        // These should be rejected by stats filtering, not bloom filter
        let outside_range: Vec<u64> = vec![1, 2, 3];
        let loaded_outside = InvertedIndex::load_shard_parquet_for_query(
            &path,
            inverted.k,
            inverted.w,
            inverted.salt,
            inverted.source_hash,
            &outside_range,
            Some(&read_opts),
        )?;

        // Stats filtering rejects these (outside min/max range)
        assert_eq!(loaded_outside.minimizers().len(), 0);

        // Now test values WITHIN the stats range but not actually present
        // These would pass stats filtering but should be caught by bloom filter
        // Note: Due to bloom filter false positives, some may slip through,
        // but with low FPP most should be rejected
        let within_range_but_absent: Vec<u64> = vec![999, 1005, 1006]; // Just outside our values

        let loaded_within = InvertedIndex::load_shard_parquet_for_query(
            &path,
            inverted.k,
            inverted.w,
            inverted.salt,
            inverted.source_hash,
            &within_range_but_absent,
            Some(&read_opts),
        )?;

        // These specific values should not be in the result
        // (the row group might be included due to stats overlap, but the values won't match)
        for &m in &within_range_but_absent {
            assert!(
                !loaded_within.minimizers().contains(&m),
                "Unexpected minimizer {} found",
                m
            );
        }

        Ok(())
    }

    /// Test bloom filter graceful fallback when file has no bloom filters.
    #[test]
    fn test_bloom_filter_graceful_fallback() -> Result<()> {
        let tmp = TempDir::new()?;
        let path = tmp.path().join("no_bloom.parquet");

        let minimizers: Vec<u64> = vec![100, 200, 300, 400, 500];

        let inverted = build_test_inverted_index(64, 50, 0, vec![(1, "test", minimizers.clone())]);

        // Save WITHOUT bloom filter
        let write_opts = ParquetWriteOptions {
            bloom_filter_enabled: false, // No bloom filter
            ..Default::default()
        };
        inverted.save_shard_parquet(&path, 0, Some(&write_opts))?;

        // Read WITH bloom filter option enabled (should gracefully fall back)
        let read_opts = ParquetReadOptions::with_bloom_filter();

        let query = vec![100, 200, 300];
        let loaded = InvertedIndex::load_shard_parquet_for_query(
            &path,
            inverted.k,
            inverted.w,
            inverted.salt,
            inverted.source_hash,
            &query,
            Some(&read_opts),
        )?;

        // Should still find the minimizers (graceful fallback to stats-only)
        for &m in &query {
            assert!(
                loaded.minimizers().contains(&m),
                "Graceful fallback failed: minimizer {} not found",
                m
            );
        }

        Ok(())
    }

    /// Test bloom filter with empty query (edge case).
    #[test]
    fn test_bloom_filter_empty_query() -> Result<()> {
        let tmp = TempDir::new()?;
        let path = tmp.path().join("bloom_empty.parquet");

        let minimizers: Vec<u64> = vec![100, 200, 300];

        let inverted = build_test_inverted_index(64, 50, 0, vec![(1, "test", minimizers)]);

        let write_opts = ParquetWriteOptions {
            bloom_filter_enabled: true,
            ..Default::default()
        };
        inverted.save_shard_parquet(&path, 0, Some(&write_opts))?;

        let read_opts = ParquetReadOptions::with_bloom_filter();

        // Empty query
        let empty_query: Vec<u64> = vec![];
        let loaded = InvertedIndex::load_shard_parquet_for_query(
            &path,
            inverted.k,
            inverted.w,
            inverted.salt,
            inverted.source_hash,
            &empty_query,
            Some(&read_opts),
        )?;

        // Empty query should return empty result
        assert_eq!(loaded.minimizers().len(), 0);

        Ok(())
    }

    /// Test bloom_filter_may_contain_any helper directly with edge cases.
    #[test]
    fn test_bloom_filter_may_contain_any_helper() {
        // Test with None bloom filter (should return true - conservative fallback)
        // When there's no bloom filter, we conservatively include the row group
        assert!(InvertedIndex::bloom_filter_may_contain_any(
            None,
            &[1, 2, 3]
        ));

        // Empty query always returns false - nothing to find regardless of bloom filter
        assert!(!InvertedIndex::bloom_filter_may_contain_any(None, &[]));
    }

    /// Test that bloom filter accepts values that ARE present (no false negatives).
    #[test]
    fn test_bloom_filter_no_false_negatives() -> Result<()> {
        let tmp = TempDir::new()?;
        let path = tmp.path().join("bloom_no_fn.parquet");

        // Create a larger set of minimizers to test bloom filter behavior
        let minimizers: Vec<u64> = (0..1000).map(|i| i * 1000).collect();

        let inverted =
            build_test_inverted_index(64, 50, 0x5555, vec![(1, "test", minimizers.clone())]);

        let write_opts = ParquetWriteOptions {
            bloom_filter_enabled: true,
            bloom_filter_fpp: 0.01,
            row_group_size: 200, // Multiple row groups
            ..Default::default()
        };
        inverted.save_shard_parquet(&path, 0, Some(&write_opts))?;

        let read_opts = ParquetReadOptions::with_bloom_filter();

        // Query with a subset of minimizers that ARE in the file
        let query: Vec<u64> = minimizers.iter().step_by(10).copied().collect();

        let loaded = InvertedIndex::load_shard_parquet_for_query(
            &path,
            inverted.k,
            inverted.w,
            inverted.salt,
            inverted.source_hash,
            &query,
            Some(&read_opts),
        )?;

        // Bloom filters guarantee NO false negatives
        // Every queried value that exists MUST be found
        for &m in &query {
            assert!(
                loaded.minimizers().contains(&m),
                "False negative: minimizer {} not found but should be present",
                m
            );
        }

        Ok(())
    }

    // ========================================================================
    // Tests for load_row_group_pairs and get_row_group_info
    // ========================================================================

    #[test]
    fn test_load_row_group_pairs_basic() -> Result<()> {
        let tmp = TempDir::new()?;
        let path = tmp.path().join("rg_pairs.parquet");

        // Create inverted index with multiple minimizers
        let inverted = build_test_inverted_index(
            64,
            50,
            0xABCD,
            vec![
                (1, "A", vec![100, 200, 300, 400, 500]),
                (2, "B", vec![200, 300, 600]),
            ],
        );
        inverted.save_shard_parquet(&path, 0, None)?;

        // Load pairs from row group 0 with specific query
        let query_minimizers = vec![200, 300, 500];
        let pairs = load_row_group_pairs(&path, 0, &query_minimizers)?;

        // Should find pairs for minimizers 200, 300, 500
        let found_mins: std::collections::HashSet<u64> = pairs.iter().map(|(m, _)| *m).collect();
        assert!(found_mins.contains(&200));
        assert!(found_mins.contains(&300));
        assert!(found_mins.contains(&500));

        // Pairs should be sorted by minimizer
        assert!(pairs.windows(2).all(|w| w[0].0 <= w[1].0));

        Ok(())
    }

    #[test]
    fn test_load_row_group_pairs_range_bounded() -> Result<()> {
        let tmp = TempDir::new()?;
        let path = tmp.path().join("rg_bounded.parquet");

        // Create a small row group with limited range
        let inverted = build_test_inverted_index(
            64,
            50,
            0,
            vec![
                (1, "A", vec![500, 600, 700]), // range [500, 700]
            ],
        );
        inverted.save_shard_parquet(&path, 0, None)?;

        // Query spans wider range than row group
        let query_minimizers = vec![100, 200, 500, 600, 700, 800, 900];
        let pairs = load_row_group_pairs(&path, 0, &query_minimizers)?;

        // Should only find pairs within row group range
        for (m, _) in &pairs {
            assert!(
                *m >= 500 && *m <= 700,
                "Minimizer {} outside expected range",
                m
            );
        }

        Ok(())
    }

    #[test]
    fn test_load_row_group_pairs_no_overlap() -> Result<()> {
        let tmp = TempDir::new()?;
        let path = tmp.path().join("rg_no_overlap.parquet");

        let inverted = build_test_inverted_index(
            64,
            50,
            0,
            vec![
                (1, "A", vec![100, 200, 300]), // range [100, 300]
            ],
        );
        inverted.save_shard_parquet(&path, 0, None)?;

        // Query outside row group range
        let query_minimizers = vec![500, 600, 700];
        let pairs = load_row_group_pairs(&path, 0, &query_minimizers)?;

        // No overlap - should return empty
        assert!(pairs.is_empty());

        Ok(())
    }

    #[test]
    fn test_load_row_group_pairs_empty_query() -> Result<()> {
        let tmp = TempDir::new()?;
        let path = tmp.path().join("rg_empty.parquet");

        let inverted = build_test_inverted_index(64, 50, 0, vec![(1, "A", vec![100, 200, 300])]);
        inverted.save_shard_parquet(&path, 0, None)?;

        // Empty query
        let pairs = load_row_group_pairs(&path, 0, &[])?;
        assert!(pairs.is_empty());

        Ok(())
    }

    #[test]
    fn test_load_row_group_pairs_with_hashset() -> Result<()> {
        let tmp = TempDir::new()?;
        let path = tmp.path().join("rg_hashset.parquet");

        let inverted =
            build_test_inverted_index(64, 50, 0, vec![(1, "A", vec![100, 200, 300, 400, 500])]);
        inverted.save_shard_parquet(&path, 0, None)?;

        let query_minimizers = vec![200, 300];

        let pairs = load_row_group_pairs(&path, 0, &query_minimizers)?;

        // Should find pairs for 200 and 300
        let found_mins: std::collections::HashSet<u64> = pairs.iter().map(|(m, _)| *m).collect();
        assert!(found_mins.contains(&200));
        assert!(found_mins.contains(&300));
        assert!(!found_mins.contains(&100));

        Ok(())
    }

    #[test]
    fn test_get_row_group_count_basic() -> Result<()> {
        let tmp = TempDir::new()?;
        let path = tmp.path().join("rg_count.parquet");

        // Create large index to span multiple row groups
        // Create buckets with different ranges
        let buckets: Vec<(u32, &str, Vec<u64>)> = (0..5u32)
            .map(|bucket_id| {
                let base = bucket_id as u64 * 100_000;
                let minimizers: Vec<u64> = (0..30000).map(|i| base + i as u64).collect();
                let name: &str = match bucket_id {
                    0 => "bucket_0",
                    1 => "bucket_1",
                    2 => "bucket_2",
                    3 => "bucket_3",
                    _ => "bucket_4",
                };
                (bucket_id, name, minimizers)
            })
            .collect();

        let inverted = build_test_inverted_index(64, 50, 0x1234, buckets);

        // Use small row group size to create multiple RGs
        let write_opts = ParquetWriteOptions {
            row_group_size: 50_000,
            ..Default::default()
        };
        inverted.save_shard_parquet(&path, 0, Some(&write_opts))?;

        let rg_ranges = get_row_group_ranges(&path)?;

        // Should have multiple row groups
        assert!(
            rg_ranges.len() > 1,
            "Expected multiple row groups, got {}",
            rg_ranges.len()
        );

        // Each RG should have valid range
        for info in &rg_ranges {
            assert!(
                info.min <= info.max,
                "RG {} has invalid range: min {} > max {}",
                info.rg_idx,
                info.min,
                info.max
            );
        }

        Ok(())
    }

    #[test]
    fn test_load_row_group_pairs_invalid_rg_index() -> Result<()> {
        let tmp = TempDir::new()?;
        let path = tmp.path().join("rg_invalid.parquet");

        let inverted = build_test_inverted_index(64, 50, 0, vec![(1, "A", vec![100, 200])]);
        inverted.save_shard_parquet(&path, 0, None)?;

        // Try to load invalid row group index
        let result = load_row_group_pairs(&path, 999, &[100, 200]);

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("out of range"), "Error: {}", err_msg);

        Ok(())
    }
}
