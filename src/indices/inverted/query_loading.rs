//! Query-aware Parquet loading with bloom filter support.

use crate::error::{Result, RypeError};
use std::path::Path;

use super::InvertedIndex;
use crate::constants::{DEFAULT_ROW_GROUP_SIZE, QUERY_HASHSET_THRESHOLD};

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
        use arrow::array::{Array, UInt32Array, UInt64Array};
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
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

        // Read Parquet footer (metadata) only - not the full file data.
        // SerializedFileReader::new(File) reads just the footer (~few KB) to get metadata.
        // The file handle is explicitly scoped so it's closed before parallel reads begin.
        //
        // When bloom filter is enabled, we keep the reader alive longer to access bloom
        // filters for each row group after statistics filtering.
        let matching_row_groups: Vec<usize> = {
            let file =
                File::open(path).map_err(|e| RypeError::io(path, "open Parquet shard", e))?;

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
            let stats_filtered = Self::find_matching_row_groups(query_minimizers, &rg_stats);

            if stats_filtered.is_empty() {
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
                    if Self::bloom_filter_may_contain_any(bf, relevant_mins) {
                        bloom_filtered.push(rg_idx);
                    } else {
                        bloom_rejected_count += 1;
                    }
                }

                // Warn if bloom filter was requested but file has no bloom filters
                if bloom_filters_found == 0 && !stats_filtered.is_empty() {
                    eprintln!(
                        "Warning: --use-bloom-filter specified but {} has no bloom filters. \
                         Rebuild index with --parquet-bloom-filter to enable bloom filtering.",
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
            } else {
                stats_filtered
            }
        }; // parquet_reader and file handle dropped here

        if matching_row_groups.is_empty() {
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

        // Filtered loading is always beneficial - even loading 90% of row groups
        // still filters individual rows within those groups, reducing memory usage.
        let use_hashset = query_minimizers.len() > QUERY_HASHSET_THRESHOLD;
        let query_set: Option<std::collections::HashSet<u64>> = if use_hashset {
            Some(query_minimizers.iter().copied().collect())
        } else {
            None
        };

        // Parallel read of matching row groups only.
        // Each row group is internally sorted by minimizer (enforced at write time).
        // Results from different row groups may overlap, so we sort after concatenation.
        // Each thread opens its own file handle; OS page cache handles deduplication.
        let path = path.to_path_buf(); // Clone path for parallel closure

        // Estimate pairs per row group for pre-allocation.
        // We expect query selectivity to filter most rows. Conservative estimate: 10%.
        let estimated_pairs_per_rg = DEFAULT_ROW_GROUP_SIZE / 10;

        let row_group_results: Vec<Result<Vec<(u64, u32)>>> = matching_row_groups
            .par_iter()
            .map(|&rg_idx| {
                let file = File::open(&path)?;
                let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
                let reader = builder.with_row_groups(vec![rg_idx]).build()?;

                let mut pairs = Vec::with_capacity(estimated_pairs_per_rg);

                for batch in reader {
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

                    for i in 0..batch.num_rows() {
                        let m = min_col.value(i);
                        // Filter: only include pairs where minimizer is in query set
                        let matches = if let Some(ref hs) = query_set {
                            hs.contains(&m)
                        } else {
                            // Binary search for small query sets
                            query_minimizers.binary_search(&m).is_ok()
                        };
                        if matches {
                            pairs.push((m, bid_col.value(i)));
                        }
                    }
                }

                Ok(pairs)
            })
            .collect();

        // Concatenate results from all row groups, adding context for failures
        let mut all_pairs: Vec<(u64, u32)> = Vec::new();

        for (idx, result) in matching_row_groups.iter().zip(row_group_results) {
            let pairs = result.map_err(|e| {
                RypeError::io(
                    path.clone(),
                    "read row group",
                    std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("row group {}: {}", idx, e),
                    ),
                )
            })?;
            all_pairs.extend(pairs);
        }

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

        // Sort concatenated results to handle overlapping row groups.
        // Multiple row groups may contain the same minimizer if a bucket list spans
        // the row group size boundary during write.
        all_pairs.sort_unstable_by_key(|&(m, _)| m);

        // Build CSR structure
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

        // Note: We intentionally don't call shrink_to_fit() here.
        // The slight memory overhead is preferable to the reallocation cost.

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
    use crate::constants::BOUNDED_QUERY_HASHSET_THRESHOLD;
    use arrow::array::{Array, UInt32Array, UInt64Array};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use parquet::file::reader::FileReader;
    use parquet::file::serialized_reader::SerializedFileReader;
    use parquet::file::statistics::Statistics;
    use std::fs::File;

    if query_minimizers.is_empty() {
        return Ok(Vec::new());
    }

    // Open file and get row group metadata
    let file = File::open(path).map_err(|e| RypeError::io(path, "open Parquet file", e))?;
    let parquet_reader = SerializedFileReader::new(file)?;
    let metadata = parquet_reader.metadata();

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

    // Drop the parquet_reader before doing more work
    drop(parquet_reader);

    // Binary search to find bounded query minimizers
    let q_start = query_minimizers.partition_point(|&m| m < rg_min);
    let q_end = query_minimizers.partition_point(|&m| m <= rg_max);

    // No overlap between query minimizers and this row group's range
    if q_start >= q_end {
        return Ok(Vec::new());
    }

    let bounded_queries = &query_minimizers[q_start..q_end];

    // Build HashSet from bounded queries for O(1) filtering when set is large enough.
    // Always build from bounded_queries (not the full query set) for efficiency.
    let local_set: Option<std::collections::HashSet<u64>> =
        if bounded_queries.len() > BOUNDED_QUERY_HASHSET_THRESHOLD {
            Some(bounded_queries.iter().copied().collect())
        } else {
            None
        };

    // Load the row group using Arrow reader
    let file = File::open(path).map_err(|e| RypeError::io(path, "open Parquet file", e))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.with_row_groups(vec![rg_idx]).build()?;

    // Estimate capacity based on row group size
    let estimated_pairs = rg_num_rows / 10; // Assume ~10% match rate
    let mut pairs = Vec::with_capacity(estimated_pairs);

    for batch in reader {
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

        for i in 0..batch.num_rows() {
            let m = min_col.value(i);

            // Filter: only include pairs where minimizer is in bounded query set
            let matches = if let Some(ref hs) = local_set {
                hs.contains(&m)
            } else {
                // Binary search for small query sets
                bounded_queries.binary_search(&m).is_ok()
            };

            if matches {
                pairs.push((m, bid_col.value(i)));
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
