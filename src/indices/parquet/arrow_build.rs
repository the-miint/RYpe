//! Build a Parquet inverted index from streaming Arrow data.
//!
//! This is the engine behind the C/FFI `rype_index_build_from_arrow` entry point.
//! It consumes chunked genome sequences delivered over Arrow (a DuckDB/DuckLake
//! pipeline that splits each genome into ordered ~64 KB blocks) plus a small
//! feature→bucket mapping, and produces a `.ryxdi` directory.
//!
//! # Input schema
//!
//! **Chunk stream** — one row per ~64 KB block, a feature's chunks delivered
//! contiguously in ascending `chunk_index`:
//!
//! | Column        | Type                              | Description                  |
//! |---------------|-----------------------------------|------------------------------|
//! | `feature_idx` | `Int64`                           | Genome / read identifier     |
//! | `chunk_index` | `Int32`                           | Block order within a feature |
//! | `chunk_data`  | `Binary`/`LargeBinary`/`Utf8`/... | Sequence bytes for the block |
//!
//! **Mapping stream** — small, one row per feature:
//!
//! | Column        | Type    | Description          |
//! |---------------|---------|----------------------|
//! | `feature_idx` | `Int64` | Genome / read id     |
//! | `bucket_name` | `Utf8`  | Target bucket label  |
//!
//! Bucket IDs are assigned internally (1..N over the sorted, sanitized unique
//! bucket names); callers neither supply nor observe them.
//!
//! # Why reassemble
//!
//! Minimizers slide over a `w + k - 1` window, so a window can straddle a chunk
//! boundary. Each feature's chunks are therefore reassembled (in verified
//! `chunk_index` order) into the full sequence before extraction — this is correct
//! across boundaries and is also required for orientation, which is a whole-sequence,
//! per-bucket decision.

use std::path::Path;

use arrow::array::{
    Array, BinaryArray, BinaryViewArray, Int32Array, Int64Array, LargeBinaryArray,
    LargeStringArray, StringArray, StringViewArray,
};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;

use crate::error::{Result as RypeResult, RypeError};
use crate::parquet_index::ParquetWriteOptions;
use crate::{
    choose_orientation_sampled, extract_dual_strand_into, extract_into, MinimizerWorkspace,
    Orientation,
};

/// Column names of the chunk stream.
const COL_FEATURE_IDX: &str = "feature_idx";
const COL_CHUNK_INDEX: &str = "chunk_index";
const COL_CHUNK_DATA: &str = "chunk_data";

/// The first `chunk_index` expected for every feature. Chunks must be contiguous
/// and 0-based; the reassembler fails loud on any deviation.
const FIRST_CHUNK_INDEX: i64 = 0;

/// Summary of a completed Arrow build, returned to callers for reporting/testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArrowBuildStats {
    /// Number of buckets written.
    pub num_buckets: u32,
    /// Number of distinct features processed from the chunk stream.
    pub num_features: u64,
    /// Total minimizers written (may include duplicates across shards).
    pub total_minimizers: u64,
    /// Number of shards written.
    pub num_shards: u32,
}

/// Build a `.ryxdi` index from streaming Arrow chunk + mapping readers.
///
/// `chunks` yields `RecordBatch`es with `feature_idx`/`chunk_index`/`chunk_data`
/// columns; `mapping` yields `feature_idx`/`bucket_name` rows and is consumed in
/// full up front. Both are plain iterators so the builder is testable without the
/// `arrow-ffi` C-stream machinery.
///
/// `max_memory` of `None` auto-detects the available budget.
#[allow(clippy::too_many_arguments)]
pub fn build_index_from_arrow(
    output_dir: &Path,
    chunks: impl Iterator<Item = Result<RecordBatch, ArrowError>>,
    mapping: impl Iterator<Item = Result<RecordBatch, ArrowError>>,
    k: usize,
    w: usize,
    salt: u64,
    orient: bool,
    max_memory: Option<usize>,
    options: Option<&ParquetWriteOptions>,
) -> RypeResult<ArrowBuildStats> {
    use crate::memory::detect_available_memory;

    let available = max_memory.unwrap_or_else(|| detect_available_memory().bytes);
    // Completed features are buffered up to this many bytes, then extracted in
    // parallel. Bounds the transient feature buffer to ~an eighth of the budget.
    let batch_target_bytes = (available / 8).max(8 * 1024 * 1024);
    build_index_from_arrow_inner(
        output_dir,
        chunks,
        mapping,
        k,
        w,
        salt,
        orient,
        available,
        batch_target_bytes,
        options,
    )
}

/// Inner builder with an explicit `batch_target_bytes` seam so tests can force tiny
/// batches (one feature each) and assert batch-size invariance.
#[allow(clippy::too_many_arguments)]
fn build_index_from_arrow_inner(
    output_dir: &Path,
    chunks: impl Iterator<Item = Result<RecordBatch, ArrowError>>,
    mapping: impl Iterator<Item = Result<RecordBatch, ArrowError>>,
    k: usize,
    w: usize,
    salt: u64,
    orient: bool,
    available: usize,
    batch_target_bytes: usize,
    options: Option<&ParquetWriteOptions>,
) -> RypeResult<ArrowBuildStats> {
    use crate::parquet_index::{
        compute_source_hash, consolidate_shards_streaming, create_index_directory,
        write_buckets_parquet, InvertedManifest, ParquetManifest, ParquetShardFormat,
        ShardAccumulator, FORMAT_MAGIC, FORMAT_VERSION, MIN_SHARD_BYTES,
    };
    use crate::BucketFileStats;
    use std::collections::HashMap;

    if !matches!(k, 16 | 32 | 64) {
        return Err(RypeError::validation(format!(
            "k must be 16, 32, or 64; got {k}"
        )));
    }
    // Resolve the whole feature→bucket mapping before touching the chunk stream.
    let mut bmap = BucketMap::from_reader(mapping)?;

    create_index_directory(output_dir)?;

    // Half the memory budget per shard, clamped to the minimum shard size; entries
    // are added in batches of roughly one shard's worth (capped) before flushing.
    let shard_size = (available / 2).max(MIN_SHARD_BYTES);
    let add_batch_entries = (shard_size / ShardAccumulator::BYTES_PER_ENTRY).clamp(1, 8_000_000);

    let opts = options.cloned().unwrap_or_default();
    let mut acc = ShardAccumulator::with_output_dir(output_dir, shard_size, Some(&opts));

    // Single-bucket builds enable the cross-shard dedup filter by default: it keys on
    // the minimizer alone (sound only with exactly one bucket) and suppresses duplicate
    // writes at the source instead of leaving every duplicate for the finalization
    // k-way merge to remove. The budget is a quarter of the memory budget — alongside
    // the accumulator (available/2) and the feature batch (available/8) that keeps the
    // working set within `available`. Once the filter fills it stops growing and the
    // merge stays the exact backstop for whatever it could not hold. Multi-bucket
    // builds skip it (the minimizer-only key would wrongly drop other buckets' entries).
    if bmap.num_buckets() == 1 {
        acc.enable_seen_filter(available / 4);
    }

    // Per-bucket metadata accumulated for the manifest + buckets.parquet.
    let mut bucket_sources: HashMap<u32, Vec<String>> = HashMap::new();
    let mut bucket_min_counts: HashMap<u32, usize> = HashMap::new();
    let mut bucket_file_lengths: HashMap<u32, Vec<u64>> = HashMap::new();
    let mut total_minimizers: u64 = 0;
    let mut num_features: u64 = 0;
    // bucket id → fixed orientation baseline (sorted, deduped minimizers from the
    // bucket's first feature). Dropped as soon as the bucket's last feature is done.
    let mut baselines: HashMap<u32, Vec<u64>> = HashMap::new();

    let mut reasm = FeatureReassembler::new();
    // Completed features buffered for the next parallel extraction batch.
    let mut pending: Vec<(i64, Vec<u8>)> = Vec::new();
    let mut pending_bytes: usize = 0;

    for batch in chunks {
        let batch = batch.map_err(|e| RypeError::validation(format!("chunk stream error: {e}")))?;
        for (fid, seq) in reasm.push_batch(&batch)? {
            pending_bytes += seq.len();
            pending.push((fid, seq));
            if pending_bytes >= batch_target_bytes {
                num_features += pending.len() as u64;
                flush_feature_batch(
                    &mut pending,
                    &mut bmap,
                    &mut baselines,
                    orient,
                    k,
                    w,
                    salt,
                    add_batch_entries,
                    &mut acc,
                    &mut bucket_sources,
                    &mut bucket_min_counts,
                    &mut bucket_file_lengths,
                    &mut total_minimizers,
                )?;
                pending_bytes = 0;
            }
        }
    }
    if let Some((fid, seq)) = reasm.finish() {
        pending.push((fid, seq));
    }
    if !pending.is_empty() {
        num_features += pending.len() as u64;
        flush_feature_batch(
            &mut pending,
            &mut bmap,
            &mut baselines,
            orient,
            k,
            w,
            salt,
            add_batch_entries,
            &mut acc,
            &mut bucket_sources,
            &mut bucket_min_counts,
            &mut bucket_file_lengths,
            &mut total_minimizers,
        )?;
    }

    // Finalize by consolidating the size-thresholded intermediate shards into final,
    // globally-deduplicated, non-overlapping shards — matching the CLI single-bucket
    // streaming build (`commands::index::consolidate_streaming_shards`). The streaming
    // accumulator dedups only within each flush buffer, so a minimizer carried by
    // features that fall in different flush windows is re-stored once per shard. Without
    // this merge a single-bucket index bloats by the average number of shards a minimizer
    // spans — measured ~60x on a real human-reference build. The k-way merge streams
    // pairs (peak memory O(num_shards * batch + shard_size)), so it does not rebuild the
    // corpus-wide footprint the windowed feed exists to avoid.
    // Flush the trailing buffer first so the filter diagnostics include the final
    // window, then read them before `finish()` consumes the accumulator. (`finish()`
    // re-flushes, but the buffer is now empty, so that is a no-op.)
    let verbose = std::env::var("RYPE_BUILD_VERBOSE").is_ok();
    acc.flush_shard()?;
    let filtered_count = acc.filtered_count();
    let seen_len = acc.seen_len();
    let seen_frozen = acc.seen_frozen();

    let intermediate = acc.finish()?;
    if verbose {
        let inter_entries: u64 = intermediate.iter().map(|s| s.num_entries).sum();
        eprintln!(
            "[arrow_build] intermediate shards: {} | entries written: {} | filtered (skipped): {} | seen retained: {} (frozen={})",
            intermediate.len(),
            inter_entries,
            filtered_count,
            seen_len,
            seen_frozen,
        );
    }
    let (shard_infos, total_entries) = if intermediate.len() <= 1 {
        // A lone shard is already sorted + deduped by ShardAccumulator::flush_shard, so
        // it already satisfies the non-overlapping / deduplicated contract.
        let total = intermediate.iter().map(|s| s.num_entries).sum();
        (intermediate, total)
    } else {
        // `bucket_id` is only a log label inside consolidation (the merge preserves each
        // pair's own bucket_id, read from the shard), so 0 is correct even for a
        // multi-bucket build.
        consolidate_shards_streaming(output_dir, &intermediate, 0, shard_size, &opts)?
    };
    let num_shards = shard_infos.len() as u32;
    // After consolidation every (minimizer, bucket_id) pair is stored exactly once, so the
    // physical entry count is the true minimizer total — the CLI reports it the same way
    // (commands::index sets the manifest's total_minimizers from the consolidated count).
    total_minimizers = total_entries;
    if verbose {
        eprintln!("[arrow_build] final shards: {num_shards} | unique minimizers: {total_entries}",);
    }

    let file_stats: HashMap<u32, BucketFileStats> = bucket_file_lengths
        .iter()
        .filter_map(|(id, lens)| BucketFileStats::from_file_lengths(lens).map(|s| (*id, s)))
        .collect();
    let file_stats_opt = if file_stats.is_empty() {
        None
    } else {
        Some(&file_stats)
    };
    write_buckets_parquet(
        output_dir,
        &bmap.bucket_names,
        &bucket_sources,
        file_stats_opt,
    )?;

    let source_hash = compute_source_hash(&bucket_min_counts);
    let num_buckets = bmap.num_buckets();

    let manifest = ParquetManifest {
        magic: FORMAT_MAGIC.to_string(),
        format_version: FORMAT_VERSION,
        k,
        w,
        salt,
        source_hash,
        num_buckets,
        total_minimizers,
        inverted: Some(InvertedManifest {
            format: ParquetShardFormat::Parquet,
            num_shards,
            total_entries,
            has_overlapping_shards: false,
            shards: shard_infos,
        }),
    };
    manifest.save(output_dir)?;

    Ok(ArrowBuildStats {
        num_buckets,
        num_features,
        total_minimizers,
        num_shards,
    })
}

/// Record one feature's chosen-strand minimizers: update per-bucket metadata and
/// stream the (minimizer, bucket_id) entries into the accumulator.
#[allow(clippy::too_many_arguments)]
fn write_feature_minimizers(
    bucket_id: u32,
    fid: i64,
    seq_len: usize,
    mins: &[u64],
    add_batch_entries: usize,
    acc: &mut crate::parquet_index::ShardAccumulator,
    bucket_sources: &mut std::collections::HashMap<u32, Vec<String>>,
    bucket_min_counts: &mut std::collections::HashMap<u32, usize>,
    bucket_file_lengths: &mut std::collections::HashMap<u32, Vec<u64>>,
    total_minimizers: &mut u64,
) -> RypeResult<()> {
    *total_minimizers += mins.len() as u64;
    *bucket_min_counts.entry(bucket_id).or_insert(0) += mins.len();
    // Source label uses the established two-part `path::seqname` form (BUCKET_SOURCE_DELIM):
    // there is no file in this data model, so the path-part is the literal "feature_idx"
    // and the seq-part is the integer id, e.g. "feature_idx::12345". This keeps
    // `index bucket-source-detail` (which splits on the delimiter) well-formed.
    bucket_sources.entry(bucket_id).or_default().push(format!(
        "feature_idx{}{}",
        crate::BUCKET_SOURCE_DELIM,
        fid
    ));
    bucket_file_lengths
        .entry(bucket_id)
        .or_default()
        .push(seq_len as u64);

    for chunk in mins.chunks(add_batch_entries) {
        acc.add_entries_from_minimizers(chunk, bucket_id);
        while acc.should_flush() {
            acc.flush_shard()?;
        }
    }
    Ok(())
}

/// Extract a byte-bounded batch of completed features in parallel and stream their
/// minimizers into the accumulator, draining `pending`.
///
/// The minimizer SET and per-feature orientation decisions are independent of how
/// features are grouped into batches: a bucket's baseline is always seeded by its
/// globally-first feature (in stream order), and orientation depends only on that
/// fixed baseline. Only minimizer *extraction* is parallel; accumulator writes and
/// completion bookkeeping stay serial (the accumulator is not thread-safe).
#[allow(clippy::too_many_arguments)]
fn flush_feature_batch(
    pending: &mut Vec<(i64, Vec<u8>)>,
    bmap: &mut BucketMap,
    baselines: &mut std::collections::HashMap<u32, Vec<u64>>,
    orient: bool,
    k: usize,
    w: usize,
    salt: u64,
    add_batch_entries: usize,
    acc: &mut crate::parquet_index::ShardAccumulator,
    bucket_sources: &mut std::collections::HashMap<u32, Vec<String>>,
    bucket_min_counts: &mut std::collections::HashMap<u32, usize>,
    bucket_file_lengths: &mut std::collections::HashMap<u32, Vec<u64>>,
    total_minimizers: &mut u64,
) -> RypeResult<()> {
    use rayon::prelude::*;
    use std::collections::HashSet;

    if pending.is_empty() {
        return Ok(());
    }

    // 1. Resolve each feature's bucket (serial; surfaces unmapped-feature errors). In
    //    orient mode, the first feature of a bucket lacking a baseline is a SEED; every
    //    other feature orients against a baseline guaranteed to exist by the orient pass.
    let mut seeds: Vec<(i64, u32, Vec<u8>)> = Vec::new();
    let mut others: Vec<(i64, u32, Vec<u8>)> = Vec::new();
    let mut seeded_this_batch: HashSet<u32> = HashSet::new();
    for (fid, seq) in pending.drain(..) {
        let bid = bmap.bucket_of(fid)?;
        if orient && !baselines.contains_key(&bid) && !seeded_this_batch.contains(&bid) {
            seeded_this_batch.insert(bid);
            seeds.push((fid, bid, seq));
        } else {
            others.push((fid, bid, seq));
        }
    }

    // 2. Extract seed baselines in parallel (distinct, independent buckets), then
    //    install each and write its forward minimizers.
    let seed_mins: Vec<(i64, u32, usize, Vec<u64>)> = seeds
        .par_iter()
        .map_init(MinimizerWorkspace::new, |ws, (fid, bid, seq)| {
            extract_into(seq, k, w, salt, ws);
            ws.buffer.sort_unstable();
            ws.buffer.dedup();
            (*fid, *bid, seq.len(), ws.buffer.clone())
        })
        .collect();
    let mut batch_bids: Vec<u32> = Vec::with_capacity(seed_mins.len() + others.len());
    for (fid, bid, len, mins) in seed_mins {
        baselines.insert(bid, mins);
        write_feature_minimizers(
            bid,
            fid,
            len,
            &baselines[&bid],
            add_batch_entries,
            acc,
            bucket_sources,
            bucket_min_counts,
            bucket_file_lengths,
            total_minimizers,
        )?;
        batch_bids.push(bid);
    }

    // 3. Extract the remaining features in parallel. For orient, decide the strand
    //    against the now-complete baselines (read-only); otherwise take forward.
    let baselines_ref: &std::collections::HashMap<u32, Vec<u64>> = baselines;
    let other_mins: Vec<(i64, u32, usize, Vec<u64>)> = others
        .par_iter()
        .map_init(MinimizerWorkspace::new, |ws, (fid, bid, seq)| {
            if orient {
                let (mut fwd, mut rc) = extract_dual_strand_into(seq, k, w, salt, ws);
                fwd.sort_unstable();
                rc.sort_unstable();
                let baseline = baselines_ref
                    .get(bid)
                    .expect("baseline is seeded before the orient pass");
                let (orientation, _overlap) = choose_orientation_sampled(baseline, &fwd, &rc);
                let mut chosen = match orientation {
                    Orientation::Forward => fwd,
                    Orientation::ReverseComplement => rc,
                };
                chosen.dedup();
                (*fid, *bid, seq.len(), chosen)
            } else {
                extract_into(seq, k, w, salt, ws);
                ws.buffer.sort_unstable();
                ws.buffer.dedup();
                (*fid, *bid, seq.len(), ws.buffer.clone())
            }
        })
        .collect();
    for (fid, bid, len, mins) in &other_mins {
        write_feature_minimizers(
            *bid,
            *fid,
            *len,
            mins,
            add_batch_entries,
            acc,
            bucket_sources,
            bucket_min_counts,
            bucket_file_lengths,
            total_minimizers,
        )?;
        batch_bids.push(*bid);
    }

    // 4. Mark each processed feature done; free a bucket's baseline once complete.
    //    Extraction has finished, so freeing a completed bucket here is safe.
    for bid in batch_bids {
        if bmap.mark_feature_done(bid) {
            baselines.remove(&bid);
        }
    }
    Ok(())
}

/// Uniform read access to a binary/string Arrow column. Bytes are copied into the
/// reassembly buffer, so there is no zero-copy lifetime to preserve here.
enum BytesColumn<'a> {
    Binary(&'a BinaryArray),
    LargeBinary(&'a LargeBinaryArray),
    String(&'a StringArray),
    LargeString(&'a LargeStringArray),
    BinaryView(&'a BinaryViewArray),
    StringView(&'a StringViewArray),
}

impl<'a> BytesColumn<'a> {
    fn from_column(batch: &'a RecordBatch, idx: usize, name: &str) -> RypeResult<Self> {
        let column = batch.column(idx);
        if let Some(a) = column.as_any().downcast_ref::<BinaryArray>() {
            Ok(BytesColumn::Binary(a))
        } else if let Some(a) = column.as_any().downcast_ref::<LargeBinaryArray>() {
            Ok(BytesColumn::LargeBinary(a))
        } else if let Some(a) = column.as_any().downcast_ref::<StringArray>() {
            Ok(BytesColumn::String(a))
        } else if let Some(a) = column.as_any().downcast_ref::<LargeStringArray>() {
            Ok(BytesColumn::LargeString(a))
        } else if let Some(a) = column.as_any().downcast_ref::<BinaryViewArray>() {
            Ok(BytesColumn::BinaryView(a))
        } else if let Some(a) = column.as_any().downcast_ref::<StringViewArray>() {
            Ok(BytesColumn::StringView(a))
        } else {
            Err(RypeError::validation(format!(
                "column '{}' must be Binary/LargeBinary/Utf8/LargeUtf8/BinaryView/Utf8View, got {:?}",
                name,
                column.data_type()
            )))
        }
    }

    #[inline]
    fn is_null(&self, i: usize) -> bool {
        match self {
            BytesColumn::Binary(a) => a.is_null(i),
            BytesColumn::LargeBinary(a) => a.is_null(i),
            BytesColumn::String(a) => a.is_null(i),
            BytesColumn::LargeString(a) => a.is_null(i),
            BytesColumn::BinaryView(a) => a.is_null(i),
            BytesColumn::StringView(a) => a.is_null(i),
        }
    }

    #[inline]
    fn value(&self, i: usize) -> &[u8] {
        match self {
            BytesColumn::Binary(a) => a.value(i),
            BytesColumn::LargeBinary(a) => a.value(i),
            BytesColumn::String(a) => a.value(i).as_bytes(),
            BytesColumn::LargeString(a) => a.value(i).as_bytes(),
            BytesColumn::BinaryView(a) => a.value(i),
            BytesColumn::StringView(a) => a.value(i).as_bytes(),
        }
    }
}

/// A `chunk_index` column, normalized to `i64` regardless of Int32/Int64 width.
enum IndexColumn<'a> {
    I32(&'a Int32Array),
    I64(&'a Int64Array),
}

impl<'a> IndexColumn<'a> {
    fn from_column(batch: &'a RecordBatch, idx: usize, name: &str) -> RypeResult<Self> {
        let column = batch.column(idx);
        if let Some(a) = column.as_any().downcast_ref::<Int32Array>() {
            Ok(IndexColumn::I32(a))
        } else if let Some(a) = column.as_any().downcast_ref::<Int64Array>() {
            Ok(IndexColumn::I64(a))
        } else {
            Err(RypeError::validation(format!(
                "column '{}' must be Int32 or Int64, got {:?}",
                name,
                column.data_type()
            )))
        }
    }

    #[inline]
    fn is_null(&self, i: usize) -> bool {
        match self {
            IndexColumn::I32(a) => a.is_null(i),
            IndexColumn::I64(a) => a.is_null(i),
        }
    }

    #[inline]
    fn value(&self, i: usize) -> i64 {
        match self {
            IndexColumn::I32(a) => a.value(i) as i64,
            IndexColumn::I64(a) => a.value(i),
        }
    }
}

/// Look up a required column index by name.
fn column_index(batch: &RecordBatch, name: &str) -> RypeResult<usize> {
    batch.schema().index_of(name).map_err(|_| {
        RypeError::validation(format!("chunk batch missing required column '{}'", name))
    })
}

/// Decode the three chunk columns from a batch into typed accessors.
fn decode_chunk_batch<'a>(
    batch: &'a RecordBatch,
) -> RypeResult<(&'a Int64Array, IndexColumn<'a>, BytesColumn<'a>)> {
    let f_idx = column_index(batch, COL_FEATURE_IDX)?;
    let c_idx = column_index(batch, COL_CHUNK_INDEX)?;
    let d_idx = column_index(batch, COL_CHUNK_DATA)?;

    let feature = batch
        .column(f_idx)
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| {
            RypeError::validation(format!(
                "column '{}' must be Int64, got {:?}",
                COL_FEATURE_IDX,
                batch.column(f_idx).data_type()
            ))
        })?;
    let chunk_index = IndexColumn::from_column(batch, c_idx, COL_CHUNK_INDEX)?;
    let chunk_data = BytesColumn::from_column(batch, d_idx, COL_CHUNK_DATA)?;
    Ok((feature, chunk_index, chunk_data))
}

// ============================================================================
// Per-feature reassembly
// ============================================================================

/// Reassembles complete per-feature sequences from ordered chunk rows.
///
/// Chunks for a feature are expected to arrive contiguously, 0-based, in ascending
/// `chunk_index`. Any gap, duplicate, regression, or a feature reappearing after it
/// has been closed is a hard error (Rule 10): silently mis-ordered chunks would
/// corrupt the genome and the resulting minimizers.
struct FeatureReassembler {
    current_feature: Option<i64>,
    /// Next `chunk_index` expected for the in-progress feature.
    expected_next: i64,
    /// Accumulated bytes for the in-progress feature.
    buffer: Vec<u8>,
    /// Features already emitted, to detect illegal reappearance.
    closed: std::collections::HashSet<i64>,
}

impl FeatureReassembler {
    fn new() -> Self {
        Self {
            current_feature: None,
            expected_next: FIRST_CHUNK_INDEX,
            buffer: Vec::new(),
            closed: std::collections::HashSet::new(),
        }
    }

    /// Feed one decoded chunk batch, returning every feature that completed within it
    /// (in stream order). The trailing in-progress feature is held until the next
    /// batch or [`finish`](Self::finish).
    fn push_batch(&mut self, batch: &RecordBatch) -> RypeResult<Vec<(i64, Vec<u8>)>> {
        let (feature, chunk_index, chunk_data) = decode_chunk_batch(batch)?;
        let mut completed = Vec::new();
        for i in 0..batch.num_rows() {
            if feature.is_null(i) {
                return Err(RypeError::validation(format!(
                    "null '{}' at chunk row {}",
                    COL_FEATURE_IDX, i
                )));
            }
            if chunk_index.is_null(i) {
                return Err(RypeError::validation(format!(
                    "null '{}' at chunk row {}",
                    COL_CHUNK_INDEX, i
                )));
            }
            if chunk_data.is_null(i) {
                return Err(RypeError::validation(format!(
                    "null '{}' at chunk row {}",
                    COL_CHUNK_DATA, i
                )));
            }
            let fid = feature.value(i);
            let cidx = chunk_index.value(i);
            let data = chunk_data.value(i);

            if self.current_feature == Some(fid) {
                // Continuation of the in-progress feature.
                if cidx != self.expected_next {
                    return Err(RypeError::validation(format!(
                        "feature {}: expected chunk_index {} but got {} (chunks must be \
                         contiguous and ascending)",
                        fid, self.expected_next, cidx
                    )));
                }
                self.buffer.extend_from_slice(data);
                self.expected_next += 1;
            } else {
                // Feature boundary: close the previous, open this one.
                if let Some(done) = self.close_current() {
                    completed.push(done);
                }
                if !self.closed.insert(fid) {
                    return Err(RypeError::validation(format!(
                        "feature {} reappeared after its chunks were already closed; a \
                         feature's chunks must be delivered contiguously",
                        fid
                    )));
                }
                if cidx != FIRST_CHUNK_INDEX {
                    return Err(RypeError::validation(format!(
                        "feature {}: first chunk_index must be {} but got {}",
                        fid, FIRST_CHUNK_INDEX, cidx
                    )));
                }
                // `close_current()` above already emptied `buffer` via mem::take
                // (and on the very first feature it starts empty), so no clear needed.
                self.current_feature = Some(fid);
                self.buffer.extend_from_slice(data);
                self.expected_next = FIRST_CHUNK_INDEX + 1;
            }
        }
        Ok(completed)
    }

    /// Emit the trailing in-progress feature, if any.
    fn finish(&mut self) -> Option<(i64, Vec<u8>)> {
        self.close_current()
    }

    /// Take the in-progress feature's bytes and reset for the next one.
    fn close_current(&mut self) -> Option<(i64, Vec<u8>)> {
        let fid = self.current_feature.take()?;
        let seq = std::mem::take(&mut self.buffer);
        self.expected_next = FIRST_CHUNK_INDEX;
        Some((fid, seq))
    }
}

// ============================================================================
// Feature → bucket mapping
// ============================================================================

/// Column name of the bucket label in the mapping stream.
const MAP_COL_BUCKET_NAME: &str = "bucket_name";

/// Replace control / non-graphic characters with `_` so bucket names are safe in
/// file paths and metadata. Mirrors `commands::helpers::sanitize_bucket_name`,
/// which lives in the binary and is unreachable from the library.
fn sanitize_bucket_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_control() || (!c.is_ascii_graphic() && !c.is_whitespace()) {
                '_'
            } else {
                c
            }
        })
        .collect()
}

/// The fully-resolved feature→bucket mapping, consumed up front from the mapping
/// stream. Holding it whole lets us assign deterministic bucket IDs and track
/// per-bucket completion (so a bucket's orientation baseline can be freed as soon
/// as its last feature is processed).
#[derive(Debug)]
struct BucketMap {
    /// feature_idx → assigned bucket id.
    feature_to_bucket: std::collections::HashMap<i64, u32>,
    /// bucket id → sanitized bucket name.
    bucket_names: std::collections::HashMap<u32, String>,
    /// bucket id → number of features not yet processed.
    remaining: std::collections::HashMap<u32, u64>,
}

impl BucketMap {
    /// Read the entire mapping stream (`feature_idx: Int64`, `bucket_name: Utf8`)
    /// and assign bucket ids `1..=N` over the sorted unique sanitized names.
    fn from_reader(
        mapping: impl Iterator<Item = Result<RecordBatch, ArrowError>>,
    ) -> RypeResult<Self> {
        use std::collections::HashMap;

        // Pass 1: feature_idx → sanitized bucket name (rejecting duplicate features).
        let mut feature_to_name: HashMap<i64, String> = HashMap::new();
        // Sanitized name → the raw name that produced it, to fail loud when two
        // *distinct* raw names collapse to the same sanitized bucket (matches the
        // CLI's validate_unique_bucket_names; silent merging would lose a bucket).
        let mut sanitized_from: HashMap<String, String> = HashMap::new();
        for batch in mapping {
            let batch =
                batch.map_err(|e| RypeError::validation(format!("mapping stream error: {e}")))?;
            let f_idx = column_index(&batch, COL_FEATURE_IDX)?;
            let n_idx = column_index(&batch, MAP_COL_BUCKET_NAME)?;
            let features = batch
                .column(f_idx)
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| {
                    RypeError::validation(format!(
                        "mapping column '{}' must be Int64, got {:?}",
                        COL_FEATURE_IDX,
                        batch.column(f_idx).data_type()
                    ))
                })?;
            let names = BytesColumn::from_column(&batch, n_idx, MAP_COL_BUCKET_NAME)?;
            for i in 0..batch.num_rows() {
                if features.is_null(i) {
                    return Err(RypeError::validation(format!(
                        "null '{}' at mapping row {}",
                        COL_FEATURE_IDX, i
                    )));
                }
                if names.is_null(i) {
                    return Err(RypeError::validation(format!(
                        "null '{}' at mapping row {}",
                        MAP_COL_BUCKET_NAME, i
                    )));
                }
                let fid = features.value(i);
                let raw = String::from_utf8_lossy(names.value(i)).into_owned();
                let name = sanitize_bucket_name(&raw);
                match sanitized_from.get(&name) {
                    Some(prev) if *prev != raw => {
                        return Err(RypeError::validation(format!(
                            "bucket names {:?} and {:?} both sanitize to {:?}; bucket names \
                             must be distinct after sanitization",
                            prev, raw, name
                        )));
                    }
                    None => {
                        sanitized_from.insert(name.clone(), raw);
                    }
                    _ => {}
                }
                if feature_to_name.insert(fid, name).is_some() {
                    return Err(RypeError::validation(format!(
                        "feature {} appears multiple times in the bucket mapping; each \
                         feature must map to exactly one bucket",
                        fid
                    )));
                }
            }
        }

        // Assign bucket ids 1..=N over the sorted unique names (deterministic).
        let mut unique: Vec<&String> = feature_to_name.values().collect();
        unique.sort_unstable();
        unique.dedup();
        let mut name_to_id: HashMap<&str, u32> = HashMap::with_capacity(unique.len());
        let mut bucket_names: HashMap<u32, String> = HashMap::with_capacity(unique.len());
        for (i, name) in unique.iter().enumerate() {
            let id = (i + 1) as u32;
            name_to_id.insert(name.as_str(), id);
            bucket_names.insert(id, (*name).clone());
        }

        // Build feature→bucket and per-bucket remaining counts.
        let mut feature_to_bucket: HashMap<i64, u32> =
            HashMap::with_capacity(feature_to_name.len());
        let mut remaining: HashMap<u32, u64> = HashMap::with_capacity(bucket_names.len());
        for (fid, name) in &feature_to_name {
            let id = name_to_id[name.as_str()];
            feature_to_bucket.insert(*fid, id);
            *remaining.entry(id).or_insert(0) += 1;
        }

        Ok(Self {
            feature_to_bucket,
            bucket_names,
            remaining,
        })
    }

    fn num_buckets(&self) -> u32 {
        self.bucket_names.len() as u32
    }

    /// Bucket id for a feature; errors if the feature has no mapping entry.
    fn bucket_of(&self, feature_idx: i64) -> RypeResult<u32> {
        self.feature_to_bucket
            .get(&feature_idx)
            .copied()
            .ok_or_else(|| {
                RypeError::validation(format!(
                    "feature {} in the chunk stream has no bucket mapping",
                    feature_idx
                ))
            })
    }

    /// Decrement a bucket's remaining-feature count; returns `true` when it reaches
    /// zero (the bucket is now complete).
    fn mark_feature_done(&mut self, bucket_id: u32) -> bool {
        match self.remaining.get_mut(&bucket_id) {
            Some(r) => {
                *r = r.saturating_sub(1);
                *r == 0
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{ArrayRef, BinaryArray, Int32Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    /// Build a chunk RecordBatch from (feature_idx, chunk_index, chunk_data) triples.
    fn chunk_batch(rows: &[(i64, i32, &[u8])]) -> RecordBatch {
        let features: Int64Array = rows.iter().map(|(f, _, _)| *f).collect();
        let indices: Int32Array = rows.iter().map(|(_, c, _)| *c).collect();
        let data: BinaryArray =
            BinaryArray::from(rows.iter().map(|(_, _, d)| *d).collect::<Vec<_>>());
        let schema = Schema::new(vec![
            Field::new(COL_FEATURE_IDX, DataType::Int64, false),
            Field::new(COL_CHUNK_INDEX, DataType::Int32, false),
            Field::new(COL_CHUNK_DATA, DataType::Binary, false),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(features) as ArrayRef,
                Arc::new(indices) as ArrayRef,
                Arc::new(data) as ArrayRef,
            ],
        )
        .unwrap()
    }

    /// Drain a reassembler across a list of batches into completed features.
    fn reassemble(batches: &[RecordBatch]) -> RypeResult<Vec<(i64, Vec<u8>)>> {
        let mut r = FeatureReassembler::new();
        let mut out = Vec::new();
        for b in batches {
            out.extend(r.push_batch(b)?);
        }
        out.extend(r.finish());
        Ok(out)
    }

    #[test]
    fn reassembles_chunks_in_order() {
        let batch = chunk_batch(&[
            (1, 0, b"ACGT"),
            (1, 1, b"TTGG"),
            (1, 2, b"CC"),
            (2, 0, b"GGGG"),
        ]);
        let out = reassemble(&[batch]).unwrap();
        assert_eq!(
            out,
            vec![(1, b"ACGTTTGGCC".to_vec()), (2, b"GGGG".to_vec())]
        );
    }

    #[test]
    fn reassembles_feature_split_across_batches() {
        let b1 = chunk_batch(&[(7, 0, b"AAAA"), (7, 1, b"CCCC")]);
        let b2 = chunk_batch(&[(7, 2, b"GGGG"), (8, 0, b"TTTT")]);
        let out = reassemble(&[b1, b2]).unwrap();
        assert_eq!(
            out,
            vec![(7, b"AAAACCCCGGGG".to_vec()), (8, b"TTTT".to_vec())]
        );
    }

    #[test]
    fn rejects_gap_in_chunk_index() {
        // 0 then 2: missing chunk 1.
        let batch = chunk_batch(&[(1, 0, b"AC"), (1, 2, b"GT")]);
        let err = reassemble(&[batch]).unwrap_err();
        assert!(err.to_string().contains("expected chunk_index 1"), "{err}");
    }

    #[test]
    fn rejects_duplicate_chunk_index() {
        let batch = chunk_batch(&[(1, 0, b"AC"), (1, 0, b"GT")]);
        let err = reassemble(&[batch]).unwrap_err();
        assert!(err.to_string().contains("expected chunk_index 1"), "{err}");
    }

    #[test]
    fn rejects_nonzero_first_chunk() {
        let batch = chunk_batch(&[(1, 1, b"AC")]);
        let err = reassemble(&[batch]).unwrap_err();
        assert!(
            err.to_string().contains("first chunk_index must be 0"),
            "{err}"
        );
    }

    #[test]
    fn rejects_feature_reappearing_after_close() {
        // feature 1, then 2, then 1 again.
        let batch = chunk_batch(&[(1, 0, b"AC"), (2, 0, b"GG"), (1, 0, b"TT")]);
        let err = reassemble(&[batch]).unwrap_err();
        assert!(err.to_string().contains("reappeared"), "{err}");
    }

    #[test]
    fn missing_column_errors() {
        let schema = Schema::new(vec![Field::new(COL_FEATURE_IDX, DataType::Int64, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(Int64Array::from(vec![1i64])) as ArrayRef],
        )
        .unwrap();
        let mut r = FeatureReassembler::new();
        let err = r.push_batch(&batch).unwrap_err();
        assert!(err.to_string().contains("missing required column"), "{err}");
    }

    // ===== Phase 2: BucketMap =====

    /// Build a mapping RecordBatch from (feature_idx, bucket_name) rows.
    fn mapping_batch(rows: &[(i64, &str)]) -> RecordBatch {
        let features: Int64Array = rows.iter().map(|(f, _)| *f).collect();
        let names = StringArray::from(rows.iter().map(|(_, n)| *n).collect::<Vec<&str>>());
        let schema = Schema::new(vec![
            Field::new(COL_FEATURE_IDX, DataType::Int64, false),
            Field::new(MAP_COL_BUCKET_NAME, DataType::Utf8, false),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(features) as ArrayRef, Arc::new(names) as ArrayRef],
        )
        .unwrap()
    }

    fn bucket_map(batches: Vec<RecordBatch>) -> RypeResult<BucketMap> {
        BucketMap::from_reader(batches.into_iter().map(Ok))
    }

    #[test]
    fn assigns_deterministic_ids_over_sorted_names() {
        // Two distinct names; "alpha" sorts before "zeta" → alpha=1, zeta=2.
        let m = bucket_map(vec![mapping_batch(&[
            (10, "zeta"),
            (20, "alpha"),
            (30, "alpha"),
        ])])
        .unwrap();
        assert_eq!(m.num_buckets(), 2);
        let alpha = m.bucket_of(20).unwrap();
        let zeta = m.bucket_of(10).unwrap();
        assert_eq!(alpha, 1);
        assert_eq!(zeta, 2);
        // Two features share "alpha".
        assert_eq!(m.bucket_of(30).unwrap(), alpha);
        assert_eq!(m.bucket_names[&alpha], "alpha");
        assert_eq!(m.bucket_names[&zeta], "zeta");
    }

    #[test]
    fn mapping_spans_multiple_batches() {
        let m = bucket_map(vec![
            mapping_batch(&[(1, "b")]),
            mapping_batch(&[(2, "a"), (3, "b")]),
        ])
        .unwrap();
        assert_eq!(m.num_buckets(), 2);
        // a=1, b=2 (sorted). features 1 and 3 share b.
        assert_eq!(m.bucket_of(2).unwrap(), 1);
        assert_eq!(m.bucket_of(1).unwrap(), m.bucket_of(3).unwrap());
    }

    #[test]
    fn unmapped_feature_errors() {
        let m = bucket_map(vec![mapping_batch(&[(1, "a")])]).unwrap();
        let err = m.bucket_of(99).unwrap_err();
        assert!(err.to_string().contains("no bucket mapping"), "{err}");
    }

    #[test]
    fn duplicate_feature_in_mapping_errors() {
        let err = bucket_map(vec![mapping_batch(&[(1, "a"), (1, "b")])]).unwrap_err();
        assert!(err.to_string().contains("multiple times"), "{err}");
    }

    #[test]
    fn completion_counter_signals_on_last_feature() {
        let mut m = bucket_map(vec![mapping_batch(&[(10, "a"), (20, "a"), (30, "b")])]).unwrap();
        let a = m.bucket_of(10).unwrap();
        let b = m.bucket_of(30).unwrap();
        // bucket a has 2 features.
        assert!(!m.mark_feature_done(a), "first of two should not complete");
        assert!(m.mark_feature_done(a), "second of two should complete");
        // bucket b has 1 feature.
        assert!(
            m.mark_feature_done(b),
            "single feature completes immediately"
        );
    }

    #[test]
    fn sanitizes_bucket_names() {
        let m = bucket_map(vec![mapping_batch(&[(1, "weird\tname")])]).unwrap();
        let id = m.bucket_of(1).unwrap();
        assert_eq!(m.bucket_names[&id], "weird_name");
    }

    #[test]
    fn distinct_names_colliding_after_sanitize_error() {
        // "n\u{1}" and "n\u{2}" both sanitize to "n_": distinct buckets that would
        // silently merge. Must fail loud (CLI parity).
        let err = bucket_map(vec![mapping_batch(&[(1, "n\u{1}"), (2, "n\u{2}")])]).unwrap_err();
        assert!(err.to_string().contains("sanitize"), "{err}");
    }

    #[test]
    fn null_mapping_values_error() {
        // Null bucket_name in the mapping must fail loud.
        let features = Int64Array::from(vec![1i64]);
        let names = StringArray::from(vec![None as Option<&str>]);
        let schema = Schema::new(vec![
            Field::new(COL_FEATURE_IDX, DataType::Int64, false),
            Field::new(MAP_COL_BUCKET_NAME, DataType::Utf8, true),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(features) as ArrayRef, Arc::new(names) as ArrayRef],
        )
        .unwrap();
        let err = BucketMap::from_reader(std::iter::once(Ok(batch))).unwrap_err();
        assert!(err.to_string().contains("null"), "{err}");
    }

    // ===== Phase 3: end-to-end build (orient off) =====

    use crate::parquet_index::merge::load_all_minimizers;
    use crate::{extract_into, MinimizerWorkspace, ShardedInvertedIndex};
    use std::collections::HashSet;

    /// Deterministic pseudo-DNA of length `n` (ACGT only), varied by `seed`.
    fn make_dna(n: usize, seed: usize) -> Vec<u8> {
        let alpha = b"ACGT";
        (0..n)
            .map(|i| alpha[(i * 7 + i / 3 + seed * 3 + (i % 5)) % 4])
            .collect()
    }

    /// One batch holding all of `seq`'s chunks for `feature`, split every `chunk_len`.
    fn single_feature_chunk_batch(feature: i64, seq: &[u8], chunk_len: usize) -> RecordBatch {
        let rows: Vec<(i64, i32, &[u8])> = seq
            .chunks(chunk_len)
            .enumerate()
            .map(|(i, p)| (feature, i as i32, p))
            .collect();
        chunk_batch(&rows)
    }

    fn reference_minimizers(seq: &[u8], k: usize, w: usize, salt: u64) -> HashSet<u64> {
        let mut ws = MinimizerWorkspace::new();
        extract_into(seq, k, w, salt, &mut ws);
        ws.buffer.iter().copied().collect()
    }

    /// Reverse-complement of an ACGT sequence.
    fn revcomp(seq: &[u8]) -> Vec<u8> {
        seq.iter()
            .rev()
            .map(|&b| match b {
                b'A' => b'T',
                b'T' => b'A',
                b'C' => b'G',
                b'G' => b'C',
                other => other,
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        out: &std::path::Path,
        chunks: Vec<RecordBatch>,
        mapping: Vec<RecordBatch>,
        k: usize,
        w: usize,
        salt: u64,
        orient: bool,
    ) -> RypeResult<ArrowBuildStats> {
        build_index_from_arrow(
            out,
            chunks.into_iter().map(Ok),
            mapping.into_iter().map(Ok),
            k,
            w,
            salt,
            orient,
            Some(64 << 20),
            None,
        )
    }

    #[test]
    fn arrow_build_minimizers_match_unchunked_reference() {
        // The crux: a chunked build must reproduce the minimizers of the un-chunked
        // sequence exactly — proving reassembly loses nothing across chunk boundaries.
        let seq = make_dna(512, 1);
        let (k, w, salt) = (16usize, 5usize, 0x5555_5555_5555_5555u64);
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("idx.ryxdi");

        let chunks = vec![single_feature_chunk_batch(1, &seq, 64)];
        let mapping = vec![mapping_batch(&[(1, "genome1")])];
        let stats = build_index_from_arrow(
            &out,
            chunks.into_iter().map(Ok),
            mapping.into_iter().map(Ok),
            k,
            w,
            salt,
            false,
            Some(64 << 20),
            None,
        )
        .unwrap();
        assert_eq!(stats.num_buckets, 1);
        assert_eq!(stats.num_features, 1);

        let idx = ShardedInvertedIndex::open(&out).unwrap();
        let got = load_all_minimizers(&idx).unwrap();
        let want = reference_minimizers(&seq, k, w, salt);
        assert!(!want.is_empty(), "test sequence should yield minimizers");
        assert_eq!(
            got, want,
            "chunked build must reproduce un-chunked minimizers exactly"
        );
    }

    #[test]
    fn arrow_build_multi_bucket_opens() {
        let (k, w, salt) = (16usize, 5usize, 0x5555_5555_5555_5555u64);
        let s1 = make_dna(200, 1);
        let s2 = make_dna(180, 2);
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("idx.ryxdi");

        let chunks = vec![
            single_feature_chunk_batch(1, &s1, 64),
            single_feature_chunk_batch(2, &s2, 64),
        ];
        let mapping = vec![mapping_batch(&[(1, "g1"), (2, "g2")])];
        let stats = build_index_from_arrow(
            &out,
            chunks.into_iter().map(Ok),
            mapping.into_iter().map(Ok),
            k,
            w,
            salt,
            false,
            Some(64 << 20),
            None,
        )
        .unwrap();
        assert_eq!(stats.num_buckets, 2);
        assert_eq!(stats.num_features, 2);

        let idx = ShardedInvertedIndex::open(&out).unwrap();
        assert!(idx.total_minimizers() > 0);
    }

    #[test]
    fn source_labels_use_feature_idx_delim_form() {
        let (k, w, salt) = (16usize, 5usize, 0x5555_5555_5555_5555u64);
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("idx.ryxdi");
        let chunks = vec![
            single_feature_chunk_batch(10, &make_dna(120, 1), 64),
            single_feature_chunk_batch(20, &make_dna(120, 2), 64),
        ];
        // Both features in one bucket "g".
        let mapping = vec![mapping_batch(&[(10, "g"), (20, "g")])];
        build_index_from_arrow(
            &out,
            chunks.into_iter().map(Ok),
            mapping.into_iter().map(Ok),
            k,
            w,
            salt,
            false,
            Some(64 << 20),
            None,
        )
        .unwrap();

        let (_names, sources, _stats) = crate::parquet_index::read_buckets_parquet(&out).unwrap();
        let bucket_sources = sources.values().next().unwrap();
        let mut got: Vec<&String> = bucket_sources.iter().collect();
        got.sort();
        assert_eq!(
            got,
            vec![
                &"feature_idx::10".to_string(),
                &"feature_idx::20".to_string()
            ]
        );
    }

    #[test]
    fn invalid_k_errors() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("idx.ryxdi");
        let chunks = vec![single_feature_chunk_batch(1, &make_dna(64, 1), 32)];
        let mapping = vec![mapping_batch(&[(1, "g")])];
        let err = build_index_from_arrow(
            &out,
            chunks.into_iter().map(Ok),
            mapping.into_iter().map(Ok),
            20, // invalid: must be 16, 32, or 64
            5,
            0,
            false,
            Some(64 << 20),
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("16, 32, or 64"), "{err}");
    }

    #[test]
    fn orientation_flips_rc_to_baseline() {
        // Bucket "g": feature 1 = S (baseline, forward), feature 2 = revcomp(S).
        // With orient on, feature 2 is flipped back, so the bucket's minimizer set
        // equals S's forward set. With orient off it does NOT (RC adds new minimizers).
        let (k, w, salt) = (16usize, 5usize, 0x5555_5555_5555_5555u64);
        let s = make_dna(300, 1);
        let rc = revcomp(&s);
        let want = reference_minimizers(&s, k, w, salt);
        assert!(!want.is_empty());

        let dir = tempfile::tempdir().unwrap();

        let out_on = dir.path().join("on.ryxdi");
        build(
            &out_on,
            vec![
                single_feature_chunk_batch(1, &s, 64),
                single_feature_chunk_batch(2, &rc, 64),
            ],
            vec![mapping_batch(&[(1, "g"), (2, "g")])],
            k,
            w,
            salt,
            true,
        )
        .unwrap();
        let got_on = load_all_minimizers(&ShardedInvertedIndex::open(&out_on).unwrap()).unwrap();
        assert_eq!(
            got_on, want,
            "orientation should flip RC feature back to baseline"
        );

        let out_off = dir.path().join("off.ryxdi");
        build(
            &out_off,
            vec![
                single_feature_chunk_batch(1, &s, 64),
                single_feature_chunk_batch(2, &rc, 64),
            ],
            vec![mapping_batch(&[(1, "g"), (2, "g")])],
            k,
            w,
            salt,
            false,
        )
        .unwrap();
        let got_off = load_all_minimizers(&ShardedInvertedIndex::open(&out_off).unwrap()).unwrap();
        assert!(
            got_off.len() > got_on.len(),
            "without orientation the RC feature adds distinct minimizers (on={}, off={})",
            got_on.len(),
            got_off.len()
        );
    }

    #[test]
    fn orientation_baseline_survives_interleaved_buckets() {
        // Stream order interleaves buckets: g1(f1=S1), g2(f2=S2), g1(f3=revcomp(S1)).
        // g1's baseline must still be alive when f3 arrives (freed only after f3),
        // so f3 flips back to S1. Premature freeing would leave f3 forward (RC) and
        // change the union.
        let (k, w, salt) = (16usize, 5usize, 0x5555_5555_5555_5555u64);
        let s1 = make_dna(300, 1);
        let s2 = make_dna(280, 2);
        let mut want = reference_minimizers(&s1, k, w, salt);
        want.extend(reference_minimizers(&s2, k, w, salt));

        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("idx.ryxdi");
        build(
            &out,
            vec![
                single_feature_chunk_batch(1, &s1, 64),
                single_feature_chunk_batch(2, &s2, 64),
                single_feature_chunk_batch(3, &revcomp(&s1), 64),
            ],
            vec![mapping_batch(&[(1, "g1"), (2, "g2"), (3, "g1")])],
            k,
            w,
            salt,
            true,
        )
        .unwrap();
        let got = load_all_minimizers(&ShardedInvertedIndex::open(&out).unwrap()).unwrap();
        assert_eq!(
            got, want,
            "interleaved orientation must keep g1's baseline live for f3"
        );
    }

    #[test]
    fn parallel_batching_is_invariant_to_batch_size() {
        // The built index must be identical whether features are extracted one-per-batch
        // (forces cross-batch baseline persistence) or all in one batch (forces the
        // within-batch seed/seeded classification). Mix of buckets, interleaving, and an
        // RC feature that must orient back to its bucket's baseline.
        let (k, w, salt) = (16usize, 5usize, 0x5555_5555_5555_5555u64);
        let s1 = make_dna(300, 1);
        let seqs: Vec<(i64, &str, Vec<u8>)> = vec![
            (1, "g1", s1.clone()),
            (2, "g2", make_dna(280, 2)),
            (3, "g1", revcomp(&s1)), // RC of feature 1 → orients back under g1's baseline
            (4, "g3", make_dna(260, 4)),
            (5, "g2", make_dna(290, 5)),
            (6, "g1", make_dna(310, 6)),
        ];

        let build_with = |batch_target_bytes: usize| -> (HashSet<u64>, ArrowBuildStats) {
            let dir = tempfile::tempdir().unwrap();
            let out = dir.path().join("idx.ryxdi");
            let chunks: Vec<RecordBatch> = seqs
                .iter()
                .map(|(f, _, s)| single_feature_chunk_batch(*f, s, 64))
                .collect();
            let map_rows: Vec<(i64, &str)> = seqs.iter().map(|(f, b, _)| (*f, *b)).collect();
            let mapping = vec![mapping_batch(&map_rows)];
            let stats = build_index_from_arrow_inner(
                &out,
                chunks.into_iter().map(Ok),
                mapping.into_iter().map(Ok),
                k,
                w,
                salt,
                true, // orient
                64 << 20,
                batch_target_bytes,
                None,
            )
            .unwrap();
            let got = load_all_minimizers(&ShardedInvertedIndex::open(&out).unwrap()).unwrap();
            (got, stats)
        };

        let (one_per_batch, stats_small) = build_with(1); // each feature flushes alone
        let (single_batch, stats_big) = build_with(1 << 30); // all features in one batch
        assert_eq!(
            one_per_batch, single_batch,
            "minimizer set must not depend on batch size"
        );
        assert_eq!(stats_small.num_features, 6);
        assert_eq!(stats_small.num_buckets, 3);
        assert_eq!(stats_small.num_features, stats_big.num_features);
        assert_eq!(stats_small.num_buckets, stats_big.num_buckets);
    }

    #[test]
    fn arrow_build_consolidates_cross_shard_duplicates() {
        // A single bucket of many IDENTICAL features makes every minimizer recur in
        // every flush window. With a tiny shard budget the streaming accumulator writes
        // several intermediate shards that each re-contain the full minimizer set; an
        // un-consolidated build stores each (minimizer, bucket) pair once per shard,
        // inflating the index by the number of shards (measured ~60x on a real
        // human-reference build) and leaving the shards minimizer-overlapping. The Arrow
        // build must consolidate — like the CLI single-bucket path — so the stored entry
        // count collapses to the true unique count and the final shards are range-disjoint.

        // High-diversity DNA so each feature yields ~one distinct minimizer per window
        // (make_dna is too periodic to reliably overflow a shard); identical across
        // features so they share every minimizer. xorshift64 — deterministic, no RNG dep.
        fn diverse_dna(n: usize, seed: u64) -> Vec<u8> {
            let alpha = b"ACGT";
            let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            (0..n)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    alpha[(x & 3) as usize]
                })
                .collect()
        }

        let (k, w, salt) = (16usize, 5usize, 0x5555_5555_5555_5555u64);
        let seq = diverse_dna(25_000, 1);
        let nfeat = 40i64;
        let chunks: Vec<RecordBatch> = (1..=nfeat)
            .map(|f| single_feature_chunk_batch(f, &seq, 64))
            .collect();
        let map_rows: Vec<(i64, &str)> = (1..=nfeat).map(|f| (f, "g")).collect();
        let mapping = vec![mapping_batch(&map_rows)];

        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("idx.ryxdi");
        // available = 2 MiB → shard_size = max(1 MiB, 1 MiB) = 1 MiB (~65k entries/flush);
        // 40 identical ~25k-minimizer features each re-add the full set, forcing many
        // intermediate flushes. batch_target_bytes large so all features extract together.
        build_index_from_arrow_inner(
            &out,
            chunks.into_iter().map(Ok),
            mapping.into_iter().map(Ok),
            k,
            w,
            salt,
            false,           // orient
            2 << 20,         // available
            8 * 1024 * 1024, // batch_target_bytes
            None,
        )
        .unwrap();

        let idx = ShardedInvertedIndex::open(&out).unwrap();
        let unique = load_all_minimizers(&idx).unwrap();
        assert!(!unique.is_empty(), "test sequence must yield minimizers");

        let manifest = idx.manifest();
        let stored: u64 = manifest
            .shards
            .iter()
            .map(|s| s.num_minimizers as u64)
            .sum();

        // Core invariant: a single-bucket index stores each minimizer exactly once.
        // Un-consolidated, `stored` is a multiple of `unique` (one copy per intermediate
        // shard the minimizer landed in).
        assert_eq!(
            stored,
            unique.len() as u64,
            "single-bucket index must store each minimizer once after consolidation \
             (stored={}, unique={})",
            stored,
            unique.len(),
        );
        assert!(
            !manifest.has_overlapping_shards,
            "consolidated shards must be marked non-overlapping"
        );
        // Final shards must be range-disjoint (consolidation invariant); shards are
        // listed in ascending minimizer order.
        for win in manifest.shards.windows(2) {
            assert!(
                win[0].min_end < win[1].min_start,
                "shards must be range-disjoint after consolidation: \
                 [{}..{}] then [{}..{}]",
                win[0].min_start,
                win[0].min_end,
                win[1].min_start,
                win[1].min_end,
            );
        }
    }
}
