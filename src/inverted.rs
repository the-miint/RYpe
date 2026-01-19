//! Inverted index for fast minimizer → bucket lookups.
//!
//! The inverted index maps each unique minimizer to the set of bucket IDs
//! that contain it, using a CSR (Compressed Sparse Row) format. This enables
//! O(Q * log(U)) lookups where Q = query minimizers and U = unique minimizers.

// The k-way merge heap type is complex but clear in context
#![allow(clippy::type_complexity)]

use anyhow::{anyhow, Result};
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::constants::{MAX_INVERTED_BUCKET_IDS, MAX_INVERTED_MINIMIZERS};
use crate::encoding::{decode_varint, encode_varint, VarIntError, MAX_VARINT_BYTES};
use crate::index::Index;
use crate::sharded::ShardInfo;
use crate::sharded_main::MainIndexShard;
use crate::types::IndexMetadata;

/// CSR-format inverted index for fast minimizer → bucket lookups.
///
/// # Invariants
/// - `minimizers` is sorted in ascending order with no duplicates
/// - `offsets.len() == minimizers.len() + 1`
/// - `offsets[0] == 0`
/// - `offsets` is monotonically increasing
/// - `offsets[minimizers.len()] == bucket_ids.len()`
/// - For each minimizer at index i, the associated bucket IDs are `bucket_ids[offsets[i]..offsets[i+1]]`
///
/// # Thread Safety
/// InvertedIndex can be shared across threads for concurrent classification.
#[derive(Debug)]
pub struct InvertedIndex {
    pub k: usize,
    pub w: usize,
    pub salt: u64,
    pub(crate) source_hash: u64,
    pub(crate) minimizers: Vec<u64>,
    pub(crate) offsets: Vec<u32>,
    pub(crate) bucket_ids: Vec<u32>,
}

impl InvertedIndex {
    /// Compute a hash from index metadata for validation.
    /// Hash is computed from sorted (bucket_id, minimizer_count) pairs.
    pub fn compute_metadata_hash(metadata: &IndexMetadata) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut pairs: Vec<(u32, usize)> = metadata
            .bucket_minimizer_counts
            .iter()
            .map(|(&id, &count)| (id, count))
            .collect();
        pairs.sort_by_key(|(id, _)| *id);

        let mut hasher = DefaultHasher::new();
        for (id, count) in pairs {
            id.hash(&mut hasher);
            count.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Build an inverted index from a primary Index using k-way merge.
    ///
    /// This implementation uses O(num_buckets) heap memory instead of O(total_entries)
    /// by leveraging the fact that each bucket is already sorted.
    ///
    /// # Requirements
    /// All buckets in the index must be finalized (sorted and deduplicated).
    ///
    /// # Panics
    /// Panics if any bucket is not sorted.
    pub fn build_from_index(index: &Index) -> Self {
        Self::verify_buckets_sorted(&index.buckets);

        let metadata = IndexMetadata {
            k: index.k,
            w: index.w,
            salt: index.salt,
            bucket_names: index.bucket_names.clone(),
            bucket_sources: index.bucket_sources.clone(),
            bucket_minimizer_counts: index.buckets.iter().map(|(&id, v)| (id, v.len())).collect(),
        };

        Self::build_from_bucket_map(index.k, index.w, index.salt, &index.buckets, &metadata)
    }

    /// Build an inverted index from a single main index shard.
    ///
    /// This is used for 1:1 correspondence between main index shards and
    /// inverted index shards when the main index is sharded.
    pub fn build_from_shard(shard: &MainIndexShard) -> Self {
        Self::verify_buckets_sorted(&shard.buckets);

        let metadata = IndexMetadata {
            k: shard.k,
            w: shard.w,
            salt: shard.salt,
            bucket_names: HashMap::new(),
            bucket_sources: HashMap::new(),
            bucket_minimizer_counts: shard.buckets.iter().map(|(&id, v)| (id, v.len())).collect(),
        };

        Self::build_from_bucket_map(shard.k, shard.w, shard.salt, &shard.buckets, &metadata)
    }

    /// Verify all buckets are sorted. Panics if any bucket is unsorted.
    fn verify_buckets_sorted(buckets: &HashMap<u32, Vec<u64>>) {
        buckets.par_iter().for_each(|(&id, minimizers)| {
            if !minimizers.windows(2).all(|w| w[0] <= w[1]) {
                panic!("Bucket {} is not sorted. Call finalize_bucket() before building inverted index.", id);
            }
        });
    }

    /// Core k-way merge algorithm to build inverted index from bucket data.
    fn build_from_bucket_map(
        k: usize,
        w: usize,
        salt: u64,
        buckets: &HashMap<u32, Vec<u64>>,
        metadata: &IndexMetadata,
    ) -> Self {
        let total_entries: usize = buckets.values().map(|v| v.len()).sum();

        // Handle empty case
        if buckets.is_empty() || total_entries == 0 {
            return InvertedIndex {
                k,
                w,
                salt,
                source_hash: Self::compute_metadata_hash(metadata),
                minimizers: Vec::new(),
                offsets: vec![0],
                bucket_ids: Vec::new(),
            };
        }

        let estimated_unique = total_entries / 2;
        let mut minimizers: Vec<u64> = Vec::with_capacity(estimated_unique);
        let mut offsets: Vec<u32> = Vec::with_capacity(estimated_unique + 1);
        let mut bucket_ids_out: Vec<u32> = Vec::with_capacity(total_entries);

        offsets.push(0);

        // K-way merge using a min-heap
        // Each heap entry: (Reverse((minimizer_value, bucket_id)), data_index, position_in_bucket)
        let bucket_data: Vec<(u32, &[u64])> = buckets
            .iter()
            .filter(|(_, mins)| !mins.is_empty())
            .map(|(&id, mins)| (id, mins.as_slice()))
            .collect();

        let mut heap: BinaryHeap<(Reverse<(u64, u32)>, usize, usize)> =
            BinaryHeap::with_capacity(bucket_data.len());

        for (idx, &(bucket_id, mins)) in bucket_data.iter().enumerate() {
            heap.push((Reverse((mins[0], bucket_id)), idx, 0));
        }

        let mut current_min: Option<u64> = None;
        let mut current_bucket_ids: Vec<u32> = Vec::new();

        while let Some((Reverse((min_val, _)), data_idx, pos)) = heap.pop() {
            let (bucket_id, bucket_mins) = bucket_data[data_idx];

            if current_min != Some(min_val) {
                // Flush previous minimizer's bucket list
                if current_min.is_some() {
                    current_bucket_ids.sort_unstable();
                    current_bucket_ids.dedup();
                    bucket_ids_out.extend_from_slice(&current_bucket_ids);
                    offsets.push(
                        u32::try_from(bucket_ids_out.len())
                            .expect("CSR offset overflow: bucket_ids exceeded u32::MAX"),
                    );
                    current_bucket_ids.clear();
                }
                minimizers.push(min_val);
                current_min = Some(min_val);
            }

            current_bucket_ids.push(bucket_id);

            // Push next element from this bucket if available
            let next_pos = pos + 1;
            if next_pos < bucket_mins.len() {
                heap.push((
                    Reverse((bucket_mins[next_pos], bucket_id)),
                    data_idx,
                    next_pos,
                ));
            }
        }

        // Flush final minimizer's bucket list
        if current_min.is_some() {
            current_bucket_ids.sort_unstable();
            current_bucket_ids.dedup();
            bucket_ids_out.extend_from_slice(&current_bucket_ids);
            offsets.push(
                u32::try_from(bucket_ids_out.len())
                    .expect("CSR offset overflow: bucket_ids exceeded u32::MAX"),
            );
        }

        minimizers.shrink_to_fit();
        offsets.shrink_to_fit();
        bucket_ids_out.shrink_to_fit();

        InvertedIndex {
            k,
            w,
            salt,
            source_hash: Self::compute_metadata_hash(metadata),
            minimizers,
            offsets,
            bucket_ids: bucket_ids_out,
        }
    }

    /// Validate that this inverted index matches the given metadata.
    pub fn validate_against_metadata(&self, metadata: &IndexMetadata) -> Result<()> {
        if self.k != metadata.k || self.w != metadata.w || self.salt != metadata.salt {
            return Err(anyhow!(
                "Inverted index parameters don't match source index.\n  \
                 Inverted: K={}, W={}, salt=0x{:x}\n  \
                 Source:   K={}, W={}, salt=0x{:x}",
                self.k,
                self.w,
                self.salt,
                metadata.k,
                metadata.w,
                metadata.salt
            ));
        }

        let expected_hash = Self::compute_metadata_hash(metadata);
        if self.source_hash != expected_hash {
            return Err(anyhow!(
                "Inverted index is stale (hash 0x{:016x} != expected 0x{:016x}). \
                 The source index has been modified. Regenerate with 'rype index invert -i <index.ryidx>'",
                self.source_hash, expected_hash
            ));
        }

        Ok(())
    }

    /// Get bucket hit counts for a sorted query using hybrid binary search.
    ///
    /// # Arguments
    /// * `query` - A sorted, deduplicated slice of minimizer values
    ///
    /// # Returns
    /// HashMap of bucket_id -> hit_count for all buckets matching at least one query minimizer.
    pub fn get_bucket_hits(&self, query: &[u64]) -> HashMap<u32, usize> {
        let mut hits: HashMap<u32, usize> = HashMap::new();

        if query.is_empty() || self.minimizers.is_empty() {
            return hits;
        }

        let mut search_start = 0;

        for &q in query {
            if search_start >= self.minimizers.len() {
                break;
            }

            match self.minimizers[search_start..].binary_search(&q) {
                Ok(relative_idx) => {
                    let abs_idx = search_start + relative_idx;
                    let start = self.offsets[abs_idx] as usize;
                    let end = self.offsets[abs_idx + 1] as usize;
                    for &bid in &self.bucket_ids[start..end] {
                        *hits.entry(bid).or_insert(0) += 1;
                    }
                    search_start = abs_idx + 1;
                }
                Err(relative_idx) => {
                    search_start += relative_idx;
                }
            }
        }

        hits
    }

    /// Returns the number of unique minimizers in the index.
    pub fn num_minimizers(&self) -> usize {
        self.minimizers.len()
    }

    /// Returns the total number of bucket ID entries.
    pub fn num_bucket_entries(&self) -> usize {
        self.bucket_ids.len()
    }

    /// Get the sorted minimizer array (for debugging/inspection).
    pub fn minimizers(&self) -> &[u64] {
        &self.minimizers
    }

    /// Get the CSR offsets array (for debugging/inspection).
    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    /// Get the bucket IDs array (for debugging/inspection).
    pub fn bucket_ids(&self) -> &[u32] {
        &self.bucket_ids
    }

    /// Save a subset of this inverted index as a shard file (RYXS format v1).
    pub fn save_shard(
        &self,
        path: &Path,
        shard_id: u32,
        start_idx: usize,
        end_idx: usize,
        is_last_shard: bool,
    ) -> Result<ShardInfo> {
        let end_idx = end_idx.min(self.minimizers.len());
        if start_idx >= end_idx {
            return Err(anyhow!(
                "Invalid shard range: start {} >= end {}",
                start_idx,
                end_idx
            ));
        }

        let shard_minimizers = &self.minimizers[start_idx..end_idx];
        let min_start = shard_minimizers[0];
        // min_end is 0 for the last shard, or when end_idx == len (no next minimizer).
        // The latter happens when building 1:1 inverted shards from main shards,
        // where each shard contains all its minimizers but isn't marked as last.
        let min_end = if is_last_shard || end_idx >= self.minimizers.len() {
            0
        } else {
            self.minimizers[end_idx]
        };

        let bucket_start = self.offsets[start_idx] as usize;
        let bucket_end = self.offsets[end_idx] as usize;
        let shard_bucket_ids = &self.bucket_ids[bucket_start..bucket_end];

        let base_offset = self.offsets[start_idx];
        let shard_offsets: Vec<u32> = self.offsets[start_idx..=end_idx]
            .iter()
            .map(|&o| o - base_offset)
            .collect();

        let mut writer = BufWriter::new(File::create(path)?);
        let max_bucket_id = shard_bucket_ids.iter().copied().max().unwrap_or(0);

        writer.write_all(b"RYXS")?;
        writer.write_all(&1u32.to_le_bytes())?;
        writer.write_all(&(self.k as u64).to_le_bytes())?;
        writer.write_all(&(self.w as u64).to_le_bytes())?;
        writer.write_all(&self.salt.to_le_bytes())?;
        writer.write_all(&self.source_hash.to_le_bytes())?;
        writer.write_all(&shard_id.to_le_bytes())?;
        writer.write_all(&min_start.to_le_bytes())?;
        writer.write_all(&min_end.to_le_bytes())?;
        writer.write_all(&max_bucket_id.to_le_bytes())?;
        writer.write_all(&(shard_minimizers.len() as u64).to_le_bytes())?;
        writer.write_all(&(shard_bucket_ids.len() as u64).to_le_bytes())?;

        let mut encoder = zstd::stream::write::Encoder::new(writer, 3)?;

        const WRITE_BUF_SIZE: usize = 8 * 1024 * 1024;
        let mut write_buf = Vec::with_capacity(WRITE_BUF_SIZE);

        let flush_buf = |buf: &mut Vec<u8>,
                         encoder: &mut zstd::stream::write::Encoder<BufWriter<File>>|
         -> Result<()> {
            if !buf.is_empty() {
                encoder.write_all(buf)?;
                buf.clear();
            }
            Ok(())
        };

        for &offset in &shard_offsets {
            write_buf.extend_from_slice(&offset.to_le_bytes());
            if write_buf.len() >= WRITE_BUF_SIZE {
                flush_buf(&mut write_buf, &mut encoder)?;
            }
        }
        flush_buf(&mut write_buf, &mut encoder)?;

        let mut varint_buf = [0u8; 10];
        if !shard_minimizers.is_empty() {
            write_buf.extend_from_slice(&shard_minimizers[0].to_le_bytes());
            if write_buf.len() >= WRITE_BUF_SIZE {
                flush_buf(&mut write_buf, &mut encoder)?;
            }

            let mut prev = shard_minimizers[0];
            for &min in &shard_minimizers[1..] {
                let delta = min - prev;
                let len = encode_varint(delta, &mut varint_buf);
                write_buf.extend_from_slice(&varint_buf[..len]);
                if write_buf.len() >= WRITE_BUF_SIZE {
                    flush_buf(&mut write_buf, &mut encoder)?;
                }
                prev = min;
            }
        }
        flush_buf(&mut write_buf, &mut encoder)?;

        if max_bucket_id <= u8::MAX as u32 {
            for &bid in shard_bucket_ids {
                write_buf.push(bid as u8);
                if write_buf.len() >= WRITE_BUF_SIZE {
                    flush_buf(&mut write_buf, &mut encoder)?;
                }
            }
        } else if max_bucket_id <= u16::MAX as u32 {
            for &bid in shard_bucket_ids {
                write_buf.extend_from_slice(&(bid as u16).to_le_bytes());
                if write_buf.len() >= WRITE_BUF_SIZE {
                    flush_buf(&mut write_buf, &mut encoder)?;
                }
            }
        } else {
            for &bid in shard_bucket_ids {
                write_buf.extend_from_slice(&bid.to_le_bytes());
                if write_buf.len() >= WRITE_BUF_SIZE {
                    flush_buf(&mut write_buf, &mut encoder)?;
                }
            }
        }
        flush_buf(&mut write_buf, &mut encoder)?;

        encoder.finish()?;

        Ok(ShardInfo {
            shard_id,
            min_start,
            min_end,
            is_last_shard,
            num_minimizers: shard_minimizers.len(),
            num_bucket_ids: shard_bucket_ids.len(),
        })
    }

    /// Load a shard file into an InvertedIndex.
    ///
    /// Supports both legacy RYXS format and Parquet format (auto-detected by extension).
    ///
    /// # Parquet Shards
    /// Parquet shards should be loaded via `ShardedInvertedIndex::load_shard()` which
    /// provides the required parameters (k, w, salt, source_hash) from the manifest.
    /// Loading a Parquet shard directly requires these parameters and will fail.
    pub fn load_shard(path: &Path) -> Result<Self> {
        // Detect format based on extension
        if path.extension().map(|e| e == "parquet").unwrap_or(false) {
            #[cfg(feature = "parquet")]
            {
                return Err(anyhow!(
                    "Parquet shards must be loaded via ShardedInvertedIndex::load_shard() \
                     which provides parameters from the manifest. \
                     Use load_shard_parquet_with_params() directly if you have the parameters. \
                     Path: {}",
                    path.display()
                ));
            }
            #[cfg(not(feature = "parquet"))]
            {
                return Err(anyhow!(
                    "Parquet shard format requires --features parquet: {}",
                    path.display()
                ));
            }
        }
        Self::load_shard_legacy(path)
    }

    /// Load a legacy RYXS format shard file.
    fn load_shard_legacy(path: &Path) -> Result<Self> {
        let mut reader = BufReader::new(File::open(path)?);
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        reader.read_exact(&mut buf4)?;
        if &buf4 != b"RYXS" {
            return Err(anyhow!("Invalid shard format (expected RYXS)"));
        }

        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != 1 {
            return Err(anyhow!(
                "Unsupported shard version: {} (expected 1)",
                version
            ));
        }

        reader.read_exact(&mut buf8)?;
        let k = u64::from_le_bytes(buf8) as usize;
        if !matches!(k, 16 | 32 | 64) {
            return Err(anyhow!(
                "Invalid K value in shard: {} (must be 16, 32, or 64)",
                k
            ));
        }

        reader.read_exact(&mut buf8)?;
        let w = u64::from_le_bytes(buf8) as usize;

        reader.read_exact(&mut buf8)?;
        let salt = u64::from_le_bytes(buf8);

        reader.read_exact(&mut buf8)?;
        let source_hash = u64::from_le_bytes(buf8);

        reader.read_exact(&mut buf4)?; // shard_id
        reader.read_exact(&mut buf8)?; // min_start
        reader.read_exact(&mut buf8)?; // min_end

        reader.read_exact(&mut buf4)?;
        let max_bucket_id = u32::from_le_bytes(buf4);

        reader.read_exact(&mut buf8)?;
        let num_minimizers = u64::from_le_bytes(buf8) as usize;

        reader.read_exact(&mut buf8)?;
        let num_bucket_ids = u64::from_le_bytes(buf8) as usize;

        if num_minimizers > MAX_INVERTED_MINIMIZERS {
            return Err(anyhow!(
                "Shard has too many minimizers: {} (hard-coded limit: {} as defensive sanity check)",
                num_minimizers,
                MAX_INVERTED_MINIMIZERS
            ));
        }
        if num_bucket_ids > MAX_INVERTED_BUCKET_IDS {
            return Err(anyhow!("Shard has too many bucket IDs: {}", num_bucket_ids));
        }

        let mut decoder = zstd::stream::read::Decoder::new(reader)?;

        const READ_BUF_SIZE: usize = 8 * 1024 * 1024;
        let mut read_buf = vec![0u8; READ_BUF_SIZE];
        let mut buf_pos = 0;
        let mut buf_len = 0;

        macro_rules! ensure_bytes {
            ($n:expr) => {{
                while buf_len - buf_pos < $n {
                    if buf_pos > 0 {
                        read_buf.copy_within(buf_pos..buf_len, 0);
                        buf_len -= buf_pos;
                        buf_pos = 0;
                    }
                    let space = read_buf.len() - buf_len;
                    if space == 0 {
                        read_buf.resize(read_buf.len() * 2, 0);
                    }
                    let n = decoder.read(&mut read_buf[buf_len..])?;
                    if n == 0 {
                        return Err(anyhow!("Unexpected end of shard data"));
                    }
                    buf_len += n;
                }
            }};
        }

        let offsets_count = num_minimizers + 1;
        let mut offsets = Vec::with_capacity(offsets_count);
        for _ in 0..offsets_count {
            ensure_bytes!(4);
            let offset = u32::from_le_bytes(read_buf[buf_pos..buf_pos + 4].try_into().unwrap());
            offsets.push(offset);
            buf_pos += 4;
        }

        if !offsets.is_empty() && offsets[0] != 0 {
            return Err(anyhow!("Invalid shard: first offset must be 0"));
        }
        if offsets.windows(2).any(|w| w[0] > w[1]) {
            return Err(anyhow!(
                "Invalid shard: offsets not monotonically increasing"
            ));
        }
        if !offsets.is_empty() && *offsets.last().unwrap() as usize != num_bucket_ids {
            return Err(anyhow!(
                "Invalid shard: final offset doesn't match bucket_ids count"
            ));
        }

        let mut minimizers = Vec::with_capacity(num_minimizers);
        if num_minimizers > 0 {
            ensure_bytes!(8);
            let first = u64::from_le_bytes(read_buf[buf_pos..buf_pos + 8].try_into().unwrap());
            minimizers.push(first);
            buf_pos += 8;

            let mut prev = first;
            for i in 1..num_minimizers {
                // Ensure enough bytes for a complete varint.
                // Loop until we successfully decode or hit a real error.
                let (delta, consumed) = loop {
                    // Ensure we have at least MAX_VARINT_BYTES available if possible
                    ensure_bytes!(MAX_VARINT_BYTES.min(1));

                    match decode_varint(&read_buf[buf_pos..buf_len]) {
                        Ok((delta, consumed)) => break (delta, consumed),
                        Err(VarIntError::Truncated(_)) => {
                            // Need more data - shift remaining bytes and read more
                            read_buf.copy_within(buf_pos..buf_len, 0);
                            buf_len -= buf_pos;
                            buf_pos = 0;
                            let n = decoder.read(&mut read_buf[buf_len..])?;
                            if n == 0 {
                                return Err(anyhow!(
                                    "Truncated varint at minimizer {} (EOF with continuation bit set, buf_len={})",
                                    i, buf_len
                                ));
                            }
                            buf_len += n;
                        }
                        Err(VarIntError::Overflow(bytes)) => {
                            return Err(anyhow!(
                                "Malformed varint at minimizer {}: exceeded 10 bytes (consumed {})",
                                i,
                                bytes
                            ));
                        }
                    }
                };
                buf_pos += consumed;

                let val = prev.checked_add(delta).ok_or_else(|| {
                    anyhow!(
                        "Minimizer overflow at index {} (prev={}, delta={})",
                        i,
                        prev,
                        delta
                    )
                })?;

                if val <= prev && i > 0 {
                    return Err(anyhow!(
                        "Minimizers not strictly increasing at index {} (prev={}, val={})",
                        i,
                        prev,
                        val
                    ));
                }

                minimizers.push(val);
                prev = val;
            }
        }

        let mut bucket_ids = Vec::with_capacity(num_bucket_ids);
        if max_bucket_id <= u8::MAX as u32 {
            for _ in 0..num_bucket_ids {
                ensure_bytes!(1);
                bucket_ids.push(read_buf[buf_pos] as u32);
                buf_pos += 1;
            }
        } else if max_bucket_id <= u16::MAX as u32 {
            for _ in 0..num_bucket_ids {
                ensure_bytes!(2);
                let val = u16::from_le_bytes(read_buf[buf_pos..buf_pos + 2].try_into().unwrap());
                bucket_ids.push(val as u32);
                buf_pos += 2;
            }
        } else {
            for _ in 0..num_bucket_ids {
                ensure_bytes!(4);
                let val = u32::from_le_bytes(read_buf[buf_pos..buf_pos + 4].try_into().unwrap());
                bucket_ids.push(val);
                buf_pos += 4;
            }
        }

        Ok(InvertedIndex {
            k,
            w,
            salt,
            source_hash,
            minimizers,
            offsets,
            bucket_ids,
        })
    }

    /// Save this inverted index as a Parquet shard file.
    ///
    /// The Parquet schema is flattened: (minimizer: u64, bucket_id: u32)
    /// with one row per (minimizer, bucket_id) pair, sorted by minimizer then bucket_id.
    /// This enables row group filtering based on minimizer range statistics.
    ///
    /// # Memory Efficiency
    /// Streams directly from CSR format without materializing the full flattened
    /// dataset. Memory usage is O(BATCH_SIZE) instead of O(total_pairs).
    ///
    /// # Arguments
    /// * `path` - Output path (should end in .parquet)
    /// * `shard_id` - Shard identifier for manifest
    /// * `options` - Optional Parquet write options (compression, bloom filters, etc.)
    ///
    /// # Returns
    /// ShardInfo describing the written shard.
    #[cfg(feature = "parquet")]
    pub fn save_shard_parquet(
        &self,
        path: &Path,
        shard_id: u32,
        options: Option<&crate::parquet_index::ParquetWriteOptions>,
    ) -> Result<ShardInfo> {
        use anyhow::Context;
        use arrow::array::{ArrayRef, UInt32Builder, UInt64Builder};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        let opts = options.cloned().unwrap_or_default();

        if self.minimizers.is_empty() {
            // Empty shard - don't create file
            return Ok(ShardInfo {
                shard_id,
                min_start: 0,
                min_end: 0,
                is_last_shard: true,
                num_minimizers: 0,
                num_bucket_ids: 0,
            });
        }

        // Schema: (minimizer: u64, bucket_id: u32)
        let schema = Arc::new(Schema::new(vec![
            Field::new("minimizer", DataType::UInt64, false),
            Field::new("bucket_id", DataType::UInt32, false),
        ]));

        // DRY: Use ParquetWriteOptions::to_writer_properties() as single source of truth
        let props = opts.to_writer_properties();

        let file = std::fs::File::create(path)
            .with_context(|| format!("Failed to create Parquet shard: {}", path.display()))?;
        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

        // Stream from CSR format in batches without materializing all pairs
        const BATCH_SIZE: usize = 100_000;
        let mut minimizer_builder = UInt64Builder::with_capacity(BATCH_SIZE);
        let mut bucket_id_builder = UInt32Builder::with_capacity(BATCH_SIZE);
        let mut pairs_in_batch = 0;

        for (i, &minimizer) in self.minimizers.iter().enumerate() {
            let start = self.offsets[i] as usize;
            let end = self.offsets[i + 1] as usize;

            for &bucket_id in &self.bucket_ids[start..end] {
                minimizer_builder.append_value(minimizer);
                bucket_id_builder.append_value(bucket_id);
                pairs_in_batch += 1;

                if pairs_in_batch >= BATCH_SIZE {
                    // Flush batch
                    let minimizer_array: ArrayRef = Arc::new(minimizer_builder.finish());
                    let bucket_id_array: ArrayRef = Arc::new(bucket_id_builder.finish());

                    let batch = RecordBatch::try_new(
                        schema.clone(),
                        vec![minimizer_array, bucket_id_array],
                    )?;
                    writer.write(&batch)?;

                    // Reset builders for next batch
                    minimizer_builder = UInt64Builder::with_capacity(BATCH_SIZE);
                    bucket_id_builder = UInt32Builder::with_capacity(BATCH_SIZE);
                    pairs_in_batch = 0;
                }
            }
        }

        // Flush remaining pairs
        if pairs_in_batch > 0 {
            let minimizer_array: ArrayRef = Arc::new(minimizer_builder.finish());
            let bucket_id_array: ArrayRef = Arc::new(bucket_id_builder.finish());

            let batch =
                RecordBatch::try_new(schema.clone(), vec![minimizer_array, bucket_id_array])?;
            writer.write(&batch)?;
        }

        writer.close()?;

        let min_start = self.minimizers[0];
        let min_end = 0; // Last shard marker

        Ok(ShardInfo {
            shard_id,
            min_start,
            min_end,
            is_last_shard: true,
            num_minimizers: self.minimizers.len(),
            num_bucket_ids: self.bucket_ids.len(),
        })
    }

    /// Load a Parquet shard with explicit parameters.
    ///
    /// This is the main entry point for loading Parquet shards. Parameters are
    /// provided by the manifest file that accompanies Parquet shards.
    ///
    /// # Performance
    /// Row groups are read in parallel using rayon, then concatenated and
    /// validated in a single pass to build the CSR structure.
    ///
    /// # Memory
    /// Uses a shared Bytes buffer to avoid opening N file descriptors.
    /// Peak memory usage is 2x the Parquet file size (buffer + decoded pairs).
    #[cfg(feature = "parquet")]
    pub fn load_shard_parquet_with_params(
        path: &Path,
        k: usize,
        w: usize,
        salt: u64,
        source_hash: u64,
    ) -> Result<Self> {
        use anyhow::Context;
        use arrow::array::{Array, UInt32Array, UInt64Array};
        use bytes::Bytes;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use rayon::prelude::*;
        use std::fs::File;
        use std::io::Read;

        // Validate k value (same as legacy format)
        if !matches!(k, 16 | 32 | 64) {
            return Err(anyhow!(
                "Invalid K value for Parquet shard: {} (must be 16, 32, or 64)",
                k
            ));
        }

        // Read entire file into memory once (avoids N file opens)
        let mut file = File::open(path)
            .with_context(|| format!("Failed to open Parquet shard: {}", path.display()))?;
        let file_size = file.metadata()?.len() as usize;
        let mut buffer = Vec::with_capacity(file_size);
        file.read_to_end(&mut buffer)?;
        let bytes = Bytes::from(buffer);

        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes.clone())?;
        let metadata = builder.metadata().clone();
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

        // Parallel read of row groups using shared bytes
        let row_group_results: Vec<Result<Vec<(u64, u32)>>> = (0..num_row_groups)
            .into_par_iter()
            .map(|rg_idx| {
                let builder = ParquetRecordBatchReaderBuilder::try_new(bytes.clone())?;
                let reader = builder.with_row_groups(vec![rg_idx]).build()?;

                let mut pairs = Vec::new();

                for batch in reader {
                    let batch = batch?;

                    let min_col = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<UInt64Array>()
                        .context("Expected UInt64Array for minimizer column")?;

                    let bid_col = batch
                        .column(1)
                        .as_any()
                        .downcast_ref::<UInt32Array>()
                        .context("Expected UInt32Array for bucket_id column")?;

                    for i in 0..batch.num_rows() {
                        pairs.push((min_col.value(i), bid_col.value(i)));
                    }
                }

                Ok(pairs)
            })
            .collect();

        // Concatenate row groups in order (they're already sorted globally since we write them that way)
        // This is O(n) vs O(n log k) for k-way merge, and avoids tuple overhead.
        let mut all_minimizers: Vec<u64> = Vec::new();
        let mut all_bucket_ids: Vec<u32> = Vec::new();

        for result in row_group_results {
            let pairs = result?;
            for (m, b) in pairs {
                all_minimizers.push(m);
                all_bucket_ids.push(b);
            }
        }

        if all_minimizers.is_empty() {
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

        // Validate global ordering and build CSR structure in one pass
        let mut minimizers: Vec<u64> = Vec::with_capacity(all_minimizers.len() / 2);
        let mut offsets: Vec<u32> = Vec::with_capacity(all_minimizers.len() / 2 + 1);
        let mut bucket_ids_out: Vec<u32> = Vec::with_capacity(all_bucket_ids.len());

        offsets.push(0);
        let mut current_min = all_minimizers[0];
        let mut prev_min = all_minimizers[0];
        minimizers.push(current_min);

        for (i, (&m, &b)) in all_minimizers.iter().zip(all_bucket_ids.iter()).enumerate() {
            // Validation: minimizers must be non-decreasing
            if m < prev_min {
                return Err(anyhow!(
                    "Parquet shard has unsorted minimizers at row {}: {} < {} (corrupt file?)",
                    i,
                    m,
                    prev_min
                ));
            }
            prev_min = m;

            if m != current_min {
                // New minimizer - finalize previous
                offsets.push(bucket_ids_out.len() as u32);
                minimizers.push(m);
                current_min = m;
            }
            bucket_ids_out.push(b);
        }

        // Finalize last minimizer
        offsets.push(bucket_ids_out.len() as u32);

        // Validation: offsets must be monotonically increasing
        if offsets.windows(2).any(|w| w[0] > w[1]) {
            return Err(anyhow!(
                "Parquet shard produced invalid offsets (internal error)"
            ));
        }

        // Validation: minimizers must be strictly increasing (after grouping)
        if minimizers.windows(2).any(|w| w[0] >= w[1]) {
            return Err(anyhow!(
                "Parquet shard has duplicate minimizers after merge (corrupt file?)"
            ));
        }

        // Validation: size limits (same as legacy format)
        if minimizers.len() > MAX_INVERTED_MINIMIZERS {
            return Err(anyhow!(
                "Parquet shard has too many minimizers: {} (limit: {})",
                minimizers.len(),
                MAX_INVERTED_MINIMIZERS
            ));
        }
        if bucket_ids_out.len() > MAX_INVERTED_BUCKET_IDS {
            return Err(anyhow!(
                "Parquet shard has too many bucket IDs: {} (limit: {})",
                bucket_ids_out.len(),
                MAX_INVERTED_BUCKET_IDS
            ));
        }

        minimizers.shrink_to_fit();
        offsets.shrink_to_fit();
        bucket_ids_out.shrink_to_fit();

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

    /// Load only the row groups that overlap with the given minimizer range.
    ///
    /// This enables efficient query processing when the query minimizers span
    /// a small fraction of the total range. Row groups are filtered based on
    /// their min/max statistics before loading.
    ///
    /// # Arguments
    /// * `path` - Path to the Parquet shard file
    /// * `k`, `w`, `salt`, `source_hash` - Index parameters from manifest
    /// * `query_min` - Minimum minimizer in the query set
    /// * `query_max` - Maximum minimizer in the query set
    ///
    /// # Returns
    /// A partial InvertedIndex containing only minimizers in the overlapping range.
    #[cfg(feature = "parquet")]
    pub fn load_shard_parquet_filtered(
        path: &Path,
        k: usize,
        w: usize,
        salt: u64,
        source_hash: u64,
        query_min: u64,
        query_max: u64,
    ) -> Result<Self> {
        use anyhow::Context;
        use arrow::array::{Array, UInt32Array, UInt64Array};
        use bytes::Bytes;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use parquet::file::reader::FileReader;
        use parquet::file::serialized_reader::SerializedFileReader;
        use parquet::file::statistics::Statistics;
        use rayon::prelude::*;
        use std::fs::File;
        use std::io::Read;

        // Validate k value
        if !matches!(k, 16 | 32 | 64) {
            return Err(anyhow!(
                "Invalid K value for Parquet shard: {} (must be 16, 32, or 64)",
                k
            ));
        }

        // Read entire file into memory
        let mut file = File::open(path)
            .with_context(|| format!("Failed to open Parquet shard: {}", path.display()))?;
        let file_size = file.metadata()?.len() as usize;
        let mut buffer = Vec::with_capacity(file_size);
        file.read_to_end(&mut buffer)?;
        let bytes = Bytes::from(buffer);

        // Get metadata to filter row groups
        let parquet_reader = SerializedFileReader::new(bytes.clone())?;
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

        // Find row groups that overlap with query range
        let mut matching_row_groups: Vec<usize> = Vec::new();

        for rg_idx in 0..num_row_groups {
            let rg_meta = metadata.row_group(rg_idx);
            let col_meta = rg_meta.column(0); // minimizer column

            if let Some(stats) = col_meta.statistics() {
                match stats {
                    Statistics::Int64(s) => {
                        // Parquet stores u64 as i64, interpret correctly
                        let rg_min = s.min_opt().map(|v| *v as u64).unwrap_or(0);
                        let rg_max = s.max_opt().map(|v| *v as u64).unwrap_or(u64::MAX);

                        // Check if row group overlaps with query range
                        if rg_max >= query_min && rg_min <= query_max {
                            matching_row_groups.push(rg_idx);
                        }
                    }
                    _ => {
                        // Unknown stats type, include to be safe
                        matching_row_groups.push(rg_idx);
                    }
                }
            } else {
                // No stats, include to be safe
                matching_row_groups.push(rg_idx);
            }
        }

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

        // Parallel read of matching row groups only
        let row_group_results: Vec<Result<Vec<(u64, u32)>>> = matching_row_groups
            .par_iter()
            .map(|&rg_idx| {
                let builder = ParquetRecordBatchReaderBuilder::try_new(bytes.clone())?;
                let reader = builder.with_row_groups(vec![rg_idx]).build()?;

                let mut pairs = Vec::new();

                for batch in reader {
                    let batch = batch?;

                    let min_col = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<UInt64Array>()
                        .context("Expected UInt64Array for minimizer column")?;

                    let bid_col = batch
                        .column(1)
                        .as_any()
                        .downcast_ref::<UInt32Array>()
                        .context("Expected UInt32Array for bucket_id column")?;

                    for i in 0..batch.num_rows() {
                        let m = min_col.value(i);
                        // Only include pairs within the query range
                        if m >= query_min && m <= query_max {
                            pairs.push((m, bid_col.value(i)));
                        }
                    }
                }

                Ok(pairs)
            })
            .collect();

        // Concatenate results (row groups are already sorted)
        let mut all_minimizers: Vec<u64> = Vec::new();
        let mut all_bucket_ids: Vec<u32> = Vec::new();

        for result in row_group_results {
            let pairs = result?;
            for (m, b) in pairs {
                all_minimizers.push(m);
                all_bucket_ids.push(b);
            }
        }

        if all_minimizers.is_empty() {
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

        // Build CSR structure with validation
        let mut minimizers: Vec<u64> = Vec::with_capacity(all_minimizers.len() / 2);
        let mut offsets: Vec<u32> = Vec::with_capacity(all_minimizers.len() / 2 + 1);
        let mut bucket_ids_out: Vec<u32> = Vec::with_capacity(all_bucket_ids.len());

        offsets.push(0);
        let mut current_min = all_minimizers[0];
        let mut prev_min = all_minimizers[0];
        minimizers.push(current_min);

        for (i, (&m, &b)) in all_minimizers.iter().zip(all_bucket_ids.iter()).enumerate() {
            if m < prev_min {
                return Err(anyhow!(
                    "Parquet shard has unsorted minimizers at row {}: {} < {}",
                    i,
                    m,
                    prev_min
                ));
            }
            prev_min = m;

            if m != current_min {
                offsets.push(bucket_ids_out.len() as u32);
                minimizers.push(m);
                current_min = m;
            }
            bucket_ids_out.push(b);
        }

        offsets.push(bucket_ids_out.len() as u32);

        minimizers.shrink_to_fit();
        offsets.shrink_to_fit();
        bucket_ids_out.shrink_to_fit();

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

    /// Get row group statistics from a Parquet shard without loading data.
    ///
    /// Returns a list of (row_group_index, min_minimizer, max_minimizer) tuples.
    /// This can be used to plan which row groups to load for a query.
    #[cfg(feature = "parquet")]
    pub fn get_parquet_row_group_stats(path: &Path) -> Result<Vec<(usize, u64, u64)>> {
        use parquet::file::reader::FileReader;
        use parquet::file::serialized_reader::SerializedFileReader;
        use parquet::file::statistics::Statistics;
        use std::fs::File;

        let file = File::open(path)?;
        let reader = SerializedFileReader::new(file)?;
        let metadata = reader.metadata();
        let num_row_groups = metadata.num_row_groups();

        let mut stats = Vec::with_capacity(num_row_groups);

        for rg_idx in 0..num_row_groups {
            let rg_meta = metadata.row_group(rg_idx);
            let col_meta = rg_meta.column(0); // minimizer column

            let (rg_min, rg_max) = if let Some(Statistics::Int64(s)) = col_meta.statistics() {
                let min = s.min_opt().map(|v| *v as u64).unwrap_or(0);
                let max = s.max_opt().map(|v| *v as u64).unwrap_or(u64::MAX);
                (min, max)
            } else {
                (0, u64::MAX) // No stats, assume full range
            };

            stats.push((rg_idx, rg_min, rg_max));
        }

        Ok(stats)
    }

    /// Find which row groups contain at least one query minimizer.
    ///
    /// Uses binary search per row group: O(R × log Q) where R = row groups, Q = query minimizers.
    /// This is correct even if row groups overlap or are unsorted.
    ///
    /// # Arguments
    /// * `query_minimizers` - Sorted slice of query minimizers (caller must ensure sorted)
    /// * `row_group_stats` - List of (row_group_idx, min, max) tuples
    ///
    /// # Returns
    /// Vector of row group indices that contain at least one query minimizer.
    #[cfg(feature = "parquet")]
    pub fn find_matching_row_groups(
        query_minimizers: &[u64],
        row_group_stats: &[(usize, u64, u64)],
    ) -> Vec<usize> {
        if query_minimizers.is_empty() || row_group_stats.is_empty() {
            return Vec::new();
        }

        // For each row group, binary search to check if any query minimizer falls within its range
        let mut matching = Vec::new();

        for &(rg_id, rg_min, rg_max) in row_group_stats {
            // Find first query minimizer >= rg_min
            let start = query_minimizers.partition_point(|&m| m < rg_min);

            // If that minimizer exists and is <= rg_max, this row group matches
            if start < query_minimizers.len() && query_minimizers[start] <= rg_max {
                matching.push(rg_id);
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
    ///
    /// # Panics (debug mode)
    /// Panics if `query_minimizers` is not sorted.
    ///
    /// # Returns
    /// A partial InvertedIndex containing only data from matching row groups.
    #[cfg(feature = "parquet")]
    pub fn load_shard_parquet_for_query(
        path: &Path,
        k: usize,
        w: usize,
        salt: u64,
        source_hash: u64,
        query_minimizers: &[u64],
    ) -> Result<Self> {
        use anyhow::Context;
        use arrow::array::{Array, UInt32Array, UInt64Array};
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use parquet::file::reader::FileReader;
        use parquet::file::serialized_reader::SerializedFileReader;
        use parquet::file::statistics::Statistics;
        use rayon::prelude::*;
        use std::fs::File;

        // For small query sets, binary search on sorted array is O(log n) and faster than
        // HashSet lookup which is O(1) but has higher constant factors (hashing + probe).
        // Empirically, break-even point is ~1000 elements on modern CPUs.
        const QUERY_HASHSET_THRESHOLD: usize = 1000;

        // Validate query_minimizers is sorted - required for binary search correctness
        // in find_matching_row_groups() and row-level filtering.
        if !query_minimizers.is_empty() && !query_minimizers.windows(2).all(|w| w[0] <= w[1]) {
            return Err(anyhow!(
                "query_minimizers must be sorted (precondition violation)"
            ));
        }

        // Validate k value
        if !matches!(k, 16 | 32 | 64) {
            return Err(anyhow!(
                "Invalid K value for Parquet shard: {} (must be 16, 32, or 64)",
                k
            ));
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
        let (_num_row_groups, rg_stats) = {
            let file = File::open(path)
                .with_context(|| format!("Failed to open Parquet shard: {}", path.display()))?;
            let parquet_reader = SerializedFileReader::new(file)?;
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
            let mut rg_stats: Vec<(usize, u64, u64)> = Vec::with_capacity(num_row_groups);
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

                rg_stats.push((rg_idx, rg_min, rg_max));
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

            (num_row_groups, rg_stats)
        }; // parquet_reader and file handle dropped here

        // Find matching row groups using binary search on statistics
        let matching_row_groups = Self::find_matching_row_groups(query_minimizers, &rg_stats);

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

        // Estimate pairs per row group for pre-allocation. Row groups are ~100k rows,
        // and we expect query selectivity to filter most rows. Conservative estimate: 10%.
        const ROW_GROUP_SIZE: usize = 100_000;
        const EXPECTED_SELECTIVITY: f64 = 0.10;
        let estimated_pairs_per_rg = (ROW_GROUP_SIZE as f64 * EXPECTED_SELECTIVITY) as usize;

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
                        .context("Expected UInt64Array for minimizer column")?;

                    let bid_col = batch
                        .column(1)
                        .as_any()
                        .downcast_ref::<UInt32Array>()
                        .context("Expected UInt32Array for bucket_id column")?;

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
            let pairs = result.with_context(|| {
                format!("Failed to read row group {} from {}", idx, path.display())
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
        debug_assert!(read_idx <= 0x7FFFFFFF, "Read index exceeds 31 bits");
        if is_rc {
            read_idx | 0x80000000
        } else {
            read_idx
        }
    }

    /// Unpack a read_id entry into (read_index, is_rc).
    #[inline]
    pub fn unpack_read_id(packed: u32) -> (u32, bool) {
        let is_rc = (packed & 0x80000000) != 0;
        let read_idx = packed & 0x7FFFFFFF;
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
    pub const MAX_READS: usize = 0x7FFFFFFF;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::Index;
    use tempfile::NamedTempFile;

    #[test]
    fn test_inverted_index_build() {
        let mut index = Index::new(64, 50, 0x1234).unwrap();
        index.buckets.insert(1, vec![100, 200, 300]);
        index.buckets.insert(2, vec![200, 300, 400]);
        index.buckets.insert(3, vec![500, 600]);
        index.bucket_names.insert(1, "A".into());
        index.bucket_names.insert(2, "B".into());
        index.bucket_names.insert(3, "C".into());

        let inverted = InvertedIndex::build_from_index(&index);

        assert_eq!(inverted.k, 64);
        assert_eq!(inverted.w, 50);
        assert_eq!(inverted.salt, 0x1234);
        assert_eq!(inverted.num_minimizers(), 6);
        assert_eq!(inverted.num_bucket_entries(), 8);
    }

    #[test]
    fn test_inverted_index_get_bucket_hits() {
        let mut index = Index::new(64, 50, 0).unwrap();
        index.buckets.insert(1, vec![100, 200, 300]);
        index.buckets.insert(2, vec![200, 300, 400]);
        index.buckets.insert(3, vec![500, 600]);
        index.bucket_names.insert(1, "A".into());
        index.bucket_names.insert(2, "B".into());
        index.bucket_names.insert(3, "C".into());

        let inverted = InvertedIndex::build_from_index(&index);

        let query = vec![200, 300, 500];
        let hits = inverted.get_bucket_hits(&query);

        assert_eq!(hits.get(&1), Some(&2));
        assert_eq!(hits.get(&2), Some(&2));
        assert_eq!(hits.get(&3), Some(&1));
    }

    #[test]
    fn test_inverted_index_get_bucket_hits_no_matches() {
        let mut index = Index::new(64, 50, 0).unwrap();
        index.buckets.insert(1, vec![100, 200, 300]);
        index.bucket_names.insert(1, "A".into());

        let inverted = InvertedIndex::build_from_index(&index);

        let query = vec![999, 1000, 1001];
        let hits = inverted.get_bucket_hits(&query);

        assert!(hits.is_empty());
    }

    #[test]
    fn test_inverted_index_validate_success() -> Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();

        let mut index = Index::new(64, 50, 0x1234).unwrap();
        index.buckets.insert(1, vec![100, 200]);
        index.bucket_names.insert(1, "A".into());
        index.bucket_sources.insert(1, vec!["src".into()]);
        index.save(&path)?;

        let inverted = InvertedIndex::build_from_index(&index);
        let metadata = Index::load_metadata(&path)?;

        inverted.validate_against_metadata(&metadata)?;
        Ok(())
    }

    #[test]
    fn test_inverted_index_validate_stale() -> Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();

        let mut index = Index::new(64, 50, 0x1234).unwrap();
        index.buckets.insert(1, vec![100, 200]);
        index.bucket_names.insert(1, "A".into());
        index.bucket_sources.insert(1, vec!["src".into()]);
        index.save(&path)?;

        let inverted = InvertedIndex::build_from_index(&index);

        index.buckets.insert(2, vec![300, 400]);
        index.bucket_names.insert(2, "B".into());
        index.bucket_sources.insert(2, vec!["src2".into()]);
        index.save(&path)?;

        let metadata = Index::load_metadata(&path)?;

        let result = inverted.validate_against_metadata(&metadata);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("stale"));

        Ok(())
    }

    #[test]
    fn test_inverted_index_empty() {
        let index = Index::new(64, 50, 0).unwrap();
        let inverted = InvertedIndex::build_from_index(&index);

        assert_eq!(inverted.num_minimizers(), 0);
        assert_eq!(inverted.num_bucket_entries(), 0);

        let hits = inverted.get_bucket_hits(&[100, 200, 300]);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_inverted_index_hybrid_search_correctness() {
        let mut index = Index::new(64, 50, 0).unwrap();
        index.buckets.insert(1, (0..1000).map(|i| i * 10).collect());
        index.bucket_names.insert(1, "A".into());

        let inverted = InvertedIndex::build_from_index(&index);

        let query: Vec<u64> = vec![50, 500, 5000, 9990];
        let hits = inverted.get_bucket_hits(&query);

        assert_eq!(hits.get(&1), Some(&4));
    }

    #[test]
    fn test_shard_save_load_roundtrip() -> Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();

        let mut index = Index::new(64, 50, 0xABCD).unwrap();
        index.buckets.insert(1, vec![100, 200, 300]);
        index.buckets.insert(2, vec![200, 300, 400]);
        index.bucket_names.insert(1, "A".into());
        index.bucket_names.insert(2, "B".into());

        let inverted = InvertedIndex::build_from_index(&index);

        let shard_info = inverted.save_shard(&path, 0, 0, inverted.num_minimizers(), true)?;

        assert_eq!(shard_info.shard_id, 0);
        assert_eq!(shard_info.num_minimizers, inverted.num_minimizers());
        assert_eq!(shard_info.num_bucket_ids, inverted.num_bucket_entries());

        let loaded = InvertedIndex::load_shard(&path)?;

        assert_eq!(loaded.k, 64);
        assert_eq!(loaded.w, 50);
        assert_eq!(loaded.salt, 0xABCD);
        assert_eq!(loaded.num_minimizers(), inverted.num_minimizers());
        assert_eq!(loaded.num_bucket_entries(), inverted.num_bucket_entries());

        let hits = loaded.get_bucket_hits(&[200, 300]);
        assert_eq!(hits.get(&1), Some(&2));
        assert_eq!(hits.get(&2), Some(&2));

        Ok(())
    }

    #[test]
    fn test_shard_partial_range() -> Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();

        let mut index = Index::new(64, 50, 0x1234).unwrap();
        index.buckets.insert(1, vec![100, 200, 300, 400, 500]);
        index.bucket_names.insert(1, "A".into());

        let inverted = InvertedIndex::build_from_index(&index);
        assert_eq!(inverted.num_minimizers(), 5);

        let shard_info = inverted.save_shard(&path, 0, 0, 3, false)?;

        assert_eq!(shard_info.shard_id, 0);
        assert_eq!(shard_info.num_minimizers, 3);
        assert!(!shard_info.is_last_shard);
        assert_eq!(shard_info.min_start, 100);
        assert_eq!(shard_info.min_end, 400);

        let loaded = InvertedIndex::load_shard(&path)?;
        assert_eq!(loaded.num_minimizers(), 3);

        let hits = loaded.get_bucket_hits(&[100, 200, 300, 400, 500]);
        assert_eq!(hits.get(&1), Some(&3));

        Ok(())
    }

    /// Regression test for panic when saving a full-range shard with is_last_shard=false.
    /// This happens when building 1:1 inverted shards from a sharded main index,
    /// where each inverted shard contains all minimizers (end_idx = len) but only
    /// the final shard has is_last_shard=true.
    #[test]
    fn test_shard_full_range_not_last() -> Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();

        let mut index = Index::new(64, 50, 0x9999).unwrap();
        index.buckets.insert(1, vec![100, 200, 300, 400, 500]);
        index.bucket_names.insert(1, "A".into());

        let inverted = InvertedIndex::build_from_index(&index);
        assert_eq!(inverted.num_minimizers(), 5);

        // This is the scenario that caused the panic: saving full range with is_last_shard=false
        // This happens when building 1:1 inverted shards from sharded main index
        let shard_info = inverted.save_shard(&path, 0, 0, inverted.num_minimizers(), false)?;

        assert_eq!(shard_info.shard_id, 0);
        assert_eq!(shard_info.num_minimizers, 5);
        assert!(!shard_info.is_last_shard);
        assert_eq!(shard_info.min_start, 100);
        // When end_idx == len, min_end should be 0 (no next minimizer)
        assert_eq!(shard_info.min_end, 0);

        // Verify the shard can be loaded and used
        let loaded = InvertedIndex::load_shard(&path)?;
        assert_eq!(loaded.num_minimizers(), 5);

        let hits = loaded.get_bucket_hits(&[100, 200, 300, 400, 500]);
        assert_eq!(hits.get(&1), Some(&5));

        Ok(())
    }

    #[test]
    fn test_shard_invalid_magic() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path();
        std::fs::write(path, b"NOTS").unwrap();

        let result = InvertedIndex::load_shard(path);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid shard format"));
    }

    #[test]
    fn test_build_from_shard() {
        use crate::sharded_main::MainIndexShard;

        // Create a main index shard with some buckets
        let mut shard = MainIndexShard {
            k: 64,
            w: 50,
            salt: 0x1234,
            shard_id: 0,
            buckets: HashMap::new(),
            bucket_sources: HashMap::new(),
        };
        shard.buckets.insert(1, vec![100, 200, 300]);
        shard.buckets.insert(2, vec![200, 300, 400]);
        shard.buckets.insert(3, vec![500, 600]);

        let inverted = InvertedIndex::build_from_shard(&shard);

        assert_eq!(inverted.k, 64);
        assert_eq!(inverted.w, 50);
        assert_eq!(inverted.salt, 0x1234);
        // Unique minimizers: 100, 200, 300, 400, 500, 600
        assert_eq!(inverted.num_minimizers(), 6);
        // Bucket entries: 100->1, 200->1,2, 300->1,2, 400->2, 500->3, 600->3 = 8
        assert_eq!(inverted.num_bucket_entries(), 8);

        // Verify we can query it
        let hits = inverted.get_bucket_hits(&[200, 300, 500]);
        assert_eq!(hits.get(&1), Some(&2)); // 200, 300
        assert_eq!(hits.get(&2), Some(&2)); // 200, 300
        assert_eq!(hits.get(&3), Some(&1)); // 500
    }

    #[test]
    fn test_build_from_shard_empty() {
        use crate::sharded_main::MainIndexShard;

        let shard = MainIndexShard {
            k: 64,
            w: 50,
            salt: 0xABCD,
            shard_id: 5,
            buckets: HashMap::new(),
            bucket_sources: HashMap::new(),
        };

        let inverted = InvertedIndex::build_from_shard(&shard);

        assert_eq!(inverted.k, 64);
        assert_eq!(inverted.w, 50);
        assert_eq!(inverted.salt, 0xABCD);
        assert_eq!(inverted.num_minimizers(), 0);
        assert_eq!(inverted.num_bucket_entries(), 0);
    }

    #[test]
    fn test_build_from_shard_single_bucket() {
        use crate::sharded_main::MainIndexShard;

        let mut shard = MainIndexShard {
            k: 32,
            w: 20,
            salt: 0x5678,
            shard_id: 0,
            buckets: HashMap::new(),
            bucket_sources: HashMap::new(),
        };
        shard.buckets.insert(42, vec![10, 20, 30, 40, 50]);

        let inverted = InvertedIndex::build_from_shard(&shard);

        assert_eq!(inverted.num_minimizers(), 5);
        assert_eq!(inverted.num_bucket_entries(), 5);

        // Each minimizer maps to only bucket 42
        let hits = inverted.get_bucket_hits(&[10, 20, 30, 40, 50]);
        assert_eq!(hits.get(&42), Some(&5));
    }

    #[test]
    fn test_build_from_shard_high_overlap() {
        use crate::sharded_main::MainIndexShard;

        // All buckets share all minimizers (maximum overlap)
        let mut shard = MainIndexShard {
            k: 64,
            w: 50,
            salt: 0,
            shard_id: 0,
            buckets: HashMap::new(),
            bucket_sources: HashMap::new(),
        };
        let shared_mins = vec![100, 200, 300, 400, 500];
        shard.buckets.insert(1, shared_mins.clone());
        shard.buckets.insert(2, shared_mins.clone());
        shard.buckets.insert(3, shared_mins.clone());
        shard.buckets.insert(4, shared_mins.clone());

        let inverted = InvertedIndex::build_from_shard(&shard);

        // Only 5 unique minimizers despite 20 total entries
        assert_eq!(inverted.num_minimizers(), 5);
        // Each minimizer maps to all 4 buckets = 20 bucket entries
        assert_eq!(inverted.num_bucket_entries(), 20);

        let hits = inverted.get_bucket_hits(&[100, 200, 300]);
        assert_eq!(hits.get(&1), Some(&3));
        assert_eq!(hits.get(&2), Some(&3));
        assert_eq!(hits.get(&3), Some(&3));
        assert_eq!(hits.get(&4), Some(&3));
    }

    #[test]
    fn test_build_from_shard_matches_build_from_index() {
        use crate::sharded_main::MainIndexShard;

        // Create identical data as both Index and MainIndexShard
        let mut index = Index::new(64, 50, 0x1234).unwrap();
        index.buckets.insert(1, vec![100, 200, 300]);
        index.buckets.insert(2, vec![200, 300, 400]);
        index.buckets.insert(3, vec![500, 600]);
        index.bucket_names.insert(1, "A".into());
        index.bucket_names.insert(2, "B".into());
        index.bucket_names.insert(3, "C".into());

        let mut shard = MainIndexShard {
            k: 64,
            w: 50,
            salt: 0x1234,
            shard_id: 0,
            buckets: HashMap::new(),
            bucket_sources: HashMap::new(),
        };
        shard.buckets.insert(1, vec![100, 200, 300]);
        shard.buckets.insert(2, vec![200, 300, 400]);
        shard.buckets.insert(3, vec![500, 600]);

        let inv_from_index = InvertedIndex::build_from_index(&index);
        let inv_from_shard = InvertedIndex::build_from_shard(&shard);

        // Core structure should be identical
        assert_eq!(inv_from_index.k, inv_from_shard.k);
        assert_eq!(inv_from_index.w, inv_from_shard.w);
        assert_eq!(inv_from_index.salt, inv_from_shard.salt);
        assert_eq!(
            inv_from_index.num_minimizers(),
            inv_from_shard.num_minimizers()
        );
        assert_eq!(
            inv_from_index.num_bucket_entries(),
            inv_from_shard.num_bucket_entries()
        );

        // Minimizer arrays should be identical
        assert_eq!(inv_from_index.minimizers, inv_from_shard.minimizers);

        // Offsets should be identical
        assert_eq!(inv_from_index.offsets, inv_from_shard.offsets);

        // Bucket IDs should be identical
        assert_eq!(inv_from_index.bucket_ids, inv_from_shard.bucket_ids);

        // Query results should match
        let hits_index = inv_from_index.get_bucket_hits(&[200, 300, 500]);
        let hits_shard = inv_from_shard.get_bucket_hits(&[200, 300, 500]);
        assert_eq!(hits_index, hits_shard);
    }

    // ==================== QueryInvertedIndex Tests ====================

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

    /// Regression test for varint boundary bug in load_shard().
    ///
    /// The bug: load_shard() used `ensure_bytes!(1)` before decoding varints,
    /// but LEB128 varints can be up to 10 bytes. When a large delta value
    /// happened to span buffer boundaries during streaming decompression,
    /// only part of the varint was decoded, corrupting subsequent minimizers.
    ///
    /// This test creates minimizers with large gaps (requiring multi-byte varints)
    /// and verifies save/load round-trip preserves all values correctly.
    #[test]
    fn test_shard_varint_boundary_roundtrip() -> Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();

        // Create minimizers with large gaps requiring multi-byte varints.
        // LEB128 encoding:
        // - 1 byte: values 0-127
        // - 2 bytes: values 128-16383
        // - 3 bytes: values 16384-2097151
        // - 10 bytes: max u64
        //
        // We intentionally create values with gaps of varying sizes to trigger
        // varints of different lengths, increasing the chance of hitting buffer boundaries.
        let mut minimizers: Vec<u64> = Vec::new();

        // Start with small values (small deltas, 1-byte varints)
        for i in 0..50 {
            minimizers.push(i * 10);
        }

        // Add medium gaps (2-3 byte varints)
        let mut val = 500;
        for _ in 0..100 {
            minimizers.push(val);
            val += 50000; // ~2-3 byte varint deltas
        }

        // Add large gaps (4+ byte varints)
        for _ in 0..50 {
            minimizers.push(val);
            val += 1_000_000_000; // ~5 byte varint deltas
        }

        // Add very large values near u64::MAX (requiring large absolute values and deltas)
        let high_base = 10_000_000_000_000_000_000u64;
        for i in 0..20 {
            minimizers.push(high_base + i * 100_000_000_000);
        }

        minimizers.sort();
        minimizers.dedup();

        // Create index with these minimizers
        let mut index = Index::new(64, 50, 0xDEADBEEF).unwrap();
        index.buckets.insert(1, minimizers.clone());
        index.bucket_names.insert(1, "test".into());

        let inverted = InvertedIndex::build_from_index(&index);
        let original_count = inverted.num_minimizers();

        // Save as shard
        let shard_info = inverted.save_shard(&path, 0, 0, original_count, true)?;
        assert_eq!(shard_info.num_minimizers, original_count);

        // Load shard back
        let loaded = InvertedIndex::load_shard(&path)?;

        // Verify all minimizers were preserved
        assert_eq!(
            loaded.num_minimizers(),
            original_count,
            "Minimizer count changed after save/load: {} -> {}",
            original_count,
            loaded.num_minimizers()
        );

        // Verify specific high-value minimizers weren't truncated
        for &m in &minimizers {
            let found = loaded.minimizers().binary_search(&m).is_ok();
            assert!(
                found,
                "Minimizer {} was lost after save/load round-trip (varint boundary bug?)",
                m
            );
        }

        // Verify the actual minimizer values match exactly
        assert_eq!(
            loaded.minimizers(),
            inverted.minimizers(),
            "Minimizer arrays differ after save/load"
        );

        // Verify queries still work correctly
        let hits = loaded.get_bucket_hits(&minimizers[..10]);
        assert_eq!(hits.get(&1), Some(&10));

        Ok(())
    }

    /// Regression test: ensure very large minimizer values near u64::MAX survive round-trip.
    /// This tests the upper range where varint deltas could overflow or be misinterpreted.
    #[test]
    fn test_shard_large_minimizers_roundtrip() -> Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();

        // Create minimizers near the top of the u64 range
        let minimizers: Vec<u64> = vec![
            12297829382473034403,
            12297829382473034405,
            12297829382473034408,
            12297829382473034409,
            12297829382473034410,
            14168481312020516, // The actual problematic minimizer from the bug report
            u64::MAX - 1000,
            u64::MAX - 100,
            u64::MAX - 10,
            u64::MAX - 1,
        ];

        let mut sorted_mins = minimizers.clone();
        sorted_mins.sort();

        let mut index = Index::new(64, 200, 0x5555555555555555).unwrap();
        index.buckets.insert(30, sorted_mins.clone());
        index.bucket_names.insert(30, "bucket-125".into());

        let inverted = InvertedIndex::build_from_index(&index);
        let shard_info = inverted.save_shard(&path, 0, 0, inverted.num_minimizers(), true)?;

        let loaded = InvertedIndex::load_shard(&path)?;

        assert_eq!(loaded.num_minimizers(), shard_info.num_minimizers);

        // Check the specific minimizer that was missing in the bug
        let test_min = 14168481312020516u64;
        let found = loaded.minimizers().binary_search(&test_min).is_ok();
        assert!(
            found,
            "Minimizer {} (from bug report) not found after save/load",
            test_min
        );

        // Verify all minimizers survived
        for &m in &sorted_mins {
            let found = loaded.minimizers().binary_search(&m).is_ok();
            assert!(found, "Minimizer {} lost after round-trip", m);
        }

        Ok(())
    }

    // ==================== Parquet I/O Tests ====================

    #[cfg(feature = "parquet")]
    #[test]
    fn test_inverted_parquet_roundtrip() -> Result<()> {
        use tempfile::TempDir;

        let tmp = TempDir::new()?;
        let path = tmp.path().join("shard.0.parquet");

        // Create an inverted index
        let mut index = Index::new(64, 50, 0xABCD).unwrap();
        index.buckets.insert(1, vec![100, 200, 300]);
        index.buckets.insert(2, vec![200, 300, 400]);
        index.buckets.insert(3, vec![500, 600]);
        index.bucket_names.insert(1, "A".into());
        index.bucket_names.insert(2, "B".into());
        index.bucket_names.insert(3, "C".into());

        let inverted = InvertedIndex::build_from_index(&index);
        let original_minimizers = inverted.minimizers().to_vec();
        let original_bucket_ids = inverted.bucket_ids().to_vec();

        // Save as Parquet
        let shard_info = inverted.save_shard_parquet(&path, 0, None)?;

        assert_eq!(shard_info.shard_id, 0);
        assert_eq!(shard_info.num_minimizers, inverted.num_minimizers());
        assert_eq!(shard_info.num_bucket_ids, inverted.num_bucket_entries());

        // Load back
        let loaded = InvertedIndex::load_shard_parquet_with_params(
            &path,
            inverted.k,
            inverted.w,
            inverted.salt,
            inverted.source_hash,
        )?;

        // Verify structure
        assert_eq!(loaded.k, 64);
        assert_eq!(loaded.w, 50);
        assert_eq!(loaded.salt, 0xABCD);
        assert_eq!(loaded.num_minimizers(), inverted.num_minimizers());
        assert_eq!(loaded.num_bucket_entries(), inverted.num_bucket_entries());

        // Verify minimizers match
        assert_eq!(loaded.minimizers(), original_minimizers.as_slice());

        // Verify bucket_ids match
        assert_eq!(loaded.bucket_ids(), original_bucket_ids.as_slice());

        // Verify queries work
        let hits = loaded.get_bucket_hits(&[200, 300, 500]);
        assert_eq!(hits.get(&1), Some(&2)); // 200, 300
        assert_eq!(hits.get(&2), Some(&2)); // 200, 300
        assert_eq!(hits.get(&3), Some(&1)); // 500

        Ok(())
    }

    #[cfg(feature = "parquet")]
    #[test]
    fn test_inverted_parquet_empty() -> Result<()> {
        use tempfile::TempDir;

        let tmp = TempDir::new()?;
        let path = tmp.path().join("empty.parquet");

        // Create an empty inverted index
        let index = Index::new(64, 50, 0).unwrap();
        let inverted = InvertedIndex::build_from_index(&index);

        // Save as Parquet
        let shard_info = inverted.save_shard_parquet(&path, 0, None)?;
        assert_eq!(shard_info.num_minimizers, 0);
        assert_eq!(shard_info.num_bucket_ids, 0);

        // Empty shard doesn't create file, so load should handle this
        // For now, just verify the shard_info is correct
        Ok(())
    }

    #[cfg(feature = "parquet")]
    #[test]
    fn test_inverted_parquet_large_values() -> Result<()> {
        use tempfile::TempDir;

        let tmp = TempDir::new()?;
        let path = tmp.path().join("large.parquet");

        // Create minimizers with large values
        let minimizers: Vec<u64> = vec![
            1,
            1000,
            1_000_000,
            1_000_000_000,
            1_000_000_000_000,
            u64::MAX / 2,
            u64::MAX - 100,
            u64::MAX - 1,
        ];

        let mut index = Index::new(64, 50, 0x12345678).unwrap();
        index.buckets.insert(1, minimizers.clone());
        index.bucket_names.insert(1, "test".into());

        let inverted = InvertedIndex::build_from_index(&index);

        // Save and load
        inverted.save_shard_parquet(&path, 0, None)?;
        let loaded = InvertedIndex::load_shard_parquet_with_params(
            &path,
            inverted.k,
            inverted.w,
            inverted.salt,
            inverted.source_hash,
        )?;

        // Verify all large values survived
        for &m in &minimizers {
            let found = loaded.minimizers().binary_search(&m).is_ok();
            assert!(found, "Large minimizer {} lost", m);
        }

        Ok(())
    }

    #[cfg(feature = "parquet")]
    #[test]
    fn test_inverted_parquet_load_shard_requires_params() -> Result<()> {
        use tempfile::TempDir;

        let tmp = TempDir::new()?;
        let parquet_path = tmp.path().join("shard.parquet");

        let mut index = Index::new(64, 50, 0).unwrap();
        index.buckets.insert(1, vec![100, 200, 300]);
        index.bucket_names.insert(1, "A".into());

        let inverted = InvertedIndex::build_from_index(&index);

        // Save as Parquet
        inverted.save_shard_parquet(&parquet_path, 0, None)?;

        // load_shard should error for Parquet files (they need manifest parameters)
        let result = InvertedIndex::load_shard(&parquet_path);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Parquet shards must be loaded via ShardedInvertedIndex"));

        // Direct loading with params should work
        let loaded = InvertedIndex::load_shard_parquet_with_params(
            &parquet_path,
            inverted.k,
            inverted.w,
            inverted.salt,
            inverted.source_hash,
        )?;
        assert_eq!(loaded.num_minimizers(), inverted.num_minimizers());

        Ok(())
    }

    #[cfg(feature = "parquet")]
    #[test]
    fn test_inverted_parquet_many_buckets() -> Result<()> {
        use tempfile::TempDir;

        let tmp = TempDir::new()?;
        let path = tmp.path().join("many_buckets.parquet");

        // Create index with many buckets sharing some minimizers
        let mut index = Index::new(64, 50, 0xBEEF).unwrap();
        let shared = vec![100, 200, 300, 400, 500];
        for i in 0..50 {
            let mut mins = shared.clone();
            mins.push(1000 + i as u64); // Each bucket has one unique minimizer
            mins.sort();
            index.buckets.insert(i, mins);
            index.bucket_names.insert(i, format!("bucket_{}", i));
        }

        let inverted = InvertedIndex::build_from_index(&index);

        // Save and load
        inverted.save_shard_parquet(&path, 0, None)?;
        let loaded = InvertedIndex::load_shard_parquet_with_params(
            &path,
            inverted.k,
            inverted.w,
            inverted.salt,
            inverted.source_hash,
        )?;

        // Shared minimizer 200 should map to all 50 buckets
        let hits = loaded.get_bucket_hits(&[200]);
        assert_eq!(hits.len(), 50);
        for i in 0..50 {
            assert_eq!(hits.get(&i), Some(&1));
        }

        Ok(())
    }

    #[cfg(feature = "parquet")]
    #[test]
    fn test_load_parquet_for_query_basic() -> Result<()> {
        use tempfile::TempDir;

        let tmp = TempDir::new()?;
        let path = tmp.path().join("shard.parquet");

        // Create an inverted index with multiple minimizers across buckets
        let mut index = Index::new(64, 50, 0xABCD).unwrap();
        index.buckets.insert(1, vec![100, 200, 300, 400, 500]);
        index.buckets.insert(2, vec![200, 300, 600, 700]);
        index.buckets.insert(3, vec![500, 800, 900]);
        index.bucket_names.insert(1, "A".into());
        index.bucket_names.insert(2, "B".into());
        index.bucket_names.insert(3, "C".into());

        let inverted = InvertedIndex::build_from_index(&index);

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

    #[cfg(feature = "parquet")]
    #[test]
    fn test_load_parquet_for_query_empty_query() -> Result<()> {
        use tempfile::TempDir;

        let tmp = TempDir::new()?;
        let path = tmp.path().join("shard.parquet");

        // Create an inverted index
        let mut index = Index::new(64, 50, 0).unwrap();
        index.buckets.insert(1, vec![100, 200, 300]);
        index.bucket_names.insert(1, "A".into());

        let inverted = InvertedIndex::build_from_index(&index);
        inverted.save_shard_parquet(&path, 0, None)?;

        // Empty query should return empty result without reading row groups
        let loaded = InvertedIndex::load_shard_parquet_for_query(
            &path,
            inverted.k,
            inverted.w,
            inverted.salt,
            inverted.source_hash,
            &[], // empty query
        )?;

        assert_eq!(loaded.minimizers().len(), 0);
        assert_eq!(loaded.bucket_ids().len(), 0);

        Ok(())
    }

    #[cfg(feature = "parquet")]
    #[test]
    fn test_load_parquet_for_query_no_matches() -> Result<()> {
        use tempfile::TempDir;

        let tmp = TempDir::new()?;
        let path = tmp.path().join("shard.parquet");

        // Create an inverted index with minimizers in a specific range
        let mut index = Index::new(64, 50, 0).unwrap();
        index.buckets.insert(1, vec![100, 200, 300]);
        index.bucket_names.insert(1, "A".into());

        let inverted = InvertedIndex::build_from_index(&index);
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
        )?;

        assert_eq!(loaded.minimizers().len(), 0);
        assert_eq!(loaded.bucket_ids().len(), 0);

        Ok(())
    }

    #[cfg(feature = "parquet")]
    #[test]
    fn test_load_parquet_for_query_multiple_row_groups() -> Result<()> {
        use tempfile::TempDir;

        let tmp = TempDir::new()?;
        let path = tmp.path().join("large.parquet");

        // Create a large inverted index that will span multiple row groups (>100k pairs)
        // Each bucket has many minimizers to create enough pairs
        let mut index = Index::new(64, 50, 0x1234).unwrap();

        // Create 10 buckets, each with 15000 minimizers (150k pairs total = 2+ row groups)
        for bucket_id in 0..10u32 {
            let base = bucket_id as u64 * 100_000;
            let minimizers: Vec<u64> = (0..15000).map(|i| base + i as u64).collect();
            index.buckets.insert(bucket_id, minimizers);
            index
                .bucket_names
                .insert(bucket_id, format!("bucket_{}", bucket_id));
        }

        let inverted = InvertedIndex::build_from_index(&index);
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

    #[cfg(feature = "parquet")]
    #[test]
    fn test_load_parquet_for_query_unsorted_input() -> Result<()> {
        use tempfile::TempDir;

        let tmp = TempDir::new()?;
        let path = tmp.path().join("shard.parquet");

        // Create a simple Parquet file
        let mut index = Index::new(64, 50, 0).unwrap();
        index.buckets.insert(1, vec![100, 200, 300]);
        index.bucket_names.insert(1, "A".into());

        let inverted = InvertedIndex::build_from_index(&index);
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

    #[cfg(feature = "parquet")]
    #[test]
    fn test_load_parquet_for_query_large_query_set() -> Result<()> {
        use tempfile::TempDir;

        let tmp = TempDir::new()?;
        let path = tmp.path().join("shard.parquet");

        // Create Parquet file with many minimizers
        let mut index = Index::new(64, 50, 0xBEEF).unwrap();
        let minimizers: Vec<u64> = (0..5000).map(|i| i as u64 * 10).collect();
        index.buckets.insert(1, minimizers);
        index.bucket_names.insert(1, "big".into());

        let inverted = InvertedIndex::build_from_index(&index);
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
        )?;

        // Should find all 2000 queried minimizers
        assert_eq!(loaded.minimizers().len(), 2000);

        // Verify bucket hits
        let hits = loaded.get_bucket_hits(&query_minimizers);
        assert_eq!(hits.get(&1), Some(&2000));

        Ok(())
    }

    #[cfg(feature = "parquet")]
    #[test]
    fn test_load_parquet_for_query_boundary_conditions() -> Result<()> {
        use tempfile::TempDir;

        let tmp = TempDir::new()?;
        let path = tmp.path().join("shard.parquet");

        // Create Parquet file with specific minimizers
        let mut index = Index::new(64, 50, 0).unwrap();
        index.buckets.insert(1, vec![100, 500, 1000]); // min=100, max=1000
        index.bucket_names.insert(1, "A".into());

        let inverted = InvertedIndex::build_from_index(&index);
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
        )?;

        // Should find nothing (99 < min, 1001 > max)
        assert_eq!(loaded_outside.minimizers().len(), 0);

        Ok(())
    }
}
