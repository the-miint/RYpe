//! Primary index structure for storing minimizer buckets.
//!
//! The `Index` struct stores a collection of named buckets, each containing
//! a set of minimizers extracted from reference sequences. It supports
//! incremental building and serialization in the v5 format.
//!
//! # Index Format
//!
//! The main index uses the single-file format (`.ryidx`):
//! - Header: magic "RYP5", version 5, k, w, salt, num_buckets
//! - Bucket metadata (uncompressed): for each bucket: minimizer_count, bucket_id, name, sources
//! - Minimizers (zstd compressed stream): delta+varint encoded minimizers per bucket

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use super::sharded_main::{plan_shards, MainIndexManifest, ShardedMainIndexBuilder};
use crate::constants::{
    BUCKET_SOURCE_DELIM, MAX_BUCKET_SIZE, MAX_NUM_BUCKETS, MAX_STRING_LENGTH, READ_BUF_SIZE,
    SINGLE_FILE_INDEX_MAGIC, SINGLE_FILE_INDEX_VERSION, WRITE_BUF_SIZE,
};
use crate::core::encoding::{decode_varint, encode_varint, VarIntError};
use crate::core::extraction::extract_into;
use crate::core::workspace::MinimizerWorkspace;
use crate::types::IndexMetadata;

/// Primary index for storing minimizer buckets.
///
/// Each bucket contains a sorted, deduplicated set of minimizers extracted
/// from reference sequences. The index supports incremental building (adding
/// records), finalization (sorting/deduping), and merging.
///
/// # File Format (v5)
/// - Header (uncompressed): magic "RYP5", version 5, k, w, salt, num_buckets
/// - Bucket metadata (uncompressed): for each bucket: minimizer_count, bucket_id, name, sources
/// - Minimizers (zstd compressed stream): delta+varint encoded minimizers per bucket
#[derive(Debug)]
pub struct Index {
    pub k: usize,
    pub w: usize,
    pub salt: u64,
    pub buckets: HashMap<u32, Vec<u64>>,
    pub bucket_names: HashMap<u32, String>,
    pub bucket_sources: HashMap<u32, Vec<String>>,
}

impl Index {
    /// Delimiter used for source names (re-exported from constants).
    pub const BUCKET_SOURCE_DELIM: &'static str = BUCKET_SOURCE_DELIM;

    /// Create a new empty index.
    ///
    /// # Arguments
    /// * `k` - K-mer size (must be 16, 32, or 64)
    /// * `w` - Window size for minimizer selection
    /// * `salt` - XOR salt applied to k-mer hashes
    ///
    /// # Errors
    /// Returns an error if k is not 16, 32, or 64.
    pub fn new(k: usize, w: usize, salt: u64) -> Result<Self> {
        if !matches!(k, 16 | 32 | 64) {
            return Err(anyhow!("K must be 16, 32, or 64 (got {})", k));
        }
        Ok(Index {
            k,
            w,
            salt,
            buckets: HashMap::new(),
            bucket_names: HashMap::new(),
            bucket_sources: HashMap::new(),
        })
    }

    /// Add a sequence record to a bucket.
    ///
    /// Extracts minimizers from the sequence and adds them to the specified bucket.
    /// Call `finalize_bucket()` after adding all records to sort and deduplicate.
    ///
    /// # Arguments
    /// * `id` - Bucket ID
    /// * `source_name` - Name of the source sequence
    /// * `sequence` - DNA sequence bytes
    /// * `ws` - Workspace for minimizer extraction
    pub fn add_record(
        &mut self,
        id: u32,
        source_name: &str,
        sequence: &[u8],
        ws: &mut MinimizerWorkspace,
    ) {
        let sources = self.bucket_sources.entry(id).or_default();
        sources.push(source_name.to_string());

        extract_into(sequence, self.k, self.w, self.salt, ws);
        let bucket = self.buckets.entry(id).or_default();
        bucket.extend_from_slice(&ws.buffer);
    }

    /// Finalize a bucket by sorting and deduplicating its minimizers.
    ///
    /// This must be called after all records have been added to a bucket
    /// and before using the index for classification or building an inverted index.
    pub fn finalize_bucket(&mut self, id: u32) {
        if let Some(sources) = self.bucket_sources.get_mut(&id) {
            sources.sort_unstable();
            sources.dedup();
        }
        if let Some(bucket) = self.buckets.get_mut(&id) {
            bucket.sort_unstable();
            bucket.dedup();
        }
    }

    /// Get the next available bucket ID.
    ///
    /// Returns one more than the current maximum bucket ID.
    ///
    /// # Errors
    /// Returns an error if the maximum bucket ID would overflow.
    pub fn next_id(&self) -> Result<u32> {
        let max_id = self.buckets.keys().max().copied().unwrap_or(0);
        max_id
            .checked_add(1)
            .ok_or_else(|| anyhow!("Bucket ID overflow: maximum ID {} reached", max_id))
    }

    /// Save the index to a file (v5 format with delta+varint+zstd compressed minimizers).
    ///
    /// Format v5:
    /// - Header (uncompressed): magic "RYP5", version 5, k, w, salt, num_buckets
    /// - Bucket metadata (uncompressed): for each bucket: minimizer_count, bucket_id, name, sources
    /// - Minimizers (zstd compressed stream): delta+varint encoded minimizers per bucket
    ///   - For each bucket: first minimizer as u64, then deltas as varints
    ///
    /// Delta encoding exploits the fact that minimizers are sorted, so consecutive
    /// values are often close together. Combined with varint and zstd, this achieves
    /// ~65% compression compared to raw u64 storage.
    pub fn save(&self, path: &Path) -> Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);
        writer.write_all(SINGLE_FILE_INDEX_MAGIC)?;
        writer.write_all(&SINGLE_FILE_INDEX_VERSION.to_le_bytes())?;
        writer.write_all(&(self.k as u64).to_le_bytes())?;
        writer.write_all(&(self.w as u64).to_le_bytes())?;
        writer.write_all(&(self.salt).to_le_bytes())?;

        let mut sorted_ids: Vec<_> = self.buckets.keys().collect();
        sorted_ids.sort();

        writer.write_all(&(sorted_ids.len() as u32).to_le_bytes())?;

        // Write all bucket metadata (uncompressed) - enables fast load_metadata()
        for id in &sorted_ids {
            let vec = &self.buckets[id];

            // Write minimizer count first (for seeking past in load_metadata)
            writer.write_all(&(vec.len() as u64).to_le_bytes())?;

            // Write bucket ID
            writer.write_all(&id.to_le_bytes())?;

            // Write name
            let name_str = self
                .bucket_names
                .get(id)
                .map(|s| s.as_str())
                .unwrap_or("unknown");
            let name_bytes = name_str.as_bytes();
            writer.write_all(&(name_bytes.len() as u64).to_le_bytes())?;
            writer.write_all(name_bytes)?;

            // Write sources
            let sources = self.bucket_sources.get(id);
            let empty = Vec::new();
            let src_vec = sources.unwrap_or(&empty);

            writer.write_all(&(src_vec.len() as u64).to_le_bytes())?;
            for src in src_vec {
                let s_bytes = src.as_bytes();
                writer.write_all(&(s_bytes.len() as u64).to_le_bytes())?;
                writer.write_all(s_bytes)?;
            }
        }

        // Write all minimizers with delta+varint encoding (zstd compressed stream)
        let mut encoder = zstd::stream::write::Encoder::new(writer, 3)?;

        let mut write_buf = Vec::with_capacity(WRITE_BUF_SIZE);
        let mut varint_buf = [0u8; 10]; // Max 10 bytes for u64 varint

        let flush_buf = |buf: &mut Vec<u8>,
                         encoder: &mut zstd::stream::write::Encoder<BufWriter<File>>|
         -> Result<()> {
            if !buf.is_empty() {
                encoder.write_all(buf)?;
                buf.clear();
            }
            Ok(())
        };

        for id in &sorted_ids {
            let vec = &self.buckets[id];
            if vec.is_empty() {
                continue;
            }

            // First minimizer: store as full u64
            write_buf.extend_from_slice(&vec[0].to_le_bytes());
            if write_buf.len() >= WRITE_BUF_SIZE {
                flush_buf(&mut write_buf, &mut encoder)?;
            }

            // Remaining minimizers: store as delta-encoded varints
            let mut prev = vec[0];
            for &val in &vec[1..] {
                let delta = val - prev; // Safe because minimizers are sorted
                let len = encode_varint(delta, &mut varint_buf);
                write_buf.extend_from_slice(&varint_buf[..len]);
                if write_buf.len() >= WRITE_BUF_SIZE {
                    flush_buf(&mut write_buf, &mut encoder)?;
                }
                prev = val;
            }
        }
        flush_buf(&mut write_buf, &mut encoder)?;

        encoder.finish()?;
        Ok(())
    }

    /// Save the index as a sharded format with memory budget-based sharding.
    ///
    /// This method creates multiple shard files based on the memory budget. Each shard
    /// contains complete buckets - a bucket's minimizers are never split across shards.
    ///
    /// # Arguments
    /// * `base_path` - Base path for output files (e.g., "index.ryidx")
    /// * `max_shard_bytes` - Maximum estimated bytes per shard
    ///
    /// # Returns
    /// The manifest for the created sharded index.
    ///
    /// # File Layout
    /// - `base_path.manifest` - Manifest file with metadata
    /// - `base_path.shard.0`, `base_path.shard.1`, ... - Shard files with minimizers
    pub fn save_sharded(
        &self,
        base_path: &Path,
        max_shard_bytes: usize,
    ) -> Result<MainIndexManifest> {
        // Plan which buckets go to which shards
        let bucket_counts: HashMap<u32, usize> =
            self.buckets.iter().map(|(&id, v)| (id, v.len())).collect();

        let shard_plan = plan_shards(&bucket_counts, max_shard_bytes);

        // Use the builder to write shards
        let mut builder =
            ShardedMainIndexBuilder::new(self.k, self.w, self.salt, base_path, max_shard_bytes)?;

        // Add buckets in shard order for efficient writing
        for (_shard_id, bucket_ids) in &shard_plan {
            for &bucket_id in bucket_ids {
                let name = self
                    .bucket_names
                    .get(&bucket_id)
                    .map(|s| s.as_str())
                    .unwrap_or("unknown");
                let sources = self
                    .bucket_sources
                    .get(&bucket_id)
                    .cloned()
                    .unwrap_or_default();
                let minimizers = self.buckets.get(&bucket_id).cloned().unwrap_or_default();

                builder.add_bucket(bucket_id, name, sources, minimizers)?;
            }
        }

        builder.finish()
    }

    /// Load an index from a file (v5 format with delta+varint+zstd compressed minimizers).
    ///
    /// This is the primary method for loading a main index file (.ryidx).
    pub fn load(path: &Path) -> Result<Self> {
        let mut reader = BufReader::new(File::open(path)?);
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        reader.read_exact(&mut buf4)?;
        if &buf4 != SINGLE_FILE_INDEX_MAGIC {
            return Err(anyhow!("Invalid Index Format (Expected RYP5)"));
        }

        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != SINGLE_FILE_INDEX_VERSION {
            return Err(anyhow!(
                "Unsupported index version: {} (expected {})",
                version,
                SINGLE_FILE_INDEX_VERSION
            ));
        }

        reader.read_exact(&mut buf8)?;
        let k = u64::from_le_bytes(buf8) as usize;
        if !matches!(k, 16 | 32 | 64) {
            return Err(anyhow!(
                "Invalid K value in index: {} (must be 16, 32, or 64)",
                k
            ));
        }
        reader.read_exact(&mut buf8)?;
        let w = u64::from_le_bytes(buf8) as usize;
        reader.read_exact(&mut buf8)?;
        let salt = u64::from_le_bytes(buf8);
        reader.read_exact(&mut buf4)?;
        let num = u32::from_le_bytes(buf4);
        if num > MAX_NUM_BUCKETS {
            return Err(anyhow!(
                "Number of buckets {} exceeds maximum {}",
                num,
                MAX_NUM_BUCKETS
            ));
        }

        let mut buckets = HashMap::new();
        let mut names = HashMap::new();
        let mut sources = HashMap::new();
        let mut bucket_order: Vec<(u32, usize)> = Vec::with_capacity(num as usize);

        // Read all bucket metadata (uncompressed)
        for _ in 0..num {
            reader.read_exact(&mut buf8)?;
            let vec_len = u64::from_le_bytes(buf8) as usize;
            if vec_len > MAX_BUCKET_SIZE {
                return Err(anyhow!(
                    "Bucket size {} exceeds maximum {}",
                    vec_len,
                    MAX_BUCKET_SIZE
                ));
            }

            reader.read_exact(&mut buf4)?;
            let id = u32::from_le_bytes(buf4);

            reader.read_exact(&mut buf8)?;
            let name_len = u64::from_le_bytes(buf8) as usize;
            if name_len > MAX_STRING_LENGTH {
                return Err(anyhow!(
                    "Bucket name length {} exceeds maximum {}",
                    name_len,
                    MAX_STRING_LENGTH
                ));
            }
            let mut nbuf = vec![0u8; name_len];
            reader.read_exact(&mut nbuf)?;
            names.insert(id, String::from_utf8(nbuf)?);

            reader.read_exact(&mut buf8)?;
            let src_count = u64::from_le_bytes(buf8) as usize;
            let mut src_list = Vec::with_capacity(src_count);
            for _ in 0..src_count {
                reader.read_exact(&mut buf8)?;
                let slen = u64::from_le_bytes(buf8) as usize;
                if slen > MAX_STRING_LENGTH {
                    return Err(anyhow!(
                        "Source string length {} exceeds maximum {}",
                        slen,
                        MAX_STRING_LENGTH
                    ));
                }
                let mut sbuf = vec![0u8; slen];
                reader.read_exact(&mut sbuf)?;
                src_list.push(String::from_utf8(sbuf)?);
            }
            sources.insert(id, src_list);

            // Remember bucket ID and size for reading minimizers later
            bucket_order.push((id, vec_len));
        }

        // Read all minimizers with delta+varint decoding (zstd compressed stream)
        let mut decoder = zstd::stream::read::Decoder::new(reader)
            .map_err(|e| anyhow!("Failed to create zstd decoder: {}", e))?;

        // Read buffer - we read in chunks and decode varints from the buffer
        let mut read_buf = vec![0u8; READ_BUF_SIZE];
        let mut buf_pos = 0usize;
        let mut buf_len = 0usize;

        for (id, vec_len) in bucket_order {
            let mut vec = Vec::with_capacity(vec_len);

            if vec_len == 0 {
                buckets.insert(id, vec);
                continue;
            }

            // Helper macro to ensure we have at least `need` bytes in buffer
            macro_rules! ensure_bytes {
                ($need:expr) => {{
                    let need = $need;
                    if buf_pos + need > buf_len {
                        // Shift remaining bytes to start
                        if buf_pos > 0 {
                            read_buf.copy_within(buf_pos..buf_len, 0);
                            buf_len -= buf_pos;
                            buf_pos = 0;
                        }
                        // Read more data until we have enough
                        while buf_len < need {
                            let n = decoder.read(&mut read_buf[buf_len..])?;
                            if n == 0 {
                                return Err(anyhow!("Unexpected end of compressed data"));
                            }
                            buf_len += n;
                        }
                    }
                }};
            }

            // Helper macro to refill buffer if running low (but don't require specific amount)
            macro_rules! refill_if_low {
                () => {{
                    if buf_pos >= buf_len {
                        // Buffer exhausted, refill
                        buf_pos = 0;
                        let n = decoder.read(&mut read_buf)?;
                        if n == 0 {
                            return Err(anyhow!("Unexpected end of compressed data"));
                        }
                        buf_len = n;
                    } else if buf_len - buf_pos < 10 {
                        // Low on data, shift and try to read more
                        read_buf.copy_within(buf_pos..buf_len, 0);
                        buf_len -= buf_pos;
                        buf_pos = 0;
                        // Try to read more (but don't fail if we can't)
                        let n = decoder.read(&mut read_buf[buf_len..])?;
                        buf_len += n;
                    }
                }};
            }

            // Read first minimizer as full u64
            ensure_bytes!(8);
            let first = u64::from_le_bytes(read_buf[buf_pos..buf_pos + 8].try_into().unwrap());
            buf_pos += 8;
            vec.push(first);

            // Read remaining minimizers as delta-encoded varints
            let mut prev = first;
            for i in 1..vec_len {
                // Loop until we successfully decode or hit a real error
                let (delta, consumed) = loop {
                    refill_if_low!();

                    match decode_varint(&read_buf[buf_pos..buf_len]) {
                        Ok((delta, consumed)) => break (delta, consumed),
                        Err(VarIntError::Truncated(_)) => {
                            // Need more data - shift and read more
                            read_buf.copy_within(buf_pos..buf_len, 0);
                            buf_len -= buf_pos;
                            buf_pos = 0;
                            let n = decoder.read(&mut read_buf[buf_len..])?;
                            if n == 0 {
                                return Err(anyhow!(
                                    "Truncated varint at bucket {} minimizer {} (EOF with continuation bit set)",
                                    id, i
                                ));
                            }
                            buf_len += n;
                        }
                        Err(VarIntError::Overflow(bytes)) => {
                            return Err(anyhow!(
                                "Malformed varint at bucket {} minimizer {}: exceeded 10 bytes (consumed {})",
                                id, i, bytes
                            ));
                        }
                    }
                };
                buf_pos += consumed;

                let val = prev.checked_add(delta).ok_or_else(|| {
                    anyhow!(
                        "Minimizer overflow at bucket {} index {} (prev={}, delta={})",
                        id,
                        i,
                        prev,
                        delta
                    )
                })?;
                vec.push(val);
                prev = val;
            }

            buckets.insert(id, vec);
        }

        Ok(Index {
            k,
            w,
            salt,
            buckets,
            bucket_names: names,
            bucket_sources: sources,
        })
    }

    /// Load only metadata from an index file (v5 format).
    ///
    /// This is much faster than load() because in v5 format all metadata is stored
    /// uncompressed before the compressed minimizers, so we can read just the metadata
    /// section and stop without touching the compressed data.
    pub fn load_metadata(path: &Path) -> Result<IndexMetadata> {
        let mut reader = BufReader::new(File::open(path)?);
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        reader.read_exact(&mut buf4)?;
        if &buf4 != SINGLE_FILE_INDEX_MAGIC {
            return Err(anyhow!("Invalid Index Format (Expected RYP5)"));
        }

        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != SINGLE_FILE_INDEX_VERSION {
            return Err(anyhow!(
                "Unsupported index version: {} (expected {})",
                version,
                SINGLE_FILE_INDEX_VERSION
            ));
        }

        reader.read_exact(&mut buf8)?;
        let k = u64::from_le_bytes(buf8) as usize;
        if !matches!(k, 16 | 32 | 64) {
            return Err(anyhow!(
                "Invalid K value in index: {} (must be 16, 32, or 64)",
                k
            ));
        }
        reader.read_exact(&mut buf8)?;
        let w = u64::from_le_bytes(buf8) as usize;
        reader.read_exact(&mut buf8)?;
        let salt = u64::from_le_bytes(buf8);
        reader.read_exact(&mut buf4)?;
        let num = u32::from_le_bytes(buf4);
        if num > MAX_NUM_BUCKETS {
            return Err(anyhow!(
                "Number of buckets {} exceeds maximum {}",
                num,
                MAX_NUM_BUCKETS
            ));
        }

        let mut names = HashMap::new();
        let mut sources = HashMap::new();
        let mut minimizer_counts = HashMap::new();

        // V5 format: all metadata is before the compressed minimizer section
        // No seeking needed - we just read the metadata and stop
        for _ in 0..num {
            reader.read_exact(&mut buf8)?;
            let vec_len = u64::from_le_bytes(buf8) as usize;
            if vec_len > MAX_BUCKET_SIZE {
                return Err(anyhow!(
                    "Bucket size {} exceeds maximum {}",
                    vec_len,
                    MAX_BUCKET_SIZE
                ));
            }

            reader.read_exact(&mut buf4)?;
            let id = u32::from_le_bytes(buf4);

            reader.read_exact(&mut buf8)?;
            let name_len = u64::from_le_bytes(buf8) as usize;
            if name_len > MAX_STRING_LENGTH {
                return Err(anyhow!(
                    "Bucket name length {} exceeds maximum {}",
                    name_len,
                    MAX_STRING_LENGTH
                ));
            }
            let mut nbuf = vec![0u8; name_len];
            reader.read_exact(&mut nbuf)?;
            names.insert(id, String::from_utf8(nbuf)?);

            reader.read_exact(&mut buf8)?;
            let src_count = u64::from_le_bytes(buf8) as usize;
            let mut src_list = Vec::with_capacity(src_count);
            for _ in 0..src_count {
                reader.read_exact(&mut buf8)?;
                let slen = u64::from_le_bytes(buf8) as usize;
                if slen > MAX_STRING_LENGTH {
                    return Err(anyhow!(
                        "Source string length {} exceeds maximum {}",
                        slen,
                        MAX_STRING_LENGTH
                    ));
                }
                let mut sbuf = vec![0u8; slen];
                reader.read_exact(&mut sbuf)?;
                src_list.push(String::from_utf8(sbuf)?);
            }
            sources.insert(id, src_list);

            // Store minimizer count (no need to seek - minimizers are after all metadata)
            minimizer_counts.insert(id, vec_len);
        }

        Ok(IndexMetadata {
            k,
            w,
            salt,
            bucket_names: names,
            bucket_sources: sources,
            bucket_minimizer_counts: minimizer_counts,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::MAX_BUCKET_SIZE;
    use tempfile::NamedTempFile;

    #[test]
    fn test_index_new_valid_k() {
        assert!(Index::new(16, 10, 0).is_ok());
        assert!(Index::new(32, 10, 0).is_ok());
        assert!(Index::new(64, 10, 0).is_ok());
    }

    #[test]
    fn test_index_new_invalid_k() {
        assert!(Index::new(15, 10, 0).is_err());
        assert!(Index::new(48, 10, 0).is_err());
    }

    #[test]
    fn test_index_save_load_roundtrip() -> Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();
        let mut index = Index::new(64, 50, 0x1234).unwrap();

        index.buckets.insert(1, vec![100, 200]);
        index.buckets.insert(2, vec![300, 400]);
        index.bucket_names.insert(1, "BucketA".into());
        index.bucket_names.insert(2, "BucketB".into());
        index
            .bucket_sources
            .insert(1, vec!["file1.fa::seq1".into()]);

        index.save(&path)?;
        let loaded = Index::load(&path)?;

        assert_eq!(loaded.w, 50);
        assert_eq!(loaded.salt, 0x1234);
        assert_eq!(loaded.buckets.len(), 2);
        assert_eq!(loaded.buckets[&1], vec![100, 200]);
        assert_eq!(loaded.bucket_sources[&1][0], "file1.fa::seq1");
        Ok(())
    }

    #[test]
    fn test_add_record_logic() {
        let mut index = Index::new(64, 5, 0).unwrap();
        let mut ws = MinimizerWorkspace::new();
        let seq = vec![b'A'; 70];

        index.add_record(1, "test_seq", &seq, &mut ws);
        index.finalize_bucket(1);

        assert!(index.buckets.contains_key(&1));
        assert!(!index.buckets[&1].is_empty());
        assert_eq!(index.bucket_sources[&1][0], "test_seq");
    }

    #[test]
    fn test_index_load_invalid_format() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path();
        std::fs::write(path, b"NOTRYP5").unwrap();

        let result = Index::load(path);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid Index Format"));
    }

    #[test]
    fn test_index_load_unsupported_version() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path();

        let mut data = b"RYP5".to_vec();
        data.extend_from_slice(&999u32.to_le_bytes());
        std::fs::write(path, data).unwrap();

        let result = Index::load(path);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Unsupported index version"));
    }

    #[test]
    fn test_index_load_oversized_bucket() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path();

        let mut data = Vec::new();
        data.extend_from_slice(b"RYP5");
        data.extend_from_slice(&5u32.to_le_bytes());
        data.extend_from_slice(&64u64.to_le_bytes());
        data.extend_from_slice(&50u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&(MAX_BUCKET_SIZE as u64 + 1).to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&4u64.to_le_bytes());
        data.extend_from_slice(b"test");
        data.extend_from_slice(&0u64.to_le_bytes());

        std::fs::write(path, data).unwrap();

        let result = Index::load(path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exceeds maximum"));
    }

    #[test]
    fn test_next_id_overflow() {
        let mut index = Index::new(64, 50, 0).unwrap();
        index.buckets.insert(u32::MAX, vec![]);

        let result = index.next_id();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("overflow"));
    }

    #[test]
    fn test_multiple_records_single_bucket() {
        let mut index = Index::new(64, 10, 0).unwrap();
        let mut ws = MinimizerWorkspace::new();

        let seq1 = vec![b'A'; 80];
        let seq2 = vec![b'T'; 80];
        let seq3 = vec![b'G'; 80];

        index.add_record(1, "file.fa::seq1", &seq1, &mut ws);
        index.add_record(1, "file.fa::seq2", &seq2, &mut ws);
        index.add_record(1, "file.fa::seq3", &seq3, &mut ws);
        index.finalize_bucket(1);

        assert!(index.buckets.contains_key(&1));
        assert_eq!(index.bucket_sources[&1].len(), 3);
        assert_eq!(index.bucket_sources[&1][0], "file.fa::seq1");
        assert_eq!(index.bucket_sources[&1][1], "file.fa::seq2");
        assert_eq!(index.bucket_sources[&1][2], "file.fa::seq3");
        assert!(!index.buckets[&1].is_empty());
    }

    #[test]
    fn test_bucket_naming_consistency() -> Result<()> {
        let mut index = Index::new(64, 50, 0).unwrap();
        let mut ws = MinimizerWorkspace::new();

        let bucket_id = 1;
        let filename = "reference.fasta";

        index.bucket_names.insert(bucket_id, filename.to_string());

        let seq1 = vec![b'A'; 80];
        let seq2 = vec![b'T'; 80];

        index.add_record(bucket_id, &format!("{}::seq1", filename), &seq1, &mut ws);
        index.add_record(bucket_id, &format!("{}::seq2", filename), &seq2, &mut ws);
        index.finalize_bucket(bucket_id);

        assert_eq!(index.bucket_names[&bucket_id], filename);
        assert_eq!(index.bucket_sources[&bucket_id].len(), 2);
        assert!(index.bucket_sources[&bucket_id][0].contains("seq1"));
        assert!(index.bucket_sources[&bucket_id][1].contains("seq2"));

        Ok(())
    }

    #[test]
    fn test_load_metadata_fast() -> Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();
        let mut index = Index::new(64, 50, 0x1234)?;

        index.buckets.insert(1, vec![100, 200, 300]);
        index.bucket_names.insert(1, "BucketA".into());
        index
            .bucket_sources
            .insert(1, vec!["file1.fa::seq1".into(), "file1.fa::seq2".into()]);

        index.buckets.insert(2, vec![400, 500]);
        index.bucket_names.insert(2, "BucketB".into());
        index
            .bucket_sources
            .insert(2, vec!["file2.fa::seq1".into()]);

        index.buckets.insert(3, vec![600, 700, 800, 900]);
        index.bucket_names.insert(3, "BucketC".into());
        index.bucket_sources.insert(
            3,
            vec![
                "file3.fa::seq1".into(),
                "file3.fa::seq2".into(),
                "file3.fa::seq3".into(),
            ],
        );

        index.save(&path)?;

        let metadata = Index::load_metadata(&path)?;

        assert_eq!(metadata.k, 64);
        assert_eq!(metadata.w, 50);
        assert_eq!(metadata.salt, 0x1234);
        assert_eq!(metadata.bucket_names.len(), 3);
        assert_eq!(metadata.bucket_names[&1], "BucketA");
        assert_eq!(metadata.bucket_names[&2], "BucketB");
        assert_eq!(metadata.bucket_names[&3], "BucketC");
        assert_eq!(metadata.bucket_sources[&1].len(), 2);
        assert_eq!(metadata.bucket_sources[&1][0], "file1.fa::seq1");
        assert_eq!(metadata.bucket_sources[&2].len(), 1);
        assert_eq!(metadata.bucket_sources[&3].len(), 3);
        assert_eq!(metadata.bucket_minimizer_counts[&1], 3);
        assert_eq!(metadata.bucket_minimizer_counts[&2], 2);
        assert_eq!(metadata.bucket_minimizer_counts[&3], 4);

        Ok(())
    }

    #[test]
    fn test_load_metadata_matches_full_load() -> Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();
        let mut index = Index::new(64, 42, 0xABCD).unwrap();

        index.buckets.insert(10, vec![1, 2, 3, 4, 5]);
        index.bucket_names.insert(10, "Test".into());
        index
            .bucket_sources
            .insert(10, vec!["src1".into(), "src2".into()]);

        index.save(&path)?;

        let metadata = Index::load_metadata(&path)?;
        let full_index = Index::load(&path)?;

        assert_eq!(metadata.w, full_index.w);
        assert_eq!(metadata.salt, full_index.salt);
        assert_eq!(metadata.bucket_names, full_index.bucket_names);
        assert_eq!(metadata.bucket_sources, full_index.bucket_sources);
        assert_eq!(
            metadata.bucket_minimizer_counts[&10],
            full_index.buckets[&10].len()
        );

        Ok(())
    }

    #[test]
    fn test_load_metadata_empty_index() -> Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();
        let index = Index::new(64, 25, 0).unwrap();

        index.save(&path)?;
        let metadata = Index::load_metadata(&path)?;

        assert_eq!(metadata.w, 25);
        assert_eq!(metadata.salt, 0);
        assert!(metadata.bucket_names.is_empty());
        assert!(metadata.bucket_sources.is_empty());
        assert!(metadata.bucket_minimizer_counts.is_empty());

        Ok(())
    }

    #[test]
    fn test_v5_format_roundtrip() -> Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();
        let mut index = Index::new(64, 50, 0x1234)?;

        index.buckets.insert(1, vec![100, 200]);
        index.buckets.insert(2, vec![300, 400, 500]);
        index.bucket_names.insert(1, "BucketA".into());
        index.bucket_names.insert(2, "BucketB".into());
        index
            .bucket_sources
            .insert(1, vec!["file1.fa::seq1".into()]);
        index
            .bucket_sources
            .insert(2, vec!["file2.fa::seq1".into(), "file2.fa::seq2".into()]);

        index.save(&path)?;
        let loaded = Index::load(&path)?;

        assert_eq!(loaded.k, 64);
        assert_eq!(loaded.w, 50);
        assert_eq!(loaded.salt, 0x1234);
        assert_eq!(loaded.buckets.len(), 2);
        assert_eq!(loaded.buckets[&1], vec![100, 200]);
        assert_eq!(loaded.buckets[&2], vec![300, 400, 500]);
        assert_eq!(loaded.bucket_names[&1], "BucketA");
        assert_eq!(loaded.bucket_sources[&1][0], "file1.fa::seq1");
        assert_eq!(loaded.bucket_sources[&2].len(), 2);

        Ok(())
    }

    #[test]
    fn test_index_k16() -> Result<()> {
        let mut index = Index::new(16, 50, 0)?;
        assert_eq!(index.k, 16);

        let mut ws = MinimizerWorkspace::new();
        let seq = vec![b'A'; 70];
        index.add_record(1, "test", &seq, &mut ws);
        index.finalize_bucket(1);

        assert!(index.buckets.contains_key(&1));
        assert!(!index.buckets[&1].is_empty());
        Ok(())
    }

    #[test]
    fn test_index_k32() -> Result<()> {
        let mut index = Index::new(32, 50, 0)?;
        assert_eq!(index.k, 32);

        let mut ws = MinimizerWorkspace::new();
        let seq = vec![b'A'; 85];
        index.add_record(1, "test", &seq, &mut ws);
        index.finalize_bucket(1);

        assert!(index.buckets.contains_key(&1));
        assert!(!index.buckets[&1].is_empty());
        Ok(())
    }

    #[test]
    fn test_reject_invalid_k() {
        assert!(Index::new(17, 50, 0).is_err());
        assert!(Index::new(48, 50, 0).is_err());
        assert!(Index::new(65, 50, 0).is_err());
        assert!(Index::new(0, 50, 0).is_err());
        assert!(Index::new(15, 50, 0).is_err());
        assert!(Index::new(33, 50, 0).is_err());
    }

    #[test]
    fn test_save_load_k32() -> Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();
        let mut index = Index::new(32, 50, 0x1234)?;

        index.buckets.insert(1, vec![100, 200]);
        index.bucket_names.insert(1, "TestBucket".into());

        index.save(&path)?;
        let loaded = Index::load(&path)?;

        assert_eq!(loaded.k, 32);
        assert_eq!(loaded.w, 50);
        assert_eq!(loaded.salt, 0x1234);
        assert_eq!(loaded.buckets[&1], vec![100, 200]);

        Ok(())
    }

    #[test]
    fn test_save_load_k16() -> Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();
        let mut index = Index::new(16, 25, 0xABCD)?;

        index.buckets.insert(1, vec![10, 20, 30]);
        index.bucket_names.insert(1, "K16Bucket".into());

        index.save(&path)?;
        let loaded = Index::load(&path)?;

        assert_eq!(loaded.k, 16);
        assert_eq!(loaded.w, 25);
        assert_eq!(loaded.salt, 0xABCD);
        assert_eq!(loaded.buckets[&1], vec![10, 20, 30]);

        Ok(())
    }

    #[test]
    fn test_load_invalid_k_from_file() -> Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path();

        let mut data = Vec::new();
        data.extend_from_slice(b"RYP5");
        data.extend_from_slice(&5u32.to_le_bytes());
        data.extend_from_slice(&48u64.to_le_bytes()); // Invalid K=48
        data.extend_from_slice(&50u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());

        std::fs::write(path, data)?;

        let result = Index::load(path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid K value"));

        Ok(())
    }
}
