//! Sharded main index structures.
//!
//! For very large indices that exceed available memory during creation, the main
//! index can be split into multiple shard files. Unlike sharded inverted indices
//! (which shard by minimizer value range), sharded main indices shard by bucket
//! assignment - each shard contains complete buckets.
//!
//! Key differences from sharded inverted index:
//! - Shards contain complete buckets (all minimizers for assigned buckets)
//! - Sharding is by memory budget, not by minimizer count
//! - 1:1 correspondence possible with inverted index shards

use anyhow::{anyhow, Context, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use crate::constants::{
    BUCKET_SOURCE_DELIM, BYTES_PER_MINIMIZER_COMPRESSED, BYTES_PER_MINIMIZER_MEMORY,
    MAIN_MANIFEST_MAGIC, MAIN_MANIFEST_VERSION, MAIN_SHARD_MAGIC, MAIN_SHARD_VERSION,
    MAX_BUCKET_SIZE, MAX_MAIN_SHARDS, MAX_NUM_BUCKETS, MAX_SOURCES_PER_BUCKET, MAX_STRING_LENGTH,
    MAX_STRING_TABLE_BYTES, MAX_STRING_TABLE_ENTRIES, READ_BUF_SIZE, WRITE_BUF_SIZE,
};
use crate::encoding::{decode_varint, encode_varint, VarIntError};
use crate::types::IndexMetadata;

/// Information about a single shard in a sharded main index.
#[derive(Debug, Clone)]
pub struct MainIndexShardInfo {
    /// Shard identifier (0-indexed)
    pub shard_id: u32,
    /// Bucket IDs contained in this shard
    pub bucket_ids: Vec<u32>,
    /// Total number of minimizers in this shard
    pub num_minimizers: usize,
    /// Compressed size in bytes (0 if not yet written)
    pub compressed_size: u64,
}

/// Manifest describing a sharded main index.
///
/// Format v2 (current):
/// - Magic: "RYPM" (4 bytes)
/// - Version: 2 (u32)
/// - k (u64), w (u64), salt (u64)
/// - num_buckets (u32), num_shards (u32), total_minimizers (u64)
/// - For each bucket (sorted by ID): bucket_id (u32), shard_id (u32), minimizer_count (u64),
///   name_len (u64), name bytes
/// - For each shard: shard_id (u32), bucket_count (u32), num_minimizers (u64), compressed_size (u64)
///
/// Note: Sources are stored in shard files, not in the manifest (moved in v2 for size reduction).
/// Format v1 is no longer supported; re-index with current version if you have v1 files.
#[derive(Debug, Clone)]
pub struct MainIndexManifest {
    pub k: usize,
    pub w: usize,
    pub salt: u64,
    pub bucket_names: HashMap<u32, String>,
    pub bucket_minimizer_counts: HashMap<u32, usize>,
    pub bucket_to_shard: HashMap<u32, u32>,
    pub shards: Vec<MainIndexShardInfo>,
    pub total_minimizers: usize,
}

impl MainIndexManifest {
    /// Save the manifest to a file (v2 format - sources stored in shards).
    pub fn save(&self, path: &Path) -> Result<()> {
        // Write to a temporary file, then atomically rename for crash safety
        let temp_path = path.with_extension("tmp");
        let mut writer = BufWriter::new(File::create(&temp_path)?);

        // Header
        writer.write_all(MAIN_MANIFEST_MAGIC)?;
        writer.write_all(&MAIN_MANIFEST_VERSION.to_le_bytes())?;

        writer.write_all(&(self.k as u64).to_le_bytes())?;
        writer.write_all(&(self.w as u64).to_le_bytes())?;
        writer.write_all(&self.salt.to_le_bytes())?;

        let num_buckets = self.bucket_names.len() as u32;
        writer.write_all(&num_buckets.to_le_bytes())?;
        writer.write_all(&(self.shards.len() as u32).to_le_bytes())?;
        writer.write_all(&(self.total_minimizers as u64).to_le_bytes())?;

        // Bucket metadata (sorted by ID) - no sources in v2
        let mut sorted_bucket_ids: Vec<_> = self.bucket_names.keys().copied().collect();
        sorted_bucket_ids.sort_unstable();

        for bucket_id in &sorted_bucket_ids {
            writer.write_all(&bucket_id.to_le_bytes())?;

            let shard_id = self.bucket_to_shard.get(bucket_id).copied().unwrap_or(0);
            writer.write_all(&shard_id.to_le_bytes())?;

            let minimizer_count = self
                .bucket_minimizer_counts
                .get(bucket_id)
                .copied()
                .unwrap_or(0);
            writer.write_all(&(minimizer_count as u64).to_le_bytes())?;

            // Name
            let name = self
                .bucket_names
                .get(bucket_id)
                .map(|s| s.as_str())
                .unwrap_or("");
            let name_bytes = name.as_bytes();
            writer.write_all(&(name_bytes.len() as u64).to_le_bytes())?;
            writer.write_all(name_bytes)?;
        }

        // Shard info
        for shard in &self.shards {
            writer.write_all(&shard.shard_id.to_le_bytes())?;
            writer.write_all(&(shard.bucket_ids.len() as u32).to_le_bytes())?;
            writer.write_all(&(shard.num_minimizers as u64).to_le_bytes())?;
            writer.write_all(&shard.compressed_size.to_le_bytes())?;
        }

        writer.flush()?;

        // fsync to ensure data is persisted to disk
        let file = writer.into_inner()?;
        file.sync_all()?;

        // Drop the file handle before rename
        drop(file);

        // Atomically rename temp file to final path
        std::fs::rename(&temp_path, path)?;

        Ok(())
    }

    /// Load a manifest from a file (supports v1 and v2 formats).
    ///
    /// Note: v1 manifests stored sources in the manifest; these are skipped on load
    /// since sources are now stored in shard files. Re-index to get source support.
    pub fn load(path: &Path) -> Result<Self> {
        let mut reader = BufReader::new(File::open(path)?);
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        // Header
        reader.read_exact(&mut buf4)?;
        if &buf4 != MAIN_MANIFEST_MAGIC {
            return Err(anyhow!(
                "Invalid main index manifest format (expected RYPM)"
            ));
        }

        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != MAIN_MANIFEST_VERSION {
            return Err(anyhow!(
                "Unsupported main index manifest version: {} (expected {}).\n\
                 Version 1 manifests are no longer supported. Re-create the sharded index:\n  \
                 rype index shard -i <single-file.ryidx> -o <output.ryidx> --max-shard-size <size>",
                version,
                MAIN_MANIFEST_VERSION
            ));
        }

        reader.read_exact(&mut buf8)?;
        let k = u64::from_le_bytes(buf8) as usize;
        if !matches!(k, 16 | 32 | 64) {
            return Err(anyhow!(
                "Invalid K value in manifest: {} (must be 16, 32, or 64)",
                k
            ));
        }

        reader.read_exact(&mut buf8)?;
        let w = u64::from_le_bytes(buf8) as usize;

        reader.read_exact(&mut buf8)?;
        let salt = u64::from_le_bytes(buf8);

        reader.read_exact(&mut buf4)?;
        let num_buckets = u32::from_le_bytes(buf4);
        if num_buckets > MAX_NUM_BUCKETS {
            return Err(anyhow!(
                "Number of buckets {} exceeds maximum {}",
                num_buckets,
                MAX_NUM_BUCKETS
            ));
        }

        reader.read_exact(&mut buf4)?;
        let num_shards = u32::from_le_bytes(buf4);
        if num_shards > MAX_MAIN_SHARDS {
            return Err(anyhow!(
                "Number of shards {} exceeds maximum {}",
                num_shards,
                MAX_MAIN_SHARDS
            ));
        }

        reader.read_exact(&mut buf8)?;
        let total_minimizers = u64::from_le_bytes(buf8) as usize;

        // Bucket metadata
        let mut bucket_names = HashMap::new();
        let mut bucket_minimizer_counts = HashMap::new();
        let mut bucket_to_shard = HashMap::new();

        for _ in 0..num_buckets {
            reader.read_exact(&mut buf4)?;
            let bucket_id = u32::from_le_bytes(buf4);

            reader.read_exact(&mut buf4)?;
            let shard_id = u32::from_le_bytes(buf4);

            reader.read_exact(&mut buf8)?;
            let minimizer_count = u64::from_le_bytes(buf8) as usize;
            if minimizer_count > MAX_BUCKET_SIZE {
                return Err(anyhow!(
                    "Bucket {} minimizer count {} exceeds maximum {}",
                    bucket_id,
                    minimizer_count,
                    MAX_BUCKET_SIZE
                ));
            }

            // Name
            reader.read_exact(&mut buf8)?;
            let name_len = u64::from_le_bytes(buf8) as usize;
            if name_len > MAX_STRING_LENGTH {
                return Err(anyhow!(
                    "Bucket name length {} exceeds maximum {}",
                    name_len,
                    MAX_STRING_LENGTH
                ));
            }
            let mut name_buf = vec![0u8; name_len];
            reader.read_exact(&mut name_buf)?;
            let name = String::from_utf8(name_buf)?;

            bucket_names.insert(bucket_id, name);
            bucket_minimizer_counts.insert(bucket_id, minimizer_count);
            bucket_to_shard.insert(bucket_id, shard_id);
        }

        // Shard info
        let mut shards = Vec::with_capacity(num_shards as usize);
        for _ in 0..num_shards {
            reader.read_exact(&mut buf4)?;
            let shard_id = u32::from_le_bytes(buf4);

            reader.read_exact(&mut buf4)?;
            let bucket_count = u32::from_le_bytes(buf4);

            reader.read_exact(&mut buf8)?;
            let num_minimizers = u64::from_le_bytes(buf8) as usize;

            reader.read_exact(&mut buf8)?;
            let compressed_size = u64::from_le_bytes(buf8);

            // Reconstruct bucket_ids from bucket_to_shard
            let bucket_ids: Vec<u32> = bucket_to_shard
                .iter()
                .filter(|(_, &s)| s == shard_id)
                .map(|(&b, _)| b)
                .collect();

            if bucket_ids.len() != bucket_count as usize {
                return Err(anyhow!(
                    "Shard {} bucket count mismatch: expected {}, found {}",
                    shard_id,
                    bucket_count,
                    bucket_ids.len()
                ));
            }

            shards.push(MainIndexShardInfo {
                shard_id,
                bucket_ids,
                num_minimizers,
                compressed_size,
            });
        }

        // Validate shard IDs are sequential
        for (i, shard) in shards.iter().enumerate() {
            if shard.shard_id != i as u32 {
                return Err(anyhow!(
                    "Invalid manifest: shard IDs not sequential (expected {}, found {})",
                    i,
                    shard.shard_id
                ));
            }
        }

        // Validate total minimizers
        let sum_minimizers: usize = shards.iter().map(|s| s.num_minimizers).sum();
        if sum_minimizers != total_minimizers {
            return Err(anyhow!(
                "Invalid manifest: shard minimizer counts sum to {}, expected {}",
                sum_minimizers,
                total_minimizers
            ));
        }

        Ok(MainIndexManifest {
            k,
            w,
            salt,
            bucket_names,
            bucket_minimizer_counts,
            bucket_to_shard,
            shards,
            total_minimizers,
        })
    }

    /// Get the path for the manifest file given a base path.
    pub fn manifest_path(base: &Path) -> PathBuf {
        let mut path = base.to_path_buf();
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        path.set_file_name(format!("{}.manifest", name));
        path
    }

    /// Get the path for a shard file given a base path and shard ID.
    pub fn shard_path(base: &Path, shard_id: u32) -> PathBuf {
        let mut path = base.to_path_buf();
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        path.set_file_name(format!("{}.shard.{}", name, shard_id));
        path
    }

    /// Check if a path appears to be a sharded main index (has .manifest file).
    pub fn is_sharded(base: &Path) -> bool {
        Self::manifest_path(base).exists()
    }

    /// Convert to IndexMetadata for compatibility with existing code.
    ///
    /// Note: bucket_sources will be empty since sources are stored in shards.
    /// Use ShardedMainIndex::get_bucket_sources() to retrieve sources.
    pub fn to_metadata(&self) -> IndexMetadata {
        IndexMetadata {
            k: self.k,
            w: self.w,
            salt: self.salt,
            bucket_names: self.bucket_names.clone(),
            bucket_sources: HashMap::new(), // Sources are in shards, not manifest
            bucket_minimizer_counts: self.bucket_minimizer_counts.clone(),
        }
    }
}

/// A single loaded shard containing a subset of buckets.
///
/// Shards now store both minimizers and sources (v2 format).
#[derive(Debug)]
pub struct MainIndexShard {
    pub k: usize,
    pub w: usize,
    pub salt: u64,
    pub shard_id: u32,
    pub buckets: HashMap<u32, Vec<u64>>,
    pub bucket_sources: HashMap<u32, Vec<String>>,
}

impl MainIndexShard {
    /// Save the shard to a file (RYPS v2 format with sources and string deduplication).
    ///
    /// Format v2:
    /// - Header (uncompressed): magic "RYPS", version 2, k, w, salt, shard_id, num_buckets
    /// - Compressed stream (zstd):
    ///   - String table: num_filenames (u32), then [len (varint), bytes]...
    ///   - For each bucket (sorted by ID):
    ///     - bucket_id (u32), minimizer_count (u64), source_count (u64)
    ///     - Sources: [filename_idx (varint), seqname_len (varint), seqname bytes]...
    ///     - Minimizers: first (u64), then deltas as varints
    pub fn save(&self, path: &Path) -> Result<u64> {
        // Write to a temporary file, then atomically rename for crash safety
        let temp_path = path.with_extension("tmp");
        let mut writer = BufWriter::new(File::create(&temp_path)?);

        // Header (uncompressed)
        writer.write_all(MAIN_SHARD_MAGIC)?;
        writer.write_all(&MAIN_SHARD_VERSION.to_le_bytes())?;
        writer.write_all(&(self.k as u64).to_le_bytes())?;
        writer.write_all(&(self.w as u64).to_le_bytes())?;
        writer.write_all(&self.salt.to_le_bytes())?;
        writer.write_all(&self.shard_id.to_le_bytes())?;
        writer.write_all(&(self.buckets.len() as u32).to_le_bytes())?;

        // Build string table from all sources (deduplicate filenames)
        let mut filename_to_idx: HashMap<String, u32> = HashMap::new();
        let mut filenames: Vec<String> = Vec::new();

        let empty_sources = Vec::new();
        for bucket_id in self.buckets.keys() {
            let sources = self.bucket_sources.get(bucket_id).unwrap_or(&empty_sources);
            for source in sources {
                let filename = source.split(BUCKET_SOURCE_DELIM).next().unwrap_or(source);
                if !filename_to_idx.contains_key(filename) {
                    // Check for string table overflow
                    let idx = u32::try_from(filenames.len()).map_err(|_| {
                        anyhow!(
                            "String table overflow: more than {} unique filenames",
                            u32::MAX
                        )
                    })?;
                    if idx >= MAX_STRING_TABLE_ENTRIES {
                        return Err(anyhow!(
                            "String table overflow: more than {} unique filenames",
                            MAX_STRING_TABLE_ENTRIES
                        ));
                    }
                    filename_to_idx.insert(filename.to_string(), idx);
                    filenames.push(filename.to_string());
                }
            }
        }

        // Compressed stream
        let mut encoder = zstd::stream::write::Encoder::new(writer, 3)?;

        let mut write_buf = Vec::with_capacity(WRITE_BUF_SIZE);
        let mut varint_buf = [0u8; 10];

        let flush_buf = |buf: &mut Vec<u8>,
                         encoder: &mut zstd::stream::write::Encoder<BufWriter<File>>|
         -> Result<()> {
            if !buf.is_empty() {
                encoder.write_all(buf)?;
                buf.clear();
            }
            Ok(())
        };

        // Write string table (with overflow check)
        let num_filenames = u32::try_from(filenames.len()).map_err(|_| {
            anyhow!(
                "String table overflow: more than {} unique filenames",
                u32::MAX
            )
        })?;
        write_buf.extend_from_slice(&num_filenames.to_le_bytes());
        for filename in &filenames {
            let bytes = filename.as_bytes();
            let len = encode_varint(bytes.len() as u64, &mut varint_buf);
            write_buf.extend_from_slice(&varint_buf[..len]);
            write_buf.extend_from_slice(bytes);
            if write_buf.len() >= WRITE_BUF_SIZE {
                flush_buf(&mut write_buf, &mut encoder)?;
            }
        }

        // Write buckets
        let mut sorted_ids: Vec<_> = self.buckets.keys().copied().collect();
        sorted_ids.sort_unstable();

        for id in &sorted_ids {
            let minimizers = &self.buckets[id];
            let sources = self.bucket_sources.get(id).unwrap_or(&empty_sources);

            // Bucket ID, minimizer count, source count
            write_buf.extend_from_slice(&id.to_le_bytes());
            write_buf.extend_from_slice(&(minimizers.len() as u64).to_le_bytes());
            write_buf.extend_from_slice(&(sources.len() as u64).to_le_bytes());

            // Sources with string deduplication
            for source in sources {
                let mut parts = source.splitn(2, BUCKET_SOURCE_DELIM);
                let filename = parts.next().unwrap_or(source);
                let seqname = parts.next().unwrap_or("");

                let filename_idx = filename_to_idx.get(filename).copied().unwrap_or(0);
                let len = encode_varint(filename_idx as u64, &mut varint_buf);
                write_buf.extend_from_slice(&varint_buf[..len]);

                let seqname_bytes = seqname.as_bytes();
                let len = encode_varint(seqname_bytes.len() as u64, &mut varint_buf);
                write_buf.extend_from_slice(&varint_buf[..len]);
                write_buf.extend_from_slice(seqname_bytes);

                if write_buf.len() >= WRITE_BUF_SIZE {
                    flush_buf(&mut write_buf, &mut encoder)?;
                }
            }

            // Minimizers
            if !minimizers.is_empty() {
                write_buf.extend_from_slice(&minimizers[0].to_le_bytes());
                if write_buf.len() >= WRITE_BUF_SIZE {
                    flush_buf(&mut write_buf, &mut encoder)?;
                }

                let mut prev = minimizers[0];
                for &val in &minimizers[1..] {
                    let delta = val - prev;
                    let len = encode_varint(delta, &mut varint_buf);
                    write_buf.extend_from_slice(&varint_buf[..len]);
                    if write_buf.len() >= WRITE_BUF_SIZE {
                        flush_buf(&mut write_buf, &mut encoder)?;
                    }
                    prev = val;
                }
            }
        }
        flush_buf(&mut write_buf, &mut encoder)?;

        let writer = encoder.finish()?;
        let file = writer.into_inner()?;

        // fsync to ensure data is persisted to disk
        file.sync_all()?;

        let compressed_size = file.metadata()?.len();

        // Drop the file handle before rename
        drop(file);

        // Atomically rename temp file to final path
        std::fs::rename(&temp_path, path)?;

        Ok(compressed_size)
    }

    /// Load a shard from a file (RYPS v2 format).
    pub fn load(path: &Path) -> Result<Self> {
        let mut reader = BufReader::new(File::open(path)?);
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        // Header
        reader.read_exact(&mut buf4)?;
        if &buf4 != MAIN_SHARD_MAGIC {
            return Err(anyhow!("Invalid main index shard format (expected RYPS)"));
        }

        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != MAIN_SHARD_VERSION {
            return Err(anyhow!(
                "Unsupported main index shard version: {} (expected {}).\n\
                 Version 1 shards are no longer supported. Re-create the sharded index:\n  \
                 rype index shard -i <single-file.ryidx> -o <output.ryidx> --max-shard-size <size>",
                version,
                MAIN_SHARD_VERSION
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

        reader.read_exact(&mut buf4)?;
        let shard_id = u32::from_le_bytes(buf4);

        reader.read_exact(&mut buf4)?;
        let num_buckets = u32::from_le_bytes(buf4);
        if num_buckets > MAX_NUM_BUCKETS {
            return Err(anyhow!(
                "Number of buckets {} exceeds maximum {}",
                num_buckets,
                MAX_NUM_BUCKETS
            ));
        }

        // Compressed stream
        let mut decoder = zstd::stream::read::Decoder::new(reader)
            .map_err(|e| anyhow!("Failed to create zstd decoder: {}", e))?;

        let mut read_buf = vec![0u8; READ_BUF_SIZE];
        let mut buf_pos = 0usize;
        let mut buf_len = 0usize;

        // Helper macros for buffered reading
        macro_rules! ensure_bytes {
            ($need:expr) => {{
                let need = $need;
                if buf_pos + need > buf_len {
                    if buf_pos > 0 {
                        read_buf.copy_within(buf_pos..buf_len, 0);
                        buf_len -= buf_pos;
                        buf_pos = 0;
                    }
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

        macro_rules! refill_if_low {
            () => {{
                if buf_pos >= buf_len {
                    buf_pos = 0;
                    let n = decoder.read(&mut read_buf)?;
                    if n == 0 {
                        return Err(anyhow!("Unexpected end of compressed data"));
                    }
                    buf_len = n;
                } else if buf_len - buf_pos < 10 {
                    read_buf.copy_within(buf_pos..buf_len, 0);
                    buf_len -= buf_pos;
                    buf_pos = 0;
                    let n = decoder.read(&mut read_buf[buf_len..])?;
                    buf_len += n;
                }
            }};
        }

        // Read string table
        ensure_bytes!(4);
        let num_filenames = u32::from_le_bytes(read_buf[buf_pos..buf_pos + 4].try_into().unwrap());
        buf_pos += 4;

        // Validate string table entry count
        if num_filenames > MAX_STRING_TABLE_ENTRIES {
            return Err(anyhow!(
                "String table has {} entries, exceeds maximum {}",
                num_filenames,
                MAX_STRING_TABLE_ENTRIES
            ));
        }

        let num_filenames = num_filenames as usize;
        let mut filenames = Vec::with_capacity(num_filenames);
        let mut total_string_bytes: usize = 0;

        for filename_idx in 0..num_filenames {
            // Decode varint with proper truncation handling
            let (str_len, consumed) = loop {
                refill_if_low!();
                match decode_varint(&read_buf[buf_pos..buf_len]) {
                    Ok((val, consumed)) => break (val, consumed),
                    Err(VarIntError::Truncated(_)) => {
                        // Need more data - shift and read more
                        read_buf.copy_within(buf_pos..buf_len, 0);
                        buf_len -= buf_pos;
                        buf_pos = 0;
                        let n = decoder.read(&mut read_buf[buf_len..])?;
                        if n == 0 {
                            return Err(anyhow!(
                                "Truncated varint at filename {} length",
                                filename_idx
                            ));
                        }
                        buf_len += n;
                    }
                    Err(VarIntError::Overflow(bytes)) => {
                        return Err(anyhow!(
                            "Malformed varint at filename {} length: exceeded 10 bytes (consumed {})",
                            filename_idx, bytes
                        ));
                    }
                }
            };
            buf_pos += consumed;

            // Validate string length with overflow-safe conversion
            let str_len = usize::try_from(str_len).map_err(|_| {
                anyhow!(
                    "Filename {} length {} overflows usize",
                    filename_idx,
                    str_len
                )
            })?;

            if str_len > MAX_STRING_LENGTH {
                return Err(anyhow!(
                    "Filename length {} exceeds maximum {}",
                    str_len,
                    MAX_STRING_LENGTH
                ));
            }

            // Track total bytes for DoS protection
            total_string_bytes = total_string_bytes.saturating_add(str_len);
            if total_string_bytes > MAX_STRING_TABLE_BYTES {
                return Err(anyhow!(
                    "String table exceeds {} bytes limit",
                    MAX_STRING_TABLE_BYTES
                ));
            }

            ensure_bytes!(str_len);
            let name = String::from_utf8(read_buf[buf_pos..buf_pos + str_len].to_vec())?;
            buf_pos += str_len;
            filenames.push(name);
        }

        let mut buckets = HashMap::new();
        let mut bucket_sources = HashMap::new();

        for _ in 0..num_buckets {
            // Read bucket ID
            ensure_bytes!(4);
            let bucket_id = u32::from_le_bytes(read_buf[buf_pos..buf_pos + 4].try_into().unwrap());
            buf_pos += 4;

            // Read minimizer count with overflow-safe conversion
            ensure_bytes!(8);
            let minimizer_count_u64 =
                u64::from_le_bytes(read_buf[buf_pos..buf_pos + 8].try_into().unwrap());
            buf_pos += 8;

            let minimizer_count = usize::try_from(minimizer_count_u64).map_err(|_| {
                anyhow!(
                    "Bucket {} minimizer count {} overflows usize",
                    bucket_id,
                    minimizer_count_u64
                )
            })?;

            if minimizer_count > MAX_BUCKET_SIZE {
                return Err(anyhow!(
                    "Bucket {} size {} exceeds maximum {}",
                    bucket_id,
                    minimizer_count,
                    MAX_BUCKET_SIZE
                ));
            }

            // Read source count and sources
            ensure_bytes!(8);
            let source_count_u64 =
                u64::from_le_bytes(read_buf[buf_pos..buf_pos + 8].try_into().unwrap());
            buf_pos += 8;

            let source_count = usize::try_from(source_count_u64).map_err(|_| {
                anyhow!(
                    "Bucket {} source count {} overflows usize",
                    bucket_id,
                    source_count_u64
                )
            })?;

            if source_count > MAX_SOURCES_PER_BUCKET {
                return Err(anyhow!(
                    "Bucket {} has {} sources, exceeds maximum {}",
                    bucket_id,
                    source_count,
                    MAX_SOURCES_PER_BUCKET
                ));
            }

            let mut sources = Vec::with_capacity(source_count);
            for source_idx in 0..source_count {
                // Decode filename index varint with proper truncation handling
                let (filename_idx_raw, consumed) = loop {
                    refill_if_low!();
                    match decode_varint(&read_buf[buf_pos..buf_len]) {
                        Ok((val, consumed)) => break (val, consumed),
                        Err(VarIntError::Truncated(_)) => {
                            read_buf.copy_within(buf_pos..buf_len, 0);
                            buf_len -= buf_pos;
                            buf_pos = 0;
                            let n = decoder.read(&mut read_buf[buf_len..])?;
                            if n == 0 {
                                return Err(anyhow!(
                                    "Truncated varint at bucket {} source {} filename index",
                                    bucket_id,
                                    source_idx
                                ));
                            }
                            buf_len += n;
                        }
                        Err(VarIntError::Overflow(bytes)) => {
                            return Err(anyhow!(
                                "Malformed varint at bucket {} source {} filename index: exceeded 10 bytes (consumed {})",
                                bucket_id, source_idx, bytes
                            ));
                        }
                    }
                };
                buf_pos += consumed;

                let filename_idx = usize::try_from(filename_idx_raw).map_err(|_| {
                    anyhow!(
                        "Bucket {} source {} filename index {} overflows usize",
                        bucket_id,
                        source_idx,
                        filename_idx_raw
                    )
                })?;

                // Validate filename index is in bounds
                if filename_idx >= filenames.len() {
                    return Err(anyhow!(
                        "Bucket {} source {} has invalid filename index {} (string table has {} entries)",
                        bucket_id, source_idx, filename_idx, filenames.len()
                    ));
                }

                // Decode seqname length varint with proper truncation handling
                let (seqname_len_raw, consumed) = loop {
                    refill_if_low!();
                    match decode_varint(&read_buf[buf_pos..buf_len]) {
                        Ok((val, consumed)) => break (val, consumed),
                        Err(VarIntError::Truncated(_)) => {
                            read_buf.copy_within(buf_pos..buf_len, 0);
                            buf_len -= buf_pos;
                            buf_pos = 0;
                            let n = decoder.read(&mut read_buf[buf_len..])?;
                            if n == 0 {
                                return Err(anyhow!(
                                    "Truncated varint at bucket {} source {} seqname length",
                                    bucket_id,
                                    source_idx
                                ));
                            }
                            buf_len += n;
                        }
                        Err(VarIntError::Overflow(bytes)) => {
                            return Err(anyhow!(
                                "Malformed varint at bucket {} source {} seqname length: exceeded 10 bytes (consumed {})",
                                bucket_id, source_idx, bytes
                            ));
                        }
                    }
                };
                buf_pos += consumed;

                let seqname_len = usize::try_from(seqname_len_raw).map_err(|_| {
                    anyhow!(
                        "Bucket {} source {} seqname length {} overflows usize",
                        bucket_id,
                        source_idx,
                        seqname_len_raw
                    )
                })?;

                if seqname_len > MAX_STRING_LENGTH {
                    return Err(anyhow!(
                        "Seqname length {} exceeds maximum {}",
                        seqname_len,
                        MAX_STRING_LENGTH
                    ));
                }

                ensure_bytes!(seqname_len);
                let seqname = String::from_utf8(read_buf[buf_pos..buf_pos + seqname_len].to_vec())?;
                buf_pos += seqname_len;

                // Reconstruct full source string (bounds already validated above)
                let filename = &filenames[filename_idx];
                let source = if seqname.is_empty() {
                    filename.clone()
                } else {
                    format!("{}{}{}", filename, BUCKET_SOURCE_DELIM, seqname)
                };
                sources.push(source);
            }

            // Read minimizers
            let mut minimizers = Vec::with_capacity(minimizer_count);
            if minimizer_count > 0 {
                ensure_bytes!(8);
                let first = u64::from_le_bytes(read_buf[buf_pos..buf_pos + 8].try_into().unwrap());
                buf_pos += 8;
                minimizers.push(first);

                let mut prev = first;
                for i in 1..minimizer_count {
                    // Decode minimizer delta varint with proper truncation handling
                    let (delta, consumed) = loop {
                        refill_if_low!();
                        match decode_varint(&read_buf[buf_pos..buf_len]) {
                            Ok((val, consumed)) => break (val, consumed),
                            Err(VarIntError::Truncated(_)) => {
                                read_buf.copy_within(buf_pos..buf_len, 0);
                                buf_len -= buf_pos;
                                buf_pos = 0;
                                let n = decoder.read(&mut read_buf[buf_len..])?;
                                if n == 0 {
                                    return Err(anyhow!(
                                        "Truncated varint at bucket {} minimizer {}",
                                        bucket_id,
                                        i
                                    ));
                                }
                                buf_len += n;
                            }
                            Err(VarIntError::Overflow(bytes)) => {
                                return Err(anyhow!(
                                    "Malformed varint at bucket {} minimizer {}: exceeded 10 bytes (consumed {})",
                                    bucket_id, i, bytes
                                ));
                            }
                        }
                    };
                    buf_pos += consumed;

                    let val = prev.checked_add(delta).ok_or_else(|| {
                        anyhow!(
                            "Minimizer overflow at bucket {} index {} (prev={}, delta={})",
                            bucket_id,
                            i,
                            prev,
                            delta
                        )
                    })?;
                    minimizers.push(val);
                    prev = val;
                }
            }

            buckets.insert(bucket_id, minimizers);
            if !sources.is_empty() {
                bucket_sources.insert(bucket_id, sources);
            }
        }

        Ok(MainIndexShard {
            k,
            w,
            salt,
            shard_id,
            buckets,
            bucket_sources,
        })
    }
}

/// Handle for a sharded main index.
///
/// This struct holds a manifest describing the shards. Shards are loaded
/// on-demand during classification via `classify_batch_sharded_main`.
#[derive(Debug)]
pub struct ShardedMainIndex {
    manifest: MainIndexManifest,
    base_path: PathBuf,
}

impl ShardedMainIndex {
    /// Open a sharded main index by loading just the manifest.
    pub fn open(base_path: &Path) -> Result<Self> {
        let manifest_path = MainIndexManifest::manifest_path(base_path);
        let manifest = MainIndexManifest::load(&manifest_path)?;

        Ok(ShardedMainIndex {
            manifest,
            base_path: base_path.to_path_buf(),
        })
    }

    /// Returns the K value (k-mer size).
    pub fn k(&self) -> usize {
        self.manifest.k
    }

    /// Returns the window size.
    pub fn w(&self) -> usize {
        self.manifest.w
    }

    /// Returns the salt value.
    pub fn salt(&self) -> u64 {
        self.manifest.salt
    }

    /// Returns the total number of shards.
    pub fn num_shards(&self) -> usize {
        self.manifest.shards.len()
    }

    /// Returns the total number of minimizers across all shards.
    pub fn total_minimizers(&self) -> usize {
        self.manifest.total_minimizers
    }

    /// Returns a reference to the manifest.
    pub fn manifest(&self) -> &MainIndexManifest {
        &self.manifest
    }

    /// Returns a reference to the base path.
    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    /// Load a specific shard by ID.
    pub fn load_shard(&self, shard_id: u32) -> Result<MainIndexShard> {
        if shard_id as usize >= self.manifest.shards.len() {
            return Err(anyhow!(
                "Shard ID {} out of range (max {})",
                shard_id,
                self.manifest.shards.len() - 1
            ));
        }
        let shard_path = MainIndexManifest::shard_path(&self.base_path, shard_id);
        MainIndexShard::load(&shard_path)
    }

    /// Convert manifest to IndexMetadata for compatibility.
    pub fn metadata(&self) -> IndexMetadata {
        self.manifest.to_metadata()
    }

    /// Add a new bucket as a new shard.
    ///
    /// Creates a new shard file containing the single bucket, updates the manifest.
    pub fn add_bucket(
        &mut self,
        bucket_id: u32,
        name: &str,
        sources: Vec<String>,
        minimizers: Vec<u64>,
    ) -> Result<()> {
        if self.manifest.bucket_names.contains_key(&bucket_id) {
            return Err(anyhow!("Bucket {} already exists in the index", bucket_id));
        }

        // Create new shard with sources
        let new_shard_id = self.manifest.shards.len() as u32;
        let minimizer_count = minimizers.len();

        let shard = MainIndexShard {
            k: self.manifest.k,
            w: self.manifest.w,
            salt: self.manifest.salt,
            shard_id: new_shard_id,
            buckets: {
                let mut buckets = HashMap::new();
                buckets.insert(bucket_id, minimizers);
                buckets
            },
            bucket_sources: {
                let mut bucket_sources = HashMap::new();
                bucket_sources.insert(bucket_id, sources);
                bucket_sources
            },
        };

        // Save new shard
        let shard_path = MainIndexManifest::shard_path(&self.base_path, new_shard_id);
        let compressed_size = shard.save(&shard_path)?;

        // Update manifest (sources stored in shard, not manifest)
        self.manifest
            .bucket_names
            .insert(bucket_id, name.to_string());
        self.manifest
            .bucket_minimizer_counts
            .insert(bucket_id, minimizer_count);
        self.manifest
            .bucket_to_shard
            .insert(bucket_id, new_shard_id);
        self.manifest.total_minimizers += minimizer_count;

        self.manifest.shards.push(MainIndexShardInfo {
            shard_id: new_shard_id,
            bucket_ids: vec![bucket_id],
            num_minimizers: minimizer_count,
            compressed_size,
        });

        // Save updated manifest
        let manifest_path = MainIndexManifest::manifest_path(&self.base_path);
        self.manifest.save(&manifest_path)?;

        Ok(())
    }

    /// Update an existing bucket by adding new minimizers and sources.
    ///
    /// Loads the shard containing the bucket, extends its minimizers and sources,
    /// sorts and deduplicates, then saves the shard back.
    pub fn update_bucket(
        &mut self,
        bucket_id: u32,
        new_sources: Vec<String>,
        new_minimizers: Vec<u64>,
    ) -> Result<()> {
        if !self.manifest.bucket_names.contains_key(&bucket_id) {
            return Err(anyhow!("Bucket {} does not exist in the index", bucket_id));
        }

        let shard_id = *self
            .manifest
            .bucket_to_shard
            .get(&bucket_id)
            .ok_or_else(|| anyhow!("Bucket {} not mapped to any shard", bucket_id))?;

        // Load the shard
        let shard_path = MainIndexManifest::shard_path(&self.base_path, shard_id);
        let mut shard = MainIndexShard::load(&shard_path)?;

        // Extend minimizers
        let bucket_mins = shard.buckets.entry(bucket_id).or_default();
        let old_count = bucket_mins.len();
        bucket_mins.extend(new_minimizers);
        bucket_mins.sort_unstable();
        bucket_mins.dedup();
        let new_count = bucket_mins.len();

        // Extend sources (stored in shard, not manifest)
        let sources = shard.bucket_sources.entry(bucket_id).or_default();
        sources.extend(new_sources);
        sources.sort_unstable();
        sources.dedup();

        // Save the shard
        let compressed_size = shard.save(&shard_path)?;

        // Update manifest - minimizer count only (no sources in manifest)
        let added_minimizers = new_count.saturating_sub(old_count);
        self.manifest
            .bucket_minimizer_counts
            .insert(bucket_id, new_count);
        self.manifest.total_minimizers += added_minimizers;

        // Update shard info
        if let Some(shard_info) = self
            .manifest
            .shards
            .iter_mut()
            .find(|s| s.shard_id == shard_id)
        {
            shard_info.num_minimizers =
                shard_info.num_minimizers.saturating_sub(old_count) + new_count;
            shard_info.compressed_size = compressed_size;
        }

        // Save updated manifest
        let manifest_path = MainIndexManifest::manifest_path(&self.base_path);
        self.manifest.save(&manifest_path)?;

        Ok(())
    }

    /// Get sources for a specific bucket by loading its shard.
    ///
    /// Returns the sources or an empty Vec if the bucket has no sources.
    pub fn get_bucket_sources(&self, bucket_id: u32) -> Result<Vec<String>> {
        let shard_id = *self
            .manifest
            .bucket_to_shard
            .get(&bucket_id)
            .ok_or_else(|| anyhow!("Bucket {} not found in index", bucket_id))?;

        let shard = self.load_shard(shard_id)?;
        Ok(shard
            .bucket_sources
            .get(&bucket_id)
            .cloned()
            .unwrap_or_default())
    }

    /// Get the next available bucket ID.
    pub fn next_id(&self) -> Result<u32> {
        let max_id = self
            .manifest
            .bucket_names
            .keys()
            .max()
            .copied()
            .unwrap_or(0);
        max_id
            .checked_add(1)
            .ok_or_else(|| anyhow!("Bucket ID overflow: maximum ID {} reached", max_id))
    }
}

/// Estimate the compressed byte size of a bucket (for output planning).
pub fn estimate_bucket_bytes(minimizer_count: usize) -> usize {
    minimizer_count * BYTES_PER_MINIMIZER_COMPRESSED
}

/// Estimate the in-memory byte size of a bucket (for memory budgeting).
///
/// This accounts for the actual memory footprint: 8 bytes per u64 minimizer.
/// Does not include source string overhead (unbounded).
pub fn estimate_bucket_memory_bytes(minimizer_count: usize) -> usize {
    minimizer_count * BYTES_PER_MINIMIZER_MEMORY
}

/// Plan shard assignment based on memory budget using first-fit-decreasing bin packing.
///
/// Returns: Vec of (shard_id, Vec<bucket_id>)
pub fn plan_shards(
    bucket_minimizer_counts: &HashMap<u32, usize>,
    max_shard_bytes: usize,
) -> Vec<(u32, Vec<u32>)> {
    // Sort buckets by estimated size (largest first) for best-fit packing
    let mut sorted: Vec<_> = bucket_minimizer_counts
        .iter()
        .map(|(&id, &count)| (id, estimate_bucket_bytes(count)))
        .collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1)); // Descending by size

    let mut shards: Vec<(usize, Vec<u32>)> = vec![]; // (current_size, bucket_ids)

    for (bucket_id, bucket_bytes) in sorted {
        // Find first shard with space
        let mut placed = false;
        for (shard_size, shard_buckets) in &mut shards {
            if *shard_size + bucket_bytes <= max_shard_bytes {
                *shard_size += bucket_bytes;
                shard_buckets.push(bucket_id);
                placed = true;
                break;
            }
        }
        if !placed {
            // Start new shard (bucket may exceed budget if it's larger than max)
            shards.push((bucket_bytes, vec![bucket_id]));
        }
    }

    // Assign sequential shard IDs and sort bucket IDs within each shard
    shards
        .into_iter()
        .enumerate()
        .map(|(i, (_, mut buckets))| {
            buckets.sort_unstable();
            (i as u32, buckets)
        })
        .collect()
}

/// Builder for creating a sharded main index with write-as-you-go semantics.
///
/// This builder accumulates buckets and writes shard files to disk when the
/// memory budget is exceeded, minimizing peak memory usage during index creation.
///
/// # Example
/// ```ignore
/// let mut builder = ShardedMainIndexBuilder::new(64, 50, 0x1234, &base_path, 1_000_000_000)?;
///
/// // Add buckets (will auto-flush shards when budget exceeded)
/// builder.add_bucket(1, "BucketA", vec!["src1".into()], minimizers1)?;
/// builder.add_bucket(2, "BucketB", vec!["src2".into()], minimizers2)?;
///
/// // Finalize: flush remaining buckets and write manifest
/// let manifest = builder.finish()?;
/// ```
pub struct ShardedMainIndexBuilder {
    k: usize,
    w: usize,
    salt: u64,
    base_path: PathBuf,
    max_shard_bytes: usize,

    // Current shard accumulator (minimizers and sources stored together)
    current_shard_id: u32,
    current_shard_buckets: HashMap<u32, Vec<u64>>,
    current_shard_sources: HashMap<u32, Vec<String>>,
    current_shard_bytes: usize,

    // Metadata accumulators (kept in memory for manifest - no sources)
    bucket_names: HashMap<u32, String>,
    bucket_minimizer_counts: HashMap<u32, usize>,
    bucket_to_shard: HashMap<u32, u32>,
    shards: Vec<MainIndexShardInfo>,
    total_minimizers: usize,

    // Temp files written (for atomic commit on finish)
    pending_shard_renames: Vec<(PathBuf, PathBuf)>, // (temp_path, final_path)
}

impl ShardedMainIndexBuilder {
    /// Create a new sharded index builder.
    ///
    /// # Arguments
    /// * `k` - K-mer size (must be 16, 32, or 64)
    /// * `w` - Window size for minimizer selection
    /// * `salt` - XOR salt applied to k-mer hashes
    /// * `base_path` - Base path for output files (e.g., "index.ryidx")
    /// * `max_shard_bytes` - Maximum estimated bytes per shard
    pub fn new(
        k: usize,
        w: usize,
        salt: u64,
        base_path: &Path,
        max_shard_bytes: usize,
    ) -> Result<Self> {
        if !matches!(k, 16 | 32 | 64) {
            return Err(anyhow!("K must be 16, 32, or 64 (got {})", k));
        }

        Ok(ShardedMainIndexBuilder {
            k,
            w,
            salt,
            base_path: base_path.to_path_buf(),
            max_shard_bytes,
            current_shard_id: 0,
            current_shard_buckets: HashMap::new(),
            current_shard_sources: HashMap::new(),
            current_shard_bytes: 0,
            bucket_names: HashMap::new(),
            bucket_minimizer_counts: HashMap::new(),
            bucket_to_shard: HashMap::new(),
            shards: Vec::new(),
            total_minimizers: 0,
            pending_shard_renames: Vec::new(),
        })
    }

    /// Add a bucket to the index.
    ///
    /// Minimizers are defensively sorted and deduplicated to ensure correctness.
    /// If the current shard exceeds the memory budget after adding this bucket,
    /// the shard is flushed to disk automatically.
    ///
    /// # Arguments
    /// * `id` - Bucket ID
    /// * `name` - Human-readable bucket name
    /// * `sources` - List of source sequences
    /// * `minimizers` - Minimizers (will be sorted and deduplicated)
    pub fn add_bucket(
        &mut self,
        id: u32,
        name: &str,
        sources: Vec<String>,
        mut minimizers: Vec<u64>,
    ) -> Result<()> {
        // Defensive: ensure minimizers are sorted and deduplicated
        minimizers.sort_unstable();
        minimizers.dedup();

        let bucket_bytes = estimate_bucket_memory_bytes(minimizers.len());
        let minimizer_count = minimizers.len();

        // Check if we need to flush current shard first
        // (but don't flush if this is the first bucket in a new shard)
        if !self.current_shard_buckets.is_empty()
            && self.current_shard_bytes + bucket_bytes > self.max_shard_bytes
        {
            self.flush_shard()?;
        }

        // Add bucket minimizers and sources to current shard
        self.current_shard_buckets.insert(id, minimizers);
        self.current_shard_sources.insert(id, sources);
        self.current_shard_bytes += bucket_bytes;

        // Store manifest metadata (no sources - they go in shards)
        self.bucket_names.insert(id, name.to_string());
        self.bucket_minimizer_counts.insert(id, minimizer_count);
        self.bucket_to_shard.insert(id, self.current_shard_id);
        self.total_minimizers += minimizer_count;

        Ok(())
    }

    /// Flush the current shard to disk.
    fn flush_shard(&mut self) -> Result<()> {
        if self.current_shard_buckets.is_empty() {
            return Ok(());
        }

        let shard = MainIndexShard {
            k: self.k,
            w: self.w,
            salt: self.salt,
            shard_id: self.current_shard_id,
            buckets: std::mem::take(&mut self.current_shard_buckets),
            bucket_sources: std::mem::take(&mut self.current_shard_sources),
        };

        // Write to temp path first (atomic commit on finish)
        let final_path = MainIndexManifest::shard_path(&self.base_path, self.current_shard_id);
        let temp_path = final_path.with_extension(format!(
            "{}.tmp",
            final_path.extension().unwrap_or_default().to_string_lossy()
        ));
        let compressed_size = shard.save(&temp_path)?;
        self.pending_shard_renames.push((temp_path, final_path));

        // Collect bucket IDs that were in this shard
        let bucket_ids: Vec<u32> = shard.buckets.keys().copied().collect();
        let num_minimizers: usize = shard.buckets.values().map(|v| v.len()).sum();

        self.shards.push(MainIndexShardInfo {
            shard_id: self.current_shard_id,
            bucket_ids,
            num_minimizers,
            compressed_size,
        });

        // Prepare for next shard
        self.current_shard_id += 1;
        self.current_shard_bytes = 0;

        Ok(())
    }

    /// Finalize the index: flush remaining buckets and write the manifest.
    ///
    /// All shard files are written to temporary paths during building, then renamed
    /// atomically on success. If this method fails, no permanent files are left behind
    /// (only `.tmp` files which can be cleaned up).
    ///
    /// Returns the manifest for the created sharded index.
    pub fn finish(mut self) -> Result<MainIndexManifest> {
        // Flush any remaining buckets
        self.flush_shard()?;

        let manifest = MainIndexManifest {
            k: self.k,
            w: self.w,
            salt: self.salt,
            bucket_names: self.bucket_names,
            bucket_minimizer_counts: self.bucket_minimizer_counts,
            bucket_to_shard: self.bucket_to_shard,
            shards: self.shards,
            total_minimizers: self.total_minimizers,
        };

        // Write manifest to temp path first
        let manifest_path = MainIndexManifest::manifest_path(&self.base_path);
        let manifest_temp = manifest_path.with_extension("manifest.tmp");
        manifest.save(&manifest_temp)?;

        // Atomically commit: rename all temp files to final paths
        // Shards first, then manifest (manifest signals completion)
        for (temp_path, final_path) in &self.pending_shard_renames {
            std::fs::rename(temp_path, final_path).with_context(|| {
                format!(
                    "Failed to rename {} to {}",
                    temp_path.display(),
                    final_path.display()
                )
            })?;
        }
        std::fs::rename(&manifest_temp, &manifest_path).with_context(|| {
            format!(
                "Failed to rename manifest {} to {}",
                manifest_temp.display(),
                manifest_path.display()
            )
        })?;

        Ok(manifest)
    }

    /// Get the current number of shards (including the one being built).
    pub fn num_shards(&self) -> usize {
        self.shards.len()
            + if self.current_shard_buckets.is_empty() {
                0
            } else {
                1
            }
    }

    /// Get the current shard's estimated byte size.
    pub fn current_shard_bytes(&self) -> usize {
        self.current_shard_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_main_manifest_save_load_roundtrip() -> Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();

        let mut manifest = MainIndexManifest {
            k: 64,
            w: 50,
            salt: 0xDEADBEEF,
            bucket_names: HashMap::new(),
            bucket_minimizer_counts: HashMap::new(),
            bucket_to_shard: HashMap::new(),
            shards: vec![
                MainIndexShardInfo {
                    shard_id: 0,
                    bucket_ids: vec![1, 2],
                    num_minimizers: 1000,
                    compressed_size: 4000,
                },
                MainIndexShardInfo {
                    shard_id: 1,
                    bucket_ids: vec![3],
                    num_minimizers: 500,
                    compressed_size: 2000,
                },
            ],
            total_minimizers: 1500,
        };

        manifest.bucket_names.insert(1, "BucketA".into());
        manifest.bucket_names.insert(2, "BucketB".into());
        manifest.bucket_names.insert(3, "BucketC".into());
        manifest.bucket_minimizer_counts.insert(1, 400);
        manifest.bucket_minimizer_counts.insert(2, 600);
        manifest.bucket_minimizer_counts.insert(3, 500);
        manifest.bucket_to_shard.insert(1, 0);
        manifest.bucket_to_shard.insert(2, 0);
        manifest.bucket_to_shard.insert(3, 1);

        manifest.save(&path)?;
        let loaded = MainIndexManifest::load(&path)?;

        assert_eq!(loaded.k, 64);
        assert_eq!(loaded.w, 50);
        assert_eq!(loaded.salt, 0xDEADBEEF);
        assert_eq!(loaded.total_minimizers, 1500);
        assert_eq!(loaded.bucket_names.len(), 3);
        assert_eq!(loaded.shards.len(), 2);
        assert_eq!(loaded.bucket_names[&1], "BucketA");
        assert_eq!(loaded.bucket_to_shard[&1], 0);
        assert_eq!(loaded.bucket_to_shard[&3], 1);

        Ok(())
    }

    #[test]
    fn test_main_shard_save_load_roundtrip() -> Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();

        let mut shard = MainIndexShard {
            k: 64,
            w: 50,
            salt: 0x1234,
            shard_id: 0,
            buckets: HashMap::new(),
            bucket_sources: HashMap::new(),
        };
        shard.buckets.insert(1, vec![100, 200, 300]);
        shard.buckets.insert(2, vec![400, 500]);
        shard
            .bucket_sources
            .insert(1, vec!["file1.fa::seq1".into(), "file1.fa::seq2".into()]);
        shard
            .bucket_sources
            .insert(2, vec!["file2.fa::seq1".into()]);

        shard.save(&path)?;
        let loaded = MainIndexShard::load(&path)?;

        assert_eq!(loaded.k, 64);
        assert_eq!(loaded.w, 50);
        assert_eq!(loaded.salt, 0x1234);
        assert_eq!(loaded.shard_id, 0);
        assert_eq!(loaded.buckets.len(), 2);
        assert_eq!(loaded.buckets[&1], vec![100, 200, 300]);
        assert_eq!(loaded.buckets[&2], vec![400, 500]);
        // Verify sources with string deduplication roundtrip
        assert_eq!(loaded.bucket_sources[&1].len(), 2);
        assert_eq!(loaded.bucket_sources[&1][0], "file1.fa::seq1");
        assert_eq!(loaded.bucket_sources[&1][1], "file1.fa::seq2");
        assert_eq!(loaded.bucket_sources[&2].len(), 1);
        assert_eq!(loaded.bucket_sources[&2][0], "file2.fa::seq1");

        Ok(())
    }

    #[test]
    fn test_plan_shards_basic() {
        let mut counts = HashMap::new();
        counts.insert(1, 100); // 400 bytes
        counts.insert(2, 200); // 800 bytes
        counts.insert(3, 150); // 600 bytes

        // Budget of 1000 bytes should create 2 shards
        let shards = plan_shards(&counts, 1000);

        assert_eq!(shards.len(), 2);
        // Largest bucket (2) goes first
        // Then bucket 3 (600 bytes) doesn't fit with bucket 2 (800), new shard
        // Bucket 1 (400 bytes) fits with bucket 3 (600)
    }

    #[test]
    fn test_plan_shards_single_large_bucket() {
        let mut counts = HashMap::new();
        counts.insert(1, 1000); // 4000 bytes - exceeds budget

        let shards = plan_shards(&counts, 1000);

        // Even if bucket exceeds budget, it gets its own shard
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0].1, vec![1]);
    }

    #[test]
    fn test_path_helpers() {
        let base = Path::new("/tmp/test.ryidx");

        let manifest_path = MainIndexManifest::manifest_path(base);
        assert_eq!(manifest_path.to_str().unwrap(), "/tmp/test.ryidx.manifest");

        let shard_path = MainIndexManifest::shard_path(base, 3);
        assert_eq!(shard_path.to_str().unwrap(), "/tmp/test.ryidx.shard.3");
    }

    #[test]
    fn test_estimate_bucket_bytes() {
        // Compressed estimate: 4 bytes per minimizer
        assert_eq!(estimate_bucket_bytes(0), 0);
        assert_eq!(estimate_bucket_bytes(100), 400);
        assert_eq!(estimate_bucket_bytes(1000), 4000);
    }

    #[test]
    fn test_estimate_bucket_memory_bytes() {
        // Memory estimate: 8 bytes per minimizer (u64)
        assert_eq!(estimate_bucket_memory_bytes(0), 0);
        assert_eq!(estimate_bucket_memory_bytes(100), 800);
        assert_eq!(estimate_bucket_memory_bytes(1000), 8000);
    }

    #[test]
    fn test_manifest_invalid_magic() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path();
        std::fs::write(path, b"NOTM").unwrap();

        let result = MainIndexManifest::load(path);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid main index manifest"));
    }

    #[test]
    fn test_shard_invalid_magic() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path();
        std::fs::write(path, b"NOTS").unwrap();

        let result = MainIndexShard::load(path);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid main index shard"));
    }

    #[test]
    fn test_manifest_v1_rejected_with_helpful_error() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path();

        // Write a v1 manifest (magic + version 1)
        let mut data = Vec::new();
        data.extend_from_slice(MAIN_MANIFEST_MAGIC);
        data.extend_from_slice(&1u32.to_le_bytes());
        std::fs::write(path, data).unwrap();

        let result = MainIndexManifest::load(path);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Version 1 manifests are no longer supported"),
            "Error should mention v1: {}",
            err
        );
        assert!(
            err.contains("rype index shard"),
            "Error should suggest re-indexing: {}",
            err
        );
    }

    #[test]
    fn test_shard_v1_rejected_with_helpful_error() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path();

        // Write a v1 shard (magic + version 1)
        let mut data = Vec::new();
        data.extend_from_slice(MAIN_SHARD_MAGIC);
        data.extend_from_slice(&1u32.to_le_bytes());
        std::fs::write(path, data).unwrap();

        let result = MainIndexShard::load(path);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Version 1 shards are no longer supported"),
            "Error should mention v1: {}",
            err
        );
        assert!(
            err.contains("rype index shard"),
            "Error should suggest re-indexing: {}",
            err
        );
    }

    #[test]
    fn test_builder_basic() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let base_path = dir.path().join("test.ryidx");

        // Create builder with 1000 byte budget
        let mut builder = ShardedMainIndexBuilder::new(64, 50, 0x1234, &base_path, 1000)?;

        // Add buckets that should span 2 shards
        builder.add_bucket(1, "BucketA", vec!["src1".into()], vec![100, 200, 300])?;
        builder.add_bucket(2, "BucketB", vec!["src2".into()], vec![400, 500])?;

        let manifest = builder.finish()?;

        assert_eq!(manifest.k, 64);
        assert_eq!(manifest.w, 50);
        assert_eq!(manifest.salt, 0x1234);
        assert_eq!(manifest.bucket_names.len(), 2);
        assert_eq!(manifest.total_minimizers, 5);

        // Verify we can open the sharded index
        let sharded = ShardedMainIndex::open(&base_path)?;
        assert_eq!(sharded.k(), 64);
        assert_eq!(sharded.total_minimizers(), 5);

        Ok(())
    }

    #[test]
    fn test_builder_auto_flush() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let base_path = dir.path().join("test.ryidx");

        // Very small budget to force multiple shards
        // Each minimizer = 8 bytes in memory, so 10 minimizers = 80 bytes
        let mut builder = ShardedMainIndexBuilder::new(64, 50, 0x1234, &base_path, 100)?;

        // Add buckets that will force flushing
        builder.add_bucket(1, "A", vec![], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10])?; // 80 bytes
        builder.add_bucket(2, "B", vec![], vec![11, 12, 13, 14, 15])?; // 40 bytes - new shard
        builder.add_bucket(3, "C", vec![], vec![21, 22, 23, 24, 25])?; // 40 bytes - fits in shard 1

        let manifest = builder.finish()?;

        // Should have 2 shards: bucket 1 alone, buckets 2+3 together
        assert_eq!(manifest.shards.len(), 2);
        assert_eq!(manifest.total_minimizers, 20);

        // Verify shard assignments
        assert_eq!(manifest.bucket_to_shard[&1], 0);
        assert_eq!(manifest.bucket_to_shard[&2], 1);
        assert_eq!(manifest.bucket_to_shard[&3], 1);

        // Verify we can load the shards
        let sharded = ShardedMainIndex::open(&base_path)?;
        let shard0 = sharded.load_shard(0)?;
        let shard1 = sharded.load_shard(1)?;

        assert_eq!(shard0.buckets.len(), 1);
        assert!(shard0.buckets.contains_key(&1));
        assert_eq!(shard1.buckets.len(), 2);
        assert!(shard1.buckets.contains_key(&2));
        assert!(shard1.buckets.contains_key(&3));

        Ok(())
    }

    /// Test that ShardedMainIndexBuilder produces identical output to Index::save_sharded().
    /// This validates that batched progressive building yields the same result as
    /// building everything in memory then sharding.
    #[test]
    fn test_builder_matches_save_sharded() -> Result<()> {
        use crate::Index;
        use tempfile::tempdir;

        let dir = tempdir()?;
        let path_a = dir.path().join("index_a.ryidx");
        let path_b = dir.path().join("index_b.ryidx");

        let k = 32;
        let w = 10;
        let salt = 0x1234u64;
        let max_shard_bytes = 500; // Small to force multiple shards

        // Create test data: 5 buckets with varying minimizers
        let bucket_data: Vec<(u32, &str, Vec<String>, Vec<u64>)> = vec![
            (
                1,
                "BucketA",
                vec!["src1".into()],
                vec![100, 200, 300, 400, 500],
            ),
            (2, "BucketB", vec!["src2".into()], vec![150, 250, 350]),
            (
                3,
                "BucketC",
                vec!["src3a".into(), "src3b".into()],
                vec![600, 700, 800, 900],
            ),
            (
                4,
                "BucketD",
                vec!["src4".into()],
                vec![50, 60, 70, 80, 90, 110],
            ),
            (5, "BucketE", vec!["src5".into()], vec![1000, 2000, 3000]),
        ];

        // Approach A: Build full Index, then save_sharded
        let mut index = Index::new(k, w, salt)?;
        for (id, name, sources, minimizers) in &bucket_data {
            index.bucket_names.insert(*id, name.to_string());
            index.bucket_sources.insert(*id, sources.clone());
            index.buckets.insert(*id, minimizers.clone());
        }
        let manifest_a = index.save_sharded(&path_a, max_shard_bytes)?;

        // Approach B: Use ShardedMainIndexBuilder (progressive)
        let mut builder = ShardedMainIndexBuilder::new(k, w, salt, &path_b, max_shard_bytes)?;
        for (id, name, sources, minimizers) in &bucket_data {
            builder.add_bucket(*id, name, sources.clone(), minimizers.clone())?;
        }
        let manifest_b = builder.finish()?;

        // Verify manifests match
        assert_eq!(manifest_a.k, manifest_b.k, "k mismatch");
        assert_eq!(manifest_a.w, manifest_b.w, "w mismatch");
        assert_eq!(manifest_a.salt, manifest_b.salt, "salt mismatch");
        assert_eq!(
            manifest_a.total_minimizers, manifest_b.total_minimizers,
            "total_minimizers mismatch"
        );
        assert_eq!(
            manifest_a.shards.len(),
            manifest_b.shards.len(),
            "shard count mismatch"
        );
        assert_eq!(
            manifest_a.bucket_names, manifest_b.bucket_names,
            "bucket_names mismatch"
        );
        assert_eq!(
            manifest_a.bucket_to_shard, manifest_b.bucket_to_shard,
            "bucket_to_shard mismatch"
        );

        // Verify each shard has same buckets and minimizers
        // Note: bucket_ids order may differ (HashMap vs insertion order) - sort for comparison
        for (shard_a, shard_b) in manifest_a.shards.iter().zip(manifest_b.shards.iter()) {
            assert_eq!(shard_a.shard_id, shard_b.shard_id, "shard_id mismatch");
            let mut ids_a = shard_a.bucket_ids.clone();
            let mut ids_b = shard_b.bucket_ids.clone();
            ids_a.sort();
            ids_b.sort();
            assert_eq!(
                ids_a, ids_b,
                "bucket_ids mismatch for shard {}",
                shard_a.shard_id
            );
            assert_eq!(
                shard_a.num_minimizers, shard_b.num_minimizers,
                "num_minimizers mismatch for shard {}",
                shard_a.shard_id
            );
        }

        // Load and verify actual shard contents match
        for shard_info in &manifest_a.shards {
            let shard_path_a = MainIndexManifest::shard_path(&path_a, shard_info.shard_id);
            let shard_path_b = MainIndexManifest::shard_path(&path_b, shard_info.shard_id);

            let shard_a = MainIndexShard::load(&shard_path_a)?;
            let shard_b = MainIndexShard::load(&shard_path_b)?;

            assert_eq!(
                shard_a.buckets, shard_b.buckets,
                "buckets mismatch in shard {}",
                shard_info.shard_id
            );
            assert_eq!(
                shard_a.bucket_sources, shard_b.bucket_sources,
                "sources mismatch in shard {}",
                shard_info.shard_id
            );
        }

        Ok(())
    }

    /// Verify bucket IDs are assigned deterministically based on sorted bucket name order.
    /// This invariant is critical for inverted index consistency.
    #[test]
    fn test_bucket_id_assignment_is_deterministic() -> Result<()> {
        use tempfile::tempdir;

        let dir = tempdir()?;
        let path = dir.path().join("test.ryidx");

        // Names intentionally out of alphabetical order
        let names = vec!["Zebra", "Apple", "Mango", "Banana"];
        let sorted_names: Vec<_> = {
            let mut v = names.clone();
            v.sort();
            v
        };

        let mut builder = ShardedMainIndexBuilder::new(32, 10, 0, &path, 10_000_000)?;

        // Add buckets in sorted order (as build_index_from_config does)
        for (i, name) in sorted_names.iter().enumerate() {
            let id = (i + 1) as u32;
            builder.add_bucket(id, name, vec![], vec![100 * id as u64])?;
        }

        let manifest = builder.finish()?;

        // Verify: bucket ID 1 = "Apple", 2 = "Banana", 3 = "Mango", 4 = "Zebra"
        assert_eq!(manifest.bucket_names[&1], "Apple");
        assert_eq!(manifest.bucket_names[&2], "Banana");
        assert_eq!(manifest.bucket_names[&3], "Mango");
        assert_eq!(manifest.bucket_names[&4], "Zebra");

        Ok(())
    }

    /// Test that batched parallel processing preserves bucket ID ordering.
    ///
    /// This simulates what build_index_from_config does: sort names, chunk into batches,
    /// process each batch with par_iter, assign IDs sequentially. The key invariant is
    /// that rayon's par_iter().collect() preserves order within each batch.
    #[test]
    fn test_batched_parallel_ordering_preserved() -> Result<()> {
        use rayon::prelude::*;
        use tempfile::tempdir;

        let dir = tempdir()?;
        let path = dir.path().join("test.ryidx");

        // 20 bucket names, intentionally unordered
        let mut names: Vec<String> = (0..20)
            .map(|i| format!("Bucket_{:02}", 19 - i)) // Reverse order: 19, 18, ..., 0
            .collect();

        // Sort (as build_index_from_config does)
        names.sort();

        // Process in small batches to exercise multiple batch iterations
        let batch_size = 3;
        let mut builder = ShardedMainIndexBuilder::new(32, 10, 0, &path, 10_000_000)?;
        let mut bucket_id = 1u32;

        for chunk in names.chunks(batch_size) {
            // Simulate build_single_bucket with par_iter (just returns name and dummy data)
            let batch_results: Vec<_> = chunk
                .par_iter()
                .map(|name| {
                    // Simulate some work
                    let minimizers = vec![bucket_id as u64 * 100];
                    (name.clone(), minimizers)
                })
                .collect();

            // Assign IDs sequentially (as build_index_from_config does)
            for (name, minimizers) in batch_results {
                builder.add_bucket(bucket_id, &name, vec![], minimizers)?;
                bucket_id += 1;
            }
        }

        let manifest = builder.finish()?;

        // Verify: bucket IDs should be assigned in sorted name order
        // Bucket_00 -> ID 1, Bucket_01 -> ID 2, ..., Bucket_19 -> ID 20
        for i in 0..20 {
            let expected_name = format!("Bucket_{:02}", i);
            let expected_id = (i + 1) as u32;
            assert_eq!(
                manifest.bucket_names[&expected_id], expected_name,
                "Bucket ID {} should be '{}', got '{}'",
                expected_id, expected_name, manifest.bucket_names[&expected_id]
            );
        }

        Ok(())
    }
}
