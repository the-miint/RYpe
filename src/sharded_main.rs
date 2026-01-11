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

use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use crate::constants::{MAX_BUCKET_SIZE, MAX_STRING_LENGTH, MAX_NUM_BUCKETS};
use crate::encoding::{encode_varint, decode_varint};
use crate::types::IndexMetadata;

/// Maximum number of shards allowed (DoS protection)
pub const MAX_MAIN_SHARDS: u32 = 10_000;

/// Default bytes per minimizer estimate for shard planning (delta+varint+zstd)
pub const BYTES_PER_MINIMIZER_ESTIMATE: usize = 4;

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
/// Format v1:
/// - Magic: "RYPM" (4 bytes)
/// - Version: 1 (u32)
/// - k (u64), w (u64), salt (u64)
/// - num_buckets (u32), num_shards (u32), total_minimizers (u64)
/// - For each bucket (sorted by ID): bucket_id (u32), shard_id (u32), minimizer_count (u64),
///   name_len (u64), name bytes, source_count (u64), [source_len (u64), source bytes]...
/// - For each shard: shard_id (u32), bucket_count (u32), num_minimizers (u64), compressed_size (u64)
#[derive(Debug, Clone)]
pub struct MainIndexManifest {
    pub k: usize,
    pub w: usize,
    pub salt: u64,
    pub bucket_names: HashMap<u32, String>,
    pub bucket_sources: HashMap<u32, Vec<String>>,
    pub bucket_minimizer_counts: HashMap<u32, usize>,
    pub bucket_to_shard: HashMap<u32, u32>,
    pub shards: Vec<MainIndexShardInfo>,
    pub total_minimizers: usize,
}

impl MainIndexManifest {
    /// Save the manifest to a file.
    pub fn save(&self, path: &Path) -> Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);

        // Header
        writer.write_all(b"RYPM")?;
        writer.write_all(&1u32.to_le_bytes())?; // Version 1

        writer.write_all(&(self.k as u64).to_le_bytes())?;
        writer.write_all(&(self.w as u64).to_le_bytes())?;
        writer.write_all(&self.salt.to_le_bytes())?;

        let num_buckets = self.bucket_names.len() as u32;
        writer.write_all(&num_buckets.to_le_bytes())?;
        writer.write_all(&(self.shards.len() as u32).to_le_bytes())?;
        writer.write_all(&(self.total_minimizers as u64).to_le_bytes())?;

        // Bucket metadata (sorted by ID)
        let mut sorted_bucket_ids: Vec<_> = self.bucket_names.keys().copied().collect();
        sorted_bucket_ids.sort_unstable();

        for bucket_id in &sorted_bucket_ids {
            writer.write_all(&bucket_id.to_le_bytes())?;

            let shard_id = self.bucket_to_shard.get(bucket_id).copied().unwrap_or(0);
            writer.write_all(&shard_id.to_le_bytes())?;

            let minimizer_count = self.bucket_minimizer_counts.get(bucket_id).copied().unwrap_or(0);
            writer.write_all(&(minimizer_count as u64).to_le_bytes())?;

            // Name
            let name = self.bucket_names.get(bucket_id).map(|s| s.as_str()).unwrap_or("");
            let name_bytes = name.as_bytes();
            writer.write_all(&(name_bytes.len() as u64).to_le_bytes())?;
            writer.write_all(name_bytes)?;

            // Sources
            let empty = Vec::new();
            let sources = self.bucket_sources.get(bucket_id).unwrap_or(&empty);
            writer.write_all(&(sources.len() as u64).to_le_bytes())?;
            for src in sources {
                let src_bytes = src.as_bytes();
                writer.write_all(&(src_bytes.len() as u64).to_le_bytes())?;
                writer.write_all(src_bytes)?;
            }
        }

        // Shard info
        for shard in &self.shards {
            writer.write_all(&shard.shard_id.to_le_bytes())?;
            writer.write_all(&(shard.bucket_ids.len() as u32).to_le_bytes())?;
            writer.write_all(&(shard.num_minimizers as u64).to_le_bytes())?;
            writer.write_all(&shard.compressed_size.to_le_bytes())?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Load a manifest from a file.
    pub fn load(path: &Path) -> Result<Self> {
        let mut reader = BufReader::new(File::open(path)?);
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        // Header
        reader.read_exact(&mut buf4)?;
        if &buf4 != b"RYPM" {
            return Err(anyhow!("Invalid main index manifest format (expected RYPM)"));
        }

        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != 1 {
            return Err(anyhow!("Unsupported main index manifest version: {} (expected 1)", version));
        }

        reader.read_exact(&mut buf8)?;
        let k = u64::from_le_bytes(buf8) as usize;
        if !matches!(k, 16 | 32 | 64) {
            return Err(anyhow!("Invalid K value in manifest: {} (must be 16, 32, or 64)", k));
        }

        reader.read_exact(&mut buf8)?;
        let w = u64::from_le_bytes(buf8) as usize;

        reader.read_exact(&mut buf8)?;
        let salt = u64::from_le_bytes(buf8);

        reader.read_exact(&mut buf4)?;
        let num_buckets = u32::from_le_bytes(buf4);
        if num_buckets > MAX_NUM_BUCKETS {
            return Err(anyhow!("Number of buckets {} exceeds maximum {}", num_buckets, MAX_NUM_BUCKETS));
        }

        reader.read_exact(&mut buf4)?;
        let num_shards = u32::from_le_bytes(buf4);
        if num_shards > MAX_MAIN_SHARDS {
            return Err(anyhow!("Number of shards {} exceeds maximum {}", num_shards, MAX_MAIN_SHARDS));
        }

        reader.read_exact(&mut buf8)?;
        let total_minimizers = u64::from_le_bytes(buf8) as usize;

        // Bucket metadata
        let mut bucket_names = HashMap::new();
        let mut bucket_sources = HashMap::new();
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
                return Err(anyhow!("Bucket {} minimizer count {} exceeds maximum {}",
                    bucket_id, minimizer_count, MAX_BUCKET_SIZE));
            }

            // Name
            reader.read_exact(&mut buf8)?;
            let name_len = u64::from_le_bytes(buf8) as usize;
            if name_len > MAX_STRING_LENGTH {
                return Err(anyhow!("Bucket name length {} exceeds maximum {}", name_len, MAX_STRING_LENGTH));
            }
            let mut name_buf = vec![0u8; name_len];
            reader.read_exact(&mut name_buf)?;
            let name = String::from_utf8(name_buf)?;

            // Sources
            reader.read_exact(&mut buf8)?;
            let source_count = u64::from_le_bytes(buf8) as usize;
            let mut sources = Vec::with_capacity(source_count);
            for _ in 0..source_count {
                reader.read_exact(&mut buf8)?;
                let src_len = u64::from_le_bytes(buf8) as usize;
                if src_len > MAX_STRING_LENGTH {
                    return Err(anyhow!("Source string length {} exceeds maximum {}", src_len, MAX_STRING_LENGTH));
                }
                let mut src_buf = vec![0u8; src_len];
                reader.read_exact(&mut src_buf)?;
                sources.push(String::from_utf8(src_buf)?);
            }

            bucket_names.insert(bucket_id, name);
            bucket_sources.insert(bucket_id, sources);
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
            let bucket_ids: Vec<u32> = bucket_to_shard.iter()
                .filter(|(_, &s)| s == shard_id)
                .map(|(&b, _)| b)
                .collect();

            if bucket_ids.len() != bucket_count as usize {
                return Err(anyhow!("Shard {} bucket count mismatch: expected {}, found {}",
                    shard_id, bucket_count, bucket_ids.len()));
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
                return Err(anyhow!("Invalid manifest: shard IDs not sequential (expected {}, found {})",
                    i, shard.shard_id));
            }
        }

        // Validate total minimizers
        let sum_minimizers: usize = shards.iter().map(|s| s.num_minimizers).sum();
        if sum_minimizers != total_minimizers {
            return Err(anyhow!("Invalid manifest: shard minimizer counts sum to {}, expected {}",
                sum_minimizers, total_minimizers));
        }

        Ok(MainIndexManifest {
            k,
            w,
            salt,
            bucket_names,
            bucket_sources,
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
    pub fn to_metadata(&self) -> IndexMetadata {
        IndexMetadata {
            k: self.k,
            w: self.w,
            salt: self.salt,
            bucket_names: self.bucket_names.clone(),
            bucket_sources: self.bucket_sources.clone(),
            bucket_minimizer_counts: self.bucket_minimizer_counts.clone(),
        }
    }
}

/// A single loaded shard containing a subset of buckets.
#[derive(Debug)]
pub struct MainIndexShard {
    pub k: usize,
    pub w: usize,
    pub salt: u64,
    pub shard_id: u32,
    pub buckets: HashMap<u32, Vec<u64>>,
}

impl MainIndexShard {
    /// Save the shard to a file (RYPS v1 format with delta+varint+zstd).
    pub fn save(&self, path: &Path) -> Result<u64> {
        let mut writer = BufWriter::new(File::create(path)?);

        // Header (uncompressed)
        writer.write_all(b"RYPS")?;
        writer.write_all(&1u32.to_le_bytes())?; // Version 1
        writer.write_all(&(self.k as u64).to_le_bytes())?;
        writer.write_all(&(self.w as u64).to_le_bytes())?;
        writer.write_all(&self.salt.to_le_bytes())?;
        writer.write_all(&self.shard_id.to_le_bytes())?;
        writer.write_all(&(self.buckets.len() as u32).to_le_bytes())?;

        // Bucket data with delta+varint encoding (zstd compressed)
        let mut encoder = zstd::stream::write::Encoder::new(writer, 3)?;

        let mut sorted_ids: Vec<_> = self.buckets.keys().copied().collect();
        sorted_ids.sort_unstable();

        const WRITE_BUF_SIZE: usize = 8 * 1024 * 1024;
        let mut write_buf = Vec::with_capacity(WRITE_BUF_SIZE);
        let mut varint_buf = [0u8; 10];

        let flush_buf = |buf: &mut Vec<u8>, encoder: &mut zstd::stream::write::Encoder<BufWriter<File>>| -> Result<()> {
            if !buf.is_empty() {
                encoder.write_all(buf)?;
                buf.clear();
            }
            Ok(())
        };

        for id in &sorted_ids {
            let vec = &self.buckets[id];

            // Bucket ID and minimizer count (uncompressed within stream)
            write_buf.extend_from_slice(&id.to_le_bytes());
            write_buf.extend_from_slice(&(vec.len() as u64).to_le_bytes());

            if vec.is_empty() {
                continue;
            }

            // First minimizer: full u64
            write_buf.extend_from_slice(&vec[0].to_le_bytes());
            if write_buf.len() >= WRITE_BUF_SIZE {
                flush_buf(&mut write_buf, &mut encoder)?;
            }

            // Remaining: delta-encoded varints
            let mut prev = vec[0];
            for &val in &vec[1..] {
                let delta = val - prev;
                let len = encode_varint(delta, &mut varint_buf);
                write_buf.extend_from_slice(&varint_buf[..len]);
                if write_buf.len() >= WRITE_BUF_SIZE {
                    flush_buf(&mut write_buf, &mut encoder)?;
                }
                prev = val;
            }
        }
        flush_buf(&mut write_buf, &mut encoder)?;

        let writer = encoder.finish()?;
        let inner = writer.into_inner()?;
        let compressed_size = inner.metadata()?.len();

        Ok(compressed_size)
    }

    /// Load a shard from a file (RYPS v1 format).
    pub fn load(path: &Path) -> Result<Self> {
        let mut reader = BufReader::new(File::open(path)?);
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        // Header
        reader.read_exact(&mut buf4)?;
        if &buf4 != b"RYPS" {
            return Err(anyhow!("Invalid main index shard format (expected RYPS)"));
        }

        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != 1 {
            return Err(anyhow!("Unsupported main index shard version: {} (expected 1)", version));
        }

        reader.read_exact(&mut buf8)?;
        let k = u64::from_le_bytes(buf8) as usize;
        if !matches!(k, 16 | 32 | 64) {
            return Err(anyhow!("Invalid K value in shard: {} (must be 16, 32, or 64)", k));
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
            return Err(anyhow!("Number of buckets {} exceeds maximum {}", num_buckets, MAX_NUM_BUCKETS));
        }

        // Bucket data with delta+varint decoding (zstd compressed)
        let mut decoder = zstd::stream::read::Decoder::new(reader)
            .map_err(|e| anyhow!("Failed to create zstd decoder: {}", e))?;

        const READ_BUF_SIZE: usize = 8 * 1024 * 1024;
        let mut read_buf = vec![0u8; READ_BUF_SIZE];
        let mut buf_pos = 0usize;
        let mut buf_len = 0usize;

        let mut buckets = HashMap::new();

        for _ in 0..num_buckets {
            // Helper to ensure bytes available
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

            // Read bucket ID
            ensure_bytes!(4);
            let bucket_id = u32::from_le_bytes(read_buf[buf_pos..buf_pos + 4].try_into().unwrap());
            buf_pos += 4;

            // Read minimizer count
            ensure_bytes!(8);
            let vec_len = u64::from_le_bytes(read_buf[buf_pos..buf_pos + 8].try_into().unwrap()) as usize;
            buf_pos += 8;

            if vec_len > MAX_BUCKET_SIZE {
                return Err(anyhow!("Bucket {} size {} exceeds maximum {}", bucket_id, vec_len, MAX_BUCKET_SIZE));
            }

            let mut vec = Vec::with_capacity(vec_len);

            if vec_len == 0 {
                buckets.insert(bucket_id, vec);
                continue;
            }

            // First minimizer
            ensure_bytes!(8);
            let first = u64::from_le_bytes(read_buf[buf_pos..buf_pos + 8].try_into().unwrap());
            buf_pos += 8;
            vec.push(first);

            // Remaining: delta-decoded varints
            let mut prev = first;
            for _ in 1..vec_len {
                refill_if_low!();
                let (delta, consumed) = decode_varint(&read_buf[buf_pos..buf_len]);
                buf_pos += consumed;
                let val = prev + delta;
                vec.push(val);
                prev = val;
            }

            buckets.insert(bucket_id, vec);
        }

        Ok(MainIndexShard {
            k,
            w,
            salt,
            shard_id,
            buckets,
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
            return Err(anyhow!("Shard ID {} out of range (max {})",
                shard_id, self.manifest.shards.len() - 1));
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

        // Create new shard
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
        };

        // Save new shard
        let shard_path = MainIndexManifest::shard_path(&self.base_path, new_shard_id);
        let compressed_size = shard.save(&shard_path)?;

        // Update manifest
        self.manifest.bucket_names.insert(bucket_id, name.to_string());
        self.manifest.bucket_sources.insert(bucket_id, sources);
        self.manifest.bucket_minimizer_counts.insert(bucket_id, minimizer_count);
        self.manifest.bucket_to_shard.insert(bucket_id, new_shard_id);
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

    /// Merge one bucket into another.
    ///
    /// If both buckets are in the same shard, modifies that shard.
    /// If in different shards, moves minimizers from src to dest shard.
    pub fn merge_buckets(&mut self, src_id: u32, dest_id: u32) -> Result<()> {
        if !self.manifest.bucket_names.contains_key(&src_id) {
            return Err(anyhow!("Source bucket {} does not exist", src_id));
        }
        if !self.manifest.bucket_names.contains_key(&dest_id) {
            return Err(anyhow!("Destination bucket {} does not exist", dest_id));
        }

        let src_shard_id = *self.manifest.bucket_to_shard.get(&src_id)
            .ok_or_else(|| anyhow!("Source bucket {} not mapped to any shard", src_id))?;
        let dest_shard_id = *self.manifest.bucket_to_shard.get(&dest_id)
            .ok_or_else(|| anyhow!("Destination bucket {} not mapped to any shard", dest_id))?;

        let src_minimizer_count = self.manifest.bucket_minimizer_counts.get(&src_id).copied().unwrap_or(0);

        if src_shard_id == dest_shard_id {
            // Same shard - load, merge, save
            let shard_path = MainIndexManifest::shard_path(&self.base_path, src_shard_id);
            let mut shard = MainIndexShard::load(&shard_path)?;

            let src_minimizers = shard.buckets.remove(&src_id)
                .ok_or_else(|| anyhow!("Source bucket {} not found in shard", src_id))?;

            let dest_vec = shard.buckets.entry(dest_id).or_default();
            dest_vec.extend(src_minimizers);
            dest_vec.sort_unstable();
            dest_vec.dedup();

            let compressed_size = shard.save(&shard_path)?;

            // Update shard info
            let shard_info = &mut self.manifest.shards[src_shard_id as usize];
            shard_info.bucket_ids.retain(|&id| id != src_id);
            shard_info.compressed_size = compressed_size;
            // Note: num_minimizers stays the same (minimizers moved within shard)
        } else {
            // Different shards - load both, move minimizers, save both
            let src_shard_path = MainIndexManifest::shard_path(&self.base_path, src_shard_id);
            let dest_shard_path = MainIndexManifest::shard_path(&self.base_path, dest_shard_id);

            let mut src_shard = MainIndexShard::load(&src_shard_path)?;
            let mut dest_shard = MainIndexShard::load(&dest_shard_path)?;

            let src_minimizers = src_shard.buckets.remove(&src_id)
                .ok_or_else(|| anyhow!("Source bucket {} not found in source shard", src_id))?;

            let dest_vec = dest_shard.buckets.entry(dest_id).or_default();
            dest_vec.extend(src_minimizers);
            dest_vec.sort_unstable();
            dest_vec.dedup();

            let src_compressed_size = src_shard.save(&src_shard_path)?;
            let dest_compressed_size = dest_shard.save(&dest_shard_path)?;

            // Update shard infos
            let src_shard_info = &mut self.manifest.shards[src_shard_id as usize];
            src_shard_info.bucket_ids.retain(|&id| id != src_id);
            src_shard_info.num_minimizers -= src_minimizer_count;
            src_shard_info.compressed_size = src_compressed_size;

            let dest_shard_info = &mut self.manifest.shards[dest_shard_id as usize];
            dest_shard_info.num_minimizers += src_minimizer_count;
            dest_shard_info.compressed_size = dest_compressed_size;
        }

        // Update manifest metadata
        self.manifest.bucket_names.remove(&src_id);
        if let Some(mut src_sources) = self.manifest.bucket_sources.remove(&src_id) {
            let dest_sources = self.manifest.bucket_sources.entry(dest_id).or_default();
            dest_sources.append(&mut src_sources);
            dest_sources.sort_unstable();
            dest_sources.dedup();
        }

        // Update dest minimizer count (add src count - may have some duplicates removed but we don't track that)
        let dest_count = self.manifest.bucket_minimizer_counts.get(&dest_id).copied().unwrap_or(0);
        self.manifest.bucket_minimizer_counts.insert(dest_id, dest_count + src_minimizer_count);
        self.manifest.bucket_minimizer_counts.remove(&src_id);

        self.manifest.bucket_to_shard.remove(&src_id);

        // Save updated manifest
        let manifest_path = MainIndexManifest::manifest_path(&self.base_path);
        self.manifest.save(&manifest_path)?;

        Ok(())
    }

    /// Get the next available bucket ID.
    pub fn next_id(&self) -> Result<u32> {
        let max_id = self.manifest.bucket_names.keys().max().copied().unwrap_or(0);
        max_id.checked_add(1)
            .ok_or_else(|| anyhow!("Bucket ID overflow: maximum ID {} reached", max_id))
    }
}

/// Estimate the compressed byte size of a bucket.
pub fn estimate_bucket_bytes(minimizer_count: usize) -> usize {
    minimizer_count * BYTES_PER_MINIMIZER_ESTIMATE
}

/// Plan shard assignment based on memory budget using first-fit-decreasing bin packing.
///
/// Returns: Vec of (shard_id, Vec<bucket_id>)
pub fn plan_shards(
    bucket_minimizer_counts: &HashMap<u32, usize>,
    max_shard_bytes: usize,
) -> Vec<(u32, Vec<u32>)> {
    // Sort buckets by estimated size (largest first) for best-fit packing
    let mut sorted: Vec<_> = bucket_minimizer_counts.iter()
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
    shards.into_iter()
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

    // Current shard accumulator
    current_shard_id: u32,
    current_shard_buckets: HashMap<u32, Vec<u64>>,
    current_shard_bytes: usize,

    // Metadata accumulators (kept in memory for manifest)
    bucket_names: HashMap<u32, String>,
    bucket_sources: HashMap<u32, Vec<String>>,
    bucket_minimizer_counts: HashMap<u32, usize>,
    bucket_to_shard: HashMap<u32, u32>,
    shards: Vec<MainIndexShardInfo>,
    total_minimizers: usize,
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
    pub fn new(k: usize, w: usize, salt: u64, base_path: &Path, max_shard_bytes: usize) -> Result<Self> {
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
            current_shard_bytes: 0,
            bucket_names: HashMap::new(),
            bucket_sources: HashMap::new(),
            bucket_minimizer_counts: HashMap::new(),
            bucket_to_shard: HashMap::new(),
            shards: Vec::new(),
            total_minimizers: 0,
        })
    }

    /// Add a finalized bucket to the index.
    ///
    /// The minimizers should already be sorted and deduplicated.
    /// If the current shard exceeds the memory budget after adding this bucket,
    /// the shard is flushed to disk automatically.
    ///
    /// # Arguments
    /// * `id` - Bucket ID
    /// * `name` - Human-readable bucket name
    /// * `sources` - List of source sequences
    /// * `minimizers` - Sorted, deduplicated minimizers
    pub fn add_bucket(
        &mut self,
        id: u32,
        name: &str,
        sources: Vec<String>,
        minimizers: Vec<u64>,
    ) -> Result<()> {
        let bucket_bytes = estimate_bucket_bytes(minimizers.len());
        let minimizer_count = minimizers.len();

        // Check if we need to flush current shard first
        // (but don't flush if this is the first bucket in a new shard)
        if !self.current_shard_buckets.is_empty()
            && self.current_shard_bytes + bucket_bytes > self.max_shard_bytes
        {
            self.flush_shard()?;
        }

        // Add bucket to current shard
        self.current_shard_buckets.insert(id, minimizers);
        self.current_shard_bytes += bucket_bytes;

        // Store metadata
        self.bucket_names.insert(id, name.to_string());
        self.bucket_sources.insert(id, sources);
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
        };

        let shard_path = MainIndexManifest::shard_path(&self.base_path, self.current_shard_id);
        let compressed_size = shard.save(&shard_path)?;

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
    /// Returns the manifest for the created sharded index.
    pub fn finish(mut self) -> Result<MainIndexManifest> {
        // Flush any remaining buckets
        self.flush_shard()?;

        let manifest = MainIndexManifest {
            k: self.k,
            w: self.w,
            salt: self.salt,
            bucket_names: self.bucket_names,
            bucket_sources: self.bucket_sources,
            bucket_minimizer_counts: self.bucket_minimizer_counts,
            bucket_to_shard: self.bucket_to_shard,
            shards: self.shards,
            total_minimizers: self.total_minimizers,
        };

        let manifest_path = MainIndexManifest::manifest_path(&self.base_path);
        manifest.save(&manifest_path)?;

        Ok(manifest)
    }

    /// Get the current number of shards (including the one being built).
    pub fn num_shards(&self) -> usize {
        self.shards.len() + if self.current_shard_buckets.is_empty() { 0 } else { 1 }
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
            bucket_sources: HashMap::new(),
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
        manifest.bucket_sources.insert(1, vec!["src1".into()]);
        manifest.bucket_sources.insert(2, vec!["src2".into()]);
        manifest.bucket_sources.insert(3, vec!["src3a".into(), "src3b".into()]);
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
        };
        shard.buckets.insert(1, vec![100, 200, 300]);
        shard.buckets.insert(2, vec![400, 500]);

        shard.save(&path)?;
        let loaded = MainIndexShard::load(&path)?;

        assert_eq!(loaded.k, 64);
        assert_eq!(loaded.w, 50);
        assert_eq!(loaded.salt, 0x1234);
        assert_eq!(loaded.shard_id, 0);
        assert_eq!(loaded.buckets.len(), 2);
        assert_eq!(loaded.buckets[&1], vec![100, 200, 300]);
        assert_eq!(loaded.buckets[&2], vec![400, 500]);

        Ok(())
    }

    #[test]
    fn test_plan_shards_basic() {
        let mut counts = HashMap::new();
        counts.insert(1, 100);  // 400 bytes
        counts.insert(2, 200);  // 800 bytes
        counts.insert(3, 150);  // 600 bytes

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
        counts.insert(1, 1000);  // 4000 bytes - exceeds budget

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
        assert_eq!(estimate_bucket_bytes(0), 0);
        assert_eq!(estimate_bucket_bytes(100), 400);
        assert_eq!(estimate_bucket_bytes(1000), 4000);
    }

    #[test]
    fn test_manifest_invalid_magic() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path();
        std::fs::write(path, b"NOTM").unwrap();

        let result = MainIndexManifest::load(path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid main index manifest"));
    }

    #[test]
    fn test_shard_invalid_magic() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path();
        std::fs::write(path, b"NOTS").unwrap();

        let result = MainIndexShard::load(path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid main index shard"));
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
        // Each minimizer ~4 bytes, so 10 minimizers = 40 bytes
        let mut builder = ShardedMainIndexBuilder::new(64, 50, 0x1234, &base_path, 50)?;

        // Add buckets that will force flushing
        builder.add_bucket(1, "A", vec![], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10])?; // 40 bytes
        builder.add_bucket(2, "B", vec![], vec![11, 12, 13, 14, 15])?; // 20 bytes - new shard
        builder.add_bucket(3, "C", vec![], vec![21, 22, 23, 24, 25])?; // 20 bytes - fits in shard 1

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
}
