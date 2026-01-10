use anyhow::{Result, anyhow};
use rayon::prelude::*;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::cmp::Reverse;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

// Expose the C-API module
pub mod c_api;

// Expose the config module for testing and CLI usage
pub mod config;

// --- CONSTANTS ---

// Maximum sizes for safety checks when loading files
const MAX_BUCKET_SIZE: usize = 1_000_000_000; // 1B minimizers (~8GB)
const MAX_STRING_LENGTH: usize = 10_000; // 10KB for names/sources
const MAX_NUM_BUCKETS: u32 = 100_000; // Reasonable upper limit

// Maximum sizes for inverted index
const MAX_INVERTED_MINIMIZERS: usize = usize::MAX; // Allow system memory to be the limit
const MAX_INVERTED_BUCKET_IDS: usize = 4_000_000_000; // 4B total bucket ID entries

// Default capacities for workspace (document the reasoning)
const DEFAULT_DEQUE_CAPACITY: usize = 128; // Typical window size range
const ESTIMATED_MINIMIZERS_PER_SEQUENCE: usize = 32; // Conservative estimate

const BASE_TO_BIT_LUT: [u64; 256] = {
    let mut lut = [u64::MAX; 256];
    lut[b'A' as usize] = 1; lut[b'a' as usize] = 1;
    lut[b'G' as usize] = 1; lut[b'g' as usize] = 1;
    lut[b'T' as usize] = 0; lut[b't' as usize] = 0;
    lut[b'C' as usize] = 0; lut[b'c' as usize] = 0;
    lut
};

// --- VARINT ENCODING (LEB128) ---

/// Encode a u64 as a variable-length integer (LEB128 format).
/// Returns the number of bytes written.
#[inline]
fn encode_varint(mut value: u64, buf: &mut [u8]) -> usize {
    let mut i = 0;
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            buf[i] = byte;
            return i + 1;
        } else {
            buf[i] = byte | 0x80;
            i += 1;
        }
    }
}

/// Decode a variable-length integer from a byte slice.
/// Returns (value, bytes_consumed). If buffer is too small or varint is malformed,
/// returns (partial_value, bytes_read) - caller must check consumed < buf.len()
/// or validate the result.
#[inline]
fn decode_varint(buf: &[u8]) -> (u64, usize) {
    let mut value: u64 = 0;
    let mut shift = 0;
    let mut i = 0;
    loop {
        // Bounds check - prevent buffer overrun
        if i >= buf.len() {
            // Truncated varint - return what we have
            return (value, i);
        }
        let byte = buf[i];
        value |= ((byte & 0x7F) as u64) << shift;
        i += 1;
        if byte & 0x80 == 0 {
            return (value, i);
        }
        shift += 7;
        if shift >= 64 {
            // Malformed varint (>10 bytes) - return what we have
            return (value, i);
        }
    }
}

// --- DATA TYPES ---

/// ID (i64), Sequence Reference, Optional Pair Sequence Reference
pub type QueryRecord<'a> = (i64, &'a [u8], Option<&'a [u8]>);

/// Lightweight metadata-only view of an Index (without minimizer data)
#[derive(Debug, Clone)]
pub struct IndexMetadata {
    pub k: usize,
    pub w: usize,
    pub salt: u64,
    pub bucket_names: HashMap<u32, String>,
    pub bucket_sources: HashMap<u32, Vec<String>>,
    pub bucket_minimizer_counts: HashMap<u32, usize>,
}

/// Query ID, Bucket ID, Score
#[derive(Debug, Clone, PartialEq)]
pub struct HitResult {
    pub query_id: i64,
    pub bucket_id: u32,
    pub score: f64,
}

// --- CORE STRUCTURES ---

#[inline(always)]
pub fn base_to_bit(byte: u8) -> u64 {
    unsafe { *BASE_TO_BIT_LUT.get_unchecked(byte as usize) }
}

/// Compute reverse complement of a k-mer in RY space
/// Returns the reversed bits in the rightmost K positions
#[inline]
fn reverse_complement(kmer: u64, k: usize) -> u64 {
    let complement = !kmer;
    match k {
        16 => complement.reverse_bits() >> 48,
        32 => complement.reverse_bits() >> 32,
        64 => complement.reverse_bits(),
        _ => panic!("Unsupported K value: {}", k),
    }
}

pub struct MinimizerWorkspace {
    q_fwd: VecDeque<(usize, u64)>,
    q_rc: VecDeque<(usize, u64)>,
    pub buffer: Vec<u64>,
}

impl MinimizerWorkspace {
    pub fn new() -> Self {
        Self {
            q_fwd: VecDeque::with_capacity(DEFAULT_DEQUE_CAPACITY),
            q_rc: VecDeque::with_capacity(DEFAULT_DEQUE_CAPACITY),
            buffer: Vec::with_capacity(DEFAULT_DEQUE_CAPACITY),
        }
    }
}

impl Default for MinimizerWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

// --- ALGORITHMS ---

pub fn extract_into(seq: &[u8], k: usize, w: usize, salt: u64, ws: &mut MinimizerWorkspace) {
    ws.buffer.clear();
    ws.q_fwd.clear();

    let len = seq.len();
    if len < k { return; }

    let mut current_val: u64 = 0;
    let mut last_min: Option<u64> = None;
    let mut valid_bases_count = 0;

    for i in 0..len {
        let bit = base_to_bit(seq[i]);

        if bit == u64::MAX {
            valid_bases_count = 0;
            ws.q_fwd.clear();
            current_val = 0;
            last_min = None;
            continue;
        }

        valid_bases_count += 1;
        current_val = (current_val << 1) | bit;

        if valid_bases_count >= k {
            let pos = i + 1 - k;
            let hash = current_val ^ salt;

            while let Some(&(p, _)) = ws.q_fwd.front() {
                if p + w <= pos { ws.q_fwd.pop_front(); } else { break; }
            }
            while let Some(&(_, v)) = ws.q_fwd.back() {
                if v >= hash { ws.q_fwd.pop_back(); } else { break; }
            }
            ws.q_fwd.push_back((pos, hash));

            if valid_bases_count >= k + w - 1 {
                if let Some(&(_, min_h)) = ws.q_fwd.front() {
                    if Some(min_h) != last_min {
                        ws.buffer.push(min_h);
                        last_min = Some(min_h);
                    }
                }
            }
        }
    }
}

pub fn extract_dual_strand_into(
    seq: &[u8],
    k: usize,
    w: usize,
    salt: u64,
    ws: &mut MinimizerWorkspace
) -> (Vec<u64>, Vec<u64>) {
    ws.q_fwd.clear();
    ws.q_rc.clear();

    let len = seq.len();
    if len < k { return (vec![], vec![]); }

    let mut fwd_mins = Vec::with_capacity(ESTIMATED_MINIMIZERS_PER_SEQUENCE);
    let mut rc_mins = Vec::with_capacity(ESTIMATED_MINIMIZERS_PER_SEQUENCE);

    let mut current_val: u64 = 0;
    let mut valid_bases_count = 0;

    let mut last_fwd: Option<u64> = None;
    let mut last_rc: Option<u64> = None;

    for i in 0..len {
        let bit = base_to_bit(seq[i]);

        if bit == u64::MAX {
            valid_bases_count = 0;
            ws.q_fwd.clear();
            ws.q_rc.clear();
            current_val = 0;
            last_fwd = None;
            last_rc = None;
            continue;
        }

        valid_bases_count += 1;
        current_val = (current_val << 1) | bit;

        if valid_bases_count >= k {
            let pos = i + 1 - k;
            let h_fwd = current_val ^ salt;
            let h_rc = reverse_complement(current_val, k) ^ salt;

            while let Some(&(p, _)) = ws.q_fwd.front() { if p + w <= pos { ws.q_fwd.pop_front(); } else { break; } }
            while let Some(&(_, v)) = ws.q_fwd.back() { if v >= h_fwd { ws.q_fwd.pop_back(); } else { break; } }
            ws.q_fwd.push_back((pos, h_fwd));

            while let Some(&(p, _)) = ws.q_rc.front() { if p + w <= pos { ws.q_rc.pop_front(); } else { break; } }
            while let Some(&(_, v)) = ws.q_rc.back() { if v >= h_rc { ws.q_rc.pop_back(); } else { break; } }
            ws.q_rc.push_back((pos, h_rc));

            if valid_bases_count >= k + w - 1 {
                if let Some(&(_, min)) = ws.q_fwd.front() {
                    if Some(min) != last_fwd { fwd_mins.push(min); last_fwd = Some(min); }
                }
                if let Some(&(_, min)) = ws.q_rc.front() {
                    if Some(min) != last_rc { rc_mins.push(min); last_rc = Some(min); }
                }
            }
        }
    }
    (fwd_mins, rc_mins)
}

pub fn get_paired_minimizers_into(
    s1: &[u8], s2: Option<&[u8]>, k: usize, w: usize, salt: u64, ws: &mut MinimizerWorkspace
) -> (Vec<u64>, Vec<u64>) {
    let (mut fwd, mut rc) = extract_dual_strand_into(s1, k, w, salt, ws);
    if let Some(seq2) = s2 {
        let (mut r2_f, mut r2_rc) = extract_dual_strand_into(seq2, k, w, salt, ws);
        fwd.append(&mut r2_rc);
        rc.append(&mut r2_f);
    }
    fwd.sort_unstable(); fwd.dedup();
    rc.sort_unstable(); rc.dedup();
    (fwd, rc)
}

pub fn count_hits(mins: &[u64], bucket: &[u64]) -> f64 {
    let mut hits = 0;
    for m in mins {
        if bucket.binary_search(m).is_ok() {
            hits += 1;
        }
    }
    hits as f64
}

// --- INDEX ---

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
    pub const BUCKET_SOURCE_DELIM: &'static str = "::";

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
            bucket_sources: HashMap::new()
        })
    }

    pub fn add_record(&mut self, id: u32, source_name: &str, sequence: &[u8], ws: &mut MinimizerWorkspace) {
        let sources = self.bucket_sources.entry(id).or_default();
        sources.push(source_name.to_string());

        extract_into(sequence, self.k, self.w, self.salt, ws);
        let bucket = self.buckets.entry(id).or_default();
        bucket.extend_from_slice(&ws.buffer);
    }

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

    pub fn merge_buckets(&mut self, src_id: u32, dest_id: u32) -> Result<()> {
        let src_vec = self.buckets.remove(&src_id)
            .ok_or_else(|| anyhow!("Source bucket {} does not exist", src_id))?;
        self.bucket_names.remove(&src_id);

        if let Some(mut src_sources) = self.bucket_sources.remove(&src_id) {
            let dest_sources = self.bucket_sources.entry(dest_id).or_default();
            dest_sources.append(&mut src_sources);
            dest_sources.sort_unstable();
            dest_sources.dedup();
        }

        let dest_vec = self.buckets.entry(dest_id).or_default();
        dest_vec.extend(src_vec);
        dest_vec.sort_unstable();
        dest_vec.dedup();

        Ok(())
    }

    pub fn next_id(&self) -> Result<u32> {
        let max_id = self.buckets.keys().max().copied().unwrap_or(0);
        max_id.checked_add(1)
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
        writer.write_all(b"RYP5")?;
        writer.write_all(&5u32.to_le_bytes())?; // Version 5
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
            let name_str = self.bucket_names.get(id).map(|s| s.as_str()).unwrap_or("unknown");
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

        const WRITE_BUF_SIZE: usize = 8 * 1024 * 1024;
        let mut write_buf = Vec::with_capacity(WRITE_BUF_SIZE);
        let mut varint_buf = [0u8; 10]; // Max 10 bytes for u64 varint

        let flush_buf = |buf: &mut Vec<u8>, encoder: &mut zstd::stream::write::Encoder<BufWriter<File>>| -> Result<()> {
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

    /// Load an index from a file (v5 format with delta+varint+zstd compressed minimizers).
    pub fn load(path: &Path) -> Result<Self> {
        let mut reader = BufReader::new(File::open(path)?);
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        reader.read_exact(&mut buf4)?;
        if &buf4 != b"RYP5" { return Err(anyhow!("Invalid Index Format (Expected RYP5)")); }

        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        const SUPPORTED_VERSION: u32 = 5;
        if version != SUPPORTED_VERSION {
            return Err(anyhow!("Unsupported index version: {} (expected {})", version, SUPPORTED_VERSION));
        }

        reader.read_exact(&mut buf8)?;
        let k = u64::from_le_bytes(buf8) as usize;
        if !matches!(k, 16 | 32 | 64) {
            return Err(anyhow!("Invalid K value in index: {} (must be 16, 32, or 64)", k));
        }
        reader.read_exact(&mut buf8)?; let w = u64::from_le_bytes(buf8) as usize;
        reader.read_exact(&mut buf8)?; let salt = u64::from_le_bytes(buf8);
        reader.read_exact(&mut buf4)?; let num = u32::from_le_bytes(buf4);
        if num > MAX_NUM_BUCKETS {
            return Err(anyhow!("Number of buckets {} exceeds maximum {}", num, MAX_NUM_BUCKETS));
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
                return Err(anyhow!("Bucket size {} exceeds maximum {}", vec_len, MAX_BUCKET_SIZE));
            }

            reader.read_exact(&mut buf4)?; let id = u32::from_le_bytes(buf4);

            reader.read_exact(&mut buf8)?; let name_len = u64::from_le_bytes(buf8) as usize;
            if name_len > MAX_STRING_LENGTH {
                return Err(anyhow!("Bucket name length {} exceeds maximum {}", name_len, MAX_STRING_LENGTH));
            }
            let mut nbuf = vec![0u8; name_len];
            reader.read_exact(&mut nbuf)?;
            names.insert(id, String::from_utf8(nbuf)?);

            reader.read_exact(&mut buf8)?; let src_count = u64::from_le_bytes(buf8) as usize;
            let mut src_list = Vec::with_capacity(src_count);
            for _ in 0..src_count {
                reader.read_exact(&mut buf8)?; let slen = u64::from_le_bytes(buf8) as usize;
                if slen > MAX_STRING_LENGTH {
                    return Err(anyhow!("Source string length {} exceeds maximum {}", slen, MAX_STRING_LENGTH));
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
        const READ_BUF_SIZE: usize = 8 * 1024 * 1024;
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
            for _ in 1..vec_len {
                // Ensure we have some bytes available
                refill_if_low!();

                let (delta, consumed) = decode_varint(&read_buf[buf_pos..buf_len]);
                buf_pos += consumed;

                let val = prev + delta;
                vec.push(val);
                prev = val;
            }

            buckets.insert(id, vec);
        }

        Ok(Index { k, w, salt, buckets, bucket_names: names, bucket_sources: sources })
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
        if &buf4 != b"RYP5" { return Err(anyhow!("Invalid Index Format (Expected RYP5)")); }

        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        const SUPPORTED_VERSION: u32 = 5;
        if version != SUPPORTED_VERSION {
            return Err(anyhow!("Unsupported index version: {} (expected {})", version, SUPPORTED_VERSION));
        }

        reader.read_exact(&mut buf8)?;
        let k = u64::from_le_bytes(buf8) as usize;
        if !matches!(k, 16 | 32 | 64) {
            return Err(anyhow!("Invalid K value in index: {} (must be 16, 32, or 64)", k));
        }
        reader.read_exact(&mut buf8)?; let w = u64::from_le_bytes(buf8) as usize;
        reader.read_exact(&mut buf8)?; let salt = u64::from_le_bytes(buf8);
        reader.read_exact(&mut buf4)?; let num = u32::from_le_bytes(buf4);
        if num > MAX_NUM_BUCKETS {
            return Err(anyhow!("Number of buckets {} exceeds maximum {}", num, MAX_NUM_BUCKETS));
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
                return Err(anyhow!("Bucket size {} exceeds maximum {}", vec_len, MAX_BUCKET_SIZE));
            }

            reader.read_exact(&mut buf4)?; let id = u32::from_le_bytes(buf4);

            reader.read_exact(&mut buf8)?; let name_len = u64::from_le_bytes(buf8) as usize;
            if name_len > MAX_STRING_LENGTH {
                return Err(anyhow!("Bucket name length {} exceeds maximum {}", name_len, MAX_STRING_LENGTH));
            }
            let mut nbuf = vec![0u8; name_len];
            reader.read_exact(&mut nbuf)?;
            names.insert(id, String::from_utf8(nbuf)?);

            reader.read_exact(&mut buf8)?; let src_count = u64::from_le_bytes(buf8) as usize;
            let mut src_list = Vec::with_capacity(src_count);
            for _ in 0..src_count {
                reader.read_exact(&mut buf8)?; let slen = u64::from_le_bytes(buf8) as usize;
                if slen > MAX_STRING_LENGTH {
                    return Err(anyhow!("Source string length {} exceeds maximum {}", slen, MAX_STRING_LENGTH));
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

// --- INVERTED INDEX ---

/// CSR-format inverted index for fast minimizer → bucket lookups.
///
/// The inverted index maps each unique minimizer to the set of bucket IDs
/// that contain it. Uses hybrid binary search for efficient batch lookups.
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
/// InvertedIndex can be shared across threads for concurrent classification (it only contains Vecs which are Sync).
#[derive(Debug)]
pub struct InvertedIndex {
    pub k: usize,
    pub w: usize,
    pub salt: u64,
    source_hash: u64,
    minimizers: Vec<u64>,      // sorted unique minimizers (ascending, no duplicates)
    offsets: Vec<u32>,         // CSR offsets (length = minimizers.len() + 1, monotonically increasing, max 4B entries)
    bucket_ids: Vec<u32>,      // flattened bucket IDs
}

impl InvertedIndex {
    /// Compute a hash from index metadata for validation.
    /// Hash is computed from sorted (bucket_id, minimizer_count) pairs.
    pub fn compute_metadata_hash(metadata: &IndexMetadata) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut pairs: Vec<(u32, usize)> = metadata.bucket_minimizer_counts
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
    /// by leveraging the fact that each bucket is already sorted. The k-way merge
    /// is O(n log k) where k = number of buckets.
    ///
    /// # Requirements
    /// All buckets in the index must be finalized (sorted and deduplicated).
    /// Call `finalize_bucket()` on each bucket before calling this function.
    ///
    /// # Panics
    /// Panics if any bucket is not sorted (indicates unfinalized bucket).
    pub fn build_from_index(index: &Index) -> Self {
        // Verify all buckets are finalized (sorted) - can be done in parallel
        let bucket_vec: Vec<_> = index.buckets.iter().collect();
        bucket_vec.par_iter().for_each(|(&id, minimizers)| {
            debug_assert!(
                minimizers.windows(2).all(|w| w[0] <= w[1]),
                "Bucket {} is not sorted. Call finalize_bucket() first.", id
            );
            // In release builds, check and panic with a clear message
            if !minimizers.windows(2).all(|w| w[0] <= w[1]) {
                panic!("Bucket {} is not sorted. Call finalize_bucket() before building inverted index.", id);
            }
        });

        // Count total entries for capacity reservation
        let total_entries: usize = index.buckets.values().map(|v| v.len()).sum();

        // Estimate unique minimizers (heuristic: assume 50% overlap on average)
        let estimated_unique = total_entries / 2;

        // Build CSR structure with reserved capacity
        let mut minimizers: Vec<u64> = Vec::with_capacity(estimated_unique);
        let mut offsets: Vec<u32> = Vec::with_capacity(estimated_unique + 1);
        let mut bucket_ids_out: Vec<u32> = Vec::with_capacity(total_entries);

        offsets.push(0);

        // Handle empty index case
        if index.buckets.is_empty() || total_entries == 0 {
            let metadata = IndexMetadata {
                k: index.k,
                w: index.w,
                salt: index.salt,
                bucket_names: index.bucket_names.clone(),
                bucket_sources: index.bucket_sources.clone(),
                bucket_minimizer_counts: HashMap::new(),
            };
            let source_hash = Self::compute_metadata_hash(&metadata);

            return InvertedIndex {
                k: index.k,
                w: index.w,
                salt: index.salt,
                source_hash,
                minimizers,
                offsets,
                bucket_ids: bucket_ids_out,
            };
        }

        // K-way merge using a min-heap
        // Heap entry: (Reverse(minimizer), Reverse(bucket_id), bucket_index_in_vec, position_in_bucket)
        // Using Reverse because BinaryHeap is a max-heap
        // bucket_index_in_vec is used to look up the bucket data

        // Collect bucket data into a vec for indexed access
        let bucket_data: Vec<(u32, &[u64])> = index.buckets
            .iter()
            .filter(|(_, mins)| !mins.is_empty())
            .map(|(&id, mins)| (id, mins.as_slice()))
            .collect();

        // Initialize heap with first element from each non-empty bucket
        // Entry: (Reverse((minimizer, bucket_id)), bucket_data_index, position)
        let mut heap: BinaryHeap<(Reverse<(u64, u32)>, usize, usize)> = BinaryHeap::with_capacity(bucket_data.len());

        for (idx, &(bucket_id, mins)) in bucket_data.iter().enumerate() {
            heap.push((Reverse((mins[0], bucket_id)), idx, 0));
        }

        // Merge and build CSR
        let mut current_min: Option<u64> = None;
        let mut current_bucket_ids: Vec<u32> = Vec::new();

        while let Some((Reverse((min_val, bucket_id)), data_idx, pos)) = heap.pop() {
            // Check if this is a new minimizer
            if current_min != Some(min_val) {
                // Finalize previous minimizer if exists
                if current_min.is_some() {
                    // Sort bucket_ids for determinism and dedup
                    current_bucket_ids.sort_unstable();
                    current_bucket_ids.dedup();
                    bucket_ids_out.extend_from_slice(&current_bucket_ids);
                    // Safe conversion: panic if bucket_ids exceeds u32::MAX (indicates corrupted/too large data)
                    offsets.push(u32::try_from(bucket_ids_out.len())
                        .expect("CSR offset overflow: bucket_ids exceeded u32::MAX"));
                    current_bucket_ids.clear();
                }
                minimizers.push(min_val);
                current_min = Some(min_val);
            }

            // Add this bucket_id to current minimizer's list
            current_bucket_ids.push(bucket_id);

            // Push next element from this bucket if available
            let next_pos = pos + 1;
            let (_, bucket_mins) = bucket_data[data_idx];
            if next_pos < bucket_mins.len() {
                let next_bucket_id = bucket_data[data_idx].0;
                heap.push((Reverse((bucket_mins[next_pos], next_bucket_id)), data_idx, next_pos));
            }
        }

        // Finalize last minimizer
        if !current_bucket_ids.is_empty() {
            current_bucket_ids.sort_unstable();
            current_bucket_ids.dedup();
            bucket_ids_out.extend_from_slice(&current_bucket_ids);
            offsets.push(u32::try_from(bucket_ids_out.len())
                .expect("CSR offset overflow: bucket_ids exceeded u32::MAX"));
        }

        // Shrink to fit to reclaim excess capacity
        minimizers.shrink_to_fit();
        offsets.shrink_to_fit();
        bucket_ids_out.shrink_to_fit();

        // Compute source hash from the index
        let metadata = IndexMetadata {
            k: index.k,
            w: index.w,
            salt: index.salt,
            bucket_names: index.bucket_names.clone(),
            bucket_sources: index.bucket_sources.clone(),
            bucket_minimizer_counts: index.buckets.iter()
                .map(|(&id, v)| (id, v.len()))
                .collect(),
        };
        let source_hash = Self::compute_metadata_hash(&metadata);

        InvertedIndex {
            k: index.k,
            w: index.w,
            salt: index.salt,
            source_hash,
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
                self.k, self.w, self.salt,
                metadata.k, metadata.w, metadata.salt
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
    /// Complexity: O(Q * log(U)) worst case, where Q = query length and U = unique minimizers.
    /// When queries have good coverage of the minimizer space, the amortized complexity
    /// approaches O(Q + Q * log(U/Q)) because the search range shrinks with each match.
    ///
    /// # Arguments
    /// * `query` - A sorted, deduplicated slice of minimizer values
    ///
    /// # Returns
    /// HashMap of bucket_id -> hit_count for all buckets that match at least one query minimizer.
    pub fn get_bucket_hits(&self, query: &[u64]) -> HashMap<u32, usize> {
        let mut hits: HashMap<u32, usize> = HashMap::new();

        if query.is_empty() || self.minimizers.is_empty() {
            return hits;
        }

        let mut search_start = 0;

        for &q in query {
            // Early exit if we've exhausted the index
            if search_start >= self.minimizers.len() {
                break;
            }

            // Binary search for q in minimizers[search_start..]
            match self.minimizers[search_start..].binary_search(&q) {
                Ok(relative_idx) => {
                    let abs_idx = search_start + relative_idx;
                    // Found! Get bucket IDs for this minimizer
                    let start = self.offsets[abs_idx] as usize;
                    let end = self.offsets[abs_idx + 1] as usize;
                    for &bid in &self.bucket_ids[start..end] {
                        *hits.entry(bid).or_insert(0) += 1;
                    }
                    // Move search_start past the found element
                    search_start = abs_idx + 1;
                }
                Err(relative_idx) => {
                    // Not found. relative_idx is the insertion point in the slice.
                    // If relative_idx == 0, the query element is smaller than all remaining
                    // index elements. Since query is sorted, ALL subsequent query elements
                    // that are <= q will also not match, so we can update search_start.
                    search_start += relative_idx;
                    // Early exit optimization: if relative_idx == 0, this query element
                    // is smaller than minimizers[search_start]. Since query is sorted,
                    // we continue to the next query element (which may be larger).
                    // No action needed here - the loop continues naturally.
                }
            }
        }

        hits
    }

    /// Save the inverted index to a file (zstd streaming compressed, u32 offsets, adaptive bucket IDs).
    ///
    /// Format v3:
    /// - Header (uncompressed, 60 bytes): magic "RYXI", version 3, k, w, salt, source_hash, max_bucket_id, num_minimizers, num_bucket_ids
    /// - Payload (zstd compressed stream to EOF): offsets (u32), minimizers (delta+varint), bucket_ids (u8/u16/u32 based on max_bucket_id)
    ///
    /// Uses streaming compression with chunked writes for efficiency.
    /// Minimizers are delta+varint encoded for ~65% compression.
    pub fn save(&self, path: &Path) -> Result<()> {
        // Validate sizes before saving
        if self.bucket_ids.len() > MAX_INVERTED_BUCKET_IDS {
            return Err(anyhow!("Cannot save: bucket_ids length {} exceeds maximum {}",
                self.bucket_ids.len(), MAX_INVERTED_BUCKET_IDS));
        }
        if self.minimizers.len() > MAX_INVERTED_MINIMIZERS {
            return Err(anyhow!("Cannot save: minimizers length {} exceeds maximum {}",
                self.minimizers.len(), MAX_INVERTED_MINIMIZERS));
        }

        // Validate minimizers are sorted and unique (required for delta encoding)
        if !self.minimizers.windows(2).all(|w| w[0] < w[1]) {
            return Err(anyhow!("Cannot save: minimizers must be sorted and unique"));
        }

        let mut writer = BufWriter::new(File::create(path)?);

        // Determine max bucket ID for adaptive sizing
        let max_bucket_id = self.bucket_ids.iter().copied().max().unwrap_or(0);

        // Magic and version
        writer.write_all(b"RYXI")?;
        writer.write_all(&3u32.to_le_bytes())?;

        // Metadata (uncompressed header)
        writer.write_all(&(self.k as u64).to_le_bytes())?;
        writer.write_all(&(self.w as u64).to_le_bytes())?;
        writer.write_all(&self.salt.to_le_bytes())?;
        writer.write_all(&self.source_hash.to_le_bytes())?;

        // Sizes and adaptive bucket ID info
        writer.write_all(&max_bucket_id.to_le_bytes())?;
        writer.write_all(&(self.minimizers.len() as u64).to_le_bytes())?;
        writer.write_all(&(self.bucket_ids.len() as u64).to_le_bytes())?;

        // Create streaming zstd encoder (level 3 is a good balance of speed/compression)
        let mut encoder = zstd::stream::write::Encoder::new(writer, 3)?;

        // Write buffer for batching small writes (8MB chunks)
        const WRITE_BUF_SIZE: usize = 8 * 1024 * 1024;
        let mut write_buf = Vec::with_capacity(WRITE_BUF_SIZE);

        // Helper closure to flush buffer when full
        let flush_buf = |buf: &mut Vec<u8>, encoder: &mut zstd::stream::write::Encoder<BufWriter<File>>| -> Result<()> {
            if !buf.is_empty() {
                encoder.write_all(buf)?;
                buf.clear();
            }
            Ok(())
        };

        // Stream offsets (u32) in chunks
        for &offset in &self.offsets {
            write_buf.extend_from_slice(&offset.to_le_bytes());
            if write_buf.len() >= WRITE_BUF_SIZE {
                flush_buf(&mut write_buf, &mut encoder)?;
            }
        }
        flush_buf(&mut write_buf, &mut encoder)?;

        // Stream minimizers using delta+varint encoding
        // First minimizer is stored as full u64, rest are delta-encoded varints
        let mut varint_buf = [0u8; 10]; // Max 10 bytes for LEB128 u64
        if !self.minimizers.is_empty() {
            // First minimizer: store as full u64
            write_buf.extend_from_slice(&self.minimizers[0].to_le_bytes());
            if write_buf.len() >= WRITE_BUF_SIZE {
                flush_buf(&mut write_buf, &mut encoder)?;
            }

            // Remaining minimizers: store as delta-encoded varints
            let mut prev = self.minimizers[0];
            for &min in &self.minimizers[1..] {
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

        // Stream bucket IDs (adaptive sizing) in chunks
        if max_bucket_id <= u8::MAX as u32 {
            for &bid in &self.bucket_ids {
                write_buf.push(bid as u8);
                if write_buf.len() >= WRITE_BUF_SIZE {
                    flush_buf(&mut write_buf, &mut encoder)?;
                }
            }
        } else if max_bucket_id <= u16::MAX as u32 {
            for &bid in &self.bucket_ids {
                write_buf.extend_from_slice(&(bid as u16).to_le_bytes());
                if write_buf.len() >= WRITE_BUF_SIZE {
                    flush_buf(&mut write_buf, &mut encoder)?;
                }
            }
        } else {
            for &bid in &self.bucket_ids {
                write_buf.extend_from_slice(&bid.to_le_bytes());
                if write_buf.len() >= WRITE_BUF_SIZE {
                    flush_buf(&mut write_buf, &mut encoder)?;
                }
            }
        }
        flush_buf(&mut write_buf, &mut encoder)?;

        // Finish compression and flush
        encoder.finish()?;

        Ok(())
    }

    /// Load an inverted index from a file using streaming decompression.
    ///
    /// Uses streaming decompression to avoid buffering the entire payload in memory.
    pub fn load(path: &Path) -> Result<Self> {
        let mut reader = BufReader::new(File::open(path)?);
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        // Magic
        reader.read_exact(&mut buf4)?;
        if &buf4 != b"RYXI" {
            return Err(anyhow!("Invalid inverted index format (expected RYXI)"));
        }

        // Version
        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != 3 {
            return Err(anyhow!("Unsupported inverted index version: {} (expected 3)", version));
        }

        // Metadata
        reader.read_exact(&mut buf8)?;
        let k = u64::from_le_bytes(buf8) as usize;
        if !matches!(k, 16 | 32 | 64) {
            return Err(anyhow!("Invalid K value in inverted index: {} (must be 16, 32, or 64)", k));
        }

        reader.read_exact(&mut buf8)?;
        let w = u64::from_le_bytes(buf8) as usize;

        reader.read_exact(&mut buf8)?;
        let salt = u64::from_le_bytes(buf8);

        reader.read_exact(&mut buf8)?;
        let source_hash = u64::from_le_bytes(buf8);

        // Sizes and adaptive bucket ID info
        reader.read_exact(&mut buf4)?;
        let max_bucket_id = u32::from_le_bytes(buf4);

        reader.read_exact(&mut buf8)?;
        let num_minimizers = u64::from_le_bytes(buf8) as usize;
        if num_minimizers > MAX_INVERTED_MINIMIZERS {
            return Err(anyhow!("Number of minimizers {} exceeds maximum {}",
                num_minimizers, MAX_INVERTED_MINIMIZERS));
        }

        reader.read_exact(&mut buf8)?;
        let num_bucket_ids = u64::from_le_bytes(buf8) as usize;
        if num_bucket_ids > MAX_INVERTED_BUCKET_IDS {
            return Err(anyhow!("Number of bucket IDs {} exceeds maximum {}",
                num_bucket_ids, MAX_INVERTED_BUCKET_IDS));
        }

        // Calculate expected allocation sizes for validation
        let offsets_len = num_minimizers.checked_add(1)
            .ok_or_else(|| anyhow!("Offset array length overflow"))?;

        // Sanity check on total allocation size (prevent DoS via huge allocations)
        let total_elements = (offsets_len as u64)
            .checked_add(num_minimizers as u64)
            .and_then(|s| s.checked_add(num_bucket_ids as u64))
            .ok_or_else(|| anyhow!("Total element count overflow"))?;
        const MAX_TOTAL_ELEMENTS: u64 = 100_000_000_000; // ~100B elements
        if total_elements > MAX_TOTAL_ELEMENTS {
            return Err(anyhow!(
                "Total element count {} exceeds maximum {}",
                total_elements, MAX_TOTAL_ELEMENTS
            ));
        }

        // Create streaming zstd decoder for the rest of the file
        let mut decoder = zstd::stream::read::Decoder::new(reader)
            .map_err(|e| anyhow!("Failed to create zstd decoder: {}", e))?;

        // Read buffer for batching small reads (8MB chunks)
        const READ_BUF_SIZE: usize = 8 * 1024 * 1024;

        // Read offsets (u32) in chunks
        let mut offsets = Vec::with_capacity(offsets_len);
        {
            let bytes_needed = offsets_len * 4;
            let mut read_buf = vec![0u8; bytes_needed.min(READ_BUF_SIZE)];
            let mut total_read = 0;

            while total_read < bytes_needed {
                let chunk_size = (bytes_needed - total_read).min(READ_BUF_SIZE);
                decoder.read_exact(&mut read_buf[..chunk_size])
                    .map_err(|e| anyhow!("Failed to read offsets: {}", e))?;

                for chunk in read_buf[..chunk_size].chunks_exact(4) {
                    offsets.push(u32::from_le_bytes(chunk.try_into().unwrap()));
                }
                total_read += chunk_size;
            }
        }

        // Validate CSR invariants
        if !offsets.is_empty() {
            if offsets[0] != 0 {
                return Err(anyhow!("Invalid CSR format: first offset must be 0, got {}", offsets[0]));
            }
            for i in 1..offsets.len() {
                if offsets[i] < offsets[i - 1] {
                    return Err(anyhow!(
                        "Invalid CSR format: offsets must be monotonically increasing (offset[{}]={} < offset[{}]={})",
                        i, offsets[i], i - 1, offsets[i - 1]
                    ));
                }
            }
            if let Some(&last_offset) = offsets.last() {
                if last_offset as usize != num_bucket_ids {
                    return Err(anyhow!(
                        "Invalid CSR format: final offset {} doesn't match bucket_ids length {}",
                        last_offset, num_bucket_ids
                    ));
                }
            }
        }

        // Read minimizers using delta+varint decoding
        // First minimizer is stored as full u64, rest are delta-encoded varints
        let mut minimizers = Vec::with_capacity(num_minimizers);
        // Stack-allocated buffer for leftover bytes from varint decoding (max 9 bytes)
        let mut leftover_buf: [u8; 16] = [0; 16];
        let mut leftover_len: usize = 0;

        if num_minimizers > 0 {
            // Read first minimizer as full u64
            decoder.read_exact(&mut buf8)
                .map_err(|e| anyhow!("Failed to read first minimizer: {}", e))?;
            let first_min = u64::from_le_bytes(buf8);
            minimizers.push(first_min);

            // Read remaining minimizers as delta-encoded varints
            if num_minimizers > 1 {
                let mut read_buf = vec![0u8; READ_BUF_SIZE];
                let mut buf_pos: usize = 0;
                let mut buf_len: usize = 0;
                let mut prev = first_min;

                // Macro to refill buffer when fewer than 10 bytes remain (max varint size)
                macro_rules! refill_if_low {
                    () => {
                        let remaining = buf_len - buf_pos;
                        if remaining < 10 && buf_len > 0 {
                            // Move remaining bytes to the front
                            if remaining > 0 && buf_pos > 0 {
                                read_buf.copy_within(buf_pos..buf_len, 0);
                            }
                            buf_len = remaining;
                            buf_pos = 0;
                            // Try to read more data (EOF returns Ok(0), Err is a real error)
                            match decoder.read(&mut read_buf[buf_len..]) {
                                Ok(n) => buf_len += n,
                                Err(e) => return Err(anyhow!("I/O error reading minimizers: {}", e)),
                            }
                        } else if buf_len == 0 {
                            // Initial fill
                            match decoder.read(&mut read_buf) {
                                Ok(n) => buf_len = n,
                                Err(e) => return Err(anyhow!("Failed to read minimizers: {}", e)),
                            }
                        }
                    };
                }

                for i in 1..num_minimizers {
                    refill_if_low!();
                    if buf_pos >= buf_len {
                        return Err(anyhow!("Unexpected end of data at minimizer {}", i));
                    }

                    let (delta, consumed) = decode_varint(&read_buf[buf_pos..buf_len]);
                    buf_pos += consumed;

                    // Use checked_add to detect overflow (corrupted data)
                    let min = prev.checked_add(delta)
                        .ok_or_else(|| anyhow!(
                            "Invalid inverted index: minimizer overflow at index {} (prev={}, delta={})",
                            i, prev, delta
                        ))?;

                    // Validate minimizers are strictly increasing (sorted, unique)
                    if min <= prev {
                        return Err(anyhow!(
                            "Invalid inverted index: minimizers not strictly increasing (minimizers[{}]={} <= minimizers[{}]={})",
                            i, min, i - 1, prev
                        ));
                    }
                    minimizers.push(min);
                    prev = min;
                }

                // Save remaining bytes for bucket ID reading (stack allocated)
                leftover_len = buf_len - buf_pos;
                leftover_buf[..leftover_len].copy_from_slice(&read_buf[buf_pos..buf_len]);
            }
        }

        // Read bucket IDs (adaptive sizing)
        // First consume any leftover bytes from varint decoding, then read from decoder
        let mut bucket_ids = Vec::with_capacity(num_bucket_ids);
        let mut leftover_pos = 0; // Position in leftover_buf

        // Helper macro to read exact bytes, consuming leftover first then decoder
        macro_rules! read_exact_with_leftover {
            ($buf:expr) => {{
                let buf: &mut [u8] = $buf;
                let mut filled = 0;
                // First, consume from leftover buffer
                if leftover_pos < leftover_len {
                    let from_leftover = (leftover_len - leftover_pos).min(buf.len());
                    buf[..from_leftover].copy_from_slice(&leftover_buf[leftover_pos..leftover_pos + from_leftover]);
                    leftover_pos += from_leftover;
                    filled = from_leftover;
                }
                // Then read remainder from decoder
                if filled < buf.len() {
                    decoder.read_exact(&mut buf[filled..])
                        .map_err(|e| anyhow!("Failed to read bucket_ids: {}", e))?;
                }
                Ok::<(), anyhow::Error>(())
            }};
        }

        if max_bucket_id <= u8::MAX as u32 {
            // u8 bucket IDs - no validation needed (any u8 is valid when max <= 255)
            let bytes_needed = num_bucket_ids;
            let mut read_buf = vec![0u8; bytes_needed.min(READ_BUF_SIZE)];
            let mut total_read = 0;

            while total_read < bytes_needed {
                let chunk_size = (bytes_needed - total_read).min(READ_BUF_SIZE);
                read_exact_with_leftover!(&mut read_buf[..chunk_size])?;

                for &b in &read_buf[..chunk_size] {
                    bucket_ids.push(b as u32);
                }
                total_read += chunk_size;
            }
        } else if max_bucket_id <= u16::MAX as u32 {
            // u16 bucket IDs
            let bytes_needed = num_bucket_ids * 2;
            let mut read_buf = vec![0u8; bytes_needed.min(READ_BUF_SIZE)];
            let mut total_read = 0;
            let mut idx = 0;

            while total_read < bytes_needed {
                let chunk_size = (bytes_needed - total_read).min(READ_BUF_SIZE);
                read_exact_with_leftover!(&mut read_buf[..chunk_size])?;

                for chunk in read_buf[..chunk_size].chunks_exact(2) {
                    let bid = u16::from_le_bytes(chunk.try_into().unwrap()) as u32;
                    if bid > max_bucket_id {
                        return Err(anyhow!(
                            "Invalid inverted index: bucket_id[{}]={} exceeds declared max_bucket_id={}",
                            idx, bid, max_bucket_id
                        ));
                    }
                    bucket_ids.push(bid);
                    idx += 1;
                }
                total_read += chunk_size;
            }

            // Validate we read exactly what we expected
            if total_read != bytes_needed {
                return Err(anyhow!(
                    "Incomplete bucket_id read: expected {} bytes, got {}",
                    bytes_needed, total_read
                ));
            }
        } else {
            // u32 bucket IDs
            let bytes_needed = num_bucket_ids * 4;
            let mut read_buf = vec![0u8; bytes_needed.min(READ_BUF_SIZE)];
            let mut total_read = 0;
            let mut idx = 0;

            while total_read < bytes_needed {
                let chunk_size = (bytes_needed - total_read).min(READ_BUF_SIZE);
                read_exact_with_leftover!(&mut read_buf[..chunk_size])?;

                for chunk in read_buf[..chunk_size].chunks_exact(4) {
                    let bid = u32::from_le_bytes(chunk.try_into().unwrap());
                    if bid > max_bucket_id {
                        return Err(anyhow!(
                            "Invalid inverted index: bucket_id[{}]={} exceeds declared max_bucket_id={}",
                            idx, bid, max_bucket_id
                        ));
                    }
                    bucket_ids.push(bid);
                    idx += 1;
                }
                total_read += chunk_size;
            }

            // Validate we read exactly what we expected
            if total_read != bytes_needed {
                return Err(anyhow!(
                    "Incomplete bucket_id read: expected {} bytes, got {}",
                    bytes_needed, total_read
                ));
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

    /// Returns the number of unique minimizers in the index.
    pub fn num_minimizers(&self) -> usize {
        self.minimizers.len()
    }

    /// Returns the total number of bucket ID entries.
    pub fn num_bucket_entries(&self) -> usize {
        self.bucket_ids.len()
    }
}

// --- LIBRARY LEVEL PROCESSING ---

pub fn classify_batch(
    engine: &Index,
    records: &[QueryRecord],
    threshold: f64
) -> Vec<HitResult> {

    let processed: Vec<_> = records.par_iter()
        .map_init(MinimizerWorkspace::new, |ws, (id, s1, s2)| {
            let (ha, hb) = get_paired_minimizers_into(s1, *s2, engine.k, engine.w, engine.salt, ws);
            (*id, ha, hb)
        }).collect();

    let mut map_a: HashMap<u64, Vec<usize>> = HashMap::new();
    let mut map_b: HashMap<u64, Vec<usize>> = HashMap::new();
    let mut uniq_mins = HashSet::new();

    for (idx, (_, ma, mb)) in processed.iter().enumerate() {
        for &m in ma { map_a.entry(m).or_default().push(idx); uniq_mins.insert(m); }
        for &m in mb { map_b.entry(m).or_default().push(idx); uniq_mins.insert(m); }
    }
    let uniq_vec: Vec<u64> = uniq_mins.into_iter().collect();

    let results: Vec<HitResult> = engine.buckets.par_iter().map(|(b_id, bucket)| {
        let mut hits = HashMap::new();

        for &m in &uniq_vec {
            if bucket.binary_search(&m).is_ok() {
                if let Some(rs) = map_a.get(&m) { for &r in rs { hits.entry(r).or_insert((0,0)).0 += 1; } }
                if let Some(rs) = map_b.get(&m) { for &r in rs { hits.entry(r).or_insert((0,0)).1 += 1; } }
            }
        }

        let mut bucket_results = Vec::new();
        for (r_idx, (hits_a, hits_b)) in hits {
            let (qid, ha, hb) = &processed[r_idx];
            let la = ha.len() as f64;
            let lb = hb.len() as f64;
            
            let score = (if la > 0.0 { hits_a as f64 / la } else { 0.0 })
                .max(if lb > 0.0 { hits_b as f64 / lb } else { 0.0 });
            
            if score >= threshold {
                bucket_results.push(HitResult { query_id: *qid, bucket_id: *b_id, score });
            }
        }
        bucket_results
    }).flatten().collect();

    results
}

/// Classify a batch of query records using an inverted index.
///
/// This function uses the inverted index for O(Q * log(U/Q)) lookups per query
/// instead of O(B * Q * log(M)) where B = buckets, Q = query minimizers, M = avg bucket size.
///
/// # Arguments
/// * `inverted` - The inverted index for minimizer → bucket lookups
/// * `records` - Batch of query records to classify
/// * `threshold` - Minimum score threshold for reporting hits
///
/// # Returns
/// Vector of HitResult for all (query, bucket) pairs meeting the threshold
pub fn classify_batch_inverted(
    inverted: &InvertedIndex,
    records: &[QueryRecord],
    threshold: f64
) -> Vec<HitResult> {
    // Extract minimizers for all queries in parallel
    let processed: Vec<_> = records.par_iter()
        .map_init(MinimizerWorkspace::new, |ws, (id, s1, s2)| {
            let (ha, hb) = get_paired_minimizers_into(s1, *s2, inverted.k, inverted.w, inverted.salt, ws);
            (*id, ha, hb)
        }).collect();

    // Process each query and collect results
    let results: Vec<HitResult> = processed.par_iter()
        .flat_map(|(query_id, fwd_mins, rc_mins)| {
            // Get bucket hits for forward strand
            let fwd_hits = inverted.get_bucket_hits(fwd_mins);
            let fwd_len = fwd_mins.len() as f64;

            // Get bucket hits for reverse complement strand
            let rc_hits = inverted.get_bucket_hits(rc_mins);
            let rc_len = rc_mins.len() as f64;

            // Merge hits into a single map (bucket_id -> (fwd_count, rc_count))
            // Reserve capacity based on combined unique buckets (upper bound: sum of both)
            let capacity = fwd_hits.len() + rc_hits.len();
            let mut scores: HashMap<u32, (usize, usize)> = HashMap::with_capacity(capacity);
            for (&bucket_id, &count) in &fwd_hits {
                scores.entry(bucket_id).or_insert((0, 0)).0 = count;
            }
            for (&bucket_id, &count) in &rc_hits {
                scores.entry(bucket_id).or_insert((0, 0)).1 = count;
            }

            // Compute scores and filter by threshold
            let mut query_results = Vec::with_capacity(scores.len());
            for (bucket_id, (fwd_count, rc_count)) in scores {
                let score = (if fwd_len > 0.0 { fwd_count as f64 / fwd_len } else { 0.0 })
                    .max(if rc_len > 0.0 { rc_count as f64 / rc_len } else { 0.0 });

                if score >= threshold {
                    query_results.push(HitResult {
                        query_id: *query_id,
                        bucket_id,
                        score,
                    });
                }
            }
            query_results
        })
        .collect();

    results
}

pub fn aggregate_batch(
    engine: &Index,
    records: &[QueryRecord],
    threshold: f64
) -> Vec<HitResult> {
    let mut global = HashSet::new();

    let batch_mins: Vec<Vec<u64>> = records.par_iter()
        .map_init(MinimizerWorkspace::new, |ws, (_, s1, s2)| {
            let (mut a, b) = get_paired_minimizers_into(s1, *s2, engine.k, engine.w, engine.salt, ws);
            a.extend(b);
            a
        }).collect();

    for v in batch_mins { for m in v { global.insert(m); } }
    
    let total = global.len() as f64;
    if total == 0.0 { return Vec::new(); }
    
    let g_vec: Vec<u64> = global.into_iter().collect();
    
    engine.buckets.par_iter().filter_map(|(id, b)| {
        let s = count_hits(&g_vec, b) / total;
        if s >= threshold { 
            Some(HitResult { query_id: -1, bucket_id: *id, score: s }) 
        } else { 
            None 
        }
    }).collect()
}


#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    // --- UTILS ---

    #[allow(dead_code)]
    fn create_temp_fasta(content: &[u8]) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("Failed to create temp file");
        file.write_all(content).expect("Failed to write to temp file");
        file
    }

    // --- BASIC LOGIC TESTS ---

    #[test]
    fn test_lut_accuracy() {
        assert_eq!(base_to_bit(b'A'), 1);
        assert_eq!(base_to_bit(b'G'), 1);
        assert_eq!(base_to_bit(b'T'), 0);
        assert_eq!(base_to_bit(b'C'), 0);
        assert_eq!(base_to_bit(b'N'), u64::MAX);
    }

    #[test]
    fn test_valid_extraction_long() {
        let mut ws = MinimizerWorkspace::new();
        let seq = vec![b'A'; 70]; 
        extract_into(&seq, 64, 5, 0, &mut ws);
        assert!(!ws.buffer.is_empty(), "Should extract minimizers from valid long seq");
    }

    #[test]
    fn test_short_sequences_ignored() {
        let mut ws = MinimizerWorkspace::new();
        let seq = vec![b'A'; 60]; 
        extract_into(&seq, 64, 5, 0, &mut ws);
        assert!(ws.buffer.is_empty(), "Should not extract minimizers from seq < K");
    }

    #[test]
    fn test_n_handling_separator() {
        let mut ws = MinimizerWorkspace::new();
        let seq_a: Vec<u8> = (0..80).map(|i| if i % 2 == 0 { b'A' } else { b'T' }).collect(); 
        let seq_b: Vec<u8> = (0..80).map(|i| if i % 3 == 0 { b'G' } else { b'C' }).collect(); 
        
        extract_into(&seq_a, 64, 5, 0, &mut ws);
        let mins_a = ws.buffer.clone();

        extract_into(&seq_b, 64, 5, 0, &mut ws);
        let mins_b = ws.buffer.clone();

        let mut seq_combined = seq_a.clone();
        seq_combined.push(b'N'); // N separator
        seq_combined.extend_from_slice(&seq_b);

        extract_into(&seq_combined, 64, 5, 0, &mut ws);
        let mins_combined = ws.buffer.clone();

        let mut expected = mins_a;
        expected.extend(mins_b);
        
        assert_eq!(mins_combined, expected, "N should act as a perfect separator");
    }

    #[test]
    fn test_dual_strand_extraction() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCC"; // Long seq
        let (fwd, rc) = extract_dual_strand_into(seq, 64, 5, 0, &mut ws);
        assert!(!fwd.is_empty());
        assert!(!rc.is_empty());
        assert_ne!(fwd, rc); 
    }

    // --- INDEX OPERATIONS & FILE IO TESTS ---

    #[test]
    fn test_index_save_load_roundtrip() -> Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();
        let mut index = Index::new(64, 50, 0x1234).unwrap();

        index.buckets.insert(1, vec![100, 200]);
        index.buckets.insert(2, vec![300, 400]);
        index.bucket_names.insert(1, "BucketA".into());
        index.bucket_names.insert(2, "BucketB".into());
        index.bucket_sources.insert(1, vec!["file1.fa::seq1".into()]);

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
    fn test_merge_buckets_logic() -> Result<()> {
        let mut index = Index::new(64, 50, 0).unwrap();
        
        index.buckets.insert(10, vec![1, 2, 3]);
        index.bucket_names.insert(10, "Source".into());
        index.bucket_sources.insert(10, vec!["s1".into()]);

        index.buckets.insert(20, vec![3, 4, 5]);
        index.bucket_names.insert(20, "Dest".into());
        index.bucket_sources.insert(20, vec!["d1".into()]);

        index.merge_buckets(10, 20)?;

        assert!(!index.buckets.contains_key(&10));
        assert!(index.buckets.contains_key(&20));
        
        let merged_vec = &index.buckets[&20];
        assert_eq!(merged_vec, &vec![1, 2, 3, 4, 5]);

        let merged_sources = &index.bucket_sources[&20];
        assert_eq!(merged_sources.len(), 2);

        Ok(())
    }

    // --- LIBRARY BATCH PROCESSING TESTS ---

    #[test]
    fn test_classify_batch_logic() {
        let mut index = Index::new(64, 10, 0).unwrap();
        index.buckets.insert(1, vec![10, 20, 30, 40, 50]); 
        index.buckets.insert(2, vec![60, 70, 80, 90, 100]); 
        
        let seq_a = vec![b'A'; 80]; 
        let mut ws = MinimizerWorkspace::new();
        
        index.add_record(1, "ref_a", &seq_a, &mut ws);
        index.finalize_bucket(1);

        let query_seq = seq_a.clone();
        let records: Vec<QueryRecord> = vec![
            (101, &query_seq, None) 
        ];

        let results = classify_batch(&index, &records, 0.5);

        assert!(!results.is_empty());
        let hit = &results[0];
        assert_eq!(hit.query_id, 101);
        assert_eq!(hit.bucket_id, 1);
        assert_eq!(hit.score, 1.0); 
    }

    #[test]
    fn test_aggregate_batch_logic() {
        let mut index = Index::new(64, 10, 0).unwrap();
        let mut ws = MinimizerWorkspace::new();
        
        // Bucket 1: Poly-A (RY Space: All 1s)
        let seq_a = vec![b'A'; 100];
        index.add_record(1, "ref_a", &seq_a, &mut ws);
        index.finalize_bucket(1);

        // Bucket 2: Alternating AT (RY Space: Alternating 1/0)
        // A=1, T=0 => 10101010...
        // This generates distinct K-mers from Poly-A
        let seq_at: Vec<u8> = (0..200).map(|i| if i % 2 == 0 { b'A' } else { b'T' }).collect();
        index.add_record(2, "ref_at", &seq_at, &mut ws);
        index.finalize_bucket(2);

        // Query: Split the AT seq
        let q1 = &seq_at[0..100];
        let q2 = &seq_at[100..200];

        let records: Vec<QueryRecord> = vec![
            (1, q1, None),
            (2, q2, None),
        ];

        let results = aggregate_batch(&index, &records, 0.5);

        // Should strictly match Bucket 2
        assert_eq!(results.len(), 1, "Should only match bucket 2");
        assert_eq!(results[0].bucket_id, 2);
        assert!(results[0].score > 0.9);
    }

    // --- ERROR PATH TESTS ---

    #[test]
    fn test_index_load_invalid_format() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path();
        std::fs::write(path, b"NOTRYP5").unwrap();

        let result = Index::load(path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid Index Format"));
    }

    #[test]
    fn test_index_load_unsupported_version() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path();

        // Write valid header with wrong version
        let mut data = b"RYP5".to_vec();
        data.extend_from_slice(&999u32.to_le_bytes());  // Bad version
        std::fs::write(path, data).unwrap();

        let result = Index::load(path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unsupported index version"));
    }

    #[test]
    fn test_index_load_oversized_bucket() {
        // Create a malicious index file claiming a huge bucket (V5 format)
        let file = NamedTempFile::new().unwrap();
        let path = file.path();

        let mut data = Vec::new();
        data.extend_from_slice(b"RYP5");
        data.extend_from_slice(&5u32.to_le_bytes());  // Version 5
        data.extend_from_slice(&64u64.to_le_bytes()); // K
        data.extend_from_slice(&50u64.to_le_bytes()); // W
        data.extend_from_slice(&0u64.to_le_bytes());  // Salt
        data.extend_from_slice(&1u32.to_le_bytes());  // 1 bucket
        // V5 Format: MinimizerCount comes FIRST (in metadata section)
        data.extend_from_slice(&(MAX_BUCKET_SIZE as u64 + 1).to_le_bytes()); // Oversized!
        data.extend_from_slice(&1u32.to_le_bytes());  // Bucket ID 1
        data.extend_from_slice(&4u64.to_le_bytes());  // Name length
        data.extend_from_slice(b"test");
        data.extend_from_slice(&0u64.to_le_bytes());  // No sources

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
    fn test_merge_buckets_nonexistent_source() {
        let mut index = Index::new(64, 50, 0).unwrap();
        index.buckets.insert(1, vec![1, 2, 3]);

        let result = index.merge_buckets(999, 1);  // 999 doesn't exist
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not exist"));
    }

    #[test]
    fn test_merge_buckets_updates_minimizer_count() -> Result<()> {
        // Verify that merge_buckets correctly updates the minimizer count
        let mut index = Index::new(64, 50, 0).unwrap();

        // Create bucket 10 with 3 minimizers
        index.buckets.insert(10, vec![1, 2, 3]);
        index.bucket_names.insert(10, "Source".into());

        // Create bucket 20 with 3 minimizers (one overlapping)
        index.buckets.insert(20, vec![3, 4, 5]);
        index.bucket_names.insert(20, "Dest".into());

        // Before merge: bucket 20 has 3 minimizers
        assert_eq!(index.buckets[&20].len(), 3);

        // Merge 10 into 20
        index.merge_buckets(10, 20)?;

        // After merge: bucket 20 should have 5 unique minimizers (deduped)
        assert_eq!(index.buckets[&20].len(), 5);
        assert_eq!(index.buckets[&20], vec![1, 2, 3, 4, 5]);

        Ok(())
    }

    #[test]
    fn test_multiple_records_single_bucket() {
        // Verify that multiple records can be added to a single bucket
        let mut index = Index::new(64, 10, 0).unwrap();
        let mut ws = MinimizerWorkspace::new();

        // Add multiple sequences to the same bucket
        let seq1 = vec![b'A'; 80];
        let seq2 = vec![b'T'; 80];
        let seq3 = vec![b'G'; 80];

        index.add_record(1, "file.fa::seq1", &seq1, &mut ws);
        index.add_record(1, "file.fa::seq2", &seq2, &mut ws);
        index.add_record(1, "file.fa::seq3", &seq3, &mut ws);
        index.finalize_bucket(1);

        // All three sequences should be in bucket 1
        assert!(index.buckets.contains_key(&1));
        assert_eq!(index.bucket_sources[&1].len(), 3);
        assert_eq!(index.bucket_sources[&1][0], "file.fa::seq1");
        assert_eq!(index.bucket_sources[&1][1], "file.fa::seq2");
        assert_eq!(index.bucket_sources[&1][2], "file.fa::seq3");

        // Bucket should have minimizers (exact count depends on sequences)
        assert!(!index.buckets[&1].is_empty());
    }

    #[test]
    fn test_bucket_naming_consistency() -> Result<()> {
        // Verify that bucket names are consistent across operations
        let mut index = Index::new(64, 50, 0).unwrap();
        let mut ws = MinimizerWorkspace::new();

        // Simulate adding a file with multiple records to a new bucket
        let bucket_id = 1;
        let filename = "reference.fasta";

        // Set the bucket name to filename (consistent behavior)
        index.bucket_names.insert(bucket_id, filename.to_string());

        // Add multiple records to this bucket
        let seq1 = vec![b'A'; 80];
        let seq2 = vec![b'T'; 80];

        index.add_record(bucket_id, &format!("{}::seq1", filename), &seq1, &mut ws);
        index.add_record(bucket_id, &format!("{}::seq2", filename), &seq2, &mut ws);
        index.finalize_bucket(bucket_id);

        // Bucket name should be the filename
        assert_eq!(index.bucket_names[&bucket_id], filename);

        // Sources should include record names
        assert_eq!(index.bucket_sources[&bucket_id].len(), 2);
        assert!(index.bucket_sources[&bucket_id][0].contains("seq1"));
        assert!(index.bucket_sources[&bucket_id][1].contains("seq2"));

        Ok(())
    }

    // --- METADATA LOADING TESTS (V3 FORMAT) ---

    #[test]
    fn test_load_metadata_fast() -> Result<()> {
        // Verify that load_metadata() can read metadata without loading minimizers
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();
        let mut index = Index::new(64, 50, 0x1234)?;

        // Create index with multiple buckets with varying minimizer counts
        index.buckets.insert(1, vec![100, 200, 300]);
        index.bucket_names.insert(1, "BucketA".into());
        index.bucket_sources.insert(1, vec!["file1.fa::seq1".into(), "file1.fa::seq2".into()]);

        index.buckets.insert(2, vec![400, 500]);
        index.bucket_names.insert(2, "BucketB".into());
        index.bucket_sources.insert(2, vec!["file2.fa::seq1".into()]);

        index.buckets.insert(3, vec![600, 700, 800, 900]);
        index.bucket_names.insert(3, "BucketC".into());
        index.bucket_sources.insert(3, vec!["file3.fa::seq1".into(), "file3.fa::seq2".into(), "file3.fa::seq3".into()]);

        index.save(&path)?;

        // Load metadata only
        let metadata = Index::load_metadata(&path)?;

        // Verify basic parameters
        assert_eq!(metadata.k, 64);
        assert_eq!(metadata.w, 50);
        assert_eq!(metadata.salt, 0x1234);

        // Verify bucket names
        assert_eq!(metadata.bucket_names.len(), 3);
        assert_eq!(metadata.bucket_names[&1], "BucketA");
        assert_eq!(metadata.bucket_names[&2], "BucketB");
        assert_eq!(metadata.bucket_names[&3], "BucketC");

        // Verify bucket sources
        assert_eq!(metadata.bucket_sources[&1].len(), 2);
        assert_eq!(metadata.bucket_sources[&1][0], "file1.fa::seq1");
        assert_eq!(metadata.bucket_sources[&2].len(), 1);
        assert_eq!(metadata.bucket_sources[&3].len(), 3);

        // Verify minimizer counts (without loading actual minimizers)
        assert_eq!(metadata.bucket_minimizer_counts[&1], 3);
        assert_eq!(metadata.bucket_minimizer_counts[&2], 2);
        assert_eq!(metadata.bucket_minimizer_counts[&3], 4);

        Ok(())
    }

    #[test]
    fn test_load_metadata_matches_full_load() -> Result<()> {
        // Verify that metadata from load_metadata() matches metadata from load()
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();
        let mut index = Index::new(64, 42, 0xABCD).unwrap();

        index.buckets.insert(10, vec![1, 2, 3, 4, 5]);
        index.bucket_names.insert(10, "Test".into());
        index.bucket_sources.insert(10, vec!["src1".into(), "src2".into()]);

        index.save(&path)?;

        let metadata = Index::load_metadata(&path)?;
        let full_index = Index::load(&path)?;

        // Verify metadata matches
        assert_eq!(metadata.w, full_index.w);
        assert_eq!(metadata.salt, full_index.salt);
        assert_eq!(metadata.bucket_names, full_index.bucket_names);
        assert_eq!(metadata.bucket_sources, full_index.bucket_sources);
        assert_eq!(metadata.bucket_minimizer_counts[&10], full_index.buckets[&10].len());

        Ok(())
    }

    #[test]
    fn test_load_metadata_empty_index() -> Result<()> {
        // Verify load_metadata works with index containing no buckets
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
        // Verify V5 format (delta+varint+zstd-compressed minimizers) can save and load correctly
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();
        let mut index = Index::new(64, 50, 0x1234)?;

        // Create realistic index
        index.buckets.insert(1, vec![100, 200]);
        index.buckets.insert(2, vec![300, 400, 500]);
        index.bucket_names.insert(1, "BucketA".into());
        index.bucket_names.insert(2, "BucketB".into());
        index.bucket_sources.insert(1, vec!["file1.fa::seq1".into()]);
        index.bucket_sources.insert(2, vec!["file2.fa::seq1".into(), "file2.fa::seq2".into()]);

        index.save(&path)?;
        let loaded = Index::load(&path)?;

        // Verify complete roundtrip
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

    // --- NEW TESTS FOR K PARAMETERIZATION ---

    #[test]
    fn test_reverse_complement() {
        // K=16: Use non-palindromic pattern
        // 0xF0AA = 1111000010101010
        // Complement: 0000111101010101 = 0x0F55
        // Reverse:    1010101011110000 = 0xAAF0
        assert_eq!(reverse_complement(0xF0AA, 16), 0xAAF0);

        // K=32: Use non-palindromic pattern
        // 0xF0F0AAAA = 11110000111100001010101010101010
        // Complement:  00001111000011110101010101010101 = 0x0F0F5555
        // Reverse:     10101010101010101111000011110000 = 0xAAAAF0F0
        assert_eq!(reverse_complement(0xF0F0AAAA, 32), 0xAAAAF0F0);

        // K=64: Test with a simple pattern
        // All 1s complement to all 0s, which reversed is still all 0s
        assert_eq!(reverse_complement(0xFFFFFFFFFFFFFFFF, 64), 0x0000000000000000);

        // All 0s complement to all 1s, which reversed is still all 1s
        assert_eq!(reverse_complement(0x0000000000000000, 64), 0xFFFFFFFFFFFFFFFF);
    }

    #[test]
    fn test_index_k16() -> Result<()> {
        let mut index = Index::new(16, 50, 0)?;
        assert_eq!(index.k, 16);

        let mut ws = MinimizerWorkspace::new();
        // Need K+W-1 = 16+50-1 = 65 bases to generate minimizers
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
        // Need K+W-1 = 32+50-1 = 81 bases to generate minimizers
        let seq = vec![b'A'; 85];
        index.add_record(1, "test", &seq, &mut ws);
        index.finalize_bucket(1);

        assert!(index.buckets.contains_key(&1));
        assert!(!index.buckets[&1].is_empty());
        Ok(())
    }

    #[test]
    fn test_reject_invalid_k() {
        // Test that invalid K values are rejected
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
    fn test_extract_minimizers_k16() {
        let mut ws = MinimizerWorkspace::new();
        // 20 bases - enough for K=16 + small window
        let seq = b"AAAATTTTGGGGCCCCAAAA";
        extract_into(seq, 16, 5, 0, &mut ws);
        assert!(!ws.buffer.is_empty(), "Should extract minimizers with K=16");
    }

    #[test]
    fn test_extract_minimizers_k32() {
        let mut ws = MinimizerWorkspace::new();
        // 40 bases - enough for K=32 + small window
        let seq = b"AAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT";
        extract_into(seq, 32, 5, 0, &mut ws);
        assert!(!ws.buffer.is_empty(), "Should extract minimizers with K=32");
    }

    #[test]
    fn test_short_seq_k16() {
        let mut ws = MinimizerWorkspace::new();
        // Only 10 bases - too short for K=16
        let seq = b"AAAATTTTGG";
        extract_into(seq, 16, 5, 0, &mut ws);
        assert!(ws.buffer.is_empty(), "Should not extract from seq < K");
    }

    #[test]
    fn test_short_seq_k32() {
        let mut ws = MinimizerWorkspace::new();
        // Only 20 bases - too short for K=32
        let seq = b"AAAATTTTGGGGCCCCAAAA";
        extract_into(seq, 32, 5, 0, &mut ws);
        assert!(ws.buffer.is_empty(), "Should not extract from seq < K");
    }

    #[test]
    fn test_load_invalid_k_from_file() -> Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path();

        // Manually create an invalid index file with K=48
        let mut data = Vec::new();
        data.extend_from_slice(b"RYP5");
        data.extend_from_slice(&5u32.to_le_bytes());  // Version 5
        data.extend_from_slice(&48u64.to_le_bytes()); // Invalid K=48
        data.extend_from_slice(&50u64.to_le_bytes()); // W
        data.extend_from_slice(&0u64.to_le_bytes());  // Salt
        data.extend_from_slice(&0u32.to_le_bytes());  // 0 buckets

        std::fs::write(path, data)?;

        let result = Index::load(path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid K value"));

        Ok(())
    }

    // --- INVERTED INDEX TESTS ---

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
        // Unique minimizers: 100, 200, 300, 400, 500, 600 = 6
        assert_eq!(inverted.num_minimizers(), 6);
        // Bucket entries: 1->100, 1->200, 1->300, 2->200, 2->300, 2->400, 3->500, 3->600 = 8
        assert_eq!(inverted.num_bucket_entries(), 8);
    }

    #[test]
    fn test_inverted_index_save_load_roundtrip() -> Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();

        let mut index = Index::new(64, 50, 0xABCD).unwrap();
        index.buckets.insert(1, vec![10, 20, 30]);
        index.buckets.insert(2, vec![20, 30, 40]);
        index.bucket_names.insert(1, "A".into());
        index.bucket_names.insert(2, "B".into());

        let inverted = InvertedIndex::build_from_index(&index);
        inverted.save(&path)?;

        let loaded = InvertedIndex::load(&path)?;

        assert_eq!(loaded.k, 64);
        assert_eq!(loaded.w, 50);
        assert_eq!(loaded.salt, 0xABCD);
        assert_eq!(loaded.num_minimizers(), inverted.num_minimizers());
        assert_eq!(loaded.num_bucket_entries(), inverted.num_bucket_entries());

        Ok(())
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

        // Query with minimizers that match multiple buckets
        let query = vec![200, 300, 500];
        let hits = inverted.get_bucket_hits(&query);

        // Bucket 1 has 200, 300 -> 2 hits
        assert_eq!(hits.get(&1), Some(&2));
        // Bucket 2 has 200, 300 -> 2 hits
        assert_eq!(hits.get(&2), Some(&2));
        // Bucket 3 has 500 -> 1 hit
        assert_eq!(hits.get(&3), Some(&1));
    }

    #[test]
    fn test_inverted_index_get_bucket_hits_no_matches() {
        let mut index = Index::new(64, 50, 0).unwrap();
        index.buckets.insert(1, vec![100, 200, 300]);
        index.bucket_names.insert(1, "A".into());

        let inverted = InvertedIndex::build_from_index(&index);

        // Query with minimizers that don't match
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

        // Should pass validation
        inverted.validate_against_metadata(&metadata)?;
        Ok(())
    }

    #[test]
    fn test_inverted_index_validate_stale() -> Result<()> {
        // Create original index
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();

        let mut index = Index::new(64, 50, 0x1234).unwrap();
        index.buckets.insert(1, vec![100, 200]);
        index.bucket_names.insert(1, "A".into());
        index.bucket_sources.insert(1, vec!["src".into()]);
        index.save(&path)?;

        // Build inverted index from original
        let inverted = InvertedIndex::build_from_index(&index);

        // Modify the index (add a bucket)
        index.buckets.insert(2, vec![300, 400]);
        index.bucket_names.insert(2, "B".into());
        index.bucket_sources.insert(2, vec!["src2".into()]);
        index.save(&path)?;

        // Load modified metadata
        let metadata = Index::load_metadata(&path)?;

        // Validation should fail because hash changed
        let result = inverted.validate_against_metadata(&metadata);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("stale"));

        Ok(())
    }

    #[test]
    fn test_inverted_index_load_invalid_format() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path();
        std::fs::write(path, b"NOTRYXI").unwrap();

        let result = InvertedIndex::load(path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid inverted index format"));
    }

    #[test]
    fn test_inverted_index_load_wrong_version() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path();

        let mut data = b"RYXI".to_vec();
        data.extend_from_slice(&999u32.to_le_bytes()); // Bad version
        std::fs::write(path, data).unwrap();

        let result = InvertedIndex::load(path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unsupported inverted index version"));
    }

    #[test]
    fn test_classify_batch_inverted_matches_regular() -> Result<()> {
        // Build a simple index
        let mut index = Index::new(64, 10, 0).unwrap();
        let mut ws = MinimizerWorkspace::new();

        let seq_a = vec![b'A'; 80];
        index.add_record(1, "ref_a", &seq_a, &mut ws);
        index.finalize_bucket(1);
        index.bucket_names.insert(1, "BucketA".into());

        let seq_b: Vec<u8> = (0..80).map(|i| if i % 2 == 0 { b'A' } else { b'T' }).collect();
        index.add_record(2, "ref_b", &seq_b, &mut ws);
        index.finalize_bucket(2);
        index.bucket_names.insert(2, "BucketB".into());

        // Build inverted index
        let inverted = InvertedIndex::build_from_index(&index);

        // Query that should match bucket 1
        let query_seq = seq_a.clone();
        let records: Vec<QueryRecord> = vec![(101, &query_seq[..], None)];

        // Run both classification methods
        let results_regular = classify_batch(&index, &records, 0.5);
        let results_inverted = classify_batch_inverted(&inverted, &records, 0.5);

        // Both should return the same results
        assert_eq!(results_regular.len(), results_inverted.len());
        if !results_regular.is_empty() {
            assert_eq!(results_regular[0].bucket_id, results_inverted[0].bucket_id);
            assert!((results_regular[0].score - results_inverted[0].score).abs() < 0.001);
        }

        Ok(())
    }

    #[test]
    fn test_inverted_index_empty() {
        let index = Index::new(64, 50, 0).unwrap();
        let inverted = InvertedIndex::build_from_index(&index);

        assert_eq!(inverted.num_minimizers(), 0);
        assert_eq!(inverted.num_bucket_entries(), 0);

        // Query against empty index
        let hits = inverted.get_bucket_hits(&[100, 200, 300]);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_inverted_index_hybrid_search_correctness() {
        // Test that hybrid search correctly handles sorted queries
        let mut index = Index::new(64, 50, 0).unwrap();
        // Create buckets with a range of minimizers
        index.buckets.insert(1, (0..1000).map(|i| i * 10).collect());
        index.bucket_names.insert(1, "A".into());

        let inverted = InvertedIndex::build_from_index(&index);

        // Query with sorted minimizers spread across the range
        let query: Vec<u64> = vec![50, 500, 5000, 9990];
        let hits = inverted.get_bucket_hits(&query);

        // 50, 500, 5000 are in range (0..10000 step 10), 9990 is also in range
        // 50/10=5 ✓, 500/10=50 ✓, 5000/10=500 ✓, 9990/10=999 ✓
        assert_eq!(hits.get(&1), Some(&4));
    }

    #[test]
    fn test_inverted_index_k16() -> Result<()> {
        // Test inverted index with K=16
        let mut index = Index::new(16, 10, 0)?;

        // Create some test data with minimizers in K=16 range (16 bits = max 65535)
        index.buckets.insert(1, vec![100, 200, 300, 500]);
        index.bucket_names.insert(1, "bucket_a".into());

        index.buckets.insert(2, vec![200, 400, 600]);
        index.bucket_names.insert(2, "bucket_b".into());

        let inverted = InvertedIndex::build_from_index(&index);

        // Verify K is preserved
        assert_eq!(inverted.k, 16);

        // Test save/load roundtrip
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();
        inverted.save(&path)?;
        let loaded = InvertedIndex::load(&path)?;

        assert_eq!(loaded.k, 16);
        assert_eq!(loaded.num_minimizers(), inverted.num_minimizers());

        // Test lookups
        let hits = loaded.get_bucket_hits(&[100, 200, 400]);
        assert_eq!(hits.get(&1), Some(&2)); // 100, 200
        assert_eq!(hits.get(&2), Some(&2)); // 200, 400

        Ok(())
    }

    #[test]
    fn test_inverted_index_k32() -> Result<()> {
        // Test inverted index with K=32
        let mut index = Index::new(32, 20, 0)?;

        // Create some test data with minimizers in K=32 range
        let large_min: u64 = 1 << 30; // Use larger values for K=32
        index.buckets.insert(1, vec![large_min, large_min + 100, large_min + 200]);
        index.bucket_names.insert(1, "bucket_a".into());

        index.buckets.insert(2, vec![large_min + 100, large_min + 300]);
        index.bucket_names.insert(2, "bucket_b".into());

        let inverted = InvertedIndex::build_from_index(&index);

        // Verify K is preserved
        assert_eq!(inverted.k, 32);

        // Test save/load roundtrip
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();
        inverted.save(&path)?;
        let loaded = InvertedIndex::load(&path)?;

        assert_eq!(loaded.k, 32);
        assert_eq!(loaded.num_minimizers(), inverted.num_minimizers());

        // Test lookups
        let hits = loaded.get_bucket_hits(&[large_min, large_min + 100, large_min + 300]);
        assert_eq!(hits.get(&1), Some(&2)); // large_min, large_min+100
        assert_eq!(hits.get(&2), Some(&2)); // large_min+100, large_min+300

        Ok(())
    }

    #[test]
    fn test_inverted_index_csr_validation() {
        // Test that load() rejects invalid CSR data
        use std::io::Write;

        // Create a file with valid v3 header but invalid CSR offsets in compressed payload
        let file = NamedTempFile::new().unwrap();
        let path = file.path();

        {
            let mut writer = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
            writer.write_all(b"RYXI").unwrap(); // Magic
            writer.write_all(&3u32.to_le_bytes()).unwrap(); // Version 3
            writer.write_all(&64u64.to_le_bytes()).unwrap(); // K
            writer.write_all(&50u64.to_le_bytes()).unwrap(); // W
            writer.write_all(&0u64.to_le_bytes()).unwrap(); // Salt
            writer.write_all(&0u64.to_le_bytes()).unwrap(); // Source hash
            writer.write_all(&0u32.to_le_bytes()).unwrap(); // max_bucket_id (u8 mode)
            writer.write_all(&2u64.to_le_bytes()).unwrap(); // num_minimizers = 2
            writer.write_all(&3u64.to_le_bytes()).unwrap(); // num_bucket_ids = 3

            // Create streaming encoder for payload
            let mut encoder = zstd::stream::write::Encoder::new(writer, 3).unwrap();

            // Offsets (u32): invalid - first offset != 0
            encoder.write_all(&1u32.to_le_bytes()).unwrap(); // Invalid: first offset != 0
            encoder.write_all(&2u32.to_le_bytes()).unwrap();
            encoder.write_all(&3u32.to_le_bytes()).unwrap();
            // Minimizers: v3 format uses delta+varint encoding
            // First minimizer as u64, second as varint delta
            encoder.write_all(&100u64.to_le_bytes()).unwrap(); // First minimizer
            let delta: u64 = 100; // delta = 200 - 100
            let mut varint_buf = [0u8; 10];
            let len = encode_varint(delta, &mut varint_buf);
            encoder.write_all(&varint_buf[..len]).unwrap(); // Second minimizer as delta
            // Bucket IDs (u8 since max_bucket_id = 0)
            encoder.write_all(&[0u8]).unwrap();
            encoder.write_all(&[0u8]).unwrap();
            encoder.write_all(&[0u8]).unwrap();

            encoder.finish().unwrap();
        }

        let result = InvertedIndex::load(path);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("first offset must be 0"), "Unexpected error: {}", err);
    }
}

