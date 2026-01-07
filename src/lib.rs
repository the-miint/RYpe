use anyhow::{Result, anyhow};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

// Expose the C-API module
pub mod c_api;

// Expose the config module for testing and CLI usage
pub mod config;

// --- CONSTANTS ---

pub const K: usize = 64;
const _: () = assert!(K == 64, "Implementation relies on u64 overflow behavior");

// Maximum sizes for safety checks when loading files
const MAX_BUCKET_SIZE: usize = 1_000_000_000; // 1B minimizers (~8GB)
const MAX_STRING_LENGTH: usize = 10_000; // 10KB for names/sources
const MAX_NUM_BUCKETS: u32 = 100_000; // Reasonable upper limit

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

// --- DATA TYPES ---

/// ID (i64), Sequence Reference, Optional Pair Sequence Reference
pub type QueryRecord<'a> = (i64, &'a [u8], Option<&'a [u8]>);

/// Lightweight metadata-only view of an Index (without minimizer data)
#[derive(Debug, Clone)]
pub struct IndexMetadata {
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

pub fn extract_into(seq: &[u8], w: usize, salt: u64, ws: &mut MinimizerWorkspace) {
    ws.buffer.clear();
    ws.q_fwd.clear();

    let len = seq.len();
    if len < K { return; }

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

        if valid_bases_count >= K {
            let pos = i + 1 - K;
            let hash = current_val ^ salt;

            while let Some(&(p, _)) = ws.q_fwd.front() {
                if p + w <= pos { ws.q_fwd.pop_front(); } else { break; }
            }
            while let Some(&(_, v)) = ws.q_fwd.back() {
                if v >= hash { ws.q_fwd.pop_back(); } else { break; }
            }
            ws.q_fwd.push_back((pos, hash));

            if valid_bases_count >= K + w - 1 {
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
    w: usize,
    salt: u64,
    ws: &mut MinimizerWorkspace
) -> (Vec<u64>, Vec<u64>) {
    ws.q_fwd.clear();
    ws.q_rc.clear();

    let len = seq.len();
    if len < K { return (vec![], vec![]); }

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

        if valid_bases_count >= K {
            let pos = i + 1 - K;
            let h_fwd = current_val ^ salt;
            let h_rc = (!current_val).reverse_bits() ^ salt;

            while let Some(&(p, _)) = ws.q_fwd.front() { if p + w <= pos { ws.q_fwd.pop_front(); } else { break; } }
            while let Some(&(_, v)) = ws.q_fwd.back() { if v >= h_fwd { ws.q_fwd.pop_back(); } else { break; } }
            ws.q_fwd.push_back((pos, h_fwd));

            while let Some(&(p, _)) = ws.q_rc.front() { if p + w <= pos { ws.q_rc.pop_front(); } else { break; } }
            while let Some(&(_, v)) = ws.q_rc.back() { if v >= h_rc { ws.q_rc.pop_back(); } else { break; } }
            ws.q_rc.push_back((pos, h_rc));

            if valid_bases_count >= K + w - 1 {
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
    s1: &[u8], s2: Option<&[u8]>, w: usize, salt: u64, ws: &mut MinimizerWorkspace
) -> (Vec<u64>, Vec<u64>) {
    let (mut fwd, mut rc) = extract_dual_strand_into(s1, w, salt, ws);
    if let Some(seq2) = s2 {
        let (mut r2_f, mut r2_rc) = extract_dual_strand_into(seq2, w, salt, ws);
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
    pub w: usize,
    pub salt: u64,
    pub buckets: HashMap<u32, Vec<u64>>,
    pub bucket_names: HashMap<u32, String>,
    pub bucket_sources: HashMap<u32, Vec<String>>,
}

impl Index {
    pub const BUCKET_SOURCE_DELIM: &'static str = "::";

    pub fn new(w: usize, salt: u64) -> Self {
        Index {
            w,
            salt,
            buckets: HashMap::new(),
            bucket_names: HashMap::new(),
            bucket_sources: HashMap::new()
        }
    }

    pub fn add_record(&mut self, id: u32, source_name: &str, sequence: &[u8], ws: &mut MinimizerWorkspace) {
        let sources = self.bucket_sources.entry(id).or_default();
        sources.push(source_name.to_string());

        extract_into(sequence, self.w, self.salt, ws);
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

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);
        writer.write_all(b"RYP3")?;
        writer.write_all(&3u32.to_le_bytes())?; // Version 3
        writer.write_all(&(K as u64).to_le_bytes())?;
        writer.write_all(&(self.w as u64).to_le_bytes())?;
        writer.write_all(&(self.salt).to_le_bytes())?;

        let mut sorted_ids: Vec<_> = self.buckets.keys().collect();
        sorted_ids.sort();

        writer.write_all(&(sorted_ids.len() as u32).to_le_bytes())?;

        // V3 Format: MinimizerCount comes FIRST for each bucket (enables seeking)
        for id in sorted_ids {
            let vec = &self.buckets[id];

            // Write minimizer count first
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

            // Write minimizers
            for val in vec {
                writer.write_all(&val.to_le_bytes())?;
            }
        }
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let mut reader = BufReader::new(File::open(path)?);
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        reader.read_exact(&mut buf4)?;
        if &buf4 != b"RYP3" { return Err(anyhow!("Invalid Index Format (Expected RYP3)")); }

        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        const SUPPORTED_VERSION: u32 = 3;
        if version != SUPPORTED_VERSION {
            return Err(anyhow!("Unsupported index version: {} (expected {})", version, SUPPORTED_VERSION));
        }

        reader.read_exact(&mut buf8)?; if u64::from_le_bytes(buf8) as usize != K { return Err(anyhow!("K mismatch")); }
        reader.read_exact(&mut buf8)?; let w = u64::from_le_bytes(buf8) as usize;
        reader.read_exact(&mut buf8)?; let salt = u64::from_le_bytes(buf8);
        reader.read_exact(&mut buf4)?; let num = u32::from_le_bytes(buf4);
        if num > MAX_NUM_BUCKETS {
            return Err(anyhow!("Number of buckets {} exceeds maximum {}", num, MAX_NUM_BUCKETS));
        }

        let mut buckets = HashMap::new();
        let mut names = HashMap::new();
        let mut sources = HashMap::new();

        // V3 format: MinimizerCount comes first for each bucket
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

            let mut vec = Vec::with_capacity(vec_len);
            let mut buf8_inner = [0u8; 8];
            for _ in 0..vec_len {
                reader.read_exact(&mut buf8_inner)?;
                vec.push(u64::from_le_bytes(buf8_inner));
            }

            buckets.insert(id, vec);
        }

        Ok(Index { w, salt, buckets, bucket_names: names, bucket_sources: sources })
    }

    pub fn load_metadata(path: &Path) -> Result<IndexMetadata> {
        use std::io::Seek;

        let mut reader = BufReader::new(File::open(path)?);
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        reader.read_exact(&mut buf4)?;
        if &buf4 != b"RYP3" { return Err(anyhow!("Invalid Index Format (Expected RYP3)")); }

        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        const SUPPORTED_VERSION: u32 = 3;
        if version != SUPPORTED_VERSION {
            return Err(anyhow!("Unsupported index version: {} (expected {})", version, SUPPORTED_VERSION));
        }

        reader.read_exact(&mut buf8)?; if u64::from_le_bytes(buf8) as usize != K { return Err(anyhow!("K mismatch")); }
        reader.read_exact(&mut buf8)?; let w = u64::from_le_bytes(buf8) as usize;
        reader.read_exact(&mut buf8)?; let salt = u64::from_le_bytes(buf8);
        reader.read_exact(&mut buf4)?; let num = u32::from_le_bytes(buf4);
        if num > MAX_NUM_BUCKETS {
            return Err(anyhow!("Number of buckets {} exceeds maximum {}", num, MAX_NUM_BUCKETS));
        }

        let mut names = HashMap::new();
        let mut sources = HashMap::new();
        let mut minimizer_counts = HashMap::new();

        // V3 format allows seeking past minimizers for fast metadata-only loading
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

            // Store minimizer count without loading minimizers
            minimizer_counts.insert(id, vec_len);

            // Seek past minimizer data (vec_len × 8 bytes)
            let bytes_to_skip = (vec_len as u64) * 8;
            reader.seek(std::io::SeekFrom::Current(bytes_to_skip as i64))?;
        }

        Ok(IndexMetadata {
            w,
            salt,
            bucket_names: names,
            bucket_sources: sources,
            bucket_minimizer_counts: minimizer_counts,
        })
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
            let (ha, hb) = get_paired_minimizers_into(s1, *s2, engine.w, engine.salt, ws);
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

pub fn aggregate_batch(
    engine: &Index,
    records: &[QueryRecord],
    threshold: f64
) -> Vec<HitResult> {
    let mut global = HashSet::new();
    
    let batch_mins: Vec<Vec<u64>> = records.par_iter()
        .map_init(MinimizerWorkspace::new, |ws, (_, s1, s2)| {
            let (mut a, b) = get_paired_minimizers_into(s1, *s2, engine.w, engine.salt, ws);
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
        extract_into(&seq, 5, 0, &mut ws);
        assert!(!ws.buffer.is_empty(), "Should extract minimizers from valid long seq");
    }

    #[test]
    fn test_short_sequences_ignored() {
        let mut ws = MinimizerWorkspace::new();
        let seq = vec![b'A'; 60]; 
        extract_into(&seq, 5, 0, &mut ws);
        assert!(ws.buffer.is_empty(), "Should not extract minimizers from seq < K");
    }

    #[test]
    fn test_n_handling_separator() {
        let mut ws = MinimizerWorkspace::new();
        let seq_a: Vec<u8> = (0..80).map(|i| if i % 2 == 0 { b'A' } else { b'T' }).collect(); 
        let seq_b: Vec<u8> = (0..80).map(|i| if i % 3 == 0 { b'G' } else { b'C' }).collect(); 
        
        extract_into(&seq_a, 5, 0, &mut ws);
        let mins_a = ws.buffer.clone();
        
        extract_into(&seq_b, 5, 0, &mut ws);
        let mins_b = ws.buffer.clone();

        let mut seq_combined = seq_a.clone();
        seq_combined.push(b'N'); // N separator
        seq_combined.extend_from_slice(&seq_b);
        
        extract_into(&seq_combined, 5, 0, &mut ws);
        let mins_combined = ws.buffer.clone();

        let mut expected = mins_a;
        expected.extend(mins_b);
        
        assert_eq!(mins_combined, expected, "N should act as a perfect separator");
    }

    #[test]
    fn test_dual_strand_extraction() {
        let mut ws = MinimizerWorkspace::new();
        let seq = b"AAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCC"; // Long seq
        let (fwd, rc) = extract_dual_strand_into(seq, 5, 0, &mut ws);
        assert!(!fwd.is_empty());
        assert!(!rc.is_empty());
        assert_ne!(fwd, rc); 
    }

    // --- INDEX OPERATIONS & FILE IO TESTS ---

    #[test]
    fn test_index_save_load_roundtrip() -> Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();
        let mut index = Index::new(50, 0x1234);
        
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
        let mut index = Index::new(5, 0);
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
        let mut index = Index::new(50, 0);
        
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
        let mut index = Index::new(10, 0);
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
        let mut index = Index::new(10, 0);
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
        std::fs::write(path, b"NOTRYP3").unwrap();

        let result = Index::load(path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid Index Format"));
    }

    #[test]
    fn test_index_load_unsupported_version() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path();

        // Write valid header with wrong version
        let mut data = b"RYP3".to_vec();
        data.extend_from_slice(&999u32.to_le_bytes());  // Bad version
        std::fs::write(path, data).unwrap();

        let result = Index::load(path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unsupported index version"));
    }

    #[test]
    fn test_index_load_oversized_bucket() {
        // Create a malicious index file claiming a huge bucket (V3 format)
        let file = NamedTempFile::new().unwrap();
        let path = file.path();

        let mut data = Vec::new();
        data.extend_from_slice(b"RYP3");
        data.extend_from_slice(&3u32.to_le_bytes());  // Version 3
        data.extend_from_slice(&64u64.to_le_bytes()); // K
        data.extend_from_slice(&50u64.to_le_bytes()); // W
        data.extend_from_slice(&0u64.to_le_bytes());  // Salt
        data.extend_from_slice(&1u32.to_le_bytes());  // 1 bucket
        // V3 Format: MinimizerCount comes FIRST
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
        let mut index = Index::new(50, 0);
        index.buckets.insert(u32::MAX, vec![]);

        let result = index.next_id();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("overflow"));
    }

    #[test]
    fn test_merge_buckets_nonexistent_source() {
        let mut index = Index::new(50, 0);
        index.buckets.insert(1, vec![1, 2, 3]);

        let result = index.merge_buckets(999, 1);  // 999 doesn't exist
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not exist"));
    }

    #[test]
    fn test_merge_buckets_updates_minimizer_count() -> Result<()> {
        // Verify that merge_buckets correctly updates the minimizer count
        let mut index = Index::new(50, 0);

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
        let mut index = Index::new(10, 0);
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
        let mut index = Index::new(50, 0);
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
        let mut index = Index::new(50, 0x1234);

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
        let mut index = Index::new(42, 0xABCD);

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
        let index = Index::new(25, 0);

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
    fn test_v3_format_roundtrip() -> Result<()> {
        // Verify V3 format can save and load correctly
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();
        let mut index = Index::new(50, 0x1234);

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
}

