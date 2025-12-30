use anyhow::{Result, anyhow};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

// --- CONSTANTS ---

pub const K: usize = 64;
const _: () = assert!(K == 64, "Implementation relies on u64 overflow behavior");

const BASE_TO_BIT_LUT: [u64; 256] = {
    let mut lut = [u64::MAX; 256];
    lut[b'A' as usize] = 1; lut[b'a' as usize] = 1;
    lut[b'G' as usize] = 1; lut[b'g' as usize] = 1;
    lut[b'T' as usize] = 0; lut[b't' as usize] = 0;
    lut[b'C' as usize] = 0; lut[b'c' as usize] = 0;
    lut
};

// --- DATA TYPES ---

/// ID (i64), Sequence, Optional Pair Sequence
pub type QueryRecord = (i64, Vec<u8>, Option<Vec<u8>>);

/// Query ID, Bucket ID, Score
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
            q_fwd: VecDeque::with_capacity(128),
            q_rc: VecDeque::with_capacity(128),
            buffer: Vec::with_capacity(128),
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

    let mut fwd_mins = Vec::with_capacity(32);
    let mut rc_mins = Vec::with_capacity(32);

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
        if !self.buckets.contains_key(&src_id) {
            return Err(anyhow!("Source bucket {} does not exist", src_id));
        }

        let src_vec = self.buckets.remove(&src_id).unwrap();
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

    pub fn next_id(&self) -> u32 {
        self.buckets.keys().max().copied().unwrap_or(0) + 1
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);
        writer.write_all(b"RYP3")?;
        writer.write_all(&2u32.to_le_bytes())?;
        writer.write_all(&(K as u64).to_le_bytes())?;
        writer.write_all(&(self.w as u64).to_le_bytes())?;
        writer.write_all(&(self.salt).to_le_bytes())?;

        let mut sorted_ids: Vec<_> = self.buckets.keys().collect();
        sorted_ids.sort();

        writer.write_all(&(sorted_ids.len() as u32).to_le_bytes())?;

        for id in sorted_ids {
            let vec = &self.buckets[id];
            writer.write_all(&id.to_le_bytes())?;

            let name_str = self.bucket_names.get(id).map(|s| s.as_str()).unwrap_or("unknown");
            let name_bytes = name_str.as_bytes();
            writer.write_all(&(name_bytes.len() as u64).to_le_bytes())?;
            writer.write_all(name_bytes)?;

            let sources = self.bucket_sources.get(id);
            let empty = Vec::new();
            let src_vec = sources.unwrap_or(&empty);

            writer.write_all(&(src_vec.len() as u64).to_le_bytes())?;
            for src in src_vec {
                let s_bytes = src.as_bytes();
                writer.write_all(&(s_bytes.len() as u64).to_le_bytes())?;
                writer.write_all(s_bytes)?;
            }

            writer.write_all(&(vec.len() as u64).to_le_bytes())?;
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
        reader.read_exact(&mut buf8)?; if u64::from_le_bytes(buf8) as usize != K { return Err(anyhow!("K mismatch")); }
        reader.read_exact(&mut buf8)?; let w = u64::from_le_bytes(buf8) as usize;
        reader.read_exact(&mut buf8)?; let salt = u64::from_le_bytes(buf8);
        reader.read_exact(&mut buf4)?; let num = u32::from_le_bytes(buf4);

        let mut buckets = HashMap::new();
        let mut names = HashMap::new();
        let mut sources = HashMap::new();

        for _ in 0..num {
            reader.read_exact(&mut buf4)?; let id = u32::from_le_bytes(buf4);

            reader.read_exact(&mut buf8)?; let name_len = u64::from_le_bytes(buf8) as usize;
            let mut nbuf = vec![0u8; name_len];
            reader.read_exact(&mut nbuf)?;
            names.insert(id, String::from_utf8(nbuf)?);

            reader.read_exact(&mut buf8)?; let src_count = u64::from_le_bytes(buf8) as usize;
            let mut src_list = Vec::with_capacity(src_count);
            for _ in 0..src_count {
                reader.read_exact(&mut buf8)?; let slen = u64::from_le_bytes(buf8) as usize;
                let mut sbuf = vec![0u8; slen];
                reader.read_exact(&mut sbuf)?;
                src_list.push(String::from_utf8(sbuf)?);
            }
            sources.insert(id, src_list);

            reader.read_exact(&mut buf8)?;
            let vec_len = u64::from_le_bytes(buf8) as usize;

            let mut vec = vec![0u64; vec_len];
            let byte_slice = unsafe {
                std::slice::from_raw_parts_mut(vec.as_mut_ptr() as *mut u8, vec_len * 8)
            };
            reader.read_exact(byte_slice)?;

            if cfg!(target_endian = "big") {
                for x in &mut vec { *x = u64::from_le(*x); }
            }

            buckets.insert(id, vec);
        }
        Ok(Index { w, salt, buckets, bucket_names: names, bucket_sources: sources })
    }
}

// --- LIBRARY LEVEL PROCESSING ---

/// Batch Classify: Processes a vector of query records and returns individual hit scores.
/// Threading: Uses `rayon::par_iter`. Caller controls threads via global pool config.
pub fn classify_batch(
    engine: &Index, 
    records: &[QueryRecord], 
    threshold: f64
) -> Vec<HitResult> {
    
    // Step 1: Extract Minimizers in Parallel
    let processed: Vec<_> = records.par_iter()
        .map_init(MinimizerWorkspace::new, |ws, (id, s1, s2)| {
            let (ha, hb) = get_paired_minimizers_into(s1, s2.as_deref(), engine.w, engine.salt, ws);
            (*id, ha, hb)
        }).collect();

    // Step 2: Build Global Map of Minimizer -> Query Index
    // This allows us to scan the Buckets (which are large) only once per unique minimizer.
    let mut map_a: HashMap<u64, Vec<usize>> = HashMap::new();
    let mut map_b: HashMap<u64, Vec<usize>> = HashMap::new();
    let mut uniq_mins = HashSet::new();

    for (idx, (_, ma, mb)) in processed.iter().enumerate() {
        for &m in ma { map_a.entry(m).or_default().push(idx); uniq_mins.insert(m); }
        for &m in mb { map_b.entry(m).or_default().push(idx); uniq_mins.insert(m); }
    }
    let uniq_vec: Vec<u64> = uniq_mins.into_iter().collect();

    // Step 3: Scan Buckets in Parallel
    let results: Vec<HitResult> = engine.buckets.par_iter().map(|(b_id, bucket)| {
        let mut hits = HashMap::new(); // Query Index -> (HitsA, HitsB)

        // Iterate unique minimizers in this batch
        for &m in &uniq_vec {
            if bucket.binary_search(&m).is_ok() {
                if let Some(rs) = map_a.get(&m) { for &r in rs { hits.entry(r).or_insert((0,0)).0 += 1; } }
                if let Some(rs) = map_b.get(&m) { for &r in rs { hits.entry(r).or_insert((0,0)).1 += 1; } }
            }
        }

        // Calculate Scores
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

/// Aggregate Classify: Treats the entire batch as one "Meta-Read" and scores it against buckets.
pub fn aggregate_batch(
    engine: &Index,
    records: &[QueryRecord],
    threshold: f64
) -> Vec<HitResult> {
    let mut global = HashSet::new();
    
    let batch_mins: Vec<Vec<u64>> = records.par_iter()
        .map_init(MinimizerWorkspace::new, |ws, (_, s1, s2)| {
            let (mut a, b) = get_paired_minimizers_into(s1, s2.as_deref(), engine.w, engine.salt, ws);
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
}

