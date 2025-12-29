use clap::{Parser, Subcommand};
use needletail::{parse_fastx_file, FastxReader};
use rayon::prelude::*;
use roaring::RoaringTreemap;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use anyhow::{Context, Result, anyhow};

// --- CONSTANTS & LOOKUP TABLES ---

const K: usize = 64; 

/// Branchless lookup table for Base -> Bit conversion (RY Space).
/// A/G (Purines) -> 1
/// T/C/N (Pyrimidines) -> 0
const BASE_TO_BIT_LUT: [u64; 256] = {
    let mut lut = [0u64; 256];
    lut[b'A' as usize] = 1;
    lut[b'a' as usize] = 1;
    lut[b'G' as usize] = 1;
    lut[b'g' as usize] = 1;
    // All others are 0
    lut
};

// --- CLI CONFIGURATION ---

#[derive(Parser)]
#[command(name = "rype")]
#[command(about = "High-performance Read Partitioning Engine (RY-Space, K=64)", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build a new index from reference FASTA files
    Index {
        /// Output path for the .ryidx index file
        #[arg(short, long)]
        output: PathBuf,

        /// List of reference FASTA files
        #[arg(short, long, required = true)]
        reference: Vec<PathBuf>,

        /// Minimizer window size
        #[arg(short, long, default_value_t = 50)]
        window: usize,

        /// Random seed for hashing (Salt)
        #[arg(short, long, default_value_t = 0x5555555555555555)]
        salt: u64,
    },

    /// Standard Classification: Processes reads individually against the index
    Classify {
        #[arg(short, long)]
        index: PathBuf,
        #[arg(short = '1', long)]
        r1: PathBuf,
        #[arg(short = '2', long)]
        r2: Option<PathBuf>,
        #[arg(short, long, default_value_t = 0.1)]
        threshold: f64,
        #[arg(short, long, default_value_t = 10_000)]
        batch_size: usize,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Batch Classification: Aggregates minimizers per batch, queries efficiently, reports per READ
    #[command(alias = "batch-classify")]
    BatchClassify {
        #[arg(short, long)]
        index: PathBuf,
        #[arg(short = '1', long)]
        r1: PathBuf,
        #[arg(short = '2', long)]
        r2: Option<PathBuf>,
        #[arg(short, long, default_value_t = 0.1)]
        threshold: f64,
        #[arg(short, long, default_value_t = 50_000)]
        batch_size: usize,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Aggregate Classification: Treats ALL reads as ONE logical query, reports per BUCKET
    #[command(alias = "aggregate")]
    AggregateClassify {
        #[arg(short, long)]
        index: PathBuf,
        #[arg(short = '1', long)]
        r1: PathBuf,
        #[arg(short = '2', long)]
        r2: Option<PathBuf>,
        #[arg(short, long, default_value_t = 0.05)]
        threshold: f64,
        #[arg(short, long, default_value_t = 50_000)]
        batch_size: usize,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

// --- MINIMIZER LOGIC (Optimized) ---

#[inline(always)]
fn base_to_bit(byte: u8) -> u64 {
    // SAFETY: byte is u8 (0..255), array is [u64; 256]. Bounds check unnecessary.
    unsafe { *BASE_TO_BIT_LUT.get_unchecked(byte as usize) }
}

/// Reusable workspace to avoid heap allocations in hot loops.
struct MinimizerWorkspace {
    q_fwd: VecDeque<(usize, u64)>, // For Forward strand (or single strand)
    q_rc: VecDeque<(usize, u64)>,  // For Reverse Complement strand
    buffer: Vec<u64>,              // General purpose output buffer
}

impl MinimizerWorkspace {
    fn new() -> Self {
        Self {
            q_fwd: VecDeque::with_capacity(128),
            q_rc: VecDeque::with_capacity(128),
            buffer: Vec::with_capacity(128),
        }
    }
}

/// Extracts minimizers into the workspace buffer (Single Strand).
fn extract_into(seq: &[u8], w: usize, salt: u64, ws: &mut MinimizerWorkspace) {
    ws.buffer.clear();
    ws.q_fwd.clear();
    
    let len = seq.len();
    if len < K { return; }
    
    let num_kmers = len - K + 1;
    let effective_w = if num_kmers < w { num_kmers } else { w };
    let mut current_val: u64 = 0;
    let mut last_min: Option<u64> = None;

    // Pre-load first K-1 bases
    for i in 0..(K - 1) { 
        current_val = (current_val << 1) | base_to_bit(seq[i]); 
    }

    // Rolling hash + Monotonic Queue
    for i in 0..num_kmers {
        let next_bit = base_to_bit(seq[i + K - 1]);
        current_val = (current_val << 1) | next_bit;
        let hash = current_val ^ salt;

        // Maintain queue: remove old elements
        while let Some(&(pos, _)) = ws.q_fwd.front() {
            if pos + effective_w <= i { ws.q_fwd.pop_front(); } else { break; }
        }
        // Maintain queue: remove larger elements (shadowed)
        while let Some(&(_, v)) = ws.q_fwd.back() {
            if v >= hash { ws.q_fwd.pop_back(); } else { break; }
        }
        ws.q_fwd.push_back((i, hash));

        // Report minimizer
        if i >= effective_w - 1 {
            if let Some(&(_, min_h)) = ws.q_fwd.front() {
                if Some(min_h) != last_min {
                    ws.buffer.push(min_h);
                    last_min = Some(min_h);
                }
            }
        }
    }
}

/// Helper to get paired minimizers using a reused workspace.
/// Returns (Hypothesis_A_Minimizers, Hypothesis_B_Minimizers)
/// Hyp A = R1_Fwd + R2_RC
/// Hyp B = R1_RC + R2_Fwd
fn get_paired_minimizers_into(
    s1: &[u8], 
    s2: Option<&Vec<u8>>, 
    w: usize, 
    salt: u64, 
    ws: &mut MinimizerWorkspace
) -> (Vec<u64>, Vec<u64>) {
    
    // 1. Extract R1 Dual Strands
    let (mut fwd, mut rc) = extract_dual_strand_into(s1, w, salt, ws);

    if let Some(seq2) = s2 {
        let (mut r2_f, mut r2_rc) = extract_dual_strand_into(seq2, w, salt, ws);
        
        // Hyp A: R1 Fwd + R2 RC
        fwd.append(&mut r2_rc);
        // Hyp B: R1 RC + R2 Fwd
        rc.append(&mut r2_f);
    }

    fwd.sort_unstable(); fwd.dedup();
    rc.sort_unstable(); rc.dedup();
    
    (fwd, rc)
}

/// Optimized dual-strand extraction. 
/// Uses `ws.q_fwd` and `ws.q_rc` to avoid allocating queues.
fn extract_dual_strand_into(seq: &[u8], w: usize, salt: u64, ws: &mut MinimizerWorkspace) -> (Vec<u64>, Vec<u64>) {
    ws.q_fwd.clear();
    ws.q_rc.clear();

    let len = seq.len();
    if len < K { return (vec![], vec![]); }
    
    let num_kmers = len - K + 1;
    let effective_w = if num_kmers < w { num_kmers } else { w };

    // These results must be allocated to be returned, but the intermediate queues are reused
    let mut fwd_mins = Vec::with_capacity(32);
    let mut rc_mins = Vec::with_capacity(32);

    let mut current_val: u64 = 0;
    
    // Pre-load
    for i in 0..(K - 1) { 
        current_val = (current_val << 1) | base_to_bit(seq[i]); 
    }

    for i in 0..num_kmers {
        let next_bit = base_to_bit(seq[i + K - 1]);
        current_val = (current_val << 1) | next_bit;
        
        let h_fwd = current_val ^ salt;
        let h_rc = (!current_val).reverse_bits() ^ salt;

        // FWD Queue
        while let Some(&(pos, _)) = ws.q_fwd.front() { if pos + effective_w <= i { ws.q_fwd.pop_front(); } else { break; } }
        while let Some(&(_, v)) = ws.q_fwd.back() { if v >= h_fwd { ws.q_fwd.pop_back(); } else { break; } }
        ws.q_fwd.push_back((i, h_fwd));

        // RC Queue
        while let Some(&(pos, _)) = ws.q_rc.front() { if pos + effective_w <= i { ws.q_rc.pop_front(); } else { break; } }
        while let Some(&(_, v)) = ws.q_rc.back() { if v >= h_rc { ws.q_rc.pop_back(); } else { break; } }
        ws.q_rc.push_back((i, h_rc));

        // Emit
        if i >= effective_w - 1 {
            if let Some(&(_, min)) = ws.q_fwd.front() {
                if fwd_mins.last() != Some(&min) { fwd_mins.push(min); }
            }
            if let Some(&(_, min)) = ws.q_rc.front() {
                if rc_mins.last() != Some(&min) { rc_mins.push(min); }
            }
        }
    }
    (fwd_mins, rc_mins)
}

// --- INDEX STRUCTURE ---

#[derive(Debug)]
struct Index {
    w: usize,
    salt: u64,
    buckets: HashMap<u32, RoaringTreemap>,
    bucket_names: HashMap<u32, String>,
}

impl Index {
    fn new(w: usize, salt: u64) -> Self {
        Index { w, salt, buckets: HashMap::new(), bucket_names: HashMap::new() }
    }

    fn add_reference_file(&mut self, path: &Path, id: u32) -> Result<()> {
        let mut reader = parse_fastx_file(path)
            .with_context(|| format!("Failed to open reference: {:?}", path))?;
        
        self.bucket_names.insert(id, path.file_name().unwrap().to_string_lossy().to_string());
        let bucket = self.buckets.entry(id).or_insert_with(RoaringTreemap::new);
        
        let mut ws = MinimizerWorkspace::new(); // Local workspace for single-threaded build

        while let Some(record) = reader.next() {
            let seqrec = record.context("Invalid FASTA record")?;
            extract_into(&seqrec.seq(), self.w, self.salt, &mut ws);
            for &m in &ws.buffer { bucket.insert(m); }
        }
        Ok(())
    }

    fn save(&self, path: &Path) -> Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);
        writer.write_all(b"RYPE")?; 
        writer.write_all(&2u32.to_le_bytes())?; // Version
        writer.write_all(&(K as u64).to_le_bytes())?;
        writer.write_all(&(self.w as u64).to_le_bytes())?;
        writer.write_all(&(self.salt).to_le_bytes())?;

        let mut sorted_ids: Vec<_> = self.buckets.keys().collect();
        sorted_ids.sort();
        writer.write_all(&(sorted_ids.len() as u32).to_le_bytes())?;

        for id in sorted_ids {
            let map = &self.buckets[id];
            writer.write_all(&id.to_le_bytes())?;
            
            let name = self.bucket_names.get(id).unwrap_or(&"Unknown".to_string()).clone();
            let name_bytes = name.as_bytes();
            writer.write_all(&(name_bytes.len() as u64).to_le_bytes())?;
            writer.write_all(name_bytes)?;

            let mut buf = Vec::new();
            map.serialize_into(&mut buf)?;
            writer.write_all(&(buf.len() as u64).to_le_bytes())?;
            writer.write_all(&buf)?;
        }
        Ok(())
    }

    fn load(path: &Path) -> Result<Self> {
        let mut reader = BufReader::new(File::open(path).with_context(|| format!("Failed to open index: {:?}", path))?);
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        reader.read_exact(&mut buf4)?;
        if &buf4 != b"RYPE" { return Err(anyhow!("Invalid index file format")); }

        reader.read_exact(&mut buf4)?; // version
        reader.read_exact(&mut buf8)?; // K
        if u64::from_le_bytes(buf8) as usize != K { return Err(anyhow!("K-mer size mismatch")); }
        
        reader.read_exact(&mut buf8)?;
        let w = u64::from_le_bytes(buf8) as usize;
        reader.read_exact(&mut buf8)?;
        let salt = u64::from_le_bytes(buf8);

        reader.read_exact(&mut buf4)?;
        let num_buckets = u32::from_le_bytes(buf4);

        let mut buckets = HashMap::new();
        let mut bucket_names = HashMap::new();

        for _ in 0..num_buckets {
            reader.read_exact(&mut buf4)?;
            let id = u32::from_le_bytes(buf4);

            reader.read_exact(&mut buf8)?;
            let mut name_buf = vec![0u8; u64::from_le_bytes(buf8) as usize];
            reader.read_exact(&mut name_buf)?;
            bucket_names.insert(id, String::from_utf8(name_buf)?);

            reader.read_exact(&mut buf8)?;
            let _map_len = u64::from_le_bytes(buf8);
            buckets.insert(id, RoaringTreemap::deserialize_from(&mut reader)?);
        }
        Ok(Index { w, salt, buckets, bucket_names })
    }
}

// --- IO ABSTRACTION ---

struct IoHandler {
    r1: Box<dyn FastxReader>,
    r2: Option<Box<dyn FastxReader>>,
    writer: Arc<Mutex<Box<dyn Write + Send>>>,
}

impl IoHandler {
    fn new(r1_path: &Path, r2_path: Option<&PathBuf>, out_path: Option<PathBuf>) -> Result<Self> {
        let r1 = parse_fastx_file(r1_path).context("Could not open R1")?;
        let r2 = if let Some(p) = r2_path {
            Some(parse_fastx_file(p).context("Could not open R2")?)
        } else {
            None
        };

        let writer: Box<dyn Write + Send> = match out_path {
            Some(p) => Box::new(BufWriter::new(File::create(p)?)),
            None => Box::new(BufWriter::new(io::stdout())),
        };

        Ok(Self { r1, r2, writer: Arc::new(Mutex::new(writer)) })
    }

    fn write_header(&self, cols: &[&str]) -> Result<()> {
        let mut w = self.writer.lock().unwrap();
        writeln!(w, "{}", cols.join("\t"))?;
        Ok(())
    }

    fn next_batch(&mut self, batch_size: usize) -> Result<Option<(Vec<String>, Vec<Vec<u8>>, Option<Vec<Vec<u8>>>)>> {
        let mut ids = Vec::with_capacity(batch_size);
        let mut s1s = Vec::with_capacity(batch_size);
        let mut s2s = if self.r2.is_some() { Some(Vec::with_capacity(batch_size)) } else { None };

        for _ in 0..batch_size {
            match self.r1.next() {
                Some(Ok(rec)) => {
                    ids.push(String::from_utf8_lossy(rec.id()).to_string());
                    s1s.push(rec.seq().into_owned());
                }
                Some(Err(e)) => return Err(anyhow!(e)),
                None => break,
            }

            if let Some(reader2) = &mut self.r2 {
                match reader2.next() {
                    Some(Ok(rec)) => s2s.as_mut().unwrap().push(rec.seq().into_owned()),
                    Some(Err(e)) => return Err(anyhow!(e)),
                    None => return Err(anyhow!("R1/R2 length mismatch")),
                }
            }
        }

        if ids.is_empty() { return Ok(None); }
        Ok(Some((ids, s1s, s2s)))
    }
}

// --- MAIN EXECUTION ---

fn main() -> Result<()> {
    let args = Cli::parse();

    match args.command {
        Commands::Index { output, reference, window, salt } => {
            let mut index = Index::new(window, salt);
            for (i, ref_path) in reference.iter().enumerate() {
                println!("[*] Indexing {:?} as bucket {}", ref_path, i+1);
                index.add_reference_file(ref_path, (i + 1) as u32)?;
            }
            println!("[*] Saving index to {:?}", output);
            index.save(&output)?;
        }

        Commands::Classify { index, r1, r2, threshold, batch_size, output } => {
            let engine = Index::load(&index)?;
            let mut io = IoHandler::new(&r1, r2.as_ref(), output)?;
            io.write_header(&["read_id", "bucket_id", "score"])?;
            run_classify(&engine, &mut io, threshold, batch_size)?;
        }

        Commands::BatchClassify { index, r1, r2, threshold, batch_size, output } => {
            let engine = Index::load(&index)?;
            let mut io = IoHandler::new(&r1, r2.as_ref(), output)?;
            io.write_header(&["read_id", "bucket_id", "score"])?;
            run_batch_classify(&engine, &mut io, threshold, batch_size)?;
        }

        Commands::AggregateClassify { index, r1, r2, threshold, batch_size, output } => {
            let engine = Index::load(&index)?;
            let mut io = IoHandler::new(&r1, r2.as_ref(), output)?;
            io.write_header(&["query_name", "bucket_id", "score"])?;
            run_aggregate_classify(&engine, &mut io, threshold, batch_size)?;
        }
    }
    Ok(())
}

// --- LOGIC IMPLEMENTATIONS ---

fn run_classify(engine: &Index, io: &mut IoHandler, threshold: f64, batch_size: usize) -> Result<()> {
    while let Some((ids, s1s, s2s)) = io.next_batch(batch_size)? {
        let results: Vec<String> = ids.par_iter().enumerate()
            // Initialize thread-local workspace
            .map_init(MinimizerWorkspace::new, |ws, (i, id)| {
                let s1 = &s1s[i];
                let s2 = s2s.as_ref().map(|v| &v[i]);
                
                let (hyp_a, hyp_b) = get_paired_minimizers_into(s1, s2, engine.w, engine.salt, ws);
                let len_a = hyp_a.len() as f64;
                let len_b = hyp_b.len() as f64;

                if len_a == 0.0 && len_b == 0.0 { return String::new(); }

                let mut out = String::with_capacity(128);
                for (b_id, bucket) in &engine.buckets {
                    let score_a = if len_a > 0.0 { count_hits(&hyp_a, bucket) / len_a } else { 0.0 };
                    let score_b = if len_b > 0.0 { count_hits(&hyp_b, bucket) / len_b } else { 0.0 };
                    let score = score_a.max(score_b);

                    if score >= threshold {
                        out.push_str(&format!("{}\t{}\t{:.4}\n", id, b_id, score));
                    }
                }
                out
            }).collect();

        let mut w = io.writer.lock().unwrap();
        for s in results { if !s.is_empty() { w.write_all(s.as_bytes())?; } }
    }
    Ok(())
}

fn run_batch_classify(engine: &Index, io: &mut IoHandler, threshold: f64, batch_size: usize) -> Result<()> {
    while let Some((ids, s1s, s2s)) = io.next_batch(batch_size)? {
        // 1. Process batch to get minimizers
        let processed: Vec<(f64, f64, Vec<u64>, Vec<u64>)> = s1s.par_iter().enumerate()
            .map_init(MinimizerWorkspace::new, |ws, (i, s1)| {
                let s2 = s2s.as_ref().map(|v| &v[i]);
                let (hyp_a, hyp_b) = get_paired_minimizers_into(s1, s2, engine.w, engine.salt, ws);
                (hyp_a.len() as f64, hyp_b.len() as f64, hyp_a, hyp_b)
            }).collect();

        // 2. Build Inverted Index (Minimizer -> Read Indices)
        let mut map_a: HashMap<u64, Vec<usize>> = HashMap::new();
        let mut map_b: HashMap<u64, Vec<usize>> = HashMap::new();
        let mut unique_mins: HashSet<u64> = HashSet::new();

        for (idx, (_, _, ma, mb)) in processed.iter().enumerate() {
            for &m in ma { map_a.entry(m).or_default().push(idx); unique_mins.insert(m); }
            for &m in mb { map_b.entry(m).or_default().push(idx); unique_mins.insert(m); }
        }
        let unique_vec: Vec<u64> = unique_mins.into_iter().collect();

        // 3. Query Buckets
        let bucket_results: Vec<String> = engine.buckets.par_iter().map(|(b_id, bucket)| {
            let mut hits_map: HashMap<usize, (u32, u32)> = HashMap::new();

            // Iterate only unique minimizers present in this batch
            for &m in &unique_vec {
                if bucket.contains(m) {
                    if let Some(reads) = map_a.get(&m) { for &r in reads { hits_map.entry(r).or_default().0 += 1; } }
                    if let Some(reads) = map_b.get(&m) { for &r in reads { hits_map.entry(r).or_default().1 += 1; } }
                }
            }

            let mut buf = String::new();
            for (r_idx, (ha, hb)) in hits_map {
                let (len_a, len_b, _, _) = &processed[r_idx];
                let score = (if *len_a > 0.0 { ha as f64 / len_a } else { 0.0 })
                    .max(if *len_b > 0.0 { hb as f64 / len_b } else { 0.0 });
                
                if score >= threshold {
                    buf.push_str(&format!("{}\t{}\t{:.4}\n", ids[r_idx], b_id, score));
                }
            }
            buf
        }).collect();

        let mut w = io.writer.lock().unwrap();
        for s in bucket_results { if !s.is_empty() { w.write_all(s.as_bytes())?; } }
    }
    Ok(())
}

fn run_aggregate_classify(engine: &Index, io: &mut IoHandler, threshold: f64, batch_size: usize) -> Result<()> {
    let mut global_mins: HashSet<u64> = HashSet::new();
    
    // Accumulate ALL minimizers from ALL reads in the input
    while let Some((_, s1s, s2s)) = io.next_batch(batch_size)? {
        let batch_mins: Vec<Vec<u64>> = s1s.par_iter().enumerate()
            .map_init(MinimizerWorkspace::new, |ws, (i, s1)| {
                let s2 = s2s.as_ref().map(|v| &v[i]);
                let (mut ma, mb) = get_paired_minimizers_into(s1, s2, engine.w, engine.salt, ws);
                ma.extend(mb);
                ma // Just dump them all together
            }).collect();

        for m_vec in batch_mins {
            for m in m_vec { global_mins.insert(m); }
        }
    }

    let total_unique = global_mins.len() as f64;
    if total_unique == 0.0 { return Ok(()); }

    let global_vec: Vec<u64> = global_mins.into_iter().collect();

    // Query buckets against global set
    let results: Vec<String> = engine.buckets.par_iter().map(|(b_id, bucket)| {
        let hits = count_hits(&global_vec, bucket);
        let score = hits / total_unique;
        
        if score >= threshold {
            return format!("global_query\t{}\t{:.4}\n", b_id, score);
        }
        String::new()
    }).collect();

    let mut w = io.writer.lock().unwrap();
    for s in results { if !s.is_empty() { w.write_all(s.as_bytes())?; } }
    
    Ok(())
}

// --- HELPERS ---

fn count_hits(mins: &[u64], bucket: &RoaringTreemap) -> f64 {
    mins.iter().filter(|m| bucket.contains(**m)).count() as f64
}

// --- UNIT TESTS ---

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_lut_accuracy() {
        assert_eq!(base_to_bit(b'A'), 1);
        assert_eq!(base_to_bit(b'G'), 1);
        assert_eq!(base_to_bit(b'a'), 1);
        assert_eq!(base_to_bit(b'g'), 1);
        assert_eq!(base_to_bit(b'T'), 0);
        assert_eq!(base_to_bit(b'C'), 0);
        assert_eq!(base_to_bit(b'N'), 0);
    }

    #[test]
    fn test_extract_into_basic() {
        let mut ws = MinimizerWorkspace::new();
        let w = 5;
        // All As = all 1s. Hash will vary by salt, but sequence is constant
        let seq = vec![b'A'; 70]; 
        
        extract_into(&seq, w, 0, &mut ws);
        // K=64, Seq=70. NumKmers = 7. W=5.
        // Should produce 2 minimizers (0..5, 2..7) roughly
        assert!(!ws.buffer.is_empty());
    }

    #[test]
    fn test_workspace_clears_correctly() {
        let mut ws = MinimizerWorkspace::new();
        extract_into(&vec![b'A'; 70], 5, 0, &mut ws);
        let first_len = ws.buffer.len();
        
        // Run again with short sequence -> should be empty
        extract_into(&vec![b'A'; 10], 5, 0, &mut ws);
        assert!(ws.buffer.is_empty());
        assert_eq!(ws.q_fwd.len(), 0);

        // Run again with valid seq
        extract_into(&vec![b'A'; 70], 5, 0, &mut ws);
        assert_eq!(ws.buffer.len(), first_len);
    }

    #[test]
    fn test_dual_strand_extraction() {
        let mut ws = MinimizerWorkspace::new();
        let seq = vec![b'A'; 70]; // All 1s
        let (fwd, rc) = extract_dual_strand_into(&seq, 10, 0x1234, &mut ws);
        
        assert!(!fwd.is_empty());
        assert!(!rc.is_empty());
    }

    #[test]
    fn test_index_io() -> Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();
        let mut index = Index::new(50, 0x1234);
        
        // Manually insert some data (simulating add_reference)
        let mut map = RoaringTreemap::new();
        map.insert(100); map.insert(200);
        index.buckets.insert(1, map);
        index.bucket_names.insert(1, "TestBucket".to_string());

        index.save(&path)?;
        let loaded = Index::load(&path)?;

        assert_eq!(loaded.w, 50);
        assert_eq!(loaded.salt, 0x1234);
        assert!(loaded.buckets.contains_key(&1));
        assert!(loaded.buckets[&1].contains(100));
        assert_eq!(loaded.bucket_names[&1], "TestBucket");
        Ok(())
    }
}

