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

// --- CONSTANTS ---
const K: usize = 64; 

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
        #[arg(short, long)]
        output: PathBuf,
        #[arg(short, long, required = true)]
        reference: Vec<PathBuf>,
        #[arg(short, long, default_value_t = 50)]
        window: usize,
        #[arg(short, long, default_value_t = 0x5555555555555555)]
        salt: u64,
    },

    /// Classify reads individually against the index
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

    /// Batch Classify: Aggregates minimizers per batch, reports per READ
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

    /// Aggregate Classify: Treats ALL reads as ONE logical query, reports per BUCKET
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
        batch_size: usize, // Used internally to control memory usage while reading
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

// --- MINIMIZER LOGIC ---

#[inline(always)]
fn base_to_bit(byte: u8) -> u64 {
    match byte {
        b'A' | b'a' | b'G' | b'g' => 1,
        _ => 0, 
    }
}

struct MinQueue {
    deque: VecDeque<(usize, u64)>,
    last_min: Option<u64>,
    window_size: usize,
}

impl MinQueue {
    fn new(window_size: usize) -> Self {
        Self {
            deque: VecDeque::with_capacity(window_size),
            last_min: None,
            window_size,
        }
    }

    #[inline(always)]
    fn push(&mut self, idx: usize, val: u64, out: &mut Vec<u64>) {
        while let Some(&(pos, _)) = self.deque.front() {
            if pos + self.window_size <= idx { self.deque.pop_front(); } else { break; }
        }
        while let Some(&(_, v)) = self.deque.back() {
            if v >= val { self.deque.pop_back(); } else { break; }
        }
        self.deque.push_back((idx, val));
        
        if idx >= self.window_size - 1 {
            if let Some(&(_, min_h)) = self.deque.front() {
                if Some(min_h) != self.last_min {
                    out.push(min_h);
                    self.last_min = Some(min_h);
                }
            }
        }
    }
}

fn extract_minimizers(seq: &[u8], w: usize, salt: u64) -> Vec<u64> {
    let len = seq.len();
    if len < K { return vec![]; }
    
    let num_kmers = len - K + 1;
    let effective_w = if num_kmers < w { num_kmers } else { w };

    let mut mins = Vec::with_capacity(num_kmers / effective_w + 2);
    let mut queue = MinQueue::new(effective_w);
    let mut current_val: u64 = 0;

    for i in 0..(K - 1) { 
        current_val = (current_val << 1) | base_to_bit(seq[i]); 
    }

    for i in 0..num_kmers {
        let next_bit = base_to_bit(seq[i + K - 1]);
        current_val = (current_val << 1) | next_bit;
        queue.push(i, current_val ^ salt, &mut mins);
    }
    mins
}

fn extract_dual_strand(seq: &[u8], w: usize, salt: u64) -> (Vec<u64>, Vec<u64>) {
    let len = seq.len();
    if len < K { return (vec![], vec![]); }
    
    let num_kmers = len - K + 1;
    let effective_w = if num_kmers < w { num_kmers } else { w };

    let mut fwd_mins = Vec::with_capacity(num_kmers / effective_w + 2);
    let mut rc_mins = Vec::with_capacity(num_kmers / effective_w + 2);
    let mut fwd_queue = MinQueue::new(effective_w);
    let mut rc_queue = MinQueue::new(effective_w);
    
    let mut current_val: u64 = 0;

    for i in 0..(K - 1) { 
        current_val = (current_val << 1) | base_to_bit(seq[i]); 
    }

    for i in 0..num_kmers {
        let next_bit = base_to_bit(seq[i + K - 1]);
        current_val = (current_val << 1) | next_bit;
        
        fwd_queue.push(i, current_val ^ salt, &mut fwd_mins);
        rc_queue.push(i, (!current_val).reverse_bits() ^ salt, &mut rc_mins);
    }
    (fwd_mins, rc_mins)
}

// --- ENGINE ---

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

        while let Some(record) = reader.next() {
            let seqrec = record.context("Invalid FASTA record")?;
            for m in extract_minimizers(&seqrec.seq(), self.w, self.salt) {
                bucket.insert(m);
            }
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
            let _map_len = u64::from_le_bytes(buf8); // skip length check for now
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

    /// Reads up to `batch_size` items. Returns `(ids, seqs1, seqs2_opt)`.
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
                println!("Indexing {:?}", ref_path);
                index.add_reference_file(ref_path, (i + 1) as u32)?;
            }
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
        let results: Vec<String> = ids.par_iter().enumerate().map(|(i, id)| {
            let s1 = &s1s[i];
            let s2 = s2s.as_ref().map(|v| &v[i]);
            
            let (hyp_a, hyp_b) = get_paired_minimizers(s1, s2, engine.w, engine.salt);
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
        let processed: Vec<(f64, f64, Vec<u64>, Vec<u64>)> = s1s.par_iter().enumerate().map(|(i, s1)| {
            let s2 = s2s.as_ref().map(|v| &v[i]);
            let (hyp_a, hyp_b) = get_paired_minimizers(s1, s2, engine.w, engine.salt);
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
        let batch_mins: Vec<Vec<u64>> = s1s.par_iter().enumerate().map(|(i, s1)| {
            let s2 = s2s.as_ref().map(|v| &v[i]);
            let (mut ma, mb) = get_paired_minimizers(s1, s2, engine.w, engine.salt);
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

fn get_paired_minimizers(s1: &[u8], s2: Option<&Vec<u8>>, w: usize, salt: u64) -> (Vec<u64>, Vec<u64>) {
    let (mut r1_f, mut r1_rc) = extract_dual_strand(s1, w, salt);
    if let Some(seq2) = s2 {
        let (mut r2_f, mut r2_rc) = extract_dual_strand(seq2, w, salt);
        
        // Hyp A: R1 Fwd + R2 RC
        r1_f.append(&mut r2_rc);
        // Hyp B: R1 RC + R2 Fwd
        r1_rc.append(&mut r2_f);
    }

    r1_f.sort_unstable(); r1_f.dedup();
    r1_rc.sort_unstable(); r1_rc.dedup();
    (r1_f, r1_rc)
}

fn count_hits(mins: &[u64], bucket: &RoaringTreemap) -> f64 {
    mins.iter().filter(|m| bucket.contains(**m)).count() as f64
}

// --- TESTS ---

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimizer_extraction() {
        let seq = vec![b'A'; 70];
        let (fwd, rc) = extract_dual_strand(&seq, 50, 0);
        assert!(!fwd.is_empty());
        assert!(!rc.is_empty());
    }

    #[test]
    fn test_pairing_logic() {
        let s1 = vec![b'A'; 70];
        let s2 = vec![b'T'; 70];
        // R2 (T..) is RC of R1 (A..). 
        // Hyp A (R1_F + R2_RC) should be uniform (all 1s or all 0s depending on hash).
        let (ha, hb) = get_paired_minimizers(&s1, Some(&s2), 10, 0);
        assert_eq!(ha.len(), hb.len()); // Symmetric
    }
}

