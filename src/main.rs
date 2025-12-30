use clap::{Parser, Subcommand};
use needletail::{parse_fastx_file, FastxReader};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Sender};
use std::thread;
use anyhow::{Context, Result, anyhow};

// --- CONSTANTS & SAFETY CHECKS ---

const K: usize = 64; 
const _: () = assert!(K == 64, "This implementation relies on u64 overflow behavior matching K=64");

const BASE_TO_BIT_LUT: [u64; 256] = {
    let mut lut = [u64::MAX; 256];
    lut[b'A' as usize] = 1; lut[b'a' as usize] = 1;
    lut[b'G' as usize] = 1; lut[b'g' as usize] = 1;
    lut[b'T' as usize] = 0; lut[b't' as usize] = 0;
    lut[b'C' as usize] = 0; lut[b'c' as usize] = 0;
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
    /// Create a new index from reference files
    Index {
        #[arg(short, long)]
        output: PathBuf,
        #[arg(short, long, required = true)]
        reference: Vec<PathBuf>,
        #[arg(short, long, default_value_t = 50)]
        window: usize,
        #[arg(short, long, default_value_t = 0x5555555555555555)]
        salt: u64,
        /// If set, each file becomes a separate bucket. If unset (default), all files merge into Bucket 1.
        #[arg(long)]
        separate_buckets: bool,
    },
    /// Display statistics and contents (sequence IDs) of an index
    IndexStats {
        #[arg(short, long)]
        index: PathBuf,
    },
    /// Add a reference file to an existing index (creates a new bucket)
    IndexBucketAdd {
        #[arg(short, long)]
        index: PathBuf,
        #[arg(short, long)]
        reference: PathBuf,
    },
    /// Merge two buckets within an index (Source -> Dest, Source is deleted)
    IndexBucketMerge {
        #[arg(short, long)]
        index: PathBuf,
        #[arg(long)]
        src: u32,
        #[arg(long)]
        dest: u32,
    },
    /// Merge multiple index files into a new index
    IndexMerge {
        #[arg(short, long)]
        output: PathBuf,
        #[arg(short, long, required = true)]
        inputs: Vec<PathBuf>,
    },
    /// Classify reads (Single Output Stream)
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
    /// Classify reads (Batch/Grouped optimized)
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
    /// Calculate aggregate score for a batch of reads against buckets
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

// --- MINIMIZER LOGIC ---

#[inline(always)]
fn base_to_bit(byte: u8) -> u64 {
    unsafe { *BASE_TO_BIT_LUT.get_unchecked(byte as usize) }
}

struct MinimizerWorkspace {
    q_fwd: VecDeque<(usize, u64)>, 
    q_rc: VecDeque<(usize, u64)>,  
    buffer: Vec<u64>,              
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

// Single Strand Extraction
fn extract_into(seq: &[u8], w: usize, salt: u64, ws: &mut MinimizerWorkspace) {
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

// Dual Strand Extraction
fn extract_dual_strand_into(seq: &[u8], w: usize, salt: u64, ws: &mut MinimizerWorkspace) -> (Vec<u64>, Vec<u64>) {
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

fn get_paired_minimizers_into(
    s1: &[u8], s2: Option<&Vec<u8>>, w: usize, salt: u64, ws: &mut MinimizerWorkspace
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

// --- INDEX STRUCTURE (SORTED VEC + Source Metadata) ---

#[derive(Debug)]
struct Index {
    w: usize,
    salt: u64,
    buckets: HashMap<u32, Vec<u64>>, 
    bucket_names: HashMap<u32, String>,
    bucket_sources: HashMap<u32, Vec<String>>, 
}

impl Index {
    fn new(w: usize, salt: u64) -> Self { 
        Index { 
            w, 
            salt, 
            buckets: HashMap::new(), 
            bucket_names: HashMap::new(),
            bucket_sources: HashMap::new()
        } 
    }
    
    fn add_reference_file(&mut self, path: &Path, id: u32) -> Result<()> {
        let mut reader = parse_fastx_file(path).with_context(|| format!("Failed to open reference: {:?}", path))?;
        
        let fname = path.file_name().unwrap().to_string_lossy().to_string();
        self.bucket_names.entry(id).or_insert_with(|| fname.clone());
        
        let mut temp_mins = Vec::new();
        let mut ws = MinimizerWorkspace::new();
        
        // Grab the vector for sources and push unconditionally
        // We will sort and dedup AFTER the loop to avoid O(N^2) scan
        let sources = self.bucket_sources.entry(id).or_default();
        
        while let Some(record) = reader.next() {
            let seqrec = record.context("Invalid FASTA")?;
            let seq_id = String::from_utf8_lossy(seqrec.id()).to_string();
            
            // Just push. We fix duplicates later.
            sources.push(format!("{}::{}", fname, seq_id));
            
            extract_into(&seqrec.seq(), self.w, self.salt, &mut ws);
            temp_mins.extend_from_slice(&ws.buffer);
        }

        // Fix duplicates in sources (O(N log N))
        sources.sort_unstable();
        sources.dedup();

        let bucket = self.buckets.entry(id).or_default();
        bucket.extend(temp_mins);
        
        // Fix duplicates in minimizers (O(N log N))
        bucket.sort_unstable();
        bucket.dedup();
        
        Ok(())
    }

    fn merge_buckets(&mut self, src_id: u32, dest_id: u32) -> Result<()> {
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

    fn next_id(&self) -> u32 {
        self.buckets.keys().max().copied().unwrap_or(0) + 1
    }

    fn save(&self, path: &Path) -> Result<()> {
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
            
            // 1. Write Bucket Name
            let name_str = self.bucket_names.get(id).map(|s| s.as_str()).unwrap_or("unknown");
            let name_bytes = name_str.as_bytes();
            writer.write_all(&(name_bytes.len() as u64).to_le_bytes())?; 
            writer.write_all(name_bytes)?;

            // 2. Write Sources List
            let sources = self.bucket_sources.get(id);
            let empty = Vec::new();
            let src_vec = sources.unwrap_or(&empty);
            
            writer.write_all(&(src_vec.len() as u64).to_le_bytes())?;
            for src in src_vec {
                let s_bytes = src.as_bytes();
                writer.write_all(&(s_bytes.len() as u64).to_le_bytes())?;
                writer.write_all(s_bytes)?;
            }
            
            // 3. Write Minimizers
            writer.write_all(&(vec.len() as u64).to_le_bytes())?; 
            for val in vec {
                writer.write_all(&val.to_le_bytes())?;
            }
        }
        Ok(())
    }

    fn load(path: &Path) -> Result<Self> {
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
            
            // 1. Read Name
            reader.read_exact(&mut buf8)?; let name_len = u64::from_le_bytes(buf8) as usize;
            let mut nbuf = vec![0u8; name_len]; 
            reader.read_exact(&mut nbuf)?;
            names.insert(id, String::from_utf8(nbuf)?);

            // 2. Read Sources
            reader.read_exact(&mut buf8)?; let src_count = u64::from_le_bytes(buf8) as usize;
            let mut src_list = Vec::with_capacity(src_count);
            for _ in 0..src_count {
                reader.read_exact(&mut buf8)?; let slen = u64::from_le_bytes(buf8) as usize;
                let mut sbuf = vec![0u8; slen];
                reader.read_exact(&mut sbuf)?;
                src_list.push(String::from_utf8(sbuf)?);
            }
            sources.insert(id, src_list);
            
            // 3. Read Minimizers
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

// --- IO ABSTRACTION ---

struct IoHandler {
    r1: Box<dyn FastxReader>,
    r2: Option<Box<dyn FastxReader>>,
    tx: Sender<Option<Vec<u8>>>, 
    writer_handle: Option<thread::JoinHandle<Result<()>>>,
}

impl IoHandler {
    fn new(r1: &Path, r2: Option<&PathBuf>, out: Option<PathBuf>) -> Result<Self> {
        let r1_r = parse_fastx_file(r1).context("R1 fail")?;
        let r2_r = if let Some(p) = r2 { Some(parse_fastx_file(p).context("R2 fail")?) } else { None };

        let (tx, rx) = channel::<Option<Vec<u8>>>();
        
        let handle = thread::spawn(move || {
            let mut w: Box<dyn Write> = match out {
                Some(p) => Box::new(BufWriter::new(File::create(p)?)),
                None => Box::new(BufWriter::new(io::stdout())),
            };
            while let Ok(Some(data)) = rx.recv() {
                if let Err(e) = w.write_all(&data) {
                    if e.kind() == io::ErrorKind::BrokenPipe { return Ok(()); }
                    return Err(e.into());
                }
            }
            if let Err(e) = w.flush() {
                if e.kind() != io::ErrorKind::BrokenPipe { return Err(e.into()); }
            }
            Ok(())
        });

        Ok(Self { r1: r1_r, r2: r2_r, tx, writer_handle: Some(handle) })
    }

    fn write(&self, data: Vec<u8>) -> Result<()> {
        self.tx.send(Some(data)).map_err(|_| {
            std::io::Error::from(std::io::ErrorKind::BrokenPipe).into()
        })
    }

    fn next_batch(&mut self, size: usize) -> Result<Option<(Vec<String>, Vec<Vec<u8>>, Option<Vec<Vec<u8>>>)>> {
        let mut ids = Vec::with_capacity(size);
        let mut s1s = Vec::with_capacity(size);
        let mut s2s = if self.r2.is_some() { Some(Vec::with_capacity(size)) } else { None };

        for _ in 0..size {
            match self.r1.next() {
                Some(Ok(rec)) => { ids.push(String::from_utf8_lossy(rec.id()).to_string()); s1s.push(rec.seq().into_owned()); },
                Some(Err(e)) => return Err(anyhow!(e)),
                None => break,
            }
            if let Some(r2) = &mut self.r2 {
                match r2.next() {
                    Some(Ok(rec)) => s2s.as_mut().unwrap().push(rec.seq().into_owned()),
                    Some(Err(e)) => return Err(anyhow!(e)),
                    None => return Err(anyhow!("R1/R2 mismatch")),
                }
            }
        }
        if ids.is_empty() { return Ok(None); }
        Ok(Some((ids, s1s, s2s)))
    }

    fn finish(mut self) -> Result<()> {
        let _ = self.tx.send(None); 
        if let Some(h) = self.writer_handle.take() { h.join().unwrap()?; }
        Ok(())
    }
}

// --- MAIN ---

fn main() -> Result<()> {
    match run_cli() {
        Ok(_) => Ok(()),
        Err(e) => {
            if let Some(io_err) = e.downcast_ref::<std::io::Error>() {
                if io_err.kind() == std::io::ErrorKind::BrokenPipe { return Ok(()); }
            }
            Err(e)
        }
    }
}

fn run_cli() -> Result<()> {
    let args = Cli::parse();

    match args.command {
        Commands::Index { output, reference, window, salt, separate_buckets } => {
            let mut index = Index::new(window, salt);
            for (i, ref_path) in reference.iter().enumerate() {
                let id = if separate_buckets { (i + 1) as u32 } else { 1 };
                println!("[*] Indexing {:?} -> Bucket {}", ref_path, id);
                index.add_reference_file(ref_path, id)?;
            }
            index.save(&output)?;
        }
        Commands::IndexStats { index } => {
            let idx = Index::load(&index)?;
            println!("Window (w): {}", idx.w);
            println!("Salt: 0x{:x}", idx.salt);
            println!("Total Buckets: {}", idx.buckets.len());
            println!("\n{:<10} {:<30} {:<15}", "BucketID", "BucketName", "UniqueMinimizers");
            println!("{:-<70}", "");
            
            let mut ids: Vec<_> = idx.buckets.keys().collect();
            ids.sort();
            for id in ids {
                let count = idx.buckets[id].len();
                let name = idx.bucket_names.get(id).map(|s| s.as_str()).unwrap_or("?");
                println!("{:<10} {:<30} {:<15}", id, name, count);
                
                if let Some(sources) = idx.bucket_sources.get(id) {
                    println!("  Sources included in this bucket:");
                    for src in sources {
                        println!("    - {}", src);
                    }
                }
                println!();
            }
        }
        Commands::IndexBucketAdd { index, reference } => {
            let mut idx = Index::load(&index)?;
            let new_id = idx.next_id();
            println!("[*] Adding {:?} -> Bucket {}", reference, new_id);
            idx.add_reference_file(&reference, new_id)?;
            idx.save(&index)?;
        }
        Commands::IndexBucketMerge { index, src, dest } => {
            let mut idx = Index::load(&index)?;
            println!("[*] Merging Bucket {} into {}", src, dest);
            idx.merge_buckets(src, dest)?;
            idx.save(&index)?;
        }
        Commands::IndexMerge { output, inputs } => {
            if inputs.is_empty() { return Err(anyhow!("No input files")); }
            let mut base = Index::load(&inputs[0])?;
            
            for path in &inputs[1..] {
                let other = Index::load(path)?;
                if other.w != base.w || other.salt != base.salt {
                    return Err(anyhow!("Index parameter mismatch (w/salt) in {:?}", path));
                }
                
                for (old_id, vec) in other.buckets {
                    let new_id = base.next_id();
                    let name = other.bucket_names.get(&old_id).cloned().unwrap_or_else(|| "imported".to_string());
                    
                    base.bucket_names.insert(new_id, name);
                    base.buckets.insert(new_id, vec);
                    
                    if let Some(srcs) = other.bucket_sources.get(&old_id) {
                        base.bucket_sources.insert(new_id, srcs.clone());
                    }
                }
            }
            base.save(&output)?;
        }
        Commands::Classify { index, r1, r2, threshold, batch_size, output } => {
            let engine = Index::load(&index)?;
            let mut io = IoHandler::new(&r1, r2.as_ref(), output)?;
            io.write(b"read_id\tbucket_id\tscore\n".to_vec())?;
            run_classify(&engine, &mut io, threshold, batch_size)?;
            io.finish()?;
        }
        Commands::BatchClassify { index, r1, r2, threshold, batch_size, output } => {
            let engine = Index::load(&index)?;
            let mut io = IoHandler::new(&r1, r2.as_ref(), output)?;
            io.write(b"read_id\tbucket_id\tscore\n".to_vec())?;
            run_batch_classify(&engine, &mut io, threshold, batch_size)?;
            io.finish()?;
        }
        Commands::AggregateClassify { index, r1, r2, threshold, batch_size, output } => {
            let engine = Index::load(&index)?;
            let mut io = IoHandler::new(&r1, r2.as_ref(), output)?;
            io.write(b"query_name\tbucket_id\tscore\n".to_vec())?;
            run_aggregate_classify(&engine, &mut io, threshold, batch_size)?;
            io.finish()?;
        }
    }
    Ok(())
}

fn run_classify(engine: &Index, io: &mut IoHandler, threshold: f64, batch_size: usize) -> Result<()> {
    while let Some((ids, s1s, s2s)) = io.next_batch(batch_size)? {
        let results: Vec<u8> = ids.par_iter().enumerate()
            .map_init(MinimizerWorkspace::new, |ws, (i, id)| {
                let s2 = s2s.as_ref().map(|v| &v[i]);
                let (hyp_a, hyp_b) = get_paired_minimizers_into(&s1s[i], s2, engine.w, engine.salt, ws);
                let (la, lb) = (hyp_a.len() as f64, hyp_b.len() as f64);
                if la == 0.0 && lb == 0.0 { return Vec::new(); }

                let mut out = Vec::with_capacity(128);
                for (b_id, bucket) in &engine.buckets {
                    let sa = if la > 0.0 { count_hits(&hyp_a, bucket) / la } else { 0.0 };
                    let sb = if lb > 0.0 { count_hits(&hyp_b, bucket) / lb } else { 0.0 };
                    let score = sa.max(sb);
                    if score >= threshold {
                        writeln!(out, "{}\t{}\t{:.4}", id, b_id, score).unwrap();
                    }
                }
                out
            }).flatten().collect();
        io.write(results)?;
    }
    Ok(())
}

fn run_batch_classify(engine: &Index, io: &mut IoHandler, threshold: f64, batch_size: usize) -> Result<()> {
    while let Some((ids, s1s, s2s)) = io.next_batch(batch_size)? {
        let processed: Vec<_> = s1s.par_iter().enumerate()
            .map_init(MinimizerWorkspace::new, |ws, (i, s1)| {
                let s2 = s2s.as_ref().map(|v| &v[i]);
                let (ha, hb) = get_paired_minimizers_into(s1, s2, engine.w, engine.salt, ws);
                (ha.len() as f64, hb.len() as f64, ha, hb)
            }).collect();

        let mut map_a: HashMap<u64, Vec<usize>> = HashMap::new();
        let mut map_b: HashMap<u64, Vec<usize>> = HashMap::new();
        let mut uniq = HashSet::new();

        for (idx, (_, _, ma, mb)) in processed.iter().enumerate() {
            for &m in ma { map_a.entry(m).or_default().push(idx); uniq.insert(m); }
            for &m in mb { map_b.entry(m).or_default().push(idx); uniq.insert(m); }
        }
        let uniq_vec: Vec<u64> = uniq.into_iter().collect();

        let batch_results: Vec<u8> = engine.buckets.par_iter().map(|(b_id, bucket)| {
            let mut hits = HashMap::new();
            for &m in &uniq_vec {
                if bucket.binary_search(&m).is_ok() {
                    if let Some(rs) = map_a.get(&m) { for &r in rs { hits.entry(r).or_insert((0,0)).0 += 1; } }
                    if let Some(rs) = map_b.get(&m) { for &r in rs { hits.entry(r).or_insert((0,0)).1 += 1; } }
                }
            }
            let mut buf = Vec::new();
            for (r, (ha, hb)) in hits {
                let (la, lb, _, _) = &processed[r];
                let score = (if *la > 0.0 { ha as f64 / la } else { 0.0 }).max(if *lb > 0.0 { hb as f64 / lb } else { 0.0 });
                if score >= threshold { writeln!(buf, "{}\t{}\t{:.4}", ids[r], b_id, score).unwrap(); }
            }
            buf
        }).flatten().collect();
        io.write(batch_results)?;
    }
    Ok(())
}

fn run_aggregate_classify(engine: &Index, io: &mut IoHandler, threshold: f64, batch_size: usize) -> Result<()> {
    let mut global = HashSet::new();
    while let Some((_, s1s, s2s)) = io.next_batch(batch_size)? {
        let batch: Vec<Vec<u64>> = s1s.par_iter().enumerate()
            .map_init(MinimizerWorkspace::new, |ws, (i, s1)| {
                let s2 = s2s.as_ref().map(|v| &v[i]);
                let (mut a, b) = get_paired_minimizers_into(s1, s2, engine.w, engine.salt, ws);
                a.extend(b); a
            }).collect();
        for v in batch { for m in v { global.insert(m); } }
    }
    let total = global.len() as f64;
    if total == 0.0 { return Ok(()); }
    let g_vec: Vec<u64> = global.into_iter().collect();
    
    let res: Vec<u8> = engine.buckets.par_iter().filter_map(|(id, b)| {
        let s = count_hits(&g_vec, b) / total;
        if s >= threshold { Some(format!("global\t{}\t{:.4}\n", id, s).into_bytes()) } else { None }
    }).flatten().collect();
    io.write(res)?;
    Ok(())
}

fn count_hits(mins: &[u64], bucket: &[u64]) -> f64 {
    let mut hits = 0;
    for m in mins {
        if bucket.binary_search(m).is_ok() {
            hits += 1;
        }
    }
    hits as f64
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
        // Simple property check: RC of poly-A/C mix should definitely be different in RY space if not palindromic
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
    fn test_add_reference_file() -> Result<()> {
        // Create dummy FASTA
        let fa = create_temp_fasta(b">seq1\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n");
        let path = fa.path();
        let filename = path.file_name().unwrap().to_string_lossy().to_string();

        let mut index = Index::new(5, 0);
        index.add_reference_file(path, 1)?;

        assert!(index.buckets.contains_key(&1));
        assert!(!index.buckets[&1].is_empty());
        
        // Verify Source Metadata
        let sources = &index.bucket_sources[&1];
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0], format!("{}::seq1", filename));

        Ok(())
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
        // Sorted and Deduped: 1, 2, 3, 4, 5
        assert_eq!(merged_vec.len(), 5);
        assert_eq!(merged_vec, &vec![1, 2, 3, 4, 5]);

        let merged_sources = &index.bucket_sources[&20];
        assert_eq!(merged_sources.len(), 2);
        assert!(merged_sources.contains(&"s1".to_string()));
        assert!(merged_sources.contains(&"d1".to_string()));

        Ok(())
    }

    #[test]
    fn test_next_id() {
        let mut index = Index::new(50, 0);
        assert_eq!(index.next_id(), 1);
        
        index.buckets.insert(1, vec![]);
        assert_eq!(index.next_id(), 2);
        
        index.buckets.insert(5, vec![]);
        assert_eq!(index.next_id(), 6);
    }
}

