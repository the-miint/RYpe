use clap::{Parser, Subcommand};
use needletail::{parse_fastx_file, FastxReader};
// REMOVED: use std::collections::{HashMap, HashSet}; <-- These caused the warning
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Sender};
use std::thread;
use anyhow::{Context, Result, anyhow};

use rype::{Index, MinimizerWorkspace, QueryRecord, classify_batch, aggregate_batch};

#[derive(Parser)]
#[command(name = "rype")]
#[command(about = "High-performance Read Partitioning Engine (RY-Space, K=64)", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Index {
        #[arg(short, long)]
        output: PathBuf,
        #[arg(short, long, required = true)]
        reference: Vec<PathBuf>,
        #[arg(short, long, default_value_t = 50)]
        window: usize,
        #[arg(short, long, default_value_t = 0x5555555555555555)]
        salt: u64,
        #[arg(long)]
        separate_buckets: bool,
    },
    IndexStats {
        #[arg(short, long)]
        index: PathBuf,
    },
    IndexBucketAdd {
        #[arg(short, long)]
        index: PathBuf,
        #[arg(short, long)]
        reference: PathBuf,
    },
    IndexBucketMerge {
        #[arg(short, long)]
        index: PathBuf,
        #[arg(long)]
        src: u32,
        #[arg(long)]
        dest: u32,
    },
    IndexMerge {
        #[arg(short, long)]
        output: PathBuf,
        #[arg(short, long, required = true)]
        inputs: Vec<PathBuf>,
    },
    Classify {
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
    // ADDED BACK: This variant was missing
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

// ... rest of file (add_reference_file_to_index, IoHandler, main implementation) remains the same

// --- HELPERS ---

fn add_reference_file_to_index(index: &mut Index, path: &Path, id: u32) -> Result<()> {
    let mut reader = parse_fastx_file(path).with_context(|| format!("Failed to open reference: {:?}", path))?;
    let fname = path.file_name().unwrap().to_string_lossy().to_string();
    index.bucket_names.entry(id).or_insert_with(|| fname.clone());
    
    let mut ws = MinimizerWorkspace::new();
    while let Some(record) = reader.next() {
        let seqrec = record.context("Invalid FASTA")?;
        let seq_id = String::from_utf8_lossy(seqrec.id()).to_string();
        let source_name = format!("{}::{}", fname, seq_id);
        index.add_record(id, &source_name, &seqrec.seq(), &mut ws);
    }
    index.finalize_bucket(id);
    Ok(())
}

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
            let _ = w.flush();
            Ok(())
        });
        Ok(Self { r1: r1_r, r2: r2_r, tx, writer_handle: Some(handle) })
    }

    fn write(&self, data: Vec<u8>) -> Result<()> {
        self.tx.send(Some(data)).map_err(|_| std::io::Error::from(std::io::ErrorKind::BrokenPipe).into())
    }

    /// Reads a batch and maps String IDs to temporary i64 (0..batch_size)
    /// Returns: (Vec<QueryRecord>, Vec<OriginalHeaderString>)
    fn next_batch_records(&mut self, size: usize) -> Result<Option<(Vec<QueryRecord>, Vec<String>)>> {
        let mut records = Vec::with_capacity(size);
        let mut headers = Vec::with_capacity(size);
        
        for i in 0..size {
            let s1_rec = match self.r1.next() {
                Some(Ok(rec)) => rec,
                Some(Err(e)) => return Err(anyhow!(e)),
                None => break,
            };

            let s2_vec = if let Some(r2) = &mut self.r2 {
                match r2.next() {
                    Some(Ok(rec)) => Some(rec.seq().into_owned()),
                    Some(Err(e)) => return Err(anyhow!(e)),
                    None => return Err(anyhow!("R1/R2 mismatch")),
                }
            } else {
                None
            };

            let header = String::from_utf8_lossy(s1_rec.id()).to_string();
            records.push((i as i64, s1_rec.seq().into_owned(), s2_vec));
            headers.push(header);
        }

        if records.is_empty() { return Ok(None); }
        Ok(Some((records, headers)))
    }

    fn finish(mut self) -> Result<()> {
        let _ = self.tx.send(None);
        if let Some(h) = self.writer_handle.take() { h.join().unwrap()?; }
        Ok(())
    }
}

fn main() -> Result<()> {
    let args = Cli::parse();

    match args.command {
        Commands::Index { output, reference, window, salt, separate_buckets } => {
            let mut index = Index::new(window, salt);
            for (i, ref_path) in reference.iter().enumerate() {
                let id = if separate_buckets { (i + 1) as u32 } else { 1 };
                println!("[*] Indexing {:?} -> Bucket {}", ref_path, id);
                add_reference_file_to_index(&mut index, ref_path, id)?;
            }
            index.save(&output)?;
        }
        Commands::IndexStats { index } => {
            let idx = Index::load(&index)?;
            println!("Window (w): {}", idx.w);
            println!("Salt: 0x{:x}", idx.salt);
            println!("Total Buckets: {}", idx.buckets.len());
            for id in idx.buckets.keys() {
                println!("Bucket {}: {} minimizers", id, idx.buckets[id].len());
            }
        }
        Commands::IndexBucketAdd { index, reference } => {
            let mut idx = Index::load(&index)?;
            let new_id = idx.next_id();
            println!("[*] Adding {:?} -> Bucket {}", reference, new_id);
            add_reference_file_to_index(&mut idx, &reference, new_id)?;
            idx.save(&index)?;
        }
        Commands::IndexBucketMerge { index, src, dest } => {
            let mut idx = Index::load(&index)?;
            idx.merge_buckets(src, dest)?;
            idx.save(&index)?;
        }
        Commands::IndexMerge { output, inputs } => {
            if inputs.is_empty() { return Err(anyhow!("No input files")); }
            let mut base = Index::load(&inputs[0])?;
            for path in &inputs[1..] {
                let other = Index::load(path)?;
                if other.w != base.w || other.salt != base.salt { return Err(anyhow!("Index mismatch")); }
                for (old_id, vec) in other.buckets {
                    let new_id = base.next_id();
                    base.bucket_names.insert(new_id, other.bucket_names.get(&old_id).cloned().unwrap_or("imported".into()));
                    base.buckets.insert(new_id, vec);
                    if let Some(srcs) = other.bucket_sources.get(&old_id) {
                        base.bucket_sources.insert(new_id, srcs.clone());
                    }
                }
            }
            base.save(&output)?;
        }
        Commands::Classify { index, r1, r2, threshold, batch_size, output } | 
        Commands::BatchClassify { index, r1, r2, threshold, batch_size, output } => {
            // Note: BatchClassify logic is now the default "high perf" way. 
            // The library's `classify_batch` handles the optimizations.
            let engine = Index::load(&index)?;
            let mut io = IoHandler::new(&r1, r2.as_ref(), output)?;
            io.write(b"read_id\tbucket_id\tscore\n".to_vec())?;
            
            while let Some((records, headers)) = io.next_batch_records(batch_size)? {
                let results = classify_batch(&engine, &records, threshold);
                
                let mut chunk_out = Vec::with_capacity(1024);
                for res in results {
                    let header = &headers[res.query_id as usize];
                    writeln!(chunk_out, "{}\t{}\t{:.4}", header, res.bucket_id, res.score).unwrap();
                }
                io.write(chunk_out)?;
            }
            io.finish()?;
        }
        Commands::AggregateClassify { index, r1, r2, threshold, batch_size, output } => {
            let engine = Index::load(&index)?;
            let mut io = IoHandler::new(&r1, r2.as_ref(), output)?;
            io.write(b"query_name\tbucket_id\tscore\n".to_vec())?;

            while let Some((records, _)) = io.next_batch_records(batch_size)? {
                let results = aggregate_batch(&engine, &records, threshold);
                
                let mut chunk_out = Vec::with_capacity(1024);
                for res in results {
                    writeln!(chunk_out, "global\t{}\t{:.4}", res.bucket_id, res.score).unwrap();
                }
                io.write(chunk_out)?;
            }
            io.finish()?;
        }
    }
    Ok(())
}

