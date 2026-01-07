use clap::{Parser, Subcommand};
use needletail::{parse_fastx_file, FastxReader};
use rayon::prelude::*;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::collections::HashSet;
use anyhow::{Context, Result, anyhow};

use rype::{Index, MinimizerWorkspace, QueryRecord, classify_batch, aggregate_batch};
use rype::config::{parse_config, validate_config, resolve_path};

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
    IndexBucketSourceDetail {
        #[arg(short, long)]
        index: PathBuf,
        #[arg(short, long, required = true)]
        bucket: u32,
        #[arg(long)]
        paths: bool,
        #[arg(long)]
        ids: bool,
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
    IndexFromConfig {
        #[arg(short, long)]
        config: PathBuf,
    },
}

// --- HELPER FUNCTIONS ---

fn add_reference_file_to_index(
    index: &mut Index,
    path: &Path,
    separate_buckets: bool,
    next_id: &mut u32
) -> Result<()> {
    let mut reader = parse_fastx_file(path).context("Failed to open reference file")?;
    let mut ws = MinimizerWorkspace::new();
    let filename = path.canonicalize().unwrap().to_string_lossy().to_string();

    while let Some(record) = reader.next() {
        let rec = record.context("Invalid record")?;
        let seq = rec.seq();
        let name = String::from_utf8_lossy(rec.id()).to_string();

        let bucket_id = if separate_buckets {
            let id = *next_id;
            *next_id += 1;
            index.bucket_names.insert(id, name.clone());
            id
        } else {
            1
        };

        if !separate_buckets {
             // Just use bucket 1 and label it with the filename if not set
             index.bucket_names.entry(1).or_insert(filename.clone());
        }

        let source_label = format!("{}{}{}", filename, Index::BUCKET_SOURCE_DELIM, name);
        index.add_record(bucket_id, &source_label, &seq, &mut ws);
    }
    
    // Finalize relevant buckets
    if separate_buckets {
        // FIX: Collect keys into a Vec first to avoid immutable borrow during mutable iteration
        let ids: Vec<u32> = index.buckets.keys().copied().collect();
        for id in ids {
            index.finalize_bucket(id);
        }
    } else {
        index.finalize_bucket(1);
    }
    
    Ok(())
}

// --- IO HANDLER ---

type OwnedRecord = (i64, Vec<u8>, Option<Vec<u8>>);

struct IoHandler {
    r1: Box<dyn FastxReader>,
    r2: Option<Box<dyn FastxReader>>,
    writer: BufWriter<Box<dyn Write>>,
}

impl IoHandler {
    fn new(r1_path: &Path, r2_path: Option<&PathBuf>, out_path: Option<PathBuf>) -> Result<Self> {
        let r1 = parse_fastx_file(r1_path).context("Failed to open R1")?;
        
        let r2 = if let Some(p) = r2_path {
            Some(parse_fastx_file(p).context("Failed to open R2")?)
        } else {
            None
        };

        let output: Box<dyn Write> = if let Some(p) = out_path {
            Box::new(File::create(p).context("Failed to create output file")?)
        } else {
            Box::new(io::stdout())
        };

        Ok(Self {
            r1,
            r2,
            writer: BufWriter::new(output),
        })
    }

    fn next_batch_records(&mut self, size: usize) -> Result<Option<(Vec<OwnedRecord>, Vec<String>)>> {
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
            // Ownership transfer: .seq().into_owned()
            records.push((i as i64, s1_rec.seq().into_owned(), s2_vec));
            headers.push(header);
        }

        if records.is_empty() { return Ok(None); }
        Ok(Some((records, headers)))
    }

    fn write(&mut self, data: Vec<u8>) -> Result<()> {
        self.writer.write_all(&data)?;
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

// --- MAIN ---

fn main() -> Result<()> {
    let args = Cli::parse();

    match args.command {
        Commands::Index { output, reference, window, salt, separate_buckets } => {
            let mut index = Index::new(window, salt);
            let mut next_id = 1;

            for ref_file in reference {
                println!("Adding reference: {:?}", ref_file);
                add_reference_file_to_index(&mut index, &ref_file, separate_buckets, &mut next_id)?;
            }

            println!("Saving index to {:?}...", output);
            index.save(&output)?;
            println!("Done.");
        }
        
        Commands::IndexStats { index } => {
            let idx = Index::load(&index)?;
            println!("Index Stats for {:?}", index);
            println!("  K: 64 (fixed)");
            println!("  Window (w): {}", idx.w);
            println!("  Salt: {:x}", idx.salt);
            println!("  Buckets: {}", idx.buckets.len());
            println!("------------------------------------------------");
            let mut sorted_ids: Vec<_> = idx.buckets.keys().collect();
            sorted_ids.sort();
            for id in sorted_ids {
                let name = idx.bucket_names.get(id).map(|s| s.as_str()).unwrap_or("unknown");
                let count = idx.buckets[id].len();
                let sources = idx.bucket_sources.get(id).map(|v| v.len()).unwrap_or(0);
                println!("  Bucket {}: '{}' ({} minimizers, {} sources)", id, name, count, sources);
            }
        }

        Commands::IndexBucketSourceDetail { index, bucket, paths, ids } => {
            let idx = Index::load(&index)?;
            let sources = idx.bucket_sources.get(&bucket).unwrap();

            if paths && ids {
                return Err(anyhow!("Cannot have --paths and --ids"));
            }
            
            if paths {
                let mut all_paths: HashSet<String> = HashSet::new();
                for source in sources {
                    let parts: Vec<_> = source.split(Index::BUCKET_SOURCE_DELIM).collect();
                    let path = parts.first().unwrap().to_string();
                    all_paths.insert(path.clone());
                }

                for path in all_paths {
                    println!("{}", path);
                }
            } else if ids {
                let mut sorted_ids: Vec<_> = idx.buckets.keys().collect();
                sorted_ids.sort();
                for id in sorted_ids {
                    println!("{}", id);
                }
            } else {
                for source in sources {
                    println!("{}", source);
                }
            }
        }

        Commands::IndexBucketAdd { index, reference } => {
            let mut idx = Index::load(&index)?;
            let next_id = idx.next_id()?;
            println!("Adding {:?} as new bucket ID {}", reference, next_id);

            // Add all records from the file to a single new bucket
            let mut reader = parse_fastx_file(&reference).context("Failed to open reference file")?;
            let mut ws = MinimizerWorkspace::new();
            let filename = reference.canonicalize().unwrap().to_string_lossy().to_string();

            // Set the bucket name to the filename for consistency with 'index' command
            idx.bucket_names.insert(next_id, filename.clone());

            while let Some(record) = reader.next() {
                let rec = record.context("Invalid record")?;
                let seq = rec.seq();
                let name = String::from_utf8_lossy(rec.id()).to_string();
                let source_label = format!("{}{}{}", filename, Index::BUCKET_SOURCE_DELIM, name);
                idx.add_record(next_id, &source_label, &seq, &mut ws);
            }

            // Finalize the new bucket
            idx.finalize_bucket(next_id);
            idx.save(&index)?;
            println!("Done. Added {} minimizers to bucket {}.",
                     idx.buckets.get(&next_id).map(|v| v.len()).unwrap_or(0), next_id);
        }

        Commands::IndexBucketMerge { index, src, dest } => {
            let mut idx = Index::load(&index)?;
            println!("Merging Bucket {} -> Bucket {}...", src, dest);
            idx.merge_buckets(src, dest)?;
            idx.save(&index)?;
            println!("Done.");
        }

        Commands::IndexMerge { output, inputs } => {
            // Logic: Load first index, then merge others into it. 
            // Warning: Salt/W must match.
            if inputs.is_empty() { return Err(anyhow!("No input indexes provided")); }
            
            println!("Loading base index: {:?}", inputs[0]);
            let mut base_idx = Index::load(&inputs[0])?;

            for path in &inputs[1..] {
                println!("Merging index: {:?}", path);
                let other_idx = Index::load(path)?;
                
                if other_idx.w != base_idx.w || other_idx.salt != base_idx.salt {
                    return Err(anyhow!("Index parameters (w/salt) mismatch in {:?}", path));
                }

                // Naive merge strategy: Re-map IDs of 'other' to not collide, then insert
                // Simple version: just append buckets with new IDs
                for (old_id, vec) in other_idx.buckets {
                    let new_id = base_idx.next_id()?;
                    base_idx.buckets.insert(new_id, vec);
                    
                    if let Some(name) = other_idx.bucket_names.get(&old_id) {
                        base_idx.bucket_names.insert(new_id, name.clone());
                    }
                    if let Some(srcs) = other_idx.bucket_sources.get(&old_id) {
                        base_idx.bucket_sources.insert(new_id, srcs.clone());
                    }
                }
            }
            base_idx.save(&output)?;
            println!("Merged index saved to {:?}", output);
        }

        Commands::Classify { index, r1, r2, threshold, batch_size, output } | 
        Commands::BatchClassify { index, r1, r2, threshold, batch_size, output } => {
            let engine = Index::load(&index)?;
            let mut io = IoHandler::new(&r1, r2.as_ref(), output)?;
            io.write(b"read_id\tbucket_id\tscore\n".to_vec())?;
            
            // 1. Get OWNED records from disk
            while let Some((owned_records, headers)) = io.next_batch_records(batch_size)? {
                
                // 2. Create REFERENCES for the library
                let batch_refs: Vec<QueryRecord> = owned_records.iter()
                    .map(|(id, s1, s2)| (*id, s1.as_slice(), s2.as_deref()))
                    .collect();

                // 3. Process
                let results = classify_batch(&engine, &batch_refs, threshold);
                
                // 4. Write Output
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

            while let Some((owned_records, _)) = io.next_batch_records(batch_size)? {

                let batch_refs: Vec<QueryRecord> = owned_records.iter()
                    .map(|(id, s1, s2)| (*id, s1.as_slice(), s2.as_deref()))
                    .collect();

                let results = aggregate_batch(&engine, &batch_refs, threshold);

                let mut chunk_out = Vec::with_capacity(1024);
                for res in results {
                    writeln!(chunk_out, "global\t{}\t{:.4}", res.bucket_id, res.score).unwrap();
                }
                io.write(chunk_out)?;
            }
            io.finish()?;
        }

        Commands::IndexFromConfig { config } => {
            build_index_from_config(&config)?;
        }
    }

    Ok(())
}

fn build_index_from_config(config_path: &Path) -> Result<()> {
    println!("Building index from config: {}", config_path.display());

    // 1. Parse and validate config
    let cfg = parse_config(config_path)?;
    let config_dir = config_path.parent()
        .ok_or_else(|| anyhow!("Invalid config path"))?;

    println!("Validating file paths...");
    validate_config(&cfg, config_dir)?;
    println!("Validation successful.");

    // 2. Sort bucket names for deterministic ordering
    let mut bucket_names: Vec<_> = cfg.buckets.keys().cloned().collect();
    bucket_names.sort();

    println!("Building {} buckets in parallel...", bucket_names.len());
    for name in &bucket_names {
        let file_count = cfg.buckets[name].files.len();
        println!("  - {}: {} file{}", name, file_count, if file_count == 1 { "" } else { "s" });
    }

    // 3. Build indices in parallel (one per bucket)
    let bucket_indices: Vec<_> = bucket_names.par_iter()
        .map(|bucket_name| {
            let mut idx = Index::new(cfg.index.window, cfg.index.salt);
            let mut ws = MinimizerWorkspace::new();

            // Process all files for this bucket
            for file_path in &cfg.buckets[bucket_name].files {
                let abs_path = resolve_path(config_dir, file_path);
                let mut reader = parse_fastx_file(&abs_path)
                    .context(format!("Failed to open file {} for bucket '{}'",
                                   abs_path.display(), bucket_name))?;

                let filename = file_path.canonicalize()
                    .unwrap()
                    .to_string_lossy()
                    .to_string();

                while let Some(record) = reader.next() {
                    let rec = record.context(format!("Invalid record in file {} (bucket '{}')",
                                                    abs_path.display(), bucket_name))?;
                    let seq_name = String::from_utf8_lossy(rec.id()).to_string();
                    let source_label = format!("{}{}{}", filename, Index::BUCKET_SOURCE_DELIM, seq_name);
                    idx.add_record(1, &source_label, &rec.seq(), &mut ws);
                }
            }

            idx.finalize_bucket(1);
            Ok::<_, anyhow::Error>((bucket_name.clone(), idx))
        })
        .collect::<Result<Vec<_>>>()?;

    println!("Processing complete. Merging buckets...");

    // 4. Merge all bucket indices into final index
    let mut final_index = Index::new(cfg.index.window, cfg.index.salt);
    for (bucket_id, (bucket_name, bucket_idx)) in bucket_indices.into_iter().enumerate() {
        let new_id = (bucket_id + 1) as u32;
        final_index.bucket_names.insert(new_id, bucket_name);

        // Transfer bucket data
        if let Some(minimizers) = bucket_idx.buckets.get(&1) {
            final_index.buckets.insert(new_id, minimizers.clone());
        }
        if let Some(sources) = bucket_idx.bucket_sources.get(&1) {
            final_index.bucket_sources.insert(new_id, sources.clone());
        }
    }

    // 5. Save final index
    println!("Saving index to {}...", cfg.index.output.display());
    final_index.save(&cfg.index.output)?;

    println!("Done! Index saved successfully.");
    println!("\nFinal statistics:");
    println!("  Buckets: {}", final_index.buckets.len());
    let total_minimizers: usize = final_index.buckets.values().map(|v| v.len()).sum();
    println!("  Total minimizers: {}", total_minimizers);

    Ok(())
}

