use clap::{Parser, Subcommand};
use needletail::{parse_fastx_file, FastxReader};
use rayon::prelude::*;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::collections::HashSet;
use anyhow::{Context, Result, anyhow};

use rype::{Index, InvertedIndex, MinimizerWorkspace, QueryRecord, classify_batch, classify_batch_inverted, classify_batch_with_query_index, classify_batch_sharded_sequential, classify_batch_sharded_merge_join, classify_batch_sharded_main, aggregate_batch, ShardManifest, ShardedInvertedIndex, MainIndexManifest, MainIndexShard, ShardedMainIndex, extract_into, extract_with_positions, Strand};
use rype::config::{parse_config, validate_config, resolve_path};
use std::collections::HashMap;
use std::io::BufRead;

mod logging;

// --- HELPER FUNCTIONS ---

/// Sanitize bucket names by replacing nonprintable characters with "_"
fn sanitize_bucket_name(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_control() || !c.is_ascii_graphic() && !c.is_whitespace() {
            '_'
        } else {
            c
        })
        .collect()
}

#[derive(Parser)]
#[command(name = "rype")]
#[command(about = "High-performance Read Partitioning Engine (RY-Space, K=16/32/64)", long_about = None)]
struct Cli {
    /// Enable verbose progress output with timestamps
    #[arg(short, long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Index operations: create, modify, and inspect indices
    #[command(subcommand)]
    Index(IndexCommands),

    /// Classification operations: classify reads against an index
    #[command(subcommand)]
    Classify(ClassifyCommands),

    /// Inspect minimizer details and matches
    #[command(subcommand)]
    Inspect(InspectCommands),
}

#[derive(Subcommand)]
enum IndexCommands {
    /// Create a new index from reference sequences
    Create {
        #[arg(short, long)]
        output: PathBuf,
        #[arg(short, long, required = true)]
        reference: Vec<PathBuf>,
        #[arg(short = 'k', long, default_value_t = 64)]
        kmer_size: usize,
        #[arg(short, long, default_value_t = 50)]
        window: usize,
        #[arg(short, long, default_value_t = 0x5555555555555555)]
        salt: u64,
        #[arg(long)]
        separate_buckets: bool,
        /// Maximum shard size in bytes (e.g., 1073741824 for 1GB). If specified, creates a sharded index.
        #[arg(long)]
        max_shard_size: Option<usize>,
    },

    /// Show index statistics
    Stats {
        #[arg(short, long)]
        index: PathBuf,
        /// Show inverted index stats (if .ryxdi exists)
        #[arg(short = 'I', long)]
        inverted: bool,
    },

    /// Show source details for a specific bucket
    BucketSourceDetail {
        #[arg(short, long)]
        index: PathBuf,
        #[arg(short, long, required = true)]
        bucket: u32,
        #[arg(long)]
        paths: bool,
        #[arg(long)]
        ids: bool,
    },

    /// Add sequences to an existing index as a new bucket
    BucketAdd {
        #[arg(short, long)]
        index: PathBuf,
        #[arg(short, long)]
        reference: PathBuf,
    },

    /// Merge two buckets within an index
    BucketMerge {
        #[arg(short, long)]
        index: PathBuf,
        #[arg(long)]
        src: u32,
        #[arg(long)]
        dest: u32,
    },

    /// Merge multiple indices into one
    Merge {
        #[arg(short, long)]
        output: PathBuf,
        #[arg(short, long, required = true)]
        inputs: Vec<PathBuf>,
    },

    /// Build index from a TOML configuration file
    FromConfig {
        #[arg(short, long)]
        config: PathBuf,
    },

    /// Create inverted index for faster classification.
    /// If the main index is sharded, inverted shards are created with 1:1 correspondence.
    Invert {
        #[arg(short, long)]
        index: PathBuf,

        /// Number of shards to split the inverted index into (default: 1 = single file).
        /// Ignored if main index is sharded (uses 1:1 correspondence instead).
        #[arg(long, default_value_t = 1)]
        shards: u32,
    },

    /// Summarize index with detailed minimizer statistics
    Summarize {
        #[arg(short, long)]
        index: PathBuf,
    },
}

#[derive(Subcommand)]
enum ClassifyCommands {
    /// Classify reads against an index (per-read output)
    Run {
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
        /// Use inverted index (.ryxdi) for faster classification
        #[arg(short = 'I', long)]
        use_inverted: bool,
        /// Use merge-join with query inverted index (requires --use-inverted)
        #[arg(short = 'M', long)]
        merge_join: bool,
    },

    /// Batch classify reads (alias for 'run')
    Batch {
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
        /// Use inverted index (.ryxdi) for faster classification
        #[arg(short = 'I', long)]
        use_inverted: bool,
        /// Use merge-join with query inverted index (requires --use-inverted)
        #[arg(short = 'M', long)]
        merge_join: bool,
    },

    /// Aggregate classification (higher sensitivity, aggregates paired-end reads)
    #[command(alias = "agg")]
    Aggregate {
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

#[derive(Subcommand)]
enum InspectCommands {
    /// Show matching minimizers between queries and buckets with reference details
    Matches {
        /// Path to the index file
        #[arg(short, long)]
        index: PathBuf,

        /// Query sequences (FASTA/FASTQ)
        #[arg(short = '1', long)]
        queries: PathBuf,

        /// File with sequence IDs to inspect (one per line)
        #[arg(long)]
        ids: PathBuf,

        /// Bucket IDs to check against (comma-separated)
        #[arg(short, long, value_delimiter = ',', required = true)]
        buckets: Vec<u32>,
    },
}

fn add_reference_file_to_index(
    index: &mut Index,
    path: &Path,
    separate_buckets: bool,
    next_id: &mut u32
) -> Result<()> {
    log::info!("Adding reference: {}", path.display());
    let mut reader = parse_fastx_file(path).context("Failed to open reference file")?;
    let mut ws = MinimizerWorkspace::new();
    let filename = path.canonicalize().unwrap().to_string_lossy().to_string();
    let mut record_count = 0;

    while let Some(record) = reader.next() {
        let rec = record.context("Invalid record")?;
        let seq = rec.seq();
        let name = String::from_utf8_lossy(rec.id()).to_string();

        let bucket_id = if separate_buckets {
            let id = *next_id;
            *next_id += 1;
            index.bucket_names.insert(id, sanitize_bucket_name(&name));
            id
        } else {
            1
        };

        if !separate_buckets {
             // Just use bucket 1 and label it with the filename if not set
             index.bucket_names.entry(1).or_insert_with(|| sanitize_bucket_name(&filename));
        }

        let source_label = format!("{}{}{}", filename, Index::BUCKET_SOURCE_DELIM, name);
        index.add_record(bucket_id, &source_label, &seq, &mut ws);

        record_count += 1;
        if record_count % 100_000 == 0 {
            log::info!("Processed {} records from {}", record_count, path.display());
        }
    }

    log::info!("Finalized bucket processing for {}: {} total records", path.display(), record_count);
    
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

    // Initialize logging based on verbose flag
    logging::init_logger(args.verbose);

    match args.command {
        Commands::Index(index_cmd) => match index_cmd {
            IndexCommands::Create { output, reference, kmer_size, window, salt, separate_buckets, max_shard_size } => {
                if !matches!(kmer_size, 16 | 32 | 64) {
                    return Err(anyhow!("K must be 16, 32, or 64 (got {})", kmer_size));
                }
                let mut index = Index::new(kmer_size, window, salt)?;
                let mut next_id = 1;

                for ref_file in reference {
                    add_reference_file_to_index(&mut index, &ref_file, separate_buckets, &mut next_id)?;
                }

                if let Some(max_bytes) = max_shard_size {
                    log::info!("Saving sharded index to {:?} (max {} bytes/shard)...", output, max_bytes);
                    let manifest = index.save_sharded(&output, max_bytes)?;
                    log::info!("Created {} shards with {} total minimizers.", manifest.shards.len(), manifest.total_minimizers);
                } else {
                    log::info!("Saving index to {:?}...", output);
                    index.save(&output)?;
                }
                log::info!("Done.");
            }

            IndexCommands::Stats { index, inverted } => {
                if inverted {
                    // Show inverted index stats
                    let inverted_path = index.with_extension("ryxdi");
                    if !inverted_path.exists() {
                        return Err(anyhow!(
                            "Inverted index not found: {:?}. Create it with 'rype index invert -i {:?}'",
                            inverted_path, index
                        ));
                    }
                    let inv = InvertedIndex::load(&inverted_path)?;
                    println!("Inverted Index Stats for {:?}", inverted_path);
                    println!("  K: {}", inv.k);
                    println!("  Window (w): {}", inv.w);
                    println!("  Salt: 0x{:x}", inv.salt);
                    println!("  Unique minimizers: {}", inv.num_minimizers());
                    println!("  Total bucket references: {}", inv.num_bucket_entries());
                    if inv.num_minimizers() > 0 {
                        println!("  Avg buckets per minimizer: {:.2}",
                            inv.num_bucket_entries() as f64 / inv.num_minimizers() as f64);
                    }
                } else {
                    // Show primary index stats - detect sharded vs single-file
                    let main_manifest_path = MainIndexManifest::manifest_path(&index);

                    let metadata = if main_manifest_path.exists() {
                        // Sharded main index
                        let manifest = MainIndexManifest::load(&main_manifest_path)?;
                        println!("Sharded Index Stats for {:?}", index);
                        println!("  K: {}", manifest.k);
                        println!("  Window (w): {}", manifest.w);
                        println!("  Salt: 0x{:x}", manifest.salt);
                        println!("  Buckets: {}", manifest.bucket_names.len());
                        println!("  Shards: {}", manifest.shards.len());
                        println!("  Total minimizers: {}", manifest.total_minimizers);

                        println!("  ------------------------------------------------");
                        println!("  Shard distribution:");
                        for shard in &manifest.shards {
                            println!("    Shard {}: {} buckets, {} minimizers, {} bytes",
                                shard.shard_id, shard.bucket_ids.len(), shard.num_minimizers, shard.compressed_size);
                        }

                        manifest.to_metadata()
                    } else {
                        // Single-file main index
                        let metadata = Index::load_metadata(&index)?;
                        println!("Index Stats for {:?}", index);
                        println!("  K: {}", metadata.k);
                        println!("  Window (w): {}", metadata.w);
                        println!("  Salt: 0x{:x}", metadata.salt);
                        println!("  Buckets: {}", metadata.bucket_names.len());
                        metadata
                    };

                    // Check if inverted index exists
                    let inverted_path = index.with_extension("ryxdi");
                    if inverted_path.exists() {
                        println!("  Inverted index: {:?} (use -I to show stats)", inverted_path);
                    }

                    println!("------------------------------------------------");
                    let mut sorted_ids: Vec<_> = metadata.bucket_names.keys().collect();
                    sorted_ids.sort();
                    for id in sorted_ids {
                        let name = metadata.bucket_names.get(id).map(|s| s.as_str()).unwrap_or("unknown");
                        let count = metadata.bucket_minimizer_counts.get(id).copied().unwrap_or(0);
                        let sources = metadata.bucket_sources.get(id).map(|v| v.len()).unwrap_or(0);
                        println!("  Bucket {}: '{}' ({} minimizers, {} sources)", id, name, count, sources);
                    }
                }
            }

            IndexCommands::BucketSourceDetail { index, bucket, paths, ids } => {
                let metadata = Index::load_metadata(&index)?;
                let sources = metadata.bucket_sources.get(&bucket).unwrap();

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
                    let mut sorted_ids: Vec<_> = metadata.bucket_names.keys().collect();
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

            IndexCommands::BucketAdd { index, reference } => {
                let main_manifest_path = MainIndexManifest::manifest_path(&index);

                if main_manifest_path.exists() {
                    // Sharded main index - add as new shard
                    let mut sharded = ShardedMainIndex::open(&index)?;
                    let next_id = sharded.next_id()?;
                    log::info!("Adding {:?} as new bucket ID {} (sharded)", reference, next_id);

                    // Extract minimizers from reference file
                    let mut reader = parse_fastx_file(&reference).context("Failed to open reference file")?;
                    let mut ws = MinimizerWorkspace::new();
                    let filename = reference.canonicalize().unwrap().to_string_lossy().to_string();
                    let mut sources = Vec::new();
                    let mut all_minimizers = Vec::new();

                    while let Some(record) = reader.next() {
                        let rec = record.context("Invalid record")?;
                        let seq = rec.seq();
                        let name = String::from_utf8_lossy(rec.id()).to_string();
                        let source_label = format!("{}{}{}", filename, Index::BUCKET_SOURCE_DELIM, name);
                        sources.push(source_label);

                        extract_into(&seq, sharded.k(), sharded.w(), sharded.salt(), &mut ws);
                        all_minimizers.extend_from_slice(&ws.buffer);
                    }

                    // Sort and deduplicate
                    all_minimizers.sort_unstable();
                    all_minimizers.dedup();
                    sources.sort_unstable();
                    sources.dedup();

                    let minimizer_count = all_minimizers.len();
                    sharded.add_bucket(next_id, &sanitize_bucket_name(&filename), sources, all_minimizers)?;

                    log::info!("Done. Added {} minimizers to bucket {} (new shard {}).",
                        minimizer_count, next_id, sharded.num_shards() - 1);
                } else {
                    // Single-file main index
                    let mut idx = Index::load(&index)?;
                    let next_id = idx.next_id()?;
                    log::info!("Adding {:?} as new bucket ID {}", reference, next_id);

                    let mut reader = parse_fastx_file(&reference).context("Failed to open reference file")?;
                    let mut ws = MinimizerWorkspace::new();
                    let filename = reference.canonicalize().unwrap().to_string_lossy().to_string();

                    idx.bucket_names.insert(next_id, sanitize_bucket_name(&filename));

                    while let Some(record) = reader.next() {
                        let rec = record.context("Invalid record")?;
                        let seq = rec.seq();
                        let name = String::from_utf8_lossy(rec.id()).to_string();
                        let source_label = format!("{}{}{}", filename, Index::BUCKET_SOURCE_DELIM, name);
                        idx.add_record(next_id, &source_label, &seq, &mut ws);
                    }

                    idx.finalize_bucket(next_id);
                    idx.save(&index)?;
                    log::info!("Done. Added {} minimizers to bucket {}.",
                             idx.buckets.get(&next_id).map(|v| v.len()).unwrap_or(0), next_id);
                }
            }

            IndexCommands::BucketMerge { index, src, dest } => {
                let main_manifest_path = MainIndexManifest::manifest_path(&index);

                if main_manifest_path.exists() {
                    // Sharded main index
                    let mut sharded = ShardedMainIndex::open(&index)?;
                    log::info!("Merging Bucket {} -> Bucket {} (sharded)...", src, dest);
                    sharded.merge_buckets(src, dest)?;
                    log::info!("Done.");
                } else {
                    // Single-file main index
                    let mut idx = Index::load(&index)?;
                    log::info!("Merging Bucket {} -> Bucket {}...", src, dest);
                    idx.merge_buckets(src, dest)?;
                    idx.save(&index)?;
                    log::info!("Done.");
                }
            }

            IndexCommands::Merge { output, inputs } => {
                // Logic: Load first index, then merge others into it.
                // Warning: Salt/W must match.
                if inputs.is_empty() { return Err(anyhow!("No input indexes provided")); }

                log::info!("Loading base index: {:?}", inputs[0]);
                let mut base_idx = Index::load(&inputs[0])?;

                for path in &inputs[1..] {
                    log::info!("Merging index: {:?}", path);
                    let other_idx = Index::load(path)?;

                    if other_idx.k != base_idx.k || other_idx.w != base_idx.w || other_idx.salt != base_idx.salt {
                        return Err(anyhow!(
                            "Index parameters mismatch: expected K={}, W={}, Salt=0x{:x}, got K={}, W={}, Salt=0x{:x}",
                            base_idx.k, base_idx.w, base_idx.salt,
                            other_idx.k, other_idx.w, other_idx.salt
                        ));
                    }

                    // Naive merge strategy: Re-map IDs of 'other' to not collide, then insert
                    // Simple version: just append buckets with new IDs
                    for (old_id, vec) in other_idx.buckets {
                        let new_id = base_idx.next_id()?;
                        base_idx.buckets.insert(new_id, vec);

                        if let Some(name) = other_idx.bucket_names.get(&old_id) {
                            base_idx.bucket_names.insert(new_id, sanitize_bucket_name(name));
                        }
                        if let Some(srcs) = other_idx.bucket_sources.get(&old_id) {
                            base_idx.bucket_sources.insert(new_id, srcs.clone());
                        }
                    }
                }
                base_idx.save(&output)?;
                log::info!("Merged index saved to {:?}", output);
            }

            IndexCommands::FromConfig { config } => {
                build_index_from_config(&config)?;
            }

            IndexCommands::Invert { index, shards } => {
                let output_path = index.with_extension("ryxdi");

                // Check if main index is sharded
                if MainIndexManifest::is_sharded(&index) {
                    log::info!("Detected sharded main index, creating 1:1 inverted shards");
                    let main_manifest = MainIndexManifest::load(&MainIndexManifest::manifest_path(&index))?;
                    log::info!("Main index has {} shards, {} buckets",
                        main_manifest.shards.len(), main_manifest.bucket_names.len());

                    // Track files we create for cleanup on error
                    let mut created_files: Vec<PathBuf> = Vec::new();

                    let result = (|| -> Result<ShardManifest> {
                        let mut inv_shards = Vec::new();
                        let mut total_minimizers = 0usize;
                        let mut total_bucket_ids = 0usize;
                        let num_shards = main_manifest.shards.len();

                        for (idx, shard_info) in main_manifest.shards.iter().enumerate() {
                            let shard_path = MainIndexManifest::shard_path(&index, shard_info.shard_id);
                            log::info!("Processing main shard {}: {} buckets, {} minimizers (raw)",
                                shard_info.shard_id, shard_info.bucket_ids.len(), shard_info.num_minimizers);

                            // Build inverted index from shard, dropping main shard immediately after
                            let inverted = {
                                let main_shard = MainIndexShard::load(&shard_path)?;
                                InvertedIndex::build_from_shard(&main_shard)
                            };

                            log::info!("  Built inverted: {} unique minimizers, {} bucket entries",
                                inverted.num_minimizers(), inverted.num_bucket_entries());

                            // Save as inverted shard with same ID
                            let inv_shard_path = ShardManifest::shard_path(&output_path, shard_info.shard_id);
                            let is_last = idx == num_shards - 1;
                            let inv_shard_info = inverted.save_shard(
                                &inv_shard_path,
                                shard_info.shard_id,
                                0,
                                inverted.num_minimizers(),
                                is_last,
                            )?;
                            created_files.push(inv_shard_path);

                            total_minimizers += inv_shard_info.num_minimizers;
                            total_bucket_ids += inv_shard_info.num_bucket_ids;
                            inv_shards.push(inv_shard_info);
                        }

                        // Create inverted manifest
                        let inv_manifest = ShardManifest {
                            k: main_manifest.k,
                            w: main_manifest.w,
                            salt: main_manifest.salt,
                            source_hash: InvertedIndex::compute_metadata_hash(&main_manifest.to_metadata()),
                            total_minimizers,
                            total_bucket_ids,
                            shards: inv_shards,
                        };

                        let manifest_path = ShardManifest::manifest_path(&output_path);
                        inv_manifest.save(&manifest_path)?;
                        created_files.push(manifest_path);

                        Ok(inv_manifest)
                    })();

                    match result {
                        Ok(inv_manifest) => {
                            log::info!("Created {} inverted shards with 1:1 correspondence:", inv_manifest.shards.len());
                            for shard in &inv_manifest.shards {
                                log::info!("  Shard {}: {} unique minimizers, {} bucket entries",
                                    shard.shard_id, shard.num_minimizers, shard.num_bucket_ids);
                            }
                        }
                        Err(e) => {
                            // Clean up partial files on error
                            for path in &created_files {
                                if path.exists() {
                                    let _ = std::fs::remove_file(path);
                                }
                            }
                            return Err(e);
                        }
                    }
                } else {
                    // Non-sharded main index
                    log::info!("Loading index from {:?}", index);
                    let idx = Index::load(&index)?;
                    log::info!("Index loaded: {} buckets", idx.buckets.len());

                    log::info!("Building inverted index...");
                    let inverted = InvertedIndex::build_from_index(&idx);
                    log::info!("Inverted index built: {} unique minimizers, {} bucket entries",
                        inverted.num_minimizers(), inverted.num_bucket_entries());

                    if shards > 1 {
                        log::info!("Saving sharded inverted index ({} shards) to {:?}", shards, output_path);
                        let manifest = inverted.save_sharded(&output_path, shards)?;
                        log::info!("Created {} shards:", manifest.shards.len());
                        for shard in &manifest.shards {
                            log::info!("  Shard {}: {} unique minimizers, {} bucket entries",
                                shard.shard_id, shard.num_minimizers, shard.num_bucket_ids);
                        }
                    } else {
                        log::info!("Saving inverted index to {:?}", output_path);
                        inverted.save(&output_path)?;
                    }
                }
                log::info!("Done.");
            }

            IndexCommands::Summarize { index } => {
                log::info!("Loading index from {:?}", index);
                let idx = Index::load(&index)?;

                // Basic info
                println!("=== Index Summary ===");
                println!("File: {:?}", index);
                println!("K: {}", idx.k);
                println!("Window (w): {}", idx.w);
                println!("Salt: 0x{:x}", idx.salt);
                println!("Buckets: {}", idx.buckets.len());

                // Collect all minimizers from all buckets (they're already sorted within buckets)
                let mut all_minimizers: Vec<u64> = Vec::new();
                let mut per_bucket_counts: Vec<(u32, usize)> = Vec::new();

                for (&id, mins) in &idx.buckets {
                    per_bucket_counts.push((id, mins.len()));
                    all_minimizers.extend(mins.iter().copied());
                }

                let total_minimizers = all_minimizers.len();
                println!("Total minimizers (with duplicates across buckets): {}", total_minimizers);

                if total_minimizers == 0 {
                    println!("No minimizers to analyze.");
                    return Ok(());
                }

                // Sort all minimizers for global analysis
                all_minimizers.sort_unstable();

                // Deduplicate to get unique minimizers
                all_minimizers.dedup();
                let unique_minimizers = all_minimizers.len();
                println!("Unique minimizers: {}", unique_minimizers);
                println!("Duplication ratio: {:.2}x", total_minimizers as f64 / unique_minimizers as f64);

                // Value range
                let min_val = *all_minimizers.first().unwrap();
                let max_val = *all_minimizers.last().unwrap();
                println!("\n=== Minimizer Value Statistics ===");
                println!("Min value: {}", min_val);
                println!("Max value: {}", max_val);
                println!("Value range: {}", max_val - min_val);

                // Bits needed for raw values
                let bits_for_max = if max_val == 0 { 1 } else { 64 - max_val.leading_zeros() };
                println!("Bits needed for max value: {}", bits_for_max);

                // Compute deltas
                if unique_minimizers > 1 {
                    println!("\n=== Delta Statistics (for compression analysis) ===");

                    let mut deltas: Vec<u64> = Vec::with_capacity(unique_minimizers - 1);
                    for i in 1..unique_minimizers {
                        deltas.push(all_minimizers[i] - all_minimizers[i - 1]);
                    }

                    let min_delta = *deltas.iter().min().unwrap();
                    let max_delta = *deltas.iter().max().unwrap();
                    let sum_delta: u128 = deltas.iter().map(|&d| d as u128).sum();
                    let mean_delta = sum_delta as f64 / deltas.len() as f64;

                    // Median delta
                    let mut sorted_deltas = deltas.clone();
                    sorted_deltas.sort_unstable();
                    let median_delta = sorted_deltas[sorted_deltas.len() / 2];

                    println!("Min delta: {}", min_delta);
                    println!("Max delta: {}", max_delta);
                    println!("Mean delta: {:.2}", mean_delta);
                    println!("Median delta: {}", median_delta);

                    // Bits needed for deltas
                    let bits_for_max_delta = if max_delta == 0 { 1 } else { 64 - max_delta.leading_zeros() };
                    println!("Bits needed for max delta: {}", bits_for_max_delta);

                    // Distribution of bits needed per delta
                    let mut bit_distribution = [0usize; 65]; // 0-64 bits
                    for &d in &deltas {
                        let bits = if d == 0 { 1 } else { 64 - d.leading_zeros() } as usize;
                        bit_distribution[bits] += 1;
                    }

                    println!("\nDelta bit-width distribution:");
                    let mut cumulative = 0usize;
                    for bits in 1..=64 {
                        if bit_distribution[bits] > 0 {
                            cumulative += bit_distribution[bits];
                            let pct = 100.0 * cumulative as f64 / deltas.len() as f64;
                            println!("  <= {} bits: {} deltas ({:.1}% cumulative)",
                                bits, bit_distribution[bits], pct);
                        }
                    }

                    // Estimate compression potential
                    println!("\n=== Compression Estimates ===");
                    let raw_bytes = unique_minimizers * 8;
                    println!("Raw storage (8 bytes/minimizer): {} bytes ({:.2} GB)",
                        raw_bytes, raw_bytes as f64 / (1024.0 * 1024.0 * 1024.0));

                    // Estimate varint-encoded delta size
                    // Varint uses 1 byte per 7 bits, roughly
                    let mut estimated_varint_bytes: usize = 8; // First value stored raw
                    for &d in &deltas {
                        let bits = if d == 0 { 1 } else { 64 - d.leading_zeros() } as usize;
                        estimated_varint_bytes += (bits + 6) / 7; // ceil(bits/7)
                    }
                    let varint_ratio = estimated_varint_bytes as f64 / raw_bytes as f64;
                    println!("Estimated delta+varint: {} bytes ({:.1}% of raw)",
                        estimated_varint_bytes, varint_ratio * 100.0);

                    // With zstd on top (rough estimate: 50-70% of varint size for sorted data)
                    let estimated_zstd = (estimated_varint_bytes as f64 * 0.6) as usize;
                    let zstd_ratio = estimated_zstd as f64 / raw_bytes as f64;
                    println!("Estimated delta+varint+zstd: ~{} bytes (~{:.1}% of raw)",
                        estimated_zstd, zstd_ratio * 100.0);
                }

                // Per-bucket summary
                println!("\n=== Per-Bucket Statistics ===");
                per_bucket_counts.sort_by_key(|(id, _)| *id);
                let total_in_buckets: usize = per_bucket_counts.iter().map(|(_, c)| c).sum();
                println!("Total minimizers across all buckets: {}", total_in_buckets);

                if per_bucket_counts.len() <= 20 {
                    for (id, count) in &per_bucket_counts {
                        let name = idx.bucket_names.get(id).map(|s| s.as_str()).unwrap_or("unknown");
                        println!("  Bucket {}: {} minimizers ({})", id, count, name);
                    }
                } else {
                    println!("  (showing first 10 and last 10 of {} buckets)", per_bucket_counts.len());
                    for (id, count) in per_bucket_counts.iter().take(10) {
                        let name = idx.bucket_names.get(id).map(|s| s.as_str()).unwrap_or("unknown");
                        println!("  Bucket {}: {} minimizers ({})", id, count, name);
                    }
                    println!("  ...");
                    for (id, count) in per_bucket_counts.iter().rev().take(10).collect::<Vec<_>>().into_iter().rev() {
                        let name = idx.bucket_names.get(id).map(|s| s.as_str()).unwrap_or("unknown");
                        println!("  Bucket {}: {} minimizers ({})", id, count, name);
                    }
                }
            }
        },

        Commands::Classify(classify_cmd) => match classify_cmd {
            ClassifyCommands::Run { index, r1, r2, threshold, batch_size, output, use_inverted, merge_join } |
            ClassifyCommands::Batch { index, r1, r2, threshold, batch_size, output, use_inverted, merge_join } => {
                if merge_join && !use_inverted {
                    return Err(anyhow!("--merge-join requires --use-inverted"));
                }
                if use_inverted {
                    // Use inverted index path - detect sharded vs single-file format
                    let inverted_path = index.with_extension("ryxdi");
                    let manifest_path = ShardManifest::manifest_path(&inverted_path);

                    // Load index metadata first (needed for both paths)
                    log::info!("Loading index metadata from {:?}", index);
                    let metadata = Index::load_metadata(&index)?;
                    log::info!("Metadata loaded: {} buckets", metadata.bucket_names.len());

                    let mut io = IoHandler::new(&r1, r2.as_ref(), output)?;
                    io.write(b"read_id\tbucket_name\tscore\n".to_vec())?;

                    let mut total_reads = 0;
                    let mut batch_num = 0;

                    if manifest_path.exists() {
                        // Sharded inverted index - use sequential loading to minimize memory
                        log::info!("Loading sharded inverted index manifest from {:?}", inverted_path);
                        let sharded = ShardedInvertedIndex::open(&inverted_path)?;
                        log::info!("Sharded index: {} shards, {} total minimizers",
                            sharded.num_shards(), sharded.total_minimizers());

                        sharded.validate_against_metadata(&metadata)?;
                        log::info!("Sharded index validated successfully");

                        if merge_join {
                            log::info!("Starting merge-join classification with sequential shard loading (batch_size={})", batch_size);
                        } else {
                            log::info!("Starting classification with sequential shard loading (batch_size={})", batch_size);
                        }

                        while let Some((owned_records, headers)) = io.next_batch_records(batch_size)? {
                            batch_num += 1;
                            let batch_read_count = owned_records.len();
                            total_reads += batch_read_count;

                            let batch_refs: Vec<QueryRecord> = owned_records.iter()
                                .map(|(id, s1, s2)| (*id, s1.as_slice(), s2.as_deref()))
                                .collect();

                            let results = if merge_join {
                                classify_batch_sharded_merge_join(&sharded, &batch_refs, threshold)?
                            } else {
                                classify_batch_sharded_sequential(&sharded, &batch_refs, threshold)?
                            };

                            let mut chunk_out = Vec::with_capacity(1024);
                            for res in results {
                                let header = &headers[res.query_id as usize];
                                let bucket_name = metadata.bucket_names.get(&res.bucket_id)
                                    .map(|s| s.as_str())
                                    .unwrap_or("unknown");
                                writeln!(chunk_out, "{}\t{}\t{:.4}", header, bucket_name, res.score).unwrap();
                            }
                            io.write(chunk_out)?;

                            log::info!("Processed batch {}: {} reads ({} total)", batch_num, batch_read_count, total_reads);
                        }
                    } else if inverted_path.exists() {
                        // Single-file inverted index
                        log::info!("Loading inverted index from {:?}", inverted_path);
                        let inverted = InvertedIndex::load(&inverted_path)?;
                        log::info!("Inverted index loaded: {} unique minimizers", inverted.num_minimizers());

                        inverted.validate_against_metadata(&metadata)?;
                        log::info!("Inverted index validated successfully");

                        if merge_join {
                            log::info!("Starting merge-join classification with inverted index (batch_size={})", batch_size);
                        } else {
                            log::info!("Starting classification with inverted index (batch_size={})", batch_size);
                        }

                        while let Some((owned_records, headers)) = io.next_batch_records(batch_size)? {
                            batch_num += 1;
                            let batch_read_count = owned_records.len();
                            total_reads += batch_read_count;

                            let batch_refs: Vec<QueryRecord> = owned_records.iter()
                                .map(|(id, s1, s2)| (*id, s1.as_slice(), s2.as_deref()))
                                .collect();

                            let results = if merge_join {
                                classify_batch_with_query_index(&inverted, &batch_refs, threshold)
                            } else {
                                classify_batch_inverted(&inverted, &batch_refs, threshold)
                            };

                            let mut chunk_out = Vec::with_capacity(1024);
                            for res in results {
                                let header = &headers[res.query_id as usize];
                                let bucket_name = metadata.bucket_names.get(&res.bucket_id)
                                    .map(|s| s.as_str())
                                    .unwrap_or("unknown");
                                writeln!(chunk_out, "{}\t{}\t{:.4}", header, bucket_name, res.score).unwrap();
                            }
                            io.write(chunk_out)?;

                            log::info!("Processed batch {}: {} reads ({} total)", batch_num, batch_read_count, total_reads);
                        }
                    } else {
                        return Err(anyhow!(
                            "Inverted index not found: {:?}. Create it with 'rype index invert -i {:?}'",
                            inverted_path, index
                        ));
                    }

                    log::info!("Classification complete: {} reads processed", total_reads);
                    io.finish()?;
                } else {
                    // Standard path - detect sharded vs single-file main index
                    let main_manifest_path = MainIndexManifest::manifest_path(&index);

                    let mut io = IoHandler::new(&r1, r2.as_ref(), output)?;
                    io.write(b"read_id\tbucket_name\tscore\n".to_vec())?;

                    let mut total_reads = 0;
                    let mut batch_num = 0;

                    if main_manifest_path.exists() {
                        // Sharded main index - use sequential shard loading
                        log::info!("Loading sharded main index from {:?}", index);
                        let sharded = ShardedMainIndex::open(&index)?;
                        log::info!("Sharded main index: {} shards, {} buckets, {} total minimizers",
                            sharded.num_shards(), sharded.manifest().bucket_names.len(), sharded.total_minimizers());

                        let metadata = sharded.metadata();

                        log::info!("Starting classification with sequential main shard loading (batch_size={})", batch_size);

                        while let Some((owned_records, headers)) = io.next_batch_records(batch_size)? {
                            batch_num += 1;
                            let batch_read_count = owned_records.len();
                            total_reads += batch_read_count;

                            let batch_refs: Vec<QueryRecord> = owned_records.iter()
                                .map(|(id, s1, s2)| (*id, s1.as_slice(), s2.as_deref()))
                                .collect();

                            let results = classify_batch_sharded_main(&sharded, &batch_refs, threshold)?;

                            let mut chunk_out = Vec::with_capacity(1024);
                            for res in results {
                                let header = &headers[res.query_id as usize];
                                let bucket_name = metadata.bucket_names.get(&res.bucket_id)
                                    .map(|s| s.as_str())
                                    .unwrap_or("unknown");
                                writeln!(chunk_out, "{}\t{}\t{:.4}", header, bucket_name, res.score).unwrap();
                            }
                            io.write(chunk_out)?;

                            log::info!("Processed batch {}: {} reads ({} total)", batch_num, batch_read_count, total_reads);
                        }
                    } else {
                        // Single-file main index
                        log::info!("Loading index from {:?}", index);
                        let engine = Index::load(&index)?;
                        log::info!("Index loaded: {} buckets", engine.buckets.len());

                        log::info!("Starting classification (batch_size={})", batch_size);

                        while let Some((owned_records, headers)) = io.next_batch_records(batch_size)? {
                            batch_num += 1;
                            let batch_read_count = owned_records.len();
                            total_reads += batch_read_count;

                            let batch_refs: Vec<QueryRecord> = owned_records.iter()
                                .map(|(id, s1, s2)| (*id, s1.as_slice(), s2.as_deref()))
                                .collect();

                            let results = classify_batch(&engine, &batch_refs, threshold);

                            let mut chunk_out = Vec::with_capacity(1024);
                            for res in results {
                                let header = &headers[res.query_id as usize];
                                let bucket_name = engine.bucket_names.get(&res.bucket_id)
                                    .map(|s| s.as_str())
                                    .unwrap_or("unknown");
                                writeln!(chunk_out, "{}\t{}\t{:.4}", header, bucket_name, res.score).unwrap();
                            }
                            io.write(chunk_out)?;

                            log::info!("Processed batch {}: {} reads ({} total)", batch_num, batch_read_count, total_reads);
                        }
                    }

                    log::info!("Classification complete: {} reads processed", total_reads);
                    io.finish()?;
                }
            }

            ClassifyCommands::Aggregate { index, r1, r2, threshold, batch_size, output } => {
                log::info!("Loading index from {:?}", index);
                let engine = Index::load(&index)?;
                log::info!("Index loaded: {} buckets", engine.buckets.len());

                let mut io = IoHandler::new(&r1, r2.as_ref(), output)?;
                io.write(b"query_name\tbucket_name\tscore\n".to_vec())?;

                let mut total_reads = 0;
                let mut batch_num = 0;

                log::info!("Starting aggregate classification (batch_size={})", batch_size);

                while let Some((owned_records, _)) = io.next_batch_records(batch_size)? {
                    batch_num += 1;
                    let batch_read_count = owned_records.len();
                    total_reads += batch_read_count;

                    let batch_refs: Vec<QueryRecord> = owned_records.iter()
                        .map(|(id, s1, s2)| (*id, s1.as_slice(), s2.as_deref()))
                        .collect();

                    let results = aggregate_batch(&engine, &batch_refs, threshold);

                    let mut chunk_out = Vec::with_capacity(1024);
                    for res in results {
                        let bucket_name = engine.bucket_names.get(&res.bucket_id)
                            .map(|s| s.as_str())
                            .unwrap_or("unknown");
                        writeln!(chunk_out, "global\t{}\t{:.4}", bucket_name, res.score).unwrap();
                    }
                    io.write(chunk_out)?;

                    log::info!("Processed batch {}: {} reads ({} total)", batch_num, batch_read_count, total_reads);
                }

                log::info!("Aggregate classification complete: {} reads processed", total_reads);
                io.finish()?;
            }
        },

        Commands::Inspect(inspect_cmd) => match inspect_cmd {
            InspectCommands::Matches { index, queries, ids, buckets } => {
                inspect_matches(&index, &queries, &ids, &buckets)?;
            }
        },
    }

    Ok(())
}

fn build_index_from_config(config_path: &Path) -> Result<()> {
    log::info!("Building index from config: {}", config_path.display());

    // 1. Parse and validate config
    let cfg = parse_config(config_path)?;
    let config_dir = config_path.parent()
        .ok_or_else(|| anyhow!("Invalid config path"))?;

    log::info!("Validating file paths...");
    validate_config(&cfg, config_dir)?;
    log::info!("Validation successful.");

    // 2. Sort bucket names for deterministic ordering
    let mut bucket_names: Vec<_> = cfg.buckets.keys().cloned().collect();
    bucket_names.sort();

    log::info!("Building {} buckets in parallel...", bucket_names.len());
    for name in &bucket_names {
        let file_count = cfg.buckets[name].files.len();
        log::info!("  - {}: {} file{}", name, file_count, if file_count == 1 { "" } else { "s" });
    }

    // 3. Build indices in parallel (one per bucket)
    let bucket_indices: Vec<_> = bucket_names.par_iter()
        .map(|bucket_name| {
            log::info!("Processing bucket '{}'...", bucket_name);
            let mut idx = Index::new(cfg.index.k, cfg.index.window, cfg.index.salt)?;
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
            let minimizer_count = idx.buckets.get(&1).map(|v| v.len()).unwrap_or(0);
            log::info!("Completed bucket '{}': {} minimizers", bucket_name, minimizer_count);
            Ok::<_, anyhow::Error>((bucket_name.clone(), idx))
        })
        .collect::<Result<Vec<_>>>()?;

    log::info!("Processing complete. Merging buckets...");

    // 4. Merge all bucket indices into final index
    let mut final_index = Index::new(cfg.index.k, cfg.index.window, cfg.index.salt)?;
    for (bucket_id, (bucket_name, bucket_idx)) in bucket_indices.into_iter().enumerate() {
        let new_id = (bucket_id + 1) as u32;
        final_index.bucket_names.insert(new_id, sanitize_bucket_name(&bucket_name));

        // Transfer bucket data
        if let Some(minimizers) = bucket_idx.buckets.get(&1) {
            final_index.buckets.insert(new_id, minimizers.clone());
        }
        if let Some(sources) = bucket_idx.bucket_sources.get(&1) {
            final_index.bucket_sources.insert(new_id, sources.clone());
        }
    }

    // 5. Save final index
    log::info!("Saving index to {}...", cfg.index.output.display());
    final_index.save(&cfg.index.output)?;

    log::info!("Done! Index saved successfully.");
    log::info!("\nFinal statistics:");
    log::info!("  Buckets: {}", final_index.buckets.len());
    let total_minimizers: usize = final_index.buckets.values().map(|v| v.len()).sum();
    log::info!("  Total minimizers: {}", total_minimizers);

    Ok(())
}

// --- INSPECT COMMAND HELPERS ---

/// A match found in a reference sequence
#[derive(Debug, Clone)]
struct ReferenceMatch {
    file_path: String,
    seq_id: String,
    position: usize,
    strand: Strand,
    kmer: String,
}

/// Extract the k-mer nucleotide string at a given position.
/// For reverse complement strand, returns the RC of the k-mer.
fn extract_kmer_string(seq: &[u8], pos: usize, k: usize, strand: Strand) -> String {
    if pos + k > seq.len() {
        return "<out-of-bounds>".to_string();
    }
    let kmer_bytes = &seq[pos..pos + k];

    match strand {
        Strand::Forward => String::from_utf8_lossy(kmer_bytes).to_string(),
        Strand::ReverseComplement => {
            kmer_bytes.iter().rev().map(|&b| match b {
                b'A' | b'a' => 'T',
                b'T' | b't' => 'A',
                b'G' | b'g' => 'C',
                b'C' | b'c' => 'G',
                other => other as char,
            }).collect()
        }
    }
}

/// Build a map of minimizer hash → all reference locations for a bucket
fn build_reference_minimizer_map(
    index: &Index,
    bucket_id: u32,
) -> Result<HashMap<u64, Vec<ReferenceMatch>>> {
    let mut map: HashMap<u64, Vec<ReferenceMatch>> = HashMap::new();
    let mut ws = MinimizerWorkspace::new();

    // Get source info for this bucket (format: "filepath::seqname")
    let sources = index.bucket_sources.get(&bucket_id)
        .ok_or_else(|| anyhow!("Bucket {} has no sources", bucket_id))?;

    // Group sources by file path
    let mut files_to_seqs: HashMap<String, HashSet<String>> = HashMap::new();
    for source in sources {
        let parts: Vec<&str> = source.split(Index::BUCKET_SOURCE_DELIM).collect();
        if parts.len() >= 2 {
            let file_path = parts[0].to_string();
            let seq_id = parts[1..].join(Index::BUCKET_SOURCE_DELIM);
            files_to_seqs.entry(file_path).or_default().insert(seq_id);
        }
    }

    // Scan each reference file
    for (file_path, target_seqs) in &files_to_seqs {
        let path = Path::new(file_path);
        if !path.exists() {
            log::warn!("Reference file not found: {}", file_path);
            continue;
        }

        let mut reader = match parse_fastx_file(path) {
            Ok(r) => r,
            Err(e) => {
                log::warn!("Failed to open reference file {}: {}", file_path, e);
                continue;
            }
        };

        while let Some(record) = reader.next() {
            let rec = match record {
                Ok(r) => r,
                Err(e) => {
                    log::warn!("Error reading record from {}: {}", file_path, e);
                    continue;
                }
            };
            let seq_id = String::from_utf8_lossy(rec.id()).to_string();

            // Only process sequences that are in this bucket's sources
            if !target_seqs.contains(&seq_id) {
                continue;
            }

            let seq = rec.seq();
            let minimizers = extract_with_positions(&seq, index.k, index.w, index.salt, &mut ws);

            for m in minimizers {
                let kmer = extract_kmer_string(&seq, m.position, index.k, m.strand);
                map.entry(m.hash).or_default().push(ReferenceMatch {
                    file_path: file_path.clone(),
                    seq_id: seq_id.clone(),
                    position: m.position,
                    strand: m.strand,
                    kmer,
                });
            }
        }
    }

    Ok(map)
}

/// Main inspect matches function
fn inspect_matches(
    index_path: &Path,
    queries_path: &Path,
    ids_file: &Path,
    bucket_filter: &[u32],
) -> Result<()> {
    // 1. Load the index
    log::info!("Loading index from {:?}", index_path);
    let index = Index::load(index_path)?;
    log::info!("Index loaded: {} buckets, K={}, W={}", index.buckets.len(), index.k, index.w);

    // 2. Load sequence IDs to inspect
    log::info!("Loading sequence IDs from {:?}", ids_file);
    let target_ids: HashSet<String> = std::io::BufReader::new(File::open(ids_file)?)
        .lines()
        .filter_map(|l| l.ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    log::info!("Loaded {} sequence IDs to inspect", target_ids.len());

    // 3. Validate bucket IDs exist
    for &bucket_id in bucket_filter {
        if !index.buckets.contains_key(&bucket_id) {
            return Err(anyhow!("Bucket {} does not exist in index", bucket_id));
        }
    }

    // 4. Build reference minimizer maps for each bucket
    log::info!("Building reference minimizer maps for {} buckets...", bucket_filter.len());
    let mut ref_maps: HashMap<u32, HashMap<u64, Vec<ReferenceMatch>>> = HashMap::new();
    for &bucket_id in bucket_filter {
        let bucket_name = index.bucket_names.get(&bucket_id)
            .map(|s| s.as_str())
            .unwrap_or("unknown");
        log::info!("  Scanning references for bucket {} ({})...", bucket_id, bucket_name);
        ref_maps.insert(bucket_id, build_reference_minimizer_map(&index, bucket_id)?);
    }
    log::info!("Reference maps built.");

    // 5. Process query sequences
    log::info!("Processing query sequences from {:?}", queries_path);
    let mut reader = parse_fastx_file(queries_path).context("Failed to open query file")?;
    let mut ws = MinimizerWorkspace::new();
    let mut queries_processed = 0;
    let mut queries_with_matches = 0;

    while let Some(record) = reader.next() {
        let rec = record.context("Invalid query record")?;
        let id = String::from_utf8_lossy(rec.id()).to_string();

        if !target_ids.contains(&id) {
            continue;
        }

        queries_processed += 1;
        let seq = rec.seq();
        let minimizers = extract_with_positions(&seq, index.k, index.w, index.salt, &mut ws);

        // Find matches
        let mut has_output = false;
        for m in &minimizers {
            // Check which buckets contain this minimizer
            let mut bucket_matches: Vec<(u32, &str, &[ReferenceMatch])> = vec![];

            for &bucket_id in bucket_filter {
                if let Some(bucket) = index.buckets.get(&bucket_id) {
                    if bucket.binary_search(&m.hash).is_ok() {
                        let name = index.bucket_names.get(&bucket_id)
                            .map(|s| s.as_str())
                            .unwrap_or("unknown");
                        let ref_matches = ref_maps.get(&bucket_id)
                            .and_then(|map| map.get(&m.hash))
                            .map(|v| v.as_slice())
                            .unwrap_or(&[]);
                        bucket_matches.push((bucket_id, name, ref_matches));
                    }
                }
            }

            if !bucket_matches.is_empty() {
                if !has_output {
                    println!(">{}", id);
                    has_output = true;
                    queries_with_matches += 1;
                }

                let query_kmer = extract_kmer_string(&seq, m.position, index.k, m.strand);
                let strand_char = if m.strand == Strand::Forward { '+' } else { '-' };

                println!("  position: {}  strand: {}  kmer: {}  minimizer: 0x{:016X}",
                    m.position, strand_char, query_kmer, m.hash);

                for (bucket_id, bucket_name, ref_matches) in bucket_matches {
                    println!("    bucket: {} (id={})", bucket_name, bucket_id);

                    if ref_matches.is_empty() {
                        println!("      (no reference positions found - file may be missing)");
                        continue;
                    }

                    // Group reference matches by file path, then by seq_id
                    let mut by_file: HashMap<&str, HashMap<&str, Vec<&ReferenceMatch>>> = HashMap::new();
                    for rm in ref_matches {
                        by_file.entry(&rm.file_path)
                            .or_default()
                            .entry(&rm.seq_id)
                            .or_default()
                            .push(rm);
                    }

                    // Output grouped by file, then by sequence
                    let mut file_paths: Vec<_> = by_file.keys().collect();
                    file_paths.sort();
                    for file_path in file_paths {
                        println!("      file: {}", file_path);
                        let seqs = &by_file[file_path];
                        let mut seq_ids: Vec<_> = seqs.keys().collect();
                        seq_ids.sort();
                        for seq_id in seq_ids {
                            println!("        ref: {}", seq_id);
                            for rm in &seqs[seq_id] {
                                let ref_strand = if rm.strand == Strand::Forward { '+' } else { '-' };
                                println!("          pos: {}  strand: {}  kmer: {}",
                                    rm.position, ref_strand, rm.kmer);
                            }
                        }
                    }
                }
            }
        }
    }

    log::info!("Inspection complete: {} queries processed, {} had matches",
        queries_processed, queries_with_matches);
    Ok(())
}

