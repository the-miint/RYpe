use clap::{Parser, Subcommand};
use needletail::{parse_fastx_file, FastxReader};
use rayon::prelude::*;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::collections::HashSet;
use anyhow::{Context, Result, anyhow};

use rype::{Index, IndexMetadata, InvertedIndex, MinimizerWorkspace, QueryRecord, classify_batch, classify_batch_inverted, classify_batch_with_query_index, classify_batch_sharded_sequential, classify_batch_sharded_merge_join, classify_batch_sharded_main, aggregate_batch, ShardManifest, ShardedInvertedIndex, MainIndexManifest, MainIndexShard, ShardedMainIndex, extract_into, extract_with_positions, Strand};
use rype::config::{parse_config, validate_config, resolve_path, parse_bucket_add_config, validate_bucket_add_config, AssignmentSettings, BestBinFallback};
use rype::memory::{parse_byte_suffix, detect_available_memory, ReadMemoryProfile, MemoryConfig, calculate_batch_config, format_bytes, MemorySource};
use std::collections::HashMap;
use std::io::BufRead;

mod logging;

// --- HELPER FUNCTIONS ---

/// Parse a byte size string from CLI (e.g., "4G", "512M", "auto").
/// Returns 0 for "auto" (signals auto-detection), bytes otherwise.
fn parse_max_memory_arg(s: &str) -> Result<usize, String> {
    parse_byte_suffix(s)
        .map(|opt| opt.unwrap_or(0))  // None (auto) -> 0
        .map_err(|e| e.to_string())
}

/// Parse a byte size string from CLI, requiring a concrete value (no "auto").
fn parse_shard_size_arg(s: &str) -> Result<usize, String> {
    match parse_byte_suffix(s) {
        Ok(Some(bytes)) => Ok(bytes),
        Ok(None) => Err("'auto' not supported for shard size".to_string()),
        Err(e) => Err(e.to_string()),
    }
}

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

/// Load metadata from either a sharded or non-sharded main index.
///
/// This helper handles the case where the main index might be sharded (with .manifest
/// and .shard.* files) or a single-file index.
fn load_index_metadata(path: &Path) -> Result<IndexMetadata> {
    if MainIndexManifest::is_sharded(path) {
        let manifest = MainIndexManifest::load(&MainIndexManifest::manifest_path(path))?;
        Ok(manifest.to_metadata())
    } else {
        Index::load_metadata(path)
    }
}

#[derive(Parser)]
#[command(name = "rype")]
#[command(about = "High-performance Read Partitioning Engine (RY-Space, K=16/32/64)")]
#[command(long_about = "Rype: High-performance genomic sequence classification using minimizer-based k-mer sketching in RY (purine/pyrimidine) space.

WORKFLOW:
  1. Create an index:     rype index create -o index.ryidx -r refs.fasta
  2. (Optional) Invert:   rype index invert -i index.ryidx
  3. Classify reads:      rype classify run -i index.ryidx -1 reads.fq

INPUT FORMATS:
  FASTA (.fa, .fasta, .fna) and FASTQ (.fq, .fastq) files are supported.
  Gzip-compressed files (.gz) are automatically detected and decompressed.

OUTPUT FORMAT (classify):
  Tab-separated: read_id<TAB>bucket_name<TAB>score
  - read_id: Sequence header (first whitespace-delimited token)
  - bucket_name: Human-readable name from index
  - score: Fraction of query minimizers matching (0.0-1.0)")]
#[command(after_help = "EXAMPLES:
  # Create index from reference genomes
  rype index create -o bacteria.ryidx -r genome1.fna -r genome2.fna -k 64 -w 50

  # Create index with one bucket per sequence
  rype index create -o genes.ryidx -r genes.fasta --separate-buckets

  # Build inverted index for faster classification
  rype index invert -i bacteria.ryidx

  # Classify single-end reads
  rype classify run -i bacteria.ryidx -1 reads.fq -t 0.1 -o results.tsv

  # Classify paired-end reads with negative filtering
  rype classify run -i bacteria.ryidx -N host.ryidx -1 R1.fq -2 R2.fq -t 0.1

  # Use inverted index for faster classification
  rype classify run -i bacteria.ryidx -I -1 reads.fq

  # Aggregate mode for higher sensitivity
  rype classify aggregate -i bacteria.ryidx -1 R1.fq -2 R2.fq -t 0.05")]
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

    /// Inspect minimizer details and matches (debugging)
    #[command(subcommand)]
    Inspect(InspectCommands),
}

#[derive(Subcommand)]
enum IndexCommands {
    /// Create a new index from reference sequences
    #[command(after_help = "EXAMPLES:
  # Basic index creation
  rype index create -o index.ryidx -r genome.fasta

  # Multiple references, all in one bucket
  rype index create -o index.ryidx -r chr1.fa -r chr2.fa

  # One bucket per sequence (e.g., for gene-level classification)
  rype index create -o genes.ryidx -r genes.fasta --separate-buckets

  # Large index with sharding (for memory-constrained systems)
  rype index create -o large.ryidx -r refs.fa --max-shard-size 1073741824")]
    Create {
        /// Output index file path (.ryidx extension recommended)
        #[arg(short, long)]
        output: PathBuf,

        /// Reference sequence files (FASTA/FASTQ, optionally gzipped).
        /// Can specify multiple times: -r file1.fa -r file2.fa
        #[arg(short, long, required = true)]
        reference: Vec<PathBuf>,

        /// K-mer size for minimizer computation. Must be 16, 32, or 64.
        /// Larger k = more specific matches, fewer false positives.
        /// Smaller k = more sensitive, may find distant homologs.
        #[arg(short = 'k', long, default_value_t = 64)]
        kmer_size: usize,

        /// Minimizer window size. Larger values = smaller index, less sensitive.
        /// Recommended: 30-100 for genomes, 20-50 for shorter sequences.
        #[arg(short, long, default_value_t = 50)]
        window: usize,

        /// XOR salt for hash randomization. Must match for index compatibility.
        /// Default is fine for most uses; change to create incompatible indices.
        #[arg(short, long, default_value_t = 0x5555555555555555)]
        salt: u64,

        /// Create one bucket per input sequence instead of one per file.
        /// Use when each sequence represents a distinct classification target
        /// (e.g., individual genes, plasmids, or genomes in a multi-FASTA).
        #[arg(long)]
        separate_buckets: bool,

        /// Maximum shard size for large indices (e.g., "1G", "512M").
        /// Creates multiple shard files loaded on-demand during classification.
        #[arg(long, value_parser = parse_shard_size_arg)]
        max_shard_size: Option<usize>,
    },

    /// Show index statistics and bucket information
    Stats {
        /// Path to index file (.ryidx)
        #[arg(short, long)]
        index: PathBuf,

        /// Show inverted index stats instead of primary index
        #[arg(short = 'I', long)]
        inverted: bool,
    },

    /// Show source file paths or sequence IDs for a bucket
    BucketSourceDetail {
        /// Path to index file (.ryidx)
        #[arg(short, long)]
        index: PathBuf,

        /// Bucket ID to inspect (from 'rype index stats' output)
        #[arg(short, long, required = true)]
        bucket: u32,

        /// Show only unique file paths (one per line)
        #[arg(long)]
        paths: bool,

        /// Show only bucket IDs (for scripting)
        #[arg(long)]
        ids: bool,
    },

    /// Add a new reference file as a new bucket to an existing index
    BucketAdd {
        /// Path to existing index file
        #[arg(short, long)]
        index: PathBuf,

        /// Reference file to add (creates a new bucket)
        #[arg(short, long)]
        reference: PathBuf,
    },

    /// Merge two buckets within an index (source absorbed into destination)
    BucketMerge {
        /// Path to index file (modified in place)
        #[arg(short, long)]
        index: PathBuf,

        /// Source bucket ID (will be removed after merge)
        #[arg(long)]
        src: u32,

        /// Destination bucket ID (receives source's minimizers)
        #[arg(long)]
        dest: u32,
    },

    /// Merge multiple indices into one (buckets renumbered to avoid conflicts)
    Merge {
        /// Output path for merged index
        #[arg(short, long)]
        output: PathBuf,

        /// Input indices to merge (must have same k, w, salt)
        #[arg(short, long, required = true)]
        inputs: Vec<PathBuf>,
    },

    /// Build index from a TOML configuration file (see CONFIG FORMAT below)
    #[command(after_help = "CONFIG FORMAT (from-config):
  [index]
  k = 64                           # K-mer size (16, 32, or 64)
  window = 50                      # Minimizer window size
  salt = 0x5555555555555555        # Hash salt (hex)
  output = \"index.ryidx\"           # Output path
  max_shard_size = 1073741824      # Optional: shard main index (bytes)

  [index.invert]                   # Optional: create inverted index
  shards = 4                       # Number of shards (default: 1)

  [buckets.BucketName]             # Define a bucket
  files = [\"ref1.fa\", \"ref2.fa\"]   # Files for this bucket

  [buckets.AnotherBucket]
  files = [\"other.fasta\"]

CLI OPTIONS OVERRIDE CONFIG FILE:
  --max-shard-size overrides [index].max_shard_size
  --invert enables inverted index creation (even without [index.invert])
  --invert-shards overrides [index.invert].shards

NOTE: Cannot combine max_shard_size with [index.invert]. When main index is
sharded, use 'rype index invert' separately (creates 1:1 inverted shards).")]
    FromConfig {
        /// Path to TOML config file
        #[arg(short, long)]
        config: PathBuf,

        /// Maximum shard size for main index (e.g., "1G", overrides config)
        #[arg(long, value_parser = parse_shard_size_arg)]
        max_shard_size: Option<usize>,

        /// Create inverted index after building (overrides config)
        #[arg(short = 'I', long)]
        invert: bool,

        /// Number of shards for inverted index (overrides config, implies --invert)
        #[arg(long)]
        invert_shards: Option<u32>,
    },

    /// Add files to existing index using TOML config (see CONFIG FORMAT below)
    #[command(after_help = "CONFIG FORMAT (bucket-add-config):
  [target]
  index = \"existing.ryidx\"         # Index to modify

  [assignment]
  mode = \"new_bucket\"              # or \"existing_bucket\" or \"best_bin\"
  bucket_name = \"MyBucket\"         # For new_bucket/existing_bucket modes
  # For best_bin mode:
  # threshold = 0.3                 # Min score to match existing bucket
  # fallback = \"create_new\"        # or \"skip\" or \"error\"

  [files]
  paths = [\"new1.fa\", \"new2.fa\"]   # Files to add")]
    BucketAddConfig {
        /// Path to TOML config file
        #[arg(short, long)]
        config: PathBuf,
    },

    /// Create inverted index for faster classification (2-10x speedup)
    #[command(after_help = "The inverted index maps minimizers to buckets instead of buckets to
minimizers, enabling O(Q log U) lookups instead of O(B × Q × log M).

USAGE:
  rype index invert -i index.ryidx      # Creates index.ryxdi
  rype classify run -i index.ryidx -I   # Use inverted index

SHARDING:
  For very large indices that exceed RAM, use --shards to split the
  inverted index into multiple files loaded on-demand during classification.")]
    Invert {
        /// Path to primary index file (.ryidx)
        #[arg(short, long)]
        index: PathBuf,

        /// Number of shards (default: 1 = single file).
        /// Use for memory-constrained classification of large indices.
        /// Ignored if main index is already sharded (uses 1:1 correspondence).
        #[arg(long, default_value_t = 1)]
        shards: u32,
    },

    /// Show detailed minimizer statistics for compression analysis
    Summarize {
        /// Path to index file (.ryidx)
        #[arg(short, long)]
        index: PathBuf,
    },
}

#[derive(Subcommand)]
enum ClassifyCommands {
    /// Classify reads against an index, one result line per read
    #[command(after_help = "OUTPUT FORMAT:
  Tab-separated values (TSV): read_id<TAB>bucket_name<TAB>score

  read_id     - First whitespace-delimited token from FASTA/FASTQ header
  bucket_name - Human-readable name from index (or filename if unnamed)
  score       - Fraction of query minimizers matching bucket (0.0-1.0)

  Only reads with score >= threshold for at least one bucket are output.
  A single read may produce multiple lines if it matches multiple buckets.

THRESHOLD GUIDANCE:
  0.05  - High sensitivity, useful for detecting distant homologs
  0.10  - Balanced (default), good for most metagenomic classification
  0.20  - High specificity, fewer false positives
  0.30+ - Very stringent, may miss true matches

WHEN TO USE 'run' vs 'aggregate':
  Use 'run' (this command) for:
  - Per-read classification results
  - Downstream analysis requiring read-level assignments
  - When you need to know which specific reads matched

  Use 'aggregate' for:
  - Sample-level composition estimates
  - Higher sensitivity (pools evidence across reads)
  - Abundance estimation")]
    Run {
        /// Path to target index (references to classify against)
        #[arg(short, long, visible_alias = "positive-index")]
        index: PathBuf,

        /// Path to negative index for contamination filtering.
        /// Minimizers matching the negative index are excluded before scoring.
        /// Use for host depletion (e.g., human reads) or adapter removal.
        /// Must have same k, w, salt as positive index.
        #[arg(short = 'N', long)]
        negative_index: Option<PathBuf>,

        /// Forward reads (FASTA/FASTQ, optionally gzipped)
        #[arg(short = '1', long)]
        r1: PathBuf,

        /// Reverse reads for paired-end data (optional)
        #[arg(short = '2', long)]
        r2: Option<PathBuf>,

        /// Minimum score threshold for reporting matches (0.0-1.0).
        /// Score = matching_minimizers / total_query_minimizers.
        /// Lower = more sensitive, higher = more specific.
        #[arg(short, long, default_value_t = 0.1)]
        threshold: f64,

        /// Maximum memory to use (e.g., "4G", "512M", "auto").
        /// Default: auto-detect available memory.
        /// Batch size is calculated automatically based on this limit.
        #[arg(long, default_value = "auto", value_parser = parse_max_memory_arg)]
        max_memory: usize,

        /// Override automatic batch size calculation.
        /// If set, uses this fixed batch size instead of adaptive sizing.
        #[arg(short, long)]
        batch_size: Option<usize>,

        /// Output file path (TSV format). Writes to stdout if not specified.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Use inverted index (.ryxdi) for faster classification.
        /// Creates index.ryxdi with: rype index invert -i index.ryidx
        /// Recommended for large indices or repeated classifications.
        #[arg(short = 'I', long)]
        use_inverted: bool,

        /// Use merge-join algorithm (requires --use-inverted).
        /// Faster when query and index have high overlap.
        /// May be slower for queries with many unique minimizers.
        #[arg(short = 'M', long)]
        merge_join: bool,
    },

    /// Batch classify reads (identical to 'run', kept for backwards compatibility)
    #[command(hide = true)]
    Batch {
        #[arg(short, long, visible_alias = "positive-index")]
        index: PathBuf,
        #[arg(short = 'N', long)]
        negative_index: Option<PathBuf>,
        #[arg(short = '1', long)]
        r1: PathBuf,
        #[arg(short = '2', long)]
        r2: Option<PathBuf>,
        #[arg(short, long, default_value_t = 0.1)]
        threshold: f64,
        #[arg(long, default_value = "auto", value_parser = parse_max_memory_arg)]
        max_memory: usize,
        #[arg(short, long)]
        batch_size: Option<usize>,
        #[arg(short, long)]
        output: Option<PathBuf>,
        #[arg(short = 'I', long)]
        use_inverted: bool,
        #[arg(short = 'M', long)]
        merge_join: bool,
    },

    /// Pool all reads for sample-level classification (higher sensitivity)
    #[command(alias = "agg")]
    #[command(after_help = "AGGREGATE vs RUN:
  'aggregate' pools minimizers from all reads before scoring, providing:
  - Higher sensitivity for low-abundance targets
  - Sample-level composition rather than per-read assignments
  - Reduced noise from individual read variation

  Use 'aggregate' when you want to know what's in a sample.
  Use 'run' when you need read-level assignments.

OUTPUT FORMAT:
  Tab-separated: query_name<TAB>bucket_name<TAB>score

  query_name is always 'global' since reads are aggregated.

THRESHOLD:
  Default 0.05 (lower than 'run') since aggregation reduces noise.
  Score represents fraction of total unique minimizers matching bucket.")]
    Aggregate {
        /// Path to target index
        #[arg(short, long, visible_alias = "positive-index")]
        index: PathBuf,

        /// Path to negative index for contamination filtering
        #[arg(short = 'N', long)]
        negative_index: Option<PathBuf>,

        /// Forward reads (FASTA/FASTQ, optionally gzipped)
        #[arg(short = '1', long)]
        r1: PathBuf,

        /// Reverse reads for paired-end data (optional)
        #[arg(short = '2', long)]
        r2: Option<PathBuf>,

        /// Minimum score threshold (default lower than 'run' since
        /// aggregation reduces noise)
        #[arg(short, long, default_value_t = 0.05)]
        threshold: f64,

        /// Maximum memory to use (e.g., "4G", "512M", "auto").
        #[arg(long, default_value = "auto", value_parser = parse_max_memory_arg)]
        max_memory: usize,

        /// Override automatic batch size calculation
        #[arg(short, long)]
        batch_size: Option<usize>,

        /// Output file path (TSV format)
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
                    // Show inverted index stats - check for sharded vs single-file
                    let inverted_path = index.with_extension("ryxdi");
                    let manifest_path = ShardManifest::manifest_path(&inverted_path);

                    if manifest_path.exists() {
                        // Sharded inverted index - manifest is already lightweight
                        let manifest = ShardManifest::load(&manifest_path)?;
                        println!("Sharded Inverted Index Stats for {:?}", inverted_path);
                        println!("  K: {}", manifest.k);
                        println!("  Window (w): {}", manifest.w);
                        println!("  Salt: 0x{:x}", manifest.salt);
                        println!("  Shards: {}", manifest.shards.len());
                        println!("  Unique minimizers: {}", manifest.total_minimizers);
                        println!("  Total bucket references: {}", manifest.total_bucket_ids);
                        if manifest.total_minimizers > 0 {
                            println!("  Avg buckets per minimizer: {:.2}",
                                manifest.total_bucket_ids as f64 / manifest.total_minimizers as f64);
                        }
                        println!("  ------------------------------------------------");
                        println!("  Shard distribution:");
                        for shard in &manifest.shards {
                            println!("    Shard {}: {} minimizers, {} bucket refs",
                                shard.shard_id, shard.num_minimizers, shard.num_bucket_ids);
                        }
                    } else if inverted_path.exists() {
                        // Single-file inverted index - use load_stats() for header only
                        let stats = InvertedIndex::load_stats(&inverted_path)?;
                        println!("Inverted Index Stats for {:?}", inverted_path);
                        println!("  K: {}", stats.k);
                        println!("  Window (w): {}", stats.w);
                        println!("  Salt: 0x{:x}", stats.salt);
                        println!("  Unique minimizers: {}", stats.num_minimizers);
                        println!("  Total bucket references: {}", stats.num_bucket_ids);
                        if stats.num_minimizers > 0 {
                            println!("  Avg buckets per minimizer: {:.2}",
                                stats.num_bucket_ids as f64 / stats.num_minimizers as f64);
                        }
                    } else {
                        return Err(anyhow!(
                            "Inverted index not found: {:?}. Create it with 'rype index invert -i {:?}'",
                            inverted_path, index
                        ));
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
                let metadata = load_index_metadata(&index)?;
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

            IndexCommands::FromConfig { config, max_shard_size, invert, invert_shards } => {
                build_index_from_config(&config, max_shard_size, invert, invert_shards)?;
            }

            IndexCommands::BucketAddConfig { config } => {
                bucket_add_from_config(&config)?;
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
            ClassifyCommands::Run { index, negative_index, r1, r2, threshold, max_memory, batch_size, output, use_inverted, merge_join } |
            ClassifyCommands::Batch { index, negative_index, r1, r2, threshold, max_memory, batch_size, output, use_inverted, merge_join } => {
                if merge_join && !use_inverted {
                    return Err(anyhow!("--merge-join requires --use-inverted"));
                }

                // Determine effective batch size: user override or adaptive
                let effective_batch_size = if let Some(bs) = batch_size {
                    log::info!("Using user-specified batch size: {}", bs);
                    bs
                } else {
                    // Load index metadata to get k, w, num_buckets
                    let metadata = load_index_metadata(&index)?;

                    // Detect or use specified memory limit (0 = auto)
                    let mem_limit = if max_memory == 0 {
                        let detected = detect_available_memory();
                        if detected.source == MemorySource::Fallback {
                            log::warn!("Could not detect available memory, using 8GB fallback. \
                                Consider specifying --max-memory explicitly.");
                        } else {
                            log::info!("Auto-detected available memory: {} (source: {:?})",
                                format_bytes(detected.bytes), detected.source);
                        }
                        detected.bytes
                    } else {
                        max_memory
                    };

                    // Sample read lengths from input files
                    let is_paired = r2.is_some();
                    let read_profile = ReadMemoryProfile::from_files(
                        &r1,
                        r2.as_deref(),
                        1000,  // sample size
                        metadata.k,
                        metadata.w,
                    ).unwrap_or_else(|| {
                        log::warn!("Could not sample read lengths, using default profile");
                        ReadMemoryProfile::default_profile(is_paired, metadata.k, metadata.w)
                    });

                    log::debug!("Read profile: avg_read_length={}, avg_query_length={}, minimizers_per_query={}",
                        read_profile.avg_read_length, read_profile.avg_query_length, read_profile.minimizers_per_query);

                    // For now, use a simple heuristic since we don't have index loaded yet
                    // We'll estimate index memory from metadata
                    let estimated_index_mem = metadata.bucket_minimizer_counts.values().sum::<usize>() * 8;
                    let num_buckets = metadata.bucket_names.len();

                    let mem_config = MemoryConfig {
                        max_memory: mem_limit,
                        num_threads: rayon::current_num_threads(),
                        index_memory: estimated_index_mem,
                        shard_reservation: 0, // Will be updated after loading index
                        read_profile,
                        num_buckets,
                    };

                    let batch_config = calculate_batch_config(&mem_config);
                    log::info!("Adaptive batch sizing: batch_size={}, estimated peak memory={}",
                        batch_config.batch_size, format_bytes(batch_config.peak_memory));
                    batch_config.batch_size
                };

                // Load negative index if provided, validate parameters, and build minimizer set
                let neg_mins: Option<HashSet<u64>> = match &negative_index {
                    None => None,
                    Some(neg_path) => {
                        log::info!("Loading negative index from {:?}", neg_path);
                        let neg = Index::load(neg_path)?;

                        // Load positive index metadata to validate parameters match
                        let pos_metadata = load_index_metadata(&index)?;
                        if neg.k != pos_metadata.k {
                            return Err(anyhow!(
                                "Negative index K ({}) does not match positive index K ({})",
                                neg.k, pos_metadata.k
                            ));
                        }
                        if neg.w != pos_metadata.w {
                            return Err(anyhow!(
                                "Negative index W ({}) does not match positive index W ({})",
                                neg.w, pos_metadata.w
                            ));
                        }
                        if neg.salt != pos_metadata.salt {
                            return Err(anyhow!(
                                "Negative index salt (0x{:x}) does not match positive index salt (0x{:x})",
                                neg.salt, pos_metadata.salt
                            ));
                        }

                        let total_mins: usize = neg.buckets.values().map(|v| v.len()).sum();
                        log::info!("Negative index loaded: {} buckets, {} minimizers",
                            neg.buckets.len(), total_mins);
                        let neg_set: HashSet<u64> = neg.buckets.values()
                            .flat_map(|v| v.iter().copied())
                            .collect();
                        log::info!("Built negative minimizer set: {} unique minimizers", neg_set.len());
                        Some(neg_set)
                    }
                };

                if use_inverted {
                    // Use inverted index path - detect sharded vs single-file format
                    let inverted_path = index.with_extension("ryxdi");
                    let manifest_path = ShardManifest::manifest_path(&inverted_path);

                    // Load index metadata first (needed for both paths)
                    log::info!("Loading index metadata from {:?}", index);
                    let metadata = load_index_metadata(&index)?;
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
                            log::info!("Starting merge-join classification with sequential shard loading (batch_size={})", effective_batch_size);
                        } else {
                            log::info!("Starting classification with sequential shard loading (batch_size={})", effective_batch_size);
                        }

                        while let Some((owned_records, headers)) = io.next_batch_records(effective_batch_size)? {
                            batch_num += 1;
                            let batch_read_count = owned_records.len();
                            total_reads += batch_read_count;

                            let batch_refs: Vec<QueryRecord> = owned_records.iter()
                                .map(|(id, s1, s2)| (*id, s1.as_slice(), s2.as_deref()))
                                .collect();

                            let results = if merge_join {
                                classify_batch_sharded_merge_join(&sharded, neg_mins.as_ref(), &batch_refs, threshold)?
                            } else {
                                classify_batch_sharded_sequential(&sharded, neg_mins.as_ref(), &batch_refs, threshold)?
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
                            log::info!("Starting merge-join classification with inverted index (batch_size={})", effective_batch_size);
                        } else {
                            log::info!("Starting classification with inverted index (batch_size={})", effective_batch_size);
                        }

                        while let Some((owned_records, headers)) = io.next_batch_records(effective_batch_size)? {
                            batch_num += 1;
                            let batch_read_count = owned_records.len();
                            total_reads += batch_read_count;

                            let batch_refs: Vec<QueryRecord> = owned_records.iter()
                                .map(|(id, s1, s2)| (*id, s1.as_slice(), s2.as_deref()))
                                .collect();

                            let results = if merge_join {
                                classify_batch_with_query_index(&inverted, neg_mins.as_ref(), &batch_refs, threshold)
                            } else {
                                classify_batch_inverted(&inverted, neg_mins.as_ref(), &batch_refs, threshold)
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

                        log::info!("Starting classification with sequential main shard loading (batch_size={})", effective_batch_size);

                        while let Some((owned_records, headers)) = io.next_batch_records(effective_batch_size)? {
                            batch_num += 1;
                            let batch_read_count = owned_records.len();
                            total_reads += batch_read_count;

                            let batch_refs: Vec<QueryRecord> = owned_records.iter()
                                .map(|(id, s1, s2)| (*id, s1.as_slice(), s2.as_deref()))
                                .collect();

                            let results = classify_batch_sharded_main(&sharded, neg_mins.as_ref(), &batch_refs, threshold)?;

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

                        log::info!("Starting classification (batch_size={})", effective_batch_size);

                        while let Some((owned_records, headers)) = io.next_batch_records(effective_batch_size)? {
                            batch_num += 1;
                            let batch_read_count = owned_records.len();
                            total_reads += batch_read_count;

                            let batch_refs: Vec<QueryRecord> = owned_records.iter()
                                .map(|(id, s1, s2)| (*id, s1.as_slice(), s2.as_deref()))
                                .collect();

                            let results = classify_batch(&engine, neg_mins.as_ref(), &batch_refs, threshold);

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

            ClassifyCommands::Aggregate { index, negative_index, r1, r2, threshold, max_memory, batch_size, output } => {
                log::info!("Loading index from {:?}", index);
                let engine = Index::load(&index)?;
                log::info!("Index loaded: {} buckets", engine.buckets.len());

                // Determine effective batch size: user override or adaptive
                let effective_batch_size = if let Some(bs) = batch_size {
                    log::info!("Using user-specified batch size: {}", bs);
                    bs
                } else {
                    // Detect or use specified memory limit (0 = auto)
                    let mem_limit = if max_memory == 0 {
                        let detected = detect_available_memory();
                        if detected.source == MemorySource::Fallback {
                            log::warn!("Could not detect available memory, using 8GB fallback. \
                                Consider specifying --max-memory explicitly.");
                        } else {
                            log::info!("Auto-detected available memory: {} (source: {:?})",
                                format_bytes(detected.bytes), detected.source);
                        }
                        detected.bytes
                    } else {
                        max_memory
                    };

                    let is_paired = r2.is_some();
                    let read_profile = ReadMemoryProfile::from_files(
                        &r1,
                        r2.as_deref(),
                        1000,
                        engine.k,
                        engine.w,
                    ).unwrap_or_else(|| {
                        log::warn!("Could not sample read lengths, using default profile");
                        ReadMemoryProfile::default_profile(is_paired, engine.k, engine.w)
                    });

                    let estimated_index_mem = engine.buckets.values().map(|v| v.len() * 8).sum::<usize>();
                    let num_buckets = engine.buckets.len();

                    let mem_config = MemoryConfig {
                        max_memory: mem_limit,
                        num_threads: rayon::current_num_threads(),
                        index_memory: estimated_index_mem,
                        shard_reservation: 0,
                        read_profile,
                        num_buckets,
                    };

                    let batch_config = calculate_batch_config(&mem_config);
                    log::info!("Adaptive batch sizing: batch_size={}, estimated peak memory={}",
                        batch_config.batch_size, format_bytes(batch_config.peak_memory));
                    batch_config.batch_size
                };

                // Load negative index if provided, validate parameters, and build minimizer set
                let neg_mins: Option<HashSet<u64>> = match &negative_index {
                    None => None,
                    Some(neg_path) => {
                        log::info!("Loading negative index from {:?}", neg_path);
                        let neg = Index::load(neg_path)?;

                        // Validate parameters match
                        if neg.k != engine.k {
                            return Err(anyhow!(
                                "Negative index K ({}) does not match positive index K ({})",
                                neg.k, engine.k
                            ));
                        }
                        if neg.w != engine.w {
                            return Err(anyhow!(
                                "Negative index W ({}) does not match positive index W ({})",
                                neg.w, engine.w
                            ));
                        }
                        if neg.salt != engine.salt {
                            return Err(anyhow!(
                                "Negative index salt (0x{:x}) does not match positive index salt (0x{:x})",
                                neg.salt, engine.salt
                            ));
                        }

                        let total_mins: usize = neg.buckets.values().map(|v| v.len()).sum();
                        log::info!("Negative index loaded: {} buckets, {} minimizers",
                            neg.buckets.len(), total_mins);
                        let neg_set: HashSet<u64> = neg.buckets.values()
                            .flat_map(|v| v.iter().copied())
                            .collect();
                        log::info!("Built negative minimizer set: {} unique minimizers", neg_set.len());
                        Some(neg_set)
                    }
                };

                let mut io = IoHandler::new(&r1, r2.as_ref(), output)?;
                io.write(b"query_name\tbucket_name\tscore\n".to_vec())?;

                let mut total_reads = 0;
                let mut batch_num = 0;

                log::info!("Starting aggregate classification (batch_size={})", effective_batch_size);

                while let Some((owned_records, _)) = io.next_batch_records(effective_batch_size)? {
                    batch_num += 1;
                    let batch_read_count = owned_records.len();
                    total_reads += batch_read_count;

                    let batch_refs: Vec<QueryRecord> = owned_records.iter()
                        .map(|(id, s1, s2)| (*id, s1.as_slice(), s2.as_deref()))
                        .collect();

                    let results = aggregate_batch(&engine, neg_mins.as_ref(), &batch_refs, threshold);

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

fn build_index_from_config(
    config_path: &Path,
    cli_max_shard_size: Option<usize>,
    cli_invert: bool,
    cli_invert_shards: Option<u32>,
) -> Result<()> {
    log::info!("Building index from config: {}", config_path.display());

    // 1. Parse and validate config
    let cfg = parse_config(config_path)?;
    let config_dir = config_path.parent()
        .ok_or_else(|| anyhow!("Invalid config path"))?;

    log::info!("Validating file paths...");
    validate_config(&cfg, config_dir)?;
    log::info!("Validation successful.");

    // Resolve CLI overrides for sharding options
    // CLI takes precedence over config file
    let max_shard_size = cli_max_shard_size.or(cfg.index.max_shard_size);

    // Determine if we should create inverted index and how many shards
    // --invert-shards implies --invert
    let should_invert = cli_invert || cli_invert_shards.is_some() || cfg.index.invert.is_some();
    let invert_shards = cli_invert_shards
        .or_else(|| cfg.index.invert.as_ref().map(|i| i.shards))
        .unwrap_or(1);

    // Disallow combining main index sharding with inverted index creation
    // When main index is sharded, inverted shards must be 1:1 with main shards,
    // which requires loading each shard separately. Use 'index invert' for this.
    if max_shard_size.is_some() && should_invert {
        return Err(anyhow!(
            "Cannot create inverted index when sharding main index.\n\
             When --max-shard-size is used, inverted shards must be 1:1 with main shards.\n\
             Run 'rype index invert -i {}' after building to create the inverted index.",
            cfg.index.output.display()
        ));
    }

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

    // 5. Save final index (with optional sharding)
    if let Some(max_bytes) = max_shard_size {
        log::info!("Saving sharded index to {:?} (max {} bytes/shard)...", cfg.index.output, max_bytes);
        let manifest = final_index.save_sharded(&cfg.index.output, max_bytes)?;
        log::info!("Created {} shards with {} total minimizers.", manifest.shards.len(), manifest.total_minimizers);
    } else {
        log::info!("Saving index to {}...", cfg.index.output.display());
        final_index.save(&cfg.index.output)?;
    }

    log::info!("Done! Index saved successfully.");
    log::info!("\nFinal statistics:");
    log::info!("  Buckets: {}", final_index.buckets.len());
    let total_minimizers: usize = final_index.buckets.values().map(|v| v.len()).sum();
    log::info!("  Total minimizers: {}", total_minimizers);

    // 6. Optionally create inverted index
    if should_invert {
        let inverted_path = cfg.index.output.with_extension("ryxdi");
        log::info!("Building inverted index...");
        let inverted = InvertedIndex::build_from_index(&final_index);
        log::info!("Inverted index built: {} unique minimizers, {} bucket entries",
            inverted.num_minimizers(), inverted.num_bucket_entries());

        if invert_shards > 1 {
            log::info!("Saving sharded inverted index ({} shards) to {:?}", invert_shards, inverted_path);
            let manifest = inverted.save_sharded(&inverted_path, invert_shards)?;
            log::info!("Created {} inverted shards:", manifest.shards.len());
            for shard in &manifest.shards {
                log::info!("  Shard {}: {} unique minimizers, {} bucket entries",
                    shard.shard_id, shard.num_minimizers, shard.num_bucket_ids);
            }
        } else {
            log::info!("Saving inverted index to {:?}", inverted_path);
            inverted.save(&inverted_path)?;
        }
    }

    Ok(())
}

// --- BUCKET ADD FROM CONFIG ---
//
// CONCURRENCY WARNING: bucket-add-config operations are NOT safe for concurrent use.
// Running multiple bucket-add-config commands simultaneously on the same index may
// result in data corruption. Use file locking or sequential processing in pipelines.

/// Represents a file assignment during bucket-add-config processing
#[derive(Debug)]
struct FileAssignment {
    file_path: PathBuf,
    bucket_id: u32,
    bucket_name: String,
    mode: &'static str,  // "new_bucket", "existing_bucket", "matched", "created"
    score: Option<f64>,
}

/// Data extracted from a single file in one pass
struct FileData {
    path: PathBuf,
    minimizers: Vec<u64>,  // sorted, deduplicated
    sources: Vec<String>,  // source labels for each sequence
}

/// Extract minimizers and source labels from a file in a single pass
fn extract_file_data(path: &Path, k: usize, w: usize, salt: u64) -> Result<FileData> {
    let mut reader = parse_fastx_file(path)
        .context(format!("Failed to open file {}", path.display()))?;

    let filename = path.canonicalize()
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .into_owned();

    let mut ws = MinimizerWorkspace::new();
    let mut all_mins = Vec::new();
    let mut sources = Vec::new();

    while let Some(record) = reader.next() {
        let rec = record.context(format!("Invalid record in file {}", path.display()))?;
        let seq_name = String::from_utf8_lossy(rec.id()).to_string();
        let source_label = format!("{}{}{}", filename, Index::BUCKET_SOURCE_DELIM, seq_name);
        sources.push(source_label);

        extract_into(&rec.seq(), k, w, salt, &mut ws);
        all_mins.extend_from_slice(&ws.buffer);
    }

    all_mins.sort_unstable();
    all_mins.dedup();

    Ok(FileData {
        path: path.to_path_buf(),
        minimizers: all_mins,
        sources,
    })
}

/// Count intersection using two-pointer merge - O(Q + B) instead of O(Q log B)
/// Both arrays must be sorted.
fn count_intersection_merge(query: &[u64], bucket: &[u64]) -> usize {
    let mut i = 0;
    let mut j = 0;
    let mut hits = 0;

    while i < query.len() && j < bucket.len() {
        match query[i].cmp(&bucket[j]) {
            std::cmp::Ordering::Equal => {
                hits += 1;
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    hits
}

/// Find bucket ID by name, with helpful error message listing available buckets
fn find_bucket_by_name(bucket_names: &HashMap<u32, String>, name: &str) -> Result<u32> {
    for (&id, bucket_name) in bucket_names {
        if bucket_name == name {
            return Ok(id);
        }
    }
    let available: Vec<_> = bucket_names.values().collect();
    Err(anyhow!(
        "Bucket '{}' not found. Available buckets: {:?}",
        name, available
    ))
}

/// Index parameters for validation
struct IndexParams {
    k: usize,
    w: usize,
    salt: u64,
}

fn bucket_add_from_config(config_path: &Path) -> Result<()> {
    log::info!("Adding files to index from config: {}", config_path.display());

    // 1. Parse and validate config
    let cfg = parse_bucket_add_config(config_path)?;
    let config_dir = config_path.parent()
        .ok_or_else(|| anyhow!("Invalid config path"))?;

    log::info!("Validating file paths...");
    validate_bucket_add_config(&cfg, config_dir)?;
    log::info!("Validation successful.");

    // 2. Resolve index path and detect type
    let index_path = resolve_path(config_dir, &cfg.target.index);

    if MainIndexManifest::is_sharded(&index_path) {
        bucket_add_sharded(&index_path, &cfg, config_dir)
    } else {
        bucket_add_single(&index_path, &cfg, config_dir)
    }
}

/// Handle bucket-add-config for single-file indices
fn bucket_add_single(
    index_path: &Path,
    cfg: &rype::config::BucketAddConfig,
    config_dir: &Path,
) -> Result<()> {
    let mut index = Index::load(index_path)?;
    let mut assignments: Vec<FileAssignment> = Vec::new();
    let params = IndexParams { k: index.k, w: index.w, salt: index.salt };

    log::info!("Loaded index: k={}, w={}, {} buckets, {} total minimizers",
        index.k, index.w, index.buckets.len(),
        index.buckets.values().map(|v| v.len()).sum::<usize>());

    match &cfg.assignment {
        AssignmentSettings::NewBucket { bucket_name } => {
            let new_id = index.next_id()?;
            let name = bucket_name.clone()
                .or_else(|| cfg.files.bucket_name.clone())
                .unwrap_or_else(|| make_bucket_name_from_files(&cfg.files.paths));
            let name = sanitize_bucket_name(&name);

            log::info!("Creating new bucket {} ('{}')", new_id, name);
            index.bucket_names.insert(new_id, name.clone());

            // Extract all file data in single pass per file
            for file_path in &cfg.files.paths {
                let abs_path = resolve_path(config_dir, file_path);
                let data = extract_file_data(&abs_path, params.k, params.w, params.salt)?;

                // Add minimizers and sources directly
                let bucket = index.buckets.entry(new_id).or_default();
                bucket.extend(&data.minimizers);
                let sources = index.bucket_sources.entry(new_id).or_default();
                sources.extend(data.sources);

                log::info!("  {} -> bucket {} ('{}') [{} minimizers]",
                    file_path.display(), new_id, name, data.minimizers.len());
                assignments.push(FileAssignment {
                    file_path: file_path.clone(),
                    bucket_id: new_id,
                    bucket_name: name.clone(),
                    mode: "new_bucket",
                    score: None,
                });
            }

            index.finalize_bucket(new_id);
        }

        AssignmentSettings::ExistingBucket { bucket_name } => {
            let bucket_id = find_bucket_by_name(&index.bucket_names, bucket_name)?;
            log::info!("Adding to existing bucket {} ('{}')", bucket_id, bucket_name);

            for file_path in &cfg.files.paths {
                let abs_path = resolve_path(config_dir, file_path);
                let data = extract_file_data(&abs_path, params.k, params.w, params.salt)?;

                let bucket = index.buckets.entry(bucket_id).or_default();
                bucket.extend(&data.minimizers);
                let sources = index.bucket_sources.entry(bucket_id).or_default();
                sources.extend(data.sources);

                log::info!("  {} -> bucket {} ('{}') [{} minimizers]",
                    file_path.display(), bucket_id, bucket_name, data.minimizers.len());
                assignments.push(FileAssignment {
                    file_path: file_path.clone(),
                    bucket_id,
                    bucket_name: bucket_name.clone(),
                    mode: "existing_bucket",
                    score: None,
                });
            }

            index.finalize_bucket(bucket_id);
        }

        AssignmentSettings::BestBin { threshold, fallback } => {
            best_bin_assign(&mut index, cfg, config_dir, params, *threshold, *fallback, &mut assignments)?;
        }
    }

    // Save updated index
    index.save(index_path)?;
    log::info!("Index saved to {}", index_path.display());

    print_assignment_summary(&assignments);
    Ok(())
}

/// Unified best-bin assignment for single-file index
fn best_bin_assign(
    index: &mut Index,
    cfg: &rype::config::BucketAddConfig,
    config_dir: &Path,
    params: IndexParams,
    threshold: f64,
    fallback: BestBinFallback,
    assignments: &mut Vec<FileAssignment>,
) -> Result<()> {
    log::info!("Best-bin mode: threshold={}, fallback={:?}", threshold, fallback);

    // Track assignments by category
    let mut matched: Vec<(FileData, u32, f64)> = Vec::new();  // file, bucket_id, score
    let mut to_create: HashMap<String, Vec<FileData>> = HashMap::new();  // stem -> files
    let mut skipped_count = 0;

    // 1. Extract all file data and find best bucket for each
    for file_path in &cfg.files.paths {
        let abs_path = resolve_path(config_dir, file_path);
        let data = extract_file_data(&abs_path, params.k, params.w, params.salt)?;

        if data.minimizers.is_empty() {
            log::warn!("{}: No minimizers extracted, skipping", file_path.display());
            skipped_count += 1;
            continue;
        }

        // Find best bucket using O(Q+B) merge
        let mut best: Option<(u32, f64)> = None;
        for (&bucket_id, bucket_mins) in &index.buckets {
            let hits = count_intersection_merge(&data.minimizers, bucket_mins);
            let score = hits as f64 / data.minimizers.len() as f64;

            if best.map(|(_, s)| score > s).unwrap_or(true) {
                best = Some((bucket_id, score));
            }
        }

        // Log and decide
        if let Some((id, score)) = best {
            let name = index.bucket_names.get(&id).map(|s| s.as_str()).unwrap_or("unknown");
            log::info!("{}: best match is bucket {} ('{}') with score {:.4}",
                file_path.display(), id, name, score);

            if score >= threshold {
                matched.push((data, id, score));
            } else {
                handle_below_threshold(file_path, &data, score, threshold, fallback,
                    &mut to_create, &mut skipped_count)?;
            }
        } else {
            // No existing buckets
            log::info!("{}: no existing buckets to match against", file_path.display());
            handle_no_buckets(file_path, data, fallback, &mut to_create, &mut skipped_count)?;
        }
    }

    // 2. Add matched files to their buckets (reusing extracted data)
    for (data, bucket_id, score) in matched {
        let bucket = index.buckets.entry(bucket_id).or_default();
        bucket.extend(&data.minimizers);
        let sources = index.bucket_sources.entry(bucket_id).or_default();
        sources.extend(data.sources);

        let name = index.bucket_names.get(&bucket_id).cloned().unwrap_or_default();
        assignments.push(FileAssignment {
            file_path: data.path,
            bucket_id,
            bucket_name: name,
            mode: "matched",
            score: Some(score),
        });
    }

    // 3. Create new buckets for unmatched files (reusing extracted data)
    for (stem, files) in to_create {
        let new_id = index.next_id()?;
        let bucket_name = sanitize_bucket_name(&stem);
        index.bucket_names.insert(new_id, bucket_name.clone());
        log::info!("Creating new bucket {} ('{}') for {} file(s)", new_id, bucket_name, files.len());

        for data in files {
            let bucket = index.buckets.entry(new_id).or_default();
            bucket.extend(&data.minimizers);
            let sources = index.bucket_sources.entry(new_id).or_default();
            sources.extend(data.sources);

            assignments.push(FileAssignment {
                file_path: data.path,
                bucket_id: new_id,
                bucket_name: bucket_name.clone(),
                mode: "created",
                score: None,
            });
        }

        index.finalize_bucket(new_id);
    }

    // 4. Finalize modified existing buckets
    let modified: HashSet<u32> = assignments.iter()
        .filter(|a| a.mode == "matched")
        .map(|a| a.bucket_id)
        .collect();
    for id in modified {
        index.finalize_bucket(id);
    }

    if skipped_count > 0 {
        log::info!("{} file(s) skipped", skipped_count);
    }

    Ok(())
}

/// Handle file that scored below threshold
fn handle_below_threshold(
    file_path: &Path,
    data: &FileData,
    score: f64,
    threshold: f64,
    fallback: BestBinFallback,
    to_create: &mut HashMap<String, Vec<FileData>>,
    skipped_count: &mut usize,
) -> Result<()> {
    match fallback {
        BestBinFallback::CreateNew => {
            let stem = get_file_stem(file_path);
            // Clone data since we need to move it
            to_create.entry(stem).or_default().push(FileData {
                path: data.path.clone(),
                minimizers: data.minimizers.clone(),
                sources: data.sources.clone(),
            });
            log::info!("{}: will create new bucket (best score {:.4} < threshold {})",
                file_path.display(), score, threshold);
        }
        BestBinFallback::Skip => {
            log::info!("{}: skipped (best score {:.4} < threshold {})",
                file_path.display(), score, threshold);
            *skipped_count += 1;
        }
        BestBinFallback::Error => {
            return Err(anyhow!(
                "{}: No bucket meets threshold {} (best: {:.4})",
                file_path.display(), threshold, score
            ));
        }
    }
    Ok(())
}

/// Handle file when no buckets exist
fn handle_no_buckets(
    file_path: &Path,
    data: FileData,
    fallback: BestBinFallback,
    to_create: &mut HashMap<String, Vec<FileData>>,
    skipped_count: &mut usize,
) -> Result<()> {
    match fallback {
        BestBinFallback::CreateNew => {
            let stem = get_file_stem(file_path);
            to_create.entry(stem).or_default().push(data);
            log::info!("{}: will create new bucket (no existing buckets)", file_path.display());
        }
        BestBinFallback::Skip => {
            // FIXED: Skip means skip, not create
            log::info!("{}: skipped (no existing buckets)", file_path.display());
            *skipped_count += 1;
        }
        BestBinFallback::Error => {
            return Err(anyhow!(
                "{}: No existing buckets to match against",
                file_path.display()
            ));
        }
    }
    Ok(())
}

/// Get file stem as string for bucket naming
fn get_file_stem(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string()
}

/// Handle bucket-add-config for sharded main indices
fn bucket_add_sharded(
    index_path: &Path,
    cfg: &rype::config::BucketAddConfig,
    config_dir: &Path,
) -> Result<()> {
    let mut sharded = ShardedMainIndex::open(index_path)?;
    let mut assignments: Vec<FileAssignment> = Vec::new();
    let params = IndexParams {
        k: sharded.k(),
        w: sharded.w(),
        salt: sharded.salt(),
    };

    log::info!("Loaded sharded index: k={}, w={}, {} shards, {} buckets",
        params.k, params.w,
        sharded.manifest().shards.len(),
        sharded.manifest().bucket_names.len());

    match &cfg.assignment {
        AssignmentSettings::NewBucket { bucket_name } => {
            let new_id = sharded.next_id()?;
            let name = bucket_name.clone()
                .or_else(|| cfg.files.bucket_name.clone())
                .unwrap_or_else(|| make_bucket_name_from_files(&cfg.files.paths));
            let name = sanitize_bucket_name(&name);

            log::info!("Creating new bucket {} ('{}') as new shard", new_id, name);

            let mut all_mins = Vec::new();
            let mut all_sources = Vec::new();

            for file_path in &cfg.files.paths {
                let abs_path = resolve_path(config_dir, file_path);
                let data = extract_file_data(&abs_path, params.k, params.w, params.salt)?;

                log::info!("  {} -> bucket {} ('{}') [{} minimizers]",
                    file_path.display(), new_id, name, data.minimizers.len());

                all_mins.extend(data.minimizers);
                all_sources.extend(data.sources);
                assignments.push(FileAssignment {
                    file_path: file_path.clone(),
                    bucket_id: new_id,
                    bucket_name: name.clone(),
                    mode: "new_bucket",
                    score: None,
                });
            }

            all_mins.sort_unstable();
            all_mins.dedup();
            all_sources.sort_unstable();
            all_sources.dedup();

            sharded.add_bucket(new_id, &name, all_sources, all_mins)?;
        }

        AssignmentSettings::ExistingBucket { bucket_name } => {
            let bucket_id = find_bucket_by_name(&sharded.manifest().bucket_names, bucket_name)?;
            log::info!("Adding to existing bucket {} ('{}')", bucket_id, bucket_name);

            let mut all_mins = Vec::new();
            let mut all_sources = Vec::new();

            for file_path in &cfg.files.paths {
                let abs_path = resolve_path(config_dir, file_path);
                let data = extract_file_data(&abs_path, params.k, params.w, params.salt)?;

                log::info!("  {} -> bucket {} ('{}') [{} minimizers]",
                    file_path.display(), bucket_id, bucket_name, data.minimizers.len());

                all_mins.extend(data.minimizers);
                all_sources.extend(data.sources);
                assignments.push(FileAssignment {
                    file_path: file_path.clone(),
                    bucket_id,
                    bucket_name: bucket_name.clone(),
                    mode: "existing_bucket",
                    score: None,
                });
            }

            all_mins.sort_unstable();
            all_mins.dedup();

            sharded.update_bucket(bucket_id, all_sources, all_mins)?;
        }

        AssignmentSettings::BestBin { threshold, fallback } => {
            best_bin_assign_sharded(&mut sharded, cfg, config_dir, params, *threshold, *fallback, &mut assignments)?;
        }
    }

    print_assignment_summary(&assignments);
    Ok(())
}

/// Best-bin assignment for sharded index - streams through shards to avoid loading all into memory
fn best_bin_assign_sharded(
    sharded: &mut ShardedMainIndex,
    cfg: &rype::config::BucketAddConfig,
    config_dir: &Path,
    params: IndexParams,
    threshold: f64,
    fallback: BestBinFallback,
    assignments: &mut Vec<FileAssignment>,
) -> Result<()> {
    log::info!("Best-bin mode (sharded): threshold={}, fallback={:?}", threshold, fallback);

    // 1. Extract all file data upfront (single pass per file)
    let mut file_data: Vec<FileData> = Vec::new();
    for file_path in &cfg.files.paths {
        let abs_path = resolve_path(config_dir, file_path);
        let data = extract_file_data(&abs_path, params.k, params.w, params.salt)?;
        file_data.push(data);
    }

    // 2. Track best match for each file across all shards
    let mut file_best: Vec<Option<(u32, f64)>> = vec![None; file_data.len()];

    // 3. Stream through shards one at a time (memory efficient)
    let shard_infos: Vec<_> = sharded.manifest().shards.iter().cloned().collect();
    log::info!("Scanning {} shards for best matches...", shard_infos.len());

    for shard_info in &shard_infos {
        let shard_path = MainIndexManifest::shard_path(&sharded.base_path(), shard_info.shard_id);
        let shard = MainIndexShard::load(&shard_path)?;

        // Check each file against buckets in this shard
        for (idx, data) in file_data.iter().enumerate() {
            if data.minimizers.is_empty() {
                continue;
            }

            for (&bucket_id, bucket_mins) in &shard.buckets {
                let hits = count_intersection_merge(&data.minimizers, bucket_mins);
                let score = hits as f64 / data.minimizers.len() as f64;

                if file_best[idx].map(|(_, s)| score > s).unwrap_or(true) {
                    file_best[idx] = Some((bucket_id, score));
                }
            }
        }
        // Shard is dropped here, freeing memory before loading next
    }

    // 4. Make assignment decisions
    let mut matched: Vec<(usize, u32, f64)> = Vec::new();  // file_idx, bucket_id, score
    let mut to_create: HashMap<String, Vec<usize>> = HashMap::new();  // stem -> file indices
    let mut skipped_count = 0;

    for (idx, data) in file_data.iter().enumerate() {
        if data.minimizers.is_empty() {
            log::warn!("{}: No minimizers extracted, skipping", data.path.display());
            skipped_count += 1;
            continue;
        }

        if let Some((bucket_id, score)) = file_best[idx] {
            let name = sharded.manifest().bucket_names.get(&bucket_id)
                .map(|s| s.as_str())
                .unwrap_or("unknown");
            log::info!("{}: best match is bucket {} ('{}') with score {:.4}",
                data.path.display(), bucket_id, name, score);

            if score >= threshold {
                matched.push((idx, bucket_id, score));
            } else {
                match fallback {
                    BestBinFallback::CreateNew => {
                        let stem = get_file_stem(&data.path);
                        to_create.entry(stem).or_default().push(idx);
                        log::info!("{}: will create new bucket (score {:.4} < threshold {})",
                            data.path.display(), score, threshold);
                    }
                    BestBinFallback::Skip => {
                        log::info!("{}: skipped (score {:.4} < threshold {})",
                            data.path.display(), score, threshold);
                        skipped_count += 1;
                    }
                    BestBinFallback::Error => {
                        return Err(anyhow!(
                            "{}: No bucket meets threshold {} (best: {:.4})",
                            data.path.display(), threshold, score
                        ));
                    }
                }
            }
        } else {
            log::info!("{}: no existing buckets to match against", data.path.display());
            match fallback {
                BestBinFallback::CreateNew => {
                    let stem = get_file_stem(&data.path);
                    to_create.entry(stem).or_default().push(idx);
                    log::info!("{}: will create new bucket (no existing buckets)", data.path.display());
                }
                BestBinFallback::Skip => {
                    log::info!("{}: skipped (no existing buckets)", data.path.display());
                    skipped_count += 1;
                }
                BestBinFallback::Error => {
                    return Err(anyhow!(
                        "{}: No existing buckets to match against",
                        data.path.display()
                    ));
                }
            }
        }
    }

    // 5. Update existing buckets with matched files (group by bucket for efficiency)
    let mut by_bucket: HashMap<u32, Vec<(usize, f64)>> = HashMap::new();
    for (idx, bucket_id, score) in matched {
        by_bucket.entry(bucket_id).or_default().push((idx, score));
    }

    for (bucket_id, files) in by_bucket {
        let bucket_name = sharded.manifest().bucket_names.get(&bucket_id).cloned().unwrap_or_default();
        log::info!("Updating bucket {} ('{}') with {} file(s)", bucket_id, bucket_name, files.len());

        let mut all_mins = Vec::new();
        let mut all_sources = Vec::new();

        for (idx, score) in &files {
            let data = &file_data[*idx];
            all_mins.extend(&data.minimizers);
            all_sources.extend(data.sources.iter().cloned());

            assignments.push(FileAssignment {
                file_path: data.path.clone(),
                bucket_id,
                bucket_name: bucket_name.clone(),
                mode: "matched",
                score: Some(*score),
            });
        }

        all_mins.sort_unstable();
        all_mins.dedup();

        sharded.update_bucket(bucket_id, all_sources, all_mins)?;
    }

    // 6. Create new buckets for unmatched files
    for (stem, indices) in to_create {
        let new_id = sharded.next_id()?;
        let bucket_name = sanitize_bucket_name(&stem);
        log::info!("Creating new bucket {} ('{}') for {} file(s)", new_id, bucket_name, indices.len());

        let mut all_mins = Vec::new();
        let mut all_sources = Vec::new();

        for idx in indices {
            let data = &file_data[idx];
            all_mins.extend(&data.minimizers);
            all_sources.extend(data.sources.iter().cloned());

            assignments.push(FileAssignment {
                file_path: data.path.clone(),
                bucket_id: new_id,
                bucket_name: bucket_name.clone(),
                mode: "created",
                score: None,
            });
        }

        all_mins.sort_unstable();
        all_mins.dedup();
        all_sources.sort_unstable();
        all_sources.dedup();

        sharded.add_bucket(new_id, &bucket_name, all_sources, all_mins)?;
    }

    if skipped_count > 0 {
        log::info!("{} file(s) skipped", skipped_count);
    }

    Ok(())
}

// --- HELPER FUNCTIONS FOR BUCKET ADD CONFIG ---

/// Generate bucket name from list of file paths
fn make_bucket_name_from_files(paths: &[PathBuf]) -> String {
    const UNNAMED_BUCKET: &str = "__unnamed__";
    if paths.is_empty() {
        return UNNAMED_BUCKET.to_string();
    }
    paths[0].file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(UNNAMED_BUCKET)
        .to_string()
}

/// Print assignment summary
fn print_assignment_summary(assignments: &[FileAssignment]) {
    if assignments.is_empty() {
        println!("\n=== Assignment Summary ===");
        println!("No files were assigned.");
        return;
    }

    println!("\n=== Assignment Summary ===");

    // Group by bucket
    let mut by_bucket: HashMap<u32, Vec<&FileAssignment>> = HashMap::new();
    for assignment in assignments {
        by_bucket.entry(assignment.bucket_id).or_default().push(assignment);
    }

    let mut bucket_ids: Vec<_> = by_bucket.keys().copied().collect();
    bucket_ids.sort();

    for bucket_id in bucket_ids {
        let files = &by_bucket[&bucket_id];
        let first = files[0];
        println!("\nBucket {} ('{}') [{}]:", bucket_id, first.bucket_name, first.mode);
        for assignment in files {
            if let Some(score) = assignment.score {
                println!("  - {} (score: {:.4})", assignment.file_path.display(), score);
            } else {
                println!("  - {}", assignment.file_path.display());
            }
        }
    }

    println!("\n{} files assigned to {} bucket(s)", assignments.len(), by_bucket.len());
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

