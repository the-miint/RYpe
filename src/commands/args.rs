//! Command-line argument definitions for the rype CLI.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

use super::helpers::{
    parse_bloom_fpp, parse_max_memory_arg, parse_shard_format, parse_shard_size_arg,
};

#[derive(Parser)]
#[command(name = "rype")]
#[command(about = "High-performance Read Partitioning Engine (RY-Space, K=16/32/64)")]
#[command(
    long_about = "Rype: High-performance genomic sequence classification using minimizer-based k-mer sketching in RY (purine/pyrimidine) space.

WORKFLOW:
  1. Create an index:     rype index create -o index.ryidx -r refs.fasta
  2. (Optional) Invert:   rype index invert -i index.ryidx
  3. Classify reads:      rype classify run -i index.ryidx -1 reads.fq

INPUT FORMATS:
  FASTA (.fa, .fasta, .fna) and FASTQ (.fq, .fastq) files are supported.
  Gzip-compressed files (.gz) are automatically detected and decompressed.
  Parquet (.parquet) with columns: read_id, sequence1, sequence2 (optional)

OUTPUT FORMAT (classify):
  Format auto-detected from extension:
  - .tsv or no extension: Plain TSV
  - .tsv.gz: Gzip-compressed TSV
  - .parquet: Apache Parquet with zstd compression
  - -: stdout (TSV)

  Tab-separated columns: read_id<TAB>bucket_name<TAB>score
  - read_id: Sequence header (first whitespace-delimited token)
  - bucket_name: Human-readable name from index
  - score: Fraction of query minimizers matching (0.0-1.0)"
)]
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
pub struct Cli {
    /// Enable verbose progress output with timestamps
    #[arg(short, long, global = true)]
    pub verbose: bool,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
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
pub enum IndexCommands {
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

        /// Create Parquet inverted index directly (bypasses main index).
        /// Output will be a directory (e.g., index.ryxdi/) containing Parquet files.
        /// This is the recommended format for large indices.
        #[arg(long)]
        parquet: bool,

        /// Parquet row group size (rows per group). Larger = better compression.
        /// Only used when --parquet is specified.
        #[arg(long, default_value_t = 100_000)]
        parquet_row_group_size: usize,

        /// Use Zstd compression instead of Snappy for Parquet files.
        /// Better compression ratio but slower. Only used with --parquet.
        #[arg(long)]
        parquet_zstd: bool,

        /// Enable bloom filters in Parquet files for faster lookups.
        /// Increases file size slightly. Only used with --parquet.
        #[arg(long)]
        parquet_bloom_filter: bool,

        /// Bloom filter false positive probability (0.0-1.0).
        /// Lower = more accurate but larger files. Only used with --parquet-bloom-filter.
        #[arg(long, default_value = "0.05", value_parser = parse_bloom_fpp)]
        parquet_bloom_fpp: f64,

        /// Print timing diagnostics to stderr for performance analysis.
        #[arg(long)]
        timing: bool,
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
  invert = true                    # Optional: create inverted index

  [buckets.BucketName]             # Define a bucket
  files = [\"ref1.fa\", \"ref2.fa\"]   # Files for this bucket

  [buckets.AnotherBucket]
  files = [\"other.fasta\"]

CLI OPTIONS OVERRIDE CONFIG FILE:
  --max-shard-size overrides [index].max_shard_size
  --invert (-I) enables inverted index creation (even without invert = true)

INVERTED INDEX SHARDING:
  Inverted shards are automatically created with 1:1 correspondence to main shards.
  - Sharded main index (--max-shard-size): creates N inverted shards (1:1)
  - Single-file main index: creates 1 inverted shard")]
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

        /// Create Parquet inverted index directly (bypasses main index).
        /// Output will be a directory (e.g., index.ryxdi/) containing Parquet files.
        #[arg(long)]
        parquet: bool,

        /// Parquet row group size (rows per group). Larger = better compression,
        /// but less effective filtering. Only used with --parquet.
        #[arg(long, default_value_t = 100_000)]
        parquet_row_group_size: usize,

        /// Enable bloom filters in Parquet files for faster lookups.
        /// Only used with --parquet.
        #[arg(long)]
        parquet_bloom_filter: bool,

        /// Bloom filter false positive probability (0.0-1.0).
        /// Only used with --parquet-bloom-filter.
        #[arg(long, default_value = "0.05", value_parser = parse_bloom_fpp)]
        parquet_bloom_fpp: f64,

        /// Print timing diagnostics to stderr for performance analysis.
        #[arg(long)]
        timing: bool,
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
    #[command(
        after_help = "The inverted index maps minimizers to buckets instead of buckets to
minimizers, enabling O(Q log U) lookups instead of O(B × Q × log M).

Creates bucket-partitioned shards:
- Sharded main index: creates 1:1 inverted shard correspondence
- Single-file main index: creates a 1-shard inverted index

FORMATS:
  legacy   - Custom RYXS binary format (default, smaller files)
  parquet  - Apache Parquet format (faster parallel I/O, requires --features parquet)

USAGE:
  rype index invert -i index.ryidx                    # Legacy format
  rype index invert -i index.ryidx --format parquet   # Parquet format
  rype classify run -i index.ryidx -I                 # Auto-detects format"
    )]
    Invert {
        /// Path to primary index file (.ryidx)
        #[arg(short, long)]
        index: PathBuf,

        /// Output format for inverted index shards
        #[arg(long, default_value = "legacy", value_parser = parse_shard_format)]
        format: String,

        /// Parquet row group size (rows per group). Larger = better compression.
        /// Only used when --format=parquet is specified.
        #[arg(long, default_value_t = 100_000)]
        parquet_row_group_size: usize,

        /// Use Zstd compression instead of Snappy for Parquet files.
        /// Better compression ratio but slower. Only used with --format=parquet.
        #[arg(long)]
        parquet_zstd: bool,

        /// Enable bloom filters in Parquet files for faster lookups.
        /// Increases file size slightly. Only used with --format=parquet.
        #[arg(long)]
        parquet_bloom_filter: bool,

        /// Bloom filter false positive probability (0.0-1.0).
        /// Lower = more accurate but larger files. Only used with --parquet-bloom-filter.
        #[arg(long, default_value = "0.05", value_parser = parse_bloom_fpp)]
        parquet_bloom_fpp: f64,

        /// Print timing diagnostics to stderr for performance analysis.
        #[arg(long)]
        timing: bool,
    },

    /// Show detailed minimizer statistics for compression analysis
    Summarize {
        /// Path to index file (.ryidx)
        #[arg(short, long)]
        index: PathBuf,
    },

    /// Convert single-file index to sharded format
    #[command(
        after_help = "Converts an existing single-file index (.ryidx) to sharded format.

This is useful for:
- Memory-constrained classification of large indices
- Enabling parallel I/O during classification
- Converting legacy indices to the more efficient sharded format

USAGE:
  rype index shard -i large.ryidx -o sharded.ryidx --max-shard-size 1G

The output will be:
  sharded.ryidx.manifest     (metadata)
  sharded.ryidx.shard.0      (first shard)
  sharded.ryidx.shard.1      (second shard)
  ..."
    )]
    Shard {
        /// Path to input single-file index (.ryidx)
        #[arg(short, long)]
        input: PathBuf,

        /// Path to output sharded index (base path for .manifest and .shard.* files)
        #[arg(short, long)]
        output: PathBuf,

        /// Maximum shard size (e.g., "1G", "512M", "100M")
        #[arg(long, value_parser = parse_shard_size_arg)]
        max_shard_size: usize,
    },
}

#[derive(Subcommand)]
pub enum ClassifyCommands {
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

        /// Forward reads. Formats: FASTA/FASTQ (.fa/.fq, optionally .gz),
        /// Parquet (.parquet) with columns: read_id, sequence1, sequence2 (optional)
        #[arg(short = '1', long)]
        r1: PathBuf,

        /// Reverse reads for paired-end data (optional).
        /// Not supported with Parquet input - use sequence2 column instead.
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

        /// Output file path. Format auto-detected from extension:
        /// - `.tsv` or no extension: Plain TSV
        /// - `.tsv.gz`: Gzip-compressed TSV
        /// - `.parquet`: Apache Parquet with zstd compression
        /// - `-`: stdout (TSV)
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

        /// Use parallel row group processing (requires --use-inverted with Parquet index).
        /// Processes each row group independently in parallel, maximizing CPU utilization.
        /// Most effective when query minimizers span entire index range (row group
        /// filtering is ineffective). Mutually exclusive with --merge-join.
        #[arg(long, conflicts_with = "merge_join")]
        parallel_rg: bool,

        /// Use bloom filters for row group filtering (requires --use-inverted with Parquet index).
        /// Reduces I/O by rejecting row groups that definitely don't contain query minimizers.
        /// Only effective if index was built with --parquet-bloom-filter.
        #[arg(long)]
        use_bloom_filter: bool,

        /// Enable parallel row group reading for Parquet input files.
        /// Processes N row groups in parallel for faster decompression.
        /// Default: 4 when enabled, 0 = disabled (sequential reading).
        /// Most effective with SSDs when decompression is CPU-bound.
        #[arg(long, default_value_t = 0)]
        parallel_input_rg: usize,

        /// Print timing diagnostics to stderr for performance analysis.
        #[arg(long)]
        timing: bool,
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
        #[arg(long, conflicts_with = "merge_join")]
        parallel_rg: bool,
        #[arg(long)]
        use_bloom_filter: bool,
        #[arg(long, default_value_t = 0)]
        parallel_input_rg: usize,
        #[arg(long)]
        timing: bool,
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

        /// Forward reads. Formats: FASTA/FASTQ (.fa/.fq, optionally .gz),
        /// Parquet (.parquet) with columns: read_id, sequence1, sequence2 (optional)
        #[arg(short = '1', long)]
        r1: PathBuf,

        /// Reverse reads for paired-end data (optional).
        /// Not supported with Parquet input - use sequence2 column instead.
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

        /// Output file path. Format auto-detected from extension:
        /// - `.tsv` or no extension: Plain TSV
        /// - `.tsv.gz`: Gzip-compressed TSV
        /// - `.parquet`: Apache Parquet with zstd compression
        /// - `-`: stdout (TSV)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
pub enum InspectCommands {
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
