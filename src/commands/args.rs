//! Command-line argument definitions for the rype CLI.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

use super::helpers::{
    parse_bloom_fpp, parse_max_memory_arg, parse_shard_size_arg, validate_trim_to,
};

#[derive(Parser)]
#[command(name = "rype")]
#[command(about = "High-performance Read Partitioning Engine (RY-Space, K=16/32/64)")]
#[command(
    long_about = "Rype: High-performance genomic sequence classification using minimizer-based k-mer sketching in RY (purine/pyrimidine) space.

WORKFLOW:
  1. Create an index:     rype index create -o index.ryxdi -r refs.fasta
  2. Classify reads:      rype classify run -i index.ryxdi -1 reads.fq

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
  rype index create -o bacteria.ryxdi -r genome1.fna -r genome2.fna -k 64 -w 50

  # Create index with one bucket per sequence
  rype index create -o genes.ryxdi -r genes.fasta --separate-buckets

  # Classify single-end reads
  rype classify run -i bacteria.ryxdi -1 reads.fq -t 0.1 -o results.tsv

  # Classify paired-end reads with negative filtering
  rype classify run -i bacteria.ryxdi -N host.ryxdi -1 R1.fq -2 R2.fq -t 0.1

  # Aggregate mode for higher sensitivity
  rype classify aggregate -i bacteria.ryxdi -1 R1.fq -2 R2.fq -t 0.05")]
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
  rype index create -o index.ryxdi -r genome.fasta

  # Multiple references, all in one bucket
  rype index create -o index.ryxdi -r chr1.fa -r chr2.fa

  # One bucket per sequence (e.g., for gene-level classification)
  rype index create -o genes.ryxdi -r genes.fasta --separate-buckets

  # Large index with sharding (for memory-constrained systems)
  rype index create -o large.ryxdi -r refs.fa --max-shard-size 1073741824")]
    Create {
        /// Output index path (.ryxdi directory will be created)
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

        /// Row group size (rows per group). Larger = better compression.
        #[arg(long, default_value_t = 100_000)]
        row_group_size: usize,

        /// Use Zstd compression instead of Snappy for Parquet files.
        /// Better compression ratio but slower.
        #[arg(long)]
        zstd: bool,

        /// Enable bloom filters for faster lookups.
        /// Increases file size slightly.
        #[arg(long)]
        bloom_filter: bool,

        /// Bloom filter false positive probability (0.0-1.0).
        /// Lower = more accurate but larger files. Only used with --bloom-filter.
        #[arg(long, default_value = "0.05", value_parser = parse_bloom_fpp)]
        bloom_fpp: f64,

        /// Print timing diagnostics to stderr for performance analysis.
        #[arg(long)]
        timing: bool,
    },

    /// Show index statistics and bucket information
    Stats {
        /// Path to index directory (.ryxdi)
        #[arg(short, long)]
        index: PathBuf,
    },

    /// Show source file paths or sequence IDs for a bucket
    BucketSourceDetail {
        /// Path to index directory (.ryxdi)
        #[arg(short, long)]
        index: PathBuf,

        /// Bucket identifier: numeric ID (e.g., '1') or exact bucket name (case-sensitive).
        /// Numeric IDs take precedence - if a bucket is named '42', use its numeric ID instead.
        #[arg(short, long, required = true)]
        bucket: String,

        /// Show only unique file paths (one per line)
        #[arg(long)]
        paths: bool,

        /// Show only bucket IDs (for scripting)
        #[arg(long)]
        ids: bool,
    },

    /// Add a new reference file as a new bucket to an existing index (development pending)
    BucketAdd {
        /// Path to existing index directory
        #[arg(short, long)]
        index: PathBuf,

        /// Reference file to add (creates a new bucket)
        #[arg(short, long)]
        reference: PathBuf,
    },

    /// Build index from a TOML configuration file (see CONFIG FORMAT below)
    #[command(after_help = "CONFIG FORMAT (from-config):
  [index]
  k = 64                           # K-mer size (16, 32, or 64)
  window = 50                      # Minimizer window size
  salt = 0x5555555555555555        # Hash salt (hex)
  output = \"index.ryxdi\"           # Output path (directory will be created)
  orient_sequences = true          # Optional: orient sequences for better overlap

  [buckets.BucketName]             # Define a bucket
  files = [\"ref1.fa\", \"ref2.fa\"]   # Files for this bucket

  [buckets.AnotherBucket]
  files = [\"other.fasta\"]

CLI OPTIONS OVERRIDE CONFIG FILE:
  --max-memory controls memory budget (auto-detected if not specified)
  --orient overrides [index].orient_sequences

SUBTRACTION MODE (--subtract-from):
  Removes minimizers present in an existing index from all buckets during build.
  Useful for host depletion: build a non-host index in one step.
  The subtraction index must have matching k, w, and salt values.

  Example: rype index from-config -c config.toml --subtract-from host.ryxdi")]
    FromConfig {
        /// Path to TOML config file
        #[arg(short, long)]
        config: PathBuf,

        /// Maximum memory to use (e.g., "8G", "512M", "auto").
        /// Controls chunk sizes for input processing.
        /// Default: auto-detect from system/cgroups/SLURM.
        #[arg(long, default_value = "auto", value_parser = parse_max_memory_arg)]
        max_memory: usize,

        /// Row group size (rows per group). Larger = better compression.
        #[arg(long, default_value_t = 100_000)]
        row_group_size: usize,

        /// Enable bloom filters for faster lookups.
        #[arg(long)]
        bloom_filter: bool,

        /// Bloom filter false positive probability (0.0-1.0).
        /// Only used with --bloom-filter.
        #[arg(long, default_value = "0.05", value_parser = parse_bloom_fpp)]
        bloom_fpp: f64,

        /// Orient sequences within buckets to maximize minimizer overlap.
        /// First sequence establishes baseline; subsequent sequences use
        /// forward or reverse-complement based on which has higher overlap.
        #[arg(long)]
        orient: bool,

        /// Print timing diagnostics to stderr for performance analysis.
        #[arg(long)]
        timing: bool,

        /// Subtract minimizers from an existing index before building.
        /// Removes any minimizer that exists in the subtraction index.
        /// Useful for host depletion: build a non-host index in one step.
        /// The subtraction index must have the same k, w, and salt values.
        #[arg(long)]
        subtract_from: Option<PathBuf>,
    },

    /// Add files to existing index using TOML config (development pending)
    #[command(after_help = "CONFIG FORMAT (bucket-add-config):
  [target]
  index = \"existing.ryxdi\"         # Index to modify

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

    /// Show detailed minimizer statistics for compression analysis
    Summarize {
        /// Path to index directory (.ryxdi)
        #[arg(short, long)]
        index: PathBuf,
    },

    /// Merge two indices into one
    #[command(after_help = "MERGE OPERATION:
  Combines all buckets from both indices into a single output index.
  Bucket IDs are renumbered sequentially (1, 2, 3...) with primary buckets first.

REQUIREMENTS:
  - Both indices must have the same k, w, and salt values
  - Bucket names must be unique across both indices (no duplicates)

SUBTRACTION MODE (--subtract-from-primary):
  When enabled, minimizers present in the primary index are removed from
  the secondary index before merging. This is useful for creating indices
  where secondary buckets only contain sequences NOT found in primary.

  Use case: Create a \"non-host\" index by subtracting host minimizers.

EXAMPLES:
  # Simple merge of two indices
  rype index merge --index-primary bacteria.ryxdi --index-secondary phage.ryxdi -o combined.ryxdi

  # Merge with subtraction (create non-host index)
  rype index merge --index-primary host.ryxdi --index-secondary sample.ryxdi \\
      -o non_host.ryxdi --subtract-from-primary

  # Merge with compression options
  rype index merge --index-primary idx1.ryxdi --index-secondary idx2.ryxdi \\
      -o merged.ryxdi --zstd --bloom-filter")]
    Merge {
        /// Path to primary index directory (.ryxdi)
        #[arg(long)]
        index_primary: PathBuf,

        /// Path to secondary index directory (.ryxdi)
        #[arg(long)]
        index_secondary: PathBuf,

        /// Output path for merged index (.ryxdi directory will be created)
        #[arg(short, long)]
        output: PathBuf,

        /// Remove minimizers from secondary that exist in primary.
        /// Useful for creating indices where secondary buckets contain
        /// only sequences NOT found in the primary index.
        #[arg(long)]
        subtract_from_primary: bool,

        /// Maximum memory to use for merge operations (e.g., "8G", "512M", or "auto").
        /// When "auto", detects available system memory.
        /// Memory-bounded merging processes secondary shards one at a time to
        /// avoid OOM on large indices with high overlap.
        #[arg(long, default_value = "auto", value_parser = parse_max_memory_arg)]
        max_memory: usize,

        /// Row group size (rows per group). Larger = better compression.
        #[arg(long, default_value_t = 100_000)]
        row_group_size: usize,

        /// Use Zstd compression instead of Snappy for Parquet files.
        #[arg(long)]
        zstd: bool,

        /// Enable bloom filters for faster lookups.
        #[arg(long)]
        bloom_filter: bool,

        /// Bloom filter false positive probability (0.0-1.0).
        #[arg(long, default_value = "0.05", value_parser = parse_bloom_fpp)]
        bloom_fpp: f64,

        /// Print timing diagnostics to stderr for performance analysis.
        #[arg(long)]
        timing: bool,
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
        /// Path to target index directory (.ryxdi)
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

        /// Use parallel row group processing.
        /// Processes each row group independently in parallel, maximizing CPU utilization.
        /// Most effective when query minimizers span entire index range (row group
        /// filtering is ineffective).
        #[arg(long)]
        parallel_rg: bool,

        /// Use bloom filters for row group filtering.
        /// Reduces I/O by rejecting row groups that definitely don't contain query minimizers.
        /// Only effective if index was built with --bloom-filter.
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

        /// Report only the single best hit per query.
        /// If multiple buckets tie for best score, one is chosen arbitrarily.
        #[arg(long)]
        best_hit: bool,

        /// Trim sequences to first N nucleotides before classification.
        /// Sequences shorter than N are skipped.
        ///
        /// For paired-end reads, R1 must be at least N bases (pairs with shorter R1 are skipped).
        /// R2 is trimmed to min(length, N) - a short R2 does not cause the pair to be skipped.
        ///
        /// Recommended: N >= k (k-mer size) to ensure minimizer extraction.
        /// Values smaller than k will produce no minimizers and yield no results.
        #[arg(long, value_parser = validate_trim_to)]
        trim_to: Option<usize>,

        /// Output wide-form matrix instead of long-form TSV.
        /// Columns: read_id, then one column per bucket (ordered by bucket_id).
        /// Each row contains scores for all buckets (0.0 if no hit).
        /// Incompatible with --threshold (all scores must be reported).
        #[arg(long)]
        wide: bool,
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
        /// Path to target index directory (.ryxdi)
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

    /// Compute log10(numerator_score / denominator_score) using two single-bucket indices
    #[command(after_help = "LOG-RATIO MODE:
  Computes log10(numerator_score / denominator_score) for each read using
  two separate single-bucket indices.

  Reads are first classified against the numerator index. Reads with zero
  numerator score are assigned -inf (fast path). If --numerator-skip-threshold
  is set, reads exceeding that threshold are assigned +inf (fast path).
  Only remaining reads are classified against the denominator index.

OUTPUT FORMAT:
  Tab-separated: read_id<TAB>log10([Num] / [Denom])<TAB>score<TAB>fast_path

  fast_path column: 'none' (exact), 'num_zero' (-inf), 'num_high' (+inf)

EDGE CASES:
  - numerator = 0 → -inf (fast path: num_zero)
  - denominator = 0, numerator > 0 → +inf
  - both = 0 → -inf (fast path: num_zero)

EXAMPLES:
  # Basic log-ratio with two indices
  rype classify log-ratio -n numerator.ryxdi -d denominator.ryxdi -1 reads.fq

  # With fast-path skip threshold
  rype classify log-ratio -n num.ryxdi -d denom.ryxdi -1 reads.fq --numerator-skip-threshold 0.01")]
    LogRatio {
        /// Path to numerator index directory (.ryxdi). Must have exactly 1 bucket.
        #[arg(short = 'n', long)]
        numerator: PathBuf,

        /// Path to denominator index directory (.ryxdi). Must have exactly 1 bucket.
        #[arg(short = 'd', long)]
        denominator: PathBuf,

        /// Forward reads. Formats: FASTA/FASTQ (.fa/.fq, optionally .gz),
        /// Parquet (.parquet) with columns: read_id, sequence1, sequence2 (optional)
        #[arg(short = '1', long)]
        r1: PathBuf,

        /// Reverse reads for paired-end data (optional).
        /// Not supported with Parquet input - use sequence2 column instead.
        #[arg(short = '2', long)]
        r2: Option<PathBuf>,

        /// Maximum memory to use (e.g., "4G", "512M", "auto").
        #[arg(long, default_value = "auto", value_parser = parse_max_memory_arg)]
        max_memory: usize,

        /// Override automatic batch size calculation.
        #[arg(short, long)]
        batch_size: Option<usize>,

        /// Output file path. Format auto-detected from extension:
        /// - `.tsv` or no extension: Plain TSV
        /// - `.tsv.gz`: Gzip-compressed TSV
        /// - `.parquet`: Apache Parquet with zstd compression
        /// - `-`: stdout (TSV)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Use parallel row group processing.
        #[arg(long)]
        parallel_rg: bool,

        /// Use bloom filters for row group filtering.
        #[arg(long)]
        use_bloom_filter: bool,

        /// Enable parallel row group reading for Parquet input files.
        #[arg(long, default_value_t = 0)]
        parallel_input_rg: usize,

        /// Print timing diagnostics to stderr.
        #[arg(long)]
        timing: bool,

        /// Trim sequences to first N nucleotides before classification.
        #[arg(long, value_parser = validate_trim_to)]
        trim_to: Option<usize>,

        /// Output passing sequences to gzipped FASTA/FASTQ.
        /// For paired-end: foo.fastq.gz creates foo.R1.fastq.gz and foo.R2.fastq.gz.
        /// Not supported with Parquet input or --trim-to.
        #[arg(long)]
        output_sequences: Option<PathBuf>,

        /// Pass sequences with POSITIVE log-ratio; excludes zero (default: pass NEGATIVE/zero).
        /// Requires --output-sequences.
        #[arg(long, requires = "output_sequences")]
        passing_is_positive: bool,

        /// Skip denominator classification for reads with numerator score >= this value.
        /// These reads are assigned +inf with fast_path=num_high.
        /// Must be between 0.0 (exclusive) and 1.0 (inclusive).
        #[arg(long)]
        numerator_skip_threshold: Option<f64>,
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
