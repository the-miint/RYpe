//! `rype classify filter`: single-pass read filtering against an index.
//!
//! Streams FASTX input through the same accumulate-then-scan classification
//! passes as `classify run`, but instead of reporting hits it writes the reads
//! themselves: by default those with NO bucket score >= threshold (host
//! depletion), or with `--invert` only those that do. Each read's header,
//! sequence and quality ride along as the accumulator's per-read metadata, so
//! the pass byte budget accounts for them and a pass's reads are written as
//! soon as that pass is classified. Input batches are sized for reads that
//! carry qualities, and the budget is checked after every read, so peak memory
//! stays within `--max-memory`. There is no second walk over the input
//! (unlike `log-ratio --output-sequences`) and no intermediate hit table, so
//! the output can be piped straight into a downstream aligner.
//!
//! Paired input is written interleaved to `--output` as `<id>/1`, `<id>/2`
//! (the layout `minimap2 -x sr` and `samtools fastq` expect), or split across
//! `--output` and `--output2`. A pair is kept or dropped as a unit, matching
//! how rype scores R1+R2 as one query. Headers are written as the base read id
//! the input reader extracts (first whitespace token, `/1` `/2` stripped), so
//! header comments are not carried through.

use anyhow::{anyhow, Result};
use flate2::write::GzEncoder;
use flate2::Compression;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use rype::memory::{format_bytes, FASTX_PREFETCH_BUFFER_SLOTS, MIN_BATCH_SIZE};
use rype::{
    classify_from_query_index, classify_from_query_index_parallel_rg, extract_batch_minimizers,
    log_timing, MetaHeapBytes, ParquetReadOptions, QueryAccumulator, ShardedInvertedIndex,
};

use super::helpers::{
    cap_input_batch, compute_effective_batch_size, is_parquet_input, load_index_for_classification,
    BatchSizeConfig, IndexLoadOptions, OutputFormat, OwnedFastxRecord, PrefetchingIoHandler,
};

/// Arguments for the classify filter command.
pub struct ClassifyFilterArgs {
    pub index: PathBuf,
    pub r1: PathBuf,
    pub r2: Option<PathBuf>,
    pub threshold: f64,
    pub max_memory: usize,
    pub batch_size: Option<usize>,
    pub output: Option<PathBuf>,
    pub output2: Option<PathBuf>,
    pub invert: bool,
    pub parallel_rg: bool,
    pub use_bloom_filter: bool,
}

/// One read (or pair) held in the accumulator until its pass is classified.
struct BufferedRead {
    header: String,
    rec: OwnedFastxRecord,
}

impl MetaHeapBytes for BufferedRead {
    fn heap_bytes(&self) -> usize {
        let r = &self.rec;
        self.header.capacity()
            + r.seq1.capacity()
            + r.qual1.as_ref().map_or(0, Vec::capacity)
            + r.seq2.as_ref().map_or(0, Vec::capacity)
            + r.qual2.as_ref().map_or(0, Vec::capacity)
    }
}

const SINK_BUF_BYTES: usize = 1 << 20;

/// Input batches resident at once: those queued in the prefetch channel, one
/// being read by the prefetch thread and one being pushed into the accumulator.
const INPUT_BATCHES_IN_FLIGHT: usize = FASTX_PREFETCH_BUFFER_SLOTS + 2;

/// Header bytes per read assumed when sizing input batches (Illumina read ids
/// run about 40-60 bytes).
const EST_HEADER_BYTES: usize = 64;

/// Output destination: stdout/plain file, or gzip when the path ends in `.gz`.
/// Gzip is single-threaded; for large outputs pipe stdout through a parallel
/// compressor (pigz/bgzip) instead.
///
/// The buffer sits in front of the gzip encoder, which runs the compressor on
/// every write call it gets: records arrive as many small writes.
#[allow(clippy::large_enum_variant)]
enum Sink {
    Plain(BufWriter<Box<dyn Write>>),
    Gzip(BufWriter<GzEncoder<File>>),
}

impl Sink {
    fn open(path: Option<&PathBuf>) -> Result<Self> {
        let path = match path {
            Some(p) if !OutputFormat::is_stdout(Some(p)) => p,
            _ => {
                let stdout: Box<dyn Write> = Box::new(io::stdout());
                return Ok(Sink::Plain(BufWriter::with_capacity(
                    SINK_BUF_BYTES,
                    stdout,
                )));
            }
        };
        let file = File::create(path)
            .map_err(|e| anyhow!("Failed to create output {}: {}", path.display(), e))?;
        if path.extension().is_some_and(|ext| ext == "gz") {
            let encoder = GzEncoder::new(file, Compression::fast());
            Ok(Sink::Gzip(BufWriter::with_capacity(
                SINK_BUF_BYTES,
                encoder,
            )))
        } else {
            let file: Box<dyn Write> = Box::new(file);
            Ok(Sink::Plain(BufWriter::with_capacity(SINK_BUF_BYTES, file)))
        }
    }

    fn writer(&mut self) -> &mut dyn Write {
        match self {
            Sink::Plain(w) => w,
            Sink::Gzip(w) => w,
        }
    }

    fn finish(self) -> Result<()> {
        match self {
            Sink::Plain(mut w) => w.flush()?,
            Sink::Gzip(w) => w
                .into_inner()
                .map_err(|e| e.into_error())?
                .finish()?
                .flush()?,
        }
        Ok(())
    }
}

/// Write one FASTQ record, or FASTA when the input carried no qualities.
fn write_record(
    w: &mut dyn Write,
    header: &str,
    mate_suffix: &[u8],
    seq: &[u8],
    qual: Option<&[u8]>,
) -> io::Result<()> {
    w.write_all(if qual.is_some() { b"@" } else { b">" })?;
    w.write_all(header.as_bytes())?;
    w.write_all(mate_suffix)?;
    w.write_all(b"\n")?;
    w.write_all(seq)?;
    if let Some(q) = qual {
        w.write_all(b"\n+\n")?;
        w.write_all(q)?;
    }
    w.write_all(b"\n")
}

#[derive(Default)]
struct FilterStats {
    reads_in: usize,
    reads_matched: usize,
    reads_written: usize,
    passes: usize,
}

/// Classify everything buffered in `acc` and write the reads that pass the
/// filter, in input order.
#[allow(clippy::too_many_arguments)]
fn flush_pass(
    acc: &mut QueryAccumulator<BufferedRead>,
    sharded: &ShardedInvertedIndex,
    parallel_rg: bool,
    threshold: f64,
    read_options: Option<&ParquetReadOptions>,
    invert: bool,
    out1: &mut Sink,
    mut out2: Option<&mut Sink>,
    stats: &mut FilterStats,
) -> Result<()> {
    if acc.is_empty() {
        return Ok(());
    }
    let (query_idx, reads) = acc.drain();
    let query_ids: Vec<i64> = (0..reads.len() as i64).collect();
    let hits = if parallel_rg {
        classify_from_query_index_parallel_rg(
            sharded,
            &query_idx,
            &query_ids,
            threshold,
            read_options,
        )?
    } else {
        classify_from_query_index(sharded, &query_idx, &query_ids, threshold, read_options)?
    };
    drop(query_idx);

    // Hits are already thresholded; a read matched if it scored in any bucket.
    let mut matched = vec![false; reads.len()];
    for hit in &hits {
        matched[hit.query_id as usize] = true;
    }
    drop(hits);
    let matched_count = matched.iter().filter(|&&m| m).count();
    let written = if invert {
        matched_count
    } else {
        reads.len() - matched_count
    };

    let t_write = Instant::now();
    for (read, &is_match) in reads.iter().zip(&matched) {
        if is_match != invert {
            continue;
        }
        let rec = &read.rec;
        let mate1: &[u8] = if rec.seq2.is_some() { b"/1" } else { b"" };
        write_record(
            out1.writer(),
            &read.header,
            mate1,
            &rec.seq1,
            rec.qual1.as_deref(),
        )?;
        if let Some(seq2) = &rec.seq2 {
            // R2 goes to --output2 when given, otherwise interleaved after R1.
            let w = match out2.as_deref_mut() {
                Some(o2) => o2.writer(),
                None => out1.writer(),
            };
            write_record(w, &read.header, b"/2", seq2, rec.qual2.as_deref())?;
        }
    }
    log_timing("filter: write", t_write.elapsed().as_millis());

    stats.passes += 1;
    stats.reads_in += reads.len();
    stats.reads_matched += matched_count;
    stats.reads_written += written;
    log::info!(
        "Filter pass {}: {} reads, {} matched, {} written",
        stats.passes,
        reads.len(),
        matched_count,
        written
    );
    Ok(())
}

pub fn run_filter(args: ClassifyFilterArgs) -> Result<()> {
    if is_parquet_input(&args.r1) {
        return Err(anyhow!(
            "classify filter writes reads back out as FASTX and only accepts FASTA/FASTQ input"
        ));
    }
    if args.output2.is_some() && args.r2.is_none() {
        return Err(anyhow!("--output2 requires paired input (-2/--r2)"));
    }

    // Always sized from --max-memory: -b only caps the reads per pass here,
    // it does not lift the byte budget as it does for `classify run`.
    let batch_result = compute_effective_batch_size(&BatchSizeConfig {
        batch_size_override: None,
        max_memory: args.max_memory,
        r1_path: &args.r1,
        r2_path: args.r2.as_deref(),
        is_parquet_input: false,
        index_path: &args.index,
        trim_to: None,
        minimum_length: None,
        is_log_ratio: false,
        denominator_index_path: None,
    })?;

    let loaded_index = load_index_for_classification(
        &args.index,
        &IndexLoadOptions {
            use_bloom_filter: args.use_bloom_filter,
            parallel_rg: args.parallel_rg,
        },
    )?;
    let sharded = loaded_index.sharded;
    let read_options = loaded_index.read_options;
    let manifest = sharded.manifest();
    let (idx_k, idx_w, idx_salt) = (manifest.k, manifest.w, manifest.salt);

    // Same per-read classification-pass reservation as `run_classify`; the
    // buffered sequences themselves are counted through `MetaHeapBytes`.
    let max_bucket_id = loaded_index
        .metadata
        .bucket_names
        .keys()
        .max()
        .copied()
        .unwrap_or(0);
    let accumulator_bytes_per_read =
        rype::memory::estimate_accumulator_bytes_per_read(max_bucket_id as usize).unwrap_or(0);
    let mut acc: QueryAccumulator<BufferedRead> =
        QueryAccumulator::with_budget(batch_result.classify_byte_budget)
            .with_max_reads(args.batch_size)
            .with_accumulator_cost_per_read(accumulator_bytes_per_read);

    // The auto batch size assumes reads without qualities that are dropped
    // once extracted. Here every read in flight also carries its qualities,
    // so shrink the batch until INPUT_BATCHES_IN_FLIGHT of them fit in the
    // input reservation the pass budget was computed net of, and no larger
    // than MAX_INPUT_BATCH_BYTES. Reads already pushed are counted in the
    // pass budget itself.
    let read_bytes =
        std::mem::size_of::<BufferedRead>() + EST_HEADER_BYTES + 2 * batch_result.avg_query_length;
    let batch_size = (batch_result.input_batch_residency / (INPUT_BATCHES_IN_FLIGHT * read_bytes))
        .max(MIN_BATCH_SIZE)
        .min(cap_input_batch(batch_result.batch_size, read_bytes));
    let batch_size = args.batch_size.map_or(batch_size, |bs| bs.min(batch_size));
    log::info!(
        "Filter pass budget: {} (input batch_size={}, threshold={}, keep={})",
        format_bytes(batch_result.classify_byte_budget as u64),
        batch_size,
        args.threshold,
        if args.invert { "matched" } else { "unmatched" }
    );

    // Keep qualities so reads can be written back out as FASTQ.
    let mut io = PrefetchingIoHandler::with_options(
        &args.r1,
        args.r2.as_ref(),
        None,
        batch_size,
        None,
        None,
        true,
    )?;

    let mut out1 = Sink::open(args.output.as_ref())?;
    let mut out2 = args
        .output2
        .as_ref()
        .map(|p| Sink::open(Some(p)))
        .transpose()?;
    let mut stats = FilterStats::default();
    let mut flush = |acc: &mut QueryAccumulator<BufferedRead>| {
        flush_pass(
            acc,
            &sharded,
            args.parallel_rg,
            args.threshold,
            read_options.as_ref(),
            args.invert,
            &mut out1,
            out2.as_mut(),
            &mut stats,
        )
    };

    loop {
        let t_io_read = Instant::now();
        let Some((records, headers)) = io.next_batch()? else {
            break;
        };
        log_timing("batch: io_read", t_io_read.elapsed().as_millis());

        let t_extract = Instant::now();
        let batch_refs: Vec<rype::QueryRecord> = records
            .iter()
            .map(|rec| (rec.query_id, rec.seq1.as_slice(), rec.seq2.as_deref()))
            .collect();
        let extracted = extract_batch_minimizers(idx_k, idx_w, idx_salt, None, &batch_refs);
        drop(batch_refs);
        log_timing("batch: extract", t_extract.elapsed().as_millis());

        for ((header, rec), (fwd, rc)) in headers.into_iter().zip(records).zip(extracted) {
            // A pair is written as one format; mixing would interleave `@` and
            // `>` records that downstream tools reject.
            if rec.seq2.is_some() && rec.qual1.is_some() != rec.qual2.is_some() {
                return Err(anyhow!(
                    "R1 and R2 must both be FASTQ or both be FASTA (read {})",
                    header
                ));
            }
            acc.push(BufferedRead { header, rec }, fwd, rc);
            // Checked per read rather than per batch, so a pass never runs
            // past its budget by more than one read.
            if acc.should_flush() {
                flush(&mut acc)?;
            }
        }
    }
    flush(&mut acc)?;

    out1.finish()?;
    if let Some(o2) = out2 {
        o2.finish()?;
    }
    io.finish()?;

    // Always reported (not gated on -v) so wrapper scripts can log counts.
    eprintln!(
        "rype classify filter: reads_in={} matched={} written={} passes={} threshold={}",
        stats.reads_in, stats.reads_matched, stats.reads_written, stats.passes, args.threshold
    );
    Ok(())
}
