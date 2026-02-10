//! Sequence output writer for filtered FASTA/FASTQ sequences.
//!
//! This module provides gzip-compressed sequence output for the `--output-sequences`
//! feature in `classify log-ratio`.

use anyhow::{anyhow, Result};
use flate2::write::GzEncoder;
use flate2::Compression;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use super::fastx_io::OwnedFastxRecord;

/// Output sequence format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeqFormat {
    FastaGz,
    FastqGz,
}

impl SeqFormat {
    /// Detect format from file extension.
    ///
    /// # Errors
    /// Returns an error if the extension is not recognized.
    pub fn detect(path: &Path) -> Result<Self> {
        let s = path.to_string_lossy().to_lowercase();
        if s.ends_with(".fasta.gz") || s.ends_with(".fa.gz") {
            Ok(Self::FastaGz)
        } else if s.ends_with(".fastq.gz") || s.ends_with(".fq.gz") {
            Ok(Self::FastqGz)
        } else {
            Err(anyhow!(
                "Output must end with .fasta.gz, .fa.gz, .fastq.gz, or .fq.gz"
            ))
        }
    }
}

/// Gzip-compressed sequence writer for FASTA or FASTQ output.
///
/// Handles both single-end and paired-end output. For paired-end, creates
/// separate R1 and R2 files (e.g., `out.R1.fastq.gz` and `out.R2.fastq.gz`).
pub struct SequenceWriter {
    r1: BufWriter<GzEncoder<File>>,
    r2: Option<BufWriter<GzEncoder<File>>>,
    format: SeqFormat,
}

impl SequenceWriter {
    /// Compute output paths for single-end or paired-end output.
    ///
    /// For paired-end, splits `foo.fastq.gz` into `foo.R1.fastq.gz` and `foo.R2.fastq.gz`.
    pub fn compute_paths(path: &Path, paired: bool) -> (PathBuf, Option<PathBuf>) {
        if !paired {
            return (path.to_path_buf(), None);
        }

        let s = path.to_string_lossy();
        let (base, ext) = if s.ends_with(".fastq.gz") {
            (s.trim_end_matches(".fastq.gz"), ".fastq.gz")
        } else if s.ends_with(".fq.gz") {
            (s.trim_end_matches(".fq.gz"), ".fq.gz")
        } else if s.ends_with(".fasta.gz") {
            (s.trim_end_matches(".fasta.gz"), ".fasta.gz")
        } else if s.ends_with(".fa.gz") {
            (s.trim_end_matches(".fa.gz"), ".fa.gz")
        } else {
            // Fallback: append .R1/.R2 before .gz
            let base = s.trim_end_matches(".gz");
            return (
                PathBuf::from(format!("{}.R1.gz", base)),
                Some(PathBuf::from(format!("{}.R2.gz", base))),
            );
        };

        (
            PathBuf::from(format!("{}.R1{}", base, ext)),
            Some(PathBuf::from(format!("{}.R2{}", base, ext))),
        )
    }

    /// Create a new sequence writer.
    ///
    /// # Arguments
    /// * `path` - Output path. For paired-end, R1/R2 paths are computed from this.
    /// * `paired` - Whether this is paired-end data.
    ///
    /// # Errors
    /// Returns an error if files cannot be created or format cannot be detected.
    pub fn new(path: &Path, paired: bool) -> Result<Self> {
        let format = SeqFormat::detect(path)?;
        let (r1_path, r2_path) = Self::compute_paths(path, paired);

        let r1_file = File::create(&r1_path)?;
        let r1_encoder = GzEncoder::new(r1_file, Compression::default());
        let r1 = BufWriter::new(r1_encoder);

        let r2 = if let Some(ref p) = r2_path {
            let r2_file = File::create(p)?;
            let r2_encoder = GzEncoder::new(r2_file, Compression::default());
            Some(BufWriter::new(r2_encoder))
        } else {
            None
        };

        Ok(Self { r1, r2, format })
    }

    /// Write a record to the output file(s).
    ///
    /// The header is passed separately because it lives in the batch's `Vec<String>`
    /// (indexed by `rec.query_id`), not in the record struct itself. This avoids
    /// duplicate storage of header data.
    ///
    /// For paired-end records, writes R1 to the first file and R2 to the second.
    /// For single-end, writes only to the first file.
    ///
    /// # Errors
    /// Returns an error if writing fails.
    pub fn write_record(&mut self, rec: &OwnedFastxRecord, header: &[u8]) -> Result<()> {
        match self.format {
            SeqFormat::FastaGz => self.write_fasta(rec, header),
            SeqFormat::FastqGz => self.write_fastq(rec, header),
        }
    }

    /// Write a record in FASTA format.
    fn write_fasta(&mut self, rec: &OwnedFastxRecord, header: &[u8]) -> Result<()> {
        // Write R1
        self.r1.write_all(b">")?;
        self.r1.write_all(header)?;
        self.r1.write_all(b"\n")?;
        self.r1.write_all(&rec.seq1)?;
        self.r1.write_all(b"\n")?;

        // Write R2 if paired
        if let (Some(ref mut r2_writer), Some(ref seq2)) = (&mut self.r2, &rec.seq2) {
            r2_writer.write_all(b">")?;
            r2_writer.write_all(header)?;
            r2_writer.write_all(b"\n")?;
            r2_writer.write_all(seq2)?;
            r2_writer.write_all(b"\n")?;
        }

        Ok(())
    }

    /// Write a record in FASTQ format.
    fn write_fastq(&mut self, rec: &OwnedFastxRecord, header: &[u8]) -> Result<()> {
        // Write R1
        self.r1.write_all(b"@")?;
        self.r1.write_all(header)?;
        self.r1.write_all(b"\n")?;
        self.r1.write_all(&rec.seq1)?;
        self.r1.write_all(b"\n+\n")?;

        // Write quality or placeholder
        if let Some(ref qual) = rec.qual1 {
            self.r1.write_all(qual)?;
        } else {
            // No quality scores - write placeholder (I = Phred 40)
            let placeholder = vec![b'I'; rec.seq1.len()];
            self.r1.write_all(&placeholder)?;
        }
        self.r1.write_all(b"\n")?;

        // Write R2 if paired
        if let (Some(ref mut r2_writer), Some(ref seq2)) = (&mut self.r2, &rec.seq2) {
            r2_writer.write_all(b"@")?;
            r2_writer.write_all(header)?;
            r2_writer.write_all(b"\n")?;
            r2_writer.write_all(seq2)?;
            r2_writer.write_all(b"\n+\n")?;

            if let Some(ref qual) = rec.qual2 {
                r2_writer.write_all(qual)?;
            } else {
                let placeholder = vec![b'I'; seq2.len()];
                r2_writer.write_all(&placeholder)?;
            }
            r2_writer.write_all(b"\n")?;
        }

        Ok(())
    }

    /// Flush buffers and finish writing.
    ///
    /// This consumes the writer to ensure no further writes are possible.
    ///
    /// # Errors
    /// Returns an error if flushing fails.
    pub fn finish(mut self) -> Result<()> {
        self.r1.flush()?;
        // Explicitly finish the gzip encoder
        let encoder = self.r1.into_inner()?;
        encoder.finish()?;

        if let Some(mut r2) = self.r2 {
            r2.flush()?;
            let encoder = r2.into_inner()?;
            encoder.finish()?;
        }

        Ok(())
    }
}

// ============================================================================
// Post-classification re-walk
// ============================================================================

/// Re-walk the input file and write sequences that are marked as passing in the tracker.
///
/// This is used after classification completes: during classification, only a compact
/// bitset tracks which reads pass the filter. After classification, this function
/// re-reads the input file and writes passing sequences to gzipped output.
///
/// Supports both FASTX (FASTA/FASTQ) and Parquet input. For Parquet input, sequences
/// are read from `read_id`, `sequence1`, and optionally `sequence2` columns.
pub fn rewalk_and_write_passing(
    r1_path: &Path,
    r2_path: Option<&Path>,
    is_parquet: bool,
    tracker: &super::passing_tracker::PassingReadTracker,
    output_path: &Path,
    paired: bool,
    expected_total_reads: usize,
) -> Result<usize> {
    if is_parquet {
        rewalk_parquet(r1_path, tracker, output_path, paired, expected_total_reads)
    } else {
        rewalk_fastx(
            r1_path,
            r2_path,
            tracker,
            output_path,
            paired,
            expected_total_reads,
        )
    }
}

/// Re-walk a FASTX input file, writing passing sequences.
fn rewalk_fastx(
    r1_path: &Path,
    r2_path: Option<&Path>,
    tracker: &super::passing_tracker::PassingReadTracker,
    output_path: &Path,
    paired: bool,
    expected_total_reads: usize,
) -> Result<usize> {
    use super::fastx_io::PrefetchingIoHandler;

    let mut reader1 = needletail::parse_fastx_file(r1_path)
        .map_err(|e| anyhow!("Failed to open R1 for re-walk: {}", e))?;
    let mut reader2 = r2_path
        .map(|p| {
            needletail::parse_fastx_file(p)
                .map_err(|e| anyhow!("Failed to open R2 for re-walk: {}", e))
        })
        .transpose()?;

    let mut writer = SequenceWriter::new(output_path, paired)?;
    let mut global_index = 0usize;
    let mut written = 0usize;

    loop {
        let Some(rec1_result) = reader1.next() else {
            break;
        };
        let rec1 = rec1_result.map_err(|e| anyhow!("Error reading R1 record: {}", e))?;

        let rec2 = if let Some(ref mut r2) = reader2 {
            let r2_result = r2
                .next()
                .ok_or_else(|| anyhow!("R2 has fewer records than R1"))?;
            Some(r2_result.map_err(|e| anyhow!("Error reading R2 record: {}", e))?)
        } else {
            None
        };

        if tracker.is_passing(global_index) {
            let header = PrefetchingIoHandler::base_read_id(rec1.id());
            let owned = OwnedFastxRecord::new(
                0,
                rec1.seq().to_vec(),
                rec1.qual().map(|q| q.to_vec()),
                rec2.as_ref().map(|r| r.seq().to_vec()),
                rec2.as_ref().and_then(|r| r.qual().map(|q| q.to_vec())),
            );
            writer.write_record(&owned, header)?;
            written += 1;
        }

        global_index += 1;
    }

    if global_index != expected_total_reads {
        return Err(anyhow!(
            "Input file changed between classification and re-walk: \
             expected {} reads, found {}",
            expected_total_reads,
            global_index
        ));
    }

    writer.finish()?;
    Ok(written)
}

/// Re-walk a Parquet input file, writing passing sequences.
fn rewalk_parquet(
    parquet_path: &Path,
    tracker: &super::passing_tracker::PassingReadTracker,
    output_path: &Path,
    paired: bool,
    expected_total_reads: usize,
) -> Result<usize> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let file = File::open(parquet_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut writer = SequenceWriter::new(output_path, paired)?;
    let mut global_index = 0usize;
    let mut written = 0usize;

    for batch_result in reader {
        let batch = batch_result?;
        let num_rows = batch.num_rows();

        let read_id_col = batch
            .column_by_name("read_id")
            .ok_or_else(|| anyhow!("Missing read_id column in Parquet"))?;
        let seq1_col = batch
            .column_by_name("sequence1")
            .ok_or_else(|| anyhow!("Missing sequence1 column in Parquet"))?;
        let seq2_col = if paired {
            batch.column_by_name("sequence2")
        } else {
            None
        };

        for row in 0..num_rows {
            if tracker.is_passing(global_index) {
                let header = get_string_value(read_id_col, row)?;
                let seq1 = get_string_value(seq1_col, row)?;
                let seq2 = seq2_col.map(|col| get_string_value(col, row)).transpose()?;

                let owned = OwnedFastxRecord::new(
                    0,
                    seq1.as_bytes().to_vec(),
                    None, // no quality in Parquet
                    seq2.map(|s| s.as_bytes().to_vec()),
                    None,
                );
                writer.write_record(&owned, header.as_bytes())?;
                written += 1;
            }
            global_index += 1;
        }
    }

    if global_index != expected_total_reads {
        return Err(anyhow!(
            "Parquet input changed between classification and re-walk: \
             expected {} reads, found {}",
            expected_total_reads,
            global_index
        ));
    }

    writer.finish()?;
    Ok(written)
}

/// Extract a string value from an Arrow column (supports both Utf8 and LargeUtf8).
///
/// Returns a borrowed `&str` to avoid allocating per-row.
fn get_string_value(col: &dyn arrow::array::Array, row: usize) -> Result<&str> {
    use arrow::array::{LargeStringArray, StringArray};

    if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
        return Ok(arr.value(row));
    }
    if let Some(arr) = col.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(arr.value(row));
    }
    Err(anyhow!(
        "Column is not a string type: {:?}",
        col.data_type()
    ))
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::read::GzDecoder;
    use std::io::Read;
    use tempfile::tempdir;

    /// Helper to read and decompress a gzipped file.
    fn read_gzipped(path: &Path) -> String {
        let file = File::open(path).unwrap();
        let mut decoder = GzDecoder::new(file);
        let mut content = String::new();
        decoder.read_to_string(&mut content).unwrap();
        content
    }

    // -------------------------------------------------------------------------
    // Tests for SeqFormat::detect
    // -------------------------------------------------------------------------

    #[test]
    fn test_format_detection_fastq_gz() {
        assert_eq!(
            SeqFormat::detect(Path::new("x.fastq.gz")).unwrap(),
            SeqFormat::FastqGz
        );
    }

    #[test]
    fn test_format_detection_fq_gz() {
        assert_eq!(
            SeqFormat::detect(Path::new("x.fq.gz")).unwrap(),
            SeqFormat::FastqGz
        );
    }

    #[test]
    fn test_format_detection_fasta_gz() {
        assert_eq!(
            SeqFormat::detect(Path::new("x.fasta.gz")).unwrap(),
            SeqFormat::FastaGz
        );
    }

    #[test]
    fn test_format_detection_fa_gz() {
        assert_eq!(
            SeqFormat::detect(Path::new("x.fa.gz")).unwrap(),
            SeqFormat::FastaGz
        );
    }

    #[test]
    fn test_format_detection_invalid() {
        assert!(SeqFormat::detect(Path::new("x.txt")).is_err());
        assert!(SeqFormat::detect(Path::new("x.fastq")).is_err());
        assert!(SeqFormat::detect(Path::new("x.gz")).is_err());
    }

    // -------------------------------------------------------------------------
    // Tests for SequenceWriter::compute_paths
    // -------------------------------------------------------------------------

    #[test]
    fn test_output_paths_single() {
        let (r1, r2) = SequenceWriter::compute_paths(Path::new("out.fastq.gz"), false);
        assert_eq!(r1, PathBuf::from("out.fastq.gz"));
        assert!(r2.is_none());
    }

    #[test]
    fn test_output_paths_paired_fastq() {
        let (r1, r2) = SequenceWriter::compute_paths(Path::new("out.fastq.gz"), true);
        assert_eq!(r1, PathBuf::from("out.R1.fastq.gz"));
        assert_eq!(r2.unwrap(), PathBuf::from("out.R2.fastq.gz"));
    }

    #[test]
    fn test_output_paths_paired_fq() {
        let (r1, r2) = SequenceWriter::compute_paths(Path::new("out.fq.gz"), true);
        assert_eq!(r1, PathBuf::from("out.R1.fq.gz"));
        assert_eq!(r2.unwrap(), PathBuf::from("out.R2.fq.gz"));
    }

    #[test]
    fn test_output_paths_paired_fasta() {
        let (r1, r2) = SequenceWriter::compute_paths(Path::new("out.fasta.gz"), true);
        assert_eq!(r1, PathBuf::from("out.R1.fasta.gz"));
        assert_eq!(r2.unwrap(), PathBuf::from("out.R2.fasta.gz"));
    }

    #[test]
    fn test_output_paths_paired_fa() {
        let (r1, r2) = SequenceWriter::compute_paths(Path::new("out.fa.gz"), true);
        assert_eq!(r1, PathBuf::from("out.R1.fa.gz"));
        assert_eq!(r2.unwrap(), PathBuf::from("out.R2.fa.gz"));
    }

    // -------------------------------------------------------------------------
    // Tests for SequenceWriter::write_record
    // -------------------------------------------------------------------------

    #[test]
    fn test_write_fastq_record_with_quality() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.fastq.gz");
        let mut w = SequenceWriter::new(&path, false).unwrap();

        let rec = OwnedFastxRecord::new(0, b"ACGT".to_vec(), Some(b"IIII".to_vec()), None, None);
        w.write_record(&rec, b"r1").unwrap();
        w.finish().unwrap();

        let content = read_gzipped(&path);
        assert_eq!(content, "@r1\nACGT\n+\nIIII\n");
    }

    #[test]
    fn test_write_fastq_record_without_quality() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.fastq.gz");
        let mut w = SequenceWriter::new(&path, false).unwrap();

        // Record without quality scores (e.g., from FASTA input)
        let rec = OwnedFastxRecord::new(0, b"ACGT".to_vec(), None, None, None);
        w.write_record(&rec, b"r1").unwrap();
        w.finish().unwrap();

        let content = read_gzipped(&path);
        // Should have placeholder quality scores
        assert_eq!(content, "@r1\nACGT\n+\nIIII\n");
    }

    #[test]
    fn test_write_fasta_record() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.fasta.gz");
        let mut w = SequenceWriter::new(&path, false).unwrap();

        let rec = OwnedFastxRecord::new(0, b"ACGT".to_vec(), None, None, None);
        w.write_record(&rec, b"seq1").unwrap();
        w.finish().unwrap();

        let content = read_gzipped(&path);
        assert_eq!(content, ">seq1\nACGT\n");
    }

    #[test]
    fn test_write_paired_fastq() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("paired.fastq.gz");
        let mut w = SequenceWriter::new(&path, true).unwrap();

        let rec = OwnedFastxRecord::new(
            0,
            b"ACGT".to_vec(),
            Some(b"IIII".to_vec()),
            Some(b"TGCA".to_vec()),
            Some(b"JJJJ".to_vec()),
        );
        w.write_record(&rec, b"read1").unwrap();
        w.finish().unwrap();

        // Check R1 file
        let r1_content = read_gzipped(&dir.path().join("paired.R1.fastq.gz"));
        assert_eq!(r1_content, "@read1\nACGT\n+\nIIII\n");

        // Check R2 file
        let r2_content = read_gzipped(&dir.path().join("paired.R2.fastq.gz"));
        assert_eq!(r2_content, "@read1\nTGCA\n+\nJJJJ\n");
    }

    #[test]
    fn test_write_paired_fasta() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("paired.fasta.gz");
        let mut w = SequenceWriter::new(&path, true).unwrap();

        let rec = OwnedFastxRecord::new(0, b"ACGT".to_vec(), None, Some(b"TGCA".to_vec()), None);
        w.write_record(&rec, b"read1").unwrap();
        w.finish().unwrap();

        // Check R1 file
        let r1_content = read_gzipped(&dir.path().join("paired.R1.fasta.gz"));
        assert_eq!(r1_content, ">read1\nACGT\n");

        // Check R2 file
        let r2_content = read_gzipped(&dir.path().join("paired.R2.fasta.gz"));
        assert_eq!(r2_content, ">read1\nTGCA\n");
    }

    #[test]
    fn test_write_multiple_records() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("multi.fastq.gz");
        let mut w = SequenceWriter::new(&path, false).unwrap();

        for i in 0..3 {
            let header = format!("read{}", i);
            let rec =
                OwnedFastxRecord::new(i, b"ACGT".to_vec(), Some(b"IIII".to_vec()), None, None);
            w.write_record(&rec, header.as_bytes()).unwrap();
        }
        w.finish().unwrap();

        let content = read_gzipped(&path);
        assert!(content.contains("@read0\nACGT\n+\nIIII\n"));
        assert!(content.contains("@read1\nACGT\n+\nIIII\n"));
        assert!(content.contains("@read2\nACGT\n+\nIIII\n"));
    }
}
