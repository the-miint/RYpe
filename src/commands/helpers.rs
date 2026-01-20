//! Helper functions and utilities for the rype CLI.

use anyhow::{anyhow, Context, Result};
use needletail::{parse_fastx_file, FastxReader};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use rype::memory::parse_byte_suffix;
use rype::{Index, IndexMetadata, MainIndexManifest};

/// Owned record type: (query_id, seq1, optional_seq2)
pub type OwnedRecord = (i64, Vec<u8>, Option<Vec<u8>>);

/// Parse a byte size string from CLI (e.g., "4G", "512M", "auto").
/// Returns 0 for "auto" (signals auto-detection), bytes otherwise.
pub fn parse_max_memory_arg(s: &str) -> Result<usize, String> {
    parse_byte_suffix(s)
        .map(|opt| opt.unwrap_or(0)) // None (auto) -> 0
        .map_err(|e| e.to_string())
}

/// Parse a byte size string from CLI, requiring a concrete value (no "auto").
pub fn parse_shard_size_arg(s: &str) -> Result<usize, String> {
    match parse_byte_suffix(s) {
        Ok(Some(bytes)) => Ok(bytes),
        Ok(None) => Err("'auto' not supported for shard size".to_string()),
        Err(e) => Err(e.to_string()),
    }
}

/// Parse bloom filter false positive probability, validating range (0.0, 1.0).
pub fn parse_bloom_fpp(s: &str) -> Result<f64, String> {
    let fpp: f64 = s
        .parse()
        .map_err(|_| format!("'{}' is not a valid number", s))?;
    if fpp <= 0.0 || fpp >= 1.0 {
        return Err(format!(
            "bloom_filter_fpp must be in (0.0, 1.0), got {}",
            fpp
        ));
    }
    Ok(fpp)
}

/// Parse shard format argument ("legacy" or "parquet")
pub fn parse_shard_format(s: &str) -> Result<String, String> {
    match s.to_lowercase().as_str() {
        "legacy" | "ryxs" => Ok("legacy".to_string()),
        "parquet" | "pq" => Ok("parquet".to_string()),
        _ => Err(format!(
            "Unknown format '{}'. Valid options: legacy, parquet",
            s
        )),
    }
}

/// Sanitize bucket names by replacing nonprintable characters with "_"
pub fn sanitize_bucket_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_control() || !c.is_ascii_graphic() && !c.is_whitespace() {
                '_'
            } else {
                c
            }
        })
        .collect()
}

/// Load metadata from Parquet, sharded, or single-file indices.
///
/// This helper handles:
/// - Parquet inverted index directories (with manifest.toml)
/// - Sharded main indices (with .manifest and .shard.* files)
/// - Single-file indices (.ryidx)
pub fn load_index_metadata(path: &Path) -> Result<IndexMetadata> {
    // Check for Parquet format first (directory with manifest.toml)
    if rype::is_parquet_index(path) {
        let manifest = rype::ParquetManifest::load(path)?;
        let (bucket_names, bucket_sources) = rype::parquet_index::read_buckets_parquet(path)?;
        return Ok(IndexMetadata {
            k: manifest.k,
            w: manifest.w,
            salt: manifest.salt,
            bucket_names,
            bucket_sources,
            bucket_minimizer_counts: HashMap::new(),
        });
    }

    // Check for sharded main index
    if MainIndexManifest::is_sharded(path) {
        let manifest = MainIndexManifest::load(&MainIndexManifest::manifest_path(path))?;
        Ok(manifest.to_metadata())
    } else {
        // Single-file index
        Ok(Index::load_metadata(path)?)
    }
}

/// I/O handler for reading FASTX files and writing output.
pub struct IoHandler {
    pub r1: Box<dyn FastxReader>,
    pub r2: Option<Box<dyn FastxReader>>,
    writer: BufWriter<Box<dyn Write>>,
}

impl IoHandler {
    pub fn new(
        r1_path: &Path,
        r2_path: Option<&PathBuf>,
        out_path: Option<PathBuf>,
    ) -> Result<Self> {
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

    pub fn next_batch_records(
        &mut self,
        size: usize,
    ) -> Result<Option<(Vec<OwnedRecord>, Vec<String>)>> {
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

        if records.is_empty() {
            return Ok(None);
        }
        Ok(Some((records, headers)))
    }

    pub fn write(&mut self, data: Vec<u8>) -> Result<()> {
        self.writer.write_all(&data)?;
        Ok(())
    }

    pub fn finish(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}
