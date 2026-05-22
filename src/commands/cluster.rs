//! CLI handler for `rype cluster`.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use needletail::parse_fastx_file;

use rype::cluster::{cluster_contigs, ClusterConfig, ClusterResult, ContigInput};
use rype::BUCKET_SOURCE_DELIM;

use super::args::ClusterArgs;

/// Run the `rype cluster` subcommand.
pub fn run_cluster(args: ClusterArgs) -> Result<()> {
    if !matches!(args.kmer_size, 16 | 32 | 64) {
        return Err(anyhow!("K must be 16, 32, or 64 (got {})", args.kmer_size));
    }
    if args.window == 0 {
        return Err(anyhow!("window must be > 0"));
    }
    if !args.threshold.is_finite() || args.threshold <= 0.0 || args.threshold > 1.0 {
        return Err(anyhow!(
            "threshold must be finite in (0.0, 1.0] (got {})",
            args.threshold
        ));
    }

    log::info!(
        "Loading contigs from {} input file(s)...",
        args.inputs.len()
    );
    let inputs = load_contigs_from_fastx(&args.inputs)?;
    let total_bytes: u64 = inputs.iter().map(|c| c.sequence.len() as u64).sum();
    log::info!(
        "Loaded {} contig(s), {:.2} MB total sequence",
        inputs.len(),
        total_bytes as f64 / 1_048_576.0,
    );

    let cfg = ClusterConfig {
        k: args.kmer_size,
        w: args.window,
        salt: args.salt,
        min_length: args.min_length,
        threshold: args.threshold,
        min_shared: args.min_shared,
        // Phase 1: CLI exposes no chain flags yet (Phase 3 adds --no-chain,
        // --chain-threshold, --chain-min-anchors). Default to chain disabled
        // at the CLI for now so behavior matches today; `strain_default()`
        // enables chain for library callers that want it.
        chain_params: None,
        min_chain_containment: None,
    };

    log::info!(
        "Building cluster index and computing edges (this is the heavy phase; \
         expect to hold ~3x total sequence size in RAM at peak)..."
    );
    let result = cluster_contigs(inputs, &cfg).context("clustering failed")?;
    let rep_count = result
        .rows
        .iter()
        .filter(|r| r.rep_contig == r.member_contig)
        .count();
    log::info!(
        "Clustering complete: {} output rows ({} representatives, {} absorbed)",
        result.rows.len(),
        rep_count,
        result.rows.len().saturating_sub(rep_count),
    );

    write_cluster_output(args.output.as_deref(), &result).with_context(|| {
        format!(
            "writing cluster TSV to {}",
            args.output
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "<stdout>".to_string())
        )
    })?;

    Ok(())
}

/// Load every sequence from every input FASTA/FASTQ as a ContigInput.
///
/// The MAG name is the file basename with common FASTA/FASTQ (and gzip)
/// extensions stripped. The contig id is `<mag>::<header_first_token>` so
/// duplicate sequence names across files don't collide.
fn load_contigs_from_fastx(files: &[std::path::PathBuf]) -> Result<Vec<ContigInput>> {
    let mut out = Vec::new();
    for (i, path) in files.iter().enumerate() {
        let mag = mag_name_from_path(path).ok_or_else(|| {
            anyhow!(
                "cannot derive MAG name from path {} (try renaming the file)",
                path.display()
            )
        })?;
        let mut reader = parse_fastx_file(path)
            .with_context(|| format!("opening FASTA/FASTQ file {}", path.display()))?;

        let mut seqs_in_file = 0usize;
        while let Some(rec_res) = reader.next() {
            let rec = rec_res.with_context(|| format!("reading record from {}", path.display()))?;
            let header_full = String::from_utf8_lossy(rec.id()).to_string();
            let header_token = header_full
                .split_ascii_whitespace()
                .next()
                .unwrap_or("")
                .to_string();
            if header_token.is_empty() {
                return Err(anyhow!(
                    "empty sequence header in {} — every sequence needs an id",
                    path.display()
                ));
            }
            let id = format!("{}{}{}", mag, BUCKET_SOURCE_DELIM, header_token);
            let sequence = rec.seq().into_owned();
            out.push(ContigInput {
                id,
                source_mag: Some(mag.clone()),
                sequence,
            });
            seqs_in_file += 1;
        }

        if (i + 1) % 100 == 0 || i + 1 == files.len() {
            log::info!(
                "  loaded {}/{} files ({} contigs so far; last file: {} seqs)",
                i + 1,
                files.len(),
                out.len(),
                seqs_in_file,
            );
        }
    }
    Ok(out)
}

/// Strip common FASTA/FASTQ extensions to derive the MAG name.
///
/// Handles `.gz` stacked on top of `.fasta` / `.fa` / `.fna` / `.fq` /
/// `.fastq`. Anything else falls through to the raw file_stem. Returns
/// `None` if the path has no usable file_name (e.g. `/`, `.`, `..`).
fn mag_name_from_path(p: &Path) -> Option<String> {
    let name = p.file_name()?.to_str()?.to_string();
    if name.is_empty() {
        return None;
    }
    let lowered = name.to_ascii_lowercase();
    for suf in [
        ".fasta.gz",
        ".fa.gz",
        ".fna.gz",
        ".fastq.gz",
        ".fq.gz",
        ".fasta",
        ".fa",
        ".fna",
        ".fastq",
        ".fq",
    ] {
        if lowered.ends_with(suf) {
            return Some(name[..name.len() - suf.len()].to_string());
        }
    }
    p.file_stem()?.to_str().map(|s| s.to_string())
}

/// Write the TSV to the given path (or stdout if `None` or `"-"`).
fn write_cluster_output(path: Option<&Path>, result: &ClusterResult) -> Result<()> {
    match path {
        Some(p) if p.as_os_str() == "-" => {
            write_tsv(&mut BufWriter::new(io::stdout().lock()), result)
        }
        None => write_tsv(&mut BufWriter::new(io::stdout().lock()), result),
        Some(p) => {
            let file = File::create(p)?;
            write_tsv(&mut BufWriter::new(file), result)
        }
    }
}

fn write_tsv<W: Write>(w: &mut W, result: &ClusterResult) -> Result<()> {
    writeln!(w, "rep_contig\tmember_contig\tsource_mag\tcontainment")?;
    for row in &result.rows {
        let mag = row.source_mag.as_deref().unwrap_or("");
        writeln!(
            w,
            "{}\t{}\t{}\t{:.6}",
            row.rep_contig, row.member_contig, mag, row.containment
        )?;
    }
    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mag_name_strips_common_extensions() {
        assert_eq!(
            mag_name_from_path(Path::new("foo.fasta")).as_deref(),
            Some("foo")
        );
        assert_eq!(
            mag_name_from_path(Path::new("foo.fa")).as_deref(),
            Some("foo")
        );
        assert_eq!(
            mag_name_from_path(Path::new("foo.fna")).as_deref(),
            Some("foo")
        );
        assert_eq!(
            mag_name_from_path(Path::new("foo.fq")).as_deref(),
            Some("foo")
        );
        assert_eq!(
            mag_name_from_path(Path::new("foo.fastq")).as_deref(),
            Some("foo")
        );
        assert_eq!(
            mag_name_from_path(Path::new("foo.fasta.gz")).as_deref(),
            Some("foo")
        );
        assert_eq!(
            mag_name_from_path(Path::new("foo.fa.gz")).as_deref(),
            Some("foo")
        );
        assert_eq!(
            mag_name_from_path(Path::new("foo.fq.gz")).as_deref(),
            Some("foo")
        );
        assert_eq!(
            mag_name_from_path(Path::new("foo.FASTA")).as_deref(),
            Some("foo")
        );
        assert_eq!(
            mag_name_from_path(Path::new("/path/to/bar.fasta")).as_deref(),
            Some("bar")
        );
        assert_eq!(
            mag_name_from_path(Path::new("foo.txt")).as_deref(),
            Some("foo")
        );
        assert_eq!(mag_name_from_path(Path::new("foo")).as_deref(), Some("foo"));
    }

    #[test]
    fn mag_name_returns_none_for_paths_without_filename() {
        assert!(mag_name_from_path(Path::new("/")).is_none());
        assert!(mag_name_from_path(Path::new("..")).is_none()); // file_stem of ".." is ""
    }
}
