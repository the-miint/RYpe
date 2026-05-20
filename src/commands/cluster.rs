//! CLI handler for `rype cluster`.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use needletail::parse_fastx_file;

use rype::cluster::{cluster_contigs, ClusterConfig, ClusterResult, ContigInput};

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

    let inputs = load_contigs_from_fastx(&args.inputs)?;
    log::info!(
        "Loaded {} contig(s) from {} input file(s) (before length filter)",
        inputs.len(),
        args.inputs.len(),
    );

    let cfg = ClusterConfig {
        k: args.kmer_size,
        w: args.window,
        salt: args.salt,
        min_length: args.min_length,
        threshold: args.threshold,
        min_shared: args.min_shared,
    };

    let result = cluster_contigs(&inputs, &cfg).context("clustering failed")?;
    log::info!(
        "Clustering complete: {} output rows ({} representatives)",
        result.rows.len(),
        result
            .rows
            .iter()
            .filter(|r| r.rep_contig == r.member_contig)
            .count(),
    );

    write_tsv(&args.output, &result)
        .with_context(|| format!("writing cluster TSV to {}", args.output.display()))?;

    Ok(())
}

/// Load every sequence from every input FASTA/FASTQ as a ContigInput.
///
/// The MAG name is the file basename with common FASTA/FASTQ (and gzip)
/// extensions stripped. The contig id is `<mag>::<header_first_token>` so
/// duplicate sequence names across files don't collide.
fn load_contigs_from_fastx(files: &[std::path::PathBuf]) -> Result<Vec<ContigInput>> {
    let mut out = Vec::new();
    for path in files {
        let mag = mag_name_from_path(path);
        let mut reader = parse_fastx_file(path)
            .with_context(|| format!("opening FASTA/FASTQ file {}", path.display()))?;

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
            let id = format!("{}::{}", mag, header_token);
            let sequence = rec.seq().into_owned();
            out.push(ContigInput {
                id,
                source_mag: Some(mag.clone()),
                sequence,
            });
        }
    }
    Ok(out)
}

/// Strip common FASTA/FASTQ extensions to derive the MAG name.
///
/// Handles `.gz` stacked on top of `.fasta` / `.fa` / `.fna` / `.fq` /
/// `.fastq`. Anything else falls through to the raw file_stem.
fn mag_name_from_path(p: &Path) -> String {
    let name = p
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();
    let lowered = name.to_ascii_lowercase();
    let strip = |suf: &str| -> Option<String> {
        if lowered.ends_with(suf) {
            Some(name[..name.len() - suf.len()].to_string())
        } else {
            None
        }
    };
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
        if let Some(s) = strip(suf) {
            return s;
        }
    }
    p.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string()
}

fn write_tsv(path: &Path, result: &ClusterResult) -> Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
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
        assert_eq!(mag_name_from_path(Path::new("foo.fasta")), "foo");
        assert_eq!(mag_name_from_path(Path::new("foo.fa")), "foo");
        assert_eq!(mag_name_from_path(Path::new("foo.fna")), "foo");
        assert_eq!(mag_name_from_path(Path::new("foo.fq")), "foo");
        assert_eq!(mag_name_from_path(Path::new("foo.fastq")), "foo");
        assert_eq!(mag_name_from_path(Path::new("foo.fasta.gz")), "foo");
        assert_eq!(mag_name_from_path(Path::new("foo.fa.gz")), "foo");
        assert_eq!(mag_name_from_path(Path::new("foo.fq.gz")), "foo");
        assert_eq!(mag_name_from_path(Path::new("foo.FASTA")), "foo");
        assert_eq!(mag_name_from_path(Path::new("/path/to/bar.fasta")), "bar");
        // Unknown extension falls through to file_stem
        assert_eq!(mag_name_from_path(Path::new("foo.txt")), "foo");
        assert_eq!(mag_name_from_path(Path::new("foo")), "foo");
    }
}
