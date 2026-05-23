//! CLI handler for `rype cluster`.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use needletail::parse_fastx_file;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use rype::arrow::cluster::{cluster_result_schema, cluster_result_to_record_batch};
use rype::cluster::{cluster_contigs, ClusterConfig, ClusterResult, ContigInput};
use rype::{ChainParams, BUCKET_SOURCE_DELIM};

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
    let output_path = validate_output_path(args.output.as_deref())?;
    if args.no_chain && args.chain_threshold.is_some() {
        return Err(anyhow!(
            "--chain-threshold requires chain to be enabled; remove --no-chain or drop --chain-threshold"
        ));
    }
    if args.no_chain && args.chain_min_anchors.is_some() {
        return Err(anyhow!(
            "--chain-min-anchors requires chain to be enabled; remove --no-chain or drop --chain-min-anchors"
        ));
    }
    if let Some(t) = args.chain_threshold {
        if !t.is_finite() || t <= 0.0 || t > 1.0 {
            return Err(anyhow!(
                "--chain-threshold must be finite in (0.0, 1.0] (got {})",
                t
            ));
        }
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

    // Chain is opt-in at the CLI to preserve the historical default behavior.
    // Any of `--chain-threshold`, `--chain-min-anchors` flips chain on with
    // `starting_for_w(window)`; `--no-chain` is explicit and overrides those.
    let chain_params = if args.no_chain {
        None
    } else if args.chain_threshold.is_some() || args.chain_min_anchors.is_some() {
        let mut params = ChainParams::starting_for_w(args.window);
        if let Some(m) = args.chain_min_anchors {
            params.min_anchors = m;
        }
        Some(params)
    } else {
        None
    };

    let cfg = ClusterConfig {
        k: args.kmer_size,
        w: args.window,
        salt: args.salt,
        min_length: args.min_length,
        threshold: args.threshold,
        min_shared: args.min_shared,
        chain_params,
        min_chain_containment: args.chain_threshold,
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

    write_cluster_parquet(&output_path, &result)
        .with_context(|| format!("writing cluster Parquet to {}", output_path.display()))?;

    Ok(())
}

/// Validate the `--output` argument and resolve it to a concrete Parquet path.
///
/// Phase 1.5 dropped TSV support; the cluster CLI now writes only Parquet.
/// `--output -` (stdout) is rejected because Parquet is binary. A `.tsv`
/// extension is rejected with a named error so users notice the format
/// change immediately rather than producing a corrupt Parquet file with a
/// `.tsv` extension.
fn validate_output_path(output: Option<&Path>) -> Result<std::path::PathBuf> {
    let path = output.ok_or_else(|| {
        anyhow!("--output is required (rype cluster writes Parquet; pass a .parquet path)")
    })?;

    if path.as_os_str() == "-" {
        return Err(anyhow!(
            "rype cluster writes Parquet (binary); stdout (`-`) is not supported. \
             Pass an explicit path ending in `.parquet`."
        ));
    }

    if let Some(ext) = path.extension() {
        let ext_lower = ext.to_ascii_lowercase();
        if ext_lower == "tsv" {
            return Err(anyhow!(
                "rype cluster now writes Parquet, not TSV (the cluster output \
                 grew nullable chain columns that don't round-trip cleanly \
                 through TSV). Use a `.parquet` extension (or no extension); \
                 read the result via duckdb/pandas/the parquet crate."
            ));
        }
    }

    Ok(path.to_path_buf())
}

/// Write the cluster result to a Parquet file using the 8-column schema
/// defined by [`cluster_result_schema`].
fn write_cluster_parquet(path: &Path, result: &ClusterResult) -> Result<()> {
    let batch = cluster_result_to_record_batch(result)
        .context("building Arrow RecordBatch from cluster result")?;
    let file = File::create(path)
        .with_context(|| format!("creating Parquet output file {}", path.display()))?;
    let buf = BufWriter::new(file);
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let schema: Arc<arrow::datatypes::Schema> = cluster_result_schema();
    let mut writer = ArrowWriter::try_new(buf, schema, Some(props))
        .context("opening Parquet writer for cluster output")?;
    writer
        .write(&batch)
        .context("writing cluster RecordBatch to Parquet")?;
    writer.close().context("closing Parquet writer")?;
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

    /// WHY: the CLI must reject `.tsv` with a named error rather than
    /// silently producing a corrupt Parquet file at a `.tsv` path. The
    /// message must mention "Parquet" so users notice the format change.
    #[test]
    fn validate_output_path_rejects_tsv_extension() {
        let err = validate_output_path(Some(Path::new("out.tsv"))).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("Parquet") && msg.contains("TSV"),
            "error must explain the format change, got: {}",
            msg
        );
    }

    /// WHY: Parquet is binary; stdout is not a useful sink. Reject `-`
    /// explicitly so a user piping the output sees a clear error rather
    /// than a binary mess on their terminal.
    #[test]
    fn validate_output_path_rejects_stdout_dash() {
        let err = validate_output_path(Some(Path::new("-"))).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("stdout") || msg.contains("`-`"),
            "error must explain stdout-rejection, got: {}",
            msg
        );
    }

    /// WHY: `--output` is now mandatory (stdout is rejected, no implicit
    /// default file). A missing argument must error early — before the
    /// expensive clustering work — with a message naming the flag.
    #[test]
    fn validate_output_path_requires_output_arg() {
        let err = validate_output_path(None).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("--output"),
            "error must name the missing flag, got: {}",
            msg
        );
    }

    /// WHY: a `.parquet` (or extensionless) path is the supported case;
    /// it must round-trip the validator unchanged.
    #[test]
    fn validate_output_path_accepts_parquet_and_extensionless() {
        let p = validate_output_path(Some(Path::new("out.parquet"))).unwrap();
        assert_eq!(p, Path::new("out.parquet"));
        let p = validate_output_path(Some(Path::new("out"))).unwrap();
        assert_eq!(p, Path::new("out"));
    }
}
