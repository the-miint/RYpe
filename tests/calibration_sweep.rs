//! Plan 1.6 — chain parameter calibration harness.
//!
//! This integration test crate ships the offline calibration harness
//! described in `localdocs/chain-calibration.md`. The harness is
//! `#[ignore]`-gated and reads real MAG data + skani-derived labels
//! from env vars at run time; this file's NON-ignored tests cover the
//! parser, the scoring helpers, the sweep grid, and the Parquet writer.
//!
//! # Skani labels TSV format (what `parse_skani_labels` expects)
//!
//! 3-column TSV, TAB-separated, NO header row. Lines starting with `#`
//! are comments and are skipped. Each data line:
//!
//! ```text
//! ref_id   query_id   ani
//! ```
//!
//! Where `ani` is in **percent** scale (the convention skani uses),
//! e.g. `95.234` for 95.234% ANI. The parser filters by an operator-
//! supplied threshold (also in percent) and returns a set of unordered
//! `(query_id, ref_id)` pairs in canonical order (lexicographic
//! `(min, max)`).
//!
//! The methodology doc gives an `awk` snippet for extracting this
//! 3-column shape from the full `skani dist` output. The harness
//! deliberately does NOT try to auto-detect skani's many output formats:
//! one strict format keeps the parser bulletproof and tests tractable.
//!
//! # Precision / recall semantics (`score_cell`)
//!
//! Strict **pair-based** counting. An "absorbed prediction" is a
//! `(rep_contig, member_contig)` pair from a `ClusterRow` where
//! `rep_contig != member_contig`. Both sides are canonicalized to
//! `(min, max)` for the join.
//!
//!   - TP = |absorbed ∩ labels|
//!   - FP = |absorbed \ labels|
//!   - FN = |labels \ absorbed|
//!
//! This evaluates the **absorption decisions the greedy made**, not
//! transitive cluster membership. A label `(A, B)` where the greedy
//! built cluster `{C, A, B}` (so the absorbed pairs are `(C, A)` and
//! `(C, B)`, not `(A, B)`) is counted as FN here. For dereplication
//! at MAG / contig scale the typical cluster is 2 members, so the
//! transitive-credit gap rarely fires; the methodology doc warns the
//! operator if their MAGs cluster wider than that.

use anyhow::{anyhow, Context, Result};
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;

/// Return a canonical-ordered pair `(min(a, b), max(a, b))` so that
/// `canonical_pair("A", "B") == canonical_pair("B", "A")`. The label
/// set and the absorbed-prediction set both use this so the join is
/// order-independent.
fn canonical_pair(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

/// Parse a 3-column skani labels TSV and return the set of positive
/// pairs (those with ANI ≥ `ani_threshold_pct`).
///
/// `ani_threshold_pct` is in percent (e.g. `95.0` for 95% ANI). Pairs
/// are returned in canonical order, so a `(A, B)` and `(B, A)` row in
/// the input collapse to one label.
///
/// # Errors
/// - File not readable.
/// - Any data line that isn't exactly 3 TAB-separated fields.
/// - Non-numeric or non-finite ANI value.
///
/// # Self-pairs
/// Rows where `ref_id == query_id` are silently dropped (skani sometimes
/// emits these and they're vacuous for calibration — a contig perfectly
/// "matches" itself).
pub fn parse_skani_labels(
    path: &Path,
    ani_threshold_pct: f64,
) -> Result<HashSet<(String, String)>> {
    let body = fs::read_to_string(path)
        .with_context(|| format!("reading skani labels TSV at {}", path.display()))?;
    parse_skani_labels_str(&body, ani_threshold_pct).with_context(|| {
        format!(
            "parsing skani labels TSV at {} (ani_threshold_pct={})",
            path.display(),
            ani_threshold_pct
        )
    })
}

/// In-process parser for the labels TSV body. Splits out for test
/// reuse on inline fixtures (no temp-file thrash).
fn parse_skani_labels_str(body: &str, ani_threshold_pct: f64) -> Result<HashSet<(String, String)>> {
    let mut out = HashSet::new();
    for (line_no, raw) in body.lines().enumerate() {
        let line = raw.trim_end();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 3 {
            return Err(anyhow!(
                "line {} has {} TAB-separated fields, expected 3: {:?}",
                line_no + 1,
                fields.len(),
                line
            ));
        }
        let ref_id = fields[0];
        let query_id = fields[1];
        let ani: f64 = fields[2].parse().map_err(|e| {
            anyhow!(
                "line {} ANI parse error: {} ({:?})",
                line_no + 1,
                e,
                fields[2]
            )
        })?;
        if !ani.is_finite() {
            return Err(anyhow!(
                "line {} ANI is non-finite ({}): {:?}",
                line_no + 1,
                ani,
                line
            ));
        }
        if ref_id == query_id {
            // Self-pair; vacuous.
            continue;
        }
        if ani >= ani_threshold_pct {
            out.insert(canonical_pair(ref_id, query_id));
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------
// Phase 2 — sweep grid + scoring + Parquet writer + ignored harness.
// ---------------------------------------------------------------------

/// Default `min_chain_containment` values swept by the harness when
/// `RYPE_CALIB_CONTAINMENT_GRID` is unset.
const DEFAULT_CONTAINMENTS: [f64; 7] = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

/// Default `ChainParams.min_anchors` values swept by the harness when
/// `RYPE_CALIB_MIN_ANCHORS_GRID` is unset.
const DEFAULT_MIN_ANCHORS: [u32; 5] = [3, 5, 10, 20, 50];

/// Default ANI threshold (percent) for declaring a skani pair a
/// positive label when `RYPE_CALIB_ANI_THRESHOLD` is unset.
const DEFAULT_ANI_THRESHOLD_PCT: f64 = 95.0;

/// Per-cell counts emitted by `score_cell`.
///
/// `fn_` (with trailing underscore) avoids the `fn` keyword while keeping
/// the field name obvious in test assertions and Parquet column-source
/// reads. The Parquet column itself is `false_negatives`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellResult {
    pub tp: u64,
    pub fp: u64,
    pub fn_: u64,
    pub total_absorbed: u64,
    pub total_positives: u64,
}

/// Score one sweep cell by intersecting the absorbed-prediction set
/// against the positive-label set. See module-level docs for the exact
/// TP/FP/FN definitions.
pub fn score_cell(
    absorbed: &HashSet<(String, String)>,
    labels: &HashSet<(String, String)>,
) -> CellResult {
    let tp = absorbed.intersection(labels).count() as u64;
    let total_absorbed = absorbed.len() as u64;
    let total_positives = labels.len() as u64;
    // `tp <= min(total_absorbed, total_positives)` by construction, so
    // these subtractions cannot underflow.
    let fp = total_absorbed - tp;
    let fn_ = total_positives - tp;
    CellResult {
        tp,
        fp,
        fn_,
        total_absorbed,
        total_positives,
    }
}

/// Derive `(precision, recall, f1)` from a `CellResult`.
///
/// Returns `NaN` for any metric whose denominator is zero, per the plan:
///   - precision is NaN when `tp + fp == 0` (no predictions).
///   - recall is NaN when `tp + fn == 0` (no positives).
///   - f1 is NaN when precision or recall is NaN, OR when `p + r == 0`
///     (both zero — happens when there are predictions and positives but
///     they don't overlap).
pub fn metrics(c: &CellResult) -> (f64, f64, f64) {
    let p_denom = c.tp + c.fp;
    let r_denom = c.tp + c.fn_;
    let precision = if p_denom == 0 {
        f64::NAN
    } else {
        c.tp as f64 / p_denom as f64
    };
    let recall = if r_denom == 0 {
        f64::NAN
    } else {
        c.tp as f64 / r_denom as f64
    };
    let f1 = if precision.is_nan() || recall.is_nan() || precision + recall == 0.0 {
        f64::NAN
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    (precision, recall, f1)
}

/// Parse a comma-separated env-var value into a `Vec<T>`. Whitespace
/// around each value is trimmed; empty tokens (from trailing commas) are
/// skipped. Returns a clear error if any token fails to parse.
fn parse_grid<T>(input: &str, label: &str) -> Result<Vec<T>>
where
    T: FromStr,
    T::Err: std::fmt::Display,
{
    let out: Vec<T> = input
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<T>()
                .map_err(|e| anyhow!("invalid {} value {:?}: {}", label, s, e))
        })
        .collect::<Result<Vec<T>>>()?;
    if out.is_empty() {
        return Err(anyhow!("{} grid is empty after parsing {:?}", label, input));
    }
    Ok(out)
}

/// Build the full sweep grid as a `Vec<(containment, min_anchors)>`,
/// honoring `RYPE_CALIB_CONTAINMENT_GRID` / `RYPE_CALIB_MIN_ANCHORS_GRID`
/// env-var overrides. Default is the 7×5 = 35-cell grid from the plan.
pub fn sweep_grid() -> Result<Vec<(f64, u32)>> {
    let containments: Vec<f64> = match std::env::var("RYPE_CALIB_CONTAINMENT_GRID") {
        Ok(s) => parse_grid::<f64>(&s, "containment")?,
        Err(_) => DEFAULT_CONTAINMENTS.to_vec(),
    };
    let min_anchors: Vec<u32> = match std::env::var("RYPE_CALIB_MIN_ANCHORS_GRID") {
        Ok(s) => parse_grid::<u32>(&s, "min_anchors")?,
        Err(_) => DEFAULT_MIN_ANCHORS.to_vec(),
    };
    let mut out = Vec::with_capacity(containments.len() * min_anchors.len());
    for &c in &containments {
        for &m in &min_anchors {
            out.push((c, m));
        }
    }
    Ok(out)
}

/// Parquet column name for the sweep `min_chain_containment` axis.
const COL_MIN_CHAIN_CONTAINMENT: &str = "min_chain_containment";
const COL_MIN_ANCHORS: &str = "min_anchors";
const COL_TRUE_POSITIVES: &str = "true_positives";
const COL_FALSE_POSITIVES: &str = "false_positives";
const COL_FALSE_NEGATIVES: &str = "false_negatives";
const COL_PRECISION: &str = "precision";
const COL_RECALL: &str = "recall";
const COL_F1: &str = "f1";
const COL_TOTAL_ABSORBED: &str = "total_absorbed";
const COL_TOTAL_POSITIVES: &str = "total_positives";

/// 10-column Arrow schema for the sweep-result Parquet file. Mirrors
/// the table in the Plan 1.6 spec verbatim.
fn sweep_result_schema() -> arrow::datatypes::SchemaRef {
    use arrow::datatypes::{DataType, Field, Schema};
    Arc::new(Schema::new(vec![
        Field::new(COL_MIN_CHAIN_CONTAINMENT, DataType::Float64, false),
        Field::new(COL_MIN_ANCHORS, DataType::UInt32, false),
        Field::new(COL_TRUE_POSITIVES, DataType::UInt64, false),
        Field::new(COL_FALSE_POSITIVES, DataType::UInt64, false),
        Field::new(COL_FALSE_NEGATIVES, DataType::UInt64, false),
        Field::new(COL_PRECISION, DataType::Float64, false),
        Field::new(COL_RECALL, DataType::Float64, false),
        Field::new(COL_F1, DataType::Float64, false),
        Field::new(COL_TOTAL_ABSORBED, DataType::UInt64, false),
        Field::new(COL_TOTAL_POSITIVES, DataType::UInt64, false),
    ]))
}

/// Write a sweep-result Parquet file at `path`. Compression is ZSTD to
/// match the rest of the project. `cells` is one entry per sweep cell;
/// metrics are derived from `CellResult` via `metrics()`.
pub fn write_sweep_results_parquet(path: &Path, cells: &[(f64, u32, CellResult)]) -> Result<()> {
    use arrow::array::{Float64Builder, UInt32Builder, UInt64Builder};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use parquet::basic::Compression;
    use parquet::file::properties::WriterProperties;
    use std::fs::File;
    use std::io::BufWriter;

    let mut containment_b = Float64Builder::with_capacity(cells.len());
    let mut min_anchors_b = UInt32Builder::with_capacity(cells.len());
    let mut tp_b = UInt64Builder::with_capacity(cells.len());
    let mut fp_b = UInt64Builder::with_capacity(cells.len());
    let mut fn_b = UInt64Builder::with_capacity(cells.len());
    let mut precision_b = Float64Builder::with_capacity(cells.len());
    let mut recall_b = Float64Builder::with_capacity(cells.len());
    let mut f1_b = Float64Builder::with_capacity(cells.len());
    let mut total_absorbed_b = UInt64Builder::with_capacity(cells.len());
    let mut total_positives_b = UInt64Builder::with_capacity(cells.len());

    for (c, m, r) in cells {
        let (p, recall, f1) = metrics(r);
        containment_b.append_value(*c);
        min_anchors_b.append_value(*m);
        tp_b.append_value(r.tp);
        fp_b.append_value(r.fp);
        fn_b.append_value(r.fn_);
        precision_b.append_value(p);
        recall_b.append_value(recall);
        f1_b.append_value(f1);
        total_absorbed_b.append_value(r.total_absorbed);
        total_positives_b.append_value(r.total_positives);
    }

    let schema = sweep_result_schema();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(containment_b.finish()),
            Arc::new(min_anchors_b.finish()),
            Arc::new(tp_b.finish()),
            Arc::new(fp_b.finish()),
            Arc::new(fn_b.finish()),
            Arc::new(precision_b.finish()),
            Arc::new(recall_b.finish()),
            Arc::new(f1_b.finish()),
            Arc::new(total_absorbed_b.finish()),
            Arc::new(total_positives_b.finish()),
        ],
    )
    .context("building sweep-result Arrow RecordBatch")?;

    let file = File::create(path)
        .with_context(|| format!("creating sweep-result Parquet file {}", path.display()))?;
    let buf = BufWriter::new(file);
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let mut writer = ArrowWriter::try_new(buf, schema, Some(props))
        .context("opening Parquet writer for sweep results")?;
    writer
        .write(&batch)
        .context("writing sweep-result RecordBatch to Parquet")?;
    writer.close().context("closing Parquet writer")?;
    Ok(())
}

// ---------------------------------------------------------------------
// Ignored integration test — the calibration harness itself.
// Reads RYPE_CALIB_GENOMES, RYPE_CALIB_LABELS, RYPE_CALIB_OUT from env.
// Optional: RYPE_CALIB_ANI_THRESHOLD, RYPE_CALIB_CONTAINMENT_GRID,
// RYPE_CALIB_MIN_ANCHORS_GRID, RYPE_CALIB_K, RYPE_CALIB_W,
// RYPE_CALIB_MIN_LENGTH, RYPE_CALIB_THRESHOLD, RYPE_CALIB_MIN_SHARED.
// See localdocs/chain-calibration.md (Phase 3) for the run book.
// ---------------------------------------------------------------------

/// Load every FASTA/FASTQ (optionally gzipped) file under `dir` as a
/// flat `Vec<ContigInput>`. The contig id is
/// `<file_stem_minus_compression>::<header_first_token>` to match the
/// CLI's `load_contigs_from_fastx` convention, so the labels TSV from
/// skani-on-the-same-files joins correctly.
#[cfg(feature = "fastx")]
fn load_mag_directory(dir: &Path) -> Result<Vec<rype::cluster::ContigInput>> {
    use needletail::parse_fastx_file;

    let mut files: Vec<std::path::PathBuf> = fs::read_dir(dir)
        .with_context(|| format!("listing MAG directory {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            if !p.is_file() {
                return false;
            }
            let lower = p.to_string_lossy().to_ascii_lowercase();
            lower.ends_with(".fasta")
                || lower.ends_with(".fasta.gz")
                || lower.ends_with(".fa")
                || lower.ends_with(".fa.gz")
                || lower.ends_with(".fna")
                || lower.ends_with(".fna.gz")
                || lower.ends_with(".fastq")
                || lower.ends_with(".fastq.gz")
                || lower.ends_with(".fq")
                || lower.ends_with(".fq.gz")
        })
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(anyhow!(
            "no FASTA/FASTQ files found under {} (extensions checked: .fasta[.gz], .fa[.gz], .fna[.gz], .fastq[.gz], .fq[.gz])",
            dir.display()
        ));
    }

    let mut out: Vec<rype::cluster::ContigInput> = Vec::new();
    for path in &files {
        let mag = mag_name_from_path(path).ok_or_else(|| {
            anyhow!(
                "cannot derive MAG name from path {} (try renaming the file)",
                path.display()
            )
        })?;
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
            let id = format!("{}{}{}", mag, rype::BUCKET_SOURCE_DELIM, header_token);
            let sequence = rec.seq().into_owned();
            out.push(rype::cluster::ContigInput {
                id,
                source_mag: Some(mag.clone()),
                sequence,
            });
        }
    }
    Ok(out)
}

/// Strip common FASTA/FASTQ extensions (including stacked `.gz`) to
/// derive the MAG name. Mirrors `src/commands/cluster.rs::mag_name_from_path`.
#[cfg(feature = "fastx")]
fn mag_name_from_path(path: &Path) -> Option<String> {
    let stem = path.file_name()?.to_string_lossy().to_string();
    let lower = stem.to_ascii_lowercase();
    let trimmed = if let Some(stripped) = lower.strip_suffix(".gz") {
        stem[..stripped.len()].to_string()
    } else {
        stem.clone()
    };
    let trimmed_lower = trimmed.to_ascii_lowercase();
    for suffix in [".fasta", ".fa", ".fna", ".fastq", ".fq"] {
        if let Some(stripped) = trimmed_lower.strip_suffix(suffix) {
            return Some(trimmed[..stripped.len()].to_string());
        }
    }
    Some(trimmed)
}

/// Build an `(absorbed_pairs, labels)`-style canonical pair set from a
/// `ClusterResult`. Each row with `rep_contig != member_contig` becomes
/// one canonical pair; representative rows (the partition's
/// self-loops) are dropped.
#[cfg(feature = "fastx")]
fn absorbed_pairs_from_result(result: &rype::cluster::ClusterResult) -> HashSet<(String, String)> {
    result
        .rows
        .iter()
        .filter(|r| r.rep_contig != r.member_contig)
        .map(|r| canonical_pair(&r.rep_contig, &r.member_contig))
        .collect()
}

/// `#[ignore]`-gated entry point. Run with:
///
/// ```bash
/// RYPE_CALIB_GENOMES=/path/to/mags \
/// RYPE_CALIB_LABELS=/path/to/labels.tsv \
/// RYPE_CALIB_OUT=scratch/calib-sweep.parquet \
/// cargo test --release --test calibration_sweep -- --ignored --nocapture
/// ```
///
/// Optional env knobs (all parse the same comma-separated grid format as
/// `parse_grid`):
///   - `RYPE_CALIB_ANI_THRESHOLD` (percent; default 95.0)
///   - `RYPE_CALIB_CONTAINMENT_GRID` (default `0.3,0.4,0.5,0.6,0.7,0.8,0.9`)
///   - `RYPE_CALIB_MIN_ANCHORS_GRID` (default `3,5,10,20,50`)
///
/// The harness deliberately has no assertions on the harness output —
/// the operator picks a value from the resulting Parquet using the
/// methodology doc. The integration's correctness is checked by the
/// non-ignored unit tests below.
#[cfg(feature = "fastx")]
#[test]
#[ignore]
fn run_calibration_sweep() -> Result<()> {
    use rype::cluster::{cluster_contigs, ClusterConfig};
    use rype::ChainParams;

    let genomes_dir = require_env_path("RYPE_CALIB_GENOMES")?;
    let labels_path = require_env_path("RYPE_CALIB_LABELS")?;
    let out_path = require_env_path("RYPE_CALIB_OUT")?;
    let ani_threshold = match std::env::var("RYPE_CALIB_ANI_THRESHOLD") {
        Ok(s) => s
            .parse::<f64>()
            .with_context(|| format!("RYPE_CALIB_ANI_THRESHOLD must parse as f64 (got {:?})", s))?,
        Err(_) => DEFAULT_ANI_THRESHOLD_PCT,
    };

    eprintln!("[calib] loading MAGs from {}", genomes_dir.display());
    let inputs = load_mag_directory(&genomes_dir)?;
    eprintln!(
        "[calib] loaded {} contigs ({:.2} MB total sequence)",
        inputs.len(),
        inputs.iter().map(|c| c.sequence.len() as f64).sum::<f64>() / 1_048_576.0,
    );

    eprintln!(
        "[calib] parsing skani labels at {} (ani >= {}%)",
        labels_path.display(),
        ani_threshold
    );
    let labels = parse_skani_labels(&labels_path, ani_threshold)?;
    eprintln!(
        "[calib] {} positive label pairs after threshold",
        labels.len()
    );

    let grid = sweep_grid()?;
    eprintln!("[calib] sweep grid: {} cells", grid.len());

    // Base config: strain defaults except chain_params/min_chain_containment,
    // which the sweep overrides per cell. Operator-overridable knobs
    // (k/w/min_length/threshold/min_shared/salt) read from env vars so the
    // harness can be retargeted at different ANI tiers without recompiling.
    let base = base_config_from_env()?;
    let chain_window = base.w;

    let mut results: Vec<(f64, u32, CellResult)> = Vec::with_capacity(grid.len());
    for (cell_idx, (containment, min_anchors)) in grid.iter().enumerate() {
        let mut chain_params = ChainParams::starting_for_w(chain_window);
        chain_params.min_anchors = *min_anchors;
        let cfg = ClusterConfig {
            chain_params: Some(chain_params),
            min_chain_containment: Some(*containment),
            ..base.clone()
        };
        let t0 = std::time::Instant::now();
        let result = cluster_contigs(inputs.clone(), &cfg).with_context(|| {
            format!(
                "cluster_contigs failed for cell ({}, {})",
                containment, min_anchors
            )
        })?;
        let absorbed = absorbed_pairs_from_result(&result);
        let cell = score_cell(&absorbed, &labels);
        let (p, r, f1) = metrics(&cell);
        eprintln!(
            "[calib] cell {}/{}: containment={:.3} min_anchors={} → tp={} fp={} fn={} P={:.4} R={:.4} F1={:.4} ({:.1}s)",
            cell_idx + 1,
            grid.len(),
            containment,
            min_anchors,
            cell.tp,
            cell.fp,
            cell.fn_,
            p,
            r,
            f1,
            t0.elapsed().as_secs_f64()
        );
        results.push((*containment, *min_anchors, cell));
    }

    eprintln!("[calib] writing sweep results to {}", out_path.display());
    write_sweep_results_parquet(&out_path, &results)?;
    eprintln!("[calib] done");
    Ok(())
}

/// Required-env helper: return the value as a `PathBuf` or a clear error.
#[cfg(feature = "fastx")]
fn require_env_path(name: &str) -> Result<std::path::PathBuf> {
    std::env::var(name)
        .map(std::path::PathBuf::from)
        .map_err(|_| anyhow!("required env var {} is not set", name))
}

/// Read base `ClusterConfig` knobs from env vars (or fall back to
/// `strain_default()`). Lets the operator sweep at different ANI tiers
/// without recompiling.
#[cfg(feature = "fastx")]
fn base_config_from_env() -> Result<rype::cluster::ClusterConfig> {
    use rype::cluster::ClusterConfig;
    let mut base = ClusterConfig::strain_default();
    if let Ok(s) = std::env::var("RYPE_CALIB_K") {
        base.k = s
            .parse()
            .with_context(|| format!("RYPE_CALIB_K must parse as usize (got {:?})", s))?;
    }
    if let Ok(s) = std::env::var("RYPE_CALIB_W") {
        base.w = s
            .parse()
            .with_context(|| format!("RYPE_CALIB_W must parse as usize (got {:?})", s))?;
    }
    if let Ok(s) = std::env::var("RYPE_CALIB_MIN_LENGTH") {
        base.min_length = s
            .parse()
            .with_context(|| format!("RYPE_CALIB_MIN_LENGTH must parse as u64 (got {:?})", s))?;
    }
    if let Ok(s) = std::env::var("RYPE_CALIB_THRESHOLD") {
        base.threshold = s
            .parse()
            .with_context(|| format!("RYPE_CALIB_THRESHOLD must parse as f64 (got {:?})", s))?;
    }
    if let Ok(s) = std::env::var("RYPE_CALIB_MIN_SHARED") {
        base.min_shared = s
            .parse()
            .with_context(|| format!("RYPE_CALIB_MIN_SHARED must parse as u64 (got {:?})", s))?;
    }
    Ok(base)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the basic round-trip. Three rows, one above threshold, one
    /// below, one self-pair. Confirms the filter, the dedup, and the
    /// self-pair exclusion all fire together.
    #[test]
    fn parse_skani_labels_basic_round_trip() {
        let body = "\
A\tB\t95.5
A\tC\t94.0
A\tA\t100.0
";
        let labels = parse_skani_labels_str(body, 95.0).unwrap();
        assert_eq!(labels.len(), 1);
        assert!(labels.contains(&("A".to_string(), "B".to_string())));
    }

    /// WHY: self-pairs (skani sometimes emits A vs A) must NEVER count
    /// as positive labels. If they did, the recall denominator would be
    /// inflated by N (one per genome) and downstream metrics would be
    /// silently wrong.
    #[test]
    fn parse_skani_labels_self_pairs_ignored() {
        let body = "\
A\tA\t100.0
B\tB\t100.0
A\tB\t99.9
";
        let labels = parse_skani_labels_str(body, 95.0).unwrap();
        assert_eq!(labels.len(), 1);
        assert!(labels.contains(&("A".to_string(), "B".to_string())));
    }

    /// WHY: the threshold is the load-bearing filter. A regression that
    /// flipped the `>=` to `>` would silently drop pairs at exactly the
    /// boundary; this test pins both sides of the boundary.
    #[test]
    fn parse_skani_labels_below_threshold_excluded() {
        let body = "\
A\tB\t94.99
A\tC\t95.0
A\tD\t95.01
";
        let labels = parse_skani_labels_str(body, 95.0).unwrap();
        assert_eq!(labels.len(), 2);
        assert!(labels.contains(&("A".to_string(), "C".to_string())));
        assert!(labels.contains(&("A".to_string(), "D".to_string())));
        assert!(!labels.contains(&("A".to_string(), "B".to_string())));
    }

    /// WHY: skani can emit both `(A, B)` and `(B, A)` (the latter from
    /// the reciprocal triangle of `skani dist`). Both must collapse to
    /// one label, otherwise duplicate-counting inflates the positives
    /// set and silently distorts every downstream metric.
    #[test]
    fn parse_skani_labels_canonical_ordering() {
        let body = "\
A\tB\t99.0
B\tA\t99.0
";
        let labels = parse_skani_labels_str(body, 95.0).unwrap();
        assert_eq!(labels.len(), 1);
        assert!(labels.contains(&("A".to_string(), "B".to_string())));
    }

    /// WHY: a malformed input must fail LOUDLY (project Rule 10).
    /// Silent skipping of a bad row would mean the operator runs the
    /// harness against a corrupted label set and never notices until
    /// the metrics look bizarre.
    #[test]
    fn parse_skani_labels_rejects_malformed_tsv() {
        // Wrong column count.
        let body = "A\tB\n";
        let err = parse_skani_labels_str(body, 95.0).unwrap_err();
        assert!(format!("{}", err).contains("expected 3"));

        // Non-numeric ANI.
        let body = "A\tB\tnot-a-number\n";
        let err = parse_skani_labels_str(body, 95.0).unwrap_err();
        assert!(format!("{}", err).contains("ANI parse error"));

        // Non-finite ANI.
        let body = "A\tB\tNaN\n";
        let err = parse_skani_labels_str(body, 95.0).unwrap_err();
        assert!(format!("{}", err).contains("non-finite"));
    }

    /// WHY: comments and blank lines must be skipped so operators can
    /// annotate the labels file (e.g. with the skani command they ran)
    /// without the parser blowing up. Pinned because a regression that
    /// dropped this would force operators to strip comments before each
    /// run — a footgun the methodology doc explicitly tells them they
    /// don't need to do.
    #[test]
    fn parse_skani_labels_skips_comments_and_blanks() {
        let body = "\
# skani triangle --slow -t 16 -o labels.tsv /path/to/mags/*.fna
# pruned to 3 columns via awk

A\tB\t99.0

# end
";
        let labels = parse_skani_labels_str(body, 95.0).unwrap();
        assert_eq!(labels.len(), 1);
    }

    /// WHY: canonical ordering is the join-key invariant. Two strings
    /// must map to the same `(min, max)` pair regardless of input order.
    /// This is the function `parse_skani_labels` AND the (future) score
    /// step both depend on; pinning it standalone catches a regression
    /// that would break the join in either site.
    #[test]
    fn canonical_pair_is_order_independent() {
        assert_eq!(canonical_pair("A", "B"), ("A".to_string(), "B".to_string()));
        assert_eq!(canonical_pair("B", "A"), ("A".to_string(), "B".to_string()));
        assert_eq!(
            canonical_pair("mag_z::ctg1", "mag_a::ctg2"),
            ("mag_a::ctg2".to_string(), "mag_z::ctg1".to_string())
        );
    }

    /// WHY: the `mut HashSet` insertion happens in a loop; reaching the
    /// end of a multi-row file with a mix of qualifying and disqualifying
    /// rows must not have weird interactions. A previous version of this
    /// kind of parser had an off-by-one where the last row was always
    /// dropped — this test guards the boundary by mixing qualifying and
    /// disqualifying rows interleaved with no trailing newline.
    #[test]
    fn parse_skani_labels_handles_mixed_rows_no_trailing_newline() {
        let body = "A\tB\t99.0\nA\tC\t50.0\nB\tC\t99.0\nD\tE\t50.0";
        let labels = parse_skani_labels_str(body, 95.0).unwrap();
        assert_eq!(labels.len(), 2);
        assert!(labels.contains(&("A".to_string(), "B".to_string())));
        assert!(labels.contains(&("B".to_string(), "C".to_string())));
    }

    // ---------------------------------------------------------------------
    // Phase 2 — score_cell / metrics / sweep_grid / Parquet writer tests.
    // ---------------------------------------------------------------------

    fn pair(a: &str, b: &str) -> (String, String) {
        canonical_pair(a, b)
    }

    /// WHY: the textbook good case. Absorbed = labels exactly: TP equals
    /// both totals, FP=FN=0, all three metrics 1.0. If this drifts the
    /// scoring is fundamentally broken — pin it.
    #[test]
    fn score_cell_perfect_classification() {
        let labels: HashSet<_> = [pair("A", "B"), pair("C", "D")].into_iter().collect();
        let absorbed: HashSet<_> = labels.clone();
        let cell = score_cell(&absorbed, &labels);
        assert_eq!(cell.tp, 2);
        assert_eq!(cell.fp, 0);
        assert_eq!(cell.fn_, 0);
        assert_eq!(cell.total_absorbed, 2);
        assert_eq!(cell.total_positives, 2);
        let (p, r, f1) = metrics(&cell);
        assert_eq!(p, 1.0);
        assert_eq!(r, 1.0);
        assert_eq!(f1, 1.0);
    }

    /// WHY: absorbed predictions exist but none match a label. Precision
    /// is well-defined zero; recall is NaN because there are no positives.
    /// A regression that returned 0.0 instead of NaN for recall here would
    /// silently let an unrelated FP-only sweep cell "tie" cells that
    /// actually have positives. Pin the NaN.
    #[test]
    fn score_cell_all_false_positives() {
        let labels: HashSet<(String, String)> = HashSet::new();
        let absorbed: HashSet<_> = [pair("A", "B"), pair("C", "D")].into_iter().collect();
        let cell = score_cell(&absorbed, &labels);
        assert_eq!(cell.tp, 0);
        assert_eq!(cell.fp, 2);
        assert_eq!(cell.fn_, 0);
        let (p, r, f1) = metrics(&cell);
        assert_eq!(p, 0.0);
        assert!(
            r.is_nan(),
            "expected NaN recall when no positives, got {}",
            r
        );
        assert!(
            f1.is_nan(),
            "expected NaN F1 when recall is NaN, got {}",
            f1
        );
    }

    /// WHY: positives exist but no predictions. Recall well-defined zero;
    /// precision NaN because there are no predictions. Symmetric guard to
    /// `all_false_positives` so a regression that flipped the precision /
    /// recall NaN branches gets caught from both sides.
    #[test]
    fn score_cell_all_false_negatives() {
        let labels: HashSet<_> = [pair("A", "B"), pair("C", "D")].into_iter().collect();
        let absorbed: HashSet<(String, String)> = HashSet::new();
        let cell = score_cell(&absorbed, &labels);
        assert_eq!(cell.tp, 0);
        assert_eq!(cell.fp, 0);
        assert_eq!(cell.fn_, 2);
        let (p, r, f1) = metrics(&cell);
        assert!(
            p.is_nan(),
            "expected NaN precision when no predictions, got {}",
            p
        );
        assert_eq!(r, 0.0);
        assert!(
            f1.is_nan(),
            "expected NaN F1 when precision is NaN, got {}",
            f1
        );
    }

    /// WHY: a sweep cell can produce zero positives AND zero predictions
    /// (very strict gate on a small input). The harness MUST emit a row
    /// for that cell rather than panicking; all three metrics NaN keeps
    /// the row present without polluting downstream sort-by-F1 queries.
    #[test]
    fn score_cell_empty_inputs() {
        let labels: HashSet<(String, String)> = HashSet::new();
        let absorbed: HashSet<(String, String)> = HashSet::new();
        let cell = score_cell(&absorbed, &labels);
        assert_eq!(cell.tp, 0);
        assert_eq!(cell.fp, 0);
        assert_eq!(cell.fn_, 0);
        let (p, r, f1) = metrics(&cell);
        assert!(p.is_nan());
        assert!(r.is_nan());
        assert!(f1.is_nan());
    }

    /// WHY: pin the four NaN-on-zero-denominator cases for `metrics`
    /// independently of `score_cell`, in case a future refactor decouples
    /// them. Includes the "both zero but nonzero TP+FP+FN" case
    /// (precision=0, recall=0, P+R=0 ⇒ F1=NaN per the plan spec).
    #[test]
    fn metrics_handles_nan_correctly() {
        // Precision NaN, recall well-defined.
        let c = CellResult {
            tp: 0,
            fp: 0,
            fn_: 5,
            total_absorbed: 0,
            total_positives: 5,
        };
        let (p, _, f1) = metrics(&c);
        assert!(p.is_nan());
        assert!(f1.is_nan());

        // Recall NaN, precision well-defined.
        let c = CellResult {
            tp: 0,
            fp: 5,
            fn_: 0,
            total_absorbed: 5,
            total_positives: 0,
        };
        let (_, r, f1) = metrics(&c);
        assert!(r.is_nan());
        assert!(f1.is_nan());

        // Both zero (no overlap but both sides nonzero): P+R=0 → F1=NaN.
        let c = CellResult {
            tp: 0,
            fp: 5,
            fn_: 5,
            total_absorbed: 5,
            total_positives: 5,
        };
        let (p, r, f1) = metrics(&c);
        assert_eq!(p, 0.0);
        assert_eq!(r, 0.0);
        assert!(
            f1.is_nan(),
            "P=0 and R=0 with P+R=0 must give F1=NaN per spec, got {}",
            f1
        );
    }

    /// WHY: the spec mandates the 7×5 default grid (35 cells). A future
    /// edit that drops a containment value or adds a min_anchors value
    /// must update the spec AND this test together — pinning the shape
    /// makes that pairing mechanical.
    #[test]
    fn sweep_grid_default_is_seven_by_five() {
        // Clear any env overrides so the test sees defaults regardless of
        // ambient shell state. (Other tests in this module don't touch
        // these vars; setting them just to be defensive.)
        unsafe {
            std::env::remove_var("RYPE_CALIB_CONTAINMENT_GRID");
            std::env::remove_var("RYPE_CALIB_MIN_ANCHORS_GRID");
        }
        let grid = sweep_grid().unwrap();
        assert_eq!(grid.len(), 35, "default grid must be 7 × 5 = 35 cells");
        // First and last cells pin the iteration order: containment is
        // the outer axis (slowest-varying), min_anchors the inner.
        assert_eq!(grid[0], (0.3, 3));
        assert_eq!(grid[34], (0.9, 50));
    }

    /// WHY: pin the env-var format the methodology doc tells operators
    /// to use. The parser is shared by both env vars, so testing it on
    /// the f64 axis is sufficient. Trailing commas and whitespace must
    /// be tolerated (operators copy-paste lists).
    #[test]
    fn sweep_grid_env_var_override_parsed() {
        // Test the inner parser directly (env vars are global state;
        // setting them in tests is racy across threads).
        let v: Vec<f64> = parse_grid("0.4, 0.6,0.8,", "containment").unwrap();
        assert_eq!(v, vec![0.4, 0.6, 0.8]);
        let v: Vec<u32> = parse_grid("3,5,10", "min_anchors").unwrap();
        assert_eq!(v, vec![3, 5, 10]);
        // Empty after parsing → loud error.
        let err = parse_grid::<f64>("  ,, ", "containment").unwrap_err();
        assert!(format!("{}", err).contains("empty"));
        // Malformed values → loud error with the offending token.
        let err = parse_grid::<u32>("3,oops,5", "min_anchors").unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("oops"),
            "error should name the bad token, got: {}",
            msg
        );
    }

    /// WHY: the harness's only persistent output is the Parquet file;
    /// schema drift breaks the methodology doc's duckdb queries. Pin
    /// the 10 columns, their dtypes, and the row count so any schema
    /// edit must update this test together with the spec.
    #[test]
    fn parquet_writer_round_trip() {
        use arrow::datatypes::DataType;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use std::fs::File;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sweep.parquet");

        // 3 cells with deliberately diverse metrics so the writer can't
        // silently drop NaN values.
        let cells = vec![
            (
                0.5_f64,
                5_u32,
                CellResult {
                    tp: 10,
                    fp: 0,
                    fn_: 0,
                    total_absorbed: 10,
                    total_positives: 10,
                },
            ),
            (
                0.7_f64,
                10_u32,
                CellResult {
                    tp: 0,
                    fp: 5,
                    fn_: 0,
                    total_absorbed: 5,
                    total_positives: 0,
                },
            ),
            (
                0.9_f64,
                20_u32,
                CellResult {
                    tp: 0,
                    fp: 0,
                    fn_: 5,
                    total_absorbed: 0,
                    total_positives: 5,
                },
            ),
        ];
        write_sweep_results_parquet(&path, &cells).unwrap();

        let file = File::open(&path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        let batches: Vec<_> = reader.collect::<Result<_, _>>().unwrap();
        assert_eq!(batches.len(), 1);
        let batch = &batches[0];
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 10);

        let schema = batch.schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(
            names,
            vec![
                "min_chain_containment",
                "min_anchors",
                "true_positives",
                "false_positives",
                "false_negatives",
                "precision",
                "recall",
                "f1",
                "total_absorbed",
                "total_positives",
            ]
        );
        assert_eq!(schema.field(0).data_type(), &DataType::Float64);
        assert_eq!(schema.field(1).data_type(), &DataType::UInt32);
        assert_eq!(schema.field(2).data_type(), &DataType::UInt64);
        assert_eq!(schema.field(5).data_type(), &DataType::Float64);
        assert_eq!(schema.field(7).data_type(), &DataType::Float64);
    }

    /// WHY: pin canonicalization for the absorbed-pair set. The harness
    /// joins absorbed pairs against labels through the same
    /// `canonical_pair` helper; if that join breaks because rep_contig
    /// and member_contig come in different orders across cluster runs,
    /// every TP silently becomes an FP. This test pins the contract.
    #[test]
    fn canonical_pair_drives_absorbed_label_join() {
        let labels: HashSet<_> = [pair("A", "B"), pair("C", "D")].into_iter().collect();
        // Absorbed pair where rep > member alphabetically still joins.
        let absorbed: HashSet<_> = [pair("B", "A"), pair("D", "C")].into_iter().collect();
        let cell = score_cell(&absorbed, &labels);
        assert_eq!(cell.tp, 2);
        assert_eq!(cell.fp, 0);
        assert_eq!(cell.fn_, 0);
    }
}
