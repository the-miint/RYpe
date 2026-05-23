//! Plan 1.6 — chain parameter calibration harness (Phase 1: parser).
//!
//! This integration test crate ships the offline calibration harness
//! described in `localdocs/chain-calibration.md`. The harness is
//! `#[ignore]`-gated and reads real MAG data + skani-derived labels from
//! env vars at run time; this file's NON-ignored tests cover only the
//! parser and helpers (Phase 1).
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

use anyhow::{anyhow, Context, Result};
use std::collections::HashSet;
use std::fs;
use std::path::Path;

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
}
