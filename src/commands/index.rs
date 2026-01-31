//! Index command handlers and helper functions.
//!
//! This module contains the implementation logic for index-related commands.
//! Only Parquet inverted index format is supported.

use anyhow::{anyhow, Context, Result};
use needletail::parse_fastx_file;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rype::config::{parse_config, resolve_path, validate_config};
use rype::parquet_index;
use rype::{
    choose_orientation_sampled, extract_dual_strand_into, extract_into, log_timing,
    merge_sorted_into, MinimizerWorkspace, Orientation, BUCKET_SOURCE_DELIM,
};

use std::collections::HashSet;

use super::helpers::sanitize_bucket_name;

/// Validate that all bucket names are unique.
///
/// Returns an error if duplicate bucket names are found, listing them all.
fn validate_unique_bucket_names(buckets: &[rype::BucketData]) -> Result<()> {
    let mut seen: HashSet<&str> = HashSet::new();
    let mut duplicates: Vec<&str> = Vec::new();

    for bucket in buckets {
        if !seen.insert(&bucket.bucket_name) && !duplicates.contains(&bucket.bucket_name.as_str()) {
            duplicates.push(&bucket.bucket_name);
        }
    }

    if duplicates.is_empty() {
        Ok(())
    } else {
        Err(anyhow!(
            "Duplicate bucket names are not allowed. Found duplicates: {:?}\n\
             Bucket names must be unique to avoid ambiguity in output formats.\n\
             Consider using --separate-buckets or renaming sequences.",
            duplicates
        ))
    }
}

// ============================================================================
// Parquet Index Creation
// ============================================================================

/// Create Parquet inverted index directly from reference files.
#[allow(clippy::too_many_arguments)]
pub fn create_parquet_index_from_refs(
    output: &Path,
    references: &[PathBuf],
    k: usize,
    w: usize,
    salt: u64,
    separate_buckets: bool,
    max_shard_bytes: Option<usize>,
    options: Option<&parquet_index::ParquetWriteOptions>,
) -> Result<()> {
    use rype::{create_parquet_inverted_index, BucketData};

    log::info!(
        "Creating Parquet inverted index at {:?} (K={}, W={}, salt={:#x})",
        output,
        k,
        w,
        salt
    );

    let mut buckets: Vec<BucketData> = Vec::new();
    let mut next_id: u32 = 1;
    let mut ws = MinimizerWorkspace::new();

    for ref_path in references {
        log::info!("Processing reference: {}", ref_path.display());
        let mut reader = parse_fastx_file(ref_path).context("Failed to open reference file")?;
        let filename = ref_path
            .canonicalize()
            .unwrap()
            .to_string_lossy()
            .to_string();

        if separate_buckets {
            while let Some(record) = reader.next() {
                let rec = record.context("Invalid record")?;
                let seq = rec.seq();
                let name = String::from_utf8_lossy(rec.id()).to_string();
                let bucket_id = next_id;
                next_id += 1;

                extract_into(&seq, k, w, salt, &mut ws);
                let mut minimizers = std::mem::take(&mut ws.buffer);
                minimizers.sort_unstable();
                minimizers.dedup();

                let source_label = format!("{}{}{}", filename, BUCKET_SOURCE_DELIM, name);
                buckets.push(BucketData {
                    bucket_id,
                    bucket_name: sanitize_bucket_name(&name),
                    sources: vec![source_label],
                    minimizers,
                });
            }
        } else {
            let bucket_id = next_id;
            next_id += 1;

            let mut all_minimizers: Vec<u64> = Vec::new();
            let mut sources: Vec<String> = Vec::new();

            while let Some(record) = reader.next() {
                let rec = record.context("Invalid record")?;
                let seq = rec.seq();
                let name = String::from_utf8_lossy(rec.id()).to_string();

                extract_into(&seq, k, w, salt, &mut ws);
                all_minimizers.extend_from_slice(&ws.buffer);

                let source_label = format!("{}{}{}", filename, BUCKET_SOURCE_DELIM, name);
                sources.push(source_label);
            }

            all_minimizers.sort_unstable();
            all_minimizers.dedup();

            buckets.push(BucketData {
                bucket_id,
                bucket_name: sanitize_bucket_name(&filename),
                sources,
                minimizers: all_minimizers,
            });
        }
    }

    let total_minimizers: usize = buckets.iter().map(|b| b.minimizers.len()).sum();
    log::info!(
        "Extracted minimizers from {} buckets ({} total)",
        buckets.len(),
        total_minimizers
    );

    // Validate bucket name uniqueness before creating index
    validate_unique_bucket_names(&buckets)?;

    let manifest =
        create_parquet_inverted_index(output, buckets, k, w, salt, max_shard_bytes, options)?;

    log::info!("Created Parquet inverted index:");
    log::info!("  Buckets: {}", manifest.num_buckets);
    if let Some(ref inv) = manifest.inverted {
        log::info!("  Shards: {}", inv.num_shards);
        log::info!("  Total entries: {}", inv.total_entries);
    }
    log::info!("Done.");

    Ok(())
}

/// Build a single bucket from its files, returning the name, minimizers, and sources.
///
/// When `orient_sequences` is true:
/// - The first sequence uses forward strand (establishes baseline)
/// - Subsequent sequences compare forward vs reverse-complement overlap with existing bucket minimizers
/// - The orientation with higher overlap is chosen
///
/// The bucket minimizers are maintained as a sorted, deduplicated Vec throughout,
/// using `merge_sorted_into` for efficient in-place merging.
fn build_single_bucket(
    bucket_name: &str,
    files: &[PathBuf],
    config_dir: &Path,
    k: usize,
    w: usize,
    salt: u64,
    orient_sequences: bool,
) -> Result<(String, Vec<u64>, Vec<String>)> {
    log::info!(
        "Processing bucket '{}'{} ...",
        bucket_name,
        if orient_sequences { " (oriented)" } else { "" }
    );
    let mut ws = MinimizerWorkspace::new();
    let mut bucket_mins: Vec<u64> = Vec::new(); // Kept sorted and deduped
    let mut sources: Vec<String> = Vec::new();
    let mut is_first_sequence = true;
    let mut sample_buffer: Vec<u64> = Vec::new(); // Reusable buffer for orientation sampling

    for file_path in files {
        let abs_path = resolve_path(config_dir, file_path);
        let mut reader = parse_fastx_file(&abs_path).context(format!(
            "Failed to open file {} for bucket '{}'",
            abs_path.display(),
            bucket_name
        ))?;

        let filename = file_path
            .canonicalize()
            .unwrap_or_else(|_| file_path.to_path_buf())
            .to_string_lossy()
            .to_string();

        while let Some(record) = reader.next() {
            let rec = record.context(format!(
                "Invalid record in file {} (bucket '{}')",
                abs_path.display(),
                bucket_name
            ))?;
            let seq_name = String::from_utf8_lossy(rec.id()).to_string();
            let source_label = format!("{}{}{}", filename, BUCKET_SOURCE_DELIM, seq_name);
            sources.push(source_label);

            let seq = rec.seq();

            if is_first_sequence || !orient_sequences {
                // Forward-only: extract, sort, merge in-place
                extract_into(&seq, k, w, salt, &mut ws);
                let mut new_mins = std::mem::take(&mut ws.buffer);
                new_mins.sort_unstable();
                merge_sorted_into(&mut bucket_mins, &new_mins);
                is_first_sequence = false;
            } else {
                // Oriented: extract both strands, sort both, choose best, merge in-place
                let (mut fwd, mut rc) = extract_dual_strand_into(&seq, k, w, salt, &mut ws);
                fwd.sort_unstable();
                rc.sort_unstable();

                let (orientation, _overlap) =
                    choose_orientation_sampled(&bucket_mins, &fwd, &rc, &mut sample_buffer);

                let chosen = match orientation {
                    Orientation::Forward => fwd,
                    Orientation::ReverseComplement => rc,
                };

                merge_sorted_into(&mut bucket_mins, &chosen);
            }
        }
    }

    // bucket_mins is already sorted and deduped from merge_sorted_into
    let minimizer_count = bucket_mins.len();
    log::info!(
        "Completed bucket '{}': {} minimizers",
        bucket_name,
        minimizer_count
    );

    Ok((bucket_name.to_string(), bucket_mins, sources))
}

/// Create Parquet inverted index directly from a TOML config file.
///
/// # Arguments
/// * `config_path` - Path to the TOML configuration file
/// * `cli_max_shard_size` - CLI override for max shard size (takes precedence over config)
/// * `options` - Parquet write options
/// * `cli_orient` - CLI override for orient sequences flag (takes precedence over config)
pub fn build_parquet_index_from_config(
    config_path: &Path,
    cli_max_shard_size: Option<usize>,
    options: Option<&parquet_index::ParquetWriteOptions>,
    cli_orient: bool,
) -> Result<()> {
    use rype::{create_parquet_inverted_index, BucketData};

    let t_total = Instant::now();

    log::info!(
        "Building Parquet index from config: {}",
        config_path.display()
    );

    let cfg = parse_config(config_path)?;
    let config_dir = config_path
        .parent()
        .ok_or_else(|| anyhow!("Invalid config path"))?;

    log::info!("Validating file paths...");
    validate_config(&cfg, config_dir)?;
    log::info!("Validation successful.");

    let max_shard_size = cli_max_shard_size.or(cfg.index.max_shard_size);

    // Determine orient_sequences: CLI --orient flag takes precedence over config file
    let orient_sequences = if cli_orient {
        true // CLI --orient flag overrides everything
    } else {
        cfg.index.orient_sequences.unwrap_or(false) // Config value, or default to false
    };
    if orient_sequences {
        log::info!("Orientation enabled: sequences will be oriented to maximize minimizer overlap");
    }

    // Change output extension from .ryidx to .ryxdi for parquet inverted index
    let output_path = cfg.index.output.with_extension("ryxdi");
    let output_path = resolve_path(config_dir, &output_path);

    log::info!(
        "Creating Parquet inverted index at {:?} (K={}, W={}, salt={:#x})",
        output_path,
        cfg.index.k,
        cfg.index.window,
        cfg.index.salt
    );

    let mut bucket_names: Vec<_> = cfg.buckets.keys().cloned().collect();
    bucket_names.sort();

    const MAX_BUCKETS: usize = 100_000;
    if bucket_names.len() > MAX_BUCKETS {
        return Err(anyhow!(
            "Too many buckets: {} exceeds maximum {}",
            bucket_names.len(),
            MAX_BUCKETS
        ));
    }

    // Build buckets in parallel
    let t_build = Instant::now();
    let bucket_results: Vec<_> = bucket_names
        .par_iter()
        .map(|bucket_name| {
            build_single_bucket(
                bucket_name,
                &cfg.buckets[bucket_name].files,
                config_dir,
                cfg.index.k,
                cfg.index.window,
                cfg.index.salt,
                orient_sequences,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    log_timing(
        "parquet_index: bucket_building",
        t_build.elapsed().as_millis(),
    );

    // Convert to BucketData format for create_parquet_inverted_index
    let buckets: Vec<BucketData> = bucket_results
        .into_iter()
        .enumerate()
        .map(|(idx, (name, minimizers, sources))| BucketData {
            bucket_id: (idx + 1) as u32,
            bucket_name: sanitize_bucket_name(&name),
            sources,
            minimizers,
        })
        .collect();

    let total_minimizers: usize = buckets.iter().map(|b| b.minimizers.len()).sum();
    log::info!(
        "Extracted minimizers from {} buckets ({} total)",
        buckets.len(),
        total_minimizers
    );

    // Validate bucket name uniqueness before creating index
    validate_unique_bucket_names(&buckets)?;

    // Create parquet index
    let t_write = Instant::now();
    let manifest = create_parquet_inverted_index(
        &output_path,
        buckets,
        cfg.index.k,
        cfg.index.window,
        cfg.index.salt,
        max_shard_size,
        options,
    )?;
    log_timing("parquet_index: write", t_write.elapsed().as_millis());

    log::info!("Created Parquet inverted index:");
    log::info!("  Buckets: {}", manifest.num_buckets);
    if let Some(ref inv) = manifest.inverted {
        log::info!("  Shards: {}", inv.num_shards);
        log::info!("  Total entries: {}", inv.total_entries);
    }

    log_timing("parquet_index: total", t_total.elapsed().as_millis());
    log::info!("Done.");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rype::BucketData;
    use std::fs::File;
    use std::io::Write;
    use tempfile::TempDir;

    /// Helper to create a simple FASTA file with one sequence
    fn create_fasta_file(dir: &Path, name: &str, seq: &[u8]) -> PathBuf {
        let path = dir.join(name);
        let mut file = File::create(&path).unwrap();
        writeln!(file, ">seq1").unwrap();
        file.write_all(seq).unwrap();
        writeln!(file).unwrap();
        path
    }

    /// Helper to create a config file for testing
    fn create_test_config(
        dir: &Path,
        output_name: &str,
        buckets: &[(&str, &[&str])],
        k: usize,
        window: usize,
    ) -> PathBuf {
        let config_path = dir.join("config.toml");
        let mut content = format!(
            r#"[index]
k = {}
window = {}
salt = 0x5555555555555555
output = "{}"

"#,
            k, window, output_name
        );

        for (bucket_name, files) in buckets {
            let files_str: Vec<String> = files.iter().map(|f| format!("\"{}\"", f)).collect();
            content.push_str(&format!(
                "[buckets.{}]\nfiles = [{}]\n\n",
                bucket_name,
                files_str.join(", ")
            ));
        }

        let mut file = File::create(&config_path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
        config_path
    }

    #[test]
    fn test_build_single_bucket_extracts_minimizers() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // Create a FASTA file with a sequence long enough for k=32, w=10
        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let fasta_path = create_fasta_file(dir, "test.fa", seq);

        let (name, minimizers, sources) = build_single_bucket(
            "TestBucket",
            &[fasta_path],
            dir,
            32, // k
            10, // w
            0x5555555555555555,
            false, // orient_sequences
        )
        .unwrap();

        assert_eq!(name, "TestBucket");
        assert!(!minimizers.is_empty(), "Should extract some minimizers");
        assert!(!sources.is_empty(), "Should have source labels");

        // Verify minimizers are sorted and deduplicated
        let mut sorted = minimizers.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            minimizers, sorted,
            "Minimizers should be sorted and deduplicated"
        );
    }

    #[test]
    fn test_bucket_result_to_bucket_data_conversion() {
        // Test that build_single_bucket output can be converted to BucketData
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let fasta_path = create_fasta_file(dir, "test.fa", seq);

        let (name, minimizers, sources) = build_single_bucket(
            "TestBucket",
            &[fasta_path],
            dir,
            32,
            10,
            0x5555555555555555,
            false,
        )
        .unwrap();

        // Convert to BucketData (this is the reuse we want)
        let bucket_data = BucketData {
            bucket_id: 1,
            bucket_name: sanitize_bucket_name(&name),
            sources,
            minimizers,
        };

        assert_eq!(bucket_data.bucket_id, 1);
        assert!(!bucket_data.minimizers.is_empty());
        assert!(bucket_data.validate().is_ok(), "BucketData should be valid");
    }

    #[test]
    fn test_build_parquet_index_from_config_creates_index() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // Create test FASTA files
        let seq1 = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let seq2 = b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA";

        create_fasta_file(dir, "ref1.fa", seq1);
        create_fasta_file(dir, "ref2.fa", seq2);

        // Create config
        let config_path = create_test_config(
            dir,
            "test_index.ryidx",
            &[("Bucket1", &["ref1.fa"]), ("Bucket2", &["ref2.fa"])],
            32,
            10,
        );

        // Build parquet index
        let result = build_parquet_index_from_config(&config_path, None, None, false);
        assert!(result.is_ok(), "Should succeed: {:?}", result);

        // Verify the parquet index was created
        let output_path = dir.join("test_index.ryxdi");
        assert!(output_path.exists(), "Parquet index directory should exist");

        // Verify manifest exists
        let manifest_path = output_path.join("manifest.toml");
        assert!(manifest_path.exists(), "Manifest should exist");
    }

    #[test]
    fn test_build_parquet_index_from_config_with_bloom_filter() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        create_fasta_file(dir, "ref.fa", seq);

        let config_path = create_test_config(
            dir,
            "bloom_test.ryidx",
            &[("TestBucket", &["ref.fa"])],
            32,
            10,
        );

        let options = parquet_index::ParquetWriteOptions {
            bloom_filter_enabled: true,
            bloom_filter_fpp: 0.05,
            ..Default::default()
        };

        let result = build_parquet_index_from_config(&config_path, None, Some(&options), false);
        assert!(
            result.is_ok(),
            "Should succeed with bloom filter: {:?}",
            result
        );

        let output_path = dir.join("bloom_test.ryxdi");
        assert!(output_path.exists());
    }

    #[test]
    fn test_build_parquet_index_from_config_invalid_file() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // Create config referencing non-existent file
        let config_path = create_test_config(
            dir,
            "test.ryidx",
            &[("TestBucket", &["nonexistent.fa"])],
            32,
            10,
        );

        let result = build_parquet_index_from_config(&config_path, None, None, false);
        assert!(result.is_err(), "Should fail with missing file");
    }

    // ============================================================================
    // Oriented Bucket Building Tests
    // ============================================================================

    #[test]
    fn test_build_single_bucket_orient_disabled_matches_original() {
        // orient=false should produce same result whether we call it with orient=false
        // This tests that the code path with orient=false doesn't break anything
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let fasta_path = create_fasta_file(dir, "test.fa", seq);

        let (name, minimizers, sources) = build_single_bucket(
            "TestBucket",
            &[fasta_path],
            dir,
            32,
            10,
            0x5555555555555555,
            false, // orient disabled
        )
        .unwrap();

        assert_eq!(name, "TestBucket");
        assert!(!minimizers.is_empty());
        assert!(!sources.is_empty());

        // Verify minimizers are sorted and deduplicated
        let mut sorted = minimizers.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(minimizers, sorted);
    }

    #[test]
    fn test_build_single_bucket_orient_enabled_produces_valid_output() {
        // orient=true should also produce valid sorted, deduplicated output
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let fasta_path = create_fasta_file(dir, "test.fa", seq);

        let (name, minimizers, sources) = build_single_bucket(
            "OrientedBucket",
            &[fasta_path],
            dir,
            32,
            10,
            0x5555555555555555,
            true, // orient enabled
        )
        .unwrap();

        assert_eq!(name, "OrientedBucket");
        assert!(!minimizers.is_empty());
        assert!(!sources.is_empty());

        // Verify minimizers are sorted and deduplicated
        let mut sorted = minimizers.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            minimizers, sorted,
            "Oriented bucket minimizers should be sorted and deduplicated"
        );
    }

    /// Helper to create a multi-sequence FASTA file
    fn create_multi_fasta_file(dir: &Path, name: &str, sequences: &[(&str, &[u8])]) -> PathBuf {
        let path = dir.join(name);
        let mut file = File::create(&path).unwrap();
        for (seq_name, seq) in sequences {
            writeln!(file, ">{}", seq_name).unwrap();
            file.write_all(seq).unwrap();
            writeln!(file).unwrap();
        }
        path
    }

    #[test]
    fn test_build_single_bucket_orient_with_multiple_sequences() {
        // Test that orientation works with multiple sequences:
        // - First sequence establishes baseline (forward)
        // - Subsequent sequences should choose orientation based on overlap
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        // Create two sequences - the second is different but should still work
        let seq1 = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let seq2 = b"TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA";

        let fasta_path =
            create_multi_fasta_file(dir, "multi.fa", &[("seq1", seq1), ("seq2", seq2)]);

        // Build without orientation
        let (_, mins_no_orient, _) = build_single_bucket(
            "NoOrient",
            &[fasta_path.clone()],
            dir,
            32,
            10,
            0x5555555555555555,
            false,
        )
        .unwrap();

        // Build with orientation
        let (_, mins_with_orient, _) = build_single_bucket(
            "WithOrient",
            &[fasta_path],
            dir,
            32,
            10,
            0x5555555555555555,
            true,
        )
        .unwrap();

        // Both should be valid (sorted and deduplicated)
        let mut sorted_no = mins_no_orient.clone();
        sorted_no.sort_unstable();
        sorted_no.dedup();
        assert_eq!(mins_no_orient, sorted_no);

        let mut sorted_with = mins_with_orient.clone();
        sorted_with.sort_unstable();
        sorted_with.dedup();
        assert_eq!(mins_with_orient, sorted_with);

        // The oriented version may have same or different minimizers
        // depending on which orientation was chosen - both are valid
        assert!(!mins_no_orient.is_empty());
        assert!(!mins_with_orient.is_empty());
    }

    #[test]
    fn test_build_single_bucket_orient_first_sequence_uses_forward() {
        // With a single sequence, orient=true and orient=false should produce
        // identical results since the first sequence always uses forward
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();

        let seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
        let fasta_path = create_fasta_file(dir, "single.fa", seq);

        let (_, mins_no_orient, _) = build_single_bucket(
            "NoOrient",
            &[fasta_path.clone()],
            dir,
            32,
            10,
            0x5555555555555555,
            false,
        )
        .unwrap();

        let (_, mins_with_orient, _) = build_single_bucket(
            "WithOrient",
            &[fasta_path],
            dir,
            32,
            10,
            0x5555555555555555,
            true,
        )
        .unwrap();

        // Single sequence: both should be identical since first seq always uses forward
        assert_eq!(
            mins_no_orient, mins_with_orient,
            "First sequence should use forward orientation in both cases"
        );
    }

    // ============================================================================
    // Bucket Name Uniqueness Validation Tests
    // ============================================================================

    #[test]
    fn test_validate_unique_bucket_names_accepts_unique_names() {
        let buckets = vec![
            BucketData {
                bucket_id: 1,
                bucket_name: "Bucket_A".to_string(),
                sources: vec![],
                minimizers: vec![],
            },
            BucketData {
                bucket_id: 2,
                bucket_name: "Bucket_B".to_string(),
                sources: vec![],
                minimizers: vec![],
            },
        ];
        assert!(validate_unique_bucket_names(&buckets).is_ok());
    }

    #[test]
    fn test_validate_unique_bucket_names_rejects_duplicates() {
        let buckets = vec![
            BucketData {
                bucket_id: 1,
                bucket_name: "Duplicate".to_string(),
                sources: vec![],
                minimizers: vec![],
            },
            BucketData {
                bucket_id: 2,
                bucket_name: "Duplicate".to_string(),
                sources: vec![],
                minimizers: vec![],
            },
        ];
        let result = validate_unique_bucket_names(&buckets);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Duplicate"));
    }

    #[test]
    fn test_validate_unique_bucket_names_empty_list() {
        let buckets: Vec<BucketData> = vec![];
        assert!(validate_unique_bucket_names(&buckets).is_ok());
    }
}
