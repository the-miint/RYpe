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
use rype::{extract_into, log_timing, MinimizerWorkspace, BUCKET_SOURCE_DELIM};

use super::helpers::sanitize_bucket_name;

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
fn build_single_bucket(
    bucket_name: &str,
    files: &[PathBuf],
    config_dir: &Path,
    k: usize,
    w: usize,
    salt: u64,
) -> Result<(String, Vec<u64>, Vec<String>)> {
    log::info!("Processing bucket '{}'...", bucket_name);
    let mut ws = MinimizerWorkspace::new();
    let mut all_minimizers: Vec<u64> = Vec::new();
    let mut sources: Vec<String> = Vec::new();

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

            extract_into(&rec.seq(), k, w, salt, &mut ws);
            all_minimizers.extend_from_slice(&ws.buffer);
        }
    }

    all_minimizers.sort_unstable();
    all_minimizers.dedup();

    let minimizer_count = all_minimizers.len();
    log::info!(
        "Completed bucket '{}': {} minimizers",
        bucket_name,
        minimizer_count
    );

    Ok((bucket_name.to_string(), all_minimizers, sources))
}

/// Create Parquet inverted index directly from a TOML config file.
pub fn build_parquet_index_from_config(
    config_path: &Path,
    cli_max_shard_size: Option<usize>,
    options: Option<&parquet_index::ParquetWriteOptions>,
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

        let (name, minimizers, sources) =
            build_single_bucket("TestBucket", &[fasta_path], dir, 32, 10, 0x5555555555555555)
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
        let result = build_parquet_index_from_config(&config_path, None, None);
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

        let result = build_parquet_index_from_config(&config_path, None, Some(&options));
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

        let result = build_parquet_index_from_config(&config_path, None, None);
        assert!(result.is_err(), "Should fail with missing file");
    }
}
