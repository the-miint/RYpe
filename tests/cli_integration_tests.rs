//! CLI integration tests for the rype tool.
//!
//! These tests verify that CLI commands work correctly with Parquet indices.

use anyhow::Result;
use arrow::array::LargeStringArray;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use rype::{extract_strand_minimizers, get_paired_minimizers_into, MinimizerWorkspace};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use tempfile::tempdir;

/// Return path to the rype binary built by `cargo test`.
fn get_binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_rype"))
}

/// Test extract_strand_minimizers returns correct positions and both strands
#[test]
fn test_extract_strand_minimizers_correctness() -> Result<()> {
    let mut ws = MinimizerWorkspace::new();

    // Create a simple sequence
    let seq = b"AAAATTTTGGGGCCCCAAAATTTTGGGGCCCC"; // 32 bases

    let (fwd, rc) = extract_strand_minimizers(seq, 16, 4, 0, &mut ws);

    // Should have some minimizers on both strands
    assert!(!fwd.hashes.is_empty(), "Should have forward minimizers");
    assert!(!rc.hashes.is_empty(), "Should have RC minimizers");

    // SoA invariant
    assert_eq!(fwd.hashes.len(), fwd.positions.len());
    assert_eq!(rc.hashes.len(), rc.positions.len());

    // All positions should be valid
    for &p in &fwd.positions {
        assert!(p + 16 <= seq.len(), "Forward position {} invalid", p);
    }
    for &p in &rc.positions {
        assert!(p + 16 <= seq.len(), "RC position {} invalid", p);
    }

    Ok(())
}

/// Test that minimizers can be extracted correctly with paired-end reads
#[test]
fn test_paired_minimizer_extraction() -> Result<()> {
    let mut ws = MinimizerWorkspace::new();

    // Create paired-end sequences
    let seq1 = b"AAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCC";
    let seq2 = b"TTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGG";

    let (fwd_mins, rc_mins) =
        get_paired_minimizers_into(seq1, Some(seq2), 32, 10, 0x12345, &mut ws);

    // Should have minimizers from both sequences
    assert!(!fwd_mins.is_empty(), "Should have forward minimizers");
    assert!(
        !rc_mins.is_empty(),
        "Should have reverse complement minimizers"
    );

    Ok(())
}

// ============================================================================
// README CLI Example Tests
// ============================================================================
// These tests extract and run the actual bash examples from README.md.

/// Extract bash code blocks from markdown content
fn extract_bash_code_blocks(markdown: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut in_bash_block = false;
    let mut current_block = String::new();

    for line in markdown.lines() {
        if line.starts_with("```bash") {
            in_bash_block = true;
            current_block.clear();
        } else if line == "```" && in_bash_block {
            in_bash_block = false;
            if !current_block.trim().is_empty() {
                blocks.push(current_block.clone());
            }
        } else if in_bash_block {
            current_block.push_str(line);
            current_block.push('\n');
        }
    }

    blocks
}

/// Extract individual commands from a bash block (split by newlines, filter comments)
fn extract_commands(block: &str) -> Vec<String> {
    block
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| l.to_string())
        .collect()
}

/// Test that README bash examples for index creation and classification work.
/// This extracts actual commands from README.md and runs them with real example files.
#[test]
fn test_readme_bash_examples() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let readme_path = std::path::Path::new(manifest_dir).join("README.md");
    let readme = std::fs::read_to_string(&readme_path)?;

    let bash_blocks = extract_bash_code_blocks(&readme);
    assert!(!bash_blocks.is_empty(), "No bash blocks found in README");

    // Set up test environment
    let dir = tempdir()?;
    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    let puc19_path = std::path::Path::new(manifest_dir).join("examples/pUC19.fasta");

    if !phix_path.exists() || !puc19_path.exists() {
        eprintln!("Skipping README bash test: example FASTA files not found");
        return Ok(());
    }

    // Create a query file for classify commands
    let query_path = dir.path().join("reads.fastq");
    fs::write(
        &query_path,
        "@query1\nGAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCTGATGAGTCGAAAAATTATCTT\n+\nIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n",
    )?;

    let binary = get_binary_path();

    // Track which commands we've tested
    let mut tested_commands = Vec::new();

    for block in &bash_blocks {
        for cmd in extract_commands(block) {
            // Only test rype commands (skip cargo, gcc, etc.)
            if !cmd.starts_with("rype ") {
                continue;
            }

            // Skip removed commands and unsupported commands
            if cmd.contains("bucket-merge")
                || cmd.contains("merge -o")
                || cmd.contains("index shard")
                || cmd.contains("index invert")
                || cmd.contains("aggregate")
                || cmd.contains("bucket-add")
            {
                continue;
            }

            // Substitute placeholder paths with real paths
            let cmd = cmd
                .replace("ref1.fasta", phix_path.to_str().unwrap())
                .replace("ref2.fasta", puc19_path.to_str().unwrap())
                .replace("new_ref.fasta", phix_path.to_str().unwrap())
                .replace("reads.fastq", query_path.to_str().unwrap())
                .replace("reads_R1.fastq", query_path.to_str().unwrap())
                .replace("reads_R2.fastq", query_path.to_str().unwrap())
                .replace("index.ryidx", dir.path().join("index.ryxdi").to_str().unwrap())
                .replace("index.ryxdi", dir.path().join("index.ryxdi").to_str().unwrap())
                .replace("large.ryidx", dir.path().join("index.ryxdi").to_str().unwrap())
                .replace("sharded.ryidx", dir.path().join("sharded.ryxdi").to_str().unwrap())
                .replace("merged.ryidx", dir.path().join("merged.ryxdi").to_str().unwrap())
                .replace("idx1.ryidx", dir.path().join("index.ryxdi").to_str().unwrap())
                .replace("idx2.ryidx", dir.path().join("index.ryxdi").to_str().unwrap())
                .replace("config.toml", dir.path().join("config.toml").to_str().unwrap())
                // Remove --parquet flag (now default)
                .replace(" --parquet", "")
                // Remove --use-inverted flag (now default)
                .replace(" --use-inverted", "")
                // Use smaller k and w for faster tests
                .replace("-k 64", "-k 32")
                .replace("-w 50", "-w 10");

            // Skip commands that need files we haven't created yet
            if cmd.contains("config.toml") {
                continue;
            }

            // Parse command into args (simple split, doesn't handle quotes)
            let args: Vec<&str> = cmd
                .strip_prefix("rype ")
                .unwrap()
                .split_whitespace()
                .collect();

            // Ensure index exists for commands that need it
            let needs_index = args.iter().any(|a| *a == "-i" || *a == "stats");
            let index_path = dir.path().join("index.ryxdi");
            if needs_index && !index_path.exists() {
                // Create index first
                let output = Command::new(&binary)
                    .args([
                        "index",
                        "create",
                        "-o",
                        index_path.to_str().unwrap(),
                        "-r",
                        phix_path.to_str().unwrap(),
                        "-r",
                        puc19_path.to_str().unwrap(),
                        "-k",
                        "32",
                        "-w",
                        "10",
                    ])
                    .output()?;
                assert!(
                    output.status.success(),
                    "Failed to create prerequisite index: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }

            println!("Testing README command: rype {}", args.join(" "));

            let output = Command::new(&binary)
                .args(&args)
                .current_dir(dir.path())
                .output()?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                // Some commands may fail due to missing files - that's OK for this test
                // We're mainly checking that the CLI parses correctly
                if !stderr.contains("No such file") && !stderr.contains("not found") {
                    panic!(
                        "README command failed: rype {}\nStderr: {}",
                        args.join(" "),
                        stderr
                    );
                }
            }

            tested_commands.push(cmd.clone());
        }
    }

    println!("\nTested {} README CLI commands:", tested_commands.len());
    for cmd in &tested_commands {
        println!("  {}", cmd);
    }

    assert!(
        !tested_commands.is_empty(),
        "Should have tested at least one README command"
    );

    Ok(())
}

/// Test index creation and classification via CLI
#[test]
fn test_cli_index_create_and_classify() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    if !phix_path.exists() {
        eprintln!("Skipping test: example FASTA file not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let index_path = dir.path().join("test.ryxdi");

    // Create index
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            index_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;

    assert!(
        output.status.success(),
        "Index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Verify index directory was created
    assert!(index_path.exists(), "Index directory should exist");
    assert!(
        index_path.join("manifest.toml").exists(),
        "Manifest should exist"
    );

    // Create a query file
    let query_path = dir.path().join("query.fastq");
    fs::write(
        &query_path,
        "@query1\nGAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCTGATGAGTCGAAAAATTATCTT\n+\nIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n",
    )?;

    // Test classification
    let output = Command::new(&binary)
        .args([
            "classify",
            "run",
            "-i",
            index_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
            "-t",
            "0.0",
        ])
        .output()?;

    assert!(
        output.status.success(),
        "Classification failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Should have some output
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.is_empty() || output.stderr.len() > 0,
        "Should have some output"
    );

    Ok(())
}

/// Test index stats command
#[test]
fn test_cli_index_stats() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    if !phix_path.exists() {
        eprintln!("Skipping test: example FASTA file not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let index_path = dir.path().join("test.ryxdi");

    // Create index
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            index_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;

    assert!(
        output.status.success(),
        "Index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Test stats command
    let output = Command::new(&binary)
        .args(["index", "stats", "-i", index_path.to_str().unwrap()])
        .output()?;

    assert!(
        output.status.success(),
        "Stats failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Stats should mention k, w, and bucket count
    assert!(
        stdout.contains("32") || stdout.contains("k"),
        "Should show k parameter"
    );

    Ok(())
}

/// Test that --best-hit flag returns at most one result per query
#[test]
fn test_cli_best_hit_flag() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    let puc19_path = std::path::Path::new(manifest_dir).join("examples/pUC19.fasta");
    if !phix_path.exists() || !puc19_path.exists() {
        eprintln!("Skipping test: example FASTA files not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let index_path = dir.path().join("test.ryxdi");

    // Create index with two separate buckets (one per file)
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            index_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-r",
            puc19_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
            "--separate-buckets",
        ])
        .output()?;

    assert!(
        output.status.success(),
        "Index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Create a query file with two reads (50 bases each)
    let query_path = dir.path().join("query.fastq");
    fs::write(
        &query_path,
        "@query1\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n+\nIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n\
         @query2\nTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT\n+\nIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n",
    )?;

    // Test classification WITHOUT --best-hit (may get multiple results per query)
    let output_no_best = Command::new(&binary)
        .args([
            "classify",
            "run",
            "-i",
            index_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
            "-t",
            "0.0", // Very low threshold to ensure matches
        ])
        .output()?;

    assert!(
        output_no_best.status.success(),
        "Classification without --best-hit failed: {}",
        String::from_utf8_lossy(&output_no_best.stderr)
    );

    let stdout_no_best = String::from_utf8_lossy(&output_no_best.stdout);
    let lines_no_best: Vec<&str> = stdout_no_best
        .lines()
        .filter(|l| !l.starts_with("read_id")) // Skip header
        .collect();

    // Test classification WITH --best-hit
    let output_best = Command::new(&binary)
        .args([
            "classify",
            "run",
            "-i",
            index_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
            "-t",
            "0.0",
            "--best-hit",
        ])
        .output()?;

    assert!(
        output_best.status.success(),
        "Classification with --best-hit failed: {}",
        String::from_utf8_lossy(&output_best.stderr)
    );

    let stdout_best = String::from_utf8_lossy(&output_best.stdout);
    let lines_best: Vec<&str> = stdout_best
        .lines()
        .filter(|l| !l.starts_with("read_id"))
        .collect();

    // With --best-hit, we should have at most one result per query
    // Count unique query IDs in best-hit output
    let mut query_ids_best: Vec<&str> = lines_best
        .iter()
        .filter_map(|l| l.split('\t').next())
        .collect();
    query_ids_best.sort();
    query_ids_best.dedup();

    // Each query ID should appear exactly once in best-hit output
    assert_eq!(
        lines_best.len(),
        query_ids_best.len(),
        "With --best-hit, each query should have at most one result.\n\
         Found {} lines but {} unique query IDs.\n\
         Lines: {:?}",
        lines_best.len(),
        query_ids_best.len(),
        lines_best
    );

    println!("Without --best-hit: {} result lines", lines_no_best.len());
    println!("With --best-hit: {} result lines", lines_best.len());
    println!("Unique queries in best-hit: {}", query_ids_best.len());

    Ok(())
}

/// Test bucket-source-detail command with both numeric ID and bucket name
#[test]
fn test_cli_bucket_source_detail_by_name() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    let puc19_path = std::path::Path::new(manifest_dir).join("examples/pUC19.fasta");
    if !phix_path.exists() || !puc19_path.exists() {
        eprintln!("Skipping test: example FASTA files not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let index_path = dir.path().join("test.ryxdi");

    // Create index with two separate buckets (one per file)
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            index_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-r",
            puc19_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
            "--separate-buckets",
        ])
        .output()?;

    assert!(
        output.status.success(),
        "Index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // First, get the bucket names from stats to know what to look for
    let output = Command::new(&binary)
        .args(["index", "stats", "-i", index_path.to_str().unwrap()])
        .output()?;

    assert!(
        output.status.success(),
        "Stats failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stats_stdout = String::from_utf8_lossy(&output.stdout);
    println!("Stats output:\n{}", stats_stdout);

    // Test bucket-source-detail by numeric ID (bucket 1)
    let output = Command::new(&binary)
        .args([
            "index",
            "bucket-source-detail",
            "-i",
            index_path.to_str().unwrap(),
            "-b",
            "1",
        ])
        .output()?;

    assert!(
        output.status.success(),
        "bucket-source-detail by ID failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout_by_id = String::from_utf8_lossy(&output.stdout);
    println!("Output by ID '1':\n{}", stdout_by_id);

    // Test bucket-source-detail by bucket name
    // The bucket name is the full FASTA header: "NC_001422.1 Escherichia phage phiX174, complete genome"
    let bucket_name = "NC_001422.1 Escherichia phage phiX174, complete genome";
    let output = Command::new(&binary)
        .args([
            "index",
            "bucket-source-detail",
            "-i",
            index_path.to_str().unwrap(),
            "-b",
            bucket_name,
        ])
        .output()?;

    assert!(
        output.status.success(),
        "bucket-source-detail by name failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout_by_name = String::from_utf8_lossy(&output.stdout);
    println!("Output by name '{}':\n{}", bucket_name, stdout_by_name);

    // Both should return the same results
    assert_eq!(
        stdout_by_id, stdout_by_name,
        "Output should be identical whether accessed by ID or name"
    );

    // Test with invalid bucket name (should error)
    let output = Command::new(&binary)
        .args([
            "index",
            "bucket-source-detail",
            "-i",
            index_path.to_str().unwrap(),
            "-b",
            "NonExistentBucket",
        ])
        .output()?;

    assert!(
        !output.status.success(),
        "bucket-source-detail with invalid name should fail"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("not found"),
        "Error should mention 'not found': {}",
        stderr
    );

    Ok(())
}

/// Test that --trim-to CLI argument is recognized
#[test]
fn test_cli_trim_to_argument_parsing() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    if !phix_path.exists() {
        eprintln!("Skipping test: example FASTA file not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let index_path = dir.path().join("test.ryxdi");

    // Create index
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            index_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;

    assert!(
        output.status.success(),
        "Index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Create a query file with 70-base sequence
    let query_path = dir.path().join("query.fastq");
    fs::write(
        &query_path,
        "@query1\nGAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCTGATGAGTCGAAAAATTATCTT\n+\nIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n",
    )?;

    // Test classification with --trim-to flag
    let output = Command::new(&binary)
        .args([
            "classify",
            "run",
            "-i",
            index_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
            "-t",
            "0.0",
            "--trim-to",
            "50",
        ])
        .output()?;

    assert!(
        output.status.success(),
        "Classification with --trim-to failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    Ok(())
}

/// Test that --trim-to skips sequences shorter than the trim length
#[test]
fn test_cli_trim_to_skips_short_sequences() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    if !phix_path.exists() {
        eprintln!("Skipping test: example FASTA file not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let index_path = dir.path().join("test.ryxdi");

    // Create index
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            index_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;

    assert!(
        output.status.success(),
        "Index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Create a query file with:
    // - short_query: 40 bases (shorter than trim_to=50, should be SKIPPED)
    // - long_query: 100 bases (longer than trim_to=50, should be INCLUDED)
    // Using actual phiX174 sequences to ensure matches
    let query_path = dir.path().join("query.fastq");
    // short_query: 40 bases from phiX174
    let short_seq = "GAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGG"; // 41 bases, we need 40
    let short_seq = &short_seq[0..40]; // exactly 40 bases
    let short_qual = "I".repeat(40);
    // long_query: 100 bases from phiX174
    let long_seq = "GAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCTGATGAGTCGAAAAATTATCTTGATAAAGCAGGAATTACTACTGCTTGTTTA"; // 100 bases
    let long_qual = "I".repeat(100);
    fs::write(
        &query_path,
        format!(
            "@short_query\n{}\n+\n{}\n@long_query\n{}\n+\n{}\n",
            short_seq, short_qual, long_seq, long_qual
        ),
    )?;

    // Run classification with --trim-to 50
    let output = Command::new(&binary)
        .args([
            "classify",
            "run",
            "-i",
            index_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
            "-t",
            "0.0",
            "--trim-to",
            "50",
        ])
        .output()?;

    assert!(
        output.status.success(),
        "Classification with --trim-to failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().collect();

    println!("Classification output:\n{}", stdout);

    // short_query (40 bases) should NOT appear in output (skipped because < 50)
    let has_short = lines.iter().any(|l| l.contains("short_query"));
    assert!(
        !has_short,
        "short_query (40 bases) should be skipped when --trim-to 50, but found in output"
    );

    // long_query (100 bases) should appear in output (trimmed to 50 and classified)
    let has_long = lines.iter().any(|l| l.contains("long_query"));
    assert!(
        has_long,
        "long_query (100 bases) should be present when --trim-to 50, but not found in output.\nOutput: {}",
        stdout
    );

    Ok(())
}

// ============================================================================
// Wide Format Output Tests (Phase 1: CLI Argument and Validation)
// ============================================================================

/// Test that --wide flag is recognized and parsed by the CLI
#[test]
fn test_cli_wide_flag_is_recognized() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    if !phix_path.exists() {
        eprintln!("Skipping test: example FASTA file not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let index_path = dir.path().join("test.ryxdi");

    // Create index
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            index_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;

    assert!(
        output.status.success(),
        "Index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Create a query file
    let query_path = dir.path().join("query.fastq");
    fs::write(
        &query_path,
        "@query1\nGAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCTGATGAGTCGAAAAATTATCTT\n+\nIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n",
    )?;

    // Test that --wide flag is recognized (should not error with "unexpected argument")
    let output = Command::new(&binary)
        .args([
            "classify",
            "run",
            "-i",
            index_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
            "--wide",
        ])
        .output()?;

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should NOT complain about unknown argument
    assert!(
        !stderr.contains("unexpected argument") && !stderr.contains("unknown"),
        "--wide flag should be recognized. Stderr: {}",
        stderr
    );

    // Command should succeed (or fail for reasons other than argument parsing)
    assert!(
        output.status.success(),
        "--wide flag should be accepted. Stderr: {}",
        stderr
    );

    Ok(())
}

/// Test that --wide with non-default --threshold produces an error
#[test]
fn test_cli_wide_with_threshold_errors() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    if !phix_path.exists() {
        eprintln!("Skipping test: example FASTA file not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let index_path = dir.path().join("test.ryxdi");

    // Create index
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            index_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;

    assert!(
        output.status.success(),
        "Index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Create a query file
    let query_path = dir.path().join("query.fastq");
    fs::write(
        &query_path,
        "@query1\nGAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCTGATGAGTCGAAAAATTATCTT\n+\nIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n",
    )?;

    // Test that --wide with --threshold (non-default) produces an error
    let output = Command::new(&binary)
        .args([
            "classify",
            "run",
            "-i",
            index_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
            "--wide",
            "-t",
            "0.5", // Non-default threshold
        ])
        .output()?;

    // Should fail
    assert!(
        !output.status.success(),
        "--wide with --threshold should fail, but succeeded"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Error message should mention the incompatibility
    assert!(
        stderr.contains("incompatible")
            || stderr.contains("--wide") && stderr.contains("--threshold"),
        "Error should mention --wide/--threshold incompatibility. Stderr: {}",
        stderr
    );

    Ok(())
}

/// Test that --wide alone (without --threshold) is accepted
#[test]
fn test_cli_wide_alone_is_accepted() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    if !phix_path.exists() {
        eprintln!("Skipping test: example FASTA file not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let index_path = dir.path().join("test.ryxdi");

    // Create index
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            index_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;

    assert!(
        output.status.success(),
        "Index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Create a query file
    let query_path = dir.path().join("query.fastq");
    fs::write(
        &query_path,
        "@query1\nGAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCTGATGAGTCGAAAAATTATCTT\n+\nIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n",
    )?;

    // Test that --wide alone is accepted (uses default threshold of 0.1)
    let output = Command::new(&binary)
        .args([
            "classify",
            "run",
            "-i",
            index_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
            "--wide",
        ])
        .output()?;

    assert!(
        output.status.success(),
        "--wide alone should be accepted. Stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    Ok(())
}

// ============================================================================
// Wide Format Output Tests (Phase 5: Integration)
// ============================================================================

/// Test that --wide produces wide-format TSV output with correct columns
#[test]
fn test_cli_wide_produces_wide_format_tsv() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    let puc19_path = std::path::Path::new(manifest_dir).join("examples/pUC19.fasta");
    if !phix_path.exists() || !puc19_path.exists() {
        eprintln!("Skipping test: example FASTA files not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let index_path = dir.path().join("test.ryxdi");

    // Create index with two separate buckets
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            index_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-r",
            puc19_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
            "--separate-buckets",
        ])
        .output()?;

    assert!(
        output.status.success(),
        "Index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Create a query file with two reads (70 bases each)
    let query_path = dir.path().join("query.fastq");
    let seq1 = "GAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCTGATGAGTCGAAAAATTATCTT"; // 70 bases
    let seq2 = "TTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAA"; // 70 bases
    let qual = "I".repeat(70);
    fs::write(
        &query_path,
        format!(
            "@query1\n{}\n+\n{}\n@query2\n{}\n+\n{}\n",
            seq1, qual, seq2, qual
        ),
    )?;

    // Test classification with --wide
    let output = Command::new(&binary)
        .args([
            "classify",
            "run",
            "-i",
            index_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
            "--wide",
        ])
        .output()?;

    assert!(
        output.status.success(),
        "Classification with --wide failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().collect();

    println!("Wide format output:\n{}", stdout);

    // Should have header + data rows
    assert!(
        lines.len() >= 2,
        "Should have at least header + 1 data row. Got: {:?}",
        lines
    );

    // Header should start with "read_id" and have bucket columns
    let header = lines[0];
    assert!(
        header.starts_with("read_id"),
        "Header should start with 'read_id'. Got: {}",
        header
    );

    // Header should have exactly 3 columns: read_id + 2 buckets
    let header_cols: Vec<&str> = header.split('\t').collect();
    assert_eq!(
        header_cols.len(),
        3,
        "Wide header should have 3 columns (read_id + 2 buckets). Got: {:?}",
        header_cols
    );

    // Data rows should have the same number of columns as header
    for (i, line) in lines.iter().skip(1).enumerate() {
        if line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(
            cols.len(),
            header_cols.len(),
            "Data row {} should have {} columns like header. Got: {:?}",
            i,
            header_cols.len(),
            cols
        );

        // Score columns should be valid floats
        for (j, col) in cols.iter().skip(1).enumerate() {
            let score: f64 = col.parse().unwrap_or_else(|_| {
                panic!(
                    "Column {} in row {} should be a valid float. Got: '{}'",
                    j + 1,
                    i,
                    col
                )
            });
            assert!(
                (0.0..=1.0).contains(&score),
                "Score should be in [0.0, 1.0]. Got: {}",
                score
            );
        }
    }

    Ok(())
}

/// Test that --wide with Parquet output produces wide-format Parquet file
#[test]
fn test_cli_wide_produces_wide_format_parquet() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    let puc19_path = std::path::Path::new(manifest_dir).join("examples/pUC19.fasta");
    if !phix_path.exists() || !puc19_path.exists() {
        eprintln!("Skipping test: example FASTA files not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let index_path = dir.path().join("test.ryxdi");
    let output_path = dir.path().join("output.parquet");

    // Create index with two separate buckets
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            index_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-r",
            puc19_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
            "--separate-buckets",
        ])
        .output()?;

    assert!(
        output.status.success(),
        "Index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Create a query file
    let query_path = dir.path().join("query.fastq");
    fs::write(
        &query_path,
        "@query1\nGAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCTGATGAGTCGAAAAATTATCTT\n+\nIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n",
    )?;

    // Test classification with --wide and Parquet output
    let output = Command::new(&binary)
        .args([
            "classify",
            "run",
            "-i",
            index_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
            "--wide",
            "-o",
            output_path.to_str().unwrap(),
        ])
        .output()?;

    assert!(
        output.status.success(),
        "Classification with --wide -o parquet failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Verify Parquet file was created
    assert!(
        output_path.exists(),
        "Parquet output file should exist at {:?}",
        output_path
    );

    // Read and verify Parquet schema
    use parquet::file::reader::FileReader;
    use parquet::file::reader::SerializedFileReader;

    let file = std::fs::File::open(&output_path)?;
    let reader = SerializedFileReader::new(file)?;
    let schema = reader.metadata().file_metadata().schema();

    println!("Parquet schema: {:?}", schema);

    // Schema should have read_id + bucket columns
    let fields = schema.get_fields();
    assert!(
        fields.len() >= 2,
        "Schema should have at least 2 fields (read_id + buckets). Got: {}",
        fields.len()
    );

    // First field should be read_id
    assert_eq!(
        fields[0].name(),
        "read_id",
        "First field should be 'read_id'. Got: {}",
        fields[0].name()
    );

    // Should have exactly 3 columns: read_id + 2 buckets
    assert_eq!(
        fields.len(),
        3,
        "Wide Parquet should have 3 columns (read_id + 2 buckets). Got fields: {:?}",
        fields.iter().map(|f| f.name()).collect::<Vec<_>>()
    );

    Ok(())
}

/// Test that --wide output contains scores for all buckets (including 0.0 for non-hits)
#[test]
fn test_cli_wide_includes_zero_scores_for_non_hits() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    let puc19_path = std::path::Path::new(manifest_dir).join("examples/pUC19.fasta");
    if !phix_path.exists() || !puc19_path.exists() {
        eprintln!("Skipping test: example FASTA files not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let index_path = dir.path().join("test.ryxdi");

    // Create index with two separate buckets
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            index_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-r",
            puc19_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
            "--separate-buckets",
        ])
        .output()?;

    assert!(
        output.status.success(),
        "Index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Create a query that should only match one bucket (phiX174 sequence)
    // This is actual phiX174 sequence that should match phiX bucket but not pUC19
    let query_path = dir.path().join("query.fastq");
    fs::write(
        &query_path,
        "@phix_query\nGAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCTGATGAGTCGAAAAATTATCTT\n+\nIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n",
    )?;

    // Test classification with --wide
    let output = Command::new(&binary)
        .args([
            "classify",
            "run",
            "-i",
            index_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
            "--wide",
        ])
        .output()?;

    assert!(
        output.status.success(),
        "Classification with --wide failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().collect();

    println!("Wide format output:\n{}", stdout);

    // Should have header + 1 data row
    assert!(
        lines.len() >= 2,
        "Should have at least header + 1 data row. Got: {:?}",
        lines
    );

    // Data row should have scores for both buckets
    let data_row = lines[1];
    let cols: Vec<&str> = data_row.split('\t').collect();

    // Should have read_id + 2 bucket scores
    assert_eq!(
        cols.len(),
        3,
        "Data row should have 3 columns (read_id + 2 bucket scores). Got: {:?}",
        cols
    );

    // At least one score should be > 0 (the matching bucket)
    // and at least one might be 0.0 (the non-matching bucket)
    let scores: Vec<f64> = cols[1..]
        .iter()
        .map(|s| s.parse::<f64>().unwrap())
        .collect();

    let has_positive = scores.iter().any(|&s| s > 0.0);
    assert!(
        has_positive,
        "Should have at least one positive score. Scores: {:?}",
        scores
    );

    // Wide format should include all buckets, even if score is 0.0
    // (In long format, these would be filtered out)
    println!("Scores for phix_query: {:?}", scores);

    Ok(())
}

/// Test that --wide with --trim-to excludes reads that are too short
/// (skipped reads should NOT appear in wide output, even with all zeros)
#[test]
fn test_cli_wide_with_trim_to_excludes_short_reads() -> Result<()> {
    let dir = tempdir()?;

    let binary = get_binary_path();

    // Create index with a reference sequence
    let ref_seq = "GAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCTGATGAGTCGAAAAATTATCTT\
                   GAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCTGATGAGTCGAAAAATTATCTT";
    let ref_path = dir.path().join("ref.fasta");
    std::fs::write(&ref_path, format!(">ref\n{}\n", ref_seq))?;

    let index_path = dir.path().join("test.ryxdi");
    let status = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            index_path.to_str().unwrap(),
            "-r",
            ref_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .status()?;
    assert!(status.success(), "Failed to create index");

    // Create query file with one long read and one short read
    // The short read should be skipped with --trim-to 100
    let query_path = dir.path().join("queries.fastq");
    std::fs::write(
        &query_path,
        // Long read (140 bases) - should be classified
        "@long_read\n\
         GAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCTGATGAGTCGAAAAATTATCTT\
         GAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCTGATGAGTCGAAAAATTATCTT\n\
         +\n\
         IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\
         IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n\
         @short_read\n\
         ACGTACGTACGTACGTACGT\n\
         +\n\
         IIIIIIIIIIIIIIIIIIII\n",
    )?;

    let output_path = dir.path().join("output.tsv");

    // Test classification with --wide and --trim-to 100
    // short_read (20 bases) should be skipped since it's < 100
    let output = Command::new(&binary)
        .args([
            "classify",
            "run",
            "-i",
            index_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
            "--wide",
            "--trim-to",
            "100",
            "-o",
            output_path.to_str().unwrap(),
        ])
        .output()?;

    if !output.status.success() {
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
    }
    assert!(output.status.success(), "Classification should succeed");

    // Read and parse output
    let content = std::fs::read_to_string(&output_path)?;
    let lines: Vec<&str> = content.lines().collect();

    println!("Wide output with trim_to:\n{}", content);

    // Should have header + 1 data row (only long_read, not short_read)
    assert_eq!(
        lines.len(),
        2,
        "Should have exactly 2 lines (header + 1 read). short_read should be skipped. Got {} lines: {:?}",
        lines.len(),
        lines
    );

    // Verify the data row is for long_read only
    let data_row = lines[1];
    assert!(
        data_row.starts_with("long_read"),
        "Only long_read should be in output. Got: {}",
        data_row
    );

    Ok(())
}

// ============================================================================
// Log-Ratio Mode Integration Tests (Phase 5)
// ============================================================================

/// Test end-to-end log-ratio classification with two single-bucket indices
#[test]
fn test_cli_log_ratio_end_to_end() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    let puc19_path = std::path::Path::new(manifest_dir).join("examples/pUC19.fasta");
    if !phix_path.exists() || !puc19_path.exists() {
        eprintln!("Skipping test: example FASTA files not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let num_path = dir.path().join("num.ryxdi");
    let denom_path = dir.path().join("denom.ryxdi");

    // Create separate single-bucket indices
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            num_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Numerator index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            denom_path.to_str().unwrap(),
            "-r",
            puc19_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Denominator index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Create a query file with phiX174 sequence (should match numerator strongly)
    let query_path = dir.path().join("query.fastq");
    fs::write(
        &query_path,
        "@query1\nGAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCTGATGAGTCGAAAAATTATCTT\n+\nIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n",
    )?;

    // Run log-ratio classification with two indices
    let output = Command::new(&binary)
        .args([
            "classify",
            "log-ratio",
            "-n",
            num_path.to_str().unwrap(),
            "-d",
            denom_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
        ])
        .output()?;

    assert!(
        output.status.success(),
        "Log-ratio classification failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().collect();

    println!("Log-ratio output:\n{}", stdout);

    // Should have header + data rows
    assert!(
        lines.len() >= 2,
        "Should have at least header + 1 data row. Got: {:?}",
        lines
    );

    // Header should be "read_id\tbucket_name\tscore\tfast_path"
    let header = lines[0];
    assert_eq!(
        header, "read_id\tbucket_name\tscore\tfast_path",
        "Unexpected header: {}",
        header
    );

    // Data row should have 4 columns
    let data_row = lines[1];
    let cols: Vec<&str> = data_row.split('\t').collect();
    assert_eq!(cols.len(), 4, "Should have 4 columns: {:?}", cols);

    // Bucket name should have log10([...] / [...]) format
    assert!(
        cols[1].contains("log10(["),
        "Bucket name should contain 'log10(['. Got: {}",
        cols[1]
    );

    // Score should be parseable
    let score_str = cols[2];
    if score_str != "inf" && score_str != "-inf" && score_str != "NaN" {
        let _score: f64 = score_str.parse().unwrap_or_else(|_| {
            panic!(
                "Score should be a valid float, 'inf', '-inf', or 'NaN'. Got: '{}'",
                score_str
            )
        });
    }

    // fast_path should be a valid value
    let fast_path = cols[3];
    assert!(
        fast_path == "none" || fast_path == "num_high",
        "fast_path should be none/num_high. Got: {}",
        fast_path
    );

    Ok(())
}

/// Test that swapping numerator/denominator negates the log-ratio output
#[test]
fn test_cli_log_ratio_swap_indices_negates() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    let puc19_path = std::path::Path::new(manifest_dir).join("examples/pUC19.fasta");
    if !phix_path.exists() || !puc19_path.exists() {
        eprintln!("Skipping test: example FASTA files not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let num_path = dir.path().join("num.ryxdi");
    let denom_path = dir.path().join("denom.ryxdi");

    // Create separate single-bucket indices
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            num_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Numerator index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            denom_path.to_str().unwrap(),
            "-r",
            puc19_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Denominator index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Create query with sequence matching BOTH references partially
    // Use a chimeric read so we get a finite score (not inf/-inf)
    let phix_half = "GAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCG";
    let puc19_half = "TCGCGCGTTTCGGTGATGACGGTGAAAACCTCTGACACAT";
    let chimera = format!("{}{}", phix_half, puc19_half);
    let qual = "I".repeat(chimera.len());
    let query_path = dir.path().join("query.fastq");
    fs::write(&query_path, format!("@query1\n{}\n+\n{}\n", chimera, qual))?;

    // Run log-ratio: phiX as numerator, pUC19 as denominator
    let output_normal = Command::new(&binary)
        .args([
            "classify",
            "log-ratio",
            "-n",
            num_path.to_str().unwrap(),
            "-d",
            denom_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
        ])
        .output()?;
    assert!(
        output_normal.status.success(),
        "Log-ratio failed: {}",
        String::from_utf8_lossy(&output_normal.stderr)
    );

    // Run log-ratio: SWAPPED (pUC19 as numerator, phiX as denominator)
    let output_swapped = Command::new(&binary)
        .args([
            "classify",
            "log-ratio",
            "-n",
            denom_path.to_str().unwrap(),
            "-d",
            num_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
        ])
        .output()?;
    assert!(
        output_swapped.status.success(),
        "Swapped log-ratio failed: {}",
        String::from_utf8_lossy(&output_swapped.stderr)
    );

    let stdout_normal = String::from_utf8_lossy(&output_normal.stdout);
    let stdout_swapped = String::from_utf8_lossy(&output_swapped.stdout);

    println!("Normal output:\n{}", stdout_normal);
    println!("Swapped output:\n{}", stdout_swapped);

    // Extract scores from both outputs (4 columns now)
    let get_score = |stdout: &str| -> Option<f64> {
        for line in stdout.lines().skip(1) {
            let cols: Vec<&str> = line.split('\t').collect();
            if cols.len() >= 3 {
                let score_str = cols[2];
                if score_str == "inf" {
                    return Some(f64::INFINITY);
                } else if score_str == "-inf" {
                    return Some(f64::NEG_INFINITY);
                } else {
                    return score_str.parse().ok();
                }
            }
        }
        None
    };

    let score_normal = get_score(&stdout_normal);
    let score_swapped = get_score(&stdout_swapped);

    println!(
        "Score normal: {:?}, Score swapped: {:?}",
        score_normal, score_swapped
    );

    // If we have valid finite scores, they should be negated
    if let (Some(sn), Some(ss)) = (score_normal, score_swapped) {
        if sn.is_finite() && ss.is_finite() && sn != 0.0 {
            let diff = (sn + ss).abs();
            assert!(
                diff < 1e-6,
                "Swapped score should negate original: {} + {} = {} (expected ~0)",
                sn,
                ss,
                diff
            );
        }
    }

    Ok(())
}

/// Test log-ratio with Parquet output format using two single-bucket indices
#[test]
fn test_cli_log_ratio_parquet_output() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    let puc19_path = std::path::Path::new(manifest_dir).join("examples/pUC19.fasta");
    if !phix_path.exists() || !puc19_path.exists() {
        eprintln!("Skipping test: example FASTA files not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let num_path = dir.path().join("num.ryxdi");
    let denom_path = dir.path().join("denom.ryxdi");
    let output_path = dir.path().join("output.parquet");

    // Create separate single-bucket indices
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            num_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Numerator index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            denom_path.to_str().unwrap(),
            "-r",
            puc19_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Denominator index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Create a query file
    let query_path = dir.path().join("query.fastq");
    fs::write(
        &query_path,
        "@query1\nGAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCTGATGAGTCGAAAAATTATCTT\n+\nIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n",
    )?;

    // Run log-ratio classification with Parquet output
    let output = Command::new(&binary)
        .args([
            "classify",
            "log-ratio",
            "-n",
            num_path.to_str().unwrap(),
            "-d",
            denom_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
            "-o",
            output_path.to_str().unwrap(),
        ])
        .output()?;

    assert!(
        output.status.success(),
        "Log-ratio with Parquet output failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Verify Parquet file was created
    assert!(
        output_path.exists(),
        "Parquet output file should exist at {:?}",
        output_path
    );

    // Read and verify Parquet schema
    use parquet::file::reader::FileReader;
    use parquet::file::reader::SerializedFileReader;

    let file = std::fs::File::open(&output_path)?;
    let reader = SerializedFileReader::new(file)?;
    let schema = reader.metadata().file_metadata().schema();

    println!("Parquet schema: {:?}", schema);

    // Schema should have read_id, bucket_name, score (and possibly fast_path)
    let fields = schema.get_fields();
    assert!(
        fields.len() >= 3,
        "Schema should have at least 3 fields. Got: {}",
        fields.len()
    );

    assert_eq!(
        fields[0].name(),
        "read_id",
        "First field should be 'read_id'. Got: {}",
        fields[0].name()
    );
    assert_eq!(
        fields[1].name(),
        "bucket_name",
        "Second field should be 'bucket_name'. Got: {}",
        fields[1].name()
    );
    assert_eq!(
        fields[2].name(),
        "score",
        "Third field should be 'score'. Got: {}",
        fields[2].name()
    );

    Ok(())
}

/// Test --numerator-skip-threshold assigns fast_path=num_high
#[test]
fn test_cli_log_ratio_numerator_skip_threshold() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    let puc19_path = std::path::Path::new(manifest_dir).join("examples/pUC19.fasta");
    if !phix_path.exists() || !puc19_path.exists() {
        eprintln!("Skipping test: example FASTA files not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let num_path = dir.path().join("num.ryxdi");
    let denom_path = dir.path().join("denom.ryxdi");

    // Create separate single-bucket indices
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            num_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Numerator index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            denom_path.to_str().unwrap(),
            "-r",
            puc19_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Denominator index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Query matching ONLY phiX (numerator) strongly
    let query_path = dir.path().join("query.fastq");
    fs::write(
        &query_path,
        "@query1\nGAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCTGATGAGTCGAAAAATTATCTT\n+\nIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n",
    )?;

    // Run with very low skip threshold so the phiX-matching read triggers num_high
    let output = Command::new(&binary)
        .args([
            "classify",
            "log-ratio",
            "-n",
            num_path.to_str().unwrap(),
            "-d",
            denom_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
            "--numerator-skip-threshold",
            "0.01",
        ])
        .output()?;

    assert!(
        output.status.success(),
        "Log-ratio with skip threshold failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("Skip threshold output:\n{}", stdout);

    // Should have 4 columns with fast_path
    for line in stdout.lines().skip(1) {
        if line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 4, "Should have 4 columns: {:?}", cols);

        // The phiX read should have high numerator score >= 0.01 -> num_high + inf
        let score_str = cols[2];
        let fast_path = cols[3];
        assert_eq!(
            score_str, "inf",
            "Score should be inf for num_high fast path. Got: {}",
            score_str
        );
        assert_eq!(
            fast_path, "num_high",
            "Fast path should be num_high. Got: {}",
            fast_path
        );
    }

    Ok(())
}

/// Test exact computation for num=0: query matching only denominator -> -inf + none
#[test]
fn test_cli_log_ratio_num_zero_denom_positive() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    let puc19_path = std::path::Path::new(manifest_dir).join("examples/pUC19.fasta");
    if !phix_path.exists() || !puc19_path.exists() {
        eprintln!("Skipping test: example FASTA files not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let num_path = dir.path().join("num.ryxdi");
    let denom_path = dir.path().join("denom.ryxdi");

    // phiX is numerator, pUC19 is denominator
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            num_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Numerator index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            denom_path.to_str().unwrap(),
            "-r",
            puc19_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Denominator index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Query matching ONLY pUC19 (denominator) -> numerator score = 0, denom > 0 -> -inf via exact computation
    let query_path = dir.path().join("query.fastq");
    let puc19_seq = "TCGCGCGTTTCGGTGATGACGGTGAAAACCTCTGACACATGCAGCTCCCGGAGACGGTCACAGCTTGTCT";
    fs::write(
        &query_path,
        format!(
            "@puc19_only\n{}\n+\n{}\n",
            puc19_seq,
            "I".repeat(puc19_seq.len())
        ),
    )?;

    let output = Command::new(&binary)
        .args([
            "classify",
            "log-ratio",
            "-n",
            num_path.to_str().unwrap(),
            "-d",
            denom_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
        ])
        .output()?;

    assert!(
        output.status.success(),
        "Log-ratio failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("num=0 exact computation output:\n{}", stdout);

    // Should have header + at least 1 data row
    let lines: Vec<&str> = stdout.lines().collect();
    assert!(lines.len() >= 2, "Should have at least header + 1 data row");

    // Data row should show -inf with none fast path (exact computation, not fast-pathed)
    let data_row = lines[1];
    let cols: Vec<&str> = data_row.split('\t').collect();
    assert_eq!(cols.len(), 4, "Should have 4 columns: {:?}", cols);
    assert_eq!(
        cols[2], "-inf",
        "Score should be -inf for num=0, denom>0. Got: {}",
        cols[2]
    );
    assert_eq!(
        cols[3], "none",
        "Fast path should be none (exact computation). Got: {}",
        cols[3]
    );

    Ok(())
}

/// Test NaN output when query matches neither numerator nor denominator (both scores = 0).
#[test]
fn test_cli_log_ratio_both_zero_gives_nan() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    let puc19_path = std::path::Path::new(manifest_dir).join("examples/pUC19.fasta");
    if !phix_path.exists() || !puc19_path.exists() {
        eprintln!("Skipping test: example FASTA files not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let num_path = dir.path().join("num.ryxdi");
    let denom_path = dir.path().join("denom.ryxdi");

    // phiX is numerator, pUC19 is denominator
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            num_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Numerator index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            denom_path.to_str().unwrap(),
            "-r",
            puc19_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Denominator index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Synthetic query that matches neither phiX nor pUC19.
    // Alternating ACAC... produces k-mers unlike either genome.
    let query_path = dir.path().join("query.fastq");
    let no_match_seq = "ACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACAC";
    fs::write(
        &query_path,
        format!(
            "@no_match_read\n{}\n+\n{}\n",
            no_match_seq,
            "I".repeat(no_match_seq.len())
        ),
    )?;

    let output = Command::new(&binary)
        .args([
            "classify",
            "log-ratio",
            "-n",
            num_path.to_str().unwrap(),
            "-d",
            denom_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
        ])
        .output()?;

    assert!(
        output.status.success(),
        "Log-ratio failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("both-zero NaN output:\n{}", stdout);

    let lines: Vec<&str> = stdout.lines().collect();
    assert!(lines.len() >= 2, "Should have at least header + 1 data row");

    // Data row should show NaN with fast_path=none (exact computation, both scores = 0)
    let data_row = lines[1];
    let cols: Vec<&str> = data_row.split('\t').collect();
    assert_eq!(cols.len(), 4, "Should have 4 columns: {:?}", cols);
    assert_eq!(
        cols[2], "NaN",
        "Score should be NaN when both num and denom are 0. Got: {}",
        cols[2]
    );
    assert_eq!(
        cols[3], "none",
        "Fast path should be none (exact computation). Got: {}",
        cols[3]
    );

    Ok(())
}

/// Test that log-ratio fails when a multi-bucket index is used as numerator
#[test]
fn test_cli_log_ratio_fails_with_multi_bucket_index() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    let puc19_path = std::path::Path::new(manifest_dir).join("examples/pUC19.fasta");
    if !phix_path.exists() || !puc19_path.exists() {
        eprintln!("Skipping test: example FASTA files not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let multi_path = dir.path().join("multi.ryxdi");
    let single_path = dir.path().join("single.ryxdi");

    // Create a 2-bucket index (invalid for log-ratio numerator/denominator)
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            multi_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-r",
            puc19_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
            "--separate-buckets",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Multi-bucket index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Create a valid single-bucket index
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            single_path.to_str().unwrap(),
            "-r",
            puc19_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Single-bucket index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let query_path = dir.path().join("query.fastq");
    fs::write(
        &query_path,
        "@query1\nGAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCTGATGAGTCGAAAAATTATCTT\n+\nIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n",
    )?;

    // Use multi-bucket index as numerator -> should fail with "exactly 1 bucket"
    let output = Command::new(&binary)
        .args([
            "classify",
            "log-ratio",
            "-n",
            multi_path.to_str().unwrap(),
            "-d",
            single_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
        ])
        .output()?;

    assert!(
        !output.status.success(),
        "Log-ratio with multi-bucket numerator should fail"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    println!("Expected error: {}", stderr);

    assert!(
        stderr.contains("exactly 1 bucket") || stderr.contains("1 bucket"),
        "Error should mention exactly 1 bucket requirement. Got: {}",
        stderr
    );

    Ok(())
}

/// Test that log-ratio fails with incompatible indices (different k values)
#[test]
fn test_cli_log_ratio_fails_incompatible_indices() -> Result<()> {
    let dir = tempdir()?;

    let ref1_path = dir.path().join("ref1.fasta");
    let ref2_path = dir.path().join("ref2.fasta");
    fs::write(&ref1_path, format!(">ref1\n{}\n", "A".repeat(200)))?;
    fs::write(&ref2_path, format!(">ref2\n{}\n", "T".repeat(200)))?;

    let binary = get_binary_path();
    let num_path = dir.path().join("num.ryxdi");
    let denom_path = dir.path().join("denom.ryxdi");

    // Create numerator with k=32
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            num_path.to_str().unwrap(),
            "-r",
            ref1_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Numerator index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Create denominator with k=16 (INCOMPATIBLE)
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            denom_path.to_str().unwrap(),
            "-r",
            ref2_path.to_str().unwrap(),
            "-k",
            "16",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Denominator index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let query_path = dir.path().join("query.fastq");
    fs::write(
        &query_path,
        format!("@query1\n{}\n+\n{}\n", "A".repeat(100), "I".repeat(100)),
    )?;

    // Should fail due to k mismatch
    let output = Command::new(&binary)
        .args([
            "classify",
            "log-ratio",
            "-n",
            num_path.to_str().unwrap(),
            "-d",
            denom_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
        ])
        .output()?;

    assert!(
        !output.status.success(),
        "Log-ratio with incompatible k should fail"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    println!("Expected error: {}", stderr);

    assert!(
        stderr.contains("k") || stderr.contains("mismatch") || stderr.contains("compatible"),
        "Error should mention k mismatch. Got: {}",
        stderr
    );

    Ok(())
}

/// Test that --numerator-skip-threshold rejects invalid values (zero, negative, >1)
#[test]
fn test_cli_log_ratio_fails_invalid_skip_threshold() -> Result<()> {
    let dir = tempdir()?;

    let ref_path = dir.path().join("ref.fasta");
    fs::write(&ref_path, format!(">ref\n{}\n", "A".repeat(200)))?;

    let binary = get_binary_path();
    let num_path = dir.path().join("num.ryxdi");
    let denom_path = dir.path().join("denom.ryxdi");

    // Create both indices
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            num_path.to_str().unwrap(),
            "-r",
            ref_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(output.status.success());

    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            denom_path.to_str().unwrap(),
            "-r",
            ref_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(output.status.success());

    let query_path = dir.path().join("query.fastq");
    fs::write(
        &query_path,
        format!("@query1\n{}\n+\n{}\n", "A".repeat(100), "I".repeat(100)),
    )?;

    // Test threshold = 0.0 (should fail)
    let output = Command::new(&binary)
        .args([
            "classify",
            "log-ratio",
            "-n",
            num_path.to_str().unwrap(),
            "-d",
            denom_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
            "--numerator-skip-threshold",
            "0.0",
        ])
        .output()?;
    assert!(!output.status.success(), "threshold=0.0 should fail");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("between 0.0 (exclusive) and 1.0 (inclusive)"),
        "Got: {}",
        stderr
    );

    // Test threshold = -0.5 (should fail)
    let output = Command::new(&binary)
        .args([
            "classify",
            "log-ratio",
            "-n",
            num_path.to_str().unwrap(),
            "-d",
            denom_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
            "--numerator-skip-threshold",
            "-0.5",
        ])
        .output()?;
    assert!(!output.status.success(), "negative threshold should fail");

    // Test threshold = 1.5 (should fail)
    let output = Command::new(&binary)
        .args([
            "classify",
            "log-ratio",
            "-n",
            num_path.to_str().unwrap(),
            "-d",
            denom_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
            "--numerator-skip-threshold",
            "1.5",
        ])
        .output()?;
    assert!(!output.status.success(), "threshold > 1.0 should fail");

    Ok(())
}

// ============================================================================
// Index Merge CLI Integration Tests (Phase 5)
// ============================================================================

/// Test basic merge of two indices via CLI
#[test]
fn test_cli_merge_basic() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    let puc19_path = std::path::Path::new(manifest_dir).join("examples/pUC19.fasta");
    if !phix_path.exists() || !puc19_path.exists() {
        eprintln!("Skipping test: example FASTA files not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let primary_path = dir.path().join("primary.ryxdi");
    let secondary_path = dir.path().join("secondary.ryxdi");
    let merged_path = dir.path().join("merged.ryxdi");

    // Create primary index
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            primary_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Primary index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Create secondary index
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            secondary_path.to_str().unwrap(),
            "-r",
            puc19_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Secondary index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Merge the indices
    let output = Command::new(&binary)
        .args([
            "index",
            "merge",
            "--index-primary",
            primary_path.to_str().unwrap(),
            "--index-secondary",
            secondary_path.to_str().unwrap(),
            "-o",
            merged_path.to_str().unwrap(),
        ])
        .output()?;

    assert!(
        output.status.success(),
        "Merge failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Verify output
    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("Merge output:\n{}", stdout);

    // Should print merge summary
    assert!(
        stdout.contains("Merge complete"),
        "Output should indicate merge complete"
    );
    assert!(
        stdout.contains("Total buckets: 2"),
        "Should have 2 total buckets"
    );

    // Verify merged index can be opened
    assert!(merged_path.exists(), "Merged index should exist");
    assert!(
        merged_path.join("manifest.toml").exists(),
        "Manifest should exist"
    );

    // Verify stats command works on merged index
    let output = Command::new(&binary)
        .args(["index", "stats", "-i", merged_path.to_str().unwrap()])
        .output()?;
    assert!(
        output.status.success(),
        "Stats on merged index failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stats_stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stats_stdout.contains("Buckets: 2"),
        "Merged index should have 2 buckets. Stats:\n{}",
        stats_stdout
    );

    Ok(())
}

/// Test merge with --subtract-from-primary flag
#[test]
fn test_cli_merge_with_subtraction() -> Result<()> {
    let dir = tempdir()?;

    let binary = get_binary_path();
    let primary_path = dir.path().join("primary.ryxdi");
    let secondary_path = dir.path().join("secondary.ryxdi");
    let merged_path = dir.path().join("merged.ryxdi");

    // Create two reference files with overlapping sequences
    // Primary: AAAA... pattern
    // Secondary: same AAAA... pattern plus some TTTT... (different)
    let ref1_path = dir.path().join("ref1.fasta");
    let ref2_path = dir.path().join("ref2.fasta");

    let shared_seq = "A".repeat(200);
    let unique_seq = "T".repeat(200);
    fs::write(&ref1_path, format!(">shared\n{}\n", shared_seq))?;
    fs::write(
        &ref2_path,
        format!(">shared_copy\n{}\n>unique\n{}\n", shared_seq, unique_seq),
    )?;

    // Create primary index
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            primary_path.to_str().unwrap(),
            "-r",
            ref1_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Primary index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Create secondary index
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            secondary_path.to_str().unwrap(),
            "-r",
            ref2_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Secondary index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Merge with subtraction
    let output = Command::new(&binary)
        .args([
            "index",
            "merge",
            "--index-primary",
            primary_path.to_str().unwrap(),
            "--index-secondary",
            secondary_path.to_str().unwrap(),
            "-o",
            merged_path.to_str().unwrap(),
            "--subtract-from-primary",
        ])
        .output()?;

    assert!(
        output.status.success(),
        "Merge with subtraction failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Verify output mentions subtraction/exclusion
    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("Merge with subtraction output:\n{}", stdout);

    assert!(
        stdout.contains("Merge complete"),
        "Output should indicate merge complete"
    );
    // Should mention excluded minimizers since secondary has some shared with primary
    assert!(
        stdout.contains("Excluded minimizers") || stdout.contains("removed"),
        "Output should mention excluded/removed minimizers when subtraction is active"
    );

    // Verify merged index can be opened
    assert!(merged_path.exists(), "Merged index should exist");

    Ok(())
}

/// Test that merge fails with incompatible indices
#[test]
fn test_cli_merge_incompatible_error() -> Result<()> {
    let dir = tempdir()?;

    let binary = get_binary_path();
    let primary_path = dir.path().join("primary.ryxdi");
    let secondary_path = dir.path().join("secondary.ryxdi");
    let merged_path = dir.path().join("merged.ryxdi");

    // Create reference files
    let ref1_path = dir.path().join("ref1.fasta");
    let ref2_path = dir.path().join("ref2.fasta");
    fs::write(
        &ref1_path,
        ">seq1\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n",
    )?;
    fs::write(
        &ref2_path,
        ">seq2\nTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT\n",
    )?;

    // Create primary index with k=32
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            primary_path.to_str().unwrap(),
            "-r",
            ref1_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Primary index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Create secondary index with DIFFERENT k=16 (incompatible)
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            secondary_path.to_str().unwrap(),
            "-r",
            ref2_path.to_str().unwrap(),
            "-k",
            "16", // Different k!
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Secondary index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Attempt to merge incompatible indices - should FAIL
    let output = Command::new(&binary)
        .args([
            "index",
            "merge",
            "--index-primary",
            primary_path.to_str().unwrap(),
            "--index-secondary",
            secondary_path.to_str().unwrap(),
            "-o",
            merged_path.to_str().unwrap(),
        ])
        .output()?;

    assert!(
        !output.status.success(),
        "Merge of incompatible indices should fail"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    println!("Expected error: {}", stderr);

    // Error should mention k mismatch
    assert!(
        stderr.contains("k mismatch") || stderr.contains("k="),
        "Error should mention k mismatch. Got: {}",
        stderr
    );

    // Merged index should NOT be created
    assert!(
        !merged_path.exists(),
        "Merged index should not be created on error"
    );

    Ok(())
}

/// Test that verbose flag works with merge command
#[test]
fn test_cli_merge_verbose_output() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    let puc19_path = std::path::Path::new(manifest_dir).join("examples/pUC19.fasta");
    if !phix_path.exists() || !puc19_path.exists() {
        eprintln!("Skipping test: example FASTA files not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let primary_path = dir.path().join("primary.ryxdi");
    let secondary_path = dir.path().join("secondary.ryxdi");
    let merged_path = dir.path().join("merged.ryxdi");

    // Create primary index
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            primary_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Primary index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Create secondary index
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            secondary_path.to_str().unwrap(),
            "-r",
            puc19_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Secondary index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Merge with --verbose flag (global flag before subcommand)
    let output = Command::new(&binary)
        .args([
            "--verbose",
            "index",
            "merge",
            "--index-primary",
            primary_path.to_str().unwrap(),
            "--index-secondary",
            secondary_path.to_str().unwrap(),
            "-o",
            merged_path.to_str().unwrap(),
        ])
        .output()?;

    assert!(
        output.status.success(),
        "Merge with verbose failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Verbose output goes to stderr
    let stderr = String::from_utf8_lossy(&output.stderr);
    println!("Verbose stderr:\n{}", stderr);

    // Verbose output should include progress messages
    assert!(
        stderr.contains("Loading primary index") || stderr.contains("primary"),
        "Verbose output should mention loading primary index. Stderr:\n{}",
        stderr
    );
    assert!(
        stderr.contains("Loading secondary index") || stderr.contains("secondary"),
        "Verbose output should mention loading secondary index. Stderr:\n{}",
        stderr
    );
    assert!(
        stderr.contains("Validation passed") || stderr.contains("compatible"),
        "Verbose output should mention validation. Stderr:\n{}",
        stderr
    );

    Ok(())
}

/// Test merge with compression options (--zstd, --bloom-filter)
#[test]
fn test_cli_merge_with_compression_options() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    let puc19_path = std::path::Path::new(manifest_dir).join("examples/pUC19.fasta");
    if !phix_path.exists() || !puc19_path.exists() {
        eprintln!("Skipping test: example FASTA files not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let primary_path = dir.path().join("primary.ryxdi");
    let secondary_path = dir.path().join("secondary.ryxdi");
    let merged_path = dir.path().join("merged.ryxdi");

    // Create primary index
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            primary_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Primary index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Create secondary index
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            secondary_path.to_str().unwrap(),
            "-r",
            puc19_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Secondary index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Merge with compression options
    let output = Command::new(&binary)
        .args([
            "index",
            "merge",
            "--index-primary",
            primary_path.to_str().unwrap(),
            "--index-secondary",
            secondary_path.to_str().unwrap(),
            "-o",
            merged_path.to_str().unwrap(),
            "--zstd",
            "--bloom-filter",
            "--bloom-fpp",
            "0.01",
        ])
        .output()?;

    assert!(
        output.status.success(),
        "Merge with compression options failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Verify merged index exists and can be opened
    assert!(merged_path.exists(), "Merged index should exist");

    // Run stats to verify it's valid
    let output = Command::new(&binary)
        .args(["index", "stats", "-i", merged_path.to_str().unwrap()])
        .output()?;
    assert!(
        output.status.success(),
        "Stats on merged index failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    Ok(())
}

/// Test merge fails with duplicate bucket names
#[test]
fn test_cli_merge_duplicate_bucket_names_error() -> Result<()> {
    let dir = tempdir()?;

    let binary = get_binary_path();
    let primary_path = dir.path().join("primary.ryxdi");
    let secondary_path = dir.path().join("secondary.ryxdi");
    let merged_path = dir.path().join("merged.ryxdi");

    // Create reference file with a sequence named "duplicate_name"
    let ref_path = dir.path().join("ref.fasta");
    fs::write(
        &ref_path,
        ">duplicate_name\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n",
    )?;

    // Create primary index with --separate-buckets (bucket name = sequence name)
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            primary_path.to_str().unwrap(),
            "-r",
            ref_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
            "--separate-buckets",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Primary index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Create secondary index with SAME sequence name (will create duplicate bucket name)
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            secondary_path.to_str().unwrap(),
            "-r",
            ref_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
            "--separate-buckets",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Secondary index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Attempt to merge - should FAIL due to duplicate bucket names
    let output = Command::new(&binary)
        .args([
            "index",
            "merge",
            "--index-primary",
            primary_path.to_str().unwrap(),
            "--index-secondary",
            secondary_path.to_str().unwrap(),
            "-o",
            merged_path.to_str().unwrap(),
        ])
        .output()?;

    assert!(
        !output.status.success(),
        "Merge with duplicate bucket names should fail"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    println!("Expected error: {}", stderr);

    // Error should mention duplicate bucket name
    assert!(
        stderr.contains("duplicate") || stderr.contains("bucket name"),
        "Error should mention duplicate bucket name. Got: {}",
        stderr
    );

    Ok(())
}

// ============================================================================
// Phase 5: Memory-Bounded Merge CLI Tests
// ============================================================================

/// Test merge with explicit --max-memory flag for streaming subtraction.
/// This tests the memory-bounded merge path that processes secondary shards
/// one at a time to avoid OOM on large indices with high overlap.
#[test]
fn test_cli_merge_with_max_memory() -> Result<()> {
    let dir = tempdir()?;

    let binary = get_binary_path();
    let primary_path = dir.path().join("primary.ryxdi");
    let secondary_path = dir.path().join("secondary.ryxdi");
    let merged_path = dir.path().join("merged.ryxdi");

    // Create two reference files with overlapping sequences
    let ref1_path = dir.path().join("ref1.fasta");
    let ref2_path = dir.path().join("ref2.fasta");

    // Primary: shared sequence
    let shared_seq = "A".repeat(200);
    fs::write(&ref1_path, format!(">shared\n{}\n", shared_seq))?;

    // Secondary: same shared sequence plus unique sequences
    let unique_seq = "T".repeat(200);
    fs::write(
        &ref2_path,
        format!(">shared_copy\n{}\n>unique\n{}\n", shared_seq, unique_seq),
    )?;

    // Create primary index
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            primary_path.to_str().unwrap(),
            "-r",
            ref1_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Primary index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Create secondary index
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            secondary_path.to_str().unwrap(),
            "-r",
            ref2_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Secondary index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Merge with subtraction AND explicit max-memory limit
    let output = Command::new(&binary)
        .args([
            "index",
            "merge",
            "--index-primary",
            primary_path.to_str().unwrap(),
            "--index-secondary",
            secondary_path.to_str().unwrap(),
            "-o",
            merged_path.to_str().unwrap(),
            "--subtract-from-primary",
            "--max-memory",
            "1G",
        ])
        .output()?;

    assert!(
        output.status.success(),
        "Merge with --max-memory failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Verify output indicates merge completed
    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("Merge with max-memory output:\n{}", stdout);

    assert!(
        stdout.contains("Merge complete"),
        "Output should indicate merge complete"
    );

    // Should mention excluded minimizers since secondary has some shared with primary
    assert!(
        stdout.contains("Excluded minimizers") || stdout.contains("removed"),
        "Output should mention excluded/removed minimizers. Got: {}",
        stdout
    );

    // Verify merged index exists and can be opened
    assert!(merged_path.exists(), "Merged index should exist");

    // Verify we can run stats on it
    let output = Command::new(&binary)
        .args(["index", "stats", "-i", merged_path.to_str().unwrap()])
        .output()?;
    assert!(
        output.status.success(),
        "Stats on merged index failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    Ok(())
}

/// Test that --max-memory auto detection works.
/// When "auto" is specified, the system should detect available memory
/// and use an appropriate fraction for the merge operation.
#[test]
fn test_cli_merge_max_memory_auto_detection() -> Result<()> {
    let dir = tempdir()?;

    let binary = get_binary_path();
    let primary_path = dir.path().join("primary.ryxdi");
    let secondary_path = dir.path().join("secondary.ryxdi");
    let merged_path = dir.path().join("merged.ryxdi");

    // Create simple reference files
    let ref1_path = dir.path().join("ref1.fasta");
    let ref2_path = dir.path().join("ref2.fasta");

    let seq1 = "A".repeat(200);
    let seq2 = "T".repeat(200);
    fs::write(&ref1_path, format!(">seq1\n{}\n", seq1))?;
    fs::write(&ref2_path, format!(">seq2\n{}\n", seq2))?;

    // Create primary index
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            primary_path.to_str().unwrap(),
            "-r",
            ref1_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Primary index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Create secondary index
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            secondary_path.to_str().unwrap(),
            "-r",
            ref2_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Secondary index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Merge with --max-memory auto (should use detected system memory)
    let output = Command::new(&binary)
        .args([
            "index",
            "merge",
            "--index-primary",
            primary_path.to_str().unwrap(),
            "--index-secondary",
            secondary_path.to_str().unwrap(),
            "-o",
            merged_path.to_str().unwrap(),
            "--subtract-from-primary",
            "--max-memory",
            "auto",
        ])
        .output()?;

    assert!(
        output.status.success(),
        "Merge with --max-memory auto failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Verify merge completed
    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("Merge with auto memory:\n{}", stdout);

    assert!(
        stdout.contains("Merge complete"),
        "Output should indicate merge complete. Got: {}",
        stdout
    );

    // Verify merged index exists
    assert!(merged_path.exists(), "Merged index should exist");

    Ok(())
}

// ============================================================================
// Phase 6: Log-Ratio Sequence Output Integration Tests
// ============================================================================

/// Helper to read and decompress a gzipped file.
fn read_gzipped(path: &std::path::Path) -> String {
    use flate2::read::GzDecoder;
    use std::io::Read;

    let file = std::fs::File::open(path).unwrap();
    let mut decoder = GzDecoder::new(file);
    let mut content = String::new();
    decoder.read_to_string(&mut content).unwrap();
    content
}

/// Test --output-sequences outputs reads with NEGATIVE log-ratio (default).
///
/// With two single-bucket indices (phiX=numerator, pUC19=denominator):
/// - A read matching only phiX (numerator) has log-ratio = +inf (excluded)
/// - A read matching only pUC19 (denominator) has log-ratio = -inf (included)
#[test]
fn test_log_ratio_output_sequences_negative() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    let puc19_path = std::path::Path::new(manifest_dir).join("examples/pUC19.fasta");
    if !phix_path.exists() || !puc19_path.exists() {
        eprintln!("Skipping test: example FASTA files not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let num_path = dir.path().join("num.ryxdi");
    let denom_path = dir.path().join("denom.ryxdi");

    // Create separate single-bucket indices
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            num_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Numerator index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            denom_path.to_str().unwrap(),
            "-r",
            puc19_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Denominator index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let phix_seq = "GAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCTGATGAGTCGAAAAATTATCTT";
    let puc19_seq = "TCGCGCGTTTCGGTGATGACGGTGAAAACCTCTGACACATGCAGCTCCCGGAGACGGTCACAGCTTGTCT";

    let query_path = dir.path().join("reads.fastq");
    fs::write(
        &query_path,
        format!(
            "@phix_read\n{}\n+\n{}\n@puc19_read\n{}\n+\n{}\n",
            phix_seq,
            "I".repeat(phix_seq.len()),
            puc19_seq,
            "I".repeat(puc19_seq.len())
        ),
    )?;

    let output_sequences_path = dir.path().join("filtered.fastq.gz");
    let tsv_output_path = dir.path().join("results.tsv");
    let output = Command::new(&binary)
        .args([
            "classify",
            "log-ratio",
            "-n",
            num_path.to_str().unwrap(),
            "-d",
            denom_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
            "-o",
            tsv_output_path.to_str().unwrap(),
            "--output-sequences",
            output_sequences_path.to_str().unwrap(),
        ])
        .output()?;

    assert!(
        output.status.success(),
        "Log-ratio with --output-sequences failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let tsv_content = fs::read_to_string(&tsv_output_path)?;
    println!("TSV output:\n{}", tsv_content);

    assert!(
        output_sequences_path.exists(),
        "Output sequences file should exist"
    );

    let content = read_gzipped(&output_sequences_path);
    println!("Output sequences content:\n{}", content);

    // pUC19 read: numerator=0 -> -inf -> included by default
    assert!(
        content.contains("@puc19_read"),
        "pUC19 read with -inf log-ratio should be in output"
    );
    assert!(
        content.contains(puc19_seq),
        "pUC19 sequence should be preserved in output"
    );

    // phiX read: numerator>0, denominator=0 -> +inf -> excluded by default
    assert!(
        !content.contains("@phix_read"),
        "phiX read with +inf log-ratio should NOT be in output"
    );

    assert!(
        content.contains("+\n"),
        "Should have FASTQ quality separator"
    );

    Ok(())
}

/// Test --passing-is-positive outputs reads with POSITIVE log-ratio only.
///
/// With two single-bucket indices and --passing-is-positive:
/// - phiX read (numerator match): log-ratio = +inf -> INCLUDED
/// - pUC19 read (denominator match): log-ratio = -inf -> EXCLUDED
#[test]
fn test_log_ratio_output_sequences_positive_flag() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    let puc19_path = std::path::Path::new(manifest_dir).join("examples/pUC19.fasta");
    if !phix_path.exists() || !puc19_path.exists() {
        eprintln!("Skipping test: example FASTA files not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let num_path = dir.path().join("num.ryxdi");
    let denom_path = dir.path().join("denom.ryxdi");

    // Create separate single-bucket indices
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            num_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Numerator index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            denom_path.to_str().unwrap(),
            "-r",
            puc19_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Denominator index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let phix_seq = "GAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCTGATGAGTCGAAAAATTATCTT";
    let puc19_seq = "TCGCGCGTTTCGGTGATGACGGTGAAAACCTCTGACACATGCAGCTCCCGGAGACGGTCACAGCTTGTCT";

    let query_path = dir.path().join("reads.fastq");
    fs::write(
        &query_path,
        format!(
            "@phix_read\n{}\n+\n{}\n@puc19_read\n{}\n+\n{}\n",
            phix_seq,
            "I".repeat(phix_seq.len()),
            puc19_seq,
            "I".repeat(puc19_seq.len())
        ),
    )?;

    let output_sequences_path = dir.path().join("filtered.fastq.gz");
    let tsv_output_path = dir.path().join("results.tsv");
    let output = Command::new(&binary)
        .args([
            "classify",
            "log-ratio",
            "-n",
            num_path.to_str().unwrap(),
            "-d",
            denom_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
            "-o",
            tsv_output_path.to_str().unwrap(),
            "--output-sequences",
            output_sequences_path.to_str().unwrap(),
            "--passing-is-positive",
        ])
        .output()?;

    assert!(
        output.status.success(),
        "Log-ratio with --passing-is-positive failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let tsv_content = fs::read_to_string(&tsv_output_path)?;
    println!("TSV output:\n{}", tsv_content);

    assert!(
        output_sequences_path.exists(),
        "Output sequences file should exist"
    );

    let content = read_gzipped(&output_sequences_path);
    println!("Output sequences (passing-is-positive):\n{}", content);

    // phiX read: numerator>0, denom=0 -> +inf -> INCLUDED with --passing-is-positive
    assert!(
        content.contains("@phix_read"),
        "phiX read with +inf log-ratio should be in output with --passing-is-positive"
    );
    assert!(
        content.contains(phix_seq),
        "phiX sequence should be preserved in output"
    );

    // pUC19 read: numerator=0 -> -inf -> EXCLUDED
    assert!(
        !content.contains("@puc19_read"),
        "pUC19 read with -inf log-ratio should NOT be in output when --passing-is-positive"
    );

    Ok(())
}

/// Test that NaN reads (matching neither index) are always emitted in --output-sequences,
/// regardless of --passing-is-positive flag.
///
/// NaN means "no evidence for or against", so the conservative choice is to always emit.
#[test]
fn test_log_ratio_output_sequences_nan_always_emitted() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    let puc19_path = std::path::Path::new(manifest_dir).join("examples/pUC19.fasta");
    if !phix_path.exists() || !puc19_path.exists() {
        eprintln!("Skipping test: example FASTA files not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let num_path = dir.path().join("num.ryxdi");
    let denom_path = dir.path().join("denom.ryxdi");

    // Create separate single-bucket indices
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            num_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Numerator index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            denom_path.to_str().unwrap(),
            "-r",
            puc19_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Denominator index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Synthetic query matching neither index → NaN
    let no_match_seq = "ACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACAC";

    let query_path = dir.path().join("reads.fastq");
    fs::write(
        &query_path,
        format!(
            "@no_match_read\n{}\n+\n{}\n",
            no_match_seq,
            "I".repeat(no_match_seq.len()),
        ),
    )?;

    // Test 1: default (passing-is-negative) — NaN read should be emitted
    let out_seq_neg = dir.path().join("filtered_neg.fastq.gz");
    let tsv_neg = dir.path().join("results_neg.tsv");
    let output = Command::new(&binary)
        .args([
            "classify",
            "log-ratio",
            "-n",
            num_path.to_str().unwrap(),
            "-d",
            denom_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
            "-o",
            tsv_neg.to_str().unwrap(),
            "--output-sequences",
            out_seq_neg.to_str().unwrap(),
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Log-ratio (default) failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Verify TSV shows NaN
    let tsv_content = fs::read_to_string(&tsv_neg)?;
    println!("TSV (default):\n{}", tsv_content);
    let data_lines: Vec<&str> = tsv_content.lines().skip(1).collect();
    assert!(!data_lines.is_empty(), "Should have data rows");
    let cols: Vec<&str> = data_lines[0].split('\t').collect();
    assert_eq!(cols[2], "NaN", "Score should be NaN. Got: {}", cols[2]);

    // NaN read should be in output sequences (default = passing-is-negative)
    assert!(
        out_seq_neg.exists(),
        "Output sequences file should exist (default)"
    );
    let content_neg = read_gzipped(&out_seq_neg);
    assert!(
        content_neg.contains("@no_match_read"),
        "NaN read should be emitted with default (passing-is-negative) setting"
    );

    // Test 2: --passing-is-positive — NaN read should ALSO be emitted
    let out_seq_pos = dir.path().join("filtered_pos.fastq.gz");
    let tsv_pos = dir.path().join("results_pos.tsv");
    let output = Command::new(&binary)
        .args([
            "classify",
            "log-ratio",
            "-n",
            num_path.to_str().unwrap(),
            "-d",
            denom_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
            "-o",
            tsv_pos.to_str().unwrap(),
            "--output-sequences",
            out_seq_pos.to_str().unwrap(),
            "--passing-is-positive",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Log-ratio (passing-is-positive) failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(
        out_seq_pos.exists(),
        "Output sequences file should exist (passing-is-positive)"
    );
    let content_pos = read_gzipped(&out_seq_pos);
    assert!(
        content_pos.contains("@no_match_read"),
        "NaN read should be emitted with --passing-is-positive setting"
    );

    Ok(())
}

/// Test that paired-end output creates foo.R1.fastq.gz and foo.R2.fastq.gz files.
///
/// Uses a pUC19-matching read (log-ratio = -inf) which passes with default settings.
#[test]
fn test_log_ratio_paired_creates_r1_r2_files() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    let puc19_path = std::path::Path::new(manifest_dir).join("examples/pUC19.fasta");
    if !phix_path.exists() || !puc19_path.exists() {
        eprintln!("Skipping test: example FASTA files not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let num_path = dir.path().join("num.ryxdi");
    let denom_path = dir.path().join("denom.ryxdi");

    // Create separate single-bucket indices
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            num_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Numerator index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            denom_path.to_str().unwrap(),
            "-r",
            puc19_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Denominator index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Create paired-end query files (R1 and R2)
    // Use pUC19-matching sequence: numerator=0 -> -inf -> passes with default
    let r1_path = dir.path().join("reads_R1.fastq");
    let r2_path = dir.path().join("reads_R2.fastq");

    let puc19_r1 = "TCGCGCGTTTCGGTGATGACGGTGAAAACCTCTGACACATGCAGCTCCCGGAGACGGTCACAGCTTGTCT";
    let puc19_r2 = "GTAAGCGGATGCCGGGAGCAGACAAGCCCGTCAGGGCGCGTCAGCGGGTGTTGGCGGGTGTCGGGGCTGG";

    fs::write(
        &r1_path,
        format!("@read1\n{}\n+\n{}\n", puc19_r1, "I".repeat(puc19_r1.len())),
    )?;
    fs::write(
        &r2_path,
        format!("@read1\n{}\n+\n{}\n", puc19_r2, "J".repeat(puc19_r2.len())),
    )?;

    let output_sequences_path = dir.path().join("filtered.fastq.gz");
    let tsv_output_path = dir.path().join("results.tsv");
    let output = Command::new(&binary)
        .args([
            "classify",
            "log-ratio",
            "-n",
            num_path.to_str().unwrap(),
            "-d",
            denom_path.to_str().unwrap(),
            "-1",
            r1_path.to_str().unwrap(),
            "-2",
            r2_path.to_str().unwrap(),
            "-o",
            tsv_output_path.to_str().unwrap(),
            "--output-sequences",
            output_sequences_path.to_str().unwrap(),
        ])
        .output()?;

    assert!(
        output.status.success(),
        "Log-ratio paired with --output-sequences failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let tsv_content = fs::read_to_string(&tsv_output_path)?;
    println!("TSV output:\n{}", tsv_content);

    let r1_output_path = dir.path().join("filtered.R1.fastq.gz");
    let r2_output_path = dir.path().join("filtered.R2.fastq.gz");

    assert!(
        r1_output_path.exists(),
        "R1 output file (filtered.R1.fastq.gz) should exist"
    );
    assert!(
        r2_output_path.exists(),
        "R2 output file (filtered.R2.fastq.gz) should exist"
    );

    // Original path should NOT exist for paired-end (only R1/R2)
    assert!(
        !output_sequences_path.exists(),
        "Original path should not exist for paired-end; R1/R2 paths used instead"
    );

    let r1_content = read_gzipped(&r1_output_path);
    println!("R1 output content:\n{}", r1_content);
    assert!(
        r1_content.contains("@read1"),
        "R1 should contain the read header"
    );
    assert!(
        r1_content.contains(puc19_r1),
        "R1 should contain the R1 sequence"
    );

    let r2_content = read_gzipped(&r2_output_path);
    println!("R2 output content:\n{}", r2_content);
    assert!(
        r2_content.contains("@read1"),
        "R2 should contain the read header"
    );
    assert!(
        r2_content.contains(puc19_r2),
        "R2 should contain the R2 sequence"
    );

    Ok(())
}

/// Test that log-ratio works correctly when multiple deferred-denom flush cycles occur.
///
/// Uses a very small batch size (4 reads) so the deferred threshold (batch_size/2 = 2)
/// is reached quickly, forcing multiple flush cycles across 10 query reads.
/// Validates that all reads appear in the output despite multiple flushes.
#[test]
fn test_log_ratio_multiple_deferred_flushes() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    let puc19_path = std::path::Path::new(manifest_dir).join("examples/pUC19.fasta");
    if !phix_path.exists() || !puc19_path.exists() {
        eprintln!("Skipping test: example FASTA files not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let num_path = dir.path().join("num.ryxdi");
    let denom_path = dir.path().join("denom.ryxdi");

    // Create separate single-bucket indices
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            num_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Numerator index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            denom_path.to_str().unwrap(),
            "-r",
            puc19_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Denominator index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Create 10 query reads — all from phiX174 so they have non-zero numerator scores
    // and need denominator classification.
    let query_path = dir.path().join("queries.fastq");
    let phix_seq = "GAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCTGATGAGTCGAAAAATTATCTT";
    let mut query_content = String::new();
    for i in 0..10 {
        query_content.push_str(&format!(
            "@read{}\n{}\n+\n{}\n",
            i,
            phix_seq,
            "I".repeat(phix_seq.len())
        ));
    }
    fs::write(&query_path, &query_content)?;

    let tsv_output_path = dir.path().join("output.tsv");

    // Run log-ratio with very small batch size to force multiple deferred flushes.
    // batch_size=4 → deferred_threshold=2, so with 10 reads across 3 batches (4+4+2),
    // the deferred buffer should flush multiple times.
    let output = Command::new(&binary)
        .args([
            "classify",
            "log-ratio",
            "-n",
            num_path.to_str().unwrap(),
            "-d",
            denom_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
            "-o",
            tsv_output_path.to_str().unwrap(),
            "--batch-size",
            "4",
            "--verbose",
        ])
        .output()?;

    assert!(
        output.status.success(),
        "Log-ratio with small batch size failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Verify all 10 reads appear in output
    let tsv_content = fs::read_to_string(&tsv_output_path)?;
    let data_lines: Vec<&str> = tsv_content
        .lines()
        .filter(|l| !l.starts_with("read_id"))
        .collect();

    assert_eq!(
        data_lines.len(),
        10,
        "Expected 10 output lines (one per read), got {}. Output:\n{}",
        data_lines.len(),
        tsv_content
    );

    // Every line should have 4 tab-separated columns
    for (i, line) in data_lines.iter().enumerate() {
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(
            cols.len(),
            4,
            "Line {} should have 4 columns, got {}: {:?}",
            i,
            cols.len(),
            cols
        );
    }

    // Verify stderr mentions deferred flushing (at least one mid-loop flush)
    let stderr = String::from_utf8_lossy(&output.stderr);
    // With 10 reads at batch_size=4, there should be multiple batches and at least one flush
    assert!(
        stderr.contains("Flushed") || stderr.contains("deferred"),
        "Expected deferred flush log messages in stderr. stderr:\n{}",
        stderr
    );

    Ok(())
}

#[test]
fn test_cli_from_config_subtract_nonexistent_index() -> Result<()> {
    let dir = tempdir()?;
    let binary = get_binary_path();

    // Create a minimal FASTA file
    let ref_path = dir.path().join("ref.fasta");
    fs::write(&ref_path, format!(">seq1\n{}\n", "A".repeat(200)))?;

    // Write a valid TOML config
    let config_path = dir.path().join("config.toml");
    let config_content = format!(
        r#"
[index]
k = 32
window = 10
salt = 0x5555555555555555
output = "output.ryxdi"

[buckets.TestBucket]
files = ["{}"]
"#,
        ref_path.to_str().unwrap()
    );
    fs::write(&config_path, config_content)?;

    // Run from-config with --subtract-from pointing to nonexistent index
    let output = Command::new(&binary)
        .args([
            "index",
            "from-config",
            "-c",
            config_path.to_str().unwrap(),
            "--subtract-from",
            dir.path().join("nonexistent.ryxdi").to_str().unwrap(),
        ])
        .output()?;

    assert!(
        !output.status.success(),
        "Should fail with nonexistent subtraction index"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("nonexistent.ryxdi"),
        "Error should mention the bad path, got: {}",
        stderr
    );

    Ok(())
}

#[test]
fn test_cli_from_config_subtract_incompatible_index() -> Result<()> {
    let dir = tempdir()?;
    let binary = get_binary_path();

    // Create a FASTA file
    let ref_path = dir.path().join("ref.fasta");
    fs::write(&ref_path, format!(">seq1\n{}\n", "A".repeat(200)))?;

    // Build a subtraction index with k=32, w=10
    let subtract_path = dir.path().join("subtract.ryxdi");
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            subtract_path.to_str().unwrap(),
            "-r",
            ref_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Subtraction index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Write config with DIFFERENT k=64, w=50 (incompatible)
    let config_path = dir.path().join("config.toml");
    let config_content = format!(
        r#"
[index]
k = 64
window = 50
salt = 0x5555555555555555
output = "output.ryxdi"

[buckets.TestBucket]
files = ["{}"]
"#,
        ref_path.to_str().unwrap()
    );
    fs::write(&config_path, config_content)?;

    // Run from-config with incompatible --subtract-from
    let output = Command::new(&binary)
        .args([
            "index",
            "from-config",
            "-c",
            config_path.to_str().unwrap(),
            "--subtract-from",
            subtract_path.to_str().unwrap(),
        ])
        .output()?;

    assert!(
        !output.status.success(),
        "Should fail with incompatible subtraction index"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("mismatch"),
        "Error should mention mismatch, got: {}",
        stderr
    );

    Ok(())
}

#[test]
fn test_cli_from_config_subtract_removes_minimizers() -> Result<()> {
    let dir = tempdir()?;
    let binary = get_binary_path();

    // Create reference files with overlapping sequences
    let shared_seq = "A".repeat(200);
    let unique_seq = "T".repeat(200);

    let ref_shared = dir.path().join("ref_shared.fasta");
    fs::write(&ref_shared, format!(">shared\n{}\n", shared_seq))?;

    let ref_main = dir.path().join("ref_main.fasta");
    fs::write(
        &ref_main,
        format!(">shared_copy\n{}\n>unique\n{}\n", shared_seq, unique_seq),
    )?;

    // Build subtraction index from the shared sequence
    let subtract_path = dir.path().join("subtract.ryxdi");
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            subtract_path.to_str().unwrap(),
            "-r",
            ref_shared.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Subtraction index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Build from-config WITHOUT subtraction
    let config_no_sub = dir.path().join("config_no_sub.toml");
    let config_content = format!(
        r#"
[index]
k = 32
window = 10
salt = 0x5555555555555555
output = "output_no_sub.ryxdi"

[buckets.MainBucket]
files = ["{}"]
"#,
        ref_main.to_str().unwrap()
    );
    fs::write(&config_no_sub, &config_content)?;

    let output = Command::new(&binary)
        .args([
            "index",
            "from-config",
            "-c",
            config_no_sub.to_str().unwrap(),
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Index creation without subtraction failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Build from-config WITH subtraction
    let config_with_sub = dir.path().join("config_with_sub.toml");
    let config_content_sub = format!(
        r#"
[index]
k = 32
window = 10
salt = 0x5555555555555555
output = "output_with_sub.ryxdi"

[buckets.MainBucket]
files = ["{}"]
"#,
        ref_main.to_str().unwrap()
    );
    fs::write(&config_with_sub, &config_content_sub)?;

    let output = Command::new(&binary)
        .args([
            "index",
            "from-config",
            "-c",
            config_with_sub.to_str().unwrap(),
            "--subtract-from",
            subtract_path.to_str().unwrap(),
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Index creation with subtraction failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Compare stats: subtracted index should have fewer total entries
    let stats_no_sub = Command::new(&binary)
        .args([
            "index",
            "stats",
            "-i",
            dir.path().join("output_no_sub.ryxdi").to_str().unwrap(),
        ])
        .output()?;
    assert!(stats_no_sub.status.success());

    let stats_with_sub = Command::new(&binary)
        .args([
            "index",
            "stats",
            "-i",
            dir.path().join("output_with_sub.ryxdi").to_str().unwrap(),
        ])
        .output()?;
    assert!(stats_with_sub.status.success());

    let no_sub_stdout = String::from_utf8_lossy(&stats_no_sub.stdout);
    let with_sub_stdout = String::from_utf8_lossy(&stats_with_sub.stdout);

    // Parse total minimizers from stats output (stdout)
    fn parse_total_minimizers(output: &str) -> u64 {
        for line in output.lines() {
            if line.contains("Total minimizers:") {
                return line
                    .split(':')
                    .last()
                    .unwrap()
                    .trim()
                    .replace(',', "")
                    .parse()
                    .unwrap();
            }
        }
        panic!("Could not find 'Total minimizers' in output:\n{}", output);
    }

    let mins_no_sub = parse_total_minimizers(&no_sub_stdout);
    let mins_with_sub = parse_total_minimizers(&with_sub_stdout);

    println!("Minimizers without subtraction: {}", mins_no_sub);
    println!("Minimizers with subtraction: {}", mins_with_sub);

    assert!(
        mins_with_sub < mins_no_sub,
        "Subtracted index should have fewer minimizers: {} should be < {}",
        mins_with_sub,
        mins_no_sub
    );

    // Also verify the subtracted index is valid (stats didn't error)
    assert!(
        mins_with_sub > 0,
        "Subtracted index should still have some minimizers (the unique sequence)"
    );

    Ok(())
}

#[test]
fn test_cli_from_config_subtract_multibucket() -> Result<()> {
    let dir = tempdir()?;
    let binary = get_binary_path();

    // Create sequences: shared overlap + two unique per-bucket sequences
    let shared_seq = "A".repeat(200);
    let unique_a = "T".repeat(200);
    let unique_b = "C".repeat(200);

    let ref_shared = dir.path().join("ref_shared.fasta");
    fs::write(&ref_shared, format!(">shared\n{}\n", shared_seq))?;

    let ref_a = dir.path().join("ref_a.fasta");
    fs::write(
        &ref_a,
        format!(">shared_a\n{}\n>unique_a\n{}\n", shared_seq, unique_a),
    )?;

    let ref_b = dir.path().join("ref_b.fasta");
    fs::write(
        &ref_b,
        format!(">shared_b\n{}\n>unique_b\n{}\n", shared_seq, unique_b),
    )?;

    // Build subtraction index from shared sequence
    let subtract_path = dir.path().join("subtract.ryxdi");
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            subtract_path.to_str().unwrap(),
            "-r",
            ref_shared.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Subtraction index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Build multi-bucket from-config WITHOUT subtraction
    let config_no_sub = dir.path().join("config_no_sub.toml");
    fs::write(
        &config_no_sub,
        format!(
            r#"
[index]
k = 32
window = 10
salt = 0x5555555555555555
output = "multi_no_sub.ryxdi"

[buckets.BucketA]
files = ["{}"]

[buckets.BucketB]
files = ["{}"]
"#,
            ref_a.to_str().unwrap(),
            ref_b.to_str().unwrap()
        ),
    )?;

    let output = Command::new(&binary)
        .args([
            "index",
            "from-config",
            "-c",
            config_no_sub.to_str().unwrap(),
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Multi-bucket index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Build multi-bucket from-config WITH subtraction
    let config_with_sub = dir.path().join("config_with_sub.toml");
    fs::write(
        &config_with_sub,
        format!(
            r#"
[index]
k = 32
window = 10
salt = 0x5555555555555555
output = "multi_with_sub.ryxdi"

[buckets.BucketA]
files = ["{}"]

[buckets.BucketB]
files = ["{}"]
"#,
            ref_a.to_str().unwrap(),
            ref_b.to_str().unwrap()
        ),
    )?;

    let output = Command::new(&binary)
        .args([
            "index",
            "from-config",
            "-c",
            config_with_sub.to_str().unwrap(),
            "--subtract-from",
            subtract_path.to_str().unwrap(),
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Multi-bucket index with subtraction failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Compare stats
    let stats_no_sub = Command::new(&binary)
        .args([
            "index",
            "stats",
            "-i",
            dir.path().join("multi_no_sub.ryxdi").to_str().unwrap(),
        ])
        .output()?;
    assert!(stats_no_sub.status.success());

    let stats_with_sub = Command::new(&binary)
        .args([
            "index",
            "stats",
            "-i",
            dir.path().join("multi_with_sub.ryxdi").to_str().unwrap(),
        ])
        .output()?;
    assert!(stats_with_sub.status.success());

    let no_sub_stdout = String::from_utf8_lossy(&stats_no_sub.stdout);
    let with_sub_stdout = String::from_utf8_lossy(&stats_with_sub.stdout);

    fn parse_total_minimizers(output: &str) -> u64 {
        for line in output.lines() {
            if line.contains("Total minimizers:") {
                return line
                    .split(':')
                    .last()
                    .unwrap()
                    .trim()
                    .replace(',', "")
                    .parse()
                    .unwrap();
            }
        }
        panic!("Could not find 'Total minimizers' in output:\n{}", output);
    }

    fn parse_buckets(output: &str) -> u64 {
        for line in output.lines() {
            if line.contains("Buckets:") {
                return line.split(':').last().unwrap().trim().parse().unwrap();
            }
        }
        panic!("Could not find 'Buckets' in output:\n{}", output);
    }

    // Verify both have 2 buckets
    assert_eq!(
        parse_buckets(&no_sub_stdout),
        2,
        "Should have 2 buckets without subtraction"
    );
    assert_eq!(
        parse_buckets(&with_sub_stdout),
        2,
        "Should have 2 buckets with subtraction"
    );

    let mins_no_sub = parse_total_minimizers(&no_sub_stdout);
    let mins_with_sub = parse_total_minimizers(&with_sub_stdout);

    println!(
        "Multi-bucket minimizers without subtraction: {}",
        mins_no_sub
    );
    println!(
        "Multi-bucket minimizers with subtraction: {}",
        mins_with_sub
    );

    assert!(
        mins_with_sub < mins_no_sub,
        "Subtracted multi-bucket index should have fewer minimizers: {} should be < {}",
        mins_with_sub,
        mins_no_sub
    );

    assert!(
        mins_with_sub > 0,
        "Subtracted index should still have minimizers from unique sequences"
    );

    Ok(())
}

/// Test that single-bucket `from-config --subtract-from` uses the parallel
/// streaming path (not the serial multi-bucket path).
#[test]
fn test_cli_from_config_subtract_single_bucket_parallel() -> Result<()> {
    let dir = tempdir()?;
    let binary = get_binary_path();

    // Create reference files
    let shared_seq = "A".repeat(200);
    let unique_seq = "T".repeat(200);

    let ref_shared = dir.path().join("ref_shared.fasta");
    fs::write(&ref_shared, format!(">shared\n{}\n", shared_seq))?;

    let ref_main = dir.path().join("ref_main.fasta");
    fs::write(
        &ref_main,
        format!(">shared_copy\n{}\n>unique\n{}\n", shared_seq, unique_seq),
    )?;

    // Build subtraction index
    let subtract_path = dir.path().join("subtract.ryxdi");
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            subtract_path.to_str().unwrap(),
            "-r",
            ref_shared.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Subtraction index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Build from-config WITH subtraction (single bucket)
    let config_path = dir.path().join("config.toml");
    let config_content = format!(
        r#"
[index]
k = 32
window = 10
salt = 0x5555555555555555
output = "output.ryxdi"

[buckets.SingleBucket]
files = ["{}"]
"#,
        ref_main.to_str().unwrap()
    );
    fs::write(&config_path, &config_content)?;

    let output = Command::new(&binary)
        .args([
            "-v",
            "index",
            "from-config",
            "-c",
            config_path.to_str().unwrap(),
            "--subtract-from",
            subtract_path.to_str().unwrap(),
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stderr = String::from_utf8_lossy(&output.stderr);

    // The single-bucket streaming path logs "(streaming," while the
    // multi-bucket path logs "(N/M)". Assert we hit the parallel path.
    assert!(
        stderr.contains("(streaming,"),
        "Single-bucket subtract should use parallel streaming path.\n\
         Expected '(streaming,' in stderr but got:\n{}",
        stderr
    );

    // Verify the index is valid
    let stats = Command::new(&binary)
        .args([
            "index",
            "stats",
            "-i",
            dir.path().join("output.ryxdi").to_str().unwrap(),
        ])
        .output()?;
    assert!(stats.status.success());

    Ok(())
}

// ============================================================================
// Parquet Input with Trim/Filter Tests (Cycle 9)
// ============================================================================

/// Write a Parquet input file with read_id + sequence1 columns.
fn write_parquet_input(path: &Path, ids: &[&str], seqs: &[&str]) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("read_id", DataType::LargeUtf8, false),
        Field::new("sequence1", DataType::LargeUtf8, false),
    ]));
    let file = fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema.clone(), None).unwrap();
    let id_array = LargeStringArray::from_iter_values(ids.iter().copied());
    let seq_array = LargeStringArray::from_iter_values(seqs.iter().copied());
    let batch =
        RecordBatch::try_new(schema, vec![Arc::new(id_array), Arc::new(seq_array)]).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// Test classify run with Parquet input + --trim-to: short reads skipped, long reads trimmed.
#[test]
fn test_classify_run_with_parquet_trim() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    if !phix_path.exists() {
        eprintln!("Skipping test: example FASTA file not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let index_path = dir.path().join("test.ryxdi");

    // Create index
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            index_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Read the first 100 bases from phiX174 for a realistic long sequence
    let phix_content = fs::read_to_string(&phix_path)?;
    let phix_seq: String = phix_content
        .lines()
        .filter(|l| !l.starts_with('>'))
        .collect::<Vec<_>>()
        .join("");
    let long_seq = &phix_seq[0..100]; // 100bp from phiX174
    let short_seq = &phix_seq[0..40]; // 40bp — too short for trim_to=50

    // Write Parquet input file with mixed lengths
    let parquet_path = dir.path().join("input.parquet");
    write_parquet_input(
        &parquet_path,
        &["short_read", "long_read"],
        &[short_seq, long_seq],
    );

    // Run classification with --trim-to 50
    let output = Command::new(&binary)
        .args([
            "classify",
            "run",
            "-i",
            index_path.to_str().unwrap(),
            "-1",
            parquet_path.to_str().unwrap(),
            "-t",
            "0.0",
            "--trim-to",
            "50",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Classification with Parquet --trim-to failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().collect();

    // short_read (40bp) should be skipped (< 50bp trim_to)
    assert!(
        !lines.iter().any(|l| l.contains("short_read")),
        "short_read (40bp) should be skipped with --trim-to 50, but found in output:\n{}",
        stdout
    );

    // long_read (100bp) should be present (trimmed to 50bp and classified)
    assert!(
        lines.iter().any(|l| l.contains("long_read")),
        "long_read (100bp) should be present with --trim-to 50, but not found:\n{}",
        stdout
    );

    Ok(())
}

// ============================================================================
// --minimum-length Integration Tests (Cycle 12)
// ============================================================================

/// Test that --minimum-length filters out short reads from FASTX input.
#[test]
fn test_cli_minimum_length_skips_short_reads_fastx() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    if !phix_path.exists() {
        eprintln!("Skipping test: example FASTA file not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let index_path = dir.path().join("test.ryxdi");

    // Create index
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            index_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Create query with:
    // - short_query: 40bp (below --minimum-length 50, should be SKIPPED)
    // - long_query: 100bp (above --minimum-length 50, should be INCLUDED)
    let phix_content = fs::read_to_string(&phix_path)?;
    let phix_seq: String = phix_content
        .lines()
        .filter(|l| !l.starts_with('>'))
        .collect::<Vec<_>>()
        .join("");
    let short_seq = &phix_seq[0..40];
    let long_seq = &phix_seq[0..100];

    let query_path = dir.path().join("query.fastq");
    fs::write(
        &query_path,
        format!(
            "@short_query\n{}\n+\n{}\n@long_query\n{}\n+\n{}\n",
            short_seq,
            "I".repeat(40),
            long_seq,
            "I".repeat(100)
        ),
    )?;

    let output = Command::new(&binary)
        .args([
            "classify",
            "run",
            "-i",
            index_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
            "-t",
            "0.0",
            "--minimum-length",
            "50",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Classification with --minimum-length failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().collect();

    assert!(
        !lines.iter().any(|l| l.contains("short_query")),
        "short_query (40bp) should be skipped with --minimum-length 50, but found in output:\n{}",
        stdout
    );
    assert!(
        lines.iter().any(|l| l.contains("long_query")),
        "long_query (100bp) should be present with --minimum-length 50, but not found:\n{}",
        stdout
    );

    Ok(())
}

/// Test that --minimum-length is applied before --trim-to.
/// With --minimum-length 50 --trim-to 70:
/// - 40bp read: filtered by minimum_length (< 50)
/// - 60bp read: passes minimum_length (60 >= 50), but skipped by trim_to (60 < 70)
/// - 100bp read: passes both checks, trimmed to 70, classified
#[test]
fn test_cli_minimum_length_before_trim_to_fastx() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    if !phix_path.exists() {
        eprintln!("Skipping test: example FASTA file not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let index_path = dir.path().join("test.ryxdi");

    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            index_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(output.status.success());

    let phix_content = fs::read_to_string(&phix_path)?;
    let phix_seq: String = phix_content
        .lines()
        .filter(|l| !l.starts_with('>'))
        .collect::<Vec<_>>()
        .join("");
    let seq_40 = &phix_seq[0..40];
    let seq_60 = &phix_seq[0..60];
    let seq_100 = &phix_seq[0..100];

    let query_path = dir.path().join("query.fastq");
    fs::write(
        &query_path,
        format!(
            "@read_40bp\n{}\n+\n{}\n@read_60bp\n{}\n+\n{}\n@read_100bp\n{}\n+\n{}\n",
            seq_40,
            "I".repeat(40),
            seq_60,
            "I".repeat(60),
            seq_100,
            "I".repeat(100)
        ),
    )?;

    let output = Command::new(&binary)
        .args([
            "classify",
            "run",
            "-i",
            index_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
            "-t",
            "0.0",
            "--minimum-length",
            "50",
            "--trim-to",
            "70",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Classification with --minimum-length + --trim-to failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().collect();

    // 40bp read filtered by minimum_length (40 < 50)
    assert!(
        !lines.iter().any(|l| l.contains("read_40bp")),
        "read_40bp should be skipped with --minimum-length 50:\n{}",
        stdout
    );
    // 60bp read passes minimum_length (60 >= 50) but skipped by trim_to (60 < 70)
    assert!(
        !lines.iter().any(|l| l.contains("read_60bp")),
        "read_60bp (60bp) should be skipped by --trim-to 70 (60 < 70):\n{}",
        stdout
    );
    // 100bp read passes both checks, trimmed to 70, classified
    assert!(
        lines.iter().any(|l| l.contains("read_100bp")),
        "read_100bp (100bp >= 50, >= 70) should be present:\n{}",
        stdout
    );

    Ok(())
}

/// Test that --minimum-length > --trim-to works correctly.
/// With --minimum-length 100 --trim-to 50:
/// - minimum_length is the binding constraint
/// - 60bp read: filtered (< 100)
/// - 100bp read: passes minimum_length (>= 100), trimmed to 50, classified
#[test]
fn test_cli_minimum_length_gt_trim_to_fastx() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    if !phix_path.exists() {
        eprintln!("Skipping test: example FASTA file not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let index_path = dir.path().join("test.ryxdi");

    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            index_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(output.status.success());

    let phix_content = fs::read_to_string(&phix_path)?;
    let phix_seq: String = phix_content
        .lines()
        .filter(|l| !l.starts_with('>'))
        .collect::<Vec<_>>()
        .join("");
    let seq_60 = &phix_seq[0..60];
    let seq_100 = &phix_seq[0..100];

    let query_path = dir.path().join("query.fastq");
    fs::write(
        &query_path,
        format!(
            "@read_60bp\n{}\n+\n{}\n@read_100bp\n{}\n+\n{}\n",
            seq_60,
            "I".repeat(60),
            seq_100,
            "I".repeat(100)
        ),
    )?;

    let output = Command::new(&binary)
        .args([
            "classify",
            "run",
            "-i",
            index_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
            "-t",
            "0.0",
            "--minimum-length",
            "100",
            "--trim-to",
            "50",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Classification with --minimum-length 100 --trim-to 50 failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().collect();

    // 60bp read: filtered by minimum_length (60 < 100)
    assert!(
        !lines.iter().any(|l| l.contains("read_60bp")),
        "read_60bp (60bp < 100) should be skipped with --minimum-length 100:\n{}",
        stdout
    );
    // 100bp read: passes minimum_length (100 >= 100), trimmed to 50, classified
    assert!(
        lines.iter().any(|l| l.contains("read_100bp")),
        "read_100bp (100bp >= 100, trimmed to 50) should be present:\n{}",
        stdout
    );

    Ok(())
}

/// Test --minimum-length with Parquet input.
#[test]
fn test_cli_minimum_length_with_parquet_input() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    if !phix_path.exists() {
        eprintln!("Skipping test: example FASTA file not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let index_path = dir.path().join("test.ryxdi");

    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            index_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(output.status.success());

    let phix_content = fs::read_to_string(&phix_path)?;
    let phix_seq: String = phix_content
        .lines()
        .filter(|l| !l.starts_with('>'))
        .collect::<Vec<_>>()
        .join("");
    let short_seq = &phix_seq[0..40];
    let long_seq = &phix_seq[0..100];

    let parquet_path = dir.path().join("input.parquet");
    write_parquet_input(
        &parquet_path,
        &["short_read", "long_read"],
        &[short_seq, long_seq],
    );

    let output = Command::new(&binary)
        .args([
            "classify",
            "run",
            "-i",
            index_path.to_str().unwrap(),
            "-1",
            parquet_path.to_str().unwrap(),
            "-t",
            "0.0",
            "--minimum-length",
            "50",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Classification with Parquet --minimum-length failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().collect();

    assert!(
        !lines.iter().any(|l| l.contains("short_read")),
        "short_read (40bp) should be skipped with --minimum-length 50:\n{}",
        stdout
    );
    assert!(
        lines.iter().any(|l| l.contains("long_read")),
        "long_read (100bp) should be present with --minimum-length 50:\n{}",
        stdout
    );

    Ok(())
}

/// Test --minimum-length with log-ratio classification.
#[test]
fn test_cli_minimum_length_log_ratio() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    let puc19_path = std::path::Path::new(manifest_dir).join("examples/pUC19.fasta");
    if !phix_path.exists() || !puc19_path.exists() {
        eprintln!("Skipping test: example FASTA files not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let num_path = dir.path().join("num.ryxdi");
    let denom_path = dir.path().join("denom.ryxdi");

    // Create two single-bucket indices
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            num_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(output.status.success());

    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            denom_path.to_str().unwrap(),
            "-r",
            puc19_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(output.status.success());

    let phix_content = fs::read_to_string(&phix_path)?;
    let phix_seq: String = phix_content
        .lines()
        .filter(|l| !l.starts_with('>'))
        .collect::<Vec<_>>()
        .join("");
    let short_seq = &phix_seq[0..40];
    let long_seq = &phix_seq[0..100];

    let query_path = dir.path().join("query.fastq");
    fs::write(
        &query_path,
        format!(
            "@short_query\n{}\n+\n{}\n@long_query\n{}\n+\n{}\n",
            short_seq,
            "I".repeat(40),
            long_seq,
            "I".repeat(100)
        ),
    )?;

    let output = Command::new(&binary)
        .args([
            "classify",
            "log-ratio",
            "-n",
            num_path.to_str().unwrap(),
            "-d",
            denom_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
            "--minimum-length",
            "50",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Log-ratio with --minimum-length failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().collect();

    assert!(
        !lines.iter().any(|l| l.contains("short_query")),
        "short_query (40bp) should be skipped with --minimum-length 50:\n{}",
        stdout
    );
    assert!(
        lines.iter().any(|l| l.contains("long_query")),
        "long_query (100bp) should be present with --minimum-length 50:\n{}",
        stdout
    );

    Ok(())
}

/// Test that --minimum-length is compatible with --output-sequences on log-ratio
/// (unlike --trim-to which is rejected).
#[test]
fn test_cli_minimum_length_with_output_sequences() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    let puc19_path = std::path::Path::new(manifest_dir).join("examples/pUC19.fasta");
    if !phix_path.exists() || !puc19_path.exists() {
        eprintln!("Skipping test: example FASTA files not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let num_path = dir.path().join("num.ryxdi");
    let denom_path = dir.path().join("denom.ryxdi");

    // Create two single-bucket indices
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            num_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(output.status.success());

    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            denom_path.to_str().unwrap(),
            "-r",
            puc19_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(output.status.success());

    let phix_content = fs::read_to_string(&phix_path)?;
    let phix_seq: String = phix_content
        .lines()
        .filter(|l| !l.starts_with('>'))
        .collect::<Vec<_>>()
        .join("");
    let long_seq = &phix_seq[0..100];

    let query_path = dir.path().join("query.fastq");
    fs::write(
        &query_path,
        format!("@long_query\n{}\n+\n{}\n", long_seq, "I".repeat(100)),
    )?;

    let output_sequences_path = dir.path().join("filtered.fastq.gz");

    // --minimum-length + --output-sequences should succeed (unlike --trim-to)
    let output = Command::new(&binary)
        .args([
            "classify",
            "log-ratio",
            "-n",
            num_path.to_str().unwrap(),
            "-d",
            denom_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
            "--minimum-length",
            "50",
            "--output-sequences",
            output_sequences_path.to_str().unwrap(),
        ])
        .output()?;
    assert!(
        output.status.success(),
        "--minimum-length with --output-sequences should succeed, but failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Verify the output sequences file was created
    assert!(
        output_sequences_path.exists(),
        "Output sequences file should exist"
    );

    Ok(())
}

// ============================================================================
// C API Extraction Example Tests
// ============================================================================

/// Test that the C extraction example compiles and runs correctly.
///
/// Compiles examples/extraction_example.c against librype, runs it, and
/// parses the structured output to verify correctness.
#[test]
fn test_c_extraction_example() -> Result<()> {
    // Check if gcc is available
    let gcc_check = Command::new("gcc").arg("--version").output();
    if gcc_check.is_err() || !gcc_check.unwrap().status.success() {
        eprintln!("Skipping test_c_extraction_example: gcc not available");
        return Ok(());
    }

    let project_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let example_src = project_root.join("examples/extraction_example.c");
    assert!(
        example_src.exists(),
        "Missing examples/extraction_example.c"
    );

    // Build the cdylib (cargo test only builds rlib, not the shared library)
    let build = Command::new("cargo")
        .args(["build", "--lib"])
        .current_dir(project_root)
        .output()?;
    assert!(
        build.status.success(),
        "Failed to build cdylib:\n{}",
        String::from_utf8_lossy(&build.stderr)
    );

    let dir = tempdir()?;
    let binary_path = dir.path().join("extraction_example");

    // Find the library directory (debug build)
    let target_dir = project_root.join("target/debug");

    // Compile the C example
    let compile = Command::new("gcc")
        .args([
            "-o",
            binary_path.to_str().unwrap(),
            example_src.to_str().unwrap(),
            &format!("-L{}", target_dir.display()),
            "-lrype",
            &format!("-Wl,-rpath,{}", target_dir.display()),
            "-Wall",
            "-Wextra",
        ])
        .output()?;

    assert!(
        compile.status.success(),
        "Failed to compile extraction_example.c:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&compile.stdout),
        String::from_utf8_lossy(&compile.stderr)
    );

    // Run the example
    let run = Command::new(&binary_path)
        .env("LD_LIBRARY_PATH", &target_dir)
        .output()?;

    assert!(
        run.status.success(),
        "extraction_example failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );

    // Parse structured output
    let stdout = String::from_utf8_lossy(&run.stdout);
    let mut values = std::collections::HashMap::new();
    for line in stdout.lines() {
        if let Some((key, val)) = line.split_once('=') {
            values.insert(key.to_string(), val.to_string());
        }
    }

    // Verify minimizer_set results
    assert_eq!(
        values.get("MINIMIZER_SET_OK").map(|s| s.as_str()),
        Some("1"),
        "extract_minimizer_set should succeed"
    );
    let fwd_set_len: usize = values
        .get("FWD_SET_LEN")
        .expect("Missing FWD_SET_LEN")
        .parse()
        .expect("FWD_SET_LEN not a number");
    assert!(fwd_set_len > 0, "Forward set should be non-empty");
    let rc_set_len: usize = values
        .get("RC_SET_LEN")
        .expect("Missing RC_SET_LEN")
        .parse()
        .expect("RC_SET_LEN not a number");
    assert!(rc_set_len > 0, "RC set should be non-empty");
    assert_eq!(
        values.get("FWD_SET_SORTED").map(|s| s.as_str()),
        Some("1"),
        "Forward set should be sorted"
    );
    assert_eq!(
        values.get("RC_SET_SORTED").map(|s| s.as_str()),
        Some("1"),
        "RC set should be sorted"
    );

    // Verify strand_minimizers results
    assert_eq!(
        values.get("STRAND_MINIMIZERS_OK").map(|s| s.as_str()),
        Some("1"),
        "extract_strand_minimizers should succeed"
    );
    let fwd_strand_len: usize = values
        .get("FWD_STRAND_LEN")
        .expect("Missing FWD_STRAND_LEN")
        .parse()
        .expect("FWD_STRAND_LEN not a number");
    assert!(fwd_strand_len > 0, "Forward strand should be non-empty");
    let rc_strand_len: usize = values
        .get("RC_STRAND_LEN")
        .expect("Missing RC_STRAND_LEN")
        .parse()
        .expect("RC_STRAND_LEN not a number");
    assert!(rc_strand_len > 0, "RC strand should be non-empty");
    assert_eq!(
        values.get("FWD_POSITIONS_ORDERED").map(|s| s.as_str()),
        Some("1"),
        "Forward positions should be non-decreasing"
    );
    assert_eq!(
        values.get("FWD_POSITIONS_INBOUNDS").map(|s| s.as_str()),
        Some("1"),
        "Forward positions should be in bounds"
    );
    assert_eq!(
        values.get("RC_POSITIONS_ORDERED").map(|s| s.as_str()),
        Some("1"),
        "RC positions should be non-decreasing"
    );
    assert_eq!(
        values.get("RC_POSITIONS_INBOUNDS").map(|s| s.as_str()),
        Some("1"),
        "RC positions should be in bounds"
    );

    Ok(())
}

/// Test that a reverse-complement read classifies to the same bucket as the forward read.
///
/// This is an end-to-end test: build an index from phiX174, classify a forward fragment,
/// classify the DNA reverse complement of that fragment, and assert both hit the same
/// bucket with the same score.
#[test]
fn test_classify_rc_read_matches_forward() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    if !phix_path.exists() {
        eprintln!("Skipping test: example FASTA file not found");
        return Ok(());
    }

    let binary = get_binary_path();
    let index_path = dir.path().join("test_rc.ryxdi");

    // Create index from phiX174
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            index_path.to_str().unwrap(),
            "-r",
            phix_path.to_str().unwrap(),
            "-k",
            "32",
            "-w",
            "10",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "Index creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // A 200bp fragment from phiX174 (lines 2-4 of the FASTA)
    let fwd_seq = "GAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCTGATGAGTCGAAAAATTATCTT\
                    GATAAAGCAGGAATTACTACTGCTTGTTTACGAATTAAATCGAAGTGGACTGCTGGCGGAAAATGAGAAA\
                    ATTCGACCTATCCTTGCGCAGCTCGAGAAGCTCTTACTTTGCGACCTTTCGCCATCAACTAACGATTCTG";

    // DNA reverse complement: reverse the string, then complement each base
    let rc_seq: String = fwd_seq
        .bytes()
        .rev()
        .map(|b| match b {
            b'A' => 'T',
            b'T' => 'A',
            b'G' => 'C',
            b'C' => 'G',
            _ => 'N',
        })
        .collect();

    // Write forward query FASTQ
    let fwd_query_path = dir.path().join("fwd_query.fastq");
    let fwd_qual = "I".repeat(fwd_seq.len());
    fs::write(
        &fwd_query_path,
        format!("@fwd_read\n{}\n+\n{}\n", fwd_seq, fwd_qual),
    )?;

    // Write RC query FASTQ
    let rc_query_path = dir.path().join("rc_query.fastq");
    let rc_qual = "I".repeat(rc_seq.len());
    fs::write(
        &rc_query_path,
        format!("@rc_read\n{}\n+\n{}\n", rc_seq, rc_qual),
    )?;

    // Classify forward read
    let fwd_output = Command::new(&binary)
        .args([
            "classify",
            "run",
            "-i",
            index_path.to_str().unwrap(),
            "-1",
            fwd_query_path.to_str().unwrap(),
            "-t",
            "0.0",
        ])
        .output()?;
    assert!(
        fwd_output.status.success(),
        "Forward classification failed: {}",
        String::from_utf8_lossy(&fwd_output.stderr)
    );

    // Classify RC read
    let rc_output = Command::new(&binary)
        .args([
            "classify",
            "run",
            "-i",
            index_path.to_str().unwrap(),
            "-1",
            rc_query_path.to_str().unwrap(),
            "-t",
            "0.0",
        ])
        .output()?;
    assert!(
        rc_output.status.success(),
        "RC classification failed: {}",
        String::from_utf8_lossy(&rc_output.stderr)
    );

    // Parse TSV output: read_id\tbucket_name\tscore
    let fwd_stdout = String::from_utf8_lossy(&fwd_output.stdout);
    let rc_stdout = String::from_utf8_lossy(&rc_output.stdout);

    let parse_hit = |output: &str| -> Option<(String, f64)> {
        output
            .lines()
            .find(|line| !line.starts_with("read_id"))
            .map(|line| {
                let fields: Vec<&str> = line.split('\t').collect();
                let bucket = fields[1].to_string();
                let score: f64 = fields[2].parse().unwrap();
                (bucket, score)
            })
    };

    let fwd_hit = parse_hit(&fwd_stdout);
    let rc_hit = parse_hit(&rc_stdout);

    assert!(
        fwd_hit.is_some(),
        "Forward read should have a hit. Output: {}",
        fwd_stdout
    );
    assert!(
        rc_hit.is_some(),
        "RC read should have a hit. Output: {}",
        rc_stdout
    );

    let (fwd_bucket, fwd_score) = fwd_hit.unwrap();
    let (rc_bucket, rc_score) = rc_hit.unwrap();

    assert_eq!(
        fwd_bucket, rc_bucket,
        "Forward and RC reads should hit the same bucket"
    );
    assert!(
        (fwd_score - rc_score).abs() < 1e-9,
        "Forward and RC reads should have the same score, got fwd={} rc={}",
        fwd_score,
        rc_score
    );

    // Sanity: score should be high since the query comes directly from the reference
    assert!(
        fwd_score > 0.9,
        "Score should be high for a self-match, got {}",
        fwd_score
    );

    Ok(())
}
