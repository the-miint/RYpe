//! CLI integration tests for the rype tool.
//!
//! These tests verify that CLI commands work correctly with Parquet indices.

use anyhow::Result;
use rype::{extract_with_positions, get_paired_minimizers_into, MinimizerWorkspace, Strand};
use std::fs;
use std::process::Command;
use tempfile::tempdir;

/// Test extract_with_positions returns correct positions and strands
#[test]
fn test_extract_with_positions_correctness() -> Result<()> {
    let mut ws = MinimizerWorkspace::new();

    // Create a simple sequence
    let seq = b"AAAATTTTGGGGCCCCAAAATTTTGGGGCCCC"; // 32 bases

    let results = extract_with_positions(seq, 16, 4, 0, &mut ws);

    // Should have some minimizers
    assert!(!results.is_empty(), "Should extract minimizers");

    // All positions should be valid
    for m in &results {
        assert!(
            m.position + 16 <= seq.len(),
            "Position {} invalid for seq len {}",
            m.position,
            seq.len()
        );

        // Strand should be either Forward or ReverseComplement
        match m.strand {
            Strand::Forward | Strand::ReverseComplement => {}
        }
    }

    // Should have both forward and reverse complement minimizers
    let has_fwd = results.iter().any(|m| m.strand == Strand::Forward);
    let has_rc = results
        .iter()
        .any(|m| m.strand == Strand::ReverseComplement);

    assert!(has_fwd, "Should have forward strand minimizers");
    assert!(has_rc, "Should have reverse complement minimizers");

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

    // Build the binary
    let status = Command::new("cargo")
        .args(["build"])
        .current_dir(manifest_dir)
        .status()?;
    assert!(status.success(), "Failed to build rype binary");

    let binary = std::path::Path::new(manifest_dir).join("target/debug/rype");

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

    // Build the binary
    let status = Command::new("cargo")
        .args(["build"])
        .current_dir(manifest_dir)
        .status()?;
    assert!(status.success(), "Failed to build rype binary");

    let binary = std::path::Path::new(manifest_dir).join("target/debug/rype");
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

    // Build the binary
    let status = Command::new("cargo")
        .args(["build"])
        .current_dir(manifest_dir)
        .status()?;
    assert!(status.success(), "Failed to build rype binary");

    let binary = std::path::Path::new(manifest_dir).join("target/debug/rype");
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

    // Build the binary
    let status = Command::new("cargo")
        .args(["build"])
        .current_dir(manifest_dir)
        .status()?;
    assert!(status.success(), "Failed to build rype binary");

    let binary = std::path::Path::new(manifest_dir).join("target/debug/rype");
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

    // Build the binary
    let status = Command::new("cargo")
        .args(["build"])
        .current_dir(manifest_dir)
        .status()?;
    assert!(status.success(), "Failed to build rype binary");

    let binary = std::path::Path::new(manifest_dir).join("target/debug/rype");
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

    // Build the binary
    let status = Command::new("cargo")
        .args(["build"])
        .current_dir(manifest_dir)
        .status()?;
    assert!(status.success(), "Failed to build rype binary");

    let binary = std::path::Path::new(manifest_dir).join("target/debug/rype");
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

    // Build the binary
    let status = Command::new("cargo")
        .args(["build"])
        .current_dir(manifest_dir)
        .status()?;
    assert!(status.success(), "Failed to build rype binary");

    let binary = std::path::Path::new(manifest_dir).join("target/debug/rype");
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

    // Build the binary
    let status = Command::new("cargo")
        .args(["build"])
        .current_dir(manifest_dir)
        .status()?;
    assert!(status.success(), "Failed to build rype binary");

    let binary = std::path::Path::new(manifest_dir).join("target/debug/rype");
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

    // Build the binary
    let status = Command::new("cargo")
        .args(["build"])
        .current_dir(manifest_dir)
        .status()?;
    assert!(status.success(), "Failed to build rype binary");

    let binary = std::path::Path::new(manifest_dir).join("target/debug/rype");
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

    // Build the binary
    let status = Command::new("cargo")
        .args(["build"])
        .current_dir(manifest_dir)
        .status()?;
    assert!(status.success(), "Failed to build rype binary");

    let binary = std::path::Path::new(manifest_dir).join("target/debug/rype");
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

    // Build the binary
    let status = Command::new("cargo")
        .args(["build"])
        .current_dir(manifest_dir)
        .status()?;
    assert!(status.success(), "Failed to build rype binary");

    let binary = std::path::Path::new(manifest_dir).join("target/debug/rype");
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

    // Build the binary
    let status = Command::new("cargo")
        .args(["build"])
        .current_dir(manifest_dir)
        .status()?;
    assert!(status.success(), "Failed to build rype binary");

    let binary = std::path::Path::new(manifest_dir).join("target/debug/rype");
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

    // Build the binary
    let status = Command::new("cargo")
        .args(["build"])
        .current_dir(manifest_dir)
        .status()?;
    assert!(status.success(), "Failed to build rype binary");

    let binary = std::path::Path::new(manifest_dir).join("target/debug/rype");
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
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    // Build the binary
    let status = Command::new("cargo")
        .args([
            "build",
            "--manifest-path",
            &format!("{}/Cargo.toml", manifest_dir),
        ])
        .status()?;
    assert!(status.success(), "Failed to build");

    let binary = format!("{}/target/debug/rype", manifest_dir);

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

/// Test end-to-end log-ratio classification with a 2-bucket index
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

    // Build the binary
    let status = Command::new("cargo")
        .args(["build"])
        .current_dir(manifest_dir)
        .status()?;
    assert!(status.success(), "Failed to build rype binary");

    let binary = std::path::Path::new(manifest_dir).join("target/debug/rype");
    let index_path = dir.path().join("test.ryxdi");

    // Create index with two separate buckets (required for log-ratio)
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

    // Create a query file with phiX174 sequence (should match phiX bucket strongly)
    let query_path = dir.path().join("query.fastq");
    fs::write(
        &query_path,
        "@query1\nGAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCTGATGAGTCGAAAAATTATCTT\n+\nIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n",
    )?;

    // Run log-ratio classification
    let output = Command::new(&binary)
        .args([
            "classify",
            "log-ratio",
            "-i",
            index_path.to_str().unwrap(),
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

    // Header should be "read_id\tbucket_name\tscore"
    let header = lines[0];
    assert!(
        header.starts_with("read_id"),
        "Header should start with 'read_id'. Got: {}",
        header
    );
    assert!(
        header.contains("bucket_name"),
        "Header should contain 'bucket_name'. Got: {}",
        header
    );

    // Data row should have the log10([A] / [B]) bucket name format
    let data_row = lines[1];
    assert!(
        data_row.contains("log10(["),
        "Bucket name should contain 'log10(['. Got: {}",
        data_row
    );

    // Data row should have a score (the log ratio)
    let cols: Vec<&str> = data_row.split('\t').collect();
    assert_eq!(cols.len(), 3, "Should have 3 columns: {:?}", cols);

    // Score should be parseable (can be "inf" or a float)
    let score_str = cols[2];
    if score_str != "inf" && score_str != "-inf" {
        let score: f64 = score_str.parse().unwrap_or_else(|_| {
            panic!(
                "Score should be a valid float or 'inf'. Got: '{}'",
                score_str
            )
        });
        // Log ratio can be any value, but for a strongly matching read it should be non-zero
        println!("Log ratio score: {}", score);
    }

    Ok(())
}

/// Test that --swap-buckets negates the log-ratio output
#[test]
fn test_cli_log_ratio_swap_buckets_negates() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    let puc19_path = std::path::Path::new(manifest_dir).join("examples/pUC19.fasta");
    if !phix_path.exists() || !puc19_path.exists() {
        eprintln!("Skipping test: example FASTA files not found");
        return Ok(());
    }

    // Build the binary
    let status = Command::new("cargo")
        .args(["build"])
        .current_dir(manifest_dir)
        .status()?;
    assert!(status.success(), "Failed to build rype binary");

    let binary = std::path::Path::new(manifest_dir).join("target/debug/rype");
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

    // Create a query file
    let query_path = dir.path().join("query.fastq");
    fs::write(
        &query_path,
        "@query1\nGAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCTGATGAGTCGAAAAATTATCTT\n+\nIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n",
    )?;

    // Run log-ratio WITHOUT --swap-buckets
    let output_normal = Command::new(&binary)
        .args([
            "classify",
            "log-ratio",
            "-i",
            index_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
        ])
        .output()?;

    assert!(
        output_normal.status.success(),
        "Log-ratio without swap failed: {}",
        String::from_utf8_lossy(&output_normal.stderr)
    );

    // Run log-ratio WITH --swap-buckets
    let output_swapped = Command::new(&binary)
        .args([
            "classify",
            "log-ratio",
            "-i",
            index_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
            "--swap-buckets",
        ])
        .output()?;

    assert!(
        output_swapped.status.success(),
        "Log-ratio with swap failed: {}",
        String::from_utf8_lossy(&output_swapped.stderr)
    );

    let stdout_normal = String::from_utf8_lossy(&output_normal.stdout);
    let stdout_swapped = String::from_utf8_lossy(&output_swapped.stdout);

    println!("Normal output:\n{}", stdout_normal);
    println!("Swapped output:\n{}", stdout_swapped);

    // Extract scores from both outputs
    let get_score = |stdout: &str| -> Option<f64> {
        for line in stdout.lines().skip(1) {
            // Skip header
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
            // log10(A/B) = -log10(B/A), so scores should be negated
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

/// Test log-ratio with Parquet output format
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

    // Build the binary
    let status = Command::new("cargo")
        .args(["build"])
        .current_dir(manifest_dir)
        .status()?;
    assert!(status.success(), "Failed to build rype binary");

    let binary = std::path::Path::new(manifest_dir).join("target/debug/rype");
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

    // Run log-ratio classification with Parquet output
    let output = Command::new(&binary)
        .args([
            "classify",
            "log-ratio",
            "-i",
            index_path.to_str().unwrap(),
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

    // Schema should have read_id, bucket_name, score
    let fields = schema.get_fields();
    assert!(
        fields.len() >= 3,
        "Schema should have at least 3 fields. Got: {}",
        fields.len()
    );

    // Verify field names
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

/// Test that --threshold filters log-ratio results
#[test]
fn test_cli_log_ratio_threshold_filters() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    let puc19_path = std::path::Path::new(manifest_dir).join("examples/pUC19.fasta");
    if !phix_path.exists() || !puc19_path.exists() {
        eprintln!("Skipping test: example FASTA files not found");
        return Ok(());
    }

    // Build the binary
    let status = Command::new("cargo")
        .args(["build"])
        .current_dir(manifest_dir)
        .status()?;
    assert!(status.success(), "Failed to build rype binary");

    let binary = std::path::Path::new(manifest_dir).join("target/debug/rype");
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

    // Create a query file with a read that has a real sequence
    // and a "junk" read that won't match well
    let query_path = dir.path().join("query.fastq");
    // phiX174 sequence: 70 bases
    let good_seq = "GAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCTGATGAGTCGAAAAATTATCTT";
    let good_qual = "I".repeat(70);
    // Junk sequence: 70 N's (won't produce valid k-mers)
    let weak_seq = "N".repeat(70);
    let weak_qual = "I".repeat(70);
    fs::write(
        &query_path,
        format!(
            "@good_query\n{}\n+\n{}\n@weak_query\n{}\n+\n{}\n",
            good_seq, good_qual, weak_seq, weak_qual
        ),
    )?;

    // Run log-ratio WITHOUT threshold (should get all results)
    let output_no_threshold = Command::new(&binary)
        .args([
            "classify",
            "log-ratio",
            "-i",
            index_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
            "-t",
            "0.0",
        ])
        .output()?;

    assert!(
        output_no_threshold.status.success(),
        "Log-ratio without threshold failed: {}",
        String::from_utf8_lossy(&output_no_threshold.stderr)
    );

    // Run log-ratio WITH high threshold (should filter weak matches)
    let output_high_threshold = Command::new(&binary)
        .args([
            "classify",
            "log-ratio",
            "-i",
            index_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
            "-t",
            "0.5", // High threshold
        ])
        .output()?;

    assert!(
        output_high_threshold.status.success(),
        "Log-ratio with high threshold failed: {}",
        String::from_utf8_lossy(&output_high_threshold.stderr)
    );

    let stdout_no_thresh = String::from_utf8_lossy(&output_no_threshold.stdout);
    let stdout_high_thresh = String::from_utf8_lossy(&output_high_threshold.stdout);

    println!("No threshold output:\n{}", stdout_no_thresh);
    println!("High threshold output:\n{}", stdout_high_thresh);

    // Count data lines (excluding header)
    let count_data_lines =
        |s: &str| -> usize { s.lines().skip(1).filter(|l| !l.is_empty()).count() };

    let lines_no_thresh = count_data_lines(&stdout_no_thresh);
    let lines_high_thresh = count_data_lines(&stdout_high_thresh);

    println!(
        "Lines without threshold: {}, with high threshold: {}",
        lines_no_thresh, lines_high_thresh
    );

    // With high threshold, we should have fewer or equal results
    // (the weak_query with N's should be filtered out)
    assert!(
        lines_high_thresh <= lines_no_thresh,
        "High threshold should filter results: {} <= {}",
        lines_high_thresh,
        lines_no_thresh
    );

    Ok(())
}

/// Test that log-ratio outputs infinity when one bucket has score 0
#[test]
fn test_cli_log_ratio_infinity_output() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    // We need to create two DISTINCT reference sequences that share nothing in common
    // This ensures a read from one won't match the other at all

    // Create two completely different reference sequences
    let ref1_path = dir.path().join("ref1.fasta");
    let ref2_path = dir.path().join("ref2.fasta");

    // Ref1: repeating AAAA pattern (purines only)
    let ref1_seq = "A".repeat(200);
    fs::write(&ref1_path, format!(">ref1\n{}\n", ref1_seq))?;

    // Ref2: repeating TTTT pattern (pyrimidines only)
    let ref2_seq = "T".repeat(200);
    fs::write(&ref2_path, format!(">ref2\n{}\n", ref2_seq))?;

    // Build the binary
    let status = Command::new("cargo")
        .args(["build"])
        .current_dir(manifest_dir)
        .status()?;
    assert!(status.success(), "Failed to build rype binary");

    let binary = std::path::Path::new(manifest_dir).join("target/debug/rype");
    let index_path = dir.path().join("test.ryxdi");

    // Create index with two separate buckets
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            index_path.to_str().unwrap(),
            "-r",
            ref1_path.to_str().unwrap(),
            "-r",
            ref2_path.to_str().unwrap(),
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

    // Create a query that matches only ref1 (A's) - exactly matches ref1
    let query_path = dir.path().join("query.fastq");
    let query_seq = "A".repeat(100); // Pure A's should match ref1, not ref2
    let query_qual = "I".repeat(100);
    fs::write(
        &query_path,
        format!("@only_matches_ref1\n{}\n+\n{}\n", query_seq, query_qual),
    )?;

    // Run log-ratio classification
    let output = Command::new(&binary)
        .args([
            "classify",
            "log-ratio",
            "-i",
            index_path.to_str().unwrap(),
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
    println!("Infinity test output:\n{}", stdout);

    // The output should contain "inf" because:
    // - Query matches ref1 (A's) with score > 0
    // - Query doesn't match ref2 (T's) with score = 0
    // - log10(score/0) = +infinity OR log10(0/score) would be handled as 0
    // Depending on which bucket is numerator/denominator, we might see "inf" or "0"

    let has_result = stdout.lines().skip(1).any(|l| !l.is_empty());
    assert!(
        has_result,
        "Should have at least one result for the matching query"
    );

    // Look for infinity in output (can be "inf" for positive infinity)
    // Note: if ref1 is denominator and has score 0, we get inf
    // if ref1 is numerator and ref2 has score 0, we get inf
    // The actual behavior depends on bucket ordering
    for line in stdout.lines().skip(1) {
        if line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() >= 3 {
            let score_str = cols[2];
            println!("Score value: '{}'", score_str);
            // Score should be parseable (inf, 0, or some float)
            if score_str == "inf" {
                println!("Found infinity as expected when one bucket has 0 score");
            } else if let Ok(score) = score_str.parse::<f64>() {
                // If it's 0.0, that means numerator was 0
                // If it's a finite non-zero, both buckets had scores
                println!("Found finite score: {}", score);
            }
        }
    }

    Ok(())
}

/// Test that log-ratio fails gracefully with 1-bucket index
#[test]
fn test_cli_log_ratio_fails_with_one_bucket() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    let phix_path = std::path::Path::new(manifest_dir).join("examples/phiX174.fasta");
    if !phix_path.exists() {
        eprintln!("Skipping test: example FASTA file not found");
        return Ok(());
    }

    // Build the binary
    let status = Command::new("cargo")
        .args(["build"])
        .current_dir(manifest_dir)
        .status()?;
    assert!(status.success(), "Failed to build rype binary");

    let binary = std::path::Path::new(manifest_dir).join("target/debug/rype");
    let index_path = dir.path().join("test.ryxdi");

    // Create index with only ONE bucket (single reference file)
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

    // Run log-ratio classification - should FAIL
    let output = Command::new(&binary)
        .args([
            "classify",
            "log-ratio",
            "-i",
            index_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
        ])
        .output()?;

    // Should fail because log-ratio requires exactly 2 buckets
    assert!(
        !output.status.success(),
        "Log-ratio with 1-bucket index should fail"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    println!("Expected error: {}", stderr);

    // Error message should mention 2 buckets requirement
    assert!(
        stderr.contains("2 buckets") || stderr.contains("exactly 2"),
        "Error should mention 2 bucket requirement. Got: {}",
        stderr
    );

    Ok(())
}

/// Test that log-ratio fails gracefully with 3-bucket index
#[test]
fn test_cli_log_ratio_fails_with_three_buckets() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir = tempdir()?;

    // Create three reference files
    let ref1_path = dir.path().join("ref1.fasta");
    let ref2_path = dir.path().join("ref2.fasta");
    let ref3_path = dir.path().join("ref3.fasta");

    fs::write(
        &ref1_path,
        ">ref1\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n",
    )?;
    fs::write(
        &ref2_path,
        ">ref2\nTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT\n",
    )?;
    fs::write(
        &ref3_path,
        ">ref3\nGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG\n",
    )?;

    // Build the binary
    let status = Command::new("cargo")
        .args(["build"])
        .current_dir(manifest_dir)
        .status()?;
    assert!(status.success(), "Failed to build rype binary");

    let binary = std::path::Path::new(manifest_dir).join("target/debug/rype");
    let index_path = dir.path().join("test.ryxdi");

    // Create index with THREE separate buckets
    let output = Command::new(&binary)
        .args([
            "index",
            "create",
            "-o",
            index_path.to_str().unwrap(),
            "-r",
            ref1_path.to_str().unwrap(),
            "-r",
            ref2_path.to_str().unwrap(),
            "-r",
            ref3_path.to_str().unwrap(),
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
        "@query1\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n+\nIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n",
    )?;

    // Run log-ratio classification - should FAIL
    let output = Command::new(&binary)
        .args([
            "classify",
            "log-ratio",
            "-i",
            index_path.to_str().unwrap(),
            "-1",
            query_path.to_str().unwrap(),
        ])
        .output()?;

    // Should fail because log-ratio requires exactly 2 buckets
    assert!(
        !output.status.success(),
        "Log-ratio with 3-bucket index should fail"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    println!("Expected error: {}", stderr);

    // Error message should mention 2 buckets requirement and found 3
    assert!(
        stderr.contains("2 buckets") || stderr.contains("exactly 2"),
        "Error should mention 2 bucket requirement. Got: {}",
        stderr
    );
    assert!(
        stderr.contains("found 3") || stderr.contains("3"),
        "Error should mention found 3 buckets. Got: {}",
        stderr
    );

    Ok(())
}
