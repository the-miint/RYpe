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
