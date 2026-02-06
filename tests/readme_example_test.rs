//! Tests that the README example code compiles and runs correctly.
//!
//! This extracts the actual Rust code from README.md and runs it,
//! ensuring the documentation stays in sync with the API.

use std::io::Write;
use std::process::Command;

/// Extract Rust code blocks from markdown content.
fn extract_rust_code_blocks(markdown: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut in_rust_block = false;
    let mut current_block = String::new();

    for line in markdown.lines() {
        if line.starts_with("```rust") {
            in_rust_block = true;
            current_block.clear();
        } else if line == "```" && in_rust_block {
            in_rust_block = false;
            if !current_block.trim().is_empty() {
                blocks.push(current_block.clone());
            }
        } else if in_rust_block {
            current_block.push_str(line);
            current_block.push('\n');
        }
    }

    blocks
}

/// Recursively copy a directory
fn copy_dir_all(src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        if ty.is_dir() {
            copy_dir_all(&entry.path(), &dst.join(entry.file_name()))?;
        } else {
            std::fs::copy(entry.path(), dst.join(entry.file_name()))?;
        }
    }
    Ok(())
}

/// Extract C code blocks from markdown content
fn extract_c_code_blocks(markdown: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut in_c_block = false;
    let mut current_block = String::new();

    for line in markdown.lines() {
        if line.starts_with("```c") && !line.contains(",") {
            in_c_block = true;
            current_block.clear();
        } else if line == "```" && in_c_block {
            in_c_block = false;
            if !current_block.trim().is_empty() {
                blocks.push(current_block.clone());
            }
        } else if in_c_block {
            current_block.push_str(line);
            current_block.push('\n');
        }
    }

    blocks
}

/// Test that README C example compiles and runs correctly.
/// This extracts the actual C code from README.md and compiles/runs it.
#[test]
fn test_readme_c_example_compiles_and_runs() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let readme_path = std::path::Path::new(manifest_dir).join("README.md");
    let readme = std::fs::read_to_string(&readme_path).expect("Failed to read README.md");

    let c_blocks = extract_c_code_blocks(&readme);
    if c_blocks.is_empty() {
        println!("No C code blocks found in README.md, skipping C test");
        return;
    }

    // Check for gcc
    let gcc_check = Command::new("gcc").arg("--version").output();
    if gcc_check.is_err() || !gcc_check.unwrap().status.success() {
        println!("Skipping C test: gcc not available");
        return;
    }

    // Build the cdylib
    let status = Command::new("cargo")
        .args(["build", "--release"])
        .current_dir(manifest_dir)
        .status()
        .expect("Failed to run cargo build");
    assert!(status.success(), "Failed to build librype");

    let lib_dir = std::path::Path::new(manifest_dir).join("target/release");
    let header_path = std::path::Path::new(manifest_dir).join("rype.h");

    if !header_path.exists() {
        println!("Skipping C test: rype.h header not found");
        return;
    }

    // Create test index for the C example to use
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let index_path = temp_dir.path().join("test.ryxdi");

    {
        use rype::{extract_into, BucketData, MinimizerWorkspace, ParquetWriteOptions};
        let mut ws = MinimizerWorkspace::new();
        // Use the same sequence that the README C example queries
        let seq = b"GAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCTGATGAGTCGAAAAATTATCTT";
        extract_into(seq, 32, 10, 0, &mut ws);
        let mut mins: Vec<u64> = ws.buffer.drain(..).collect();
        mins.sort();
        mins.dedup();

        let buckets = vec![BucketData {
            bucket_id: 1,
            bucket_name: "phiX174".to_string(),
            sources: vec!["phiX174_fragment".to_string()],
            minimizers: mins,
        }];

        let options = ParquetWriteOptions::default();
        rype::create_parquet_inverted_index(&index_path, buckets, 32, 10, 0, None, Some(&options))
            .expect("Failed to create index");
    }

    for (i, c_code) in c_blocks.iter().enumerate() {
        // Update header include path to absolute path
        let c_code = c_code.replace(
            "#include \"rype.h\"",
            &format!("#include \"{}\"", header_path.display()),
        );

        // Write the C code to a file
        let c_file = temp_dir.path().join(format!("readme_example_{}.c", i));
        std::fs::write(&c_file, &c_code).expect("Failed to write C file");

        // Compile with gcc
        let output_binary = temp_dir.path().join(format!("readme_example_{}", i));
        let compile_output = Command::new("gcc")
            .args([
                "-o",
                output_binary.to_str().unwrap(),
                c_file.to_str().unwrap(),
                "-L",
                lib_dir.to_str().unwrap(),
                "-lrype",
                "-Wl,-rpath",
                lib_dir.to_str().unwrap(),
            ])
            .output()
            .expect("Failed to run gcc");

        if !compile_output.status.success() {
            let stderr = String::from_utf8_lossy(&compile_output.stderr);
            panic!(
                "Failed to compile README C example {}:\n\nC Code:\n{}\n\nCompiler error:\n{}",
                i + 1,
                c_code,
                stderr
            );
        }

        // Run the compiled program with the test index
        let run_output = Command::new(&output_binary)
            .arg(index_path.to_str().unwrap())
            .env("LD_LIBRARY_PATH", &lib_dir)
            .output()
            .expect("Failed to run compiled C example");

        if !run_output.status.success() {
            let stderr = String::from_utf8_lossy(&run_output.stderr);
            let stdout = String::from_utf8_lossy(&run_output.stdout);
            panic!(
                "README C example {} crashed:\n\nStdout:\n{}\n\nStderr:\n{}",
                i + 1,
                stdout,
                stderr
            );
        }

        let stdout = String::from_utf8_lossy(&run_output.stdout);
        println!("README C example {} output:\n{}", i + 1, stdout);
    }

    println!("All README C examples compiled and ran successfully");
}

#[test]
fn test_readme_rust_examples_compile_and_run() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let readme_path = std::path::Path::new(manifest_dir).join("README.md");
    let readme = std::fs::read_to_string(&readme_path).expect("Failed to read README.md");
    let code_blocks = extract_rust_code_blocks(&readme);

    assert!(
        !code_blocks.is_empty(),
        "No Rust code blocks found in README.md"
    );

    for (i, code) in code_blocks.iter().enumerate() {
        // Create a temporary directory for this test
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");

        // Copy the examples directory so example files are available
        let examples_src = std::path::Path::new(manifest_dir).join("examples");
        let examples_dst = temp_dir.path().join("examples");
        if examples_src.exists() {
            copy_dir_all(&examples_src, &examples_dst).expect("Failed to copy examples directory");
        }

        // Create the cargo project structure
        let cargo_toml = temp_dir.path().join("Cargo.toml");
        let src_dir = temp_dir.path().join("src");
        std::fs::create_dir_all(&src_dir).expect("Failed to create src dir");
        let main_rs = src_dir.join("main.rs");

        // Write the code to main.rs
        let mut file = std::fs::File::create(&main_rs).expect("Failed to create main.rs");
        file.write_all(code.as_bytes())
            .expect("Failed to write example code");

        let cargo_content = format!(
            r#"[package]
name = "readme_example_{}"
version = "0.1.0"
edition = "2021"

[dependencies]
rype = {{ path = "{}" }}
needletail = "0.6"
anyhow = "1.0"
tempfile = "3"
"#,
            i, manifest_dir
        );

        std::fs::write(&cargo_toml, cargo_content).expect("Failed to write Cargo.toml");

        // Build and run the example
        let output = Command::new("cargo")
            .args(["run", "--release"])
            .current_dir(temp_dir.path())
            .output()
            .expect("Failed to execute cargo run");

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            panic!(
                "README example {} failed to compile/run:\n\nCode:\n{}\n\nStderr:\n{}\n\nStdout:\n{}",
                i + 1,
                code,
                stderr,
                stdout
            );
        }

        println!("README example {} passed", i + 1);
    }
}
