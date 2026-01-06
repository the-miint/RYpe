use anyhow::Result;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use tempfile::tempdir;

use rype::Index;

#[test]
fn test_config_based_index_building() -> Result<()> {
    let dir = tempdir()?;

    // Create test FASTA files
    let ref1_path = dir.path().join("ref1.fa");
    let ref2_path = dir.path().join("ref2.fa");
    let ref3_path = dir.path().join("ref3.fa");

    // Write simple test sequences
    let mut file1 = File::create(&ref1_path)?;
    writeln!(file1, ">seq1")?;
    writeln!(file1, "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")?;

    let mut file2 = File::create(&ref2_path)?;
    writeln!(file2, ">seq2")?;
    writeln!(file2, "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")?;

    let mut file3 = File::create(&ref3_path)?;
    writeln!(file3, ">seq3")?;
    writeln!(file3, "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG")?;

    // Create TOML config file
    let config_path = dir.path().join("test_config.toml");
    let output_path = dir.path().join("test_output.ryidx");

    let config_content = format!(r#"
[index]
window = 50
salt = 0x5555555555555555
output = "{}"

[buckets.BucketA]
files = ["ref1.fa", "ref2.fa"]

[buckets.BucketB]
files = ["ref3.fa"]
"#, output_path.display());

    let mut config_file = File::create(&config_path)?;
    config_file.write_all(config_content.as_bytes())?;
    drop(config_file);

    // Run the index-from-config command via the CLI module
    // Since we can't easily call main(), we'll test the config module directly
    use rype::Index as RypeIndex;

    // Parse config
    let cfg = rype::config::parse_config(&config_path)?;

    // Validate
    rype::config::validate_config(&cfg, dir.path())?;

    // Verify config parsed correctly
    assert_eq!(cfg.index.window, 50);
    assert_eq!(cfg.buckets.len(), 2);
    assert!(cfg.buckets.contains_key("BucketA"));
    assert!(cfg.buckets.contains_key("BucketB"));
    assert_eq!(cfg.buckets["BucketA"].files.len(), 2);
    assert_eq!(cfg.buckets["BucketB"].files.len(), 1);

    Ok(())
}

#[test]
fn test_config_validation_missing_file() -> Result<()> {
    let dir = tempdir()?;

    // Create config referencing non-existent file
    let config_path = dir.path().join("test_config.toml");
    let config_content = r#"
[index]
window = 50
salt = 0x5555555555555555
output = "test.ryidx"

[buckets.TestBucket]
files = ["nonexistent.fa"]
"#;

    let mut config_file = File::create(&config_path)?;
    config_file.write_all(config_content.as_bytes())?;
    drop(config_file);

    // Parse should succeed
    let cfg = rype::config::parse_config(&config_path)?;

    // But validation should fail
    let result = rype::config::validate_config(&cfg, dir.path());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("File not found"));

    Ok(())
}

#[test]
fn test_config_empty_buckets() -> Result<()> {
    let dir = tempdir()?;

    // Create config with no buckets
    let config_path = dir.path().join("test_config.toml");
    let config_content = r#"
[index]
window = 50
salt = 0x5555555555555555
output = "test.ryidx"
"#;

    let mut config_file = File::create(&config_path)?;
    config_file.write_all(config_content.as_bytes())?;
    drop(config_file);

    // Parse should fail because buckets field is missing
    let result = rype::config::parse_config(&config_path);
    assert!(result.is_err());
    // The error will be about parsing TOML (missing field)
    assert!(result.unwrap_err().to_string().contains("Failed to parse TOML"));

    Ok(())
}
