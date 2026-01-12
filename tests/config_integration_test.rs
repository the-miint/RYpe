use anyhow::Result;
use std::fs::File;
use std::io::Write;
use tempfile::tempdir;

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

// --- BUCKET ADD CONFIG TESTS ---

#[test]
fn test_bucket_add_config_parse_new_bucket() -> Result<()> {
    let dir = tempdir()?;

    // Create a dummy index file
    let index_path = dir.path().join("test.ryidx");
    File::create(&index_path)?;

    // Create a test FASTA file
    let ref_path = dir.path().join("ref.fa");
    let mut file = File::create(&ref_path)?;
    writeln!(file, ">seq1\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")?;

    // Create bucket-add config
    let config_path = dir.path().join("add_config.toml");
    let config_content = format!(r#"
[target]
index = "{}"

[assignment]
mode = "new_bucket"
bucket_name = "NewBucket"

[files]
paths = ["ref.fa"]
"#, index_path.display());

    let mut config_file = File::create(&config_path)?;
    config_file.write_all(config_content.as_bytes())?;
    drop(config_file);

    let cfg = rype::config::parse_bucket_add_config(&config_path)?;

    assert_eq!(cfg.files.paths.len(), 1);
    match &cfg.assignment {
        rype::config::AssignmentSettings::NewBucket { bucket_name } => {
            assert_eq!(bucket_name.as_deref(), Some("NewBucket"));
        }
        _ => panic!("Expected NewBucket mode"),
    }

    rype::config::validate_bucket_add_config(&cfg, dir.path())?;

    Ok(())
}

#[test]
fn test_bucket_add_config_parse_existing_bucket() -> Result<()> {
    let dir = tempdir()?;

    // Create a dummy index file
    let index_path = dir.path().join("test.ryidx");
    File::create(&index_path)?;

    // Create a test FASTA file
    let ref_path = dir.path().join("ref.fa");
    let mut file = File::create(&ref_path)?;
    writeln!(file, ">seq1\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")?;

    // Create bucket-add config
    let config_path = dir.path().join("add_config.toml");
    let config_content = format!(r#"
[target]
index = "{}"

[assignment]
mode = "existing_bucket"
bucket_name = "ExistingBucket"

[files]
paths = ["ref.fa"]
"#, index_path.display());

    let mut config_file = File::create(&config_path)?;
    config_file.write_all(config_content.as_bytes())?;
    drop(config_file);

    let cfg = rype::config::parse_bucket_add_config(&config_path)?;

    match &cfg.assignment {
        rype::config::AssignmentSettings::ExistingBucket { bucket_name } => {
            assert_eq!(bucket_name, "ExistingBucket");
        }
        _ => panic!("Expected ExistingBucket mode"),
    }

    Ok(())
}

#[test]
fn test_bucket_add_config_parse_best_bin() -> Result<()> {
    let dir = tempdir()?;

    // Create a dummy index file
    let index_path = dir.path().join("test.ryidx");
    File::create(&index_path)?;

    // Create a test FASTA file
    let ref_path = dir.path().join("ref.fa");
    let mut file = File::create(&ref_path)?;
    writeln!(file, ">seq1\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")?;

    // Create bucket-add config with best_bin mode
    let config_path = dir.path().join("add_config.toml");
    let config_content = format!(r#"
[target]
index = "{}"

[assignment]
mode = "best_bin"
threshold = 0.25
fallback = "create_new"

[files]
paths = ["ref.fa"]
"#, index_path.display());

    let mut config_file = File::create(&config_path)?;
    config_file.write_all(config_content.as_bytes())?;
    drop(config_file);

    let cfg = rype::config::parse_bucket_add_config(&config_path)?;

    match &cfg.assignment {
        rype::config::AssignmentSettings::BestBin { threshold, fallback } => {
            assert!((threshold - 0.25).abs() < 0.001);
            assert_eq!(*fallback, rype::config::BestBinFallback::CreateNew);
        }
        _ => panic!("Expected BestBin mode"),
    }

    Ok(())
}

#[test]
fn test_bucket_add_config_parse_best_bin_skip() -> Result<()> {
    let dir = tempdir()?;

    // Create a dummy index file
    let index_path = dir.path().join("test.ryidx");
    File::create(&index_path)?;

    // Create a test FASTA file
    let ref_path = dir.path().join("ref.fa");
    let mut file = File::create(&ref_path)?;
    writeln!(file, ">seq1\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")?;

    // Create bucket-add config with skip fallback
    let config_path = dir.path().join("add_config.toml");
    let config_content = format!(r#"
[target]
index = "{}"

[assignment]
mode = "best_bin"
threshold = 0.5
fallback = "skip"

[files]
paths = ["ref.fa"]
"#, index_path.display());

    let mut config_file = File::create(&config_path)?;
    config_file.write_all(config_content.as_bytes())?;
    drop(config_file);

    let cfg = rype::config::parse_bucket_add_config(&config_path)?;

    match &cfg.assignment {
        rype::config::AssignmentSettings::BestBin { threshold, fallback } => {
            assert!((threshold - 0.5).abs() < 0.001);
            assert_eq!(*fallback, rype::config::BestBinFallback::Skip);
        }
        _ => panic!("Expected BestBin mode"),
    }

    Ok(())
}

#[test]
fn test_bucket_add_config_invalid_threshold() -> Result<()> {
    let dir = tempdir()?;

    // Create a dummy index file
    let index_path = dir.path().join("test.ryidx");
    File::create(&index_path)?;

    // Create config with invalid threshold
    let config_path = dir.path().join("add_config.toml");
    let config_content = format!(r#"
[target]
index = "{}"

[assignment]
mode = "best_bin"
threshold = 1.5

[files]
paths = ["nonexistent.fa"]
"#, index_path.display());

    let mut config_file = File::create(&config_path)?;
    config_file.write_all(config_content.as_bytes())?;
    drop(config_file);

    let result = rype::config::parse_bucket_add_config(&config_path);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("threshold must be between 0.0 and 1.0"));

    Ok(())
}

#[test]
fn test_bucket_add_config_validation_missing_index() -> Result<()> {
    let dir = tempdir()?;

    // Create a test FASTA file
    let ref_path = dir.path().join("ref.fa");
    let mut file = File::create(&ref_path)?;
    writeln!(file, ">seq1\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")?;

    // Create config pointing to non-existent index
    let config_path = dir.path().join("add_config.toml");
    let config_content = r#"
[target]
index = "nonexistent.ryidx"

[assignment]
mode = "new_bucket"

[files]
paths = ["ref.fa"]
"#;

    let mut config_file = File::create(&config_path)?;
    config_file.write_all(config_content.as_bytes())?;
    drop(config_file);

    let cfg = rype::config::parse_bucket_add_config(&config_path)?;
    let result = rype::config::validate_bucket_add_config(&cfg, dir.path());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Target index not found"));

    Ok(())
}

#[test]
fn test_bucket_add_config_validation_missing_file() -> Result<()> {
    let dir = tempdir()?;

    // Create a dummy index file
    let index_path = dir.path().join("test.ryidx");
    File::create(&index_path)?;

    // Create config pointing to non-existent file
    let config_path = dir.path().join("add_config.toml");
    let config_content = format!(r#"
[target]
index = "{}"

[assignment]
mode = "new_bucket"

[files]
paths = ["nonexistent.fa"]
"#, index_path.display());

    let mut config_file = File::create(&config_path)?;
    config_file.write_all(config_content.as_bytes())?;
    drop(config_file);

    let cfg = rype::config::parse_bucket_add_config(&config_path)?;
    let result = rype::config::validate_bucket_add_config(&cfg, dir.path());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("File not found"));

    Ok(())
}

#[test]
fn test_bucket_add_config_empty_files() -> Result<()> {
    let dir = tempdir()?;

    // Create config with empty files list
    let config_path = dir.path().join("add_config.toml");
    let config_content = r#"
[target]
index = "test.ryidx"

[assignment]
mode = "new_bucket"

[files]
paths = []
"#;

    let mut config_file = File::create(&config_path)?;
    config_file.write_all(config_content.as_bytes())?;
    drop(config_file);

    let result = rype::config::parse_bucket_add_config(&config_path);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("at least one file"));

    Ok(())
}
