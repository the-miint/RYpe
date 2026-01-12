use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
pub struct ConfigFile {
    pub index: IndexSettings,
    pub buckets: HashMap<String, BucketDefinition>,
}

#[derive(Debug, Deserialize)]
pub struct IndexSettings {
    #[serde(default = "default_k")]
    pub k: usize,
    pub window: usize,
    pub salt: u64,
    pub output: PathBuf,
}

fn default_k() -> usize {
    64
}

#[derive(Debug, Deserialize)]
pub struct BucketDefinition {
    pub files: Vec<PathBuf>,
}

pub fn parse_config(path: &Path) -> Result<ConfigFile> {
    let contents = fs::read_to_string(path)
        .context(format!("Failed to read config file: {}", path.display()))?;

    let config: ConfigFile = toml::from_str(&contents)
        .context("Failed to parse TOML config")?;

    if config.buckets.is_empty() {
        return Err(anyhow!("Config must define at least one bucket"));
    }

    if !matches!(config.index.k, 16 | 32 | 64) {
        return Err(anyhow!("Config error: k must be 16, 32, or 64 (got {})", config.index.k));
    }

    Ok(config)
}

pub fn validate_config(config: &ConfigFile, config_dir: &Path) -> Result<()> {
    for (bucket_name, bucket_def) in &config.buckets {
        if bucket_def.files.is_empty() {
            return Err(anyhow!("Bucket '{}' has no files", bucket_name));
        }

        for file_path in &bucket_def.files {
            let abs_path = resolve_path(config_dir, file_path);
            if !abs_path.exists() {
                return Err(anyhow!(
                    "File not found for bucket '{}': {}",
                    bucket_name,
                    abs_path.display()
                ));
            }
        }
    }

    Ok(())
}

pub fn resolve_path(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

// ============================================================================
// BucketAddConfig - for adding files to an existing index
// ============================================================================

/// Configuration for bucket-add-config command
#[derive(Debug, Deserialize)]
pub struct BucketAddConfig {
    pub target: BucketAddTarget,
    pub assignment: AssignmentSettings,
    pub files: FileList,
}

#[derive(Debug, Deserialize)]
pub struct BucketAddTarget {
    pub index: PathBuf,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "mode")]
pub enum AssignmentSettings {
    #[serde(rename = "new_bucket")]
    NewBucket {
        /// Optional name for the new bucket (defaults to first filename)
        bucket_name: Option<String>,
    },
    #[serde(rename = "existing_bucket")]
    ExistingBucket {
        /// Name of existing bucket to add files to
        bucket_name: String,
    },
    #[serde(rename = "best_bin")]
    BestBin {
        /// Query-centric overlap threshold (0.0-1.0)
        threshold: f64,
        /// What to do when no bucket meets threshold
        #[serde(default)]
        fallback: BestBinFallback,
    },
}

#[derive(Debug, Deserialize, Clone, Copy, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum BestBinFallback {
    #[default]
    CreateNew,
    Skip,
    Error,
}

#[derive(Debug, Deserialize)]
pub struct FileList {
    pub paths: Vec<PathBuf>,
    /// Optional bucket name override (for new_bucket mode)
    pub bucket_name: Option<String>,
}

pub fn parse_bucket_add_config(path: &Path) -> Result<BucketAddConfig> {
    let contents = fs::read_to_string(path)
        .context(format!("Failed to read config file: {}", path.display()))?;

    let config: BucketAddConfig = toml::from_str(&contents)
        .context("Failed to parse TOML config")?;

    // Validate threshold for best_bin mode
    if let AssignmentSettings::BestBin { threshold, .. } = &config.assignment {
        if *threshold < 0.0 || *threshold > 1.0 {
            return Err(anyhow!(
                "Config error: threshold must be between 0.0 and 1.0 (got {})",
                threshold
            ));
        }
    }

    if config.files.paths.is_empty() {
        return Err(anyhow!("Config must specify at least one file in [files].paths"));
    }

    Ok(config)
}

pub fn validate_bucket_add_config(config: &BucketAddConfig, config_dir: &Path) -> Result<()> {
    // Check target index exists
    let index_path = resolve_path(config_dir, &config.target.index);
    if !index_path.exists() {
        return Err(anyhow!(
            "Target index not found: {}",
            index_path.display()
        ));
    }

    // Check all file paths exist
    for file_path in &config.files.paths {
        let abs_path = resolve_path(config_dir, file_path);
        if !abs_path.exists() {
            return Err(anyhow!(
                "File not found: {}",
                abs_path.display()
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_parse_valid_config() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("config.toml");

        let config_content = r#"
[index]
window = 50
salt = 0x5555555555555555
output = "test.ryidx"

[buckets.TestBucket]
files = ["test.fa"]
"#;

        let mut file = File::create(&config_path).unwrap();
        file.write_all(config_content.as_bytes()).unwrap();

        let result = parse_config(&config_path);
        assert!(result.is_ok());

        let config = result.unwrap();
        assert_eq!(config.index.window, 50);
        assert_eq!(config.buckets.len(), 1);
        assert!(config.buckets.contains_key("TestBucket"));
    }

    #[test]
    fn test_parse_empty_buckets() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("config.toml");

        let config_content = r#"
[index]
window = 50
salt = 0x5555555555555555
output = "test.ryidx"
"#;

        let mut file = File::create(&config_path).unwrap();
        file.write_all(config_content.as_bytes()).unwrap();

        let result = parse_config(&config_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_path() {
        let base = Path::new("/home/user");

        // Relative path
        let relative = Path::new("file.txt");
        assert_eq!(resolve_path(base, relative), PathBuf::from("/home/user/file.txt"));

        // Absolute path
        let absolute = Path::new("/tmp/file.txt");
        assert_eq!(resolve_path(base, absolute), PathBuf::from("/tmp/file.txt"));
    }
}
