//! Argument parsing utilities for CLI commands.

use rype::memory::parse_byte_suffix;

/// Parse a byte size string from CLI (e.g., "4G", "512M", "auto").
/// Returns 0 for "auto" (signals auto-detection), bytes otherwise.
pub fn parse_max_memory_arg(s: &str) -> Result<usize, String> {
    parse_byte_suffix(s)
        .map(|opt| opt.unwrap_or(0)) // None (auto) -> 0
        .map_err(|e| e.to_string())
}

/// Parse a byte size string from CLI, requiring a concrete value (no "auto").
pub fn parse_shard_size_arg(s: &str) -> Result<usize, String> {
    match parse_byte_suffix(s) {
        Ok(Some(bytes)) => Ok(bytes),
        Ok(None) => Err("'auto' not supported for shard size".to_string()),
        Err(e) => Err(e.to_string()),
    }
}

/// Parse bloom filter false positive probability, validating range (0.0, 1.0).
pub fn parse_bloom_fpp(s: &str) -> Result<f64, String> {
    let fpp: f64 = s
        .parse()
        .map_err(|_| format!("'{}' is not a valid number", s))?;
    if fpp <= 0.0 || fpp >= 1.0 {
        return Err(format!(
            "bloom_filter_fpp must be in (0.0, 1.0), got {}",
            fpp
        ));
    }
    Ok(fpp)
}

/// Shared validation for positive-length arguments (--trim-to, --minimum-length).
///
/// - Must be a positive integer greater than 0
/// - Values smaller than typical k-mer sizes (16) will produce a warning but are allowed
fn validate_positive_length(s: &str, flag_name: &str) -> Result<usize, String> {
    let val: usize = s
        .parse()
        .map_err(|_| format!("'{}' is not a valid positive integer", s))?;
    if val == 0 {
        return Err(format!("{} must be greater than 0", flag_name));
    }
    if val < 16 {
        eprintln!(
            "Warning: --{} {} is smaller than the minimum k-mer size (16). \
             This will likely produce no classification results.",
            flag_name, val
        );
    }
    Ok(val)
}

/// Validate the --trim-to argument.
pub fn validate_trim_to(s: &str) -> Result<usize, String> {
    validate_positive_length(s, "trim-to")
}

/// Validate the --minimum-length argument.
pub fn validate_minimum_length(s: &str) -> Result<usize, String> {
    validate_positive_length(s, "minimum-length")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_minimum_length_valid() {
        assert_eq!(validate_minimum_length("100").unwrap(), 100);
        assert_eq!(validate_minimum_length("1").unwrap(), 1);
        assert_eq!(validate_minimum_length("16").unwrap(), 16);
    }

    #[test]
    fn test_validate_minimum_length_zero() {
        let err = validate_minimum_length("0").unwrap_err();
        assert!(err.contains("must be greater than 0"));
    }

    #[test]
    fn test_validate_minimum_length_non_numeric() {
        let err = validate_minimum_length("abc").unwrap_err();
        assert!(err.contains("not a valid positive integer"));
    }

    #[test]
    fn test_validate_trim_to_valid() {
        assert_eq!(validate_trim_to("100").unwrap(), 100);
        assert_eq!(validate_trim_to("1").unwrap(), 1);
    }

    #[test]
    fn test_validate_trim_to_zero() {
        let err = validate_trim_to("0").unwrap_err();
        assert!(err.contains("must be greater than 0"));
    }

    #[test]
    fn test_validate_trim_to_non_numeric() {
        let err = validate_trim_to("abc").unwrap_err();
        assert!(err.contains("not a valid positive integer"));
    }
}
