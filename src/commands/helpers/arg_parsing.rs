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

/// Validate the --trim-to argument.
///
/// - Must be a positive integer greater than 0
/// - Values smaller than typical k-mer sizes (16) will produce a warning but are allowed
pub fn validate_trim_to(s: &str) -> Result<usize, String> {
    let val: usize = s
        .parse()
        .map_err(|_| format!("'{}' is not a valid positive integer", s))?;
    if val == 0 {
        return Err("trim_to must be greater than 0".to_string());
    }
    // Warn about very small values that won't produce useful results
    // (minimum k-mer size is 16, so anything less won't generate minimizers)
    if val < 16 {
        eprintln!(
            "Warning: --trim-to {} is smaller than the minimum k-mer size (16). \
             This will likely produce no classification results.",
            val
        );
    }
    Ok(val)
}
