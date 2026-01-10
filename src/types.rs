//! Core types used throughout the rype library.

use std::collections::HashMap;

/// ID (i64), Sequence Reference, Optional Pair Sequence Reference
pub type QueryRecord<'a> = (i64, &'a [u8], Option<&'a [u8]>);

/// Lightweight metadata-only view of an Index (without minimizer data)
#[derive(Debug, Clone)]
pub struct IndexMetadata {
    pub k: usize,
    pub w: usize,
    pub salt: u64,
    pub bucket_names: HashMap<u32, String>,
    pub bucket_sources: HashMap<u32, Vec<String>>,
    pub bucket_minimizer_counts: HashMap<u32, usize>,
}

/// Query ID, Bucket ID, Score
#[derive(Debug, Clone, PartialEq)]
pub struct HitResult {
    pub query_id: i64,
    pub bucket_id: u32,
    pub score: f64,
}
