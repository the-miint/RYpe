//! End-to-end Plan 1.2 ↔ Plan 1.1 closure.
//!
//! Builds a deterministic sequence, runs `extract_into_with_positions` +
//! `pairs_into_cluster_bucket_arrays` to produce the parallel arrays Plan
//! 1.1's writer expects, constructs a `ClusterBucketData`, writes a `.ryci`,
//! reads it back through `ClusterParquetIndex`, and proves that every
//! `(minimizer, _, position)` triple recovered from disk decodes back to
//! the right k-mer in the original sequence.
//!
//! This is the load-bearing integration test for Plan 1.2: positions are
//! preserved end-to-end and they actually point at the k-mer that hashes
//! to the claimed value.

use rype::parquet_cluster_index::{
    create_cluster_parquet_index, ClusterBucketData, ClusterParquetIndex,
};
use rype::{extract_into_with_positions, MinimizerWorkspace};
use std::path::PathBuf;
use tempfile::TempDir;

/// Compute the RY-encoded hash of a k-mer at a given offset in `seq`, using
/// the same algorithm `extract_into` uses internally. Test-only — re-derived
/// here to keep the integration test independent of the extractor's internals.
///
/// Returns `None` if the k-mer spans an invalid base (N etc.); panics on an
/// out-of-bounds offset since that's caller error in the test.
fn kmer_hash(seq: &[u8], pos: usize, k: usize, salt: u64) -> Option<u64> {
    let k_mask = if k == 64 { u64::MAX } else { (1u64 << k) - 1 };
    let mut val: u64 = 0;
    for &b in &seq[pos..pos + k] {
        // RY-space mapping per src/core/encoding.rs BASE_TO_BIT_LUT:
        // purines (A/G) → 1, pyrimidines (C/T) → 0.
        let bit = match b {
            b'A' | b'a' | b'G' | b'g' => 1u64,
            b'C' | b'c' | b'T' | b't' => 0u64,
            _ => return None,
        };
        val = ((val << 1) | bit) & k_mask;
    }
    Some(val ^ salt)
}

/// Deterministic non-trivial DNA sequence — alternating-ish pattern that
/// gives a mix of purine and pyrimidine bases so minimizer extraction
/// produces a non-trivial set of (hash, position) pairs (not all-zero).
fn make_test_seq() -> Vec<u8> {
    // 520 bases: 130 copies of "ACGT". Periodic enough to ensure
    // non-consecutive duplicate hashes (dedup-helper actually exercised).
    let mut seq = Vec::with_capacity(520);
    for _ in 0..130 {
        seq.extend_from_slice(b"ACGT");
    }
    seq
}

#[test]
fn extract_into_with_positions_feeds_ryci_round_trip() {
    // Use a scratch tempdir under the project's `scratch/` so /tmp isn't
    // touched (CLAUDE.md operational rule).
    let scratch_base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scratch");
    std::fs::create_dir_all(&scratch_base).unwrap();
    let tmp = TempDir::new_in(&scratch_base).unwrap();
    let dir = tmp.path().join("roundtrip.ryci");

    let seq = make_test_seq();
    let k: usize = 16;
    let w: usize = 5;
    let salt: u64 = 0;

    // Plan 1.2: extract + pair-up.
    let mut ws = MinimizerWorkspace::new();
    extract_into_with_positions(&seq, k, w, salt, &mut ws).expect("extract must succeed");
    assert!(
        !ws.buffer.is_empty(),
        "test seq must produce at least one minimizer"
    );

    let mut hashes = std::mem::take(&mut ws.buffer);
    let mut positions = std::mem::take(&mut ws.positions_fwd);
    let extracted_count_before_dedup = hashes.len();
    rype::pairs_into_cluster_bucket_arrays(&mut hashes, &mut positions);
    assert_eq!(hashes.len(), positions.len());
    assert!(
        hashes.len() <= extracted_count_before_dedup,
        "dedup may only shrink, not grow"
    );
    // Sorted by hash (load-bearing: Plan 1.1's ClusterBucketData::validate
    // requires this).
    for win in hashes.windows(2) {
        assert!(
            win[0] < win[1],
            "hashes must be strictly ascending after the helper"
        );
    }

    // Plan 1.1: construct + validate + write.
    let bucket = ClusterBucketData {
        bucket_id: 0,
        bucket_name: "roundtrip".into(),
        sources: vec!["roundtrip.fa".into()],
        minimizers: hashes.clone(),
        positions: positions.clone(),
    };
    bucket
        .validate()
        .expect("Plan 1.2 output must satisfy Plan 1.1's ClusterBucketData::validate");

    let manifest =
        create_cluster_parquet_index(&dir, vec![bucket], k, w, salt).expect("create cluster index");
    assert_eq!(manifest.total_minimizers as usize, hashes.len());

    // Read back through the public reader and prove every triple's position
    // actually points at a k-mer whose RY hash equals the claimed minimizer.
    let index = ClusterParquetIndex::open(&dir).expect("open .ryci");
    assert_eq!(index.num_shards(), 1);
    let triples = index.load_shard(0).expect("load_shard(0)");
    assert_eq!(triples.len(), hashes.len());

    // Build a quick lookup of (hash → first position) from our pre-write
    // arrays, so we can also check the reader saw the same data.
    let expected: std::collections::HashMap<u64, u32> = hashes
        .iter()
        .zip(positions.iter())
        .map(|(&h, &p)| (h, p))
        .collect();

    for (mz, bucket_id, position) in triples {
        assert_eq!(bucket_id, 0, "single-bucket index — bucket_id must be 0");
        // 1. The reader saw the same hash → position we wrote.
        assert_eq!(
            expected.get(&mz).copied(),
            Some(position),
            "reader returned a (hash, position) pair we didn't write"
        );
        // 2. Load-bearing: the position actually points at a k-mer in the
        // source sequence whose RY hash equals the claimed minimizer. This
        // is the end-to-end proof that positions survived extraction, dedup,
        // serialization, and deserialization without drift.
        let recomputed = kmer_hash(&seq, position as usize, k, salt)
            .expect("test seq has no Ns; kmer_hash must succeed");
        assert_eq!(
            recomputed, mz,
            "position {} in seq does not decode to claimed hash {:#x} (got {:#x})",
            position, mz, recomputed
        );
    }
}
