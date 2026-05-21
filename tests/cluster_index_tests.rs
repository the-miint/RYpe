//! Integration tests for the `.ryci` cluster-index format.
//!
//! These tests live outside the implementation module so they exercise
//! the public crate API the same way downstream consumers will.

use arrow::array::{Array, UInt32Array, UInt64Array};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rype::parquet_cluster_index::{
    create_cluster_index_directory, create_cluster_parquet_index, files as cluster_files,
    is_cluster_parquet_index, ClusterBucketData, ClusterParquetManifest,
};
use rype::{is_parquet_index, ParquetManifest, ShardedInvertedIndex};
use std::collections::HashSet;
use std::fs::File;
use std::path::Path;
use tempfile::TempDir;

/// Read all `(minimizer, bucket_id, position)` triples from a `.ryci` shard.
///
/// Lives in the test crate so Phase 2 can verify shard contents directly,
/// without depending on the Phase 4 reader.
fn read_shard_triples(path: &Path) -> Vec<(u64, u32, u32)> {
    let file = File::open(path).expect("open cluster shard");
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .expect("open parquet reader")
        .build()
        .expect("build parquet reader");

    let mut triples = Vec::new();
    for batch_result in reader {
        let batch = batch_result.expect("read batch");
        let minimizers = batch
            .column_by_name("minimizer")
            .expect("minimizer column missing")
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("minimizer should be UInt64");
        let bucket_ids = batch
            .column_by_name("bucket_id")
            .expect("bucket_id column missing")
            .as_any()
            .downcast_ref::<UInt32Array>()
            .expect("bucket_id should be UInt32");
        let positions = batch
            .column_by_name("position")
            .expect("position column missing")
            .as_any()
            .downcast_ref::<UInt32Array>()
            .expect("position should be UInt32");

        for i in 0..batch.num_rows() {
            triples.push((minimizers.value(i), bucket_ids.value(i), positions.value(i)));
        }
    }
    triples
}

/// WHY: the two on-disk formats must never be confused at the detection layer.
/// Confusion here would let a classify-time reader interpret a chain-index
/// shard's 3-column schema as a 2-column schema, silently misreading positions
/// as bucket_ids.
#[test]
fn detection_does_not_cross_match_between_ryxdi_and_ryci() {
    let tmp = TempDir::new().unwrap();

    // Build a .ryxdi (classify) manifest in one directory.
    let ryxdi_dir = tmp.path().join("classify.ryxdi");
    rype::parquet_index::create_index_directory(&ryxdi_dir).unwrap();
    ParquetManifest::new(64, 50, 0).save(&ryxdi_dir).unwrap();

    // Build a .ryci (cluster) manifest in another directory.
    let ryci_dir = tmp.path().join("chain.ryci");
    create_cluster_index_directory(&ryci_dir).unwrap();
    ClusterParquetManifest::new(64, 50, 0)
        .save(&ryci_dir)
        .unwrap();

    // Each format's detector must accept only its own.
    assert!(is_parquet_index(&ryxdi_dir));
    assert!(!is_parquet_index(&ryci_dir));
    assert!(is_cluster_parquet_index(&ryci_dir));
    assert!(!is_cluster_parquet_index(&ryxdi_dir));
}

/// WHY: a user who hands a `.ryci` directory to the classify-side reader
/// must get an actionable error pointing them at the right reader, not
/// a confusing TOML-parse failure or — worse — a successful open that
/// later misinterprets shard data.
#[test]
fn sharded_index_open_rejects_ryci_with_clear_error() {
    let tmp = TempDir::new().unwrap();
    let ryci_dir = tmp.path().join("chain.ryci");
    create_cluster_index_directory(&ryci_dir).unwrap();
    ClusterParquetManifest::new(64, 50, 0)
        .save(&ryci_dir)
        .unwrap();

    let err = ShardedInvertedIndex::open(&ryci_dir)
        .expect_err("ShardedInvertedIndex must refuse a .ryci directory");
    let msg = format!("{}", err);
    assert!(
        msg.contains(".ryci"),
        "error must mention the .ryci format so the user knows what they handed in: {}",
        msg
    );
}

// ----- Phase 2 tests: writer + single-shard sequential create_cluster_parquet_index -----

/// WHY: smallest possible round-trip — one minimizer, one position, one bucket.
/// If this fails, the column order / schema / encoding is fundamentally wrong.
#[test]
fn create_cluster_index_single_minimizer_round_trip() {
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().join("single.ryci");

    let bucket = ClusterBucketData {
        bucket_id: 0,
        bucket_name: "g1".to_string(),
        sources: vec!["g1.fa".to_string()],
        minimizers: vec![0xDEAD_BEEF_u64],
        positions: vec![42u32],
    };

    let manifest = create_cluster_parquet_index(&dir, vec![bucket], 64, 50, 0x5555_5555_5555_5555)
        .expect("create cluster index");

    assert_eq!(manifest.k, 64);
    assert_eq!(manifest.w, 50);
    assert_eq!(manifest.num_buckets, 1);
    assert_eq!(manifest.total_minimizers, 1);
    let inv = manifest
        .inverted
        .as_ref()
        .expect("inverted manifest must be set");
    assert_eq!(inv.num_shards, 1);
    assert_eq!(inv.total_entries, 1);
    assert_eq!(inv.shards.len(), 1);
    assert_eq!(inv.shards[0].num_entries, 1);
    assert_eq!(inv.shards[0].min_minimizer, 0xDEAD_BEEF);
    assert_eq!(inv.shards[0].max_minimizer, 0xDEAD_BEEF);

    let shard_path = dir
        .join(cluster_files::INVERTED_DIR)
        .join(cluster_files::inverted_shard(0));
    let triples = read_shard_triples(&shard_path);
    assert_eq!(triples, vec![(0xDEAD_BEEF_u64, 0u32, 42u32)]);

    // Manifest survives load — we'd lose the writer's invariants otherwise.
    let loaded = ClusterParquetManifest::load(&dir).expect("load manifest");
    assert_eq!(loaded.total_minimizers, 1);
}

/// WHY: multi-bucket varied positions exercises the per-row `(minimizer, bucket_id, position)`
/// alignment. A bug that scrambled the parallel-array pairing would be hidden by the
/// single-minimizer test because there's only one row.
#[test]
fn create_cluster_index_multi_bucket_round_trip() {
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().join("multi.ryci");

    let b0 = ClusterBucketData {
        bucket_id: 0,
        bucket_name: "a".to_string(),
        sources: vec!["a.fa".to_string()],
        minimizers: vec![100, 300, 500],
        positions: vec![1, 17, 999],
    };
    let b1 = ClusterBucketData {
        bucket_id: 1,
        bucket_name: "b".to_string(),
        sources: vec!["b.fa".to_string()],
        minimizers: vec![200, 300, 400], // 300 shared with bucket 0 — sort must keep both
        positions: vec![7, 8, 9],
    };

    // Expected: every triple from inputs, identified by (minimizer, bucket_id) → position.
    let mut expected: HashSet<(u64, u32, u32)> = HashSet::new();
    expected.insert((100, 0, 1));
    expected.insert((300, 0, 17));
    expected.insert((500, 0, 999));
    expected.insert((200, 1, 7));
    expected.insert((300, 1, 8));
    expected.insert((400, 1, 9));

    let manifest =
        create_cluster_parquet_index(&dir, vec![b0, b1], 64, 50, 0).expect("create cluster index");

    assert_eq!(manifest.num_buckets, 2);
    assert_eq!(manifest.total_minimizers, 6);
    let inv = manifest.inverted.as_ref().unwrap();
    assert_eq!(inv.num_shards, 1);
    assert_eq!(inv.total_entries, 6);

    let shard_path = dir
        .join(cluster_files::INVERTED_DIR)
        .join(cluster_files::inverted_shard(0));
    let triples = read_shard_triples(&shard_path);
    let actual: HashSet<(u64, u32, u32)> = triples.iter().copied().collect();
    assert_eq!(actual, expected, "round-trip lost or corrupted triples");

    // Shard is sorted by (minimizer, bucket_id) — the merge-join contract.
    let mut sorted = triples.clone();
    sorted.sort_by_key(|&(m, b, _)| (m, b));
    assert_eq!(
        triples, sorted,
        "shard not sorted by (minimizer, bucket_id)"
    );
}

/// WHY: a contig that yielded no minimizers (e.g. shorter than k) is legal upstream.
/// The writer must accept an entirely empty input rather than panicking on min/max.
#[test]
fn create_cluster_index_empty_input_succeeds() {
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().join("empty.ryci");

    let manifest =
        create_cluster_parquet_index(&dir, vec![], 64, 50, 0).expect("create empty cluster index");
    assert_eq!(manifest.num_buckets, 0);
    assert_eq!(manifest.total_minimizers, 0);
    let inv = manifest.inverted.as_ref().unwrap();
    assert_eq!(inv.total_entries, 0);
    // Empty input is allowed; matches .ryxdi's single-empty-shard convention.
}

/// WHY: positions are u32 and minimap2/skani conventions allow positions in long contigs
/// to approach the u32 ceiling. The writer must not lose or saturate the high bits.
/// Overflow handling is Plan 1.2 (extractor side); here we only verify storage fidelity.
#[test]
fn create_cluster_index_accepts_u32_max_position() {
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().join("max-pos.ryci");

    let bucket = ClusterBucketData {
        bucket_id: 0,
        bucket_name: "big".to_string(),
        sources: vec!["big.fa".to_string()],
        minimizers: vec![1, 2, 3],
        positions: vec![0, u32::MAX / 2, u32::MAX],
    };

    create_cluster_parquet_index(&dir, vec![bucket], 64, 50, 0).expect("create cluster index");

    let shard_path = dir
        .join(cluster_files::INVERTED_DIR)
        .join(cluster_files::inverted_shard(0));
    let triples = read_shard_triples(&shard_path);
    let actual: HashSet<(u64, u32, u32)> = triples.iter().copied().collect();
    let mut expected: HashSet<(u64, u32, u32)> = HashSet::new();
    expected.insert((1, 0, 0));
    expected.insert((2, 0, u32::MAX / 2));
    expected.insert((3, 0, u32::MAX));
    assert_eq!(actual, expected);
}

/// WHY: the merge-join shard contract requires unique minimizers within a bucket.
/// A caller that passes duplicates (and so implicitly two positions for one minimizer
/// in one bucket) must be rejected loudly at the writer boundary, not silently
/// produce a shard with rows that violate the sort/dedup invariant.
#[test]
fn create_cluster_index_rejects_duplicate_minimizers_within_bucket() {
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().join("dup.ryci");

    let bucket = ClusterBucketData {
        bucket_id: 0,
        bucket_name: "dup".to_string(),
        sources: vec!["dup.fa".to_string()],
        // Same minimizer twice → two positions for one (minimizer, bucket_id).
        minimizers: vec![100, 100],
        positions: vec![5, 10],
    };

    let err = create_cluster_parquet_index(&dir, vec![bucket], 64, 50, 0)
        .expect_err("duplicate minimizer within bucket must be rejected");
    let msg = format!("{}", err);
    assert!(
        msg.contains("duplicate"),
        "error should mention duplicate (caught by ClusterBucketData::validate): {}",
        msg
    );
}
