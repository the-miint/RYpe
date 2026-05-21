//! Integration tests for the `.ryci` cluster-index format.
//!
//! These tests live outside the implementation module so they exercise
//! the public crate API the same way downstream consumers will.

use arrow::array::{Array, UInt32Array, UInt64Array};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rype::parquet_cluster_index::{
    create_cluster_index_directory, create_cluster_parquet_index,
    create_cluster_parquet_index_with_options, files as cluster_files, is_cluster_parquet_index,
    ClusterBucketData, ClusterParquetManifest, ClusterParquetWriteOptions,
};
use rype::{is_parquet_index, ParquetCompression, ParquetManifest, ShardedInvertedIndex};
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
/// The writer must accept an entirely empty input rather than panicking on min/max,
/// AND must still produce a readable zero-row shard file (so the Phase 4 reader has
/// something to open).
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
    // Empty-range sentinel: min > max signals "this shard covers no minimizers",
    // distinguishable from a shard that legitimately holds minimizer 0.
    assert_eq!(inv.shards[0].min_minimizer, u64::MAX);
    assert_eq!(inv.shards[0].max_minimizer, 0);

    // The shard file must exist and be a readable zero-row Parquet file.
    let shard_path = dir
        .join(cluster_files::INVERTED_DIR)
        .join(cluster_files::inverted_shard(0));
    assert!(shard_path.exists(), "empty shard file must be on disk");
    let triples = read_shard_triples(&shard_path);
    assert!(triples.is_empty(), "empty shard must have zero rows");
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

/// WHY: two `ClusterBucketData` with the same `bucket_id` would silently overwrite
/// each other in the bucket-metadata HashMap while every triple from the overwritten
/// bucket still reaches the shard — the shard would reference a `bucket_id` whose
/// name lives in the manifest pointing at the *wrong* bucket's sources. The writer
/// must reject this loudly at the boundary.
#[test]
fn create_cluster_index_rejects_duplicate_bucket_id_across_inputs() {
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().join("dup-bid.ryci");

    let b0 = ClusterBucketData {
        bucket_id: 7,
        bucket_name: "alpha".to_string(),
        sources: vec!["alpha.fa".to_string()],
        minimizers: vec![1, 2],
        positions: vec![0, 0],
    };
    let b1 = ClusterBucketData {
        bucket_id: 7, // same id, different name/sources — must be rejected.
        bucket_name: "beta".to_string(),
        sources: vec!["beta.fa".to_string()],
        minimizers: vec![3, 4],
        positions: vec![0, 0],
    };

    let err = create_cluster_parquet_index(&dir, vec![b0, b1], 64, 50, 0)
        .expect_err("duplicate bucket_id must be rejected");
    let msg = format!("{}", err);
    assert!(
        msg.contains("duplicate bucket_id"),
        "error should mention duplicate bucket_id: {}",
        msg
    );
}

/// WHY: the `_with_options` path is a separate entry point. A bug that silently
/// drops options (e.g. forgot to thread them into the writer) would be invisible
/// from the no-options path because defaults happen to work. Pass Zstd through
/// end-to-end and verify the index still round-trips correctly.
#[test]
fn create_cluster_index_with_zstd_options_round_trips() {
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().join("zstd.ryci");

    let bucket = ClusterBucketData {
        bucket_id: 0,
        bucket_name: "g".to_string(),
        sources: vec!["g.fa".to_string()],
        minimizers: vec![10, 20, 30],
        positions: vec![100, 200, 300],
    };

    let opts = ClusterParquetWriteOptions {
        compression: ParquetCompression::Zstd,
        ..Default::default()
    };

    let manifest =
        create_cluster_parquet_index_with_options(&dir, vec![bucket], 64, 50, 0, None, Some(&opts))
            .expect("create cluster index with options");
    assert_eq!(manifest.total_minimizers, 3);

    let shard_path = dir
        .join(cluster_files::INVERTED_DIR)
        .join(cluster_files::inverted_shard(0));
    let triples = read_shard_triples(&shard_path);
    let actual: std::collections::HashSet<(u64, u32, u32)> = triples.iter().copied().collect();
    let mut expected = std::collections::HashSet::new();
    expected.insert((10u64, 0u32, 100u32));
    expected.insert((20u64, 0u32, 200u32));
    expected.insert((30u64, 0u32, 300u32));
    assert_eq!(actual, expected);
}

// ----- Phase 3 tests: multi-shard sequential k-way merge -----

/// WHY: this is the load-bearing test for Phase 3. When max_shard_bytes is set,
/// the writer must roll over to a new shard, the manifest must accurately reflect
/// per-shard ranges, and no triple may be lost or duplicated across the boundary.
/// A bug here corrupts every downstream consumer.
#[test]
fn create_cluster_index_rolls_over_at_max_shard_bytes() {
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().join("multi-shard.ryci");

    // Two buckets, 1000 minimizers each, interleaved by parity to force the
    // merge join to alternate buckets across the run. Small row_group_size
    // makes bytes_written update frequently enough that a tiny max_shard_bytes
    // can trigger rollover with this small dataset.
    let n: u64 = 1000;
    let b0_mins: Vec<u64> = (0..n).map(|i| 2 * i).collect();
    let b0_pos: Vec<u32> = (0..n).map(|i| i as u32).collect();
    let b1_mins: Vec<u64> = (0..n).map(|i| 2 * i + 1).collect();
    let b1_pos: Vec<u32> = (0..n).map(|i| (i + 100_000) as u32).collect();

    let b0 = ClusterBucketData {
        bucket_id: 0,
        bucket_name: "a".into(),
        sources: vec!["a.fa".into()],
        minimizers: b0_mins.clone(),
        positions: b0_pos.clone(),
    };
    let b1 = ClusterBucketData {
        bucket_id: 1,
        bucket_name: "b".into(),
        sources: vec!["b.fa".into()],
        minimizers: b1_mins.clone(),
        positions: b1_pos.clone(),
    };

    let opts = ClusterParquetWriteOptions {
        row_group_size: 50,
        ..Default::default()
    };

    let manifest = create_cluster_parquet_index_with_options(
        &dir,
        vec![b0, b1],
        64,
        50,
        0,
        Some(2048), // max_shard_bytes — small enough to force rollover with row_group_size=50
        Some(&opts),
    )
    .expect("create multi-shard cluster index");

    let inv = manifest
        .inverted
        .as_ref()
        .expect("inverted manifest must be set");
    assert!(
        inv.num_shards >= 2,
        "expected ≥2 shards with max_shard_bytes=2048, got {}",
        inv.num_shards
    );
    assert!(
        inv.has_overlapping_shards,
        "sequential multi-shard must set has_overlapping_shards=true"
    );
    assert_eq!(inv.shards.len() as u32, inv.num_shards);

    // Per-shard: file exists, manifest min/max/num_entries matches actual file data.
    let mut all_triples: std::collections::HashSet<(u64, u32, u32)> =
        std::collections::HashSet::new();
    let mut total_entries: u64 = 0;
    for shard in &inv.shards {
        let shard_path = dir
            .join(cluster_files::INVERTED_DIR)
            .join(cluster_files::inverted_shard(shard.shard_id));
        assert!(
            shard_path.exists(),
            "shard file {} missing on disk",
            shard.shard_id
        );
        let triples = read_shard_triples(&shard_path);
        assert_eq!(
            triples.len() as u64,
            shard.num_entries,
            "shard {} num_entries mismatch",
            shard.shard_id
        );
        // Manifest claims must be honest — they drive the Phase 3+ range-skip path.
        let actual_min = triples.iter().map(|t| t.0).min().expect("nonempty shard");
        let actual_max = triples.iter().map(|t| t.0).max().expect("nonempty shard");
        assert_eq!(
            actual_min, shard.min_minimizer,
            "shard {} min_minimizer mismatch (manifest={:#x}, actual={:#x})",
            shard.shard_id, shard.min_minimizer, actual_min
        );
        assert_eq!(
            actual_max, shard.max_minimizer,
            "shard {} max_minimizer mismatch (manifest={:#x}, actual={:#x})",
            shard.shard_id, shard.max_minimizer, actual_max
        );
        for t in triples {
            all_triples.insert(t);
        }
        total_entries += shard.num_entries;
    }

    assert_eq!(total_entries, 2 * n, "sum of per-shard entries != total");
    assert_eq!(inv.total_entries, 2 * n);
    assert_eq!(manifest.total_minimizers, 2 * n);

    // Every input triple must appear exactly once across the union of shards.
    let mut expected: std::collections::HashSet<(u64, u32, u32)> = std::collections::HashSet::new();
    for i in 0..n as usize {
        expected.insert((b0_mins[i], 0, b0_pos[i]));
        expected.insert((b1_mins[i], 1, b1_pos[i]));
    }
    assert_eq!(
        all_triples, expected,
        "union of shard triples doesn't match input"
    );

    // Shard ranges must be ascending (sequential mode emits shards in sorted order).
    for w in inv.shards.windows(2) {
        assert!(
            w[0].max_minimizer <= w[1].min_minimizer,
            "shard {} ends at {:#x} but shard {} starts at {:#x} — ranges out of order",
            w[0].shard_id,
            w[0].max_minimizer,
            w[1].shard_id,
            w[1].min_minimizer
        );
    }
}
