//! Integration tests for the `.ryci` cluster-index format.
//!
//! These tests live outside the implementation module so they exercise
//! the public crate API the same way downstream consumers will.

use rype::parquet_cluster_index::{
    create_cluster_index_directory, is_cluster_parquet_index, ClusterParquetManifest,
};
use rype::{is_parquet_index, ParquetManifest, ShardedInvertedIndex};
use tempfile::TempDir;

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
