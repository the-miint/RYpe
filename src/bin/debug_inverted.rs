//! Debug tool for inspecting Parquet inverted indices.

use anyhow::Result;
use rype::{
    get_paired_minimizers_into, is_parquet_index, MinimizerWorkspace, ShardedInvertedIndex,
};
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: debug_inverted <index.ryxdi>");
        std::process::exit(1);
    }

    let index_path = PathBuf::from(&args[1]);

    println!("=== File paths ===");
    println!("Index path: {:?}", index_path);
    println!("Is Parquet index: {}", is_parquet_index(&index_path));

    // Check if it's a Parquet index
    if !is_parquet_index(&index_path) {
        eprintln!("Error: Not a Parquet index at {:?}", index_path);
        eprintln!("Expected a directory with manifest.toml");
        std::process::exit(1);
    }

    // Load the Parquet inverted index
    println!("\n=== Inverted Index Manifest ===");
    let sharded = ShardedInvertedIndex::open(&index_path)?;
    let manifest = sharded.manifest();

    println!("K: {}", manifest.k);
    println!("W: {}", manifest.w);
    println!("Salt: 0x{:016x}", manifest.salt);
    println!("Source hash: 0x{:016x}", manifest.source_hash);
    println!("Total minimizers: {}", manifest.total_minimizers);
    println!("Total bucket IDs: {}", manifest.total_bucket_ids);
    println!("Num shards: {}", manifest.shards.len());
    println!("Num buckets: {}", manifest.bucket_names.len());

    for (i, shard) in manifest.shards.iter().enumerate() {
        println!(
            "  Shard {}: id={}, min_start={}, min_end={}, num_mins={}, num_bids={}",
            i,
            shard.shard_id,
            shard.min_start,
            shard.min_end,
            shard.num_minimizers,
            shard.num_bucket_ids
        );
    }

    // Display bucket names
    println!("\n=== Buckets ===");
    let mut bucket_ids: Vec<_> = manifest.bucket_names.keys().collect();
    bucket_ids.sort();
    for id in bucket_ids.iter().take(10) {
        let name = manifest.bucket_names.get(id).unwrap();
        println!("  Bucket {}: {}", id, name);
    }
    if bucket_ids.len() > 10 {
        println!("  ... and {} more buckets", bucket_ids.len() - 10);
    }

    // Load first shard and check its contents
    if !manifest.shards.is_empty() {
        println!("\n=== Loading first shard for inspection ===");
        let shard = sharded.load_shard(0)?;
        println!("Shard loaded successfully");
        println!("  K: {}", shard.k);
        println!("  W: {}", shard.w);
        println!("  Salt: 0x{:016x}", shard.salt);
        println!("  Num minimizers: {}", shard.num_minimizers());
        println!("  Num bucket entries: {}", shard.num_bucket_entries());
    }

    // Test query extraction
    println!("\n=== Test Query ===");
    let test_seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
    let mut ws = MinimizerWorkspace::new();

    let (fwd, rc) = get_paired_minimizers_into(
        test_seq,
        None,
        manifest.k,
        manifest.w,
        manifest.salt,
        &mut ws,
    );

    println!(
        "Using manifest params (K={}, W={}, salt=0x{:016x}):",
        manifest.k, manifest.w, manifest.salt
    );
    println!("  Forward minimizers: {}", fwd.len());
    println!("  RC minimizers: {}", rc.len());
    if !fwd.is_empty() {
        println!("  First 5 fwd: {:?}", &fwd[..5.min(fwd.len())]);
    }

    // Try to find matches in first shard
    if !manifest.shards.is_empty() {
        let shard = sharded.load_shard(0)?;

        println!("\n=== Testing get_bucket_hits on first shard ===");
        let hits = shard.get_bucket_hits(&fwd);
        println!("Forward hits: {} unique buckets", hits.len());
        if !hits.is_empty() {
            let mut hit_vec: Vec<_> = hits.iter().collect();
            hit_vec.sort_by_key(|(_, &c)| std::cmp::Reverse(c));
            println!("Top hits: {:?}", &hit_vec[..5.min(hit_vec.len())]);
        }
    }

    Ok(())
}
