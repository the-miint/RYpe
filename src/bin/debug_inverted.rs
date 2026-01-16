use anyhow::Result;
use rype::{
    get_paired_minimizers_into, InvertedIndex, MainIndexManifest, MinimizerWorkspace,
    ShardManifest, ShardedInvertedIndex,
};
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: debug_inverted <index.ryidx>");
        std::process::exit(1);
    }

    let index_path = PathBuf::from(&args[1]);

    // Check what files exist
    let ryxdi_path = index_path.with_extension("ryxdi");
    let ryxdi_manifest = ryxdi_path.with_file_name(format!(
        "{}.manifest",
        ryxdi_path.file_name().unwrap().to_string_lossy()
    ));
    let main_manifest_path = index_path.with_file_name(format!(
        "{}.manifest",
        index_path.file_name().unwrap().to_string_lossy()
    ));

    println!("=== File paths ===");
    println!("Index path: {:?}", index_path);
    println!(
        "Main manifest: {:?} (exists: {})",
        main_manifest_path,
        main_manifest_path.exists()
    );
    println!(
        "Inverted path: {:?} (exists: {})",
        ryxdi_path,
        ryxdi_path.exists()
    );
    println!(
        "Inverted manifest: {:?} (exists: {})",
        ryxdi_manifest,
        ryxdi_manifest.exists()
    );

    // Load main manifest
    println!("\n=== Main Index Manifest ===");
    if main_manifest_path.exists() {
        let main_manifest = MainIndexManifest::load(&main_manifest_path)?;
        println!("K: {}", main_manifest.k);
        println!("W: {}", main_manifest.w);
        println!("Salt: 0x{:016x}", main_manifest.salt);
        println!("Buckets: {}", main_manifest.bucket_names.len());
        println!("Shards: {}", main_manifest.shards.len());
        println!("Total minimizers: {}", main_manifest.total_minimizers);
    } else {
        println!("Main manifest not found");
    }

    // Load inverted manifest
    println!("\n=== Inverted Index Manifest ===");
    if ryxdi_manifest.exists() {
        let sharded = ShardedInvertedIndex::open(&ryxdi_path)?;
        let manifest = sharded.manifest();
        println!("K: {}", manifest.k);
        println!("W: {}", manifest.w);
        println!("Salt: 0x{:016x}", manifest.salt);
        println!("Source hash: 0x{:016x}", manifest.source_hash);
        println!("Total minimizers: {}", manifest.total_minimizers);
        println!("Total bucket IDs: {}", manifest.total_bucket_ids);
        println!("Num shards: {}", manifest.shards.len());

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

        // Load first shard and check its contents
        if !manifest.shards.is_empty() {
            println!("\n=== Loading first shard for inspection ===");
            let shard_path = ShardManifest::shard_path(&ryxdi_path, 0);
            println!("Shard path: {:?}", shard_path);

            let shard = InvertedIndex::load_shard(&shard_path)?;
            println!("Shard loaded successfully");
            println!("  K: {}", shard.k);
            println!("  W: {}", shard.w);
            println!("  Salt: 0x{:016x}", shard.salt);
            println!("  Num minimizers: {}", shard.num_minimizers());
            println!("  Num bucket entries: {}", shard.num_bucket_entries());

            // Note: minimizers field is private, so we can only see the count
            println!("  (minimizers field is private - cannot inspect values directly)");
        }
    }

    // Test query extraction
    println!("\n=== Test Query ===");
    let test_seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";

    // Get parameters from main manifest
    if main_manifest_path.exists() {
        let main_manifest = MainIndexManifest::load(&main_manifest_path)?;
        let mut ws = MinimizerWorkspace::new();

        let (fwd, rc) = get_paired_minimizers_into(
            test_seq,
            None,
            main_manifest.k,
            main_manifest.w,
            main_manifest.salt,
            &mut ws,
        );

        println!(
            "Using main manifest params (K={}, W={}, salt=0x{:016x}):",
            main_manifest.k, main_manifest.w, main_manifest.salt
        );
        println!("  Forward minimizers: {}", fwd.len());
        println!("  RC minimizers: {}", rc.len());
        if !fwd.is_empty() {
            println!("  First 5 fwd: {:?}", &fwd[..5.min(fwd.len())]);
        }
    }

    if ryxdi_manifest.exists() {
        let sharded = ShardedInvertedIndex::open(&ryxdi_path)?;
        let manifest = sharded.manifest();
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
            "Using inverted manifest params (K={}, W={}, salt=0x{:016x}):",
            manifest.k, manifest.w, manifest.salt
        );
        println!("  Forward minimizers: {}", fwd.len());
        println!("  RC minimizers: {}", rc.len());
        if !fwd.is_empty() {
            println!("  First 5 fwd: {:?}", &fwd[..5.min(fwd.len())]);
        }

        // Try to find matches in first shard
        if !manifest.shards.is_empty() {
            let shard_path = ShardManifest::shard_path(&ryxdi_path, 0);
            let shard = InvertedIndex::load_shard(&shard_path)?;

            println!("\n=== Testing get_bucket_hits on first shard ===");
            let hits = shard.get_bucket_hits(&fwd);
            println!("Forward hits: {} unique buckets", hits.len());
            if !hits.is_empty() {
                let mut hit_vec: Vec<_> = hits.iter().collect();
                hit_vec.sort_by_key(|(_, &c)| std::cmp::Reverse(c));
                println!("Top hits: {:?}", &hit_vec[..5.min(hit_vec.len())]);
            }

            // Can't do manual binary search since minimizers field is private
            // The get_bucket_hits call above shows if there are matches
        }
    }

    Ok(())
}
