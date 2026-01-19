//! Benchmark the fixed Parquet implementation
//!
//! Tests the streaming save and k-way merge load implementations.

use anyhow::Result;
use rype::InvertedIndex;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <shard_path>", args[0]);
        eprintln!(
            "Example: {} perf-data/n100-w200-fixed.ryxdi.shard.0",
            args[0]
        );
        std::process::exit(1);
    }

    let shard_path = Path::new(&args[1]);
    let parquet_path = shard_path.with_extension("bench.parquet");

    println!("=== Benchmarking Fixed Parquet Implementation ===\n");

    // 1. Load legacy format
    println!("1. Loading legacy format: {}", shard_path.display());
    let t1 = Instant::now();
    let index = InvertedIndex::load_shard(shard_path)?;
    let legacy_load_ms = t1.elapsed().as_millis();
    println!(
        "   Loaded {} minimizers, {} bucket_ids in {}ms",
        index.minimizers().len(),
        index.bucket_ids().len(),
        legacy_load_ms
    );

    let legacy_size = std::fs::metadata(shard_path)?.len();
    println!("   File size: {:.2} MB", legacy_size as f64 / 1_000_000.0);

    // 2. Save as Parquet (our streaming implementation)
    println!(
        "\n2. Saving as Parquet (streaming): {}",
        parquet_path.display()
    );
    let t2 = Instant::now();
    let shard_info = index.save_shard_parquet(&parquet_path, 0, None)?;
    let save_ms = t2.elapsed().as_millis();
    let parquet_size = std::fs::metadata(&parquet_path)?.len();
    println!(
        "   Saved {} minimizers, {} bucket_ids in {}ms",
        shard_info.num_minimizers, shard_info.num_bucket_ids, save_ms
    );
    println!(
        "   Parquet size: {:.2} MB ({:.1}% of legacy)",
        parquet_size as f64 / 1_000_000.0,
        parquet_size as f64 / legacy_size as f64 * 100.0
    );

    // 3. Load Parquet (our k-way merge implementation)
    println!("\n3. Loading Parquet (k-way merge + parallel row groups)...");
    let t3 = Instant::now();
    // Use dummy source_hash - it's only used for validation against manifest
    let loaded = InvertedIndex::load_shard_parquet_with_params(
        &parquet_path,
        index.k,
        index.w,
        index.salt,
        0, // source_hash not needed for standalone benchmark
    )?;
    let parquet_load_ms = t3.elapsed().as_millis();
    println!(
        "   Loaded {} minimizers, {} bucket_ids in {}ms",
        loaded.minimizers().len(),
        loaded.bucket_ids().len(),
        parquet_load_ms
    );

    // 4. Verify correctness
    println!("\n4. Verifying correctness...");
    assert_eq!(
        index.minimizers().len(),
        loaded.minimizers().len(),
        "Minimizer count mismatch"
    );
    assert_eq!(
        index.bucket_ids().len(),
        loaded.bucket_ids().len(),
        "Bucket ID count mismatch"
    );
    assert_eq!(index.minimizers(), loaded.minimizers(), "Minimizers differ");
    assert_eq!(index.bucket_ids(), loaded.bucket_ids(), "Bucket IDs differ");
    println!("   All data verified!");

    // Summary
    println!("\n=== Performance Summary ===");
    println!("| Metric              | Legacy      | Parquet     | Improvement |");
    println!("|---------------------|-------------|-------------|-------------|");
    println!(
        "| File size (MB)      | {:>11.2} | {:>11.2} | {:>10.1}% |",
        legacy_size as f64 / 1_000_000.0,
        parquet_size as f64 / 1_000_000.0,
        (1.0 - parquet_size as f64 / legacy_size as f64) * 100.0
    );
    println!(
        "| Load time (ms)      | {:>11} | {:>11} | {:>10.2}x |",
        legacy_load_ms,
        parquet_load_ms,
        legacy_load_ms as f64 / parquet_load_ms as f64
    );
    println!(
        "| Save time (ms)      |         N/A | {:>11} |         N/A |",
        save_ms
    );

    // Cleanup
    std::fs::remove_file(&parquet_path).ok();

    Ok(())
}
